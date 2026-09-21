"""
EdgeBERT: compact transformer for network-flow sequences with XAI-friendly attention.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Final, Iterator, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants (memory target: small footprint for edge deployment)
# ---------------------------------------------------------------------------

DEFAULT_D_MODEL: Final[int] = 128
DEFAULT_NHEAD: Final[int] = 4
DEFAULT_NUM_ENCODER_LAYERS: Final[int] = 3
DEFAULT_DIM_FEEDFORWARD: Final[int] = 512
DEFAULT_TOKEN_EMB_DIM: Final[int] = 32
DEFAULT_SEQ_LEN: Final[int] = 5
DEFAULT_DROPOUT: Final[float] = 0.1


@contextmanager
def _mha_slow_path_if_quantized_ffn(encoder_layers: nn.ModuleList) -> Iterator[None]:
    """
    Dynamic ``nn.Linear`` quantization replaces weights with packed params; the
    fused MHA fast path in ``nn.TransformerEncoderLayer`` then mis-detects tensor
    arguments. Disable the fast path only when quantized FFN linears are present.
    """
    from torch.ao.nn.quantized.dynamic.modules.linear import Linear as DynamicQuantizedLinear

    need_slow = any(
        isinstance(getattr(layer, "linear1", None), DynamicQuantizedLinear)
        for layer in encoder_layers
    )
    if not need_slow:
        yield
        return
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FinalSelfAttentionBlock(nn.Module):
    """
    Transformer encoder block with explicit multi-head attention weights.

    Mirrors ``nn.TransformerEncoderLayer`` (post-norm, batch_first) but uses
    ``nn.MultiheadAttention(..., need_weights=True)`` for the self-attention step.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = self._get_activation(activation)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name!r}")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_mid = self.activation(self.linear1(x))
        ffn_mid = self.dropout_ffn(ffn_mid)
        ffn_out = self.linear2(ffn_mid)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


class EdgeBERT(nn.Module):
    """
    Edge-oriented BERT-style model for sequences ``(batch, 5, 4)``.

    Channels: ``[:, :, 0]`` protocol token id, ``[:, :, 1]`` port token id,
    ``[:, :, 2:]`` two continuous features.

    Forward returns ``(logits, final_layer_attention_weights)`` for explainability.
    """

    def __init__(
        self,
        protocol_vocab_size: int,
        port_vocab_size: int,
        *,
        token_embedding_dim: int = DEFAULT_TOKEN_EMB_DIM,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        max_seq_len: int = DEFAULT_SEQ_LEN,
        pooling: Literal["mean", "last"] = "mean",
        encoder_activation: str = "relu",
    ) -> None:
        super().__init__()
        if protocol_vocab_size < 1 or port_vocab_size < 1:
            raise ValueError("Vocabulary sizes must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.d_model = d_model
        self.pooling = pooling

        self.protocol_embed = nn.Embedding(protocol_vocab_size, token_embedding_dim)
        self.port_embed = nn.Embedding(port_vocab_size, token_embedding_dim)
        fused_in = token_embedding_dim * 2 + 2
        self.input_proj = nn.Linear(fused_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(max_seq_len, 16), dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=encoder_activation,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(3)
            ]
        )

        self.final_block = FinalSelfAttentionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=encoder_activation,
        )
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 3 or x.size(-1) != 4:
            raise ValueError(f"Expected x shape (batch, seq, 4), got {tuple(x.shape)}.")

        proto = x[:, :, 0].long().clamp(min=0, max=self.protocol_embed.num_embeddings - 1)
        port = x[:, :, 1].long().clamp(min=0, max=self.port_embed.num_embeddings - 1)
        cont = x[:, :, 2:].float()

        p_emb = self.protocol_embed(proto)
        t_emb = self.port_embed(port)
        fused = torch.cat([p_emb, t_emb, cont], dim=-1)
        h: Tensor = self.input_proj(fused)
        h = self.pos_encoder(h)

        with _mha_slow_path_if_quantized_ffn(self.encoder_layers):
            for layer in self.encoder_layers:
                h = layer(h)

        h, attn_weights = self.final_block(h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h[:, -1, :]

        logits: Tensor = self.classifier(pooled)
        return logits, attn_weights


def train_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    lr: float,
) -> None:
    """
    Standard supervised training with binary labels (Benign=0, Malicious=1).

    Expects ``dataloader`` batches of ``(x, y)`` where ``x`` matches ``EdgeBERT``
    input and ``y`` is ``(batch,)`` or ``(batch, 1)`` with integer or float 0/1.
    """
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x)
            loss = criterion(logits.view(-1), batch_y)
            loss.backward()
            optimizer.step()


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization to all :class:`nn.Linear` layers (CPU).

    Returns a new quantized module; the original ``model`` is unchanged when
    ``inplace=False``.

    Note: ``EdgeBERT`` disables the MHA fused fast path during the first three
    encoder layers when FFN linears are dynamically quantized, because the
    fast-path compatibility check in ``nn.TransformerEncoderLayer`` expects
    plain :class:`~torch.nn.Parameter` weights on the FFN linears.
    """
    model_cpu = model.cpu().eval()
    return torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )


__all__ = [
    "DEFAULT_D_MODEL",
    "DEFAULT_DIM_FEEDFORWARD",
    "DEFAULT_NHEAD",
    "DEFAULT_NUM_ENCODER_LAYERS",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_TOKEN_EMB_DIM",
    "EdgeBERT",
    "FinalSelfAttentionBlock",
    "PositionalEncoding",
    "quantize_model",
    "train_model",
]
