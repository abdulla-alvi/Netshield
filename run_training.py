"""
Proof-of-concept training: NetworkBehaviorDataset → EdgeBERT → dynamic quantization bundle.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from sqlalchemy import text
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from app.common.db import get_db_session
from data_pipeline import NSSTokenizer, NetworkBehaviorDataset
from model_engine import EdgeBERT, quantize_model, train_model


class _BinaryBenignAttackDataset(Dataset[tuple[Tensor, Tensor]]):
    """Maps multi-class string labels to binary: BENIGN=0, any attack=1 (for ``BCEWithLogitsLoss``)."""

    def __init__(self, base: NetworkBehaviorDataset) -> None:
        self._base = base
        benign_id: int | None = None
        for name, idx in base.label_mapping.items():
            if str(name).strip().upper() == "BENIGN":
                benign_id = idx
                break
        if benign_id is None:
            raise ValueError(
                "No label named 'BENIGN' found in dataset.label_mapping; cannot build binary targets."
            )
        self._benign_id = benign_id

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x, y = self._base[index]
        y_i = int(y.item())
        y_bin = 0.0 if y_i == self._benign_id else 1.0
        return x, torch.tensor(y_bin, dtype=torch.float32)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)
    env_dir = os.getenv("TRAINING_DATA_DIR", "").strip()
    data_dir = Path(env_dir) if env_dir else (project_root / "data")
    data_dir = data_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(
            f"ERROR: Training data directory does not exist: {data_dir}\n"
            "  Create it and add CICIDS-style *.csv files, or set TRAINING_DATA_DIR in .env.\n"
            "  See data/README.md for required columns.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dataset = NetworkBehaviorDataset(data_dir)
    if len(dataset) == 0:
        print(
            f"ERROR: No training windows built from {data_dir} (need enough rows for window size 5).\n"
            "  Add more CSV rows or verify columns: Protocol, Destination Port, "
            "Flow Duration, Flow Packets/s, Label (with BENIGN).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    train_data: Dataset[tuple[Tensor, Tensor]] = _BinaryBenignAttackDataset(dataset)

    loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    tok = dataset.tokenizer
    if not isinstance(tok, NSSTokenizer):
        raise TypeError("Expected NSSTokenizer on NetworkBehaviorDataset for state_dict export.")
    n_proto = tok.protocol_vocab_size
    n_port = tok.port_vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeBERT(
        protocol_vocab_size=n_proto,
        port_vocab_size=n_port,
    ).to(device)

    train_model(model, loader, epochs=2, lr=1e-4)

    quantized = quantize_model(model)

    bundle = {
        "model": quantized,
        "tokenizer_state": tok.state_dict(),
        "scaler": dataset.scaler_,
    }
    out_path = Path(__file__).resolve().parent / "edge_bert_quantized.pt"
    torch.save(bundle, out_path)

    model_version = f"edge-bert-v2-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    try:
        with get_db_session() as session:
            session.execute(text("UPDATE model_registry SET is_active = FALSE WHERE is_active = TRUE"))
            session.execute(
                text(
                    """
                    INSERT INTO model_registry (
                        model_version,
                        artifact_path,
                        tokenizer_path,
                        scaler_path,
                        trained_on_range,
                        metrics_json,
                        is_active
                    ) VALUES (
                        :model_version,
                        :artifact_path,
                        :tokenizer_path,
                        :scaler_path,
                        :trained_on_range,
                        :metrics_json,
                        :is_active
                    )
                    """
                ),
                {
                    "model_version": model_version,
                    "artifact_path": str(out_path),
                    "tokenizer_path": str(out_path),
                    "scaler_path": str(out_path),
                    "trained_on_range": "local-data",
                    "metrics_json": None,
                    "is_active": True,
                },
            )
            session.commit()
        print(f"Model registered in DB as active: {model_version}")
    except Exception as exc:
        print(f"Warning: model trained and saved, but DB registration failed: {exc}")


if __name__ == "__main__":
    main()
