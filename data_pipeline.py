"""
CICIDS2017-oriented network behavior data pipeline.

Network-as-a-Language: protocol numbers (IANA, e.g. 6=TCP, 17=UDP) and ports are
mapped to dense semantic token IDs via NSSTokenizer; flow duration and packet
rate are MinMax-scaled. Sequences use a sliding window over flow rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

# ---------------------------------------------------------------------------
# Column names (CICIDS2017 CSV; allow overrides via NetworkBehaviorDataset)
# ---------------------------------------------------------------------------

DEFAULT_PROTOCOL_COL: Final[str] = "Protocol"
DEFAULT_PORT_COL: Final[str] = "Destination Port"
DEFAULT_FLOW_DURATION_COL: Final[str] = "Flow Duration"
DEFAULT_PACKET_RATE_COL: Final[str] = "Flow Packets/s"
DEFAULT_LABEL_COL: Final[str] = "Label"

# Convention: CSVs live under a ``data`` directory (e.g. ``/data`` on Linux servers).
DEFAULT_DATA_DIR: Final[Path] = Path("/data")

# Common IANA protocol numbers — used for stable, interpretable token ordering
_IANA_ORDER: Final[tuple[int, ...]] = (
    1,  # ICMP
    6,  # TCP
    17,  # UDP
    41,  # IPv6
    47,  # GRE
    50,  # ESP
    51,  # AH
    58,  # IPv6-ICMP
)


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Maps logical roles to CSV header names."""

    protocol: str = DEFAULT_PROTOCOL_COL
    port: str = DEFAULT_PORT_COL
    flow_duration: str = DEFAULT_FLOW_DURATION_COL
    packet_rate: str = DEFAULT_PACKET_RATE_COL
    label: str = DEFAULT_LABEL_COL


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Dependency inversion: dataset depends on tokenization abstraction."""

    def fit(self, df: pd.DataFrame, columns: ColumnMapping) -> None: ...

    def transform_protocol(self, series: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]: ...

    def transform_port(self, series: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]: ...

    @property
    def protocol_vocab_size(self) -> int: ...

    @property
    def port_vocab_size(self) -> int: ...


class NSSTokenizer:
    """
    Maps Protocol (IANA numbers) and Port to contiguous integer token IDs.

    Protocol: known IANA values get priority ordering (TCP=6, UDP=17, …); any
    other value seen in ``fit`` is assigned subsequent IDs — preserving
    "Network-as-a-Language" semantics for standard protocols.
    Port: each distinct port value in the fitted data receives a token ID.
    """

    def __init__(self) -> None:
        self._protocol_to_id: dict[int, int] = {}
        self._port_to_id: dict[int, int] = {}
        self._next_port_id: int = 0
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, columns: ColumnMapping) -> None:
        proto_col = columns.protocol
        port_col = columns.port
        if proto_col not in df.columns or port_col not in df.columns:
            missing = {c for c in (proto_col, port_col) if c not in df.columns}
            raise KeyError(f"Missing columns for NSSTokenizer: {missing}")

        protocols = pd.to_numeric(df[proto_col], errors="coerce").dropna().astype(np.int64)
        ports = pd.to_numeric(df[port_col], errors="coerce").dropna().astype(np.int64)

        self._protocol_to_id = self._build_protocol_mapping(protocols.unique())
        self._port_to_id = {}
        self._next_port_id = 0
        for p in sorted(ports.unique()):
            self._port_to_id[int(p)] = self._next_port_id
            self._next_port_id += 1

        self._fitted = True

    def _build_protocol_mapping(self, unique_protocols: Sequence[int] | np.ndarray[Any, Any]) -> dict[int, int]:
        seen: set[int] = set(int(x) for x in unique_protocols)
        ordered: list[int] = []
        for iana in _IANA_ORDER:
            if iana in seen:
                ordered.append(iana)
        for u in sorted(seen):
            if u not in ordered:
                ordered.append(u)
        return {p: idx for idx, p in enumerate(ordered)}

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("NSSTokenizer.fit must be called before transform.")

    def transform_protocol(self, series: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]:
        self._require_fitted()
        numeric = pd.to_numeric(series, errors="coerce")
        out = np.empty(len(series), dtype=np.int64)
        unk = len(self._protocol_to_id)
        for i, v in enumerate(numeric):
            if pd.isna(v):
                out[i] = unk
                continue
            pid = int(v)
            out[i] = self._protocol_to_id.get(pid, unk)
        return out

    def transform_port(self, series: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]:
        self._require_fitted()
        numeric = pd.to_numeric(series, errors="coerce")
        out = np.empty(len(series), dtype=np.int64)
        unk = len(self._port_to_id)
        for i, v in enumerate(numeric):
            if pd.isna(v):
                out[i] = unk
                continue
            pid = int(v)
            out[i] = self._port_to_id.get(pid, unk)
        return out

    @property
    def protocol_vocab_size(self) -> int:
        self._require_fitted()
        return len(self._protocol_to_id) + 1

    @property
    def port_vocab_size(self) -> int:
        self._require_fitted()
        return len(self._port_to_id) + 1

    def state_dict(self) -> dict[str, Any]:
        """Serializable tokenizer state for deployment (e.g. live inference)."""
        return {
            "protocol_to_id": dict(self._protocol_to_id),
            "port_to_id": dict(self._port_to_id),
            "next_port_id": self._next_port_id,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore tokenizer from :meth:`state_dict`."""
        self._protocol_to_id = {int(k): int(v) for k, v in state["protocol_to_id"].items()}
        self._port_to_id = {int(k): int(v) for k, v in state["port_to_id"].items()}
        self._next_port_id = int(state["next_port_id"])
        self._fitted = bool(state["fitted"])


def _load_csv_paths(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    return paths


def _concat_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def _ensure_protocol_column(df: pd.DataFrame, columns: ColumnMapping) -> pd.DataFrame:
    """Some CIC-ISCX exports omit ``Protocol``; default to TCP (IANA 6) for flow rows."""
    if columns.protocol in df.columns:
        return df
    out = df.copy()
    out[columns.protocol] = np.int64(6)
    return out


def _clean_numeric_frame(
    df: pd.DataFrame,
    columns: ColumnMapping,
) -> pd.DataFrame:
    """Drop rows with missing required fields; coerce numerics. Keeps ``Label`` if present."""
    need = [
        columns.protocol,
        columns.port,
        columns.flow_duration,
        columns.packet_rate,
    ]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Required column missing: {c!r}")
    out = df.copy()
    for c in need:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=need).reset_index(drop=True)
    return out


def _encode_labels(series: pd.Series) -> tuple[np.ndarray[Any, np.dtype[np.int64]], dict[str, int]]:
    uniques = sorted(series.astype(str).unique())
    str_to_id: dict[str, int] = {name: i for i, name in enumerate(uniques)}
    ids = np.array([str_to_id[str(v)] for v in series], dtype=np.int64)
    return ids, str_to_id


def _sliding_window_indices(n_rows: int, window_size: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    if n_rows < window_size:
        return np.empty((0, window_size), dtype=np.int64)
    # Stride-1 windows: [0..4], [1..5], ...
    starts = np.arange(0, n_rows - window_size + 1, dtype=np.int64)
    idx = starts[:, None] + np.arange(window_size, dtype=np.int64)
    return idx


class NetworkBehaviorDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
    """
    Flow-level CICIDS-style CSVs under ``data_dir`` (all ``*.csv`` concatenated).

    Each item is a behavioral sequence: shape ``(window_size, n_features)`` with
    ``n_features = 4``: ``[protocol_id, port_id, flow_duration_scaled, packet_rate_scaled]``.
    Use ``torch.utils.data.DataLoader`` with ``batch_size=B`` to obtain
    ``(B, window_size, n_features)``.

    Missing values: rows with NaN in required numeric columns are dropped.
    """

    WINDOW_SIZE: Final[int] = 5
    N_FEATURES: Final[int] = 4

    def __init__(
        self,
        data_dir: str | Path,
        columns: ColumnMapping | None = None,
        tokenizer: TokenizerProtocol | None = None,
        scaler: MinMaxScaler | None = None,
        fit_tokenizer: bool = True,
        fit_scaler: bool = True,
        include_label: bool = True,
    ) -> None:
        self._columns: ColumnMapping = columns or ColumnMapping()
        self._tokenizer: TokenizerProtocol = tokenizer if tokenizer is not None else NSSTokenizer()

        data_path = Path(data_dir)
        df_raw = _ensure_protocol_column(_concat_csvs(_load_csv_paths(data_path)), self._columns)
        df = _clean_numeric_frame(df_raw, self._columns)

        if fit_tokenizer:
            self._tokenizer.fit(df, self._columns)

        proto_ids = self._tokenizer.transform_protocol(df[self._columns.protocol])
        port_ids = self._tokenizer.transform_port(df[self._columns.port])

        dur = df[self._columns.flow_duration].to_numpy(dtype=np.float64).reshape(-1, 1)
        rate = df[self._columns.packet_rate].to_numpy(dtype=np.float64).reshape(-1, 1)
        num_stack = np.hstack([dur, rate])

        self._scaler: MinMaxScaler = scaler if scaler is not None else MinMaxScaler()
        if fit_scaler:
            self._scaler.fit(num_stack)
        elif scaler is None:
            raise ValueError("When fit_scaler is False, pass a pre-fitted MinMaxScaler via scaler=.")
        scaled = self._scaler.transform(num_stack).astype(np.float32)

        self._proto = proto_ids.astype(np.float32)
        self._port = port_ids.astype(np.float32)
        self._scaled_num = scaled
        self._window_idx = _sliding_window_indices(len(df), self.WINDOW_SIZE)

        self._labels: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self._label_map: dict[str, int] = {}
        if include_label and self._columns.label in df.columns:
            self._labels, self._label_map = _encode_labels(df[self._columns.label])

    def __len__(self) -> int:
        return int(self._window_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rows = self._window_idx[index]
        p = self._proto[rows]
        po = self._port[rows]
        sn = self._scaled_num[rows]
        feat = np.stack([p, po, sn[:, 0], sn[:, 1]], axis=1).astype(np.float32)
        x = torch.from_numpy(feat)

        if self._labels is not None:
            y = torch.tensor(self._labels[rows[-1]], dtype=torch.long)
        else:
            y = torch.tensor(-1, dtype=torch.long)
        return x, y

    @property
    def feature_dim(self) -> int:
        return self.N_FEATURES

    @property
    def label_mapping(self) -> dict[str, int]:
        return dict(self._label_map)

    @property
    def scaler_(self) -> MinMaxScaler:
        return self._scaler

    @property
    def tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer


__all__ = [
    "ColumnMapping",
    "DEFAULT_DATA_DIR",
    "NSSTokenizer",
    "NetworkBehaviorDataset",
    "TokenizerProtocol",
]
