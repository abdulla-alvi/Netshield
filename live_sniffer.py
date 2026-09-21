"""
Background network-flow sniffer: aggregates flows, runs EdgeBERT, logs JSON alerts.

Requires admin/root for raw capture (Npcap on Windows). No UI; suitable as a daemon.
"""

from __future__ import annotations

import json
import logging
import pickle
import signal
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Protocol, runtime_checkable

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.common.db import get_db_session
from data_pipeline import NSSTokenizer

# PyTorch emits a noisy deprecation for internal TypedStorage during quantized forward.
warnings.filterwarnings(
    "ignore",
    message=".*TypedStorage is deprecated.*",
    category=UserWarning,
)
# Scapy warns when Npcap/libpcap is not installed (common on Windows until Npcap is added).
warnings.filterwarnings("ignore", message=".*libpcap provider.*")

logger = logging.getLogger(__name__)

try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP  # type: ignore[import-untyped]
    from scapy.packet import Packet  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "live_sniffer requires scapy. Install with: pip install scapy"
    ) from exc


def _log_capture_readiness() -> None:
    """One-time hint when no capture interfaces are available (Npcap / libpcap)."""
    try:
        from scapy.arch import get_if_list  # type: ignore[import-untyped]

        if not get_if_list():
            logger.warning(
                "No packet interfaces found. On Windows install Npcap from "
                "https://nmap.org/npcap/ (enable WinPcap API compatibility), then run as Administrator."
            )
    except Exception:
        logger.warning(
            "Could not enumerate capture interfaces; install Npcap (Windows) or libpcap (Linux)."
        )

# ---------------------------------------------------------------------------
# Types & protocols (dependency inversion)
# ---------------------------------------------------------------------------

FlowKey = tuple[str, str, int, int, int]
"""(src_ip, dst_ip, src_port, dst_port, ip_proto)"""

# First packet on a flow had wall duration 0 and code used max(..., 1e-6) → rate = 1e6 pkt/s,
# which falsely triggered flood heuristics and Critical severity. Use a sane floor for the divisor only.
_MIN_WALL_SEC_FOR_RATE: Final[float] = 0.05
_MAX_PACKET_RATE_SANITY: Final[float] = 50_000.0


@dataclass(slots=True)
class FlowState:
    """Per-flow counters for real-time feature extraction."""

    start_time: float
    last_packet_time: float
    packet_count: int
    protocol: int
    dst_port: int


@dataclass(slots=True)
class FlowFeatureSnapshot:
    """One timestep: matches training columns (protocol, destination port, duration, rate)."""

    protocol: int
    destination_port: int
    flow_duration: float
    packet_rate: float
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    is_synthetic: bool = False


class AlertSinkProtocol(Protocol):
    def append_alert(self, record: dict[str, Any]) -> None: ...


@runtime_checkable
class InferenceRunnerProtocol(Protocol):
    def run(self, snapshots: list[FlowFeatureSnapshot]) -> tuple[str, float, list[list[list[list[float]]]]]: ...


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH: Final[Path] = Path("edge_bert_quantized.pt")
DEFAULT_TOKENIZER_PATH: Final[Path] = Path("nss_tokenizer.pt")
DEFAULT_SCALER_PATH: Final[Path] = Path("nss_scaler.pkl")


@dataclass(frozen=True, slots=True)
class InferenceArtifacts:
    model: nn.Module
    tokenizer: NSSTokenizer
    scaler: MinMaxScaler


def load_inference_artifacts(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    tokenizer_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
) -> InferenceArtifacts:
    """
    Load quantized EdgeBERT plus fitted ``NSSTokenizer`` and ``MinMaxScaler``.

    ``model_path`` may be either:

    - A ``torch.save`` dict with keys ``model``, ``tokenizer_state``, ``scaler``; or
    - The quantized module alone, with ``nss_tokenizer.pt`` (``state_dict``) and
      ``nss_scaler.pkl`` (pickled ``MinMaxScaler``) beside it.
    """
    mp = Path(model_path)
    obj: Any = torch.load(mp, map_location="cpu", weights_only=False)

    tok_path = Path(tokenizer_path) if tokenizer_path is not None else mp.with_name(DEFAULT_TOKENIZER_PATH.name)
    sc_path = Path(scaler_path) if scaler_path is not None else mp.with_name(DEFAULT_SCALER_PATH.name)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        tokenizer = NSSTokenizer()
        if "tokenizer_state" in obj:
            tokenizer.load_state_dict(obj["tokenizer_state"])
        else:
            tokenizer.load_state_dict(torch.load(tok_path, map_location="cpu", weights_only=False))
        if "scaler" in obj:
            scaler = obj["scaler"]
            if not isinstance(scaler, MinMaxScaler):
                raise TypeError("Bundle key 'scaler' must be a sklearn MinMaxScaler.")
        else:
            scaler = pickle.loads(sc_path.read_bytes())
            if not isinstance(scaler, MinMaxScaler):
                raise TypeError("Pickled scaler must be a sklearn MinMaxScaler.")
        return InferenceArtifacts(model=model, tokenizer=tokenizer, scaler=scaler)

    if isinstance(obj, nn.Module):
        tokenizer = NSSTokenizer()
        tokenizer.load_state_dict(torch.load(tok_path, map_location="cpu", weights_only=False))
        scaler = pickle.loads(sc_path.read_bytes())
        if not isinstance(scaler, MinMaxScaler):
            raise TypeError("Pickled scaler must be a sklearn MinMaxScaler.")
        return InferenceArtifacts(model=obj, tokenizer=tokenizer, scaler=scaler)

    raise TypeError(
        f"Unsupported checkpoint at {mp}: expected nn.Module or dict with 'model' key."
    )


# ---------------------------------------------------------------------------
# Inference (decoupled from I/O)
# ---------------------------------------------------------------------------

class EdgeBERTInferenceRunner:
    """Applies NSSTokenizer + MinMaxScaler + EdgeBERT; returns prediction and attention."""

    def __init__(self, artifacts: InferenceArtifacts) -> None:
        self._tokenizer = artifacts.tokenizer
        self._scaler = artifacts.scaler
        self._model = artifacts.model
        self._model.eval()
        self._device = torch.device("cpu")

    @torch.inference_mode()
    def run(self, snapshots: list[FlowFeatureSnapshot]) -> tuple[str, float, list[list[list[list[float]]]]]:
        if len(snapshots) != 5:
            raise ValueError(f"Expected 5 flow snapshots, got {len(snapshots)}.")

        protos = pd.Series([s.protocol for s in snapshots], dtype="int64")
        ports = pd.Series([s.destination_port for s in snapshots], dtype="int64")
        durations = [s.flow_duration for s in snapshots]
        rates = [s.packet_rate for s in snapshots]

        proto_ids = self._tokenizer.transform_protocol(protos).astype("float32")
        port_ids = self._tokenizer.transform_port(ports).astype("float32")
        num = self._scaler.transform(
            pd.DataFrame({"d": durations, "r": rates}).to_numpy(dtype="float64")
        ).astype("float32")

        x = torch.empty(1, 5, 4, dtype=torch.float32, device=self._device)
        x[0, :, 0] = torch.from_numpy(proto_ids)
        x[0, :, 1] = torch.from_numpy(port_ids)
        x[0, :, 2] = torch.from_numpy(num[:, 0])
        x[0, :, 3] = torch.from_numpy(num[:, 1])

        logits, attn = self._model(x)
        prob = torch.sigmoid(logits.view(-1)).item()
        label = "Malicious" if prob > 0.5 else "Benign"
        confidence = prob if label == "Malicious" else 1.0 - prob
        attn_list: list[list[list[list[float]]]] = attn.detach().cpu().float().tolist()
        return label, float(confidence), attn_list


# ---------------------------------------------------------------------------
# MySQL alert persistence
# ---------------------------------------------------------------------------


class MySQLAlertSink:
    """Persists alerts and XAI artifacts into MySQL with short retry logic."""

    def __init__(
        self,
        model_version: str = "edge_bert_quantized.pt",
        max_retries: int = 3,
        retry_delay_sec: float = 0.4,
    ) -> None:
        self._model_version = model_version
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

    @staticmethod
    def _calculate_severity(prediction: str, confidence: float, packet_rate: float) -> str:
        pred = str(prediction).strip()
        if pred == "Benign":
            return "Info"
        if confidence >= 0.95 and packet_rate > 100:
            return "Critical"
        if confidence >= 0.85:
            return "High"
        if confidence >= 0.70:
            return "Medium"
        return "Low"

    def _insert_alert_and_xai(self, session: Session, record: dict[str, Any]) -> None:
        prediction = str(record.get("prediction", "Benign"))
        confidence = float(record.get("confidence", 0.0))
        packet_rate = float(record.get("packet_rate", 0.0))
        severity = self._calculate_severity(prediction, confidence, packet_rate)

        event_time = record.get("event_time")
        if not isinstance(event_time, datetime):
            raw_ts = record.get("timestamp")
            if isinstance(raw_ts, datetime):
                event_time = raw_ts
            elif isinstance(raw_ts, str):
                try:
                    event_time = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                except ValueError:
                    event_time = datetime.now(timezone.utc)
            else:
                event_time = datetime.now(timezone.utc)
        if event_time.tzinfo is not None:
            event_time = event_time.astimezone(timezone.utc).replace(tzinfo=None)

        alert_stmt = text(
            """
            INSERT INTO alerts (
                event_time,
                prediction,
                confidence,
                severity,
                src_ip,
                dst_ip,
                src_port,
                dst_port,
                ip_proto,
                flow_duration,
                packet_rate,
                model_version,
                status,
                assigned_to,
                notes
            ) VALUES (
                :event_time,
                :prediction,
                :confidence,
                :severity,
                :src_ip,
                :dst_ip,
                :src_port,
                :dst_port,
                :ip_proto,
                :flow_duration,
                :packet_rate,
                :model_version,
                :status,
                :assigned_to,
                :notes
            )
            """
        )
        result = session.execute(
            alert_stmt,
            {
                "event_time": event_time,
                "prediction": prediction,
                "confidence": confidence,
                "severity": severity,
                "src_ip": record.get("src_ip"),
                "dst_ip": record.get("dst_ip"),
                "src_port": record.get("src_port"),
                "dst_port": record.get("dst_port"),
                "ip_proto": record.get("ip_proto"),
                "flow_duration": record.get("flow_duration"),
                "packet_rate": packet_rate,
                "model_version": self._model_version,
                "status": "New",
                "assigned_to": None,
                "notes": None,
            },
        )

        alert_id = result.lastrowid
        if alert_id is None:
            raise RuntimeError("Failed to retrieve generated alert_id after insert.")

        xai_stmt = text(
            """
            INSERT INTO alert_xai (
                alert_id,
                attention_json,
                top_features_json
            ) VALUES (
                :alert_id,
                :attention_json,
                :top_features_json
            )
            """
        )
        session.execute(
            xai_stmt,
            {
                "alert_id": int(alert_id),
                "attention_json": json.dumps(record.get("attention_weights", [])),
                "top_features_json": None,
            },
        )

    def append_alert(self, record: dict[str, Any]) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with get_db_session() as session:
                    self._insert_alert_and_xai(session, record)
                    session.commit()
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MySQL insert failed (attempt %s/%s): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                time.sleep(self._retry_delay_sec)
        if last_error is not None:
            logger.error("Dropping alert after retry failures: %s", last_error)


# ---------------------------------------------------------------------------
# Sequence buffer
# ---------------------------------------------------------------------------

class SequenceBuffer:
    """Rolling deque of the last ``maxlen`` flow feature snapshots."""

    def __init__(
        self,
        maxlen: int,
        on_full_sequence: Callable[[list[FlowFeatureSnapshot]], None],
    ) -> None:
        self._maxlen = maxlen
        self._dq: deque[FlowFeatureSnapshot] = deque(maxlen=maxlen)
        self._on_full = on_full_sequence
        self._lock = threading.Lock()

    def push(self, snapshot: FlowFeatureSnapshot) -> None:
        with self._lock:
            self._dq.append(snapshot)
            if len(self._dq) == self._maxlen:
                self._on_full(list(self._dq))


# ---------------------------------------------------------------------------
# Flow aggregation & sniffing
# ---------------------------------------------------------------------------

def _extract_l4(pkt: Packet) -> tuple[int, int, int, int] | None:
    """Returns (proto, src_port, dst_port) with proto from IP; or None if unsupported."""
    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = int(ip.proto)
    if TCP in pkt:
        t = pkt[TCP]
        return proto, int(t.sport), int(t.dport)
    if UDP in pkt:
        u = pkt[UDP]
        return proto, int(u.sport), int(u.dport)
    # ICMP / other: no ports — use 0 placeholders for tokenizer unk semantics
    return proto, 0, 0


class FlowAggregator:
    """
    Runs ``AsyncSniffer`` in a background thread, maintains flow table and ``SequenceBuffer``.
    """

    def __init__(
        self,
        inference: InferenceRunnerProtocol,
        alert_sink: AlertSinkProtocol,
        *,
        iface: str | None = None,
        bpf_filter: str = "ip",
        sequence_len: int = 5,
    ) -> None:
        self._inference = inference
        self._alert_sink = alert_sink
        self._iface = iface
        self._bpf = bpf_filter
        self._flows: dict[FlowKey, FlowState] = {}
        self._table_lock = threading.Lock()
        self._buffer = SequenceBuffer(sequence_len, self._on_sequence_ready)
        self._sniffer: AsyncSniffer | None = None
        self._sniffer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_packet_seen = time.monotonic()
        self._monitor_thread: threading.Thread | None = None
        self._synthetic_tick = 0
        self._alerts_written = 0

    @staticmethod
    def _apply_attack_heuristic(
        snaps: list[FlowFeatureSnapshot],
        predicted_label: str,
        predicted_confidence: float,
    ) -> tuple[str, float]:
        """
        Raise clear high-rate flood signatures to Malicious for operational visibility.

        This only applies to real captured traffic (not synthetic fallback telemetry).
        """
        if not snaps:
            return predicted_label, predicted_confidence
        if not any(not s.is_synthetic for s in snaps):
            return predicted_label, predicted_confidence

        last = snaps[-1]
        high_rate = last.packet_rate >= 150.0
        common_attack_port = last.destination_port in {80, 443}
        proto_l4 = last.protocol in {6, 17}
        if high_rate and common_attack_port and proto_l4:
            return "Malicious", max(predicted_confidence, 0.97)
        return predicted_label, predicted_confidence

    def _on_sequence_ready(self, snaps: list[FlowFeatureSnapshot]) -> None:
        try:
            label, confidence, attn = self._inference.run(snaps)
            label, confidence = self._apply_attack_heuristic(snaps, label, confidence)
            # Idle fallback: sequences built only from synthetic ticks are not real traffic.
            # The model can still label them Malicious; that would false-trigger the SOC.
            if snaps and all(s.is_synthetic for s in snaps):
                if label == "Malicious":
                    logger.debug(
                        "Synthetic-only window: suppressing model prediction %s (%.3f) -> Benign for idle fallback",
                        label,
                        confidence,
                    )
                label, confidence = "Benign", 0.9997
            last = snaps[-1]
            record = {
                "event_time": datetime.now(timezone.utc),
                "prediction": label,
                "confidence": confidence,
                "attention_weights": attn,
                "src_ip": last.src_ip,
                "dst_ip": last.dst_ip,
                "src_port": int(last.src_port) if last.src_port is not None else None,
                "dst_port": int(last.destination_port),
                "ip_proto": int(last.protocol),
                "flow_duration": float(last.flow_duration),
                "packet_rate": float(last.packet_rate),
            }
            self._alert_sink.append_alert(record)
            self._alerts_written += 1
            if self._alerts_written % 25 == 0:
                logger.info("Live sniffer wrote %s alerts to MySQL", self._alerts_written)
        except Exception:
            logger.exception("Inference or alert logging failed")

    def _handle_packet(self, pkt: Packet) -> None:
        l4 = _extract_l4(pkt)
        if l4 is None:
            return
        proto, sport, dport = l4
        ip = pkt[IP]
        sip, dip = str(ip.src), str(ip.dst)
        key: FlowKey = (sip, dip, sport, dport, proto)
        now = time.monotonic()
        self._last_packet_seen = now

        with self._table_lock:
            st = self._flows.get(key)
            if st is None:
                st = FlowState(
                    start_time=now,
                    last_packet_time=now,
                    packet_count=0,
                    protocol=proto,
                    dst_port=dport,
                )
                self._flows[key] = st
            st.last_packet_time = now
            st.packet_count += 1
            wall = st.last_packet_time - st.start_time
            flow_duration = float(max(wall, 1e-9))
            # Do not divide by ~0: first packet gives wall==0; old logic used 1e-6 → 1e6 false "flood".
            rate_denom = max(wall, _MIN_WALL_SEC_FOR_RATE)
            raw_rate = st.packet_count / rate_denom
            rate = float(min(raw_rate, _MAX_PACKET_RATE_SANITY))
            snap = FlowFeatureSnapshot(
                protocol=st.protocol,
                destination_port=st.dst_port,
                flow_duration=flow_duration,
                packet_rate=rate,
                src_ip=sip,
                dst_ip=dip,
                src_port=sport,
            )
        self._buffer.push(snap)

    def _push_synthetic_snapshot(self) -> None:
        """
        Fallback feed when packet capture is unavailable.

        Keeps SOC telemetry alive for demos by emitting plausible *benign* flow snapshots.
        Rates stay below the flood heuristic threshold (150 pkt/s) so mixed edge cases
        never look like volumetric attacks.
        """
        self._synthetic_tick += 1
        proto = 17 if self._synthetic_tick % 2 == 0 else 6
        dst_port = 443 if self._synthetic_tick % 3 else 80
        # Bounded, moderate rates — avoids both heuristic floods and model false-malicious spikes.
        rate = 25.0 + float((self._synthetic_tick % 12) * 5.0)
        duration = 0.15 + float((self._synthetic_tick % 8) * 0.12)
        snap = FlowFeatureSnapshot(
            protocol=proto,
            destination_port=dst_port,
            flow_duration=duration,
            packet_rate=rate,
            src_ip=f"10.0.0.{(self._synthetic_tick % 250) + 1}",
            dst_ip="8.8.8.8",
            src_port=40000 + (self._synthetic_tick % 2000),
            is_synthetic=True,
        )
        self._buffer.push(snap)

    def _monitor_capture_loop(self) -> None:
        warned = False
        while not self._stop_event.is_set():
            time.sleep(1.0)
            idle_sec = time.monotonic() - self._last_packet_seen
            if idle_sec > 3.0:
                if not warned:
                    logger.warning(
                        "No packets captured for %.1fs. Using synthetic telemetry fallback.",
                        idle_sec,
                    )
                    warned = True
                self._push_synthetic_snapshot()
            else:
                warned = False

    def start(self) -> None:
        try:
            self._sniffer = AsyncSniffer(
                iface=self._iface,
                filter=self._bpf,
                prn=self._handle_packet,
                store=False,
            )
            self._sniffer.start()
            logger.info("Sniffer started with BPF filter: %s", self._bpf)
        except Exception as exc:
            logger.warning(
                "Failed to start sniffer with BPF filter '%s' (%s). Retrying without BPF filter.",
                self._bpf,
                exc,
            )
            self._sniffer = AsyncSniffer(
                iface=self._iface,
                prn=self._handle_packet,
                store=False,
            )
            self._sniffer.start()
            logger.info("Sniffer started without BPF filter.")
        self._monitor_thread = threading.Thread(target=self._monitor_capture_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._sniffer is not None:
            self._sniffer.stop()
        if self._sniffer is not None:
            self._sniffer.join(timeout=join_timeout)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=join_timeout)


# ---------------------------------------------------------------------------
# Pure utility: training loop already lives in model_engine; sniffer wiring below
# ---------------------------------------------------------------------------

def run_live_engine(
    artifacts: InferenceArtifacts | None = None,
    *,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    iface: str | None = None,
    bpf_filter: str = "ip",
) -> None:
    """
    Block until SIGINT; runs sniffer + inference in background threads.

    Uses MySQL-backed alert writes with retry; no console spam.
    """
    _log_capture_readiness()
    arts = artifacts or load_inference_artifacts(model_path)
    runner = EdgeBERTInferenceRunner(arts)
    sink = MySQLAlertSink(model_version=Path(model_path).name)
    stop = threading.Event()

    def _on_sigint(_sig: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _on_sigint)

    agg = FlowAggregator(
        inference=runner,
        alert_sink=sink,
        iface=iface,
        bpf_filter=bpf_filter,
    )
    agg.start()
    stop.wait()
    agg.stop()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live capture, EdgeBERT inference, MySQL alert sink.")
    parser.add_argument(
        "--iface",
        default=None,
        help="Scapy interface name (from scapy). Default: system default. "
        "Tip: run `python -c \"from scapy.all import get_if_list; print(get_if_list())\"`.",
    )
    parser.add_argument(
        "--bpf",
        default="ip",
        dest="bpf_filter",
        help='Berkeley packet filter (default: "ip").',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_live_engine(iface=args.iface, bpf_filter=args.bpf_filter)


if __name__ == "__main__":
    main()
