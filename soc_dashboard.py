"""
NetShield AI — SOC Command Center dashboard.
Reads exclusively from MySQL via app/common/db.  No JSON file dependency.
"""

from __future__ import annotations

import html
import inspect
import ipaddress
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from sqlalchemy import text

from app.common.db import get_db_session

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PLOTLY_DARK = "plotly_dark"

SEVERITY_COLOR: dict[str, str] = {
    "Critical": "#ff2244",
    "High":     "#ff7700",
    "Medium":   "#f5c518",
    "Low":      "#00aaff",
    "Info":     "#22cc88",
}

PROTOCOL_MAP: dict[int, str] = {1: "ICMP", 6: "TCP", 17: "UDP", 41: "IPv6", 47: "GRE"}

ALERT_STATUSES = ["New", "InReview", "Escalated", "Resolved", "FalsePositive"]

# Newest rows for map, radar, mixed analyst queue (not the attack banner — see ATTACK_ALERT_WINDOW_SEC).
# Heavy sniffing can push older rows out of this sliding window.
DASHBOARD_ALERT_FETCH_LIMIT = 250

# Rows shown in the analyst table and offered in the status dropdown.
ANALYST_QUEUE_DISPLAY = 100

# Arcs + scatter points on the threat map (one per row). Cap keeps the map readable.
MAP_PROJECTION_MAX = 25

# Attack banner + red alarm + header "Active Threats" use malicious rows whose
# event_time is within this many seconds (UTC). Independent of the mixed 250-row fetch.
ATTACK_ALERT_WINDOW_SEC = 90  # 1.5 minutes

# Dedicated malicious-only table (SQL filter); still visible after the attack UI clears.
MALICIOUS_ONLY_TABLE_ROWS = 100


def _section_box():
    """Visual grouping for a section; avoids orphan HTML divs that Streamlit cannot nest."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


_CONTAINER_KEY_PARAM = "key" in inspect.signature(st.container).parameters


@contextmanager
def _keyed_slot(key: str) -> Iterator[None]:
    """
    Stable Streamlit container identity across reruns (helps avoid duplicate HTML / charts
    when auto-refresh or threat UI toggles confuse reconciliation).
    """
    if _CONTAINER_KEY_PARAM:
        with st.container(key=key):
            yield
    else:
        yield


# ──────────────────────────────────────────────────────────────────────────────
# CSS injection
# ──────────────────────────────────────────────────────────────────────────────

def _inject_styles(threat_active: bool) -> None:
    alarm_css = """
        @keyframes soc-alarm-bg {
            0%, 100% {
                background-color: #120505 !important;
                box-shadow: inset 0 0 100px rgba(180, 0, 0, 0.22);
            }
            50% {
                background-color: #240808 !important;
                box-shadow: inset 0 0 160px rgba(255, 60, 60, 0.42);
            }
        }
        @keyframes soc-alarm-vignette {
            0%, 100% { opacity: 0.35; }
            50% { opacity: 0.65; }
        }
        [data-testid="stAppViewContainer"] {
            animation: soc-alarm-bg 1.1s ease-in-out infinite !important;
            position: relative;
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            pointer-events: none;
            position: fixed;
            inset: 0;
            z-index: 9998;
            background: radial-gradient(ellipse at center, transparent 40%, rgba(120, 0, 0, 0.55) 100%);
            animation: soc-alarm-vignette 1.1s ease-in-out infinite;
        }
        [data-testid="stHeader"] {
            background-color: rgba(40, 5, 5, 0.92) !important;
            border-bottom: 1px solid #ff224488 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ff6b7a !important;
            text-shadow: 0 0 12px rgba(255, 80, 100, 0.35);
        }
        div[data-testid="stMetricLabel"] {
            color: #fca5a5 !important;
        }
        h1, h2, h3 {
            color: #fecaca !important;
            text-shadow: 0 0 20px rgba(255, 0, 0, 0.25);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(255, 80, 90, 0.45) !important;
            background: rgba(35, 8, 10, 0.88) !important;
            box-shadow: 0 0 24px rgba(255, 0, 0, 0.12);
        }
        .hacker-terminal {
            border-color: #ff224466 !important;
            color: #ff8585 !important;
        }
    """ if threat_active else ""

    st.markdown(
        f"""
        <style>
        /* ── true-black base ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: #050505 !important;
            color: #e2e8f0;
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(5,5,5,0.95) !important;
            border-bottom: 1px solid #00f3ff33;
        }}
        /* hide streamlit footer */
        footer {{ visibility: hidden; }}

        /* ── Analyst queue: HTML table (Streamlit dataframe grid is often unreadable on dark) ── */
        .soc-queue-wrap {{
            max-height: 520px;
            overflow: auto;
            border: 1px solid rgba(0, 243, 255, 0.22);
            border-radius: 10px;
            background: rgba(12, 14, 20, 0.97);
        }}
        .soc-queue-wrap::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        .soc-queue-wrap::-webkit-scrollbar-track {{
            background: rgba(15, 15, 22, 0.9);
            border-radius: 4px;
        }}
        .soc-queue-wrap::-webkit-scrollbar-thumb {{
            background: #3d4f60;
            border-radius: 4px;
        }}
        table.soc-queue-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        table.soc-queue-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: rgba(22, 30, 42, 0.98);
            color: #7dd3fc;
            font-weight: 700;
            text-align: left;
            padding: 11px 14px;
            border-bottom: 1px solid rgba(0, 243, 255, 0.35);
        }}
        table.soc-queue-table td {{
            color: #f1f5f9;
            padding: 9px 14px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }}
        table.soc-queue-table tbody tr:hover td {{
            background: rgba(0, 243, 255, 0.07);
        }}
        table.soc-queue-table td.soc-pred-mal {{
            color: #fca5a5 !important;
            font-weight: 600;
        }}
        table.soc-queue-table td.soc-pred-benign {{
            color: #5eead4 !important;
        }}

        /* ── Selectbox / inputs (Base Web) — dark glass even if theme lags ── */
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {{
            color: #cbd5e1 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background-color: rgba(22, 24, 32, 0.95) !important;
            border-color: rgba(0, 243, 255, 0.28) !important;
            color: #e2e8f0 !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"] [aria-selected="true"],
        [data-testid="stSelectbox"] [data-baseweb="select"] span {{
            color: #e2e8f0 !important;
        }}
        /* Dropdown list popover */
        ul[role="listbox"],
        div[data-baseweb="popover"] ul {{
            background-color: rgba(22, 24, 32, 0.98) !important;
            border: 1px solid rgba(0, 243, 255, 0.25) !important;
        }}
        li[role="option"] {{
            background-color: transparent !important;
            color: #e2e8f0 !important;
        }}
        li[role="option"]:hover {{
            background-color: rgba(0, 243, 255, 0.12) !important;
        }}

        /* Streamlit bordered sections (replaces orphan glass-card divs) */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(20, 20, 25, 0.82) !important;
            border: 1px solid #00f3ff44 !important;
            border-radius: 12px !important;
            padding: 14px 18px 18px 18px !important;
            margin-bottom: 14px !important;
        }}

        /* ── metrics ── */
        div[data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #00f3ff !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: #94a3b8 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .metric-threat div[data-testid="stMetricValue"] {{
            color: #ff4466 !important;
        }}
        div[data-testid="stMetricDelta"] {{
            font-size: 0.75rem !important;
        }}

        /* ── hacker terminal ── */
        .hacker-terminal {{
            background: #000000;
            border: 1px solid #00ff4155;
            border-radius: 8px;
            padding: 14px 18px;
            font-family: 'Courier New', 'Lucida Console', monospace;
            font-size: 0.78rem;
            color: #00ff41;
            line-height: 1.7;
            max-height: 200px;
            overflow-y: auto;
        }}

        /* ── severity badges ── */
        .badge-critical {{ color:#ff2244; font-weight:700; }}
        .badge-high     {{ color:#ff7700; font-weight:700; }}
        .badge-medium   {{ color:#f5c518; font-weight:700; }}
        .badge-low      {{ color:#00aaff; font-weight:700; }}
        .badge-info     {{ color:#22cc88; font-weight:700; }}

        /* Section headings: no side bar — clean typography only */
        h2, h3 {{
            border-left: none !important;
            padding-left: 0 !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.35rem !important;
            color: #eaf2fb !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
        }}

        /* Alert / info boxes: readable on dark (Streamlit default blue is low-contrast) */
        div[data-testid="stAlert"],
        div.stAlert {{
            background-color: rgba(15, 23, 42, 0.92) !important;
            border: 1px solid rgba(0, 243, 255, 0.35) !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div,
        div.stAlert p,
        div.stAlert div {{
            color: #e2e8f0 !important;
        }}

        /* Critical attack banner (custom HTML, not st.error chrome) */
        @keyframes soc-banner-pulse {{
            0%, 100% {{
                opacity: 1;
                box-shadow: 0 0 0 0 rgba(255, 40, 40, 0.55), inset 0 0 40px rgba(255, 0, 0, 0.15);
            }}
            50% {{
                opacity: 1;
                box-shadow: 0 0 28px 4px rgba(255, 80, 80, 0.75), inset 0 0 70px rgba(255, 60, 60, 0.28);
            }}
        }}
        @keyframes soc-banner-text {{
            0%, 100% {{ color: #fff5f5; text-shadow: 0 0 8px rgba(255,100,100,0.9); }}
            50% {{ color: #ffcccc; text-shadow: 0 0 20px rgba(255,50,50,1); }}
        }}
        .soc-critical-banner {{
            position: relative;
            z-index: 10000;
            margin: 0 0 18px 0;
            padding: 16px 22px;
            border-radius: 10px;
            border: 2px solid #ff3344;
            background: linear-gradient(135deg, #3d0a0f 0%, #1a0305 50%, #2d080c 100%);
            animation: soc-banner-pulse 0.95s ease-in-out infinite;
        }}
        .soc-critical-banner .soc-cb-kicker {{
            font-size: 0.72rem;
            letter-spacing: 0.35em;
            color: #ff6b6b;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .soc-critical-banner .soc-cb-title {{
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            margin: 0 0 8px 0;
            animation: soc-banner-text 0.95s ease-in-out infinite;
        }}
        .soc-critical-banner .soc-cb-body {{
            font-size: 0.98rem;
            color: #fecaca;
            line-height: 1.45;
            margin: 0;
        }}
        .soc-critical-banner .soc-cb-count {{
            display: inline-block;
            margin-top: 10px;
            padding: 6px 14px;
            border-radius: 6px;
            background: rgba(255, 0, 0, 0.25);
            border: 1px solid #ff5555;
            font-weight: 800;
            color: #fff;
            font-size: 0.9rem;
        }}

        /* Analyst workflow — save button: low-key, readable (no strong glow) */
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stButton"] button[data-testid="baseButton-primary"] {{
            background: rgba(18, 24, 32, 0.88) !important;
            color: #e2e8f0 !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            letter-spacing: 0.02em;
            border: 1px solid rgba(100, 116, 139, 0.55) !important;
            box-shadow: none !important;
            padding: 0.48rem 0.85rem !important;
            min-height: 2.45rem;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover,
        div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover {{
            background: rgba(28, 38, 48, 0.95) !important;
            border-color: rgba(34, 211, 238, 0.35) !important;
            color: #f8fafc !important;
        }}

        /* ── full-app alarm (threat_active) ── */
        {alarm_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MySQL data fetchers (cached 2 s TTL)
# ──────────────────────────────────────────────────────────────────────────────

def _rows_to_alerts_df(rows: list[Any]) -> pd.DataFrame:
    """Normalize SQL rows into the standard alerts DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "id", "event_time", "prediction", "confidence", "severity",
        "src_ip", "dst_ip", "src_port", "dst_port", "ip_proto",
        "flow_duration", "packet_rate", "model_version", "status",
    ])
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["packet_rate"] = pd.to_numeric(df["packet_rate"], errors="coerce").fillna(0.0)
    df["flow_duration"] = pd.to_numeric(df["flow_duration"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=2)
def fetch_latest_alerts(limit: int = 50) -> pd.DataFrame:
    """Return the most recent *limit* rows from alerts, newest-first."""
    try:
        with get_db_session() as session:
            rows = session.execute(
                text(
                    """
                    SELECT id, event_time, prediction, confidence, severity,
                           src_ip, dst_ip, src_port, dst_port, ip_proto,
                           flow_duration, packet_rate, model_version, status
                    FROM alerts
                    ORDER BY event_time DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).fetchall()
        return _rows_to_alerts_df(list(rows))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=2)
def fetch_malicious_alerts(limit: int) -> pd.DataFrame:
    """Newest malicious-only rows (full history slice) for the dedicated table and XAI."""
    try:
        with get_db_session() as session:
            rows = session.execute(
                text(
                    """
                    SELECT id, event_time, prediction, confidence, severity,
                           src_ip, dst_ip, src_port, dst_port, ip_proto,
                           flow_duration, packet_rate, model_version, status
                    FROM alerts
                    WHERE prediction = 'Malicious'
                    ORDER BY event_time DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).fetchall()
        return _rows_to_alerts_df(list(rows))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=2)
def fetch_malicious_in_recent_window(window_sec: int) -> pd.DataFrame:
    """
    Malicious rows with event_time in the last `window_sec` seconds (rolling, UTC).
    Drives the attack banner, alarm styling, and Active Threats metric — not the mixed queue.
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_sec)
        cutoff_sql = cutoff.replace(tzinfo=None)
        with get_db_session() as session:
            rows = session.execute(
                text(
                    """
                    SELECT id, event_time, prediction, confidence, severity,
                           src_ip, dst_ip, src_port, dst_port, ip_proto,
                           flow_duration, packet_rate, model_version, status
                    FROM alerts
                    WHERE prediction = 'Malicious' AND event_time >= :cutoff
                    ORDER BY event_time DESC
                    """
                ),
                {"cutoff": cutoff_sql},
            ).fetchall()
        return _rows_to_alerts_df(list(rows))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=2)
def fetch_total_alert_count() -> int:
    """Return total number of alert rows in DB (lifetime count)."""
    try:
        with get_db_session() as session:
            value = session.execute(text("SELECT COUNT(*) FROM alerts")).scalar()
        return int(value or 0)
    except Exception:
        return 0


@st.cache_data(ttl=2)
def fetch_xai_for_alert(alert_id: int) -> np.ndarray | None:
    """Return averaged 5×5 attention matrix for a given alert_id, or None."""
    try:
        with get_db_session() as session:
            row = session.execute(
                text(
                    "SELECT attention_json FROM alert_xai WHERE alert_id = :aid LIMIT 1"
                ),
                {"aid": alert_id},
            ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        w = np.asarray(raw, dtype=np.float64)
        if w.size == 0:
            return None
        if w.ndim == 4:
            return np.mean(w, axis=(0, 1))
        if w.ndim == 3:
            return np.mean(w, axis=0)
        if w.ndim == 2 and w.shape == (5, 5):
            return w
        return None
    except Exception:
        return None


@st.cache_data(ttl=2)
def fetch_active_model_version() -> str:
    try:
        with get_db_session() as session:
            row = session.execute(
                text(
                    "SELECT model_version FROM model_registry WHERE is_active = TRUE LIMIT 1"
                )
            ).fetchone()
        return str(row[0]) if row else "—"
    except Exception:
        return "—"


# ──────────────────────────────────────────────────────────────────────────────
# DB write helpers (analyst workflow)
# ──────────────────────────────────────────────────────────────────────────────

def update_alert_status(alert_id: int, new_status: str, actor: str = "analyst") -> bool:
    """Update alert status and log the action in analyst_actions."""
    try:
        with get_db_session() as session:
            session.execute(
                text("UPDATE alerts SET status = :status WHERE id = :aid"),
                {"status": new_status, "aid": alert_id},
            )
            session.execute(
                text(
                    """
                    INSERT INTO analyst_actions
                        (alert_id, action_type, action_payload, actor)
                    VALUES
                        (:alert_id, :action_type, :payload, :actor)
                    """
                ),
                {
                    "alert_id": alert_id,
                    "action_type": "status_change",
                    "payload": json.dumps({"new_status": new_status}),
                    "actor": actor,
                },
            )
            session.commit()
        return True
    except Exception as exc:
        st.error(f"DB write failed: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_soc_anchor() -> tuple[float, float, str]:
    """
    SOC anchor location from environment, defaults to Hyderabad.

    Optional env vars:
    - SOC_LAT
    - SOC_LON
    - SOC_CITY
    """
    try:
        lat = float(os.getenv("SOC_LAT", "17.3850"))
        lon = float(os.getenv("SOC_LON", "78.4867"))
    except ValueError:
        lat, lon = 17.3850, 78.4867
    city = os.getenv("SOC_CITY", "Hyderabad")
    return lat, lon, city


def _is_private_or_local_ip(ip: str | None) -> bool:
    if not ip:
        return True
    try:
        parsed = ipaddress.ip_address(ip)
        return bool(
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        )
    except ValueError:
        return True


@st.cache_data(ttl=86400, show_spinner=False)
def _resolve_public_ip_geo(ip: str) -> tuple[float, float] | None:
    """
    Resolve public IP to real coordinates using ipapi.co.
    Returns None on network/API failure so we safely fall back to mock.
    """
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=1.8)
        if resp.status_code != 200:
            return None
        body = resp.json()
        lat = body.get("latitude")
        lon = body.get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception:
        return None


def mock_geo_coordinates(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Map IPs to random global coordinates purely for visual demo.
    Private / missing IPs get randomised lat/lon within real country bounding boxes.
    Returns (arc_data_list, scatter_data_list).
    """
    CITY_POOL = [
        (40.7128, -74.0060),   # New York
        (51.5074, -0.1278),    # London
        (35.6762, 139.6503),   # Tokyo
        (48.8566, 2.3522),     # Paris
        (-33.8688, 151.2093),  # Sydney
        (55.7558, 37.6173),    # Moscow
        (28.6139, 77.2090),    # Delhi
        (31.2304, 121.4737),   # Shanghai
        (1.3521, 103.8198),    # Singapore
        (-23.5505, -46.6333),  # São Paulo
        (19.4326, -99.1332),   # Mexico City
        (6.5244, 3.3792),      # Lagos
        (53.3498, -6.2603),    # Dublin
        (37.7749, -122.4194),  # San Francisco
        (41.0082, 28.9784),    # Istanbul
    ]

    rng = random.Random(42)

    def _mock_latlon(ip: str | None) -> tuple[float, float]:
        if not ip:
            return rng.choice(CITY_POOL)
        parts = ip.split(".")
        if len(parts) == 4 and parts[-1].isdigit():
            seed = int(parts[-1]) % len(CITY_POOL)
            lat, lon = CITY_POOL[seed]
            return lat + rng.uniform(-2, 2), lon + rng.uniform(-2, 2)
        # deterministic fallback based on string hash
        seed = abs(hash(ip)) % len(CITY_POOL)
        lat, lon = CITY_POOL[seed]
        return lat + rng.uniform(-2, 2), lon + rng.uniform(-2, 2)

    def _ip_to_latlon(ip: str | None) -> tuple[float, float]:
        if _is_private_or_local_ip(ip):
            return _mock_latlon(ip)
        if not ip:
            return _mock_latlon(ip)
        resolved = _resolve_public_ip_geo(ip)
        if resolved is not None:
            return resolved
        return _mock_latlon(ip)

    arcs: list[dict] = []
    points: list[dict] = []
    home_lat, home_lon, home_city = _get_soc_anchor()

    for _, row in df.iterrows():
        src_lat, src_lon = _ip_to_latlon(row.get("src_ip"))
        is_mal = str(row.get("prediction", "")) == "Malicious"
        color = [255, 34, 68, 200] if is_mal else [0, 200, 255, 120]
        arcs.append({
            "sourcePosition": [src_lon, src_lat],
            "targetPosition": [home_lon, home_lat],
            "color": color,
        })
        points.append({
            "position": [src_lon, src_lat],
            "color": color,
            "label": (
                f"{row.get('src_ip','?')} [{row.get('prediction','')}] -> "
                f"SOC:{home_city}"
            ),
        })

    return arcs, points


def _slice_df_for_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limit map arcs/markers to MAP_PROJECTION_MAX. Prefer newest malicious rows, then benign.
    `df` is already newest-first from SQL.
    """
    if df.empty or len(df) <= MAP_PROJECTION_MAX:
        return df
    pred = df["prediction"].astype(str)
    mal = df[pred == "Malicious"]
    benign = df[pred != "Malicious"]
    parts: list[pd.DataFrame] = []
    if not mal.empty:
        parts.append(mal.head(MAP_PROJECTION_MAX))
    need = MAP_PROJECTION_MAX - sum(len(p) for p in parts)
    if need > 0 and not benign.empty:
        parts.append(benign.head(need))
    if not parts:
        return df.head(MAP_PROJECTION_MAX)
    out = pd.concat(parts, axis=0)
    return out.drop_duplicates(subset=["id"], keep="first").head(MAP_PROJECTION_MAX)


def _build_threat_map(df: pd.DataFrame) -> pdk.Deck | None:
    if df.empty:
        return None
    map_df = _slice_df_for_map(df)
    arcs, points = mock_geo_coordinates(map_df)
    home_lat, home_lon, _ = _get_soc_anchor()
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs,
        get_source_position="sourcePosition",
        get_target_position="targetPosition",
        get_source_color="color",
        get_target_color="color",
        get_width=1.5,
        pickable=True,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="position",
        get_fill_color="color",
        get_radius=85000,
        pickable=True,
    )
    view = pdk.ViewState(latitude=home_lat, longitude=home_lon, zoom=2.2, pitch=25)
    return pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view,
        # Keep map rendering reliable without requiring a Mapbox token.
        map_style=None,
        tooltip={"text": "{label}"},
    )


def _build_radar_chart(df: pd.DataFrame) -> go.Figure:
    """
    Attack-signature radar: one trace per prediction class.

    Raw port (e.g. 443) and duration (seconds) were previously divided by 65535 and 60.
    That makes typical values ~0.006 on the same 0–1 radial scale as confidence (~1),
    so those axes looked “stuck at zero” even when data was fine. We use log scaling
    so port/duration are visually comparable on the chart.
    """
    categories = ["Confidence", "Packet Rate", "Flow Duration", "Protocol Score", "Port Risk"]

    fig = go.Figure()

    for label, color, fill_color in [
        ("Benign",    "#00f3ff", "rgba(0,243,255,0.15)"),
        ("Malicious", "#ff2244", "rgba(255,34,68,0.20)"),
    ]:
        sub = df[df["prediction"].astype(str) == label]
        if sub.empty:
            continue

        conf  = float(sub["confidence"].mean())
        rate  = float(sub["packet_rate"].clip(upper=500).mean() / 500)
        dur_s = float(sub["flow_duration"].clip(lower=0, upper=86400).fillna(0).mean())
        dur   = float(np.log1p(dur_s) / np.log1p(3600.0))
        proto = float(sub["ip_proto"].fillna(6).clip(upper=17).mean() / 17)
        dport = float(sub["dst_port"].fillna(80).clip(lower=0, upper=65535).mean())
        port  = float(np.log1p(dport) / np.log1p(65535.0))

        values = [conf, rate, dur, proto, port]
        values += [values[0]]   # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2),
            name=label,
        ))

    fig.update_layout(
        template=PLOTLY_DARK,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(10,10,15,0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                color="#94a3b8",
                gridcolor="#1e293b",
                tickfont=dict(size=11, color="#cbd5e1"),
            ),
            angularaxis=dict(
                color="#94a3b8",
                gridcolor="#1e293b",
                tickfont=dict(size=11, color="#cbd5e1"),
                categoryorder="array",
                categoryarray=categories,
            ),
        ),
        legend=dict(font=dict(color="#94a3b8", size=11)),
        margin=dict(l=28, r=28, t=12, b=28),
        height=360,
        title=None,
    )
    return fig


def _build_xai_heatmap(mat: np.ndarray) -> go.Figure:
    """Magma XAI attention heatmap — no axis labels (machine-vision aesthetic)."""
    fig = go.Figure(go.Heatmap(
        z=mat,
        colorscale="Magma",
        showscale=True,
        colorbar=dict(
            thickness=12,
            tickfont=dict(color="#64748b", size=10),
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
        ),
    ))
    fig.update_layout(
        template=PLOTLY_DARK,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=10, t=12, b=10),
        title=None,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def _render_queue_table(display_df: pd.DataFrame, *, slot_key: str = "soc_queue_table") -> None:
    """High-contrast HTML table — avoids Streamlit dataframe canvas/grid contrast bugs."""
    cols = list(display_df.columns)
    th = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows_html: list[str] = []
    for _, row in display_df.iterrows():
        tds: list[str] = []
        for c in cols:
            val = row[c]
            esc = html.escape(str(val))
            if c == "Prediction" and str(val) == "Malicious":
                tds.append(f'<td class="soc-pred-mal">{esc}</td>')
            elif c == "Prediction" and str(val) == "Benign":
                tds.append(f'<td class="soc-pred-benign">{esc}</td>')
            else:
                tds.append(f"<td>{esc}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    body = "\n".join(rows_html)
    html_content = (
        f'<div class="soc-queue-wrap"><table class="soc-queue-table">'
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )
    with _keyed_slot(slot_key):
        st.markdown(html_content, unsafe_allow_html=True)


def _show_pydeck_map(deck: pdk.Deck) -> None:
    """Keyed chart so auto-reruns replace the same widget (avoids stacked Deck.gl maps)."""
    try:
        st.pydeck_chart(deck, width="stretch", height=480, key="soc_global_threat_map")
    except TypeError:
        try:
            st.pydeck_chart(deck, width="stretch", key="soc_global_threat_map")
        except TypeError:
            st.pydeck_chart(deck, width="stretch")


def _show_plotly(fig: go.Figure, *, chart_key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=chart_key)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


def _render_terminal_ticker(df: pd.DataFrame) -> None:
    """Hacker-terminal style scrolling telemetry for the last 5 rows."""
    if df.empty:
        st.markdown(
            '<div class="hacker-terminal">[SYS] Waiting for telemetry feed...</div>',
            unsafe_allow_html=True,
        )
        return

    lines: list[str] = []
    for _, row in df.tail(5).iterrows():
        ts     = str(row.get("event_time", ""))[:19].replace("T", " ")
        proto  = PROTOCOL_MAP.get(int(row["ip_proto"]) if pd.notna(row.get("ip_proto")) else 0, "???")
        src    = row.get("src_ip") or "?.?.?.?"
        dst    = row.get("dst_ip") or "?.?.?.?"
        dport  = int(row["dst_port"]) if pd.notna(row.get("dst_port")) else 0
        conf   = float(row["confidence"]) if pd.notna(row.get("confidence")) else 0.0
        pred   = str(row.get("prediction", "?")).upper()
        sev    = str(row.get("severity", "?")).upper()
        status = str(row.get("status", "?")).upper()
        raw_line = (
            f"[{ts}] PROTO:{proto:<4} | {src} -> {dst}:{dport} | "
            f"{pred} | CONF:{conf:.3f} | SEV:{sev} | {status}"
        )
        lines.append(html.escape(raw_line))

    content = "<br>".join(lines)
    st.markdown(
        f'<div class="hacker-terminal">{content}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Header section
# ──────────────────────────────────────────────────────────────────────────────

def _render_critical_banner(threats: int, window_sec: int) -> None:
    """High-visibility incident strip — not Streamlit st.error (no duplicate chrome)."""
    wlabel = f"{window_sec}s (~{window_sec / 60:.1f} min)" if window_sec < 3600 else f"{window_sec // 3600}h"
    st.markdown(
        f"""
        <div class="soc-critical-banner" role="alert">
            <div class="soc-cb-kicker">SOC BREACH WATCH · IMMEDIATE ACTION</div>
            <div class="soc-cb-title">CRITICAL ATTACK · HOSTILE TRAFFIC CONFIRMED</div>
            <p class="soc-cb-body">
                Edge-BERT classified <strong>{threats}</strong> flow(s) as
                <strong>Malicious</strong> with <code>event_time</code> in the last <strong>{wlabel}</strong> (rolling UTC).
                Treat as an active incident: preserve evidence, escalate per your IR playbook, and begin containment.
            </p>
            <span class="soc-cb-count">MALICIOUS IN WINDOW: {threats}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header(total_flows: int, threats: int, latest_conf: float | None) -> None:
    grad = (
        "linear-gradient(90deg,#ff3355,#ffb020,#ff2244)"
        if threats > 0
        else "linear-gradient(90deg,#00f3ff,#7c3aed)"
    )
    st.markdown(
        f"""
        <div style="text-align:center; padding: 10px 0 4px 0;">
            <span style="font-size:2.4rem; font-weight:900;
                         background: {grad};
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                         background-clip:text;">
                🛡️ NetShield AI — Command Center
            </span>
        </div>
        <div style="text-align:center; color:#475569; font-size:0.85rem;
                    letter-spacing:0.12em; margin-bottom:12px;">
            INTELLIGENT BEHAVIOR-CENTRIC PROTECTION · EDGE-BERT INFERENCE · XAI FORENSICS
        </div>
        """,
        unsafe_allow_html=True,
    )

    with _section_box():
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total Flows Analyzed", f"{total_flows:,}")

        with c2:
            st.metric(
                "Active Threats",
                f"{threats:,}",
                delta=f"{threats} malicious" if threats else "all clear",
                delta_color="inverse" if threats else "off",
            )

        with c3:
            if latest_conf is not None:
                st.metric(
                    "Latest Threat Conf.",
                    f"{latest_conf * 100:.1f}%",
                    delta="above threshold" if latest_conf > 0.5 else "below threshold",
                    delta_color="inverse" if latest_conf > 0.5 else "off",
                )
            else:
                st.metric("Latest Threat Conf.", "—", delta="no threats yet")

        with c4:
            model_ver = fetch_active_model_version()
            st.metric("Active Model", model_ver[:20] if len(model_ver) > 20 else model_ver)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NetShield AI SOC",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🛡️",
    )

    # ── fetch data ──────────────────────────────────────────────────────────
    df = fetch_latest_alerts(DASHBOARD_ALERT_FETCH_LIMIT)
    mal_df = fetch_malicious_alerts(MALICIOUS_ONLY_TABLE_ROWS)
    recent_mal = fetch_malicious_in_recent_window(ATTACK_ALERT_WINDOW_SEC)
    total_alert_count = fetch_total_alert_count()

    total_flows = total_alert_count
    threats = len(recent_mal)
    threat_active = threats > 0
    if threat_active:
        latest_conf = float(recent_mal.iloc[0]["confidence"])
    else:
        latest_conf = None

    # ── styles ──────────────────────────────────────────────────────────────
    _inject_styles(threat_active)

    # ── critical incident strip (above fold) ──────────────────────────────
    if threat_active:
        _render_critical_banner(threats, ATTACK_ALERT_WINDOW_SEC)

    # ── header ──────────────────────────────────────────────────────────────
    _render_header(total_flows, threats, latest_conf)

    # ── global threat map ───────────────────────────────────────────────────
    with _section_box():
        st.subheader("Global Threat Map")
        if df.empty:
            st.info("No data yet — start the live sniffer to populate the map.")
        else:
            st.caption(
                f"Map shows up to **{MAP_PROJECTION_MAX}** source projections (newest **Malicious** first, "
                "then **Benign**) so arcs stay readable."
            )
            deck = _build_threat_map(df)
            if deck:
                _show_pydeck_map(deck)

    # ── radar + XAI ─────────────────────────────────────────────────────────
    col_radar, col_xai = st.columns([1, 1], gap="medium")

    with col_radar:
        with _section_box():
            st.subheader("Attack Signature Radar")
            if df.empty:
                st.info("Awaiting data for radar chart.")
            else:
                _show_plotly(_build_radar_chart(df), chart_key="soc_radar_chart")

    with col_xai:
        with _section_box():
            st.subheader("XAI Forensics · Edge-BERT Attention")
            mat: np.ndarray | None = None
            if not mal_df.empty:
                latest_alert_id = int(mal_df.iloc[0]["id"])
                mat = fetch_xai_for_alert(latest_alert_id)
            if mat is not None and mat.shape == (5, 5):
                _show_plotly(_build_xai_heatmap(mat), chart_key="soc_xai_heatmap")
            else:
                st.info("No malicious alert with a valid 5×5 attention matrix yet.")

    # ── analyst workflow: alert table + status update ────────────────────────
    with _section_box():
        st.subheader(f"Analyst Queue — Last {ANALYST_QUEUE_DISPLAY} Alerts")
        st.caption(
            f"Newest {ANALYST_QUEUE_DISPLAY} rows from the DB (same sliding window as charts above; "
            f"backend loads up to {DASHBOARD_ALERT_FETCH_LIMIT} recent rows). "
            "Under heavy traffic, older alerts roll off this view even though they still exist in MySQL."
        )

        if df.empty:
            st.info("No alerts in the database yet.")
        else:
            display_cols = ["id", "event_time", "prediction", "confidence",
                            "severity", "dst_port", "ip_proto", "packet_rate", "status"]
            display_df = df[display_cols].head(ANALYST_QUEUE_DISPLAY).copy()
            display_df["confidence"] = display_df["confidence"].map(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "—"
            )
            display_df["ip_proto"] = display_df["ip_proto"].map(
                lambda x: PROTOCOL_MAP.get(int(x), str(x)) if pd.notna(x) else "?"
            )
            display_df["event_time"] = display_df["event_time"].astype(str).str[:19]
            display_df.columns = ["ID", "Event Time", "Prediction", "Confidence",
                                  "Severity", "Dst Port", "Protocol", "Pkt Rate", "Status"]
            _render_queue_table(display_df, slot_key="soc_queue_mixed_alerts")

    with _section_box():
        st.subheader(f"Malicious flows — last {MALICIOUS_ONLY_TABLE_ROWS}")
        st.caption(
            f"Direct SQL: `prediction = 'Malicious'`, newest **{MALICIOUS_ONLY_TABLE_ROWS}** rows (IPs + ports). "
            "This list **stays visible** after the red attack UI clears. "
            f"The banner/alarms use only malicious rows whose **event_time** falls in the last "
            f"**{ATTACK_ALERT_WINDOW_SEC} seconds** (~1.5 min, rolling UTC)."
        )
        if mal_df.empty:
            st.info("No malicious flows in the database yet.")
        else:
            mcols = [
                "id", "event_time", "src_ip", "dst_ip", "src_port", "dst_port",
                "confidence", "severity", "ip_proto", "packet_rate", "status",
            ]
            mdf = mal_df[mcols].copy()
            mdf["confidence"] = mdf["confidence"].map(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "—"
            )
            mdf["ip_proto"] = mdf["ip_proto"].map(
                lambda x: PROTOCOL_MAP.get(int(x), str(x)) if pd.notna(x) else "?"
            )
            mdf["event_time"] = mdf["event_time"].astype(str).str[:19]
            mdf.columns = [
                "ID", "Event Time", "Src IP", "Dst IP", "Src Port", "Dst Port",
                "Confidence", "Severity", "Protocol", "Pkt Rate", "Status",
            ]
            _render_queue_table(mdf, slot_key="soc_queue_malicious_only")

    with _section_box():
        if not df.empty:
            st.markdown("##### Analyst workflow — update alert status")

            wu_col1, wu_col2, wu_col3 = st.columns([2.2, 2.2, 1.4])
            with wu_col1:
                alert_ids = df["id"].head(ANALYST_QUEUE_DISPLAY).astype(int).tolist()
                selected_id = st.selectbox("Alert ID", options=alert_ids, key="sel_alert_id")
            with wu_col2:
                new_status = st.selectbox("New status", options=ALERT_STATUSES, key="sel_new_status")
            with wu_col3:
                st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
                clicked = st.button(
                    "Save status",
                    key="btn_update_status",
                    width="stretch",
                    type="primary",
                )
                if clicked:
                    if update_alert_status(int(selected_id), new_status):
                        st.success(f"Alert #{selected_id} → **{new_status}** (saved to DB + audit log).")
                        st.cache_data.clear()
                        st.rerun()

    # ── hacker terminal ticker ───────────────────────────────────────────────
    with _section_box():
        st.subheader("Live Telemetry Feed")
        _render_terminal_ticker(df)

    # ── auto-refresh ─────────────────────────────────────────────────────────
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
