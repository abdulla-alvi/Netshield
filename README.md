# NetShield AI: Intelligent Behavior-Centric Network Protection

End-to-end pipeline: **live flow capture → EdgeBERT inference → MySQL → Streamlit SOC dashboard** with attention-based explainability (XAI).

## Requirements

| Component | Notes |
|-----------|--------|
| **Python** | 3.10 or newer |
| **MySQL** | 8.x recommended (5.7+ may work) |
| **OS** | Windows, Linux, or macOS |
| **GPU** | Optional (training/inference use CPU by default if no CUDA) |

**Windows live capture:** install **[Npcap](https://nmap.org/npcap/)** (WinPcap API compatible mode). Run capture as **Administrator** when needed.

## Quick start (new machine)

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`cryptography` is listed explicitly for PyMySQL secure authentication.

### 4. Configure environment

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Linux / macOS
```

Edit `.env` and set:

- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `MYSQL_DB` — default in tooling is `netshield_ai_soc` (must match what you create)
- `MYSQL_PORT` if not `3306`

Optional:

- `SOC_LAT`, `SOC_LON`, `SOC_CITY` — map “home” for the dashboard
- `TRAINING_DATA_DIR` — override CSV location for training (default: `./data`)

**Never commit `.env`.** It is listed in `.gitignore`.

### 5. Initialize the database

Ensure the MySQL server is running and the user can create databases (or create the DB manually).

```bash
python init_db.py
```

This creates the database named in **`MYSQL_DB`** (or `netshield_ai_soc` if unset) and applies `sql/schema.sql`.  
**`init_db.py` and `app/common/db.py` must agree on the same `MYSQL_DB` value.**

### 6. Model artifact

`edge_bert_quantized.pt` is **gitignored** (large binary). On a fresh clone you must either:

- **Train:** add CICIDS-style `*.csv` under `data/` (see `data/README.md`), then:

  ```bash
  python run_training.py
  ```

  This writes `edge_bert_quantized.pt` and registers an active row in `model_registry` when MySQL is configured.

- **Copy** `edge_bert_quantized.pt` from a trusted machine into the project root (and ensure `model_registry` has an active model, or run training once).

### 7. Verify setup (optional)

```bash
python scripts/check_setup.py
python scripts/check_setup.py --mysql
```

### 8. Run components

**SOC dashboard (reads MySQL only):**

```bash
streamlit run soc_dashboard.py
```

**Live sniffer (needs model file + Npcap on Windows + often admin):**

```bash
python live_sniffer.py
```

Run from the **project root** so relative paths to `edge_bert_quantized.pt` resolve.

**Lab / demo inserts (optional):**

```bash
python attack_sim.py
```

## Repository layout

| Path | Purpose |
|------|--------|
| `app/common/db.py` | SQLAlchemy engine; reads `MYSQL_*` from `.env` |
| `sql/schema.sql` | Tables: `alerts`, `alert_xai`, `model_registry`, etc. |
| `init_db.py` | Create DB + apply schema |
| `data/` | Training CSVs (not committed if large) |
| `data_pipeline.py` | NSS tokenizer, `NetworkBehaviorDataset`, sliding windows |
| `model_engine.py` | EdgeBERT |
| `run_training.py` | Train, quantize, save bundle, register model |
| `live_sniffer.py` | Scapy capture, inference, MySQL sink |
| `soc_dashboard.py` | Streamlit UI |
| `scripts/check_setup.py` | Environment smoke test |

## Cross-platform notes

- **Line endings:** Git `core.autocrlf` on Windows is fine; Python handles text files normally.
- **Paths:** Training respects `TRAINING_DATA_DIR` and `Path.expanduser()` for home-relative paths.
- **Torch:** `pip install torch` may download a large wheel; use official [PyTorch install](https://pytorch.org/) if you need a specific CUDA build.
- **Single Streamlit instance** per machine avoids duplicate UI glitches during auto-refresh.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `Access denied` (MySQL) | User, password, host, and that the user has rights on `MYSQL_DB` |
| `Unknown database` | Run `init_db.py` or create DB manually; match `MYSQL_DB` |
| Sniffer sees no interfaces | Npcap installed, reboot, run as Administrator |
| Training exits immediately | `data/*.csv` present and columns match `data/README.md` |
| Dashboard empty | Sniffer or `attack_sim.py` writing rows; DB credentials in `.env` |
| Import errors | Virtualenv activated; `pip install -r requirements.txt` |

## License

Add your team’s license here if applicable.
