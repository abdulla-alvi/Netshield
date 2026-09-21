from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.common.mysql_password import resolve_mysql_password

# Default DB name if MYSQL_DB is unset — must match MYSQL_DB in .env for the app.
DEFAULT_DB_NAME = "netshield_ai_soc"


def _load_env(project_root: Path) -> dict[str, str]:
    load_dotenv(project_root / ".env", override=False)

    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    port = os.getenv("MYSQL_PORT", "3306")
    password = resolve_mysql_password(project_root=project_root)

    missing = [
        key
        for key, value in {
            "MYSQL_HOST": host,
            "MYSQL_USER": user,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required env variables: {', '.join(missing)}")

    return {
        "host": host or "",
        "user": user or "",
        "password": password,
        "port": port,
    }


def _server_engine(cfg: dict[str, str]) -> Engine:
    url = (
        f"mysql+pymysql://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['host']}:{cfg['port']}/mysql?charset=utf8mb4"
    )
    return create_engine(url, future=True, pool_pre_ping=True)


def _db_engine(cfg: dict[str, str], db_name: str) -> Engine:
    url = (
        f"mysql+pymysql://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['host']}:{cfg['port']}/{quote_plus(db_name, safe='_')}?charset=utf8mb4"
    )
    return create_engine(url, future=True, pool_pre_ping=True)


def _read_sql_statements(schema_path: Path) -> list[str]:
    raw = schema_path.read_text(encoding="utf-8")
    statements: list[str] = []
    buf: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt.endswith(";"):
                stmt = stmt[:-1]
            if stmt:
                statements.append(stmt)
            buf = []
    if buf:
        stmt = "\n".join(buf).strip()
        if stmt:
            statements.append(stmt)
    return statements


def main() -> None:
    project_root = Path(__file__).resolve().parent
    schema_path = project_root / "sql" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    cfg = _load_env(project_root)
    db_name = (os.getenv("MYSQL_DB") or DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME

    server_engine = _server_engine(cfg)
    with server_engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    server_engine.dispose()

    db_engine = _db_engine(cfg, db_name)
    statements = _read_sql_statements(schema_path)
    with db_engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    db_engine.dispose()

    print(f"Database initialized successfully: {db_name}")
    print(f"Ensure MYSQL_DB={db_name!r} in .env matches (app/common/db.py uses MYSQL_DB).")
    print(f"Applied schema: {schema_path}")


if __name__ == "__main__":
    main()
