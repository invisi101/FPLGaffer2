"""SQLite connection manager with WAL mode and foreign keys."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.paths import DB_PATH


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing (handles mid-run DB deletion)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='season'"
    ).fetchone()
    if row is None:
        from src.db.migrations import apply_migrations
        apply_migrations(conn)


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with WAL journal mode, FKs, and Row factory.

    Parameters
    ----------
    db_path:
        Path to the database file.  Defaults to ``DB_PATH`` from
        :mod:`src.paths`.
    """
    db = db_path or DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


@contextmanager
def connect(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and closes it on exit.

    Usage::

        with connect() as conn:
            conn.execute("SELECT ...")
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()
