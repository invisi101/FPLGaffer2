"""TTL-based cache manager for local file caching.

Provides helpers to check freshness and resolve cache file paths.
All cached files live under ``CACHE_DIR`` (see :mod:`src.paths`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.paths import CACHE_DIR
from src.config import cache_cfg
from src.logging_config import get_logger

logger = get_logger(__name__)


def cache_path(name: str) -> Path:
    """Return the full path for a named cache file inside ``CACHE_DIR``."""
    return CACHE_DIR / name


def is_cache_fresh(path: Path, max_age: int | None = None) -> bool:
    """Return ``True`` if *path* exists and is younger than *max_age* seconds.

    Parameters
    ----------
    path:
        File to check.
    max_age:
        Maximum age in seconds.  Falls back to
        :pyattr:`CacheConfig.github_csv` when *None*.
    """
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < (max_age if max_age is not None else cache_cfg.github_csv)


def ensure_cache_dir() -> None:
    """Create ``CACHE_DIR`` if it does not exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── JSON helpers ────────────────────────────────────────────────────────

def read_json_cache(path: Path) -> dict | list | None:
    """Read a JSON cache file, returning ``None`` on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_cache(path: Path, data: dict | list) -> None:
    """Write *data* as JSON to *path*, creating ``CACHE_DIR`` first."""
    ensure_cache_dir()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── CSV helpers ─────────────────────────────────────────────────────────

def read_csv_cache(path: Path) -> pd.DataFrame:
    """Read a cached CSV file."""
    return pd.read_csv(path, encoding="utf-8")


def write_csv_cache(path: Path, text: str) -> None:
    """Write raw CSV text to *path*, creating ``CACHE_DIR`` first."""
    ensure_cache_dir()
    path.write_text(text, encoding="utf-8")
