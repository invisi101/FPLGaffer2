"""Season detection and enumeration helpers.

Determines the current FPL season from bootstrap data or the system date,
and provides utilities to list available training seasons.
"""

from __future__ import annotations

import json
from datetime import datetime

from src.paths import CACHE_DIR
from src.config import data_cfg


def detect_current_season(bootstrap: dict | None = None) -> str:
    """Detect the current FPL season.

    Strategy:
    1. If *bootstrap* is supplied, use GW1 ``deadline_time`` year.
    2. Otherwise, try reading a cached bootstrap file.
    3. Fall back to the system date (June+ = current year, else previous).
    """
    if bootstrap is None:
        cache = CACHE_DIR / "fpl_api_bootstrap.json"
        if cache.exists():
            try:
                bootstrap = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass

    if bootstrap:
        events = bootstrap.get("events", [])
        if events:
            deadline = events[0].get("deadline_time", "")
            if len(deadline) >= 4 and deadline[:4].isdigit():
                y = int(deadline[:4])
                return f"{y}-{y + 1}"

    # Date fallback
    now = datetime.now()
    y = now.year if now.month >= 6 else now.year - 1
    return f"{y}-{y + 1}"


def get_all_seasons(current: str) -> list[str]:
    """Return seasons to fetch, oldest first.

    Capped at :pyattr:`DataConfig.max_seasons` and floored at
    :pyattr:`DataConfig.earliest_season`.
    """
    current_start = int(current.split("-")[0])
    earliest_start = int(data_cfg.earliest_season.split("-")[0])
    first = max(earliest_start, current_start - data_cfg.max_seasons + 1)
    return [f"{y}-{y + 1}" for y in range(first, current_start + 1)]


def get_previous_season(current: str) -> str:
    """Return the season immediately before *current*."""
    start = int(current.split("-")[0])
    return f"{start - 1}-{start}"
