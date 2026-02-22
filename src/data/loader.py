"""Data loading orchestrator — single entry point for all data fetching.

Combines FPL API data, GitHub CSV season data, and season detection into
the same dict format that v1's ``load_all_data()`` returned, so downstream
code works unchanged.

Return format::

    {
        "api":            {"bootstrap": {...}, "fixtures": {...}},
        "current_season": "2025-2026",
        "seasons":        ["2024-2025", "2025-2026"],
        "2024-2025":      {"players": DataFrame, "playermatchstats": DataFrame, ...},
        "2025-2026":      {"players": DataFrame, "playermatchstats": DataFrame, ...},
    }
"""

from __future__ import annotations

from src.data.fpl_api import fetch_fpl_api
from src.data.github_csv import fetch_season_data
from src.data.season_detection import detect_current_season, get_all_seasons
from src.logging_config import get_logger

logger = get_logger(__name__)

# The public FPL API endpoints fetched by default.
_API_ENDPOINTS = ("bootstrap", "fixtures")


def load_all_data(force: bool = False) -> dict:
    """Main entry point: fetch everything needed for prediction.

    Parameters
    ----------
    force:
        When ``True``, bypass all caches and re-fetch from source.

    Returns
    -------
    dict
        Contains ``"api"``, ``"current_season"``, ``"seasons"`` keys, plus
        one key per season label (e.g. ``"2024-2025"``) holding the season
        data dict.
    """
    # ── 1. FPL API ──────────────────────────────────────────────────
    logger.info("Fetching FPL API data...")
    api_data = {ep: fetch_fpl_api(ep, force=force) for ep in _API_ENDPOINTS}

    # ── 2. Season detection ─────────────────────────────────────────
    current = detect_current_season(api_data.get("bootstrap"))
    seasons = get_all_seasons(current)

    result: dict = {
        "api": api_data,
        "current_season": current,
        "seasons": seasons,
    }

    # ── 3. Per-season CSV data ──────────────────────────────────────
    for season in seasons:
        logger.info("Fetching %s data...", season)
        try:
            sdata = fetch_season_data(season, force=force)
            if not sdata:
                logger.info("  %s: no data available, skipping", season)
            result[season] = sdata
        except Exception as exc:  # noqa: BLE001
            logger.warning("  Skipping %s: %s", season, exc)
            result[season] = {}

    logger.info("Data loading complete.")
    return result
