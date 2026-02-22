"""GitHub CSV fetcher — download season data from FPL-Core-Insights.

Supports two repository layouts:
- **flat** (2024-2025): all CSVs in category sub-folders.
- **per-GW** (2025-2026+): root-level summary files plus per-gameweek
  match data under ``By Gameweek/GW{n}/``.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.config import cache_cfg, data_cfg
from src.data.cache import (
    cache_path,
    is_cache_fresh,
    read_csv_cache,
    write_csv_cache,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Layout constants ────────────────────────────────────────────────────

_FLAT_FILES: dict[str, str] = {
    "players": "players/players.csv",
    "playermatchstats": "playermatchstats/playermatchstats.csv",
    "matches": "matches/matches.csv",
    "playerstats": "playerstats/playerstats.csv",
    "teams": "teams/teams.csv",
}

_PER_GW_ROOT_FILES: dict[str, str] = {
    "players": "players.csv",
    "playerstats": "playerstats.csv",
    "teams": "teams.csv",
}

_PER_GW_GW_FILES: list[str] = ["playermatchstats.csv", "matches.csv"]


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_layout(season: str) -> str:
    """Return ``'flat'`` for flat-layout seasons, ``'per_gw'`` otherwise."""
    return "flat" if season in data_cfg.flat_layout_seasons else "per_gw"


def _fetch_url(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _fetch_csv(url: str, cf: "Path", force: bool = False) -> pd.DataFrame:  # noqa: F821
    """Download a CSV from *url*, caching the raw text.

    Includes the Bug 60 fix from v1: the response is validated as a real
    CSV before it is written to the cache, preventing HTML error pages
    from poisoning the cache.
    """
    if not force and is_cache_fresh(cf, max_age=cache_cfg.github_csv):
        return read_csv_cache(cf)

    logger.info("Fetching %s", url)
    try:
        resp = _fetch_url(url)
        # Validate: must parse as CSV with >= 2 columns
        df = pd.read_csv(io.StringIO(resp.text))
        if len(df.columns) < 2:
            raise ValueError(
                f"Response does not look like valid CSV ({len(df.columns)} cols)"
            )
        write_csv_cache(cf, resp.text)
        return df
    except (requests.RequestException, OSError, ValueError) as exc:
        if cf.exists():
            logger.warning(
                "Fetch failed (%s), using stale cache for %s", exc, cf.name
            )
            return read_csv_cache(cf)
        raise


# ── Max-GW detection ────────────────────────────────────────────────────

def _detect_max_gw(season: str, force: bool = False) -> int:
    """Detect the latest available gameweek for a per-GW layout season."""
    cf = cache_path(f"{season}_playerstats.csv")
    url = f"{data_cfg.github_base}/{season}/playerstats.csv"
    df = _fetch_csv(url, cf, force=force)
    if df.empty or "gw" not in df.columns:
        return 0
    max_gw = df["gw"].max()
    if pd.isna(max_gw):
        return 0
    return int(max_gw)


# ── Public API ──────────────────────────────────────────────────────────

def fetch_season_data(
    season: str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all CSV data for *season*, auto-detecting layout.

    Returns a dict mapping dataset name (e.g. ``"players"``,
    ``"playermatchstats"``) to the corresponding DataFrame.  Returns an
    empty dict if the season does not exist on the data source.
    """
    layout = _get_layout(season)
    data: dict[str, pd.DataFrame] = {}

    # ── Probe: check season exists ──────────────────────────────────
    if layout == "flat":
        probe_key = next(iter(_FLAT_FILES))
        probe_path = _FLAT_FILES[probe_key]
    else:
        probe_key = next(iter(_PER_GW_ROOT_FILES))
        probe_path = _PER_GW_ROOT_FILES[probe_key]

    probe_url = f"{data_cfg.github_base}/{season}/{probe_path}"
    probe_cache = cache_path(f"{season}_{probe_key}.csv")
    try:
        _fetch_csv(probe_url, probe_cache, force=force)
    except requests.RequestException:
        return {}

    # ── Flat layout ─────────────────────────────────────────────────
    if layout == "flat":
        for key, path in _FLAT_FILES.items():
            url = f"{data_cfg.github_base}/{season}/{path}"
            cf = cache_path(f"{season}_{key}.csv")
            try:
                data[key] = _fetch_csv(url, cf, force=force)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Could not fetch %s/%s: %s", season, key, exc)
        return data

    # ── Per-GW layout ───────────────────────────────────────────────
    max_gw = _detect_max_gw(season, force=force)
    logger.info("%s: detected %d gameweeks", season, max_gw)

    # Root-level files
    for key, path in _PER_GW_ROOT_FILES.items():
        url = f"{data_cfg.github_base}/{season}/{path}"
        cf = cache_path(f"{season}_{key}.csv")
        try:
            data[key] = _fetch_csv(url, cf, force=force)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Could not fetch %s/%s: %s", season, key, exc)

    # Per-GW files (concatenated across all gameweeks)
    for filename in _PER_GW_GW_FILES:
        key = filename.replace(".csv", "")
        frames: list[pd.DataFrame] = []
        for gw in range(1, max_gw + 1):
            url = (
                f"{data_cfg.github_base}/{season}/"
                f"By Gameweek/GW{gw}/{filename}"
            )
            cf = cache_path(f"{season}_gw{gw}_{filename}")
            try:
                df = _fetch_csv(url, cf, force=force)
                if "gameweek" not in df.columns:
                    df["gameweek"] = gw
                frames.append(df)
            except (requests.RequestException, ValueError):
                pass
        if frames:
            data[key] = pd.concat(frames, ignore_index=True)

    return data
