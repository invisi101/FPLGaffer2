"""FPL API client — fetch and cache data from fantasy.premierleague.com.

All public endpoints use a 30-minute file cache.  Manager-specific endpoints
use a lightweight in-memory TTL cache (60 s) with thread-safe stale fallback.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from src.config import cache_cfg, data_cfg
from src.data.cache import (
    cache_path,
    is_cache_fresh,
    read_json_cache,
    write_json_cache,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Internal constants ──────────────────────────────────────────────────

_API_BASE = data_cfg.fpl_api_base

_ENDPOINTS: dict[str, str] = {
    "bootstrap": f"{_API_BASE}/bootstrap-static/",
    "fixtures": f"{_API_BASE}/fixtures/",
}

_REQUEST_TIMEOUT = 30  # seconds


# ── Low-level HTTP ──────────────────────────────────────────────────────

def _fetch_url(url: str, timeout: int = _REQUEST_TIMEOUT) -> requests.Response:
    """GET *url*, raise on HTTP errors."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


# ── Public endpoints (file-cached) ──────────────────────────────────────

def fetch_fpl_api(endpoint: str, force: bool = False) -> dict:
    """Fetch JSON from a public FPL API endpoint with file-based caching.

    Parameters
    ----------
    endpoint:
        Key into ``_ENDPOINTS`` (``"bootstrap"`` or ``"fixtures"``).
    force:
        Bypass the cache and re-fetch.
    """
    cp = cache_path(f"fpl_api_{endpoint}.json")
    if not force and is_cache_fresh(cp, max_age=cache_cfg.fpl_api):
        data = read_json_cache(cp)
        if data is not None:
            return data

    url = _ENDPOINTS[endpoint]
    logger.info("Fetching %s", url)
    try:
        resp = _fetch_url(url)
        data = resp.json()
        write_json_cache(cp, data)
    except (requests.RequestException, OSError) as exc:
        if cp.exists():
            logger.warning("Fetch failed (%s), using stale cache for %s", exc, cp.name)
            cached = read_json_cache(cp)
            if cached is not None:
                return cached
        raise
    return data


# ── Event live data (file-cached) ───────────────────────────────────────

def fetch_event_live(event: int, force: bool = False) -> dict:
    """Fetch per-player live stats for gameweek *event*."""
    cp = cache_path(f"fpl_api_event_{event}_live.json")
    if not force and is_cache_fresh(cp, max_age=cache_cfg.fpl_api):
        data = read_json_cache(cp)
        if data is not None:
            return data

    url = f"{_API_BASE}/event/{event}/live/"
    logger.info("Fetching %s", url)
    try:
        resp = _fetch_url(url)
        data = resp.json()
        write_json_cache(cp, data)
    except (requests.RequestException, OSError) as exc:
        if cp.exists():
            logger.warning("Fetch failed (%s), using stale cache", exc)
            cached = read_json_cache(cp)
            if cached is not None:
                return cached
        raise
    return data


# ── Manager endpoints (in-memory TTL cache) ─────────────────────────────

_manager_cache: dict[str, tuple[object, float]] = {}
_manager_cache_lock = threading.Lock()
_MAX_MANAGER_CACHE_SIZE = 200


def _cached_manager_fetch(cache_key: str, fetch_fn):
    """Thread-safe in-memory TTL cache with stale-data fallback.

    If the live fetch fails, the most recent cached value is returned
    instead of propagating the exception (Bug 62 fix from v1).
    """
    now = time.time()
    ttl = cache_cfg.manager_api

    # Check cache
    with _manager_cache_lock:
        if cache_key in _manager_cache:
            data, ts = _manager_cache[cache_key]
            if now - ts < ttl:
                return data

    # Fetch live
    try:
        data = fetch_fn()
    except requests.RequestException:
        # Stale fallback
        with _manager_cache_lock:
            if cache_key in _manager_cache:
                return _manager_cache[cache_key][0]
        raise

    # Store
    with _manager_cache_lock:
        _manager_cache[cache_key] = (data, now)
        # Evict stale entries when cache grows large (Bug 63 fix)
        if len(_manager_cache) > _MAX_MANAGER_CACHE_SIZE:
            stale = [k for k, (_, ts) in _manager_cache.items() if now - ts > ttl]
            for k in stale:
                del _manager_cache[k]

    return data


def fetch_manager_entry(manager_id: int) -> dict:
    """Fetch manager overview (name, bank, value, current_event)."""
    url = f"{_API_BASE}/entry/{manager_id}/"
    return _cached_manager_fetch(
        f"entry_{manager_id}",
        lambda: _fetch_url(url).json(),
    )


def fetch_manager_picks(manager_id: int, event: int) -> dict:
    """Fetch manager's 15 picks for gameweek *event*."""
    url = f"{_API_BASE}/entry/{manager_id}/event/{event}/picks/"
    return _cached_manager_fetch(
        f"picks_{manager_id}_{event}",
        lambda: _fetch_url(url).json(),
    )


def fetch_manager_history(manager_id: int) -> dict:
    """Fetch per-GW history (transfers, chips) for FT calculation."""
    url = f"{_API_BASE}/entry/{manager_id}/history/"
    return _cached_manager_fetch(
        f"history_{manager_id}",
        lambda: _fetch_url(url).json(),
    )


def fetch_manager_transfers(manager_id: int) -> list[dict]:
    """Fetch all transfers made by a manager this season."""
    url = f"{_API_BASE}/entry/{manager_id}/transfers/"
    return _cached_manager_fetch(
        f"transfers_{manager_id}",
        lambda: _fetch_url(url).json(),
    )


def fetch_player_summary(player_id: int) -> dict:
    """Fetch element summary (per-GW history, upcoming fixtures) for one player."""
    url = f"{_API_BASE}/element-summary/{player_id}/"
    return _cached_manager_fetch(
        f"player_summary_{player_id}",
        lambda: _fetch_url(url).json(),
    )


# ── Bulk element-summary fetch (NEW in v2) ──────────────────────────────

def fetch_all_element_summaries(
    player_ids: list[int],
    max_workers: int = 8,
) -> dict[int, dict]:
    """Concurrently fetch element-summary data for multiple players.

    Uses :class:`~concurrent.futures.ThreadPoolExecutor` to parallelise
    requests.  Each individual call goes through
    :func:`fetch_player_summary` (which uses the in-memory TTL cache),
    so repeated calls within the TTL window are free.

    Parameters
    ----------
    player_ids:
        FPL element IDs to fetch.
    max_workers:
        Maximum concurrent threads.  The FPL API is rate-limited, so
        keep this modest (default 8).

    Returns
    -------
    dict[int, dict]
        Mapping of ``player_id -> element-summary JSON``.  Players whose
        fetch failed are silently omitted.
    """
    results: dict[int, dict] = {}

    if not player_ids:
        return results

    logger.info(
        "Fetching element summaries for %d players (workers=%d)",
        len(player_ids),
        max_workers,
    )

    def _fetch_one(pid: int) -> tuple[int, dict | None]:
        try:
            return pid, fetch_player_summary(pid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Element-summary fetch failed for player %d: %s", pid, exc)
            return pid, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, pid): pid for pid in player_ids}
        for future in as_completed(futures):
            pid, data = future.result()
            if data is not None:
                results[pid] = data

    logger.info(
        "Fetched %d / %d element summaries successfully",
        len(results),
        len(player_ids),
    )
    return results
