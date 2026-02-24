"""Shared helpers for API blueprints."""

import json

import numpy as np
import pandas as pd

from src.paths import CACHE_DIR, OUTPUT_DIR
from src.api.sse import pipeline_cache, pipeline_lock

ELEMENT_TYPE_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def safe_num(val, decimals=0):
    """Convert a value to a number, returning 0 for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if decimals > 0:
        return round(float(val), decimals)
    return int(val)


def scrub_nan(obj):
    """Recursively replace NaN/Inf with None in dicts/lists."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: scrub_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_nan(v) for v in obj]
    return obj


def load_bootstrap():
    """Load bootstrap data from cache, return dict or None."""
    path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_next_gw(bootstrap=None):
    """Determine the next gameweek to predict for from bootstrap events.

    Uses ``is_next`` flag first (the GW *after* the one in progress),
    then falls back to current + 1 if no ``is_next`` is set.
    Returns None when the season is over (GW38 finished).
    """
    if bootstrap is None:
        bootstrap = load_bootstrap()
    if not bootstrap:
        return None
    for ev in bootstrap.get("events", []):
        if ev.get("is_next"):
            return ev["id"]
    # Fallback: one after the current GW
    for ev in bootstrap.get("events", []):
        if ev.get("is_current"):
            next_id = ev["id"] + 1
            return next_id if next_id <= 38 else None
    return None


def get_team_map(season=None):
    """Return {team_code: short_name} from cached bootstrap."""
    bootstrap = load_bootstrap()
    if not bootstrap:
        return {}
    return {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}


def get_next_fixtures(n=3):
    """Return {team_code: [opp_short_1, ...]} for next n GWs."""
    bootstrap = load_bootstrap()
    if not bootstrap:
        return {}
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    if not fixtures_path.exists():
        return {}

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    team_map = {t["id"]: t for t in bootstrap.get("teams", [])}
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    next_gw = get_next_gw(bootstrap)
    if not next_gw:
        return {}

    target_gws = set(range(next_gw, next_gw + n))
    result = {}
    for f in sorted(fixtures, key=lambda x: (x.get("event") or 0, x.get("kickoff_time") or "")):
        if f.get("event") not in target_gws:
            continue
        h_code = id_to_code.get(f["team_h"])
        a_code = id_to_code.get(f["team_a"])
        h_short = team_map.get(f["team_h"], {}).get("short_name", "?")
        a_short = team_map.get(f["team_a"], {}).get("short_name", "?")
        if h_code is not None:
            result.setdefault(h_code, []).append(f"{a_short}(H)")
        if a_code is not None:
            result.setdefault(a_code, []).append(f"{h_short}(A)")
    return result


def load_predictions_from_csv():
    """Load predictions.csv into a DataFrame, or return None."""
    path = OUTPUT_DIR / "predictions.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def ensure_pipeline_data():
    """Ensure pipeline_cache has df and features loaded."""
    with pipeline_lock:
        if "df" in pipeline_cache and pipeline_cache["df"] is not None:
            return True

    # Try loading from disk
    from src.data.loader import load_all_data
    from src.features.builder import build_features

    try:
        data = load_all_data()
        df = build_features(data)
        with pipeline_lock:
            pipeline_cache["df"] = df
            pipeline_cache["data"] = data
        return True
    except Exception:
        return False


def resolve_current_squad_event(history: dict, current_event: int) -> tuple[int, bool]:
    """Detect Free Hit reversion: if FH was played in current_event, use pre-FH squad.

    After a Free Hit, the squad reverts to the pre-FH state. When planning for
    the next GW, we need picks from the GW before the FH was played.

    Returns (squad_event, fh_reverted).
    """
    if not current_event or current_event < 2:
        return current_event, False
    chips = history.get("chips", [])
    for chip in chips:
        if chip.get("name") == "freehit" and chip.get("event") == current_event:
            return current_event - 1, True
    return current_event, False


def calculate_free_transfers(history: dict) -> int:
    """Calculate free transfers available for the next GW.

    Walks through history["current"] (one entry per GW played).
    WC/FH preserve FTs at pre-chip count.
    """
    current = history.get("current", [])
    chips = history.get("chips", [])
    chip_events = {c["event"] for c in chips if c.get("name") in ("wildcard", "freehit")}

    first_event = current[0].get("event", 1) if current else 1
    ft = 1
    for i, gw_entry in enumerate(current):
        event = gw_entry.get("event")

        if event in chip_events:
            continue

        transfers_cost = gw_entry.get("event_transfers_cost", 0)
        transfers_made = gw_entry.get("event_transfers", 0)

        paid = transfers_cost // 4 if transfers_cost > 0 else 0
        free_used = transfers_made - paid
        ft = ft - free_used
        ft = max(ft, 0)

        if i == 0 and first_event > 1:
            pass
        else:
            ft = min(ft + 1, 5)

    return max(ft, 1)


def optimize_starting_xi(squad, pred_key="predicted_next_gw_points",
                         captain_key="captain_score"):
    """Pick the best formation-valid XI + captain from a 15-player squad."""
    import copy
    players = [copy.copy(p) for p in squad]

    by_pos = {}
    for p in players:
        by_pos.setdefault(p.get("position", ""), []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: (p.get(pred_key) or 0), reverse=True)

    best_pts = -1
    best_xi_ids = set()

    gkps = by_pos.get("GKP", [])
    defs = by_pos.get("DEF", [])
    mids = by_pos.get("MID", [])
    fwds = by_pos.get("FWD", [])

    for d in range(3, min(6, len(defs) + 1)):
        for m in range(2, min(6, len(mids) + 1)):
            f = 10 - d - m
            if f < 1 or f > 3 or f > len(fwds):
                continue
            xi = gkps[:1] + defs[:d] + mids[:m] + fwds[:f]
            pts = sum((p.get(pred_key) or 0) for p in xi)
            if pts > best_pts:
                best_pts = pts
                best_xi_ids = {p["player_id"] for p in xi}

    for p in players:
        p["starter"] = p["player_id"] in best_xi_ids
        p["is_captain"] = False
        p["is_vice_captain"] = False

    starters = [p for p in players if p["starter"]]
    if starters:
        cap_key = captain_key if any(p.get(captain_key) for p in starters) else pred_key
        starters.sort(key=lambda p: (p.get(cap_key) or 0), reverse=True)
        starters[0]["is_captain"] = True
        if len(starters) > 1:
            starters[1]["is_vice_captain"] = True

    return players


def require_manager_id(args_or_body, source="args"):
    """Extract and validate manager_id. Returns (int, None) or (None, error_tuple)."""
    if source == "args":
        manager_id = args_or_body.get("manager_id")
    else:
        manager_id = args_or_body.get("manager_id")
    if not manager_id:
        return None, ({"error": "manager_id is required."}, 400)
    try:
        return int(manager_id), None
    except (TypeError, ValueError):
        return None, ({"error": "manager_id must be an integer."}, 400)
