"""Result recording — import actual picks and compare to recommendation.

Extracted from v1's ``SeasonManager.record_actual_results()``.
"""

from __future__ import annotations

import json
from typing import Callable

from src.data.fpl_api import (
    fetch_event_live,
    fetch_fpl_api,
    fetch_manager_entry,
    fetch_manager_history,
    fetch_manager_picks,
    fetch_manager_transfers,
)
from src.db.repositories import (
    OutcomeRepository,
    RecommendationRepository,
    SeasonRepository,
    SnapshotRepository,
)
from src.logging_config import get_logger
from src.season.manager import ELEMENT_TYPE_MAP, SeasonManager

logger = get_logger(__name__)


def record_results(
    manager_id: int,
    season_repo: SeasonRepository,
    snapshot_repo: SnapshotRepository,
    rec_repo: RecommendationRepository,
    outcome_repo: OutcomeRepository,
    season_name: str = "",
    progress_fn: Callable[[str], None] | None = None,
) -> dict:
    """Post-GW: import actual picks/results and compare to recommendation.

    Parameters
    ----------
    manager_id:
        FPL manager ID.
    season_repo, snapshot_repo, rec_repo, outcome_repo:
        Repository instances for DB access.
    season_name:
        Season to record for.  Defaults to current season.
    progress_fn:
        Optional progress callback.

    Returns
    -------
    dict
        Recorded GW data and outcome comparison.
    """

    def log(msg: str) -> None:
        if progress_fn:
            progress_fn(msg)
        logger.info(msg)

    season = season_repo.get_season(manager_id, season_name)
    if not season:
        raise ValueError("No active season.")
    season_id = season["id"]

    entry = fetch_manager_entry(manager_id)
    current_event = entry.get("current_event")
    if not current_event:
        raise ValueError("No current event.")

    log(f"Recording results for GW{current_event}...")

    picks_data = fetch_manager_picks(manager_id, current_event)
    history = fetch_manager_history(manager_id)

    # Fetch bootstrap for player info (names, teams, costs)
    bootstrap = fetch_fpl_api("bootstrap", force=True)
    elements_map = SeasonManager._get_elements_map(bootstrap)
    id_to_code, _, code_to_short = SeasonManager._get_team_maps(bootstrap)

    # Bug 54 fix: use live event data for accurate per-GW points
    live_points_map: dict[int, int] = {}
    try:
        live_data = fetch_event_live(current_event, force=True)
        for el_live in live_data.get("elements", []):
            live_points_map[el_live["id"]] = (
                el_live.get("stats", {}).get("total_points", 0)
            )
        log(
            f"  Loaded live points for GW{current_event} "
            f"({len(live_points_map)} players)"
        )
    except Exception as exc:
        log(
            f"  Warning: could not fetch live event data ({exc}), "
            "falling back to bootstrap"
        )

    # Build squad
    picks = picks_data.get("picks", [])
    squad = []
    captain_id = None
    captain_name = None
    for pick in picks:
        eid = pick.get("element")
        el = elements_map.get(eid, {})
        tid = el.get("team")
        tc = id_to_code.get(tid)
        pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")
        # Use live event points when available, fall back to bootstrap
        raw_pts = live_points_map.get(eid, el.get("event_points", 0))
        player = {
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": pos,
            "team_code": tc,
            "team": code_to_short.get(tc, ""),
            "cost": el.get("now_cost", 0) / 10,
            "starter": pick.get("position", 12) <= 11,
            "is_captain": pick.get("is_captain", False),
            "multiplier": pick.get("multiplier", 1),
            "event_points": raw_pts * pick.get("multiplier", 1),
        }
        squad.append(player)
        if pick.get("is_captain"):
            captain_id = eid
            captain_name = el.get("web_name", "Unknown")

    # Find GW data from history
    gw_entries = history.get("current", [])
    gw_data = next((g for g in gw_entries if g["event"] == current_event), {})
    chip_map = {c["event"]: c["name"] for c in history.get("chips", [])}

    entry_hist = picks_data.get("entry_history", {})

    # Fetch transfers for this GW
    all_transfers = fetch_manager_transfers(manager_id)
    gw_transfers = [t for t in all_transfers if t["event"] == current_event]
    t_in_list = []
    t_out_list = []
    for t in gw_transfers:
        el_in = elements_map.get(t["element_in"], {})
        el_out = elements_map.get(t["element_out"], {})
        t_in_list.append({
            "player_id": t["element_in"],
            "web_name": el_in.get("web_name", "Unknown"),
            "cost": t.get("element_in_cost", 0) / 10,
        })
        t_out_list.append({
            "player_id": t["element_out"],
            "web_name": el_out.get("web_name", "Unknown"),
            "cost": t.get("element_out_cost", 0) / 10,
        })

    # Save snapshot
    snapshot_repo.save_gw_snapshot(
        season_id=season_id,
        gameweek=current_event,
        squad_json=json.dumps(squad),
        bank=(
            entry_hist.get("bank", gw_data.get("bank", 0)) / 10
            if entry_hist
            else gw_data.get("bank", 0) / 10
        ),
        team_value=(
            (
                entry_hist.get("value", gw_data.get("value", 0))
                - entry_hist.get("bank", gw_data.get("bank", 0))
            )
            / 10
            if entry_hist
            else (gw_data.get("value", 0) - gw_data.get("bank", 0)) / 10
        ),
        free_transfers=SeasonManager._calculate_free_transfers(history),
        chip_used=chip_map.get(current_event),
        points=gw_data.get("points"),
        total_points=gw_data.get("total_points"),
        overall_rank=gw_data.get("overall_rank"),
        transfers_in_json=json.dumps(t_in_list) if t_in_list else None,
        transfers_out_json=json.dumps(t_out_list) if t_out_list else None,
        captain_id=captain_id,
        captain_name=captain_name,
        transfers_cost=gw_data.get("event_transfers_cost", 0),
    )

    season_repo.update_season_gw(season_id, current_event)

    # Compare to recommendation
    rec = rec_repo.get_recommendation(season_id, current_event)
    outcome: dict = {}
    if rec:
        log("  Comparing to recommendation...")
        actual_points = gw_data.get("points", 0)
        recommended_points = rec.get("predicted_points", 0)
        point_delta = round(
            (actual_points or 0) - (recommended_points or 0), 1
        )

        # Check if captain was followed
        followed_captain = 1 if captain_id == rec.get("captain_id") else 0

        # Check if transfers were followed
        rec_transfers = json.loads(rec.get("transfers_json") or "[]")
        rec_in_ids = {
            t["in"]["player_id"]
            for t in rec_transfers
            if t.get("in", {}).get("player_id")
        }
        actual_squad_ids = {p["player_id"] for p in squad}
        # Bug 87 fix: When recommendation was to bank FT (0 transfers),
        # only count as followed if user also made 0 transfers
        if not rec_in_ids:
            followed_transfers = 1 if not gw_transfers else 0
        else:
            followed_transfers = 1 if rec_in_ids.issubset(actual_squad_ids) else 0

        # WC/FH squad comparison: compare full recommended squad to actual
        rec_chip = rec.get("chip_suggestion") if rec else None
        rec_new_squad = rec.get("new_squad_json") if rec else None
        if rec_chip in ("wildcard", "freehit") and rec_new_squad:
            try:
                rec_squad = json.loads(rec_new_squad)
                rec_squad_ids = {
                    p["player_id"] for p in rec_squad if "player_id" in p
                }
                if rec_squad_ids:
                    overlap = rec_squad_ids & actual_squad_ids
                    followed_transfers = (
                        1 if len(overlap) >= 13 else 0  # 13/15 threshold
                    )
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        actual_chip = chip_map.get(current_event)
        recommended_chip = rec.get("chip_suggestion")
        followed_chip = 1 if actual_chip == recommended_chip else 0

        outcome_repo.save_outcome(
            season_id=season_id,
            gameweek=current_event,
            followed_transfers=followed_transfers,
            followed_captain=followed_captain,
            followed_chip=followed_chip,
            recommended_points=recommended_points,
            actual_points=actual_points,
            point_delta=point_delta,
        )
        outcome = {
            "followed_transfers": followed_transfers,
            "followed_captain": followed_captain,
            "followed_chip": followed_chip,
            "recommended_points": recommended_points,
            "actual_points": actual_points,
            "point_delta": point_delta,
        }

    log(f"GW{current_event} results recorded.")
    return {
        "gameweek": current_event,
        "points": gw_data.get("points"),
        "total_points": gw_data.get("total_points"),
        "overall_rank": gw_data.get("overall_rank"),
        "outcome": outcome,
    }
