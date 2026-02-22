"""Reactive re-planning — detect plan invalidation and adjust availability.

Ported from v1 strategy.py (detect_plan_invalidation, apply_availability_adjustments)
and season_manager.py (check_plan_health).
"""

from __future__ import annotations

import json

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def detect_plan_invalidation(
    current_plan: dict,
    new_predictions: dict[int, pd.DataFrame],
    fixture_calendar: list[dict],
    squad_changes: dict | None = None,
) -> list[dict]:
    """Detect changes that invalidate the current strategic plan.

    Returns list of {severity, type, description, affected_gws}.
    Severity: 'critical' (auto-replan), 'moderate' (warning), 'minor' (logged).
    """
    triggers: list[dict] = []

    if not current_plan or "timeline" not in current_plan:
        return triggers

    timeline = current_plan["timeline"]

    # Check for fixture changes (DGW/BGW)
    fx_by_gw: dict[int, list[dict]] = {}
    for f in fixture_calendar:
        gw = f["gameweek"]
        if gw not in fx_by_gw:
            fx_by_gw[gw] = []
        fx_by_gw[gw].append(f)

    # Check chip GW changes
    chip_schedule = current_plan.get("chip_schedule", {})
    for chip_name, chip_gw in chip_schedule.items():
        if chip_gw in fx_by_gw:
            gw_fixtures = fx_by_gw[chip_gw]
            n_dgw = sum(1 for f in gw_fixtures if f.get("is_dgw"))
            n_bgw = sum(1 for f in gw_fixtures if f.get("is_bgw"))

            # BB without DGW is unusual but legitimate (strong bench week).
            # Only flag as moderate warning, not critical.
            if chip_name == "bboost" and n_dgw == 0:
                triggers.append({
                    "severity": "moderate",
                    "type": "fixture_change",
                    "description": (
                        f"BB planned for GW{chip_gw} without DGW fixtures "
                        "-- verify bench strength justifies it"
                    ),
                    "affected_gws": [chip_gw],
                })

            # If FH was planned for a GW that's no longer a BGW
            if chip_name == "freehit" and n_bgw == 0:
                triggers.append({
                    "severity": "moderate",
                    "type": "fixture_change",
                    "description": (
                        f"FH planned for GW{chip_gw} but no BGW fixtures found"
                    ),
                    "affected_gws": [chip_gw],
                })

    # Check for significant prediction shifts
    for entry in timeline:
        gw = entry["gw"]
        if gw not in new_predictions:
            continue

        new_preds = new_predictions[gw]
        old_captain_id = entry.get("captain_id")
        if old_captain_id:
            new_cap_pred = new_preds[new_preds["player_id"] == old_captain_id]
            if not new_cap_pred.empty:
                new_pts = new_cap_pred.iloc[0]["predicted_points"]
                old_pts = entry.get("captain_points", 0)
                if old_pts > 0 and new_pts < old_pts * 0.5:
                    triggers.append({
                        "severity": "critical",
                        "type": "prediction_shift",
                        "description": (
                            f"GW{gw} captain prediction dropped >50% "
                            f"({old_pts:.1f}->{new_pts:.1f})"
                        ),
                        "affected_gws": [gw],
                    })
            else:
                # Captain missing from predictions entirely
                old_pts = entry.get("captain_points", 0)
                if old_pts > 0:
                    triggers.append({
                        "severity": "critical",
                        "type": "prediction_shift",
                        "description": (
                            f"GW{gw} captain (id={old_captain_id}) "
                            "missing from predictions"
                        ),
                        "affected_gws": [gw],
                    })

        # Check planned transfers: are players being transferred in still worthwhile?
        transfers_in = entry.get("transfers_in", [])
        for t in transfers_in:
            pid = t.get("player_id")
            if pid:
                new_pred = new_preds[new_preds["player_id"] == pid]
                if not new_pred.empty:
                    new_pts = new_pred.iloc[0]["predicted_points"]
                    old_pts = t.get("predicted_points", 0)
                    if old_pts > 0 and new_pts < old_pts * 0.5:
                        triggers.append({
                            "severity": "moderate",
                            "type": "prediction_shift",
                            "description": (
                                f"GW{gw} transfer target "
                                f"{t.get('web_name', '?')} prediction dropped >50%"
                            ),
                            "affected_gws": [gw],
                        })

    # Check for squad changes (injuries, suspensions)
    if squad_changes:
        for player_id, change in squad_changes.items():
            status = change.get("status", "a")
            chance = change.get("chance_of_playing", 100)
            name = change.get("web_name", "Unknown")

            if status in ("i", "s", "u", "n") or (
                chance is not None and chance < 25
            ):
                # Check ALL planned squads this player appears in
                affected_gws = [
                    entry["gw"]
                    for entry in timeline
                    if int(player_id)
                    in [int(x) for x in entry.get("squad_ids", [])]
                ]
                if affected_gws:
                    triggers.append({
                        "severity": "critical",
                        "type": "injury",
                        "description": (
                            f"{name} injured/unavailable -- in planned squads "
                            f"GW{','.join(str(g) for g in affected_gws)}"
                        ),
                        "affected_gws": affected_gws,
                    })
            elif chance is not None and chance < 50:
                affected_gws = [
                    entry["gw"]
                    for entry in timeline
                    if int(player_id)
                    in [int(x) for x in entry.get("squad_ids", [])]
                ]
                if affected_gws:
                    triggers.append({
                        "severity": "moderate",
                        "type": "doubt",
                        "description": (
                            f"{name} doubtful ({chance}% chance) -- in squads "
                            f"GW{','.join(str(g) for g in affected_gws)}"
                        ),
                        "affected_gws": affected_gws,
                    })

    # Sort by severity
    severity_order = {"critical": 0, "moderate": 1, "minor": 2}
    triggers.sort(key=lambda t: severity_order.get(t["severity"], 9))

    return triggers


def apply_availability_adjustments(
    future_predictions: dict[int, pd.DataFrame],
    bootstrap_elements: list[dict],
) -> dict[int, pd.DataFrame]:
    """Zero out predictions for injured/unavailable players.

    - chance_of_playing < 50%: zero for GW+1
    - status == 'i' (injured): zero for all GWs
    """
    # Build availability map
    injured_ids: set[int] = set()
    doubtful_ids: set[int] = set()
    for el in bootstrap_elements:
        status = el.get("status", "a")
        chance = el.get("chance_of_playing_next_round")
        pid = el["id"]

        if status in ("i", "s", "u", "n"):
            injured_ids.add(pid)
        elif chance is not None and chance < 50:
            doubtful_ids.add(pid)

    adjusted: dict[int, pd.DataFrame] = {}
    gws = sorted(future_predictions.keys())
    for i, gw in enumerate(gws):
        gw_df = future_predictions[gw].copy()

        # Zero injured players for all GWs -- catch all prediction columns
        injured_mask = gw_df["player_id"].isin(injured_ids)
        pred_cols = [
            c
            for c in gw_df.columns
            if c.startswith("predicted_") or c in ("captain_score",)
        ]
        for col in pred_cols:
            gw_df.loc[injured_mask, col] = 0.0

        # Zero doubtful players for GW+1 only -- catch all prediction columns
        if i == 0:
            doubtful_mask = gw_df["player_id"].isin(doubtful_ids)
            for col in pred_cols:
                gw_df.loc[doubtful_mask, col] = 0.0

        adjusted[gw] = gw_df

    return adjusted


def check_plan_health(
    plan_json_str: str | None,
    bootstrap: dict,
    fixture_calendar: list[dict],
) -> dict:
    """Lightweight check: is the current strategic plan still valid?

    Uses bootstrap availability data + stored plan. Does NOT regenerate
    predictions (expensive). Returns {healthy, triggers, summary}.
    """
    empty_result = {
        "healthy": True,
        "triggers": [],
        "summary": {"critical": 0, "moderate": 0},
    }

    if not plan_json_str:
        return empty_result

    try:
        current_plan = json.loads(plan_json_str)
    except (json.JSONDecodeError, TypeError):
        return empty_result

    elements = bootstrap.get("elements", [])

    # Build squad_changes from injured/doubtful players
    squad_changes: dict[int, dict] = {}
    for el in elements:
        status = el.get("status", "a")
        chance = el.get("chance_of_playing_next_round")
        if status != "a" or (chance is not None and chance < 75):
            squad_changes[el["id"]] = {
                "status": status,
                "chance_of_playing": chance,
                "web_name": el.get("web_name", "Unknown"),
            }

    # Call detect_plan_invalidation (without new predictions --
    # fixture/injury checks only)
    triggers = detect_plan_invalidation(
        current_plan,
        new_predictions={},
        fixture_calendar=fixture_calendar,
        squad_changes=squad_changes,
    )

    critical = sum(1 for t in triggers if t["severity"] == "critical")
    moderate = sum(1 for t in triggers if t["severity"] == "moderate")

    return {
        "healthy": critical == 0,
        "triggers": triggers,
        "summary": {"critical": critical, "moderate": moderate},
    }
