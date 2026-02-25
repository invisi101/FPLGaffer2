"""Reactive availability checking — injury detection and prediction zeroing.

Simplified from the v1 plan-invalidation system. Only two concerns remain:
1. Zero predictions for injured/unavailable players across future GWs.
2. Check a squad for injured/doubtful players (bootstrap availability).
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def apply_availability_adjustments(
    future_predictions: dict[int, pd.DataFrame],
    bootstrap_elements: list[dict],
) -> dict[int, pd.DataFrame]:
    """Zero out predictions for injured/unavailable players.

    - status in (i, s, u, n): zero for ALL future GWs
    - chance_of_playing < 50%: zero for GW+1 only
    """
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

        # Zero injured players for all GWs — catch all prediction columns
        injured_mask = gw_df["player_id"].isin(injured_ids)
        pred_cols = [
            c
            for c in gw_df.columns
            if c.startswith("predicted_") or c in ("captain_score",)
        ]
        for col in pred_cols:
            gw_df.loc[injured_mask, col] = 0.0

        # Zero doubtful players for GW+1 only
        if i == 0:
            doubtful_mask = gw_df["player_id"].isin(doubtful_ids)
            for col in pred_cols:
                gw_df.loc[doubtful_mask, col] = 0.0

        adjusted[gw] = gw_df

    return adjusted


def check_squad_injuries(
    bootstrap: dict, squad_ids: set[int],
) -> list[dict]:
    """Check if any squad players are injured/doubtful.

    Returns list of ``{player_id, web_name, status, chance_of_playing}``.
    """
    elements = {el["id"]: el for el in bootstrap.get("elements", [])}
    issues: list[dict] = []
    for pid in squad_ids:
        el = elements.get(pid)
        if not el:
            continue
        chance = el.get("chance_of_playing_next_round")
        status = el.get("status", "a")
        if status in ("i", "s", "n") or (chance is not None and chance < 75):
            issues.append({
                "player_id": pid,
                "web_name": el.get("web_name", "Unknown"),
                "status": status,
                "chance_of_playing": chance,
            })
    return issues
