"""Features derived from the FPL playerstats snapshot.

Handles ICT/BPS cumulative-to-delta conversion, chance of playing,
set piece involvement, availability rate, net transfers, and transfer
momentum.  Includes bootstrap ICT patching for stale GitHub data.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_playerstats_features(
    playerstats: pd.DataFrame,
    bootstrap_elements: list[dict] | None = None,
) -> pd.DataFrame:
    """Extract per-player-per-GW features from the FPL playerstats snapshot.

    When *bootstrap_elements* is provided (from FPL API bootstrap-static),
    the latest GW's cumulative ICT/BPS values are overridden with fresh API
    data.  The GitHub playerstats CSV can have stale ICT for some players,
    causing zero per-GW deltas and broken predictions.
    """
    feature_cols = {
        "event_points": "event_points",
        "form": "player_form",
        "bonus": "player_bonus",
        "bps": "player_bps",
        "ep_next": "ep_next",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict_index": "ict_index",
        "now_cost": "cost",
        "chance_of_playing_next_round": "chance_of_playing",
        "selected_by_percent": "ownership",
        "minutes": "cumulative_minutes",
        "clean_sheets_per_90": "clean_sheets_per_90",
        "starts_per_90": "starts_per_90",
        "yellow_cards": "yellow_cards",
        "transfers_in_event": "transfers_in_event",
        "transfers_out_event": "transfers_out_event",
        "expected_goals_conceded_per_90": "xgc_per_90",
        "saves_per_90": "saves_per_90",
        "total_points": "total_points",
    }

    # Set piece involvement
    set_piece_cols = {
        "penalties_order": "penalties_order",
        "corners_and_indirect_freekicks_order": "corners_order",
        "direct_freekicks_order": "freekicks_order",
    }

    available = {}
    for src, dst in {**feature_cols, **set_piece_cols}.items():
        if src in playerstats.columns:
            available[src] = dst

    result = playerstats[["id", "gw"]].copy()
    result = result.rename(columns={"id": "player_id", "gw": "gameweek"})

    # Carry web_name through if available
    if "web_name" in playerstats.columns:
        result["web_name"] = playerstats["web_name"]

    for src, dst in available.items():
        result[dst] = pd.to_numeric(playerstats[src], errors="coerce")

    # Players without injury flags are fully fit — NaN means 100%
    if "chance_of_playing" in result.columns:
        result["chance_of_playing"] = result["chance_of_playing"].fillna(100)

    # Set piece involvement flag (1 if any set piece role <= 2)
    sp_cols = [v for k, v in set_piece_cols.items() if k in playerstats.columns]
    if sp_cols:
        for c in sp_cols:
            result[c] = result[c].fillna(99)
        result["set_piece_involvement"] = (result[sp_cols].min(axis=1) <= 2).astype(int)

    # Override latest GW cumulative ICT/BPS with fresh FPL API data
    if bootstrap_elements is not None:
        _patch_ict_bps(result, bootstrap_elements)

    # Convert cumulative season totals to per-GW deltas
    for cum_col in ["influence", "creativity", "threat", "ict_index", "player_bps"]:
        if cum_col in result.columns:
            result = result.sort_values(["player_id", "gameweek"])
            # Forward-fill within each player to avoid accumulated deltas across GW gaps
            filled = result.groupby("player_id")[cum_col].ffill()
            result[f"gw_{cum_col}"] = filled.groupby(result["player_id"]).diff()
            first_mask = result[f"gw_{cum_col}"].isna()
            result.loc[first_mask, f"gw_{cum_col}"] = result.loc[first_mask, cum_col]
            result[f"gw_{cum_col}"] = result[f"gw_{cum_col}"].clip(lower=0)

    # Availability consistency: fraction of recent GWs where player featured
    if "cumulative_minutes" in result.columns:
        result = result.sort_values(["player_id", "gameweek"])
        gw_mins = result.groupby("player_id")["cumulative_minutes"].diff()
        first = gw_mins.isna()
        gw_mins = gw_mins.copy()  # force writeable copy for pandas CoW
        gw_mins.loc[first] = result.loc[first, "cumulative_minutes"]
        result["availability_rate_last5"] = (
            (gw_mins > 0).astype(float)
            .groupby(result["player_id"])
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    # Net transfers and transfer momentum
    if "transfers_in_event" in result.columns and "transfers_out_event" in result.columns:
        result["net_transfers"] = (
            result["transfers_in_event"].fillna(0) - result["transfers_out_event"].fillna(0)
        )
        result = result.sort_values(["player_id", "gameweek"])
        result["transfer_momentum"] = (
            result.groupby("player_id")["net_transfers"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    return result


def _patch_ict_bps(
    result: pd.DataFrame, bootstrap_elements: list[dict],
) -> None:
    """Override latest GW cumulative ICT/BPS with fresh FPL API bootstrap data."""
    max_gw = result["gameweek"].max()
    api_lookup: dict[int, dict[str, float]] = {}
    for el in bootstrap_elements:
        api_lookup[el["id"]] = {
            "influence": float(el.get("influence", 0)),
            "creativity": float(el.get("creativity", 0)),
            "threat": float(el.get("threat", 0)),
            "ict_index": float(el.get("ict_index", 0)),
            "player_bps": int(el.get("bps", 0)),
        }
    latest_mask = result["gameweek"] == max_gw
    patched = 0
    for col in ["influence", "creativity", "threat", "ict_index", "player_bps"]:
        if col not in result.columns:
            continue
        for idx in result.index[latest_mask]:
            pid = result.at[idx, "player_id"]
            if pid in api_lookup:
                old_val = result.at[idx, col]
                new_val = api_lookup[pid][col]
                if old_val != new_val:
                    result.at[idx, col] = new_val
                    patched += 1
    if patched:
        log.info("Patched %d stale ICT/BPS values in GW%d from FPL API", patched, max_gw)
