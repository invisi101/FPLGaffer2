"""Multi-GW prediction — 3-GW and 8-GW horizons.

Builds offset snapshots with per-GW opponent data and runs the 1-GW ensemble
model at each future gameweek, applying confidence decay.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import POSITION_GROUPS, prediction as pred_cfg
from src.ml.prediction import _ensemble_predict_position

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Offset snapshot builder
# ---------------------------------------------------------------------------

def _build_offset_snapshot(
    current: pd.DataFrame,
    df: pd.DataFrame,
    target_gw: int,
    fixture_map: pd.DataFrame,
    fdr_map: pd.DataFrame,
    elo: pd.DataFrame,
    opp_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Build a prediction snapshot with opponent data for a specific future GW.

    Takes the current GW snapshot (player features) and swaps in
    fixture/opponent columns for *target_gw* so the 1-GW model can predict
    that future GW.
    """
    # Columns that are fixture-specific and must be replaced
    fixture_cols = [
        "opponent_code", "is_home", "fdr", "opponent_elo",
        "next_gw_fixture_count",
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        "xg_x_opp_goals_conceded", "chances_x_opp_big_chances",
        "cs_opportunity", "venue_matched_form",
    ]
    opp_cols = [c for c in current.columns if c.startswith("opp_")]
    vs_opp_cols = [c for c in current.columns if c.startswith("vs_opponent_")]
    drop_cols = [c for c in fixture_cols + opp_cols + vs_opp_cols if c in current.columns]

    # Deduplicate to one row per player (drop DGW fixture splits)
    snapshot = (
        current.drop(columns=drop_cols)
        .drop_duplicates(subset=["player_id"], keep="first")
        .copy()
    )

    # Look up fixtures for the target GW
    gw_fixtures = fixture_map[fixture_map["gameweek"] == target_gw][
        ["team_code", "opponent_code", "is_home"]
    ].copy()

    if gw_fixtures.empty:
        return pd.DataFrame()

    # Count fixtures per team (DGW detection)
    fx_counts = gw_fixtures.groupby("team_code").size().reset_index(
        name="next_gw_fixture_count"
    )

    # Merge fixtures — DGW teams get multiple rows
    snapshot = snapshot.merge(gw_fixtures, on="team_code", how="inner")
    snapshot = snapshot.merge(fx_counts, on="team_code", how="left")
    snapshot["next_gw_fixture_count"] = (
        snapshot["next_gw_fixture_count"].fillna(1).astype(int)
    )

    if snapshot.empty:
        return pd.DataFrame()

    # Add FDR per fixture
    if (
        not fdr_map.empty
        and "team_code" in fdr_map.columns
        and "opponent_code" in fdr_map.columns
    ):
        fdr_lookup = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
        fdr_lookup = fdr_lookup.dropna(subset=["opponent_code"])
        fdr_lookup["opponent_code"] = fdr_lookup["opponent_code"].astype(int)
        fdr_gw = fdr_lookup[fdr_lookup["gameweek"] == target_gw].drop(columns=["gameweek"])
        fdr_gw = fdr_gw.drop_duplicates(subset=["team_code", "opponent_code"], keep="first")
        snapshot = snapshot.merge(fdr_gw, on=["team_code", "opponent_code"], how="left")
    if "fdr" not in snapshot.columns:
        snapshot["fdr"] = 3.0
    snapshot["fdr"] = snapshot["fdr"].fillna(3.0)

    # Add opponent Elo
    if not elo.empty and "team_code" in elo.columns:
        opp_elo = elo.rename(
            columns={"team_code": "opponent_code", "team_elo": "opponent_elo"}
        )
        snapshot = snapshot.merge(opp_elo, on="opponent_code", how="left")
    if "opponent_elo" not in snapshot.columns:
        snapshot["opponent_elo"] = 1500.0
    snapshot["opponent_elo"] = snapshot["opponent_elo"].fillna(1500.0)

    # Add opponent rolling features
    if not opp_rolling.empty:
        opp_feats = opp_rolling.rename(columns={"team_code": "opponent_code"})
        opp_latest = (
            opp_feats.sort_values("gameweek")
            .drop_duplicates(subset=["opponent_code"], keep="last")
            .drop(columns=["gameweek"])
        )
        snapshot = snapshot.merge(opp_latest, on="opponent_code", how="left")

    # Look up vs_opponent_* history from df for the new opponents
    if vs_opp_cols:
        vs_src_cols = ["player_id", "opponent_code", "gameweek"] + vs_opp_cols
        vs_src = df[[c for c in vs_src_cols if c in df.columns]].copy()
        vs_src = vs_src.dropna(subset=["opponent_code"])
        vs_src = vs_src.dropna(
            subset=[c for c in vs_opp_cols if c in vs_src.columns], how="all"
        )
        if not vs_src.empty:
            vs_src["opponent_code"] = vs_src["opponent_code"].astype(int)
            snapshot["opponent_code"] = snapshot["opponent_code"].astype(int)
            vs_src = vs_src.sort_values(["player_id", "opponent_code", "gameweek"])
            vs_latest = vs_src.drop_duplicates(
                subset=["player_id", "opponent_code"], keep="last"
            )
            vs_latest = vs_latest.drop(columns=["gameweek"], errors="ignore")
            snapshot = snapshot.merge(
                vs_latest, on=["player_id", "opponent_code"], how="left"
            )

    # Recompute interaction features
    if "player_xg_last3" in snapshot.columns and "opp_goals_conceded_last3" in snapshot.columns:
        snapshot["xg_x_opp_goals_conceded"] = (
            snapshot["player_xg_last3"] * snapshot["opp_goals_conceded_last3"]
        )
    if "player_chances_created_last3" in snapshot.columns and "opp_big_chances_allowed_last3" in snapshot.columns:
        snapshot["chances_x_opp_big_chances"] = (
            snapshot["player_chances_created_last3"] * snapshot["opp_big_chances_allowed_last3"]
        )
    if "opp_xg_last3" in snapshot.columns:
        snapshot["cs_opportunity"] = 1.0 / (snapshot["opp_xg_last3"] + 0.1)
    elif "opp_opponent_xg_last3" in snapshot.columns:
        snapshot["cs_opportunity"] = 1.0 / (snapshot["opp_opponent_xg_last3"] + 0.1)
    if "home_xg_form" in snapshot.columns and "away_xg_form" in snapshot.columns:
        snapshot["venue_matched_form"] = np.where(
            snapshot["is_home"] == 1, snapshot["home_xg_form"], snapshot["away_xg_form"]
        )

    # Recompute multi-GW lookahead features relative to the target GW
    ahead_gws = fixture_map[
        (fixture_map["gameweek"] > target_gw)
        & (fixture_map["gameweek"] <= target_gw + 3)
    ]
    if not ahead_gws.empty:
        for tc in snapshot["team_code"].unique():
            team_ahead = ahead_gws[ahead_gws["team_code"] == tc]
            mask = snapshot["team_code"] == tc
            if not team_ahead.empty:
                snapshot.loc[mask, "avg_fdr_next3"] = team_ahead.merge(
                    fdr_map[
                        fdr_map["gameweek"].isin(team_ahead["gameweek"].unique())
                    ][["team_code", "gameweek", "opponent_code", "fdr"]].dropna(subset=["fdr"]),
                    on=["team_code", "gameweek", "opponent_code"],
                    how="left",
                )["fdr"].mean()
                snapshot.loc[mask, "home_pct_next3"] = team_ahead["is_home"].mean()
                if not elo.empty and "team_code" in elo.columns:
                    opp_elo_map = dict(zip(elo["team_code"], elo["team_elo"]))
                    snapshot.loc[mask, "avg_opponent_elo_next3"] = (
                        team_ahead["opponent_code"].map(opp_elo_map).mean()
                    )
            else:
                snapshot.loc[mask, "avg_fdr_next3"] = 3.0
                snapshot.loc[mask, "home_pct_next3"] = 0.5
                snapshot.loc[mask, "avg_opponent_elo_next3"] = 1500.0

    # Fill any remaining NaN lookahead features
    snapshot["avg_fdr_next3"] = snapshot.get(
        "avg_fdr_next3", pd.Series(3.0, index=snapshot.index)
    ).fillna(3.0)
    snapshot["home_pct_next3"] = snapshot.get(
        "home_pct_next3", pd.Series(0.5, index=snapshot.index)
    ).fillna(0.5)
    snapshot["avg_opponent_elo_next3"] = snapshot.get(
        "avg_opponent_elo_next3", pd.Series(1500.0, index=snapshot.index)
    ).fillna(1500.0)

    return snapshot


# ---------------------------------------------------------------------------
# 3-GW predictions
# ---------------------------------------------------------------------------

def predict_3gw(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Predict next-3-GW points by summing three 1-GW predictions.

    Returns a tuple of:
      - DataFrame with player_id and predicted_next_3gw_points
      - List of per-GW detail DataFrames (for export)
    """
    fixture_map = fixture_context["fixture_map"]
    fdr_map = fixture_context["fdr_map"]
    elo = fixture_context["elo"]
    opp_rolling = fixture_context["opp_rolling"]

    per_gw_preds: list[pd.DataFrame] = []
    per_gw_detail: list[pd.DataFrame] = []

    for offset in range(1, 4):
        target_gw = latest_gw + offset
        snapshot = _build_offset_snapshot(
            current, df, target_gw, fixture_map, fdr_map, elo, opp_rolling,
        )
        if snapshot.empty:
            log.info("  GW%d: no fixtures found, skipping", target_gw)
            continue

        gw_preds: list[pd.DataFrame] = []
        for position in POSITION_GROUPS:
            preds = _ensemble_predict_position(snapshot, position)
            if not preds.empty:
                gw_preds.append(preds[["player_id", "predicted_next_gw_points"]].copy())

        if gw_preds:
            gw_df = pd.concat(gw_preds, ignore_index=True)
            # Apply confidence decay
            decay = 0.95 ** (offset - 1)
            gw_df["predicted_next_gw_points"] *= decay
            gw_df = gw_df.rename(columns={"predicted_next_gw_points": f"pred_gw{target_gw}"})
            per_gw_preds.append(gw_df)
            n_players = len(gw_df)
            avg_pts = gw_df[f"pred_gw{target_gw}"].mean()
            log.info(
                "  GW%d: %d players, avg %.2f pts (decay=%.4f)",
                target_gw, n_players, avg_pts, decay,
            )

            # Collect fixture info for detail export
            fixture_info = snapshot[["player_id", "team_code"]].drop_duplicates(
                subset=["player_id"], keep="first"
            )
            gw_fixtures = fixture_map[fixture_map["gameweek"] == target_gw][
                ["team_code", "opponent_code", "is_home"]
            ].drop_duplicates()
            fdr_gw = (
                fdr_map[fdr_map["gameweek"] == target_gw][
                    ["team_code", "opponent_code", "fdr"]
                ].drop_duplicates()
                if not fdr_map.empty
                else pd.DataFrame()
            )
            if not fdr_gw.empty:
                gw_fixtures = gw_fixtures.merge(
                    fdr_gw, on=["team_code", "opponent_code"], how="left"
                )
            else:
                gw_fixtures["fdr"] = 3

            fixture_info = fixture_info.merge(gw_fixtures, on="team_code", how="left")
            fixture_info = fixture_info.merge(
                gw_df.rename(columns={f"pred_gw{target_gw}": "predicted_pts"}),
                on="player_id",
                how="left",
            )
            fixture_info["gw"] = target_gw
            per_gw_detail.append(fixture_info)

    if not per_gw_preds:
        return pd.DataFrame(columns=["player_id", "predicted_next_3gw_points"]), []

    # Merge all per-GW predictions and sum
    merged = per_gw_preds[0]
    for extra in per_gw_preds[1:]:
        merged = merged.merge(extra, on="player_id", how="outer")

    pred_cols = [c for c in merged.columns if c.startswith("pred_gw")]
    merged["predicted_next_3gw_points"] = merged[pred_cols].sum(axis=1)

    return merged[["player_id", "predicted_next_3gw_points"]], per_gw_detail


# ---------------------------------------------------------------------------
# Multi-GW predictions (arbitrary horizon)
# ---------------------------------------------------------------------------

def predict_multi_gw(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
    max_gw: int = 8,
) -> dict[int, pd.DataFrame]:
    """Predict points for GW+1 through GW+max_gw.

    Returns ``{gw: DataFrame}`` where each DataFrame has
    ``player_id``, ``predicted_points``, ``confidence`` columns.
    """
    fixture_map = fixture_context["fixture_map"]
    fdr_map = fixture_context["fdr_map"]
    elo = fixture_context["elo"]
    opp_rolling = fixture_context["opp_rolling"]

    predictions: dict[int, pd.DataFrame] = {}
    for offset in range(1, max_gw + 1):
        target_gw = latest_gw + offset
        if target_gw > 38:
            break

        snapshot = _build_offset_snapshot(
            current, df, target_gw, fixture_map, fdr_map, elo, opp_rolling,
        )
        if snapshot.empty:
            continue

        gw_preds: list[pd.DataFrame] = []
        for position in POSITION_GROUPS:
            preds = _ensemble_predict_position(snapshot, position)
            if not preds.empty:
                gw_preds.append(preds[["player_id", "predicted_next_gw_points"]].copy())

        if not gw_preds:
            continue

        result = pd.concat(gw_preds, ignore_index=True)

        # Confidence decay
        confidence = 0.95 ** (offset - 1)
        result["predicted_points"] = result["predicted_next_gw_points"] * confidence
        result["confidence"] = confidence
        result = result.drop(columns=["predicted_next_gw_points"])

        predictions[target_gw] = result
        n = len(result)
        avg = result["predicted_points"].mean()
        log.info(
            "  GW%d: %d players, avg %.2f pts (conf %.2f)",
            target_gw, n, avg, confidence,
        )

    return predictions


def predict_8gw(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
) -> dict[int, pd.DataFrame]:
    """Predict points for the next 8 gameweeks.

    Convenience wrapper around :func:`predict_multi_gw` with ``max_gw=8``.
    """
    return predict_multi_gw(current, df, fixture_context, latest_gw, max_gw=8)
