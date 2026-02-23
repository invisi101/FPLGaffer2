"""Prediction pipeline — 1-GW predictions with ensemble blending.

Generates per-player predictions using the 85/15 mean/decomposed ensemble,
adds quantile-based captain scores, prediction intervals, and availability
adjustments.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_FILL_DEFAULTS,
    POSITION_GROUPS,
    decomposed,
    ensemble,
)
from src.data.season_detection import detect_current_season
from src.ml.decomposed import predict_decomposed
from src.ml.model_store import load_model, load_sub_model

log = logging.getLogger(__name__)

CURRENT_SEASON = detect_current_season()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_gw(
    df: pd.DataFrame,
    season: str | None = None,
    *,
    data: dict | None = None,
) -> int:
    """Find the latest GW with complete data for *season*.

    When the FPL API bootstrap is available (via *data*), GWs that are not
    yet ``data_checked`` are skipped so predictions use finalised features.
    """
    if season is None:
        season = CURRENT_SEASON

    season_df = df[df["season"] == season]
    if season_df.empty:
        return int(df["gameweek"].max())

    max_gw = int(season_df["gameweek"].max())

    if data is not None:
        events: dict[int, dict] = {}
        api = data.get("api", {})
        bootstrap = api.get("bootstrap", {})
        for ev in bootstrap.get("events", []):
            events[ev["id"]] = ev
        if events:
            for gw in range(max_gw, 0, -1):
                ev = events.get(gw)
                if ev and ev.get("data_checked", False):
                    if gw < max_gw:
                        log.info(
                            "  GW%d not data_checked, using GW%d as prediction base",
                            max_gw, gw,
                        )
                    return gw

    return max_gw


def predict_for_position(
    snapshot: pd.DataFrame,
    position: str,
    target: str,
    model_dict: dict | None = None,
    *,
    suffix: str = "",
) -> pd.DataFrame:
    """Generate predictions for all players of a given position.

    Returns DataFrame with player_id, predicted points column, and player info.
    """
    if model_dict is None:
        model_dict = load_model(position, target, suffix=suffix)
    if model_dict is None:
        return pd.DataFrame()

    model = model_dict["model"]
    features = model_dict["features"]

    pos_df = snapshot[snapshot["position_clean"] == position].copy()
    available_feats = [c for c in features if c in pos_df.columns]

    if not available_feats:
        return pd.DataFrame()

    # Warn if significant features are missing
    missing_feats = [c for c in features if c not in pos_df.columns]
    if missing_feats and len(missing_feats) > len(features) * 0.1:
        log.warning(
            "  %s/%s%s: %d/%d features missing: %s...",
            position, target, suffix,
            len(missing_feats), len(features), missing_feats[:5],
        )

    for c in available_feats:
        pos_df[c] = pos_df[c].fillna(FEATURE_FILL_DEFAULTS.get(c, 0))

    # Handle missing features: use NaN so XGBoost follows its learned default
    X = np.full((len(pos_df), len(features)), np.nan)
    for i, f in enumerate(features):
        if f in pos_df.columns:
            X[:, i] = pos_df[f].values
        elif f in FEATURE_FILL_DEFAULTS:
            X[:, i] = FEATURE_FILL_DEFAULTS[f]

    pred_col = f"predicted_{target}{suffix}"
    pos_df[pred_col] = model.predict(X).clip(min=0)

    # DGW: sum per-fixture predictions
    if pos_df.duplicated(subset=["player_id"], keep=False).any():
        agg_pred = pos_df.groupby("player_id")[pred_col].sum()
        meta_cols = [c for c in pos_df.columns if c != pred_col]
        deduped = pos_df[meta_cols].drop_duplicates(subset=["player_id"], keep="first")
        deduped = deduped.set_index("player_id")
        deduped[pred_col] = agg_pred
        pos_df = deduped.reset_index()

    return pos_df


def _ensemble_predict_position(
    snapshot: pd.DataFrame,
    position: str,
    decomp_cache: dict | None = None,
) -> pd.DataFrame:
    """Predict for a position using the 85/15 mean/decomposed ensemble blend.

    If *decomp_cache* dict is provided, stores the full decomposed prediction
    DataFrame under ``key=position`` to avoid redundant recomputation.
    """
    pred_col = "predicted_next_gw_points"

    components = decomposed.components.get(position, [])
    has_sub = (
        all(load_sub_model(position, comp) is not None for comp in components)
        if components
        else False
    )

    decomp_preds = predict_decomposed(snapshot, position) if has_sub else pd.DataFrame()
    if decomp_cache is not None and not decomp_preds.empty:
        decomp_cache[position] = decomp_preds

    model_dict = load_model(position, "next_gw_points")
    mean_preds = (
        predict_for_position(snapshot, position, "next_gw_points", model_dict)
        if model_dict is not None
        else pd.DataFrame()
    )

    w_d = ensemble.decomposed_weight
    w_m = 1 - w_d

    if not decomp_preds.empty and not mean_preds.empty:
        merged = decomp_preds[["player_id", pred_col]].merge(
            mean_preds[["player_id", pred_col]],
            on="player_id",
            suffixes=("_decomp", "_mean"),
            how="outer",
        )
        d_col = f"{pred_col}_decomp"
        m_col = f"{pred_col}_mean"
        merged[pred_col] = np.where(
            merged[d_col].notna() & merged[m_col].notna(),
            w_d * merged[d_col] + w_m * merged[m_col],
            merged[m_col].fillna(merged[d_col]),
        )
        meta_cols = [c for c in decomp_preds.columns if c != pred_col]
        return decomp_preds[meta_cols].merge(
            merged[["player_id", pred_col]], on="player_id", how="outer"
        )
    elif not decomp_preds.empty:
        return decomp_preds
    elif not mean_preds.empty:
        return mean_preds
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Full prediction generation
# ---------------------------------------------------------------------------

def _build_player_output(
    df: pd.DataFrame,
    row: pd.Series,
    position_map: dict[int, str],
    team_map: dict[int, str],
) -> dict:
    """Build the API output dict for a single player row."""
    pid = int(row["player_id"])
    return {
        "player_id": pid,
        "web_name": row.get("web_name", str(pid)),
        "position": row.get("position", position_map.get(pid, "MID")),
        "team": team_map.get(int(row.get("team_code", 0)), ""),
        "cost": float(row.get("cost", 0)),
        "form": float(row.get("player_form", 0)),
        "predicted_next_gw_points": round(float(row.get("predicted_next_gw_points", 0)), 2),
        "predicted_next_3gw_points": (
            round(float(row["predicted_next_3gw_points"]), 2)
            if pd.notna(row.get("predicted_next_3gw_points"))
            else None
        ),
        "captain_score": round(float(row.get("captain_score", 0)), 2),
        "prediction_low": (
            round(float(row["prediction_low"]), 2)
            if pd.notna(row.get("prediction_low"))
            else None
        ),
        "prediction_high": (
            round(float(row["prediction_high"]), 2)
            if pd.notna(row.get("prediction_high"))
            else None
        ),
        "ep_next": float(row.get("ep_next", 0)),
        "fdr": int(row.get("fdr", 3)),
        "is_home": bool(row.get("is_home", False)),
    }


def generate_predictions(
    df: pd.DataFrame,
    data: dict | None = None,
) -> dict:
    """Generate predictions for all players for the upcoming gameweek(s).

    Returns a dict with keys: ``players`` (list of player dicts),
    ``gameweek`` (int), ``component_details`` (dict), ``per_gw_detail`` (list).
    """
    latest_gw = get_latest_gw(df, data=data)
    log.info("Latest gameweek in data: GW%d", latest_gw)

    current = df[(df["season"] == CURRENT_SEASON) & (df["gameweek"] == latest_gw)].copy()
    if current.empty:
        current = df[df["gameweek"] == latest_gw].copy()

    unique_players = current.drop_duplicates(subset=["player_id"], keep="first")
    dgw_count = (
        (unique_players["next_gw_fixture_count"] > 1).sum()
        if "next_gw_fixture_count" in unique_players.columns
        else 0
    )
    log.info("Players in current GW: %d (%d with double GW)", len(unique_players), dgw_count)

    # --- 1-GW predictions: ensemble blend ---
    all_preds: list[pd.DataFrame] = []
    _decomp_cache: dict[str, pd.DataFrame] = {}
    for position in POSITION_GROUPS:
        pred_col = "predicted_next_gw_points"
        keep = ["player_id", "position_clean", pred_col]

        preds = _ensemble_predict_position(current, position, decomp_cache=_decomp_cache)
        if not preds.empty:
            if "web_name" in preds.columns and "web_name" not in keep:
                keep.insert(1, "web_name")
            all_preds.append(preds[keep].copy())
            log.info("  %s: %d players", position, len(preds))
        else:
            log.info("  No trained model for %s/next_gw_points", position)

    if not all_preds:
        log.warning("No predictions generated.")
        return {"players": [], "gameweek": latest_gw, "component_details": {}, "per_gw_detail": []}

    result = pd.concat(all_preds, ignore_index=True)

    # --- Prediction intervals from walk-forward residuals ---
    intervals: list[pd.DataFrame] = []
    for position in POSITION_GROUPS:
        model_dict = load_model(position, "next_gw_points")
        if model_dict is None:
            continue
        q10 = model_dict.get("residual_q10", 0.0)
        q90 = model_dict.get("residual_q90", 0.0)
        if q10 == 0.0 and q90 == 0.0:
            continue
        pos_mask = result["position_clean"] == position
        if not pos_mask.any():
            continue
        pos_rows = result.loc[pos_mask].copy()

        bin_edges = model_dict.get("bin_edges")
        residual_bins = model_dict.get("residual_bins")
        if bin_edges and residual_bins:
            pred_vals = pos_rows["predicted_next_gw_points"].values
            bins = np.digitize(pred_vals, bin_edges)
            q10s = np.array([residual_bins.get(b, {"q10": q10})["q10"] for b in bins])
            q90s = np.array([residual_bins.get(b, {"q90": q90})["q90"] for b in bins])
            pos_rows["prediction_low"] = (pred_vals + q10s).clip(min=0)
            pos_rows["prediction_high"] = (pred_vals + q90s).clip(min=0)
        else:
            pos_rows["prediction_low"] = (pos_rows["predicted_next_gw_points"] + q10).clip(lower=0)
            pos_rows["prediction_high"] = (pos_rows["predicted_next_gw_points"] + q90).clip(lower=0)

        intervals.append(pos_rows[["player_id", "prediction_low", "prediction_high"]])
    if intervals:
        interval_df = pd.concat(intervals, ignore_index=True)
        result = result.merge(interval_df, on="player_id", how="left")

    # --- Decomposed component details (for export) ---
    component_details: dict[int, dict] = {}
    for position in POSITION_GROUPS:
        decomp = _decomp_cache.get(position, pd.DataFrame())
        if not decomp.empty:
            sub_cols = [c for c in decomp.columns if c.startswith("sub_") or c.startswith("pts_")]
            keep_cols = ["player_id"] + sub_cols + ["p_plays", "p_60plus"]
            keep_cols = [c for c in keep_cols if c in decomp.columns]
            for _, row in decomp[keep_cols].iterrows():
                pid = int(row["player_id"])
                component_details[pid] = {
                    c: round(float(row[c]), 4) if pd.notna(row[c]) else 0
                    for c in keep_cols
                    if c != "player_id"
                }

    # --- Quantile predictions for captain scoring (MID/FWD only) ---
    q80_preds: list[pd.DataFrame] = []
    for position in ["MID", "FWD"]:
        q_model = load_model(position, "next_gw_points", suffix="_q80")
        if q_model is None:
            continue
        q_preds = predict_for_position(
            current, position, "next_gw_points", q_model, suffix="_q80",
        )
        if not q_preds.empty:
            q80_preds.append(
                q_preds[["player_id", "predicted_next_gw_points_q80"]].copy()
            )

    if q80_preds:
        q80_df = pd.concat(q80_preds, ignore_index=True)
        result = result.merge(q80_df, on="player_id", how="left")

    # Composite captain score
    if "predicted_next_gw_points_q80" in result.columns:
        result["captain_score"] = (
            ensemble.captain_mean_weight * result["predicted_next_gw_points"]
            + ensemble.captain_q80_weight * result["predicted_next_gw_points_q80"].fillna(
                result["predicted_next_gw_points"]
            )
        )
    elif "predicted_next_gw_points" in result.columns:
        result["captain_score"] = result["predicted_next_gw_points"]

    # --- Availability adjustments: zero predictions for unavailable players ---
    unavailable_ids: set[int] = set()
    if data is not None:
        bootstrap_elements = (
            data.get("api", {}).get("bootstrap", {}).get("elements", [])
        )
        if bootstrap_elements:
            for el in bootstrap_elements:
                status = el.get("status", "a")
                chance = el.get("chance_of_playing_next_round")
                pid = el["id"]
                if status in ("i", "s", "u", "n"):
                    unavailable_ids.add(pid)
                elif chance is not None and chance < 50:
                    unavailable_ids.add(pid)

            if unavailable_ids:
                mask = result["player_id"].isin(unavailable_ids)
                n_zeroed = mask.sum()
                pred_cols = [
                    c for c in result.columns
                    if c.startswith("predicted_") or c.startswith("prediction_")
                    or c in ("captain_score",)
                ]
                for col in pred_cols:
                    result.loc[mask, col] = 0.0
                if n_zeroed > 0:
                    log.info("  Zeroed predictions for %d unavailable/doubtful players", n_zeroed)

    # --- 3-GW predictions: sum of per-GW predictions with decay ---
    per_gw_detail: list[pd.DataFrame] = []
    if data is not None:
        try:
            from src.features import get_fixture_context
            from src.ml.multi_gw import predict_3gw

            fixture_context = get_fixture_context(data)
            pred_3gw, per_gw_detail = predict_3gw(current, df, fixture_context, latest_gw)
            if not pred_3gw.empty:
                result = result.merge(pred_3gw, on="player_id", how="left")
                log.info("3-GW predictions merged for %d players", len(pred_3gw))
        except Exception:
            log.warning("Could not generate 3-GW predictions", exc_info=True)

    # --- Re-zero 3-GW predictions for unavailable players (C4 fix) ---
    # The availability zeroing above runs before the 3-GW merge, so
    # predicted_next_3gw_points can be non-zero for injured players.
    if unavailable_ids and "predicted_next_3gw_points" in result.columns:
        mask_3gw = result["player_id"].isin(unavailable_ids)
        result.loc[mask_3gw, "predicted_next_3gw_points"] = 0.0

    return {
        "players": result,
        "gameweek": latest_gw,
        "component_details": component_details,
        "per_gw_detail": per_gw_detail,
    }
