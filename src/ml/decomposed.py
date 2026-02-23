"""Decomposed sub-model predictions combined via FPL scoring rules.

Ports ``predict_decomposed()`` from v1's ``model.py`` identically — Poisson CDF
for DefCon, P(plays) gating, soft calibration caps, DGW summation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import FEATURE_FILL_DEFAULTS, decomposed, fpl_scoring
from src.ml.model_store import load_sub_model

log = logging.getLogger(__name__)


def predict_decomposed(
    snapshot: pd.DataFrame,
    position: str,
    sub_models: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Generate decomposed predictions for *position* and combine via FPL rules.

    When *sub_models* is provided (dict mapping component name -> model_dict),
    uses those directly instead of loading from disk.  This allows the backtest
    to pass in freshly-trained in-memory models.

    Handles DGW players the same way as the mean model: each fixture row gets
    predicted independently, then summed per player.
    """
    components = decomposed.components.get(position, [])
    if not components:
        return pd.DataFrame()

    scoring = fpl_scoring.scoring[position]
    pos_df = snapshot[snapshot["position_clean"] == position].copy()
    if pos_df.empty:
        return pd.DataFrame()

    # Load sub-models from disk if not provided
    component_preds: dict[str, bool] = {}
    if sub_models is None:
        sub_models = {}
        for comp in components:
            model_dict = load_sub_model(position, comp)
            if model_dict is not None:
                sub_models[comp] = model_dict

    # Predict each component
    for comp, model_dict in sub_models.items():
        model = model_dict["model"]
        features = model_dict["features"]
        available = [c for c in features if c in pos_df.columns]
        if not available:
            continue

        for c in available:
            pos_df[c] = pos_df[c].fillna(FEATURE_FILL_DEFAULTS.get(c, 0))

        X = np.full((len(pos_df), len(features)), np.nan)
        for i, f in enumerate(features):
            if f in pos_df.columns:
                X[:, i] = pos_df[f].values
            elif f in FEATURE_FILL_DEFAULTS:
                X[:, i] = FEATURE_FILL_DEFAULTS[f]

        pos_df[f"sub_{comp}"] = model.predict(X).clip(min=0)
        component_preds[comp] = True

    if not component_preds:
        return pd.DataFrame()

    # --- Combine using FPL scoring rules ---

    # P(plays): Use chance_of_playing when it signals doubt (< 100), otherwise
    # fall back to availability_rate_last5 to filter bench warmers.
    cop = (
        pos_df["chance_of_playing"].fillna(100) / 100.0
        if "chance_of_playing" in pos_df.columns
        else pd.Series(0.8, index=pos_df.index)
    )
    avail = (
        pos_df["availability_rate_last5"].fillna(0.75)
        if "availability_rate_last5" in pos_df.columns
        else pd.Series(0.75, index=pos_df.index)
    )
    pos_df["p_plays"] = pd.Series(
        np.where(cop < 1.0, cop, avail), index=pos_df.index
    ).clip(0, 1)

    # Per-player P(60+ | plays) from minutes history
    if "player_minutes_played_last5" in pos_df.columns:
        _p60_rate = (pos_df["player_minutes_played_last5"].fillna(0) / 90.0).clip(0.1, 0.99)
        pos_df["p_60plus"] = pos_df["p_plays"] * _p60_rate
    else:
        pos_df["p_60plus"] = pos_df["p_plays"] * 0.85

    # Appearance points: E[appearance] = P(60+)*2 + (P(plays)-P(60+))*1
    pos_df["pts_appearance"] = (
        pos_df["p_60plus"] * 2
        + (pos_df["p_plays"] - pos_df["p_60plus"]).clip(lower=0) * 1
    )

    # Goals
    if "sub_goals" in pos_df.columns:
        pos_df["pts_goals"] = pos_df["p_plays"] * pos_df["sub_goals"] * scoring["goal"]
    else:
        pos_df["pts_goals"] = 0.0

    # Assists
    if "sub_assists" in pos_df.columns:
        pos_df["pts_assists"] = pos_df["p_plays"] * pos_df["sub_assists"] * scoring["assist"]
    else:
        pos_df["pts_assists"] = 0.0

    # Clean sheets (only if player plays 60+)
    if "sub_cs" in pos_df.columns and scoring["cs"] > 0:
        pos_df["pts_cs"] = pos_df["p_60plus"] * pos_df["sub_cs"] * scoring["cs"]
    else:
        pos_df["pts_cs"] = 0.0

    # Goals conceded penalty (GKP/DEF only)
    # Use continuous E[GC]/2 instead of floor(E[GC]/2)
    if "sub_goals_conceded" in pos_df.columns and scoring["gc_per_2"] != 0:
        pos_df["pts_gc"] = pos_df["p_60plus"] * (pos_df["sub_goals_conceded"] / 2) * scoring["gc_per_2"]
    else:
        pos_df["pts_gc"] = 0.0

    # Saves (GKP only) — continuous E[saves]/3
    if "sub_saves" in pos_df.columns and scoring["save_per_3"] > 0:
        pos_df["pts_saves"] = pos_df["p_plays"] * (pos_df["sub_saves"] / 3) * scoring["save_per_3"]
    else:
        pos_df["pts_saves"] = 0.0

    # Bonus
    if "sub_bonus" in pos_df.columns:
        pos_df["pts_bonus"] = pos_df["p_plays"] * pos_df["sub_bonus"]
    else:
        pos_df["pts_bonus"] = 0.0

    # DefCon: +2 pts if CBIT/CBIRT >= threshold via Poisson CDF
    if "sub_defcon" in pos_df.columns and scoring.get("defcon", 0) > 0:
        from scipy.stats import poisson

        threshold = scoring.get("defcon_threshold", 10)
        expected_cbit = pos_df["sub_defcon"].clip(lower=0.01)
        # P(CBIT >= threshold) = 1 - CDF(threshold - 1)
        p_defcon = 1.0 - poisson.cdf(threshold - 1, mu=expected_cbit)
        pos_df["pts_defcon"] = pos_df["p_plays"] * p_defcon * scoring["defcon"]
    else:
        pos_df["pts_defcon"] = 0.0

    # Total
    pos_df["predicted_next_gw_points"] = (
        pos_df["pts_appearance"]
        + pos_df["pts_goals"]
        + pos_df["pts_assists"]
        + pos_df["pts_cs"]
        + pos_df["pts_gc"]
        + pos_df["pts_saves"]
        + pos_df["pts_bonus"]
        + pos_df["pts_defcon"]
    ).clip(lower=0)

    pred_col = "predicted_next_gw_points"

    # DGW: sum per-fixture predictions
    if pos_df.duplicated(subset=["player_id"], keep=False).any():
        sub_cols = [c for c in pos_df.columns if c.startswith("sub_") or c.startswith("pts_")]
        sum_cols = [pred_col] + sub_cols
        agg = pos_df.groupby("player_id")[sum_cols].sum()
        meta_cols = [c for c in pos_df.columns if c not in sum_cols]
        deduped = pos_df[meta_cols].drop_duplicates(subset=["player_id"], keep="first")
        deduped = deduped.set_index("player_id")
        for c in sum_cols:
            deduped[c] = agg[c]
        pos_df = deduped.reset_index()

    # DGW-aware soft calibration cap (applied after DGW summation so DGW
    # predictions aren't suppressed by a cap designed for single fixtures)
    cap = decomposed.soft_caps.get(position, 10.0)
    if "next_gw_fixture_count" in pos_df.columns:
        eff_cap = cap * pos_df["next_gw_fixture_count"].clip(lower=1)
        over = pos_df[pred_col] > eff_cap
        if over.any():
            pos_df.loc[over, pred_col] = (
                eff_cap[over] + (pos_df.loc[over, pred_col] - eff_cap[over]) * 0.5
            )
    else:
        over = pos_df[pred_col] > cap
        if over.any():
            pos_df.loc[over, pred_col] = cap + (pos_df.loc[over, pred_col] - cap) * 0.5

    return pos_df
