"""Model training — mean regression, quantile, and decomposed sub-models.

Produces identical models to v1's ``model.py``.  All magic numbers come from
:pymod:`src.config` and all persistence goes through :pymod:`src.ml.model_store`.
"""

from __future__ import annotations

import logging
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import (
    DEFAULT_FEATURES,
    FEATURE_FILL_DEFAULTS,
    POSITION_GROUPS,
    SUB_MODEL_FEATURES,
    decomposed,
    xgb,
)
from src.data.season_detection import detect_current_season
from src.ml.model_store import load_sub_model, save_model, save_sub_model

log = logging.getLogger(__name__)

_N_JOBS = 1 if getattr(sys, "frozen", False) else -1
CURRENT_SEASON = detect_current_season()

# XGBoost parameter grid for tuning
PARAM_GRID = {
    "n_estimators": list(xgb.tune_n_estimators),
    "max_depth": list(xgb.tune_max_depth),
    "learning_rate": list(xgb.tune_learning_rate),
    "subsample": [xgb.subsample],
    "colsample_bytree": [xgb.colsample_bytree],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _season_weight(season: str, current: str) -> float:
    """Weight: 1.0 for current, 0.5 for previous, 0.25 for two back, etc."""
    cur_year = int(current.split("-")[0])
    s_year = int(season.split("-")[0])
    age = cur_year - s_year
    return 0.5 ** age


def _prepare_position_data(
    df: pd.DataFrame,
    position: str,
    target: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Filter and prepare data for a specific position/target."""
    pos_df = df[df["position_clean"] == position].copy()
    pos_df = pos_df.dropna(subset=[target])

    # DGW handling: keep per-fixture rows (aligned with inference).
    # Divide target by fixture count so each row represents a single
    # fixture's contribution to total GW points.
    group_keys = ["player_id", "season", "gameweek"]
    if pos_df.duplicated(subset=group_keys, keep=False).any():
        if "next_gw_fixture_count" in pos_df.columns:
            fc = pos_df["next_gw_fixture_count"].clip(lower=1)
            pos_df[target] = pos_df[target] / fc
    else:
        pos_df = pos_df.drop_duplicates(subset=group_keys, keep="first")

    available_feats = [c for c in feature_cols if c in pos_df.columns]
    # Require at least half the features to be non-null
    pos_df = pos_df.dropna(subset=available_feats, thresh=(len(available_feats) + 1) // 2)
    for c in available_feats:
        pos_df[c] = pos_df[c].fillna(FEATURE_FILL_DEFAULTS.get(c, 0))

    return pos_df, available_feats


def _walk_forward_splits(
    df: pd.DataFrame,
    min_train_gws: int | None = None,
):
    """Walk-forward validation splits across seasons."""
    if min_train_gws is None:
        min_train_gws = xgb.min_train_gws

    df = df.copy()
    season_order = sorted(df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    df["_seq_gw"] = df["season"].map(season_map) * 100 + df["gameweek"]

    seq_gws = sorted(df["_seq_gw"].unique())
    if len(seq_gws) < min_train_gws + 1:
        return

    for i in range(min_train_gws, len(seq_gws)):
        train_gws = set(seq_gws[:i])
        test_gw = seq_gws[i]
        train_mask = df["_seq_gw"].isin(train_gws)
        test_mask = df["_seq_gw"] == test_gw
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            yield train_mask, test_mask


# ---------------------------------------------------------------------------
# Mean regression
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    position: str,
    target: str,
    feature_cols: list[str] | None = None,
    tune: bool = False,
) -> dict:
    """Train an XGBoost model for a position/target combination.

    Returns dict with: model, features, mae, spearman, position, target.
    """
    from xgboost import XGBRegressor

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])

    pos_df, available_feats = _prepare_position_data(df, position, target, feature_cols)

    if len(pos_df) < 50:
        log.info("    %s/%s: insufficient data (%d rows)", position, target, len(pos_df))
        return {}

    log.info("    %s/%s: %d rows, %d features", position, target, len(pos_df), len(available_feats))

    # Season-based sample weights
    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    # Sort by time for temporal ordering
    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    # Tune hyperparameters (if requested)
    if tune:
        log.info("    Tuning hyperparameters...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = XGBRegressor(
                objective="reg:pseudohubererror",
                random_state=xgb.random_state, verbosity=xgb.verbosity,
            )
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(
                base_model, PARAM_GRID, cv=tscv, scoring="neg_mean_absolute_error",
                n_jobs=_N_JOBS, verbose=0,
            )
            grid.fit(X_all, y_all, sample_weight=w_all)
            best_params = grid.best_params_
            log.info("    Best params: %s", best_params)
    else:
        best_params = {
            "n_estimators": xgb.n_estimators,
            "max_depth": xgb.max_depth,
            "learning_rate": xgb.learning_rate,
            "subsample": xgb.subsample,
            "colsample_bytree": xgb.colsample_bytree,
        }

    # Walk-forward validation — last N splits
    maes: list[float] = []
    spearmans: list[float] = []
    all_pred_resid: list[tuple[float, float]] = []
    all_splits = list(_walk_forward_splits(pos_df))
    for train_mask, test_mask in all_splits[-xgb.walk_forward_splits:]:
        X_train = pos_df.loc[train_mask, available_feats].values
        y_train = pos_df.loc[train_mask, target].values
        w_train = pos_df.loc[train_mask, "_sample_weight"].values
        X_test = pos_df.loc[test_mask, available_feats].values
        y_test = pos_df.loc[test_mask, target].values

        # Early stopping: use last 20% of training fold as validation
        val_size = max(1, int(len(X_train) * 0.2))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        w_tr = w_train[:-val_size]

        model = XGBRegressor(
            **best_params, objective="reg:pseudohubererror",
            random_state=xgb.random_state, verbosity=xgb.verbosity,
            early_stopping_rounds=xgb.early_stopping_rounds,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        residuals = y_test - preds
        for pred_val, resid_val in zip(preds, residuals):
            all_pred_resid.append((float(pred_val), float(resid_val)))
        if len(y_test) >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(y_test, preds).correlation
            if not np.isnan(rho):
                spearmans.append(float(rho))

    walk_forward_mae = np.mean(maes) if maes else float("nan")
    avg_spearman = np.mean(spearmans) if spearmans else float("nan")
    log.info(
        "    Walk-forward MAE: %.3f, Spearman: %.3f (over %d splits)",
        walk_forward_mae, avg_spearman, len(maes),
    )

    # Residual percentiles for prediction intervals (80% PI)
    all_resid_vals = [r for _, r in all_pred_resid]
    residual_q10 = float(np.percentile(all_resid_vals, 10)) if all_resid_vals else 0.0
    residual_q90 = float(np.percentile(all_resid_vals, 90)) if all_resid_vals else 0.0

    # Conditional (heteroscedastic) intervals binned by predicted value
    residual_bins: dict[int, dict[str, float]] = {}
    bin_edges: list[float] = []
    if len(all_pred_resid) >= 30:
        pr = np.array(all_pred_resid)
        pred_vals, resid_vals = pr[:, 0], pr[:, 1]
        bin_edges = np.percentile(pred_vals, [33, 67]).tolist()
        bins = np.digitize(pred_vals, bin_edges)
        for b in range(3):
            mask = bins == b
            if mask.sum() >= 10:
                residual_bins[b] = {
                    "q10": float(np.percentile(resid_vals[mask], 10)),
                    "q90": float(np.percentile(resid_vals[mask], 90)),
                }

    # Isotonic calibration from walk-forward residuals
    calibrator = None
    if len(all_pred_resid) >= 50:
        pr_cal = np.array(all_pred_resid)
        pred_vals_cal = pr_cal[:, 0]
        actual_vals_cal = pred_vals_cal + pr_cal[:, 1]  # residual = actual - pred
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(pred_vals_cal, actual_vals_cal)
        log.info("    Fitted isotonic calibrator from %d walk-forward predictions", len(all_pred_resid))

    # Holdout: last 3 sequential GWs
    seq_gws = sorted(pos_df["_seq_gw"].unique())
    holdout_gws = set(seq_gws[-3:])
    holdout_mask = pos_df["_seq_gw"].isin(holdout_gws)
    train_mask_ho = ~holdout_mask
    if train_mask_ho.sum() >= 50 and holdout_mask.sum() >= 10:
        X_ho_train = pos_df.loc[train_mask_ho, available_feats].values
        y_ho_train = pos_df.loc[train_mask_ho, target].values
        w_ho_train = pos_df.loc[train_mask_ho, "_sample_weight"].values

        # Early stopping: use last 20% of holdout training data as validation
        ho_val_size = max(1, int(len(X_ho_train) * 0.2))
        X_ho_tr, X_ho_val = X_ho_train[:-ho_val_size], X_ho_train[-ho_val_size:]
        y_ho_tr, y_ho_val = y_ho_train[:-ho_val_size], y_ho_train[-ho_val_size:]
        w_ho_tr = w_ho_train[:-ho_val_size]

        ho_model = XGBRegressor(
            **best_params, objective="reg:pseudohubererror",
            random_state=xgb.random_state, verbosity=xgb.verbosity,
            early_stopping_rounds=xgb.early_stopping_rounds,
        )
        ho_model.fit(
            X_ho_tr, y_ho_tr, sample_weight=w_ho_tr,
            eval_set=[(X_ho_val, y_ho_val)], verbose=False,
        )
        ho_preds = ho_model.predict(pos_df.loc[holdout_mask, available_feats].values)
        holdout_mae = mean_absolute_error(pos_df.loc[holdout_mask, target].values, ho_preds)
        log.info("    Holdout MAE (last 3 GWs): %.3f", holdout_mae)

    # Train final model on all data (last 15% as early-stopping validation)
    val_size_final = max(1, int(len(X_all) * 0.15))
    final_model = XGBRegressor(
        **best_params, objective="reg:pseudohubererror",
        random_state=xgb.random_state, verbosity=xgb.verbosity,
        early_stopping_rounds=xgb.early_stopping_rounds,
    )
    final_model.fit(
        X_all[:-val_size_final], y_all[:-val_size_final],
        sample_weight=w_all[:-val_size_final],
        eval_set=[(X_all[-val_size_final:], y_all[-val_size_final:])],
        verbose=False,
    )

    # Persist
    save_model(final_model, position, target, metadata={
        "features": available_feats,
        "residual_q10": residual_q10,
        "residual_q90": residual_q90,
        "residual_bins": residual_bins,
        "bin_edges": bin_edges,
        "calibrator": calibrator,
    })

    return {
        "model": final_model,
        "features": available_feats,
        "mae": walk_forward_mae,
        "spearman": avg_spearman,
        "position": position,
        "target": target,
    }


def train_all_models(df: pd.DataFrame, tune: bool = False) -> list[dict]:
    """Train 1-GW mean-regression models for all positions."""
    results = []
    for position in POSITION_GROUPS:
        target = "next_gw_points"
        log.info("  Training %s — %s...", position, target)
        result = train_model(df, position, target, tune=tune)
        if result:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------

def train_quantile_model(
    df: pd.DataFrame,
    position: str,
    target: str = "next_gw_points",
    feature_cols: list[str] | None = None,
    alpha: float = 0.80,
    tune: bool = False,
) -> dict:
    """Train an XGBoost quantile regression model for captain picking.

    Returns dict with: model, features, mae, calibration, position, target.
    """
    from xgboost import XGBRegressor

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])

    pos_df, available_feats = _prepare_position_data(df, position, target, feature_cols)

    suffix = f"_q{int(alpha * 100)}"
    if len(pos_df) < 50:
        log.info("    %s/%s q%d: insufficient data (%d rows)", position, target, int(alpha * 100), len(pos_df))
        return {}

    log.info("    %s/%s%s: %d rows, %d features", position, target, suffix, len(pos_df), len(available_feats))

    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    # Walk-forward validation — last N splits
    maes: list[float] = []
    calibration_vals: list[float] = []
    all_splits = list(_walk_forward_splits(pos_df))
    for train_mask, test_mask in all_splits[-xgb.walk_forward_splits:]:
        X_train = pos_df.loc[train_mask, available_feats].values
        y_train = pos_df.loc[train_mask, target].values
        w_train = pos_df.loc[train_mask, "_sample_weight"].values
        X_test = pos_df.loc[test_mask, available_feats].values
        y_test = pos_df.loc[test_mask, target].values

        m = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha,
            n_estimators=xgb.n_estimators, max_depth=xgb.max_depth,
            learning_rate=xgb.learning_rate, subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
            random_state=xgb.random_state, verbosity=xgb.verbosity,
        )
        m.fit(X_train, y_train, sample_weight=w_train)
        preds = m.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        calibration_vals.append(float((y_test <= preds).mean()))

    walk_forward_mae = np.mean(maes) if maes else float("nan")
    avg_calibration = np.mean(calibration_vals) if calibration_vals else float("nan")
    log.info("    Walk-forward MAE: %.3f (over %d splits)", walk_forward_mae, len(maes))
    log.info(
        "    Calibration: %.1f%% of actuals below q%d prediction (target: %.0f%%)",
        avg_calibration * 100, int(alpha * 100), alpha * 100,
    )

    # Train final model on all data (last 15% as early-stopping validation)
    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    val_size_final = max(1, int(len(X_all) * 0.15))
    final_model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=xgb.n_estimators, max_depth=xgb.max_depth,
        learning_rate=xgb.learning_rate, subsample=xgb.subsample,
        colsample_bytree=xgb.colsample_bytree,
        random_state=xgb.random_state, verbosity=xgb.verbosity,
        early_stopping_rounds=xgb.early_stopping_rounds,
    )
    final_model.fit(
        X_all[:-val_size_final], y_all[:-val_size_final],
        sample_weight=w_all[:-val_size_final],
        eval_set=[(X_all[-val_size_final:], y_all[-val_size_final:])],
        verbose=False,
    )

    save_model(final_model, position, target, metadata={
        "features": available_feats,
    }, suffix=suffix)

    return {
        "model": final_model,
        "features": available_feats,
        "mae": walk_forward_mae,
        "calibration": avg_calibration,
        "position": position,
        "target": target,
    }


def train_all_quantile_models(
    df: pd.DataFrame, alpha: float = 0.80, tune: bool = False,
) -> list[dict]:
    """Train quantile models for MID and FWD (captain-relevant positions)."""
    results = []
    for position in ["MID", "FWD"]:
        log.info("  Training %s — quantile q%d...", position, int(alpha * 100))
        result = train_quantile_model(df, position, alpha=alpha, tune=tune)
        if result:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Decomposed sub-models
# ---------------------------------------------------------------------------

def train_sub_model(
    df: pd.DataFrame,
    position: str,
    component: str,
    tune: bool = False,
) -> dict:
    """Train a single decomposed sub-model for a position/component."""
    from xgboost import XGBRegressor

    target = decomposed.target_columns[component]
    # MID/FWD DefCon uses CBIRT (5 components incl. recoveries) at threshold 12
    if component == "defcon" and position in ("MID", "FWD"):
        target = "next_gw_cbirt"

    feature_cols = SUB_MODEL_FEATURES.get(component, [])
    pos_df, available_feats = _prepare_position_data(df, position, target, feature_cols)

    # Train only on rows where the player actually played.
    if "next_gw_minutes" in pos_df.columns:
        pos_df = pos_df[pos_df["next_gw_minutes"] > 0].copy()

    if len(pos_df) < 30:
        log.info("    %s/sub_%s: insufficient data (%d rows)", position, component, len(pos_df))
        return {}

    log.info("    %s/sub_%s: %d rows, %d features", position, component, len(pos_df), len(available_feats))

    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    objective = decomposed.objectives.get(component, "reg:squarederror")
    obj_params: dict = {"objective": objective}
    if objective == "binary:logistic":
        obj_params["eval_metric"] = "logloss"

    params = {
        "n_estimators": xgb.n_estimators,
        "max_depth": xgb.sub_model_max_depth,
        "learning_rate": xgb.learning_rate,
        "subsample": xgb.subsample,
        "colsample_bytree": xgb.colsample_bytree,
    }

    if tune:
        log.info("    Tuning %s...", component)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = XGBRegressor(**obj_params, random_state=xgb.random_state, verbosity=xgb.verbosity)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(
                base, PARAM_GRID, cv=tscv, scoring="neg_mean_absolute_error",
                n_jobs=_N_JOBS, verbose=0,
            )
            grid.fit(X_all, y_all, sample_weight=w_all)
            params = grid.best_params_

    # Walk-forward MAE + Spearman
    maes: list[float] = []
    spearmans: list[float] = []
    n_splits = 0
    for train_mask, test_mask in _walk_forward_splits(pos_df):
        X_tr = pos_df.loc[train_mask, available_feats].values
        y_tr = pos_df.loc[train_mask, target].values
        w_tr = pos_df.loc[train_mask, "_sample_weight"].values
        X_te = pos_df.loc[test_mask, available_feats].values
        y_te = pos_df.loc[test_mask, target].values

        m = XGBRegressor(**params, **obj_params, random_state=xgb.random_state, verbosity=xgb.verbosity)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = m.predict(X_te)
        maes.append(mean_absolute_error(y_te, preds))
        if len(y_te) >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(y_te, preds).correlation
            if not np.isnan(rho):
                spearmans.append(float(rho))
        n_splits += 1
        if n_splits >= 15:
            break

    wf_mae = np.mean(maes) if maes else float("nan")
    avg_spearman = np.mean(spearmans) if spearmans else float("nan")
    log.info(
        "    Walk-forward MAE: %.4f, Spearman: %.3f (%d splits)",
        wf_mae, avg_spearman, n_splits,
    )

    # Holdout: last 3 sequential GWs
    seq_gws = sorted(pos_df["_seq_gw"].unique())
    holdout_gws = set(seq_gws[-3:])
    ho_mask = pos_df["_seq_gw"].isin(holdout_gws)
    tr_mask = ~ho_mask
    if tr_mask.sum() >= 30 and ho_mask.sum() >= 5:
        ho_m = XGBRegressor(**params, **obj_params, random_state=xgb.random_state, verbosity=xgb.verbosity)
        ho_m.fit(
            pos_df.loc[tr_mask, available_feats].values,
            pos_df.loc[tr_mask, target].values,
            sample_weight=pos_df.loc[tr_mask, "_sample_weight"].values,
        )
        ho_preds = ho_m.predict(pos_df.loc[ho_mask, available_feats].values)
        holdout_mae = mean_absolute_error(pos_df.loc[ho_mask, target].values, ho_preds)
        log.info("    Holdout MAE (last 3 GWs): %.4f", holdout_mae)

    # Final model on all data
    final = XGBRegressor(**params, **obj_params, random_state=xgb.random_state, verbosity=xgb.verbosity)
    final.fit(X_all, y_all, sample_weight=w_all)

    # Build empirical P(CBIT >= threshold) lookup for defcon sub-models
    defcon_cdf = None
    if component == "defcon" and len(X_all) >= 50:
        from src.config import fpl_scoring as _fpl_scoring

        scoring_pos = _fpl_scoring.scoring.get(position, {})
        threshold = scoring_pos.get("defcon_threshold", 10)

        preds_all = final.predict(X_all).clip(min=0)
        actuals_hit = (y_all >= threshold).astype(float)

        # Bin by predicted CBIT value, compute empirical P(hit) per bin
        _bin_edges = np.percentile(preds_all, np.arange(0, 101, 10))
        _bin_edges = np.unique(_bin_edges)
        bins = np.digitize(preds_all, _bin_edges)

        defcon_cdf = {}
        for b in np.unique(bins):
            mask = bins == b
            if mask.sum() >= 5:
                defcon_cdf[str(int(b))] = {
                    "p_hit": float(actuals_hit[mask].mean()),
                    "mean_pred": float(preds_all[mask].mean()),
                    "n": int(mask.sum()),
                }
        defcon_cdf["_bin_edges"] = _bin_edges.tolist()
        log.info(
            "    Built empirical DefCon CDF: %d bins, threshold=%d",
            len(defcon_cdf) - 1, threshold,
        )

    save_sub_model(final, position, component, metadata={
        "features": available_feats,
        "defcon_cdf": defcon_cdf,
    })

    return {
        "model": final,
        "features": available_feats,
        "mae": wf_mae,
        "position": position,
        "component": component,
    }


def train_all_sub_models(df: pd.DataFrame, tune: bool = False) -> list[dict]:
    """Train all decomposed sub-models for all positions."""
    results = []
    for position in POSITION_GROUPS:
        components = decomposed.components.get(position, [])
        log.info("  Training %s sub-models (%s)...", position, ", ".join(components))
        for comp in components:
            result = train_sub_model(df, position, comp, tune=tune)
            if result:
                results.append(result)
    return results
