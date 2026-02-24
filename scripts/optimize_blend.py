#!/usr/bin/env python3
"""Grid search for optimal ensemble blend weight.

Runs one backtest pass with raw mean/decomposed predictions saved, then
re-blends at multiple candidate weights to find the optimal
decomposed_weight without retraining models.

Usage:
    .venv/bin/python scripts/optimize_blend.py [--start-gw 10] [--end-gw 27]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import POSITION_GROUPS, ensemble
from src.data.loader import load_all_data
from src.features.builder import build_features
from src.ml.backtest import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    pred_df: pd.DataFrame,
    pred_col: str = "predicted_next_gw_points",
) -> dict:
    """Compute MAE, Spearman, top-11 pts, and captain hit rate for one GW."""
    if pred_df.empty or "actual" not in pred_df.columns:
        return {}

    mae = float(np.abs(pred_df[pred_col] - pred_df["actual"]).mean())

    played_mask = pred_df["actual"] > 0
    mae_played = (
        float(np.abs(
            pred_df.loc[played_mask, pred_col] - pred_df.loc[played_mask, "actual"],
        ).mean())
        if played_mask.any()
        else mae
    )

    _sp = spearmanr(pred_df["actual"], pred_df[pred_col]).correlation
    spearman_rho = float(_sp) if not np.isnan(_sp) else 0.0

    model_top11 = pred_df.nlargest(11, pred_col)
    actual_top11 = pred_df.nlargest(11, "actual")
    top11_pts = float(model_top11["actual"].sum())
    actual_best = float(actual_top11["actual"].sum())

    # Captain: use captain_score if available, else pred_col
    cap_col = "captain_score" if "captain_score" in pred_df.columns else pred_col
    captain = pred_df.nlargest(1, cap_col).iloc[0]
    actual_top3_ids = set(pred_df.nlargest(3, "actual")["player_id"])
    captain_in_top3 = bool(captain["player_id"] in actual_top3_ids)

    # Per-position MAE
    pos_maes = {}
    if "position" in pred_df.columns:
        for pos in POSITION_GROUPS:
            pos_rows = pred_df[pred_df["position"] == pos]
            if not pos_rows.empty:
                pos_maes[pos] = float(np.abs(pos_rows[pred_col] - pos_rows["actual"]).mean())

    return {
        "mae": mae,
        "mae_played": mae_played,
        "spearman": spearman_rho,
        "top11_pts": top11_pts,
        "actual_best": actual_best,
        "captain_in_top3": captain_in_top3,
        "pos_maes": pos_maes,
    }


def _reblend_gw(
    raw: dict,
    weight: float | dict[str, float],
) -> pd.DataFrame:
    """Re-blend raw mean/decomp predictions at a given weight.

    *weight* can be a single float (global) or a dict mapping position to
    weight (per-position).
    """
    mean_df = raw["mean"]
    decomp_df = raw["decomp"]
    actuals = raw["actuals"]
    q80_df = raw["q80"]

    if mean_df.empty:
        return pd.DataFrame()

    # Start from mean predictions
    result = mean_df[["player_id", "mean_pred", "position"]].copy()

    # Merge decomposed predictions
    if not decomp_df.empty:
        result = result.merge(
            decomp_df[["player_id", "decomp_pred"]],
            on="player_id",
            how="outer",
        )
    else:
        result["decomp_pred"] = np.nan

    # Blend
    if isinstance(weight, dict):
        # Per-position weights
        blended = []
        for pos in result["position"].dropna().unique():
            pos_mask = result["position"] == pos
            w_d = weight.get(pos, 0.15)
            pos_rows = result[pos_mask].copy()
            pos_rows["predicted_next_gw_points"] = np.where(
                pos_rows["decomp_pred"].notna() & pos_rows["mean_pred"].notna(),
                w_d * pos_rows["decomp_pred"] + (1 - w_d) * pos_rows["mean_pred"],
                pos_rows["mean_pred"].fillna(pos_rows["decomp_pred"]),
            )
            blended.append(pos_rows)
        result = pd.concat(blended, ignore_index=True) if blended else result
    else:
        w_d = weight
        result["predicted_next_gw_points"] = np.where(
            result["decomp_pred"].notna() & result["mean_pred"].notna(),
            w_d * result["decomp_pred"] + (1 - w_d) * result["mean_pred"],
            result["mean_pred"].fillna(result["decomp_pred"]),
        )

    # Add actuals
    result["actual"] = result["player_id"].map(actuals)
    result = result.dropna(subset=["actual"])

    # Add captain score (Q80 blend if available)
    pred_col = "predicted_next_gw_points"
    if not q80_df.empty and "predicted_next_gw_points_q80" in q80_df.columns:
        result = result.merge(
            q80_df[["player_id", "predicted_next_gw_points_q80"]],
            on="player_id",
            how="left",
        )
        result["captain_score"] = (
            ensemble.captain_mean_weight * result[pred_col]
            + ensemble.captain_q80_weight
            * result["predicted_next_gw_points_q80"].fillna(result[pred_col])
        )
    else:
        result["captain_score"] = result[pred_col]

    return result


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(
    raw_preds: dict[int, dict],
    candidate_weights: list[float],
) -> pd.DataFrame:
    """Run grid search over global blend weights.

    Returns a DataFrame with one row per weight and summary metrics.
    """
    rows = []
    for w in candidate_weights:
        gw_metrics = []
        for gw, raw in sorted(raw_preds.items()):
            pred_df = _reblend_gw(raw, w)
            if pred_df.empty:
                continue
            m = _compute_metrics(pred_df)
            if m:
                m["gw"] = gw
                gw_metrics.append(m)

        if not gw_metrics:
            continue

        n = len(gw_metrics)
        rows.append({
            "weight": w,
            "mae": round(np.mean([m["mae"] for m in gw_metrics]), 4),
            "mae_played": round(np.mean([m["mae_played"] for m in gw_metrics]), 4),
            "spearman": round(np.mean([m["spearman"] for m in gw_metrics]), 4),
            "top11_pts": round(np.mean([m["top11_pts"] for m in gw_metrics]), 1),
            "captain_pct": round(
                sum(m["captain_in_top3"] for m in gw_metrics) / n * 100, 1,
            ),
            "n_gws": n,
        })

    return pd.DataFrame(rows)


def run_per_position_search(
    raw_preds: dict[int, dict],
    candidate_weights: list[float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Greedy per-position weight optimization.

    For each position, finds the best weight while holding others at
    default. Then does a refinement pass. Returns results table and
    best per-position weights.
    """
    # Start with the default weight for all positions
    best_weights: dict[str, float] = {pos: ensemble.decomposed_weight for pos in POSITION_GROUPS}

    # Greedy: optimize one position at a time
    for pos in POSITION_GROUPS:
        best_mae = float("inf")
        best_w = best_weights[pos]
        for w in candidate_weights:
            test_weights = best_weights.copy()
            test_weights[pos] = w

            gw_maes = []
            for gw, raw in sorted(raw_preds.items()):
                pred_df = _reblend_gw(raw, test_weights)
                if pred_df.empty:
                    continue
                # Only measure this position's MAE
                pos_rows = pred_df[pred_df["position"] == pos]
                if not pos_rows.empty:
                    mae = float(np.abs(
                        pos_rows["predicted_next_gw_points"] - pos_rows["actual"],
                    ).mean())
                    gw_maes.append(mae)

            if gw_maes:
                avg_mae = np.mean(gw_maes)
                if avg_mae < best_mae:
                    best_mae = avg_mae
                    best_w = w
        best_weights[pos] = best_w

    # Evaluate the per-position result
    gw_metrics = []
    for gw, raw in sorted(raw_preds.items()):
        pred_df = _reblend_gw(raw, best_weights)
        if pred_df.empty:
            continue
        m = _compute_metrics(pred_df)
        if m:
            gw_metrics.append(m)

    n = len(gw_metrics)
    result_row = {
        "weights": best_weights,
        "mae": round(np.mean([m["mae"] for m in gw_metrics]), 4) if gw_metrics else None,
        "mae_played": round(np.mean([m["mae_played"] for m in gw_metrics]), 4) if gw_metrics else None,
        "spearman": round(np.mean([m["spearman"] for m in gw_metrics]), 4) if gw_metrics else None,
        "top11_pts": round(np.mean([m["top11_pts"] for m in gw_metrics]), 1) if gw_metrics else None,
        "captain_pct": round(
            sum(m["captain_in_top3"] for m in gw_metrics) / n * 100, 1,
        ) if gw_metrics else None,
    }

    # Also compute per-position MAE for the best weights
    pos_results = []
    for pos in POSITION_GROUPS:
        pos_maes = [m["pos_maes"].get(pos) for m in gw_metrics if pos in m.get("pos_maes", {})]
        if pos_maes:
            pos_results.append({
                "position": pos,
                "weight": best_weights[pos],
                "mae": round(np.mean(pos_maes), 4),
            })

    return pd.DataFrame(pos_results), best_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble blend weight")
    parser.add_argument("--start-gw", type=int, default=10,
                        help="First GW to backtest (default: 10)")
    parser.add_argument("--end-gw", type=int, default=27,
                        help="Last GW to backtest (default: 27)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Ensemble Blend Weight Optimization (L9)")
    log.info("=" * 60)

    # Step 1: Load data + build features
    log.info("Loading data...")
    data = load_all_data()
    log.info("Building features...")
    df = build_features(data)
    log.info("Features built: %d rows", len(df))

    # Step 2: Run backtest ONCE with raw predictions saved
    log.info("Running backtest (GW%d-%d) with raw predictions...", args.start_gw, args.end_gw)
    t0 = time.time()
    bt_result = run_backtest(
        df,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        save_raw_preds=True,
    )
    elapsed = time.time() - t0
    log.info("Backtest complete in %.1f seconds", elapsed)

    raw_preds = bt_result.get("raw_preds", {})
    if not raw_preds:
        log.error("No raw predictions saved. Cannot run grid search.")
        sys.exit(1)

    baseline_summary = bt_result["summary"]
    log.info(
        "Baseline (w=%.2f): MAE=%.4f Spearman=%.4f Top11=%.1f Captain%%=%.0f%%",
        ensemble.decomposed_weight,
        baseline_summary["model_avg_mae"],
        baseline_summary["avg_spearman"],
        baseline_summary["model_avg_top11_pts"],
        baseline_summary["captain_hit_rate"] * 100,
    )

    # Step 3: Grid search over global weights
    log.info("")
    log.info("=" * 60)
    log.info("Global Weight Grid Search")
    log.info("=" * 60)
    candidate_weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = run_grid_search(raw_preds, candidate_weights)

    # Print results table
    print("\n" + "=" * 80)
    print("GLOBAL BLEND WEIGHT RESULTS")
    print("=" * 80)
    print(f"{'Weight':>8} {'MAE':>8} {'MAE(pl)':>8} {'Spearman':>9} {'Top11 Pts':>10} {'Cap Top3%':>10}")
    print("-" * 63)
    for _, row in results.iterrows():
        marker = " <-- current" if row["weight"] == ensemble.decomposed_weight else ""
        print(
            f"{row['weight']:>8.2f} {row['mae']:>8.4f} {row['mae_played']:>8.4f} "
            f"{row['spearman']:>9.4f} {row['top11_pts']:>10.1f} {row['captain_pct']:>9.1f}%{marker}"
        )

    # Find best weight
    best_row = results.loc[results["mae"].idxmin()]
    best_by_spearman = results.loc[results["spearman"].idxmax()]
    best_by_top11 = results.loc[results["top11_pts"].idxmax()]

    print(f"\nBest by MAE:      w={best_row['weight']:.2f} (MAE={best_row['mae']:.4f})")
    print(f"Best by Spearman: w={best_by_spearman['weight']:.2f} (rho={best_by_spearman['spearman']:.4f})")
    print(f"Best by Top11:    w={best_by_top11['weight']:.2f} (pts={best_by_top11['top11_pts']:.1f})")

    # Step 4: Per-position weight optimization
    log.info("")
    log.info("=" * 60)
    log.info("Per-Position Weight Optimization")
    log.info("=" * 60)
    pos_candidate_weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    pos_results, best_pos_weights = run_per_position_search(raw_preds, pos_candidate_weights)

    print("\n" + "=" * 80)
    print("PER-POSITION WEIGHT RESULTS")
    print("=" * 80)
    if not pos_results.empty:
        print(f"{'Position':>10} {'Best Weight':>12} {'MAE':>8}")
        print("-" * 34)
        for _, row in pos_results.iterrows():
            print(f"{row['position']:>10} {row['weight']:>12.2f} {row['mae']:>8.4f}")

        # Compare per-position vs best global
        print(f"\nPer-position weights: {best_pos_weights}")

        # Evaluate per-position result on full metrics
        gw_metrics_pp = []
        for gw, raw in sorted(raw_preds.items()):
            pred_df = _reblend_gw(raw, best_pos_weights)
            if pred_df.empty:
                continue
            m = _compute_metrics(pred_df)
            if m:
                gw_metrics_pp.append(m)

        if gw_metrics_pp:
            n_pp = len(gw_metrics_pp)
            pp_mae = np.mean([m["mae"] for m in gw_metrics_pp])
            pp_sp = np.mean([m["spearman"] for m in gw_metrics_pp])
            pp_top11 = np.mean([m["top11_pts"] for m in gw_metrics_pp])
            pp_cap = sum(m["captain_in_top3"] for m in gw_metrics_pp) / n_pp * 100
            print(f"\nPer-position overall: MAE={pp_mae:.4f} Spearman={pp_sp:.4f} "
                  f"Top11={pp_top11:.1f} Captain={pp_cap:.1f}%")

    # Step 5: Summary recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    current_w = ensemble.decomposed_weight
    current_row = results[results["weight"] == current_w]
    if not current_row.empty:
        current_mae = current_row.iloc[0]["mae"]
        best_mae_val = best_row["mae"]
        improvement = current_mae - best_mae_val
        print(f"Current weight: {current_w:.2f} (MAE={current_mae:.4f})")
        print(f"Best weight:    {best_row['weight']:.2f} (MAE={best_mae_val:.4f})")
        print(f"MAE improvement: {improvement:.4f} ({'better' if improvement > 0 else 'no improvement'})")

        if improvement > 0.001:
            print(f"\nSuggestion: Update decomposed_weight to {best_row['weight']:.2f} in src/config.py")
        else:
            print(f"\nSuggestion: Current weight {current_w:.2f} is near-optimal. No change needed.")


if __name__ == "__main__":
    main()
