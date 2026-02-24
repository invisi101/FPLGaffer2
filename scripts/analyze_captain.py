#!/usr/bin/env python3
"""Analyze Q80 captain score accuracy and test alternative captain formulas.

Runs one backtest pass with raw predictions saved, then evaluates:
1. Q80 calibration: when Q80 = X, how often do players exceed X?
2. Captain hit rate under different weight formulas
3. Q80/mean ratio analysis: are extreme ratios reliable?
4. Total captain points per strategy across all backtest GWs

Usage:
    .venv/bin/python scripts/analyze_captain.py [--start-gw 10] [--end-gw 27]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensemble
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
# Build per-GW prediction DataFrames from raw backtest data
# ---------------------------------------------------------------------------

def build_gw_df(raw: dict, blend_weight: float = 0.30) -> pd.DataFrame:
    """Merge mean, decomp, Q80, and actuals into one DataFrame for a GW."""
    mean_df = raw["mean"]
    decomp_df = raw["decomp"]
    actuals = raw["actuals"]
    q80_df = raw["q80"]

    if mean_df.empty:
        return pd.DataFrame()

    result = mean_df[["player_id", "mean_pred", "position"]].copy()

    # Merge decomposed
    if not decomp_df.empty:
        result = result.merge(
            decomp_df[["player_id", "decomp_pred"]],
            on="player_id", how="outer",
        )
    else:
        result["decomp_pred"] = np.nan

    # Blend to get predicted_next_gw_points
    w_d = blend_weight
    result["predicted_next_gw_points"] = np.where(
        result["decomp_pred"].notna() & result["mean_pred"].notna(),
        w_d * result["decomp_pred"] + (1 - w_d) * result["mean_pred"],
        result["mean_pred"].fillna(result["decomp_pred"]),
    )

    # Add Q80
    if not q80_df.empty and "predicted_next_gw_points_q80" in q80_df.columns:
        result = result.merge(
            q80_df[["player_id", "predicted_next_gw_points_q80"]],
            on="player_id", how="left",
        )
    else:
        result["predicted_next_gw_points_q80"] = np.nan

    # Add actuals
    result["actual"] = result["player_id"].map(actuals)
    result = result.dropna(subset=["actual"])

    return result


def compute_captain_score(
    df: pd.DataFrame,
    mean_weight: float,
    q80_weight: float,
    q80_cap: float | None = None,
) -> pd.Series:
    """Compute captain score with given weights and optional Q80 cap."""
    mean = df["predicted_next_gw_points"]
    q80 = df["predicted_next_gw_points_q80"].fillna(mean)

    if q80_cap is not None:
        q80 = q80.clip(upper=q80_cap * mean.clip(lower=0.5))

    return mean_weight * mean + q80_weight * q80


# ---------------------------------------------------------------------------
# Test 1: Q80 Calibration
# ---------------------------------------------------------------------------

def test_q80_calibration(all_gw_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """When Q80 = X, do players exceed X about 20% of the time?

    Bins predictions by Q80 value and checks actual exceedance rates.
    """
    rows = []
    for gw, df in sorted(all_gw_dfs.items()):
        q80_mask = df["predicted_next_gw_points_q80"].notna()
        gw_df = df[q80_mask].copy()
        if gw_df.empty:
            continue
        for _, row in gw_df.iterrows():
            rows.append({
                "gw": gw,
                "player_id": row["player_id"],
                "position": row["position"],
                "mean_pred": row["predicted_next_gw_points"],
                "q80_pred": row["predicted_next_gw_points_q80"],
                "actual": row["actual"],
                "exceeded_q80": row["actual"] >= row["predicted_next_gw_points_q80"],
                "exceeded_mean": row["actual"] >= row["predicted_next_gw_points"],
            })

    if not rows:
        return pd.DataFrame()

    pool = pd.DataFrame(rows)

    # Bin by Q80 prediction level
    bins = [0, 2, 4, 6, 8, 10, 15, 50]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15+"]
    pool["q80_bin"] = pd.cut(pool["q80_pred"], bins=bins, labels=labels, right=False)

    summary = pool.groupby("q80_bin", observed=True).agg(
        count=("exceeded_q80", "size"),
        exceedance_rate=("exceeded_q80", "mean"),
        avg_q80=("q80_pred", "mean"),
        avg_actual=("actual", "mean"),
        avg_mean=("mean_pred", "mean"),
    ).reset_index()

    # Overall calibration
    overall_rate = pool["exceeded_q80"].mean()
    summary.attrs["overall_exceedance"] = overall_rate
    summary.attrs["expected_exceedance"] = 0.20
    summary.attrs["n_total"] = len(pool)

    return summary


# ---------------------------------------------------------------------------
# Test 2: Q80/Mean Ratio Analysis
# ---------------------------------------------------------------------------

def test_q80_ratio(all_gw_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Analyze players where Q80/mean is extreme. Do they actually deliver?"""
    rows = []
    for gw, df in sorted(all_gw_dfs.items()):
        q80_mask = df["predicted_next_gw_points_q80"].notna()
        gw_df = df[q80_mask].copy()
        if gw_df.empty:
            continue
        gw_df["ratio"] = gw_df["predicted_next_gw_points_q80"] / gw_df[
            "predicted_next_gw_points"
        ].clip(lower=0.5)
        for _, row in gw_df.iterrows():
            rows.append({
                "gw": gw,
                "player_id": row["player_id"],
                "position": row["position"],
                "mean_pred": row["predicted_next_gw_points"],
                "q80_pred": row["predicted_next_gw_points_q80"],
                "ratio": row["ratio"],
                "actual": row["actual"],
            })

    if not rows:
        return pd.DataFrame()

    pool = pd.DataFrame(rows)

    # Bin by ratio
    bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 100]
    labels = ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0+"]
    pool["ratio_bin"] = pd.cut(pool["ratio"], bins=bins, labels=labels, right=False)

    summary = pool.groupby("ratio_bin", observed=True).agg(
        count=("actual", "size"),
        avg_mean=("mean_pred", "mean"),
        avg_q80=("q80_pred", "mean"),
        avg_actual=("actual", "mean"),
        avg_ratio=("ratio", "mean"),
        pct_beat_mean=("actual", lambda x: (x > pool.loc[x.index, "mean_pred"]).mean()),
    ).reset_index()

    return summary


# ---------------------------------------------------------------------------
# Test 3: Captain Formula Comparison
# ---------------------------------------------------------------------------

def test_captain_formulas(
    all_gw_dfs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Test different captain selection strategies and measure total points."""

    strategies = {
        "pure_mean":     {"mean_w": 1.0, "q80_w": 0.0, "cap": None},
        "0.7/0.3":       {"mean_w": 0.7, "q80_w": 0.3, "cap": None},
        "0.5/0.5":       {"mean_w": 0.5, "q80_w": 0.5, "cap": None},
        "0.4/0.6 (cur)": {"mean_w": 0.4, "q80_w": 0.6, "cap": None},
        "0.3/0.7":       {"mean_w": 0.3, "q80_w": 0.7, "cap": None},
        "pure_q80":      {"mean_w": 0.0, "q80_w": 1.0, "cap": None},
        "0.4/0.6 cap2x": {"mean_w": 0.4, "q80_w": 0.6, "cap": 2.0},
        "0.4/0.6 cap3x": {"mean_w": 0.4, "q80_w": 0.6, "cap": 3.0},
    }

    results = []
    gw_details = {name: [] for name in strategies}

    for name, params in strategies.items():
        total_captain_pts = 0
        total_best_captain_pts = 0
        captain_correct = 0
        captain_in_top3 = 0
        captain_in_top5 = 0
        n_gws = 0

        for gw, df in sorted(all_gw_dfs.items()):
            if df.empty:
                continue

            # Compute captain score for this strategy
            cap_score = compute_captain_score(
                df, params["mean_w"], params["q80_w"], params["cap"],
            )
            df = df.copy()
            df["test_captain_score"] = cap_score

            # Pick captain: highest captain score among players with mean > 0
            viable = df[df["predicted_next_gw_points"] > 0]
            if viable.empty:
                continue

            captain = viable.nlargest(1, "test_captain_score").iloc[0]
            captain_pts = captain["actual"] * 2  # Captain doubles

            # Best possible captain
            best_captain = viable.nlargest(1, "actual").iloc[0]
            best_pts = best_captain["actual"] * 2

            # Top actual scorers
            top3_ids = set(viable.nlargest(3, "actual")["player_id"])
            top5_ids = set(viable.nlargest(5, "actual")["player_id"])

            total_captain_pts += captain_pts
            total_best_captain_pts += best_pts
            captain_correct += (captain["player_id"] == best_captain["player_id"])
            captain_in_top3 += (captain["player_id"] in top3_ids)
            captain_in_top5 += (captain["player_id"] in top5_ids)
            n_gws += 1

            gw_details[name].append({
                "gw": gw,
                "captain_name": captain.get("player_id"),
                "captain_pts": captain_pts,
                "best_pts": best_pts,
                "captain_mean": captain["predicted_next_gw_points"],
                "captain_q80": captain.get("predicted_next_gw_points_q80"),
                "captain_actual": captain["actual"],
            })

        if n_gws > 0:
            results.append({
                "strategy": name,
                "total_captain_pts": total_captain_pts,
                "total_best_pts": total_best_captain_pts,
                "capture_pct": round(total_captain_pts / total_best_captain_pts * 100, 1)
                if total_best_captain_pts > 0 else 0,
                "pts_per_gw": round(total_captain_pts / n_gws, 2),
                "correct_pct": round(captain_correct / n_gws * 100, 1),
                "top3_pct": round(captain_in_top3 / n_gws * 100, 1),
                "top5_pct": round(captain_in_top5 / n_gws * 100, 1),
                "n_gws": n_gws,
            })

    return pd.DataFrame(results), gw_details


# ---------------------------------------------------------------------------
# Test 4: Per-GW Captain Comparison (pure_mean vs current formula)
# ---------------------------------------------------------------------------

def print_gw_captain_comparison(
    gw_details: dict,
    all_gw_dfs: dict[int, pd.DataFrame],
):
    """Show per-GW detail where pure_mean and current formula disagree."""
    mean_gws = {d["gw"]: d for d in gw_details.get("pure_mean", [])}
    current_gws = {d["gw"]: d for d in gw_details.get("0.4/0.6 (cur)", [])}

    disagreements = []
    for gw in sorted(mean_gws.keys()):
        m = mean_gws.get(gw)
        c = current_gws.get(gw)
        if not m or not c:
            continue
        if m["captain_name"] != c["captain_name"]:
            diff = c["captain_pts"] - m["captain_pts"]
            disagreements.append({
                "gw": gw,
                "mean_captain": m["captain_name"],
                "mean_captain_pts": m["captain_pts"],
                "mean_captain_pred": m["captain_mean"],
                "q80_captain": c["captain_name"],
                "q80_captain_pts": c["captain_pts"],
                "q80_captain_pred": c["captain_mean"],
                "q80_captain_q80": c["captain_q80"],
                "pts_diff": diff,
            })

    return disagreements


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Q80 captain score accuracy",
    )
    parser.add_argument("--start-gw", type=int, default=10,
                        help="First GW to backtest (default: 10)")
    parser.add_argument("--end-gw", type=int, default=27,
                        help="Last GW to backtest (default: 27)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Captain Score Analysis — Q80 Evaluation")
    log.info("=" * 70)
    log.info("Current formula: captain_score = %.1f * mean + %.1f * Q80",
             ensemble.captain_mean_weight, ensemble.captain_q80_weight)

    # Step 1: Load data + build features
    log.info("Loading data...")
    data = load_all_data()
    log.info("Building features...")
    df = build_features(data)
    log.info("Features built: %d rows", len(df))

    # Step 2: Run backtest with raw predictions
    log.info("Running backtest (GW%d-%d) with raw predictions...",
             args.start_gw, args.end_gw)
    t0 = time.time()
    bt_result = run_backtest(
        df, start_gw=args.start_gw, end_gw=args.end_gw,
        save_raw_preds=True,
    )
    elapsed = time.time() - t0
    log.info("Backtest complete in %.1f seconds", elapsed)

    raw_preds = bt_result.get("raw_preds", {})
    if not raw_preds:
        log.error("No raw predictions saved.")
        sys.exit(1)

    # Step 3: Build per-GW DataFrames
    log.info("Building per-GW prediction DataFrames...")
    all_gw_dfs = {}
    for gw, raw in sorted(raw_preds.items()):
        gw_df = build_gw_df(raw, blend_weight=ensemble.decomposed_weight)
        if not gw_df.empty:
            all_gw_dfs[gw] = gw_df

    log.info("Built DataFrames for %d GWs", len(all_gw_dfs))

    # ===================================================================
    # TEST 1: Q80 Calibration
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Q80 CALIBRATION")
    print("If Q80 is the 80th percentile, players should exceed it ~20% of the time")
    print("=" * 80)

    cal = test_q80_calibration(all_gw_dfs)
    if not cal.empty:
        print(f"\n{'Q80 Bin':>10} {'Count':>7} {'Exceed%':>8} {'Avg Q80':>8} "
              f"{'Avg Mean':>9} {'Avg Actual':>11}")
        print("-" * 63)
        for _, row in cal.iterrows():
            print(f"{row['q80_bin']:>10} {int(row['count']):>7} "
                  f"{row['exceedance_rate']*100:>7.1f}% {row['avg_q80']:>8.1f} "
                  f"{row['avg_mean']:>9.2f} {row['avg_actual']:>11.2f}")

        overall = cal.attrs.get("overall_exceedance", 0)
        n_total = cal.attrs.get("n_total", 0)
        print(f"\nOverall: {overall*100:.1f}% exceed Q80 (expected ~20%) "
              f"— n={n_total}")
        if overall < 0.15:
            print("VERDICT: Q80 is too HIGH (overpredicting upside)")
        elif overall > 0.25:
            print("VERDICT: Q80 is too LOW (underpredicting upside)")
        else:
            print("VERDICT: Q80 is reasonably calibrated")

    # ===================================================================
    # TEST 2: Q80/Mean Ratio Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Q80/MEAN RATIO ANALYSIS")
    print("Do players with extreme Q80/mean ratios actually deliver?")
    print("=" * 80)

    ratio_df = test_q80_ratio(all_gw_dfs)
    if not ratio_df.empty:
        print(f"\n{'Ratio Bin':>10} {'Count':>7} {'Avg Mean':>9} {'Avg Q80':>8} "
              f"{'Avg Actual':>11} {'Beat Mean%':>11}")
        print("-" * 66)
        for _, row in ratio_df.iterrows():
            print(f"{row['ratio_bin']:>10} {int(row['count']):>7} "
                  f"{row['avg_mean']:>9.2f} {row['avg_q80']:>8.1f} "
                  f"{row['avg_actual']:>11.2f} {row['pct_beat_mean']*100:>10.1f}%")

    # ===================================================================
    # TEST 3: Captain Formula Comparison
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 3: CAPTAIN FORMULA COMPARISON")
    print("Total captain points across all backtest GWs under each strategy")
    print("=" * 80)

    formula_results, gw_details = test_captain_formulas(all_gw_dfs)
    if not formula_results.empty:
        print(f"\n{'Strategy':>18} {'Total Pts':>10} {'Pts/GW':>8} {'Capture%':>9} "
              f"{'Correct%':>9} {'Top3%':>7} {'Top5%':>7}")
        print("-" * 78)
        for _, row in formula_results.iterrows():
            marker = " <--" if row["strategy"] == "0.4/0.6 (cur)" else ""
            print(f"{row['strategy']:>18} {row['total_captain_pts']:>10.0f} "
                  f"{row['pts_per_gw']:>8.2f} {row['capture_pct']:>8.1f}% "
                  f"{row['correct_pct']:>8.1f}% {row['top3_pct']:>6.1f}% "
                  f"{row['top5_pct']:>6.1f}%{marker}")

        # Find best
        best = formula_results.loc[formula_results["total_captain_pts"].idxmax()]
        worst = formula_results.loc[formula_results["total_captain_pts"].idxmin()]
        current = formula_results[formula_results["strategy"] == "0.4/0.6 (cur)"]

        print(f"\nBest:    {best['strategy']} ({best['total_captain_pts']:.0f} pts)")
        print(f"Worst:   {worst['strategy']} ({worst['total_captain_pts']:.0f} pts)")
        if not current.empty:
            cur = current.iloc[0]
            diff = best["total_captain_pts"] - cur["total_captain_pts"]
            print(f"Current: {cur['strategy']} ({cur['total_captain_pts']:.0f} pts)")
            if diff > 0:
                print(f"Gap: {diff:.0f} pts over {cur['n_gws']:.0f} GWs "
                      f"({diff/cur['n_gws']:.1f} pts/GW)")

    # ===================================================================
    # TEST 4: Per-GW Disagreements
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 4: GWs WHERE PURE MEAN vs Q80 FORMULA DISAGREE")
    print("Shows each GW where the two strategies picked different captains")
    print("=" * 80)

    disagreements = print_gw_captain_comparison(gw_details, all_gw_dfs)
    if disagreements:
        q80_wins = sum(1 for d in disagreements if d["pts_diff"] > 0)
        mean_wins = sum(1 for d in disagreements if d["pts_diff"] < 0)
        draws = sum(1 for d in disagreements if d["pts_diff"] == 0)

        print(f"\n{'GW':>4} {'Mean Pick':>12} {'Pts':>5} {'Pred':>6} "
              f"{'Q80 Pick':>12} {'Pts':>5} {'Pred':>6} {'Q80':>6} {'Diff':>6}")
        print("-" * 78)
        for d in disagreements:
            diff_str = f"+{d['pts_diff']:.0f}" if d['pts_diff'] > 0 else f"{d['pts_diff']:.0f}"
            winner = "Q80" if d["pts_diff"] > 0 else "MEAN" if d["pts_diff"] < 0 else "DRAW"
            print(f"{d['gw']:>4} {d['mean_captain']:>12} {d['mean_captain_pts']:>5.0f} "
                  f"{d['mean_captain_pred']:>6.2f} "
                  f"{d['q80_captain']:>12} {d['q80_captain_pts']:>5.0f} "
                  f"{d['q80_captain_pred']:>6.2f} "
                  f"{d['q80_captain_q80']:>6.1f} {diff_str:>6} {winner}")

        total_diff = sum(d["pts_diff"] for d in disagreements)
        print(f"\nDisagreements: {len(disagreements)} GWs")
        print(f"Q80 wins: {q80_wins}, Mean wins: {mean_wins}, Draws: {draws}")
        print(f"Net pts from using Q80 formula: {total_diff:+.0f}")
    else:
        print("\nNo disagreements found — both strategies always pick the same captain.")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if not formula_results.empty:
        best = formula_results.loc[formula_results["total_captain_pts"].idxmax()]
        print(f"Best captain strategy: {best['strategy']}")
        print(f"  Total pts: {best['total_captain_pts']:.0f}")
        print(f"  Pts/GW: {best['pts_per_gw']:.2f}")
        print(f"  Top-3 accuracy: {best['top3_pct']:.1f}%")

    if not cal.empty:
        overall = cal.attrs.get("overall_exceedance", 0)
        print(f"\nQ80 calibration: {overall*100:.1f}% exceedance (target: 20%)")


if __name__ == "__main__":
    main()
