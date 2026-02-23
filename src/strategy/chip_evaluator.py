"""Chip evaluation across all remaining GWs.

Ported from v1 strategy.py ChipEvaluator class.
Evaluates BB, TC, FH, WC values using model predictions (near-term)
and fixture heuristics (far-term), with synergy detection (WC->BB, FH+WC).
"""

from __future__ import annotations

import pandas as pd

from src.config import strategy_cfg
from src.logging_config import get_logger
from src.solver.squad import solve_milp_team

logger = get_logger(__name__)

# ── Heuristic base values (calibrated to match near-term predictions) ──
_BB_HEURISTIC_BASE = 8.0
_TC_HEURISTIC_BASE = 5.0
_FH_HEURISTIC_BASE = 3.0
_WC_HEURISTIC_BASE = 3.0

# ── Synergy parameters ────────────────────────────────────────────────
_BB_SYNERGY_BOOST = 0.3  # 30% boost from WC->BB squad optimization
_FH_WC_MIN_COMBINED = 10  # Minimum combined value to report FH+WC synergy
_MAX_SYNERGIES = 10


class ChipEvaluator:
    """Evaluate chip value across all remaining GWs in the season."""

    def evaluate_all_chips(
        self,
        current_squad_ids: set[int],
        total_budget: float,
        available_chips: set[str],
        future_predictions: dict[int, pd.DataFrame],
        fixture_calendar: list[dict],
    ) -> dict[str, dict[int, float]]:
        """Return {chip_name: {gw: estimated_value}} for all remaining GWs.

        For GWs within the prediction horizon, uses model predictions.
        For GWs beyond, uses fixture-calendar heuristics only.
        """
        if not available_chips:
            return {}

        # Build fixture lookup: gw -> {team_id -> fixture_info}
        fx_by_gw = self._build_fixture_lookup(fixture_calendar)

        # Get the set of GWs we have predictions for
        pred_gws = sorted(future_predictions.keys())
        if not pred_gws:
            return {}

        # All GWs from fixture calendar
        all_gws = sorted(fx_by_gw.keys())
        if not all_gws:
            all_gws = pred_gws

        # Filter to current half-season only (chips expire at half boundary)
        if all_gws:
            current_gw = min(pred_gws) if pred_gws else min(all_gws)
            if current_gw <= 19:
                all_gws = [gw for gw in all_gws if gw <= 19]
            else:
                all_gws = [gw for gw in all_gws if gw >= 20]

        chip_values: dict[str, dict[int, float]] = {}

        if "bboost" in available_chips:
            chip_values["bboost"] = self._evaluate_bench_boost(
                current_squad_ids, total_budget, future_predictions,
                fx_by_gw, all_gws, pred_gws,
            )

        if "3xc" in available_chips:
            chip_values["3xc"] = self._evaluate_triple_captain(
                current_squad_ids, future_predictions, fx_by_gw,
                all_gws, pred_gws,
            )

        if "freehit" in available_chips:
            chip_values["freehit"] = self._evaluate_free_hit(
                current_squad_ids, total_budget, future_predictions,
                fx_by_gw, all_gws, pred_gws,
            )

        if "wildcard" in available_chips:
            chip_values["wildcard"] = self._evaluate_wildcard(
                current_squad_ids, total_budget, future_predictions,
                fx_by_gw, all_gws, pred_gws,
            )

        # Late-season chip urgency: unused chips approaching expiry
        current_gw = min(pred_gws) if pred_gws else min(all_gws)
        if current_gw >= strategy_cfg.late_season_gw:
            # Urgency ramps from 1.0 at GW33 to ~1.6 at GW38
            gws_past = current_gw - strategy_cfg.late_season_gw
            urgency = 1.0 + 0.12 * gws_past
            for chip_name in chip_values:
                for gw in chip_values[chip_name]:
                    chip_values[chip_name][gw] = round(
                        chip_values[chip_name][gw] * urgency, 1,
                    )

        return chip_values

    # ── Fixture helpers ───────────────────────────────────────────────

    def _build_fixture_lookup(
        self, fixture_calendar: list[dict],
    ) -> dict[int, dict[int, dict]]:
        """Build gw -> {team_id -> fixture_info} lookup."""
        fx_by_gw: dict[int, dict[int, dict]] = {}
        for f in fixture_calendar:
            gw = f["gameweek"]
            tid = f["team_id"]
            if gw not in fx_by_gw:
                fx_by_gw[gw] = {}
            fx_by_gw[gw][tid] = {
                "fixture_count": f.get("fixture_count", 1),
                "is_dgw": f.get("is_dgw", 0),
                "is_bgw": f.get("is_bgw", 0),
                "fdr_avg": f.get("fdr_avg"),
            }
        return fx_by_gw

    def _count_dgw_teams(self, fx_by_gw: dict, gw: int) -> int:
        """Count how many teams have a DGW in the given GW."""
        if gw not in fx_by_gw:
            return 0
        return sum(1 for t in fx_by_gw[gw].values() if t.get("is_dgw"))

    def _count_bgw_teams(self, fx_by_gw: dict, gw: int) -> int:
        """Count how many teams have a BGW in the given GW."""
        if gw not in fx_by_gw:
            return 0
        return sum(1 for t in fx_by_gw[gw].values() if t.get("is_bgw"))

    def _avg_fdr(self, fx_by_gw: dict, gw: int) -> float:
        """Average FDR across all teams with fixtures in a GW."""
        if gw not in fx_by_gw:
            return 3.0
        fdrs = [
            t["fdr_avg"]
            for t in fx_by_gw[gw].values()
            if t.get("fdr_avg") is not None and not t.get("is_bgw")
        ]
        return sum(fdrs) / len(fdrs) if fdrs else 3.0

    # ── Per-chip evaluation ───────────────────────────────────────────

    def _evaluate_bench_boost(
        self, current_squad_ids, total_budget, future_predictions,
        fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Bench Boost value per GW.

        Within prediction horizon: uses current squad bench.
        Beyond prediction horizon: DGW count heuristic.
        """
        from src.strategy.transfer_planner import MultiWeekPlanner

        values: dict[int, float] = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]

                # Use current squad's bench for all prediction GWs
                squad_preds = gw_df[
                    gw_df["player_id"].isin(current_squad_ids)
                ]
                if len(squad_preds) >= 11:
                    xi = MultiWeekPlanner._select_formation_xi(squad_preds)
                    xi_ids = set(xi.index)
                    bench = squad_preds.loc[
                        ~squad_preds.index.isin(xi_ids)
                    ]
                    bench_pts = bench["predicted_points"].sum()
                else:
                    bench_pts = (
                        squad_preds["predicted_points"].sum() * 0.25
                        if not squad_preds.empty
                        else 0
                    )

                # DGW value is already captured in predicted_points
                values[gw] = round(bench_pts, 1)
            else:
                # Heuristic: base bench value boosted by DGW count
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                n_bgw = self._count_bgw_teams(fx_by_gw, gw)
                values[gw] = round(
                    _BB_HEURISTIC_BASE
                    * (1 + n_dgw * 0.4)
                    * (1 - n_bgw * 0.05),
                    1,
                )

        return values

    def _evaluate_triple_captain(
        self, current_squad_ids, future_predictions, fx_by_gw,
        all_gws, pred_gws,
    ) -> dict[int, float]:
        """Triple Captain value per GW.

        Within prediction horizon: best player's predicted points (extra 1x).
        DGW players get a 30% boost since conservative predictions
        undervalue TC timing on double gameweeks.
        Beyond: heuristic based on DGW + low FDR.
        """
        from src.strategy.transfer_planner import MultiWeekPlanner

        values: dict[int, float] = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                # TC can only be used on a starter in your squad
                candidates = gw_df[
                    gw_df["player_id"].isin(current_squad_ids)
                ]
                xi = (
                    MultiWeekPlanner._select_formation_xi(candidates)
                    if not candidates.empty
                    else candidates
                )
                if not xi.empty:
                    # Use captain_score to identify the best captain, but
                    # the TC chip value is the extra predicted_points
                    # (one additional multiply: 3x instead of 2x).
                    score_col = (
                        "captain_score"
                        if "captain_score" in xi.columns
                        else "predicted_points"
                    )
                    best_idx = xi[score_col].idxmax()
                    best = xi.loc[best_idx, "predicted_points"]

                    # DGW boost: TC on a DGW player is more valuable
                    # because the player scores in two matches.
                    n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                    if n_dgw > 0 and "team_code" in xi.columns:
                        candidate_row = xi.loc[best_idx]
                        tc_code = candidate_row.get("team_code")
                        if tc_code and gw in fx_by_gw:
                            tc_fx = fx_by_gw[gw].get(int(tc_code), {})
                            if tc_fx.get("is_dgw"):
                                best *= 1.3

                    values[gw] = round(best, 1)
                else:
                    values[gw] = 0.0
            else:
                # Heuristic: base TC value from premium captain
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                avg_fdr = self._avg_fdr(fx_by_gw, gw)
                fdr_factor = max(0, (3.5 - avg_fdr) / 2)
                values[gw] = round(
                    _TC_HEURISTIC_BASE * (1 + n_dgw * 0.4) * (1 + fdr_factor),
                    1,
                )

        return values

    def _evaluate_free_hit(
        self, current_squad_ids, total_budget, future_predictions,
        fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Free Hit value per GW.

        Within prediction horizon: unconstrained best XI minus current squad
        points. Beyond: heuristic based on BGW count.
        """
        from src.strategy.transfer_planner import MultiWeekPlanner

        values: dict[int, float] = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                # Current squad points this GW (MILP for formation-correct
                # baseline)
                squad_preds = gw_df[
                    gw_df["player_id"].isin(current_squad_ids)
                ]
                current_pts = 0.0
                cap_col = (
                    "captain_score"
                    if "captain_score" in gw_df.columns
                    else None
                )
                if (
                    not squad_preds.empty
                    and "position" in squad_preds.columns
                    and "cost" in squad_preds.columns
                ):
                    curr_result = solve_milp_team(
                        squad_preds, "predicted_points",
                        budget=9999, captain_col=cap_col,
                    )
                    current_pts = (
                        curr_result["starting_points"] if curr_result else 0
                    )
                if current_pts == 0 and not squad_preds.empty:
                    current_pts = (
                        MultiWeekPlanner._squad_points_with_captain(
                            squad_preds,
                        )
                    )

                # Solve unconstrained best XI
                pool = gw_df.copy()
                if "position" in pool.columns and "cost" in pool.columns:
                    fh_result = solve_milp_team(
                        pool, "predicted_points",
                        budget=total_budget, captain_col=cap_col,
                    )
                    if fh_result:
                        fh_pts = fh_result["starting_points"]
                        values[gw] = round(max(0, fh_pts - current_pts), 1)
                    else:
                        values[gw] = 0.0
                else:
                    # Can't solve without position/cost, use estimate
                    all_top11 = gw_df.nlargest(
                        11, "predicted_points",
                    )["predicted_points"].sum()
                    values[gw] = round(max(0, all_top11 - current_pts), 1)
            else:
                # Heuristic: FH is great in BGWs, moderate otherwise
                n_bgw = self._count_bgw_teams(fx_by_gw, gw)
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                values[gw] = round(
                    _FH_HEURISTIC_BASE + n_bgw * 2.0 + n_dgw * 0.5, 1,
                )

        return values

    def _evaluate_wildcard(
        self, current_squad_ids, total_budget, future_predictions,
        fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Wildcard value per GW.

        Within prediction horizon: solve best squad over next 3 GWs minus
        current. Beyond: heuristic based on upcoming fixture swings.
        """
        from src.strategy.transfer_planner import MultiWeekPlanner

        values: dict[int, float] = {}

        for gw in all_gws:
            if gw in future_predictions:
                # Sum predictions over next 3 GWs from this point
                look_ahead_gws = [
                    g for g in pred_gws if gw <= g <= gw + 2
                ]
                if not look_ahead_gws:
                    values[gw] = 0.0
                    continue

                # Build combined prediction (sum over 3 GWs)
                combined = None
                for lag in look_ahead_gws:
                    if lag in future_predictions:
                        lag_df = future_predictions[lag][
                            ["player_id", "predicted_points"]
                        ].copy()
                        lag_df = lag_df.rename(
                            columns={"predicted_points": f"pts_{lag}"},
                        )
                        if combined is None:
                            combined = lag_df
                        else:
                            combined = combined.merge(
                                lag_df, on="player_id", how="outer",
                            )

                if combined is None:
                    values[gw] = 0.0
                    continue

                pts_cols = [
                    c for c in combined.columns if c.startswith("pts_")
                ]
                combined["total_pts"] = combined[pts_cols].sum(axis=1)

                # Current squad value over these GWs (formation-constrained)
                first_gw_preds = future_predictions[look_ahead_gws[0]]
                squad_combined = combined[
                    combined["player_id"].isin(current_squad_ids)
                ].copy()
                squad_combined = squad_combined.rename(
                    columns={"total_pts": "predicted_points"},
                )
                if "position" in first_gw_preds.columns:
                    pos_map = first_gw_preds.drop_duplicates(
                        "player_id",
                    ).set_index("player_id")["position"]
                    squad_combined["position"] = squad_combined[
                        "player_id"
                    ].map(pos_map)
                    squad_combined = squad_combined.dropna(
                        subset=["position"],
                    )
                if len(squad_combined) >= 11:
                    # Captain-aware points for fair WC comparison
                    current_3gw = (
                        MultiWeekPlanner._squad_points_with_captain(
                            squad_combined,
                        )
                    )
                else:
                    current_3gw = squad_combined["predicted_points"].sum()
                squad_combined = squad_combined.rename(
                    columns={"predicted_points": "total_pts"},
                )

                # Solve best squad for the combined period
                first_gw_df = future_predictions[look_ahead_gws[0]]
                meta_cols = [
                    c
                    for c in first_gw_df.columns
                    if c
                    in (
                        "player_id", "position", "cost",
                        "team_code", "team",
                    )
                ]
                if (
                    "position" in first_gw_df.columns
                    and "cost" in first_gw_df.columns
                ):
                    pool = combined.merge(
                        first_gw_df[meta_cols].drop_duplicates("player_id"),
                        on="player_id",
                        how="left",
                    )
                    pool = pool.dropna(subset=["position", "cost"])
                    # Include captain optimization in WC evaluation
                    wc_cap_col = (
                        "captain_score"
                        if "captain_score" in pool.columns
                        else None
                    )
                    wc_result = solve_milp_team(
                        pool, "total_pts",
                        budget=total_budget, captain_col=wc_cap_col,
                    )
                    if wc_result:
                        values[gw] = round(
                            max(
                                0,
                                wc_result["starting_points"] - current_3gw,
                            ),
                            1,
                        )
                    else:
                        values[gw] = 0.0
                else:
                    values[gw] = 0.0
            else:
                # Heuristic: WC value based on fixture difficulty swings
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                avg_fdr = self._avg_fdr(fx_by_gw, gw)
                fdr_improvement = max(0, (3.0 - avg_fdr))
                values[gw] = round(
                    _WC_HEURISTIC_BASE + n_dgw * 1.5 + fdr_improvement * 2,
                    1,
                )

        return values

    # ── Synergy evaluation ────────────────────────────────────────────

    def evaluate_chip_synergies(
        self,
        chip_values: dict[str, dict[int, float]],
        available_chips: set[str],
    ) -> list[dict]:
        """Evaluate chip pairings and synergies.

        WC in GW X + BB in GW X+1..X+3: WC value increases
        FH for BGW + WC for nearby DGW: complementary strategy
        """
        synergies: list[dict] = []

        if "wildcard" in available_chips and "bboost" in available_chips:
            wc_vals = chip_values.get("wildcard", {})
            bb_vals = chip_values.get("bboost", {})
            for wc_gw, wc_val in wc_vals.items():
                # Look for BB opportunities 1-3 GWs after WC
                for bb_offset in range(1, 4):
                    bb_gw = wc_gw + bb_offset
                    # Reject synergies crossing half-season boundary
                    if (wc_gw <= 19) != (bb_gw <= 19):
                        continue
                    if bb_gw in bb_vals:
                        bb_val = bb_vals[bb_gw]
                        # WC->BB synergy: WC can build a BB-optimized squad
                        synergy_bonus = bb_val * _BB_SYNERGY_BOOST
                        combined = wc_val + bb_val + synergy_bonus
                        synergies.append({
                            "chips": ["wildcard", "bboost"],
                            "gws": [wc_gw, bb_gw],
                            "individual_values": [
                                round(wc_val, 1),
                                round(bb_val, 1),
                            ],
                            "synergy_bonus": round(synergy_bonus, 1),
                            "combined_value": round(combined, 1),
                            "description": (
                                f"WC GW{wc_gw} -> BB GW{bb_gw}: "
                                "build BB-optimized squad"
                            ),
                        })

        if "freehit" in available_chips and "wildcard" in available_chips:
            fh_vals = chip_values.get("freehit", {})
            wc_vals = chip_values.get("wildcard", {})
            for fh_gw, fh_val in fh_vals.items():
                # FH for awkward GW, WC nearby to restructure
                for wc_offset in range(-2, 3):
                    if wc_offset == 0:
                        continue
                    wc_gw = fh_gw + wc_offset
                    # Reject synergies crossing half-season boundary
                    if (fh_gw <= 19) != (wc_gw <= 19):
                        continue
                    if wc_gw in wc_vals:
                        wc_val = wc_vals[wc_gw]
                        combined = fh_val + wc_val
                        if combined > _FH_WC_MIN_COMBINED:
                            synergies.append({
                                "chips": ["freehit", "wildcard"],
                                "gws": [fh_gw, wc_gw],
                                "individual_values": [
                                    round(fh_val, 1),
                                    round(wc_val, 1),
                                ],
                                "synergy_bonus": 0.0,
                                "combined_value": round(combined, 1),
                                "description": (
                                    f"FH GW{fh_gw} + WC GW{wc_gw}: "
                                    "complementary strategy"
                                ),
                            })

        # Sort by combined value descending
        synergies.sort(key=lambda s: s["combined_value"], reverse=True)
        return synergies[:_MAX_SYNERGIES]
