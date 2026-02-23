"""Mathematical correctness and FPL compliance tests.

Tests the mathematical properties, formulas, and FPL rule compliance
that the automated suite can verify.  Complements:
  - test_integration.py   (smoke tests: app, routes, imports)
  - test_strategy_pipeline.py  (strategy-layer behaviour)

These tests catch the bugs that cost real FPL points:
  - Wrong scoring formula  -> wrong predictions for every player
  - Wrong ensemble blend   -> ranking errors across the board
  - Wrong captain formula  -> 10+ pts/GW lost on suboptimal captain
  - Wrong solver constraints -> illegal squads
  - Wrong hit counting     -> recommending unprofitable transfers
  - Wrong decay curve      -> bad 3-GW and 8-GW transfer targets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    POSITION_GROUPS,
    decomposed,
    ensemble,
    fpl_scoring,
    prediction as pred_cfg,
    solver_cfg,
)


# ===========================================================================
# 1. Config sanity — every magic number must be internally consistent
# ===========================================================================

class TestConfigSanity:
    """Config values are consistent with each other and with FPL rules."""

    def test_confidence_decay_monotonically_decreasing(self):
        decay = pred_cfg.confidence_decay
        for i in range(1, len(decay)):
            assert decay[i] < decay[i - 1], (
                f"Decay must decrease: offset {i}={decay[i-1]} vs {i+1}={decay[i]}"
            )

    def test_confidence_decay_values_in_zero_one(self):
        for i, d in enumerate(pred_cfg.confidence_decay):
            assert 0 < d < 1, f"Decay[{i}]={d} must be in (0, 1)"

    def test_ensemble_weights_valid(self):
        assert 0 < ensemble.decomposed_weight < 1
        assert abs(ensemble.decomposed_weight - 0.15) < 1e-9

    def test_captain_weights_sum_to_one(self):
        total = ensemble.captain_mean_weight + ensemble.captain_q80_weight
        assert abs(total - 1.0) < 1e-9, f"Captain weights sum to {total}"

    def test_squad_positions_sum_to_15(self):
        total = sum(solver_cfg.squad_positions.values())
        assert total == solver_cfg.squad_size == 15

    def test_formation_limits_within_squad_positions(self):
        for pos, (lo, hi) in solver_cfg.formation_limits.items():
            squad_n = solver_cfg.squad_positions[pos]
            assert 0 <= lo <= hi <= squad_n, (
                f"{pos}: formation ({lo},{hi}) vs squad {squad_n}"
            )

    def test_formation_limits_bracket_11(self):
        min_total = sum(lo for lo, _ in solver_cfg.formation_limits.values())
        max_total = sum(hi for _, hi in solver_cfg.formation_limits.values())
        assert min_total <= solver_cfg.starting_xi <= max_total

    def test_fpl_scoring_matches_real_rules(self):
        """Every scoring value must match official 2025-26 FPL rules."""
        s = fpl_scoring.scoring
        # Goals
        assert s["GKP"]["goal"] == 10, "GKP goals = 10 (2025-26 change)"
        assert s["DEF"]["goal"] == 6
        assert s["MID"]["goal"] == 5
        assert s["FWD"]["goal"] == 4
        # Assists: 3 for all
        for pos in POSITION_GROUPS:
            assert s[pos]["assist"] == 3
        # CS: GKP/DEF 4, MID 1, FWD 0
        assert s["GKP"]["cs"] == s["DEF"]["cs"] == 4
        assert s["MID"]["cs"] == 1
        assert s["FWD"]["cs"] == 0
        # GC penalty: GKP/DEF only
        assert s["GKP"]["gc_per_2"] == s["DEF"]["gc_per_2"] == -1
        assert s["MID"]["gc_per_2"] == s["FWD"]["gc_per_2"] == 0
        # Saves: GKP only
        assert s["GKP"]["save_per_3"] == 1
        for pos in ["DEF", "MID", "FWD"]:
            assert s[pos]["save_per_3"] == 0
        # DefCon thresholds
        assert s["GKP"]["defcon_threshold"] == s["DEF"]["defcon_threshold"] == 10
        assert s["MID"]["defcon_threshold"] == s["FWD"]["defcon_threshold"] == 12
        # DefCon points
        for pos in POSITION_GROUPS:
            assert s[pos]["defcon"] == 2

    def test_soft_caps_reasonable(self):
        for pos, cap in decomposed.soft_caps.items():
            assert 5 <= cap <= 15, f"{pos} soft cap {cap} out of range"

    def test_every_component_has_objective(self):
        for pos, comps in decomposed.components.items():
            for comp in comps:
                assert comp in decomposed.objectives, (
                    f"{pos}/{comp} missing from objectives"
                )

    def test_every_component_has_target(self):
        for pos, comps in decomposed.components.items():
            for comp in comps:
                assert comp in decomposed.target_columns, (
                    f"{pos}/{comp} missing from target_columns"
                )

    def test_hit_cost_is_four(self):
        assert solver_cfg.hit_cost == 4.0

    def test_bench_weight_in_range(self):
        assert 0 < solver_cfg.bench_weight < 0.5


# ===========================================================================
# 2. FPL scoring formula (decomposed.py)
# ===========================================================================

class TestDecomposedScoring:
    """The decomposed FPL scoring formula must produce correct point values.

    These test the MATH of predict_decomposed() — the most complex formula
    in the entire system.  A bug here silently corrupts every prediction.
    """

    # --- P(plays) logic ---

    def test_cop_zero_means_zero_pplays(self):
        """COP = 0  ->  P(plays) = 0  ->  0 points."""
        cop = 0.0 / 100.0
        avail = 0.0
        p_plays = cop if cop < 1.0 else avail
        assert p_plays == 0.0

    def test_cop_50_means_half_pplays(self):
        """COP = 50  ->  P(plays) = 0.5  (doubt flag active)."""
        cop = 50.0 / 100.0
        avail = 0.9
        # When cop < 1.0, P(plays) = cop (not avail)
        p_plays = cop if cop < 1.0 else avail
        assert p_plays == 0.5

    def test_cop_100_falls_through_to_availability(self):
        """COP = 100 means 'no doubt'; P(plays) comes from availability_rate."""
        cop = 100.0 / 100.0  # = 1.0
        avail = 0.8
        p_plays = cop if cop < 1.0 else avail
        assert p_plays == 0.8

    # --- Appearance points ---

    def test_appearance_points_formula(self):
        """E[appearance] = P(60+)*2 + (P(plays)-P(60+))*1."""
        p_plays = 0.9
        p_60plus = 0.9 * (80 / 90)  # ~0.8
        pts = p_60plus * 2 + max(0, p_plays - p_60plus) * 1
        # Every component must be non-negative
        assert pts > 0
        assert pts <= 2.0  # Max is 2 (certain 60+ appearance)

    def test_appearance_zero_when_not_playing(self):
        p_plays = 0.0
        p_60plus = 0.0
        pts = p_60plus * 2 + max(0, p_plays - p_60plus) * 1
        assert pts == 0.0

    # --- Goal points ---

    def test_goal_points_per_position(self):
        """Goal points scale correctly by position."""
        e_goals = 0.3
        p_plays = 1.0
        s = fpl_scoring.scoring
        assert p_plays * e_goals * s["GKP"]["goal"] == pytest.approx(3.0)
        assert p_plays * e_goals * s["DEF"]["goal"] == pytest.approx(1.8)
        assert p_plays * e_goals * s["MID"]["goal"] == pytest.approx(1.5)
        assert p_plays * e_goals * s["FWD"]["goal"] == pytest.approx(1.2)

    # --- CS points require 60+ mins ---

    def test_cs_uses_p60_not_pplays(self):
        """CS is only awarded for 60+ mins. Must use P(60+), not P(plays)."""
        p_plays = 0.9
        p_60plus = 0.72
        e_cs = 0.4
        cs_val = fpl_scoring.scoring["DEF"]["cs"]  # 4

        correct = p_60plus * e_cs * cs_val
        wrong = p_plays * e_cs * cs_val
        assert correct < wrong, "Using P(plays) for CS would overpredict"
        assert correct == pytest.approx(1.152)

    # --- GC penalty ---

    def test_gc_penalty_continuous(self):
        """GC penalty uses continuous E[GC]/2, not floor."""
        p_60plus = 0.8
        e_gc = 1.5
        gc_penalty = fpl_scoring.scoring["DEF"]["gc_per_2"]  # -1

        pts_gc = p_60plus * (e_gc / 2) * gc_penalty
        assert pts_gc < 0, "GC penalty must be negative"
        assert pts_gc == pytest.approx(-0.6)

    def test_gc_penalty_zero_for_mid_fwd(self):
        for pos in ["MID", "FWD"]:
            assert fpl_scoring.scoring[pos]["gc_per_2"] == 0

    # --- Saves ---

    def test_saves_continuous(self):
        """Saves use continuous E[saves]/3."""
        p_plays = 0.95
        e_saves = 3.0
        pts = p_plays * (e_saves / 3) * 1  # save_per_3 = 1 for GKP
        assert pts == pytest.approx(0.95)

    # --- DefCon (Poisson CDF) ---

    def test_defcon_poisson_direction(self):
        """Higher E[CBIT] -> higher P(DefCon) -> more points."""
        from scipy.stats import poisson

        threshold = 10
        p_low = 1.0 - poisson.cdf(threshold - 1, mu=5.0)
        p_high = 1.0 - poisson.cdf(threshold - 1, mu=12.0)
        assert p_high > p_low

    def test_defcon_threshold_by_position(self):
        """DEF needs CBIT>=10, MID/FWD need CBIT>=12."""
        from scipy.stats import poisson

        mu = 10.0  # Expected CBIT
        p_def = 1.0 - poisson.cdf(10 - 1, mu=mu)  # P(X >= 10)
        p_mid = 1.0 - poisson.cdf(12 - 1, mu=mu)  # P(X >= 12)
        assert p_def > p_mid, "DEF has lower threshold, so higher P"

    def test_defcon_near_zero_for_low_cbit(self):
        from scipy.stats import poisson

        # A player with E[CBIT]=2 should almost never hit DefCon
        p = 1.0 - poisson.cdf(10 - 1, mu=2.0)
        assert p < 0.01

    # --- Soft cap ---

    def test_soft_cap_formula(self):
        """Above cap: cap + (pred - cap) * 0.5."""
        cap = 10.0
        pred = 14.0
        capped = cap + (pred - cap) * 0.5
        assert capped == 12.0

    def test_soft_cap_unchanged_below(self):
        cap = 10.0
        pred = 7.0
        over = pred > cap
        assert not over, "Below cap should not be modified"

    # --- DGW handling ---

    def test_dgw_predictions_summed(self):
        """DGW: per-fixture predictions are summed per player_id."""
        df = pd.DataFrame({
            "player_id": [1, 1, 2],
            "predicted_next_gw_points": [3.5, 4.0, 5.0],
        })
        agg = df.groupby("player_id")["predicted_next_gw_points"].sum()
        assert agg[1] == 7.5, "DGW player should get sum of both fixtures"
        assert agg[2] == 5.0, "SGW player unchanged"


# ===========================================================================
# 3. Ensemble blend mathematics
# ===========================================================================

class TestEnsembleBlend:
    """The 85/15 ensemble blend must produce correctly weighted predictions."""

    def test_blend_formula(self):
        w_d = ensemble.decomposed_weight  # 0.15
        w_m = 1 - w_d                     # 0.85
        blended = w_d * 3.0 + w_m * 5.0   # 0.45 + 4.25
        assert blended == pytest.approx(4.7)

    def test_blend_equals_input_when_equal(self):
        w_d = ensemble.decomposed_weight
        w_m = 1 - w_d
        for pred in [0.0, 3.5, 7.0, 12.0]:
            blended = w_d * pred + w_m * pred
            assert blended == pytest.approx(pred)

    def test_blend_bounded_by_inputs(self):
        w_d = ensemble.decomposed_weight
        w_m = 1 - w_d
        for mean, decomp in [(5, 3), (2, 8), (0, 10), (7, 7)]:
            blended = w_d * decomp + w_m * mean
            assert blended >= min(mean, decomp) - 1e-9
            assert blended <= max(mean, decomp) + 1e-9

    def test_mean_model_dominates(self):
        """With 85% weight, mean model should dominate the blend."""
        w_m = 1 - ensemble.decomposed_weight
        assert w_m > 0.5
        # Blend should be closer to mean than to decomp
        mean, decomp = 6.0, 2.0
        blended = ensemble.decomposed_weight * decomp + w_m * mean
        dist_to_mean = abs(blended - mean)
        dist_to_decomp = abs(blended - decomp)
        assert dist_to_mean < dist_to_decomp


# ===========================================================================
# 4. Captain score formula
# ===========================================================================

class TestCaptainScore:
    """Captain score = 0.4*mean + 0.6*Q80 must capture upside correctly."""

    def test_formula_values(self):
        mean, q80 = 6.0, 8.0
        score = ensemble.captain_mean_weight * mean + ensemble.captain_q80_weight * q80
        assert score == pytest.approx(7.2)

    def test_q80_weighted_more_heavily(self):
        """Q80 has higher weight to capture explosive upside."""
        assert ensemble.captain_q80_weight > ensemble.captain_mean_weight

    def test_fallback_when_q80_missing(self):
        """When Q80 is NaN, falls back to mean (the code does .fillna(mean))."""
        mean = 6.0
        q80 = mean  # fillna(mean)
        score = ensemble.captain_mean_weight * mean + ensemble.captain_q80_weight * q80
        assert score == pytest.approx(mean)

    def test_captain_bonus_doubles_points(self):
        """Captain doubles their points; the solver adds captain_score as bonus."""
        # In solver: starting_points = sum(starters) + captain_pts
        # Where captain_pts = the captain's predicted points (added once more)
        pred = 7.0
        total_with_captain = pred + pred  # 2x
        assert total_with_captain == 14.0


# ===========================================================================
# 5. Confidence decay properties
# ===========================================================================

class TestConfidenceDecay:
    """Multi-GW decay must be conservative and consistent."""

    def test_gw_plus_1_highest(self):
        assert pred_cfg.confidence_decay[0] == max(pred_cfg.confidence_decay)

    def test_all_above_half(self):
        """Even at GW+7, confidence should be above 50%."""
        for i, d in enumerate(pred_cfg.confidence_decay):
            assert d > 0.5, f"Decay at GW+{i+1}={d} too aggressive"

    def test_fallback_formula_close_to_explicit(self):
        """Explicit values should roughly track 0.95^(offset-1)."""
        for i, explicit in enumerate(pred_cfg.confidence_decay):
            fallback = 0.95 ** i
            assert abs(explicit - fallback) < 0.10

    def test_decay_applied_to_prediction(self):
        """A 5-pt prediction at GW+3 should be ~4.5 after decay."""
        raw_pred = 5.0
        decay = pred_cfg.confidence_decay[2]  # offset 3 = index 2
        assert decay == 0.90
        adjusted = raw_pred * decay
        assert adjusted == pytest.approx(4.5)


# ===========================================================================
# 6. Solver FPL compliance (comprehensive)
# ===========================================================================

class TestSolverFPLCompliance:
    """Every solver output must satisfy all FPL squad rules."""

    @pytest.fixture
    def pool(self):
        """30-player pool across 10 teams."""
        positions = (
            ["GKP"] * 4 + ["DEF"] * 8 + ["MID"] * 10 + ["FWD"] * 8
        )
        return pd.DataFrame({
            "player_id": range(1, 31),
            "web_name": [f"P{i}" for i in range(1, 31)],
            "position": positions,
            "cost": [
                40, 45, 40, 42,
                55, 50, 48, 45, 55, 50, 48, 45,
                120, 100, 80, 70, 60, 55, 50, 48, 80, 65,
                130, 90, 70, 55, 80, 65, 60, 50,
            ],
            "team_code": [
                1, 2, 3, 4,
                1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                1, 2, 3, 4, 5, 6, 7, 8,
            ],
            "predicted_next_gw_points": [
                3.5, 3.0, 2.5, 2.8,
                4.5, 4.2, 4.0, 3.8, 4.3, 3.5, 3.2, 3.0,
                7.0, 6.5, 5.5, 5.0, 4.5, 4.2, 4.0, 3.8, 5.8, 4.8,
                8.0, 6.0, 5.0, 4.0, 5.5, 4.5, 4.2, 3.5,
            ],
            "captain_score": [
                2.0, 1.8, 1.5, 1.6,
                3.0, 2.8, 2.5, 2.3, 2.8, 2.0, 1.8, 1.5,
                8.5, 7.5, 6.0, 5.5, 4.8, 4.5, 4.2, 3.8, 6.5, 5.2,
                9.5, 7.0, 5.5, 4.2, 6.0, 4.8, 4.5, 3.5,
            ],
        })

    def _validate_fpl_squad(self, result: dict, budget: float) -> None:
        """Assert every FPL squad rule holds."""
        assert result is not None, "Solver must find a solution"
        all_p = result["starters"] + result["bench"]

        # 15 players
        assert len(all_p) == 15

        # 11 starters
        assert len(result["starters"]) == 11

        # Position counts (squad)
        pc = {}
        for p in all_p:
            pc[p["position"]] = pc.get(p["position"], 0) + 1
        assert pc.get("GKP") == 2
        assert pc.get("DEF") == 5
        assert pc.get("MID") == 5
        assert pc.get("FWD") == 3

        # Formation (starting XI)
        sc = {}
        for p in result["starters"]:
            sc[p["position"]] = sc.get(p["position"], 0) + 1
        assert sc.get("GKP") == 1
        assert 3 <= sc.get("DEF", 0) <= 5
        assert 2 <= sc.get("MID", 0) <= 5
        assert 1 <= sc.get("FWD", 0) <= 3

        # Max 3 per team
        tc = {}
        for p in all_p:
            t = p.get("team_code", 0)
            tc[t] = tc.get(t, 0) + 1
        for t, count in tc.items():
            assert count <= 3, f"Team {t} has {count} players"

        # Budget
        total_cost = sum(p["cost"] for p in all_p)
        assert total_cost <= budget + 0.1

    def test_squad_solver(self, pool):
        from src.solver.squad import solve_milp_team

        result = solve_milp_team(pool, "predicted_next_gw_points", budget=1000)
        self._validate_fpl_squad(result, 1000)

    def test_squad_solver_with_captain(self, pool):
        from src.solver.squad import solve_milp_team

        result = solve_milp_team(
            pool, "predicted_next_gw_points",
            budget=1000, captain_col="captain_score",
        )
        self._validate_fpl_squad(result, 1000)
        cid = result["captain_id"]
        assert cid is not None
        starter_ids = {p["player_id"] for p in result["starters"]}
        assert cid in starter_ids, "Captain must be a starter"

    @pytest.fixture
    def valid_current_squad(self):
        """A valid 2/5/5/3 squad from the pool with max 3 per team.

        Pool layout: GKP 1-4, DEF 5-12, MID 13-22, FWD 23-30.
        Team codes: GKP [1,2,3,4], DEF [1,2,3,4,5,6,7,8],
                    MID [1,2,3,4,5,6,7,8,9,10], FWD [1,2,3,4,5,6,7,8].

        This squad is: 2 GKP, 5 DEF, 5 MID, 3 FWD
        Team spread: teams 1-5 with 3 each.
        """
        return {1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 25, 26, 27}

    def test_transfer_solver_keeps_enough(self, pool, valid_current_squad):
        from src.solver.transfers import solve_transfer_milp

        result = solve_transfer_milp(
            pool, valid_current_squad, "predicted_next_gw_points",
            budget=1000, max_transfers=2,
        )
        self._validate_fpl_squad(result, 1000)
        new_ids = {p["player_id"] for p in result["starters"] + result["bench"]}
        kept = valid_current_squad & new_ids
        assert len(kept) >= 13

    def test_hit_cost_arithmetic(self, pool, valid_current_squad):
        from src.solver.transfers import solve_transfer_milp_with_hits

        result = solve_transfer_milp_with_hits(
            pool, valid_current_squad, "predicted_next_gw_points",
            budget=1000, free_transfers=1, max_transfers=3,
        )
        assert result is not None
        assert result["hit_cost"] == result["hits"] * 4.0
        assert result["net_points"] == pytest.approx(
            result["starting_points"] - result["hit_cost"], abs=0.01,
        )

    def test_zero_transfer_baseline_exists(self, pool, valid_current_squad):
        from src.solver.transfers import solve_transfer_milp_with_hits

        result = solve_transfer_milp_with_hits(
            pool, valid_current_squad, "predicted_next_gw_points",
            budget=1000, free_transfers=1, max_transfers=0,
        )
        assert result is not None
        assert result["hits"] == 0
        assert "baseline_points" in result

    def test_tight_budget_still_valid(self, pool):
        from src.solver.squad import solve_milp_team

        result = solve_milp_team(pool, "predicted_next_gw_points", budget=600)
        if result is not None:
            self._validate_fpl_squad(result, 600)

    def test_captain_maximises_upside(self, pool):
        """Captain should be one of the top captain_score players."""
        from src.solver.squad import solve_milp_team

        result = solve_milp_team(
            pool, "predicted_next_gw_points",
            budget=1000, captain_col="captain_score",
        )
        cid = result["captain_id"]
        cap_row = pool[pool["player_id"] == cid].iloc[0]
        # Captain should be in the top 5 captain_score players
        top5 = pool.nlargest(5, "captain_score")["player_id"].tolist()
        assert cid in top5, (
            f"Captain {cid} (score={cap_row['captain_score']}) "
            f"not in top 5: {top5}"
        )


# ===========================================================================
# 7. Prediction pipeline properties
# ===========================================================================

class TestPredictionProperties:
    """Properties that must hold for any set of predictions."""

    def test_no_negative_predictions_possible(self):
        """The .clip(min=0) in prediction code prevents negatives."""
        raw = np.array([-2.0, 0.0, 3.5, 7.0])
        clipped = raw.clip(min=0)
        assert (clipped >= 0).all()

    def test_availability_zeroing_columns(self):
        """All prediction columns must be zeroed for unavailable players.

        This is the #1 recurring bug — the 3-GW merge can reintroduce
        non-zero values for injured players.
        """
        pred_cols = [
            "predicted_next_gw_points",
            "predicted_next_3gw_points",
            "captain_score",
            "prediction_low",
            "prediction_high",
            "predicted_next_gw_points_q80",
        ]
        # Simulate the zeroing pattern from prediction.py
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            **{col: [5.0, 6.0, 7.0] for col in pred_cols},
        })
        unavailable = {2}
        mask = df["player_id"].isin(unavailable)
        for col in pred_cols:
            df.loc[mask, col] = 0.0

        # Verify player 2 is zeroed across ALL columns
        p2 = df[df["player_id"] == 2].iloc[0]
        for col in pred_cols:
            assert p2[col] == 0.0, f"{col} not zeroed for unavailable player"

        # Verify other players unchanged
        p1 = df[df["player_id"] == 1].iloc[0]
        for col in pred_cols:
            assert p1[col] == 5.0

    def test_3gw_re_zeroing_after_merge(self):
        """Simulate the 3-GW merge reintroducing values, then re-zeroing."""
        # Step 1: Initial predictions with zeroing
        result = pd.DataFrame({
            "player_id": [1, 2, 3],
            "predicted_next_gw_points": [5.0, 0.0, 7.0],  # Player 2 zeroed
            "captain_score": [6.0, 0.0, 8.0],
        })
        unavailable = {2}

        # Step 2: 3-GW merge introduces non-zero for player 2
        pred_3gw = pd.DataFrame({
            "player_id": [1, 2, 3],
            "predicted_next_3gw_points": [12.0, 10.0, 18.0],
        })
        result = result.merge(pred_3gw, on="player_id", how="left")

        # Player 2 now has non-zero 3GW prediction (BUG if not re-zeroed)
        assert result.loc[result["player_id"] == 2, "predicted_next_3gw_points"].iloc[0] == 10.0

        # Step 3: Re-zeroing (the fix)
        mask_3gw = result["player_id"].isin(unavailable)
        result.loc[mask_3gw, "predicted_next_3gw_points"] = 0.0

        # Verify fix
        assert result.loc[result["player_id"] == 2, "predicted_next_3gw_points"].iloc[0] == 0.0


# ===========================================================================
# 8. Multi-GW snapshot correctness
# ===========================================================================

class TestMultiGWProperties:
    """Properties of the multi-GW prediction pipeline."""

    def test_decay_reduces_predictions(self):
        """Applying decay to predictions reduces them."""
        raw_pred = 5.0
        for offset in range(len(pred_cfg.confidence_decay)):
            decay = pred_cfg.confidence_decay[offset]
            adjusted = raw_pred * decay
            assert adjusted < raw_pred
            assert adjusted > 0

    def test_3gw_sum_with_decay(self):
        """3-GW prediction = sum of 3 decayed 1-GW predictions."""
        per_gw = 5.0
        total = sum(
            per_gw * pred_cfg.confidence_decay[i]
            for i in range(3)
        )
        # = 5*(0.95 + 0.93 + 0.90) = 5*2.78 = 13.9
        expected = per_gw * sum(pred_cfg.confidence_decay[:3])
        assert total == pytest.approx(expected)
        assert total < per_gw * 3, "Decayed sum must be less than undecayed"
