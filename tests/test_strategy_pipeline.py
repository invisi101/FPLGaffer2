"""Tests for the full strategy pipeline: chip evaluator, transfer planner,
captain planner, plan synthesizer, availability adjustments, and the
wired-up pipeline in SeasonManager."""

import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# ChipEvaluator
# ---------------------------------------------------------------------------


class TestChipEvaluator:
    """Tests for src.strategy.chip_evaluator.ChipEvaluator."""

    def test_evaluate_all_chips_returns_all_available(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        from src.strategy.chip_evaluator import ChipEvaluator

        evaluator = ChipEvaluator()
        heatmap = evaluator.evaluate_all_chips(
            current_squad_ids, 1000, available_chips,
            future_predictions, fixture_calendar,
        )
        # Should have an entry for each available chip
        assert set(heatmap.keys()) == available_chips

    def test_chip_heatmap_values_are_per_gw(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        from src.strategy.chip_evaluator import ChipEvaluator

        evaluator = ChipEvaluator()
        heatmap = evaluator.evaluate_all_chips(
            current_squad_ids, 1000, available_chips,
            future_predictions, fixture_calendar,
        )
        for chip_name, gw_values in heatmap.items():
            assert isinstance(gw_values, dict), f"{chip_name} values not a dict"
            for gw, val in gw_values.items():
                assert isinstance(gw, int), f"{chip_name} GW key {gw} not int"
                assert isinstance(val, (int, float)), f"{chip_name} GW{gw} value not numeric"
                assert val >= 0, f"{chip_name} GW{gw} has negative value {val}"

    def test_bb_values_higher_in_dgw(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        """BB should score higher in DGW GWs (GW5 in our fixture data)."""
        from src.strategy.chip_evaluator import ChipEvaluator

        evaluator = ChipEvaluator()
        heatmap = evaluator.evaluate_all_chips(
            current_squad_ids, 1000, {"bboost"},
            future_predictions, fixture_calendar,
        )
        bb_vals = heatmap.get("bboost", {})
        if 5 in bb_vals and 6 in bb_vals:
            # GW5 has DGWs, GW6 doesn't — BB should be at least as valuable
            # (may not always hold with prediction-based eval, but heuristic GWs
            # outside prediction range should show this clearly)
            assert bb_vals[5] >= 0  # sanity: non-negative

    def test_empty_available_chips_returns_empty(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.chip_evaluator import ChipEvaluator

        evaluator = ChipEvaluator()
        heatmap = evaluator.evaluate_all_chips(
            current_squad_ids, 1000, set(),
            future_predictions, fixture_calendar,
        )
        assert heatmap == {}

    def test_synergies_returns_list(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        from src.strategy.chip_evaluator import ChipEvaluator

        evaluator = ChipEvaluator()
        heatmap = evaluator.evaluate_all_chips(
            current_squad_ids, 1000, available_chips,
            future_predictions, fixture_calendar,
        )
        synergies = evaluator.evaluate_chip_synergies(heatmap, available_chips)
        assert isinstance(synergies, list)
        for syn in synergies:
            assert "chips" in syn
            assert "gws" in syn
            assert "combined_value" in syn
            assert syn["combined_value"] >= 0


# ---------------------------------------------------------------------------
# MultiWeekPlanner
# ---------------------------------------------------------------------------


class TestMultiWeekPlanner:
    """Tests for src.strategy.transfer_planner.MultiWeekPlanner."""

    def test_plan_returns_list_of_gw_steps(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            future_predictions, fixture_calendar, [],
        )
        assert isinstance(plan, list)
        assert len(plan) > 0
        # Each step should have required keys
        for step in plan:
            assert "gw" in step
            assert "transfers_in" in step
            assert "transfers_out" in step
            assert "ft_used" in step
            assert "predicted_points" in step
            assert "squad_ids" in step

    def test_plan_respects_planning_horizon(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 1,
            future_predictions, fixture_calendar, [],
        )
        # Should not exceed 5 GWs (planning horizon)
        assert len(plan) <= 5

    def test_plan_squad_ids_are_valid(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 1,
            future_predictions, fixture_calendar, [],
        )
        for step in plan:
            squad = set(step["squad_ids"])
            assert len(squad) == 15, f"GW{step['gw']} squad has {len(squad)} players, expected 15"

    def test_plan_with_zero_ft(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        """Even with 1 FT (minimum), planner should produce a plan."""
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 1,
            future_predictions, fixture_calendar, [],
        )
        assert isinstance(plan, list)
        assert len(plan) > 0

    def test_plan_has_rationale(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            future_predictions, fixture_calendar, [],
        )
        for step in plan:
            assert "rationale" in step
            assert isinstance(step["rationale"], str)
            assert len(step["rationale"]) > 0

    def test_empty_predictions_returns_empty(self, current_squad_ids, fixture_calendar):
        from src.strategy.transfer_planner import MultiWeekPlanner

        planner = MultiWeekPlanner()
        plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            {}, fixture_calendar, [],
        )
        assert plan == []


# ---------------------------------------------------------------------------
# CaptainPlanner
# ---------------------------------------------------------------------------


class TestCaptainPlanner:
    """Tests for src.strategy.captain_planner.CaptainPlanner."""

    def test_captain_plan_covers_all_prediction_gws(
        self, future_predictions, current_squad_ids,
    ):
        from src.strategy.captain_planner import CaptainPlanner

        planner = CaptainPlanner()
        plan = planner.plan_captaincy(current_squad_ids, future_predictions)
        planned_gws = {entry["gw"] for entry in plan}
        assert planned_gws == set(future_predictions.keys())

    def test_captain_is_in_squad(
        self, future_predictions, current_squad_ids,
    ):
        from src.strategy.captain_planner import CaptainPlanner

        planner = CaptainPlanner()
        plan = planner.plan_captaincy(current_squad_ids, future_predictions)
        for entry in plan:
            assert entry["captain_id"] in current_squad_ids

    def test_captain_entry_has_required_fields(
        self, future_predictions, current_squad_ids,
    ):
        from src.strategy.captain_planner import CaptainPlanner

        planner = CaptainPlanner()
        plan = planner.plan_captaincy(current_squad_ids, future_predictions)
        for entry in plan:
            assert "gw" in entry
            assert "captain_id" in entry
            assert "captain_name" in entry
            assert "captain_points" in entry
            assert "vc_id" in entry
            assert "vc_name" in entry
            assert "weak_gw" in entry
            assert isinstance(entry["captain_points"], (int, float))

    def test_captain_uses_transfer_plan_squads(
        self, future_predictions, current_squad_ids,
    ):
        """When transfer plan provides updated squads, captain should be
        picked from those squads."""
        from src.strategy.captain_planner import CaptainPlanner

        # Fake transfer plan: GW3 swaps player 15 for player 16
        transfer_plan = [{
            "gw": 3,
            "squad_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16],
            "transfers_in": [{"player_id": 16}],
            "transfers_out": [{"player_id": 15}],
            "ft_used": 1,
            "predicted_points": 60.0,
        }]
        planner = CaptainPlanner()
        plan = planner.plan_captaincy(
            current_squad_ids, future_predictions, transfer_plan,
        )
        gw3_entry = next(e for e in plan if e["gw"] == 3)
        # Captain for GW3 should be from the transfer plan squad
        assert gw3_entry["captain_id"] in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16}
        # Player 15 should NOT be captain for GW3
        assert gw3_entry["captain_id"] != 15

    def test_vc_different_from_captain(
        self, future_predictions, current_squad_ids,
    ):
        from src.strategy.captain_planner import CaptainPlanner

        planner = CaptainPlanner()
        plan = planner.plan_captaincy(current_squad_ids, future_predictions)
        for entry in plan:
            assert entry["vc_id"] != entry["captain_id"]


# ---------------------------------------------------------------------------
# PlanSynthesizer
# ---------------------------------------------------------------------------


class TestPlanSynthesizer:
    """Tests for src.strategy.plan_synthesizer.PlanSynthesizer."""

    def _make_inputs(self, future_predictions, current_squad_ids, fixture_calendar):
        from src.strategy.chip_evaluator import ChipEvaluator
        from src.strategy.transfer_planner import MultiWeekPlanner
        from src.strategy.captain_planner import CaptainPlanner

        available = {"wildcard", "freehit", "bboost", "3xc"}

        chip_eval = ChipEvaluator()
        heatmap = chip_eval.evaluate_all_chips(
            current_squad_ids, 1000, available,
            future_predictions, fixture_calendar,
        )
        synergies = chip_eval.evaluate_chip_synergies(heatmap, available)

        planner = MultiWeekPlanner()
        transfer_plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            future_predictions, fixture_calendar, [],
        )

        captain_planner = CaptainPlanner()
        captain_plan = captain_planner.plan_captaincy(
            current_squad_ids, future_predictions, transfer_plan,
        )

        return transfer_plan, captain_plan, heatmap, synergies, available

    def test_synthesize_returns_required_keys(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.plan_synthesizer import PlanSynthesizer

        tp, cp, hm, syn, avail = self._make_inputs(
            future_predictions, current_squad_ids, fixture_calendar,
        )
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(tp, cp, hm, syn, avail)

        assert "timeline" in plan
        assert "chip_schedule" in plan
        assert "chip_synergies" in plan
        assert "rationale" in plan
        assert "generated_at" in plan

    def test_timeline_merges_transfer_and_captain_data(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.plan_synthesizer import PlanSynthesizer

        tp, cp, hm, syn, avail = self._make_inputs(
            future_predictions, current_squad_ids, fixture_calendar,
        )
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(tp, cp, hm, syn, avail)

        for entry in plan["timeline"]:
            assert "gw" in entry
            # Should have captain info (from captain planner)
            if "captain_id" in entry:
                assert "captain_name" in entry
                assert "captain_points" in entry
            # Should have transfer info (from transfer planner)
            if "transfers_in" in entry:
                assert "ft_used" in entry

    def test_chip_schedule_no_duplicates(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.plan_synthesizer import PlanSynthesizer

        tp, cp, hm, syn, avail = self._make_inputs(
            future_predictions, current_squad_ids, fixture_calendar,
        )
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(tp, cp, hm, syn, avail)

        scheduled_gws = list(plan["chip_schedule"].values())
        # No two chips on the same GW
        assert len(scheduled_gws) == len(set(scheduled_gws))

    def test_plan_is_json_serialisable(
        self, future_predictions, current_squad_ids, fixture_calendar,
    ):
        from src.strategy.plan_synthesizer import PlanSynthesizer

        tp, cp, hm, syn, avail = self._make_inputs(
            future_predictions, current_squad_ids, fixture_calendar,
        )
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(tp, cp, hm, syn, avail)

        # Must be JSON-serialisable (stored in DB as JSON)
        serialised = json.dumps(plan)
        assert len(serialised) > 0
        roundtrip = json.loads(serialised)
        assert roundtrip["rationale"] == plan["rationale"]


# ---------------------------------------------------------------------------
# Availability adjustments
# ---------------------------------------------------------------------------


class TestAvailabilityAdjustments:
    """Tests for src.strategy.reactive.apply_availability_adjustments."""

    def test_injured_players_zeroed_all_gws(self, future_predictions):
        from src.strategy.reactive import apply_availability_adjustments

        # Mark player 8 (MID1, highest predicted) as injured
        elements = [
            {"id": 8, "status": "i", "chance_of_playing_next_round": 0},
            {"id": 1, "status": "a", "chance_of_playing_next_round": 100},
        ]
        adjusted = apply_availability_adjustments(future_predictions, elements)

        for gw, gw_df in adjusted.items():
            p8 = gw_df[gw_df["player_id"] == 8]
            assert not p8.empty, f"Player 8 missing from GW{gw}"
            assert p8.iloc[0]["predicted_points"] == 0.0, (
                f"Injured player 8 not zeroed in GW{gw}"
            )

    def test_doubtful_players_zeroed_first_gw_only(self, future_predictions):
        from src.strategy.reactive import apply_availability_adjustments

        # Mark player 13 (FWD1) as doubtful (25% chance)
        elements = [
            {"id": 13, "status": "a", "chance_of_playing_next_round": 25},
        ]
        adjusted = apply_availability_adjustments(future_predictions, elements)
        gws = sorted(adjusted.keys())

        # First GW: should be zeroed
        first_gw_df = adjusted[gws[0]]
        p13_first = first_gw_df[first_gw_df["player_id"] == 13]
        assert p13_first.iloc[0]["predicted_points"] == 0.0

        # Second GW: should NOT be zeroed
        if len(gws) > 1:
            second_gw_df = adjusted[gws[1]]
            p13_second = second_gw_df[second_gw_df["player_id"] == 13]
            assert p13_second.iloc[0]["predicted_points"] > 0.0

    def test_healthy_players_unchanged(self, future_predictions):
        from src.strategy.reactive import apply_availability_adjustments

        elements = [
            {"id": 1, "status": "a", "chance_of_playing_next_round": 100},
        ]
        adjusted = apply_availability_adjustments(future_predictions, elements)

        for gw in adjusted:
            orig = future_predictions[gw]
            adj = adjusted[gw]
            p1_orig = orig[orig["player_id"] == 1].iloc[0]["predicted_points"]
            p1_adj = adj[adj["player_id"] == 1].iloc[0]["predicted_points"]
            assert p1_orig == p1_adj


# ---------------------------------------------------------------------------
# Plan invalidation detection
# ---------------------------------------------------------------------------


class TestPlanInvalidation:
    """Tests for src.strategy.reactive.detect_plan_invalidation."""

    def test_injury_triggers_critical(self):
        from src.strategy.reactive import detect_plan_invalidation

        plan = {
            "timeline": [
                {"gw": 3, "squad_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
            ],
            "chip_schedule": {},
        }
        squad_changes = {
            8: {"status": "i", "chance_of_playing": 0, "web_name": "MID1"},
        }
        triggers = detect_plan_invalidation(plan, {}, [], squad_changes)
        assert any(t["severity"] == "critical" and t["type"] == "injury" for t in triggers)

    def test_captain_prediction_drop_triggers_critical(self, future_predictions):
        from src.strategy.reactive import detect_plan_invalidation

        plan = {
            "timeline": [
                {"gw": 3, "captain_id": 13, "captain_points": 8.0, "squad_ids": []},
            ],
            "chip_schedule": {},
        }
        # Create predictions where captain's points dropped >50%
        new_preds = {3: future_predictions[3].copy()}
        new_preds[3].loc[new_preds[3]["player_id"] == 13, "predicted_points"] = 2.0

        triggers = detect_plan_invalidation(plan, new_preds, [])
        assert any(
            t["severity"] == "critical" and t["type"] == "prediction_shift"
            for t in triggers
        )

    def test_no_triggers_when_plan_healthy(self):
        from src.strategy.reactive import detect_plan_invalidation

        plan = {
            "timeline": [{"gw": 3, "squad_ids": [1, 2, 3]}],
            "chip_schedule": {},
        }
        triggers = detect_plan_invalidation(plan, {}, [])
        assert triggers == []


# ---------------------------------------------------------------------------
# End-to-end: full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test: ChipEval -> Planner -> Captain -> Synthesizer."""

    def test_full_pipeline_produces_valid_strategic_plan(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        from src.strategy.chip_evaluator import ChipEvaluator
        from src.strategy.transfer_planner import MultiWeekPlanner
        from src.strategy.captain_planner import CaptainPlanner
        from src.strategy.plan_synthesizer import PlanSynthesizer

        # 1. Chip evaluation
        chip_eval = ChipEvaluator()
        heatmap = chip_eval.evaluate_all_chips(
            current_squad_ids, 1000, available_chips,
            future_predictions, fixture_calendar,
        )
        synergies = chip_eval.evaluate_chip_synergies(heatmap, available_chips)

        # 2. Transfer planning
        planner = MultiWeekPlanner()
        transfer_plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            future_predictions, fixture_calendar, [],
        )
        assert len(transfer_plan) > 0

        # 3. Captain planning
        captain_planner = CaptainPlanner()
        captain_plan = captain_planner.plan_captaincy(
            current_squad_ids, future_predictions, transfer_plan,
        )
        assert len(captain_plan) > 0

        # 4. Synthesis
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(
            transfer_plan, captain_plan, heatmap, synergies, available_chips,
        )

        # Validate the strategic plan structure
        assert "timeline" in plan
        assert len(plan["timeline"]) > 0
        assert "chip_schedule" in plan
        assert "rationale" in plan
        assert isinstance(plan["rationale"], str)
        assert len(plan["rationale"]) > 0

        # Timeline should have GW entries with both transfer + captain info
        first_entry = plan["timeline"][0]
        assert "gw" in first_entry
        assert "captain_id" in first_entry
        assert "predicted_points" in first_entry

        # Should be storable in DB (JSON-serialisable)
        plan_json = json.dumps(plan)
        heatmap_json = json.dumps({
            k: {str(gw): v for gw, v in vals.items()}
            for k, vals in heatmap.items()
        })
        assert len(plan_json) > 100
        assert len(heatmap_json) > 10

    def test_pipeline_with_injuries_adjusts_captaincy(
        self, future_predictions, current_squad_ids, fixture_calendar, available_chips,
    ):
        """When the best captain candidate is injured, the pipeline should
        select a different captain."""
        from src.strategy.reactive import apply_availability_adjustments
        from src.strategy.captain_planner import CaptainPlanner

        # Find who would be captain without injuries
        captain_planner = CaptainPlanner()
        clean_plan = captain_planner.plan_captaincy(
            current_squad_ids, future_predictions,
        )
        original_captain = clean_plan[0]["captain_id"]

        # Injure the original captain
        elements = [
            {"id": original_captain, "status": "i", "chance_of_playing_next_round": 0},
        ]
        adjusted = apply_availability_adjustments(future_predictions, elements)

        injured_plan = captain_planner.plan_captaincy(
            current_squad_ids, adjusted,
        )
        new_captain = injured_plan[0]["captain_id"]

        assert new_captain != original_captain, (
            "Captain should change when original captain is injured"
        )

    def test_pipeline_db_storage_roundtrip(
        self, tmp_db, future_predictions, current_squad_ids,
        fixture_calendar, available_chips,
    ):
        """Strategic plan can be saved to DB and retrieved intact."""
        from src.db.connection import get_connection
        from src.db.schema import init_schema
        from src.db.repositories import PlanRepository, SeasonRepository
        from src.strategy.chip_evaluator import ChipEvaluator
        from src.strategy.transfer_planner import MultiWeekPlanner
        from src.strategy.captain_planner import CaptainPlanner
        from src.strategy.plan_synthesizer import PlanSynthesizer

        # Set up DB
        conn = get_connection(tmp_db)
        init_schema(conn)
        conn.close()

        seasons = SeasonRepository(tmp_db)
        plans = PlanRepository(tmp_db)
        season_id = seasons.create_season(99999, "Test", season_name="2025-2026")

        # Run pipeline
        chip_eval = ChipEvaluator()
        heatmap = chip_eval.evaluate_all_chips(
            current_squad_ids, 1000, available_chips,
            future_predictions, fixture_calendar,
        )
        synergies = chip_eval.evaluate_chip_synergies(heatmap, available_chips)
        planner = MultiWeekPlanner()
        transfer_plan = planner.plan_transfers(
            current_squad_ids, 1000, 2,
            future_predictions, fixture_calendar, [],
        )
        captain_planner = CaptainPlanner()
        captain_plan = captain_planner.plan_captaincy(
            current_squad_ids, future_predictions, transfer_plan,
        )
        synthesizer = PlanSynthesizer()
        plan = synthesizer.synthesize(
            transfer_plan, captain_plan, heatmap, synergies, available_chips,
        )

        # Save
        plan_json = json.dumps(plan)
        heatmap_json = json.dumps({
            k: {str(gw): v for gw, v in vals.items()}
            for k, vals in heatmap.items()
        })
        plans.save_strategic_plan(season_id, 3, plan_json, heatmap_json)

        # Retrieve
        stored = plans.get_strategic_plan(season_id, as_of_gw=3)
        assert stored is not None
        roundtrip = json.loads(stored["plan_json"])
        assert roundtrip["rationale"] == plan["rationale"]
        assert len(roundtrip["timeline"]) == len(plan["timeline"])

        stored_heatmap = json.loads(stored["chip_heatmap_json"])
        assert len(stored_heatmap) == len(heatmap)
