"""Tests for the strategy pipeline: transfer planner and availability adjustments.

Note: ChipEvaluator, CaptainPlanner, PlanSynthesizer, and the full pipeline
tests were removed as part of the season-manager-v2 redesign (those modules
are no longer used).
"""

import json

import pandas as pd
import pytest


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
