"""Tests for GW lifecycle state machine."""

import pytest

from src.season.state_machine import GWPhase, can_transition, next_phase, detect_phase


# ---------------------------------------------------------------------------
# GWPhase enum
# ---------------------------------------------------------------------------

class TestGWPhase:
    def test_all_phases_exist(self):
        assert GWPhase.PLANNING
        assert GWPhase.READY
        assert GWPhase.LIVE
        assert GWPhase.COMPLETE
        assert GWPhase.SEASON_OVER

    def test_phases_are_strings(self):
        for phase in GWPhase:
            assert isinstance(phase.value, str)

    def test_phase_count(self):
        assert len(GWPhase) == 5


# ---------------------------------------------------------------------------
# can_transition
# ---------------------------------------------------------------------------

class TestCanTransition:
    # Valid transitions
    def test_planning_to_ready(self):
        assert can_transition(GWPhase.PLANNING, GWPhase.READY) is True

    def test_ready_to_live(self):
        assert can_transition(GWPhase.READY, GWPhase.LIVE) is True

    def test_live_to_complete(self):
        assert can_transition(GWPhase.LIVE, GWPhase.COMPLETE) is True

    def test_complete_to_planning(self):
        assert can_transition(GWPhase.COMPLETE, GWPhase.PLANNING) is True

    def test_complete_to_season_over(self):
        assert can_transition(GWPhase.COMPLETE, GWPhase.SEASON_OVER) is True

    # Invalid transitions
    def test_planning_to_live_invalid(self):
        assert can_transition(GWPhase.PLANNING, GWPhase.LIVE) is False

    def test_ready_to_complete_invalid(self):
        assert can_transition(GWPhase.READY, GWPhase.COMPLETE) is False

    def test_live_to_planning_invalid(self):
        assert can_transition(GWPhase.LIVE, GWPhase.PLANNING) is False

    def test_season_over_to_planning_invalid(self):
        assert can_transition(GWPhase.SEASON_OVER, GWPhase.PLANNING) is False

    def test_season_over_to_anything_invalid(self):
        for phase in GWPhase:
            assert can_transition(GWPhase.SEASON_OVER, phase) is False

    def test_self_transition_invalid(self):
        for phase in GWPhase:
            assert can_transition(phase, phase) is False


# ---------------------------------------------------------------------------
# next_phase
# ---------------------------------------------------------------------------

class TestNextPhase:
    def test_planning_next_is_ready(self):
        assert next_phase(GWPhase.PLANNING) == GWPhase.READY

    def test_ready_next_is_live(self):
        assert next_phase(GWPhase.READY) == GWPhase.LIVE

    def test_live_next_is_complete(self):
        assert next_phase(GWPhase.LIVE) == GWPhase.COMPLETE

    def test_complete_next_is_planning(self):
        assert next_phase(GWPhase.COMPLETE, is_final_gw=False) == GWPhase.PLANNING

    def test_complete_final_gw_is_season_over(self):
        assert next_phase(GWPhase.COMPLETE, is_final_gw=True) == GWPhase.SEASON_OVER

    def test_season_over_next_is_none(self):
        assert next_phase(GWPhase.SEASON_OVER) is None


# ---------------------------------------------------------------------------
# detect_phase
# ---------------------------------------------------------------------------

class TestDetectPhase:
    def test_planning_no_recommendation(self):
        result = detect_phase(
            has_recommendation=False,
            deadline_passed=False,
            all_fixtures_finished=False,
        )
        assert result == GWPhase.PLANNING

    def test_ready_has_recommendation(self):
        result = detect_phase(
            has_recommendation=True,
            deadline_passed=False,
            all_fixtures_finished=False,
        )
        assert result == GWPhase.READY

    def test_live_deadline_passed(self):
        result = detect_phase(
            has_recommendation=True,
            deadline_passed=True,
            all_fixtures_finished=False,
        )
        assert result == GWPhase.LIVE

    def test_live_deadline_passed_no_recommendation(self):
        """Deadline passed without recommendation is still LIVE."""
        result = detect_phase(
            has_recommendation=False,
            deadline_passed=True,
            all_fixtures_finished=False,
        )
        assert result == GWPhase.LIVE

    def test_complete_all_finished(self):
        result = detect_phase(
            has_recommendation=True,
            deadline_passed=True,
            all_fixtures_finished=True,
        )
        assert result == GWPhase.COMPLETE

    def test_complete_all_finished_no_recommendation(self):
        """All fixtures done = COMPLETE regardless of recommendation."""
        result = detect_phase(
            has_recommendation=False,
            deadline_passed=True,
            all_fixtures_finished=True,
        )
        assert result == GWPhase.COMPLETE

    def test_fixtures_finished_implies_deadline_passed(self):
        """Edge case: fixtures finished but deadline not flagged.
        In reality this shouldn't happen, but the function should
        still return COMPLETE since all_fixtures_finished is the
        strongest signal."""
        result = detect_phase(
            has_recommendation=False,
            deadline_passed=False,
            all_fixtures_finished=True,
        )
        assert result == GWPhase.COMPLETE
