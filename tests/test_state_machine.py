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


# ---------------------------------------------------------------------------
# PlannedSquadRepository
# ---------------------------------------------------------------------------

class TestPlannedSquadRepository:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = tmp_path / "test.db"
        from src.db.connection import connect
        with connect(path) as conn:
            from src.db.schema import init_schema
            from src.db.migrations import apply_migrations
            init_schema(conn)
            apply_migrations(conn)
        return path

    @pytest.fixture
    def season_id(self, db_path):
        """Create a season row so FK constraints are satisfied."""
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        return repo.create_season(manager_id=999, season_name="2025-2026")

    def test_save_and_get(self, db_path, season_id):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        squad = {"players": [{"id": 1, "name": "Salah"}], "captain_id": 1, "chip": None}
        repo.save_planned_squad(season_id=season_id, gw=10, squad_json=squad, source="recommended")
        result = repo.get_planned_squad(season_id=season_id, gw=10)
        assert result is not None
        assert result["source"] == "recommended"
        assert result["squad_json"]["captain_id"] == 1

    def test_upsert_overwrites(self, db_path, season_id):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        squad1 = {"players": [], "captain_id": 1}
        repo.save_planned_squad(season_id, 10, squad1, "recommended")
        squad2 = {"players": [], "captain_id": 2}
        repo.save_planned_squad(season_id, 10, squad2, "user_override")
        result = repo.get_planned_squad(season_id, 10)
        assert result["source"] == "user_override"
        assert result["squad_json"]["captain_id"] == 2

    def test_get_nonexistent(self, db_path, season_id):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        assert repo.get_planned_squad(season_id, 99) is None

    def test_delete(self, db_path, season_id):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        repo.save_planned_squad(season_id, 10, {"x": 1}, "recommended")
        repo.delete_planned_squad(season_id, 10)
        assert repo.get_planned_squad(season_id, 10) is None


# ---------------------------------------------------------------------------
# SeasonPhase
# ---------------------------------------------------------------------------

class TestSeasonPhase:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = tmp_path / "test.db"
        from src.db.connection import connect
        with connect(path) as conn:
            from src.db.schema import init_schema
            from src.db.migrations import apply_migrations
            init_schema(conn)
            apply_migrations(conn)
        return path

    def test_new_season_has_planning_phase(self, db_path):
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        repo.create_season(manager_id=123, season_name="2025-2026")
        season = repo.get_season(123, "2025-2026")
        assert season["phase"] == "planning"

    def test_update_phase(self, db_path):
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        sid = repo.create_season(manager_id=123, season_name="2025-2026")
        repo.update_phase(sid, "ready")
        season = repo.get_season(123, "2025-2026")
        assert season["phase"] == "ready"
