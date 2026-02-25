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


# ---------------------------------------------------------------------------
# SeasonManagerV2
# ---------------------------------------------------------------------------

class TestSeasonManagerV2:
    """Tests for the v2 state-machine-driven season manager."""

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
    def bootstrap(self):
        """Minimal bootstrap payload for testing."""
        return {
            "events": [
                {
                    "id": 25,
                    "is_current": True,
                    "is_next": False,
                    "deadline_time": "2026-02-14T10:00:00Z",
                    "finished": True,
                },
                {
                    "id": 26,
                    "is_current": False,
                    "is_next": True,
                    "deadline_time": "2026-02-28T11:30:00Z",
                    "finished": False,
                },
            ],
            "elements": [
                {"id": 1, "web_name": "Salah", "status": "a"},
                {"id": 2, "web_name": "Haaland", "status": "a"},
                {"id": 3, "web_name": "Saka", "status": "i"},
            ],
            "teams": [],
        }

    @pytest.fixture
    def fixtures_gw25_unfinished(self):
        """Fixtures for GW25 where not all are finished."""
        return [
            {"event": 25, "finished": True, "id": 1},
            {"event": 25, "finished": False, "id": 2},
        ]

    @pytest.fixture
    def fixtures_gw25_finished(self):
        """Fixtures for GW25 where all are finished."""
        return [
            {"event": 25, "finished": True, "id": 1},
            {"event": 25, "finished": True, "id": 2},
        ]

    def test_init_creates_repos(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert mgr.seasons is not None
        assert mgr.planned_squads is not None
        assert mgr.recommendations is not None
        assert mgr.snapshots is not None
        assert mgr.outcomes is not None
        assert mgr.prices is not None
        assert mgr.fixtures is not None
        assert mgr.dashboard is not None
        assert mgr.watchlist is not None

    def test_get_status_no_season(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        status = mgr.get_status(manager_id=99999)
        assert status["active"] is False
        assert status["phase"] is None

    def test_get_status_with_season(self, db_path, bootstrap, monkeypatch):
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository

        SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")

        # Mock bootstrap and fixtures so we don't need real cache.
        monkeypatch.setattr(
            "src.season.manager_v2.load_bootstrap", lambda: bootstrap
        )
        monkeypatch.setattr(
            "src.season.manager_v2.get_next_gw",
            lambda bs: 26,
        )

        mgr = SeasonManagerV2(db_path=db_path)
        # Mock fixture loading to return empty (GW25 not finished).
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        status = mgr.get_status(manager_id=123)
        assert status["active"] is True
        assert "phase" in status
        assert status["gw"] == 26
        assert status["current_gw"] == 25
        assert status["season_id"] is not None
        assert status["manager_id"] == 123

    def test_get_status_detects_planning_phase(self, db_path, bootstrap, monkeypatch):
        """No recommendation and deadline not passed -> PLANNING."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository

        SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")

        # Set deadline far in the future.
        bootstrap["events"][1]["deadline_time"] = "2099-12-31T23:59:59Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        status = mgr.get_status(manager_id=123)
        assert status["phase"] == "planning"
        assert status["has_recommendation"] is False

    def test_get_status_detects_ready_phase(self, db_path, bootstrap, monkeypatch):
        """Has recommendation, deadline not passed -> READY."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, RecommendationRepository

        sid = SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")
        RecommendationRepository(db_path).save_recommendation(
            season_id=sid, gameweek=26, captain_id=1, captain_name="Salah",
        )

        bootstrap["events"][1]["deadline_time"] = "2099-12-31T23:59:59Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        status = mgr.get_status(manager_id=123)
        assert status["phase"] == "ready"
        assert status["has_recommendation"] is True

    def test_tick_no_season_returns_empty(self, db_path, monkeypatch):
        from src.season.manager_v2 import SeasonManagerV2

        # Mock bootstrap so get_status works without cache.
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: None)

        mgr = SeasonManagerV2(db_path=db_path)
        alerts = mgr.tick(manager_id=99999)
        assert alerts == []

    def test_tick_planning_returns_empty_stub(self, db_path, bootstrap, monkeypatch):
        """PLANNING tick is a stub -- returns [] for now."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository

        SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")

        bootstrap["events"][1]["deadline_time"] = "2099-12-31T23:59:59Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        alerts = mgr.tick(manager_id=123)
        assert alerts == []

    def test_tick_ready_detects_injury(self, db_path, bootstrap, monkeypatch):
        """READY tick detects an injured captain and returns a critical alert."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, RecommendationRepository, PlannedSquadRepository

        sid = SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")
        RecommendationRepository(db_path).save_recommendation(
            season_id=sid, gameweek=26, captain_id=3, captain_name="Saka",
        )
        # Save a planned squad with Saka as captain.
        PlannedSquadRepository(db_path).save_planned_squad(
            season_id=sid, gw=26,
            squad_json={
                "players": [
                    {"player_id": 1, "starter": True, "is_captain": False},
                    {"player_id": 3, "starter": True, "is_captain": True},
                ],
                "captain_id": 3,
            },
        )

        # Saka is injured in bootstrap (status "i", element id 3).
        bootstrap["events"][1]["deadline_time"] = "2099-12-31T23:59:59Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        alerts = mgr.tick(manager_id=123)
        # Should have at least one injury alert for Saka.
        injury_alerts = [a for a in alerts if a["type"] == "injury"]
        assert len(injury_alerts) >= 1
        saka_alert = [a for a in injury_alerts if a["player_id"] == 3]
        assert len(saka_alert) == 1
        assert saka_alert[0]["severity"] == "critical"
        assert "Saka" in saka_alert[0]["message"]

    def test_tick_ready_no_alert_when_healthy(self, db_path, bootstrap, monkeypatch):
        """READY tick returns no alerts when all planned players are available."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, RecommendationRepository, PlannedSquadRepository

        sid = SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")
        RecommendationRepository(db_path).save_recommendation(
            season_id=sid, gameweek=26, captain_id=1, captain_name="Salah",
        )
        PlannedSquadRepository(db_path).save_planned_squad(
            season_id=sid, gw=26,
            squad_json={
                "players": [
                    {"player_id": 1, "starter": True, "is_captain": True},
                    {"player_id": 2, "starter": True, "is_captain": False},
                ],
                "captain_id": 1,
            },
        )

        bootstrap["events"][1]["deadline_time"] = "2099-12-31T23:59:59Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: [])

        alerts = mgr.tick(manager_id=123)
        assert alerts == []

    def test_tick_live_detects_gw_complete(self, db_path, bootstrap, fixtures_gw25_finished, monkeypatch):
        """LIVE tick transitions to COMPLETE when all fixtures are finished."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, RecommendationRepository

        sid = SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")
        # Need a recommendation so phase is at least READY, and deadline passed so LIVE.
        RecommendationRepository(db_path).save_recommendation(
            season_id=sid, gameweek=26, captain_id=1, captain_name="Salah",
        )

        # Deadline in the past -> LIVE phase.
        bootstrap["events"][1]["deadline_time"] = "2020-01-01T00:00:00Z"
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)
        monkeypatch.setattr("src.season.manager_v2.get_next_gw", lambda bs: 26)

        mgr = SeasonManagerV2(db_path=db_path)
        # GW25 fixtures all finished -> detect_phase returns COMPLETE,
        # but the LIVE tick logic also checks _is_gw_finished.
        monkeypatch.setattr(mgr, "_load_fixtures", lambda: fixtures_gw25_finished)

        # First get_status should detect COMPLETE (all_fixtures_finished).
        status = mgr.get_status(manager_id=123)
        assert status["phase"] == "complete"

    def test_is_gw_finished_true(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        mgr._load_fixtures = lambda: [
            {"event": 10, "finished": True},
            {"event": 10, "finished": True},
        ]
        assert mgr._is_gw_finished(10) is True

    def test_is_gw_finished_false(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        mgr._load_fixtures = lambda: [
            {"event": 10, "finished": True},
            {"event": 10, "finished": False},
        ]
        assert mgr._is_gw_finished(10) is False

    def test_is_gw_finished_no_fixtures(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        mgr._load_fixtures = lambda: []
        assert mgr._is_gw_finished(10) is False

    def test_get_deadline_parses_iso(self, db_path, bootstrap):
        from src.season.manager_v2 import SeasonManagerV2
        from datetime import datetime, timezone
        mgr = SeasonManagerV2(db_path=db_path)
        dl = mgr._get_deadline(bootstrap, 26)
        assert dl is not None
        assert dl.tzinfo is not None
        assert dl.year == 2026
        assert dl.month == 2
        assert dl.day == 28

    def test_get_deadline_returns_none_for_missing_gw(self, db_path, bootstrap):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert mgr._get_deadline(bootstrap, 99) is None

    def test_get_deadline_returns_none_for_none_bootstrap(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert mgr._get_deadline(None, 26) is None
