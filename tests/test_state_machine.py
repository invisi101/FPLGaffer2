"""Tests for GW lifecycle state machine."""

import copy
import json

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


# ---------------------------------------------------------------------------
# User Action Methods
# ---------------------------------------------------------------------------

def _make_test_players():
    """Build a 15-player squad for testing (2 GKP, 5 DEF, 5 MID, 3 FWD)."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = []
    for i, pos in enumerate(positions):
        players.append({
            "player_id": 100 + i,
            "web_name": f"Player{i}",
            "position": pos,
            "team_code": (i % 10) + 1,
            "cost": 5.0 + (i * 0.5),
            "predicted_next_gw_points": 4.0 + (i * 0.3),
            "captain_score": 4.5 + (i * 0.3),
            "starter": i < 11,  # first 11 are starters
            "is_captain": i == 14,  # FWD with highest predicted pts
            "is_vice_captain": i == 13,
        })
    return players


def _make_test_squad_json(players=None):
    """Build a complete planned-squad JSON dict."""
    if players is None:
        players = _make_test_players()
    return {
        "players": players,
        "captain_id": 114,
        "vice_captain_id": 113,
        "chip": None,
        "transfers_in": [],
        "transfers_out": [],
        "free_transfers": 1,
        "hits": 0,
        "predicted_points": 60.0,
        "budget": 100.0,
        "bank": 5.0,
    }


class TestUserActions:
    """Tests for user action methods on SeasonManagerV2."""

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
    def setup(self, db_path):
        """Create a season in READY phase with a planned squad and recommendation."""
        from src.db.repositories import (
            PlannedSquadRepository,
            RecommendationRepository,
            SeasonRepository,
        )

        repo = SeasonRepository(db_path)
        sid = repo.create_season(manager_id=123, season_name="2025-2026")
        repo.update_phase(sid, "ready")

        squad_json = _make_test_squad_json()

        PlannedSquadRepository(db_path).save_planned_squad(
            sid, 10, squad_json, "recommended",
        )
        RecommendationRepository(db_path).save_recommendation(
            sid, 10, new_squad_json=json.dumps(squad_json),
        )

        return db_path, sid

    @pytest.fixture
    def mgr(self, setup, monkeypatch):
        """A SeasonManagerV2 configured for testing (bootstrap mocked)."""
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        m = SeasonManagerV2(db_path=db_path)
        # Mock bootstrap so _get_next_gw_for_season works without cache.
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: None)
        # Patch the season's current_gw so fallback logic returns GW 10.
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)
        return m

    # ---- _require_ready_phase ----

    def test_require_ready_phase_no_season(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        _, err = mgr._require_ready_phase(99999)
        assert err is not None
        assert "No active season" in err["error"]

    def test_require_ready_phase_wrong_phase(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        sid = repo.create_season(manager_id=777, season_name="2025-2026")
        # Phase defaults to "planning", not "ready".
        mgr = SeasonManagerV2(db_path=db_path)
        _, err = mgr._require_ready_phase(777)
        assert err is not None
        assert "READY phase" in err["error"]

    def test_require_ready_phase_ok(self, setup):
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        season, err = mgr._require_ready_phase(123)
        assert err is None
        assert season is not None
        assert season["id"] == sid

    # ---- _calculate_predicted_points ----

    def test_calculate_predicted_points_normal(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        squad_json = _make_test_squad_json()
        # Captain (player 114, idx 14) has predicted_next_gw_points = 4.0 + 14*0.3 = 8.2
        # Starters are idx 0-10 (players 100-110), captain is 114 (idx 14, not a starter!)
        # Wait -- captain is idx 14 but starters are idx < 11. So captain is NOT a starter.
        # Let's fix this for the test by setting idx 10 as captain.
        squad_json["captain_id"] = 110
        for p in squad_json["players"]:
            p["is_captain"] = p["player_id"] == 110

        pts = mgr._calculate_predicted_points(squad_json)
        # Starters: idx 0-10. Captain is idx 10 (doubled).
        # idx 0: 4.0, idx 1: 4.3, ..., idx 10: 7.0
        # Sum of idx 0-10 = sum(4.0 + 0.3*i for i in 0..10) = 11*4.0 + 0.3*(10*11/2) = 44.0 + 16.5 = 60.5
        # Captain double adds extra 7.0 (already counted once in sum).
        # Total = 60.5 + 7.0 = 67.5
        assert pts == 67.5

    def test_calculate_predicted_points_triple_captain(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        squad_json = _make_test_squad_json()
        squad_json["chip"] = "3xc"
        squad_json["captain_id"] = 110
        for p in squad_json["players"]:
            p["is_captain"] = p["player_id"] == 110

        pts = mgr._calculate_predicted_points(squad_json)
        # Same as above but captain tripled: 60.5 + 2*7.0 = 74.5
        assert pts == 74.5

    def test_calculate_predicted_points_bench_boost(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        squad_json = _make_test_squad_json()
        squad_json["chip"] = "bboost"
        squad_json["captain_id"] = 110
        for p in squad_json["players"]:
            p["is_captain"] = p["player_id"] == 110

        pts = mgr._calculate_predicted_points(squad_json)
        # Starters (idx 0-10): 60.5 + captain bonus 7.0 = 67.5
        # Bench (idx 11-14): 7.3 + 7.6 + 7.9 + 8.2 = 31.0
        # Total = 67.5 + 31.0 = 98.5
        assert pts == 98.5

    # ---- accept_transfers ----

    def test_accept_transfers(self, mgr):
        result = mgr.accept_transfers(123)
        assert result.get("status") == "accepted"
        assert "planned_squad" in result

    def test_accept_transfers_wrong_phase(self, db_path):
        """Accepting transfers when not in READY phase returns error."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        repo.create_season(manager_id=888, season_name="2025-2026")
        # Phase is "planning" by default.
        mgr = SeasonManagerV2(db_path=db_path)
        result = mgr.accept_transfers(888)
        assert "error" in result

    def test_accept_transfers_no_planned_squad(self, db_path):
        """Accepting with no planned squad returns error."""
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        sid = repo.create_season(manager_id=889, season_name="2025-2026")
        repo.update_phase(sid, "ready")
        repo.update_season_gw = lambda *a: None  # no-op
        from src.db.repositories import SeasonRepository as SR2
        SR2(db_path).update_season_gw(sid, 9)
        mgr = SeasonManagerV2(db_path=db_path)
        # Mock bootstrap to None so fallback GW logic works.
        import src.season.manager_v2 as mod
        original = mod.load_bootstrap
        mod.load_bootstrap = lambda: None
        try:
            result = mgr.accept_transfers(889)
            assert "error" in result
        finally:
            mod.load_bootstrap = original

    # ---- set_captain ----

    def test_set_captain(self, mgr):
        # Player 107 is idx 7 (DEF), a starter.
        result = mgr.set_captain(123, 107)
        assert result.get("status") == "captain_set"
        assert result.get("captain_id") == 107
        squad = result["planned_squad"]
        assert squad["captain_id"] == 107
        # Verify the flag is correct on the player.
        cap = [p for p in squad["players"] if p["player_id"] == 107]
        assert len(cap) == 1
        assert cap[0]["is_captain"] is True
        # Verify exactly one captain.
        captains = [p for p in squad["players"] if p.get("is_captain")]
        assert len(captains) == 1

    def test_set_captain_not_in_squad(self, mgr):
        result = mgr.set_captain(123, 999)
        assert "error" in result

    def test_set_captain_bench_player(self, mgr):
        # Player 111 is idx 11, a bench player (starter=False).
        result = mgr.set_captain(123, 111)
        assert "error" in result
        assert "starting XI" in result["error"]

    def test_set_captain_updates_vice_captain(self, mgr):
        result = mgr.set_captain(123, 105)
        squad = result["planned_squad"]
        # There should be exactly one VC, and it should NOT be the captain.
        vcs = [p for p in squad["players"] if p.get("is_vice_captain")]
        assert len(vcs) == 1
        assert vcs[0]["player_id"] != 105

    def test_set_captain_recalculates_points(self, mgr):
        r1 = mgr.set_captain(123, 105)
        r2 = mgr.set_captain(123, 110)
        # Different captains should produce different predicted points
        # (unless they have identical predicted_next_gw_points).
        pts1 = r1["planned_squad"]["predicted_points"]
        pts2 = r2["planned_squad"]["predicted_points"]
        # Player 105 has 5.5 pts, player 110 has 7.0 pts.
        # Captain doubles, so difference = 7.0 - 5.5 = 1.5.
        assert pts2 > pts1

    # ---- lock_chip ----

    def test_lock_chip_validates_name(self, mgr):
        result = mgr.lock_chip(123, "invalid_chip")
        assert "error" in result
        assert "Invalid chip" in result["error"]

    def test_lock_chip_bboost(self, mgr):
        result = mgr.lock_chip(123, "bboost")
        assert result.get("status") == "chip_locked"
        assert result.get("chip") == "bboost"
        assert result["planned_squad"]["chip"] == "bboost"

    def test_lock_chip_3xc(self, mgr):
        result = mgr.lock_chip(123, "3xc")
        assert result.get("status") == "chip_locked"
        assert result.get("chip") == "3xc"
        assert result["planned_squad"]["chip"] == "3xc"

    def test_lock_chip_already_used(self, setup):
        """Locking a chip that was already used returns error."""
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SnapshotRepository, SeasonRepository
        # Record a snapshot where bboost was used in GW5.
        SnapshotRepository(db_path).save_gw_snapshot(
            season_id=sid, gameweek=5, chip_used="bboost",
        )
        SeasonRepository(db_path).update_season_gw(sid, 9)
        import src.season.manager_v2 as mod
        original = mod.load_bootstrap
        mod.load_bootstrap = lambda: None
        try:
            mgr = SeasonManagerV2(db_path=db_path)
            result = mgr.lock_chip(123, "bboost")
            assert "error" in result
            assert "already used" in result["error"]
        finally:
            mod.load_bootstrap = original

    def test_lock_chip_bboost_increases_points(self, mgr):
        """Bench boost should increase predicted points by including bench."""
        # First lock bboost.
        result = mgr.lock_chip(123, "bboost")
        bb_pts = result["planned_squad"]["predicted_points"]
        # Unlock and check normal points.
        result2 = mgr.unlock_chip(123)
        normal_pts = result2["planned_squad"]["predicted_points"]
        assert bb_pts > normal_pts

    # ---- unlock_chip ----

    def test_unlock_chip(self, mgr):
        mgr.lock_chip(123, "3xc")
        result = mgr.unlock_chip(123)
        assert result.get("status") == "chip_unlocked"
        assert result.get("old_chip") == "3xc"
        assert result["planned_squad"]["chip"] is None

    def test_unlock_chip_when_none(self, mgr):
        result = mgr.unlock_chip(123)
        assert result.get("status") == "no_chip"

    # ---- undo_transfers ----

    def test_undo_transfers(self, mgr):
        result = mgr.undo_transfers(123)
        assert result.get("status") == "reverted"
        assert "planned_squad" in result
        # Source should be "recommended" after undo.
        from src.db.repositories import PlannedSquadRepository
        planned = PlannedSquadRepository(mgr.db_path).get_planned_squad(
            result["planned_squad"].get("captain_id") and 1 or 1, 10,  # use season_id
        )
        # Just verify the method returned data.
        assert result["planned_squad"]["transfers_in"] == []

    def test_undo_transfers_no_recommendation(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, PlannedSquadRepository
        repo = SeasonRepository(db_path)
        sid = repo.create_season(manager_id=890, season_name="2025-2026")
        repo.update_phase(sid, "ready")
        repo.update_season_gw = lambda *a: None
        SeasonRepository(db_path).update_season_gw(sid, 9)
        PlannedSquadRepository(db_path).save_planned_squad(
            sid, 10, _make_test_squad_json(), "user_override",
        )
        import src.season.manager_v2 as mod
        original = mod.load_bootstrap
        mod.load_bootstrap = lambda: None
        try:
            mgr = SeasonManagerV2(db_path=db_path)
            result = mgr.undo_transfers(890)
            assert "error" in result
        finally:
            mod.load_bootstrap = original

    # ---- make_transfer ----

    def test_make_transfer_basic(self, setup, monkeypatch):
        """Swap two players of the same position."""
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        # Bootstrap with player 200 (MID, cost 6.0, team 8).
        bootstrap = {
            "events": [],
            "teams": [{"id": t, "code": t} for t in range(1, 21)],
            "elements": [
                {
                    "id": 200,
                    "web_name": "NewMID",
                    "element_type": 3,  # MID
                    "now_cost": 60,  # 6.0m
                    "team": 8,
                },
            ],
        }
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)

        mgr = SeasonManagerV2(db_path=db_path)
        # Player 107 is idx 7 = MID (positions: GKP*2, DEF*5, MID*5 -> idx 7 is MID).
        result = mgr.make_transfer(123, player_out_id=107, player_in_id=200)
        assert result.get("status") == "transfer_made"
        squad = result["planned_squad"]
        # Player 200 should be in the squad.
        pids = [p["player_id"] for p in squad["players"]]
        assert 200 in pids
        assert 107 not in pids
        # Transfer tracked.
        assert len(squad["transfers_in"]) == 1
        assert squad["transfers_in"][0]["player_id"] == 200
        assert len(squad["transfers_out"]) == 1
        assert squad["transfers_out"][0]["player_id"] == 107

    def test_make_transfer_position_mismatch(self, setup, monkeypatch):
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        bootstrap = {
            "events": [],
            "teams": [{"id": t, "code": t} for t in range(1, 21)],
            "elements": [
                {"id": 200, "web_name": "NewFWD", "element_type": 4, "now_cost": 60, "team": 8},
            ],
        }
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)

        mgr = SeasonManagerV2(db_path=db_path)
        # Player 107 is MID, 200 is FWD -> position mismatch.
        result = mgr.make_transfer(123, player_out_id=107, player_in_id=200)
        assert "error" in result
        assert "Position mismatch" in result["error"]

    def test_make_transfer_budget_exceeded(self, setup, monkeypatch):
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        bootstrap = {
            "events": [],
            "teams": [{"id": t, "code": t} for t in range(1, 21)],
            "elements": [
                {
                    "id": 200, "web_name": "ExpensiveMID",
                    "element_type": 3, "now_cost": 200, "team": 8,  # 20.0m
                },
            ],
        }
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)

        mgr = SeasonManagerV2(db_path=db_path)
        # Player 107 cost = 5.0 + 7*0.5 = 8.5m, bank = 5.0m.
        # Available = 8.5 + 5.0 = 13.5m. Incoming costs 20.0m -> over budget.
        result = mgr.make_transfer(123, player_out_id=107, player_in_id=200)
        assert "error" in result
        assert "budget" in result["error"].lower()

    def test_make_transfer_team_limit(self, setup, monkeypatch):
        """Cannot have more than 3 players from the same team."""
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository, PlannedSquadRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        # Modify squad so 3 players already belong to team_code 8.
        squad_json = _make_test_squad_json()
        squad_json["players"][2]["team_code"] = 8  # DEF
        squad_json["players"][3]["team_code"] = 8  # DEF
        squad_json["players"][4]["team_code"] = 8  # DEF
        PlannedSquadRepository(db_path).save_planned_squad(sid, 10, squad_json, "recommended")

        bootstrap = {
            "events": [],
            "teams": [{"id": t, "code": t} for t in range(1, 21)],
            "elements": [
                {
                    "id": 200, "web_name": "AnotherTeam8",
                    "element_type": 3, "now_cost": 60, "team": 8,  # Team code 8
                },
            ],
        }
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)

        mgr = SeasonManagerV2(db_path=db_path)
        # Player 107 (idx 7, team_code = 8 from 7%10+1=8) is MID.
        # After removing 107 (team 8), we still have 3 from team 8 (idx 2,3,4).
        result = mgr.make_transfer(123, player_out_id=107, player_in_id=200)
        assert "error" in result
        assert "Team limit" in result["error"]

    def test_make_transfer_player_not_in_squad(self, setup, monkeypatch):
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: None)
        mgr = SeasonManagerV2(db_path=db_path)
        result = mgr.make_transfer(123, player_out_id=999, player_in_id=200)
        assert "error" in result
        assert "not in the squad" in result["error"]

    def test_make_transfer_tracks_hits(self, setup, monkeypatch):
        """Second transfer incurs a hit (only 1 free transfer)."""
        db_path, sid = setup
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).update_season_gw(sid, 9)

        bootstrap = {
            "events": [],
            "teams": [{"id": t, "code": t} for t in range(1, 21)],
            "elements": [
                {"id": 200, "web_name": "NewMID1", "element_type": 3, "now_cost": 60, "team": 15},
                {"id": 201, "web_name": "NewMID2", "element_type": 3, "now_cost": 60, "team": 16},
            ],
        }
        monkeypatch.setattr("src.season.manager_v2.load_bootstrap", lambda: bootstrap)

        mgr = SeasonManagerV2(db_path=db_path)
        # First transfer (free).
        r1 = mgr.make_transfer(123, player_out_id=107, player_in_id=200)
        assert r1.get("status") == "transfer_made"
        assert r1["planned_squad"]["hits"] == 0

        # Second transfer (costs a hit).
        r2 = mgr.make_transfer(123, player_out_id=108, player_in_id=201)
        assert r2.get("status") == "transfer_made"
        assert r2["planned_squad"]["hits"] == 1


# ---------------------------------------------------------------------------
# init_season
# ---------------------------------------------------------------------------

class TestInitSeason:
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

    def test_init_season_method_exists(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert hasattr(mgr, "init_season")
        assert callable(mgr.init_season)

    def test_track_prices_simple_method_exists(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert hasattr(mgr, "_track_prices_simple")
        assert callable(mgr._track_prices_simple)
