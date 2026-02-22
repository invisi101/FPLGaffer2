"""Shared test fixtures for FPLGaffer2."""

import pandas as pd
import pytest


@pytest.fixture
def sample_players_df():
    """Minimal player DataFrame for solver/prediction tests."""
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "web_name": [
            "GK1", "GK2", "DEF1", "DEF2", "DEF3", "DEF4", "DEF5",
            "MID1", "MID2", "MID3", "MID4", "MID5",
            "FWD1", "FWD2", "FWD3",
            "DEF6", "MID6", "FWD4", "GK3", "MID7",
        ],
        "position": [
            "GKP", "GKP", "DEF", "DEF", "DEF", "DEF", "DEF",
            "MID", "MID", "MID", "MID", "MID",
            "FWD", "FWD", "FWD",
            "DEF", "MID", "FWD", "GKP", "MID",
        ],
        "cost": [
            45, 40, 60, 55, 50, 45, 40,
            120, 100, 80, 65, 50,
            130, 90, 55,
            45, 55, 60, 40, 50,
        ],
        "team_code": [
            1, 2, 1, 2, 3, 4, 5,
            1, 2, 3, 4, 5,
            1, 2, 3,
            6, 6, 7, 8, 9,
        ],
        "predicted_next_gw_points": [
            3.5, 3.0, 4.5, 4.2, 4.0, 3.8, 3.5,
            7.0, 6.5, 5.5, 5.0, 4.5,
            8.0, 6.0, 4.0,
            3.2, 4.8, 5.5, 2.8, 4.3,
        ],
        "captain_score": [
            2.0, 1.8, 3.0, 2.8, 2.5, 2.3, 2.0,
            8.5, 7.5, 6.0, 5.5, 4.8,
            9.5, 7.0, 4.2,
            2.0, 5.0, 6.0, 1.5, 4.5,
        ],
    })


@pytest.fixture
def bootstrap_data():
    """Minimal FPL bootstrap-static response."""
    return {
        "events": [
            {"id": 1, "deadline_time": "2025-08-16T10:00:00Z", "data_checked": True,
             "is_current": False, "is_next": False, "finished": True},
            {"id": 2, "deadline_time": "2025-08-23T10:00:00Z", "data_checked": True,
             "is_current": True, "is_next": False, "finished": True},
            {"id": 3, "deadline_time": "2025-08-30T10:00:00Z", "data_checked": False,
             "is_current": False, "is_next": True, "finished": False},
        ],
        "elements": [
            {"id": 1, "web_name": "GK1", "element_type": 1, "team": 1,
             "team_code": 1, "now_cost": 45, "form": "3.5",
             "chance_of_playing_next_round": 100,
             "ep_next": "3.5", "selected_by_percent": "5.0",
             "transfers_in_event": 100, "transfers_out_event": 50},
        ],
        "teams": [
            {"id": 1, "code": 1, "name": "Team A", "short_name": "TMA", "strength": 4},
        ],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
    }


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary database path for DB tests."""
    return tmp_path / "test_season.db"
