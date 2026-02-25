"""Integration tests for FPLGaffer2."""

import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    from src.api import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_app_creates(app):
    """App factory creates without errors."""
    assert app is not None


def test_api_routes_exist(app):
    """All expected API routes are registered."""
    rules = {r.rule for r in app.url_map.iter_rules()}
    expected = [
        "/api/predictions",
        "/api/refresh-data",
        "/api/train",
        "/api/model-info",
        "/api/best-team",
        "/api/my-team",
        "/api/transfer-recommendations",
        "/api/status",
        "/api/monsters",
        "/api/season/init",
        "/api/season/status",
        "/api/season/dashboard",
        "/api/season/strategic-plan",
        "/api/season/prices",
        "/api/season/watchlist",
        "/api/v2/season/init",
        "/api/v2/season/status",
        "/api/v2/season/tick",
        "/api/v2/season/accept-transfers",
        "/api/v2/season/make-transfer",
        "/api/v2/season/undo-transfers",
        "/api/v2/season/lock-chip",
        "/api/v2/season/unlock-chip",
        "/api/v2/season/set-captain",
        "/api/v2/season/fixture-lookahead",
        "/api/v2/season/history",
        "/api/v2/season/available-players",
        "/api/v2/season/delete",
    ]
    for route in expected:
        assert route in rules, f"Missing route: {route}"


def test_predictions_endpoint(client):
    """GET /api/predictions returns 200 with data or 400 when no CSV exists."""
    resp = client.get("/api/predictions")
    assert resp.status_code in (200, 400)
    data = resp.get_json()
    if resp.status_code == 400:
        assert "error" in data
    else:
        assert "players" in data


def test_model_info_endpoint(client):
    """GET /api/model-info returns model list."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_season_status_requires_manager_id(client):
    """GET /api/season/status requires manager_id parameter."""
    resp = client.get("/api/season/status")
    assert resp.status_code == 400


def test_season_status_with_manager_id(client):
    """GET /api/season/status returns inactive for unknown manager."""
    resp = client.get("/api/season/status?manager_id=999999")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("active") is False


def test_v2_status_endpoint(client):
    """GET /api/v2/season/status returns 200 with phase info."""
    resp = client.get("/api/v2/season/status?manager_id=12345")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active" in data
    # No season initialized, so active should be False
    assert data["active"] is False


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

def test_squad_solver(sample_players_df):
    """MILP solver builds a valid 15-player squad."""
    from src.solver.squad import solve_milp_team

    result = solve_milp_team(
        sample_players_df,
        target_col="predicted_next_gw_points",
        budget=1000,
    )
    assert result is not None
    assert "starters" in result
    assert "bench" in result
    assert len(result["starters"]) == 11
    assert len(result["bench"]) == 4
    assert result["total_cost"] <= 1000


def test_squad_solver_with_captain(sample_players_df):
    """MILP solver selects a captain when captain_col provided."""
    from src.solver.squad import solve_milp_team

    result = solve_milp_team(
        sample_players_df,
        target_col="predicted_next_gw_points",
        budget=1000,
        captain_col="captain_score",
    )
    assert result is not None
    assert result.get("captain_id") is not None
    # Captain must be a starter
    starter_ids = [p["player_id"] for p in result["starters"]]
    assert result["captain_id"] in starter_ids


def test_transfer_solver(sample_players_df):
    """Transfer MILP keeps most of current squad."""
    from src.solver.transfers import solve_transfer_milp

    # Use first 15 as "current squad"
    current_ids = set(sample_players_df["player_id"].tolist()[:15])
    current_squad = sample_players_df[
        sample_players_df["player_id"].isin(current_ids)
    ]
    budget = current_squad["cost"].sum()

    result = solve_transfer_milp(
        sample_players_df,
        current_player_ids=current_ids,
        target_col="predicted_next_gw_points",
        budget=budget,
        max_transfers=2,
    )
    assert result is not None
    # Should keep at least 13 of the original 15
    new_ids = set(p["player_id"] for p in result["starters"] + result["bench"])
    kept = current_ids & new_ids
    assert len(kept) >= 13


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

def test_squad_validation():
    """FPL rule validator catches invalid squads."""
    from src.schemas.fpl_rules import validate_squad

    player_ids = list(range(1, 16))
    positions = (
        ["GKP", "GKP"] +
        ["DEF"] * 5 +
        ["MID"] * 5 +
        ["FWD"] * 3
    )
    team_codes = list(range(1, 16))  # All different teams
    total_cost = 1000

    errors = validate_squad(player_ids, positions, team_codes, total_cost, budget=1000)
    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_squad_validation_catches_wrong_count():
    """FPL rule validator detects wrong number of players."""
    from src.schemas.fpl_rules import validate_squad

    player_ids = list(range(1, 11))  # Only 10 players
    positions = ["MID"] * 10
    team_codes = list(range(1, 11))
    total_cost = 500

    errors = validate_squad(player_ids, positions, team_codes, total_cost, budget=1000)
    assert len(errors) > 0


def test_squad_validation_catches_team_limit():
    """FPL rule validator detects >3 players from same team."""
    from src.schemas.fpl_rules import validate_squad

    player_ids = list(range(1, 16))
    positions = (
        ["GKP", "GKP"] +
        ["DEF"] * 5 +
        ["MID"] * 5 +
        ["FWD"] * 3
    )
    # First 4 players from same team
    team_codes = [1, 1, 1, 1] + list(range(2, 13))
    total_cost = 1000

    errors = validate_squad(player_ids, positions, team_codes, total_cost, budget=1000)
    assert any("3" in str(e) or "team" in str(e).lower() for e in errors)


# ---------------------------------------------------------------------------
# DB tests
# ---------------------------------------------------------------------------

def test_db_schema_creation(tmp_db):
    """Database schema creates all tables."""
    from src.db.connection import get_connection
    from src.db.schema import init_schema

    conn = get_connection(tmp_db)
    init_schema(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    expected = {
        "season", "gw_snapshot", "recommendation",
        "recommendation_outcome", "price_tracker",
        "fixture_calendar", "strategic_plan", "plan_changelog",
        "watchlist",
    }
    assert expected.issubset(tables), f"Missing: {expected - tables}"


def test_season_repository(tmp_db):
    """Season repository CRUD operations."""
    from src.db.connection import get_connection
    from src.db.schema import init_schema
    from src.db.repositories import SeasonRepository

    conn = get_connection(tmp_db)
    init_schema(conn)
    conn.close()

    repo = SeasonRepository(tmp_db)

    # No season initially
    assert repo.get_season(12345) is None

    # Create season
    repo.create_season(12345, "Test Season 2025-2026", start_gw=1)
    season = repo.get_season(12345)
    assert season is not None
    assert season["manager_id"] == 12345

    # Delete
    repo.delete_season(12345)
    assert repo.get_season(12345) is None


# ---------------------------------------------------------------------------
# Feature registry tests
# ---------------------------------------------------------------------------

def test_feature_registry():
    """Feature registry provides position-specific feature lists."""
    from src.features.registry import get_features_for_position, get_sub_model_features

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        features = get_features_for_position(pos)
        assert len(features) > 0, f"No features for {pos}"

    # Sub-model features return a list for a given component
    for component in ["goals", "assists", "cs", "bonus"]:
        sub_features = get_sub_model_features(component)
        assert isinstance(sub_features, list)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_loads():
    """Config module loads all configuration dataclasses."""
    from src.config import (
        XGBConfig, EnsembleConfig, SolverConfig,
        CacheConfig, PredictionConfig,
    )

    assert XGBConfig.n_estimators > 0
    assert 0 < EnsembleConfig.decomposed_weight < 1
    assert SolverConfig.bench_weight > 0
    assert CacheConfig.fpl_api > 0
    assert PredictionConfig.pool_size > 0


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------

def test_all_modules_import():
    """Every source module imports without error."""
    import importlib

    modules = [
        "src.paths", "src.config", "src.logging_config",
        "src.data.cache", "src.data.fpl_api", "src.data.github_csv",
        "src.data.loader", "src.data.season_detection",
        "src.features.registry", "src.features.builder",
        "src.ml.training", "src.ml.prediction", "src.ml.multi_gw",
        "src.ml.decomposed", "src.ml.model_store", "src.ml.backtest",
        "src.solver.squad", "src.solver.transfers", "src.solver.formation",
        "src.solver.validator",
        "src.strategy.chip_evaluator", "src.strategy.transfer_planner",
        "src.strategy.captain_planner", "src.strategy.plan_synthesizer",
        "src.strategy.reactive", "src.strategy.price_tracker",
        "src.season.manager", "src.season.manager_v2", "src.season.state_machine",
        "src.season.recorder", "src.season.dashboard",
        "src.season.fixtures", "src.season.preseason",
        "src.db.connection", "src.db.schema", "src.db.migrations",
        "src.db.repositories",
        "src.api",
    ]
    for mod in modules:
        importlib.import_module(mod)
