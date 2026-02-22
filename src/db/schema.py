"""Database schema — all CREATE TABLE statements.

Extracted from v1's ``SeasonDB._init_tables()`` with all migrated columns
(team_code, free_transfers, base_points, current_xi_points) included
in the canonical schema.
"""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS season (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manager_id INTEGER NOT NULL,
    manager_name TEXT,
    team_name TEXT,
    season_name TEXT NOT NULL,
    start_gw INTEGER NOT NULL DEFAULT 1,
    current_gw INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(manager_id, season_name)
);

CREATE TABLE IF NOT EXISTS gw_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    squad_json TEXT,
    bank REAL,
    team_value REAL,
    free_transfers INTEGER,
    chip_used TEXT,
    points INTEGER,
    total_points INTEGER,
    overall_rank INTEGER,
    transfers_in_json TEXT,
    transfers_out_json TEXT,
    captain_id INTEGER,
    captain_name TEXT,
    transfers_cost INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season_id, gameweek)
);

CREATE TABLE IF NOT EXISTS recommendation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    transfers_json TEXT,
    captain_id INTEGER,
    captain_name TEXT,
    chip_suggestion TEXT,
    chip_values_json TEXT,
    bank_analysis_json TEXT,
    new_squad_json TEXT,
    predicted_points REAL,
    base_points REAL,
    current_xi_points REAL,
    free_transfers INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season_id, gameweek)
);

CREATE TABLE IF NOT EXISTS recommendation_outcome (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    followed_transfers INTEGER,
    followed_captain INTEGER,
    followed_chip INTEGER,
    recommended_points REAL,
    actual_points REAL,
    point_delta REAL,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season_id, gameweek)
);

CREATE TABLE IF NOT EXISTS price_tracker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    web_name TEXT,
    team_code INTEGER,
    price REAL NOT NULL,
    transfers_in_event INTEGER,
    transfers_out_event INTEGER,
    snapshot_date TEXT NOT NULL DEFAULT (date('now')),
    UNIQUE(season_id, player_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS fixture_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL,
    team_code INTEGER,
    team_short TEXT,
    gameweek INTEGER NOT NULL,
    fixture_count INTEGER NOT NULL DEFAULT 1,
    opponents_json TEXT,
    fdr_avg REAL,
    is_dgw INTEGER DEFAULT 0,
    is_bgw INTEGER DEFAULT 0,
    UNIQUE(season_id, team_id, gameweek)
);

CREATE TABLE IF NOT EXISTS strategic_plan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    as_of_gw INTEGER NOT NULL,
    plan_json TEXT,
    chip_heatmap_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season_id, as_of_gw)
);

CREATE TABLE IF NOT EXISTS plan_changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    gameweek INTEGER NOT NULL,
    change_type TEXT NOT NULL,
    description TEXT,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    web_name TEXT,
    team_code INTEGER,
    price_when_added REAL,
    added_date TEXT NOT NULL DEFAULT (date('now')),
    UNIQUE(season_id, player_id)
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Execute the full schema DDL on *conn*."""
    conn.executescript(SCHEMA_SQL)
