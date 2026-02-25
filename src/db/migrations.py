"""Versioned database migration system.

Replaces the try/except ALTER TABLE pattern from v1 with a proper
``schema_version`` table and numbered migration functions.
"""

from __future__ import annotations

import sqlite3

from src.db.schema import init_schema
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Version tracking table ─────────────────────────────────────────────

_VERSION_DDL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL DEFAULT 0
);
"""


def _ensure_version_table(conn: sqlite3.Connection) -> None:
    """Create the ``schema_version`` table if it does not exist."""
    conn.executescript(_VERSION_DDL)
    # Seed row if empty
    row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (id, version) VALUES (1, 0)")
        conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version (0 means brand-new database)."""
    _ensure_version_table(conn)
    row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    return row[0] if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Update the stored schema version."""
    conn.execute(
        "UPDATE schema_version SET version = ? WHERE id = 1", (version,)
    )
    conn.commit()


# ── Migrations ─────────────────────────────────────────────────────────

def _migration_001_initial_schema(conn: sqlite3.Connection) -> None:
    """Migration 1: create all tables (season, gw_snapshot, recommendation,
    recommendation_outcome, price_tracker, fixture_calendar,
    strategic_plan, plan_changelog, watchlist).
    """
    init_schema(conn)


def _migration_002_planned_squad_and_phase(conn: sqlite3.Connection) -> None:
    """Add planned_squad table and phase column to season."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS planned_squad (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season_id INTEGER NOT NULL REFERENCES season(id) ON DELETE CASCADE,
            gameweek INTEGER NOT NULL,
            squad_json TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'recommended',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season_id, gameweek)
        )
    """)
    try:
        conn.execute("ALTER TABLE season ADD COLUMN phase TEXT NOT NULL DEFAULT 'planning'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()


def _migration_003_drop_strategy_tables(conn: sqlite3.Connection) -> None:
    """Remove strategic_plan and plan_changelog tables (v2 redesign)."""
    conn.execute("DROP TABLE IF EXISTS strategic_plan")
    conn.execute("DROP TABLE IF EXISTS plan_changelog")
    conn.commit()


# Registry: version number -> migration function.
# Each migration brings the DB from (version - 1) to (version).
_MIGRATIONS: dict[int, callable] = {
    1: _migration_001_initial_schema,
    2: _migration_002_planned_squad_and_phase,
    3: _migration_003_drop_strategy_tables,
}

LATEST_VERSION: int = max(_MIGRATIONS)


# ── Public API ─────────────────────────────────────────────────────────

def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to bring the database up to date.

    Safe to call on every startup — already-applied migrations are
    skipped.
    """
    current = get_schema_version(conn)

    if current >= LATEST_VERSION:
        return

    for version in range(current + 1, LATEST_VERSION + 1):
        migration_fn = _MIGRATIONS.get(version)
        if migration_fn is None:
            raise RuntimeError(
                f"Missing migration function for version {version}"
            )
        logger.info("Applying migration %d: %s", version, migration_fn.__doc__.strip().split('\n')[0])
        migration_fn(conn)
        _set_schema_version(conn, version)

    logger.info("Database schema is now at version %d", LATEST_VERSION)
