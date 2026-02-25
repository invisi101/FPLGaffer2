"""Database layer — connection, schema, migrations, and repositories."""

from src.db.connection import connect, get_connection
from src.db.migrations import apply_migrations, get_schema_version
from src.db.repositories import (
    DashboardRepository,
    FixtureRepository,
    OutcomeRepository,
    PlannedSquadRepository,
    PlanRepository,
    PriceRepository,
    RecommendationRepository,
    SeasonRepository,
    SnapshotRepository,
    WatchlistRepository,
)
from src.db.schema import SCHEMA_SQL, init_schema

__all__ = [
    "connect",
    "get_connection",
    "apply_migrations",
    "get_schema_version",
    "init_schema",
    "SCHEMA_SQL",
    "SeasonRepository",
    "SnapshotRepository",
    "RecommendationRepository",
    "OutcomeRepository",
    "PriceRepository",
    "FixtureRepository",
    "DashboardRepository",
    "PlanRepository",
    "PlannedSquadRepository",
    "WatchlistRepository",
]
