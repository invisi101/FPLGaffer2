"""Repository classes — one per database table.

All SQL queries are carried over verbatim from v1's ``SeasonDB``.
Each repository takes a ``db_path`` in ``__init__`` and uses
:func:`src.db.connection.connect` for every operation.
"""

from __future__ import annotations

from pathlib import Path

from src.db.connection import connect
from src.paths import DB_PATH


# ---------------------------------------------------------------------------
# SeasonRepository
# ---------------------------------------------------------------------------

class SeasonRepository:
    """CRUD for the ``season`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def create_season(
        self,
        manager_id: int,
        manager_name: str = "",
        team_name: str = "",
        season_name: str = "",
        start_gw: int = 1,
    ) -> int:
        if not season_name:
            from src.data.season_detection import detect_current_season
            season_name = detect_current_season()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO season
                   (manager_id, manager_name, team_name, season_name, start_gw, current_gw)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(manager_id, season_name) DO UPDATE SET
                     manager_name=excluded.manager_name,
                     team_name=excluded.team_name""",
                (manager_id, manager_name, team_name, season_name, start_gw, start_gw),
            )
            season_id = cur.lastrowid
            if not season_id:
                row = conn.execute(
                    "SELECT id FROM season WHERE manager_id=? AND season_name=?",
                    (manager_id, season_name),
                ).fetchone()
                season_id = row[0]
            conn.commit()
        return season_id

    def get_season(self, manager_id: int, season_name: str = "") -> dict | None:
        if not season_name:
            from src.data.season_detection import detect_current_season
            season_name = detect_current_season()
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM season WHERE manager_id=? AND season_name=?",
                (manager_id, season_name),
            ).fetchone()
        return dict(row) if row else None

    def delete_season(self, manager_id: int, season_name: str = "") -> None:
        if not season_name:
            from src.data.season_detection import detect_current_season
            season_name = detect_current_season()
        with connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM season WHERE manager_id=? AND season_name=?",
                (manager_id, season_name),
            )
            conn.commit()

    def update_season_gw(self, season_id: int, current_gw: int) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE season SET current_gw=? WHERE id=?",
                (current_gw, season_id),
            )
            conn.commit()

    def clear_generated_data(self, season_id: int) -> None:
        """Clear recommendations, outcomes, strategic plans, and changelog.

        Called on season re-init so stale generated data does not persist.
        Preserves snapshots (actual historical data), prices, and fixtures.
        """
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM recommendation WHERE season_id=?", (season_id,))
            conn.execute("DELETE FROM recommendation_outcome WHERE season_id=?", (season_id,))
            conn.execute("DELETE FROM strategic_plan WHERE season_id=?", (season_id,))
            conn.execute("DELETE FROM plan_changelog WHERE season_id=?", (season_id,))
            conn.commit()


# ---------------------------------------------------------------------------
# SnapshotRepository
# ---------------------------------------------------------------------------

class SnapshotRepository:
    """CRUD for the ``gw_snapshot`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_gw_snapshot(self, season_id: int, gameweek: int, **kwargs) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO gw_snapshot
                   (season_id, gameweek, squad_json, bank, team_value, free_transfers,
                    chip_used, points, total_points, overall_rank,
                    transfers_in_json, transfers_out_json, captain_id, captain_name,
                    transfers_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(season_id, gameweek) DO UPDATE SET
                     squad_json=excluded.squad_json,
                     bank=excluded.bank,
                     team_value=excluded.team_value,
                     free_transfers=excluded.free_transfers,
                     chip_used=excluded.chip_used,
                     points=excluded.points,
                     total_points=excluded.total_points,
                     overall_rank=excluded.overall_rank,
                     transfers_in_json=excluded.transfers_in_json,
                     transfers_out_json=excluded.transfers_out_json,
                     captain_id=excluded.captain_id,
                     captain_name=excluded.captain_name,
                     transfers_cost=excluded.transfers_cost,
                     created_at=datetime('now')""",
                (
                    season_id, gameweek,
                    kwargs.get("squad_json"),
                    kwargs.get("bank"),
                    kwargs.get("team_value"),
                    kwargs.get("free_transfers"),
                    kwargs.get("chip_used"),
                    kwargs.get("points"),
                    kwargs.get("total_points"),
                    kwargs.get("overall_rank"),
                    kwargs.get("transfers_in_json"),
                    kwargs.get("transfers_out_json"),
                    kwargs.get("captain_id"),
                    kwargs.get("captain_name"),
                    kwargs.get("transfers_cost", 0),
                ),
            )
            conn.commit()

    def get_snapshots(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM gw_snapshot WHERE season_id=? ORDER BY gameweek",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshot(self, season_id: int, gameweek: int) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM gw_snapshot WHERE season_id=? AND gameweek=?",
                (season_id, gameweek),
            ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# RecommendationRepository
# ---------------------------------------------------------------------------

class RecommendationRepository:
    """CRUD for the ``recommendation`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_recommendation(self, season_id: int, gameweek: int, **kwargs) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO recommendation
                   (season_id, gameweek, transfers_json, captain_id, captain_name,
                    chip_suggestion, chip_values_json, bank_analysis_json,
                    new_squad_json, predicted_points, base_points,
                    current_xi_points, free_transfers)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(season_id, gameweek) DO UPDATE SET
                     transfers_json=excluded.transfers_json,
                     captain_id=excluded.captain_id,
                     captain_name=excluded.captain_name,
                     chip_suggestion=excluded.chip_suggestion,
                     chip_values_json=excluded.chip_values_json,
                     bank_analysis_json=excluded.bank_analysis_json,
                     new_squad_json=excluded.new_squad_json,
                     predicted_points=excluded.predicted_points,
                     base_points=excluded.base_points,
                     current_xi_points=excluded.current_xi_points,
                     free_transfers=excluded.free_transfers,
                     created_at=datetime('now')""",
                (
                    season_id, gameweek,
                    kwargs.get("transfers_json"),
                    kwargs.get("captain_id"),
                    kwargs.get("captain_name"),
                    kwargs.get("chip_suggestion"),
                    kwargs.get("chip_values_json"),
                    kwargs.get("bank_analysis_json"),
                    kwargs.get("new_squad_json"),
                    kwargs.get("predicted_points"),
                    kwargs.get("base_points"),
                    kwargs.get("current_xi_points"),
                    kwargs.get("free_transfers"),
                ),
            )
            conn.commit()

    def get_recommendations(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM recommendation WHERE season_id=? ORDER BY gameweek",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recommendation(self, season_id: int, gameweek: int) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM recommendation WHERE season_id=? AND gameweek=?",
                (season_id, gameweek),
            ).fetchone()
        return dict(row) if row else None

    def update_chip_suggestion(
        self, season_id: int, gameweek: int, chip: str | None,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE recommendation SET chip_suggestion=? WHERE season_id=? AND gameweek=?",
                (chip, season_id, gameweek),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# OutcomeRepository
# ---------------------------------------------------------------------------

class OutcomeRepository:
    """CRUD for the ``recommendation_outcome`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_outcome(self, season_id: int, gameweek: int, **kwargs) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO recommendation_outcome
                   (season_id, gameweek, followed_transfers, followed_captain,
                    followed_chip, recommended_points, actual_points, point_delta,
                    details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(season_id, gameweek) DO UPDATE SET
                     followed_transfers=excluded.followed_transfers,
                     followed_captain=excluded.followed_captain,
                     followed_chip=excluded.followed_chip,
                     recommended_points=excluded.recommended_points,
                     actual_points=excluded.actual_points,
                     point_delta=excluded.point_delta,
                     details_json=excluded.details_json,
                     created_at=datetime('now')""",
                (
                    season_id, gameweek,
                    kwargs.get("followed_transfers"),
                    kwargs.get("followed_captain"),
                    kwargs.get("followed_chip"),
                    kwargs.get("recommended_points"),
                    kwargs.get("actual_points"),
                    kwargs.get("point_delta"),
                    kwargs.get("details_json"),
                ),
            )
            conn.commit()

    def get_outcomes(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM recommendation_outcome WHERE season_id=? ORDER BY gameweek",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PriceRepository
# ---------------------------------------------------------------------------

class PriceRepository:
    """CRUD for the ``price_tracker`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_price_snapshot(self, season_id: int, player_id: int, **kwargs) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO price_tracker
                   (season_id, player_id, web_name, team_code, price,
                    transfers_in_event, transfers_out_event, snapshot_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, date('now'))
                   ON CONFLICT(season_id, player_id, snapshot_date) DO UPDATE SET
                     web_name=excluded.web_name,
                     team_code=excluded.team_code,
                     price=excluded.price,
                     transfers_in_event=excluded.transfers_in_event,
                     transfers_out_event=excluded.transfers_out_event""",
                (
                    season_id, player_id,
                    kwargs.get("web_name"),
                    kwargs.get("team_code"),
                    kwargs.get("price"),
                    kwargs.get("transfers_in_event"),
                    kwargs.get("transfers_out_event"),
                ),
            )
            conn.commit()

    def save_price_snapshots_bulk(self, season_id: int, players: list[dict]) -> None:
        with connect(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO price_tracker
                   (season_id, player_id, web_name, team_code, price,
                    transfers_in_event, transfers_out_event, snapshot_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, date('now'))
                   ON CONFLICT(season_id, player_id, snapshot_date) DO UPDATE SET
                     web_name=excluded.web_name,
                     team_code=excluded.team_code,
                     price=excluded.price,
                     transfers_in_event=excluded.transfers_in_event,
                     transfers_out_event=excluded.transfers_out_event""",
                [
                    (
                        season_id, p["player_id"], p.get("web_name"), p.get("team_code"),
                        p["price"], p.get("transfers_in_event", 0),
                        p.get("transfers_out_event", 0),
                    )
                    for p in players
                ],
            )
            conn.commit()

    def get_price_history(
        self, season_id: int, player_id: int | None = None
    ) -> list[dict]:
        with connect(self.db_path) as conn:
            if player_id is not None:
                rows = conn.execute(
                    """SELECT * FROM price_tracker
                       WHERE season_id=? AND player_id=?
                       ORDER BY snapshot_date""",
                    (season_id, player_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM price_tracker
                       WHERE season_id=?
                       ORDER BY snapshot_date, player_id""",
                    (season_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_prices(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT pt.* FROM price_tracker pt
                   INNER JOIN (
                       SELECT player_id, MAX(snapshot_date) as max_date
                       FROM price_tracker WHERE season_id=?
                       GROUP BY player_id
                   ) latest ON pt.player_id = latest.player_id
                           AND pt.snapshot_date = latest.max_date
                   WHERE pt.season_id=?
                   ORDER BY pt.web_name""",
                (season_id, season_id),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# FixtureRepository
# ---------------------------------------------------------------------------

class FixtureRepository:
    """CRUD for the ``fixture_calendar`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_fixture_calendar(self, season_id: int, fixtures: list[dict]) -> None:
        with connect(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO fixture_calendar
                   (season_id, team_id, team_code, team_short, gameweek, fixture_count,
                    opponents_json, fdr_avg, is_dgw, is_bgw)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(season_id, team_id, gameweek) DO UPDATE SET
                     team_code=excluded.team_code,
                     team_short=excluded.team_short,
                     fixture_count=excluded.fixture_count,
                     opponents_json=excluded.opponents_json,
                     fdr_avg=excluded.fdr_avg,
                     is_dgw=excluded.is_dgw,
                     is_bgw=excluded.is_bgw""",
                [
                    (
                        season_id, f["team_id"], f.get("team_code"),
                        f.get("team_short"), f["gameweek"],
                        f.get("fixture_count", 1), f.get("opponents_json"),
                        f.get("fdr_avg"), f.get("is_dgw", 0), f.get("is_bgw", 0),
                    )
                    for f in fixtures
                ],
            )
            conn.commit()

    def get_fixture_calendar(
        self,
        season_id: int,
        from_gw: int | None = None,
        to_gw: int | None = None,
    ) -> list[dict]:
        with connect(self.db_path) as conn:
            query = "SELECT * FROM fixture_calendar WHERE season_id=?"
            params: list = [season_id]
            if from_gw is not None:
                query += " AND gameweek >= ?"
                params.append(from_gw)
            if to_gw is not None:
                query += " AND gameweek <= ?"
                params.append(to_gw)
            query += " ORDER BY gameweek, team_short"
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PlanRepository
# ---------------------------------------------------------------------------

class PlanRepository:
    """CRUD for ``strategic_plan`` and ``plan_changelog`` tables."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def save_strategic_plan(
        self,
        season_id: int,
        as_of_gw: int,
        plan_json: str,
        chip_heatmap_json: str,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO strategic_plan
                   (season_id, as_of_gw, plan_json, chip_heatmap_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(season_id, as_of_gw) DO UPDATE SET
                     plan_json=excluded.plan_json,
                     chip_heatmap_json=excluded.chip_heatmap_json,
                     created_at=datetime('now')""",
                (season_id, as_of_gw, plan_json, chip_heatmap_json),
            )
            conn.commit()

    def get_strategic_plan(
        self, season_id: int, as_of_gw: int | None = None
    ) -> dict | None:
        with connect(self.db_path) as conn:
            if as_of_gw is not None:
                row = conn.execute(
                    "SELECT * FROM strategic_plan WHERE season_id=? AND as_of_gw=?",
                    (season_id, as_of_gw),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM strategic_plan WHERE season_id=? ORDER BY as_of_gw DESC LIMIT 1",
                    (season_id,),
                ).fetchone()
        return dict(row) if row else None

    def save_plan_change(
        self,
        season_id: int,
        gameweek: int,
        change_type: str,
        description: str,
        old_value: str = "",
        new_value: str = "",
        reason: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO plan_changelog
                   (season_id, gameweek, change_type, description, old_value, new_value, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (season_id, gameweek, change_type, description, old_value, new_value, reason),
            )
            conn.commit()

    def get_plan_changelog(self, season_id: int, limit: int = 50) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM plan_changelog WHERE season_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (season_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# DashboardRepository
# ---------------------------------------------------------------------------

class DashboardRepository:
    """Aggregation queries for the Season dashboard."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def get_rank_history(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT gameweek, overall_rank, total_points
                   FROM gw_snapshot WHERE season_id=?
                   ORDER BY gameweek""",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_budget_history(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT gameweek, bank, team_value,
                          COALESCE(bank, 0) + COALESCE(team_value, 0) as total_value
                   FROM gw_snapshot WHERE season_id=?
                   ORDER BY gameweek""",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chips_status(self, season_id: int) -> list[dict]:
        """Return chips used (from snapshots) with GW info."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT gameweek, chip_used FROM gw_snapshot
                   WHERE season_id=? AND chip_used IS NOT NULL AND chip_used != ''
                   ORDER BY gameweek""",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recommendation_accuracy(self, season_id: int) -> dict:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN followed_transfers = 1 THEN 1 ELSE 0 END) as transfers_followed,
                          SUM(CASE WHEN followed_captain = 1 THEN 1 ELSE 0 END) as captain_followed,
                          AVG(point_delta) as avg_delta,
                          AVG(recommended_points) as avg_rec_points,
                          AVG(actual_points) as avg_actual_points
                   FROM recommendation_outcome WHERE season_id=?""",
                (season_id,),
            ).fetchone()
        return dict(rows) if rows else {}

    def get_transfer_history(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT gs.gameweek, gs.transfers_in_json, gs.transfers_out_json,
                          gs.transfers_cost, gs.chip_used,
                          ro.followed_transfers, ro.point_delta
                   FROM gw_snapshot gs
                   LEFT JOIN recommendation_outcome ro
                       ON gs.season_id = ro.season_id AND gs.gameweek = ro.gameweek
                   WHERE gs.season_id=?
                   ORDER BY gs.gameweek""",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# WatchlistRepository
# ---------------------------------------------------------------------------

class WatchlistRepository:
    """CRUD for the ``watchlist`` table."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def get_watchlist(self, season_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE season_id=? ORDER BY added_date DESC",
                (season_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_to_watchlist(
        self, season_id: int, player_id: int, **kwargs
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO watchlist
                   (season_id, player_id, web_name, team_code, price_when_added)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(season_id, player_id) DO UPDATE SET
                     web_name=excluded.web_name,
                     team_code=excluded.team_code,
                     price_when_added=excluded.price_when_added""",
                (
                    season_id, player_id,
                    kwargs.get("web_name"),
                    kwargs.get("team_code"),
                    kwargs.get("price_when_added"),
                ),
            )
            conn.commit()

    def remove_from_watchlist(self, season_id: int, player_id: int) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM watchlist WHERE season_id=? AND player_id=?",
                (season_id, player_id),
            )
            conn.commit()
