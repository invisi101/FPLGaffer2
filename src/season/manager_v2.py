"""Season Manager v2 -- state-machine-driven GW lifecycle.

Replaces the monolithic SeasonManager with a thin orchestrator around
a GW state machine (PLANNING -> READY -> LIVE -> COMPLETE -> repeat).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.api.helpers import get_next_gw, load_bootstrap
from src.db.connection import connect
from src.db.migrations import apply_migrations
from src.db.repositories import (
    DashboardRepository,
    FixtureRepository,
    OutcomeRepository,
    PlannedSquadRepository,
    PriceRepository,
    RecommendationRepository,
    SeasonRepository,
    SnapshotRepository,
    WatchlistRepository,
)
from src.paths import CACHE_DIR, DB_PATH
from src.season.state_machine import GWPhase, can_transition, detect_phase

logger = logging.getLogger(__name__)


class SeasonManagerV2:
    """State-machine-driven season orchestrator.

    Each call to :meth:`tick` inspects the current GW phase and
    performs exactly one phase's work (generate recommendation,
    check injuries, detect GW completion, record results).
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Run migrations to ensure schema is current.
        with connect(self.db_path) as conn:
            apply_migrations(conn)

        # Repositories -- one per table.
        self.seasons = SeasonRepository(self.db_path)
        self.snapshots = SnapshotRepository(self.db_path)
        self.recommendations = RecommendationRepository(self.db_path)
        self.outcomes = OutcomeRepository(self.db_path)
        self.prices = PriceRepository(self.db_path)
        self.fixtures = FixtureRepository(self.db_path)
        self.planned_squads = PlannedSquadRepository(self.db_path)
        self.dashboard = DashboardRepository(self.db_path)
        self.watchlist = WatchlistRepository(self.db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self, manager_id: int) -> dict:
        """Return the current season status for *manager_id*.

        Detects the GW phase from real-world signals (deadline,
        fixture completion, recommendation existence) and updates
        the stored phase if it has changed.
        """
        season = self.seasons.get_season(manager_id)
        if not season:
            return {"active": False, "phase": None}

        season_id = season["id"]

        # Bootstrap / GW detection
        bootstrap = load_bootstrap()
        next_gw = get_next_gw(bootstrap) if bootstrap else None

        # Current GW is the one before next (or the latest finished).
        current_gw = (next_gw - 1) if next_gw and next_gw > 1 else (next_gw or 1)

        # Deadline
        deadline = self._get_deadline(bootstrap, next_gw) if bootstrap and next_gw else None
        deadline_passed = (
            deadline is not None and datetime.now(timezone.utc) > deadline
        )

        # Fixture completion for current GW
        all_fixtures_finished = self._is_gw_finished(current_gw) if current_gw else False

        # Recommendation existence for next GW
        rec = self.recommendations.get_recommendation(season_id, next_gw) if next_gw else None
        has_recommendation = rec is not None

        # Detect phase
        detected = detect_phase(
            has_recommendation=has_recommendation,
            deadline_passed=deadline_passed,
            all_fixtures_finished=all_fixtures_finished,
        )

        # Update stored phase if changed.
        stored_phase_str = season.get("phase", GWPhase.PLANNING.value)
        if detected.value != stored_phase_str:
            if can_transition(GWPhase(stored_phase_str), detected):
                self.seasons.update_phase(season_id, detected.value)
                logger.info(
                    "Phase transition: %s -> %s (GW%s, manager %d)",
                    stored_phase_str, detected.value, next_gw, manager_id,
                )
            else:
                # Forced update -- detect_phase is authoritative.
                self.seasons.update_phase(season_id, detected.value)
                logger.warning(
                    "Forced phase update: %s -> %s (GW%s, manager %d)",
                    stored_phase_str, detected.value, next_gw, manager_id,
                )

        # Planned squad
        planned = (
            self.planned_squads.get_planned_squad(season_id, next_gw)
            if next_gw
            else None
        )

        return {
            "active": True,
            "phase": detected.value,
            "gw": next_gw,
            "current_gw": current_gw,
            "deadline": deadline.isoformat() if deadline else None,
            "deadline_passed": deadline_passed,
            "has_recommendation": has_recommendation,
            "planned_squad": planned,
            "season_id": season_id,
            "manager_id": manager_id,
        }

    def tick(self, manager_id: int, progress_fn=None) -> list[dict]:
        """Run one tick of the state machine.

        Inspects the current phase and performs the appropriate work.
        Returns a list of alert dicts (suitable for SSE broadcast).
        """
        status = self.get_status(manager_id)
        if not status["active"]:
            return []

        phase = status["phase"]
        season = self.seasons.get_season(manager_id)

        if phase == GWPhase.PLANNING.value:
            return self._tick_planning(manager_id, season, status, progress_fn)
        if phase == GWPhase.READY.value:
            return self._tick_ready(manager_id, season, status)
        if phase == GWPhase.LIVE.value:
            return self._tick_live(manager_id, season, status)
        if phase == GWPhase.COMPLETE.value:
            return self._tick_complete(manager_id, season, status)
        # SEASON_OVER -- nothing to do.
        return []

    # ------------------------------------------------------------------
    # Phase tick handlers
    # ------------------------------------------------------------------

    def _tick_planning(self, manager_id, season, status, progress_fn=None):
        """Generate predictions and recommendations. (Implemented in Task 13.)"""
        logger.info("PLANNING phase: recommendation generation not yet implemented")
        return []

    def _tick_ready(self, manager_id, season, status):
        """Check for injuries in the planned squad.

        Compares planned squad players against bootstrap ``elements``
        for availability changes since the recommendation was generated.
        """
        alerts: list[dict] = []

        bootstrap = load_bootstrap()
        if not bootstrap:
            return alerts

        season_id = status["season_id"]
        next_gw = status.get("gw")
        if not next_gw:
            return alerts

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return alerts

        squad_data = planned.get("squad_json", {})
        players = squad_data.get("players", [])
        if not players:
            return alerts

        # Build lookup of element availability from bootstrap.
        elements_by_id: dict[int, dict] = {
            el["id"]: el for el in bootstrap.get("elements", [])
        }

        for player in players:
            pid = player.get("player_id") or player.get("id")
            if pid is None:
                continue

            el = elements_by_id.get(int(pid))
            if el is None:
                continue

            status_code = el.get("status", "a")  # a=available, i=injured, s=suspended, u=unavailable
            web_name = el.get("web_name", f"Player {pid}")
            is_starter = player.get("starter", False)
            is_captain = player.get("is_captain", False)

            if status_code in ("i", "s", "u"):
                severity = "critical" if (is_captain or is_starter) else "warning"
                status_label = {
                    "i": "injured",
                    "s": "suspended",
                    "u": "unavailable",
                }.get(status_code, "unavailable")

                role = ""
                if is_captain:
                    role = "Captain "
                elif is_starter:
                    role = "Starter "

                alerts.append({
                    "type": "injury",
                    "player_id": int(pid),
                    "message": f"{role}{web_name} is now {status_label}",
                    "severity": severity,
                })

        return alerts

    def _tick_live(self, manager_id, season, status):
        """Check if all fixtures for the current GW are finished."""
        alerts: list[dict] = []

        current_gw = status.get("current_gw")
        if not current_gw:
            return alerts

        if self._is_gw_finished(current_gw):
            season_id = status["season_id"]
            self.seasons.update_phase(season_id, GWPhase.COMPLETE.value)
            alerts.append({
                "type": "gw_complete",
                "message": f"GW{current_gw} complete",
            })
            logger.info("GW%d all fixtures finished -- transitioning to COMPLETE", current_gw)

        return alerts

    def _tick_complete(self, manager_id, season, status):
        """Record results and advance to next GW. (Implemented in Task 14.)"""
        logger.info("COMPLETE phase: result recording not yet implemented")
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_fixtures(self) -> list[dict]:
        """Load fixtures from cache."""
        path = CACHE_DIR / "fpl_api_fixtures.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load fixtures from %s", path)
            return []

    def _is_gw_finished(self, gw: int) -> bool:
        """Return True if all fixtures for *gw* have ``finished == True``."""
        fixtures = self._load_fixtures()
        gw_fixtures = [f for f in fixtures if f.get("event") == gw]
        if not gw_fixtures:
            return False
        return all(f.get("finished", False) for f in gw_fixtures)

    def _get_deadline(self, bootstrap: dict | None, gw: int) -> datetime | None:
        """Parse the deadline for *gw* from bootstrap events."""
        if not bootstrap:
            return None
        for ev in bootstrap.get("events", []):
            if ev.get("id") == gw:
                dl_str = ev.get("deadline_time")
                if dl_str:
                    try:
                        # FPL API format: "2025-08-16T10:00:00Z"
                        return datetime.fromisoformat(
                            dl_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        return None
        return None
