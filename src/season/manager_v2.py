"""Season Manager v2 -- state-machine-driven GW lifecycle.

Replaces the monolithic SeasonManager with a thin orchestrator around
a GW state machine (PLANNING -> READY -> LIVE -> COMPLETE -> repeat).
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.api.helpers import (
    ELEMENT_TYPE_MAP,
    get_next_gw,
    load_bootstrap,
    optimize_starting_xi,
)
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

VALID_CHIPS = {"bboost", "3xc", "freehit", "wildcard"}

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

    # ------------------------------------------------------------------
    # User action methods — callable during READY phase
    # ------------------------------------------------------------------

    def _require_ready_phase(self, manager_id: int) -> tuple[dict | None, dict | None]:
        """Validate that the season is in READY phase using the DB directly.

        Returns ``(season_dict, None)`` on success, or ``(None, error_dict)``
        if the phase check fails.  Uses the DB ``season.phase`` column to
        avoid a bootstrap dependency (which is unavailable in tests).
        """
        season = self.seasons.get_season(manager_id)
        if not season:
            return None, {"error": "No active season for this manager"}
        if season.get("phase") != GWPhase.READY.value:
            return None, {"error": "Can only do this in READY phase"}
        return season, None

    def _get_next_gw_for_season(self, season: dict) -> int | None:
        """Return the next GW, trying bootstrap first then falling back to DB."""
        bootstrap = load_bootstrap()
        if bootstrap:
            gw = get_next_gw(bootstrap)
            if gw:
                return gw
        # Fallback: current_gw from the season row + 1.
        current = season.get("current_gw")
        if current and current < 38:
            return current + 1
        return None

    def _calculate_predicted_points(self, squad_json: dict) -> float:
        """Calculate total predicted points for the planned squad.

        - Sums starting XI predicted points.
        - Doubles captain's points (or triples if chip is ``"3xc"``).
        - If chip is ``"bboost"``, adds bench predicted points too.
        """
        players = squad_json.get("players", [])
        chip = squad_json.get("chip")
        captain_id = squad_json.get("captain_id")

        total = 0.0
        for p in players:
            pts = p.get("predicted_next_gw_points", 0) or 0
            pid = p.get("player_id")
            is_starter = p.get("starter", False)

            if is_starter:
                if pid == captain_id:
                    multiplier = 3 if chip == "3xc" else 2
                    total += pts * multiplier
                else:
                    total += pts
            elif chip == "bboost":
                # Bench boost: bench players also score.
                total += pts

        return round(total, 1)

    def accept_transfers(self, manager_id: int) -> dict:
        """Mark the planned squad as accepted (no changes)."""
        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return {"error": "No recommendation available to accept"}

        squad_json = planned["squad_json"]
        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source="accepted",
        )
        return {"status": "accepted", "planned_squad": squad_json}

    def make_transfer(
        self, manager_id: int, player_out_id: int, player_in_id: int,
    ) -> dict:
        """Swap *player_out_id* for *player_in_id* in the planned squad.

        Validates: position match, budget, team limit (max 3 from one team).
        Re-optimises the starting XI and recalculates predicted points.
        """
        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return {"error": "No planned squad to modify"}

        squad_json = copy.deepcopy(planned["squad_json"])
        players = squad_json.get("players", [])

        # --- Find the outgoing player ---
        out_player = None
        out_idx = None
        for i, p in enumerate(players):
            if p.get("player_id") == player_out_id:
                out_player = p
                out_idx = i
                break
        if out_player is None:
            return {"error": f"Player {player_out_id} is not in the squad"}

        # --- Load bootstrap to get incoming player details ---
        bootstrap = load_bootstrap()
        if not bootstrap:
            return {"error": "Cannot load player data (bootstrap unavailable)"}

        elements_by_id = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {
            t["id"]: t["code"] for t in bootstrap.get("teams", [])
        }

        in_el = elements_by_id.get(player_in_id)
        if not in_el:
            return {"error": f"Player {player_in_id} not found in FPL data"}

        in_position = ELEMENT_TYPE_MAP.get(in_el.get("element_type"), "")
        in_cost = round(in_el.get("now_cost", 0) / 10, 1)
        in_team_code = id_to_code.get(in_el.get("team"))
        in_web_name = in_el.get("web_name", f"Player {player_in_id}")

        # --- Validate same position ---
        out_position = out_player.get("position", "")
        if in_position != out_position:
            return {
                "error": (
                    f"Position mismatch: {out_player.get('web_name', '')} is "
                    f"{out_position}, but {in_web_name} is {in_position}"
                ),
            }

        # --- Validate budget ---
        bank = squad_json.get("bank", 0) or 0
        out_cost = out_player.get("cost", 0) or 0
        new_bank = round(bank + out_cost - in_cost, 1)
        if new_bank < 0:
            return {
                "error": (
                    f"Insufficient budget: need {in_cost}m, have "
                    f"{round(bank + out_cost, 1)}m (bank {bank}m + sell {out_cost}m)"
                ),
            }

        # --- Validate team limit (max 3 from one club) ---
        team_counts: dict[int, int] = {}
        for p in players:
            if p.get("player_id") == player_out_id:
                continue  # Skip the player being removed.
            tc = p.get("team_code")
            if tc is not None:
                team_counts[tc] = team_counts.get(tc, 0) + 1
        if in_team_code is not None:
            if team_counts.get(in_team_code, 0) >= 3:
                return {
                    "error": (
                        f"Team limit: already have 3 players from team "
                        f"code {in_team_code}"
                    ),
                }

        # --- Perform the swap ---
        in_player = {
            "player_id": player_in_id,
            "web_name": in_web_name,
            "position": in_position,
            "team_code": in_team_code,
            "cost": in_cost,
            "predicted_next_gw_points": 0.0,  # Will be filled if predictions exist.
            "captain_score": 0.0,
            "starter": False,
            "is_captain": False,
            "is_vice_captain": False,
        }
        players[out_idx] = in_player

        # --- Track transfer in/out lists ---
        transfers_in = list(squad_json.get("transfers_in", []))
        transfers_out = list(squad_json.get("transfers_out", []))
        transfers_in.append({
            "player_id": player_in_id,
            "web_name": in_web_name,
            "position": in_position,
            "cost": in_cost,
        })
        transfers_out.append({
            "player_id": player_out_id,
            "web_name": out_player.get("web_name", ""),
            "position": out_position,
            "cost": out_cost,
        })
        squad_json["transfers_in"] = transfers_in
        squad_json["transfers_out"] = transfers_out

        # --- Update hits ---
        free_transfers = squad_json.get("free_transfers", 1) or 1
        total_transfers = len(transfers_in)
        extra = max(0, total_transfers - free_transfers)
        squad_json["hits"] = extra

        # --- Update bank ---
        squad_json["bank"] = new_bank

        # --- Re-optimise starting XI ---
        squad_json["players"] = optimize_starting_xi(players)

        # --- Update captain/vc IDs from optimised players ---
        for p in squad_json["players"]:
            if p.get("is_captain"):
                squad_json["captain_id"] = p["player_id"]
            if p.get("is_vice_captain"):
                squad_json["vice_captain_id"] = p["player_id"]

        # --- Recalculate predicted points ---
        squad_json["predicted_points"] = self._calculate_predicted_points(squad_json)

        # --- Save ---
        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source="user_override",
        )
        return {"status": "transfer_made", "planned_squad": squad_json}

    def undo_transfers(self, manager_id: int) -> dict:
        """Reset the planned squad to the original recommendation."""
        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        rec = self.recommendations.get_recommendation(season_id, next_gw)
        if not rec:
            return {"error": "No original recommendation to revert to"}

        # Rebuild squad_json from the recommendation.
        raw = rec.get("new_squad_json")
        if not raw:
            return {"error": "Recommendation has no squad data"}
        try:
            squad_json = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {"error": "Recommendation squad data is corrupted"}

        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source="recommended",
        )
        return {"status": "reverted", "planned_squad": squad_json}

    def lock_chip(self, manager_id: int, chip: str) -> dict:
        """Activate a chip on the planned squad for the next GW."""
        if chip not in VALID_CHIPS:
            return {
                "error": (
                    f"Invalid chip '{chip}'. Must be one of: "
                    f"{', '.join(sorted(VALID_CHIPS))}"
                ),
            }

        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        # --- Check chip not already used this half-season ---
        chips_used = self.dashboard.get_chips_status(season_id)
        for used in chips_used:
            if used.get("chip_used") == chip:
                return {
                    "error": (
                        f"Chip '{chip}' was already used in GW"
                        f"{used.get('gameweek', '?')}"
                    ),
                }

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return {"error": "No planned squad to modify"}

        squad_json = copy.deepcopy(planned["squad_json"])
        squad_json["chip"] = chip

        # Recalculate predicted points with chip effect.
        squad_json["predicted_points"] = self._calculate_predicted_points(squad_json)

        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source=planned.get("source", "recommended"),
        )
        return {"status": "chip_locked", "chip": chip, "planned_squad": squad_json}

    def unlock_chip(self, manager_id: int) -> dict:
        """Remove any active chip from the planned squad."""
        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return {"error": "No planned squad to modify"}

        squad_json = copy.deepcopy(planned["squad_json"])
        old_chip = squad_json.get("chip")
        if not old_chip:
            return {"status": "no_chip", "planned_squad": squad_json}

        squad_json["chip"] = None

        # Recalculate without chip effect.
        squad_json["predicted_points"] = self._calculate_predicted_points(squad_json)

        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source=planned.get("source", "recommended"),
        )
        return {"status": "chip_unlocked", "old_chip": old_chip, "planned_squad": squad_json}

    def set_captain(self, manager_id: int, player_id: int) -> dict:
        """Set a new captain. Player must be in the starting XI."""
        season, err = self._require_ready_phase(manager_id)
        if err:
            return err

        season_id = season["id"]
        next_gw = self._get_next_gw_for_season(season)
        if not next_gw:
            return {"error": "Cannot determine next gameweek"}

        planned = self.planned_squads.get_planned_squad(season_id, next_gw)
        if not planned:
            return {"error": "No planned squad to modify"}

        squad_json = copy.deepcopy(planned["squad_json"])
        players = squad_json.get("players", [])

        # Find the target player and validate they are a starter.
        target = None
        for p in players:
            if p.get("player_id") == player_id:
                target = p
                break
        if target is None:
            return {"error": f"Player {player_id} is not in the squad"}
        if not target.get("starter", False):
            return {"error": f"Player {player_id} is not in the starting XI"}

        # Clear old captain/VC flags.
        old_captain_id = squad_json.get("captain_id")
        for p in players:
            p["is_captain"] = False
            p["is_vice_captain"] = False

        # Set new captain.
        target["is_captain"] = True
        squad_json["captain_id"] = player_id

        # Pick new VC: highest captain_score (or predicted pts) among
        # starters who are NOT the new captain.
        starters = [
            p for p in players
            if p.get("starter") and p.get("player_id") != player_id
        ]
        if starters:
            vc_key = "captain_score" if any(
                p.get("captain_score") for p in starters
            ) else "predicted_next_gw_points"
            starters.sort(key=lambda p: (p.get(vc_key) or 0), reverse=True)
            starters[0]["is_vice_captain"] = True
            squad_json["vice_captain_id"] = starters[0]["player_id"]

        # Recalculate predicted points.
        squad_json["predicted_points"] = self._calculate_predicted_points(squad_json)

        # Preserve existing source unless it was the default.
        source = planned.get("source", "recommended")
        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source=source,
        )
        return {"status": "captain_set", "captain_id": player_id, "planned_squad": squad_json}
