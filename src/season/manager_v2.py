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
        """Generate predictions + recommendation, save planned squad, transition to READY."""
        import numpy as np
        import pandas as pd

        from src.api.helpers import calculate_free_transfers, resolve_current_squad_event
        from src.data.fpl_api import (
            fetch_fpl_api,
            fetch_manager_entry,
            fetch_manager_history,
            fetch_manager_picks,
        )
        from src.data.loader import load_all_data
        from src.features.builder import build_features, get_fixture_context
        from src.ml.multi_gw import predict_multi_gw
        from src.ml.prediction import generate_predictions
        from src.paths import OUTPUT_DIR
        from src.season.fixtures import save_fixture_calendar
        from src.strategy.reactive import apply_availability_adjustments

        alerts: list[dict] = []
        season_id = season["id"]
        next_gw = status.get("gw")

        def log(msg: str, **kw) -> None:
            if progress_fn:
                progress_fn(msg, **kw)
            logger.info(msg)

        if not next_gw:
            logger.error("Cannot determine next GW for manager %d", manager_id)
            return alerts

        # ── 1. Load data + build features + generate predictions ─────────
        log("Loading data...")
        data = load_all_data(force=True)
        df = build_features(data)

        log("Generating predictions...")
        pred_result = generate_predictions(df, data)
        players_df = pred_result.get("players")
        if players_df is None or (hasattr(players_df, "empty") and players_df.empty):
            logger.error("Prediction generation failed")
            return alerts

        # Save predictions CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(players_df, "to_csv"):
            players_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

        # ── 2. Fetch current squad (with Free Hit reversion) ─────────────
        log("Fetching current squad...")
        bootstrap = load_bootstrap()
        if not bootstrap:
            logger.error("Bootstrap data unavailable")
            return alerts

        entry = fetch_manager_entry(manager_id)
        if not entry:
            logger.error("Could not fetch manager entry for %d", manager_id)
            return alerts

        current_event = entry.get("current_event")
        if not current_event:
            logger.error("Manager %d has no current event", manager_id)
            return alerts

        history = fetch_manager_history(manager_id)
        if not history:
            logger.error("Could not fetch history for manager %d", manager_id)
            return alerts

        # Detect Free Hit reversion
        squad_event, fh_reverted = resolve_current_squad_event(history, current_event)
        if fh_reverted:
            log(f"Free Hit played in GW{current_event} -- using reverted GW{squad_event} squad")

        picks_data = fetch_manager_picks(manager_id, squad_event)
        if not picks_data:
            logger.error("Could not fetch picks for GW%d", squad_event)
            return alerts

        free_transfers = calculate_free_transfers(history)

        elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

        picks = picks_data.get("picks", [])
        entry_history = picks_data.get("entry_history", {})

        # ── 3. Calculate budget (CRITICAL: use entry_history value) ──────
        bank = round(entry_history.get("bank", 0) / 10, 1)
        current_squad_ids: set[int] = set()
        current_squad_cost = 0.0
        for pick in picks:
            eid = pick.get("element")
            current_squad_ids.add(eid)
            el = elements_map.get(eid, {})
            current_squad_cost += el.get("now_cost", 0) / 10

        api_value = entry_history.get("value")
        if api_value:
            total_budget = round(api_value / 10, 1)
        else:
            total_budget = round(bank + current_squad_cost, 1)

        # ── 4. Enrich predictions with bootstrap data ────────────────────
        log("Enriching predictions...")
        if "position_clean" in players_df.columns and "position" not in players_df.columns:
            players_df["position"] = players_df["position_clean"]
        if "cost" not in players_df.columns:
            players_df["cost"] = None
        if "team_code" not in players_df.columns:
            players_df["team_code"] = None
        if "web_name" not in players_df.columns:
            players_df["web_name"] = None

        for idx, row in players_df.iterrows():
            el = elements_map.get(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
            if el:
                if pd.isna(row.get("cost")) or row.get("cost") is None:
                    players_df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
                if pd.isna(row.get("team_code")) or row.get("team_code") is None:
                    players_df.at[idx, "team_code"] = id_to_code.get(el.get("team"))
                if pd.isna(row.get("web_name")) or row.get("web_name") is None:
                    players_df.at[idx, "web_name"] = el.get("web_name", f"ID{el['id']}")

        # Build prediction lookup for enriching player dicts
        pred_lookup: dict[int, dict] = {}
        for _, prow in players_df.iterrows():
            pid = int(prow.get("player_id", 0))
            if pid:
                pred_lookup[pid] = {
                    "predicted_next_gw_points": round(float(prow.get("predicted_next_gw_points") or 0), 2),
                    "predicted_next_3gw_points": (
                        round(float(prow["predicted_next_3gw_points"]), 2)
                        if pd.notna(prow.get("predicted_next_3gw_points"))
                        else None
                    ),
                    "captain_score": round(float(prow.get("captain_score") or 0), 2),
                }

        # Build opponent map for next GW
        fixtures_list = self._load_fixtures()
        opponent_map: dict = {}
        id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        for f in fixtures_list:
            if f.get("event") == next_gw:
                h_code = id_to_code.get(f.get("team_h"))
                a_code = id_to_code.get(f.get("team_a"))
                a_short = id_to_short.get(f.get("team_a"), "?")
                h_short = id_to_short.get(f.get("team_h"), "?")
                if h_code is not None:
                    opponent_map[h_code] = f"{a_short}(H)"
                if a_code is not None:
                    opponent_map[a_code] = f"{h_short}(A)"

        # ── 5. Generate multi-GW predictions for planner ─────────────────
        log("Generating multi-GW predictions...")
        fixture_context = get_fixture_context(data)
        latest_gw = pred_result.get("gameweek", next_gw - 1)
        current_gw_df = df[
            df["gameweek"] == latest_gw
        ].drop_duplicates("player_id", keep="first")

        future_predictions = predict_multi_gw(
            current=current_gw_df,
            df=df,
            fixture_context=fixture_context,
            latest_gw=latest_gw,
            max_gw=8,
        )

        # Keep only GWs from next_gw onward
        if future_predictions:
            future_predictions = {
                gw: gw_df for gw, gw_df in future_predictions.items() if gw >= next_gw
            }

        # Replace GW+1 with the exact predictions from the Predictions tab
        if future_predictions and next_gw in future_predictions:
            gw1_cols = ["player_id", "predicted_next_gw_points",
                        "position", "cost", "team_code", "web_name"]
            if "captain_score" in players_df.columns:
                gw1_cols.append("captain_score")
            available_cols = [c for c in gw1_cols if c in players_df.columns]
            gw1_from_pred = players_df[available_cols].drop_duplicates("player_id").copy()
            gw1_from_pred["predicted_points"] = gw1_from_pred["predicted_next_gw_points"]
            gw1_from_pred["confidence"] = 1.0
            if "team_code" in gw1_from_pred.columns:
                gw1_from_pred["team"] = (
                    gw1_from_pred["team_code"].map(code_to_short).fillna("")
                )
            future_predictions[next_gw] = gw1_from_pred

        # Enrich GW+2 onwards with bootstrap data
        enrich_rows = []
        for el in bootstrap.get("elements", []):
            enrich_rows.append({
                "player_id": el["id"],
                "web_name": el.get("web_name", "Unknown"),
                "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                "cost": round(el.get("now_cost", 0) / 10, 1),
                "team_code": id_to_code.get(el.get("team")),
            })
        enrich_df = pd.DataFrame(enrich_rows)

        if future_predictions:
            for gw in future_predictions:
                if gw == next_gw:
                    continue
                gw_df = future_predictions[gw]
                merge_cols = [
                    c for c in enrich_df.columns
                    if c not in gw_df.columns or c == "player_id"
                ]
                gw_df = gw_df.merge(enrich_df[merge_cols], on="player_id", how="left")
                if "captain_score" not in gw_df.columns:
                    gw_df["captain_score"] = gw_df["predicted_points"]
                else:
                    gw_df["captain_score"] = gw_df["predicted_points"]
                future_predictions[gw] = gw_df

        # ── 6. Apply availability adjustments ────────────────────────────
        log("Adjusting for injuries/availability...")
        if future_predictions:
            future_predictions = apply_availability_adjustments(
                future_predictions, bootstrap.get("elements", [])
            )

        # ── 7. Run MultiWeekPlanner ──────────────────────────────────────
        transfers_json = "[]"
        captain_id = None
        captain_name = None
        predicted_points = 0.0
        base_points = 0.0
        current_xi_points = 0.0
        new_squad_json = None
        transfer_result = None
        used_planner = False

        if future_predictions:
            log("Planning transfers over 5 GWs...")
            try:
                from src.strategy.transfer_planner import MultiWeekPlanner

                fixture_calendar = self.fixtures.get_fixture_calendar(
                    season_id, from_gw=next_gw
                )

                planner = MultiWeekPlanner()
                transfer_plan = planner.plan_transfers(
                    current_squad_ids=current_squad_ids,
                    total_budget=total_budget,
                    free_transfers=free_transfers,
                    future_predictions=future_predictions,
                    fixture_calendar=fixture_calendar,
                    price_alerts=[],
                    chip_plan=None,
                )

                # Extract GW+1 from the plan
                if transfer_plan:
                    gw1_step = next(
                        (s for s in transfer_plan if s.get("gw") == next_gw), None
                    )
                    if gw1_step is None and transfer_plan:
                        gw1_step = transfer_plan[0]

                    if gw1_step:
                        used_planner = True
                        predicted_points = gw1_step.get("predicted_points", 0)
                        captain_id = gw1_step.get("captain_id")
                        if captain_id:
                            el = elements_map.get(captain_id, {})
                            captain_name = el.get("web_name", "Unknown")

            except Exception as exc:
                logger.warning("MultiWeekPlanner failed: %s", exc, exc_info=True)
                used_planner = False

        # ── 8. Fallback: single-GW MILP solver ──────────────────────────
        if not used_planner:
            log("Running single-GW transfer solver (fallback)...")
            from src.solver.transfers import solve_transfer_milp_with_hits

            target = "predicted_next_gw_points"
            pool = players_df.dropna(subset=["position", "cost", target]).copy()
            captain_col = "captain_score" if "captain_score" in pool.columns else None

            transfer_result = solve_transfer_milp_with_hits(
                pool,
                current_squad_ids,
                target,
                budget=total_budget,
                free_transfers=free_transfers,
                max_transfers=min(free_transfers + 2, 5),
                captain_col=captain_col,
            )

            if transfer_result:
                captain_id = transfer_result.get("captain_id")
                if captain_id:
                    el = elements_map.get(captain_id, {})
                    captain_name = el.get("web_name", "Unknown")
                predicted_points = transfer_result.get("starting_points", 0)
                base_points = transfer_result.get("starting_points", 0)
                current_xi_points = transfer_result.get("baseline_points", 0)

        # ── 9. Build enriched squad + transfer list ──────────────────────
        log("Building recommendation...")

        def _enrich_player(pid: int) -> dict:
            """Build a rich player dict from bootstrap + predictions."""
            el = elements_map.get(pid, {})
            tc = id_to_code.get(el.get("team"))
            preds = pred_lookup.get(pid, {})
            return {
                "player_id": pid,
                "web_name": el.get("web_name", f"ID{pid}"),
                "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                "team_code": tc,
                "team": code_to_short.get(tc, ""),
                "cost": round(el.get("now_cost", 0) / 10, 1),
                "predicted_next_gw_points": preds.get("predicted_next_gw_points", 0),
                "predicted_next_3gw_points": preds.get("predicted_next_3gw_points"),
                "captain_score": preds.get("captain_score", 0),
                "opponent": opponent_map.get(tc, ""),
            }

        if used_planner and gw1_step:
            # Build transfer pairs from planner
            transfers = []
            t_in_list = gw1_step.get("transfers_in", [])
            t_out_list = gw1_step.get("transfers_out", [])
            max_len = max(len(t_out_list), len(t_in_list))
            for i in range(max_len):
                out_entry = (
                    _enrich_player(t_out_list[i].get("player_id"))
                    if i < len(t_out_list) else {}
                )
                in_entry = (
                    _enrich_player(t_in_list[i].get("player_id"))
                    if i < len(t_in_list) else {}
                )
                transfers.append({"out": out_entry, "in": in_entry})
            transfers_json = json.dumps(transfers)

            # Build enriched squad from planner squad_ids
            if gw1_step.get("squad_ids"):
                enriched_squad = [
                    _enrich_player(pid) for pid in gw1_step["squad_ids"]
                ]
                new_squad_json = json.dumps(enriched_squad)

        elif transfer_result:
            # Build transfer pairs from MILP solver
            transfers = []
            out_ids = list(transfer_result.get("transfers_out_ids", set()))
            in_ids = list(transfer_result.get("transfers_in_ids", set()))
            max_len = max(len(out_ids), len(in_ids))
            for i in range(max_len):
                out_entry = _enrich_player(out_ids[i]) if i < len(out_ids) else {}
                in_entry = _enrich_player(in_ids[i]) if i < len(in_ids) else {}
                transfers.append({"out": out_entry, "in": in_entry})
            transfers_json = json.dumps(transfers)

            # Enrich new squad with full player data
            enriched_squad = []
            for p in transfer_result.get("players", []):
                pid = int(p.get("player_id", 0))
                el = elements_map.get(pid, {})
                tc = p.get("team_code") or id_to_code.get(el.get("team"))
                preds = pred_lookup.get(pid, {})
                enriched_squad.append({
                    "player_id": pid,
                    "web_name": el.get("web_name", str(p.get("web_name", "Unknown"))),
                    "position": p.get("position", ELEMENT_TYPE_MAP.get(el.get("element_type"), "")),
                    "team_code": tc,
                    "team": code_to_short.get(tc, ""),
                    "cost": round(float(p.get("cost", el.get("now_cost", 0) / 10)), 1),
                    "predicted_next_gw_points": preds.get(
                        "predicted_next_gw_points",
                        round(float(p.get("predicted_next_gw_points", 0)), 2),
                    ),
                    "predicted_next_3gw_points": preds.get("predicted_next_3gw_points"),
                    "captain_score": preds.get(
                        "captain_score",
                        round(float(p.get("captain_score", 0)), 2),
                    ),
                    "starter": bool(p.get("starter", False)),
                    "is_captain": pid == captain_id if captain_id else False,
                    "is_vice_captain": False,
                    "opponent": opponent_map.get(tc, ""),
                })

            # Set vice captain
            vc_candidates = [
                p for p in enriched_squad if p["starter"] and not p["is_captain"]
            ]
            if vc_candidates:
                vc_candidates.sort(
                    key=lambda p: p.get("captain_score", 0), reverse=True
                )
                vc_candidates[0]["is_vice_captain"] = True
            new_squad_json = json.dumps(enriched_squad)

        # ── 10. Save recommendation to DB ────────────────────────────────
        log("Saving recommendation...")
        hits = 0
        if transfer_result:
            hits = transfer_result.get("hits", 0)
        elif used_planner and gw1_step:
            hits = gw1_step.get("hits", 0)

        self.recommendations.save_recommendation(
            season_id=season_id,
            gameweek=next_gw,
            transfers_json=transfers_json,
            captain_id=captain_id,
            captain_name=captain_name,
            chip_suggestion=None,
            chip_values_json="{}",
            bank_analysis_json="{}",
            new_squad_json=new_squad_json,
            predicted_points=predicted_points,
            base_points=base_points,
            current_xi_points=current_xi_points,
            free_transfers=free_transfers,
        )

        # ── 11. Build planned squad JSON and save ────────────────────────
        squad_json = self._build_planned_squad(
            new_squad_json=new_squad_json,
            captain_id=captain_id,
            bank=bank,
            free_transfers=free_transfers,
            transfers_json=transfers_json,
            hits=hits,
            predicted_points=predicted_points,
        )

        self.planned_squads.save_planned_squad(
            season_id, next_gw, squad_json, source="recommended",
        )

        # ── 12. Update fixture calendar + track prices ───────────────────
        log("Updating fixtures and prices...")
        if bootstrap and fixtures_list:
            try:
                save_fixture_calendar(
                    season_id, bootstrap, fixtures_list, self.fixtures,
                )
            except Exception as exc:
                logger.warning("Fixture calendar update failed: %s", exc)

        self._track_prices_simple(season_id, manager_id, bootstrap)
        self.seasons.update_season_gw(season_id, current_event)

        # ── 13. Transition to READY ──────────────────────────────────────
        self.seasons.update_phase(season_id, GWPhase.READY.value)
        log("Recommendation complete.")

        alerts.append({
            "type": "recommendation_ready",
            "message": (
                f"GW{next_gw} recommendation ready: "
                f"{captain_name or 'TBD'} (C), "
                f"{round(predicted_points, 1)} predicted pts"
            ),
        })
        return alerts

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
        """Record results for the just-completed GW and advance phase.

        1. Fetch actual picks, history, live event data from FPL API
        2. Build squad with live points (captain doubling applied)
        3. Save gw_snapshot
        4. Compare to recommendation (if one exists) and save outcome
        5. Update season current_gw
        6. Transition to PLANNING (or SEASON_OVER if GW38)
        """
        from src.api.helpers import calculate_free_transfers
        from src.data.fpl_api import (
            fetch_event_live,
            fetch_fpl_api,
            fetch_manager_history,
            fetch_manager_picks,
            fetch_manager_transfers,
        )

        alerts: list[dict] = []
        season_id = status["season_id"]
        completed_gw = status.get("current_gw")

        if not completed_gw:
            logger.error("No current_gw in status — cannot record results")
            return alerts

        logger.info(
            "COMPLETE phase: recording results for GW%d (manager %d)",
            completed_gw, manager_id,
        )

        # ── 1. Fetch data from FPL API ────────────────────────────────────
        try:
            bootstrap = fetch_fpl_api("bootstrap", force=True)
        except Exception as exc:
            logger.error("Failed to fetch bootstrap: %s", exc)
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: bootstrap unavailable",
            })
            return alerts

        if not bootstrap:
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: bootstrap unavailable",
            })
            return alerts

        try:
            picks_data = fetch_manager_picks(manager_id, completed_gw)
        except Exception as exc:
            logger.error("Failed to fetch picks for GW%d: %s", completed_gw, exc)
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: could not fetch picks",
            })
            return alerts

        if not picks_data:
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: no picks data",
            })
            return alerts

        try:
            history = fetch_manager_history(manager_id)
        except Exception as exc:
            logger.error("Failed to fetch history: %s", exc)
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: could not fetch history",
            })
            return alerts

        if not history:
            alerts.append({
                "type": "error",
                "message": f"Failed to record GW{completed_gw}: no history data",
            })
            return alerts

        # ── 2. Build lookup maps ──────────────────────────────────────────
        elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

        # Live points (more accurate than bootstrap event_points)
        live_points_map: dict[int, int] = {}
        try:
            live_data = fetch_event_live(completed_gw, force=True)
            for el_live in live_data.get("elements", []):
                live_points_map[el_live["id"]] = (
                    el_live.get("stats", {}).get("total_points", 0)
                )
            logger.info(
                "Loaded live points for GW%d (%d players)",
                completed_gw, len(live_points_map),
            )
        except Exception as exc:
            logger.warning(
                "Could not fetch live event data (%s), falling back to bootstrap",
                exc,
            )

        # ── 3. Build squad from picks ─────────────────────────────────────
        picks = picks_data.get("picks", [])
        squad = []
        captain_id = None
        captain_name = None
        for pick in picks:
            eid = pick.get("element")
            el = elements_map.get(eid, {})
            tid = el.get("team")
            tc = id_to_code.get(tid)
            pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")
            # Use live event points when available, fall back to bootstrap
            raw_pts = live_points_map.get(eid, el.get("event_points", 0))
            player = {
                "player_id": eid,
                "web_name": el.get("web_name", "Unknown"),
                "position": pos,
                "team_code": tc,
                "team": code_to_short.get(tc, ""),
                "cost": el.get("now_cost", 0) / 10,
                "starter": pick.get("position", 12) <= 11,
                "is_captain": pick.get("is_captain", False),
                "multiplier": pick.get("multiplier", 1),
                "event_points": raw_pts * pick.get("multiplier", 1),
            }
            squad.append(player)
            if pick.get("is_captain"):
                captain_id = eid
                captain_name = el.get("web_name", "Unknown")

        # ── 4. GW data from history ───────────────────────────────────────
        gw_entries = history.get("current", [])
        gw_data = next(
            (g for g in gw_entries if g["event"] == completed_gw), {}
        )
        chip_map = {c["event"]: c["name"] for c in history.get("chips", [])}

        entry_hist = picks_data.get("entry_history", {})

        # ── 5. Fetch transfers for this GW ────────────────────────────────
        t_in_list = []
        t_out_list = []
        try:
            all_transfers = fetch_manager_transfers(manager_id)
            gw_transfers = [
                t for t in all_transfers if t["event"] == completed_gw
            ]
            for t in gw_transfers:
                el_in = elements_map.get(t["element_in"], {})
                el_out = elements_map.get(t["element_out"], {})
                t_in_list.append({
                    "player_id": t["element_in"],
                    "web_name": el_in.get("web_name", "Unknown"),
                    "cost": t.get("element_in_cost", 0) / 10,
                })
                t_out_list.append({
                    "player_id": t["element_out"],
                    "web_name": el_out.get("web_name", "Unknown"),
                    "cost": t.get("element_out_cost", 0) / 10,
                })
        except Exception as exc:
            logger.warning("Could not fetch transfers: %s", exc)
            gw_transfers = []

        # ── 6. Save gw_snapshot ───────────────────────────────────────────
        free_transfers = calculate_free_transfers(history)

        self.snapshots.save_gw_snapshot(
            season_id=season_id,
            gameweek=completed_gw,
            squad_json=json.dumps(squad),
            bank=(
                entry_hist.get("bank", gw_data.get("bank", 0)) / 10
                if entry_hist
                else gw_data.get("bank", 0) / 10
            ),
            team_value=(
                (
                    entry_hist.get("value", gw_data.get("value", 0))
                    - entry_hist.get("bank", gw_data.get("bank", 0))
                )
                / 10
                if entry_hist
                else (gw_data.get("value", 0) - gw_data.get("bank", 0)) / 10
            ),
            free_transfers=free_transfers,
            chip_used=chip_map.get(completed_gw),
            points=gw_data.get("points"),
            total_points=gw_data.get("total_points"),
            overall_rank=gw_data.get("overall_rank"),
            transfers_in_json=json.dumps(t_in_list) if t_in_list else None,
            transfers_out_json=json.dumps(t_out_list) if t_out_list else None,
            captain_id=captain_id,
            captain_name=captain_name,
            transfers_cost=gw_data.get("event_transfers_cost", 0),
        )

        # ── 7. Compare to recommendation and save outcome ─────────────────
        rec = self.recommendations.get_recommendation(season_id, completed_gw)
        if rec:
            actual_points = gw_data.get("points", 0)
            recommended_points = rec.get("predicted_points", 0)
            point_delta = round(
                (actual_points or 0) - (recommended_points or 0), 1
            )

            # Check if captain was followed
            followed_captain = (
                1 if captain_id == rec.get("captain_id") else 0
            )

            # Check if transfers were followed
            rec_transfers = json.loads(rec.get("transfers_json") or "[]")
            rec_in_ids = {
                t["in"]["player_id"]
                for t in rec_transfers
                if t.get("in", {}).get("player_id")
            }
            actual_squad_ids = {p["player_id"] for p in squad}
            if not rec_in_ids:
                # Recommendation was to bank FT -- only followed if 0 transfers
                followed_transfers = 1 if not gw_transfers else 0
            else:
                followed_transfers = (
                    1 if rec_in_ids.issubset(actual_squad_ids) else 0
                )

            # WC/FH squad comparison
            rec_chip = rec.get("chip_suggestion")
            rec_new_squad = rec.get("new_squad_json")
            if rec_chip in ("wildcard", "freehit") and rec_new_squad:
                try:
                    rec_squad = json.loads(rec_new_squad)
                    rec_squad_ids = {
                        p["player_id"]
                        for p in rec_squad
                        if "player_id" in p
                    }
                    if rec_squad_ids:
                        overlap = rec_squad_ids & actual_squad_ids
                        followed_transfers = (
                            1 if len(overlap) >= 13 else 0
                        )
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            actual_chip = chip_map.get(completed_gw)
            recommended_chip = rec.get("chip_suggestion")
            followed_chip = 1 if actual_chip == recommended_chip else 0

            self.outcomes.save_outcome(
                season_id=season_id,
                gameweek=completed_gw,
                followed_transfers=followed_transfers,
                followed_captain=followed_captain,
                followed_chip=followed_chip,
                recommended_points=recommended_points,
                actual_points=actual_points,
                point_delta=point_delta,
            )

            logger.info(
                "GW%d outcome: actual=%s, recommended=%s, delta=%s, "
                "followed_captain=%s, followed_transfers=%s",
                completed_gw, actual_points, recommended_points,
                point_delta, followed_captain, followed_transfers,
            )

        # ── 8. Update season and transition phase ─────────────────────────
        self.seasons.update_season_gw(season_id, completed_gw)

        if completed_gw >= 38:
            self.seasons.update_phase(season_id, GWPhase.SEASON_OVER.value)
            logger.info("GW38 complete — season over")
            alerts.append({
                "type": "season_over",
                "message": "Season complete! All 38 gameweeks finished.",
            })
        else:
            self.seasons.update_phase(season_id, GWPhase.PLANNING.value)
            logger.info(
                "GW%d recorded — transitioning to PLANNING for GW%d",
                completed_gw, completed_gw + 1,
            )

        alerts.append({
            "type": "gw_recorded",
            "message": (
                f"GW{completed_gw} results recorded"
                f" — {gw_data.get('points', '?')} pts"
                f" (total: {gw_data.get('total_points', '?')})"
            ),
            "gameweek": completed_gw,
            "points": gw_data.get("points"),
            "total_points": gw_data.get("total_points"),
            "overall_rank": gw_data.get("overall_rank"),
        })

        return alerts

    # ------------------------------------------------------------------
    # Season initialisation
    # ------------------------------------------------------------------

    def init_season(self, manager_id: int, progress_fn=None) -> dict:
        """Initialize or re-initialize a season from FPL API data.

        Fetches manager info, backfills GW history, saves fixtures/prices.
        Sets phase to PLANNING so tick() will generate the first recommendation.
        """
        from src.data.fpl_api import (
            fetch_manager_entry, fetch_manager_history,
            fetch_manager_picks, fetch_manager_transfers, fetch_fpl_api,
        )
        from src.season.fixtures import save_fixture_calendar

        def log(msg, **kw):
            if progress_fn:
                progress_fn(msg, **kw)
            logger.info(msg)

        # 1. Fetch manager data
        log("Fetching manager data...")
        entry = fetch_manager_entry(manager_id)
        if not entry:
            return {"error": f"Manager {manager_id} not found"}

        history = fetch_manager_history(manager_id)
        if not history:
            return {"error": f"Could not fetch history for manager {manager_id}"}

        # 2. Detect season info
        manager_name = (
            f"{entry.get('player_first_name', '')} "
            f"{entry.get('player_last_name', '')}"
        ).strip()
        team_name = entry.get("name", "")

        current_events = history.get("current", [])
        if not current_events:
            # Pre-season: create season record, set phase to PLANNING
            log("Pre-season detected, creating season record...")
            season_id = self.seasons.create_season(
                manager_id=manager_id,
                manager_name=manager_name,
                team_name=team_name,
                start_gw=1,
            )
            self.seasons.update_phase(season_id, GWPhase.PLANNING.value)
            return {"status": "initialized", "season_id": season_id, "pre_season": True}

        start_gw = current_events[0].get("event", 1)
        latest_gw = current_events[-1].get("event", start_gw)

        # 3. Create season record
        log(f"Creating season (GW{start_gw}-{latest_gw})...")
        season_id = self.seasons.create_season(
            manager_id=manager_id,
            manager_name=manager_name,
            team_name=team_name,
            start_gw=start_gw,
        )
        self.seasons.update_season_gw(season_id, latest_gw)
        self.seasons.clear_generated_data(season_id)

        # 4. Refresh FPL API data to ensure cache is fresh
        log("Refreshing FPL data...")
        fetch_fpl_api(force=True)
        bootstrap = load_bootstrap()
        if not bootstrap:
            return {"error": "Could not load bootstrap data"}

        elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

        # 5. Fetch transfers for the season
        transfers = fetch_manager_transfers(manager_id) or []
        transfers_by_gw: dict[int, list] = {}
        for t in transfers:
            gw = t.get("event")
            transfers_by_gw.setdefault(gw, []).append(t)

        # 6. Backfill GW snapshots
        chips = history.get("chips", [])
        chip_by_gw = {c["event"]: c["name"] for c in chips}

        for gw_entry in current_events:
            gw = gw_entry.get("event")
            log(f"Backfilling GW{gw}...", progress=gw / latest_gw)

            try:
                picks_data = fetch_manager_picks(manager_id, gw)
            except Exception:
                logger.warning("Could not fetch picks for GW%d", gw)
                continue

            picks = picks_data.get("picks", []) if picks_data else []
            entry_hist = picks_data.get("entry_history", {}) if picks_data else {}

            # Build squad JSON
            squad = []
            captain_id = None
            captain_name = None
            for pick in picks:
                pid = pick["element"]
                el = elements_map.get(pid, {})
                team_id = el.get("team")
                pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "?")
                multiplier = pick.get("multiplier", 1)

                player_dict = {
                    "player_id": pid,
                    "web_name": el.get("web_name", f"ID{pid}"),
                    "position": pos,
                    "team_code": id_to_code.get(team_id),
                    "cost": round(el.get("now_cost", 0) / 10, 1),
                    "starter": pick.get("position", 12) <= 11,
                    "is_captain": pick.get("is_captain", False),
                    "multiplier": multiplier,
                }
                squad.append(player_dict)

                if pick.get("is_captain"):
                    captain_id = pid
                    captain_name = el.get("web_name", f"ID{pid}")

            # GW transfers
            gw_transfers = transfers_by_gw.get(gw, [])
            transfers_in = [
                {
                    "player_id": t["element_in"],
                    "web_name": elements_map.get(t["element_in"], {}).get("web_name", "?"),
                }
                for t in gw_transfers
            ]
            transfers_out = [
                {
                    "player_id": t["element_out"],
                    "web_name": elements_map.get(t["element_out"], {}).get("web_name", "?"),
                }
                for t in gw_transfers
            ]

            self.snapshots.save_gw_snapshot(
                season_id=season_id,
                gameweek=gw,
                squad_json=json.dumps(squad),
                bank=round(entry_hist.get("bank", 0) / 10, 1),
                team_value=round(entry_hist.get("value", 0) / 10, 1),
                free_transfers=None,  # Can't reliably calculate retroactively
                chip_used=chip_by_gw.get(gw),
                points=entry_hist.get("points"),
                total_points=entry_hist.get("total_points"),
                overall_rank=entry_hist.get("overall_rank"),
                transfers_in_json=json.dumps(transfers_in),
                transfers_out_json=json.dumps(transfers_out),
                captain_id=captain_id,
                captain_name=captain_name,
                transfers_cost=entry_hist.get("event_transfers_cost", 0),
            )

        # 7. Save fixture calendar
        log("Saving fixtures...")
        fixtures_raw = self._load_fixtures()
        if bootstrap and fixtures_raw:
            save_fixture_calendar(
                season_id=season_id,
                bootstrap=bootstrap,
                fixtures=fixtures_raw,
                fixture_repo=self.fixtures,
            )

        # 8. Track prices
        log("Tracking prices...")
        self._track_prices_simple(season_id, manager_id, bootstrap)

        # 9. Set phase to PLANNING
        self.seasons.update_phase(season_id, GWPhase.PLANNING.value)

        log("Season initialized!", progress=1.0)
        return {"status": "initialized", "season_id": season_id, "latest_gw": latest_gw}

    def _track_prices_simple(self, season_id: int, manager_id: int, bootstrap: dict | None) -> None:
        """Snapshot prices for current squad + watchlist players."""
        if not bootstrap:
            return
        elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

        # Get latest snapshot squad
        snapshots = self.snapshots.get_snapshots(season_id)
        squad_ids: set[int] = set()
        if snapshots:
            latest = snapshots[-1]
            try:
                squad = json.loads(latest.get("squad_json", "[]"))
                squad_ids = {
                    p.get("player_id") or p.get("id")
                    for p in squad
                    if isinstance(p, dict)
                }
            except (TypeError, json.JSONDecodeError):
                pass

        # Add watchlist
        watchlist = self.watchlist.get_watchlist(season_id)
        watchlist_ids = {w["player_id"] for w in watchlist}
        all_ids = squad_ids | watchlist_ids

        if not all_ids:
            return

        price_snapshots = []
        for pid in all_ids:
            el = elements_map.get(pid)
            if not el:
                continue
            price_snapshots.append({
                "player_id": pid,
                "web_name": el.get("web_name"),
                "team_code": id_to_code.get(el.get("team")),
                "price": round(el.get("now_cost", 0) / 10, 1),
                "transfers_in_event": el.get("transfers_in_event", 0),
                "transfers_out_event": el.get("transfers_out_event", 0),
            })

        if price_snapshots:
            self.prices.save_price_snapshots_bulk(season_id, price_snapshots)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_planned_squad(
        self,
        new_squad_json: str | None,
        captain_id: int | None,
        bank: float,
        free_transfers: int,
        transfers_json: str,
        hits: int,
        predicted_points: float,
    ) -> dict:
        """Build the planned squad dict from the recommendation data.

        Parses the enriched squad from *new_squad_json* (JSON string of
        player dicts), applies ``optimize_starting_xi``, and packages
        everything into the standard planned-squad structure.
        """
        players: list[dict] = []
        if new_squad_json:
            try:
                players = json.loads(new_squad_json) if isinstance(new_squad_json, str) else new_squad_json
            except (json.JSONDecodeError, TypeError):
                players = []

        # Ensure every player has required keys
        for p in players:
            p.setdefault("player_id", 0)
            p.setdefault("starter", False)
            p.setdefault("is_captain", False)
            p.setdefault("is_vice_captain", False)
            p.setdefault("predicted_next_gw_points", 0)
            p.setdefault("captain_score", 0)

        # Optimize starting XI + captain
        if players:
            players = optimize_starting_xi(players)

        # Extract captain/vc IDs from optimised squad
        vc_id = None
        for p in players:
            if p.get("is_captain"):
                captain_id = p["player_id"]
            if p.get("is_vice_captain"):
                vc_id = p["player_id"]

        # Build transfer lists for the squad JSON
        transfers_in = []
        transfers_out = []
        try:
            raw_transfers = json.loads(transfers_json) if isinstance(transfers_json, str) else transfers_json
            for t in raw_transfers:
                if t.get("in"):
                    transfers_in.append(t["in"])
                if t.get("out"):
                    transfers_out.append(t["out"])
        except (json.JSONDecodeError, TypeError):
            pass

        squad_json = {
            "players": players,
            "captain_id": captain_id,
            "vice_captain_id": vc_id,
            "bank": bank,
            "free_transfers": free_transfers,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "hits": hits,
            "chip": None,
            "predicted_points": self._calculate_predicted_points({
                "players": players,
                "captain_id": captain_id,
                "chip": None,
            }) if players else round(predicted_points, 1),
        }
        return squad_json

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
