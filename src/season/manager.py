"""Core season orchestration — init, refresh, next-GW detection.

Ported from v1's ``SeasonManager`` class (first ~300 lines), using
repository classes instead of direct DB calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.fpl_api import (
    fetch_fpl_api,
    fetch_manager_entry,
    fetch_manager_history,
    fetch_manager_picks,
    fetch_manager_transfers,
)
from src.data.season_detection import detect_current_season
from src.db.connection import connect
from src.db.migrations import apply_migrations
from src.db.repositories import (
    DashboardRepository,
    FixtureRepository,
    OutcomeRepository,
    PlanRepository,
    PriceRepository,
    RecommendationRepository,
    SeasonRepository,
    SnapshotRepository,
    WatchlistRepository,
)
from src.logging_config import get_logger
from src.paths import CACHE_DIR, DB_PATH

logger = get_logger(__name__)

ELEMENT_TYPE_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
ALL_CHIPS = {"wildcard", "freehit", "bboost", "3xc"}


class SeasonManager:
    """Orchestrates season-long FPL management.

    Uses repository classes from :mod:`src.db.repositories` for all
    database access.
    """

    def __init__(
        self,
        manager_id: int | None = None,
        season_name: str = "",
        db_path: Path | None = None,
    ):
        self.manager_id = manager_id
        self.season_name = season_name or detect_current_season()
        self.db_path = db_path or DB_PATH

        # Ensure schema is up to date
        with connect(self.db_path) as conn:
            apply_migrations(conn)

        # Instantiate repositories
        self.seasons = SeasonRepository(self.db_path)
        self.snapshots = SnapshotRepository(self.db_path)
        self.recommendations = RecommendationRepository(self.db_path)
        self.outcomes = OutcomeRepository(self.db_path)
        self.prices = PriceRepository(self.db_path)
        self.fixtures = FixtureRepository(self.db_path)
        self.plans = PlanRepository(self.db_path)
        self.dashboard_repo = DashboardRepository(self.db_path)
        self.watchlist = WatchlistRepository(self.db_path)

    # -------------------------------------------------------------------
    # Bootstrap helpers
    # -------------------------------------------------------------------

    def _load_bootstrap(self) -> dict:
        bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
        if not bootstrap_path.exists():
            raise FileNotFoundError(
                "No cached bootstrap data. Click 'Get Latest Data' first."
            )
        return json.loads(bootstrap_path.read_text(encoding="utf-8"))

    def _load_fixtures(self) -> list:
        fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
        if not fixtures_path.exists():
            return []
        return json.loads(fixtures_path.read_text(encoding="utf-8"))

    @staticmethod
    def _get_elements_map(bootstrap: dict) -> dict:
        return {el["id"]: el for el in bootstrap.get("elements", [])}

    @staticmethod
    def _get_team_maps(bootstrap: dict):
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}
        return id_to_code, id_to_short, code_to_short

    @staticmethod
    def _get_next_gw(bootstrap: dict) -> int | None:
        for event in bootstrap.get("events", []):
            if event.get("is_next"):
                return event["id"]
        # If no 'is_next', fall back to the one after 'is_current'
        for event in bootstrap.get("events", []):
            if event.get("is_current"):
                next_id = event["id"] + 1
                if next_id > 38:
                    return None  # Season is over
                return next_id
        return None

    @staticmethod
    def _calculate_free_transfers(history: dict) -> int:
        current = history.get("current", [])
        chips = history.get("chips", [])
        chip_events = {
            c["event"]
            for c in chips
            if c.get("name") in ("wildcard", "freehit")
        }

        first_event = current[0].get("event", 1) if current else 1
        ft = 1
        for i, gw_entry in enumerate(current):
            event = gw_entry.get("event")
            if event in chip_events:
                # WC/FH: FTs preserved at pre-chip count, no accrual
                continue
            transfers_made = gw_entry.get("event_transfers", 0)
            transfers_cost = gw_entry.get("event_transfers_cost", 0)
            paid = transfers_cost // 4 if transfers_cost > 0 else 0
            free_used = transfers_made - paid
            ft = ft - free_used
            ft = max(ft, 0)  # Prevent negative propagation from API inconsistencies
            # Mid-season joiner: first GW's FT was consumed by team creation
            if i == 0 and first_event > 1:
                ft = max(ft, 0)
            else:
                ft = min(ft + 1, 5)
        return max(ft, 1)

    # -------------------------------------------------------------------
    # Initialize Season
    # -------------------------------------------------------------------

    def init_season(
        self,
        manager_id: int | None = None,
        progress_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """Backfill season history from FPL API.

        Parameters
        ----------
        manager_id:
            FPL manager ID.  Falls back to ``self.manager_id``.
        progress_fn:
            Optional callable for progress updates.

        Returns
        -------
        dict
            Season info including ``season_id``, ``gws_backfilled``, etc.
        """
        mid = manager_id or self.manager_id
        if not mid:
            raise ValueError("No manager_id provided.")

        def log(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)
            logger.info(msg)

        log(f"Fetching manager {mid} info...")
        entry = fetch_manager_entry(mid)
        manager_name = (
            f"{entry.get('player_first_name', '')} "
            f"{entry.get('player_last_name', '')}"
        ).strip()
        team_name = entry.get("name", "")
        current_event = entry.get("current_event")

        # H5 fix: Pre-season — generate pre-season plan instead of returning error
        if not current_event:
            log("Season not started yet — switching to pre-season mode.")
            return self.generate_preseason_plan(mid, progress_fn=progress_fn)

        log(f"Manager: {manager_name} ({team_name})")
        log(f"Current GW: {current_event}")

        log("Fetching season history...")
        history = fetch_manager_history(mid)
        gw_entries = history.get("current", [])
        chips_used = history.get("chips", [])

        # Create season record
        start_gw = gw_entries[0]["event"] if gw_entries else 1
        season_id = self.seasons.create_season(
            manager_id=mid,
            manager_name=manager_name,
            team_name=team_name,
            season_name=self.season_name,
            start_gw=start_gw,
        )
        self.seasons.update_season_gw(season_id, current_event)

        # Clear stale recommendations, strategic plans, and outcomes so
        # the UI does not show outdated action plans after re-import.
        self.seasons.clear_generated_data(season_id)

        log(
            f"Season created (ID: {season_id}). "
            f"Backfilling {len(gw_entries)} gameweeks..."
        )

        # Build chip map: event -> chip_name
        chip_map = {c["event"]: c["name"] for c in chips_used}

        # Load bootstrap for element lookups
        bootstrap = self._load_bootstrap()
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        # Fetch all transfers and group by GW
        log("Fetching transfer history...")
        all_transfers = fetch_manager_transfers(mid)
        transfers_by_gw: dict[int, list] = {}
        for t in all_transfers:
            gw = t["event"]
            transfers_by_gw.setdefault(gw, []).append(t)

        # Backfill each played GW
        for i, gw_data in enumerate(gw_entries):
            gw = gw_data["event"]
            log(f"  Backfilling GW{gw} ({i + 1}/{len(gw_entries)})...")

            # Fetch picks for this GW
            try:
                picks_data = fetch_manager_picks(mid, gw)
                time.sleep(0.3)  # Rate limiting
            except Exception as exc:
                log(f"    Warning: Could not fetch picks for GW{gw}: {exc}")
                picks_data = {}

            # Build squad JSON
            picks = picks_data.get("picks", [])
            squad = []
            captain_id = None
            captain_name = None
            for pick in picks:
                eid = pick.get("element")
                el = elements_map.get(eid, {})
                tid = el.get("team")
                tc = id_to_code.get(tid)
                ts = code_to_short.get(tc, "")
                pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")

                player = {
                    "player_id": eid,
                    "web_name": el.get("web_name", "Unknown"),
                    "position": pos,
                    "team_code": tc,
                    "team": ts,
                    # NOTE: Historical backfill uses current now_cost, not
                    # the price at that GW.  FPL public API has no historical
                    # price data.  Prices will be approximate for past GWs.
                    "cost": el.get("now_cost", 0) / 10,
                    "starter": pick.get("position", 12) <= 11,
                    "is_captain": pick.get("is_captain", False),
                    "multiplier": pick.get("multiplier", 1),
                }
                squad.append(player)

                if pick.get("is_captain"):
                    captain_id = eid
                    captain_name = el.get("web_name", "Unknown")

            # Extract entry history
            entry_hist = picks_data.get("entry_history", {})

            # Build transfer in/out lists for this GW
            gw_transfers = transfers_by_gw.get(gw, [])
            t_in_list = []
            t_out_list = []
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

            self.snapshots.save_gw_snapshot(
                season_id=season_id,
                gameweek=gw,
                squad_json=json.dumps(squad) if squad else None,
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
                free_transfers=None,  # Calculated on demand
                chip_used=chip_map.get(gw),
                points=gw_data.get("points"),
                total_points=gw_data.get("total_points"),
                overall_rank=gw_data.get("overall_rank"),
                transfers_in_json=json.dumps(t_in_list) if t_in_list else None,
                transfers_out_json=json.dumps(t_out_list) if t_out_list else None,
                captain_id=captain_id,
                captain_name=captain_name,
                transfers_cost=gw_data.get("event_transfers_cost", 0),
            )

        # Build fixture calendar
        log("Building fixture calendar...")
        from src.season.fixtures import save_fixture_calendar

        save_fixture_calendar(season_id, bootstrap, self._load_fixtures(), self.fixtures)

        # Track current prices
        log("Tracking squad prices...")
        self._track_prices(season_id, mid, bootstrap)

        log(f"Season initialization complete! {len(gw_entries)} GWs backfilled.")
        return {
            "season_id": season_id,
            "manager_id": mid,
            "manager_name": manager_name,
            "team_name": team_name,
            "start_gw": start_gw,
            "current_gw": current_event,
            "gws_backfilled": len(gw_entries),
        }

    # -------------------------------------------------------------------
    # Refresh Data
    # -------------------------------------------------------------------

    def refresh_data(
        self,
        manager_id: int | None = None,
        force: bool = False,
        progress_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """Re-fetch data, update fixture calendar, check plan health.

        Parameters
        ----------
        manager_id:
            FPL manager ID.  Falls back to ``self.manager_id``.
        force:
            Force cache bypass on API fetches.
        progress_fn:
            Optional progress callback.

        Returns
        -------
        dict
            Result summary with keys like ``next_gw`` and ``plan_health``.
        """
        mid = manager_id or self.manager_id
        if not mid:
            raise ValueError("No manager_id provided.")

        def log(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)
            logger.info(msg)

        log("Refreshing FPL API data...")
        bootstrap = fetch_fpl_api("bootstrap", force=force)
        fetch_fpl_api("fixtures", force=force)

        next_gw = self._get_next_gw(bootstrap)

        season = self.seasons.get_season(mid, self.season_name)
        if not season:
            log("No active season found.")
            return {"next_gw": next_gw, "plan_health": None}

        season_id = season["id"]

        # Update fixture calendar
        log("Updating fixture calendar...")
        from src.season.fixtures import save_fixture_calendar

        save_fixture_calendar(
            season_id, bootstrap, self._load_fixtures(), self.fixtures
        )

        # Track prices
        log("Tracking prices...")
        self._track_prices(season_id, mid, bootstrap)

        # Check plan health (lightweight, no prediction regeneration)
        log("Checking plan health...")
        plan_health = self._check_plan_health(season_id, bootstrap)

        log("Data refresh complete.")
        return {
            "next_gw": next_gw,
            "plan_health": plan_health,
        }

    # -------------------------------------------------------------------
    # Plan Health Check
    # -------------------------------------------------------------------

    def _check_plan_health(self, season_id: int, bootstrap: dict) -> dict:
        """Lightweight plan health check using bootstrap availability data.

        Does NOT regenerate predictions.  Returns
        ``{healthy, triggers, summary}``.
        """
        plan_row = self.plans.get_strategic_plan(season_id)
        if not plan_row or not plan_row.get("plan_json"):
            return {
                "healthy": True,
                "triggers": [],
                "summary": {"critical": 0, "moderate": 0},
            }

        try:
            current_plan = json.loads(plan_row["plan_json"])
        except (json.JSONDecodeError, TypeError):
            return {
                "healthy": True,
                "triggers": [],
                "summary": {"critical": 0, "moderate": 0},
            }

        elements = bootstrap.get("elements", [])

        # Build squad_changes from injured/doubtful players
        squad_changes: dict[int, dict] = {}
        for el in elements:
            status = el.get("status", "a")
            chance = el.get("chance_of_playing_next_round")
            if status != "a" or (chance is not None and chance < 75):
                squad_changes[el["id"]] = {
                    "status": status,
                    "chance_of_playing": chance,
                    "web_name": el.get("web_name", "Unknown"),
                }

        # Get fixture calendar for fixture change detection
        fixture_calendar = self.fixtures.get_fixture_calendar(season_id)

        # Call detect_plan_invalidation (without new predictions)
        try:
            from src.strategy import detect_plan_invalidation

            triggers = detect_plan_invalidation(
                current_plan,
                new_predictions={},
                fixture_calendar=fixture_calendar,
                squad_changes=squad_changes,
            )
        except ImportError:
            logger.warning("strategy module not available for plan health check")
            triggers = []

        critical = sum(1 for t in triggers if t["severity"] == "critical")
        moderate = sum(1 for t in triggers if t["severity"] == "moderate")

        return {
            "healthy": critical == 0,
            "triggers": triggers,
            "summary": {"critical": critical, "moderate": moderate},
        }

    # -------------------------------------------------------------------
    # Price Tracking
    # -------------------------------------------------------------------

    def _track_prices(
        self, season_id: int, manager_id: int, bootstrap: dict
    ) -> None:
        """Snapshot prices for squad players + watchlist players."""
        elements = bootstrap.get("elements", [])
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

        # Get current squad
        entry = fetch_manager_entry(manager_id)
        current_event = entry.get("current_event")
        squad_ids: set[int] = set()
        if current_event:
            try:
                picks_data = fetch_manager_picks(manager_id, current_event)
                squad_ids = {p["element"] for p in picks_data.get("picks", [])}
            except Exception:
                pass

        # H4 fix: Include watchlist players alongside squad
        watchlist = self.watchlist.get_watchlist(season_id)
        watchlist_ids = {w["player_id"] for w in watchlist}
        track_ids = squad_ids | watchlist_ids

        players = []
        for el in elements:
            if el["id"] in track_ids:
                tid = el.get("team")
                players.append({
                    "player_id": el["id"],
                    "web_name": el.get("web_name"),
                    "team_code": id_to_code.get(tid),
                    "price": el.get("now_cost", 0) / 10,
                    "transfers_in_event": el.get("transfers_in_event", 0),
                    "transfers_out_event": el.get("transfers_out_event", 0),
                })

        if players:
            self.prices.save_price_snapshots_bulk(season_id, players)

    # -------------------------------------------------------------------
    # Full Strategy Pipeline
    # -------------------------------------------------------------------

    def _run_strategy_pipeline(
        self,
        log,
        season_id: int,
        next_gw: int,
        data: dict,
        df: pd.DataFrame,
        pred_result: dict,
        bootstrap: dict,
        elements_map: dict,
        id_to_code: dict,
        code_to_short: dict,
        current_squad_ids: set[int],
        total_budget: float,
        free_transfers: int,
        history: dict,
        captain_id: int | None,
        captain_name: str | None,
        predicted_points: float,
        current_xi_points: float,
        transfer_result: dict | None,
        pred_lookup: dict,
        chip_suggestion: str | None,
        players_df: pd.DataFrame | None = None,
    ) -> dict | None:
        """Run the full multi-GW strategy pipeline and save results.

        Returns the strategic_plan dict (or None on failure) so that
        ``generate_recommendation`` can extract GW+1 transfers from it.
        """
        from src.features.builder import get_fixture_context
        from src.ml.multi_gw import predict_multi_gw
        from src.strategy.reactive import apply_availability_adjustments
        from src.strategy.chip_evaluator import ChipEvaluator
        from src.strategy.transfer_planner import MultiWeekPlanner
        from src.strategy.captain_planner import CaptainPlanner
        from src.strategy.plan_synthesizer import PlanSynthesizer

        # --- Step 1: Generate multi-GW predictions ---
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

        if not future_predictions:
            raise RuntimeError("Multi-GW predictions returned empty")

        # Drop predictions for current/past GWs — only plan from next_gw onward
        future_predictions = {
            gw: gw_df for gw, gw_df in future_predictions.items() if gw >= next_gw
        }
        if not future_predictions:
            raise RuntimeError("No future predictions for GW >= %d" % next_gw)

        # --- Step 2: Replace GW+1 with UI predictions ---
        # Use the exact same numbers shown in the Predictions tab for GW+1,
        # so the strategic plan matches the recommendation (like v1 does).
        if players_df is not None and next_gw in future_predictions:
            gw1_cols = ["player_id", "predicted_next_gw_points",
                        "position", "cost", "team_code", "web_name"]
            if "captain_score" in players_df.columns:
                gw1_cols.append("captain_score")
            available_cols = [c for c in gw1_cols if c in players_df.columns]
            gw1_from_pred = players_df[available_cols].drop_duplicates(
                "player_id"
            ).copy()
            gw1_from_pred["predicted_points"] = gw1_from_pred["predicted_next_gw_points"]
            gw1_from_pred["confidence"] = 1.0
            if "team_code" in gw1_from_pred.columns:
                gw1_from_pred["team"] = (
                    gw1_from_pred["team_code"].map(code_to_short).fillna("")
                )
            future_predictions[next_gw] = gw1_from_pred

        # --- Step 2b: Enrich GW+2 onwards with bootstrap data ---
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

        first_pred_gw = min(future_predictions.keys())
        for gw in future_predictions:
            if gw == next_gw:
                continue  # Already built from players_df
            gw_df = future_predictions[gw]
            # Merge bootstrap enrichment (only columns not already present)
            merge_cols = [c for c in enrich_df.columns if c not in gw_df.columns or c == "player_id"]
            gw_df = gw_df.merge(enrich_df[merge_cols], on="player_id", how="left")
            # captain_score is GW+1 specific (uses Q80 quantile which is
            # fixture-specific).  For GW+2 onwards, fall back to
            # predicted_points so the captain planner doesn't use stale Q80.
            if "captain_score" not in gw_df.columns:
                gw_df["captain_score"] = gw_df["predicted_points"]
            else:
                gw_df["captain_score"] = gw_df["predicted_points"]
            future_predictions[gw] = gw_df

        # --- Step 3: Apply availability adjustments ---
        log("Adjusting for injuries/availability...")
        future_predictions = apply_availability_adjustments(
            future_predictions, bootstrap.get("elements", [])
        )

        # --- Step 4: Determine available chips ---
        # All 4 chips reset at the half-season boundary (GW20).
        # Each chip can be used once per half (GW1-19 and GW20-38).
        chips_used_list = history.get("chips", [])
        available_chips: set[str] = set()
        for chip in ALL_CHIPS:
            chip_events = [c["event"] for c in chips_used_list if c["name"] == chip]
            if next_gw <= 19:
                if not any(e <= 19 for e in chip_events):
                    available_chips.add(chip)
            else:
                if not any(e >= 20 for e in chip_events):
                    available_chips.add(chip)

        # --- Step 5: Get fixture calendar from DB ---
        fixture_calendar = self.fixtures.get_fixture_calendar(
            season_id, from_gw=next_gw
        )

        # --- Step 6: Run ChipEvaluator ---
        log("Evaluating chip strategies...")
        chip_eval = ChipEvaluator()
        chip_heatmap = chip_eval.evaluate_all_chips(
            current_squad_ids, total_budget, available_chips,
            future_predictions, fixture_calendar,
        )
        chip_synergies = chip_eval.evaluate_chip_synergies(
            chip_heatmap, available_chips
        )

        # --- Step 7: Get price alerts ---
        price_alerts = self.get_price_alerts(season_id)

        # --- H1 fix: Build chip plan from heatmap for transfer planner ---
        chip_plan_for_planner = None
        if chip_heatmap:
            synthesizer_pre = PlanSynthesizer()
            chip_schedule_pre = synthesizer_pre._plan_chip_schedule(
                chip_heatmap, chip_synergies, available_chips,
            )
            # Invert chip_schedule from {chip: gw} to {gw: chip}
            chip_gws_by_gw = {gw: chip for chip, gw in chip_schedule_pre.items()}
            chip_plan_for_planner = {"chip_gws": chip_gws_by_gw}

        # --- Step 8: Run MultiWeekPlanner ---
        log("Planning transfers over 5 GWs...")
        planner = MultiWeekPlanner()
        transfer_plan = planner.plan_transfers(
            current_squad_ids, total_budget, free_transfers,
            future_predictions, fixture_calendar, price_alerts,
            chip_plan=chip_plan_for_planner,
        )

        # --- Step 9: Run CaptainPlanner ---
        log("Planning captaincy...")
        captain_planner = CaptainPlanner()
        captain_plan = captain_planner.plan_captaincy(
            current_squad_ids, future_predictions, transfer_plan,
        )

        # --- Step 10: Run PlanSynthesizer ---
        log("Synthesizing strategic plan...")
        synthesizer = PlanSynthesizer()
        strategic_plan = synthesizer.synthesize(
            transfer_plan, captain_plan,
            chip_heatmap, chip_synergies, available_chips,
        )

        # --- Step 11: Extract chip for next GW ---
        chip_schedule = strategic_plan.get("chip_schedule", {})
        next_gw_chip = None
        for chip_name, chip_gw in chip_schedule.items():
            if chip_gw == next_gw:
                next_gw_chip = chip_name
                break

        # Update the recommendation's chip_suggestion if pipeline found one
        if next_gw_chip and next_gw_chip != chip_suggestion:
            self.recommendations.update_chip_suggestion(
                season_id, next_gw, next_gw_chip
            )

        # --- Step 12: Save strategic plan + changelog ---
        log("Saving strategic plan...")
        # Serialise chip_heatmap with string keys for JSON
        heatmap_serialisable = {
            k: {str(gw): v for gw, v in vals.items()}
            for k, vals in chip_heatmap.items()
        }

        # H2 fix: Compare old vs new plans and log specific changes
        old_plan_row = self.plans.get_strategic_plan(season_id)
        old_plan = None
        if old_plan_row and old_plan_row.get("plan_json"):
            try:
                old_plan = json.loads(old_plan_row["plan_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        self.plans.save_strategic_plan(
            season_id=season_id,
            as_of_gw=next_gw,
            plan_json=json.dumps(strategic_plan),
            chip_heatmap_json=json.dumps(heatmap_serialisable),
        )

        # Log detailed plan changes if previous plan exists
        if old_plan:
            self._log_plan_changes(season_id, old_plan, strategic_plan, next_gw)

        n_timeline = len(strategic_plan.get("timeline", []))
        n_chips = len(chip_schedule)
        log(f"Strategy pipeline complete: {n_timeline} GW timeline, {n_chips} chips scheduled.")

        # Stash future predictions for bank analysis (M5)
        self._last_future_predictions = future_predictions

        return strategic_plan

    def _log_plan_changes(
        self,
        season_id: int,
        old_plan: dict,
        new_plan: dict,
        next_gw: int,
    ) -> None:
        """Compare old and new strategic plans and log specific differences.

        Detects chip reschedules and captain changes, mirroring v1 behaviour.
        """
        old_schedule = old_plan.get("chip_schedule", {})
        new_schedule = new_plan.get("chip_schedule", {})

        for chip in set(list(old_schedule.keys()) + list(new_schedule.keys())):
            old_gw = old_schedule.get(chip)
            new_gw = new_schedule.get(chip)
            if old_gw != new_gw:
                self.plans.save_plan_change(
                    season_id=season_id,
                    gameweek=next_gw,
                    change_type="chip_schedule",
                    description=f"{chip} rescheduled",
                    old_value=f"GW{old_gw}" if old_gw else "unscheduled",
                    new_value=f"GW{new_gw}" if new_gw else "unscheduled",
                    reason="Updated predictions/fixtures",
                )

        # Compare captain plan
        old_timeline = old_plan.get("timeline", [])
        new_timeline = new_plan.get("timeline", [])
        old_caps = {e["gw"]: e.get("captain_name") for e in old_timeline if "captain_name" in e}
        new_caps = {e["gw"]: e.get("captain_name") for e in new_timeline if "captain_name" in e}
        for gw in set(list(old_caps.keys()) + list(new_caps.keys())):
            if old_caps.get(gw) != new_caps.get(gw) and old_caps.get(gw) and new_caps.get(gw):
                self.plans.save_plan_change(
                    season_id=season_id,
                    gameweek=next_gw,
                    change_type="captain",
                    description=f"GW{gw} captain changed",
                    old_value=old_caps.get(gw, ""),
                    new_value=new_caps.get(gw, ""),
                    reason="Updated predictions",
                )

    def _save_fallback_strategic_plan(
        self,
        log,
        season_id: int,
        next_gw: int,
        elements_map: dict,
        id_to_code: dict,
        pred_lookup: dict,
        current_squad_ids: set[int],
        free_transfers: int,
        captain_id: int | None,
        captain_name: str | None,
        predicted_points: float,
        current_xi_points: float,
        transfer_result: dict | None,
        chip_suggestion: str | None,
    ) -> None:
        """Save a single-GW stub strategic plan (fallback when pipeline fails)."""
        log("Saving single-GW fallback strategic plan...")

        transfers_in_list = []
        transfers_out_list = []
        if transfer_result:
            for pid in transfer_result.get("transfers_in_ids", set()):
                el = elements_map.get(pid, {})
                tc = id_to_code.get(el.get("team"))
                preds = pred_lookup.get(pid, {})
                transfers_in_list.append({
                    "player_id": pid,
                    "web_name": el.get("web_name", "Unknown"),
                    "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                    "cost": round(el.get("now_cost", 0) / 10, 1),
                    "predicted_points": preds.get("predicted_next_gw_points", 0),
                })
            for pid in transfer_result.get("transfers_out_ids", set()):
                el = elements_map.get(pid, {})
                transfers_out_list.append({
                    "player_id": pid,
                    "web_name": el.get("web_name", "Unknown"),
                })

        n_transfers = len(transfers_in_list)
        ft_used = min(n_transfers, free_transfers)
        timeline_entry = {
            "gw": next_gw,
            "confidence": 0.85,
            "ft_available": free_transfers,
            "ft_used": ft_used,
            "transfers_in": transfers_in_list,
            "transfers_out": transfers_out_list,
            "captain_name": captain_name,
            "captain_points": pred_lookup.get(captain_id, {}).get(
                "predicted_next_gw_points", 0
            ) if captain_id else 0,
            "predicted_points": predicted_points,
            "chip": chip_suggestion,
        }

        gain = predicted_points - current_xi_points
        rationale_parts = []
        if n_transfers > 0:
            rationale_parts.append(
                f"Recommending {n_transfers} transfer{'s' if n_transfers > 1 else ''} "
                f"for GW{next_gw}"
            )
        if gain > 0.5:
            rationale_parts.append(f"Expected gain of +{gain:.1f} pts over current XI")
        if captain_name:
            rationale_parts.append(f"Captain: {captain_name}")
        plan_rationale = ". ".join(rationale_parts) + "." if rationale_parts else "No changes recommended."

        strategic_plan = {
            "rationale": plan_rationale,
            "timeline": [timeline_entry],
            "chip_schedule": {},
            "chip_synergies": [],
        }
        if chip_suggestion:
            strategic_plan["chip_schedule"][chip_suggestion] = next_gw

        self.plans.save_strategic_plan(
            season_id=season_id,
            as_of_gw=next_gw,
            plan_json=json.dumps(strategic_plan),
            chip_heatmap_json="{}",
        )

    # -------------------------------------------------------------------
    # Bank vs Use Analysis
    # -------------------------------------------------------------------

    def _analyze_bank_vs_use(
        self,
        players_df: pd.DataFrame,
        current_squad_ids: set[int],
        total_budget: float,
        free_transfers: int,
        code_to_short: dict,
        gw2_pool: pd.DataFrame | None = None,
    ) -> dict:
        """Multi-week lookahead: compare FT allocation strategies over 2 weeks.

        For each option (use 0, 1, 2... FTs this week), simulates:
          Week 1: solve with N transfers -> squad + points
          Week 2: solve with rolled-over FTs from new squad -> points
        Picks the strategy that maximizes total points over both weeks.
        """
        from src.solver.transfers import solve_transfer_milp

        gw_col = "predicted_next_gw_points"
        if gw_col not in players_df.columns:
            return {"recommendation": "insufficient_data"}

        pool = players_df.dropna(subset=["position", "cost", gw_col]).copy()
        if "team_code" in pool.columns:
            pool["team"] = pool["team_code"].map(code_to_short).fillna("")

        # Current XI points (week 1 baseline with 0 transfers)
        current_pred = pool[pool["player_id"].isin(current_squad_ids)]
        current_xi_pts = (
            current_pred.nlargest(min(11, len(current_pred)), gw_col)[gw_col].sum()
            if not current_pred.empty
            else 0
        )
        # Add captain bonus to baseline for fair comparison
        if not current_pred.empty and len(current_pred) >= 1:
            cap_score_col = "captain_score" if "captain_score" in current_pred.columns else gw_col
            top11 = current_pred.nlargest(min(11, len(current_pred)), gw_col)
            if not top11.empty:
                best_cap_idx = top11[cap_score_col].idxmax()
                current_xi_pts += top11.loc[best_cap_idx, gw_col]

        # Simulate each strategy: use 0..min(ft, 3) FTs this week
        max_use = min(free_transfers, 3)
        strategies = []

        cap_col = "captain_score" if "captain_score" in pool.columns else None

        for use_now in range(0, max_use + 1):
            # Week 1: solve with use_now transfers
            if use_now == 0:
                week1_pts = current_xi_pts
                week1_squad_ids = current_squad_ids
                week1_budget = total_budget
            else:
                result_w1 = solve_transfer_milp(
                    pool, current_squad_ids, gw_col,
                    budget=total_budget, max_transfers=use_now,
                    captain_col=cap_col,
                )
                if result_w1:
                    week1_pts = result_w1["starting_points"]
                    week1_squad_ids = {p["player_id"] for p in result_w1["players"]}
                    week1_budget = total_budget
                else:
                    week1_pts = current_xi_pts
                    week1_squad_ids = current_squad_ids
                    week1_budget = total_budget

            # FTs available next week: unused FTs roll over + 1 new, capped at 5
            remaining = free_transfers - use_now
            next_week_fts = min(remaining + 1, 5)

            # Week 2: solve from week 1's squad with next_week_fts transfers
            # M5: Use GW+2 predictions when available for more accurate week 2
            w2_pool = pool
            w2_col = gw_col
            w2_cap_col = cap_col
            if gw2_pool is not None and "predicted_points" in gw2_pool.columns:
                w2_df = gw2_pool.copy()
                if "position" in w2_df.columns and "cost" in w2_df.columns:
                    w2_col = "predicted_points"
                    w2_cap_col = (
                        "captain_score"
                        if "captain_score" in w2_df.columns
                        else None
                    )
                    w2_pool = w2_df.dropna(subset=["position", "cost", w2_col])
            result_w2 = solve_transfer_milp(
                w2_pool, week1_squad_ids, w2_col,
                budget=week1_budget, max_transfers=next_week_fts,
                captain_col=w2_cap_col,
            )
            week2_pts = result_w2["starting_points"] if result_w2 else 0

            total_pts = round(week1_pts + week2_pts, 2)

            strategies.append({
                "use_this_week": use_now,
                "fts_next_week": next_week_fts,
                "week1_points": round(week1_pts, 2),
                "week2_points": round(week2_pts, 2),
                "total_2week": total_pts,
            })

        # Pick best strategy
        best = max(strategies, key=lambda s: s["total_2week"])
        best_use = best["use_this_week"]

        # Determine recommendation
        if best_use == 0:
            rec = "bank"
        else:
            rec = f"use_{best_use}"

        # Build comparison summary
        bank_strat = next((s for s in strategies if s["use_this_week"] == 0), None)
        use1_strat = next((s for s in strategies if s["use_this_week"] == 1), None)

        return {
            "recommendation": rec,
            "best_strategy": best,
            "strategies": strategies,
            "free_transfers": free_transfers,
            "horizon": "2 weeks",
            "gain_from_transfer": round(
                (use1_strat["total_2week"] - bank_strat["total_2week"])
                if use1_strat and bank_strat
                else 0,
                2,
            ),
        }

    # -------------------------------------------------------------------
    # Public Facade Methods (used by API blueprints)
    # -------------------------------------------------------------------

    def get_dashboard(self, manager_id: int) -> dict:
        """Full dashboard data for the Season tab."""
        from src.season.dashboard import get_dashboard

        return get_dashboard(
            manager_id,
            self.seasons,
            self.snapshots,
            self.recommendations,
            self.outcomes,
            self.dashboard_repo,
            self.season_name,
        )

    def record_actual_results(
        self,
        manager_id: int,
        progress_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """Post-GW: import actual picks/results and compare to recommendation."""
        from src.season.recorder import record_results

        return record_results(
            manager_id,
            self.seasons,
            self.snapshots,
            self.recommendations,
            self.outcomes,
            self.season_name,
            progress_fn,
        )

    def generate_preseason_plan(
        self,
        manager_id: int,
        progress_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """Pre-GW1: select initial squad and full season chip plan."""
        from src.season.preseason import generate_preseason_plan

        return generate_preseason_plan(
            manager_id,
            self.seasons,
            self.recommendations,
            self.fixtures,
            self.plans,
            progress_fn,
        )

    def generate_recommendation(
        self,
        manager_id: int,
        progress_fn: Callable[[str], None] | None = None,
    ) -> dict:
        """Generate strategic plan + transfer recommendation for next GW.

        Orchestrates: data refresh -> predictions -> strategic pipeline ->
        extract GW+1 from plan -> DB store.  Falls back to single-GW MILP
        if the pipeline fails.
        """

        def log(msg: str) -> None:
            if progress_fn:
                progress_fn(msg)
            logger.info(msg)

        mid = manager_id or self.manager_id
        if not mid:
            raise ValueError("No manager_id provided.")

        season = self.seasons.get_season(mid, self.season_name)
        if not season:
            raise ValueError("No active season. Initialize season first.")
        season_id = season["id"]

        # 1) Load data and generate predictions
        log("Loading data...")
        from src.data.loader import load_all_data
        from src.features.builder import build_features
        from src.ml.prediction import generate_predictions

        data = load_all_data(force=True)
        df = build_features(data)

        log("Generating predictions...")
        result = generate_predictions(df, data)
        players_df = result.get("players")
        if players_df is None or (hasattr(players_df, "empty") and players_df.empty):
            raise ValueError("Prediction generation failed.")

        bootstrap = self._load_bootstrap()
        next_gw = self._get_next_gw(bootstrap)
        if not next_gw:
            raise ValueError("Could not determine next GW.")

        # Save predictions to output
        from src.paths import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(players_df, "to_csv"):
            players_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

        # 2) Get current squad
        log("Fetching current squad...")
        entry = fetch_manager_entry(mid)
        current_event = entry.get("current_event")
        if not current_event:
            raise ValueError("Manager has no current event.")

        picks_data = fetch_manager_picks(mid, current_event)
        history = fetch_manager_history(mid)
        free_transfers = self._calculate_free_transfers(history)

        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        picks = picks_data.get("picks", [])
        entry_history = picks_data.get("entry_history", {})
        bank = entry_history.get("bank", 0) / 10

        current_squad_ids = set()
        current_squad_cost = 0.0
        for pick in picks:
            eid = pick.get("element")
            current_squad_ids.add(eid)
            el = elements_map.get(eid, {})
            current_squad_cost += el.get("now_cost", 0) / 10

        # C1 fix: Use entry_history "value" (includes 50% price-rise rule)
        # when available, fall back to sum(now_cost) + bank (slightly optimistic).
        api_value = entry_history.get("value")
        if api_value:
            total_budget = round(api_value / 10, 1)
        else:
            total_budget = round(bank + current_squad_cost, 1)

        # Build opponent map: team_code -> "OPP(H/A)" for next GW
        fixtures_list = self._load_fixtures()
        opponent_map: dict = {}
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

        # Normalise position and enrich with cost/team_code from bootstrap
        if "position_clean" in players_df.columns and "position" not in players_df.columns:
            players_df["position"] = players_df["position_clean"]
        if "cost" not in players_df.columns:
            players_df["cost"] = None
        if "team_code" not in players_df.columns:
            players_df["team_code"] = None
        for idx, row in players_df.iterrows():
            el = elements_map.get(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
            if el:
                if pd.isna(row.get("cost")) or row.get("cost") is None:
                    players_df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
                if pd.isna(row.get("team_code")) or row.get("team_code") is None:
                    players_df.at[idx, "team_code"] = id_to_code.get(el.get("team"))

        target = "predicted_next_gw_points"

        # Build a lookup for prediction data by player_id
        pred_lookup: dict[int, dict] = {}
        if hasattr(players_df, "iterrows"):
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

        # 3) Initialise recommendation defaults
        transfers_json = "[]"
        captain_id = None
        captain_name = None
        chip_suggestion = None
        predicted_points = 0.0
        base_points = 0.0
        current_xi_points = 0.0
        new_squad_json = None
        chip_values_json = "{}"
        bank_analysis_json = "{}"
        transfer_result = None

        # C2 fix: Run strategy pipeline FIRST and extract GW+1 from it.
        # Only fall back to single-GW solver if pipeline fails.
        strategic_plan = None
        try:
            strategic_plan = self._run_strategy_pipeline(
                log, season_id, next_gw, data, df, result,
                bootstrap, elements_map, id_to_code, code_to_short,
                current_squad_ids, total_budget, free_transfers,
                history, captain_id, captain_name,
                predicted_points, current_xi_points,
                transfer_result, pred_lookup, chip_suggestion,
                players_df,
            )
        except Exception as exc:
            logger.warning("Strategy pipeline failed: %s", exc)
            strategic_plan = None

        # Try to extract GW+1 transfers from the strategic plan timeline
        used_pipeline = False
        if strategic_plan and strategic_plan.get("timeline"):
            timeline = strategic_plan["timeline"]
            gw1_entry = next(
                (e for e in timeline if e.get("gw") == next_gw), None
            )
            if gw1_entry is None and timeline:
                gw1_entry = timeline[0]

            if gw1_entry:
                used_pipeline = True
                predicted_points = gw1_entry.get("predicted_points", 0)

                # Extract captain from timeline
                captain_name = gw1_entry.get("captain_name")
                captain_id_from_plan = gw1_entry.get("captain_id")
                if captain_id_from_plan:
                    captain_id = captain_id_from_plan
                    if not captain_name:
                        el = elements_map.get(captain_id, {})
                        captain_name = el.get("web_name", "Unknown")

                # Extract chip from plan chip_schedule
                chip_schedule = strategic_plan.get("chip_schedule", {})
                for cn, cg in chip_schedule.items():
                    if cg == next_gw:
                        chip_suggestion = cn
                        break

                # Build transfer pairs from pipeline
                transfers = []
                def _enrich_player(pid: int) -> dict:
                    """Build a rich player dict from bootstrap + predictions."""
                    el = elements_map.get(pid, {})
                    tc = id_to_code.get(el.get("team"))
                    preds = pred_lookup.get(pid, {})
                    return {
                        "player_id": pid,
                        "web_name": el.get("web_name", "Unknown"),
                        "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                        "team_code": tc,
                        "team": code_to_short.get(tc, ""),
                        "cost": round(el.get("now_cost", 0) / 10, 1),
                        "total_points": el.get("total_points", 0),
                        "event_points": el.get("event_points", 0),
                        "predicted_next_gw_points": preds.get("predicted_next_gw_points", 0),
                        "predicted_next_3gw_points": preds.get("predicted_next_3gw_points"),
                        "captain_score": preds.get("captain_score", 0),
                        "opponent": opponent_map.get(tc, ""),
                    }

                t_in_list = gw1_entry.get("transfers_in", [])
                t_out_list = gw1_entry.get("transfers_out", [])
                max_len = max(len(t_out_list), len(t_in_list))
                for i in range(max_len):
                    out_entry = _enrich_player(t_out_list[i].get("player_id")) if i < len(t_out_list) else {}
                    in_entry = _enrich_player(t_in_list[i].get("player_id")) if i < len(t_in_list) else {}
                    transfers.append({"out": out_entry, "in": in_entry})

                transfers_json = json.dumps(transfers)

                # Build new squad from pipeline if available
                if gw1_entry.get("new_squad"):
                    new_squad_json = json.dumps(gw1_entry["new_squad"])
                elif gw1_entry.get("squad_ids"):
                    # Build enriched squad from squad_ids
                    enriched_squad = []
                    squad_ids_plan = set(gw1_entry["squad_ids"])
                    for pid in squad_ids_plan:
                        enriched_squad.append(_enrich_player(pid))
                    new_squad_json = json.dumps(enriched_squad)

        # Fall back to single-GW solver if pipeline didn't produce GW+1 data
        if not used_pipeline:
            log("Running single-GW transfer solver (fallback)...")
            from src.solver.transfers import solve_transfer_milp_with_hits

            pool = players_df.dropna(subset=["position", "cost", target]).copy()
            captain_col = "captain_score" if "captain_score" in pool.columns else None

            transfer_result = solve_transfer_milp_with_hits(
                pool,
                current_squad_ids,
                target,
                budget=total_budget,
                free_transfers=free_transfers,
                max_transfers=4,
                captain_col=captain_col,
            )

            if transfer_result:
                log("Building recommendation...")

                def _enrich_player_fb(pid: int) -> dict:
                    """Build a rich player dict from bootstrap + predictions."""
                    el = elements_map.get(pid, {})
                    tc = id_to_code.get(el.get("team"))
                    preds = pred_lookup.get(pid, {})
                    return {
                        "player_id": pid,
                        "web_name": el.get("web_name", "Unknown"),
                        "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                        "team_code": tc,
                        "team": code_to_short.get(tc, ""),
                        "cost": round(el.get("now_cost", 0) / 10, 1),
                        "total_points": el.get("total_points", 0),
                        "event_points": el.get("event_points", 0),
                        "predicted_next_gw_points": preds.get("predicted_next_gw_points", 0),
                        "predicted_next_3gw_points": preds.get("predicted_next_3gw_points"),
                        "captain_score": preds.get("captain_score", 0),
                        "opponent": opponent_map.get(tc, ""),
                    }

                # Build transfer list
                transfers = []
                out_ids = list(transfer_result.get("transfers_out_ids", set()))
                in_ids = list(transfer_result.get("transfers_in_ids", set()))
                max_len = max(len(out_ids), len(in_ids))
                for i in range(max_len):
                    out_entry = _enrich_player_fb(out_ids[i]) if i < len(out_ids) else {}
                    in_entry = _enrich_player_fb(in_ids[i]) if i < len(in_ids) else {}
                    transfers.append({"out": out_entry, "in": in_entry})

                transfers_json = json.dumps(transfers)
                captain_id = transfer_result.get("captain_id")
                if captain_id:
                    el = elements_map.get(captain_id, {})
                    captain_name = el.get("web_name", "Unknown")

                # Points: baseline (no changes) vs after transfers
                current_xi_points = transfer_result.get("baseline_points", 0)
                base_points = transfer_result.get("starting_points", 0)
                predicted_points = transfer_result.get("starting_points", 0)

                # Enrich new_squad with full player data for pitch rendering
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
                        "total_points": el.get("total_points", 0),
                        "event_points": el.get("event_points", 0),
                        "starter": bool(p.get("starter", False)),
                        "is_captain": pid == captain_id if captain_id else False,
                        "is_vice_captain": False,
                        "opponent": opponent_map.get(tc, ""),
                    })
                # Set vice captain (highest captain_score starter after captain)
                vc_candidates = [
                    p for p in enriched_squad if p["starter"] and not p["is_captain"]
                ]
                if vc_candidates:
                    vc_candidates.sort(
                        key=lambda p: p.get("captain_score", 0), reverse=True
                    )
                    vc_candidates[0]["is_vice_captain"] = True
                new_squad_json = json.dumps(enriched_squad)

            # Save fallback strategic plan if pipeline didn't run
            if not strategic_plan:
                self._save_fallback_strategic_plan(
                    log, season_id, next_gw,
                    elements_map, id_to_code, pred_lookup,
                    current_squad_ids, free_transfers,
                    captain_id, captain_name,
                    predicted_points, current_xi_points,
                    transfer_result, chip_suggestion,
                )

        # H3 fix: Run bank vs use analysis
        log("Running bank vs use analysis...")
        try:
            # M5: Use GW+2 predictions for week 2 comparison when available
            gw2_pool = None
            fp = getattr(self, "_last_future_predictions", None)
            if fp and next_gw and (next_gw + 1) in fp:
                gw2_pool = fp[next_gw + 1]
            bank_analysis = self._analyze_bank_vs_use(
                players_df, current_squad_ids, total_budget, free_transfers,
                code_to_short, gw2_pool=gw2_pool,
            )
            bank_analysis_json = json.dumps(bank_analysis)
        except Exception as exc:
            logger.warning("Bank analysis failed: %s", exc)
            bank_analysis_json = "{}"

        # 5) Store recommendation
        log("Saving recommendation...")
        self.recommendations.save_recommendation(
            season_id=season_id,
            gameweek=next_gw,
            transfers_json=transfers_json,
            captain_id=captain_id,
            captain_name=captain_name,
            chip_suggestion=chip_suggestion,
            chip_values_json=chip_values_json,
            bank_analysis_json=bank_analysis_json,
            new_squad_json=new_squad_json,
            predicted_points=predicted_points,
            base_points=base_points,
            current_xi_points=current_xi_points,
            free_transfers=free_transfers,
        )

        # 6) Update fixture calendar
        log("Updating fixtures...")
        from src.season.fixtures import save_fixture_calendar

        save_fixture_calendar(
            season_id, bootstrap, self._load_fixtures(), self.fixtures
        )

        # 7) Track prices
        log("Tracking prices...")
        self._track_prices(season_id, mid, bootstrap)

        self.seasons.update_season_gw(season_id, current_event)
        log("Recommendation complete.")
        return {
            "gameweek": next_gw,
            "predicted_points": predicted_points,
            "captain": captain_name,
            "n_transfers": len(json.loads(transfers_json)) if transfers_json != "[]" else 0,
        }

    def get_action_plan(self, manager_id: int) -> dict:
        """Build a human-readable action plan from the latest recommendation."""
        season = self.seasons.get_season(manager_id, self.season_name)
        if not season:
            return {"error": "No active season."}
        season_id = season["id"]

        # M2 fix: Look up recommendation by actual next_gw, not recs[-1]
        try:
            bootstrap = self._load_bootstrap()
            next_gw = self._get_next_gw(bootstrap)
        except Exception:
            bootstrap = None
            next_gw = None

        if next_gw:
            latest = self.recommendations.get_recommendation(season_id, next_gw)
        else:
            latest = None

        # Fall back to most recent recommendation if next_gw lookup fails
        if not latest:
            recs = self.recommendations.get_recommendations(season_id)
            if not recs:
                return {"error": "No recommendations yet. Generate one first."}
            latest = recs[-1]

        gw = latest.get("gameweek")

        transfers = []
        try:
            transfers = json.loads(latest.get("transfers_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            pass

        # Get GW deadline from bootstrap
        deadline = None
        try:
            if bootstrap is None:
                bootstrap = self._load_bootstrap()
            for ev in bootstrap.get("events", []):
                if ev.get("id") == gw:
                    deadline = ev.get("deadline_time")
                    break
        except Exception:
            pass

        steps: list[dict] = []
        # Transfer steps
        if transfers:
            for t in transfers:
                out_info = t.get("out", {})
                in_info = t.get("in", {})
                out_name = out_info.get("web_name", "?")
                in_name = in_info.get("web_name", "?")
                if out_name != "?" and in_name != "?":
                    desc = f"Transfer OUT {out_name} → IN {in_name}"
                elif in_name != "?":
                    desc = f"Transfer IN {in_name}"
                else:
                    continue
                step = {"action": "transfer", "description": desc}
                step["player_out"] = out_info
                step["player_in"] = in_info
                steps.append(step)
        else:
            steps.append({
                "action": "transfer",
                "description": "Bank your free transfer (no transfers recommended)",
            })

        # Captain step
        captain_name = latest.get("captain_name")
        if captain_name:
            steps.append({
                "action": "captain",
                "description": f"Set captain: {captain_name}",
                "captain_id": latest.get("captain_id"),
            })

        # Chip step
        chip = latest.get("chip_suggestion")
        if chip:
            chip_labels = {
                "wildcard": "Wildcard",
                "freehit": "Free Hit",
                "bboost": "Bench Boost",
                "3xc": "Triple Captain",
            }
            step = {
                "action": "chip",
                "description": f"Activate chip: {chip_labels.get(chip, chip)}",
            }
            # Attach new squad for WC/FH pitch rendering
            if chip in ("wildcard", "freehit"):
                try:
                    new_squad = json.loads(latest.get("new_squad_json") or "[]")
                    step["new_squad"] = new_squad
                except (json.JSONDecodeError, TypeError):
                    pass
            steps.append(step)

        # Build rationale
        n_transfers = len(transfers) if transfers else 0
        ft = latest.get("free_transfers") or 1
        pts_current = latest.get("current_xi_points") or 0
        pts_after = latest.get("predicted_points") or 0
        gain = pts_after - pts_current
        rationale_parts = []
        if n_transfers > 0:
            rationale_parts.append(
                f"{n_transfers} transfer{'s' if n_transfers > 1 else ''} "
                f"using {min(n_transfers, ft)} of {ft} free transfer{'s' if ft > 1 else ''}"
            )
        if gain > 0.5:
            rationale_parts.append(f"Expected gain: +{gain:.1f} pts")
        rationale = ". ".join(rationale_parts) + "." if rationale_parts else None

        return {
            "gameweek": gw,
            "deadline": deadline,
            "steps": steps,
            "rationale": rationale,
            "predicted_points": latest.get("predicted_points"),
            "captain": captain_name,
            "chip": chip,
            "free_transfers": latest.get("free_transfers"),
        }

    def get_outcomes(self, manager_id: int) -> list[dict]:
        """Return all recorded outcomes for the active season."""
        season = self.seasons.get_season(manager_id, self.season_name)
        if not season:
            return []
        return self.outcomes.get_outcomes(season["id"])

    def check_plan_health(self, manager_id: int) -> dict:
        """Lightweight plan health check using bootstrap availability data."""
        season = self.seasons.get_season(manager_id, self.season_name)
        if not season:
            return {
                "healthy": True,
                "triggers": [],
                "summary": {"critical": 0, "moderate": 0},
            }

        bootstrap = self._load_bootstrap()
        return self._check_plan_health(season["id"], bootstrap)

    def get_price_alerts(self, season_id: int) -> list[dict]:
        """Price alerts based on net transfer volume."""
        from src.strategy.price_tracker import get_price_alerts

        bootstrap = self._load_bootstrap()
        return get_price_alerts(bootstrap)

    def predict_price_changes(self, season_id: int) -> list[dict]:
        """Ownership-based price change predictions."""
        from src.strategy.price_tracker import predict_price_changes

        bootstrap = self._load_bootstrap()
        return predict_price_changes(bootstrap)

    def get_price_history(
        self,
        season_id: int,
        player_ids: list[int] | None = None,
        days: int = 14,
    ) -> dict:
        """Price history for tracked players."""
        from src.strategy.price_tracker import get_price_history

        all_history = self.prices.get_price_history(season_id)
        return get_price_history(all_history, player_ids=player_ids, days=days)

    def track_prices(self, season_id: int, manager_id: int) -> None:
        """Public wrapper: snapshot squad prices."""
        bootstrap = self._load_bootstrap()
        self._track_prices(season_id, manager_id, bootstrap)

    def update_fixture_calendar(self, season_id: int) -> None:
        """Rebuild and save fixture calendar."""
        from src.season.fixtures import save_fixture_calendar

        bootstrap = self._load_bootstrap()
        fixtures = self._load_fixtures()
        save_fixture_calendar(season_id, bootstrap, fixtures, self.fixtures)
