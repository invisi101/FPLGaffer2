"""Pre-season planning — initial squad selection and chip plan.

Extracted from v1's ``SeasonManager.generate_preseason_plan()``.
Stubbed for now: full implementation requires the solver and strategy
modules which are not yet ported to v2.
"""

from __future__ import annotations

import json
from typing import Callable

from src.data.fpl_api import fetch_fpl_api, fetch_manager_entry
from src.data.season_detection import detect_current_season
from src.db.repositories import (
    FixtureRepository,
    PlanRepository,
    RecommendationRepository,
    SeasonRepository,
)
from src.logging_config import get_logger
from src.season.fixtures import save_fixture_calendar
from src.season.manager import ELEMENT_TYPE_MAP, SeasonManager
from src.utils.nan_handling import scrub_nan

logger = get_logger(__name__)


def generate_preseason_plan(
    manager_id: int,
    season_repo: SeasonRepository,
    rec_repo: RecommendationRepository,
    fixture_repo: FixtureRepository,
    plan_repo: PlanRepository,
    progress_fn: Callable[[str], None] | None = None,
) -> dict:
    """Pre-GW1: select initial squad and full season chip plan.

    Parameters
    ----------
    manager_id:
        FPL manager ID.
    season_repo, rec_repo, fixture_repo, plan_repo:
        Repository instances for DB access.
    progress_fn:
        Optional progress callback.

    Returns
    -------
    dict
        Pre-season plan including ``initial_squad``, ``chip_schedule``, etc.
    """

    def log(msg: str) -> None:
        if progress_fn:
            progress_fn(msg)
        logger.info(msg)

    bootstrap = fetch_fpl_api("bootstrap")
    elements = bootstrap.get("elements", [])
    elements_map = SeasonManager._get_elements_map(bootstrap)
    id_to_code, id_to_short, code_to_short = SeasonManager._get_team_maps(bootstrap)

    # Check if season already started
    next_gw = 1
    for event in bootstrap.get("events", []):
        if event.get("is_current") or event.get("is_next"):
            next_gw = event["id"]
            break

    log("Pre-season plan: building player pool...")

    # Build a price-based heuristic pool (full solver requires ml module)
    rows = []
    for el in elements:
        cost = el.get("now_cost", 0) / 10
        tid = el.get("team")
        rows.append({
            "player_id": el["id"],
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), "MID"),
            "cost": cost,
            "team_code": id_to_code.get(tid),
            "team": id_to_short.get(tid, ""),
            "predicted_next_gw_points": round((cost / 10) * 0.5, 2),
        })

    # Create or update season record
    entry = None
    try:
        entry = fetch_manager_entry(manager_id)
    except Exception:
        pass
    manager_name = ""
    team_name = ""
    if entry:
        manager_name = (
            f"{entry.get('player_first_name', '')} "
            f"{entry.get('player_last_name', '')}"
        ).strip()
        team_name = entry.get("name", "")

    season_id = season_repo.create_season(
        manager_id=manager_id,
        manager_name=manager_name,
        team_name=team_name,
        season_name=detect_current_season(),
        start_gw=1,
    )

    # Build fixture calendar
    log("  Building fixture calendar...")
    from src.data.cache import read_json_cache
    from src.paths import CACHE_DIR

    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    fixtures_data = []
    if fixtures_path.exists():
        fixtures_data = read_json_cache(fixtures_path) or []
    save_fixture_calendar(season_id, bootstrap, fixtures_data, fixture_repo)

    # Save a basic recommendation for GW1 (stub — no MILP solver yet)
    log("  Saving pre-season recommendation stub...")
    rec_repo.save_recommendation(
        season_id=season_id,
        gameweek=1,
        transfers_json=json.dumps([]),
        captain_id=None,
        captain_name=None,
        chip_suggestion=None,
        chip_values_json=json.dumps({}),
        bank_analysis_json=json.dumps({}),
        new_squad_json=None,
        predicted_points=0.0,
        base_points=0.0,
        current_xi_points=0.0,
        free_transfers=1,
    )

    # Save a stub strategic plan
    strategic_plan = {
        "timeline": [],
        "chip_schedule": {},
        "chip_synergies": [],
        "rationale": "Pre-season stub. Full plan requires model training.",
        "generated_at": __import__("datetime").datetime.now().isoformat(
            timespec="seconds"
        ),
    }
    plan_repo.save_strategic_plan(
        season_id=season_id,
        as_of_gw=1,
        plan_json=json.dumps(strategic_plan),
        chip_heatmap_json=json.dumps({}),
    )

    log("Pre-season plan complete (stub — train models for full plan).")
    return {
        "season_id": season_id,
        "manager_id": manager_id,
        "initial_squad": None,  # Requires solver
        "chip_schedule": {},
        "chip_heatmap": {},
    }
