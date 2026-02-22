"""Compare blueprint — GW compare endpoint."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import (
    ELEMENT_TYPE_MAP,
    ensure_pipeline_data,
    get_team_map,
    load_bootstrap,
    scrub_nan,
)
from src.api.sse import pipeline_cache
from src.logging_config import get_logger
from src.paths import CACHE_DIR

log = get_logger(__name__)

compare_bp = Blueprint("compare", __name__)


@compare_bp.route("/api/gw-compare", methods=["POST"])
def api_gw_compare():
    """Compare a manager's actual FPL team against the hindsight-best for a GW."""
    from src.data.fpl_api import fetch_manager_picks
    from src.solver.squad import solve_milp_team

    body = request.get_json(silent=True) or {}
    try:
        manager_id = int(body.get("manager_id", 0))
        gameweek = int(body.get("gameweek", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id and gameweek must be integers."}), 400

    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    if not (1 <= gameweek <= 38):
        return jsonify({"error": "gameweek must be between 1 and 38."}), 400

    try:
        picks_data = fetch_manager_picks(manager_id, gameweek)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks for GW{gameweek}: {exc}"}), 404

    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached bootstrap data. Refresh data first."}), 400

    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    from src.data.season_detection import detect_current_season
    season = body.get("season", detect_current_season())
    team_map = get_team_map(season)

    if not ensure_pipeline_data():
        return jsonify({"error": "Could not load pipeline data."}), 500

    df = pipeline_cache["df"]
    season_gw = df[(df["season"] == season) & (df["gameweek"] == gameweek)]
    if season_gw.empty:
        return jsonify({"error": f"No data for {season} GW{gameweek} in feature matrix."}), 404

    actuals = season_gw[["player_id", "event_points"]].drop_duplicates("player_id", keep="first")
    actuals_map = dict(zip(actuals["player_id"], actuals["event_points"]))

    # Manager's picks
    picks = picks_data.get("picks", [])
    entry_history = picks_data.get("entry_history", {})
    budget = round(entry_history.get("value", 0) / 10, 1)

    my_squad = []
    for pick in picks:
        eid = pick.get("element")
        el = elements_map.get(eid, {})
        team_id = el.get("team")
        tc = team_id_to_code.get(team_id)

        # Use historical cost from feature matrix when available
        cost_series = season_gw.loc[season_gw["player_id"] == eid, "cost"]
        cost = float(cost_series.iloc[0]) if not cost_series.empty else el.get("now_cost", 0) / 10

        my_squad.append({
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
            "team_code": tc,
            "team": team_map.get(tc, ""),
            "cost": cost,
            "actual": actuals_map.get(eid, 0),
            "multiplier": pick.get("multiplier", 1),
            "starter": pick.get("position", 12) <= 11,
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
        })

    my_starters = [p for p in my_squad if p["starter"]]
    my_bench = [p for p in my_squad if not p["starter"]]
    my_starting_actual = round(
        sum((p["actual"] or 0) * p["multiplier"] for p in my_starters), 1,
    )

    # Build pool for hindsight-best
    pool_df = season_gw.copy()
    pool_df = pool_df.drop(columns=["position"], errors="ignore")
    pool_df = pool_df.rename(columns={"position_clean": "position", "event_points": "actual"})
    if "team_code" in pool_df.columns:
        pool_df["team"] = pool_df["team_code"].map(team_map).fillna("")
    keep_cols = ["player_id", "web_name", "position", "team_code", "team", "cost", "actual"]
    keep_cols = [c for c in keep_cols if c in pool_df.columns]
    pool_df = pool_df[keep_cols].drop_duplicates("player_id", keep="first")

    best_result = solve_milp_team(pool_df, "actual", budget=budget)
    if best_result is None:
        return jsonify({"error": "Could not solve hindsight-best team."}), 500

    # Captain bonus: best scorer doubles
    best_starters_pts = [p.get("actual", 0) or 0 for p in best_result["starters"]]
    best_captain_pts = max(best_starters_pts) if best_starters_pts else 0
    best_starting_actual = round(sum(best_starters_pts) + best_captain_pts, 1)
    for p in best_result["starters"]:
        if (p.get("actual", 0) or 0) == best_captain_pts and best_captain_pts > 0:
            p["is_captain"] = True
            p["multiplier"] = 2
            break

    my_ids = {p["player_id"] for p in my_starters}
    best_ids = {p["player_id"] for p in best_result["starters"]}
    overlap_ids = sorted(my_ids & best_ids)
    capture_pct = round(
        (my_starting_actual / best_starting_actual) * 100, 1,
    ) if best_starting_actual > 0 else 0

    return jsonify({
        "gameweek": gameweek,
        "budget": budget,
        "my_team": {
            "starters": scrub_nan(my_starters),
            "bench": scrub_nan(my_bench),
            "starting_actual": my_starting_actual,
        },
        "best_team": {
            "starters": best_result["starters"],
            "bench": best_result["bench"],
            "starting_actual": best_starting_actual,
        },
        "overlap_player_ids": overlap_ids,
        "overlap_count": len(overlap_ids),
        "capture_pct": capture_pct,
    })
