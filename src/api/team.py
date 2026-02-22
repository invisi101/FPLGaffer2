"""Team blueprint — best-team, my-team, transfer-recommendations."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import (
    ELEMENT_TYPE_MAP,
    calculate_free_transfers,
    get_next_fixtures,
    get_next_gw,
    get_team_map,
    load_bootstrap,
    load_predictions_from_csv,
    optimize_starting_xi,
    safe_num,
    scrub_nan,
)
from src.logging_config import get_logger
from src.paths import CACHE_DIR

log = get_logger(__name__)

team_bp = Blueprint("team", __name__)


# ---------------------------------------------------------------------------
# Best Team
# ---------------------------------------------------------------------------

@team_bp.route("/api/best-team", methods=["POST"])
def api_best_team():
    """MILP optimal squad from scratch."""
    from src.solver.squad import solve_milp_team

    body = request.get_json(silent=True) or {}
    target = body.get("target", "predicted_next_gw_points")
    budget = body.get("budget", 100.0)

    pred_df = load_predictions_from_csv()
    if pred_df is None or pred_df.empty:
        return jsonify({"error": "No predictions available. Train models first."}), 400

    pool = pred_df.dropna(subset=["position", "cost", target]).copy()

    team_map = get_team_map()
    if "team_code" in pool.columns:
        pool["team"] = pool["team_code"].map(team_map).fillna("")

    fixture_map = get_next_fixtures(3)
    if fixture_map and "team_code" in pool.columns:
        pool["opponent"] = pool["team_code"].map(
            lambda tc: fixture_map.get(tc, [""])[0] if fixture_map.get(tc) else ""
        )
        pool["next_3_fixtures"] = pool["team_code"].map(
            lambda tc: ", ".join(fixture_map.get(tc, []))
        )

    captain_col = "captain_score" if "captain_score" in pool.columns else None
    result = solve_milp_team(pool, target, budget=budget, captain_col=captain_col)
    if result is None:
        return jsonify({"error": "Could not find a valid team."}), 400

    return jsonify(scrub_nan({
        "starters": result["starters"],
        "bench": result["bench"],
        "total_cost": result["total_cost"],
        "starting_points": result["starting_points"],
        "captain_id": result.get("captain_id"),
        "target": target,
        "next_gw": get_next_gw(),
    }))


# ---------------------------------------------------------------------------
# My Team
# ---------------------------------------------------------------------------

@team_bp.route("/api/my-team")
def api_my_team():
    """Import manager's FPL squad with predictions."""
    from src.data.fpl_api import fetch_manager_entry, fetch_manager_history, fetch_manager_picks

    manager_id = request.args.get("manager_id", type=int)
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400

    try:
        entry = fetch_manager_entry(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch manager {manager_id}: {exc}"}), 404

    current_event = entry.get("current_event")
    if not current_event:
        return jsonify({"error": "Manager has no current event."}), 400

    try:
        picks_data = fetch_manager_picks(manager_id, current_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks: {exc}"}), 404

    try:
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch history: {exc}"}), 404

    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    team_map = get_team_map()
    free_transfers = calculate_free_transfers(history)

    entry_history = picks_data.get("entry_history", {})
    bank = entry_history.get("bank", 0) / 10
    squad = []

    for pick in picks_data.get("picks", []):
        eid = pick.get("element")
        el = elements_map.get(eid, {})
        tid = el.get("team")
        tc = team_id_to_code.get(tid)
        squad.append({
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
            "team_code": tc,
            "team": team_map.get(tc, ""),
            "cost": round(el.get("now_cost", 0) / 10, 1),
            "total_points": el.get("total_points", 0),
            "event_points": el.get("event_points", 0),
            "starter": pick.get("position", 12) <= 11,
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": pick.get("multiplier", 1),
        })

    # Enrich with predictions
    pred_df = load_predictions_from_csv()
    if pred_df is not None and not pred_df.empty:
        pred_map = {int(row["player_id"]): row.to_dict() for _, row in pred_df.iterrows()}
        for p in squad:
            pr = pred_map.get(p["player_id"], {})
            p["predicted_next_gw_points"] = safe_num(pr.get("predicted_next_gw_points", 0), 2)
            p["captain_score"] = safe_num(pr.get("captain_score", 0), 2)
            p["predicted_next_3gw_points"] = safe_num(pr.get("predicted_next_3gw_points", 0), 2)

    # Optimized XI
    optimized = optimize_starting_xi(squad)

    return jsonify(scrub_nan({
        "squad": squad,
        "optimized": optimized,
        "bank": bank,
        "free_transfers": free_transfers,
        "manager": {
            "name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
            "team_name": entry.get("name", ""),
            "overall_points": entry.get("summary_overall_points", 0),
            "overall_rank": entry.get("summary_overall_rank", 0),
        },
        "next_gw": get_next_gw(),
    }))


# ---------------------------------------------------------------------------
# Transfer Recommendations
# ---------------------------------------------------------------------------

@team_bp.route("/api/transfer-recommendations", methods=["POST"])
def api_transfer_recommendations():
    """MILP transfer solver with hit-aware wrapper."""
    from src.data.fpl_api import fetch_manager_entry, fetch_manager_history, fetch_manager_picks
    from src.solver.squad import solve_milp_team
    from src.solver.transfers import solve_transfer_milp, solve_transfer_milp_with_hits

    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400

    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    max_transfers = body.get("max_transfers", 4)
    wildcard = body.get("wildcard", False)
    target = body.get("target", "predicted_next_gw_points")

    try:
        entry = fetch_manager_entry(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch manager {manager_id}: {exc}"}), 404

    current_event = entry.get("current_event")
    if not current_event:
        return jsonify({"error": "Manager has no current event."}), 400

    try:
        picks_data = fetch_manager_picks(manager_id, current_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks: {exc}"}), 404

    try:
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch history: {exc}"}), 404

    free_transfers = calculate_free_transfers(history)

    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    team_map = get_team_map()

    entry_history = picks_data.get("entry_history", {})
    bank = entry_history.get("bank", 0) / 10

    picks = picks_data.get("picks", [])
    current_squad_ids = set()
    current_squad_map = {}
    current_squad_cost = 0.0
    for pick in picks:
        eid = pick.get("element")
        current_squad_ids.add(eid)
        el = elements_map.get(eid, {})
        tid = el.get("team")
        tc = team_id_to_code.get(tid)
        player_cost = el.get("now_cost", 0) / 10
        current_squad_cost += player_cost
        current_squad_map[eid] = {
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
            "team_code": tc,
            "cost": round(player_cost, 1),
            "starter": pick.get("position", 12) <= 11,
        }

    total_budget = round(bank + current_squad_cost, 1)

    pred_df = load_predictions_from_csv()
    if pred_df is None or pred_df.empty:
        return jsonify({"error": "No predictions available. Train models first."}), 400

    pool = pred_df.dropna(subset=["position", "cost", target]).copy()

    if "team_code" in pool.columns:
        pool["team"] = pool["team_code"].map(team_map).fillna("")

    fixture_map = get_next_fixtures(3)
    if fixture_map and "team_code" in pool.columns:
        pool["opponent"] = pool["team_code"].map(
            lambda tc: fixture_map.get(tc, [""])[0] if fixture_map.get(tc) else ""
        )
        pool["next_3_fixtures"] = pool["team_code"].map(
            lambda tc: ", ".join(fixture_map.get(tc, []))
        )

    # Current XI points before transfers
    pred_map = {int(row["player_id"]): row.to_dict() for _, row in pred_df.iterrows()}
    current_starters = [p for p in current_squad_map.values() if p["starter"]]
    current_xi_points = round(
        sum(pred_map.get(p["player_id"], {}).get(target, 0) or 0 for p in current_starters), 2
    )
    for pick in picks:
        if pick.get("is_captain"):
            cap_pred = pred_map.get(pick["element"], {}).get(target, 0) or 0
            current_xi_points = round(current_xi_points + cap_pred, 2)
            break

    # Solve
    captain_col = "captain_score" if "captain_score" in pool.columns else None
    if wildcard:
        result = solve_transfer_milp(
            pool, current_squad_ids, target,
            budget=total_budget, max_transfers=max_transfers,
            captain_col=captain_col,
        )
    else:
        result = solve_transfer_milp_with_hits(
            pool, current_squad_ids, target,
            budget=total_budget, free_transfers=free_transfers,
            max_transfers=max_transfers,
            captain_col=captain_col,
        )
    if result is None:
        return jsonify({"error": "Could not find a valid transfer solution."}), 400

    # Build transfer lists
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    out_list = sorted(result["transfers_out_ids"])
    in_list = sorted(result["transfers_in_ids"])
    new_squad_map = {p["player_id"]: p for p in result["players"]}

    out_players = []
    for pid in out_list:
        info = current_squad_map.get(pid, {}).copy()
        info["team"] = team_map.get(info.get("team_code"), "")
        info[target] = pred_map.get(pid, {}).get(target)
        out_players.append(info)

    in_players = [new_squad_map.get(pid, {}) for pid in in_list]

    out_players.sort(key=lambda p: (pos_order.get(p.get("position"), 9), -(p.get("cost") or 0)))
    in_players.sort(key=lambda p: (pos_order.get(p.get("position"), 9), -(p.get("cost") or 0)))

    for i, out_p in enumerate(out_players):
        in_p = in_players[i] if i < len(in_players) else {}
        out_p["replaced_by"] = in_p.get("web_name", "?")

    for i, in_p in enumerate(in_players):
        out_p = out_players[i] if i < len(out_players) else {}
        in_p["replaces"] = out_p.get("web_name", "?")

    n_transfers = len(in_list)
    points_hit = result.get("hit_cost", max(0, n_transfers - free_transfers) * 4)
    points_gained = round(result["starting_points"] - current_xi_points, 2)
    net_gain = round(points_gained - points_hit, 2)

    return jsonify(scrub_nan({
        "transfers_in": in_players,
        "transfers_out": out_players,
        "new_squad": {"starters": result["starters"], "bench": result["bench"]},
        "current_xi_points": current_xi_points,
        "new_xi_points": result["starting_points"],
        "points_gained": points_gained,
        "free_transfers": free_transfers,
        "n_transfers": n_transfers,
        "points_hit": points_hit,
        "net_gain": net_gain,
        "budget_before": total_budget,
        "budget_after": result["total_cost"],
        "bank_after": round(total_budget - result["total_cost"], 1),
        "target": target,
        "next_gw": get_next_gw(),
        "wildcard": wildcard,
        "transfers_in_ids": list(result["transfers_in_ids"]),
    }))
