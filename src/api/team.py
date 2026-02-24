"""Team blueprint — best-team, my-team, transfer-recommendations."""

import json

import pandas as pd
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
    resolve_current_squad_event,
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

    # Normalise position column name (CSV may have position_clean)
    if "position_clean" in pred_df.columns and "position" not in pred_df.columns:
        pred_df["position"] = pred_df["position_clean"]

    # Enrich with bootstrap data (cost, team_code) when missing from CSV
    bootstrap = load_bootstrap()
    if bootstrap:
        el_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        if "cost" not in pred_df.columns:
            pred_df["cost"] = None
        if "team_code" not in pred_df.columns:
            pred_df["team_code"] = None
        for idx, row in pred_df.iterrows():
            el = el_map.get(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
            if el:
                if pd.isna(row.get("cost")) or row.get("cost") is None:
                    pred_df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
                if pd.isna(row.get("team_code")) or row.get("team_code") is None:
                    pred_df.at[idx, "team_code"] = id_to_code.get(el.get("team"))

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

    # Compute both GW and 3-GW totals for the summary panel
    players = result.get("players", result["starters"] + result["bench"])
    starters = [p for p in players if p.get("starter")]
    captain_id = result.get("captain_id")

    def _sum_xi(col):
        return round(sum(
            (p.get(col) or 0) * (2 if p.get("player_id") == captain_id else 1)
            for p in starters
        ), 1)

    return jsonify(scrub_nan({
        "players": players,
        "starters": result["starters"],
        "bench": result["bench"],
        "total_cost": result["total_cost"],
        "starting_points": result["starting_points"],
        "starting_gw_points": _sum_xi("predicted_next_gw_points"),
        "starting_gw3_points": _sum_xi("predicted_next_3gw_points"),
        "remaining": round(budget - result["total_cost"], 1),
        "captain_id": captain_id,
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
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch history: {exc}"}), 404

    # Detect Free Hit reversion: after FH, squad reverts to pre-FH state
    squad_event, fh_reverted = resolve_current_squad_event(history, current_event)

    # Always fetch actual GW picks (what was played) for the actual pitch
    try:
        actual_picks_data = fetch_manager_picks(manager_id, current_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks: {exc}"}), 404

    # For planning: use reverted (pre-FH) picks if FH was played
    if fh_reverted:
        try:
            planning_picks_data = fetch_manager_picks(manager_id, squad_event)
        except Exception as exc:
            return jsonify({"error": f"Could not fetch pre-FH picks: {exc}"}), 404
    else:
        planning_picks_data = actual_picks_data

    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    team_map = get_team_map()
    free_transfers = calculate_free_transfers(history)

    # Budget from planning picks (reverted state after FH)
    planning_entry_history = planning_picks_data.get("entry_history", {})
    bank = planning_entry_history.get("bank", 0) / 10

    # Build fixture/FDR/opponent maps for the next GW
    next_gw = get_next_gw(bootstrap)
    fdr_map = {}
    home_map = {}
    opp_map = {}
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    if next_gw and fixtures_path.exists():
        all_fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}
        for f in all_fixtures:
            if f.get("event") == next_gw:
                h_code = team_id_to_code.get(f["team_h"])
                a_code = team_id_to_code.get(f["team_a"])
                if h_code:
                    fdr_map[h_code] = f.get("team_h_difficulty", 3)
                    home_map[h_code] = True
                    opp_map[h_code] = code_to_short.get(a_code, "")
                if a_code:
                    fdr_map[a_code] = f.get("team_a_difficulty", 3)
                    home_map[a_code] = False
                    opp_map[a_code] = code_to_short.get(h_code, "")

    def _build_squad(picks_data):
        """Build squad list from picks data."""
        result = []
        for pick in picks_data.get("picks", []):
            eid = pick.get("element")
            el = elements_map.get(eid, {})
            tid = el.get("team")
            tc = team_id_to_code.get(tid)
            multiplier = pick.get("multiplier", 1)
            raw_pts = el.get("event_points", 0)
            result.append({
                "player_id": eid,
                "web_name": el.get("web_name", "Unknown"),
                "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                "team_code": tc,
                "team": team_map.get(tc, ""),
                "cost": round(el.get("now_cost", 0) / 10, 1),
                "total_points": el.get("total_points", 0),
                "event_points_raw": raw_pts,
                "event_points": raw_pts * multiplier,
                "starter": pick.get("position", 12) <= 11,
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "multiplier": multiplier,
                "status": el.get("status", "a"),
                "news": el.get("news", ""),
                "chance_of_playing": el.get("chance_of_playing_next_round"),
                "fdr": fdr_map.get(tc),
                "opponent": opp_map.get(tc, ""),
            })
        return result

    # Actual squad: what was played in current_event (FH team if FH was used)
    squad = _build_squad(actual_picks_data)

    # Planning squad: reverted team for optimization (pre-FH if FH was used)
    if fh_reverted:
        planning_squad = _build_squad(planning_picks_data)
    else:
        planning_squad = squad

    # Enrich both squads with predictions
    pred_df = load_predictions_from_csv()
    if pred_df is not None and not pred_df.empty:
        pred_map = {int(row["player_id"]): row.to_dict() for _, row in pred_df.iterrows()}
        for s in ([squad] if not fh_reverted else [squad, planning_squad]):
            for p in s:
                pr = pred_map.get(p["player_id"], {})
                p["predicted_next_gw_points"] = safe_num(pr.get("predicted_next_gw_points", 0), 2)
                p["captain_score"] = safe_num(pr.get("captain_score", 0), 2)
                p["predicted_next_3gw_points"] = safe_num(pr.get("predicted_next_3gw_points", 0), 2)
                p["predicted_next_gw_points_q80"] = safe_num(pr.get("predicted_next_gw_points_q80", 0), 2)

    # Optimized XI from the planning squad (reverted team after FH)
    optimized = optimize_starting_xi(planning_squad)

    # Computed aggregates the frontend expects
    squad_value = round(sum(p["cost"] for p in planning_squad), 1)
    sell_value = round(planning_entry_history.get("value", 0) / 10, 1)
    active_chip = actual_picks_data.get("active_chip")

    xi_actual_gw = sum(
        p["event_points"] for p in squad if p["starter"]
    )
    xi_pred_gw = round(sum(
        p.get("predicted_next_gw_points", 0) * (2 if p["is_captain"] else 1)
        for p in optimized if p["starter"]
    ), 1)
    xi_pred_3gw = round(sum(
        p.get("predicted_next_3gw_points", 0) * (2 if p["is_captain"] else 1)
        for p in optimized if p["starter"]
    ), 1)

    resp = {
        "squad": squad,
        "optimized_squad": optimized,
        "bank": bank,
        "free_transfers": free_transfers,
        "manager": {
            "name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
            "team_name": entry.get("name", ""),
            "overall_points": entry.get("summary_overall_points", 0),
            "overall_rank": entry.get("summary_overall_rank", 0),
        },
        "next_gw": get_next_gw(),
        "current_event": current_event,
        "squad_value": squad_value,
        "sell_value": sell_value,
        "xi_actual_gw": xi_actual_gw,
        "xi_pred_gw": xi_pred_gw,
        "xi_pred_3gw": xi_pred_3gw,
        "active_chip": active_chip,
    }
    if fh_reverted:
        resp["fh_reverted"] = True
        resp["fh_event"] = current_event

    return jsonify(scrub_nan(resp))


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
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch history: {exc}"}), 404

    # Detect Free Hit reversion: after FH, squad reverts to pre-FH state
    squad_event, _fh_reverted = resolve_current_squad_event(history, current_event)

    try:
        picks_data = fetch_manager_picks(manager_id, squad_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks: {exc}"}), 404

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

    # Normalise position column name (CSV may have position_clean)
    if "position_clean" in pred_df.columns and "position" not in pred_df.columns:
        pred_df["position"] = pred_df["position_clean"]

    # Enrich with bootstrap data (cost, team_code) when missing from CSV
    if bootstrap:
        el_map_enrich = {el["id"]: el for el in bootstrap.get("elements", [])}
        if "cost" not in pred_df.columns:
            pred_df["cost"] = None
        if "team_code" not in pred_df.columns:
            pred_df["team_code"] = None
        for idx, row in pred_df.iterrows():
            el = el_map_enrich.get(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
            if el:
                if pd.isna(row.get("cost")) or row.get("cost") is None:
                    pred_df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
                if pd.isna(row.get("team_code")) or row.get("team_code") is None:
                    pred_df.at[idx, "team_code"] = team_id_to_code.get(el.get("team"))

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
