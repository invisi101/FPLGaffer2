"""Core blueprint — predictions, refresh, train, status, monsters, scores."""

import json

import pandas as pd
from flask import Blueprint, Response, jsonify, request

from src.api.helpers import (
    ELEMENT_TYPE_MAP,
    get_next_fixtures,
    get_next_gw,
    get_team_map,
    load_bootstrap,
    load_predictions_from_csv,
    safe_num,
    scrub_nan,
)
from src.api.sse import broadcast, create_sse_stream, pipeline_cache, pipeline_lock, run_in_background
from src.logging_config import get_logger
from src.paths import CACHE_DIR, OUTPUT_DIR

log = get_logger(__name__)

core_bp = Blueprint("core", __name__)


def _save_predictions(result: dict) -> None:
    """Save prediction results to CSV and detail JSON for later retrieval."""
    import json as _json

    players_df = result.get("players")
    if players_df is None or (hasattr(players_df, "empty") and players_df.empty):
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(players_df, "to_csv"):
        players_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    component_details = result.get("component_details", {})
    if component_details:
        detail = {
            "latest_gw": result.get("gameweek", 0),
            "players": {str(k): v for k, v in component_details.items()},
        }
        (OUTPUT_DIR / "predictions_detail.json").write_text(
            _json.dumps(detail), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@core_bp.route("/api/predictions")
def api_predictions():
    """Return player predictions with optional filters."""
    pred_df = load_predictions_from_csv()
    if pred_df is None or pred_df.empty:
        return jsonify({"error": "No predictions available. Train models first."}), 400

    bootstrap = load_bootstrap()
    team_map = get_team_map()

    # Normalise position column name (CSV may have position_clean)
    if "position_clean" in pred_df.columns and "position" not in pred_df.columns:
        pred_df["position"] = pred_df["position_clean"]

    # Enrich with bootstrap data (cost, form, points, team, fixtures)
    if bootstrap:
        el_map = {el["id"]: el for el in bootstrap.get("elements", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        for col in ("cost", "team_code", "player_form", "ep_next", "total_points",
                     "event_points", "ownership", "chance_of_playing"):
            if col not in pred_df.columns:
                pred_df[col] = None
        for idx, row in pred_df.iterrows():
            el = el_map.get(int(row["player_id"])) if pd.notna(row.get("player_id")) else None
            if el:
                if pd.isna(row.get("cost")) or row.get("cost") is None:
                    pred_df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
                if pd.isna(row.get("team_code")) or row.get("team_code") is None:
                    pred_df.at[idx, "team_code"] = id_to_code.get(el.get("team"))
                pred_df.at[idx, "player_form"] = safe_num(el.get("form", 0), 1)
                pred_df.at[idx, "ep_next"] = safe_num(el.get("ep_next", 0), 1)
                pred_df.at[idx, "total_points"] = el.get("total_points", 0)
                pred_df.at[idx, "event_points"] = el.get("event_points", 0)
                pred_df.at[idx, "ownership"] = safe_num(el.get("selected_by_percent", 0), 1)
                pred_df.at[idx, "chance_of_playing"] = el.get("chance_of_playing_next_round")

    # Enrich with team names
    if "team_code" in pred_df.columns:
        pred_df["team"] = pred_df["team_code"].map(team_map).fillna("")

    # Enrich with next fixtures and FDR
    fixture_map = get_next_fixtures(3)
    next_gw = get_next_gw(bootstrap)
    if fixture_map and "team_code" in pred_df.columns:
        pred_df["next_3_fixtures"] = pred_df["team_code"].map(
            lambda tc: ", ".join(fixture_map.get(tc, []))
        )
    if bootstrap and next_gw:
        fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
        if fixtures_path.exists():
            import json as _json
            all_fixtures = _json.loads(fixtures_path.read_text(encoding="utf-8"))
            id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
            code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}
            fdr_map = {}
            home_map = {}
            opp_map = {}
            for f in all_fixtures:
                if f.get("event") == next_gw:
                    h_code = id_to_code.get(f["team_h"])
                    a_code = id_to_code.get(f["team_a"])
                    if h_code:
                        fdr_map[h_code] = f.get("team_h_difficulty", 3)
                        home_map[h_code] = True
                        opp_map[h_code] = code_to_short.get(a_code, "")
                    if a_code:
                        fdr_map[a_code] = f.get("team_a_difficulty", 3)
                        home_map[a_code] = False
                        opp_map[a_code] = code_to_short.get(h_code, "")
            if "team_code" in pred_df.columns:
                pred_df["fdr"] = pred_df["team_code"].map(fdr_map)
                pred_df["is_home"] = pred_df["team_code"].map(home_map)
                pred_df["opponent"] = pred_df["team_code"].map(opp_map)

    # Apply filters
    position = request.args.get("position")
    if position:
        pred_df = pred_df[pred_df["position"] == position]

    search = request.args.get("search", "").lower()
    if search:
        pred_df = pred_df[pred_df["web_name"].str.lower().str.contains(search, na=False)]

    sort_by = request.args.get("sort_by", "predicted_next_gw_points")
    if sort_by in pred_df.columns:
        pred_df = pred_df.sort_values(sort_by, ascending=False)

    records = pred_df.head(500).to_dict(orient="records")

    return jsonify({
        "players": scrub_nan(records),
        "next_gw": next_gw,
        "count": len(records),
    })


# ---------------------------------------------------------------------------
# Refresh Data
# ---------------------------------------------------------------------------

@core_bp.route("/api/refresh-data", methods=["POST"])
def api_refresh_data():
    """Re-fetch data, rebuild predictions, check plan health."""
    def do_refresh():
        from src.data.loader import load_all_data
        from src.features.builder import build_features
        from src.ml.prediction import generate_predictions

        broadcast("Fetching latest data...", event="progress")
        data = load_all_data(force=True)

        broadcast("Building features...", event="progress")
        df = build_features(data)

        with pipeline_lock:
            pipeline_cache["df"] = df
            pipeline_cache["data"] = data

        broadcast("Generating predictions...", event="progress")
        result = generate_predictions(df, data)
        _save_predictions(result)

        broadcast("Data refresh complete.", event="progress")

    started = run_in_background("Refresh Data", do_refresh)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

@core_bp.route("/api/train", methods=["POST"])
def api_train():
    """Train all model tiers and generate predictions."""
    def do_train():
        from src.data.loader import load_all_data
        from src.features.builder import build_features
        from src.ml.prediction import generate_predictions
        from src.ml.training import train_all_models, train_all_quantile_models, train_all_sub_models

        broadcast("Loading data...", event="progress")
        data = load_all_data()

        broadcast("Building features...", event="progress")
        df = build_features(data)

        with pipeline_lock:
            pipeline_cache["df"] = df
            pipeline_cache["data"] = data

        broadcast("Training mean models...", event="progress")
        train_all_models(df)

        broadcast("Training quantile models...", event="progress")
        train_all_quantile_models(df)

        broadcast("Training decomposed sub-models...", event="progress")
        train_all_sub_models(df)

        broadcast("Generating predictions...", event="progress")
        result = generate_predictions(df, data)
        _save_predictions(result)

        broadcast("Training complete.", event="progress")

    started = run_in_background("Train Models", do_train)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# SSE Status Stream
# ---------------------------------------------------------------------------

@core_bp.route("/api/status")
def api_status():
    """SSE event stream for live progress."""
    return Response(create_sse_stream(), content_type="text/event-stream")


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

@core_bp.route("/api/model-info")
def api_model_info():
    """Return info about trained models and data status."""
    import time

    from src.paths import MODEL_DIR

    bootstrap = load_bootstrap()
    next_gw = get_next_gw(bootstrap)

    # Model files with exists flag
    expected_models = ["mean_GKP", "mean_DEF", "mean_MID", "mean_FWD",
                       "quantile_MID", "quantile_FWD"]
    models = []
    if MODEL_DIR.exists():
        existing = {p.stem for p in MODEL_DIR.glob("*.joblib")}
        for name in expected_models:
            models.append({
                "name": name,
                "exists": name in existing,
            })
    else:
        models = [{"name": n, "exists": False} for n in expected_models]

    # Cache age
    cache_age = None
    cache_max = 1800  # 30 minutes
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if bootstrap_path.exists():
        cache_age = int(time.time() - bootstrap_path.stat().st_mtime)

    # Season detection
    current_season = "2025-2026"
    available_seasons = ["2024-2025", "2025-2026"]

    return jsonify({
        "models": models,
        "next_gw": next_gw,
        "current_season": current_season,
        "available_seasons": available_seasons,
        "cache_age_seconds": cache_age,
        "cache_max_age_seconds": cache_max,
    })


# ---------------------------------------------------------------------------
# Monsters
# ---------------------------------------------------------------------------

@core_bp.route("/api/monsters")
def api_monsters():
    """Return top 3 players for each monster category."""
    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No data available. Refresh data first."})

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map = {t.get("code"): t.get("short_name", "") for t in bootstrap.get("teams", [])}

    rows = []
    for el in bootstrap.get("elements", []):
        if (el.get("minutes") or 0) < 90:
            continue
        rows.append({
            "player_id": el["id"],
            "web_name": el.get("web_name", "Unknown"),
            "position": pos_map.get(el.get("element_type"), ""),
            "team_code": el.get("team_code", 0),
            "cost": round((el.get("now_cost", 0) or 0) / 10, 1),
            "photo_code": el.get("code", 0),
            "total_points": el.get("total_points", 0) or 0,
            "goals": el.get("goals_scored", 0) or 0,
            "assists": el.get("assists", 0) or 0,
            "clean_sheets": el.get("clean_sheets", 0) or 0,
            "bonus": el.get("bonus", 0) or 0,
            "saves": el.get("saves", 0) or 0,
            "defcon": el.get("defensive_contribution", 0) or 0,
            "penalties_order": el.get("penalties_order") or 99,
            "corners_order": el.get("corners_and_indirect_freekicks_order") or 99,
            "fk_order": el.get("direct_freekicks_order") or 99,
        })
    if not rows:
        return jsonify({"error": "No player data available."})

    df = pd.DataFrame(rows)

    def _card(row, stat_value, stat_label):
        return {
            "player_id": int(row["player_id"]),
            "web_name": row["web_name"],
            "position": row["position"],
            "team_code": int(row["team_code"]),
            "team_short": team_map.get(int(row["team_code"]), ""),
            "cost": row["cost"],
            "photo_code": row["photo_code"],
            "total_pts": int(row["total_points"]),
            "stat_value": round(float(stat_value), 2) if isinstance(stat_value, float) else int(stat_value),
            "stat_label": stat_label,
        }

    categories = []

    # 1. Goal Monsters (MID + FWD)
    attackers = df[df["position"].isin(["MID", "FWD"])]
    top = attackers.nlargest(3, "goals")
    categories.append({
        "id": "goal_monsters", "title": "Goal Monsters", "emoji": "\u26bd",
        "subtitle": "Top scorers this season",
        "players": [_card(r, r["goals"], "Goals") for _, r in top.iterrows()],
    })

    # 2. DefCon Monsters (DEF + MID)
    def_mids = df[df["position"].isin(["DEF", "MID"])]
    top = def_mids.nlargest(3, "defcon")
    categories.append({
        "id": "defcon_monsters", "title": "DefCon Monsters", "emoji": "\U0001f6e1\ufe0f",
        "subtitle": "Defensive contribution machines",
        "players": [_card(r, r["defcon"], "DefCon Pts") for _, r in top.iterrows()],
    })

    # 3. Closet Strikers (DEF by goals)
    defs = df[df["position"] == "DEF"]
    top = defs.nlargest(3, "goals")
    categories.append({
        "id": "closet_strikers", "title": "Closet Strikers", "emoji": "\U0001f575\ufe0f",
        "subtitle": "Defenders who think they're forwards",
        "players": [_card(r, r["goals"], "Goals") for _, r in top.iterrows()],
    })

    # 4. Assist Kings
    top = df.nlargest(3, "assists")
    categories.append({
        "id": "assist_kings", "title": "Assist Kings", "emoji": "\U0001f451",
        "subtitle": "The creators and providers",
        "players": [_card(r, r["assists"], "Assists") for _, r in top.iterrows()],
    })

    # 5. Set Piece Merchants
    df_sp = df.copy()
    df_sp["_sp_score"] = (
        (df_sp["penalties_order"] <= 1).astype(int) * 3
        + (df_sp["corners_order"] <= 2).astype(int) * 2
        + (df_sp["fk_order"] <= 2).astype(int) * 2
    )
    df_sp["_sp_rank"] = df_sp["_sp_score"] + df_sp["total_points"] * 0.01
    top = df_sp[df_sp["_sp_score"] > 0].nlargest(3, "_sp_rank")
    if len(top) < 3:
        top = df_sp.nlargest(3, "_sp_rank")
    sp_players = []
    for _, r in top.iterrows():
        duties = []
        if r["penalties_order"] <= 1:
            duties.append("PEN")
        if r["corners_order"] <= 2:
            duties.append("CRN")
        if r["fk_order"] <= 2:
            duties.append("FK")
        sp_players.append(_card(r, r["_sp_score"], ", ".join(duties) if duties else "SET"))
    categories.append({
        "id": "set_piece_merchants", "title": "Set Piece Merchants", "emoji": "\U0001f3af",
        "subtitle": "Dead ball specialists", "players": sp_players,
    })

    # 6. Value Monsters
    df_val = df[df["cost"] > 0].copy()
    df_val["_value"] = df_val["total_points"] / df_val["cost"]
    top = df_val.nlargest(3, "_value")
    categories.append({
        "id": "value_monsters", "title": "Value Monsters", "emoji": "\U0001f4b0",
        "subtitle": "Best bang for your buck",
        "players": [_card(r, round(r["_value"], 2), "Pts/\u00a3m") for _, r in top.iterrows()],
    })

    # 7. Clean Sheet Machines
    cs_players = df[df["position"].isin(["DEF", "GKP"])]
    top = cs_players.nlargest(3, "clean_sheets")
    categories.append({
        "id": "clean_sheet_machines", "title": "Clean Sheet Machines", "emoji": "\U0001f9e4",
        "subtitle": "The brick walls",
        "players": [_card(r, r["clean_sheets"], "Clean Sheets") for _, r in top.iterrows()],
    })

    return jsonify({"categories": scrub_nan(categories)})


# ---------------------------------------------------------------------------
# PL Table
# ---------------------------------------------------------------------------

@core_bp.route("/api/pl-table")
def api_pl_table():
    """Compute Premier League standings from fixture results."""
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    bootstrap = load_bootstrap()
    if not fixtures_path.exists() or not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    team_info = {t["id"]: {"name": t["name"], "short": t["short_name"]} for t in bootstrap.get("teams", [])}

    stats = {t_id: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "form": []}
             for t_id in team_info}

    finished = sorted(
        [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None],
        key=lambda f: (f.get("event", 0), f.get("kickoff_time", "")),
    )

    for f in finished:
        h, a = f["team_h"], f["team_a"]
        hs, as_ = f["team_h_score"], f["team_a_score"]
        if h not in stats or a not in stats:
            continue
        stats[h]["p"] += 1
        stats[h]["gf"] += hs
        stats[h]["ga"] += as_
        stats[a]["p"] += 1
        stats[a]["gf"] += as_
        stats[a]["ga"] += hs
        if hs > as_:
            stats[h]["w"] += 1; stats[h]["pts"] += 3; stats[h]["form"].append("W")
            stats[a]["l"] += 1; stats[a]["form"].append("L")
        elif hs < as_:
            stats[a]["w"] += 1; stats[a]["pts"] += 3; stats[a]["form"].append("W")
            stats[h]["l"] += 1; stats[h]["form"].append("L")
        else:
            stats[h]["d"] += 1; stats[h]["pts"] += 1; stats[h]["form"].append("D")
            stats[a]["d"] += 1; stats[a]["pts"] += 1; stats[a]["form"].append("D")

    table = []
    for t_id, s in stats.items():
        info = team_info.get(t_id, {})
        gd = s["gf"] - s["ga"]
        table.append({
            "team_id": t_id, "team": info.get("name", ""), "short": info.get("short", ""),
            "p": s["p"], "w": s["w"], "d": s["d"], "l": s["l"],
            "gf": s["gf"], "ga": s["ga"], "gd": gd, "points": s["pts"],
            "form": s["form"][-5:],
        })
    table.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
    for i, row in enumerate(table):
        row["pos"] = i + 1

    return jsonify({"table": table})


# ---------------------------------------------------------------------------
# GW Scores
# ---------------------------------------------------------------------------

def _extract_stat(stats_list, stat_name, side):
    for stat in stats_list:
        if stat.get("identifier") == stat_name:
            return stat.get(side, [])
    return []


def _build_match_detail(f, team_info, element_map):
    h_id, a_id = f["team_h"], f["team_a"]
    h_info = team_info.get(h_id, {})
    a_info = team_info.get(a_id, {})
    stats_list = f.get("stats", [])

    match = {
        "home_team": h_info.get("name", ""), "home_short": h_info.get("short", ""),
        "away_team": a_info.get("name", ""), "away_short": a_info.get("short", ""),
        "home_score": f.get("team_h_score"), "away_score": f.get("team_a_score"),
        "kickoff": f.get("kickoff_time"),
        "finished": f.get("finished", False), "started": f.get("started", False),
    }

    for stat_key, output_key in [
        ("goals_scored", "goals"), ("assists", "assists"),
        ("yellow_cards", "yellow_cards"), ("red_cards", "red_cards"),
        ("bonus", "bonus"),
    ]:
        home_entries = _extract_stat(stats_list, stat_key, "h")
        away_entries = _extract_stat(stats_list, stat_key, "a")
        if stat_key == "bonus":
            match[output_key] = {
                "home": [{"player": element_map.get(e["element"], "?"), "points": e["value"]} for e in home_entries],
                "away": [{"player": element_map.get(e["element"], "?"), "points": e["value"]} for e in away_entries],
            }
        elif stat_key == "goals_scored":
            match[output_key] = {
                "home": [{"player": element_map.get(e["element"], "?"), "count": e["value"]} for e in home_entries],
                "away": [{"player": element_map.get(e["element"], "?"), "count": e["value"]} for e in away_entries],
            }
        else:
            match[output_key] = {
                "home": [{"player": element_map.get(e["element"], "?")} for e in home_entries],
                "away": [{"player": element_map.get(e["element"], "?")} for e in away_entries],
            }

    og_home = _extract_stat(stats_list, "own_goals", "h")
    og_away = _extract_stat(stats_list, "own_goals", "a")
    match["own_goals"] = {
        "home": [{"player": element_map.get(e["element"], "?")} for e in og_home],
        "away": [{"player": element_map.get(e["element"], "?")} for e in og_away],
    }
    return match


@core_bp.route("/api/gw-scores")
def api_gw_scores():
    """Match details for a specific gameweek."""
    gameweek = request.args.get("gameweek", type=int)
    if not gameweek:
        next_gw = get_next_gw()
        gameweek = (next_gw - 1) if next_gw and next_gw > 1 else 1

    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    bootstrap = load_bootstrap()
    if not fixtures_path.exists() or not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    team_info = {t["id"]: {"name": t["name"], "short": t["short_name"]} for t in bootstrap.get("teams", [])}
    element_map = {el["id"]: el.get("web_name", "Unknown") for el in bootstrap.get("elements", [])}

    gw_fixtures = sorted(
        [f for f in fixtures if f.get("event") == gameweek],
        key=lambda f: f.get("kickoff_time") or "",
    )
    matches = [_build_match_detail(f, team_info, element_map) for f in gw_fixtures]
    return jsonify({"matches": matches, "gameweek": gameweek})


# ---------------------------------------------------------------------------
# Team Form
# ---------------------------------------------------------------------------

@core_bp.route("/api/team-form")
def api_team_form():
    """Last 10 GWs of results for all 20 teams."""
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    bootstrap = load_bootstrap()
    if not fixtures_path.exists() or not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    team_info = {t["id"]: {"name": t["name"], "short": t["short_name"], "code": t.get("code")}
                 for t in bootstrap.get("teams", [])}
    element_map = {el["id"]: el.get("web_name", "Unknown") for el in bootstrap.get("elements", [])}

    finished_gws = sorted({f["event"] for f in fixtures if f.get("finished") and f.get("event")})
    gw_set = set(finished_gws)
    last_10_gws = set(finished_gws[-10:]) if len(finished_gws) > 10 else set(finished_gws)

    teams = {}
    for t_id, t_info in team_info.items():
        teams[t_id] = {
            "team_id": t_id, "name": t_info["name"], "short": t_info["short"],
            "code": t_info.get("code"), "form_points": 0,
            "results": {str(gw): [] for gw in finished_gws},
        }

    for f in fixtures:
        gw = f.get("event")
        if not gw or gw not in gw_set or not f.get("finished"):
            continue
        h_id, a_id = f["team_h"], f["team_a"]
        h_score = f.get("team_h_score", 0) or 0
        a_score = f.get("team_a_score", 0) or 0
        match_detail = _build_match_detail(f, team_info, element_map)

        if h_score > a_score:
            h_result, a_result = "W", "L"
        elif h_score < a_score:
            h_result, a_result = "L", "W"
        else:
            h_result, a_result = "D", "D"

        gw_key = str(gw)
        for t_id, is_home, score_for, score_against, result, opp_id in [
            (h_id, True, h_score, a_score, h_result, a_id),
            (a_id, False, a_score, h_score, a_result, h_id),
        ]:
            if t_id not in teams:
                continue
            teams[t_id]["results"][gw_key].append({
                "fixture_id": f.get("id"),
                "opponent_short": team_info.get(opp_id, {}).get("short", "?"),
                "is_home": is_home, "score_for": score_for,
                "score_against": score_against, "result": result,
                "match_detail": match_detail,
            })
            if gw in last_10_gws:
                teams[t_id]["form_points"] += 3 if result == "W" else (1 if result == "D" else 0)

    return jsonify({
        "gw_columns": finished_gws,
        "teams": sorted(teams.values(), key=lambda t: -t["form_points"]),
    })


# ---------------------------------------------------------------------------
# Players - Teams & Detail
# ---------------------------------------------------------------------------

@core_bp.route("/api/players/teams")
def api_players_teams():
    """Return all 20 teams with players grouped by position."""
    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_info = {}
    for t in bootstrap.get("teams", []):
        team_info[t["id"]] = {
            "id": t["id"], "code": t.get("code"), "name": t["name"],
            "short_name": t["short_name"],
            "players": {"GKP": [], "DEF": [], "MID": [], "FWD": []},
            "player_count": 0,
        }

    for el in bootstrap.get("elements", []):
        tid = el.get("team")
        if tid not in team_info:
            continue
        pos = pos_map.get(el.get("element_type"), "MID")
        team_info[tid]["players"][pos].append({
            "id": el["id"], "code": el.get("code"),
            "web_name": el.get("web_name", ""),
            "first_name": el.get("first_name", ""),
            "second_name": el.get("second_name", ""),
            "position": pos,
            "now_cost": round(el.get("now_cost", 0) / 10, 1),
            "total_points": el.get("total_points", 0),
            "goals_scored": el.get("goals_scored", 0),
            "assists": el.get("assists", 0),
            "clean_sheets": el.get("clean_sheets", 0),
            "minutes": el.get("minutes", 0),
            "bonus": el.get("bonus", 0),
            "form": el.get("form", "0.0"),
            "points_per_game": el.get("points_per_game", "0.0"),
            "selected_by_percent": el.get("selected_by_percent", "0.0"),
            "status": el.get("status", "a"),
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            "news": el.get("news", ""),
            "event_points": el.get("event_points", 0),
            "starts": el.get("starts", 0),
        })
        team_info[tid]["player_count"] += 1

    for t in team_info.values():
        for pos in t["players"]:
            t["players"][pos].sort(key=lambda p: -p["total_points"])

    teams_list = sorted(team_info.values(), key=lambda t: t["name"])
    return jsonify({"teams": teams_list})


@core_bp.route("/api/players/<int:player_id>/detail")
def api_player_detail(player_id):
    """Comprehensive player detail: info, per-GW history, fixtures, predictions."""
    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No cached data. Refresh data first."}), 400

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map = {t["id"]: t for t in bootstrap.get("teams", [])}

    player_el = None
    for el in bootstrap.get("elements", []):
        if el["id"] == player_id:
            player_el = el
            break
    if not player_el:
        return jsonify({"error": f"Player {player_id} not found."}), 404

    team = team_map.get(player_el.get("team"), {})
    player_info = {
        "id": player_el["id"], "code": player_el.get("code"),
        "web_name": player_el.get("web_name", ""),
        "first_name": player_el.get("first_name", ""),
        "second_name": player_el.get("second_name", ""),
        "position": pos_map.get(player_el.get("element_type"), "MID"),
        "team_id": player_el.get("team"),
        "team_name": team.get("name", ""), "team_short": team.get("short_name", ""),
        "team_code": team.get("code"),
        "now_cost": round(player_el.get("now_cost", 0) / 10, 1),
        "total_points": player_el.get("total_points", 0),
        "goals_scored": player_el.get("goals_scored", 0),
        "assists": player_el.get("assists", 0),
        "clean_sheets": player_el.get("clean_sheets", 0),
        "minutes": player_el.get("minutes", 0),
        "bonus": player_el.get("bonus", 0),
        "form": player_el.get("form", "0.0"),
        "status": player_el.get("status", "a"),
        "chance_of_playing_next_round": player_el.get("chance_of_playing_next_round"),
        "news": player_el.get("news", ""),
        "expected_goals": player_el.get("expected_goals", "0.00"),
        "expected_assists": player_el.get("expected_assists", "0.00"),
        "ict_index": player_el.get("ict_index", "0.0"),
        "selected_by_percent": player_el.get("selected_by_percent", "0.0"),
    }

    # Per-GW history from element-summary API
    gw_history = []
    try:
        from src.data.fpl_api import fetch_player_summary
        summary = fetch_player_summary(player_id)
        for h in summary.get("history", []):
            gw_history.append({
                "gw": h.get("round"),
                "event_points": h.get("total_points", 0),
                "minutes": h.get("minutes", 0),
                "goals_scored": h.get("goals_scored", 0),
                "assists": h.get("assists", 0),
                "clean_sheets": h.get("clean_sheets", 0),
                "goals_conceded": h.get("goals_conceded", 0),
                "bonus": h.get("bonus", 0),
                "bps": h.get("bps", 0),
                "saves": h.get("saves", 0),
                "starts": h.get("starts", 0),
                "opponent_team": h.get("opponent_team"),
                "was_home": h.get("was_home"),
                "xg": safe_num(h.get("expected_goals", 0), 2),
                "xa": safe_num(h.get("expected_assists", 0), 2),
            })
    except Exception:
        pass

    # Upcoming fixtures
    upcoming = []
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    if fixtures_path.exists():
        try:
            fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
            player_team_id = player_info["team_id"]
            unfinished = sorted(
                [f for f in fixtures if not f.get("finished") and f.get("event")
                 and (f["team_h"] == player_team_id or f["team_a"] == player_team_id)],
                key=lambda f: f["event"],
            )
            for f in unfinished[:8]:
                if f["team_h"] == player_team_id:
                    opp_team = team_map.get(f["team_a"], {})
                    is_home = True
                    fdr = f.get("team_h_difficulty", 3)
                else:
                    opp_team = team_map.get(f["team_h"], {})
                    is_home = False
                    fdr = f.get("team_a_difficulty", 3)
                upcoming.append({
                    "gw": f["event"], "opponent_short": opp_team.get("short_name", "?"),
                    "is_home": is_home, "fdr": fdr,
                })
        except Exception:
            pass

    # Predictions
    predictions = {}
    pred_df = load_predictions_from_csv()
    if pred_df is not None:
        p_row = pred_df[pred_df["player_id"] == player_id]
        if not p_row.empty:
            pr = p_row.iloc[0]
            predictions = {
                "predicted_next_gw_points": safe_num(pr.get("predicted_next_gw_points", 0), 2),
                "captain_score": safe_num(pr.get("captain_score", 0), 2),
                "predicted_next_3gw_points": safe_num(pr.get("predicted_next_3gw_points", 0), 2),
                "prediction_low": safe_num(pr.get("prediction_low"), 2),
                "prediction_high": safe_num(pr.get("prediction_high"), 2),
                "q80": safe_num(pr.get("predicted_next_gw_points_q80"), 2),
            }

    # Price history
    price_history = []
    try:
        from src.data.fpl_api import fetch_player_summary
        summary = fetch_player_summary(player_id)
        price_by_gw = {}
        for h in summary.get("history", []):
            gw = h.get("round")
            price_by_gw[gw] = round(h.get("value", 0) / 10, 1)
        for gw in sorted(price_by_gw):
            price_history.append({"gw": gw, "price": price_by_gw[gw]})
    except Exception:
        pass

    return jsonify({
        "player": player_info,
        "gw_history": gw_history,
        "upcoming_fixtures": upcoming,
        "predictions": predictions,
        "price_history": price_history,
    })


# ---------------------------------------------------------------------------
# Player Explain ("but, how?")
# ---------------------------------------------------------------------------

@core_bp.route("/api/players/<int:player_id>/explain")
def api_player_explain(player_id):
    """Return a full prediction breakdown for the explainer page."""
    from src.config import FEATURE_LABELS, ensemble as ens_cfg
    from src.ml.model_store import load_model as _load_model

    bootstrap = load_bootstrap()
    if not bootstrap:
        return jsonify({"error": "No data available. Refresh data first."}), 400

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map_full = {t["id"]: t for t in bootstrap.get("teams", [])}
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

    # Find the player in bootstrap
    player_el = None
    for el in bootstrap.get("elements", []):
        if el["id"] == player_id:
            player_el = el
            break
    if not player_el:
        return jsonify({"error": f"Player {player_id} not found."}), 404

    position = pos_map.get(player_el.get("element_type"), "MID")
    team = team_map_full.get(player_el.get("team"), {})
    web_name = player_el.get("web_name", "Unknown")

    # Load component details from predictions_detail.json
    detail_path = OUTPUT_DIR / "predictions_detail.json"
    components = {}
    if detail_path.exists():
        try:
            raw = json.loads(detail_path.read_text(encoding="utf-8"))
            components = raw.get("players", {}).get(str(player_id), {})
        except Exception:
            pass

    # Load predictions CSV for the player's final prediction
    pred_df = load_predictions_from_csv()
    final_pred = 0
    captain_score = 0
    pred_low = None
    pred_high = None
    if pred_df is not None:
        p_row = pred_df[pred_df["player_id"] == player_id]
        if not p_row.empty:
            pr = p_row.iloc[0]
            final_pred = safe_num(pr.get("predicted_next_gw_points", 0), 2)
            captain_score = safe_num(pr.get("captain_score", 0), 2)
            pred_low = safe_num(pr.get("prediction_low"), 2)
            pred_high = safe_num(pr.get("prediction_high"), 2)

    # Build mean/decomposed model breakdown
    mean_pred = components.get("mean_pred", final_pred)
    decomp_pred = components.get("decomp_pred", 0)
    ensemble_pred = components.get("ensemble_pred", final_pred)
    w_d = ens_cfg.decomposed_weight
    w_m = 1 - w_d

    mean_model_info = {"prediction": round(float(mean_pred), 2), "weight": w_m}
    decomp_model_info = {
        "prediction": round(float(decomp_pred), 2),
        "weight": w_d,
        "components": {},
        "p_plays": round(float(components.get("p_plays", 0)), 2),
        "p_60plus": round(float(components.get("p_60plus", 0)), 2),
    }

    # Build component breakdown
    comp_names = {
        "appearance": "Appearance",
        "goals": "Goals",
        "assists": "Assists",
        "cs": "Clean Sheets",
        "gc": "Goals Conceded",
        "saves": "Saves",
        "bonus": "Bonus",
        "defcon": "DefCon",
    }
    for comp_key, label in comp_names.items():
        sub_val = components.get(f"sub_{comp_key}", 0)
        pts_val = components.get(f"pts_{comp_key}", 0)
        if sub_val or pts_val:
            decomp_model_info["components"][comp_key] = {
                "label": label,
                "raw": round(float(sub_val), 4),
                "pts": round(float(pts_val), 2),
            }

    # Captain info
    q80_val = components.get("q80_pred")
    captain_info = {
        "score": round(float(captain_score), 2),
        "q80_prediction": round(float(q80_val), 2) if q80_val else None,
        "formula": (
            f"{ens_cfg.captain_mean_weight} x {round(float(mean_pred), 2)} + "
            f"{ens_cfg.captain_q80_weight} x {round(float(q80_val), 2)}"
        ) if q80_val else None,
    }

    # Confidence interval
    confidence = {
        "low": pred_low,
        "high": pred_high,
    }

    # Top features from the mean model
    top_features = []
    model_dict = _load_model(position, "next_gw_points")
    if model_dict is not None:
        model = model_dict["model"]
        features = model_dict["features"]
        importances = model.feature_importances_

        # Get the player's current feature values from predictions_detail.json
        feature_values = components.get("feature_values", {})

        # Sort by importance and take top 10
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        for idx in sorted_idx[:10]:
            fname = features[idx]
            top_features.append({
                "name": fname,
                "label": FEATURE_LABELS.get(fname, fname.replace("_", " ").title()),
                "importance": round(float(importances[idx]), 4),
                "value": feature_values.get(fname),
            })

    # Fixture info
    fixture_info = {}
    next_gw = get_next_gw(bootstrap)
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    if fixtures_path.exists() and next_gw:
        try:
            all_fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
            player_team_id = player_el.get("team")
            for f in all_fixtures:
                if f.get("event") == next_gw:
                    if f["team_h"] == player_team_id:
                        opp_code = id_to_code.get(f["team_a"])
                        fixture_info = {
                            "opponent": code_to_short.get(opp_code, "?"),
                            "is_home": True,
                            "fdr": f.get("team_h_difficulty", 3),
                            "is_dgw": False,
                        }
                        break
                    elif f["team_a"] == player_team_id:
                        opp_code = id_to_code.get(f["team_h"])
                        fixture_info = {
                            "opponent": code_to_short.get(opp_code, "?"),
                            "is_home": False,
                            "fdr": f.get("team_a_difficulty", 3),
                            "is_dgw": False,
                        }
                        break
            # Check for DGW
            dgw_count = sum(
                1 for f in all_fixtures
                if f.get("event") == next_gw
                and (f["team_h"] == player_team_id or f["team_a"] == player_team_id)
            )
            if dgw_count > 1:
                fixture_info["is_dgw"] = True
        except Exception:
            pass

    return jsonify(scrub_nan({
        "player_id": player_id,
        "web_name": web_name,
        "position": position,
        "team_code": team.get("code"),
        "team_short": team.get("short_name", ""),
        "photo_code": player_el.get("code", 0),
        "final_prediction": round(float(final_pred), 2),
        "mean_model": mean_model_info,
        "decomposed_model": decomp_model_info,
        "captain": captain_info,
        "confidence": confidence,
        "top_features": top_features,
        "fixture": fixture_info,
    }))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@core_bp.route("/")
def index():
    """Serve the single-page frontend."""
    from flask import render_template
    return render_template("index.html")
