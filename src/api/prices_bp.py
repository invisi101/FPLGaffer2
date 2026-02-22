"""Prices blueprint — price tracking, predictions, watchlist."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import load_bootstrap, require_manager_id
from src.logging_config import get_logger

log = get_logger(__name__)

prices_bp = Blueprint("prices", __name__)


def _get_mgr():
    from src.season.manager import SeasonManager
    if not hasattr(_get_mgr, "_mgr"):
        _get_mgr._mgr = SeasonManager()
    return _get_mgr._mgr


def _require_season(mgr, manager_id):
    season = mgr.seasons.get_season(manager_id)
    if not season:
        return None, (jsonify({"error": "No active season."}), 404)
    return season, None


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

@prices_bp.route("/prices")
def api_season_prices():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    prices = mgr.prices.get_latest_prices(season["id"])
    alerts = mgr.get_price_alerts(season["id"])
    return jsonify({"prices": prices, "alerts": alerts})


@prices_bp.route("/update-prices", methods=["POST"])
def api_season_update_prices():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    mgr.track_prices(season["id"], manager_id)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Price Predictions
# ---------------------------------------------------------------------------

@prices_bp.route("/price-predictions")
def api_season_price_predictions():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    predictions = mgr.predict_price_changes(season["id"])
    risers = [p for p in predictions if p["direction"] == "rise"]
    fallers = [p for p in predictions if p["direction"] == "fall"]
    return jsonify({"predictions": predictions, "risers": risers, "fallers": fallers})


# ---------------------------------------------------------------------------
# Price History
# ---------------------------------------------------------------------------

@prices_bp.route("/price-history")
def api_season_price_history():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    player_ids_str = request.args.get("player_ids", "")
    player_ids = None
    if player_ids_str:
        try:
            player_ids = [int(x.strip()) for x in player_ids_str.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "player_ids must be comma-separated integers."}), 400

    days = request.args.get("days", 14, type=int)
    history = mgr.get_price_history(season["id"], player_ids=player_ids, days=days)
    return jsonify({"history": history})


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@prices_bp.route("/watchlist")
def api_season_watchlist():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    watchlist = mgr.watchlist.get_watchlist(season["id"])

    try:
        bootstrap = load_bootstrap()
        if bootstrap:
            elements = {e["id"]: e for e in bootstrap.get("elements", [])}
            id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
            code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}
            for w in watchlist:
                el = elements.get(w["player_id"])
                if el:
                    w["current_price"] = el.get("now_cost", 0) / 10
                    tc = id_to_code.get(el.get("team"))
                    w["team_short"] = code_to_short.get(tc, "")
    except Exception:
        pass

    return jsonify({"watchlist": watchlist})


@prices_bp.route("/watchlist/add", methods=["POST"])
def api_season_watchlist_add():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    player_id = body.get("player_id")
    if not player_id:
        return jsonify({"error": "player_id is required."}), 400
    try:
        player_id = int(player_id)
    except (TypeError, ValueError):
        return jsonify({"error": "player_id must be an integer."}), 400

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    web_name = ""
    team_code = None
    price = None
    try:
        bootstrap = load_bootstrap()
        if bootstrap:
            id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
            for el in bootstrap.get("elements", []):
                if el["id"] == player_id:
                    web_name = el.get("web_name", "")
                    team_code = id_to_code.get(el.get("team"))
                    price = el.get("now_cost", 0) / 10
                    break
    except Exception:
        pass

    mgr.watchlist.add_to_watchlist(
        season["id"], player_id,
        web_name=web_name, team_code=team_code, price_when_added=price,
    )
    return jsonify({"status": "ok"})


@prices_bp.route("/watchlist/remove", methods=["POST"])
def api_season_watchlist_remove():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    player_id = body.get("player_id")
    if not player_id:
        return jsonify({"error": "player_id is required."}), 400
    try:
        player_id = int(player_id)
    except (TypeError, ValueError):
        return jsonify({"error": "player_id must be an integer."}), 400

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    mgr.watchlist.remove_from_watchlist(season["id"], player_id)
    return jsonify({"status": "ok"})
