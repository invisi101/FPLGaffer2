"""Standalone Prices blueprint — works without an active season.

Provides price data, predictions, and watchlist functionality using only
the FPL bootstrap API and a manager_id-keyed watchlist table.
"""

from flask import Blueprint, jsonify, request

from src.api.helpers import load_bootstrap, require_manager_id, resolve_current_squad_event
from src.data.fpl_api import fetch_fpl_api, fetch_manager_history, fetch_manager_picks
from src.db.repositories import StandaloneWatchlistRepository
from src.logging_config import get_logger
from src.strategy.price_tracker import get_price_alerts, predict_price_changes

log = get_logger(__name__)

prices_standalone_bp = Blueprint("prices_standalone", __name__)

_watchlist_repo: StandaloneWatchlistRepository | None = None


def _get_watchlist_repo() -> StandaloneWatchlistRepository:
    global _watchlist_repo
    if _watchlist_repo is None:
        _watchlist_repo = StandaloneWatchlistRepository()
    return _watchlist_repo


def _fetch_squad_ids(manager_id: int, bootstrap: dict) -> set[int]:
    """Fetch current squad player IDs from FPL API, with FH reversion."""
    try:
        history = fetch_manager_history(manager_id)
        current_entries = history.get("current", [])
        if not current_entries:
            return set()
        current_event = current_entries[-1].get("event", 1)
        squad_event, _ = resolve_current_squad_event(history, current_event)
        picks_data = fetch_manager_picks(manager_id, squad_event)
        picks = picks_data.get("picks", [])
        return {p["element"] for p in picks}
    except Exception as exc:
        log.warning("Could not fetch squad for manager %s: %s", manager_id, exc)
        return set()


# ---------------------------------------------------------------------------
# Live Prices (squad + watchlist)
# ---------------------------------------------------------------------------

@prices_standalone_bp.route("/live")
def api_prices_live():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    bootstrap = load_bootstrap()
    if not bootstrap:
        try:
            bootstrap = fetch_fpl_api("bootstrap")
        except Exception:
            return jsonify({"error": "Could not load bootstrap data."}), 503

    elements = {el["id"]: el for el in bootstrap.get("elements", [])}
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

    # Get squad + watchlist player IDs
    squad_ids = _fetch_squad_ids(manager_id, bootstrap)
    watchlist = _get_watchlist_repo().get_watchlist(manager_id)
    wl_ids = {w["player_id"] for w in watchlist}
    track_ids = squad_ids | wl_ids

    # Build price rows from bootstrap
    prices = []
    for pid in track_ids:
        el = elements.get(pid)
        if not el:
            continue
        tc = id_to_code.get(el.get("team"))
        prices.append({
            "player_id": el["id"],
            "web_name": el.get("web_name"),
            "team_code": tc,
            "team_short": code_to_short.get(tc, ""),
            "price": el.get("now_cost", 0) / 10,
            "transfers_in_event": el.get("transfers_in_event", 0),
            "transfers_out_event": el.get("transfers_out_event", 0),
            "in_squad": pid in squad_ids,
            "in_watchlist": pid in wl_ids,
        })
    prices.sort(key=lambda p: p.get("web_name") or "")

    alerts = get_price_alerts(bootstrap)
    return jsonify({"prices": prices, "alerts": alerts})


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

@prices_standalone_bp.route("/refresh", methods=["POST"])
def api_prices_refresh():
    try:
        fetch_fpl_api("bootstrap", force=True)
    except Exception as exc:
        log.warning("Bootstrap refresh failed: %s", exc)
        return jsonify({"error": "Refresh failed."}), 503
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Price Predictions
# ---------------------------------------------------------------------------

@prices_standalone_bp.route("/predictions")
def api_prices_predictions():
    bootstrap = load_bootstrap()
    if not bootstrap:
        try:
            bootstrap = fetch_fpl_api("bootstrap")
        except Exception:
            return jsonify({"error": "Could not load bootstrap data."}), 503

    predictions = predict_price_changes(bootstrap)
    risers = [p for p in predictions if p["direction"] == "rise"]
    fallers = [p for p in predictions if p["direction"] == "fall"]
    return jsonify({"predictions": predictions, "risers": risers, "fallers": fallers})


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@prices_standalone_bp.route("/watchlist")
def api_prices_watchlist():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    repo = _get_watchlist_repo()
    watchlist = repo.get_watchlist(manager_id)

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


@prices_standalone_bp.route("/watchlist/add", methods=["POST"])
def api_prices_watchlist_add():
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

    _get_watchlist_repo().add_to_watchlist(
        manager_id, player_id,
        web_name=web_name, team_code=team_code, price_when_added=price,
    )
    return jsonify({"status": "ok"})


@prices_standalone_bp.route("/watchlist/remove", methods=["POST"])
def api_prices_watchlist_remove():
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

    _get_watchlist_repo().remove_from_watchlist(manager_id, player_id)
    return jsonify({"status": "ok"})
