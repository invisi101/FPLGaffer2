"""Season management v2 blueprint -- state-machine-driven GW lifecycle."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import require_manager_id
from src.api.sse import broadcast, run_in_background
from src.logging_config import get_logger

log = get_logger(__name__)

season_v2_bp = Blueprint("season_v2", __name__)


def _get_mgr():
    """Lazy-init SeasonManager singleton."""
    from src.season.manager import SeasonManager
    if not hasattr(_get_mgr, "_mgr"):
        _get_mgr._mgr = SeasonManager()
    return _get_mgr._mgr


# ---------------------------------------------------------------------------
# Init / Status / Delete
# ---------------------------------------------------------------------------

@season_v2_bp.route("/init", methods=["POST"])
def api_v2_init():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()

    def do_init():
        mgr.init_season(manager_id, progress_fn=broadcast)

    started = run_in_background("Season Init V2", do_init)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@season_v2_bp.route("/status")
def api_v2_status():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.get_status(manager_id)
    return jsonify(result)


@season_v2_bp.route("/delete", methods=["DELETE"])
def api_v2_delete():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    mgr.seasons.delete_season(manager_id)
    return jsonify({"status": "deleted"})


# ---------------------------------------------------------------------------
# Tick (manual trigger)
# ---------------------------------------------------------------------------

@season_v2_bp.route("/tick", methods=["POST"])
def api_v2_tick():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    force_replan = bool(body.get("force_replan", False))
    mgr = _get_mgr()

    def do_tick():
        alerts = mgr.tick(manager_id, progress_fn=broadcast, force_replan=force_replan)
        for alert in alerts:
            broadcast(alert.get("message", str(alert)), event="alert")

    started = run_in_background("Tick V2", do_tick)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# User actions (READY phase)
# ---------------------------------------------------------------------------

@season_v2_bp.route("/accept-transfers", methods=["POST"])
def api_v2_accept_transfers():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.accept_transfers(manager_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/make-transfer", methods=["POST"])
def api_v2_make_transfer():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    player_out_id = body.get("player_out_id")
    player_in_id = body.get("player_in_id")
    if not player_out_id or not player_in_id:
        return jsonify({"error": "player_out_id and player_in_id are required."}), 400

    try:
        player_out_id = int(player_out_id)
        player_in_id = int(player_in_id)
    except (TypeError, ValueError):
        return jsonify({"error": "player_out_id and player_in_id must be integers."}), 400

    mgr = _get_mgr()
    result = mgr.make_transfer(manager_id, player_out_id, player_in_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/make-transfers", methods=["POST"])
def api_v2_make_transfers():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    players_out = body.get("players_out")
    players_in = body.get("players_in")
    if not players_out or not players_in:
        return jsonify({"error": "players_out and players_in are required."}), 400

    try:
        players_out = [int(x) for x in players_out]
        players_in = [int(x) for x in players_in]
    except (TypeError, ValueError):
        return jsonify({"error": "players_out and players_in must be lists of integers."}), 400

    mgr = _get_mgr()
    result = mgr.make_transfers(manager_id, players_out, players_in)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/undo-transfers", methods=["POST"])
def api_v2_undo_transfers():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.undo_transfers(manager_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/lock-chip", methods=["POST"])
def api_v2_lock_chip():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    chip = body.get("chip")
    if not chip:
        return jsonify({"error": "chip is required."}), 400

    mgr = _get_mgr()
    result = mgr.lock_chip(manager_id, chip)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/unlock-chip", methods=["POST"])
def api_v2_unlock_chip():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.unlock_chip(manager_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@season_v2_bp.route("/set-captain", methods=["POST"])
def api_v2_set_captain():
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
    result = mgr.set_captain(manager_id, player_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


# ---------------------------------------------------------------------------
# Stubs (not yet implemented)
# ---------------------------------------------------------------------------

@season_v2_bp.route("/fixture-lookahead")
def api_v2_fixture_lookahead():
    return jsonify({"error": "Not yet implemented"}), 501


@season_v2_bp.route("/history")
def api_v2_history():
    return jsonify({"error": "Not yet implemented"}), 501


@season_v2_bp.route("/available-players")
def api_v2_available_players():
    return jsonify({"error": "Not yet implemented"}), 501
