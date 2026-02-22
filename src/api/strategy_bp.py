"""Strategy blueprint — strategic plan, action plan, plan health, preseason."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import require_manager_id
from src.api.sse import broadcast, run_in_background
from src.logging_config import get_logger

log = get_logger(__name__)

strategy_bp = Blueprint("strategy", __name__)


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
# Strategic Plan (GET/POST)
# ---------------------------------------------------------------------------

@strategy_bp.route("/season/strategic-plan", methods=["GET", "POST"])
def api_season_strategic_plan():
    """Generate (POST) or fetch (GET) strategic plan."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        manager_id, err = require_manager_id(body, "body")
        if err:
            return jsonify(err[0]), err[1]

        mgr = _get_mgr()

        def do_plan():
            mgr.generate_recommendation(manager_id, progress_fn=broadcast)

        started = run_in_background("Strategic Plan", do_plan)
        if not started:
            return jsonify({"error": "Another task is already running."}), 409
        return jsonify({"status": "started"})

    # GET
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    plan_row = mgr.plans.get_strategic_plan(season["id"])
    if not plan_row:
        return jsonify({"error": "No strategic plan generated yet."}), 404

    plan = {}
    try:
        plan = json.loads(plan_row.get("plan_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    changelog = mgr.plans.get_plan_changelog(season["id"], limit=20)

    return jsonify({
        "plan": plan,
        "as_of_gw": plan_row.get("as_of_gw"),
        "created_at": plan_row.get("created_at"),
        "changelog": changelog,
    })


# ---------------------------------------------------------------------------
# Action Plan
# ---------------------------------------------------------------------------

@strategy_bp.route("/season/action-plan")
def api_season_action_plan():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.get_action_plan(manager_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

@strategy_bp.route("/season/outcomes")
def api_season_outcomes():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    outcomes = mgr.get_outcomes(manager_id)
    return jsonify({"outcomes": outcomes})


# ---------------------------------------------------------------------------
# Plan Health
# ---------------------------------------------------------------------------

@strategy_bp.route("/season/plan-health")
def api_season_plan_health():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    result = mgr.check_plan_health(manager_id)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Plan Changelog
# ---------------------------------------------------------------------------

@strategy_bp.route("/season/plan-changelog")
def api_season_plan_changelog():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    changelog = mgr.plans.get_plan_changelog(season["id"])
    return jsonify({"changelog": changelog})


# ---------------------------------------------------------------------------
# Pre-Season
# ---------------------------------------------------------------------------

@strategy_bp.route("/preseason/generate", methods=["POST"])
def api_preseason_generate():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()

    def do_preseason():
        mgr.generate_preseason_plan(manager_id, progress_fn=broadcast)

    started = run_in_background("Pre-Season Plan", do_preseason)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@strategy_bp.route("/preseason/result")
def api_preseason_result():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    rec = mgr.recommendations.get_recommendation(season["id"], 1)
    if not rec:
        return jsonify({"error": "No pre-season plan generated yet."}), 404

    squad = []
    try:
        squad = json.loads(rec.get("new_squad_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        pass

    plan_row = mgr.plans.get_strategic_plan(season["id"])
    chip_schedule = {}
    if plan_row:
        try:
            plan = json.loads(plan_row.get("plan_json") or "{}")
            chip_schedule = plan.get("chip_schedule", {})
        except (json.JSONDecodeError, TypeError):
            pass

    return jsonify({
        "initial_squad": squad,
        "predicted_points": rec.get("predicted_points"),
        "captain": {"id": rec.get("captain_id"), "name": rec.get("captain_name")},
        "chip_schedule": chip_schedule,
    })
