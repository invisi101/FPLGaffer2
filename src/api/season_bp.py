"""Season management blueprint — init, dashboard, recommendations, snapshots."""

import json

from flask import Blueprint, jsonify, request

from src.api.helpers import get_next_gw, require_manager_id
from src.api.sse import broadcast, run_in_background
from src.logging_config import get_logger

log = get_logger(__name__)

season_bp = Blueprint("season", __name__)


def _get_mgr():
    """Lazy-init SeasonManager singleton."""
    from src.season.manager import SeasonManager
    if not hasattr(_get_mgr, "_mgr"):
        _get_mgr._mgr = SeasonManager()
    return _get_mgr._mgr


def _require_season(mgr, manager_id):
    """Get active season or return (None, error_response)."""
    season = mgr.seasons.get_season(manager_id)
    if not season:
        return None, (jsonify({"error": "No active season."}), 404)
    return season, None


# ---------------------------------------------------------------------------
# Init / Status / Delete
# ---------------------------------------------------------------------------

@season_bp.route("/init", methods=["POST"])
def api_season_init():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()

    def do_init():
        mgr.init_season(manager_id, progress_fn=broadcast)

    started = run_in_background("Season Init", do_init)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@season_bp.route("/status")
def api_season_status():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season = mgr.seasons.get_season(manager_id)
    if not season:
        return jsonify({"active": False})
    return jsonify({"active": True, "season": season})


@season_bp.route("/delete", methods=["DELETE"])
def api_season_delete():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    mgr.seasons.delete_season(manager_id)
    return jsonify({"status": "deleted"})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@season_bp.route("/dashboard")
def api_season_dashboard():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    dashboard = mgr.get_dashboard(manager_id)
    return jsonify(dashboard)


# ---------------------------------------------------------------------------
# Generate Recommendation
# ---------------------------------------------------------------------------

@season_bp.route("/generate-recommendation", methods=["POST"])
def api_season_generate_recommendation():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()

    def do_recommend():
        mgr.tick(manager_id, progress_fn=broadcast)

    started = run_in_background("Generate Recommendation", do_recommend)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# Record Results
# ---------------------------------------------------------------------------

@season_bp.route("/record-results", methods=["POST"])
def api_season_record_results():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()

    def do_record():
        mgr.tick(manager_id, progress_fn=broadcast)

    started = run_in_background("Record Results", do_record)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


# ---------------------------------------------------------------------------
# Recommendations / Snapshots / Fixtures / Chips
# ---------------------------------------------------------------------------

@season_bp.route("/recommendations")
def api_season_recommendations():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    recs = mgr.recommendations.get_recommendations(season["id"])
    return jsonify({"recommendations": recs})


@season_bp.route("/snapshots")
def api_season_snapshots():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    snaps = mgr.snapshots.get_snapshots(season["id"])
    return jsonify({"snapshots": snaps})


@season_bp.route("/fixtures")
def api_season_fixtures():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    from_gw = request.args.get("from_gw", type=int)
    to_gw = request.args.get("to_gw", type=int)
    fixtures = mgr.fixtures.get_fixture_calendar(season["id"], from_gw=from_gw, to_gw=to_gw)
    return jsonify({"fixtures": fixtures})


@season_bp.route("/chips")
def api_season_chips():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    chips_used = mgr.dashboard.get_chips_status(season["id"])
    next_gw = get_next_gw()
    if next_gw is None:
        next_gw = 1

    current_half_start = 1 if next_gw <= 19 else 20
    current_half_end = 19 if next_gw <= 19 else 38
    other_half_start = 20 if next_gw <= 19 else 1
    other_half_end = 38 if next_gw <= 19 else 19

    all_chips = [
        {"name": "wildcard", "label": "Wildcard"},
        {"name": "freehit", "label": "Free Hit"},
        {"name": "bboost", "label": "Bench Boost"},
        {"name": "3xc", "label": "Triple Captain"},
    ]
    result = []
    for chip in all_chips:
        chip_uses = [c for c in chips_used if c["chip_used"] == chip["name"]]
        current_half_use = next(
            (c for c in chip_uses if current_half_start <= c["gameweek"] <= current_half_end), None
        )
        other_half_use = next(
            (c for c in chip_uses if other_half_start <= c["gameweek"] <= other_half_end), None
        )
        result.append({
            **chip,
            "used": current_half_use is not None,
            "used_gw": current_half_use["gameweek"] if current_half_use else None,
            "used_other_half": other_half_use is not None,
            "used_other_gw": other_half_use["gameweek"] if other_half_use else None,
        })

    recs = mgr.recommendations.get_recommendations(season["id"])
    chip_values = {}
    if recs:
        try:
            chip_values = json.loads(recs[-1].get("chip_values_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

    return jsonify({"chips": result, "chip_values": chip_values})


# ---------------------------------------------------------------------------
# GW Detail / Transfer History / Bank Analysis
# ---------------------------------------------------------------------------

@season_bp.route("/gw-detail")
def api_season_gw_detail():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    gameweek = request.args.get("gameweek", type=int)
    if not gameweek:
        return jsonify({"error": "gameweek is required."}), 400

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    snapshot = mgr.snapshots.get_snapshot(season["id"], gameweek)
    rec = mgr.recommendations.get_recommendation(season["id"], gameweek)
    return jsonify({"snapshot": snapshot, "recommendation": rec})


@season_bp.route("/transfer-history")
def api_season_transfer_history():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    history = mgr.dashboard.get_transfer_history(season["id"])
    return jsonify({"transfer_history": history})


@season_bp.route("/bank-analysis")
def api_season_bank_analysis():
    manager_id, err = require_manager_id(request.args)
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    recs = mgr.recommendations.get_recommendations(season["id"])
    if not recs:
        return jsonify({"error": "No recommendations yet."}), 404

    latest = recs[-1]
    try:
        analysis = json.loads(latest.get("bank_analysis_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        analysis = {}
    return jsonify({"bank_analysis": analysis, "gameweek": latest.get("gameweek")})


# ---------------------------------------------------------------------------
# Update fixtures / prices
# ---------------------------------------------------------------------------

@season_bp.route("/update-fixtures", methods=["POST"])
def api_season_update_fixtures():
    body = request.get_json(silent=True) or {}
    manager_id, err = require_manager_id(body, "body")
    if err:
        return jsonify(err[0]), err[1]

    mgr = _get_mgr()
    season, serr = _require_season(mgr, manager_id)
    if serr:
        return serr

    mgr.update_fixture_calendar(season["id"])
    return jsonify({"status": "ok"})
