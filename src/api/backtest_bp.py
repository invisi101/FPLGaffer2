"""Backtest blueprint — walk-forward backtesting endpoints."""

from flask import Blueprint, jsonify, request

from src.api.sse import broadcast, run_in_background
from src.logging_config import get_logger

log = get_logger(__name__)

backtest_bp = Blueprint("backtest", __name__)


@backtest_bp.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run walk-forward backtest."""
    import src.api.sse as sse_module

    body = request.get_json(silent=True) or {}
    start_gw = body.get("start_gw", 10)
    end_gw = body.get("end_gw", 27)
    seasons = body.get("seasons")

    def do_backtest():
        from src.data.loader import load_all_data
        from src.features.builder import build_features
        from src.ml.backtest import run_backtest

        broadcast("Loading data for backtest...", event="progress")
        data = load_all_data()
        df = build_features(data)

        broadcast("Running backtest...", event="progress")
        results = run_backtest(
            df, start_gw=start_gw, end_gw=end_gw,
            seasons=seasons,
        )
        sse_module.backtest_results = results

    started = run_in_background("Backtest", do_backtest)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@backtest_bp.route("/api/backtest-results")
def api_backtest_results():
    """Return backtest results."""
    import src.api.sse as sse_module
    if sse_module.backtest_results is None:
        return jsonify({"error": "No backtest results. Run a backtest first."}), 404
    return jsonify(sse_module.backtest_results)
