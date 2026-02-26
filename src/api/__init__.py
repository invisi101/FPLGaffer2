"""Flask application factory."""

import os

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="../templates")

    from src.api.middleware import register_middleware
    register_middleware(app)

    from src.api.core import core_bp
    from src.api.team import team_bp
    from src.api.season_bp import season_bp
    from src.api.strategy_bp import strategy_bp
    from src.api.prices_bp import prices_bp
    from src.api.backtest_bp import backtest_bp
    from src.api.compare_bp import compare_bp
    from src.api.season_v2_bp import season_v2_bp
    from src.api.prices_standalone_bp import prices_standalone_bp

    app.register_blueprint(core_bp)
    app.register_blueprint(team_bp)
    app.register_blueprint(season_bp, url_prefix="/api/season")
    app.register_blueprint(strategy_bp, url_prefix="/api")
    app.register_blueprint(prices_bp, url_prefix="/api/season")
    app.register_blueprint(backtest_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(season_v2_bp, url_prefix="/api/v2/season")
    app.register_blueprint(prices_standalone_bp, url_prefix="/api/prices")

    # Start background scheduler if GAFFER_MANAGER_ID is set.
    manager_id_str = os.environ.get("GAFFER_MANAGER_ID")
    if manager_id_str:
        try:
            from src.paths import DB_PATH
            from src.season.scheduler import start_scheduler

            start_scheduler(int(manager_id_str), DB_PATH)
        except (ValueError, TypeError):
            pass  # Invalid manager_id — skip scheduler

    return app
