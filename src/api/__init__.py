"""Flask application factory."""

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

    app.register_blueprint(core_bp)
    app.register_blueprint(team_bp)
    app.register_blueprint(season_bp, url_prefix="/api/season")
    app.register_blueprint(strategy_bp, url_prefix="/api")
    app.register_blueprint(prices_bp, url_prefix="/api/season")
    app.register_blueprint(backtest_bp)
    app.register_blueprint(compare_bp)

    return app
