"""Flask middleware — no-cache headers, error handlers, JSON encoding."""

import math

from flask import Flask, jsonify, request


class SafeJSONProvider:
    """JSON provider that handles NaN/Inf in responses."""

    @staticmethod
    def default(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def register_middleware(app: Flask) -> None:
    """Register middleware on the Flask app."""

    @app.after_request
    def add_no_cache_headers(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(_):
        return jsonify({"error": "Internal server error"}), 500
