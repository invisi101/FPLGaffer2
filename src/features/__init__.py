"""Feature engineering package — modular feature construction pipeline.

Public API:
    build_features(data) -> DataFrame
    get_fixture_context(data) -> dict
    get_feature_columns(df) -> list[str]
"""

from src.features.builder import build_features, get_fixture_context, get_feature_columns

__all__ = ["build_features", "get_fixture_context", "get_feature_columns"]
