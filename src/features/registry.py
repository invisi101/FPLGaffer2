"""Declarative feature registry — single source of truth for feature metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import DEFAULT_FEATURES, SUB_MODEL_FEATURES, FEATURE_FILL_DEFAULTS

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------
ALL_POSITIONS = frozenset({"GKP", "DEF", "MID", "FWD"})


@dataclass(frozen=True)
class FeatureDef:
    """Metadata for a single model feature."""

    name: str
    source: str  # "element_history", "pms_rolling", "playerstats", "team_stats", etc.
    positions: frozenset[str]  # which positions use this feature
    sub_models: frozenset[str]  # which sub-models use it (empty = main model only)
    fill_default: float = 0.0


# ---------------------------------------------------------------------------
# Infer source from feature name
# ---------------------------------------------------------------------------
def _infer_source(name: str) -> str:
    """Best-effort source inference from feature naming conventions."""
    if name.startswith("player_") and "_last" in name:
        return "pms_rolling"
    if name.startswith("ewm_player_"):
        return "ewm"
    if name.startswith("opp_"):
        return "team_stats"
    if name.startswith("team_") and "_last" in name:
        return "team_stats"
    if name.startswith("gw_"):
        return "playerstats"
    if name.startswith("vs_opponent_"):
        return "opponent_history"
    if name in ("home_xg_form", "away_xg_form", "venue_matched_form"):
        return "venue_form"
    if name in ("xg_volatility_last5", "form_acceleration", "big_chance_frequency_last5"):
        return "upside"
    if name in ("xg_x_opp_goals_conceded", "chances_x_opp_big_chances", "cs_opportunity"):
        return "interactions"
    if name in ("days_rest", "fixture_congestion"):
        return "rest_congestion"
    if name in ("is_home", "fdr", "opponent_elo", "next_gw_fixture_count"):
        return "fixture_context"
    if name in ("player_form", "cost", "chance_of_playing", "ownership",
                "set_piece_involvement", "clean_sheets_per_90", "starts_per_90",
                "xgc_per_90", "saves_per_90", "transfers_in_event",
                "net_transfers", "transfer_momentum", "availability_rate_last5",
                "ep_next", "cumulative_minutes"):
        return "playerstats"
    if name == "season_progress":
        return "derived"
    if name == "team_form_5":
        return "team_form"
    if name.startswith("pos_") or name == "minutes_availability":
        return "derived"
    return "unknown"


# ---------------------------------------------------------------------------
# Build the registry from config
# ---------------------------------------------------------------------------
def _build_registry() -> list[FeatureDef]:
    """Populate the registry from DEFAULT_FEATURES and SUB_MODEL_FEATURES."""
    # Collect all feature names and the positions / sub-models that use them
    feat_positions: dict[str, set[str]] = {}
    feat_sub_models: dict[str, set[str]] = {}

    # Main model features
    for pos, feats in DEFAULT_FEATURES.items():
        for f in feats:
            feat_positions.setdefault(f, set()).add(pos)

    # Sub-model features
    for component, feats in SUB_MODEL_FEATURES.items():
        for f in feats:
            feat_positions.setdefault(f, set())  # may already exist
            feat_sub_models.setdefault(f, set()).add(component)

    registry: list[FeatureDef] = []
    for name in sorted(feat_positions):
        positions = frozenset(feat_positions[name])
        sub_models = frozenset(feat_sub_models.get(name, set()))
        source = _infer_source(name)
        fill = FEATURE_FILL_DEFAULTS.get(name, 0.0)
        registry.append(FeatureDef(
            name=name,
            source=source,
            positions=positions,
            sub_models=sub_models,
            fill_default=fill,
        ))

    return registry


FEATURE_REGISTRY: list[FeatureDef] = _build_registry()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_features_for_position(position: str) -> list[str]:
    """Return the default feature list for a given position (main model)."""
    return list(DEFAULT_FEATURES.get(position, []))


def get_sub_model_features(component: str) -> list[str]:
    """Return the feature list for a given sub-model component."""
    return list(SUB_MODEL_FEATURES.get(component, []))


def get_all_feature_names() -> list[str]:
    """Return all unique feature names across the entire registry."""
    return [fd.name for fd in FEATURE_REGISTRY]


def get_features_by_source(source: str) -> list[str]:
    """Return all feature names from a given source."""
    return [fd.name for fd in FEATURE_REGISTRY if fd.source == source]
