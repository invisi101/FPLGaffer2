"""Central configuration — every magic number in one place."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    verbosity: int = 0
    early_stopping_rounds: int = 20

    # Sub-model overrides
    sub_model_max_depth: int = 4

    # Tuning grid (used when tune=True)
    tune_n_estimators: tuple[int, ...] = (100, 200)
    tune_max_depth: tuple[int, ...] = (4, 6)
    tune_learning_rate: tuple[float, ...] = (0.05, 0.1)

    # Training
    min_train_gws: int = 10
    walk_forward_splits: int = 20


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleConfig:
    decomposed_weight: float = 0.15  # 85/15 mean/decomposed blend
    captain_mean_weight: float = 0.4
    captain_q80_weight: float = 0.6


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverConfig:
    bench_weight: float = 0.25
    hit_cost: float = 4.0
    max_budget: float = 1000.0
    team_cap: int = 3
    squad_size: int = 15
    starting_xi: int = 11
    squad_positions: dict[str, int] = field(default_factory=lambda: {
        "GKP": 2, "DEF": 5, "MID": 5, "FWD": 3,
    })
    formation_limits: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3),
    })


# ---------------------------------------------------------------------------
# Cache TTLs (seconds)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheConfig:
    github_csv: int = 6 * 3600       # 6 hours
    fpl_api: int = 30 * 60           # 30 minutes
    element_summary: int = 30 * 60   # 30 minutes
    manager_api: int = 60            # 1 minute


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionConfig:
    confidence_decay: tuple[float, ...] = (0.95, 0.93, 0.90, 0.87, 0.83, 0.80, 0.77)
    horizons: tuple[int, ...] = (1, 3, 8)
    pool_size: int = 200  # Top players for multi-GW planning


# ---------------------------------------------------------------------------
# FPL Scoring Rules (points per action by position)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPLScoringRules:
    scoring: dict[str, dict[str, int | float]] = field(default_factory=lambda: {
        "GKP": {
            "appearance": 2, "goal": 10, "assist": 3, "cs": 4,
            "gc_per_2": -1, "save_per_3": 1, "defcon": 2,
            "defcon_threshold": 10,
        },
        "DEF": {
            "appearance": 2, "goal": 6, "assist": 3, "cs": 4,
            "gc_per_2": -1, "save_per_3": 0, "defcon": 2,
            "defcon_threshold": 10,
        },
        "MID": {
            "appearance": 2, "goal": 5, "assist": 3, "cs": 1,
            "gc_per_2": 0, "save_per_3": 0, "defcon": 2,
            "defcon_threshold": 12,
        },
        "FWD": {
            "appearance": 2, "goal": 4, "assist": 3, "cs": 0,
            "gc_per_2": 0, "save_per_3": 0, "defcon": 2,
            "defcon_threshold": 12,
        },
    })


# ---------------------------------------------------------------------------
# Decomposed sub-models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecomposedConfig:
    # Which sub-models to train per position
    components: dict[str, list[str]] = field(default_factory=lambda: {
        "GKP": ["cs", "goals_conceded", "saves", "bonus"],
        "DEF": ["goals", "assists", "cs", "goals_conceded", "bonus", "defcon"],
        "MID": ["goals", "assists", "cs", "bonus", "defcon"],
        "FWD": ["goals", "assists", "bonus", "defcon"],
    })

    # XGBoost objective per component
    objectives: dict[str, str] = field(default_factory=lambda: {
        "goals": "count:poisson",
        "assists": "count:poisson",
        "cs": "reg:squarederror",
        "bonus": "count:poisson",
        "goals_conceded": "count:poisson",
        "saves": "count:poisson",
        "defcon": "count:poisson",
    })

    # Soft calibration caps per position
    soft_caps: dict[str, float] = field(default_factory=lambda: {
        "GKP": 7.0, "DEF": 8.0, "MID": 10.0, "FWD": 10.0,
    })

    # Target column mapping
    target_columns: dict[str, str] = field(default_factory=lambda: {
        "goals": "next_gw_goals",
        "assists": "next_gw_assists",
        "cs": "next_gw_cs",
        "bonus": "next_gw_bonus",
        "goals_conceded": "next_gw_goals_conceded",
        "saves": "next_gw_saves",
        "defcon": "next_gw_cbit",
    })


# ---------------------------------------------------------------------------
# Sub-model feature sets
# ---------------------------------------------------------------------------
SUB_MODEL_FEATURES: dict[str, list[str]] = {
    "goals": [
        "player_xg_last3", "player_xg_last5", "ewm_player_xg_last3",
        "player_xgot_last3", "player_shots_on_target_last3",
        "player_shots_on_target_last5", "player_touches_opposition_box_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "xg_x_opp_goals_conceded", "is_home", "fdr", "opponent_elo",
        "set_piece_involvement", "big_chance_frequency_last5",
        "player_form", "gw_threat", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "cost", "team_goals_scored_last3",
    ],
    "assists": [
        "player_xa_last3", "player_xa_last5", "ewm_player_xa_last3",
        "player_chances_created_last3", "ewm_player_chances_created_last3",
        "player_successful_dribbles_last3", "player_accurate_crosses_last3",
        "opp_big_chances_allowed_last3", "chances_x_opp_big_chances",
        "is_home", "fdr", "opponent_elo", "gw_creativity",
        "player_form", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
    ],
    "cs": [
        "opp_opponent_xg_last3", "opp_opponent_shots_on_target_last3",
        "opp_goals_conceded_last3",
        "team_clean_sheet_last3", "cs_opportunity",
        "is_home", "fdr", "opponent_elo",
        "clean_sheets_per_90", "xgc_per_90",
        "player_minutes_played_last3", "starts_per_90",
        "next_gw_fixture_count",
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "opp_big_chances_allowed_last3",
        "opp_goals_scored_last3", "opp_xg_last3",
    ],
    "bonus": [
        "gw_player_bps", "gw_ict_index", "gw_influence", "gw_threat", "gw_creativity",
        "player_form", "player_xg_last3", "player_xa_last3",
        "player_goals_last3", "player_assists_last3",
        "is_home", "cost", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
    ],
    "goals_conceded": [
        "opp_opponent_xg_last3", "opp_goals_conceded_last3",
        "opp_big_chances_allowed_last3",
        "team_clean_sheet_last3", "cs_opportunity",
        "is_home", "fdr", "opponent_elo",
        "xgc_per_90", "next_gw_fixture_count",
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
    ],
    "saves": [
        "player_saves_last3", "player_saves_last5", "saves_per_90",
        "opp_opponent_xg_last3", "opp_opponent_shots_on_target_last3",
        "opp_goals_conceded_last3",
        "is_home", "fdr", "opponent_elo",
        "next_gw_fixture_count",
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "opp_big_chances_allowed_last3",
    ],
    "defcon": [
        "player_cbit_last3", "player_cbit_last5",
        "player_recoveries_last3", "player_recoveries_last5",
        "player_clearances_last3", "player_blocks_last3",
        "player_interceptions_last3", "player_tackles_won_last3",
        "player_aerial_duels_won_last3",
        "is_home", "fdr", "opponent_elo",
        "opp_goals_scored_last3", "opp_xg_last3",
        "opp_big_chances_allowed_last3",
        "player_minutes_played_last3", "starts_per_90",
        "next_gw_fixture_count",
    ],
}

# ---------------------------------------------------------------------------
# Default feature sets per position (for mean regression model)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES: dict[str, list[str]] = {
    "GKP": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "opp_opponent_xg_last3", "opp_goals_conceded_last3",
        "player_minutes_played_last3", "chance_of_playing", "cs_opportunity",
        "opp_opponent_shots_on_target_last3", "gw_influence", "ownership",
        "team_form_5", "next_gw_fixture_count", "season_progress",
        "player_saves_last3", "player_saves_last5", "saves_per_90",
        "clean_sheets_per_90", "starts_per_90", "xgc_per_90",
        "team_clean_sheet_last3", "team_goals_scored_last3",
        "player_minutes_played_last5",
        "transfer_momentum",
        "days_rest", "fixture_congestion",
        "venue_matched_form",
        "availability_rate_last5",
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "transfers_in_event", "net_transfers",
        "opp_big_chances_allowed_last3",
        "opp_goals_scored_last3", "opp_xg_last3",
    ],
    "DEF": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_chances_created_last3", "player_clearances_last3",
        "player_interceptions_last3", "player_tackles_won_last3",
        "player_blocks_last3", "player_cbit_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "cs_opportunity", "player_minutes_played_last3",
        "chance_of_playing", "gw_influence", "gw_threat", "gw_creativity",
        "set_piece_involvement", "team_form_5",
        "next_gw_fixture_count", "season_progress",
        "clean_sheets_per_90", "starts_per_90", "xgc_per_90",
        "team_clean_sheet_last3", "team_goals_scored_last3", "team_xg_last3",
        "player_xg_last5", "player_xa_last5", "player_minutes_played_last5",
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        "transfer_momentum",
        "days_rest", "fixture_congestion",
        "venue_matched_form",
        "vs_opponent_matches", "vs_opponent_goals_avg", "vs_opponent_xg_avg",
        "transfers_in_event", "net_transfers", "ownership",
        "gw_ict_index",
        "opp_big_chances_allowed_last3",
        "opp_goals_scored_last3", "opp_xg_last3",
        "availability_rate_last5",
    ],
    "MID": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_xgot_last3", "player_shots_on_target_last3",
        "player_chances_created_last3", "player_touches_opposition_box_last3",
        "player_successful_dribbles_last3", "player_accurate_crosses_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "opp_big_chances_allowed_last3",
        "xg_x_opp_goals_conceded", "chances_x_opp_big_chances",
        "player_minutes_played_last3", "chance_of_playing",
        "gw_influence", "gw_threat", "gw_creativity", "gw_ict_index",
        "set_piece_involvement", "team_form_5",
        "next_gw_fixture_count", "season_progress",
        "starts_per_90", "transfers_in_event",
        "team_goals_scored_last3", "team_xg_last3", "team_big_chances_last3",
        "player_xg_last5", "player_xa_last5", "player_shots_on_target_last5",
        "player_minutes_played_last5",
        "player_touches_opposition_box_last5", "player_total_shots_last3",
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        "ewm_player_xgot_last3", "ewm_player_chances_created_last3",
        "ewm_player_shots_on_target_last3",
        "net_transfers", "transfer_momentum",
        "days_rest", "fixture_congestion",
        "home_xg_form", "away_xg_form", "venue_matched_form",
        "vs_opponent_xg_avg", "vs_opponent_goals_avg", "vs_opponent_matches",
        "xg_volatility_last5", "form_acceleration", "big_chance_frequency_last5",
        "availability_rate_last5",
        "player_cbit_last3", "player_tackles_won_last3", "player_interceptions_last3",
    ],
    "FWD": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_xgot_last3", "player_total_shots_last3",
        "player_shots_on_target_last3", "player_touches_opposition_box_last3",
        "player_big_chances_missed_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "opp_big_chances_allowed_last3",
        "xg_x_opp_goals_conceded",
        "player_minutes_played_last3", "chance_of_playing",
        "gw_influence", "gw_threat", "gw_creativity", "gw_ict_index",
        "set_piece_involvement", "chances_x_opp_big_chances",
        "team_form_5", "next_gw_fixture_count", "season_progress",
        "starts_per_90", "transfers_in_event",
        "team_goals_scored_last3", "team_xg_last3", "team_big_chances_last3",
        "player_xg_last5", "player_xa_last5", "player_shots_on_target_last5",
        "player_touches_opposition_box_last5", "player_minutes_played_last5",
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        "ewm_player_xgot_last3", "ewm_player_chances_created_last3",
        "ewm_player_shots_on_target_last3",
        "net_transfers", "transfer_momentum",
        "days_rest", "fixture_congestion",
        "home_xg_form", "away_xg_form", "venue_matched_form",
        "vs_opponent_xg_avg", "vs_opponent_goals_avg", "vs_opponent_matches",
        "xg_volatility_last5", "form_acceleration", "big_chance_frequency_last5",
        "availability_rate_last5",
    ],
}


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    github_base: str = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
    fpl_api_base: str = "https://fantasy.premierleague.com/api"
    earliest_season: str = "2024-2025"
    max_seasons: int = 2
    flat_layout_seasons: frozenset[str] = frozenset({"2024-2025"})


# ---------------------------------------------------------------------------
# Strategy / planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    planning_horizon: int = 5        # GWs ahead for transfer planner
    max_hits_per_gw: int = 2         # Max hit transfers explored per GW
    ft_max_bank: int = 5             # Max banked free transfers
    late_season_gw: int = 33         # GW from which late-season mode activates
    late_season_hit_cost: float = 3.0  # Reduced hit cost in late season


# ---------------------------------------------------------------------------
# Position groups
# ---------------------------------------------------------------------------
POSITION_GROUPS: list[str] = ["GKP", "DEF", "MID", "FWD"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
PLAYER_ROLLING_COLS: list[str] = [
    "xg", "xa", "xgot", "total_shots", "shots_on_target",
    "touches_opposition_box", "chances_created", "successful_dribbles",
    "accurate_crosses", "tackles_won", "interceptions", "recoveries",
    "clearances", "minutes_played", "goals", "assists",
    "big_chances_missed", "accurate_passes", "final_third_passes",
    "blocks", "aerial_duels_won", "saves",
]
PLAYER_ROLLING_WINDOWS: list[int] = [3, 5]
OPPONENT_ROLLING_WINDOWS: list[int] = [3, 5]
EWM_RAW_COLS: list[str] = ["xg", "xa", "xgot", "chances_created", "shots_on_target"]
EWM_SPAN: int = 5

# Fill defaults for features where 0 is semantically wrong
FEATURE_FILL_DEFAULTS: dict[str, float] = {
    "opponent_elo": 1500.0,
    "fdr": 3.0,
}


# ---------------------------------------------------------------------------
# Singleton instances (importable as `from src.config import xgb, ensemble, ...`)
# ---------------------------------------------------------------------------
xgb = XGBConfig()
ensemble = EnsembleConfig()
solver_cfg = SolverConfig()
cache_cfg = CacheConfig()
prediction = PredictionConfig()
fpl_scoring = FPLScoringRules()
decomposed = DecomposedConfig()
data_cfg = DataConfig()
strategy_cfg = StrategyConfig()
