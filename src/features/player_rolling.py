"""Player-level rolling stat features from player match stats (PMS).

Computes rolling averages over recent gameweeks for each player.
All rolling features use shift(1) to prevent data leakage — a row
at GW N only contains data from GW N-1 and earlier.
"""

from __future__ import annotations

import pandas as pd

from src.config import PLAYER_ROLLING_COLS, PLAYER_ROLLING_WINDOWS
from src.logging_config import get_logger

log = get_logger(__name__)


def add_gameweek_to_pms(pms: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add gameweek column to playermatchstats by joining on match_id."""
    if "gameweek" in pms.columns:
        return pms
    match_gw = matches[["match_id", "gameweek"]].drop_duplicates()
    return pms.merge(match_gw, on="match_id", how="left")


def build_player_rolling_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages of player stats over recent gameweeks.

    Returns a DataFrame with player_id, gameweek, and rolling features.
    Each row represents the rolling stats *as of* that gameweek (using data
    from prior gameweeks only — no data leakage).

    DGW handling: stats are summed per player per GW before rolling, so
    DGW output counts fully in rolling windows rather than being averaged down.
    """
    # Ensure numeric and fill NaN with 0
    for col in PLAYER_ROLLING_COLS:
        if col in pms.columns:
            pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    available_cols = [c for c in PLAYER_ROLLING_COLS if c in pms.columns]

    # Aggregate per player per gameweek (handles DGW — sum)
    agg = pms.groupby(["player_id", "gameweek"])[available_cols].sum().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    # Add composite CBIT (clearances + blocks + interceptions + tackles_won)
    # for defensive contribution (DefCon) prediction
    _cbit_cols = ["clearances", "blocks", "interceptions", "tackles_won"]
    _cbit_available = [c for c in _cbit_cols if c in agg.columns]
    if _cbit_available:
        agg["cbit"] = agg[_cbit_available].sum(axis=1)
        available_cols = available_cols + ["cbit"]

    # Compute rolling averages (shift by 1 to avoid leakage — only past data)
    result_frames = [agg[["player_id", "gameweek"]]]
    for window in PLAYER_ROLLING_WINDOWS:
        for col in available_cols:
            feat_name = f"player_{col}_last{window}"
            rolled = (
                agg.groupby("player_id")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    return pd.concat(result_frames, axis=1)


def build_ewm_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute exponentially weighted means of key raw stats.

    Applies shift(1) + ewm(span=5) directly on per-GW aggregated raw stats
    so the EWM operates on actual match data rather than already-smoothed
    rolling averages (which would double-smooth the signal).

    Returns DataFrame with player_id, gameweek, and ewm features.
    """
    from src.config import EWM_RAW_COLS, EWM_SPAN

    available = [c for c in EWM_RAW_COLS if c in pms.columns]
    if not available:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # Aggregate per player per GW (handles DGW — sum for full GW output)
    agg = pms.groupby(["player_id", "gameweek"])[available].sum().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    result = agg[["player_id", "gameweek"]].copy()
    for col in available:
        # Map raw stat name to the feature name used in DEFAULT_FEATURES
        feat_name = f"ewm_player_{col}_last3"
        result[feat_name] = (
            agg.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).ewm(span=EWM_SPAN, min_periods=1).mean())
        )

    return result
