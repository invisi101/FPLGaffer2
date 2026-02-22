"""Upside / explosive potential features.

Captures volatility, form acceleration, and big-chance frequency to
identify players who are likely to produce hauls rather than steady
low scores.  All features use shift(1) to prevent leakage.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_upside_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute features that capture explosive/upside potential.

    - xg_volatility_last5: std of xG over last 5 matches (high = explosive)
    - form_acceleration: xG last 3 minus xG last 5 (positive = upward trend)
    - big_chance_frequency_last5: rolling mean of (goals + big_chances_missed)

    DGW handling: stats are summed per player per GW before computing.
    Returns DataFrame with player_id, gameweek, and upside features.
    """
    needed = ["xg", "goals", "big_chances_missed"]
    available = [c for c in needed if c in pms.columns]
    if "xg" not in available:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    for col in available:
        pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    # Aggregate per player per GW (handles DGW — sum for full GW output)
    agg = pms.groupby(["player_id", "gameweek"])[available].sum().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    result = agg[["player_id", "gameweek"]].copy()

    # xG volatility: std of last 5 shifted xG values
    result["xg_volatility_last5"] = (
        agg.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    )

    # Form acceleration: difference between short and long rolling xG
    xg_last3 = agg.groupby("player_id")["xg"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    xg_last5 = agg.groupby("player_id")["xg"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    result["form_acceleration"] = xg_last3 - xg_last5

    # Big chance frequency: rolling mean of (goals + big_chances_missed)
    if "goals" in available and "big_chances_missed" in available:
        agg["_big_chance_total"] = agg["goals"] + agg["big_chances_missed"]
        result["big_chance_frequency_last5"] = (
            agg.groupby("player_id")["_big_chance_total"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    return result
