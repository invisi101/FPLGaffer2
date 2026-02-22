"""Opponent-adjusted interaction features.

These combine a player's attacking output with the opponent's defensive
weakness to capture match-up-specific scoring opportunity.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features to the assembled DataFrame in-place.

    Creates:
      - xg_x_opp_goals_conceded: player xG * opponent goals conceded
      - chances_x_opp_big_chances: player chances created * opponent big chances allowed
      - cs_opportunity: 1 / (opponent attacking xG + 0.1)

    Returns the same DataFrame with new columns added.
    """
    if "player_xg_last3" in df.columns and "opp_goals_conceded_last3" in df.columns:
        df["xg_x_opp_goals_conceded"] = df["player_xg_last3"] * df["opp_goals_conceded_last3"]

    if "player_chances_created_last3" in df.columns and "opp_big_chances_allowed_last3" in df.columns:
        df["chances_x_opp_big_chances"] = (
            df["player_chances_created_last3"] * df["opp_big_chances_allowed_last3"]
        )

    # cs_opportunity: inverse of opponent's attacking xG (low opp attack = high CS chance)
    # Prefer opp_xg_last3 (direct attacking output) over opp_opponent_xg_last3 (xG conceded)
    if "opp_xg_last3" in df.columns:
        df["cs_opportunity"] = 1.0 / (df["opp_xg_last3"] + 0.1)
    elif "opp_opponent_xg_last3" in df.columns:
        df["cs_opportunity"] = 1.0 / (df["opp_opponent_xg_last3"] + 0.1)

    return df
