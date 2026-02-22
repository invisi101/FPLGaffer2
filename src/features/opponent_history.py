"""Player opponent-specific history features.

Computes a player's historical performance against each specific opponent
using expanding means.  Since matches vs a specific opponent are rare
(2-4 per 2 seasons), expanding mean is more appropriate than rolling.
shift(1) prevents leakage.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_opponent_history_features(
    pms: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Compute player's historical performance vs each specific opponent.

    Returns DataFrame with player_id, gameweek, opponent_code,
    vs_opponent_xg_avg, vs_opponent_goals_avg, vs_opponent_matches.
    """
    if pms.empty or matches.empty:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    if not all(c in pms.columns for c in ["player_id", "match_id", "gameweek"]):
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    # Build match -> (team -> opponent) mapping
    match_opponents: list[dict] = []
    for _, m in matches.iterrows():
        mid = m.get("match_id")
        if pd.isna(mid):
            continue
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.notna(home) and pd.notna(away):
            match_opponents.append({"match_id": mid, "team_code": int(home), "opponent_code": int(away)})
            match_opponents.append({"match_id": mid, "team_code": int(away), "opponent_code": int(home)})

    if not match_opponents:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    opp_df = pd.DataFrame(match_opponents)

    pm = pms[["player_id", "match_id", "gameweek"]].copy()
    for col in ["xg", "goals"]:
        if col in pms.columns:
            pm[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)
        else:
            pm[col] = 0

    # Get player's team from players table
    if not players.empty and "player_id" in players.columns and "team_code" in players.columns:
        player_team = players[["player_id", "team_code"]].drop_duplicates(subset=["player_id"])
        pm = pm.merge(player_team, on="player_id", how="left")
    else:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    pm = pm.dropna(subset=["team_code"])
    pm["team_code"] = pm["team_code"].astype(int)

    # Merge to get opponent per match
    pm = pm.merge(opp_df, on=["match_id", "team_code"], how="left")
    pm = pm.dropna(subset=["opponent_code"])
    pm["opponent_code"] = pm["opponent_code"].astype(int)
    pm = pm.sort_values(["player_id", "opponent_code", "gameweek"])

    # Expanding mean EXCLUDING current match per player-opponent pair.
    # shift(1) ensures GW N features only include data from GW N-1 and before.
    pm["vs_opponent_xg_avg"] = (
        pm.groupby(["player_id", "opponent_code"])["xg"]
        .transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    )
    pm["vs_opponent_goals_avg"] = (
        pm.groupby(["player_id", "opponent_code"])["goals"]
        .transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
    )
    # Total matches played against this opponent (excluding current)
    pm["vs_opponent_matches"] = (
        pm.groupby(["player_id", "opponent_code"]).cumcount()
    )

    result = pm[["player_id", "gameweek", "opponent_code",
                 "vs_opponent_xg_avg", "vs_opponent_goals_avg",
                 "vs_opponent_matches"]].copy()

    return result
