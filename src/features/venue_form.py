"""Home/away venue form features.

Computes separate rolling xG form for home and away appearances per player.
venue_matched_form (which picks the appropriate form based on the next
fixture's venue) is built during assembly when is_home is available.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_home_away_form(
    pms: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Compute separate home/away rolling xG form per player.

    Joins PMS with matches to determine venue, then computes rolling 5-match
    xG averages separately for home and away appearances.  Forward-fills each
    so that rows always have a value.

    Returns DataFrame with player_id, gameweek, home_xg_form, away_xg_form.
    """
    if pms.empty or matches.empty:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    if "xg" not in pms.columns or "match_id" not in pms.columns:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # Get venue (is_home) for each match from the match data
    match_venue: list[dict] = []
    for _, m in matches.iterrows():
        mid = m.get("match_id")
        if pd.isna(mid):
            continue
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.notna(home):
            match_venue.append({"match_id": mid, "team_code": int(home), "is_home_venue": 1})
        if pd.notna(away):
            match_venue.append({"match_id": mid, "team_code": int(away), "is_home_venue": 0})

    if not match_venue:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    venue_df = pd.DataFrame(match_venue)

    # Merge PMS with venue info — need player's team to determine venue
    pm = pms[["player_id", "match_id", "gameweek", "xg"]].copy()
    pm["xg"] = pd.to_numeric(pm["xg"], errors="coerce").fillna(0)

    # Get player's team from players table
    if not players.empty and "player_id" in players.columns and "team_code" in players.columns:
        player_team = players[["player_id", "team_code"]].drop_duplicates(subset=["player_id"])
        pm = pm.merge(player_team, on="player_id", how="left")
    else:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    pm = pm.dropna(subset=["team_code"])
    pm["team_code"] = pm["team_code"].astype(int)

    pm = pm.merge(venue_df, on=["match_id", "team_code"], how="left")
    pm = pm.dropna(subset=["is_home_venue"])

    # Aggregate per player per GW per venue
    agg = pm.groupby(["player_id", "gameweek", "is_home_venue"])["xg"].sum().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    # Compute rolling xG form for home games
    home = agg[agg["is_home_venue"] == 1].copy()
    home["home_xg_form"] = (
        home.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Compute rolling xG form for away games
    away = agg[agg["is_home_venue"] == 0].copy()
    away["away_xg_form"] = (
        away.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Merge home/away forms back per player-GW and forward-fill
    all_pgw = pm[["player_id", "gameweek"]].drop_duplicates().sort_values(["player_id", "gameweek"])
    all_pgw = all_pgw.merge(
        home[["player_id", "gameweek", "home_xg_form"]],
        on=["player_id", "gameweek"], how="left",
    )
    all_pgw = all_pgw.merge(
        away[["player_id", "gameweek", "away_xg_form"]],
        on=["player_id", "gameweek"], how="left",
    )
    all_pgw = all_pgw.sort_values(["player_id", "gameweek"])
    all_pgw["home_xg_form"] = all_pgw.groupby("player_id")["home_xg_form"].ffill()
    all_pgw["away_xg_form"] = all_pgw.groupby("player_id")["away_xg_form"].ffill()

    return all_pgw[["player_id", "gameweek", "home_xg_form", "away_xg_form"]]
