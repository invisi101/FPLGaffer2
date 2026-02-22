"""Team-level rolling stat features — own-team and opponent.

Converts match results into per-team-per-GW rows with defensive/offensive
stats, then computes rolling averages for opponent difficulty assessment and
own-team strength.  All rolling features use shift(1) to prevent leakage.
"""

from __future__ import annotations

import pandas as pd

from src.config import OPPONENT_ROLLING_WINDOWS
from src.logging_config import get_logger

log = get_logger(__name__)


def _safe(row: pd.Series, col: str, default: float = 0) -> float:
    """Extract a value from a row, returning *default* for NaN/missing."""
    val = row.get(col, default)
    return default if pd.isna(val) else val


def build_team_match_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert matches into per-team-per-gameweek rows with defensive/offensive stats.

    For each match, creates two rows: one for the home team, one for the away team.
    """
    rows: list[dict] = []
    for _, m in matches.iterrows():
        gw = m.get("gameweek")
        if pd.isna(gw):
            continue
        gw = int(gw)
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.isna(home) or pd.isna(away):
            continue

        # Home team perspective
        rows.append({
            "team_code": int(home), "gameweek": gw, "is_home": True,
            "goals_scored": _safe(m, "home_score"),
            "goals_conceded": _safe(m, "away_score"),
            "xg": _safe(m, "home_expected_goals_xg"),
            "xg_conceded": _safe(m, "away_expected_goals_xg"),
            "big_chances": _safe(m, "home_big_chances"),
            "big_chances_allowed": _safe(m, "away_big_chances"),
            "shots_inside_box": _safe(m, "home_shots_inside_box"),
            "shots_inside_box_allowed": _safe(m, "away_shots_inside_box"),
            "shots_on_target": _safe(m, "home_shots_on_target"),
            "accurate_crosses": _safe(m, "home_accurate_crosses"),
            "accurate_crosses_allowed": _safe(m, "away_accurate_crosses"),
            "clean_sheet": 1 if _safe(m, "away_score") == 0 else 0,
            "opponent_xg": _safe(m, "away_expected_goals_xg"),
            "opponent_big_chances": _safe(m, "away_big_chances"),
            "opponent_shots_on_target": _safe(m, "away_shots_on_target"),
        })
        # Away team perspective
        rows.append({
            "team_code": int(away), "gameweek": gw, "is_home": False,
            "goals_scored": _safe(m, "away_score"),
            "goals_conceded": _safe(m, "home_score"),
            "xg": _safe(m, "away_expected_goals_xg"),
            "xg_conceded": _safe(m, "home_expected_goals_xg"),
            "big_chances": _safe(m, "away_big_chances"),
            "big_chances_allowed": _safe(m, "home_big_chances"),
            "shots_inside_box": _safe(m, "away_shots_inside_box"),
            "shots_inside_box_allowed": _safe(m, "home_shots_inside_box"),
            "shots_on_target": _safe(m, "away_shots_on_target"),
            "accurate_crosses": _safe(m, "away_accurate_crosses"),
            "accurate_crosses_allowed": _safe(m, "home_accurate_crosses"),
            "clean_sheet": 1 if _safe(m, "home_score") == 0 else 0,
            "opponent_xg": _safe(m, "home_expected_goals_xg"),
            "opponent_big_chances": _safe(m, "home_big_chances"),
            "opponent_shots_on_target": _safe(m, "home_shots_on_target"),
        })

    return pd.DataFrame(rows)


def build_opponent_rolling_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling team stats for opponent difficulty assessment.

    Returns per-team-per-gameweek rolling stats using only past data.
    DGW handling: stats are summed per team per GW before rolling.
    """
    team_stats = team_stats.sort_values(["team_code", "gameweek"])

    roll_cols = [
        "goals_conceded", "xg_conceded", "big_chances_allowed",
        "shots_inside_box_allowed", "accurate_crosses_allowed",
        "clean_sheet", "opponent_xg", "opponent_big_chances",
        "opponent_shots_on_target",
        # Opponent attacking output — directly predicts CS probability
        "goals_scored", "xg",
    ]

    # Aggregate per team per GW (handles DGW — sum for full GW output)
    agg = team_stats.groupby(["team_code", "gameweek"])[roll_cols].sum().reset_index()
    agg = agg.sort_values(["team_code", "gameweek"])

    result_frames = [agg[["team_code", "gameweek"]]]
    for window in OPPONENT_ROLLING_WINDOWS:
        for col in roll_cols:
            feat_name = f"opp_{col}_last{window}"
            rolled = (
                agg.groupby("team_code")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    return pd.concat(result_frames, axis=1)


def build_own_team_rolling_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling own-team attacking/defensive stats.

    Returns per-team-per-gameweek rolling stats using only past data.
    DGW handling: stats are summed per team per GW before rolling.
    """
    team_stats = team_stats.sort_values(["team_code", "gameweek"])
    own_cols = ["goals_scored", "xg", "big_chances", "shots_on_target", "clean_sheet"]
    available = [c for c in own_cols if c in team_stats.columns]

    agg = team_stats.groupby(["team_code", "gameweek"])[available].sum().reset_index()
    agg = agg.sort_values(["team_code", "gameweek"])

    result_frames = [agg[["team_code", "gameweek"]]]
    for window in [3, 5]:
        for col in available:
            feat_name = f"team_{col}_last{window}"
            rolled = (
                agg.groupby("team_code")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    return pd.concat(result_frames, axis=1)
