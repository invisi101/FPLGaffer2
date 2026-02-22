"""Rest days and fixture congestion features.

Uses kickoff_time from matches to compute days rest between matches and
fixture congestion rate.  Per-match granularity ensures DGW fixtures
get individual rest values rather than being collapsed.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_rest_days_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute days rest and fixture congestion per team per gameweek.

    Uses kickoff_time from matches. days_rest is computed per-match so
    DGW fixtures get individual values. fixture_congestion is the
    inverse of rolling 3-match average rest.

    Returns DataFrame with team_code, gameweek, opponent_code, days_rest,
    fixture_congestion.
    """
    if "kickoff_time" not in matches.columns:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    m = matches.copy()
    m["kickoff_dt"] = pd.to_datetime(m["kickoff_time"], errors="coerce")
    m = m.dropna(subset=["kickoff_dt", "gameweek"])

    # Build per-team rows from both home and away perspectives.
    # Include opponent_code so DGW rest can be merged per-fixture.
    rows: list[dict] = []
    for _, row in m.iterrows():
        gw = int(row["gameweek"])
        dt = row["kickoff_dt"]
        home = row.get("home_team")
        away = row.get("away_team")
        if pd.notna(home) and pd.notna(away):
            rows.append({"team_code": int(home), "gameweek": gw, "kickoff_dt": dt, "opponent_code": int(away)})
            rows.append({"team_code": int(away), "gameweek": gw, "kickoff_dt": dt, "opponent_code": int(home)})
        elif pd.notna(home):
            rows.append({"team_code": int(home), "gameweek": gw, "kickoff_dt": dt, "opponent_code": -1})
        elif pd.notna(away):
            rows.append({"team_code": int(away), "gameweek": gw, "kickoff_dt": dt, "opponent_code": -1})

    if not rows:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    team_matches = pd.DataFrame(rows)

    # Keep per-match rows so DGW fixtures get individual rest values.
    team_matches = team_matches.sort_values(["team_code", "kickoff_dt"])

    # Days since previous match (per-match, not per-GW)
    team_matches["days_rest"] = (
        team_matches.groupby("team_code")["kickoff_dt"]
        .diff()
        .dt.total_seconds() / 86400.0
    )
    team_matches["days_rest"] = team_matches["days_rest"].fillna(7.0)

    # Fixture congestion = inverse of rolling 3-match average rest (per-match)
    team_matches["fixture_congestion"] = (
        team_matches.groupby("team_code")["days_rest"]
        .transform(lambda s: 1.0 / s.rolling(3, min_periods=1).mean())
    )

    return team_matches[["team_code", "gameweek", "opponent_code", "days_rest", "fixture_congestion"]]
