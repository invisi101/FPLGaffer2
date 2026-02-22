"""Fixture context features — maps, FDR, Elo, and multi-GW lookahead."""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_fixture_map(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping of team_code -> gameweek -> opponent_code, is_home."""
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
        rows.append({"team_code": int(home), "gameweek": gw, "opponent_code": int(away), "is_home": 1})
        rows.append({"team_code": int(away), "gameweek": gw, "opponent_code": int(home), "is_home": 0})
    return pd.DataFrame(rows)


def build_fdr_map(api_fixtures: list[dict]) -> pd.DataFrame:
    """Build FDR (Fixture Difficulty Rating) from FPL API fixtures."""
    rows: list[dict] = []
    for fx in api_fixtures:
        event = fx.get("event")
        if event is None:
            continue
        # Home team
        rows.append({
            "team_id": fx["team_h"],
            "gameweek": event,
            "fdr": fx.get("team_h_difficulty", 3),
            "opponent_team_id": fx["team_a"],
        })
        # Away team
        rows.append({
            "team_id": fx["team_a"],
            "gameweek": event,
            "fdr": fx.get("team_a_difficulty", 3),
            "opponent_team_id": fx["team_h"],
        })
    return pd.DataFrame(rows)


def build_elo_features(teams: pd.DataFrame) -> pd.DataFrame:
    """Extract Elo ratings per team."""
    cols = ["code", "elo"]
    available = [c for c in cols if c in teams.columns]
    if "elo" not in available:
        return pd.DataFrame()
    result = teams[available].copy()
    result = result.rename(columns={"code": "team_code", "elo": "team_elo"})
    return result


def build_team_id_to_code_map(teams: pd.DataFrame) -> dict[int, int]:
    """Map FPL team id -> team code."""
    return dict(zip(teams["id"], teams["code"]))


def build_next3_features(
    fixture_map: pd.DataFrame,
    fdr_map: pd.DataFrame,
    elo: pd.DataFrame,
) -> pd.DataFrame:
    """Compute lookahead features over the next 3 GWs for the 3-GW target.

    Returns DataFrame with team_code, gameweek, and:
      - fixture_count_next3: total fixtures in next 3 GWs
      - home_pct_next3: fraction of those fixtures at home
      - avg_fdr_next3: average FDR across next 3 GWs
      - avg_opponent_elo_next3: average opponent Elo across next 3 GWs
    """
    if fixture_map.empty:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    fm = fixture_map.copy()
    if not fdr_map.empty and "team_code" in fdr_map.columns and "opponent_code" in fdr_map.columns:
        # Use per-fixture FDR keyed by opponent so DGW fixtures get
        # their individual difficulty ratings, not just the first one
        fdr_lookup = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
        fdr_lookup = fdr_lookup.dropna(subset=["opponent_code"])
        fdr_lookup["opponent_code"] = fdr_lookup["opponent_code"].astype(int)
        fdr_lookup = fdr_lookup.drop_duplicates(
            subset=["team_code", "gameweek", "opponent_code"], keep="first"
        )
        fm = fm.merge(fdr_lookup, on=["team_code", "gameweek", "opponent_code"], how="left")

    if not elo.empty and "team_elo" in elo.columns:
        elo_dict = dict(zip(elo["team_code"], elo["team_elo"]))
        fm["opponent_elo"] = fm["opponent_code"].map(elo_dict)

    rows: list[dict] = []
    for team, tf in fm.groupby("team_code"):
        tf = tf.sort_values("gameweek")
        gws = sorted(tf["gameweek"].unique())
        for gw in gws:
            ahead = tf[(tf["gameweek"] > gw) & (tf["gameweek"] <= gw + 3)]
            if ahead.empty:
                continue
            entry: dict = {"team_code": int(team), "gameweek": int(gw)}
            entry["fixture_count_next3"] = len(ahead)
            entry["home_pct_next3"] = float(ahead["is_home"].mean())
            if "fdr" in ahead.columns:
                fdr_vals = ahead["fdr"].fillna(3.0)
                entry["avg_fdr_next3"] = float(fdr_vals.mean())
            if "opponent_elo" in ahead.columns and ahead["opponent_elo"].notna().any():
                entry["avg_opponent_elo_next3"] = float(ahead["opponent_elo"].mean())
            rows.append(entry)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["team_code", "gameweek"])
