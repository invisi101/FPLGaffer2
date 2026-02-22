"""Target variable construction for model training.

Includes both the main regression target (next_gw_points) and the
decomposed per-component targets (goals, assists, CS, bonus, etc.)
used by the decomposed sub-models.

CRITICAL: These targets are shifted forward so that a row at GW N
contains what happens in GW N+1.  Any change here directly affects
model correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_config import get_logger

log = get_logger(__name__)


def build_targets(playerstats: pd.DataFrame) -> pd.DataFrame:
    """Build target variables: next_gw_points and next_3gw_points.

    Uses explicit gameweek-number lookup instead of row-shift so that
    gaps in a player's gameweek sequence (missed GWs, blanks) don't
    misalign the target.
    """
    ps = playerstats[["id", "gw", "event_points"]].copy()
    ps = ps.rename(columns={"id": "player_id", "gw": "gameweek"})
    ps["event_points"] = pd.to_numeric(ps["event_points"], errors="coerce").fillna(0)
    ps = ps.sort_values(["player_id", "gameweek"])

    # Build a lookup: (player_id, gameweek) -> event_points
    pts_lookup = ps.set_index(["player_id", "gameweek"])["event_points"]

    # next_gw_points: points scored in gameweek + 1
    ps["next_gw_points"] = ps.apply(
        lambda r: pts_lookup.get((r["player_id"], r["gameweek"] + 1)), axis=1
    )

    # next_3gw_points: sum of points in gameweek+1, +2, +3
    for offset in [1, 2, 3]:
        ps[f"pts_gw_plus{offset}"] = ps.apply(
            lambda r, o=offset: pts_lookup.get((r["player_id"], r["gameweek"] + o)),
            axis=1,
        )
    shift_cols = ["pts_gw_plus1", "pts_gw_plus2", "pts_gw_plus3"]
    # Only valid if all 3 future GWs exist for this player
    ps["next_3gw_points"] = ps[shift_cols].sum(axis=1)
    ps.loc[ps[shift_cols].isna().any(axis=1), "next_3gw_points"] = np.nan
    ps = ps.drop(columns=shift_cols)

    return ps[["player_id", "gameweek", "next_gw_points", "next_3gw_points"]]


def build_decomposed_targets(
    pms: pd.DataFrame,
    playerstats: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-component targets for decomposed scoring models.

    Creates next-GW targets for each FPL point source from PMS (per-match data),
    playerstats (cumulative data), and matches (team results for clean sheets).
    Targets are aligned so that a row at gameweek=N contains what happens in
    gameweek N+1.

    Returns DataFrame with player_id, gameweek, and component targets:
      next_gw_minutes, next_gw_goals, next_gw_assists, next_gw_cs,
      next_gw_bonus, next_gw_goals_conceded, next_gw_saves
    """
    # --- Per-GW stats from PMS (available both seasons) ---
    pms_cols: dict[str, str] = {
        "goals": "gw_goals",
        "assists": "gw_assists",
        "minutes_played": "gw_minutes",
        "saves": "gw_saves",
    }
    # Add CBIT (defensive contributions) to PMS columns
    _cbit_cols = ["clearances", "blocks", "interceptions", "tackles_won"]
    _cbit_available = [c for c in _cbit_cols if c in pms.columns]
    if _cbit_available:
        for c in _cbit_available:
            pms[c] = pd.to_numeric(pms[c], errors="coerce").fillna(0)
        pms["cbit"] = pms[_cbit_available].sum(axis=1)
        pms_cols["cbit"] = "gw_cbit"
        # CBIRT = CBIT + recoveries (used for MID/FWD DefCon at threshold 12)
        if "recoveries" in pms.columns:
            pms["recoveries"] = pd.to_numeric(pms["recoveries"], errors="coerce").fillna(0)
            pms["cbirt"] = pms["cbit"] + pms["recoveries"]
        else:
            pms["cbirt"] = pms["cbit"]
        pms_cols["cbirt"] = "gw_cbirt"

    available_pms = {k: v for k, v in pms_cols.items() if k in pms.columns}

    if available_pms and "gameweek" in pms.columns:
        for col in available_pms:
            pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

        # Aggregate per player per GW (sum across matches for DGW)
        pms_agg = pms.groupby(["player_id", "gameweek"])[
            list(available_pms.keys())
        ].sum().reset_index()
        pms_agg = pms_agg.rename(columns=available_pms)

        # --- Clean sheet: derive from match results + player minutes ---
        pms_agg = _build_clean_sheet_targets(pms, pms_agg, matches, players)
    else:
        pms_agg = pd.DataFrame(columns=["player_id", "gameweek"])

    # --- Bonus from playerstats cumulative diff (available both seasons) ---
    if "bonus" in playerstats.columns:
        pms_agg = _add_bonus_targets(pms_agg, playerstats)

    if pms_agg.empty:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # --- Expand to full player-GW universe from playerstats ---
    all_player_gws = playerstats[["id", "gw"]].copy()
    all_player_gws = all_player_gws.rename(columns={"id": "player_id", "gw": "gameweek"})
    all_player_gws = all_player_gws.drop_duplicates(subset=["player_id", "gameweek"])

    pms_agg = all_player_gws.merge(pms_agg, on=["player_id", "gameweek"], how="left")
    gw_cols = [c for c in pms_agg.columns if c.startswith("gw_")]
    # Fill most gw_ columns with 0 for non-starters. But gw_goals_conceded
    # should stay NaN for non-starters and transferred players.
    gw_fill_cols = [c for c in gw_cols if c != "gw_goals_conceded"]
    for col in gw_fill_cols:
        pms_agg[col] = pms_agg[col].fillna(0)

    pms_agg = pms_agg.sort_values(["player_id", "gameweek"])

    # --- Build next-GW targets via lookup (same approach as build_targets) ---
    for col in gw_cols:
        target_name = col.replace("gw_", "next_gw_")
        lookup = pms_agg.set_index(["player_id", "gameweek"])[col]
        pms_agg[target_name] = pms_agg.apply(
            lambda r, lu=lookup, tn=target_name: lu.get(
                (r["player_id"], r["gameweek"] + 1)
            ),
            axis=1,
        )

    target_cols = [c for c in pms_agg.columns if c.startswith("next_gw_")]
    return pms_agg[["player_id", "gameweek"] + target_cols]


def _build_clean_sheet_targets(
    pms: pd.DataFrame,
    pms_agg: pd.DataFrame,
    matches: pd.DataFrame | None,
    players: pd.DataFrame | None,
) -> pd.DataFrame:
    """Derive clean sheet and goals_conceded targets from match results.

    PMS goals_conceded is only populated for GKP, so instead we check
    whether the player's team conceded 0 goals in each match and the
    player played 60+ minutes.
    """
    if (
        matches is None or matches.empty
        or players is None or players.empty
        or "minutes_played" not in pms.columns
        or "match_id" not in pms.columns
    ):
        return pms_agg

    # Build per-match team goals conceded from match results
    match_gc_rows: list[dict] = []
    for _, m in matches.iterrows():
        mid = m.get("match_id")
        if pd.isna(mid):
            continue
        home_team = m.get("home_team")
        away_team = m.get("away_team")
        home_score = m.get("home_score", 0)
        away_score = m.get("away_score", 0)
        if pd.notna(home_team) and pd.notna(away_score):
            match_gc_rows.append(
                {"match_id": mid, "team_code": int(home_team),
                 "team_goals_conceded": int(away_score)}
            )
        if pd.notna(away_team) and pd.notna(home_score):
            match_gc_rows.append(
                {"match_id": mid, "team_code": int(away_team),
                 "team_goals_conceded": int(home_score)}
            )

    if not match_gc_rows:
        return pms_agg

    match_gc = pd.DataFrame(match_gc_rows)
    # Map player_id -> team_code
    pid_to_team = dict(zip(players["player_id"], players["team_code"]))
    pms_cs = pms[["player_id", "match_id", "gameweek", "minutes_played"]].copy()
    pms_cs["team_code"] = pms_cs["player_id"].map(pid_to_team)
    pms_cs = pms_cs.merge(match_gc, on=["match_id", "team_code"], how="left")
    # Don't fillna(0) — unknown goals_conceded should NOT count as clean sheet.
    pms_cs["_match_cs"] = (
        (pms_cs["team_goals_conceded"] == 0)
        & (pms_cs["minutes_played"] >= 60)
    ).astype(int)
    cs_grp = pms_cs.groupby(["player_id", "gameweek"])
    cs_agg = pd.DataFrame({
        "gw_cs": cs_grp["_match_cs"].sum(),
        # min_count=1: if ALL values are NaN, result is NaN instead of 0
        "gw_goals_conceded": cs_grp["team_goals_conceded"].sum(min_count=1),
    }).reset_index()
    pms_agg = pms_agg.merge(cs_agg, on=["player_id", "gameweek"], how="left")
    pms_agg["gw_cs"] = pms_agg["gw_cs"].fillna(0).astype(int)
    # Don't fillna(0) for gw_goals_conceded — NaN means unknown

    return pms_agg


def _add_bonus_targets(
    pms_agg: pd.DataFrame, playerstats: pd.DataFrame,
) -> pd.DataFrame:
    """Add bonus targets from playerstats cumulative diff."""
    ps_bonus = playerstats[["id", "gw", "bonus"]].copy()
    ps_bonus = ps_bonus.rename(columns={"id": "player_id", "gw": "gameweek"})
    ps_bonus["bonus"] = pd.to_numeric(ps_bonus["bonus"], errors="coerce").fillna(0)
    ps_bonus = ps_bonus.sort_values(["player_id", "gameweek"])
    # Forward-fill within each player to avoid accumulated deltas across GW gaps
    filled_bonus = ps_bonus.groupby("player_id")["bonus"].ffill()
    ps_bonus["gw_bonus"] = filled_bonus.groupby(ps_bonus["player_id"]).diff()
    # First GW diff is NaN — use the raw value (it IS the per-GW value for GW1)
    first_mask = ps_bonus["gw_bonus"].isna()
    ps_bonus.loc[first_mask, "gw_bonus"] = ps_bonus.loc[first_mask, "bonus"]
    ps_bonus["gw_bonus"] = ps_bonus["gw_bonus"].clip(lower=0)

    if not pms_agg.empty:
        pms_agg = pms_agg.merge(
            ps_bonus[["player_id", "gameweek", "gw_bonus"]],
            on=["player_id", "gameweek"], how="outer",
        )
    else:
        pms_agg = ps_bonus[["player_id", "gameweek", "gw_bonus"]].copy()

    return pms_agg
