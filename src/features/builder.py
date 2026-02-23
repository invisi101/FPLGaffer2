"""Feature matrix orchestrator — assembles all feature modules into a single DataFrame.

This is the main entry point for feature engineering.  It calls all
feature modules, merges them together, handles cross-season forward-fill
with decay, and produces the final feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_FEATURES, SUB_MODEL_FEATURES
from src.logging_config import get_logger

from src.features.player_rolling import (
    add_gameweek_to_pms,
    build_player_rolling_features,
    build_ewm_features,
)
from src.features.playerstats import build_playerstats_features
from src.features.team_stats import (
    build_team_match_stats,
    build_opponent_rolling_features,
    build_own_team_rolling_features,
)
from src.features.fixture_context import (
    build_fixture_map,
    build_fdr_map,
    build_elo_features,
    build_team_id_to_code_map,
    build_next3_features,
)
from src.features.targets import build_targets, build_decomposed_targets
from src.features.interactions import build_interaction_features
from src.features.upside import build_upside_features
from src.features.opponent_history import build_opponent_history_features
from src.features.venue_form import build_home_away_form
from src.features.rest_congestion import build_rest_days_features

log = get_logger(__name__)

# Cross-season decay factor applied per GW to carried-over rolling values
CROSS_SEASON_DECAY = 0.90
# Off-season gap in weeks for decay distance calculation
OFF_SEASON_GAP = 10


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
def build_features(data: dict) -> pd.DataFrame:
    """Build the full feature matrix from raw data.

    Args:
        data: dict from load_all_data() with season keys, 'api',
              'current_season', 'seasons'

    Returns:
        DataFrame with one row per player per gameweek (per fixture for DGW),
        including features and targets.
    """
    all_frames: list[pd.DataFrame] = []

    current_season = data.get("current_season", "2025-2026")
    seasons = data.get("seasons", [current_season])

    for season_label in seasons:
        season = data.get(season_label, {})
        if not season:
            continue

        players = season.get("players", pd.DataFrame())
        pms = season.get("playermatchstats", pd.DataFrame())
        matches = season.get("matches", pd.DataFrame())
        playerstats = season.get("playerstats", pd.DataFrame())
        teams = season.get("teams", pd.DataFrame())

        if players.empty or playerstats.empty:
            log.info("Skipping %s: missing core data", season_label)
            continue

        log.info("Building features for %s...", season_label)

        # Filter to Premier League matches only
        matches, pms = _filter_prem_only(matches, pms)

        # Filter matches to finished ones only for historical data
        finished_matches = _get_finished_matches(matches)

        # Filter PMS to PL matches only
        if not pms.empty and not finished_matches.empty and "match_id" in pms.columns:
            prem_match_ids = set(matches["match_id"].dropna())
            pms_before = len(pms)
            pms = pms[pms["match_id"].isin(prem_match_ids)].copy()
            filtered_out = pms_before - len(pms)
            if filtered_out > 0:
                log.info("Filtering out %d non-PL player match stat rows", filtered_out)

        # 1. Add gameweek to playermatchstats
        if not pms.empty and not finished_matches.empty:
            pms = add_gameweek_to_pms(pms, finished_matches)
            pms = pms.dropna(subset=["gameweek"])
            pms["gameweek"] = pms["gameweek"].astype(int)

        # 2. Build all player-level features
        player_rolling = (
            build_player_rolling_features(pms.copy())
            if not pms.empty
            else pd.DataFrame(columns=["player_id", "gameweek"])
        )
        ewm_features = (
            build_ewm_features(pms.copy())
            if not pms.empty
            else pd.DataFrame(columns=["player_id", "gameweek"])
        )
        upside_features = (
            build_upside_features(pms.copy())
            if not pms.empty
            else pd.DataFrame(columns=["player_id", "gameweek"])
        )
        home_away_form = (
            build_home_away_form(pms, finished_matches, players)
            if not pms.empty and not finished_matches.empty
            else pd.DataFrame(columns=["player_id", "gameweek"])
        )
        opponent_history = (
            build_opponent_history_features(pms, finished_matches, players)
            if not pms.empty and not finished_matches.empty
            else pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])
        )

        # 3. Team-level features
        if not finished_matches.empty:
            team_stats = build_team_match_stats(finished_matches)
            opp_rolling = build_opponent_rolling_features(team_stats)
            own_team_rolling = build_own_team_rolling_features(team_stats)
        else:
            opp_rolling = pd.DataFrame(columns=["team_code", "gameweek"])
            own_team_rolling = pd.DataFrame(columns=["team_code", "gameweek"])

        # Rest days use ALL matches (including upcoming scheduled)
        rest_days = (
            build_rest_days_features(matches)
            if not matches.empty
            else pd.DataFrame(columns=["team_code", "gameweek"])
        )

        # Team id -> code mapping
        team_id_to_code = build_team_id_to_code_map(teams) if not teams.empty else {}

        # 4. Fixture map + future fixtures from API
        fixture_map = (
            build_fixture_map(matches)
            if not matches.empty
            else pd.DataFrame(columns=["team_code", "gameweek", "opponent_code", "is_home"])
        )
        if season_label == current_season:
            fixture_map = _supplement_future_fixtures(
                fixture_map, data, team_id_to_code
            )

        # 5. Playerstats features (form, BPS, ICT, cost, etc.)
        api_elements = None
        if season_label == current_season:
            api_elements = data.get("api", {}).get("bootstrap", {}).get("elements")
        ps_features = build_playerstats_features(playerstats, bootstrap_elements=api_elements)

        # 6. Targets
        targets = build_targets(playerstats)

        # 7. Elo ratings
        elo = build_elo_features(teams) if not teams.empty else pd.DataFrame()

        # 8. FDR from API (current season only)
        api_fixtures = data["api"].get("fixtures", []) if season_label == current_season else []
        fdr_map = build_fdr_map(api_fixtures) if api_fixtures else pd.DataFrame()
        if not fdr_map.empty and team_id_to_code:
            fdr_map["team_code"] = fdr_map["team_id"].map(team_id_to_code)
            fdr_map["opponent_code"] = fdr_map["opponent_team_id"].map(team_id_to_code)
            fdr_map = fdr_map.dropna(subset=["team_code"])
            fdr_map["team_code"] = fdr_map["team_code"].astype(int)

        # --- Assemble ---
        df = _assemble_features(
            ps_features=ps_features,
            players=players,
            fixture_map=fixture_map,
            fdr_map=fdr_map,
            player_rolling=player_rolling,
            ewm_features=ewm_features,
            upside_features=upside_features,
            home_away_form=home_away_form,
            opponent_history=opponent_history,
            opp_rolling=opp_rolling,
            own_team_rolling=own_team_rolling,
            rest_days=rest_days,
            elo=elo,
            targets=targets,
            pms=pms,
            playerstats=playerstats,
            finished_matches=finished_matches,
            next3=build_next3_features(fixture_map, fdr_map, elo),
            season_label=season_label,
        )

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Cross-season forward-fill with decay
    if len(all_frames) > 1:
        combined = _cross_season_ffill(combined)

    # Post-processing
    log.info("Total rows before target filter: %d", len(combined))
    log.info("Rows with next_gw_points: %d", combined["next_gw_points"].notna().sum())

    # Filter out non-playing players
    if "player_minutes_played_last5" in combined.columns:
        before = len(combined)
        combined = combined[combined["player_minutes_played_last5"] > 0].copy()
        log.info("Filtered out %d non-playing rows (0 mins in last 5 GWs)", before - len(combined))

    # Defragment
    combined = combined.copy()

    return combined


# -----------------------------------------------------------------------
# Assembly helper
# -----------------------------------------------------------------------
def _assemble_features(
    *,
    ps_features: pd.DataFrame,
    players: pd.DataFrame,
    fixture_map: pd.DataFrame,
    fdr_map: pd.DataFrame,
    player_rolling: pd.DataFrame,
    ewm_features: pd.DataFrame,
    upside_features: pd.DataFrame,
    home_away_form: pd.DataFrame,
    opponent_history: pd.DataFrame,
    opp_rolling: pd.DataFrame,
    own_team_rolling: pd.DataFrame,
    rest_days: pd.DataFrame,
    elo: pd.DataFrame,
    targets: pd.DataFrame,
    pms: pd.DataFrame,
    playerstats: pd.DataFrame,
    finished_matches: pd.DataFrame,
    next3: pd.DataFrame,
    season_label: str,
) -> pd.DataFrame:
    """Merge all feature sources into a single DataFrame for one season."""
    df = ps_features.copy()

    # Add player info (team, position)
    df = df.merge(
        players[["player_id", "team_code", "position"]],
        on="player_id", how="left"
    )

    # Add NEXT fixture info (opponent, is_home) — shifted so row at GW N
    # gets fixture for GW N+1
    if not fixture_map.empty:
        df = _merge_fixture_data(df, fixture_map, fdr_map)

    # Add player rolling features
    if not player_rolling.empty:
        df = df.merge(player_rolling, on=["player_id", "gameweek"], how="left")
        rolling_cols = [c for c in player_rolling.columns
                       if c.startswith("player_") and c != "player_id"]
        if rolling_cols:
            df = df.sort_values(["player_id", "gameweek"])
            df[rolling_cols] = df.groupby("player_id")[rolling_cols].ffill()

    # Add EWM features
    if not ewm_features.empty:
        df = df.merge(ewm_features, on=["player_id", "gameweek"], how="left")
        ewm_cols = [c for c in ewm_features.columns if c.startswith("ewm_")]
        if ewm_cols:
            df = df.sort_values(["player_id", "gameweek"])
            df[ewm_cols] = df.groupby("player_id")[ewm_cols].ffill()

    # Add upside features
    if not upside_features.empty:
        df = df.merge(upside_features, on=["player_id", "gameweek"], how="left")
        upside_cols = [c for c in upside_features.columns
                       if c not in ("player_id", "gameweek")]
        if upside_cols:
            df = df.sort_values(["player_id", "gameweek"])
            df[upside_cols] = df.groupby("player_id")[upside_cols].ffill()

    # Add home/away form
    if not home_away_form.empty:
        df = df.merge(home_away_form, on=["player_id", "gameweek"], how="left")
        for col in ["home_xg_form", "away_xg_form"]:
            if col in df.columns:
                df = df.sort_values(["player_id", "gameweek"])
                df[col] = df.groupby("player_id")[col].ffill()

    # Add opponent rolling features for the NEXT-GW opponent
    if not opp_rolling.empty and "opponent_code" in df.columns:
        opp_feats = opp_rolling.rename(columns={"team_code": "opponent_code"})
        df = df.merge(opp_feats, on=["opponent_code", "gameweek"], how="left")
        opp_cols = [c for c in opp_feats.columns if c.startswith("opp_")]
        if opp_cols:
            df = df.sort_values(["opponent_code", "gameweek"])
            df[opp_cols] = df.groupby("opponent_code")[opp_cols].ffill()

    # Add own-team rolling features
    if not own_team_rolling.empty:
        df = df.merge(own_team_rolling, on=["team_code", "gameweek"], how="left")
        own_team_cols = [c for c in own_team_rolling.columns
                        if c.startswith("team_") and c != "team_code"]
        if own_team_cols:
            df = df.sort_values(["team_code", "gameweek"])
            df[own_team_cols] = df.groupby("team_code")[own_team_cols].ffill()

    # Add rest days / fixture congestion (shifted: rest at GW N+1 on row GW N)
    if not rest_days.empty:
        rest_shifted = rest_days.copy()
        rest_shifted["gameweek"] = rest_shifted["gameweek"] - 1
        merge_keys = ["team_code", "gameweek"]
        if "opponent_code" in df.columns:
            merge_keys.append("opponent_code")
        df = df.merge(
            rest_shifted[merge_keys + ["days_rest", "fixture_congestion"]],
            on=merge_keys, how="left"
        )
        df["days_rest"] = df["days_rest"].fillna(7.0)
        df["fixture_congestion"] = df["fixture_congestion"].fillna(1.0 / 7.0)

    # Add opponent-specific history via merge_asof
    if not opponent_history.empty and "opponent_code" in df.columns:
        df = _merge_opponent_history(df, opponent_history)

    # Build venue_matched_form
    if "home_xg_form" in df.columns and "away_xg_form" in df.columns and "is_home" in df.columns:
        df["venue_matched_form"] = np.where(
            df["is_home"] == 1, df["home_xg_form"], df["away_xg_form"]
        )

    # Add opponent Elo
    if not elo.empty and "opponent_code" in df.columns:
        opp_elo = elo.rename(columns={"team_code": "opponent_code", "team_elo": "opponent_elo"})
        df = df.merge(opp_elo, on="opponent_code", how="left")

    # Multi-GW lookahead features
    if not next3.empty:
        df = df.merge(next3, on=["team_code", "gameweek"], how="left")

    # Add targets
    df = df.merge(targets, on=["player_id", "gameweek"], how="left")

    # Add decomposed targets
    if not pms.empty:
        decomposed = build_decomposed_targets(
            pms, playerstats, matches=finished_matches, players=players,
        )
        if not decomposed.empty:
            df = df.merge(decomposed, on=["player_id", "gameweek"], how="left")

    # Add season label
    df["season"] = season_label

    # Season progress indicator
    df["season_progress"] = df["gameweek"] / 38.0

    # Team form (rolling avg of team total event_points)
    if "event_points" in df.columns and "team_code" in df.columns:
        df = _add_team_form(df)

    # Interaction features
    df = build_interaction_features(df)

    # Position one-hot encoding
    pos_map = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}
    df["position_clean"] = df["position"].map(pos_map).fillna("UNK")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position_clean"] == pos).astype(int)

    # Minutes availability proxy
    if "player_minutes_played_last3" in df.columns:
        df["minutes_availability"] = df["player_minutes_played_last3"] / 90.0

    return df


# -----------------------------------------------------------------------
# Fixture merging
# -----------------------------------------------------------------------
def _merge_fixture_data(
    df: pd.DataFrame,
    fixture_map: pd.DataFrame,
    fdr_map: pd.DataFrame,
) -> pd.DataFrame:
    """Merge next-GW fixture data (opponent, is_home, FDR, fixture count)."""
    # Count fixtures per team per GW to detect DGWs (2+) and BGWs (0)
    fixture_counts = (
        fixture_map.groupby(["team_code", "gameweek"])
        .size()
        .reset_index(name="next_gw_fixture_count")
    )
    fixture_counts["gameweek"] = fixture_counts["gameweek"] - 1

    next_fixture = fixture_map[["team_code", "gameweek", "opponent_code", "is_home"]].copy()
    next_fixture["gameweek"] = next_fixture["gameweek"] - 1  # GW N+1 fixture -> attach to GW N row

    # Add per-fixture FDR
    if not fdr_map.empty and "opponent_code" in fdr_map.columns:
        fdr_for_fixture = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
        fdr_for_fixture = fdr_for_fixture.dropna(subset=["opponent_code"])
        fdr_for_fixture["opponent_code"] = fdr_for_fixture["opponent_code"].astype(int)
        fdr_for_fixture["gameweek"] = fdr_for_fixture["gameweek"] - 1
        fdr_for_fixture = fdr_for_fixture.drop_duplicates(
            subset=["team_code", "gameweek", "opponent_code"], keep="first"
        )
        next_fixture = next_fixture.merge(
            fdr_for_fixture,
            on=["team_code", "gameweek", "opponent_code"],
            how="left"
        )

    # No dedup — DGW players get one row per fixture
    df = df.merge(next_fixture, on=["team_code", "gameweek"], how="left")

    # Merge DGW fixture count
    df = df.merge(fixture_counts, on=["team_code", "gameweek"], how="left")

    # Handle BGW detection
    if not fixture_map.empty:
        known_gws = set(fixture_map["gameweek"].unique())
        if known_gws:
            max_known_gw = max(known_gws)
            bgw_mask = (
                df["next_gw_fixture_count"].isna()
                & (df["gameweek"] + 1 <= max_known_gw)
            )
            df.loc[bgw_mask, "next_gw_fixture_count"] = 0
    df["next_gw_fixture_count"] = df["next_gw_fixture_count"].fillna(1).astype(int)

    return df


def _merge_opponent_history(
    df: pd.DataFrame, opponent_history: pd.DataFrame,
) -> pd.DataFrame:
    """Merge opponent history via merge_asof to avoid leakage."""
    df["opponent_code"] = df["opponent_code"].astype("Int64")
    opponent_history["opponent_code"] = opponent_history["opponent_code"].astype("Int64")
    df = df.reset_index(drop=True)
    df["_orig_order"] = df.index
    df = df.sort_values("gameweek")
    opponent_history = opponent_history.sort_values("gameweek")
    df = pd.merge_asof(
        df,
        opponent_history,
        on="gameweek",
        by=["player_id", "opponent_code"],
        direction="backward",
    )
    df = df.sort_values("_orig_order").drop(columns=["_orig_order"])
    return df


def _add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add team_form_5: rolling avg of team total event_points."""
    # Deduplicate before summing so DGW players aren't double-counted
    team_pts = (
        df.drop_duplicates(subset=["player_id", "gameweek"], keep="first")
        .groupby(["team_code", "gameweek"])["event_points"]
        .sum()
        .reset_index()
        .sort_values(["team_code", "gameweek"])
    )
    team_pts["team_form_5"] = (
        team_pts.groupby("team_code")["event_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.merge(
        team_pts[["team_code", "gameweek", "team_form_5"]],
        on=["team_code", "gameweek"], how="left",
    )
    return df


# -----------------------------------------------------------------------
# Future fixture supplement
# -----------------------------------------------------------------------
def _supplement_future_fixtures(
    fixture_map: pd.DataFrame,
    data: dict,
    team_id_to_code: dict,
) -> pd.DataFrame:
    """Add future fixtures from FPL API to the fixture map."""
    api_fixtures_raw = data["api"].get("fixtures", [])
    if not api_fixtures_raw or not team_id_to_code:
        return fixture_map

    future_rows: list[dict] = []
    existing_keys: set[tuple] = set()
    if not fixture_map.empty:
        existing_keys = set(zip(
            fixture_map["team_code"], fixture_map["gameweek"],
            fixture_map["opponent_code"]
        ))
    for fx in api_fixtures_raw:
        event = fx.get("event")
        if event is None:
            continue
        h_code = team_id_to_code.get(fx.get("team_h"))
        a_code = team_id_to_code.get(fx.get("team_a"))
        if h_code is None or a_code is None:
            continue
        if (int(h_code), int(event), int(a_code)) not in existing_keys:
            future_rows.append({"team_code": int(h_code), "gameweek": int(event),
                                "opponent_code": int(a_code), "is_home": 1})
        if (int(a_code), int(event), int(h_code)) not in existing_keys:
            future_rows.append({"team_code": int(a_code), "gameweek": int(event),
                                "opponent_code": int(h_code), "is_home": 0})
    if future_rows:
        fixture_map = pd.concat([fixture_map, pd.DataFrame(future_rows)], ignore_index=True)

    return fixture_map


# -----------------------------------------------------------------------
# Cross-season forward-fill with decay
# -----------------------------------------------------------------------
def _cross_season_ffill(combined: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill rolling features across seasons with per-GW decay."""

    def _ffill_with_decay(group_df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
        """Forward-fill across seasons with per-GW decay on carried values."""
        group_df = group_df.sort_values([group_col, "season", "gameweek"])

        # Build a monotonically increasing GW counter across seasons
        season_order = {s: i for i, s in enumerate(
            sorted(group_df["season"].unique())
        )}
        group_df["_global_gw"] = (
            group_df["season"].map(season_order) * (38 + OFF_SEASON_GAP) + group_df["gameweek"]
        )

        for col in cols:
            filled = group_df.groupby(group_col)[col].ffill()
            was_nan = group_df[col].isna()
            is_filled = filled.notna()
            carried = was_nan & is_filled

            if carried.any():
                # Only apply decay to CROSS-SEASON carry-over
                last_real_season = group_df.groupby(group_col)["season"].transform(
                    lambda s: s.where(group_df.loc[s.index, col].notna()).ffill()
                )
                cross_season = carried & (group_df["season"] != last_real_season)

                if cross_season.any():
                    last_real_gw = group_df.groupby(group_col)["_global_gw"].transform(
                        lambda s: s.where(group_df.loc[s.index, col].notna()).ffill()
                    )
                    distance = group_df["_global_gw"] - last_real_gw
                    decay = CROSS_SEASON_DECAY ** distance.clip(lower=0)
                    group_df[col] = np.where(cross_season, filled * decay, filled)
                else:
                    group_df[col] = filled
            else:
                group_df[col] = filled

        group_df = group_df.drop(columns=["_global_gw"])
        return group_df

    combined = combined.sort_values(["player_id", "season", "gameweek"])

    # Player rolling features
    player_rolling_cols = [c for c in combined.columns
                           if c.startswith("player_") and "_last" in c]
    if player_rolling_cols:
        combined = _ffill_with_decay(combined, player_rolling_cols, "player_id")

    # Opponent rolling features
    opp_rolling_cols = [c for c in combined.columns
                        if c.startswith("opp_") and "_last" in c]
    if opp_rolling_cols and "opponent_code" in combined.columns:
        combined = _ffill_with_decay(combined, opp_rolling_cols, "opponent_code")

    # Team form
    if "team_form_5" in combined.columns:
        combined = _ffill_with_decay(combined, ["team_form_5"], "team_code")

    # Own-team rolling features
    own_team_cols = [c for c in combined.columns
                     if c.startswith("team_") and "_last" in c]
    if own_team_cols:
        combined = _ffill_with_decay(combined, own_team_cols, "team_code")

    # Upside features
    upside_cols = [c for c in combined.columns
                   if c in ("xg_volatility_last5", "form_acceleration",
                            "big_chance_frequency_last5")]
    if upside_cols:
        combined = _ffill_with_decay(combined, upside_cols, "player_id")

    # EWM features
    ewm_cols = [c for c in combined.columns if c.startswith("ewm_")]
    if ewm_cols:
        combined = _ffill_with_decay(combined, ewm_cols, "player_id")

    # Home/away form
    ha_form_cols = [c for c in ["home_xg_form", "away_xg_form"] if c in combined.columns]
    if ha_form_cols:
        combined = _ffill_with_decay(combined, ha_form_cols, "player_id")

    # Recompute venue_matched_form after cross-season ffill
    if "home_xg_form" in combined.columns and "away_xg_form" in combined.columns and "is_home" in combined.columns:
        combined["venue_matched_form"] = np.where(
            combined["is_home"] == 1, combined["home_xg_form"], combined["away_xg_form"]
        )

    # Transfer momentum
    if "transfer_momentum" in combined.columns:
        combined = _ffill_with_decay(combined, ["transfer_momentum"], "player_id")

    # Availability rate
    if "availability_rate_last5" in combined.columns:
        combined = _ffill_with_decay(combined, ["availability_rate_last5"], "player_id")

    return combined


# -----------------------------------------------------------------------
# Data filtering helpers
# -----------------------------------------------------------------------
def _filter_prem_only(
    matches: pd.DataFrame, pms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter matches and PMS to Premier League only."""
    if not matches.empty and "tournament" in matches.columns:
        non_prem = len(matches) - (matches["tournament"] == "prem").sum()
        if non_prem > 0:
            log.info("Filtering out %d non-PL matches (CL, EFL Cup, etc.)", non_prem)
            matches = matches[matches["tournament"] == "prem"].copy()
    return matches, pms


def _get_finished_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Return only finished matches from the matches DataFrame."""
    if not matches.empty and "finished" in matches.columns:
        return matches[matches["finished"].astype(str).str.lower() == "true"].copy()
    return matches.copy()


# -----------------------------------------------------------------------
# Fixture context extraction (used by predict.py for multi-GW)
# -----------------------------------------------------------------------
def get_fixture_context(data: dict) -> dict:
    """Extract fixture context needed for multi-GW predictions.

    Returns dict with:
      - fixture_map: DataFrame(team_code, gameweek, opponent_code, is_home)
      - fdr_map: DataFrame(team_code, gameweek, opponent_code, fdr)
      - elo: DataFrame(team_code, team_elo)
      - opp_rolling: DataFrame(opponent_code, gameweek, opp_* columns)
    """
    current_season = data.get("current_season", "2025-2026")
    season = data.get(current_season, {})
    matches = season.get("matches", pd.DataFrame())
    teams_df = season.get("teams", pd.DataFrame())
    team_id_to_code = build_team_id_to_code_map(teams_df) if not teams_df.empty else {}

    # Filter to PL matches
    if not matches.empty and "tournament" in matches.columns:
        matches = matches[matches["tournament"] == "prem"].copy()

    # Build fixture_map from played matches
    fixture_map = (
        build_fixture_map(matches)
        if not matches.empty
        else pd.DataFrame(columns=["team_code", "gameweek", "opponent_code", "is_home"])
    )

    # Add future fixtures from API
    api_fixtures_raw = data["api"].get("fixtures", [])
    if api_fixtures_raw and team_id_to_code:
        future_rows: list[dict] = []
        existing_keys: set[tuple] = set()
        if not fixture_map.empty:
            existing_keys = set(zip(
                fixture_map["team_code"], fixture_map["gameweek"],
                fixture_map["opponent_code"]
            ))
        for fx in api_fixtures_raw:
            event = fx.get("event")
            if event is None:
                continue
            h_code = team_id_to_code.get(fx.get("team_h"))
            a_code = team_id_to_code.get(fx.get("team_a"))
            if h_code is None or a_code is None:
                continue
            if (int(h_code), int(event), int(a_code)) not in existing_keys:
                future_rows.append({"team_code": int(h_code), "gameweek": int(event),
                                    "opponent_code": int(a_code), "is_home": 1})
            if (int(a_code), int(event), int(h_code)) not in existing_keys:
                future_rows.append({"team_code": int(a_code), "gameweek": int(event),
                                    "opponent_code": int(h_code), "is_home": 0})
        if future_rows:
            fixture_map = pd.concat([fixture_map, pd.DataFrame(future_rows)], ignore_index=True)

    # Build FDR map
    fdr_map = build_fdr_map(api_fixtures_raw) if api_fixtures_raw else pd.DataFrame()
    if not fdr_map.empty and team_id_to_code:
        fdr_map["team_code"] = fdr_map["team_id"].map(team_id_to_code)
        fdr_map["opponent_code"] = fdr_map["opponent_team_id"].map(team_id_to_code)
        fdr_map = fdr_map.dropna(subset=["team_code"])
        fdr_map["team_code"] = fdr_map["team_code"].astype(int)

    # Build Elo
    elo = build_elo_features(teams_df) if not teams_df.empty else pd.DataFrame()

    # Build opponent rolling features
    if not matches.empty:
        finished_matches = _get_finished_matches(matches)
        team_stats = build_team_match_stats(finished_matches)
        opp_rolling = build_opponent_rolling_features(team_stats)
    else:
        opp_rolling = pd.DataFrame(columns=["team_code", "gameweek"])

    return {
        "fixture_map": fixture_map,
        "fdr_map": fdr_map,
        "elo": elo,
        "opp_rolling": opp_rolling,
    }


# -----------------------------------------------------------------------
# Feature column extraction
# -----------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding targets, IDs, metadata)."""
    exclude = {
        "player_id", "gameweek", "season", "team_code", "opponent_code",
        "position", "position_clean", "next_gw_points", "next_3gw_points",
        "event_points", "web_name", "cumulative_minutes", "ep_next",
        # Decomposed targets (future data — must not be used as features)
        "next_gw_minutes", "next_gw_goals", "next_gw_assists", "next_gw_cs",
        "next_gw_bonus", "next_gw_goals_conceded", "next_gw_saves",
        "next_gw_cbit", "next_gw_cbirt",
    }
    # Also exclude set piece order raw columns (we use the binary flag)
    exclude.update({"penalties_order", "corners_order", "freekicks_order",
                     "transfers_out_event", "total_points"})
    # Exclude cumulative season totals (proxies for total_points)
    exclude.update({"influence", "creativity", "threat", "ict_index",
                     "player_bps", "player_bonus"})
    # yellow_cards is a cumulative season total, not a per-GW stat
    exclude.add("yellow_cards")

    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(feature_cols)
