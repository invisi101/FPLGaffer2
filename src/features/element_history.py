"""Features from the FPL element-summary endpoint (current season).

The element-summary endpoint returns per-GW history for each player
including minutes, goals, assists, clean_sheets, bonus, bps, ICT
components, saves, and expected stats.  This module builds rolling
features from that data, producing column names that MATCH the
PMS-derived names (player_xg_last3, etc.) for model compatibility.

Key difference from PMS: element-summary gives per-GW values directly,
so there is no need for cumulative-to-delta conversion (unlike playerstats
ICT/BPS).
"""

from __future__ import annotations

import pandas as pd

from src.config import PLAYER_ROLLING_WINDOWS
from src.logging_config import get_logger

log = get_logger(__name__)

# Mapping from element-summary field names to the internal stat names
# used by the rolling feature builders.  The rolling feature output
# names follow the pattern "player_{stat}_last{window}".
_ELEMENT_HISTORY_COLS: dict[str, str] = {
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goal_involvements": "xgi",
    "goals_scored": "goals",
    "assists": "assists",
    "minutes": "minutes_played",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "total_points": "total_points",
}

# Stats that are available from element-summary but NOT from PMS.
# These can supplement or override PMS features.
_ELEMENT_ONLY_STATS = {"clean_sheets", "goals_conceded", "bonus", "bps",
                       "influence", "creativity", "threat", "ict_index",
                       "total_points", "xgi"}


def build_element_history_features(
    bootstrap: dict,
    element_summaries: dict[int, dict],
) -> pd.DataFrame:
    """Build rolling features from element-summary per-GW history.

    Args:
        bootstrap: FPL API bootstrap-static data (for player metadata).
        element_summaries: Dict mapping player_id -> element-summary response.
            Each value has a "history" key containing per-GW records.

    Returns:
        DataFrame with player_id, gameweek, and rolling features whose
        column names match the PMS-derived convention (player_xg_last3, etc.)
    """
    if not element_summaries:
        log.debug("No element summaries provided — skipping element history features")
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # Collect all per-GW history rows
    all_rows: list[dict] = []
    for player_id, summary in element_summaries.items():
        history = summary.get("history", [])
        if not history:
            continue
        for entry in history:
            row: dict = {
                "player_id": int(player_id),
                "gameweek": int(entry.get("round", 0)),
            }
            # Map element-summary fields to internal stat names
            for api_field, internal_name in _ELEMENT_HISTORY_COLS.items():
                val = entry.get(api_field)
                if val is not None:
                    row[internal_name] = float(val)
                else:
                    row[internal_name] = 0.0
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["player_id", "gameweek"])

    # Available stat columns
    stat_cols = [c for c in df.columns if c not in ("player_id", "gameweek")]

    # Aggregate per player per GW (sum handles DGW — multiple fixture rows
    # in the same GW get summed, matching PMS rolling behaviour)
    agg = df.groupby(["player_id", "gameweek"])[stat_cols].sum().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    # Add composite CBIT if constituent parts are available from PMS.
    # Element-summary does not have clearances/blocks/interceptions/tackles_won
    # so CBIT is NOT built here — it comes from PMS only.

    # Build rolling features with shift(1) to prevent leakage
    result_frames = [agg[["player_id", "gameweek"]]]

    # Core rolling stats (matching PMS rolling feature names)
    rolling_stats = [s for s in stat_cols if s not in _ELEMENT_ONLY_STATS]

    for window in PLAYER_ROLLING_WINDOWS:
        for col in rolling_stats:
            feat_name = f"player_{col}_last{window}"
            rolled = (
                agg.groupby("player_id")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    # EWM features (matching the ewm_ naming convention)
    from src.config import EWM_RAW_COLS, EWM_SPAN
    ewm_available = [c for c in EWM_RAW_COLS if c in agg.columns]
    for col in ewm_available:
        feat_name = f"ewm_player_{col}_last3"
        ewm_vals = (
            agg.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).ewm(span=EWM_SPAN, min_periods=1).mean())
        )
        result_frames.append(ewm_vals.rename(feat_name))

    # Per-GW ICT deltas — element-summary gives per-GW values directly,
    # so no cumulative-to-delta conversion needed.  Just rename to match
    # the gw_ column names that playerstats produces.
    ict_map = {
        "influence": "gw_influence",
        "creativity": "gw_creativity",
        "threat": "gw_threat",
        "ict_index": "gw_ict_index",
        "bps": "gw_player_bps",
    }
    for src_col, dst_col in ict_map.items():
        if src_col in agg.columns:
            # Use shifted values (GW N row gets GW N-1 ICT)
            shifted = (
                agg.groupby("player_id")[src_col]
                .transform(lambda s: s.shift(1))
            )
            result_frames.append(shifted.rename(dst_col))

    # Bonus rolling (different from gw_bonus target — this is for features)
    if "bonus" in agg.columns:
        for window in PLAYER_ROLLING_WINDOWS:
            feat_name = f"player_bonus_last{window}"
            rolled = (
                agg.groupby("player_id")["bonus"]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    result = pd.concat(result_frames, axis=1)

    # Deduplicate any columns that got added twice
    result = result.loc[:, ~result.columns.duplicated()]

    log.info(
        "Built element-history features: %d players, %d GWs, %d feature columns",
        result["player_id"].nunique(),
        result["gameweek"].nunique(),
        len([c for c in result.columns if c not in ("player_id", "gameweek")]),
    )

    return result
