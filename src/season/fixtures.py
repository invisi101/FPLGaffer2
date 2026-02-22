"""Fixture calendar building and persistence.

Extracted from v1's ``SeasonManager.update_fixture_calendar()``.
"""

from __future__ import annotations

import json

from src.db.repositories import FixtureRepository
from src.logging_config import get_logger

logger = get_logger(__name__)


def build_fixture_calendar(
    bootstrap: dict, fixtures: list[dict]
) -> list[dict]:
    """Parse fixtures API data into a flat list of per-team-per-GW records.

    Parameters
    ----------
    bootstrap:
        FPL bootstrap-static response (teams list needed).
    fixtures:
        FPL fixtures API response.

    Returns
    -------
    list[dict]
        Records suitable for ``FixtureRepository.save_fixture_calendar``.
    """
    if not fixtures:
        return []

    id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    # Build per-team per-GW fixture data
    team_gw: dict[int, dict[int, list]] = {}  # team_id -> gw -> [fixtures]
    for f in fixtures:
        gw = f.get("event")
        if not gw:
            continue
        for side, opp_side, tag in [
            ("team_h", "team_a", "H"),
            ("team_a", "team_h", "A"),
        ]:
            tid = f[side]
            opp_id = f[opp_side]
            opp_name = id_to_short.get(opp_id, "?")
            fdr = f.get(f"team_{tag.lower()}_difficulty", 3)
            if tid not in team_gw:
                team_gw[tid] = {}
            if gw not in team_gw[tid]:
                team_gw[tid][gw] = []
            team_gw[tid][gw].append({
                "opponent": f"{opp_name} ({tag})",
                "fdr": fdr,
            })

    # Collect all GWs
    all_gws: set[int] = set()
    for tid in team_gw:
        for gw in team_gw[tid]:
            all_gws.add(gw)

    # Build flat records
    records: list[dict] = []
    for tid, gw_data in team_gw.items():
        team_short = id_to_short.get(tid, "?")
        for gw in sorted(all_gws):
            fxs = gw_data.get(gw, [])
            if not fxs:
                # BGW: no fixture this GW
                records.append({
                    "team_id": tid,
                    "team_code": id_to_code.get(tid),
                    "team_short": team_short,
                    "gameweek": gw,
                    "fixture_count": 0,
                    "opponents_json": json.dumps([]),
                    "fdr_avg": None,
                    "is_dgw": 0,
                    "is_bgw": 1,
                })
            else:
                avg_fdr = round(
                    sum(fx["fdr"] for fx in fxs) / len(fxs), 1
                )
                records.append({
                    "team_id": tid,
                    "team_code": id_to_code.get(tid),
                    "team_short": team_short,
                    "gameweek": gw,
                    "fixture_count": len(fxs),
                    "opponents_json": json.dumps(
                        [fx["opponent"] for fx in fxs]
                    ),
                    "fdr_avg": avg_fdr,
                    "is_dgw": 1 if len(fxs) >= 2 else 0,
                    "is_bgw": 0,
                })

    return records


def save_fixture_calendar(
    season_id: int,
    bootstrap: dict,
    fixtures: list[dict],
    fixture_repo: FixtureRepository,
) -> None:
    """Build the fixture calendar and persist it via the repository.

    Parameters
    ----------
    season_id:
        Database season ID.
    bootstrap:
        FPL bootstrap-static response.
    fixtures:
        FPL fixtures API response.
    fixture_repo:
        Repository instance for persistence.
    """
    records = build_fixture_calendar(bootstrap, fixtures)
    if records:
        fixture_repo.save_fixture_calendar(season_id, records)
        logger.info(
            "Saved %d fixture calendar records for season %d",
            len(records),
            season_id,
        )
