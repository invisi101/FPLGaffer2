"""Dashboard aggregation — rank history, budget, chips, accuracy.

Extracted from v1's ``SeasonManager.get_season_dashboard()``.
"""

from __future__ import annotations

from src.db.repositories import (
    DashboardRepository,
    OutcomeRepository,
    RecommendationRepository,
    SeasonRepository,
    SnapshotRepository,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def get_dashboard(
    manager_id: int,
    season_repo: SeasonRepository,
    snapshot_repo: SnapshotRepository,
    rec_repo: RecommendationRepository,
    outcome_repo: OutcomeRepository,
    dashboard_repo: DashboardRepository,
    season_name: str = "",
) -> dict:
    """Full dashboard data for the Season tab.

    Parameters
    ----------
    manager_id:
        FPL manager ID.
    season_repo ... dashboard_repo:
        Repository instances.
    season_name:
        Season to query.  Defaults to current.

    Returns
    -------
    dict
        Dashboard payload matching v1's ``get_season_dashboard`` contract.
    """
    season = season_repo.get_season(manager_id, season_name)
    if not season:
        return {"error": "No active season."}
    season_id = season["id"]

    snapshots = snapshot_repo.get_snapshots(season_id)
    rank_history = dashboard_repo.get_rank_history(season_id)
    budget_history = dashboard_repo.get_budget_history(season_id)
    chips_used = dashboard_repo.get_chips_status(season_id)
    accuracy = dashboard_repo.get_recommendation_accuracy(season_id)
    recommendations = rec_repo.get_recommendations(season_id)
    outcomes = outcome_repo.get_outcomes(season_id)

    # Compute available chips with half-season reset awareness
    current_gw = season.get("current_gw", 1)
    # Bug 53 fix: Use next_gw for chip availability check
    next_gw = min(current_gw + 1, 38)
    all_chips_list = [
        {"name": "wildcard", "label": "Wildcard"},
        {"name": "freehit", "label": "Free Hit"},
        {"name": "bboost", "label": "Bench Boost"},
        {"name": "3xc", "label": "Triple Captain"},
    ]
    chips_status = []
    for chip in all_chips_list:
        chip_events = [
            c["gameweek"] for c in chips_used if c["chip_used"] == chip["name"]
        ]
        if next_gw <= 19:
            used = any(e <= 19 for e in chip_events)
            used_in = next((e for e in chip_events if e <= 19), None)
        else:
            used = any(e >= 20 for e in chip_events)
            used_in = next((e for e in chip_events if e >= 20), None)
        chips_status.append({
            "name": chip["name"],
            "label": chip["label"],
            "used": used,
            "used_gw": used_in,
        })

    # Latest snapshot for summary
    latest = snapshots[-1] if snapshots else {}

    # Points per GW
    points_per_gw = [
        {"gameweek": s["gameweek"], "points": s.get("points", 0)}
        for s in snapshots
    ]

    # Build accuracy history from outcomes
    accuracy_history = []
    for o in outcomes:
        if (
            o.get("recommended_points") is not None
            and o.get("actual_points") is not None
        ):
            accuracy_history.append({
                "gameweek": o["gameweek"],
                "predicted_points": o["recommended_points"],
                "actual_points": o["actual_points"],
                "delta": o.get("point_delta", 0),
            })

    return {
        "season": season,
        "summary": {
            "overall_rank": latest.get("overall_rank"),
            "total_points": latest.get("total_points"),
            "bank": latest.get("bank"),
            "team_value": latest.get("team_value"),
            "gameweek": latest.get("gameweek"),
        },
        "rank_history": rank_history,
        "budget_history": budget_history,
        "points_per_gw": points_per_gw,
        "chips_status": chips_status,
        "accuracy": accuracy,
        "accuracy_history": accuracy_history,
        "recommendations_count": len(recommendations),
        "outcomes_count": len(outcomes),
    }
