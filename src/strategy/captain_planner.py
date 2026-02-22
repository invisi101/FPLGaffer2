"""Captain planning across the prediction horizon.

Ported from v1 strategy.py CaptainPlanner class.
Pre-plans captaincy using transfer plan squads and flags weak captain GWs.
"""

from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


class CaptainPlanner:
    """Pre-plan captaincy across the prediction horizon."""

    def plan_captaincy(
        self,
        current_squad_ids: set[int],
        future_predictions: dict[int, pd.DataFrame],
        transfer_plan: list[dict] | None = None,
        select_formation_xi_fn=None,
    ) -> list[dict]:
        """Plan captaincy across future GWs.

        Parameters
        ----------
        current_squad_ids:
            Set of player IDs in the current squad.
        future_predictions:
            {gw: DataFrame} of predictions for each future gameweek.
        transfer_plan:
            Optional transfer plan list from MultiWeekPlanner. When provided,
            the captain is picked from the planned squad (not just current).
        select_formation_xi_fn:
            Callable that selects a formation-valid starting XI from a
            squad DataFrame.  Imported from transfer_planner to avoid
            circular imports.

        Returns
        -------
        list of {gw, captain_id, captain_name, captain_points,
                 vc_id, vc_name, confidence, weak_gw}.
        """
        # Lazy import to avoid circular dependency
        if select_formation_xi_fn is None:
            from src.strategy.transfer_planner import MultiWeekPlanner

            select_formation_xi_fn = MultiWeekPlanner._select_formation_xi

        captain_plan: list[dict] = []
        pred_gws = sorted(future_predictions.keys())

        for gw in pred_gws:
            gw_df = future_predictions[gw]

            # Use transfer plan squad if available for this GW
            squad_ids = current_squad_ids
            if transfer_plan:
                for step in transfer_plan:
                    if step["gw"] == gw and step.get("squad_ids"):
                        squad_ids = set(step["squad_ids"])
                        break

            squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)].copy()
            if squad_preds.empty:
                continue

            # Captain must be a starter -- select formation-valid XI first
            xi = select_formation_xi_fn(squad_preds)
            if xi.empty:
                continue

            # Sort by captain_score (upside-weighted) if available,
            # else predicted_points
            score_col = (
                "captain_score"
                if "captain_score" in xi.columns
                else "predicted_points"
            )
            xi = xi.sort_values(score_col, ascending=False)

            captain = xi.iloc[0]
            vc = xi.iloc[1] if len(xi) > 1 else captain

            captain_pts = float(captain["predicted_points"])
            confidence = float(captain.get("confidence", 1.0))

            # Flag weak captain GWs (captain predicted < 4 pts)
            weak_gw = bool(captain_pts < 4.0)

            captain_plan.append({
                "gw": gw,
                "captain_id": int(captain["player_id"]),
                "captain_name": captain.get("web_name", "Unknown"),
                "captain_points": round(captain_pts, 2),
                "vc_id": int(vc["player_id"]),
                "vc_name": vc.get("web_name", "Unknown"),
                "confidence": round(confidence, 2),
                "weak_gw": weak_gw,
            })

        return captain_plan
