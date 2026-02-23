"""MILP solver for optimal FPL squad selection.

Ported from v1's solver.py — solve_milp_team().
Two-tier MILP with optional captain optimization.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import Bounds as ScipyBounds, LinearConstraint, milp

from src.config import solver_cfg
from src.utils.nan_handling import scrub_nan

logger = logging.getLogger(__name__)

SUB_WEIGHT = solver_cfg.bench_weight


def solve_milp_team(
    player_df: pd.DataFrame,
    target_col: str,
    budget: float = solver_cfg.max_budget,
    team_cap: int = solver_cfg.team_cap,
    captain_col: str | None = None,
) -> dict | None:
    """Solve two-tier MILP for optimal squad selection.

    When captain_col is provided, jointly optimizes captain selection
    alongside squad/starter decisions (3n variables instead of 2n).

    Returns {"starters": [...], "bench": [...], "total_cost": float,
    "starting_points": float, "captain_id": int | None} or None on failure.
    """
    required = ["position", "cost", target_col]
    if not all(c in player_df.columns for c in required):
        return None

    df = player_df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    pred = df[target_col].values.astype(float)

    # Check if captain optimization is possible
    use_captain = captain_col and captain_col in df.columns
    if use_captain:
        # Fill NaN captain scores with the target prediction as fallback
        captain_scores = df[captain_col].fillna(df[target_col]).values.astype(float)
        captain_bonus = captain_scores  # Captain doubles points; bonus = full captain_score

    if use_captain:
        # 3n variables: x_i (squad), s_i (starter), c_i (captain)
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
            -captain_bonus,
        ])
        integrality = np.ones(3 * n)
        nvars = 3 * n
    else:
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
        ])
        integrality = np.ones(2 * n)
        nvars = 2 * n

    A_rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add_constraint(coeffs: np.ndarray, lb: float, ub: float) -> None:
        # Pad to nvars if needed
        if len(coeffs) < nvars:
            coeffs = np.concatenate([coeffs, np.zeros(nvars - len(coeffs))])
        A_rows.append(coeffs)
        lbs.append(lb)
        ubs.append(ub)

    zeros = np.zeros(n)
    costs = df["cost"].values.astype(float)

    # Budget constraint
    add_constraint(np.concatenate([costs, zeros]), 0, budget)

    # Position counts (squad): exactly 2 GKP, 5 DEF, 5 MID, 3 FWD
    squad_req = solver_cfg.squad_positions
    for pos, count in squad_req.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([pos_mask, zeros]), count, count)

    # Max players from same team
    if "team_code" in df.columns:
        for tc in df["team_code"].unique():
            team_mask = (df["team_code"] == tc).astype(float).values
            add_constraint(np.concatenate([team_mask, zeros]), 0, team_cap)
    else:
        logger.warning(
            "team_code missing from solver input; max-3-per-team constraint disabled"
        )

    # Exactly 11 starters
    add_constraint(np.concatenate([zeros, np.ones(n)]), solver_cfg.starting_xi, solver_cfg.starting_xi)

    # Formation constraints (starting XI)
    formation_limits = solver_cfg.formation_limits
    for pos, (lo, hi) in formation_limits.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([zeros, pos_mask]), lo, hi)

    # s_i <= x_i (can only start if in squad)
    for i in range(n):
        row = np.zeros(nvars)
        row[i] = -1.0      # -x_i
        row[n + i] = 1.0   # +s_i
        A_rows.append(row)
        lbs.append(-np.inf)
        ubs.append(0)

    # Captain constraints
    if use_captain:
        # sum(c_i) == 1
        cap_sum = np.zeros(nvars)
        cap_sum[2 * n:] = 1.0
        A_rows.append(cap_sum)
        lbs.append(1)
        ubs.append(1)

        # c_i <= s_i (captain must be a starter)
        for i in range(n):
            row = np.zeros(nvars)
            row[n + i] = -1.0      # -s_i
            row[2 * n + i] = 1.0   # +c_i
            A_rows.append(row)
            lbs.append(-np.inf)
            ubs.append(0)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lbs, ubs)
    variable_bounds = ScipyBounds(lb=0, ub=1)

    result = milp(
        c_obj, integrality=integrality,
        bounds=variable_bounds, constraints=constraints,
    )

    if not result.success:
        return None

    x_vals = result.x[:n]
    s_vals = result.x[n:2 * n]
    squad_mask = x_vals > 0.5
    starter_mask = s_vals > 0.5

    captain_id = None
    if use_captain:
        c_vals = result.x[2 * n:]
        cap_idx = np.where(c_vals > 0.5)[0]
        if len(cap_idx) > 0 and "player_id" in df.columns:
            captain_id = int(df.iloc[cap_idx[0]]["player_id"])

    team_df = df[squad_mask].copy()
    team_df["starter"] = starter_mask[squad_mask]

    float_cols = team_df.select_dtypes(include="float").columns
    team_df[float_cols] = team_df[float_cols].round(2)

    starters = team_df[team_df["starter"]]
    bench = team_df[~team_df["starter"]]

    # Sort bench: GK first, then outfield ordered for optimal auto-subs.
    # Positions at their formation minimum need bench coverage first, since
    # only a same-position sub can replace them without breaking formation.
    bench_gk = bench[bench["position"] == "GKP"]
    bench_outfield = bench[bench["position"] != "GKP"].copy()
    if not bench_outfield.empty:
        starter_pos_counts = starters["position"].value_counts().to_dict()
        constrained = set()
        for pos, (lo, _) in solver_cfg.formation_limits.items():
            if pos != "GKP" and starter_pos_counts.get(pos, 0) <= lo:
                constrained.add(pos)
        bench_outfield["_priority"] = bench_outfield["position"].apply(
            lambda p: 1 if p in constrained else 0
        )
        bench_outfield = bench_outfield.sort_values(
            ["_priority", target_col], ascending=[False, False],
        ).drop(columns=["_priority"])
    bench = pd.concat([bench_gk, bench_outfield])

    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    team_df["_pos_order"] = team_df["position"].map(pos_order)
    team_df = team_df.sort_values(
        ["starter", "_pos_order", target_col], ascending=[False, True, False],
    )
    team_df = team_df.drop(columns=["_pos_order"])

    # Include captain bonus in starting_points (captain doubles their points)
    base_pts = starters[target_col].sum()
    captain_pts = 0.0
    captain_opt_pts = 0.0  # Upside-weighted score used by the optimization objective
    if captain_id and "player_id" in starters.columns:
        cap_match = starters.loc[starters["player_id"] == captain_id, target_col]
        if not cap_match.empty:
            captain_pts = cap_match.iloc[0]
        if use_captain and captain_col and captain_col in starters.columns:
            cap_score = starters.loc[
                starters["player_id"] == captain_id, captain_col
            ]
            if not cap_score.empty:
                captain_opt_pts = cap_score.iloc[0]

    return {
        "starters": scrub_nan(starters.to_dict(orient="records")),
        "bench": scrub_nan(bench.to_dict(orient="records")),
        "total_cost": round(team_df["cost"].sum(), 1),
        "starting_points": round(base_pts + captain_pts, 2),
        "optimization_score": (
            round(base_pts + captain_opt_pts, 2)
            if use_captain and captain_id
            else None
        ),
        "players": scrub_nan(team_df.to_dict(orient="records")),
        "captain_id": captain_id,
    }
