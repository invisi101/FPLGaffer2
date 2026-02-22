"""Formation selection and bench ordering helpers."""

from __future__ import annotations

import pandas as pd

from src.config import solver_cfg

POS_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def select_best_xi(
    squad_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Select the best 11 starters from a 15-player squad DataFrame.

    Tries all valid formations and picks the one that maximises total
    predicted points (target_col).

    Returns a DataFrame of 11 rows (the starters), sorted by position
    then predicted points.
    """
    formation_limits = solver_cfg.formation_limits
    positions = list(formation_limits.keys())

    # Group players by position, sorted by prediction descending
    by_pos: dict[str, pd.DataFrame] = {}
    for pos in positions:
        pos_df = squad_df[squad_df["position"] == pos].sort_values(
            target_col, ascending=False,
        )
        by_pos[pos] = pos_df

    best_xi: pd.DataFrame | None = None
    best_pts = -float("inf")

    # Enumerate valid formations
    gkp_lo, gkp_hi = formation_limits["GKP"]
    def_lo, def_hi = formation_limits["DEF"]
    mid_lo, mid_hi = formation_limits["MID"]
    fwd_lo, fwd_hi = formation_limits["FWD"]

    for n_def in range(def_lo, def_hi + 1):
        for n_mid in range(mid_lo, mid_hi + 1):
            for n_fwd in range(fwd_lo, fwd_hi + 1):
                n_gkp = gkp_lo  # Always 1 GKP
                if n_gkp + n_def + n_mid + n_fwd != solver_cfg.starting_xi:
                    continue

                # Check we have enough players in each position
                if (
                    len(by_pos["GKP"]) < n_gkp
                    or len(by_pos["DEF"]) < n_def
                    or len(by_pos["MID"]) < n_mid
                    or len(by_pos["FWD"]) < n_fwd
                ):
                    continue

                xi = pd.concat([
                    by_pos["GKP"].head(n_gkp),
                    by_pos["DEF"].head(n_def),
                    by_pos["MID"].head(n_mid),
                    by_pos["FWD"].head(n_fwd),
                ])
                pts = xi[target_col].sum()
                if pts > best_pts:
                    best_pts = pts
                    best_xi = xi

    if best_xi is None:
        # Fallback: just take top 11 (shouldn't happen with valid squad)
        return squad_df.sort_values(target_col, ascending=False).head(
            solver_cfg.starting_xi,
        )

    # Sort by position order then predicted points
    best_xi = best_xi.copy()
    best_xi["_pos_order"] = best_xi["position"].map(POS_ORDER)
    best_xi = best_xi.sort_values(
        ["_pos_order", target_col], ascending=[True, False],
    )
    return best_xi.drop(columns=["_pos_order"])


def order_bench(
    bench_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Order bench players: GKP first, then outfield by descending points.

    FPL auto-sub rules require bench GK in position 1, then outfield
    ordered by manager preference (we use predicted points).
    """
    bench_gk = bench_df[bench_df["position"] == "GKP"]
    bench_outfield = bench_df[bench_df["position"] != "GKP"].sort_values(
        target_col, ascending=False,
    )
    return pd.concat([bench_gk, bench_outfield]).reset_index(drop=True)


def get_formation_string(starters_df: pd.DataFrame) -> str:
    """Return formation string like '3-4-3' from a starters DataFrame.

    Only counts outfield positions (DEF-MID-FWD).
    """
    pos_counts: dict[str, int] = {}
    for pos in starters_df["position"]:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    n_def = pos_counts.get("DEF", 0)
    n_mid = pos_counts.get("MID", 0)
    n_fwd = pos_counts.get("FWD", 0)
    return f"{n_def}-{n_mid}-{n_fwd}"
