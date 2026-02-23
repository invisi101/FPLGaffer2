"""Rolling 5-GW transfer planner with FT banking and fixture awareness.

Ported from v1 strategy.py MultiWeekPlanner class.
Tree search across all valid FT allocation sequences, simulating each path
and picking the one that maximizes total points.
"""

from __future__ import annotations

import pandas as pd

from src.config import solver_cfg, strategy_cfg
from src.logging_config import get_logger
from src.solver.squad import solve_milp_team

logger = get_logger(__name__)

# ── Transfer planner constants (from config) ──────────────────────────
_PLANNING_HORIZON = strategy_cfg.planning_horizon
_MAX_HITS_PER_GW = strategy_cfg.max_hits_per_gw
_FT_MAX_BANK = strategy_cfg.ft_max_bank

# Per-GW transfer cap to keep search tractable
_MAX_TRANSFERS_PER_GW = 3

# Top players for pool reduction
_TOP_POOL_SIZE = 200


class MultiWeekPlanner:
    """Rolling 5-GW transfer planner with FT banking, fixture swings,
    and price awareness."""

    def plan_transfers(
        self,
        current_squad_ids: set[int],
        total_budget: float,
        free_transfers: int,
        future_predictions: dict[int, pd.DataFrame],
        fixture_calendar: list[dict],
        price_alerts: list[dict],
        chip_plan: dict | None = None,
        solve_transfer_fn=None,
    ) -> list[dict]:
        """Plan transfers over next 5 GWs with forward simulation.

        Parameters
        ----------
        solve_transfer_fn:
            Callable for ``solve_transfer_milp_with_hits``.  Passed in
            to avoid circular imports; if *None*, imported lazily from
            ``src.solver.transfers``.

        Returns list of {gw, transfers_in, transfers_out, ft_strategy,
        rationale, squad_ids, predicted_points}.
        """
        if solve_transfer_fn is None:
            from src.solver.transfers import solve_transfer_milp_with_hits

            solve_transfer_fn = solve_transfer_milp_with_hits

        pred_gws = sorted(future_predictions.keys())
        if len(pred_gws) < 1:
            return []

        # Build price bonus map: player_id -> bonus points
        price_bonus = self._build_price_bonus(price_alerts)

        # Build fixture swing bonus map for each GW, keyed by team_code
        fx_lookup: dict[int, dict[int, dict]] = {}
        for f in fixture_calendar:
            gw = f["gameweek"]
            if gw not in fx_lookup:
                fx_lookup[gw] = {}
            # Use team_code if available, fall back to team_id
            key = f.get("team_code") or f.get("team_id")
            if key is not None:
                fx_lookup[gw][int(key)] = f

        # Take first N prediction GWs for planning
        plan_gws = pred_gws[:_PLANNING_HORIZON]

        # Reduce player pool to top ~200 by predicted points for efficiency
        first_gw_df = future_predictions[plan_gws[0]]
        top_pool_ids = set(
            first_gw_df.nlargest(
                _TOP_POOL_SIZE, "predicted_points",
            )["player_id"].tolist()
        )
        # Always include current squad
        top_pool_ids |= current_squad_ids

        # Filter predictions to reduced pool
        filtered_preds: dict[int, pd.DataFrame] = {}
        for gw in plan_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                filtered_preds[gw] = gw_df[
                    gw_df["player_id"].isin(top_pool_ids)
                ].copy()

        # Generate all valid FT allocation sequences across the horizon
        ft_plans = self._generate_ft_plans(plan_gws, free_transfers, chip_plan)

        best_path = None
        best_total = -float("inf")

        for ft_plan in ft_plans:
            path = self._simulate_path(
                plan_gws, filtered_preds, current_squad_ids,
                total_budget, free_transfers, ft_plan,
                price_bonus, fx_lookup, chip_plan=chip_plan,
                solve_transfer_fn=solve_transfer_fn,
            )
            if path is None:
                continue

            total_pts = sum(step["predicted_points"] for step in path)
            if total_pts > best_total:
                best_total = total_pts
                best_path = path

        if best_path is None:
            return []

        # Annotate with rationale
        for step in best_path:
            step["rationale"] = self._build_rationale(step, free_transfers)

        return best_path

    # ── FT plan generation ────────────────────────────────────────────

    def _generate_ft_plans(
        self,
        plan_gws: list[int],
        initial_ft: int,
        chip_plan: dict | None = None,
    ) -> list[list[int]]:
        """Generate all valid FT allocation sequences for the planning
        horizon.

        Each plan is a list of ints (one per GW) specifying how many FTs
        to use. Chip GWs are fixed at 0 (chip logic handles them
        separately). Normal GWs try 0 through min(ft+1, 3) to allow
        exploring one hit transfer.
        """
        plans: list[list[int]] = []

        def recurse(idx: int, ft: int, current: list[int]):
            if idx >= len(plan_gws):
                plans.append(list(current))
                return

            gw = plan_gws[idx]
            gw_chip = None
            if chip_plan:
                gw_chip = chip_plan.get("chip_gws", {}).get(gw)

            if gw_chip in ("wildcard", "freehit"):
                # Chip GW: FTs preserved at pre-chip count, no accrual
                current.append(0)
                recurse(idx + 1, ft, current)
                current.pop()
            else:
                max_use = min(ft + _MAX_HITS_PER_GW, _MAX_TRANSFERS_PER_GW)
                for use in range(0, max_use + 1):
                    current.append(use)
                    next_ft = max(min(ft - use + 1, _FT_MAX_BANK), 1)
                    recurse(idx + 1, next_ft, current)
                    current.pop()

        recurse(0, initial_ft, [])
        return plans

    # ── Price bonus ───────────────────────────────────────────────────

    def _build_price_bonus(
        self, price_alerts: list[dict],
    ) -> dict[int, float]:
        """Convert price alerts/predictions to bonus points for likely
        risers.

        Supports both old format (net_transfers only) and new format
        (with probability).
        """
        bonus: dict[int, float] = {}
        for alert in price_alerts:
            if alert.get("direction") == "rise":
                if "probability" in alert:
                    # New probability-based formula
                    prob = alert.get("probability", 0)
                    change = abs(alert.get("estimated_change", 0.1))
                    bonus[alert["player_id"]] = round(prob * change * 3.0, 2)
                else:
                    # Legacy: rough estimate from net transfers
                    net = alert.get("net_transfers", 0)
                    estimated_rise = min(0.3, net / 100000)
                    bonus[alert["player_id"]] = round(
                        estimated_rise * 3.0, 2,
                    )
        return bonus

    # ── Formation XI selection (static, used by other modules) ───────

    @staticmethod
    def _select_formation_xi(squad_preds: pd.DataFrame) -> pd.DataFrame:
        """Select a formation-valid starting XI from a squad, maximizing
        predicted points.

        Returns a DataFrame of 11 starters respecting 1 GKP, 3-5 DEF,
        2-5 MID, 1-3 FWD. Falls back to nlargest(11) if position data
        is missing.
        """
        if "position" not in squad_preds.columns or len(squad_preds) < 11:
            return squad_preds.nlargest(
                min(11, len(squad_preds)), "predicted_points",
            )

        # Pick mandatory minimums: 1 GKP, 3 DEF, 2 MID, 1 FWD = 7
        pos_groups: dict[str, pd.DataFrame] = {}
        for pos in ("GKP", "DEF", "MID", "FWD"):
            grp = squad_preds[squad_preds["position"] == pos].nlargest(
                {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}.get(pos, 5),
                "predicted_points",
            )
            pos_groups[pos] = grp

        mins = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
        starters_idx: set = set()
        for pos, n in mins.items():
            starters_idx.update(pos_groups[pos].head(n).index)

        # Fill remaining 4 spots from best available extras
        # (respecting position maxima)
        maxs = {"DEF": 5, "MID": 5, "FWD": 3}
        picked = {pos: n for pos, n in mins.items()}
        extras: list[tuple[float, object, str]] = []
        for pos in ("DEF", "MID", "FWD"):
            remaining = pos_groups[pos].iloc[mins[pos]:]
            for idx_val in remaining.index:
                if picked[pos] < maxs[pos]:
                    extras.append((
                        remaining.loc[idx_val, "predicted_points"],
                        idx_val,
                        pos,
                    ))
        extras.sort(reverse=True, key=lambda x: x[0])
        for pts_val, idx_val, pos in extras:
            if len(starters_idx) >= 11:
                break
            # Re-check position max before adding (safety guard)
            if picked[pos] < maxs.get(pos, 99):
                starters_idx.add(idx_val)
                picked[pos] += 1

        return squad_preds.loc[list(starters_idx)]

    @staticmethod
    def _squad_points_with_captain(
        squad_preds: pd.DataFrame,
    ) -> float:
        """Compute starting XI points with captain bonus, respecting
        formation constraints.

        Uses captain_score (upside-weighted) for captain selection when
        available, consistent with the MILP solver. The bonus is the
        captain's predicted_points (i.e. actual expected extra points
        from doubling).
        """
        if len(squad_preds) < 11:
            return (
                squad_preds["predicted_points"].sum()
                if not squad_preds.empty
                else 0
            )

        top11 = MultiWeekPlanner._select_formation_xi(squad_preds)
        pts = top11["predicted_points"].sum()
        # Use captain_score for selection (consistent with MILP),
        # but bonus is the captain's predicted_points.
        score_col = (
            "captain_score"
            if "captain_score" in top11.columns
            else "predicted_points"
        )
        # Guard against all-NaN captain_score
        try:
            captain_idx = top11[score_col].idxmax()
        except ValueError:
            try:
                captain_idx = top11["predicted_points"].idxmax()
            except ValueError:
                return pts  # No valid captain, return XI points only
        captain_bonus = top11.loc[captain_idx, "predicted_points"]
        return pts + captain_bonus

    # ── Path simulation ───────────────────────────────────────────────

    def _simulate_path(
        self,
        plan_gws,
        filtered_preds,
        current_squad_ids,
        total_budget,
        free_transfers,
        ft_plan,
        price_bonus,
        fx_lookup,
        chip_plan=None,
        solve_transfer_fn=None,
    ) -> list[dict] | None:
        """Simulate a transfer path over 5 GWs given a per-GW FT
        allocation plan."""
        path: list[dict] = []
        squad_ids = set(current_squad_ids)
        budget = total_budget
        ft = free_transfers
        late_season = bool(
            plan_gws and plan_gws[0] >= strategy_cfg.late_season_gw
        )

        for i, gw in enumerate(plan_gws):
            if gw not in filtered_preds:
                break

            gw_df = filtered_preds[gw].copy()

            # Check if this GW has a WC/FH chip planned
            gw_chip = None
            if chip_plan:
                gw_chip = chip_plan.get("chip_gws", {}).get(gw)

            # Apply price bonus with decay (most urgent for immediate GW)
            if price_bonus and i < 3:
                decay = (1.0, 0.5, 0.25)[i]
                gw_df["predicted_points"] = gw_df.apply(
                    lambda r, d=decay: r["predicted_points"]
                    + price_bonus.get(r["player_id"], 0) * d,
                    axis=1,
                )

            # Apply fixture swing bonus
            if gw in fx_lookup:
                gw_fx = fx_lookup[gw]

                def _fx_bonus(row):
                    if "team_code" not in row or pd.isna(
                        row.get("team_code"),
                    ):
                        return 0
                    fx = gw_fx.get(int(row["team_code"]), {})
                    fdr = fx.get("fdr_avg", 3.0)
                    if fdr is None:
                        return 0
                    # Bonus for easy fixtures (FDR < 3)
                    return max(0, (3.0 - fdr) * 0.3)

                if "team_code" in gw_df.columns:
                    gw_df["predicted_points"] = (
                        gw_df["predicted_points"]
                        + gw_df.apply(_fx_bonus, axis=1)
                    )

            # WC/FH GWs: solve full squad from scratch
            if gw_chip in ("wildcard", "freehit"):
                step = self._simulate_chip_gw(
                    gw, gw_chip, gw_df, squad_ids, budget, ft,
                )
                if step is None:
                    return None
                path.append(step)

                if gw_chip == "wildcard":
                    # WC permanently changes squad
                    squad_ids = set(step["squad_ids"])
                # FH: squad reverts (squad_ids stays unchanged)
                # FTs preserved for both WC and FH
                continue

            # Use the pre-planned FT allocation
            use_now = ft_plan[i]

            if use_now == 0:
                step = self._simulate_bank_gw(
                    gw, gw_chip, gw_df, squad_ids, ft,
                )
                path.append(step)
                # Roll forward FTs
                ft = min(ft + 1, _FT_MAX_BANK)
            else:
                step = self._simulate_transfer_gw(
                    gw, gw_chip, gw_df, squad_ids, budget, ft, use_now,
                    solve_transfer_fn, late_season=late_season,
                )
                if step is None:
                    return None
                path.append(step)

                actual_transfers = step["ft_used"]
                if actual_transfers > 0:
                    squad_ids = set(step["squad_ids"])
                    ft = min(ft - actual_transfers + 1, _FT_MAX_BANK)
                    ft = max(ft, 1)
                else:
                    ft = min(ft + 1, _FT_MAX_BANK)

        return path if path else None

    def _simulate_chip_gw(
        self, gw, gw_chip, gw_df, squad_ids, budget, ft,
    ) -> dict | None:
        """Simulate a WC/FH chip GW.

        Falls back to current squad's predicted points when the MILP
        solver fails or when position/cost columns are missing, keeping
        the planning path alive.  Only returns *None* when the
        prediction data is completely empty.
        """
        pool = gw_df.dropna(subset=["predicted_points"])

        result = None
        if "position" in pool.columns and "cost" in pool.columns:
            cap_col = (
                "captain_score"
                if "captain_score" in pool.columns
                else None
            )
            result = solve_milp_team(
                pool, "predicted_points",
                budget=budget, captain_col=cap_col,
            )

        if result:
            pts = result["starting_points"]
            new_squad_ids = {p["player_id"] for p in result["players"]}
            return {
                "gw": gw,
                "transfers_in": [],
                "transfers_out": [],
                "ft_used": 0,
                "ft_available": ft,
                "predicted_points": round(pts, 2),
                "base_points": round(pts, 2),
                "squad_ids": list(new_squad_ids),
                "chip": gw_chip,
                "new_squad": result["players"],
            }

        # Solver failed or columns missing -- fall back to current
        # squad's predicted points so the path stays alive.
        squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)]
        pts = self._squad_points_with_captain(squad_preds)
        return {
            "gw": gw,
            "transfers_in": [],
            "transfers_out": [],
            "ft_used": 0,
            "ft_available": ft,
            "predicted_points": round(pts, 2),
            "base_points": round(pts, 2),
            "squad_ids": list(squad_ids),
            "chip": gw_chip,
        }

    def _simulate_bank_gw(
        self, gw, gw_chip, gw_df, squad_ids, ft,
    ) -> dict:
        """Simulate a GW where FTs are banked (no transfers)."""
        squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)]
        pts = self._squad_points_with_captain(squad_preds)
        base_pts = pts

        # BB/TC chip adjustments
        if gw_chip == "bboost" and len(squad_preds) > 11:
            top11 = self._select_formation_xi(squad_preds)
            bench = squad_preds[~squad_preds.index.isin(top11.index)]
            pts += bench["predicted_points"].sum()
        elif gw_chip == "3xc":
            # TC gives 3x instead of 2x -- add one more captain's
            # predicted_points. Pick captain from starting XI only.
            top11 = self._select_formation_xi(squad_preds)
            score_col = (
                "captain_score"
                if "captain_score" in top11.columns
                else "predicted_points"
            )
            if not top11.empty:
                cap_idx = top11[score_col].idxmax()
                pts += top11.loc[cap_idx, "predicted_points"]

        return {
            "gw": gw,
            "transfers_in": [],
            "transfers_out": [],
            "ft_used": 0,
            "ft_available": ft,
            "predicted_points": round(pts, 2),
            "base_points": round(base_pts, 2),
            "squad_ids": list(squad_ids),
            "chip": gw_chip,
        }

    def _simulate_transfer_gw(
        self, gw, gw_chip, gw_df, squad_ids, budget, ft, use_now,
        solve_transfer_fn, *, late_season: bool = False,
    ) -> dict | None:
        """Simulate a GW with active transfers."""
        pool = gw_df.dropna(subset=["predicted_points"])
        if "position" not in pool.columns or "cost" not in pool.columns:
            return None

        cap_col = (
            "captain_score" if "captain_score" in pool.columns else None
        )
        result = solve_transfer_fn(
            pool, squad_ids, "predicted_points",
            budget=budget, free_transfers=ft,
            max_transfers=use_now, captain_col=cap_col,
        )
        if result:
            pts = result.get("net_points", result["starting_points"])
            base_pts = pts

            # BB/TC chip adjustments on transfer GWs
            if gw_chip == "bboost":
                bench_pts = sum(
                    p.get("predicted_points") or 0
                    for p in result.get("bench", [])
                )
                pts += bench_pts
            elif gw_chip == "3xc" and result.get("captain_id"):
                # TC: add one more captain's predicted_points (3x total)
                for p in result.get("starters", []):
                    if p.get("player_id") == result["captain_id"]:
                        pts += p.get("predicted_points") or 0
                        break

            new_squad_ids = {p["player_id"] for p in result["players"]}
            transfers_out = squad_ids - new_squad_ids
            transfers_in = new_squad_ids - squad_ids

            # Late-season: reduce effective hit cost (solver applied -4,
            # but in late season we value hits at -3)
            if late_season:
                hits = max(0, len(transfers_in) - ft)
                if hits > 0:
                    discount = hits * (
                        solver_cfg.hit_cost - strategy_cfg.late_season_hit_cost
                    )
                    pts += discount

            # Build player metadata lookup from gw_df for transfers_out
            out_meta: dict[int, dict] = {}
            for pid in transfers_out:
                match = gw_df.loc[gw_df["player_id"] == pid]
                if not match.empty:
                    row = match.iloc[0]
                    out_meta[pid] = {
                        "web_name": row.get("web_name", "Unknown"),
                        "position": row.get("position", ""),
                        "cost": (
                            float(row["cost"])
                            if "cost" in row and pd.notna(row.get("cost"))
                            else 0
                        ),
                    }

            return {
                "gw": gw,
                "transfers_in": [
                    {
                        "player_id": p["player_id"],
                        "web_name": p.get("web_name", "") or "",
                        "position": p.get("position", "") or "",
                        "cost": p.get("cost") or 0,
                        "team_code": p.get("team_code"),
                        "predicted_points": round(
                            p.get("predicted_points") or 0, 2,
                        ),
                        "captain_score": round(
                            p.get("captain_score") or 0, 2,
                        ),
                    }
                    for p in result["players"]
                    if p["player_id"] in transfers_in
                ],
                "transfers_out": [
                    {
                        "player_id": pid,
                        "web_name": (
                            out_meta.get(pid, {}).get("web_name") or "Unknown"
                        ),
                        "position": (
                            out_meta.get(pid, {}).get("position") or ""
                        ),
                        "cost": out_meta.get(pid, {}).get("cost") or 0,
                    }
                    for pid in transfers_out
                ],
                "ft_used": len(transfers_in),
                "ft_available": ft,
                "predicted_points": round(pts, 2),
                "base_points": round(base_pts, 2),
                "squad_ids": list(new_squad_ids),
                "chip": gw_chip,
            }
        else:
            # Solver failed, keep current squad
            squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)]
            pts = self._squad_points_with_captain(squad_preds)
            base_pts = pts

            # Apply BB/TC even when solver fails
            if gw_chip == "bboost":
                top11 = self._select_formation_xi(squad_preds)
                bench = squad_preds[~squad_preds.index.isin(top11.index)]
                pts += bench["predicted_points"].sum()
            elif gw_chip == "3xc" and not squad_preds.empty:
                top11 = self._select_formation_xi(squad_preds)
                score_col = (
                    "captain_score"
                    if "captain_score" in top11.columns
                    else "predicted_points"
                )
                if not top11.empty:
                    cap_idx = top11[score_col].idxmax()
                    pts += top11.loc[cap_idx, "predicted_points"]

            return {
                "gw": gw,
                "transfers_in": [],
                "transfers_out": [],
                "ft_used": 0,
                "ft_available": ft,
                "predicted_points": round(pts, 2),
                "base_points": round(base_pts, 2),
                "squad_ids": list(squad_ids),
                "chip": gw_chip,
            }

    # ── Rationale ─────────────────────────────────────────────────────

    def _build_rationale(self, step: dict, original_ft: int) -> str:
        """Build natural-language rationale for a planning step."""
        gw = step["gw"]
        ft_used = step["ft_used"]
        ft_avail = step["ft_available"]
        transfers_in = step.get("transfers_in", [])
        chip = step.get("chip")

        if chip in ("wildcard", "freehit"):
            chip_label = "Wildcard" if chip == "wildcard" else "Free Hit"
            return f"GW{gw}: Activate {chip_label} -- full squad rebuild"

        if ft_used == 0:
            if ft_avail < _FT_MAX_BANK:
                return (
                    f"GW{gw}: Bank transfer (save for next week, "
                    f"{ft_avail}->{min(ft_avail + 1, _FT_MAX_BANK)} FTs)"
                )
            else:
                return (
                    f"GW{gw}: No valuable transfers found "
                    f"(FTs maxed at {_FT_MAX_BANK})"
                )
        else:
            names = [t.get("web_name", "?") for t in transfers_in[:3]]
            names_str = ", ".join(names)
            return f"GW{gw}: Use {ft_used} FT(s) -- bring in {names_str}"
