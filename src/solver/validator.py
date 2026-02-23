"""Post-solve FPL rule validation.

Validates solver output against FPL rules to catch any constraint
violations that slipped through the MILP (e.g. due to rounding or
data issues).
"""

from __future__ import annotations

from src.config import solver_cfg


def validate_solver_output(
    result: dict,
    budget: float,
    current_squad_ids: set[int] | None = None,
    free_transfers: int | None = None,
) -> list[str]:
    """Validate a solver result dict against FPL rules.

    Returns an empty list if valid, or a list of error strings describing
    each violation found.
    """
    errors: list[str] = []

    if result is None:
        return ["Solver returned None (no feasible solution)"]

    players = result.get("players", [])
    starters = result.get("starters", [])
    bench = result.get("bench", [])
    total_cost = result.get("total_cost", 0.0)
    captain_id = result.get("captain_id")

    # --- Squad size ---
    if len(players) != solver_cfg.squad_size:
        errors.append(
            f"Squad size: expected {solver_cfg.squad_size}, got {len(players)}"
        )

    # --- Position counts (full squad) ---
    squad_positions = solver_cfg.squad_positions
    pos_counts: dict[str, int] = {}
    for p in players:
        pos = p.get("position", "")
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    for pos, required in squad_positions.items():
        actual = pos_counts.get(pos, 0)
        if actual != required:
            errors.append(
                f"Squad {pos}: need {required}, got {actual}"
            )

    # --- Team cap ---
    team_counts: dict[int, int] = {}
    for p in players:
        tc = p.get("team_code")
        if tc is not None:
            team_counts[tc] = team_counts.get(tc, 0) + 1
    for tc, count in team_counts.items():
        if count > solver_cfg.team_cap:
            errors.append(
                f"Team {tc}: {count} players (max {solver_cfg.team_cap})"
            )

    # --- Budget ---
    if total_cost > budget + 0.1:  # Small tolerance for floating point
        errors.append(
            f"Over budget: {total_cost} > {budget}"
        )

    # --- Starting XI count ---
    if len(starters) != solver_cfg.starting_xi:
        errors.append(
            f"Starting XI: expected {solver_cfg.starting_xi}, got {len(starters)}"
        )

    # --- Formation validity ---
    formation_limits = solver_cfg.formation_limits
    starter_pos_counts: dict[str, int] = {}
    for p in starters:
        pos = p.get("position", "")
        starter_pos_counts[pos] = starter_pos_counts.get(pos, 0) + 1
    for pos, (lo, hi) in formation_limits.items():
        actual = starter_pos_counts.get(pos, 0)
        if actual < lo or actual > hi:
            errors.append(
                f"Formation {pos}: need {lo}-{hi}, got {actual}"
            )

    # --- Captain is a starter ---
    if captain_id is not None:
        starter_ids = {p.get("player_id") for p in starters}
        if captain_id not in starter_ids:
            errors.append(
                f"Captain {captain_id} is not in the starting XI"
            )

    # --- Bench size ---
    expected_bench = solver_cfg.squad_size - solver_cfg.starting_xi
    if len(bench) != expected_bench:
        errors.append(
            f"Bench size: expected {expected_bench}, got {len(bench)}"
        )

    # --- Unique player IDs ---
    player_ids = [p.get("player_id") for p in players]
    if len(set(player_ids)) != len(player_ids):
        errors.append("Duplicate player IDs in squad")

    # --- Transfer count check (if applicable) ---
    if current_squad_ids is not None and free_transfers is not None:
        # Trust the solver's reported hits (which correctly accounts for
        # forced replacements) rather than recomputing from raw set
        # differences, which would overcount hits when unavailable players
        # are force-swapped out.
        reported_hits = result.get("hits", 0)
        hit_cost = result.get("hit_cost", 0.0)
        expected_hit_cost = reported_hits * solver_cfg.hit_cost
        if abs(hit_cost - expected_hit_cost) > 0.1:
            errors.append(
                f"Hit cost mismatch: result says {hit_cost}, "
                f"expected {expected_hit_cost} ({reported_hits} hits)"
            )

    return errors
