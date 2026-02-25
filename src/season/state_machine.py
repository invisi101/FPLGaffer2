"""GW lifecycle state machine.

Phases:
    PLANNING → READY → LIVE → COMPLETE → PLANNING (next GW)
                                       → SEASON_OVER (GW38)
"""

from __future__ import annotations

from enum import Enum


class GWPhase(str, Enum):
    PLANNING = "planning"
    READY = "ready"
    LIVE = "live"
    COMPLETE = "complete"
    SEASON_OVER = "season_over"


# Valid (from -> {to, ...}) transitions.
_TRANSITIONS: dict[GWPhase, set[GWPhase]] = {
    GWPhase.PLANNING: {GWPhase.READY},
    GWPhase.READY: {GWPhase.LIVE},
    GWPhase.LIVE: {GWPhase.COMPLETE},
    GWPhase.COMPLETE: {GWPhase.PLANNING, GWPhase.SEASON_OVER},
    GWPhase.SEASON_OVER: set(),
}


def can_transition(from_phase: GWPhase, to_phase: GWPhase) -> bool:
    """Return True if *from_phase* → *to_phase* is a valid transition."""
    return to_phase in _TRANSITIONS.get(from_phase, set())


def next_phase(phase: GWPhase, *, is_final_gw: bool = False) -> GWPhase | None:
    """Return the default next phase.

    COMPLETE branches: SEASON_OVER when *is_final_gw*, else PLANNING.
    SEASON_OVER has no successor (returns None).
    """
    if phase is GWPhase.PLANNING:
        return GWPhase.READY
    if phase is GWPhase.READY:
        return GWPhase.LIVE
    if phase is GWPhase.LIVE:
        return GWPhase.COMPLETE
    if phase is GWPhase.COMPLETE:
        return GWPhase.SEASON_OVER if is_final_gw else GWPhase.PLANNING
    return None  # SEASON_OVER


def detect_phase(
    *,
    has_recommendation: bool,
    deadline_passed: bool,
    all_fixtures_finished: bool,
) -> GWPhase:
    """Detect the current GW phase from real-world state.

    Priority (strongest signal first):
        1. all_fixtures_finished → COMPLETE
        2. deadline_passed       → LIVE
        3. has_recommendation    → READY
        4. else                  → PLANNING
    """
    if all_fixtures_finished:
        return GWPhase.COMPLETE
    if deadline_passed:
        return GWPhase.LIVE
    if has_recommendation:
        return GWPhase.READY
    return GWPhase.PLANNING
