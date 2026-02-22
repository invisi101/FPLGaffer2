"""FPL rule constants and Pydantic validators.

Encodes the official FPL game rules as constants and validation models,
used by the solver, strategy, and season management layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Squad composition
SQUAD_SIZE = 15
STARTING_XI = 11
BENCH_SIZE = SQUAD_SIZE - STARTING_XI  # 4

# Position requirements (squad)
SQUAD_POSITIONS: dict[str, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Formation limits (starting XI)
FORMATION_LIMITS: dict[str, tuple[int, int]] = {
    "GKP": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

# Team cap: max players from one club
TEAM_CAP = 3

# Budget (in 0.1m units, e.g. 1000 = 100.0m)
DEFAULT_BUDGET = 1000.0

# Transfers
HIT_COST = 4.0  # Points deducted per extra transfer
MAX_FREE_TRANSFERS = 5  # Maximum banked FTs
DEFAULT_FREE_TRANSFERS = 1  # FTs gained per gameweek

# Season structure
TOTAL_GAMEWEEKS = 38
FIRST_HALF_END = 19  # GW1-19 is first half
SECOND_HALF_START = 20  # GW20-38 is second half


# ---------------------------------------------------------------------------
# Chip definitions
# ---------------------------------------------------------------------------

class ChipType(str, Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


# Chip rules
CHIPS_PER_HALF = {ChipType.WILDCARD, ChipType.FREE_HIT, ChipType.BENCH_BOOST, ChipType.TRIPLE_CAPTAIN}

# Chips that preserve FTs (no reset, no accrual)
FT_PRESERVING_CHIPS = {ChipType.WILDCARD, ChipType.FREE_HIT}

# Chips that are one-GW only (no permanent squad change)
ONE_GW_CHIPS = {ChipType.BENCH_BOOST, ChipType.TRIPLE_CAPTAIN}

# Chips that revert squad after use
SQUAD_REVERTING_CHIPS = {ChipType.FREE_HIT}

# Chips that allow unlimited transfers
UNLIMITED_TRANSFER_CHIPS = {ChipType.WILDCARD, ChipType.FREE_HIT}

# Chips that make permanent squad changes
PERMANENT_SQUAD_CHIPS = {ChipType.WILDCARD}


def get_chip_half(gameweek: int) -> int:
    """Return which half a gameweek belongs to (1 or 2)."""
    if gameweek <= FIRST_HALF_END:
        return 1
    return 2


def chips_available_in_half(
    used_chips: dict[ChipType, int | None],
    half: int,
) -> list[ChipType]:
    """Return which chips are still available in the given half.

    used_chips maps ChipType -> gameweek it was used (None if unused).
    Each chip can be used once per half (GW1-19, GW20-38).
    """
    available = []
    half_range = range(1, FIRST_HALF_END + 1) if half == 1 else range(SECOND_HALF_START, TOTAL_GAMEWEEKS + 1)
    for chip in ChipType:
        used_gw = used_chips.get(chip)
        if used_gw is None or used_gw not in half_range:
            available.append(chip)
    return available


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

class SquadValidation(BaseModel):
    """Validates a complete 15-player squad."""

    player_ids: list[int] = Field(..., min_length=SQUAD_SIZE, max_length=SQUAD_SIZE)
    positions: list[str]  # Parallel to player_ids
    team_codes: list[int]  # Parallel to player_ids
    total_cost: float
    budget: float = DEFAULT_BUDGET

    @model_validator(mode="after")
    def validate_squad_rules(self) -> "SquadValidation":
        errors: list[str] = []

        # Position counts
        pos_counts: dict[str, int] = {}
        for pos in self.positions:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for pos, required in SQUAD_POSITIONS.items():
            actual = pos_counts.get(pos, 0)
            if actual != required:
                errors.append(f"Need {required} {pos}, got {actual}")

        # Team cap
        team_counts: dict[int, int] = {}
        for tc in self.team_codes:
            team_counts[tc] = team_counts.get(tc, 0) + 1
        for tc, count in team_counts.items():
            if count > TEAM_CAP:
                errors.append(f"Team {tc} has {count} players (max {TEAM_CAP})")

        # Budget
        if self.total_cost > self.budget:
            errors.append(f"Over budget: {self.total_cost} > {self.budget}")

        # Unique players
        if len(set(self.player_ids)) != len(self.player_ids):
            errors.append("Duplicate player IDs in squad")

        if errors:
            raise ValueError("; ".join(errors))
        return self


class FormationValidation(BaseModel):
    """Validates a starting XI formation."""

    starter_positions: list[str] = Field(..., min_length=STARTING_XI, max_length=STARTING_XI)

    @model_validator(mode="after")
    def validate_formation_rules(self) -> "FormationValidation":
        errors: list[str] = []

        pos_counts: dict[str, int] = {}
        for pos in self.starter_positions:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        for pos, (lo, hi) in FORMATION_LIMITS.items():
            actual = pos_counts.get(pos, 0)
            if actual < lo or actual > hi:
                errors.append(f"{pos}: need {lo}-{hi}, got {actual}")

        if errors:
            raise ValueError("; ".join(errors))
        return self


class TransferValidation(BaseModel):
    """Validates transfer legality (hit calculation)."""

    num_transfers: int = Field(..., ge=0)
    free_transfers: int = Field(..., ge=0, le=MAX_FREE_TRANSFERS)
    chip_active: ChipType | None = None

    @property
    def hits(self) -> int:
        """Number of point-costing hits."""
        if self.chip_active in UNLIMITED_TRANSFER_CHIPS:
            return 0
        return max(0, self.num_transfers - self.free_transfers)

    @property
    def hit_cost(self) -> float:
        """Total point cost of hits."""
        return self.hits * HIT_COST

    @property
    def remaining_free_transfers(self) -> int:
        """FTs remaining after these transfers (before next GW accrual)."""
        if self.chip_active in UNLIMITED_TRANSFER_CHIPS:
            return self.free_transfers  # FTs preserved
        return max(0, self.free_transfers - self.num_transfers)


class ChipValidation(BaseModel):
    """Validates chip usage legality."""

    chip: ChipType
    gameweek: int = Field(..., ge=1, le=TOTAL_GAMEWEEKS)
    used_chips: dict[str, int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_chip_rules(self) -> "ChipValidation":
        errors: list[str] = []
        half = get_chip_half(self.gameweek)
        half_range = range(1, FIRST_HALF_END + 1) if half == 1 else range(SECOND_HALF_START, TOTAL_GAMEWEEKS + 1)

        # Check if this chip was already used in this half
        chip_key = self.chip.value
        used_gw = self.used_chips.get(chip_key)
        if used_gw is not None and used_gw in half_range:
            errors.append(
                f"{self.chip.value} already used in GW{used_gw} "
                f"(half {half})"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def _extract_errors(exc: Exception) -> list[str]:
    """Extract human-readable error messages from a Pydantic ValidationError."""
    # Pydantic wraps ValueError messages in its own format.
    # Try to extract the original semicolon-joined message.
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        errors = []
        for err in exc.errors():
            msg = err.get("msg", "")
            # Pydantic prefixes with "Value error, "
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, "):]
            errors.extend(msg.split("; "))
        return errors
    return [str(exc)]


def validate_squad(
    player_ids: list[int],
    positions: list[str],
    team_codes: list[int],
    total_cost: float,
    budget: float = DEFAULT_BUDGET,
) -> list[str]:
    """Validate a squad and return list of error strings (empty if valid)."""
    try:
        SquadValidation(
            player_ids=player_ids,
            positions=positions,
            team_codes=team_codes,
            total_cost=total_cost,
            budget=budget,
        )
        return []
    except Exception as e:
        return _extract_errors(e)


def validate_formation(starter_positions: list[str]) -> list[str]:
    """Validate a formation and return list of error strings (empty if valid)."""
    try:
        FormationValidation(starter_positions=starter_positions)
        return []
    except Exception as e:
        return _extract_errors(e)


def validate_transfers(
    num_transfers: int,
    free_transfers: int,
    chip_active: ChipType | None = None,
) -> dict[str, Any]:
    """Validate transfers and return hit information.

    Returns dict with keys: hits, hit_cost, remaining_free_transfers.
    """
    tv = TransferValidation(
        num_transfers=num_transfers,
        free_transfers=free_transfers,
        chip_active=chip_active,
    )
    return {
        "hits": tv.hits,
        "hit_cost": tv.hit_cost,
        "remaining_free_transfers": tv.remaining_free_transfers,
    }


def compute_next_gw_fts(
    current_fts: int,
    transfers_made: int,
    chip_active: ChipType | None = None,
) -> int:
    """Compute free transfers available for the NEXT gameweek.

    FPL rules:
    - You get +1 FT per GW, max 5 banked.
    - WC and FH preserve FTs at pre-chip count (no reset, no accrual).
    - Normal transfers: remaining FTs + 1, capped at 5.
    """
    if chip_active in FT_PRESERVING_CHIPS:
        # FTs preserved at pre-chip level
        return min(current_fts + 1, MAX_FREE_TRANSFERS)

    remaining = max(0, current_fts - transfers_made)
    return min(remaining + DEFAULT_FREE_TRANSFERS, MAX_FREE_TRANSFERS)
