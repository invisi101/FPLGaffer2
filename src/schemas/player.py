"""Pydantic schemas for FPL player, squad, and transfer data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Player(BaseModel):
    """Core player representation used across solver and strategy layers."""

    player_id: int
    web_name: str
    position: str  # GKP, DEF, MID, FWD
    cost: float  # Price in 0.1m units (e.g. 100 = 10.0m)
    team_code: int
    predicted_points: float = 0.0
    captain_score: float | None = None
    chance_of_playing: float | None = None
    ownership: float | None = None


class Squad(BaseModel):
    """A complete 15-player FPL squad with starter/bench split."""

    players: list[Player] = Field(default_factory=list)
    starters: list[Player] = Field(default_factory=list)
    bench: list[Player] = Field(default_factory=list)
    captain_id: int | None = None
    vice_captain_id: int | None = None
    total_cost: float = 0.0
    starting_points: float = 0.0

    @property
    def squad_ids(self) -> set[int]:
        """Set of player IDs in the squad."""
        return {p.player_id for p in self.players}


class Transfer(BaseModel):
    """A single transfer: one player out, one player in."""

    player_in: Player
    player_out: Player

    @property
    def cost_delta(self) -> float:
        """Cost change from this transfer (positive = more expensive)."""
        return self.player_in.cost - self.player_out.cost


class TransferPlan(BaseModel):
    """A set of transfers for a single gameweek."""

    gameweek: int
    transfers: list[Transfer] = Field(default_factory=list)
    free_transfers_used: int = 0
    hits: int = 0
    hit_cost: float = 0.0

    @property
    def total_transfers(self) -> int:
        return len(self.transfers)
