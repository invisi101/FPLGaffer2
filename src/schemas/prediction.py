"""Pydantic schemas for prediction output."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRow(BaseModel):
    """Single player prediction for one gameweek."""

    player_id: int
    web_name: str
    position: str  # GKP, DEF, MID, FWD
    team_code: int | None = None
    cost: float | None = None
    chance_of_playing: float | None = None

    # Core predictions
    predicted_next_gw_points: float = 0.0
    predicted_next_3gw_points: float | None = None
    captain_score: float | None = None
    q80: float | None = None

    # Decomposed components (optional)
    pred_goals: float | None = None
    pred_assists: float | None = None
    pred_cs: float | None = None
    pred_bonus: float | None = None
    pred_goals_conceded: float | None = None
    pred_saves: float | None = None
    pred_defcon: float | None = None

    # Market data
    ownership: float | None = None
    transfer_momentum: float | None = None


class PredictionOutput(BaseModel):
    """Complete prediction output for a gameweek."""

    players: list[PredictionRow] = Field(default_factory=list)
    gameweek: int
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str | None = None
    season: str | None = None

    @property
    def count(self) -> int:
        return len(self.players)

    def top_n(self, n: int = 10) -> list[PredictionRow]:
        """Return top N players by predicted points."""
        return sorted(
            self.players,
            key=lambda p: p.predicted_next_gw_points,
            reverse=True,
        )[:n]
