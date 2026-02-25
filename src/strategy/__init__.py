"""Strategy layer — transfer planning, availability adjustments, and price tracking.

Re-exports the main classes and functions for convenience::

    from src.strategy import MultiWeekPlanner, apply_availability_adjustments, ...
"""

from src.strategy.price_tracker import (
    get_price_alerts,
    get_price_history,
    predict_price_changes,
    track_prices,
)
from src.strategy.reactive import (
    apply_availability_adjustments,
    check_squad_injuries,
)
from src.strategy.transfer_planner import MultiWeekPlanner

__all__ = [
    # Classes
    "MultiWeekPlanner",
    # Reactive functions
    "apply_availability_adjustments",
    "check_squad_injuries",
    # Price functions
    "track_prices",
    "get_price_alerts",
    "predict_price_changes",
    "get_price_history",
]
