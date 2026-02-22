"""Strategy layer — chip evaluation, transfer planning, captain planning,
plan synthesis, reactive re-planning, and price tracking.

Re-exports the main classes and functions for convenience::

    from src.strategy import ChipEvaluator, MultiWeekPlanner, ...
"""

from src.strategy.captain_planner import CaptainPlanner
from src.strategy.chip_evaluator import ChipEvaluator
from src.strategy.plan_synthesizer import PlanSynthesizer
from src.strategy.price_tracker import (
    get_price_alerts,
    get_price_history,
    predict_price_changes,
    track_prices,
)
from src.strategy.reactive import (
    apply_availability_adjustments,
    check_plan_health,
    detect_plan_invalidation,
)
from src.strategy.transfer_planner import MultiWeekPlanner

__all__ = [
    # Classes
    "ChipEvaluator",
    "MultiWeekPlanner",
    "CaptainPlanner",
    "PlanSynthesizer",
    # Reactive functions
    "detect_plan_invalidation",
    "apply_availability_adjustments",
    "check_plan_health",
    # Price functions
    "track_prices",
    "get_price_alerts",
    "predict_price_changes",
    "get_price_history",
]
