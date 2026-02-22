"""NaN/Inf scrubbing utilities for JSON-safe output."""

import math
from typing import Any

import numpy as np


def scrub_nan(records: list[dict]) -> list[dict]:
    """Replace NaN/inf with None in a list of dicts for valid JSON."""
    result = []
    for row in records:
        cleaned = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                cleaned[k] = None
            elif isinstance(v, (np.floating, np.integer)):
                fv = float(v)
                cleaned[k] = None if (math.isnan(fv) or math.isinf(fv)) else fv
            else:
                cleaned[k] = v
        result.append(cleaned)
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, returning default for NaN/None/inf."""
    if value is None:
        return default
    try:
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def safe_round(value: Any, decimals: int = 2, default: float = 0.0) -> float:
    """Round a value safely, returning default for NaN/None."""
    return round(safe_float(value, default), decimals)
