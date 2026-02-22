"""Common DataFrame operations used across modules."""

import pandas as pd


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Merge DataFrames, handling empty right gracefully."""
    if right.empty:
        return left
    return left.merge(right, **kwargs)


def ensure_columns(df: pd.DataFrame, columns: list[str], fill: float = 0.0) -> pd.DataFrame:
    """Ensure all columns exist, filling missing ones with a default."""
    for col in columns:
        if col not in df.columns:
            df[col] = fill
    return df
