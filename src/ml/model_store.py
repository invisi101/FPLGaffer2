"""Save and load trained models with metadata.

Provides a thin persistence layer over joblib, standardising file names
and making it easy to swap storage backends later.
"""

from __future__ import annotations

import logging

import joblib

from src.paths import MODEL_DIR

log = logging.getLogger(__name__)


def _ensure_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -- Mean / quantile models --------------------------------------------------

def save_model(model, position: str, target: str, metadata: dict, *, suffix: str = "") -> None:
    """Persist a trained model (mean or quantile) to *MODEL_DIR*.

    *metadata* is merged with the model object into a single joblib blob.
    The optional *suffix* (e.g. ``"_q80"``) distinguishes quantile variants.
    """
    _ensure_dir()
    path = MODEL_DIR / f"xgb_{position}_{target}{suffix}.joblib"
    payload = {"model": model, **metadata}
    joblib.dump(payload, path)
    log.info("Model saved to %s", path)


def load_model(position: str, target: str, *, suffix: str = "") -> dict | None:
    """Load a mean/quantile model from disk.  Returns *None* if absent."""
    path = MODEL_DIR / f"xgb_{position}_{target}{suffix}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


# -- Decomposed sub-models ---------------------------------------------------

def save_sub_model(model, position: str, component: str, metadata: dict) -> None:
    """Persist a decomposed sub-model to *MODEL_DIR*."""
    _ensure_dir()
    path = MODEL_DIR / f"xgb_{position}_sub_{component}.joblib"
    payload = {"model": model, **metadata}
    joblib.dump(payload, path)
    log.info("Sub-model saved to %s", path)


def load_sub_model(position: str, component: str) -> dict | None:
    """Load a decomposed sub-model from disk.  Returns *None* if absent."""
    path = MODEL_DIR / f"xgb_{position}_sub_{component}.joblib"
    if path.exists():
        return joblib.load(path)
    return None
