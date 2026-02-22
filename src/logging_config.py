"""Python logging configuration for the FPL Gaffer application."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with console handler."""
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
