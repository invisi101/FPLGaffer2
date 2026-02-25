"""Background scheduler that drives the GW state machine."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_scheduler(
    manager_id: int,
    db_path: Path,
    interval_seconds: int = 300,
) -> None:
    """Start the background tick loop.

    Calls ``SeasonManager.tick()`` every *interval_seconds* (default 5 min).
    The thread is a daemon so it dies with the process.
    """
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return  # Already running

    _stop_event.clear()

    def _loop() -> None:
        from src.api.sse import broadcast
        from src.season.manager import SeasonManager

        mgr = SeasonManager(db_path=db_path)
        while not _stop_event.is_set():
            try:
                alerts = mgr.tick(manager_id)
                for alert in alerts:
                    broadcast(alert.get("message", str(alert)), event="alert")
            except Exception:
                logger.exception("Scheduler tick failed")
            _stop_event.wait(interval_seconds)

    _scheduler_thread = threading.Thread(
        target=_loop, daemon=True, name="gw-scheduler",
    )
    _scheduler_thread.start()
    logger.info(
        "Scheduler started: manager_id=%d, interval=%ds",
        manager_id, interval_seconds,
    )


def stop_scheduler() -> None:
    """Stop the background tick loop."""
    _stop_event.set()
    logger.info("Scheduler stop requested")
