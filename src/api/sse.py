"""SSE broadcasting and background task management."""

import json
import queue
import threading
from typing import Callable

from src.logging_config import get_logger

log = get_logger(__name__)

_task_lock = threading.Lock()
_current_task: dict | None = None
_sse_queues: list[queue.Queue] = []
_sse_queues_lock = threading.Lock()

# Pipeline cache (data + features)
pipeline_cache: dict = {}
pipeline_lock = threading.Lock()

# Backtest results
backtest_results: dict | None = None


def broadcast(msg: str, event: str = "progress") -> None:
    """Send an SSE message to all connected clients."""
    data = json.dumps({"message": msg, "event": event})
    with _sse_queues_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


def run_in_background(name: str, fn: Callable) -> bool:
    """Run fn in a background thread, guarded by _task_lock."""
    global _current_task

    if not _task_lock.acquire(blocking=False):
        return False

    def wrapper():
        global _current_task
        try:
            broadcast(f"Starting: {name}", event="task_start")
            fn()
            broadcast(f"Finished: {name}", event="task_done")
        except Exception as exc:
            log.error("Background task %s failed: %s", name, exc, exc_info=True)
            broadcast(f"Error: {exc}", event="task_error")
        finally:
            _current_task = None
            _task_lock.release()

    t = threading.Thread(target=wrapper, daemon=True)
    _current_task = {"name": name, "thread": t}
    t.start()
    return True


def get_current_task() -> dict | None:
    """Return info about currently running task, or None."""
    return _current_task


def create_sse_stream():
    """Create an SSE event stream generator."""
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sse_queues_lock:
        _sse_queues.append(q)

    def stream():
        try:
            task = _current_task
            if task:
                payload = json.dumps({"message": f"Running: {task['name']}", "event": "status"})
                yield f"data: {payload}\n\n"
            else:
                payload = json.dumps({"message": "Idle", "event": "status"})
                yield f"data: {payload}\n\n"
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield f": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_queues_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    return stream()
