# Season Manager v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 1,721-line SeasonManager god class with a state-machine-driven autopilot that auto-detects GW completion, auto-generates recommendations, and lets the user accept or override transfers before each deadline.

**Architecture:** A GW state machine (`PLANNING → READY → LIVE → COMPLETE → repeat`) drives all automation. The `tick()` method is called periodically and advances phases based on real-world events (fixtures finishing, deadlines passing). The MILP solver + MultiWeekPlanner still power transfer recommendations with 5-GW lookahead. ChipEvaluator, CaptainPlanner, and PlanSynthesizer are removed — chips are user-controlled, captain is part of the 1-GW recommendation.

**Tech Stack:** Python 3.12, Flask, SQLite, XGBoost, PuLP (MILP), existing SSE infrastructure.

**Branch:** `season-manager-v2` (all work isolated from main)

**Design doc:** `docs/plans/2026-02-25-season-manager-redesign.md`

---

## Phase 1: Branch + State Machine Foundation

### Task 1: Create Branch

**Files:**
- None (git operation only)

**Step 1: Create and checkout new branch**

```bash
git checkout -b season-manager-v2
```

**Step 2: Verify branch**

```bash
git branch --show-current
```
Expected: `season-manager-v2`

**Step 3: Run existing tests to establish baseline**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: 103+ tests pass, 0 failures

**Step 4: Commit (empty marker)**

```bash
git commit --allow-empty -m "chore: start season-manager-v2 branch"
```

---

### Task 2: State Machine Module

**Files:**
- Create: `src/season/state_machine.py`
- Test: `tests/test_state_machine.py`

**Step 1: Write tests for phase enum and transitions**

```python
# tests/test_state_machine.py
"""Tests for GW state machine."""
import pytest
from src.season.state_machine import GWPhase, can_transition, next_phase


class TestGWPhase:
    def test_all_phases_exist(self):
        assert GWPhase.PLANNING is not None
        assert GWPhase.READY is not None
        assert GWPhase.LIVE is not None
        assert GWPhase.COMPLETE is not None
        assert GWPhase.SEASON_OVER is not None

    def test_valid_transitions(self):
        assert can_transition(GWPhase.PLANNING, GWPhase.READY)
        assert can_transition(GWPhase.READY, GWPhase.LIVE)
        assert can_transition(GWPhase.LIVE, GWPhase.COMPLETE)
        assert can_transition(GWPhase.COMPLETE, GWPhase.PLANNING)
        assert can_transition(GWPhase.COMPLETE, GWPhase.SEASON_OVER)

    def test_invalid_transitions(self):
        assert not can_transition(GWPhase.PLANNING, GWPhase.LIVE)
        assert not can_transition(GWPhase.READY, GWPhase.COMPLETE)
        assert not can_transition(GWPhase.LIVE, GWPhase.PLANNING)
        assert not can_transition(GWPhase.SEASON_OVER, GWPhase.PLANNING)

    def test_next_phase(self):
        assert next_phase(GWPhase.PLANNING) == GWPhase.READY
        assert next_phase(GWPhase.READY) == GWPhase.LIVE
        assert next_phase(GWPhase.LIVE) == GWPhase.COMPLETE
        assert next_phase(GWPhase.SEASON_OVER) is None

    def test_complete_branches(self):
        """COMPLETE can go to PLANNING (next GW) or SEASON_OVER (GW38)."""
        assert next_phase(GWPhase.COMPLETE, is_final_gw=False) == GWPhase.PLANNING
        assert next_phase(GWPhase.COMPLETE, is_final_gw=True) == GWPhase.SEASON_OVER


class TestPhaseDetection:
    def test_detect_phase_from_bootstrap_pre_deadline(self):
        """Before deadline, with a recommendation, should be READY."""
        from src.season.state_machine import detect_phase
        # deadline in the future, recommendation exists
        phase = detect_phase(
            has_recommendation=True,
            deadline_passed=False,
            all_fixtures_finished=False,
        )
        assert phase == GWPhase.READY

    def test_detect_phase_no_recommendation(self):
        """No recommendation yet means PLANNING."""
        from src.season.state_machine import detect_phase
        phase = detect_phase(
            has_recommendation=False,
            deadline_passed=False,
            all_fixtures_finished=False,
        )
        assert phase == GWPhase.PLANNING

    def test_detect_phase_deadline_passed(self):
        """After deadline, before fixtures done = LIVE."""
        from src.season.state_machine import detect_phase
        phase = detect_phase(
            has_recommendation=True,
            deadline_passed=True,
            all_fixtures_finished=False,
        )
        assert phase == GWPhase.LIVE

    def test_detect_phase_all_done(self):
        """All fixtures finished = COMPLETE."""
        from src.season.state_machine import detect_phase
        phase = detect_phase(
            has_recommendation=True,
            deadline_passed=True,
            all_fixtures_finished=True,
        )
        assert phase == GWPhase.COMPLETE
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py -v
```
Expected: FAIL (module doesn't exist yet)

**Step 3: Implement state machine**

```python
# src/season/state_machine.py
"""GW lifecycle state machine.

Phases:
    PLANNING → READY → LIVE → COMPLETE → PLANNING (next GW)
                                       → SEASON_OVER (GW38)
"""
from enum import Enum


class GWPhase(str, Enum):
    """Gameweek lifecycle phase."""
    PLANNING = "planning"       # Generating predictions + recommendations
    READY = "ready"             # Recommendation available, user reviews
    LIVE = "live"               # Deadline passed, GW in progress
    COMPLETE = "complete"       # All fixtures done, recording results
    SEASON_OVER = "season_over" # GW38 complete


# Valid transitions: {from_phase: {to_phase, ...}}
_TRANSITIONS: dict[GWPhase, set[GWPhase]] = {
    GWPhase.PLANNING: {GWPhase.READY},
    GWPhase.READY: {GWPhase.LIVE},
    GWPhase.LIVE: {GWPhase.COMPLETE},
    GWPhase.COMPLETE: {GWPhase.PLANNING, GWPhase.SEASON_OVER},
    GWPhase.SEASON_OVER: set(),
}


def can_transition(from_phase: GWPhase, to_phase: GWPhase) -> bool:
    """Check if a phase transition is valid."""
    return to_phase in _TRANSITIONS.get(from_phase, set())


def next_phase(phase: GWPhase, *, is_final_gw: bool = False) -> GWPhase | None:
    """Get the default next phase.

    COMPLETE branches: PLANNING if more GWs remain, SEASON_OVER if GW38.
    """
    if phase == GWPhase.COMPLETE:
        return GWPhase.SEASON_OVER if is_final_gw else GWPhase.PLANNING
    nxt = {
        GWPhase.PLANNING: GWPhase.READY,
        GWPhase.READY: GWPhase.LIVE,
        GWPhase.LIVE: GWPhase.COMPLETE,
    }
    return nxt.get(phase)


def detect_phase(
    *,
    has_recommendation: bool,
    deadline_passed: bool,
    all_fixtures_finished: bool,
) -> GWPhase:
    """Detect current phase from real-world state.

    Used on startup or when recovering from a crash to determine
    where we are in the GW cycle.
    """
    if all_fixtures_finished and deadline_passed:
        return GWPhase.COMPLETE
    if deadline_passed:
        return GWPhase.LIVE
    if has_recommendation:
        return GWPhase.READY
    return GWPhase.PLANNING
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py -v
```
Expected: All pass

**Step 5: Run full test suite to verify no regressions**

```bash
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v
```
Expected: All pass (new module doesn't affect existing code)

**Step 6: Commit**

```bash
git add src/season/state_machine.py tests/test_state_machine.py
git commit -m "feat: add GW state machine module with phase enum and transitions"
```

---

### Task 3: DB Schema — Planned Squad Table + Phase Tracking

**Files:**
- Modify: `src/db/schema.py` — add `planned_squad` table, add `phase` column to `season` table
- Modify: `src/db/migrations.py` — add migration 2
- Modify: `src/db/repositories.py` — add `PlannedSquadRepository`
- Test: `tests/test_state_machine.py` (extend)

**Step 1: Write tests for planned squad DB operations**

Add to `tests/test_state_machine.py`:

```python
import tempfile, os


class TestPlannedSquadRepository:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = str(tmp_path / "test.db")
        from src.db.schema import init_schema
        from src.db.migrations import apply_migrations
        init_schema(path)
        apply_migrations(path)
        return path

    def test_save_and_get_planned_squad(self, db_path):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        squad = {
            "players": [{"id": 1, "name": "Salah"}],
            "captain_id": 1,
            "chip": None,
            "transfers_in": [],
            "transfers_out": [],
        }
        repo.save_planned_squad(season_id=1, gw=10, squad_json=squad, source="recommended")
        result = repo.get_planned_squad(season_id=1, gw=10)
        assert result is not None
        assert result["source"] == "recommended"
        assert result["squad_json"]["captain_id"] == 1

    def test_update_planned_squad(self, db_path):
        from src.db.repositories import PlannedSquadRepository
        repo = PlannedSquadRepository(db_path)
        squad1 = {"players": [], "captain_id": 1, "chip": None,
                  "transfers_in": [], "transfers_out": []}
        repo.save_planned_squad(season_id=1, gw=10, squad_json=squad1, source="recommended")

        squad2 = {"players": [], "captain_id": 2, "chip": "bboost",
                  "transfers_in": [10], "transfers_out": [5]}
        repo.save_planned_squad(season_id=1, gw=10, squad_json=squad2, source="user_override")

        result = repo.get_planned_squad(season_id=1, gw=10)
        assert result["source"] == "user_override"
        assert result["squad_json"]["captain_id"] == 2

    def test_save_and_get_phase(self, db_path):
        from src.db.repositories import SeasonRepository
        repo = SeasonRepository(db_path)
        repo.create_season(manager_id=123, season_name="2025-2026")
        season = repo.get_season(123)
        # New seasons start in PLANNING phase
        assert season.get("phase", "planning") == "planning"
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py::TestPlannedSquadRepository -v
```

**Step 3: Add `planned_squad` table to schema.py**

Add to `SCHEMA_SQL` in `src/db/schema.py`:

```sql
CREATE TABLE IF NOT EXISTS planned_squad (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    squad_json TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'recommended',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(season_id, gameweek)
);
```

**Step 4: Add `phase` column to `season` table in schema.py**

Modify the `season` CREATE TABLE to include:
```sql
phase TEXT NOT NULL DEFAULT 'planning'
```

**Step 5: Add migration 2 in `migrations.py`**

```python
def _migration_002_planned_squad_and_phase(db_path: str):
    """Add planned_squad table and phase column to season."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS planned_squad (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season_id INTEGER NOT NULL,
                gameweek INTEGER NOT NULL,
                squad_json TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'recommended',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(season_id, gameweek)
            )
        """)
        # Add phase column to season table (SQLite ALTER TABLE)
        try:
            conn.execute("ALTER TABLE season ADD COLUMN phase TEXT NOT NULL DEFAULT 'planning'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
    finally:
        conn.close()
```

Register in `_MIGRATIONS`:
```python
_MIGRATIONS = {
    1: _migration_001_initial_schema,
    2: _migration_002_planned_squad_and_phase,
}
```

**Step 6: Add `PlannedSquadRepository` to `repositories.py`**

```python
class PlannedSquadRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def save_planned_squad(self, season_id: int, gw: int,
                           squad_json: dict, source: str = "recommended"):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO planned_squad (season_id, gameweek, squad_json, source, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(season_id, gameweek) DO UPDATE SET
                   squad_json=excluded.squad_json, source=excluded.source,
                   updated_at=datetime('now')""",
                (season_id, gw, json.dumps(squad_json), source),
            )
            conn.commit()
        finally:
            conn.close()

    def get_planned_squad(self, season_id: int, gw: int) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT squad_json, source, updated_at FROM planned_squad WHERE season_id=? AND gameweek=?",
                (season_id, gw),
            ).fetchone()
            if not row:
                return None
            return {
                "squad_json": json.loads(row[0]),
                "source": row[1],
                "updated_at": row[2],
            }
        finally:
            conn.close()

    def delete_planned_squad(self, season_id: int, gw: int):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "DELETE FROM planned_squad WHERE season_id=? AND gameweek=?",
                (season_id, gw),
            )
            conn.commit()
        finally:
            conn.close()
```

**Step 7: Update SeasonRepository to read/write phase**

Add `update_phase()` and modify `get_season()` to include `phase` in the returned dict.

**Step 8: Run tests**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py -v
```
Expected: All pass

**Step 9: Run full test suite**

```bash
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v
```
Expected: All pass (additive changes only — existing tables/repos unchanged)

**Step 10: Commit**

```bash
git add src/db/schema.py src/db/migrations.py src/db/repositories.py tests/test_state_machine.py
git commit -m "feat: add planned_squad table, phase column, and PlannedSquadRepository"
```

---

## Phase 2: New SeasonManager Core

### Task 4: Slim SeasonManager — tick() and get_status()

This is the core rewrite. The new `SeasonManager` is thin orchestration around the state machine. The heavy work (predictions, MILP) is delegated to pipeline functions.

**Files:**
- Create: `src/season/manager_v2.py` (new file, parallel to old `manager.py`)
- Modify: `tests/test_state_machine.py` (extend with SeasonManager tests)

**Why a new file?** We keep `manager.py` intact for comparison and fallback. The API layer will switch to `manager_v2.py` when ready.

**Step 1: Write tests for the new SeasonManager skeleton**

Add to `tests/test_state_machine.py`:

```python
class TestSeasonManagerV2:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = str(tmp_path / "test.db")
        from src.db.schema import init_schema
        from src.db.migrations import apply_migrations
        init_schema(path)
        apply_migrations(path)
        return path

    def test_init_creates_repos(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        assert mgr.seasons is not None
        assert mgr.planned_squads is not None

    def test_get_status_no_season(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        mgr = SeasonManagerV2(db_path=db_path)
        status = mgr.get_status(manager_id=12345)
        assert status["active"] is False

    def test_get_status_with_season(self, db_path):
        from src.season.manager_v2 import SeasonManagerV2
        from src.db.repositories import SeasonRepository
        SeasonRepository(db_path).create_season(manager_id=123, season_name="2025-2026")
        mgr = SeasonManagerV2(db_path=db_path)
        status = mgr.get_status(manager_id=123)
        assert status["active"] is True
        assert status["phase"] == "planning"
```

**Step 2: Implement `SeasonManagerV2`**

Create `src/season/manager_v2.py` with:
- `__init__` — instantiate repos (including new `PlannedSquadRepository`)
- `get_status(manager_id)` — return phase, GW info, deadline, squad
- `tick(manager_id, progress_fn)` — check phase, call appropriate `_tick_*` method
- `_tick_planning(...)` — the heavy path: load data, build features, generate predictions, run MILP solver with 5-GW lookahead, save recommendation + planned squad
- `_tick_ready(...)` — lightweight: refresh bootstrap, check injuries, re-recommend if needed
- `_tick_live(...)` — fetch actual picks, compare to planned
- `_tick_complete(...)` — record results, advance to next GW

The `_tick_planning()` method contains the core logic from the current `generate_recommendation()` but without ChipEvaluator, CaptainPlanner, or PlanSynthesizer. It calls:
1. `load_all_data()` + `build_features()` + `generate_predictions()` (from existing ML pipeline)
2. `predict_multi_gw()` for 5-GW horizon
3. `apply_availability_adjustments()` for injuries (simplified from reactive.py)
4. `MultiWeekPlanner.plan_transfers()` for 5-GW-aware transfer recs (chip_plan=None, no chip strategy)
5. Extract GW+1 from planner output: transfers, captain, bench order
6. `solve_transfer_milp_with_hits()` as the primary 1-GW solver (with captain_col for joint captain optimization)
7. Save recommendation + planned squad to DB

**Key difference from old manager:** The MultiWeekPlanner is called for intelligence only — its output influences the MILP solver's player pool and transfer timing but is NOT shown to the user as a plan. The user sees the 1-GW recommendation.

**Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py -v
```

**Step 4: Commit**

```bash
git add src/season/manager_v2.py tests/test_state_machine.py
git commit -m "feat: add SeasonManagerV2 with tick() state machine and get_status()"
```

---

### Task 5: User Action Methods

**Files:**
- Modify: `src/season/manager_v2.py`
- Modify: `tests/test_state_machine.py`

Implement the READY-phase user actions:

**`accept_transfers(manager_id)`:**
- Read recommended planned squad from DB
- Set source to "accepted"
- Return updated planned squad with predictions

**`make_transfer(manager_id, player_out_id, player_in_id)`:**
- Load current planned squad
- Validate: player_out in squad, player_in not in squad, budget allows, position matches
- Swap players
- Re-run `optimize_starting_xi()` for new XI + bench order
- Re-compute predicted points
- Save with source="user_override"
- Return updated planned squad

**`undo_transfers(manager_id)`:**
- Reset planned squad to original recommendation (source="recommended")
- Return reset squad

**`lock_chip(manager_id, chip)`:**
- Validate chip is available (not used in a previous GW, correct half-season)
- Store chip on planned squad
- For WC: mark all transfers as free (no hits)
- For FH: run unconstrained MILP to build optimal squad from scratch
- For BB: include all 15 in predicted points
- For TC: triple captain's predicted points
- Return updated planned squad

**`unlock_chip(manager_id)`:**
- Remove chip from planned squad
- Revert predictions to normal
- Return updated squad

**`set_captain(manager_id, player_id)`:**
- Validate player is in starting XI
- Update captain_id on planned squad
- Recalculate predicted points (old captain back to 1x, new captain 2x)
- Return updated squad

Tests should cover:
- Each action modifies planned squad correctly
- Actions only allowed in READY phase (return error in other phases)
- Budget validation on make_transfer
- Position validation on make_transfer
- Chip availability validation

**Step 1: Write tests for each user action** (in `tests/test_state_machine.py`)

**Step 2: Implement each method**

**Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/test_state_machine.py -v
```

**Step 4: Commit**

```bash
git add src/season/manager_v2.py tests/test_state_machine.py
git commit -m "feat: add user action methods (accept, override, chip, captain)"
```

---

### Task 6: init_season() for V2

**Files:**
- Modify: `src/season/manager_v2.py`
- Modify: `tests/test_state_machine.py`

Port `init_season()` from old manager. Same backfill logic:
1. Fetch manager entry + history from FPL API
2. Create season record with phase="planning"
3. Backfill gw_snapshot for each played GW
4. Save fixture calendar
5. Track prices
6. Immediately run `_tick_planning()` to generate first recommendation

Simplifications vs old:
- No call to `generate_preseason_plan()` (pre-GW1 is just PLANNING phase)
- No call to `_run_strategy_pipeline()` — tick_planning handles it
- Phase set to "planning" then auto-advances to "ready" after recommendation generated

**Step 1: Write test**

```python
def test_init_season_creates_season_and_advances(self, db_path):
    """init_season should create season record."""
    from src.season.manager_v2 import SeasonManagerV2
    mgr = SeasonManagerV2(db_path=db_path)
    # This will fail in test without FPL API mocking — needs mock fixtures
    # See integration test pattern in test_integration.py
```

Note: Full init_season testing requires FPL API mocks. Write a focused unit test that mocks `fetch_manager_entry`, `fetch_manager_history`, etc. and verifies the DB state after init.

**Step 2: Implement**

**Step 3: Run tests**

**Step 4: Commit**

```bash
git commit -m "feat: add init_season() to SeasonManagerV2"
```

---

## Phase 3: API Layer

### Task 7: New API Blueprint

**Files:**
- Create: `src/api/season_v2_bp.py` (new blueprint, parallel to old)
- Modify: `src/api/__init__.py` (register new blueprint)

Create a new blueprint `season_v2` that exposes the new SeasonManager API:

| Method | Endpoint | Handler |
|--------|----------|---------|
| POST | `/api/v2/season/init` | `mgr.init_season(manager_id)` |
| GET | `/api/v2/season/status` | `mgr.get_status(manager_id)` |
| POST | `/api/v2/season/tick` | `mgr.tick(manager_id)` (manual trigger) |
| POST | `/api/v2/season/accept-transfers` | `mgr.accept_transfers(manager_id)` |
| POST | `/api/v2/season/make-transfer` | `mgr.make_transfer(manager_id, out, in)` |
| POST | `/api/v2/season/undo-transfers` | `mgr.undo_transfers(manager_id)` |
| POST | `/api/v2/season/lock-chip` | `mgr.lock_chip(manager_id, chip)` |
| POST | `/api/v2/season/unlock-chip` | `mgr.unlock_chip(manager_id)` |
| POST | `/api/v2/season/set-captain` | `mgr.set_captain(manager_id, player_id)` |
| GET | `/api/v2/season/fixture-lookahead` | `mgr.get_fixture_lookahead(manager_id)` |
| GET | `/api/v2/season/history` | `mgr.get_history(manager_id)` |
| GET | `/api/v2/season/available-players` | Player pool for transfer override (filtered by budget/position) |
| DELETE | `/api/v2/season/delete` | Delete season data |

**Key design decisions:**
- `/api/v2/` prefix so old and new coexist during development
- `tick` is also exposed as a POST endpoint for manual triggering (besides the background scheduler)
- `available-players` returns the filtered player pool for the transfer override UI — sorted by predicted points, filtered by budget remaining and position of the outgoing player

**Step 1: Write the blueprint** with proper error handling (wrong phase, validation errors)

**Step 2: Register in `create_app()`** alongside existing blueprints

**Step 3: Write integration test**

```python
def test_v2_status_endpoint(self):
    resp = self.client.get("/api/v2/season/status?manager_id=12345")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active" in data
    assert "phase" in data
```

**Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_integration.py -v
```

**Step 5: Commit**

```bash
git commit -m "feat: add v2 season API blueprint with new endpoints"
```

---

### Task 8: Background Scheduler

**Files:**
- Create: `src/season/scheduler.py`
- Modify: `src/api/__init__.py` (start scheduler on app creation)

Implement a background thread that calls `tick()` periodically:

```python
# src/season/scheduler.py
"""Background scheduler that drives the GW state machine."""
import threading
import time
import logging

logger = logging.getLogger(__name__)

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_scheduler(manager_id: int, db_path: str, interval_seconds: int = 300):
    """Start the background tick loop.

    Calls SeasonManagerV2.tick() every `interval_seconds` (default 5 min).
    """
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return  # Already running

    _stop_event.clear()

    def _loop():
        from src.season.manager_v2 import SeasonManagerV2
        from src.api.sse import broadcast
        mgr = SeasonManagerV2(db_path=db_path)
        while not _stop_event.is_set():
            try:
                alerts = mgr.tick(manager_id)
                for alert in alerts:
                    broadcast(alert, event="alert")
            except Exception:
                logger.exception("Scheduler tick failed")
            _stop_event.wait(interval_seconds)

    _scheduler_thread = threading.Thread(target=_loop, daemon=True, name="gw-scheduler")
    _scheduler_thread.start()


def stop_scheduler():
    """Stop the background tick loop."""
    _stop_event.set()
```

**Design note:** The scheduler is simple — a daemon thread with a sleep loop. No external dependencies (no APScheduler). The `tick()` method is idempotent, so calling it too often is harmless. The interval can be configured but 5 minutes is a reasonable default (FPL API cache is 30 min anyway).

**Step 1: Implement scheduler**

**Step 2: Wire into `create_app()`** — start scheduler if a manager_id is configured (e.g. from environment variable or config)

**Step 3: Test manually** — start the server, verify scheduler thread starts, verify tick() is called

**Step 4: Commit**

```bash
git commit -m "feat: add background scheduler for automatic GW state machine ticking"
```

---

## Phase 4: Remove Strategy Layer

### Task 9: Remove ChipEvaluator, CaptainPlanner, PlanSynthesizer

**Files:**
- Delete: `src/strategy/chip_evaluator.py`
- Delete: `src/strategy/captain_planner.py`
- Delete: `src/strategy/plan_synthesizer.py`
- Modify: `src/strategy/__init__.py` (remove imports)
- Modify: `tests/test_strategy_pipeline.py` (remove tests for deleted modules)
- Modify: `tests/test_integration.py` (remove import checks for deleted modules)

**Step 1: Remove test classes for deleted modules**

In `tests/test_strategy_pipeline.py`, remove:
- `TestChipEvaluator` (5 tests)
- `TestCaptainPlanner` (5 tests)
- `TestPlanSynthesizer` (4 tests)
- `TestFullPipeline` (3 tests — these test the full ChipEval→Planner→Captain→Synthesizer chain)

Keep:
- `TestMultiWeekPlanner` (6 tests — planner is being kept)
- `TestAvailabilityAdjustments` (3 tests — availability is being kept)
- `TestPlanInvalidation` — simplify to just test injury detection (remove plan-specific tests)

**Step 2: Delete the strategy files**

```bash
git rm src/strategy/chip_evaluator.py
git rm src/strategy/captain_planner.py
git rm src/strategy/plan_synthesizer.py
```

**Step 3: Update `src/strategy/__init__.py`**

Remove imports of deleted modules. Keep `transfer_planner`, `reactive`, `price_tracker`.

**Step 4: Update `tests/test_integration.py`**

In `test_all_modules_import`, remove:
```python
import src.strategy.chip_evaluator
import src.strategy.captain_planner
import src.strategy.plan_synthesizer
```

**Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: All remaining tests pass. Test count will decrease (17 tests removed).

**Step 6: Commit**

```bash
git commit -m "refactor: remove ChipEvaluator, CaptainPlanner, PlanSynthesizer"
```

---

### Task 10: Simplify reactive.py

**Files:**
- Modify: `src/strategy/reactive.py`
- Modify: `tests/test_strategy_pipeline.py`

Keep:
- `apply_availability_adjustments()` — still needed for zeroing injured players
- Simple injury detection (is player injured/doubtful?)

Remove:
- `detect_plan_invalidation()` — no "plan" to invalidate anymore
- `check_plan_health()` — replaced by simple bootstrap injury check
- All the plan-specific logic (BB without DGW, FH without BGW, captain drop detection)

Replace with a simpler function:

```python
def check_squad_injuries(bootstrap: dict, squad_ids: set[int]) -> list[dict]:
    """Check if any squad players are injured/doubtful.

    Returns list of {player_id, web_name, status, chance_of_playing}.
    """
    elements = {el["id"]: el for el in bootstrap.get("elements", [])}
    issues = []
    for pid in squad_ids:
        el = elements.get(pid)
        if not el:
            continue
        chance = el.get("chance_of_playing_next_round")
        status = el.get("status", "a")
        if status in ("i", "s", "n") or (chance is not None and chance < 75):
            issues.append({
                "player_id": pid,
                "web_name": el.get("web_name", "Unknown"),
                "status": status,
                "chance_of_playing": chance,
            })
    return issues
```

**Step 1: Update tests** — simplify `TestPlanInvalidation` to test `check_squad_injuries`

**Step 2: Simplify `reactive.py`**

**Step 3: Run tests**

**Step 4: Commit**

```bash
git commit -m "refactor: simplify reactive.py to injury checking only"
```

---

### Task 11: Remove Strategic Plan DB Tables

**Files:**
- Modify: `src/db/schema.py` — remove `strategic_plan` and `plan_changelog` CREATE TABLE
- Modify: `src/db/repositories.py` — remove `PlanRepository` class
- Modify: `src/db/migrations.py` — add migration 3 to drop tables
- Modify: `src/db/repositories.py` — remove `strategic_plan` and `plan_changelog` from `clear_generated_data()`
- Modify: `tests/test_integration.py` — update `test_db_schema_creation`

**Step 1: Add migration 3**

```python
def _migration_003_drop_strategy_tables(db_path: str):
    """Remove strategic_plan and plan_changelog tables."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS strategic_plan")
        conn.execute("DROP TABLE IF EXISTS plan_changelog")
        conn.commit()
    finally:
        conn.close()
```

**Step 2: Remove `PlanRepository` from `repositories.py`**

**Step 3: Update `clear_generated_data()` in `SeasonRepository`** — remove references to `strategic_plan` and `plan_changelog`

**Step 4: Update `test_db_schema_creation`** — remove assertion for `strategic_plan` and `plan_changelog` tables, add assertion for `planned_squad`

**Step 5: Run tests**

**Step 6: Commit**

```bash
git commit -m "refactor: remove strategic_plan and plan_changelog DB tables"
```

---

### Task 12: Remove Old Strategy API Endpoints

**Files:**
- Modify: `src/api/strategy_bp.py` — remove endpoints that no longer have backing logic
- Modify: `tests/test_integration.py` — update route existence test

Remove these endpoints:
- `GET/POST /api/season/strategic-plan` — no strategic plans anymore
- `GET /api/season/plan-health` — replaced by injury check in tick()
- `GET /api/season/plan-changelog` — no changelog anymore

Keep (but will migrate to v2 later):
- `GET /api/season/action-plan` — still useful, will be reimplemented in v2
- `GET /api/season/outcomes` — still useful
- `POST /api/preseason/generate` — may be useful for pre-GW1

**Step 1: Remove endpoints**

**Step 2: Update route test**

**Step 3: Run tests**

**Step 4: Commit**

```bash
git commit -m "refactor: remove strategic-plan, plan-health, plan-changelog endpoints"
```

---

## Phase 5: Wire Everything Together

### Task 13: Connect V2 Manager to Existing ML Pipeline

**Files:**
- Modify: `src/season/manager_v2.py`

The `_tick_planning()` method needs to call the full ML pipeline. This is the core logic ported from the old `generate_recommendation()` but simplified:

1. `load_all_data()` → `build_features()` → `generate_predictions()` (unchanged)
2. Save predictions CSV
3. Fetch manager's current squad from FPL API (with FH reversion)
4. Calculate budget from `entry_history["value"]`
5. Enrich predictions with bootstrap data (cost, team_code)
6. Generate multi-GW predictions for 5-GW lookahead
7. Apply availability adjustments (simplified reactive.py)
8. Call `MultiWeekPlanner.plan_transfers()` — with chip_plan=None (no chip strategy)
9. Call `solve_transfer_milp_with_hits()` with captain_col for joint captain optimization
10. Build recommendation: transfers, captain, bench order, predicted points
11. Save recommendation to DB
12. Save planned squad to DB (source="recommended")
13. Transition to READY phase

**Key simplification vs old `generate_recommendation()`:**
- No ChipEvaluator call
- No CaptainPlanner call
- No PlanSynthesizer call
- No `_save_fallback_strategic_plan()`
- No `_log_plan_changes()`
- Still calls `_analyze_bank_vs_use()` (this logic stays — it's useful)

**Step 1: Implement `_tick_planning()`**

**Step 2: Test with actual data** (manual integration test — start server, trigger tick)

**Step 3: Commit**

```bash
git commit -m "feat: connect V2 manager to ML pipeline in _tick_planning()"
```

---

### Task 14: Auto-Record Results in _tick_complete()

**Files:**
- Modify: `src/season/manager_v2.py`

When the state machine enters COMPLETE:
1. Fetch actual picks from FPL API for the just-completed GW
2. Compare to the planned squad (what did the user actually do?)
3. Detect any chips played via FPL API (`active_chip` field)
4. Record outcome: actual points, predicted points, delta
5. Save gw_snapshot
6. Advance to PLANNING for next GW (or SEASON_OVER if GW38)

Port the core logic from `src/season/recorder.py` but auto-trigger it.

**Step 1: Implement `_tick_complete()`**

**Step 2: Implement `_tick_live()`** — lightweight: check if all fixtures finished, transition if so

**Step 3: Test**

**Step 4: Commit**

```bash
git commit -m "feat: auto-record results in _tick_complete()"
```

---

## Phase 6: UI

### Task 15: New Season UI — READY Phase View

**Files:**
- Modify: `src/templates/index.html`

This is the largest UI change. The Season tab gets a complete rework:

**Replace the old sub-tabs** (Overview, Workflow, Fixtures, Transfer History, Chips, Prices, Strategy) with a single GW-centric view:

**Layout:**

```
┌──────────────────────────────────────────────────┐
│  GW 26 — READY              Deadline: Sat 11:30  │
├──────────────────────────────────────────────────┤
│                                                   │
│  RECOMMENDED TRANSFERS                            │
│  ┌─────────────────────────────────────────────┐ │
│  │ OUT: Player A (4.5m) → IN: Player B (5.2m)  │ │
│  │ Free Transfer • +1.3 predicted pts           │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  [Accept Recommendations]  [Make My Own]          │
│                                                   │
│  CAPTAIN: Salah (7.2 pts pred)                    │
│  BENCH: GKP, Player C, Player D, Player E        │
│                                                   │
│  CHIP: [BB] [TC] [FH] [WC]    (click to lock)   │
│                                                   │
│  PREDICTED SQUAD POINTS: 58.3                     │
│                                                   │
│  ── FIXTURE LOOKAHEAD (5 GW) ──                   │
│  [Fixture difficulty grid]                        │
│                                                   │
│  ── HISTORY ──                                    │
│  GW25: Predicted 54.2 | Actual 61 | +6.8         │
│  GW24: Predicted 48.7 | Actual 45 | -3.7         │
│                                                   │
└──────────────────────────────────────────────────┘
```

**Transfer Override Modal:**

When "Make My Own" is clicked:
- Show current squad as a list
- Click a player → they're highlighted as "out"
- Show available replacements (filtered by position, budget)
- Click a replacement → transfer is made
- Show updated predicted points
- "Add Another Transfer" button (with hit cost warning)
- "Reset to Recommendations" button

**Implementation approach:**
- New JavaScript functions: `renderReadyView()`, `renderTransferOverride()`, `renderChipSelector()`
- Call `/api/v2/season/status` on tab load to get phase + full state
- User actions call the appropriate v2 endpoints
- SSE alerts update the view automatically

**Step 1: Build the READY phase view HTML/CSS/JS**

**Step 2: Build the transfer override modal**

**Step 3: Build the chip lock-in UI**

**Step 4: Build the history section**

**Step 5: Wire up all v2 API calls**

**Step 6: Test manually in browser**

**Step 7: Commit**

```bash
git commit -m "feat: new GW-centric Season UI with READY phase view"
```

---

### Task 16: Phase-Aware UI

**Files:**
- Modify: `src/templates/index.html`

The UI should show different content based on the current phase:

- **PLANNING**: "Generating recommendations..." with progress spinner (via SSE)
- **READY**: Full recommendation view with accept/override controls (Task 15)
- **LIVE**: "GW in progress. Deadline has passed." + live score tracking placeholder
- **COMPLETE**: "GW complete. Recording results..." → auto-transitions
- **SEASON_OVER**: Season summary

Poll `/api/v2/season/status` periodically (or use SSE alerts) to detect phase changes and re-render.

**Step 1: Implement phase-switching logic**

**Step 2: Build each phase's view**

**Step 3: Test each phase manually**

**Step 4: Commit**

```bash
git commit -m "feat: phase-aware UI rendering for all GW states"
```

---

## Phase 7: Cleanup and Testing

### Task 17: Remove Old Manager References

**Files:**
- Modify: `src/api/season_bp.py` — switch from `manager.py` to `manager_v2.py`
- Modify: `src/api/strategy_bp.py` — switch remaining endpoints
- Modify: `src/api/__init__.py` — remove old blueprint registration if fully migrated

Once the v2 manager and UI are working, switch the old API endpoints to use the new manager. This is the point of no return — old behavior is replaced.

**Approach:** Rather than changing all endpoints at once, keep the v2 prefix active and redirect old endpoints one by one. Once all old endpoints are migrated or removed, clean up.

**Step 1: Migrate remaining useful endpoints** (dashboard, snapshots, prices, etc.) to use V2 manager

**Step 2: Remove old `/api/season/` routes that are fully replaced**

**Step 3: Remove v2 prefix** — rename `/api/v2/season/` to `/api/season/`

**Step 4: Run all tests**

**Step 5: Commit**

```bash
git commit -m "refactor: switch all API endpoints to SeasonManagerV2"
```

---

### Task 18: Final Test Suite Update

**Files:**
- Modify: `tests/test_integration.py`
- Modify: `tests/test_state_machine.py`
- Verify: `tests/test_correctness.py` (should be unchanged)

**Step 1: Update integration tests** for new route structure

**Step 2: Add integration tests** for v2 endpoints (status, accept, transfer, chip)

**Step 3: Verify correctness tests** still pass (ML pipeline unchanged)

**Step 4: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -v
```
Expected: All pass

**Step 5: Commit**

```bash
git commit -m "test: update test suite for season manager v2"
```

---

### Task 19: Clean Up Old Code

**Files:**
- Delete: `src/season/manager.py` (old manager)
- Delete: `src/season/preseason.py` (pre-season plan generation — replaced by tick_planning at GW1)
- Modify: `src/season/recorder.py` — may be absorbed into manager_v2's `_tick_complete()`
- Rename: `src/season/manager_v2.py` → `src/season/manager.py`

**Step 1: Delete old files**

**Step 2: Rename manager_v2 to manager**

**Step 3: Update all imports**

**Step 4: Run full test suite**

**Step 5: Final commit**

```bash
git commit -m "refactor: remove old SeasonManager, rename v2 to final"
```

---

## Summary

| Phase | Tasks | What it achieves |
|-------|-------|-----------------|
| 1: Foundation | Tasks 1-3 | Branch, state machine, DB schema |
| 2: Core | Tasks 4-6 | New SeasonManager with tick(), user actions, init |
| 3: API | Tasks 7-8 | V2 endpoints + background scheduler |
| 4: Remove | Tasks 9-12 | Kill strategy layer, simplify DB |
| 5: Wire | Tasks 13-14 | Connect ML pipeline, auto-record |
| 6: UI | Tasks 15-16 | New GW-centric UI |
| 7: Cleanup | Tasks 17-19 | Migrate endpoints, final cleanup |

**Estimated test count change:** Current 103 → ~90 (remove 17 strategy tests, add ~4 state machine/integration tests). May increase as more v2-specific tests are added.

**Risk mitigation:** All work on `season-manager-v2` branch. Old code on `main` is untouched. Can switch back at any point by checking out `main`.
