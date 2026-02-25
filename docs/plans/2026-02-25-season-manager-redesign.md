# Season Manager Redesign

**Date**: 2026-02-25
**Status**: Design — awaiting approval

---

## Problem

The current `SeasonManager` is a 1,721-line god class with 27 methods that mixes orchestration, data loading, strategy execution, and price tracking. It has no concept of where it is in the gameweek cycle. Every action is user-triggered — you have to remember what to click and when.

The strategy layer (ChipEvaluator, CaptainPlanner, PlanSynthesizer, plan health) builds a multi-week timeline the user doesn't want. The user wants the app to be their FPL brain: recommend everything for the next GW, let them accept or override, and auto-record results.

## Design Goals

1. **Autopilot**: The app auto-detects GW completion, auto-pulls actual team, auto-generates recommendations, auto-records results.
2. **The app is the brain**: It recommends transfers (including hits/banking), captain, bench order — informed by 5-GW lookahead. User's goal is to accept recommendations and execute on the FPL website.
3. **Override when needed**: User can make their own transfers instead, with a clean pick-player-out / pick-replacement-in flow. Changes are reversible until deadline.
4. **Manual chip control**: User decides when to play chips. App remembers and predicts accordingly. No season-long chip strategy.
5. **FPL API is ground truth**: After deadline, the app pulls actual picks from the API. If the user played a chip but forgot to lock it in the app, the API catch corrects the record.

## GW Lifecycle — State Machine

```
PLANNING ──→ READY ──→ LIVE ──→ COMPLETE ──→ PLANNING (next GW)
```

### PLANNING (automatic, no user action)

Triggered when previous GW completes (or on first run / season init).

1. Detect GW finished (all fixtures show `finished` in bootstrap)
2. Fetch actual team + scores from FPL API for the completed GW
3. Detect any chips played (compare locked-in state vs API data)
4. Record results: actual points, delta from prediction, transfers made
5. Refresh data (bootstrap, fixtures, match stats)
6. Generate predictions (1-GW ensemble + multi-GW)
7. Run MILP solver with 5-GW lookahead for transfer recommendations
   - Tracks free transfers (FT banking, max 5)
   - Evaluates 0..N transfers including hit cost analysis
   - Picks captain + vice captain
   - Orders bench optimally
8. Save recommendation
9. Emit alert: "GW{N} recommendation ready"
10. Transition to READY

### READY (user interacts here)

The only phase requiring user action. Deadline countdown active.

**What the user sees:**
- Recommended transfers (or "bank your FT") with predicted point impact
- Recommended captain + vice captain
- Bench order
- Fixture lookahead (5-GW horizon with difficulty ratings)
- Predicted squad points for next GW

**What the user can do:**
- **Accept recommendations** (one click) — planned squad set to app's recommendation
- **Override transfers** — pick a player out from current squad, choose replacement from pool (sorted by predicted pts, filtered by budget/position). Can make multiple transfers. Hit cost shown.
- **Change mind freely** — undo overrides, try different transfers, revert to app recommendation. Nothing is locked until the user executes on the FPL website.
- **Lock in a chip** — BB, TC, FH, or WC. App adjusts predictions (BB = all 15 score, TC = captain x3, FH = unconstrained squad rebuild, WC = unlimited free transfers). Used chips greyed out.
- **Pick captain** — if they disagree with the app's choice

**Background monitoring during READY:**
- Periodic bootstrap refresh (every 30 min)
- If material change detected (key player injured, price change):
  - Re-run recommendations
  - Emit alert: "Recommendation updated — {reason}"

### LIVE (automatic, no user action)

Triggered when GW deadline passes (detected from bootstrap `deadline_time`).

1. Auto-fetch actual picks from FPL API to see what user did
2. Log what was executed vs what was recommended
3. Track live scores (future enhancement)
4. When all fixtures complete → transition to COMPLETE

### COMPLETE (automatic, no user action)

Triggered when all GW fixtures show `finished`.

1. Auto-compare predicted vs actual points
2. Save outcome record
3. If GW < 38: auto-advance to PLANNING for next GW
4. If GW == 38: season over

## Transfer Recommendation Engine

The transfer recommender keeps its current intelligence but loses the strategy-plan wrapper:

**Keeps:**
- FT tracking and banking logic (1 FT per GW, max 5 banked, -4 per hit)
- `solve_transfer_milp_with_hits()` — evaluates 0..N transfers, picks best option
- 5-GW lookahead via `MultiWeekPlanner` — influences which players to target
- Multi-GW predictions with confidence decay
- Bank-vs-use analysis (save FT vs spend now)
- Price pressure awareness (buy before rise, sell before drop)
- Captain optimization in the MILP solver

**Kills:**
- `ChipEvaluator` — no app-driven chip strategy or heatmap
- `CaptainPlanner` — no multi-GW captain calendar
- `PlanSynthesizer` — no unified timeline, rationale, or changelog
- Plan health / plan invalidation system — replaced by simple injury re-check during READY
- Strategic plan DB table and changelog

## Chip Handling

Simple manual system:

- Before deadline: user can lock in a chip via the UI
- Effect on predictions:
  - **Bench Boost**: All 15 players contribute to predicted score
  - **Triple Captain**: Captain prediction x3
  - **Free Hit**: MILP solver builds unconstrained optimal squad (reverts after)
  - **Wildcard**: Unlimited transfers with no hit cost
- Used chips greyed out, available chips clickable
- **Safety net**: Post-GW API pull is ground truth. If user played a chip on the FPL website but forgot to lock it in the app, the app detects it from the API response (`active_chip` field) and corrects the record.
- Half-season chip reset (GW1-19 and GW20-38) tracked automatically.

## New SeasonManager API

The class shrinks from 27 methods to ~10:

```python
class SeasonManager:
    # --- Lifecycle ---
    def tick(self) -> list[Alert]:
        """Background scheduler entry point.
        Checks current phase, advances if conditions met.
        Returns alerts for SSE broadcast."""

    def get_status(self) -> GWStatus:
        """Current phase, GW number, deadline, squad, predictions,
        recommendation, chip availability."""

    def init_season(self, manager_id) -> None:
        """First-time setup. Backfill history from FPL API."""

    # --- User actions (READY phase only) ---
    def accept_transfers(self) -> PlannedSquad:
        """Accept app's recommended transfers."""

    def make_transfer(self, player_out_id, player_in_id) -> PlannedSquad:
        """Manual transfer. Returns updated squad + predictions."""

    def undo_transfers(self) -> PlannedSquad:
        """Reset to app's recommendation."""

    def lock_chip(self, chip: str) -> PlannedSquad:
        """Lock in a chip for upcoming GW. Predictions update."""

    def unlock_chip(self) -> PlannedSquad:
        """Remove locked chip."""

    def set_captain(self, player_id) -> PlannedSquad:
        """Override captain pick."""

    # --- Read-only ---
    def get_fixture_lookahead(self) -> FixtureLookahead:
        """5-GW fixture difficulty grid for context."""

    def get_history(self) -> list[GWResult]:
        """Past GW results with predicted vs actual."""
```

## What Gets Removed

| Component | Action | Reason |
|---|---|---|
| `src/strategy/chip_evaluator.py` | Remove | No app-driven chip strategy |
| `src/strategy/captain_planner.py` | Remove | No multi-GW captain calendar |
| `src/strategy/plan_synthesizer.py` | Remove | No unified plan timeline |
| `src/strategy/reactive.py` | Simplify | Just injury re-check, no plan invalidation |
| `strategic_plan` DB table | Remove | No strategic plans to store |
| `plan_changelog` DB table | Remove | No plan diffs to track |
| Strategy tab in UI | Remove | Replaced by READY phase view |
| Plan health endpoint | Remove | Replaced by injury monitoring in READY |

## What Gets Kept

| Component | Notes |
|---|---|
| ML pipeline (predictions) | Core of the app, unchanged |
| MILP solver | Powers transfer + captain recs |
| Multi-GW predictions | Feeds 5-GW lookahead for transfer quality |
| `MultiWeekPlanner` | Internal engine for transfer recs (not user-facing plan) |
| FPL API data fetching | Unchanged |
| Fixture calendar | Stays as user-visible lookahead |
| Price tracker | Useful context, simplified |
| `src/strategy/transfer_planner.py` | Core logic stays, plan output format changes |

## What Gets Added

| Component | Purpose |
|---|---|
| `src/season/state_machine.py` | GW phase enum + transition logic |
| `src/season/scheduler.py` | Background tick loop (APScheduler or thread timer) |
| Planned squad state | In-memory + DB working squad that user edits during READY |
| Transfer override UI | Pick player out → pick replacement flow |
| Chip lock-in UI | Simple chip selector with prediction update |

## UI Changes (High Level)

The main experience shifts from tab-based exploration to a GW-centric workflow:

- **Primary view**: Current GW status — what phase, what to do
- **READY view**: Recommended squad with accept/override controls, chip lock-in, fixture lookahead, predicted points
- **History view**: Past GWs with actual vs predicted
- **Predictions tab**: Keep as-is (player-level prediction table)
- **Monsters tab**: Keep as-is
- **Kill**: Season tab (sub-tabs: Overview, Workflow, Strategy, Chips planner), GW Compare (folded into history)

## Implementation Notes

- All work on a **new branch** (`season-manager-v2`) so we can discard if the old app is better
- Incremental approach: refactor backend first, then UI
- Keep all tests passing at each step — modify tests as the API surface changes
- The existing `test_strategy_pipeline.py` tests for ChipEvaluator, CaptainPlanner, PlanSynthesizer will be removed alongside those modules
