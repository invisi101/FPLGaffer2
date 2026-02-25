# FPL Gaffer 2 — Claude Code Notes

## Project Goal

Build a fully autonomous FPL manager. This is NOT just a prediction tool — it should think and plan like a real FPL manager across the entire season:

- **Transfer planning**: Rolling 5-GW horizon with FT banking, price awareness, and fixture swings
- **Squad building**: Shape the squad toward upcoming fixture runs, not just next GW
- **Captain planning**: Joint captain optimization in the MILP solver
- **Chip decisions**: User-controlled — the app surfaces predictions but the manager decides when to play chips
- **Price awareness**: Ownership-based price change predictions with probability scores
- **Reactive adjustments**: Auto-detect injuries in planned squad — SSE-driven alerts during READY phase
- **Outcome tracking**: Auto-record what was recommended vs what happened after each GW, track model accuracy over time
- **State machine autopilot**: Background scheduler ticks through PLANNING → READY → LIVE → COMPLETE each GW

The app runs on a GW state machine. Each gameweek cycles through phases automatically: generate recommendation (PLANNING), let the user review/override transfers and captain (READY), monitor live fixtures (LIVE), record results (COMPLETE). The user interacts primarily during the READY phase.

## Repos

- **This project**: FPLGaffer2 (v2 rebuild with modular architecture)
- **Original app**: `/Users/neil/fplmanager` — the monolithic v1 this was rebuilt from. Use as reference when in doubt.

## Honesty Rules

Never dismiss, reframe, or minimize failing tests. A failing test is a failing test — fix it or explain exactly why it fails. Do not say "all tests pass" when they don't. Do not use phrases like "pre-existing unrelated failure" to hand-wave away red output. That is lying.

- All tests must pass. 0 failures.
- If a test is wrong, fix the test and explain why.
- If the code is wrong, fix the code.
- Never ignore test output.
- Verify endpoints actually work end-to-end before claiming they're done.
- Check the frontend expects the same fields the backend returns.
- Compare against the original fplmanager app when in doubt — that's what this was built from.

## Audit Rules

When running a full audit:
- **List EVERY finding** — every single issue discovered, not just the priorities. Present the complete list to the user.
- **Do NOT auto-fix** — the user decides what to fix. Present findings, wait for instructions.
- **Explain in plain English** — assume the user is not a machine learning engineer. Every finding must include: what the issue is, why it matters in FPL terms, and what the fix would do. No jargon without explanation.
- **When instructed to fix**: explain each change before or while making it — what you're changing, why, and what effect it will have.

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `FLASK_APP=src.api .venv/bin/python -m flask run --port 9876` (serves on `http://127.0.0.1:9876`)
- **Port 9876**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9876 | xargs kill -9` before starting
- **Tests**: `.venv/bin/python -m pytest tests/ -v` (176 tests across 4 files)
- **No build step**: Frontend is a single file at `src/templates/index.html` (inline CSS + JS). Just edit and refresh.
- **Flush the cache**: When told to flush/clear the cache, delete ALL of the following for a complete clean slate:
  - `cache/*` — cached FPL API + GitHub CSV data
  - `output/*` — predictions.csv, predictions_detail.json, season.db
  - `models/*` — trained .joblib model files
  - All `__pycache__/` directories throughout the project

## My Manager ID

12904702

---

## Architecture Overview

Three-layer system: **Data -> Features/Models -> Strategy/Solver**, backed by SQLite and served via Flask with blueprints.

### Project Structure

```
src/
├── __init__.py
├── __main__.py              # Entry point
├── config.py                # ALL magic numbers — dataclass configs
├── paths.py                 # PyInstaller-aware path resolution
├── logging_config.py        # Centralized logging
│
├── api/                     # Flask blueprints (8 blueprints, 58 endpoints)
│   ├── __init__.py          # create_app() factory, blueprint registration, scheduler startup
│   ├── core.py              # Predictions, training, status, monsters, PL table
│   ├── team.py              # Best team, my team, transfer recommendations
│   ├── season_bp.py         # Season init, dashboard, recommendations, snapshots
│   ├── season_v2_bp.py      # V2 state-machine endpoints (tick, user actions, status)
│   ├── strategy_bp.py       # Action plan, outcomes, preseason (simplified)
│   ├── prices_bp.py         # Prices, predictions, history, watchlist
│   ├── backtest_bp.py       # Walk-forward backtesting
│   ├── compare_bp.py        # GW compare (actual vs hindsight-best)
│   ├── helpers.py           # Shared: safe_num, scrub_nan, load_bootstrap, get_next_gw, resolve_current_squad_event
│   ├── middleware.py         # CORS, error handlers
│   └── sse.py               # Server-Sent Events, background task runner
│
├── data/                    # Data fetching and caching
│   ├── cache.py             # TTL-based file cache
│   ├── fpl_api.py           # FPL API client (bootstrap, fixtures, manager)
│   ├── github_csv.py        # GitHub CSV data (FPL-Core-Insights)
│   ├── loader.py            # load_all_data() — unified data loading
│   └── season_detection.py  # Detect current season from bootstrap
│
├── features/                # Feature engineering (~100+ features)
│   ├── builder.py           # build_features() — orchestrates all feature modules
│   ├── registry.py          # Position-specific feature lists, sub-model features
│   ├── player_rolling.py    # Rolling stats (3/5 GW windows)
│   ├── team_stats.py        # Team-level rolling stats
│   ├── playerstats.py       # Per-90 stats, bootstrap features (form, COP, cost)
│   ├── fixture_context.py   # FDR, opponent elo, is_home, fixture count
│   ├── opponent_history.py  # Historical performance vs specific opponents
│   ├── interactions.py      # Cross-feature interactions (xG x opp_gc)
│   ├── rest_congestion.py   # Days rest, fixture congestion
│   ├── upside.py            # Volatility, form acceleration, big chances
│   ├── venue_form.py        # Home/away form splits
│   └── targets.py           # Target variable creation (shifted by 1 GW)
│
├── ml/                      # Machine learning
│   ├── training.py          # XGBoost training: mean, quantile, sub-models
│   ├── prediction.py        # 1-GW prediction pipeline with ensemble blend
│   ├── multi_gw.py          # 3-GW and 8-GW horizon predictions
│   ├── decomposed.py        # Decomposed sub-model prediction + FPL scoring
│   ├── model_store.py       # Model save/load (.joblib)
│   └── backtest.py          # Walk-forward backtesting framework
│
├── solver/                  # MILP optimization
│   ├── squad.py             # solve_milp_team() — optimal squad from scratch
│   ├── transfers.py         # solve_transfer_milp() + with_hits wrapper
│   ├── formation.py         # Formation enumeration and validation
│   └── validator.py         # Squad rule validation
│
├── schemas/                 # Data validation
│   ├── fpl_rules.py         # FPL squad rules (positions, team cap, budget)
│   ├── player.py            # Player data schemas
│   └── prediction.py        # Prediction output schemas
│
├── strategy/                # Transfer planning + availability
│   ├── transfer_planner.py  # Multi-week rolling transfer planner (5-GW horizon)
│   ├── reactive.py          # Injury checking + availability adjustments (simplified)
│   └── price_tracker.py     # Price alerts, ownership-based predictions
│
├── season/                  # Season orchestration (state-machine-driven)
│   ├── manager.py           # SeasonManager — state-machine GW lifecycle orchestrator
│   ├── state_machine.py     # GWPhase enum, transitions, phase detection
│   ├── scheduler.py         # Background daemon thread (tick every 5 min)
│   ├── dashboard.py         # Dashboard data aggregation
│   └── fixtures.py          # Fixture calendar builder
│
├── db/                      # Database layer
│   ├── connection.py        # SQLite connection helper
│   ├── schema.py            # Table definitions (CREATE TABLE)
│   ├── migrations.py        # Schema migrations (ALTER TABLE)
│   └── repositories.py      # 9 repository classes for CRUD
│
├── utils/                   # Shared utilities
│   ├── dataframe_helpers.py # DataFrame utilities
│   └── nan_handling.py      # NaN/Inf scrubbing
│
└── templates/
    └── index.html           # Entire frontend (single file, ~5500 lines)

models/     # Saved .joblib model files (gitignored)
output/     # predictions.csv, season.db (gitignored)
cache/      # Cached data: 6h GitHub CSVs, 30m FPL API (gitignored)
tests/      # 103 tests: correctness (57), integration (17), strategy pipeline (29)
```

---

## Data Pipeline

### Sources
1. **GitHub (olbauday/FPL-Core-Insights)**: `https://github.com/olbauday/FPL-Core-Insights` — Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Raw URL base: `https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data`. CSVs per season: `playerstats.csv`, `player_match_stats.csv`, `match_stats.csv`. Cached 6 hours.
2. **FPL API** (public, no auth): Current player data (prices, form, injuries, ownership), fixtures, manager picks/history/transfers. Cached 30 minutes.

### Data Loading (`src/data/loader.py`)
`load_all_data()` fetches from both sources and returns a unified dict with keys: `player_match_stats`, `player_stats`, `match_stats`, `api` (bootstrap, fixtures).

### Feature Engineering (`src/features/`)
100+ features per player per GW, built by `build_features()` which orchestrates 9 feature modules:
- **Player rolling** (3/5 GW windows): xG, xA, xGOT, shots, touches, dribbles, crosses, tackles, goals, assists
- **EWM features** (span=5): Exponentially weighted xG, xA, xGOT
- **Upside/volatility**: xG volatility, form acceleration, big chance frequency
- **Home/away form**: Venue-matched rolling stats
- **Opponent history**: Player's historical performance vs specific opponents
- **Team rolling**: Goals scored, xG, clean sheets, big chances
- **Opponent stats**: Defensive (xG conceded, shots conceded) AND attacking (goals scored, xG)
- **CBIT composite** (clearances + blocks + interceptions + tackles) for DefCon prediction
- **Rest/congestion**: Days rest, fixture congestion rate
- **Fixture context**: FDR, is_home, opponent_elo, multi-GW lookahead
- **ICT/BPS**: Influence, creativity, threat, bonus points
- **Market data**: Ownership, transfer momentum
- **Availability**: Chance of playing, availability rate
- **Interaction features**: xG x opponent goals conceded, CS opportunity

All features shifted by 1 GW to prevent leakage. DGW handling: multiple rows per fixture, targets divided by fixture count, predictions summed.

**Non-PL match filtering**: The GitHub CSV source includes ALL competitive matches (Champions League, EFL Cup, FA Cup, etc.). Since FPL points only come from Premier League matches, `build_features()` in `builder.py` filters these out at the start of every feature build. The log messages "Filtering out N non-PL matches" and "Filtering out N non-PL player match stat rows" appear every time because the raw CSVs are loaded from cache (which includes all competitions) and the filtering is applied in-memory. This is correct behavior — the log is informational, not an error.

---

## Configuration (`src/config.py`)

Every magic number is defined in frozen dataclass configs:

| Config | Key Values |
|--------|------------|
| `XGBConfig` | 150 trees, depth 5, lr 0.1, subsample 0.8, walk-forward 20 splits, early stopping 20 rounds |
| `EnsembleConfig` | 70/30 mean/decomposed blend (empirically optimised), captain 0.7 mean + 0.3 Q80 (empirically optimised) |
| `SolverConfig` | 0.25 bench weight, -4 hit cost, max budget 1000, 3 per team |
| `CacheConfig` | GitHub CSV 6h, FPL API 30m, manager API 1m |
| `PredictionConfig` | Confidence decay 0.95->0.77, pool size 200 |
| `DecomposedConfig` | Position-specific components, Poisson/logistic objectives, DGW-aware soft caps |
| `DataConfig` | GitHub CSV base URL, FPL API base URL, earliest season 2024-2025 |
| `StrategyConfig` | 5-GW planning horizon, max 2 hits/GW, 5 max banked FTs, late-season GW33+, late-season hit cost 3.0 |
| `FPLScoringRules` | Full FPL points per action by position (incl DefCon thresholds) |

Import as singletons: `from src.config import xgb, ensemble, solver_cfg, ...`

---

## Model Architecture

### Tier 1: Mean Regression (Primary)
- 4 models (one per position) for `next_gw_points`
- XGBoost `reg:pseudohubererror` (Huber loss), walk-forward CV (last 20 splits)
- Sample weighting: current season 1.0, previous 0.5
- Fixed hyperparameters: 150 trees, depth 5, lr 0.1, subsample 0.8
- Early stopping: 20 rounds, using last 20% of training fold as validation (15% for final model)
- 3-GW predictions derived at inference by summing three 1-GW predictions with per-GW opponent data

### Tier 2: Quantile Models (Captain Picks)
- MID + FWD only, 80th percentile of next_gw_points
- `captain_score = 0.7 x mean + 0.3 x Q80` — empirically optimised (0.7/0.3 beat 0.4/0.6 by +12 pts over 18 GWs in walk-forward backtest)

### Tier 3: Decomposed Sub-Models
- Position-specific component models predicting individual scoring elements:
  - **GKP**: cs, goals_conceded, saves, bonus
  - **DEF**: goals, assists, cs, goals_conceded, bonus, defcon
  - **MID**: goals, assists, cs, bonus, defcon
  - **FWD**: goals, assists, bonus, defcon
- Poisson objectives for count data (goals, assists, bonus, saves, goals_conceded, defcon), binary:logistic for CS
- DefCon: Poisson CDF predicts P(CBIT >= threshold) where threshold = 10 (GKP/DEF) or 12 (MID/FWD), scores +2 pts
- Combined via FPL scoring rules with playing probability weighting
- DGW-aware soft calibration caps per position (GKP=7, DEF=8, MID=10, FWD=10), scaled by fixture count

### Ensemble
- Production predictions use a **70/30 blend** of mean regression and decomposed sub-models (empirically optimised via L9 grid search — 0.30 beat 0.15 on MAE, Spearman, and top-11 points)
- Both models contribute meaningfully; the decomposed model adds component-level signal beyond what the mean model captures alone

### Multi-GW Predictions (`src/ml/multi_gw.py`)
- 3-GW: Sum of three 1-GW predictions with correct opponent data per offset
- 8-GW horizon: Model predictions via `predict_multi_gw()` for up to 8 future GWs
- Confidence decay is configurable via `PredictionConfig.confidence_decay` tuple in `config.py`: `(0.95, 0.93, 0.90, 0.87, 0.83, 0.80, 0.77)`. Falls back to `0.95^(offset-1)` beyond the tuple length.
- `_build_offset_snapshot()` builds a per-GW snapshot by swapping fixture/opponent columns for each future GW, recomputing interaction features (xg_x_opp_goals_conceded, cs_opportunity, venue_matched_form) and multi-GW lookahead features (avg_fdr_next3, home_pct_next3, avg_opponent_elo_next3)

### Prediction Intervals
- Walk-forward residuals stored with each model (q10, q90 per prediction bin)
- Binned intervals: higher-predicted players get wider intervals
- Used for prediction_low / prediction_high in API output

---

## MILP Solver (`src/solver/`)

### `squad.py: solve_milp_team()` — Optimal squad from scratch
Two-tier MILP with optional captain optimization:
- **Variables**: `x_i` (in squad), `s_i` (starter), `c_i` (captain, when `captain_col` provided)
- **Objective**: max(0.75 x starting XI pts + 0.25 x bench pts + captain bonus)
- **Constraints**: Budget, positions (2/5/5/3), max 3 per team, 11 starters, formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), exactly 1 captain who is a starter
- **Returns**: starters, bench, total_cost, starting_points, captain_id
- **Bench ordering**: GKP first, then outfield ordered by position priority (formation-constrained positions first) then by predicted points. This ensures auto-subs maintain valid formations.

### `transfers.py: solve_transfer_milp()` — Optimal transfers
Same as above plus: `sum(x_i * is_current_i) >= 15 - max_transfers` — keeps at least (15 - N) current players.
- Budget = bank + sum(now_cost of current squad)
- Also supports joint captain optimization

### `transfers.py: solve_transfer_milp_with_hits()` — Hit-aware wrapper
Evaluates 0..max_transfers, subtracts forced replacements from hit count, applies -4 per hit. Compares all options including 0-transfer baseline. Returns `baseline_points` (current XI predicted points with no changes) alongside the best option.

### `formation.py` — Formation enumeration
Enumerates all valid formations and validates starting XI constraints.

### `validator.py` — Squad rule validation
Validates squad against FPL rules (positions, team cap, budget, player count).

---

## Strategy Layer (`src/strategy/`)

The strategy layer was simplified in the v2 redesign. ChipEvaluator, CaptainPlanner, and PlanSynthesizer were removed — chip decisions are now user-controlled, and captain selection is handled directly by the MILP solver. Only the transfer planner, availability checking, and price tracking remain.

### MultiWeekPlanner (`transfer_planner.py`)
Rolling **5-GW** transfer planner:
- Tree search: generates all valid FT allocation sequences, simulates each, picks the path maximizing total points
- Considers: FT banking (save vs spend at every GW), fixture swings, price change probability
- Price bonus extends across 3 GWs with decay (1.0, 0.5, 0.25) instead of GW+1 only
- Late-season hit discount: Effective hit cost reduced from 4 to 3 in GW33+ (fewer future GWs to amortize)
- Supports up to 2 hits per GW for fixture swing strategies
- Reduces pool to top 200 players for efficiency
- Passes `captain_col` to MILP solver for captain-aware squad building

### Availability Checking (`reactive.py`)
Simplified from v1's plan-invalidation system to two focused functions:
- `apply_availability_adjustments()`: Zeros predictions for injured/suspended players across all future GWs; zeros doubtful players for GW+1 only
- `check_squad_injuries()`: Checks bootstrap availability for squad players (used during READY phase injury alerts)

---

## Season Manager (`src/season/manager.py`)

State-machine-driven season orchestrator. Central class: `SeasonManager` (~1050 lines).

### GW State Machine (`src/season/state_machine.py`)

Each gameweek cycles through 4 phases:

```
PLANNING → READY → LIVE → COMPLETE → PLANNING (next GW)
                                    → SEASON_OVER (GW38)
```

Phase detection uses real-world signals (not stored state):
1. `all_fixtures_finished` → COMPLETE
2. `deadline_passed` → LIVE
3. `has_recommendation` → READY
4. else → PLANNING

The stored phase is updated to match detected phase on every `get_status()` call. Invalid transitions are forced (detect_phase is authoritative).

### Background Scheduler (`src/season/scheduler.py`)

A daemon thread calls `tick()` every 5 minutes. Started automatically when `GAFFER_MANAGER_ID` env var is set. The scheduler creates its own `SeasonManager` instance and broadcasts SSE alerts for any events (injury warnings, GW completion, etc.).

### `tick()` — The Core Dispatch

Each call inspects the current phase and performs exactly one phase's work:

| Phase | Handler | What it does |
|-------|---------|-------------|
| PLANNING | `_tick_planning()` | Load data → build features → generate predictions → fetch squad → run MultiWeekPlanner (or fallback MILP solver) → save recommendation + planned squad → transition to READY |
| READY | `_tick_ready()` | Check planned squad players against bootstrap for injury/availability changes → return alerts |
| LIVE | `_tick_live()` | Check if all GW fixtures are finished → transition to COMPLETE |
| COMPLETE | `_tick_complete()` | Fetch actual picks + live points → build squad with captain doubling → save gw_snapshot → compare to recommendation → save outcome → transition to PLANNING (or SEASON_OVER) |

### `_tick_planning()` — The Critical Orchestration

This is the most complex method. Understanding its flow is essential for debugging:

1. **Load data + build features + generate predictions** (1-GW ensemble + 3-GW)
2. **Fetch current squad** from FPL API (picks, history, entry_history) with Free Hit reversion
3. **Calculate budget** using `entry_history["value"]` (real selling value with 50% profit rule), NOT `sum(now_cost)`
4. **Enrich predictions** with bootstrap data (cost, team_code, web_name)
5. **Generate 8-GW multi-GW predictions** for the planner
6. **Replace GW+1** predictions with exact numbers from the Predictions tab (consistency)
7. **Apply availability adjustments** (zero injured players across all future GWs)
8. **Run MultiWeekPlanner** → 5-GW transfer plan with FT banking → extract GW+1 step
9. **Fallback**: If planner fails, fall back to single-GW MILP solver (`solve_transfer_milp_with_hits`)
10. **Build enriched squad + transfer list** with full player data
11. **Save recommendation** to DB + **save planned squad** to `planned_squad` table
12. **Update fixtures + track prices**, transition to READY

### `_tick_complete()` — Auto-Recording Results

Fully automatic post-GW recording:
1. Fetch actual picks, live event data, transfers from FPL API
2. Build squad with live points (captain doubling applied via `multiplier`)
3. Save `gw_snapshot` with bank, team value, rank, captain, transfers
4. Compare to recommendation: followed_captain, followed_transfers, followed_chip
5. Save `recommendation_outcome` with point delta
6. Transition to PLANNING (or SEASON_OVER if GW38)

**Chip tracking logic**: Only counts as "followed" when a chip was explicitly recommended. If no chip was recommended and none was played, that's a match. If no chip was recommended but one was played, that's not followed.

### User Action Methods (READY phase only)

All user actions are gated by `_require_ready_phase()` which checks the DB phase column:

| Method | What it does |
|--------|-------------|
| `accept_transfers()` | Mark planned squad as accepted (saves with source="accepted") |
| `make_transfer(out_id, in_id)` | Swap players with validation (position, budget, team limit) → re-optimise XI → recalculate points |
| `undo_transfers()` | Reset planned squad to original recommendation |
| `set_captain(player_id)` | Set new captain (must be starter) → auto-pick VC → recalculate points |
| `lock_chip(chip)` | Activate chip (bboost/3xc/freehit/wildcard) → recalculate points with chip effect |
| `unlock_chip()` | Remove active chip → recalculate points |

### Other Key Methods
- `init_season()` — Backfills season history from FPL API; pre-season creates record and sets PLANNING phase
- `get_status()` — Phase detection + status dict (gw, deadline, planned_squad, etc.)
- `get_action_plan()` — Builds human-readable steps from latest recommendation
- `get_dashboard()` — Aggregated dashboard data (rank, points, budget, accuracy)
- `get_outcomes()` — All recorded outcomes for the season
- `_track_prices_simple()` — Snapshots prices for squad players **AND watchlist players**

---

## Database Schema (`src/db/`)

8 SQLite tables defined in `schema.py`, with versioned migrations in `migrations.py` (3 migrations, tracked via `schema_version` table):

| Table | Purpose |
|-------|---------|
| `season` | Manager seasons (id, manager_id, name, start_gw, current_gw, **phase**) |
| `gw_snapshot` | Per-GW state (squad_json, bank, team_value, points, rank, captain, transfers) |
| `recommendation` | Pre-GW advice (transfers_json, captain, chip, predicted/base/current_xi_points) |
| `recommendation_outcome` | Post-GW tracking (followed_transfers, actual_points, point_delta) |
| `price_tracker` | Player price snapshots (price, transfers_in/out, snapshot_date) |
| `fixture_calendar` | GW x team fixture grid (fixture_count, fdr_avg, is_dgw, is_bgw) |
| `watchlist` | User watchlist for price tracking |
| `planned_squad` | User-editable squad for next GW (squad_json, source, updated_at) |

**Removed tables**: `strategic_plan` and `plan_changelog` (dropped in migration 003 — v2 redesign).

9 repository classes in `repositories.py`: `SeasonRepository`, `SnapshotRepository`, `RecommendationRepository`, `OutcomeRepository`, `PriceRepository`, `FixtureRepository`, `PlannedSquadRepository`, `DashboardRepository`, `WatchlistRepository`.

---

## API Endpoints

### Core (`core.py`)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/predictions` | Player predictions with filters/sorting |
| POST | `/api/refresh-data` | Re-fetch data, rebuild predictions, check plan health |
| POST | `/api/train` | Train all model tiers + generate predictions |
| GET | `/api/status` | SSE stream for live progress |
| GET | `/api/model-info` | Trained model metadata |
| GET | `/api/monsters` | Top 3 players across 8 monster categories |
| GET | `/api/pl-table` | Premier League standings |
| GET | `/api/gw-scores` | GW scores for all teams |
| GET | `/api/team-form` | Team form data |
| GET | `/api/players/teams` | Player team assignments |
| GET | `/api/players/<id>/detail` | Detailed player view |

### Team (`team.py`)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/best-team` | MILP optimal squad |
| GET | `/api/my-team?manager_id=ID` | Import manager's FPL squad |
| POST | `/api/transfer-recommendations` | MILP transfer solver (with captain optimization) |

### Season Management (`season_bp.py`) — Legacy endpoints, now backed by new SeasonManager
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/season/init` | Backfill season history |
| GET | `/api/season/status` | Check if season exists |
| DELETE | `/api/season/delete` | Delete season data |
| GET | `/api/season/dashboard` | Full dashboard (rank, budget, accuracy) |
| POST | `/api/season/generate-recommendation` | Runs tick() (replaces old generate_recommendation) |
| POST | `/api/season/record-results` | Runs tick() (replaces old record_actual_results) |
| GET | `/api/season/recommendations` | All recommendations for season |
| GET | `/api/season/snapshots` | All GW snapshots |
| GET | `/api/season/fixtures` | Fixture calendar |
| GET | `/api/season/chips` | Chip status + values |
| GET | `/api/season/gw-detail` | Detail for specific GW |
| GET | `/api/season/transfer-history` | Transfer history |
| GET | `/api/season/bank-analysis` | Bank analysis |
| POST | `/api/season/update-fixtures` | Rebuild fixture calendar |

### Season V2 (`season_v2_bp.py`) — State-machine-driven endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v2/season/init` | Initialize season from FPL API |
| GET | `/api/v2/season/status` | Phase-aware status (gw, phase, deadline, planned_squad) |
| DELETE | `/api/v2/season/delete` | Delete season data |
| POST | `/api/v2/season/tick` | Manual trigger of state machine tick |
| POST | `/api/v2/season/accept-transfers` | Accept recommended transfers as-is |
| POST | `/api/v2/season/make-transfer` | Override: swap player_out for player_in |
| POST | `/api/v2/season/undo-transfers` | Reset squad to original recommendation |
| POST | `/api/v2/season/lock-chip` | Activate a chip (bboost/3xc/freehit/wildcard) |
| POST | `/api/v2/season/unlock-chip` | Remove active chip |
| POST | `/api/v2/season/set-captain` | Set captain (must be starter) |
| GET | `/api/v2/season/fixture-lookahead` | Stub (not yet implemented) |
| GET | `/api/v2/season/history` | Stub (not yet implemented) |
| GET | `/api/v2/season/available-players` | Stub (not yet implemented) |

### Strategy (`strategy_bp.py`) — Simplified
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/season/action-plan` | Clear action items for next GW |
| GET | `/api/season/outcomes` | All recorded outcomes |
| POST | `/api/preseason/generate` | Runs tick() for pre-season |

### Prices (`prices_bp.py`)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/season/prices` | Latest prices + alerts |
| POST | `/api/season/update-prices` | Refresh price data |
| GET | `/api/season/price-predictions` | Ownership-based price predictions |
| GET | `/api/season/price-history` | Price movement history |
| GET | `/api/season/watchlist` | Get watchlist |
| POST | `/api/season/watchlist/add` | Add to watchlist |
| POST | `/api/season/watchlist/remove` | Remove from watchlist |

### Other
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/backtest` | Walk-forward backtesting |
| GET | `/api/backtest-results` | Fetch backtest results |
| POST | `/api/gw-compare` | Compare manager's actual vs hindsight-best |

---

## UI Structure (`src/templates/index.html`)

Single-file frontend (~5500 lines) with dark theme, CSS variables, SSE progress, localStorage persistence.

### Main Tabs
1. **Predictions** — Sortable player table, position filters, search, 1-GW and 3-GW columns
2. **Best Team** — MILP squad selector with pitch visualization (starting_gw_points, starting_gw3_points, remaining budget)
3. **GW Compare** — Compare actual FPL team vs hindsight-best for any past GW
4. **My Team** — Import FPL squad, dual-pitch (actual pts / predicted), transfer recs, season overview
5. **Season** — Full season management dashboard
6. **Monsters** — Top 3 players across 8 categories with podium layout

### Season Tab (Phase-Aware GW View)
The Season tab is a single GW-centric view that adapts to the current phase:
- **GW Header**: Phase badge (PLANNING/READY/LIVE/COMPLETE/SEASON OVER), deadline countdown, predicted points
- **PLANNING phase**: "Generating recommendation..." with progress indicator
- **READY phase**: Transfer cards (recommended + overrides), captain picker, chip toggles, transfer override modal, "Accept" button. Auto-polls every 60 seconds for injury alerts.
- **LIVE phase**: Live points display (auto-polls every 60 seconds)
- **COMPLETE phase**: Results summary with actual vs predicted points
- **SEASON OVER**: Final season summary
- **History section**: Collapsible past GW results at the bottom

---

## Critical Data Flow Patterns

### Predictions CSV vs Bootstrap Enrichment
The predictions CSV (`output/predictions.csv`) has `position_clean` (not `position`) and no `cost` column. **Every consumer of this CSV must enrich from bootstrap:**

```python
# Pattern: enrich predictions with bootstrap data
elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
if "position_clean" in df.columns and "position" not in df.columns:
    df["position"] = df["position_clean"]
if "cost" not in df.columns:
    for idx, row in df.iterrows():
        el = elements_map.get(int(row["player_id"]))
        if el:
            df.at[idx, "cost"] = round(el.get("now_cost", 0) / 10, 1)
            df.at[idx, "team_code"] = id_to_code.get(el.get("team"))
```

This pattern is required in: `core.py` (predictions endpoint), `team.py` (best-team, transfer-recs), `season/manager.py` (_tick_planning).

### FPL API event_points and Captain Doubling
FPL API `event_points` in bootstrap elements is the RAW score (not multiplied by captain). Captain doubling must be applied in backend:
```python
raw_pts = el.get("event_points", 0)
multiplier = pick.get("multiplier", 1)  # 2 for captain, 3 for TC
event_points = raw_pts * multiplier
```

### Gameweek Detection
Use `is_next` flag from bootstrap events, not `is_current` + 1:
```python
for ev in bootstrap.get("events", []):
    if ev.get("is_next"):
        return ev["id"]
# Fallback: current + 1
for ev in bootstrap.get("events", []):
    if ev.get("is_current"):
        return ev["id"] + 1 if ev["id"] < 38 else None
```

---

## Critical Invariants (Must Be Preserved)

These patterns were identified through audit. Violating any of them causes real bugs.

### 1. Availability Zeroing Must Cover ALL Prediction Columns

In `ml/prediction.py`, injured/unavailable players are zeroed BEFORE the 3-GW merge. Since 3-GW predictions come from `multi_gw.py` (which doesn't know about availability), they must be re-zeroed AFTER the merge:

```python
# In generate_predictions():
# First zeroing: covers predicted_next_gw_points, captain_score, intervals
unavailable_ids: set[int] = set()
# ... populate from bootstrap ...
result.loc[mask, col] = 0.0  # For all pred cols

# 3-GW merge happens here (can introduce non-zero values for injured players)
result = result.merge(pred_3gw, on="player_id", how="left")

# Second zeroing: MUST happen after 3-GW merge
if unavailable_ids and "predicted_next_3gw_points" in result.columns:
    mask_3gw = result["player_id"].isin(unavailable_ids)
    result.loc[mask_3gw, "predicted_next_3gw_points"] = 0.0
```

**If you add ANY new prediction column that gets merged after the first zeroing, you MUST add a re-zeroing step.**

### 2. Budget = Selling Value, Not Sum of now_cost

FPL only gives you 50% of price rises when selling. The public API `now_cost` is the BUY price. Use `entry_history["value"]` from the picks endpoint for accurate budget:

```python
# In _tick_planning():
api_value = entry_history.get("value")
if api_value:
    total_budget = round(api_value / 10, 1)  # Correct: includes 50% rule
else:
    total_budget = round(bank + current_squad_cost, 1)  # Fallback: slightly optimistic
```

### 3. MultiWeekPlanner First, Single-GW Solver as Fallback

`_tick_planning()` runs the MultiWeekPlanner FIRST, then extracts GW+1 from the plan. The single-GW MILP solver is ONLY the fallback when the planner fails. Do NOT reverse this — the planner produces better recommendations because it considers future GWs.

### 4. Chip GW Solver Failures Must Have Fallbacks

In `transfer_planner.py`, `_simulate_chip_gw()` can fail if the MILP solver finds no feasible solution. This must NOT abort the entire planning path — it falls back to the current squad's predicted points:

```python
# In _simulate_chip_gw():
if result is None:
    # Fall back to current squad points, don't return None
    return current_squad_points
```

### 5. Confidence Decay Must Use Config

Multi-GW predictions in `ml/multi_gw.py` use `PredictionConfig.confidence_decay` from `config.py`. The tuple `(0.95, 0.93, 0.90, ...)` is index-accessed by `offset - 1`, with fallback to `0.95^(offset-1)` for offsets beyond the tuple. Do NOT hardcode decay values.

### 6. Validator Trusts Solver's Hit Count

In `solver/validator.py`, hit cost validation uses `result.get("hits", 0)` from the solver, NOT recomputed from `current_squad_ids - new_squad_ids`. Recomputing overcounts because forced replacements (unavailable players) are not hits.

### 7. pandas Copy-on-Write (CoW) Safety

When modifying a column created via boolean operations on a DataFrame slice, you MUST `.copy()` the intermediate result to avoid pandas CoW warnings/bugs:

```python
# In playerstats.py:
gw_mins = gw_mins.copy()  # Required before boolean assignment
gw_mins["played"] = gw_mins["minutes_played"] > 0
```

### 8. Price Tracking Must Include Watchlist

`_track_prices_simple()` in `season/manager.py` snapshots prices for squad players AND watchlist players. Both sets must be included for price alerts to work correctly.

### 9. Free Hit Squad Revert Must Use Pre-FH Picks

After playing Free Hit, the squad reverts to the pre-FH squad. Any code fetching the "current squad" for planning purposes must call `resolve_current_squad_event()` (in `api/helpers.py`) or `SeasonManager._resolve_current_squad_event()` to detect FH usage and fetch picks from the pre-FH gameweek instead. This affects:

- `api/team.py: api_my_team()` — fetches BOTH actual (FH GW) picks for display AND reverted picks for planning/optimization
- `api/team.py: api_transfer_recommendations()` — uses reverted picks as the base squad for transfer solving
- `season/manager.py: _tick_planning()` — uses reverted picks for the transfer planner
- `season/manager.py: _track_prices()` — uses reverted squad for price tracking

The My Team page uses a dual-fetch pattern: `actual_picks_data` (FH GW) for the "actual" pitch display, `planning_picks_data` (pre-FH GW) for the optimized squad and transfer recommendations.

---

## Testing

```bash
# Kill leftover server
lsof -ti:9876 | xargs kill -9

# Run all tests (103 total, ~10 min)
.venv/bin/python -m pytest tests/ -v

# Run fast tests only (~2 sec)
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py tests/test_state_machine.py -v

# Start server
FLASK_APP=src.api .venv/bin/python -m flask run --port 9876
```

### Test Structure (4 files, 176 tests)

**`test_correctness.py`** (57 tests, <1 sec) — Mathematical correctness and FPL compliance:
- Config sanity: decay curve, ensemble weights, captain weights, squad positions, FPL scoring rules, soft caps, DefCon thresholds, late-season config
- Decomposed scoring formula: P(plays) logic, appearance points, goal/CS/GC/saves/DefCon formulas, soft cap math, DGW summing
- Ensemble blend: weighted average, boundary properties, mean model dominance
- Captain score: formula values, Q80 weighting, NaN fallback, doubling mechanics
- Confidence decay: monotonicity, bounds, fallback consistency
- Solver FPL compliance: full rule validation on 30-player pool, captain constraints, transfer keeping, hit cost arithmetic, baseline existence
- Prediction properties: no negatives, availability zeroing, 3-GW re-zeroing after merge

**`test_integration.py`** (17 tests, ~1 sec) — Smoke tests:
- Flask app creation and route registration (including v2 routes)
- API endpoints (predictions, model-info, season/status, v2/season/status)
- MILP solver (squad, transfers, captain)
- FPL rules validation (squad count, team limits)
- Database schema creation and CRUD
- Feature registry, config loading, module imports

**`test_state_machine.py`** (89 tests, ~1 sec) — Season Manager v2 behaviour:
- GWPhase: enum values, string conversion, valid/invalid transitions, next_phase, detect_phase priority
- SeasonManager: get_status (active/inactive), tick dispatch (planning/ready/live/complete)
- User actions: accept_transfers, make_transfer (position/budget/team validation, hit tracking), undo_transfers, set_captain (starter check, VC auto-pick, points recalculation), lock/unlock chip
- Init season: method existence, price tracking
- Scheduler: start/stop lifecycle
- TickLive: transitions when finished, no transition when pending
- TickComplete: records results, season over at GW38, no recommendation handling, API failure resilience

**`test_strategy_pipeline.py`** (13 tests, ~5 min) — Strategy layer behaviour:
- MultiWeekPlanner: plan steps, horizon, squad IDs, zero FT, rationale, empty predictions
- Availability: injured zeroed all GWs, doubtful first GW only, healthy unchanged
- Squad injury checking: injured, doubtful, healthy, non-squad players

### Key test commands
```bash
# V2 status (phase-aware)
curl -s "http://127.0.0.1:9876/api/v2/season/status?manager_id=12904702"

# Manual tick (trigger state machine)
curl -s -X POST http://127.0.0.1:9876/api/v2/season/tick -H 'Content-Type: application/json' -d '{"manager_id":12904702}'

# Action plan
curl -s "http://127.0.0.1:9876/api/season/action-plan?manager_id=12904702"

# Dashboard
curl -s "http://127.0.0.1:9876/api/season/dashboard?manager_id=12904702"

# Run fast tests only (~2 sec)
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py tests/test_state_machine.py -v
```

---

## FPL Rules Reference

Key FPL rules that affect codebase logic:

- **Free Transfers (FTs)**: 1 FT per GW. Unused FTs roll over, max 5 banked. Extra transfers cost -4 pts each.
- **Wildcard**: Unlimited transfers for one GW. Squad changes permanent. **FTs preserved** at pre-chip count.
- **Free Hit**: Unlimited transfers for one GW. Squad **reverts** to pre-FH squad next GW. FTs preserved.
- **Bench Boost**: All 15 players score (not just starting 11). One GW only.
- **Triple Captain**: Captain scores 3x instead of 2x. One GW only.
- **Half-season reset**: All 4 chips available once per half (GW1-19 and GW20-38), 8 total per season.
- **Captain**: Doubles selected player's points. Vice-captain activates only if captain gets 0 minutes.
- **DGW/BGW**: Double GW = team plays twice (points summed). Blank GW = team doesn't play.
- **DefCon (2025-2026)**: +2 pts if CBIT (clearances + blocks + interceptions + tackles) >= threshold. Threshold: 10 for GKP/DEF, 12 for MID/FWD.
- **GKP goals**: Worth 10 pts (changed from 6 in 2025-2026).

---

## Known Limitations

- **Static team_code for transferred players**: Uses player's CURRENT team from bootstrap. Pre-transfer matches get wrong team assignment. Fix requires per-match data not in public API.
- **fixture_congestion includes predicted match's own rest**: Borderline design choice — fixture schedule is known in advance.
- **FT planner explores at most 2 hits per GW**: Taking 3+ hits is almost never profitable. Max 3 transfers per GW keeps search tractable.
- **3-GW prediction is a simple sum**: No adjustment for form regression or rotation risk.
- **Selling prices partially use `now_cost`**: Budget calculation uses `entry_history["value"]` (correct). However, the MILP solver still uses `now_cost` for individual player costs since the public API doesn't provide per-player selling prices (50% profit sharing on price rises). This means the solver may slightly overestimate available budget when selling players whose prices have risen.

---

## Audit Findings (23 Feb 2026)

Full audit completed: 103 tests passed, 0 failures. 0 bugs found. All 8 previously identified regression patterns confirmed NOT reintroduced. Below are all findings ranked by severity with precise instructions for fixing each one.

### Last Audit Regression Check

| Pattern | Status | Evidence |
|---------|--------|----------|
| Availability zeroing before 3-GW merge | PASS | `ml/prediction.py` (first pass + second pass after 3-GW merge) |
| Budget using now_cost sum | PASS | `season/manager.py:280` uses `entry_history["value"]` |
| MultiWeekPlanner first, MILP fallback | PASS | `season/manager.py:461` — planner runs first, fallback at :461 |
| Solver failure kills planning | PASS | `strategy/transfer_planner.py` falls back to current squad pts |
| Hit cost recomputed from sets | PASS | `solver/validator.py` trusts solver's hits |
| Watchlist excluded from prices | PASS | `season/manager.py:1256` includes watchlist in price tracking |
| pandas CoW without .copy() | PASS | `features/playerstats.py` has .copy() |
| FH squad not reverted for planning | PASS | `season/manager.py:251` calls `resolve_current_squad_event()` |

---

### HIGH PRIORITY — Should Fix First

#### ~~H1. Post-Hoc Prediction Calibration (~30-35 pts/season, Low effort)~~ FIXED

**Problem**: 40% overshoot at 5+ predicted points (pred=5.95, actual=4.24). No post-hoc calibration exists. The soft caps in `decomposed.py:184-195` only affect 15% of the ensemble (decomposed weight), so they barely move the final number. This cascades to captain picks, transfer recs, and chip timing — all are based on inflated predictions at the top end.

**Root causes**: (a) XGBoost Huber loss suppresses tail learning, (b) soft caps only on decomposed model, (c) no calibration correction on 1-GW predictions.

**Fix**: Add isotonic regression calibration using existing walk-forward residuals.

1. In `ml/training.py` after line 243: `all_pred_resid` is a list of `(predicted, residual)` tuples collected during walk-forward CV (lines 208-209). Convert to `(predicted, actual)` pairs and fit a calibrator:
```python
from sklearn.isotonic import IsotonicRegression
pr = np.array(all_pred_resid)
pred_vals, resid_vals = pr[:, 0], pr[:, 1]
actual_vals = pred_vals + resid_vals  # residual = actual - predicted
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(pred_vals, actual_vals)
```
Save `calibrator` in the model metadata dict (alongside `residual_bins`, `bin_edges`).

2. In `ml/model_store.py`: The calibrator is already a picklable sklearn object — it will serialize naturally with joblib alongside the XGBoost model. Just include it in the metadata dict: `"calibrator": calibrator`.

3. In `ml/prediction.py` at line 122: After `model.predict(X).clip(min=0)`, apply calibration:
```python
pos_df[pred_col] = model.predict(X).clip(min=0)
if calibrator is not None:
    pos_df[pred_col] = calibrator.predict(pos_df[pred_col].values).clip(min=0)
```
Load calibrator from metadata when loading each position model.

4. ~30 lines of code across 3 files. Monotonic — preserves player rankings, only corrects magnitude. Guard with `if len(all_pred_resid) >= 50` to prevent degenerate fits.

**Verify with**: Run benchmarks. `model_avg_mae` should decrease. Check that predictions at 5+ pts are no longer systematically inflated.

#### ~~H2. Quantile Model (Q80) Lacks Early Stopping (~15-20 pts/season, Low effort)~~ FIXED

**Problem**: The mean model uses `early_stopping_rounds=20` with a validation split (`training.py:202-203`), but the Q80 quantile model trains all 150 trees with no early stopping (`training.py:392-400`). Q80 directly drives captain selection via `captain_score = 0.7*mean + 0.3*Q80`. An overfitted Q80 produces unreliable captain scores — it may flag "explosive" weeks that are actually noise.

**Fix**: Copy the validation-split + early-stopping pattern from `train_model` (lines 274-286) into `train_quantile_model` (lines 388-400).

Current code at line 393-400 (NO early stopping):
```python
final_model = XGBRegressor(
    objective="reg:quantileerror", quantile_alpha=alpha,
    n_estimators=xgb.n_estimators, max_depth=xgb.max_depth, ...
)
final_model.fit(X_all, y_all, sample_weight=w_all)
```

Should become (matching mean model pattern at lines 274-286):
```python
val_size_final = max(1, int(len(X_all) * 0.15))
final_model = XGBRegressor(
    objective="reg:quantileerror", quantile_alpha=alpha,
    n_estimators=xgb.n_estimators, max_depth=xgb.max_depth, ...,
    early_stopping_rounds=xgb.early_stopping_rounds,
)
final_model.fit(
    X_all[:-val_size_final], y_all[:-val_size_final],
    sample_weight=w_all[:-val_size_final],
    eval_set=[(X_all[-val_size_final:], y_all[-val_size_final:])],
    verbose=False,
)
```

**Verify with**: Run benchmarks. `captain_hit_rate` should improve or stay same. Check that Q80 predictions are not inflated vs actuals.

#### ~~H3. Backtest Uses Different Objective Than Production (~0 pts but breaks benchmarking)~~ FIXED

**Problem**: Backtest trains with `reg:squarederror` but production uses `reg:pseudohubererror` (`ml/backtest.py`). Additionally, backtest ensemble uses inner merge (not outer like production at `prediction.py`). This means backtest results don't perfectly reflect production model behavior — benchmarking protocol comparisons are slightly misleading.

**Fix**:
1. In `ml/backtest.py` line 117: Change `objective="reg:squarederror"` to `objective="reg:pseudohubererror"`.
2. In `ml/backtest.py` line 870: Change `how="inner"` to `how="outer"` (matching production `prediction.py`).
3. Small change, 2 lines.

**Verify with**: Run backtest before and after. Results will shift slightly (this is expected — the new results are more representative of production).

---

### MEDIUM PRIORITY — Should Investigate

#### ~~M1. Wildcard Evaluated Over 3 GWs, Not Full Planning Horizon (~10-15 pts/season, Medium effort)~~ FIXED

**Problem**: `chip_evaluator.py:361` limits WC look-ahead to `gw <= g <= gw + 2` (3 GWs). A WC permanently restructures the squad — its value accrues over every remaining GW. This systematically undervalues WCs, leading to late/never usage and missed WC->BB synergies.

**Fix**: Extended WC look-ahead to 5 GWs with confidence decay applied consistently to both new and current squad evaluations.

#### ~~M2. Missing xA x opp_GC Interaction Feature (~15-20 pts/season, Low effort)~~ FIXED

**Problem**: `features/interactions.py:26-27` has `xg_x_opp_goals_conceded` but no equivalent for assists. Missing interactions: `xa_x_opp_goals_conceded`, `xg_overperformance`, `form_x_fixture`.

**Fix**: Added all three interaction features in `interactions.py`, registered in `DEFAULT_FEATURES` and `SUB_MODEL_FEATURES` in `config.py`.

#### ~~M3. Backtest Ensemble Uses Inner Merge (Not Outer) (~0 pts but breaks benchmarking)~~ FIXED

**Problem**: Production ensemble uses `how="outer"` merge; backtest uses inner merge, potentially dropping players. Backtest may undercount prediction coverage.

**Fix**: Changed `how="inner"` to `how="outer"` in `ml/backtest.py` to match production. Also fixed backtest objective from `reg:squarederror` to `reg:pseudohubererror`.

#### ~~M4. Transfer Planner Aborts Path on Missing Position/Cost Data (Medium effort)~~ FIXED

**Problem**: `transfer_planner.py` — if any predictions lack position/cost columns, `_simulate_transfer_gw` returns `None`, aborting the entire 5-GW planning path. Should fall back gracefully.

**Fix**: Replaced `return None` with fallback that uses current squad's predicted points.

#### ~~M5. Late-Season Hit Discount Uses len(transfers_in) Not Solver's Hits (Low effort)~~ FIXED

**Problem**: `transfer_planner.py` recomputes hits from raw set difference, which includes forced replacements. Should use `result.get("hits", 0)` from solver output.

**Fix**: Replaced `len(transfers_in)` with `result.get("hits", 0)`.

#### ~~M6. Bank Analysis Uses Same Predictions for Both Weeks (Low effort)~~ FIXED

**Problem**: "Save FT" vs "Use FT" comparison uses identical predictions for both weeks — doesn't account for fixture changes between GW+1 and GW+2.

**Fix**: Uses GW+2 predictions for the week 2 comparison when multi-GW predictions are available.

#### ~~M7. Bench Order Not Optimized for Auto-Sub Probability (~7-8 pts/season, Medium effort)~~ FIXED

**Problem**: `solver/squad.py` orders bench by predicted points, not `P(auto_sub_needed) * predicted_points`. Auto-subs trigger ~15 of 38 GWs.

**Fix**: Modified bench sort to use `expected_sub_value = P(needed) * predicted_points` using `availability_rate_last5`.

#### ~~M8. No End-of-Season Behavioral Change (GW35-38) (~5-10 pts/season, Medium effort)~~ FIXED

**Problem**: No differential captaincy, chip dumping urgency, or aggressive transfer strategy for final GWs when chasing rank.

**Fix**: Added `rank_chasing_gw` to `StrategyConfig`. When active: reduced hit cost, differential captain bias, forced chip usage past GW36.

#### ~~M9. No Differential/Ownership-Aware Captain Scoring (~varies, Medium effort)~~ FIXED

**Problem**: `strategy_mode` config existed but was unused. Ownership data available but not integrated into captain scoring. In mini-league play, low-ownership captain hauls gain massive rank.

**Fix**:
1. Added `differential_alpha: float = 0.3` to `EnsembleConfig` in `config.py`.
2. Added `ownership` to MID and FWD feature lists (was only in GKP/DEF).
3. In `prediction.py`, apply ownership-based boost when `strategy_mode == "mini_league"`: `captain_score *= 1 + 0.3 * (1 - ownership/100)`.

#### ~~M10. DefCon CBIT Poisson CDF Undervalues DefCon (~3-5 pts/season, Medium effort)~~ FIXED

**Problem**: `decomposed.py` used Poisson CDF for P(CBIT >= threshold). CBIT values (0-20+) are overdispersed (variance > mean), so Poisson underestimates P(CBIT >= threshold) for high-CBIT defenders like Saliba, Gabriel, Van Dijk.

**Fix**:
1. In `training.py`, during defcon sub-model training, build empirical P(CBIT >= threshold) lookup from training data, saved in model metadata as `defcon_cdf`.
2. In `decomposed.py`, use empirical CDF lookup instead of Poisson CDF, with Poisson fallback.

---

### LOW PRIORITY — Minor Code Concerns

#### ~~L1. days_rest, fixture_congestion, chance_of_playing Not in FEATURE_FILL_DEFAULTS~~ FIXED

**Problem**: These features use ad-hoc `.fillna()` instead of the central config. Not a bug, but inconsistent with the pattern.

**Fix**: Added to `FEATURE_FILL_DEFAULTS` in `config.py`: `"days_rest": 7.0`, `"fixture_congestion": 0.143`, `"chance_of_playing": 100.0`.

#### ~~L2. fdr fillna Duplicated~~ FIXED

**Problem**: FDR fillna applied in two places — redundant but not harmful.

**Fix**: Removed the builder-level fill; `FEATURE_FILL_DEFAULTS` handles it.

#### ~~L3. Dead Code in _simulate_chip_gw~~ FIXED

**Problem**: `transfer_planner.py` None-return checks can never be reached due to upstream logic.

**Fix**: Removed dead `if step is None: return None` checks.

#### ~~L4. Own-Team Rolling Windows Hardcoded~~ FIXED

**Problem**: `team_stats.py` uses hardcoded `[3, 5]` instead of a config constant. Works correctly but isn't configurable.

**Fix**: Now uses `OPPONENT_ROLLING_WINDOWS` from config (already imported).

#### ~~L5. EWM Feature Naming Misnomer~~ FIXED

**Problem**: Features named `_last3` suffix but EWM uses `span=5`. Misleading but no functional impact.

**Fix**: Renamed to `_ewm5` suffix in `player_rolling.py` and all references in `config.py` (DEFAULT_FEATURES, SUB_MODEL_FEATURES, FEATURE_LABELS). Requires model retraining.

#### ~~L6. Saves Sub-Model May Be Overdispersed for Poisson~~ FIXED

**Problem**: Save counts (3-5 per match) have higher variance than Poisson expects.

**Fix**: Changed saves objective from `count:poisson` to `reg:squarederror` in `DecomposedConfig`. Requires model retraining.

#### ~~L7. Bonus Sub-Model Uses Poisson for Bounded 0-3 Data~~ FIXED

**Problem**: Bonus points are bounded at 0-3 but modeled with Poisson (unbounded).

**Fix**: Changed bonus objective from `count:poisson` to `reg:squarederror` in `DecomposedConfig`. Requires model retraining.

#### ~~L8. GKP Feature-to-Sample Ratio (~1:43)~~ FIXED

**Problem**: GKP had 40 features for ~1,500 training rows (~1:38 ratio). Potential overfitting risk.

**Fix**: Pruned 8 low-importance features (gw_influence, season_progress, transfer_momentum, fixture_congestion, transfers_in_event, net_transfers). Down to 32 features (~1:47 ratio). Requires model retraining.

#### ~~L9. 85/15 Ensemble Blend Not Empirically Optimized~~ FIXED

**Problem**: `decomposed_weight=0.15` set by judgement, not grid search. May not be optimal.

**Fix**: Built `scripts/optimize_blend.py` and ran a walk-forward grid search over GW10-27 (18 gameweeks, 500+ players per GW). The backtest framework (`src/ml/backtest.py`) now supports `save_raw_preds=True` to save raw per-component predictions before blending — this means models train once (~80 seconds) and re-blending at different weights is instant.

**Grid search results** (7 candidate weights, all players, all GWs):

| Weight | MAE | Spearman | Top-11 Pts |
|--------|------|----------|------------|
| 0.00 (mean only) | 1.0860 | 0.7377 | 51.3 |
| 0.05 | 1.0815 | 0.7397 | 50.6 |
| 0.10 | 1.0776 | 0.7412 | 50.1 |
| 0.15 (old default) | 1.0743 | 0.7423 | 50.9 |
| 0.20 | 1.0718 | 0.7432 | 51.1 |
| 0.25 | 1.0700 | 0.7437 | 51.6 |
| **0.30 (new default)** | **1.0688** | **0.7439** | **52.7** |

Every metric improves monotonically from 0.00 to 0.30: lower MAE (more accurate predictions), higher Spearman (better player rankings), and higher top-11 points (better squad selection). Captain hit rate was unchanged across all weights (11.1%).

Per-position greedy search confirmed: GKP=0.30, DEF=0.30, MID=0.25, FWD=0.30 — all positions favour more decomposed weight. Since per-position weights only marginally improved over a uniform 0.30, we used the simpler global weight to avoid overfitting to one backtest window.

**Conclusion**: Updated `EnsembleConfig.decomposed_weight` from `0.15` to `0.30`. The decomposed model (which predicts goals, assists, clean sheets, etc. separately via FPL scoring rules) contributes more useful signal than the original gut-feel 85/15 split gave it credit for. The blend is now 70/30.

**Re-run**: `.venv/bin/python scripts/optimize_blend.py [--start-gw N] [--end-gw N]` to re-validate after any model or feature changes.

#### Captain Score Weights — Empirically Optimised

Captain formula changed from `0.4*mean + 0.6*Q80` to `0.7*mean + 0.3*Q80` based on walk-forward analysis using `scripts/analyze_captain.py`. Over GW10-27 (18 GWs), the 0.7/0.3 formula scored 256 pts vs 244 for the old 0.4/0.6 formula. Pure mean scored 242, pure Q80 scored 222. The Q80 model adds value but was previously overweighted.

**Re-run**: `.venv/bin/python scripts/analyze_captain.py` to re-validate captain weights after model changes.

#### L10. Rotation Risk Modeled Only by availability_rate_last5 — WON'T FIX

**Problem**: No manager-specific rotation model. 5-GW availability window may miss systematic patterns. No cup rotation awareness.

**Won't fix**: Manager-specific rotation data isn't available in the public FPL API. Reverse-engineering rotation patterns from historical lineups is noisy and unreliable (e.g. Pep roulette). `availability_rate_last5` is a reasonable proxy given data constraints.

#### L11. Transfer Timing Doesn't Integrate Price Predictions — WON'T FIX

**Problem**: Price predictions exist but aren't fed into transfer timing decisions. No sell-before-drop logic.

**Won't fix**: The FPL price algorithm is proprietary and relies on daily transfer granularity the public API doesn't provide (only per-GW totals). Accurate prediction requires polling every 4-6 hours and filtering wildcard transfers — a different class of effort. The existing `_build_price_bonus()` in `transfer_planner.py` already nudges the solver to buy-before-rise using ownership + net transfers, which is a reasonable approximation (~3-5 pts/season ceiling for further improvements).

#### ~~L12. CLAUDE.md Documentation Inconsistency~~ FIXED

**Problem**: Audit checklist said "Bench weight (0.1)" but config has `bench_weight=0.25`.

**Fix**: Corrected — the old audit checklist text has been replaced.

---

### Top 5 Improvements by Expected Impact

All high and medium priority findings have been implemented (H1-H3, M1-M10). Run benchmarks to verify impact.

---

## Build Pipeline & Releases

GitHub Actions workflow builds Windows and macOS executables via PyInstaller. Triggers on release creation or manual `workflow_dispatch`.

### Key files
- **`.github/workflows/build-exe.yml`** — Two parallel jobs (windows-latest, macos-latest). Python 3.12.
- **`gaffer-windows.spec`** — PyInstaller spec for Windows (`GafferAI.exe`)
- **`gaffer-mac.spec`** — PyInstaller spec for macOS (`GafferAI.app` bundle)
- **`launcher.py`** — Entrypoint for PyInstaller builds. Starts Flask server and opens browser. If the server is already running, just opens the browser and exits.
- **`setup-mac.sh`** — Run once after cloning. Creates venv, installs deps, installs `GafferAI.app` to `/Applications`.

### Creating a release
```bash
gh release create v1.1.0 --title "v1.1.0" --notes "Release notes here"
```

### Rebuilding a release
`gh release delete` does NOT delete the git tag. You MUST delete both:
```bash
gh release delete v1.1.0 --yes
git push origin --delete v1.1.0
git tag -d v1.1.0
gh release create v1.1.0 --title "v1.1.0" --notes "Release notes here"
```

### macOS launcher behavior
The launcher script in `GafferAI.app` (and `launcher.py`) detects if the server is already running on port 9876. If yes, it just opens a new browser tab and exits — it does NOT kill and restart the server. This means closing the browser does not stop the server. To stop the server: `lsof -ti:9876 | xargs kill -9`.

### PyInstaller frozen-mode path detection
`src/paths.py` handles this globally:
```python
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
```

All path references go through `src.paths` — no module should compute its own paths. If a new file reads/writes to `output/`, `models/`, `cache/`, or `*.db`, it must use `src.paths`.

### Spec file gotchas
- **xgboost.testing**: `collect_all("xgboost")` imports `xgboost.testing` which calls `pytest.importorskip("hypothesis")`. Fixed with `filter_submodules=lambda name: "testing" not in name`.
- **Hidden imports**: All `src.*` modules must be listed explicitly in the spec files (including `src.season.state_machine`, `src.season.scheduler`, `src.api.season_v2_bp`).

---

## Full Audit Prompt

When the user says **"run full audit"**, you MUST execute every phase below in full. No shortcuts. No skipping. No summarising without reading code. Every checklist item requires reading the actual source file and citing file:line in your findings. If you cannot complete a phase, say so explicitly — do not silently skip it.

This audit has 5 mandatory phases. Each phase has a hard gate — you MUST complete it and report findings before moving to the next phase. You MUST use the Task tool to dispatch parallel agents where specified — do not attempt to do parallel work sequentially in the main thread.

---

### PHASE 1: Run All Tests (HARD GATE — nothing else starts until this passes)

Run this exact command:

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected**: 100+ tests, 0 failures. If ANY test fails, stop and fix it before proceeding. Do not proceed to Phase 2 with failing tests. Report the exact pass/fail count.

Test coverage:
- `test_correctness.py` — FPL scoring formulas, ensemble blend math, captain score, confidence decay, solver FPL compliance, prediction properties
- `test_integration.py` — Flask app, API routes, solver smoke tests, DB, config, imports
- `test_strategy_pipeline.py` — ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer, availability, plan invalidation, full E2E

---

### PHASE 2: Code Correctness (MANDATORY: dispatch exactly 3 parallel agents using the Task tool)

You MUST dispatch all 3 agents in a single message using the Task tool. Each agent MUST read every file listed and check every item. Each agent MUST return findings in this format for every item:

```
[ITEM] Data leakage — shift(1) on rolling calculations
[STATUS] PASS | BUG | CONCERN
[FILE:LINE] src/features/player_rolling.py:45
[EVIDENCE] Traced shift(1) call at line 45, applied after groupby rolling. Correct.
```

If an agent finds nothing wrong for an item, it still reports PASS with the file:line it checked. No silent passes.

#### Agent 1: Data & Features

**Files to read (ALL of them):** Every `.py` file in `src/features/` and `src/data/`.

**Checklist — answer every item:**

1. **Data leakage**: For every rolling/expanding/cumulative/EWM calculation, trace the `shift(1)` call. Cite the exact line. If shift is missing, that is a Critical bug. Check merge keys — does any merge allow future GW data into the training row?
2. **NaN handling**: For every feature column, verify fill defaults come from `FEATURE_FILL_DEFAULTS` in `src/config.py`. Flag any ad-hoc `.fillna(0)` that should use the central config.
3. **DGW targets**: In `src/features/targets.py`, verify target variables are divided by `next_gw_fixture_count`. Cite the line.
4. **Rolling windows**: For every `rolling(N)` or `.tail(N)` call, verify N matches intent (3 or 5 GW). Verify `.min_periods` is set correctly. Check for off-by-one.
5. **Interaction features**: Find `xg_x_opp_goals_conceded`, `chances_x_opp_big_chances`, `cs_opportunity` wherever they are computed. Verify the formula. Verify the opponent data is for the UPCOMING opponent, not a stale value.
6. **Cross-season boundaries**: Find where seasons are concatenated or where GW 1 data is used. Verify rolling stats don't leak across seasons unless intended.

#### Agent 2: ML Pipeline & Predictions

**Files to read (ALL of them):** `src/ml/prediction.py`, `src/ml/multi_gw.py`, `src/ml/decomposed.py`, `src/ml/training.py`, `src/ml/model_store.py`, `src/ml/backtest.py`.

**Checklist — answer every item:**

1. **Ensemble blend**: Read `_ensemble_predict_position` in `prediction.py`. Verify weights match `EnsembleConfig.decomposed_weight` (0.30). Verify the `how="outer"` merge doesn't drop players. Verify the `np.where` fallback logic.
2. **Availability zeroing**: In `generate_predictions()`, find where predictions are zeroed for unavailable players. Verify ALL prediction columns are zeroed (predicted_*, prediction_*, captain_score). **CRITICAL**: Verify that `predicted_next_3gw_points` is re-zeroed AFTER the 3-GW merge at the bottom of the function. Cite both zeroing locations with line numbers.
3. **DGW prediction summing**: In both `predict_for_position` and `predict_decomposed`, find the DGW deduplication logic. Verify predictions are SUMMED (not averaged) per player. Cite lines.
4. **Multi-GW snapshots**: Read `_build_offset_snapshot` in `multi_gw.py` end to end. Verify: (a) fixture columns are dropped and replaced, (b) interaction features are recomputed, (c) lookahead features (avg_fdr_next3, home_pct_next3, avg_opponent_elo_next3) are recomputed relative to `target_gw` not `latest_gw`.
5. **Confidence decay**: Find every use of `pred_cfg.confidence_decay`. Verify the fallback for offsets beyond the tuple length. Verify decay is applied as multiplication, not addition.
6. **Prediction intervals**: Find where `prediction_low` and `prediction_high` are computed. Verify `clip(min=0)` is applied. Verify binned intervals use the correct bin lookup.
7. **Decomposed scoring formula**: Read `predict_decomposed` in `decomposed.py` line by line. For each component (goals, assists, cs, goals_conceded, saves, bonus, defcon), verify: (a) the FPL points multiplier matches `FPLScoringRules` in config.py, (b) P(plays) gating is applied, (c) P(60+) is used for CS and goals_conceded, (d) DefCon uses Poisson CDF with correct threshold (10 for GKP/DEF, 12 for MID/FWD), (e) soft caps are applied correctly.

#### Agent 3: Solver, Strategy & Season

**Files to read (ALL of them):** Every `.py` file in `src/solver/`, every `.py` file in `src/strategy/`, and `src/season/manager.py`.

**Checklist — answer every item:**

1. **FPL rules in solver**: Read `solve_milp_team` in `squad.py`. Verify constraints: 15 players, 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per team, 11 starters, formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), budget, captain must be a starter. Cite the line for each constraint.
2. **Transfer hit counting**: Read `solve_transfer_milp_with_hits` in `transfers.py`. Verify forced replacements (unavailable players) are not counted as paid hits. Trace the `forced` variable. Cite lines.
3. **State tracking in multi-GW simulation**: Read `MultiWeekPlanner` in `transfer_planner.py`. For each simulated GW, verify: FTs increment correctly (+1 per GW, max 5), budget is preserved (not recalculated from squad value), WC makes transfers permanent, FH reverts squad.
4. **`_tick_planning()` flow**: Read this function in `season/manager.py`. Verify: (a) MultiWeekPlanner runs FIRST, (b) GW+1 is extracted from plan, (c) single-GW solver is ONLY used as fallback, (d) budget comes from `entry_history["value"]` not `sum(now_cost)`. Also verify `_tick_complete()` correctly records results and transitions phase.
5. **Solver fallbacks**: Find every call to `solve_milp_team`, `solve_transfer_milp`, `solve_transfer_milp_with_hits`. Verify each caller handles a `None` return. Find `_simulate_chip_gw` — verify it falls back to current squad points on solver failure.
6. **Phase transitions**: Read `state_machine.py` and `manager.py`. Verify: detect_phase priority is correct (COMPLETE > LIVE > READY > PLANNING), forced phase updates are logged, `_tick_complete()` transitions to SEASON_OVER at GW38 and PLANNING otherwise.
7. **Database integrity**: In `season/manager.py`, verify every DB query filters by correct `season_id` and `manager_id`. In `_track_prices`, verify watchlist players are included.
8. **Previously found bugs — regression check**: Verify EACH of these patterns has NOT been reintroduced:

| Pattern | Where to Check | What Goes Wrong |
|---------|---------------|-----------------|
| Availability zeroing before 3-GW merge | `ml/prediction.py` | Injured players get non-zero 3-GW predictions |
| Budget using now_cost sum | `season/manager.py` _tick_planning | Budget ~5% too high |
| MultiWeekPlanner first, MILP fallback | `season/manager.py` _tick_planning | Loses multi-GW intelligence |
| Solver failure kills planning | `strategy/transfer_planner.py` | One bad GW aborts entire 5-GW plan |
| Hit cost recomputed from sets | `solver/validator.py` | Forced replacements counted as paid hits |
| Watchlist excluded from prices | `season/manager.py` _track_prices_simple | Price alerts miss watched players |
| pandas CoW without .copy() | `features/playerstats.py` | Silent data corruption |
| FH squad not reverted for planning | `season/manager.py` _tick_planning | Planning uses FH squad instead of reverted pre-FH squad |

---

### PHASE 3: Mathematical & Statistical Foundations (MANDATORY: dispatch exactly 2 parallel agents using the Task tool)

These agents evaluate whether the mathematical and statistical choices are SOUND, not just bug-free. Each item requires a verdict: SOUND / QUESTIONABLE / WRONG, with reasoning.

#### Agent 4: Model Architecture & Feature Engineering

**Files to read:** `src/config.py` (all config values), `src/ml/decomposed.py`, `src/ml/training.py`, `src/features/registry.py`, at least 3 feature modules from `src/features/`.

**Checklist — answer every item with a verdict and reasoning:**

1. **Distribution choices** — For each decomposed component, state the objective used and evaluate:
   - Goals (Poisson, lambda ~0.1-0.3): Appropriate for low-count data? Or would Negative Binomial handle overdispersion better?
   - Saves (Poisson, lambda ~3-5): Higher counts, more variance. Still Poisson-appropriate?
   - DefCon CBIT (Poisson, range 0-20+): May be overdispersed. Does the Poisson CDF produce calibrated P(CBIT >= threshold)?
   - Clean sheets (squarederror): CS is effectively a probability. Is MSE the right loss for probabilities? Would log-loss be better?
   - Bonus (Poisson): Range 0-3. Is Poisson appropriate for bounded count data?
   - For each: cite the objective in `DecomposedConfig` and evaluate whether it's the right choice.

2. **Feature engineering quality** — Read the feature definitions and evaluate:
   - Rolling windows (3, 5 GW): Are these optimal window sizes for Premier League match data? What's the bias/variance tradeoff?
   - EWM span=5: Does exponential weighting add predictive value over simple rolling means? Is span=5 the right decay rate?
   - Interaction features: Are xG x opp_GC and chances x opp_big_chances the most informative interactions? What about xA x opp_GC, or form x fixture difficulty?
   - Feature count per position: How many features does each position model use? With ~500 training rows per position, is there overfitting risk? What's the features-to-samples ratio?
   - Dead weight features: Are there features that likely have near-zero importance? (e.g., highly correlated pairs, noisy single-match stats)

3. **Ensemble architecture** — Evaluate:
   - The 85/15 mean/decomposed blend: Is this weight empirically justified? What would happen at 80/20 or 90/10?
   - Simple weighted average vs alternatives: Would stacking (train a meta-learner on both outputs) be better? Would conditional switching (use decomposed when mean is uncertain) help?
   - When does the decomposed model add value? (Hypothesis: rotation/bench zone players where component breakdown reveals hidden value)

#### Agent 5: Calibration, Weights & Hyperparameters

**Files to read:** `src/config.py` (every tunable parameter), `src/ml/training.py`, `src/ml/prediction.py`, `src/solver/squad.py`.

**Checklist — answer every item with a verdict and reasoning:**

1. **Model calibration**: Known overprediction at 5+ points (pred=5.95, actual=4.24). Is this because of: (a) XGBoost regression to mean at extremes, (b) soft caps not aggressive enough, (c) training data distribution, or (d) something else? What would fix it?
2. **Soft caps** (GKP=7, DEF=8, MID=10, FWD=10): Are these suppressing genuine haul predictions? Should caps be DGW-aware (2x cap for DGW)? What percentage of predictions hit the cap?
3. **Confidence decay** (0.95, 0.93, 0.90, 0.87, 0.83, 0.80, 0.77): Is this curve too aggressive or too conservative? Should it vary by position (FWD rotate more)? Is the decay shape (roughly linear) correct, or should it be exponential/sigmoid?
4. **Captain formula** (0.7 * mean + 0.3 * Q80): Empirically optimised from 0.4/0.6 to 0.7/0.3 via walk-forward backtest (+12 pts over 18 GWs). Only MID/FWD have Q80 models — should GKP/DEF have captaincy scoring? (Probably not, but justify.)
5. **XGBoost hyperparameters** (n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8): Overfitting or underfitting? With walk-forward using 20 splits, is there enough data per fold? Would early stopping improve robustness?
6. **Sample weighting** (current_season=1.0, previous=0.5): Does this help or hurt? Would 0.7/0.3 or 0.8/0.2 be better? Does the game change enough season-to-season to justify heavy discounting?
7. **Bench weight** (0.25): What does this produce in practice? Is 0.25 too low (terrible bench = bad auto-subs) or too high (weakens starters)? What's the expected auto-sub frequency in FPL?
8. **Hit cost** (-4 points): Is this always the correct penalty? A hit to bring in a player for 5 good fixtures is different from one that helps for 1 GW. Does the planner account for amortized hit value?

---

### PHASE 4: FPL Domain Intelligence (MANDATORY: dispatch exactly 1 agent using the Task tool)

This agent does NOT check code. It evaluates whether the system would make GOOD FPL DECISIONS. It thinks like a top-1000 FPL manager who has played for 10+ years.

#### Agent 6: Think Like an Elite FPL Manager

**Files to read:** `src/strategy/transfer_planner.py`, `src/season/manager.py`, `src/season/state_machine.py`, `src/ml/prediction.py`, `src/config.py`.

**Checklist — answer every item with a verdict (GOOD / SUBOPTIMAL / WRONG) and what a top manager would do differently:**

1. **Captain selection**: Is the system picking the highest-EV captain? Is it too conservative (always Salah/Haaland regardless of fixture) or too bold (picking differentials without justification)? Are DGW captains valued correctly — a DGW Salah should be worth roughly 2x his SGW value, is it?
2. **Transfer timing**: Does the system sell before price drops and buy before price rises? How does price prediction quality affect transfer timing? Is the system making transfers too early (wasting information) or too late (missing price rises)?
3. **Chip timing — evaluate each chip:**
   - BB: Is it saved for DGWs with a strong bench? How does the system detect "good BB gameweeks"?
   - TC: Is it saved for DGW premium players? Or does it just pick the best SGW? A TC on a DGW Haaland >> TC on SGW Haaland.
   - FH: Is it used for BGWs (when your team has many blanks) or bad fixture clusters? Is the FH squad actually optimal or constrained?
   - WC: Is it used to reshape before a fixture swing, or just reactively to fix injuries? A proactive WC is worth 50+ pts more than a reactive one.
4. **Chip synergies**: Does the system detect WC->BB combos (wildcard to build a bench-boost-optimized squad)? FH+WC in adjacent weeks?
5. **Fixture awareness**: Is FDR actually predictive? Do "easy" fixtures produce more points in practice? Should the system weight form MORE than fixtures, or vice versa?
6. **Rotation risk**: How does the system handle rotation-prone players (Pep roulette, cup rotation)? Is `availability_rate_last5` sufficient, or should there be a manager-specific rotation model?
7. **Differential thinking**: Should ownership matter? In H2H/mini-league, differentials win. In OR, template is safer. Is there a mode toggle? Should there be?
8. **Hit tolerance**: Is -4 always correct? A -8 to bring in two players for a 5-GW fixture swing can return +20. Does the 5-GW planner evaluate amortized hit value?
9. **Form vs fixtures**: A player in great form with hard fixtures — does the model handle this correctly? Does it overweight recent form or overweight fixtures?
10. **Bench order and auto-subs**: Is bench order optimised for auto-sub probability? The GKP-only-replaces-GKP rule affects optimal bench composition. Is this modelled?
11. **End-of-season strategy**: Does the system change behaviour in GW35-38? (More aggressive, chip dumping, differential captaincy for rank chasing)

---

### PHASE 5: Improvement Discovery (MANDATORY: dispatch exactly 1 agent using the Task tool)

This agent synthesises everything from Phases 2-4 and identifies the highest-impact improvements.

#### Agent 7: What's the Biggest Opportunity?

**Input**: Findings from all previous agents (pass the collected findings as context).

**Task**: Identify the **top 5 improvements** ranked by expected FPL points gained per season. For each improvement, provide:
- **What**: Concrete description of the change
- **Why**: Evidence from the audit findings or FPL theory
- **Expected impact**: Estimated points per season (be specific — "captain improvement of 1 pt/GW = 38 pts/season")
- **Effort**: Low / Medium / High
- **Risk**: What could go wrong

Categories to consider:
- Feature ideas: xA x opp_GC, xG overperformance, progressive carries, set piece taker ID, rotation pattern detection
- Model ideas: LightGBM comparison, stacking, position-specific hyperparameters, time-series features
- Captain improvements: Bayesian selection, variance-aware captaincy, fixture-specific history
- Transfer improvements: Price change integration, multi-week chaining, amortized hit value
- Strategic improvements: Differential mode, rank-chasing mode, template awareness, auto-sub optimisation

---

### FINAL OUTPUT (MANDATORY — do not skip this)

After all 7 agents have returned, compile the complete audit report. This is not optional.

**Section 1: Test Results**
- Exact pass/fail count from Phase 1

**Section 2: All Findings**
For each finding from Phases 2-4:
1. **Category**: Bug / Mathematical Issue / FPL Domain Issue / Improvement Opportunity
2. **Severity/Impact**: Critical / High / Medium (for bugs) or Expected Points Impact (for improvements)
3. **File:line** (for code issues)
4. **What's wrong / What could be better**
5. **Evidence** (cite code lines, math, or FPL theory — no hand-waving)
6. **Suggested fix / approach**

**Section 3: Regression Check**
For each previously found bug pattern, state PASS or REINTRODUCED:

| Pattern | Status | Evidence |
|---------|--------|----------|
| Availability zeroing before 3-GW merge | PASS/REINTRODUCED | file:line |
| Budget using now_cost sum | PASS/REINTRODUCED | file:line |
| Single-GW solver as primary path | PASS/REINTRODUCED | file:line |
| Chip plan not passed to planner | PASS/REINTRODUCED | file:line |
| Solver failure kills planning | PASS/REINTRODUCED | file:line |
| Hit cost recomputed from sets | PASS/REINTRODUCED | file:line |
| Watchlist excluded from prices | PASS/REINTRODUCED | file:line |
| pandas CoW without .copy() | PASS/REINTRODUCED | file:line |
| FH squad not reverted for planning | PASS/REINTRODUCED | file:line |

**Section 4: Top 5 Actions**
Ranked by expected improvement to season performance:
- **Must Fix**: Bugs that produce wrong results
- **Should Investigate**: Mathematical/strategic questions with evidence they matter
- **Could Explore**: Improvement ideas with estimated point impact

---

## Benchmarking Protocol

When the user says **"run benchmarks"** or **"validate the model"**, execute this protocol. Run it after any model change, feature engineering change, or methodology fix.

### Step 1: Targeted Tests
Write a standalone Python script (in `/private/tmp/`) that tests each specific change with focused assertions:
- Feature changes: Verify features present/absent in trained model feature lists
- Computation fixes: Compare fixed computation against expected values
- Logic changes: Test edge cases (COP=0, COP=50, COP=100)
- Sanity checks: No negative predictions, no absurdly high predictions (>15 pts 1-GW)

### Step 2: Walk-Forward Backtest (Before/After)
```bash
# Kill leftover processes
lsof -ti:9876 | xargs kill -9

# Baseline
git stash
FLASK_APP=src.api .venv/bin/python -m flask run --port 9876 &
sleep 5
curl -s -X POST http://127.0.0.1:9876/api/backtest -H 'Content-Type: application/json' -d '{"start_gw":10,"end_gw":27}'
# Wait for completion (check /api/status SSE stream)
curl -s http://127.0.0.1:9876/api/backtest-results > /private/tmp/backtest_baseline.json
lsof -ti:9876 | xargs kill -9

# After changes
git stash pop
FLASK_APP=src.api .venv/bin/python -m flask run --port 9876 &
sleep 5
curl -s -X POST http://127.0.0.1:9876/api/backtest -H 'Content-Type: application/json' -d '{"start_gw":10,"end_gw":27}'
curl -s http://127.0.0.1:9876/api/backtest-results > /private/tmp/backtest_postfix.json
```

### Key Metrics
| Metric | Direction | Importance |
|--------|-----------|------------|
| `model_avg_mae` | Lower | Primary |
| `model_avg_mae_played` | Lower | Primary |
| `avg_spearman` | Higher | Primary |
| `avg_ndcg_top20` | Higher | High |
| `model_avg_top11_pts` | Higher | High |
| `model_capture_pct` | Higher | High |
| `model_wins` vs `ep_wins` | More wins | Medium |
| `captain_hit_rate` | Higher | Medium |
| `mae_pvalue` | Lower | Validation |

### Pass criteria
- `model_avg_mae` must not increase (same or lower)
- `model_avg_top11_pts` must not decrease
- No more than 2 individual GWs should regress in MAE
- Per-position MAE: no position should regress by more than 0.05

### Quick Reference
```bash
# Fetch results summary
curl -s http://127.0.0.1:9876/api/backtest-results | python3 -c "
import sys,json; r=json.load(sys.stdin); s=r['summary']
print(f'MAE={s[\"model_avg_mae\"]:.4f} rho={s[\"avg_spearman\"]:.4f} top11={s[\"model_avg_top11_pts\"]:.1f} cap%={s[\"model_capture_pct\"]:.1f}%')
print(f'vs ep: {s[\"model_wins\"]}W-{s[\"ep_wins\"]}L | MAE p={s.get(\"mae_pvalue\",\"?\")}')
"

# Spot-check top predicted players
curl -s http://127.0.0.1:9876/api/predictions | python3 -c "
import sys,json; d=json.load(sys.stdin)['players']
d.sort(key=lambda x: x.get('predicted_next_gw_points',0), reverse=True)
for p in d[:10]:
    print(f\"{p['web_name']:15} {p['position']:3} {p.get('predicted_next_gw_points',0):.2f} pts\")
"
```

---

## Remaining TODO

### Authenticated FPL API Access
For autonomous execution:
1. FPL Authentication (email/password -> session cookies)
2. Write API endpoints (execute transfers, set captain, activate chips)
3. Exact selling prices (50% profit sharing on price rises — per-player, not just total)
4. "Execute All" button with confirmation flow
5. Safety guardrails (deadline awareness, rollback info, dry-run mode)
