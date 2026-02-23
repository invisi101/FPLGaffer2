# FPL Gaffer 2 — Claude Code Notes

## Project Goal

Build a fully autonomous FPL manager. This is NOT just a prediction tool — it should think and plan like a real FPL manager across the entire season:

- **Transfer planning**: Rolling 5-GW horizon with FT banking, price awareness, and fixture swings
- **Squad building**: Shape the squad toward upcoming fixture runs, not just next GW
- **Captain planning**: Joint captain optimization in the MILP solver + pre-planned captaincy calendar
- **Chip strategy**: Evaluate all 4 chips across every remaining GW using DGW/BGW awareness, squad-specific predictions, and chip synergies (WC->BB, FH+WC)
- **Price awareness**: Ownership-based price change predictions with probability scores
- **Reactive adjustments**: Auto-detect injuries, fixture changes, and prediction shifts that invalidate the plan — with SSE-driven alerts and one-click replan
- **Outcome tracking**: Record what was recommended vs what happened, track model accuracy over time

Every decision (transfer, captain, bench order, chip) is made in context of the bigger picture. The app produces a rolling multi-week plan that constantly recalculates as new information comes in.

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

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `FLASK_APP=src.api .venv/bin/python -m flask run --port 9874` (serves on `http://127.0.0.1:9874`)
- **Port 9874**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9874 | xargs kill -9` before starting
- **Tests**: `.venv/bin/python -m pytest tests/test_integration.py -v` (17 integration tests)
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
├── api/                     # Flask blueprints (7 blueprints, 45+ endpoints)
│   ├── __init__.py          # create_app() factory, blueprint registration
│   ├── core.py              # Predictions, training, status, monsters, PL table
│   ├── team.py              # Best team, my team, transfer recommendations
│   ├── season_bp.py         # Season init, dashboard, recommendations, snapshots
│   ├── strategy_bp.py       # Strategic plan, action plan, plan health, preseason
│   ├── prices_bp.py         # Prices, predictions, history, watchlist
│   ├── backtest_bp.py       # Walk-forward backtesting
│   ├── compare_bp.py        # GW compare (actual vs hindsight-best)
│   ├── helpers.py           # Shared: safe_num, scrub_nan, load_bootstrap, get_next_gw
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
├── strategy/                # Strategic planning brain
│   ├── chip_evaluator.py    # Evaluate all 4 chips across remaining GWs
│   ├── transfer_planner.py  # Multi-week rolling transfer planner
│   ├── captain_planner.py   # Captain optimization across horizon
│   ├── plan_synthesizer.py  # Combine into coherent plan + changelog
│   ├── reactive.py          # Detect plan invalidation (injuries, fixtures)
│   └── price_tracker.py     # Price alerts, ownership-based predictions
│
├── season/                  # Season orchestration
│   ├── manager.py           # SeasonManager class — the central orchestrator
│   ├── dashboard.py         # Dashboard data aggregation
│   ├── recorder.py          # Record actual results, compare to recommendation
│   ├── fixtures.py          # Fixture calendar builder
│   └── preseason.py         # Pre-GW1 squad selection + chip plan
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
tests/      # Integration tests (17 tests)
```

---

## Data Pipeline

### Sources
1. **GitHub (FPL-Core-Insights)**: Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Cached 6 hours.
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

---

## Configuration (`src/config.py`)

Every magic number is defined in frozen dataclass configs:

| Config | Key Values |
|--------|------------|
| `XGBConfig` | 150 trees, depth 5, lr 0.1, subsample 0.8, walk-forward 20 splits |
| `EnsembleConfig` | 85/15 mean/decomposed blend, captain 0.4 mean + 0.6 Q80 |
| `SolverConfig` | 0.1 bench weight, -4 hit cost, max budget 1000, 3 per team |
| `CacheConfig` | GitHub CSV 6h, FPL API 30m, manager API 1m |
| `PredictionConfig` | Confidence decay 0.95->0.77, pool size 200 |
| `DecomposedConfig` | Position-specific components, Poisson/squarederror objectives, soft caps |
| `DataConfig` | GitHub CSV base URL, FPL API base URL, earliest season 2024-2025 |
| `StrategyConfig` | 5-GW planning horizon, max 1 hit/GW, 5 max banked FTs |
| `FPLScoringRules` | Full FPL points per action by position (incl DefCon thresholds) |

Import as singletons: `from src.config import xgb, ensemble, solver_cfg, ...`

---

## Model Architecture

### Tier 1: Mean Regression (Primary)
- 4 models (one per position) for `next_gw_points`
- XGBoost `reg:squarederror`, walk-forward CV (last 20 splits)
- Sample weighting: current season 1.0, previous 0.5
- Fixed hyperparameters: 150 trees, depth 5, lr 0.1, subsample 0.8
- 3-GW predictions derived at inference by summing three 1-GW predictions with per-GW opponent data

### Tier 2: Quantile Models (Captain Picks)
- MID + FWD only, 80th percentile of next_gw_points
- `captain_score = 0.4 x mean + 0.6 x Q80` — captures explosive upside

### Tier 3: Decomposed Sub-Models
- Position-specific component models predicting individual scoring elements:
  - **GKP**: cs, goals_conceded, saves, bonus
  - **DEF**: goals, assists, cs, goals_conceded, bonus, defcon
  - **MID**: goals, assists, cs, bonus, defcon
  - **FWD**: goals, assists, bonus, defcon
- Poisson objectives for count data (goals, assists, bonus, saves, goals_conceded, defcon), squarederror for CS
- DefCon: Poisson CDF predicts P(CBIT >= threshold) where threshold = 10 (GKP/DEF) or 12 (MID/FWD), scores +2 pts
- Combined via FPL scoring rules with playing probability weighting
- Soft calibration caps per position (GKP=7, DEF=8, MID=10, FWD=10)

### Ensemble
- Production predictions use an **85/15 blend** of mean regression and decomposed sub-models
- Mean model drives prediction accuracy; decomposed weight preserves ranking signal in the bench/rotation zone

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
- **Objective**: max(0.9 x starting XI pts + 0.1 x bench pts + captain bonus)
- **Constraints**: Budget, positions (2/5/5/3), max 3 per team, 11 starters, formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), exactly 1 captain who is a starter
- **Returns**: starters, bench, total_cost, starting_points, captain_id

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

## Strategic Planning Brain (`src/strategy/`)

### ChipEvaluator (`chip_evaluator.py`)
Evaluates all 4 chips (BB, TC, FH, WC) across every remaining GW:
- Near-term: Uses model predictions (solve MILP for FH/WC, bench sums for BB, best starter for TC)
- Far-term: Fixture heuristics (DGW count, BGW count, FDR)
- Synergies: WC->BB (build BB-optimized squad), FH+WC (complementary strategy)

### MultiWeekPlanner (`transfer_planner.py`)
Rolling **5-GW** transfer planner:
- Tree search: generates all valid FT allocation sequences, simulates each, picks the path maximizing total points
- Considers: FT banking (save vs spend at every GW), fixture swings, price change probability
- Reduces pool to top 200 players for efficiency
- Passes `captain_col` to MILP solver for captain-aware squad building

### CaptainPlanner (`captain_planner.py`)
Pre-plans captaincy across the prediction horizon:
- Uses transfer plan squads to pick captain from the planned squad
- Flags weak captain GWs (predicted < 4 pts)

### PlanSynthesizer (`plan_synthesizer.py`)
Combines all plans into a coherent timeline:
- Chip schedule (synergy-aware)
- Natural-language rationale explaining the overall strategy
- Comparison with previous plan -> changelog

### Reactive Re-planning (`reactive.py`)
- `detect_plan_invalidation()`: Checks injuries (critical), fixture changes (BB without DGW), prediction shifts (>50% captain drop), doubtful players
- `apply_availability_adjustments()`: Zeros predictions for injured/suspended players
- `check_plan_health()`: Lightweight check using bootstrap data (no prediction regeneration)
- Auto-triggers on data refresh via SSE `plan_invalidated` events

---

## Season Manager (`src/season/manager.py`)

Orchestrates everything for a full season. Central class: `SeasonManager`.

### Weekly Workflow
1. **Refresh Data** -> updates cache, detects availability issues, checks plan health
2. **Generate Recommendation** -> full strategy pipeline -> extract GW+1 -> DB store
3. **Review Action Plan** -> clear steps (transfer X out / Y in, set captain to Z, activate chip)
4. **Make Moves** -> user executes on FPL website
5. **Record Results** -> imports actual picks, compares to recommendation, tracks accuracy

### `generate_recommendation()` — The Critical Orchestration

This is the most complex method. Understanding its flow is essential for debugging:

1. **Load data + build features + generate predictions** (1-GW ensemble + 3-GW)
2. **Fetch current squad** from FPL API (picks, history, entry_history)
3. **Calculate budget** using `entry_history["value"]` (real selling value with 50% profit rule), NOT `sum(now_cost)` which is optimistically high
4. **Run `_run_strategy_pipeline()` FIRST** — this is the primary path:
   - Generate 8-GW multi-GW predictions
   - Replace GW+1 predictions with the exact numbers from the Predictions tab (consistency)
   - Enrich GW+2+ with bootstrap data (web_name, position, cost, team_code)
   - Apply availability adjustments (zero injured players across all future GWs)
   - Determine available chips (half-season boundary aware)
   - Run ChipEvaluator -> chip heatmap + synergies
   - Build `chip_plan` from heatmap and pass to MultiWeekPlanner
   - Run MultiWeekPlanner -> 5-GW transfer plan with FT banking
   - Run CaptainPlanner -> captaincy across horizon
   - Run PlanSynthesizer -> unified timeline + rationale + chip schedule
   - Save strategic plan + detect/log plan changes vs previous plan
   - **Return** the strategic plan so GW+1 can be extracted
5. **Extract GW+1** from the strategic plan timeline (transfers, captain, chip)
6. **Fallback**: If pipeline fails, fall back to single-GW MILP solver (`solve_transfer_milp_with_hits`) and save a stub strategic plan
7. **Run bank-vs-use analysis** (2-week FT allocation comparison)
8. **Save recommendation** to DB

### Other Key Methods
- `init_season()` — Backfills season history from FPL API; pre-season calls `generate_preseason_plan()` instead of erroring
- `get_action_plan()` — Looks up recommendation by next_gw (not just latest), builds human-readable steps
- `record_actual_results()` — Post-GW import + comparison to recommendation
- `get_dashboard()` — Aggregated dashboard data (rank, points, budget, accuracy)
- `_log_plan_changes()` — Compares old vs new strategic plans, logs chip reschedules and captain changes to plan_changelog

### Price Tracking
- `track_prices()`: Snapshots prices for squad players **AND watchlist players**
- `get_price_alerts()`: Raw net-transfer threshold alerts
- `predict_price_changes()`: Ownership-based algorithm approximation
- `get_price_history()`: Historical snapshots with date/price/net_transfers

---

## Database Schema (`src/db/`)

9 SQLite tables defined in `schema.py`, with migrations in `migrations.py`:

| Table | Purpose |
|-------|---------|
| `season` | Manager seasons (id, manager_id, name, start_gw, current_gw) |
| `gw_snapshot` | Per-GW state (squad_json, bank, team_value, points, rank, captain, transfers) |
| `recommendation` | Pre-GW advice (transfers_json, captain, chip, predicted/base/current_xi_points) |
| `recommendation_outcome` | Post-GW tracking (followed_transfers, actual_points, point_delta) |
| `price_tracker` | Player price snapshots (price, transfers_in/out, snapshot_date) |
| `fixture_calendar` | GW x team fixture grid (fixture_count, fdr_avg, is_dgw, is_bgw) |
| `strategic_plan` | Full plan JSON + chip heatmap JSON (per season per GW) |
| `plan_changelog` | Plan change history (chip reschedule, captain change, reason) |
| `watchlist` | User watchlist for price tracking |

9 repository classes in `repositories.py`: `SeasonRepository`, `SnapshotRepository`, `RecommendationRepository`, `OutcomeRepository`, `PriceRepository`, `FixtureRepository`, `PlanRepository`, `DashboardRepository`, `WatchlistRepository`.

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

### Season Management (`season_bp.py`)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/season/init` | Backfill season history |
| GET | `/api/season/status` | Check if season exists |
| DELETE | `/api/season/delete` | Delete season data |
| GET | `/api/season/dashboard` | Full dashboard (rank, budget, accuracy) |
| POST | `/api/season/generate-recommendation` | Generate strategic plan + recommendation |
| POST | `/api/season/record-results` | Import actual results, compare to advice |
| GET | `/api/season/recommendations` | All recommendations for season |
| GET | `/api/season/snapshots` | All GW snapshots |
| GET | `/api/season/fixtures` | Fixture calendar |
| GET | `/api/season/chips` | Chip status + values |
| GET | `/api/season/gw-detail` | Detail for specific GW |
| GET | `/api/season/transfer-history` | Transfer history |
| GET | `/api/season/bank-analysis` | Bank analysis |
| POST | `/api/season/update-fixtures` | Rebuild fixture calendar |

### Strategic Planning (`strategy_bp.py`)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/api/season/strategic-plan` | Fetch/generate full strategic plan |
| GET | `/api/season/action-plan` | Clear action items for next GW |
| GET | `/api/season/outcomes` | All recorded outcomes |
| GET | `/api/season/plan-health` | Check plan validity (injuries/fixtures) |
| GET | `/api/season/plan-changelog` | Plan change history |
| POST | `/api/preseason/generate` | Pre-season initial squad + chip plan |
| GET | `/api/preseason/result` | Pre-season results |

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

### Season Sub-tabs
- **Overview**: Rank chart, points bar chart, budget chart, prediction accuracy
- **Workflow**: Step indicators (Refresh -> Recommend -> Review -> Execute -> Record), action plan, outcomes
- **Fixtures**: GW x team fixture grid
- **Transfers**: Transfer history table
- **Chips**: Status (used/available) + values
- **Prices**: Alerts, ownership-based predictions, price history chart, squad prices
- **Strategy**: Plan health banner, rationale, transfer timeline cards, captain plan, chip schedule + synergies, changelog

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

This pattern is required in: `core.py` (predictions endpoint), `team.py` (best-team, transfer-recs), `season/manager.py` (generate_recommendation).

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
# In generate_recommendation():
api_value = entry_history.get("value")
if api_value:
    total_budget = round(api_value / 10, 1)  # Correct: includes 50% rule
else:
    total_budget = round(bank + current_squad_cost, 1)  # Fallback: slightly optimistic
```

### 3. Strategy Pipeline First, Single-GW Solver as Fallback

`generate_recommendation()` runs the full multi-GW strategy pipeline FIRST, then extracts GW+1 from the timeline. The single-GW MILP solver is ONLY the fallback when the pipeline fails. Do NOT reverse this — the pipeline produces better recommendations because it considers future GWs.

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

### 8. price_tracker Must Include season_id

`strategy/price_tracker.py` functions that return player dicts must include `"season_id": season_id` so the repository can correctly filter snapshots.

---

## Testing

```bash
# Kill leftover server
lsof -ti:9874 | xargs kill -9

# Run all tests (99 total, ~5 min)
.venv/bin/python -m pytest tests/ -v

# Run fast tests only (~1 sec)
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v

# Start server
FLASK_APP=src.api .venv/bin/python -m flask run --port 9874
```

### Test Structure (3 files, 99 tests)

**`test_correctness.py`** (53 tests, <1 sec) — Mathematical correctness and FPL compliance:
- Config sanity: decay curve, ensemble weights, captain weights, squad positions, FPL scoring rules, soft caps, DefCon thresholds
- Decomposed scoring formula: P(plays) logic, appearance points, goal/CS/GC/saves/DefCon formulas, soft cap math, DGW summing
- Ensemble blend: weighted average, boundary properties, mean model dominance
- Captain score: formula values, Q80 weighting, NaN fallback, doubling mechanics
- Confidence decay: monotonicity, bounds, fallback consistency
- Solver FPL compliance: full rule validation on 30-player pool, captain constraints, transfer keeping, hit cost arithmetic, baseline existence
- Prediction properties: no negatives, availability zeroing, 3-GW re-zeroing after merge

**`test_integration.py`** (17 tests, ~1 sec) — Smoke tests:
- Flask app creation and route registration
- API endpoints (predictions, model-info, season/status)
- MILP solver (squad, transfers, captain)
- FPL rules validation (squad count, team limits)
- Database schema creation and CRUD
- Feature registry, config loading, module imports

**`test_strategy_pipeline.py`** (29 tests, ~5 min) — Strategy layer behaviour:
- ChipEvaluator: heatmap, per-GW values, BB in DGW, empty chips, synergies
- MultiWeekPlanner: plan steps, horizon, squad IDs, zero FT, rationale, empty predictions
- CaptainPlanner: coverage, captain in squad, required fields, transfer plan squads, VC != captain
- PlanSynthesizer: required keys, timeline merges, no duplicate chip GWs, JSON serialisable
- Availability: injured zeroed all GWs, doubtful first GW only, healthy unchanged
- Plan invalidation: injury triggers, captain drop triggers, no triggers when healthy
- Full pipeline: valid plan, injuries adjust captaincy, DB storage roundtrip

### Key test commands
```bash
# Action plan
curl -s "http://127.0.0.1:9874/api/season/action-plan?manager_id=12904702"

# Strategic plan
curl -s "http://127.0.0.1:9874/api/season/strategic-plan?manager_id=12904702"

# Plan health
curl -s "http://127.0.0.1:9874/api/season/plan-health?manager_id=12904702"

# Dashboard
curl -s "http://127.0.0.1:9874/api/season/dashboard?manager_id=12904702"
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
- **FT planner explores at most 1 hit per GW**: Taking 2+ hits is almost never profitable.
- **3-GW prediction is a simple sum**: No adjustment for form regression or rotation risk.
- **Selling prices partially use `now_cost`**: Budget calculation uses `entry_history["value"]` (correct). However, the MILP solver still uses `now_cost` for individual player costs since the public API doesn't provide per-player selling prices (50% profit sharing on price rises). This means the solver may slightly overestimate available budget when selling players whose prices have risen.

---

## Build Pipeline & Releases

GitHub Actions workflow builds Windows and macOS executables via PyInstaller.

### Key files
- `.github/workflows/build-exe.yml` — Two parallel jobs (windows-latest, macos-latest)
- `fpl-predictor.spec` / `gaffer-mac.spec` — PyInstaller specs
- `launcher.py` — Entrypoint for builds (starts Flask + opens browser)

### PyInstaller frozen-mode path detection
`src/paths.py` handles this globally:
```python
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
```

All path references go through `src.paths` — no module should compute its own paths.

---

## Full Audit Prompt

When the user says **"run full audit"**, execute the following comprehensive audit covering **code correctness, mathematical soundness, FPL domain intelligence, and improvement opportunities**. This is not just a bug hunt — it evaluates whether the system would make good FPL decisions in practice.

### Phase 1: Automated Verification (Run First — Blocks Everything Else)

```bash
.venv/bin/python -m pytest tests/ -v
```

All tests must pass. If any fail, fix them before proceeding. The test suite covers:
- `test_correctness.py` (53 tests) — FPL scoring formulas, ensemble blend math, captain score, confidence decay, solver FPL compliance, prediction properties
- `test_integration.py` (17 tests) — Flask app, API routes, solver smoke tests, DB, config, imports
- `test_strategy_pipeline.py` (29 tests) — ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer, availability, plan invalidation, full E2E

### Phase 2: Code Correctness (Dispatch 3 Parallel Agents)

#### Agent 1: Data & Features

Read every file in `src/features/` and `src/data/`. Check:

- **Data leakage**: Every feature must only use GW N-1 and earlier when predicting GW N+1. Trace `shift(1)` on all rolling/expanding/cumulative calculations. Check merge keys.
- **NaN handling**: Consistent fill defaults from `FEATURE_FILL_DEFAULTS`. No silent NaN propagation into model inputs.
- **DGW targets**: Target variables divided by `next_gw_fixture_count` so DGW matches don't get double-counted during training.
- **Rolling windows**: Verify 3-GW and 5-GW windows use correct slice boundaries. No off-by-one.
- **Interaction features**: `xg_x_opp_goals_conceded`, `chances_x_opp_big_chances`, `cs_opportunity` — verify formulas match intent and don't use stale opponent data.
- **Cross-season boundaries**: GW 1 of season 2 correctly follows GW 38 of season 1. Rolling stats reset or carry over correctly.

#### Agent 2: ML Pipeline & Predictions

Read `src/ml/prediction.py`, `src/ml/multi_gw.py`, `src/ml/decomposed.py`, `src/ml/training.py`. Check:

- **Ensemble blend**: 85/15 mean/decomposed, weighted correctly. `_ensemble_predict_position` merges with `how="outer"` — verify no player is dropped.
- **Availability zeroing**: ALL prediction columns zeroed for unavailable players. **CRITICAL**: 3-GW predictions re-zeroed AFTER the merge (they come from `multi_gw.py` which doesn't know about availability).
- **DGW prediction summing**: Per-fixture predictions summed, not averaged. Both `predict_for_position` and `predict_decomposed` handle this.
- **Multi-GW snapshots**: `_build_offset_snapshot` correctly swaps fixture/opponent columns for each future GW. Interaction features recomputed. Lookahead features (avg_fdr_next3, etc.) recomputed relative to target GW.
- **Confidence decay**: Uses `PredictionConfig.confidence_decay` tuple from config. Falls back correctly for offsets beyond tuple length.
- **Prediction intervals**: Binned by prediction magnitude. Bounds are reasonable (low >= 0).
- **Decomposed scoring formula**: P(plays) gating, P(60+) for CS/GC, Poisson CDF for DefCon, soft caps. Compare formula against actual FPL scoring rules line by line.

#### Agent 3: Solver, Strategy & Season

Read `src/solver/`, `src/strategy/`, `src/season/manager.py`. Check:

- **FPL rules in solver**: 15 players, 2/5/5/3, max 3 per team, formation limits, budget, captain is a starter.
- **Transfer hit counting**: Forced replacements (unavailable players dropped by `dropna`) not counted as paid hits.
- **State tracking**: FTs, budget, squad across multi-GW simulation. WC=permanent, FH=reverts, BB/TC=one-GW.
- **`generate_recommendation()` flow**: Strategy pipeline runs FIRST, GW+1 extracted from timeline, single-GW solver is ONLY the fallback. Budget uses `entry_history["value"]`.
- **Solver fallbacks**: Every MILP call handles `None` returns. `_simulate_chip_gw()` falls back to current squad points.
- **Chip evaluation**: BB in DGW, FH in BGW, WC over 3-GW window. Synergies don't cross half-season boundary.
- **Database integrity**: Correct season_id/manager_id filtering. price_tracker includes season_id.
- **Frontend/backend contract**: predictions.csv has `position_clean` not `position` — must alias. No `cost` column — must enrich from bootstrap.

### Phase 3: Mathematical & Statistical Foundations (Dispatch 2 Parallel Agents)

#### Agent 4: Model Architecture & Feature Engineering

Question whether the mathematical choices are sound. Don't just check for bugs — evaluate whether the approach is optimal.

- **Distribution choices**:
  - Poisson for goals (range 0-4, low lambda ~0.1-0.3): Appropriate? Or is Negative Binomial better for overdispersion?
  - Poisson for saves (range 0-10, higher lambda): Still count data, but more variance.
  - Poisson for DefCon CBIT (range 0-20+): May be overdispersed. Does the Poisson CDF produce calibrated probabilities?
  - squarederror for CS: Correct — CS is effectively a probability (DGW makes it fractional).
  - Are there components where the wrong objective is used?

- **Feature engineering quality**:
  - Rolling windows (3, 5 GW): Too short = noisy, too long = stale. Are these optimal for Premier League data?
  - EWM span=5: Is this the right decay rate? Does it add value over simple rolling?
  - Interaction features (xG x opp_GC, chances x opp_big_chances): Are these the right interactions? What about xA x opp_GC?
  - Feature count per position (GKP ~30, DEF ~40, MID ~50, FWD ~40): Overfitting risk with ~500 training GWs per position?
  - Which features have highest importance? Are any dead weight?

- **Ensemble architecture**:
  - 85/15 blend: Empirically justified? What would the backtest say about 80/20 or 90/10?
  - Simple weighted average: Is this optimal? Alternatives: stacking, switching based on confidence.
  - When does the decomposed model outperform mean? (Rotation/bench players where component breakdown matters)

#### Agent 5: Calibration, Weights & Hyperparameters

Quantitative review of every tunable parameter.

- **Calibration**: Is the model well-calibrated? (predicted 5 pts => actual ~5 pts). Known overprediction at high end.
- **Soft caps** (GKP=7, DEF=8, MID=10, FWD=10): Suppressing genuine haul predictions? How often does the model predict above cap? Should caps be DGW-aware (higher cap for DGW)?
- **Confidence decay** (0.95, 0.93, 0.90, 0.87, 0.83, 0.80, 0.77): Too aggressive or conservative? Should it vary by position (FWD may decay faster due to rotation)?
- **Captain formula** (0.4 mean + 0.6 Q80): Right balance? Only MID/FWD have Q80 models — should DEF/GKP be eligible for captaincy scoring?
- **XGBoost hyperparameters** (150 trees, depth 5, lr 0.1): Overfitting or underfitting? Walk-forward with 20 splits — enough data per fold?
- **Sample weighting** (current season 1.0, previous 0.5): Helping? What about 0.7/0.3?
- **Bench weight** (0.1): Does this produce good benches? Too low = terrible bench players. Too high = starters suffer. What's the optimal weight considering auto-subs?

### Phase 4: FPL Domain Intelligence (1 Agent)

#### Agent 6: Think Like an Elite FPL Manager

Don't just check if the code works. Ask: **would a top-1000 FPL manager make these same decisions?**

- **Captain picks**: Is the system picking the highest-EV captain? Too conservative (always Salah/Haaland) or too bold? Are DGW captains valued correctly (should be ~2x, not just "best player")?
- **Transfer timing**: Does the system sell before price drops and buy before rises? Is the price prediction model actually profitable, or is it noise?
- **Chip timing**:
  - BB: Saved for DGWs? Is DGW detection working?
  - TC: Saved for DGW premium players? Or just best single-GW pick?
  - FH: Used in BGWs? Or to navigate bad fixture clusters?
  - WC: Reshaping squad before fixture swings? Or just fixing a bad squad?
- **Fixture awareness**: Is FDR actually predictive? Are "easy" fixtures really easier? (Check backtest `by_difficulty` diagnostic when available)
- **Rotation risk**: Is the system accounting for rotation? A player who plays 60 mins every other game is less valuable than one who plays 90 every game. How well does `availability_rate_last5` capture this?
- **Differential thinking**: Should the system consider ownership? In mini-league context, differentials matter. In OR context, they don't. Is there a toggle?
- **Hit tolerance**: Is -4 per hit always the right cost? A hit that brings in a player for 5 good fixtures is different from one that helps for 1 GW. Does the 5-GW planner account for this?
- **Form vs fixtures**: Is the model balancing recent form against upcoming fixtures correctly? A player in great form with hard fixtures — what happens?
- **Bench order**: Is auto-sub optimisation considered? The 4th-sub rule (GKP only replaces GKP) affects optimal bench composition.

### Phase 5: Improvement Discovery (1 Agent)

#### Agent 7: What's the Biggest Opportunity?

Based on everything above, identify the **top 3 improvements** ranked by expected points gained per season:

Categories to consider:
- **Feature ideas**: xA x opp_GC, xG overperformance (goals - xG), progressive carries, set piece taker identification, manager rotation patterns, team strength relative model
- **Model ideas**: LightGBM comparison, model stacking instead of simple blend, position-specific hyperparameters, time-series features (form trajectory)
- **Captain improvements**: Bayesian captain selection, variance-aware captaincy, fixture-specific captain history analysis
- **Transfer improvements**: Price change integration into transfer timing, multi-week transfer chaining (buy A to fund C via selling B)
- **Strategic improvements**: Mini-league differential mode, rank-chasing mode (late season), template team awareness, auto-sub optimisation

### Previously Found Bug Patterns (Check for Recurrence)

These bugs have been found and fixed. Specifically verify they haven't been reintroduced:

| Pattern | Where to Check | What Goes Wrong |
|---------|---------------|-----------------|
| Availability zeroing before 3-GW merge | `ml/prediction.py` | Injured players get non-zero 3-GW predictions |
| Budget using now_cost sum | `season/manager.py` generate_recommendation | Budget ~5% too high |
| Single-GW solver as primary path | `season/manager.py` generate_recommendation | Loses multi-GW intelligence |
| Chip plan not passed to planner | `season/manager.py` _run_strategy_pipeline | Planner ignores chip schedule |
| Solver failure kills planning | `strategy/transfer_planner.py` | One bad GW aborts entire 5-GW plan |
| Hit cost recomputed from sets | `solver/validator.py` | Forced replacements counted as paid hits |
| Watchlist excluded from prices | `season/manager.py` _track_prices | Price alerts miss watched players |
| pandas CoW without .copy() | `features/playerstats.py` | Silent data corruption |

### Output Format

For each finding:
1. **Category**: Bug / Mathematical Issue / FPL Domain Issue / Improvement Opportunity
2. **Severity/Impact**: Critical / High / Medium (for bugs) or Expected Points Impact (for improvements)
3. **File:line** (for code issues)
4. **What's wrong / What could be better**
5. **Evidence** (why you believe this is an issue — cite code, math, or FPL theory)
6. **Suggested fix / approach**

**Final synthesis**: Top 5 highest-impact actions ranked by expected improvement to season performance. Group as: Must Fix (bugs), Should Investigate (mathematical/strategic questions), Could Explore (improvement ideas).

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
lsof -ti:9874 | xargs kill -9

# Baseline
git stash
FLASK_APP=src.api .venv/bin/python -m flask run --port 9874 &
sleep 5
curl -s -X POST http://127.0.0.1:9874/api/backtest -H 'Content-Type: application/json' -d '{"start_gw":10,"end_gw":27}'
# Wait for completion (check /api/status SSE stream)
curl -s http://127.0.0.1:9874/api/backtest-results > /private/tmp/backtest_baseline.json
lsof -ti:9874 | xargs kill -9

# After changes
git stash pop
FLASK_APP=src.api .venv/bin/python -m flask run --port 9874 &
sleep 5
curl -s -X POST http://127.0.0.1:9874/api/backtest -H 'Content-Type: application/json' -d '{"start_gw":10,"end_gw":27}'
curl -s http://127.0.0.1:9874/api/backtest-results > /private/tmp/backtest_postfix.json
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
curl -s http://127.0.0.1:9874/api/backtest-results | python3 -c "
import sys,json; r=json.load(sys.stdin); s=r['summary']
print(f'MAE={s[\"model_avg_mae\"]:.4f} rho={s[\"avg_spearman\"]:.4f} top11={s[\"model_avg_top11_pts\"]:.1f} cap%={s[\"model_capture_pct\"]:.1f}%')
print(f'vs ep: {s[\"model_wins\"]}W-{s[\"ep_wins\"]}L | MAE p={s.get(\"mae_pvalue\",\"?\")}')
"

# Spot-check top predicted players
curl -s http://127.0.0.1:9874/api/predictions | python3 -c "
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
