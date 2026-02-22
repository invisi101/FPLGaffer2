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
│   ├── playerstats.py       # Per-90 stats, element history
│   ├── element_history.py   # FPL element summary features
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
100+ features per player per GW, built by `build_features()` which orchestrates 10 feature modules:
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

### Multi-GW Predictions
- 3-GW: Sum of three 1-GW predictions with correct opponent data per offset, 0.95 decay per GW
- 8-GW horizon: Model predictions for near-term, fixture heuristics for distant GWs
- Confidence decays with distance (0.95 -> 0.77 at GW+5)

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
2. **Generate Recommendation** -> predictions, transfer solver, action plan, strategic plan, stores in DB
3. **Review Action Plan** -> clear steps (transfer X out / Y in, set captain to Z, activate chip)
4. **Make Moves** -> user executes on FPL website
5. **Record Results** -> imports actual picks, compares to recommendation, tracks accuracy

### Key Methods
- `init_season()` — Backfills season history from FPL API (fetches all GW picks, transfers, chips)
- `generate_recommendation()` — Full pipeline: load data -> predict -> solver -> save recommendation + strategic plan
- `get_action_plan()` — Builds human-readable action plan with steps ({action, description}), deadline, rationale
- `record_actual_results()` — Post-GW import + comparison to recommendation
- `get_dashboard()` — Aggregated dashboard data (rank, points, budget, accuracy)

### Price Tracking
- `track_prices()`: Snapshots prices for squad players
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

## Testing

```bash
# Kill leftover server
lsof -ti:9874 | xargs kill -9

# Run all tests
.venv/bin/python -m pytest tests/test_integration.py -v

# Start server
FLASK_APP=src.api .venv/bin/python -m flask run --port 9874
```

### Test Structure (`tests/test_integration.py`)
17 tests covering:
- Flask app creation and route registration
- API endpoints (predictions, model-info, season/status)
- MILP solver (squad, transfers, captain)
- FPL rules validation (squad count, team limits)
- Database schema creation and CRUD
- Feature registry (position-specific features)
- Config loading (all dataclass configs)
- Module imports (all 69 source modules import without error)

### Key test commands
```bash
# Action plan
curl -s "http://127.0.0.1:9874/api/season/action-plan?manager_id=1364335"

# Strategic plan
curl -s "http://127.0.0.1:9874/api/season/strategic-plan?manager_id=1364335"

# Plan health
curl -s "http://127.0.0.1:9874/api/season/plan-health?manager_id=1364335"

# Dashboard
curl -s "http://127.0.0.1:9874/api/season/dashboard?manager_id=1364335"
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
- **Selling prices use `now_cost`**: Public API doesn't provide actual selling prices (50% profit sharing on price rises).
- **Strategic plan is currently single-GW**: The full multi-GW planning pipeline (chip evaluator, transfer planner, captain planner, plan synthesizer) is implemented but not yet wired into `generate_recommendation()`. Current strategic plan shows only the next GW.

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

When the user says **"run full audit"**, execute the following comprehensive audit. Use parallel agents.

### What to check

**1. FPL Rule Compliance** — Trace each rule to the code that handles it.

**2. Data Leakage** — Every feature must only use GW N-1 and earlier data when predicting GW N+1. Check `shift(1)` on all rolling calculations.

**3. Off-by-One Errors** — Verify gameweek offsets, loop bounds, cross-season boundaries.

**4. State Tracking** — Budget preservation, FT tracking, squad updates across simulations.

**5. Hardcoded Values** — Check for magic numbers that should be in `config.py`.

**6. DGW/BGW Handling** — Stats summed (not averaged) for DGW. BGW teams handled gracefully.

**7. Injury/Availability Propagation** — All prediction columns zeroed. Propagates to all future GWs.

**8. Cache Staleness** — Is cached data fresh enough for each operation?

**9. Database Integrity** — No orphaned rows. Correct season_id/manager_id filtering.

**10. MILP Solver Correctness** — Captain bonus, bench weight, all constraints correct.

**11. Frontend/Backend Contract** — Every field the frontend reads is returned by the backend.

### Output format
For each issue: Severity (Critical/High/Medium), File:line, What's wrong, Expected behavior, Suggested fix.

---

## Benchmarking Protocol

When the user says **"run benchmarks"**, execute:

### Step 1: Targeted Tests
Write a script testing each specific change with focused assertions.

### Step 2: Walk-Forward Backtest (Before/After)
```bash
# Baseline
git stash && .venv/bin/python -m flask run --port 9874
# POST /api/backtest {"start_gw":10,"end_gw":27}
# Save results

# After changes
git stash pop && .venv/bin/python -m flask run --port 9874
# Same backtest, compare
```

### Key Metrics
| Metric | Direction | Importance |
|--------|-----------|------------|
| `model_avg_mae` | Lower | Primary |
| `avg_spearman` | Higher | Primary |
| `model_avg_top11_pts` | Higher | High |
| `model_capture_pct` | Higher | High |
| `captain_hit_rate` | Higher | Medium |

### Pass criteria
- MAE must not increase
- Top-11 points must not decrease
- No more than 2 individual GWs should regress

---

## Remaining TODO

### Full Multi-GW Strategic Planning
The strategy modules (`chip_evaluator.py`, `transfer_planner.py`, `captain_planner.py`, `plan_synthesizer.py`) are implemented but not yet wired into `generate_recommendation()`. Currently, the recommendation generates a single-GW plan. To enable full planning:
1. Wire ChipEvaluator to produce chip heatmap
2. Wire MultiWeekPlanner for 5-GW transfer plan
3. Wire CaptainPlanner for pre-planned captaincy
4. Wire PlanSynthesizer to combine everything into the strategic plan

### Authenticated FPL API Access
For autonomous execution:
1. FPL Authentication (email/password -> session cookies)
2. Write API endpoints (execute transfers, set captain, activate chips)
3. Exact selling prices (50% profit sharing on price rises)
4. "Execute All" button with confirmation flow
5. Safety guardrails (deadline awareness, rollback info, dry-run mode)
