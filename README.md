# FPL Gaffer - the brAIn v2

Autonomous Fantasy Premier League manager powered by XGBoost predictions and MILP optimization. Plans transfers, captaincy, and chip strategy across a rolling 5-gameweek horizon.

Complete modular rebuild of [FPL Gaffer v1](https://github.com/invisi101/FPLGaffer) — same brain, cleaner architecture.

Built with Python, Flask, scipy MILP, and a single-file vanilla JS frontend.

---

## What It Does

- **Predicts** individual player points using position-specific XGBoost models (mean + quantile + decomposed sub-models), with 100+ engineered features per player per gameweek
- **Optimizes** squad selection, transfers, and captaincy jointly via mixed-integer linear programming
- **Plans ahead** with a 5-GW rolling transfer planner that considers FT banking, fixture swings, and price movements
- **Evaluates chips** (Wildcard, Bench Boost, Triple Captain, Free Hit) across every remaining GW with DGW/BGW awareness and synergy detection
- **Tracks your season** — records recommendations vs actual results, rank trajectory, budget evolution, and model accuracy over time
- **Reacts** to injuries, fixture changes, and prediction shifts with auto-replan detection and alerts
- **Validates** every recommendation against FPL rules using Pydantic schemas before it reaches you

---

## What's New in v2

- **Modular architecture** — 54 focused files instead of 10 monoliths. No file exceeds 1,700 lines.
- **Flask blueprints** — 7 domain-specific blueprints replace a 2,800-line `app.py`
- **Central config** — every hyperparameter, threshold, and weight in one file (`src/config.py`)
- **FPL rule validation** — Pydantic schemas validate every solver output (squad, formation, transfers, chips)
- **Proper DB migrations** — versioned migrations replace try/except ALTER TABLE
- **99 automated tests** — mathematical correctness, integration smoke tests, and full strategy pipeline coverage
- **Python logging** — structured logging replaces all print() statements

---

## Installation

### Windows

1. Download `GafferAI-Windows.zip` from the [latest release](https://github.com/invisi101/FPLGaffer2/releases/latest)
2. Extract the zip
3. Run `GafferAI.exe` — the app opens in your browser automatically

No Python installation required.

### macOS

**Option A — Download the app:**

1. Download `GafferAI-macOS.zip` from the [latest release](https://github.com/invisi101/FPLGaffer2/releases/latest)
2. Extract the zip
3. Move `GafferAI.app` to your Applications folder
4. Launch from Spotlight, Launchpad, or the Applications folder — the app opens in your browser automatically

No Python installation required.

**Option B — Build from source:**

```bash
git clone https://github.com/invisi101/FPLGaffer2.git
cd FPLGaffer2
./setup-mac.sh
```

This creates a virtual environment, installs dependencies, and installs **GafferAI.app** to your Applications folder.

### Linux

```bash
git clone https://github.com/invisi101/FPLGaffer2.git
cd FPLGaffer2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src
```

Open http://127.0.0.1:9876 in your browser.

---

## First Run

1. **Get Latest Data** — downloads player stats, fixtures, and team data from the FPL API and GitHub historical CSVs
2. **Train Models** — trains XGBoost models for all positions (takes a few minutes)
3. Predictions appear automatically — you're ready to go

Retrain periodically as the season progresses.

---

## Features

### Predictions

Sortable, searchable table of every player's predicted points. Filter by position. Columns include cost, form, predicted next GW and 3 GW points, fixture difficulty, upcoming opponents, captain scores, and prediction intervals.

### Best Team

MILP solver builds the optimal 15-player squad within budget, with joint captain optimization. Displayed on an interactive pitch visualization.

### GW Compare

Pick any past gameweek and see your actual FPL team side-by-side with the highest-scoring possible team for that GW. Shows dual pitch visualization with overlap highlighting, capture percentage, and bench for both sides.

### My Team

Import your FPL squad by manager ID. Dual-pitch view showing actual GW points (with captain multiplier) alongside predicted next GW points. Shows squad value, bank, and free transfers.

**Transfer Recommender** — finds optimal transfers using the MILP solver. Set max transfers, choose 1GW or 3GW optimization, optionally enable Wildcard mode. Shows each transfer with predicted points gained, hit cost, and net gain.

### Season Manager

Track your entire FPL season from any gameweek:

- **Overview** — rank trajectory, points-per-GW, budget evolution, and model accuracy charts
- **Strategy** — 5-GW transfer timeline with captain plan, chip schedule with synergy annotations, chip heatmap, and plan changelog. Auto-replan alerts when injuries or fixture changes invalidate the current plan
- **Weekly Workflow** — generate transfer/captain/chip recommendations, then record actual results after the gameweek
- **Fixtures** — FDR-colored grid for all 20 teams with DGW/BGW detection
- **Prices** — ownership-based price change predictions with probability scores and price history charts
- **Transfer History** — complete log with cost, hits, and recommendation adherence
- **Chips** — tracks usage and estimates remaining chip value

### Monsters

Top 3 players across 8 categories: Goal Monsters, DefCon Monsters, Closet Strikers, Assist Kings, Set Piece Merchants, Captain Monsters, Value Monsters, Clean Sheet Machines.

---

## Architecture

Modular three-layer system: **Data -> Features/Models -> Strategy/Solver**, backed by SQLite and served via Flask blueprints.

```
src/
├── api/              # 7 Flask blueprints (core, team, season, strategy, prices, backtest, compare)
├── data/             # FPL API + GitHub CSV fetchers with caching
├── features/         # 12 feature modules (rolling, EWM, interactions, opponent, venue, etc.)
├── ml/               # XGBoost training, ensemble prediction, multi-GW, decomposed sub-models
├── solver/           # MILP squad + transfer solvers with FPL rule validation
├── strategy/         # ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer
├── season/           # Season orchestration, recording, dashboard, fixtures
├── schemas/          # Pydantic validation for FPL rules
├── db/               # SQLite with versioned migrations
├── config.py         # Every hyperparameter in one place
└── templates/
    └── index.html    # Single-file frontend (vanilla JS, dark theme, SSE)
```

| Layer | What It Does |
|-------|-------------|
| Data | FPL API + GitHub CSVs, cached (30m / 6h), 100+ features per player per GW |
| Models | 4 position-specific mean models, 2 quantile (Q80) for captaincy, ~20 decomposed sub-models |
| Strategy | ChipEvaluator, 5-GW MultiWeekPlanner, CaptainPlanner, PlanSynthesizer, reactive re-planning |
| Solver | scipy MILP with joint captain optimization (3n decision variables) |
| Validation | Pydantic schemas enforce FPL rules on every solver output |
| Storage | SQLite with versioned migrations for season tracking, plans, outcomes, price history |
| Frontend | Single HTML file, vanilla JS, dark theme, canvas charts, SSE for live progress |

---

## Testing

```bash
# Run all tests (99 tests)
.venv/bin/python -m pytest tests/ -v

# Fast tests only (70 tests, <2s)
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v
```

| File | Tests | Coverage |
|------|-------|----------|
| `test_correctness.py` | 53 | FPL scoring formulas, ensemble blend math, solver FPL compliance |
| `test_integration.py` | 17 | Flask app, API routes, solver smoke tests, DB, config |
| `test_strategy_pipeline.py` | 29 | Chip evaluation, transfer planning, captain planning, E2E strategy |

---

## CLI

```bash
.venv/bin/python -m src                              # start the server
```

The server runs at http://127.0.0.1:9876.

---

## Building from Source

### PyInstaller (Windows/macOS executables)

```bash
pip install pyinstaller
pyinstaller gaffer-windows.spec   # Windows
pyinstaller gaffer-mac.spec       # macOS (.app bundle)
```

### GitHub Actions

Releases automatically build Windows and macOS executables. Create a release on GitHub and the workflow attaches `GafferAI-Windows.zip` and `GafferAI-macOS.zip`.

```bash
gh release create v1.0.0 --title "v1.0.0" --notes "Initial release of FPL Gaffer v2"
```

---

## License

MIT
