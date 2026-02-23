# FPL Gaffer — the brAIn

Autonomous Fantasy Premier League manager powered by XGBoost predictions and MILP optimization. Plans transfers, captaincy, and chip strategy across a rolling 5-gameweek horizon.

Built with Python, Flask, scipy MILP, and a single-file vanilla JS frontend.

---

## What It Does

- **Predicts** individual player points using position-specific XGBoost models with 100+ engineered features per player per gameweek
- **Optimizes** squad selection, transfers, and captaincy jointly via mixed-integer linear programming
- **Plans ahead** with a 5-GW rolling transfer planner that considers FT banking, fixture swings, and price movements
- **Evaluates chips** (Wildcard, Bench Boost, Triple Captain, Free Hit) across every remaining GW with DGW/BGW awareness
- **Explains predictions** — see exactly how the model arrives at every player's score, from raw features to final points
- **Tracks your season** — records recommendations vs actual results, rank trajectory, budget, and model accuracy
- **Reacts** to injuries, fixture changes, and prediction shifts with auto-replan detection and alerts
- **Backtests** — walk-forward validation across any GW range so you can see how the model actually performs

---

## Installation

### Windows

1. Download `GafferAI-Windows.zip` from the [latest release](https://github.com/invisi101/FPLGaffer/releases/latest)
2. Extract the zip
3. Run `GafferAI.exe` — the app opens in your browser automatically

No Python installation required.

### macOS

**Option A — Download the app:**

1. Download `GafferAI-macOS.zip` from the [latest release](https://github.com/invisi101/FPLGaffer/releases/latest)
2. Extract the zip
3. Move `GafferAI.app` to your Applications folder
4. Launch from Spotlight, Launchpad, or the Applications folder

No Python installation required.

**Option B — Build from source:**

```bash
git clone https://github.com/invisi101/FPLGaffer.git
cd FPLGaffer
./setup-mac.sh
```

This creates a virtual environment, installs dependencies, and installs **GafferAI.app** to your Applications folder.

### Linux

```bash
git clone https://github.com/invisi101/FPLGaffer.git
cd FPLGaffer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src
```

Open http://127.0.0.1:9876 in your browser.

---

## First Run

1. **Get Latest Data** — downloads player stats, fixtures, and team data
2. **Train Models** — trains XGBoost models for all positions (takes a few minutes)
3. Predictions appear automatically — you're ready to go

Retrain weekly as the season progresses.

---

## Features

### Predictions

Sortable, searchable table of every player's predicted points. Filter by position. Columns include cost, form, predicted next GW and 3 GW points, fixture difficulty, upcoming opponents, captain scores, and prediction intervals.

### Best Team

MILP solver builds the optimal 15-player squad within budget, with joint captain optimization. Displayed on an interactive pitch visualization.

### GW Compare

Pick any past gameweek and see your actual FPL team side-by-side with the highest-scoring possible team for that GW. Dual pitch visualization with overlap highlighting and capture percentage.

### My Team

Import your FPL squad by manager ID. Dual-pitch view showing actual GW points alongside predicted next GW points. Shows squad value, bank, and free transfers.

**Transfer Recommender** — finds optimal transfers using the MILP solver. Set max transfers, choose 1GW or 3GW optimization, optionally enable Wildcard mode.

### Backtest

Walk-forward backtesting across any GW range. Trains fresh models at each step and compares against FPL's own EP metric. Live-streaming results with per-GW detail, position breakdowns, and statistical significance testing.

### "But, How?"

Prediction explainer that shows exactly how the model arrives at every player's score:
- Mean model vs component model breakdown with the ensemble blend formula
- Per-component point contributions (goals, assists, clean sheets, bonus, DefCon)
- Playing probability and 60+ minute probability
- Captain score formula with Q80 upside
- Top feature importance with actual player values

### Season Manager

Track your entire FPL season:

- **Overview** — rank trajectory, points-per-GW, budget evolution, and model accuracy charts
- **Strategy** — 5-GW transfer timeline, captain plan, chip schedule with synergies, and plan changelog
- **Weekly Workflow** — generate recommendations, then record actual results after each gameweek
- **Fixtures** — FDR-colored grid for all 20 teams with DGW/BGW detection
- **Prices** — ownership-based price change predictions with probability scores and history charts
- **Transfer History** — complete log with cost, hits, and recommendation adherence
- **Chips** — tracks usage and estimates remaining chip value

### Monsters

Top 3 players across 8 categories: Goal Monsters, DefCon Monsters, Closet Strikers, Assist Kings, Set Piece Merchants, Captain Monsters, Value Monsters, Clean Sheet Machines.

---

## Architecture

Three-layer system: **Data -> Features/Models -> Strategy/Solver**, backed by SQLite and served via Flask blueprints.

```
src/
├── api/              # 7 Flask blueprints (core, team, season, strategy, prices, backtest, compare)
├── data/             # FPL API + GitHub CSV fetchers with caching
├── features/         # 12 feature modules (rolling, EWM, interactions, opponent, venue, etc.)
├── ml/               # XGBoost training, ensemble prediction, multi-GW, decomposed sub-models, backtest
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
| Models | 4 mean models, 2 quantile (Q80) for captaincy, ~20 decomposed sub-models, 85/15 ensemble |
| Strategy | 5-GW planner, chip evaluator, captain planner, reactive re-planning |
| Solver | scipy MILP with joint captain optimization |
| Frontend | Single HTML file, vanilla JS, dark theme, canvas charts, SSE for live progress |

---

## Testing

```bash
# Run all tests (103 tests)
.venv/bin/python -m pytest tests/ -v

# Fast tests only (74 tests, <2s)
.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v
```

---

## Building from Source

```bash
pip install pyinstaller
pyinstaller gaffer-windows.spec   # Windows
pyinstaller gaffer-mac.spec       # macOS (.app bundle)
```

Releases automatically build Windows and macOS executables via GitHub Actions.

---

## License

MIT
