# Backtest UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Backtest page to the frontend that lets users run walk-forward backtests, see live per-GW results streaming in, and explore rich diagnostic breakdowns.

**Architecture:** The backtest backend already exists (`POST /api/backtest`, `GET /api/backtest-results`). We need to: (1) add per-GW SSE broadcasting so results stream to the frontend live, (2) add a Backtest button in the action bar + a new tab panel, (3) build the results UI with scoreboard, GW timeline cards, and deep-dive diagnostics. Everything lives in `index.html` (CSS + HTML + JS) and a small change to `backtest_bp.py` + `backtest.py` for SSE streaming.

**Tech Stack:** Flask SSE (existing), vanilla JS, CSS (existing dark theme variables)

---

### Task 1: Add per-GW SSE broadcasting to backtest

The backtest currently broadcasts only "Loading data..." and "Running backtest..." — no per-GW progress. We need a callback so each completed GW is broadcast via SSE, allowing the frontend to show live results.

**Files:**
- Modify: `src/ml/backtest.py` — add `progress_callback` parameter to `run_backtest()` and `_run_season_backtest()`
- Modify: `src/api/backtest_bp.py` — pass a callback that broadcasts per-GW results via SSE

**Step 1: Add callback to `_run_season_backtest()`**

In `src/ml/backtest.py`, modify the function signature at line 309:

```python
def _run_season_backtest(
    df: pd.DataFrame,
    start_gw: int,
    end_gw: int,
    season: str,
    progress_callback=None,
) -> tuple[list[dict], pd.DataFrame]:
```

After `gameweek_results.append(gw_result)` at line 594, add:

```python
        if progress_callback:
            progress_callback(gw_result)
```

**Step 2: Add callback to `run_backtest()`**

Modify the signature at line 929:

```python
def run_backtest(
    df: pd.DataFrame,
    start_gw: int = 5,
    end_gw: int = 25,
    season: str = "",
    seasons: list[str] | None = None,
    progress_callback=None,
) -> dict:
```

Pass it through at line 963:

```python
        gw_results, pooled = _run_season_backtest(df, start_gw, end_gw, s, progress_callback=progress_callback)
```

**Step 3: Update `backtest_bp.py` to broadcast per-GW results**

In `src/api/backtest_bp.py`, modify the `do_backtest()` function inside `api_backtest()`:

```python
    def do_backtest():
        import json as _json

        from src.data.loader import load_all_data
        from src.features.builder import build_features
        from src.ml.backtest import run_backtest

        broadcast("Loading data for backtest...", event="progress")
        data = load_all_data()
        df = build_features(data)

        total_gws = end_gw - start_gw + 1

        def on_gw_complete(gw_result):
            gw = gw_result["gw"]
            idx = gw - start_gw + 1
            broadcast(
                f"GW {gw} complete ({idx}/{total_gws})",
                event="progress",
            )
            broadcast(
                _json.dumps(gw_result),
                event="backtest_gw",
            )

        broadcast("Running backtest...", event="progress")
        results = run_backtest(
            df, start_gw=start_gw, end_gw=end_gw,
            seasons=seasons,
            progress_callback=on_gw_complete,
        )
        sse_module.backtest_results = results
```

**Step 4: Run tests to verify nothing broke**

Run: `.venv/bin/python -m pytest tests/test_correctness.py tests/test_integration.py -v`
Expected: All pass (the callback is optional, defaults to None, so existing code paths are unchanged)

**Step 5: Commit**

```bash
git add src/ml/backtest.py src/api/backtest_bp.py
git commit -m "feat: add per-GW SSE broadcasting to backtest"
```

---

### Task 2: Add Backtest button to action bar and tab

Add the Backtest button with a visual separator in the action bar, a tab entry, and an empty panel placeholder.

**Files:**
- Modify: `src/templates/index.html`

**Step 1: Add button to action bar**

Find this block around line 1619:

```html
  <button class="btn btn-primary" id="btnTrain" onclick="trainModels()">🧠 Train Models</button>
  <div class="status-area">
```

Insert between them:

```html
  <div style="border-left:1px solid var(--border);height:24px;margin:0 8px"></div>
  <button class="btn" id="btnBacktest" onclick="switchTab({dataset:{view:'backtest'},classList:{add:function(){},remove:function(){}}});document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));">🧪 Backtest</button>
```

**Step 2: Add the backtest panel HTML**

After the GW Compare Panel (after line 1710, after `</div>` of gwComparePanel), add:

```html
<!-- Backtest Panel -->
<div class="content backtest-panel" id="backtestPanel">
  <div class="bt-controls">
    <label>From GW:</label>
    <input type="number" class="budget-input" id="btStartGW" value="9" min="2" max="37" style="width:70px">
    <label>To GW:</label>
    <input type="number" class="budget-input" id="btEndGW" value="26" min="3" max="38" style="width:70px">
    <button class="btn btn-primary" id="btnRunBacktest" onclick="runBacktest()">Run Backtest</button>
  </div>

  <!-- Methodology explainer -->
  <div class="bt-explainer" id="btExplainer">
    <div class="bt-explainer-header">
      <h3>How the Backtest Works</h3>
      <p class="bt-explainer-subtitle">A time machine for your prediction model</p>
    </div>
    <div class="bt-explainer-body">
      <div class="bt-explainer-section">
        <div class="bt-explainer-icon">🔬</div>
        <div>
          <strong>Walk-Forward Testing</strong>
          <p>For each past gameweek, we go back in time. The model is retrained from scratch using <em>only data that was available before that gameweek</em> &mdash; no peeking at the future. Then it predicts the upcoming GW, and we compare those predictions against what actually happened. This is repeated for every gameweek in your range.</p>
        </div>
      </div>
      <div class="bt-explainer-section">
        <div class="bt-explainer-icon">🎯</div>
        <div>
          <strong>What We Measure</strong>
          <p><strong>Prediction Accuracy (MAE)</strong> &mdash; Mean Absolute Error. If the model predicts a player will score 5 points and he scores 3, the error is 2. Lower is better. We compare our model's MAE against FPL's own prediction system (ep_next) and two simple baselines.</p>
          <p><strong>Player Ranking (Spearman)</strong> &mdash; How well does the model order players from best to worst? A score of 1.0 would mean perfect ranking. Even if exact point predictions are slightly off, good ranking means you're picking the right players.</p>
          <p><strong>Top-11 Points</strong> &mdash; If you picked the 11 highest-predicted players each week, how many actual points would they have scored? Compared against the hindsight-best 11 (the theoretical maximum).</p>
          <p><strong>Captain Accuracy</strong> &mdash; How often does the model's captain pick finish in the actual top 3 scorers?</p>
        </div>
      </div>
      <div class="bt-explainer-section">
        <div class="bt-explainer-icon">⚡</div>
        <div>
          <strong>How to Read the Results</strong>
          <p>The model is tested against <strong>four rivals</strong>: FPL's built-in prediction (ep_next), a player's recent form, their last-3-gameweek average, and the position average. Every GW is a head-to-head matchup. The scoreboard shows the win/loss record.</p>
          <p>A good model should: beat FPL's ep_next on MAE most weeks, rank players well (Spearman &gt; 0.6), and capture 85%+ of the theoretical best team's points.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Results area (hidden until backtest runs) -->
  <div id="btResults" style="display:none">
    <!-- Scoreboard -->
    <div class="bt-scoreboard" id="btScoreboard"></div>

    <!-- GW Timeline -->
    <div class="bt-timeline-header" id="btTimelineHeader" style="display:none">
      <h3>Gameweek Timeline</h3>
      <div class="bt-progress-bar"><div class="bt-progress-fill" id="btProgressFill"></div></div>
    </div>
    <div class="bt-timeline" id="btTimeline"></div>

    <!-- Deep Dive (appears after completion) -->
    <div id="btDeepDive" style="display:none">
      <div class="bt-section" id="btPositionBreakdown"></div>
      <div class="bt-section" id="btCalibration"></div>
      <div class="bt-section" id="btCaptainAnalysis"></div>
      <div class="bt-section" id="btBiggestMisses"></div>
      <div class="bt-section" id="btThreeGW"></div>
    </div>
  </div>
</div>
```

**Step 3: Add backtest panel to `switchTab()`**

Find the `switchTab` function (around line 2684). Before the closing `}` of the function (around line 2768), add:

```javascript
  const backtestPanel = document.getElementById('backtestPanel');
  if (view === 'backtest') {
    backtestPanel.classList.add('visible');
  } else {
    backtestPanel.classList.remove('visible');
  }
```

**Step 4: Also disable the Backtest button when a task is running**

In the `setTaskRunning` function (around line 2114), find the line:

```javascript
  const btns = [document.getElementById('btnRefresh'), document.getElementById('btnTrain')];
```

Change it to:

```javascript
  const btns = [document.getElementById('btnRefresh'), document.getElementById('btnTrain'), document.getElementById('btnBacktest'), document.getElementById('btnRunBacktest')];
```

**Step 5: Verify the panel shows/hides**

Start the server and verify:
- Backtest button appears in action bar with a divider
- Clicking it shows the backtest panel with the explainer
- Clicking other tabs hides it

**Step 6: Commit**

```bash
git add src/templates/index.html
git commit -m "feat: add backtest button, tab, and methodology explainer"
```

---

### Task 3: Add backtest CSS

Add all the CSS styles needed for the backtest page — controls, explainer, scoreboard, timeline cards, deep-dive sections.

**Files:**
- Modify: `src/templates/index.html` — add CSS rules in the `<style>` block

**Step 1: Add CSS**

Find the closing `</style>` tag (around line 600-ish in the CSS block). Before it, add:

```css
  /* ---- Backtest ---- */
  .backtest-panel { display: none; max-width: 1200px; }
  .backtest-panel.visible { display: block; }

  .bt-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .bt-controls label { font-size: 13px; color: var(--text2); font-weight: 500; }

  /* Explainer */
  .bt-explainer {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
  }
  .bt-explainer-header h3 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
    background: linear-gradient(135deg, var(--text) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .bt-explainer-subtitle {
    font-size: 14px;
    color: var(--text2);
    margin-bottom: 20px;
    font-style: italic;
  }
  .bt-explainer-body { display: flex; flex-direction: column; gap: 18px; }
  .bt-explainer-section {
    display: flex;
    gap: 14px;
    align-items: flex-start;
  }
  .bt-explainer-icon {
    font-size: 24px;
    flex-shrink: 0;
    width: 36px;
    text-align: center;
    padding-top: 2px;
  }
  .bt-explainer-section strong {
    font-size: 14px;
    color: var(--text);
    display: block;
    margin-bottom: 4px;
  }
  .bt-explainer-section p {
    font-size: 13px;
    color: var(--text2);
    line-height: 1.6;
    margin-bottom: 6px;
  }
  .bt-explainer-section em { color: var(--accent2); font-style: normal; font-weight: 500; }

  /* Scoreboard */
  .bt-scoreboard {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    align-items: center;
  }
  .bt-score-side {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .bt-score-side.right { text-align: right; }
  .bt-score-team {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }
  .bt-score-team.model { color: var(--green); }
  .bt-score-team.ep { color: var(--text2); }
  .bt-score-record {
    font-family: 'Outfit', monospace;
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
  }
  .bt-score-record.win { color: var(--green); }
  .bt-score-record.lose { color: var(--text2); }
  .bt-score-vs {
    font-size: 16px;
    color: var(--text2);
    font-weight: 600;
    padding: 0 24px;
    text-align: center;
  }
  .bt-score-stats {
    display: flex;
    gap: 24px;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    grid-column: 1 / -1;
    justify-content: center;
    flex-wrap: wrap;
  }
  .bt-stat-card {
    text-align: center;
    padding: 12px 18px;
    background: var(--surface2);
    border-radius: 8px;
    min-width: 110px;
  }
  .bt-stat-val {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
  }
  .bt-stat-val.good { color: var(--green); }
  .bt-stat-val.neutral { color: var(--text); }
  .bt-stat-val.bad { color: var(--red); }
  .bt-stat-label {
    font-size: 11px;
    color: var(--text2);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .bt-stat-sub {
    font-size: 11px;
    color: var(--text2);
    margin-top: 2px;
  }
  .bt-verdict {
    grid-column: 1 / -1;
    text-align: center;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 12px;
  }
  .bt-verdict.positive { background: rgba(45, 212, 160, 0.1); color: var(--green); border: 1px solid rgba(45, 212, 160, 0.2); }
  .bt-verdict.negative { background: rgba(248, 113, 113, 0.1); color: var(--red); border: 1px solid rgba(248, 113, 113, 0.2); }
  .bt-verdict.neutral { background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }

  /* Progress bar */
  .bt-timeline-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
  }
  .bt-timeline-header h3 { font-size: 16px; font-weight: 600; white-space: nowrap; }
  .bt-progress-bar {
    flex: 1;
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
  }
  .bt-progress-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--green));
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  /* GW Timeline cards */
  .bt-timeline {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    margin-bottom: 24px;
  }
  .bt-gw-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    transition: transform 0.2s, border-color 0.2s;
    animation: btCardIn 0.35s ease-out;
  }
  .bt-gw-card:hover { transform: translateY(-2px); }
  .bt-gw-card.win { border-left: 3px solid var(--green); }
  .bt-gw-card.lose { border-left: 3px solid var(--red); }
  .bt-gw-card.tie { border-left: 3px solid var(--yellow); }
  @keyframes btCardIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .bt-gw-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .bt-gw-num {
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
  }
  .bt-gw-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .bt-gw-badge.win { background: rgba(45, 212, 160, 0.15); color: var(--green); }
  .bt-gw-badge.lose { background: rgba(248, 113, 113, 0.15); color: var(--red); }
  .bt-gw-badge.tie { background: rgba(251, 191, 36, 0.15); color: var(--yellow); }
  .bt-gw-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text2);
    padding: 2px 0;
  }
  .bt-gw-row .val { color: var(--text); font-weight: 500; }
  .bt-gw-captain {
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text2);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .bt-gw-captain .captain-pts {
    font-weight: 600;
  }
  .bt-gw-captain .captain-pts.hit { color: var(--green); }
  .bt-gw-captain .captain-pts.miss { color: var(--text2); }

  /* Deep dive sections */
  .bt-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
  }
  .bt-section h3 {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 16px;
    color: var(--text);
  }
  .bt-section-subtitle {
    font-size: 13px;
    color: var(--text2);
    margin-top: -12px;
    margin-bottom: 16px;
  }

  /* Position breakdown table */
  .bt-pos-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .bt-pos-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  .bt-pos-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }
  .bt-pos-table tr:last-child td { border-bottom: none; }
  .bt-better { color: var(--green); font-weight: 600; }
  .bt-worse { color: var(--red); font-weight: 600; }

  /* Calibration bars */
  .bt-cal-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .bt-cal-label { width: 50px; color: var(--text2); text-align: right; font-weight: 500; }
  .bt-cal-bars { flex: 1; position: relative; height: 28px; }
  .bt-cal-bar {
    position: absolute;
    top: 0;
    height: 12px;
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .bt-cal-bar.predicted { background: var(--accent); opacity: 0.7; }
  .bt-cal-bar.actual { top: 14px; background: var(--green); opacity: 0.7; }
  .bt-cal-values { width: 140px; font-size: 11px; color: var(--text2); }

  /* Misses table */
  .bt-miss-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .bt-miss-col h4 {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .bt-miss-col.over h4 { color: var(--red); }
  .bt-miss-col.under h4 { color: var(--yellow); }
  .bt-miss-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .bt-miss-item:last-child { border-bottom: none; }
  .bt-miss-name { color: var(--text); font-weight: 500; }
  .bt-miss-gw { color: var(--text2); font-size: 11px; margin-left: 6px; }
  .bt-miss-nums { display: flex; gap: 10px; font-size: 12px; }
  .bt-miss-pred { color: var(--accent2); }
  .bt-miss-actual { color: var(--text); }
```

**Step 2: Commit**

```bash
git add src/templates/index.html
git commit -m "feat: add backtest page CSS styles"
```

---

### Task 4: Implement the backtest JavaScript

Add all the JS functions: `runBacktest()`, SSE listener for `backtest_gw` events, scoreboard rendering, GW card rendering, and deep-dive rendering.

**Files:**
- Modify: `src/templates/index.html` — add JS functions

**Step 1: Add backtest JavaScript**

Find the last JS section (before `</script>`, which is near the end of the file). Add the following JS block:

```javascript
// ---------------------------------------------------------------------------
// Backtest
// ---------------------------------------------------------------------------
let backtestGWResults = [];
let backtestTotalGWs = 0;
let backtestStartGW = 0;

function runBacktest() {
  const startGW = parseInt(document.getElementById('btStartGW').value) || 9;
  const endGW = parseInt(document.getElementById('btEndGW').value) || 26;

  if (startGW >= endGW || startGW < 2 || endGW > 38) {
    alert('Invalid GW range. Start must be >= 2, End <= 38, and Start < End.');
    return;
  }

  backtestGWResults = [];
  backtestTotalGWs = endGW - startGW + 1;
  backtestStartGW = startGW;

  // Show results area, hide explainer
  document.getElementById('btResults').style.display = 'block';
  document.getElementById('btExplainer').style.display = 'none';
  document.getElementById('btScoreboard').innerHTML = '<div class="empty-state"><p>Starting backtest...</p></div>';
  document.getElementById('btTimeline').innerHTML = '';
  document.getElementById('btTimelineHeader').style.display = 'flex';
  document.getElementById('btProgressFill').style.width = '0%';
  document.getElementById('btDeepDive').style.display = 'none';

  fetch('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start_gw: startGW, end_gw: endGW }),
  }).then(r => r.json()).then(d => {
    if (d.error) {
      document.getElementById('btScoreboard').innerHTML = friendlyError(d.error, 'Backtest');
      return;
    }
    // Backtest started — results will stream via SSE
  }).catch(err => {
    document.getElementById('btScoreboard').innerHTML = friendlyError(err.message, 'Backtest');
  });
}

function handleBacktestGW(gwData) {
  backtestGWResults.push(gwData);
  const pct = Math.round((backtestGWResults.length / backtestTotalGWs) * 100);
  document.getElementById('btProgressFill').style.width = pct + '%';

  // Render the live scoreboard from accumulated results
  renderBTScoreboardLive();

  // Add the GW card
  renderBTGWCard(gwData);
}

function handleBacktestDone() {
  // Fetch the full results for deep dive
  fetch('/api/backtest-results').then(r => r.json()).then(d => {
    if (d.error) return;
    renderBTScoreboard(d.summary);
    renderBTDeepDive(d);
    // Show explainer again as collapsed reference
    document.getElementById('btExplainer').style.display = '';
  });
}

function renderBTScoreboardLive() {
  const results = backtestGWResults;
  const n = results.length;
  if (n === 0) return;

  const wins = results.filter(r => r.winner === 'MODEL').length;
  const losses = results.filter(r => r.winner === 'ep_next').length;
  const ties = results.filter(r => r.winner === 'TIE').length;
  const avgMAE = (results.reduce((s, r) => s + r.model_mae, 0) / n).toFixed(3);
  const avgEpMAE = (results.reduce((s, r) => s + r.ep_mae, 0) / n).toFixed(3);
  const avgSpearman = (results.reduce((s, r) => s + r.spearman_rho, 0) / n).toFixed(3);
  const avgTop11 = (results.reduce((s, r) => s + r.model_top11_pts, 0) / n).toFixed(1);
  const avgActual = (results.reduce((s, r) => s + r.actual_best_pts, 0) / n).toFixed(1);
  const capHits = results.filter(r => r.captain_in_top3).length;
  const capRate = Math.round((capHits / n) * 100);

  const maeClass = parseFloat(avgMAE) < parseFloat(avgEpMAE) ? 'good' : 'bad';
  const capture = avgActual > 0 ? ((avgTop11 / avgActual) * 100).toFixed(1) : '0';

  document.getElementById('btScoreboard').innerHTML = `
    <div class="bt-score-side">
      <div class="bt-score-team model">Gaffer AI</div>
      <div class="bt-score-record ${wins >= losses ? 'win' : 'lose'}">${wins}</div>
    </div>
    <div class="bt-score-vs">
      <div style="font-size:12px;color:var(--text2);margin-bottom:4px">${n} GWs tested</div>
      vs
      <div style="font-size:11px;color:var(--text2);margin-top:4px">${ties > 0 ? ties + ' tie' + (ties > 1 ? 's' : '') : ''}</div>
    </div>
    <div class="bt-score-side right">
      <div class="bt-score-team ep">FPL Predicted (ep)</div>
      <div class="bt-score-record ${losses > wins ? 'win' : 'lose'}">${losses}</div>
    </div>
    <div class="bt-score-stats">
      <div class="bt-stat-card">
        <div class="bt-stat-val ${maeClass}">${avgMAE}</div>
        <div class="bt-stat-label">MAE</div>
        <div class="bt-stat-sub">vs ${avgEpMAE} ep</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val ${parseFloat(avgSpearman) > 0.6 ? 'good' : 'neutral'}">${avgSpearman}</div>
        <div class="bt-stat-label">Ranking</div>
        <div class="bt-stat-sub">Spearman rho</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val neutral">${avgTop11}</div>
        <div class="bt-stat-label">Top-11 Pts</div>
        <div class="bt-stat-sub">${capture}% of best</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val ${capRate >= 30 ? 'good' : 'neutral'}">${capRate}%</div>
        <div class="bt-stat-label">Captain</div>
        <div class="bt-stat-sub">in actual top 3</div>
      </div>
    </div>
  `;
}

function renderBTGWCard(gw) {
  const timeline = document.getElementById('btTimeline');
  const winClass = gw.winner === 'MODEL' ? 'win' : gw.winner === 'ep_next' ? 'lose' : 'tie';
  const badgeText = gw.winner === 'MODEL' ? 'Win' : gw.winner === 'ep_next' ? 'Loss' : 'Tie';
  const capClass = gw.captain_in_top3 ? 'hit' : 'miss';

  const card = document.createElement('div');
  card.className = `bt-gw-card ${winClass}`;
  card.innerHTML = `
    <div class="bt-gw-header">
      <span class="bt-gw-num">GW ${gw.gw}</span>
      <span class="bt-gw-badge ${winClass}">${badgeText}</span>
    </div>
    <div class="bt-gw-row"><span>MAE</span><span class="val">${gw.model_mae.toFixed(3)} <span style="color:var(--text2);font-weight:400">vs ${gw.ep_mae.toFixed(3)}</span></span></div>
    <div class="bt-gw-row"><span>Ranking</span><span class="val">${gw.spearman_rho.toFixed(3)}</span></div>
    <div class="bt-gw-row"><span>Top-11</span><span class="val">${gw.model_top11_pts.toFixed(0)} / ${gw.actual_best_pts.toFixed(0)}</span></div>
    <div class="bt-gw-captain">
      <span>C: ${gw.captain_name}</span>
      <span class="captain-pts ${capClass}">${gw.captain_actual.toFixed(0)} pts</span>
    </div>
  `;
  timeline.appendChild(card);
}

function renderBTScoreboard(s) {
  // Final scoreboard with full stats (replaces live version)
  const maeClass = s.model_avg_mae < s.ep_avg_mae ? 'good' : 'bad';
  const capture = s.model_capture_pct;
  const capRate = Math.round(s.captain_hit_rate * 100);
  const pval = s.mae_pvalue != null ? s.mae_pvalue : 1;
  const sigText = pval < 0.01 ? 'Highly significant (p<0.01)' : pval < 0.05 ? 'Significant (p<0.05)' : pval < 0.1 ? 'Marginally significant' : 'Not statistically significant';
  const sigClass = pval < 0.05 ? 'good' : 'neutral';

  let verdictClass, verdictText;
  if (s.model_wins > s.ep_wins && pval < 0.05) {
    verdictClass = 'positive';
    verdictText = `Gaffer AI outperforms FPL predictions with statistical significance. ${sigText} (p=${pval}).`;
  } else if (s.model_wins > s.ep_wins) {
    verdictClass = 'positive';
    verdictText = `Gaffer AI wins more gameweeks but the margin is not yet statistically conclusive. ${sigText}.`;
  } else if (s.model_wins < s.ep_wins) {
    verdictClass = 'negative';
    verdictText = `FPL predictions outperformed this run. Consider retraining or expanding the test range.`;
  } else {
    verdictClass = 'neutral';
    verdictText = `Dead even — neither model has a clear edge over this range.`;
  }

  document.getElementById('btScoreboard').innerHTML = `
    <div class="bt-score-side">
      <div class="bt-score-team model">Gaffer AI</div>
      <div class="bt-score-record ${s.model_wins >= s.ep_wins ? 'win' : 'lose'}">${s.model_wins}</div>
    </div>
    <div class="bt-score-vs">
      <div style="font-size:12px;color:var(--text2);margin-bottom:4px">${s.n_gameweeks} GWs tested</div>
      vs
      <div style="font-size:11px;color:var(--text2);margin-top:4px">${s.ties > 0 ? s.ties + ' tie' + (s.ties > 1 ? 's' : '') : ''}</div>
    </div>
    <div class="bt-score-side right">
      <div class="bt-score-team ep">FPL Predicted (ep)</div>
      <div class="bt-score-record ${s.ep_wins > s.model_wins ? 'win' : 'lose'}">${s.ep_wins}</div>
    </div>
    <div class="bt-score-stats">
      <div class="bt-stat-card">
        <div class="bt-stat-val ${maeClass}">${s.model_avg_mae}</div>
        <div class="bt-stat-label">MAE</div>
        <div class="bt-stat-sub">vs ${s.ep_avg_mae} ep</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val ${parseFloat(s.avg_spearman) > 0.6 ? 'good' : 'neutral'}">${s.avg_spearman}</div>
        <div class="bt-stat-label">Ranking</div>
        <div class="bt-stat-sub">vs ${s.avg_ep_spearman} ep</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val neutral">${s.model_avg_top11_pts}</div>
        <div class="bt-stat-label">Top-11 Pts/GW</div>
        <div class="bt-stat-sub">${capture}% of best</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val ${capRate >= 30 ? 'good' : 'neutral'}">${capRate}%</div>
        <div class="bt-stat-label">Captain</div>
        <div class="bt-stat-sub">in actual top 3</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-val ${sigClass}">${pval != null ? 'p=' + pval : 'n/a'}</div>
        <div class="bt-stat-label">Significance</div>
        <div class="bt-stat-sub">Wilcoxon test</div>
      </div>
    </div>
    <div class="bt-verdict ${verdictClass}">${verdictText}</div>
  `;
}

function renderBTDeepDive(data) {
  document.getElementById('btDeepDive').style.display = 'block';

  // --- Position Breakdown ---
  const pos = data.by_position || {};
  const posOrder = ['GKP', 'DEF', 'MID', 'FWD'];
  let posHtml = '<h3>Accuracy by Position</h3>';
  posHtml += '<p class="bt-section-subtitle">MAE (Mean Absolute Error) broken down by position — lower is better</p>';
  posHtml += '<table class="bt-pos-table"><thead><tr><th>Position</th><th>Model MAE</th><th>FPL ep MAE</th><th>Form MAE</th><th>Last 3 MAE</th><th>Avg Players</th><th>Verdict</th></tr></thead><tbody>';
  for (const p of posOrder) {
    if (!pos[p]) continue;
    const d = pos[p];
    const better = d.model_mae < d.ep_mae;
    const diff = (d.ep_mae - d.model_mae).toFixed(3);
    posHtml += `<tr>
      <td><span class="pos-tag ${p}">${p}</span></td>
      <td class="${better ? 'bt-better' : 'bt-worse'}">${d.model_mae}</td>
      <td>${d.ep_mae}</td>
      <td>${d.form_mae}</td>
      <td>${d.last3_mae}</td>
      <td style="color:var(--text2)">${d.avg_players}</td>
      <td class="${better ? 'bt-better' : 'bt-worse'}">${better ? '+' + diff + ' better' : diff + ' worse'}</td>
    </tr>`;
  }
  posHtml += '</tbody></table>';
  document.getElementById('btPositionBreakdown').innerHTML = posHtml;

  // --- Calibration ---
  const cal = (data.diagnostics || {}).calibration || [];
  if (cal.length > 0) {
    const maxVal = Math.max(...cal.map(c => Math.max(c.predicted_avg, c.actual_avg)), 1);
    let calHtml = '<h3>Calibration</h3>';
    calHtml += '<p class="bt-section-subtitle">Does the model predict the right number of points? Predicted (purple) vs Actual (green) by prediction range</p>';
    for (const bucket of cal) {
      const predPct = (bucket.predicted_avg / maxVal) * 100;
      const actPct = (bucket.actual_avg / maxVal) * 100;
      const delta = (bucket.predicted_avg - bucket.actual_avg).toFixed(2);
      const dir = parseFloat(delta) > 0.1 ? 'over' : parseFloat(delta) < -0.1 ? 'under' : 'ok';
      const dirColor = dir === 'over' ? 'color:var(--red)' : dir === 'under' ? 'color:var(--yellow)' : 'color:var(--green)';
      calHtml += `<div class="bt-cal-row">
        <div class="bt-cal-label">${bucket.bin}</div>
        <div class="bt-cal-bars">
          <div class="bt-cal-bar predicted" style="width:${predPct}%"></div>
          <div class="bt-cal-bar actual" style="width:${actPct}%"></div>
        </div>
        <div class="bt-cal-values">pred ${bucket.predicted_avg.toFixed(2)} / actual ${bucket.actual_avg.toFixed(2)} <span style="${dirColor};font-weight:600">${dir === 'over' ? '▲' : dir === 'under' ? '▼' : '✓'}</span> (n=${bucket.count})</div>
      </div>`;
    }
    document.getElementById('btCalibration').innerHTML = calHtml;
  }

  // --- Captain Analysis ---
  const cap = (data.diagnostics || {}).captain_analysis || {};
  if (cap.avg_captain_pts != null) {
    let capHtml = '<h3>Captain Analysis</h3>';
    capHtml += '<p class="bt-section-subtitle">How good is the model at picking the right captain each week?</p>';
    capHtml += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px">';
    capHtml += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${cap.avg_captain_pts}</div><div class="bt-stat-label">Avg Captain Pts</div></div>`;
    capHtml += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${cap.avg_best_captain_pts}</div><div class="bt-stat-label">Best Possible</div></div>`;
    capHtml += `<div class="bt-stat-card"><div class="bt-stat-val ${cap.captain_pts_lost <= 2 ? 'good' : 'bad'}">${cap.captain_pts_lost}</div><div class="bt-stat-label">Pts Lost / GW</div></div>`;
    if (cap.ep_avg_captain_pts != null) {
      capHtml += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${cap.ep_avg_captain_pts}</div><div class="bt-stat-label">FPL ep Captain</div></div>`;
    }
    capHtml += `<div class="bt-stat-card"><div class="bt-stat-val ${cap.captain_in_top3_pct >= 30 ? 'good' : 'neutral'}">${cap.captain_in_top3_pct}%</div><div class="bt-stat-label">In Top 3</div></div>`;
    capHtml += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${cap.captain_in_top10_pct}%</div><div class="bt-stat-label">In Top 10</div></div>`;
    capHtml += '</div>';

    if (cap.worst_captain_gws && cap.worst_captain_gws.length > 0) {
      capHtml += '<div style="font-size:13px;font-weight:600;margin-bottom:8px;color:var(--text2)">Biggest Captain Misses</div>';
      for (const w of cap.worst_captain_gws.slice(0, 5)) {
        const lost = (w.best_pts - w.pts).toFixed(0);
        capHtml += `<div class="bt-miss-item"><span><span class="bt-miss-name">${w.captain}</span><span class="bt-miss-gw">GW${w.gw}</span> scored ${w.pts} pts</span><span style="color:var(--red)">missed ${w.best} (${w.best_pts} pts, -${lost})</span></div>`;
      }
    }
    document.getElementById('btCaptainAnalysis').innerHTML = capHtml;
  }

  // --- Biggest Misses ---
  const misses = (data.diagnostics || {}).biggest_misses || {};
  const over = misses.overpredicted || [];
  const under = misses.underpredicted || [];
  if (over.length > 0 || under.length > 0) {
    let missHtml = '<h3>Biggest Prediction Misses</h3>';
    missHtml += '<p class="bt-section-subtitle">The model\'s worst individual predictions — overpredicted (expected points that didn\'t arrive) and underpredicted (surprise hauls)</p>';
    missHtml += '<div class="bt-miss-grid">';

    missHtml += '<div class="bt-miss-col over"><h4>Overpredicted</h4>';
    for (const m of over.slice(0, 8)) {
      missHtml += `<div class="bt-miss-item"><span><span class="bt-miss-name">${m.web_name}</span><span class="bt-miss-gw">GW${m.gw}</span></span><span class="bt-miss-nums"><span class="bt-miss-pred">${m.predicted} pred</span><span class="bt-miss-actual">${m.actual} actual</span></span></div>`;
    }
    missHtml += '</div>';

    missHtml += '<div class="bt-miss-col under"><h4>Underpredicted</h4>';
    for (const m of under.slice(0, 8)) {
      missHtml += `<div class="bt-miss-item"><span><span class="bt-miss-name">${m.web_name}</span><span class="bt-miss-gw">GW${m.gw}</span></span><span class="bt-miss-nums"><span class="bt-miss-pred">${m.predicted} pred</span><span class="bt-miss-actual">${m.actual} actual</span></span></div>`;
    }
    missHtml += '</div></div>';
    document.getElementById('btBiggestMisses').innerHTML = missHtml;
  }

  // --- 3-GW Rolling ---
  const gw3 = data.backtest_3gw;
  if (gw3 && gw3.n_windows > 0) {
    let html3 = '<h3>3-Gameweek Rolling Backtest</h3>';
    html3 += '<p class="bt-section-subtitle">How accurately does the model predict over 3-week windows? Important for transfer planning and bench boost timing.</p>';
    html3 += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px">';
    html3 += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${gw3.avg_mae_3gw}</div><div class="bt-stat-label">3-GW MAE</div></div>`;
    html3 += `<div class="bt-stat-card"><div class="bt-stat-val ${gw3.avg_spearman_3gw > 0.7 ? 'good' : 'neutral'}">${gw3.avg_spearman_3gw}</div><div class="bt-stat-label">3-GW Ranking</div></div>`;
    html3 += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${gw3.avg_model_top11_pts}</div><div class="bt-stat-label">Avg Top-11 Pts</div></div>`;
    html3 += `<div class="bt-stat-card"><div class="bt-stat-val neutral">${gw3.capture_pct_3gw}%</div><div class="bt-stat-label">Capture Rate</div></div>`;
    html3 += '</div>';
    document.getElementById('btThreeGW').innerHTML = html3;
  }
}
```

**Step 2: Add `backtest_gw` event handling to SSE listener**

In the `connectSSE()` function, find the `else if (evt === 'progress')` block (around line 2103). After it, add:

```javascript
    else if (evt === 'backtest_gw') {
      try {
        const gwData = JSON.parse(msg);
        if (gwData && gwData.gw) handleBacktestGW(gwData);
      } catch(e) { /* not JSON, ignore */ }
    }
```

**Step 3: Add backtest completion hook to `task_done` handler**

In the SSE handler, inside the `if (evt === 'task_done')` block (around line 2093), within the `setTimeout` callback, add:

```javascript
          if (backtestGWResults.length > 0) handleBacktestDone();
```

**Step 4: Verify end-to-end**

- Start server
- Click Backtest, set GW 9-26, click Run Backtest
- Verify progress bar fills, GW cards stream in, scoreboard updates live
- Verify deep dive sections appear when complete

**Step 5: Commit**

```bash
git add src/templates/index.html
git commit -m "feat: add backtest JavaScript — live results, scoreboard, deep dive"
```

---

### Task 5: Run all tests and final polish

**Files:**
- All modified files from previous tasks

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All 103 tests pass

**Step 2: Manual end-to-end test**

```bash
lsof -ti:9876 | xargs kill -9
FLASK_APP=src.api .venv/bin/python -m flask run --port 9876
```

Test checklist:
- [ ] Backtest button visible in action bar with divider gap
- [ ] Clicking Backtest shows the panel with methodology explainer
- [ ] Switching to other tabs hides backtest panel
- [ ] Run Backtest with GW 9-26 starts the task
- [ ] Progress bar fills as GWs complete
- [ ] GW cards animate in one by one
- [ ] Scoreboard updates live with running stats
- [ ] After completion, deep dive sections appear (position breakdown, calibration, captain, misses, 3-GW)
- [ ] Clicking other tabs and back preserves results
- [ ] Buttons disabled during backtest run

**Step 3: Commit any polish fixes**

```bash
git add -A
git commit -m "feat: backtest UI — final polish and verification"
```
