# Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all 5 improvements identified by the full audit, estimated 33-63 pts/season combined.

**Architecture:** Config-driven changes (bench weight, max hits), model training changes (early stopping), strategy layer enhancements (TC DGW bias, price timing, end-of-season mode). All changes are additive — no existing logic is removed.

**Tech Stack:** Python, XGBoost, scipy MILP, pandas, pytest

---

### Task 1: Model Calibration — Early Stopping for XGBoost

**Files:**
- Modify: `src/config.py:12` (XGBConfig class)
- Modify: `src/ml/training.py:193-197` (walk-forward CV fit), `src/ml/training.py:258-262` (final model fit), `src/ml/training.py:244-252` (holdout fit)
- Test: `tests/test_correctness.py`

**Step 1: Add early_stopping_rounds to XGBConfig**

In `src/config.py`, add a field to XGBConfig:

```python
@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    verbosity: int = 0
    early_stopping_rounds: int = 20  # NEW: stop if val loss doesn't improve
```

**Step 2: Add early stopping to walk-forward CV in train_model**

In `src/ml/training.py`, modify the walk-forward CV loop (lines 186-208). Split each fold's training data into train/val (last 20% of train as val), pass `eval_set` and `early_stopping_rounds` to `model.fit()`:

```python
    for train_mask, test_mask in all_splits[-xgb.walk_forward_splits:]:
        X_train = pos_df.loc[train_mask, available_feats].values
        y_train = pos_df.loc[train_mask, target].values
        w_train = pos_df.loc[train_mask, "_sample_weight"].values
        X_test = pos_df.loc[test_mask, available_feats].values
        y_test = pos_df.loc[test_mask, target].values

        # Early stopping: use last 20% of training fold as validation
        val_size = max(1, int(len(X_train) * 0.2))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        w_tr = w_train[:-val_size]

        model = XGBRegressor(
            **best_params, objective="reg:squarederror",
            random_state=xgb.random_state, verbosity=xgb.verbosity,
            early_stopping_rounds=xgb.early_stopping_rounds,
        )
        model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], verbose=False,
        )
        preds = model.predict(X_test)
```

**Step 3: Add early stopping to holdout model (lines 244-252)**

```python
    if train_mask_ho.sum() >= 50 and holdout_mask.sum() >= 10:
        ho_X = pos_df.loc[train_mask_ho, available_feats].values
        ho_y = pos_df.loc[train_mask_ho, target].values
        ho_w = pos_df.loc[train_mask_ho, "_sample_weight"].values
        # Early stopping val split
        ho_val_size = max(1, int(len(ho_X) * 0.2))
        ho_model = XGBRegressor(
            **best_params, objective="reg:squarederror",
            random_state=xgb.random_state, verbosity=xgb.verbosity,
            early_stopping_rounds=xgb.early_stopping_rounds,
        )
        ho_model.fit(
            ho_X[:-ho_val_size], ho_y[:-ho_val_size],
            sample_weight=ho_w[:-ho_val_size],
            eval_set=[(ho_X[-ho_val_size:], ho_y[-ho_val_size:])],
            verbose=False,
        )
```

**Step 4: Add early stopping to final model (lines 257-262)**

For the final model, use all data but still apply early stopping with a held-out validation portion:

```python
    # Train final model on all data (with early stopping using last 15% as val)
    val_size_final = max(1, int(len(X_all) * 0.15))
    final_model = XGBRegressor(
        **best_params, objective="reg:squarederror",
        random_state=xgb.random_state, verbosity=xgb.verbosity,
        early_stopping_rounds=xgb.early_stopping_rounds,
    )
    final_model.fit(
        X_all[:-val_size_final], y_all[:-val_size_final],
        sample_weight=w_all[:-val_size_final],
        eval_set=[(X_all[-val_size_final:], y_all[-val_size_final:])],
        verbose=False,
    )
```

**Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All 99 tests pass.

**Step 6: Commit**

```bash
git add src/config.py src/ml/training.py
git commit -m "feat: add early stopping (20 rounds) to XGBoost training"
```

---

### Task 2: Increase Bench Weight

**Files:**
- Modify: `src/config.py:49` (SolverConfig.bench_weight)
- Test: `tests/test_correctness.py:125-126`

**Step 1: Update bench_weight from 0.1 to 0.25**

In `src/config.py` line 49:
```python
    bench_weight: float = 0.25
```

**Step 2: Verify test still passes**

The existing test at `tests/test_correctness.py:125-126` asserts `0 < bench_weight < 0.5`, so 0.25 passes.

```bash
.venv/bin/python -m pytest tests/test_correctness.py::TestConfigSanity::test_bench_weight_in_range -v
```

Expected: PASS

**Step 3: Run full tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All 99 tests pass.

**Step 4: Commit**

```bash
git add src/config.py
git commit -m "feat: increase bench weight from 0.1 to 0.25 for better auto-sub coverage"
```

---

### Task 3: Allow Multi-Hit Transfer Strategies

**Files:**
- Modify: `src/config.py:354` (StrategyConfig.max_hits_per_gw)
- Test: `tests/test_strategy_pipeline.py` (verify planner still works)

**Step 1: Update max_hits_per_gw from 1 to 2**

In `src/config.py` line 354:
```python
    max_hits_per_gw: int = 2         # Max hit transfers explored per GW
```

**Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass. The planner tree search will explore more paths but the `_MAX_TRANSFERS_PER_GW = 3` cap (line 24 of transfer_planner.py) keeps it bounded.

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: allow multi-hit transfers (max 2 hits/GW) for fixture swing strategies"
```

---

### Task 4: TC DGW Bias + Price Timing Across Planning Horizon

**Files:**
- Modify: `src/strategy/chip_evaluator.py:192-241` (_evaluate_triple_captain)
- Modify: `src/strategy/transfer_planner.py:332` (price bonus application)
- Test: `tests/test_strategy_pipeline.py`

**Step 1: Add DGW multiplier to TC evaluation**

In `src/strategy/chip_evaluator.py`, modify `_evaluate_triple_captain`. After computing the best player's predicted points (line 227-228), apply a DGW boost when fixture count > 1:

```python
    def _evaluate_triple_captain(
        self, current_squad_ids, future_predictions, fx_by_gw,
        all_gws, pred_gws,
    ) -> dict[int, float]:
        """Triple Captain value per GW.

        Within prediction horizon: best player's predicted points (extra 1x).
        DGW players get a boost since TC on DGW ~1.8x SGW value.
        Beyond: heuristic based on DGW + low FDR.
        """
        from src.strategy.transfer_planner import MultiWeekPlanner

        values: dict[int, float] = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                # TC can only be used on a starter in your squad
                candidates = gw_df[
                    gw_df["player_id"].isin(current_squad_ids)
                ]
                xi = (
                    MultiWeekPlanner._select_formation_xi(candidates)
                    if not candidates.empty
                    else candidates
                )
                if not xi.empty:
                    # Use captain_score to identify the best captain, but
                    # the TC chip value is the extra predicted_points
                    # (one additional multiply: 3x instead of 2x).
                    score_col = (
                        "captain_score"
                        if "captain_score" in xi.columns
                        else "predicted_points"
                    )
                    best_idx = xi[score_col].idxmax()
                    best = xi.loc[best_idx, "predicted_points"]

                    # DGW boost: TC on a DGW player is more valuable
                    # because the player scores in two matches.
                    # The model predictions already sum both fixtures,
                    # but conservative DGW predictions undervalue TC timing.
                    n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                    if n_dgw > 0:
                        # Check if the best TC candidate is in a DGW team
                        candidate_row = xi.loc[best_idx]
                        is_dgw_candidate = False
                        if "team_code" in xi.columns:
                            tc_code = candidate_row.get("team_code")
                            if tc_code and gw in fx_by_gw:
                                tc_fx = fx_by_gw[gw].get(int(tc_code), {})
                                is_dgw_candidate = bool(tc_fx.get("is_dgw"))
                        if is_dgw_candidate:
                            best *= 1.3  # 30% DGW boost to account for conservative predictions

                    values[gw] = round(best, 1)
                else:
                    values[gw] = 0.0
            else:
                # Heuristic: base TC value from premium captain
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                avg_fdr = self._avg_fdr(fx_by_gw, gw)
                fdr_factor = max(0, (3.5 - avg_fdr) / 2)
                values[gw] = round(
                    _TC_HEURISTIC_BASE * (1 + n_dgw * 0.4) * (1 + fdr_factor),
                    1,
                )

        return values
```

**Step 2: Extend price bonus beyond GW+1 with decay**

In `src/strategy/transfer_planner.py`, modify line 332. Instead of only applying price bonus for `i == 0`, apply it with decay across the first 3 GWs:

```python
            # Apply price bonus with decay (most urgent for immediate GW)
            if price_bonus and i < 3:
                decay = {0: 1.0, 1: 0.5, 2: 0.25}[i]
                gw_df["predicted_points"] = gw_df.apply(
                    lambda r, d=decay: r["predicted_points"]
                    + price_bonus.get(r["player_id"], 0) * d,
                    axis=1,
                )
```

**Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add src/strategy/chip_evaluator.py src/strategy/transfer_planner.py
git commit -m "feat: add TC DGW bias (30% boost) and extend price timing across 3 GWs"
```

---

### Task 5: End-of-Season Strategy Mode (GW33-38)

**Files:**
- Modify: `src/config.py:351-355` (StrategyConfig — add late_season_gw, late_season_hit_cost)
- Modify: `src/strategy/chip_evaluator.py:32-95` (add chip urgency multiplier)
- Modify: `src/strategy/transfer_planner.py:300-404` (_simulate_path — use reduced hit cost)
- Modify: `src/strategy/captain_planner.py` (higher Q80 weight in late season)
- Test: `tests/test_correctness.py`, `tests/test_strategy_pipeline.py`

**Step 1: Add late-season config to StrategyConfig**

In `src/config.py`, expand StrategyConfig:

```python
@dataclass(frozen=True)
class StrategyConfig:
    planning_horizon: int = 5        # GWs ahead for transfer planner
    max_hits_per_gw: int = 2         # Max hit transfers explored per GW
    ft_max_bank: int = 5             # Max banked free transfers
    late_season_gw: int = 33         # GW from which late-season mode activates
    late_season_hit_cost: float = 3.0  # Reduced hit cost in late season (amortized over fewer GWs)
```

**Step 2: Add chip urgency multiplier to ChipEvaluator**

In `src/strategy/chip_evaluator.py`, modify `evaluate_all_chips` to apply an urgency multiplier to chip values in late season. Unused chips become more valuable as they approach expiry.

After computing all chip values (before the `return` at line 95), add:

```python
        # Late-season chip urgency: unused chips approaching expiry get a boost
        current_gw = min(pred_gws) if pred_gws else min(all_gws)
        if current_gw >= strategy_cfg.late_season_gw:
            # Half-season boundary for current half
            half_end = 19 if current_gw <= 19 else 38
            gws_remaining = max(1, half_end - current_gw + 1)
            # Urgency ramps from 1.0 at GW33 to ~1.6 at GW38
            urgency = 1.0 + 0.12 * max(0, strategy_cfg.late_season_gw + 5 - gws_remaining)
            for chip_name in chip_values:
                for gw in chip_values[chip_name]:
                    chip_values[chip_name][gw] = round(
                        chip_values[chip_name][gw] * urgency, 1,
                    )
```

Add the import at the top of the file:
```python
from src.config import strategy_cfg
```

**Step 3: Use reduced hit cost in late season**

In `src/strategy/transfer_planner.py`, the hit cost is applied by the MILP solver in `src/solver/transfers.py`. The planner passes through to `solve_transfer_milp_with_hits`. We need to pass a `hit_cost` override in late season.

First, modify `_simulate_transfer_gw` to accept and pass a hit_cost parameter. In the `_simulate_path` method, detect late season and pass reduced hit cost:

In `_simulate_path` (around line 318), add late season detection:

```python
        # Late-season: use reduced hit cost
        from src.config import strategy_cfg as strat_cfg
        late_season = plan_gws and plan_gws[0] >= strat_cfg.late_season_gw
```

Then in the transfer GW simulation call (line 388-391), pass it through. However, since `solve_transfer_milp_with_hits` uses `solver_cfg.hit_cost` internally, we need a simpler approach.

**Better approach:** Override at the config level during late season isn't clean. Instead, in `_simulate_path`, adjust predicted_points upward to offset the reduced hit penalty. The MILP solver applies -4 per hit. In late season, we want -3. So after the solver returns, we add back +1 per hit to compensate:

In `_simulate_transfer_gw`, after the solver returns (around line 518), add:

```python
            # Late-season hit cost reduction
            hits = max(0, len(transfers_in) - ft)
            if late_season and hits > 0:
                hit_discount = hits * (solver_cfg.hit_cost - strat_cfg.late_season_hit_cost)
                pts += hit_discount
```

This requires passing `late_season` as a parameter to `_simulate_transfer_gw`. Update the signature and call site.

**Step 4: Write a test for late-season config**

Add to `tests/test_correctness.py` in `TestConfigSanity`:

```python
    def test_late_season_config_valid(self):
        from src.config import strategy_cfg
        assert 30 <= strategy_cfg.late_season_gw <= 36
        assert 0 < strategy_cfg.late_season_hit_cost < solver_cfg.hit_cost
```

**Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (100 now with the new test).

**Step 6: Commit**

```bash
git add src/config.py src/strategy/chip_evaluator.py src/strategy/transfer_planner.py tests/test_correctness.py
git commit -m "feat: add end-of-season strategy mode (GW33+) with chip urgency and reduced hit cost"
```

---

### Task 6: Run Full Test Suite and Verify

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 100 tests pass, 0 failures.

**Step 2: Verify no regressions in critical invariants**

Spot-check that the 8 regression patterns from the audit are still intact:
- Availability zeroing after 3-GW merge in `ml/prediction.py`
- Budget from `entry_history["value"]` in `season/manager.py`
- Strategy pipeline runs first in `generate_recommendation()`
- Chip plan passed to planner
- Solver failure fallbacks in `transfer_planner.py`
- Hit cost from solver result in `validator.py`
- Watchlist in prices
- pandas CoW .copy() calls

---

### Task 7: Update CLAUDE.md with new config values

**Files:**
- Modify: `CLAUDE.md`

Update the config table in CLAUDE.md to reflect new values:
- `SolverConfig`: bench_weight 0.1 -> 0.25
- `StrategyConfig`: max_hits_per_gw 1 -> 2, add late_season_gw=33, late_season_hit_cost=3.0
- `XGBConfig`: add early_stopping_rounds=20
- Mention TC DGW bias and price timing extension in the chip evaluator and transfer planner sections
