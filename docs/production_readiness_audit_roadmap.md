# GRU Alpha Production Readiness Audit

Recorded date: 2026-05-29

## Verdict

Current status:

- Research alpha prototype: pass.
- Course/project deliverable: pass.
- Production tradable strategy: fail.

The current best research configuration is:

- Score model: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/`
- Strategy overlay: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/`
- Portfolio rule for reporting: strict tradable mask plus K20 keep=2x turnover buffer.
- Main analysis outputs: `outputs/analysis/turnover_overlay_eval/`

This version is a credible research baseline because it has a functioning sequence model, a fixed head-saturation issue, strict tradable-sample filtering, transaction-cost proxy evaluation, and turnover-buffer analysis.

It is not production-ready because the evidence does not yet prove stable residual alpha, executable trading returns, or robustness outside the favorable 2025-2026 test regime.

## Why It Cannot Be Delivered As A Live Strategy

### 1. Validation-Test Inversion Is A Red Flag

The most dangerous pattern is not simply that validation is weak. The deeper issue is the asymmetric performance profile:

- Validation portfolio returns remain poor, with large drawdowns.
- Test portfolio returns are very strong across K values.
- The old full62 feature set works best in test, while cleaner neutralized variants weaken or fail.
- The final choice has passed through multiple rounds of model, feature, mask, K, and turnover-buffer selection.

This creates three major risks.

First, test-set snooping risk is high. The test set has effectively been inspected repeatedly through ReLU, GELU, LeakyReLU slope variants, strict-mask overlay, clean alpha-only data, residualized data, K10/K20/K30, and keep-multiplier variants. Once the test set influences design choices, it is no longer a clean final holdout.

Second, regime-overfit risk is high. The strong 2025-2026 test result may reflect a specific market style, such as growth, technology, liquidity recovery, small-cap behavior, momentum, or industry rotation. The model may be harvesting a favorable regime rather than a stable cross-regime stock-selection signal.

Third, point-in-time leakage is not fully ruled out. Current proxy backtests merge predictions with label-side `future_return`; this is acceptable only if every feature, mask, stock-pool decision, and label boundary is proven point-in-time. That audit has not yet been completed.

Required conclusion:

> The current test result is useful research evidence, not production alpha proof.

### 2. K30 Keep=3 Low Turnover High Excess Is Suspicious

The K30 keep=3 setting reports very low top-leg turnover and positive excess. This should not be treated as strong evidence of live tradability.

The rank-buffer rule keeps existing names while their current rank remains within `ceil(K * keep_multiplier)`. For K30 keep=3, a name can remain in the portfolio while ranked inside Top90. This is no longer a strict Top30 strategy. It is a path-dependent slow-turnover portfolio initialized by earlier model selections.

The low-turnover high-excess result may be driven by:

- A lucky initial basket during the favorable test regime.
- Path dependence rather than continuing prediction skill.
- Survivorship bias in the stock pool.
- Unrealistic execution assumptions.
- Locked limit-up names whose returns are counted but could not have been bought.
- Locked limit-down names whose losses are not fully reflected through failed exits.
- Missing capacity and market-impact constraints.

Current backtest limitations:

- Uses label-side holding-period `future_return`.
- Uses equal-weight selected names.
- Applies cost as `turnover * cost_bps`.
- Does not simulate order placement, partial fills, open/VWAP execution, queue priority, or intraday liquidity.
- Does not fully model limit-up buy failures or limit-down sell failures.
- Does not model portfolio weight drift.
- Does not model borrow availability or borrow fees for long-short results.

Required conclusion:

> K30 keep=3 is a sensitivity check, not a production portfolio.

### 3. Neutralized Features Underperforming Is A Style-Risk Warning

The cleaner feature sets and residualized variants did not beat the old full62 feature set in the test period. This is not automatically a reason to keep the old features without concern.

The more conservative interpretation is:

> The old full62 model may be relying on unneutralized style, liquidity, size, turnover, volatility, industry, or short-term return exposures.

These exposures can look like alpha in one regime and behave like beta in another. If market leadership rotates away from the style favored by the old full62 feature set, live performance can degrade quickly.

Risky feature families include:

- Turnover and turnover volatility.
- Amount and volume.
- Market capitalization.
- Volatility and amplitude.
- Limit-position features.
- Short-horizon return and momentum/reversal features.
- Industry-relative liquidity or money-flow variables.

Required conclusion:

> The old full62 feature set is currently the best research signal, but its alpha is not yet proven to be residual alpha.

## Production Readiness Gaps

Blocking gaps before any live-strategy claim:

- No full point-in-time field audit.
- No purged and embargoed walk-forward validation.
- No locked test-set discipline after strategy selection.
- No production-grade execution simulation.
- No formal style and industry exposure constraints.
- No portfolio optimizer with risk model and transaction-cost model.
- No capacity analysis.
- No borrow model for long-short results.
- No proof that validation-period drawdown is an acceptable live risk.

## Roadmap

### P0: Stop Optimizing Test Performance

Freeze the current test result as a historical research observation. Future choices must not be made from 2025-2026 test performance.

From this point forward, the 2025-2026 period is a sealed test-set lockbox. It has already been inspected too often to support further model, feature, K, keep-multiplier, capacity, participation, execution-mode, or reporting-language decisions. Treat all existing 2025-2026 results as contaminated research history, not as an active selection signal.

Immediate actions:

- Treat `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/` as the frozen score baseline.
- Treat `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/` as the frozen proxy strategy baseline.
- Do not promote additional K or keep-multiplier settings based on test results.
- Do not promote capacity, participation, execution, risk-control, or portfolio settings based on 2025-2026 results.
- Base all future choices only on pre-declared purged and embargoed walk-forward validation.
- Open the 2025-2026 lockbox exactly once after the full model-selection protocol is frozen.
- Record the lockbox opening date, code commit, config files, random seeds, selected candidate, and acceptance thresholds before running the final evaluation.
- If any choice changes after opening the lockbox, the lockbox result becomes invalid for final proof and must be reported only as exploratory.

Lockbox rule:

> 2025-2026 is no longer a tuning, selection, or promotion input. It is a sealed final holdout. The only admissible path is: pre-register the walk-forward selection protocol, select the candidate using walk-forward validation only, freeze code and configs, then open the 2025-2026 lockbox once for final reporting.

### P1: Point-In-Time Audit

Create a dedicated audit script:

```bash
python scripts/audit_point_in_time.py
```

Required checks:

| Check | Requirement |
| --- | --- |
| Feature timestamp | Every feature must be available no later than signal time. |
| Label boundary | `future_return` must begin after executable signal time. |
| Strict mask timestamp | Tradable mask must use only information known at or before decision time. |
| Stock pool membership | Universe construction must not use future survivorship. |
| ST/suspension status | Status flags must be point-in-time. |
| Limit-up/down state | Execution filters must not use future-day information unless explicitly modeling next-day execution failure. |
| Adjustment factors | Price adjustment must not leak future corporate-action information. |

Suggested output:

```text
outputs/audit/point_in_time/field_audit.csv
outputs/audit/point_in_time/leakage_findings.md
outputs/audit/point_in_time/suspect_features.txt
```

Acceptance criteria:

- No unresolved future-looking features in the main model.
- No future-looking strict-mask filters.
- Label construction documented with exact signal time and execution time.

### P2: Purged And Embargoed Walk-Forward Validation

Replace the single train/validation/test interpretation with purged and embargoed walk-forward validation. The current single split is too fragile to support a production-readiness claim.

Proposed model-selection folds:

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| F1 | 2016-2019 | 2020 | 2021 |
| F2 | 2016-2020 | 2021 | 2022 |
| F3 | 2016-2021 | 2022 | 2023 |
| F4 | 2016-2022 | 2023 | 2024 |

Reserved final lockbox:

| Period | Role | Rule |
| --- | --- | --- |
| 2025-2026 | Final holdout | Open exactly once after the walk-forward protocol, candidate, code, configs, seeds, and acceptance thresholds are frozen. |

Use an embargo window at every validation/test boundary. If the label is a future 5-trading-day return, isolate at least 5 to 20 trading days around each boundary so adjacent samples cannot share overlapping future-return information. Start with `--embargo-days 20` as the conservative default, then report sensitivity to shorter embargo windows only as diagnostics.

Environment:

```bash
conda activate dl_env
```

Implementation tasks:

- Add `split_scheme: walk_forward` support to the dataset builder.
- Add fixed fold definitions for F1-F4 using the table above.
- Exclude 2025-2026 from all model-selection, parameter-selection, and diagnostic-ranking outputs.
- Purge overlapping label windows before applying the embargo.
- Export fold-specific sequence NPZ files so each fold has independent train, validation, and test arrays.
- Add `--fold` support to `scripts/train_sequence.py`.
- Add a fold summary script that computes fold mean, t-stat, positive-fold ratio, worst fold, and fold-level portfolio excess.
- Store all fold manifests with exact date ranges, embargo days, row counts, stock counts, and dropped-sample counts.

Suggested commands:

```bash
conda activate dl_env
python scripts/build_model_datasets.py --split-scheme walk_forward --embargo-days 20
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F1
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F2
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F3
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F4
python scripts/summarize_walk_forward.py
```

Acceptance criteria:

- Do not select or promote the best fold. The evaluation unit is the full walk-forward panel.
- Mean RankIC across folds is positive.
- At least 70% of folds have positive RankIC.
- RankIC fold-level t-stat is reported.
- Worst fold is not disastrous under the pre-declared drawdown and excess-return thresholds.
- Portfolio excess is positive in a majority of folds.
- Results do not use 2025-2026 until the final lockbox opening.
- Fold-level results include both validation and test behavior; model and portfolio choices cannot be made from later test folds.
- The final selected candidate is declared from the full F1-F4 walk-forward panel before the 2025-2026 lockbox is opened.

### P3: Formal Style Exposure Report

Do not rely on feature neutralization alone. First measure the exposures precisely.

Required exposures:

- Industry.
- Log market cap.
- Liquidity and amount.
- Turnover.
- Volatility.
- Beta.
- Short-term momentum.
- Medium-term momentum.
- Value proxies.
- Limit-position and tradability proxies.

Create:

```bash
python scripts/report_style_exposure.py \
  --predictions outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/predictions.parquet \
  --output-dir outputs/audit/style_exposure
```

Required outputs:

```text
outputs/audit/style_exposure/top_bottom_exposure_by_day.csv
outputs/audit/style_exposure/exposure_summary_by_split.csv
outputs/audit/style_exposure/exposure_by_year.csv
outputs/audit/style_exposure/exposure_findings.md
```

Acceptance criteria:

- Top-minus-bottom industry and style exposures are reported.
- Exposure stability is evaluated by year and regime.
- Returns are decomposed into style return and residual return.

### P4: Portfolio Optimizer Instead Of Raw Top-K

The model should produce alpha scores. Risk control should happen at the portfolio layer.

Add:

```bash
python scripts/optimize_portfolio.py
```

Initial optimization objective:

```text
maximize:
    alpha' w
    - lambda_risk * w' Sigma w
    - lambda_tc * transaction_cost(w - w_prev)
```

Initial constraints:

```text
sum(w) = 1
0 <= w_i <= single_name_cap
industry exposure within bounds
style exposure within bounds
turnover <= turnover_cap
participation <= participation_cap
no buy if limit-up locked
no sell if limit-down locked
```

Minimum risk model:

- Industry factor covariance.
- Style factor exposures.
- Shrunk residual covariance.
- Rolling volatility estimates.

Design of experiments:

| Alpha | Risk Control | Portfolio |
| --- | --- | --- |
| old_full62 score | none | TopK |
| old_full62 score | industry neutral | optimizer |
| old_full62 score | industry + size | optimizer |
| old_full62 score | industry + size + liquidity + volatility + momentum | optimizer |
| alpha-only score | same controls | optimizer |
| residualized score | same controls | optimizer |

Acceptance criteria:

- Style exposure is materially reduced.
- Turnover is controlled.
- Excess return remains positive in most walk-forward folds.
- Validation-period drawdown improves versus raw Top-K.

### P5: Production-Like Backtest

Replace label-return replay with an execution-aware simulator.

Required data columns:

- Next open.
- Next VWAP.
- Daily high/low.
- Daily amount and volume.
- Limit-up and limit-down flags at open and close.
- Suspension flag.
- ST flag.
- Corporate-action adjustment factors.

Required execution rules:

```text
If buy order and next open is locked limit-up: fill = 0
If sell order and next open is locked limit-down: fill = 0
If suspended: fill = 0
If order notional > participation_cap * daily_amount: partial fill
Execution price = next VWAP plus slippage
Unfilled orders carry forward or are cancelled according to config
```

Suggested slippage models:

| Model | Formula |
| --- | --- |
| Linear | `slippage = a * participation` |
| Square-root | `slippage = b * volatility * sqrt(participation)` |
| Stress | `max(linear, square_root) + limit_penalty` |

Suggested command:

```bash
python scripts/backtest_execution_sim.py \
  --orders outputs/portfolio/orders.parquet \
  --market-data data/mart/datasets/dataset_v20260526.parquet \
  --slippage-model square_root \
  --participation-cap 0.03 \
  --cost-bps 20 \
  --output-dir outputs/backtest/execution_sim
```

Acceptance criteria:

- Report filled quantity, rejected quantity, and partial-fill rate.
- Report turnover after actual fills.
- Report realized slippage.
- Report limit-lock missed alpha.
- Report capacity sensitivity.

### P6: Long-Short Results Must Be Downgraded

Current long-short results are diagnostic only.

Before any long-short production claim, add:

- Borrow availability.
- Borrow fee.
- Locate failure probability.
- Short-sale restrictions.
- Recall risk.
- Short-side transaction costs.

Until then, official performance should focus on long-only excess versus a tradable universe.

## Research Design Matrix

Run the following matrix after P0-P3 are complete:

| Experiment | Score | Universe | Portfolio | Backtest | Goal |
| --- | --- | --- | --- | --- | --- |
| E1 | old_full62 | strict | TopK K20 | proxy | frozen baseline |
| E2 | old_full62 | strict | K20 keep=2 | proxy | current mainline |
| E3 | old_full62 | strict | optimizer industry-neutral | proxy | isolate industry risk |
| E4 | old_full62 | strict | optimizer style-neutral | proxy | isolate residual alpha |
| E5 | alpha-only | strict | optimizer style-neutral | proxy | clean-feature comparison |
| E6 | residualized | strict | optimizer style-neutral | proxy | residual-feature comparison |
| E7 | old_full62 | strict | optimizer style-neutral | execution sim | execution haircut |
| E8 | ensemble | strict | optimizer style-neutral | execution sim | final candidate |

Promotion criteria for production candidate:

- Positive excess return in most walk-forward folds.
- No single fold dominates total performance.
- Style-adjusted residual return remains positive.
- Execution-aware returns remain positive after 20 bps cost and slippage.
- Turnover and participation are within capacity limits.
- Drawdown is acceptable in validation and stress regimes.
- No unresolved point-in-time leakage findings.

## Current Reporting Language

Recommended language:

> The project has produced a credible GRU-based research alpha prototype. The best current configuration uses the old full62 feature set, a LeakyReLU head with negative slope 0.005, strict tradable-sample filtering, and a K20 keep=2x turnover buffer. The result is suitable as a research baseline and course-project deliverable.

Do not write:

> The strategy is production-ready or directly tradable.

Required disclaimer:

> The current result has not yet passed point-in-time audit, purged walk-forward validation, formal style-risk attribution, production-like execution simulation, or capacity analysis. Test-period performance may be affected by test-set snooping and favorable regime exposure.

## Immediate Checklist

- [x] Freeze the current test result.
- [x] Add point-in-time audit script.
- [ ] Build purged and embargoed walk-forward folds.
- [ ] Generate style exposure reports.
- [ ] Build initial portfolio optimizer.
- [x] Replace proxy backtest with execution-aware simulation.
- [ ] Downgrade long-short results to diagnostic status.
- [ ] Re-evaluate only after the above controls are in place.

## Point-In-Time Audit Run 2026-05-29

Command:

```bash
conda activate dl_env
python scripts/audit_point_in_time.py
```

Generated outputs:

```text
outputs/audit/point_in_time/field_audit.csv
outputs/audit/point_in_time/feature_column_audit.csv
outputs/audit/point_in_time/negative_shift_audit.csv
outputs/audit/point_in_time/suspect_features.txt
outputs/audit/point_in_time/leakage_findings.md
```

Verdict:

```text
PASS_WITH_WARNINGS
```

Summary:

| Severity | Count |
| --- | ---: |
| Blocker | 0 |
| Warning | 4 |
| Pass | 9 |

Confirmed pass checks:

- Feature config declares `future_shift_allowed=false`.
- Feature config declares `dataset_requires_lagged_features_only=true`.
- Mart dataset feature columns all use the `lag1_` prefix.
- Static code scan found no unapproved negative feature shifts in audited mart/backtest scripts.
- `add_lagged_features` uses grouped `shift(1)` for `lag1_` feature creation.
- Mart dataset has unique `trade_date + ts_code` keys.
- Label table has unique `trade_date + ts_code` keys.
- `label_rel_return = future_return - benchmark_future_return` identity check passes.
- Strict mask filter log has unique `trade_date + ts_code + split` keys.

Open warnings:

| ID | Issue | Required follow-up |
| --- | --- | --- |
| `CODE002` | Labels are future close-to-close returns, not executable T+1 open/VWAP returns. | Add execution-price labels before production-like backtest. |
| `LBL003` | Label table lacks `next_open_return`, `next_vwap_return`, `buy_executable`, `sell_executable`, and fillability columns. | Build execution label table. |
| `MSK002` | `mask_locked_limit` is based on same-date state, not next-session fill simulation. | Treat strict mask as conservative sample filter only; add next-open executable flags. |
| `DATA003` | Dataset contains 42 style or microstructure-sensitive features. | Run style, liquidity, and execution attribution before production use. |

Interpretation:

The first point-in-time audit did not find direct future-feature leakage in the audited construction path. However, it does not clear the strategy for production because current labels and backtests remain close-to-close proxy evaluations. The next required step is not more model tuning; it is to build execution-aware labels and then rerun backtests under next-open or next-VWAP assumptions with fillability constraints.

## T+1 Fill Simulation Run 2026-05-29

Execution label build command:

```bash
conda activate dl_env
python scripts/build_execution_labels.py
```

Generated execution labels:

```text
data/mart/labels/execution_labels_v20260526.parquet
data/mart/labels/execution_labels_v20260526_manifest.json
```

Execution label summary:

| Field | Value |
| --- | ---: |
| Rows | 10,719,660 |
| Trade dates | 2,521 |
| Stocks | 5,761 |
| Date range | 20160104..20260525 |
| Buy executable rate | 0.9586 |
| Sell executable rate | 0.9673 |
| Execution return coverage | 0.9973 |

Fill simulation command:

```bash
conda activate dl_env
python scripts/backtest_t1_fill_sim.py \
  --predictions outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet \
  --execution-labels data/mart/labels/execution_labels_v20260526.parquet \
  --output-dir outputs/backtest/t1_fill_sim/gru_l20_slope0005_k20_keep2 \
  --k 20 \
  --keep-multiplier 2 \
  --cost-bps 10 \
  --slippage-bps 5 \
  --portfolio-nav 10000000 \
  --participation-cap 0.03 \
  --rebalance-stride 5
```

Generated outputs:

```text
outputs/backtest/t1_fill_sim/gru_l20_slope0005_k20_keep2/t1_fill_metrics.json
outputs/backtest/t1_fill_sim/gru_l20_slope0005_k20_keep2/t1_fill_periods.csv
```

Simulation assumptions:

- T-day close signal.
- T+1 open fill attempt.
- 5-trading-day holding horizon.
- Entry and exit return approximation: T+1 open to T+5 close.
- Buy blocked when T+1 is not executable or locked limit-up.
- Sell blocked when T+1 is not executable or locked limit-down.
- Orders above 3% of next-day amount are partially filled.
- Cost model: 10 bps explicit cost plus 5 bps filled-turnover slippage.
- Portfolio NAV: 10,000,000.

Mainline K20 keep=2x result:

| Split | Net annualized | Excess vs benchmark ann. | Excess vs executable universe ann. | Max drawdown | Desired turnover | Filled turnover | Avg buy rejects | Avg sell rejects | Avg partial fills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | -0.2271 | -0.2141 | -0.1483 | -0.5699 | 1.1532 | 0.0648 | 0.0206 | 3.7835 | 34.7732 |
| test | 0.4106 | -0.1109 | -0.0159 | -0.1924 | 1.1832 | 0.1434 | 0.0152 | 3.7121 | 32.1061 |

Interpretation:

The execution-aware result materially weakens the previous proxy conclusion. Test split still has positive absolute return, but it no longer shows positive excess versus the benchmark or the executable equal-weight universe. Validation remains deeply negative. The large gap between desired turnover and filled turnover also shows that the previous turnover-buffer result was not enough to model tradability: capacity and partial fills dominate actual implementation.

Production implication:

> The current mainline should not be described as a tradable strategy. Under first-pass T+1 fill simulation, its test-period absolute return is mostly market/regime participation, not robust executable stock-selection excess.

Immediate next steps:

1. Run capacity sensitivity at NAV 1m, 10m, 50m, and 100m.
2. Run participation sensitivity at 1%, 3%, 5%, and 10%.
3. Add next-VWAP execution mode and compare against T+1 open.
4. Start style attribution on execution-aware returns rather than proxy close-to-close returns.

## T+1 Fill K/Keep Matrix Run 2026-05-29

Command:

```bash
conda activate dl_env
python scripts/backtest_t1_fill_sim.py \
  --predictions outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet \
  --execution-labels data/mart/labels/execution_labels_v20260526.parquet \
  --output-dir outputs/backtest/t1_fill_sim/gru_l20_slope0005_k_keep_matrix_nav10m_part3pct \
  --k "10,20,30" \
  --keep-multiplier "1,1.5,2,3" \
  --cost-bps 10 \
  --slippage-bps 5 \
  --portfolio-nav 10000000 \
  --participation-cap 0.03 \
  --rebalance-stride 5
```

Generated outputs:

```text
outputs/backtest/t1_fill_sim/gru_l20_slope0005_k_keep_matrix_nav10m_part3pct/t1_fill_metrics.json
outputs/backtest/t1_fill_sim/gru_l20_slope0005_k_keep_matrix_nav10m_part3pct/t1_fill_periods.csv
outputs/backtest/t1_fill_sim/gru_l20_slope0005_k_keep_matrix_nav10m_part3pct/t1_fill_matrix_summary.csv
```

Test split summary:

| K | Keep | Net ann. | Excess vs benchmark ann. | Excess vs executable universe ann. | Max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1.0x | 0.6898 | 0.0723 | 0.1813 | -0.1976 |
| 10 | 1.5x | 0.7167 | 0.0901 | 0.2006 | -0.2009 |
| 10 | 2.0x | 0.6150 | 0.0203 | 0.1253 | -0.1972 |
| 10 | 3.0x | 0.5794 | -0.0005 | 0.1018 | -0.2009 |
| 20 | 1.0x | 0.3119 | -0.1691 | -0.0841 | -0.1867 |
| 20 | 1.5x | 0.3769 | -0.1310 | -0.0391 | -0.1887 |
| 20 | 2.0x | 0.4106 | -0.1109 | -0.0159 | -0.1924 |
| 20 | 3.0x | 0.3567 | -0.1497 | -0.0591 | -0.1755 |
| 30 | 1.0x | 0.5420 | -0.0248 | 0.0784 | -0.1907 |
| 30 | 1.5x | 0.4305 | -0.1005 | -0.0037 | -0.1885 |
| 30 | 2.0x | 0.4754 | -0.0727 | 0.0264 | -0.1915 |
| 30 | 3.0x | 0.6477 | 0.0393 | 0.1489 | -0.1987 |

Validation split summary:

| K | Keep | Net ann. | Excess vs benchmark ann. | Excess vs executable universe ann. | Max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1.0x | -0.2509 | -0.2420 | -0.1794 | -0.5841 |
| 10 | 1.5x | -0.2514 | -0.2422 | -0.1796 | -0.5847 |
| 10 | 2.0x | -0.2569 | -0.2474 | -0.1851 | -0.5906 |
| 10 | 3.0x | -0.2465 | -0.2358 | -0.1725 | -0.5810 |
| 20 | 1.0x | -0.2362 | -0.2240 | -0.1594 | -0.5814 |
| 20 | 1.5x | -0.2302 | -0.2184 | -0.1530 | -0.5733 |
| 20 | 2.0x | -0.2271 | -0.2141 | -0.1483 | -0.5699 |
| 20 | 3.0x | -0.2205 | -0.2092 | -0.1429 | -0.5650 |
| 30 | 1.0x | -0.2037 | -0.1921 | -0.1241 | -0.5565 |
| 30 | 1.5x | -0.1919 | -0.1822 | -0.1132 | -0.5463 |
| 30 | 2.0x | -0.2133 | -0.2027 | -0.1351 | -0.5637 |
| 30 | 3.0x | -0.1661 | -0.1578 | -0.0869 | -0.5009 |

Matrix interpretation:

- The originally promoted K20 keep=2x does not survive T+1 execution constraints as an excess-return strategy.
- Test split has several positive-excess survivors, especially K10 keep=1.5x and K30 keep=3x.
- Validation split is negative for every K/keep combination, including all positive test survivors.
- K10 keep variants now look strongest on test, but this reverses the earlier proxy-backtest preference and increases suspicion of test-regime specificity.
- K30 keep=3x again shows positive test excess, but validation remains negative and the result is path-dependent.

Production implication:

> The K/keep matrix does not rescue the strategy. It identifies candidate diagnostics for further attribution, but no setting qualifies as production-ready because all settings fail validation under T+1 fill simulation.

Historical diagnostic candidates, now frozen:

| Candidate | Reason | Risk |
| --- | --- | --- |
| K10 keep=1.5x | Best test excess versus executable universe and benchmark. | Narrow head, likely regime/test-snooping risk. |
| K30 keep=3x | Positive test excess with wider basket and lower churn. | Strong path dependence and negative validation. |
| K30 keep=1x | Positive test excess versus executable universe. | Still negative vs benchmark and validation. |

Next required step:

Do not run further selection work from these test-positive candidates. They may remain in the document as historical diagnostics, but future candidate choice must come only from the pre-declared walk-forward validation panel. Capacity and participation sensitivity should be run only after a candidate is selected by walk-forward validation, with the grid also pre-declared before any 2025-2026 lockbox opening.

## Full62 Attribution Audit Run 2026-05-29

Command:

```bash
conda activate dl_env
python scripts/audit_full62_attribution.py
```

Generated outputs:

```text
outputs/audit/full62_attribution/candidate_selected_snapshots.parquet
outputs/audit/full62_attribution/style_exposure_selected.csv
outputs/audit/full62_attribution/style_liquidity_vs_universe.csv
outputs/audit/full62_attribution/liquidity_bucket_returns.csv
outputs/audit/full62_attribution/execution_attribution.csv
outputs/audit/full62_attribution/full62_attribution_findings.md
outputs/audit/full62_attribution/manifest.json
```

Scope:

- Base signal: old full62 GRU score, LeakyReLU head slope=0.005.
- Candidates: K10 keep=1.5x, K30 keep=3x, K30 keep=1x.
- Return lens: T+1 fill simulation periods from the execution-aware K/keep matrix.
- Attribution lens: selected target names versus executable universe. This is selection attribution, not yet post-fill weight attribution.

### Execution Attribution

| Candidate | Split | Net mean | Excess vs benchmark mean | Excess vs executable universe mean | Desired turnover | Filled turnover | Fill ratio | Avg sell rejects | Avg partial fills |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K10 keep=1.5x | test | 0.011724 | 0.002019 | 0.003974 | 1.662486 | 0.142826 | 0.085911 | 1.500000 | 23.242424 |
| K30 keep=3x | test | 0.010570 | 0.000865 | 0.002820 | 0.305066 | 0.041279 | 0.135311 | 4.409091 | 11.954545 |
| K30 keep=1x | test | 0.009298 | -0.000407 | 0.001549 | 1.343890 | 0.220053 | 0.163744 | 7.833333 | 53.318182 |
| K10 keep=1.5x | validation | -0.004418 | -0.005325 | -0.003747 | 1.679360 | 0.046814 | 0.027876 | 1.525773 | 23.195876 |
| K30 keep=3x | validation | -0.002387 | -0.003294 | -0.001717 | 0.250957 | 0.023238 | 0.092596 | 2.618557 | 12.783505 |
| K30 keep=1x | validation | -0.003164 | -0.004071 | -0.002493 | 1.415092 | 0.104329 | 0.073726 | 9.082474 | 59.371134 |

Execution conclusion:

The first-order cost drag is not the main killer. The larger issue is implementation distortion: filled turnover is only about 2.8% to 16.4% of desired turnover depending on split and candidate. This means the realized portfolio is materially different from the intended score-ranked portfolio. K30 keep=3x is the cleanest candidate from a churn perspective, but even it has validation net mean below zero and non-trivial blocked/partial execution.

### Style Attribution

Main selected-minus-universe patterns:

| Candidate | Split | Dominant exposure pattern |
| --- | --- | --- |
| K10 keep=1.5x | test | Larger and more liquid names; lower turnover; lower recent activity/volatility. |
| K30 keep=3x | test | Stable large-cap/liquid tilt; lower turnover; low-churn defensive profile. |
| K30 keep=1x | test | More liquid than universe but still lower turnover and lower volume-log exposure. |
| K30 keep=3x | validation | Strong large-cap tilt, strong low-turnover tilt, but negative returns. |

Representative exposures:

- K10 keep=1.5x test selects higher `lag1_amount_60d_mean` by about 589k and higher `lag1_log_circ_mv` by 0.222 versus universe.
- K30 keep=3x test selects higher `lag1_amount_20d_mean` by about 517k and higher `lag1_log_circ_mv` by 0.215 versus universe.
- K30 keep=3x validation selects higher `lag1_log_circ_mv` by 0.337 and lower `lag1_turnover_60d_mean` by 0.373 versus universe.
- All three candidates consistently show lower turnover-rate exposure than the executable universe.

Style conclusion:

The current full62 result does not look like a pure micro-cap or junk-stock artifact. It is closer to a large/liquid/low-turnover/low-volatility selection profile with some reversal or defensive character. That is better than a microcap illusion, but it is still not robust alpha: the same broad style profile works in test and fails in validation.

### Liquidity Attribution

Executable-universe bucket returns show strong regime dependence:

| Split | Bucket feature | Low bucket mean | Mid bucket mean | High bucket mean | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| test | `lag1_amount_20d_mean` | 0.002826 | 0.009575 | 0.012447 | Test rewarded liquid names. |
| test | `lag1_turnover_rate_f` | 0.003134 | 0.009072 | 0.012643 | Test rewarded high turnover. |
| test | `lag1_ret_20d_std` | 0.003883 | 0.007617 | 0.013331 | Test rewarded high volatility. |
| validation | `lag1_amount_20d_mean` | -0.001156 | -0.001651 | -0.000407 | Liquidity helped only weakly. |
| validation | `lag1_turnover_rate_f` | -0.002243 | -0.001764 | 0.000790 | High turnover helped validation universe, while selected portfolios stayed low-turnover. |
| validation | `lag1_log_circ_mv` | -0.000720 | -0.000610 | -0.001873 | Large-cap bucket hurt validation. |

Liquidity conclusion:

The profit is not concentrated in illiquid microcaps. In the test period, the whole executable universe rewarded high liquidity, high turnover, and high realized volatility buckets. The strategy's selected basket is not simply chasing those buckets; however, its positive test excess is still embedded in a favorable liquidity/volatility regime that did not hold in validation.

### PM Verdict

This attribution audit strengthens the "not production-ready" verdict.

What is ruled out:

- The result is probably not just a microcap garbage-stock backtest artifact.
- The result is not primarily explained by explicit 10 bps cost plus 5 bps slippage drag.

What remains dangerous:

- The full62 alpha is style-regime dependent: large/liquid/low-turnover exposure is not consistently rewarded.
- Execution constraints materially reshape the portfolio because most desired turnover is not filled under the 3% participation cap.
- Validation remains negative for every diagnostic candidate even after moving strict tradability constraints into T+1 execution.
- The test-positive candidates are frozen historical diagnostics, not deployable strategies and not eligible as future selection inputs.

Required next experiments:

| Priority | Experiment | Stop condition |
| ---: | --- | --- |
| 1 | Walk-forward purged validation with fixed model-selection protocol using only pre-2025 folds. | Stop promotion if winners rotate by fold or require test-period selection. |
| 2 | Barra-lite residual alpha regression on selected returns and score deciles. Controls: Size, Momentum, Beta, Volatility, Liquidity, Industry. | Stop promotion if residual alpha is statistically weak or flips sign by fold. |
| 3 | Post-fill holdings attribution instead of target-selection attribution. | Stop promotion if realized holdings are dominated by fill artifacts rather than model rank. |
| 4 | Capacity and participation sensitivity on the walk-forward-selected candidate. NAV and participation grid must be declared before the final lockbox opening. | Stop promotion if excess survives only at unrealistic capacity or participation. |

## Canonical Labels And Capacity Matrix Run 2026-05-29

### Canonical Label Table

Command:

```bash
conda activate dl_env
python scripts/build_canonical_labels.py
```

Generated outputs:

```text
data/mart/labels/labels_canonical_v20260526.parquet
data/mart/labels/labels_canonical_v20260526_manifest.json
```

Manifest summary:

| Metric | Value |
| --- | ---: |
| Rows | 242,938 |
| Trade dates | 2,521 |
| Stocks | 259 |
| Date range | 20160104..20260525 |
| Duplicate keys | 0 |
| Buy executable rate | 0.994694 |
| Sell executable rate | 0.996316 |
| Next-open return coverage | 0.997942 |
| Next-VWAP return coverage | 0.997942 |

Canonical fields added on top of close-to-close research labels:

```text
label_rel_return_close_to_close5
next_open_return_5d
next_vwap_return_5d
execution_excess_open_to_close5
benchmark_next_open_return_5d
buy_executable
sell_executable
next_open
next_vwap
next_amount
next_vol
next_is_limit_up
next_is_limit_down
```

Interpretation:

The execution-label warning is now addressed at schema level. The canonical label table is aligned to the original research universe, so it has 242,938 rows rather than the full-market execution-label table's 10.7 million rows.

### Canonical PIT Audit

Command:

```bash
conda activate dl_env
python scripts/audit_point_in_time.py \
  --labels data/mart/labels/labels_canonical_v20260526.parquet \
  --out-dir outputs/audit/point_in_time_canonical_labels
```

Generated outputs:

```text
outputs/audit/point_in_time_canonical_labels/field_audit.csv
outputs/audit/point_in_time_canonical_labels/feature_column_audit.csv
outputs/audit/point_in_time_canonical_labels/negative_shift_audit.csv
outputs/audit/point_in_time_canonical_labels/suspect_features.txt
outputs/audit/point_in_time_canonical_labels/leakage_findings.md
```

Audit result:

| Metric | Before | After canonical labels |
| --- | ---: | ---: |
| Blockers | 0 | 0 |
| Warnings | 4 | 1 |
| Pass checks | 9 | 10 |

Resolved or downgraded findings:

| Finding | New status | Reason |
| --- | --- | --- |
| `LBL003` execution labels missing | PASS | Canonical table contains next-open/VWAP returns, executable flags, and fillability fields. |
| `MSK002` same-day strict mask | INFO | Strict mask is documented as a conservative sample filter; T+1 executable flags now drive production-like backtests. |
| `CODE002` close-to-close label | INFO | Close-to-close labels remain acceptable as supervised targets when execution-aware labels drive evaluation. |

Remaining warning:

| Finding | Status | Interpretation |
| --- | --- | --- |
| `DATA003` style/microstructure-sensitive features | WARNING | This cannot be solved by label schema. It requires style attribution, residual alpha regression, and portfolio-level risk control. |

### Capacity And Participation Matrix

Command:

```bash
conda activate dl_env
python scripts/run_capacity_participation_matrix.py \
  --labels data/mart/labels/labels_canonical_v20260526.parquet \
  --output-dir outputs/backtest/t1_fill_sim/capacity_participation_matrix_canonical \
  --k "10,30" \
  --keep-multiplier "1,1.5,3" \
  --portfolio-nav "1000000,10000000,50000000,100000000" \
  --participation-cap "0.01,0.03,0.05,0.10" \
  --cost-bps 10 \
  --slippage-bps 5 \
  --rebalance-stride 5
```

Generated outputs:

```text
outputs/backtest/t1_fill_sim/capacity_participation_matrix_canonical/capacity_participation_periods.csv
outputs/backtest/t1_fill_sim/capacity_participation_matrix_canonical/capacity_participation_summary.csv
outputs/backtest/t1_fill_sim/capacity_participation_matrix_canonical/manifest.json
```

Matrix size:

| Metric | Value |
| --- | ---: |
| Period rows | 15,648 |
| Summary rows | 192 |
| NAV grid | 1m, 10m, 50m, 100m |
| Participation grid | 1%, 3%, 5%, 10% |
| K grid | 10, 30 |
| Keep grid | 1.0x, 1.5x, 3.0x |

Key capacity results for diagnostic candidates:

| Candidate | Split | Case | NAV | Participation | Net ann. | Excess vs benchmark ann. | Excess vs exec universe ann. | Max drawdown | Avg filled turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K10 keep=1.5x | test | Best exec excess | 1m | 5% | 0.8994 | 0.1980 | 0.3211 | -0.2041 | 0.6925 |
| K10 keep=1.5x | test | Worst exec excess | 100m | 1% | 0.6028 | 0.0170 | 0.1193 | -0.1996 | 0.0055 |
| K10 keep=1.5x | validation | Best exec excess | 1m | 10% | -0.1025 | -0.0940 | -0.0189 | -0.4875 | 0.6729 |
| K30 keep=3x | validation | Best exec excess | 50m | 1% | -0.1312 | -0.1214 | -0.0504 | -0.5207 | 0.0043 |
| K30 keep=1x | validation | Best exec excess | 100m | 1% | -0.1349 | -0.1277 | -0.0574 | -0.5248 | 0.0041 |

Capacity interpretation:

- K10 keep=1.5x remains test-positive across the matrix, but the validation split is negative even in its best capacity/participation setting.
- K30 keep=3x has lower turnover and wider basket, but validation still fails with large drawdown.
- K30 keep=1x does not justify promotion: validation remains negative and test performance is weaker than K10 keep=1.5x.
- Very low filled turnover at high NAV/low participation shows that capacity constraints can turn the strategy into a stale-holding portfolio. Positive test returns under this condition should not be interpreted as clean alpha.

PM decision after this run:

> Canonical label schema and PIT audit quality improved materially, but the strategy remains a research prototype. Capacity sensitivity does not rescue validation. Further model tuning is not the next bottleneck; residual style alpha and walk-forward robustness are.

## Barra-lite Residual Alpha And Optimizer Run 2026-05-29

### Residual Alpha Audit

Command:

```bash
conda activate dl_env
python scripts/audit_barra_lite_residual_alpha.py
```

Generated outputs:

```text
outputs/audit/barra_lite_residual_alpha/daily_residual_ic.csv
outputs/audit/barra_lite_residual_alpha/decile_returns.csv
outputs/audit/barra_lite_residual_alpha/residual_summary.csv
outputs/audit/barra_lite_residual_alpha/residual_alpha_findings.md
outputs/audit/barra_lite_residual_alpha/manifest.json
```

Target:

```text
execution_excess_open_to_close5
```

Residual IC summary:

| Split | Control set | Mean raw IC | Mean residual IC | Score coef after controls | Positive residual IC rate |
| --- | --- | ---: | ---: | ---: | ---: |
| test | none | 0.045503 | 0.045503 | 0.002323 | 0.6292 |
| test | size | 0.045503 | 0.047406 | 0.002340 | 0.6353 |
| test | size_liquidity | 0.045503 | 0.068503 | 0.004088 | 0.6748 |
| test | full_style | 0.045503 | 0.042046 | 0.002969 | 0.6140 |
| test | industry_proxy_full_style | 0.045503 | 0.057947 | 0.003734 | 0.6535 |
| validation | none | 0.011288 | 0.011288 | 0.000289 | 0.5289 |
| validation | size | 0.011288 | 0.012128 | 0.000281 | 0.5310 |
| validation | size_liquidity | 0.011288 | 0.009563 | 0.000454 | 0.5269 |
| validation | full_style | 0.011288 | 0.004680 | 0.000476 | 0.5000 |
| validation | industry_proxy_full_style | 0.011288 | 0.001508 | 0.000227 | 0.4917 |

Interpretation:

- Test residual IC remains strong after style controls, and even improves under size/liquidity or industry-proxy controls.
- Validation raw IC is already weak; after full-style or industry-proxy full-style controls, residual IC nearly disappears.
- This is strong evidence that the test-period score is not the same object as the validation-period score. The model is partly capturing a regime-specific style/liquidity structure rather than stable residual alpha.
- The correct next research question is no longer "can full62 predict in test?", but "can residual alpha survive across folds after style controls?"

### Barra-lite Optimizer Experiment

Command:

```bash
conda activate dl_env
python scripts/optimize_portfolio.py \
  --output-dir outputs/backtest/optimizer/barra_lite_full62 \
  --risk-control "none,industry_proxy,industry_size,industry_size_liquidity_vol_mom" \
  --k "10,30" \
  --style-penalty "0,0.05,0.10,0.20" \
  --turnover-penalty "0,0.02" \
  --portfolio-nav 10000000 \
  --participation-cap 0.03 \
  --cost-bps 10 \
  --slippage-bps 5 \
  --rebalance-stride 5
```

Generated outputs:

```text
outputs/backtest/optimizer/barra_lite_full62/optimizer_periods.csv
outputs/backtest/optimizer/barra_lite_full62/optimizer_summary.csv
outputs/backtest/optimizer/barra_lite_full62/manifest.json
```

Method:

This first optimizer is a transparent Barra-lite constrained re-ranker, not a full QP optimizer. It uses:

- model score as alpha;
- industry-proxy and style exposures as soft penalties;
- turnover penalty for names not already held;
- T+1 canonical executable labels;
- 10m NAV and 3% participation cap;
- 10 bps cost plus 5 bps slippage.

Best test rows:

| Split | Risk control | K | Style penalty | Turnover penalty | Net ann. | Excess vs benchmark ann. | Excess vs exec universe ann. | Max DD | Full-style exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | industry_proxy | 10 | 0.20 | 0.02 | 0.7134 | 0.0891 | 0.1991 | -0.1981 | 0.4112 |
| test | industry_proxy | 10 | 0.10 | 0.02 | 0.7106 | 0.0864 | 0.1966 | -0.1976 | 0.4068 |
| test | industry_proxy | 10 | 0.10 | 0.00 | 0.7104 | 0.0863 | 0.1965 | -0.1976 | 0.4067 |

Best validation rows:

| Split | Risk control | K | Style penalty | Turnover penalty | Net ann. | Excess vs benchmark ann. | Excess vs exec universe ann. | Max DD | Full-style exposure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | industry_proxy | 30 | 0.20 | 0.02 | -0.1946 | -0.1822 | -0.1133 | -0.5504 | 0.2851 |
| validation | industry_proxy | 30 | 0.10 | 0.02 | -0.1950 | -0.1827 | -0.1138 | -0.5497 | 0.2898 |
| validation | industry_proxy | 30 | 0.20 | 0.00 | -0.1957 | -0.1834 | -0.1147 | -0.5495 | 0.2862 |

Control-effect diagnostic:

| Split | Portfolio | Excess vs exec universe ann. | Full-style exposure |
| --- | --- | ---: | ---: |
| validation | K30 none, turnover penalty 0.02 | -0.1216 | 0.2887 |
| validation | K30 industry_proxy, style 0.20, turnover 0.02 | -0.1133 | 0.2851 |
| validation | K30 industry_size_liquidity_vol_mom, style 0.20, turnover 0.02 | -0.1198 | 0.2858 |

Optimizer conclusion:

- Soft style penalties do not materially reduce full-style exposure.
- Validation improves only slightly versus the unconstrained K30 row, and remains deeply negative.
- Test remains strong under K10 industry-proxy control, but this is not enough because validation failure persists.
- The current re-ranker is useful as a diagnostic baseline, but it is not a sufficient portfolio optimizer.

PM verdict after residual and optimizer experiments:

> The full62 score contains test-period residual signal, but that residual signal is not stable in validation after broad style controls. Soft portfolio penalties are too weak to solve the problem. The next optimizer must use hard exposure caps or a real QP formulation; otherwise, continued tuning is mostly curve fitting around a regime break.

Next required experiment:

| Priority | Experiment | Requirement |
| ---: | --- | --- |
| 1 | Hard-constrained exposure optimizer | Explicit caps on size, liquidity, turnover, volatility, momentum, beta, and industry-proxy exposures versus universe. |
| 2 | Residualized alpha portfolio | Use residualized score from daily cross-sectional regression, not raw score, then apply the same hard caps. |
| 3 | Walk-forward folds | Repeat residual IC and optimizer evaluation across purged folds, not just the current validation/test split. |
| 4 | Post-fill holdings attribution | Attribute realized weights after partial fills, not just target or selected names. |

## Hard-Constrained Residual Portfolio Run 2026-05-29

### Residualized Score Construction

Command:

```bash
conda activate dl_env
python scripts/build_residualized_predictions.py \
  --control-set industry_proxy_full_style \
  --output outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/predictions.parquet
```

Generated outputs:

```text
outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/predictions.parquet
outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/manifest.json
```

Residualization controls:

```text
lag1_industry_turnover_rank
lag1_industry_amount_rank
lag1_industry_pb_rank
lag1_industry_mv_rank
lag1_log_circ_mv
lag1_log_total_mv
lag1_amount_log
lag1_amount_rank_pct
lag1_turnover_rate_f
lag1_ret_20d_std
lag1_ret_20d
lag1_beta_60d
lag1_pb_winsor
```

Output summary:

| Metric | Value |
| --- | ---: |
| Rows | 72,218 |
| Date count | 813 |
| Score | Daily cross-sectional residualized `pred_score` |
| Raw score retained | `pred_score_raw` |

### Hard-Constrained Optimizer

Commands:

```bash
conda activate dl_env
python scripts/optimize_portfolio_hard_constraints.py \
  --predictions outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet \
  --output-dir outputs/backtest/optimizer/hard_constraints_raw_full62 \
  --style-set "industry_proxy,industry_size,full_style" \
  --exposure-cap "0.15,0.25,0.35,0.50" \
  --k "10,30" \
  --portfolio-nav 10000000 \
  --participation-cap 0.03 \
  --cost-bps 10 \
  --slippage-bps 5 \
  --rebalance-stride 5

python scripts/optimize_portfolio_hard_constraints.py \
  --predictions outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_resid_industry_proxy_full_style/predictions.parquet \
  --output-dir outputs/backtest/optimizer/hard_constraints_resid_full62 \
  --style-set "industry_proxy,industry_size,full_style" \
  --exposure-cap "0.15,0.25,0.35,0.50" \
  --k "10,30" \
  --portfolio-nav 10000000 \
  --participation-cap 0.03 \
  --cost-bps 10 \
  --slippage-bps 5 \
  --rebalance-stride 5
```

Generated outputs:

```text
outputs/backtest/optimizer/hard_constraints_raw_full62/hard_constraint_periods.csv
outputs/backtest/optimizer/hard_constraints_raw_full62/hard_constraint_summary.csv
outputs/backtest/optimizer/hard_constraints_raw_full62/manifest.json

outputs/backtest/optimizer/hard_constraints_resid_full62/hard_constraint_periods.csv
outputs/backtest/optimizer/hard_constraints_resid_full62/hard_constraint_summary.csv
outputs/backtest/optimizer/hard_constraints_resid_full62/manifest.json
```

Method:

- Greedy hard-constrained selector over the top score candidate pool.
- Checks equal-weight target exposure against same-day universe z-scored style exposures.
- T+1 canonical execution labels, 10m NAV, 3% participation cap, 10 bps cost plus 5 bps slippage.
- This is still not a true QP optimizer. It is a hard-constraint diagnostic bridge.

Best rows by split and alpha:

| Alpha | Split | Style set | K | Exposure cap | Net ann. | Excess vs exec universe ann. | Max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | test | full_style | 10 | 0.50 | 0.6701 | 0.1678 | -0.1868 |
| raw | validation | industry_proxy | 30 | 0.50 | -0.0363 | 0.0517 | -0.4288 |
| residualized | test | industry_proxy | 30 | 0.25 | 0.6180 | 0.1334 | -0.2059 |
| residualized | validation | industry_size | 30 | 0.25 | -0.0222 | 0.0654 | -0.3978 |

Top validation rows:

| Alpha | Style set | K | Cap | Net ann. | Excess vs benchmark ann. | Excess vs exec universe ann. | Max DD | Avg max target exposure z | Fallback rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residualized | industry_size | 30 | 0.25 | -0.0222 | -0.0180 | 0.0654 | -0.3978 | 0.1774 | 0.9794 |
| residualized | industry_proxy | 30 | 0.25 | -0.0273 | -0.0239 | 0.0591 | -0.3940 | 0.1601 | 0.9691 |
| residualized | industry_proxy | 30 | 0.15 | -0.0282 | -0.0251 | 0.0577 | -0.3909 | 0.1607 | 1.0000 |
| residualized | industry_size | 30 | 0.15 | -0.0292 | -0.0261 | 0.0566 | -0.3929 | 0.1767 | 1.0000 |
| raw | industry_proxy | 30 | 0.50 | -0.0363 | -0.0314 | 0.0517 | -0.4288 | 0.2295 | 0.2268 |

Interpretation:

- This is the first portfolio-construction result where validation excess versus executable universe turns positive in a meaningful way.
- The improvement comes from the combination of residualized score and hard style/industry-size exposure selection, not from raw full62 alone.
- Absolute validation return is still slightly negative, and max drawdown remains too large for production.
- The high fallback rate shows that the greedy hard-constraint selector often cannot fill K=30 under strict caps without relaxing the sequential selection path. This is an engineering warning: a proper optimizer should solve the feasible set globally rather than greedily.
- Test performance declines versus the unconstrained raw K10 result, which is acceptable if validation robustness improves.

PM verdict:

> The project has moved from "proxy alpha fails validation" to "residualized alpha plus hard exposure control may contain a usable research signal." It is still not production-ready, but this is now the most promising branch. The next step is not another GRU ablation; it is a real feasible-set optimizer and walk-forward robustness test.

Next required step:

| Priority | Experiment | Why |
| ---: | --- | --- |
| 1 | Replace greedy hard selector with QP or linear constrained optimizer. | Current fallback rate is too high; feasibility should be solved globally. |
| 2 | Allow underinvestment/cash when hard caps cannot fill K names. | Avoid forced fallback that weakens the meaning of hard constraints. |
| 3 | Run purged walk-forward folds for residualized score + hard constraints. | Confirm the validation improvement is not one split artifact. |
| 4 | Attribute post-fill realized holdings. | Verify actual filled weights retain lower style exposure after partial fills. |
