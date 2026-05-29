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

Freeze the current test result as a historical research observation. Future choices should not be made from 2025-2026 test performance.

Immediate actions:

- Treat `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/` as the frozen score baseline.
- Treat `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/` as the frozen proxy strategy baseline.
- Do not promote additional K or keep-multiplier settings based on test results.
- Use the current test window only once more after the next validation framework is rebuilt.

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

Replace the single train/validation/test interpretation with walk-forward validation.

Proposed folds:

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| F1 | 2016-2019 | 2020 | 2021 |
| F2 | 2016-2020 | 2021 | 2022 |
| F3 | 2016-2021 | 2022 | 2023 |
| F4 | 2016-2022 | 2023 | 2024 |
| F5 | 2016-2023 | 2024 | 2025 |
| F6 | 2016-2024 | 2025 | 2026 |

Use an embargo window at every split boundary. For 5-day labels, start with at least 20 trading days of embargo.

Implementation tasks:

- Add fold definitions to dataset builder.
- Export fold-specific sequence NPZ files.
- Add `--fold` support to `scripts/train_sequence.py`.
- Add a fold summary script.

Suggested commands:

```bash
python scripts/build_model_datasets.py --split-scheme walk_forward --embargo-days 20
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F1
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F2
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F3
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F4
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F5
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda --fold F6
python scripts/summarize_walk_forward.py
```

Acceptance criteria:

- Mean RankIC across folds is positive.
- At least 70% of folds have positive RankIC.
- Portfolio excess is positive in a majority of folds.
- Worst-fold drawdown is acceptable.
- Results do not rely only on 2025-2026.

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
- [ ] Replace proxy backtest with execution-aware simulation.
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
