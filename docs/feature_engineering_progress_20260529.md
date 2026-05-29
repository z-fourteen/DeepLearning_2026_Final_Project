# Feature Engineering Progress - 2026-05-29

## Scope

This report tracks the feature-engineering transition after the GRU head was frozen to the LeakyReLU slope 0.005 variant. The current goal is to move from model-head selection to feature validity, residual alpha, tradability controls, and strict universe filtering.

## Current Status

### 1. Neutralized IC / RankIC

Status: completed for `advanced_sequence_fixed`.

Implemented and run:

```bash
conda run -n dl_env python scripts/run_factor_validation.py --data-version v20260526 --feature-set advanced_sequence_fixed --stage neutralized --skip-quantile --skip-extended-quantile --neutralized-jobs 4 --neutralized-chunk-size 16
```

Output directory:

```text
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/
```

Important outputs:

```text
factor_ic_rankic.csv
factor_neutralized_ic.csv
factor_neutralization_decay.csv
factor_validation_summary.json
validation_profile.json
```

Run profile:

| Item | Value |
| --- | ---: |
| Rows | 241,643 |
| Trade dates | 2,516 |
| Stocks | 259 |
| Advanced features | 62 |
| Neutralized output features | 59 |
| Runtime | 172.6 seconds |
| Neutralized skipped | false |

The three missing neutralized output features are expected because they were used as style exposures or skipped as direct exposure columns.

Neutralization currently controls:

- Industry, joined from `data/lake/core/chinext_pool/chinext_pool_scd2.parquet` when the mart dataset has no `industry` column.
- Size: `lag1_log_circ_mv`, fallback `lag1_log_total_mv`, fallback `lag1_industry_mv_rank`.
- Liquidity: `lag1_turnover_rate_f`, fallback `lag1_turnover_rate`, `lag1_turnover_20d_mean`, `lag1_turnover_60d_mean`, `lag1_amount_log`, `lag1_amount_rank_pct`.
- Volatility: `lag1_ret_20d_std`, fallback `lag1_ret_60d_std`, `lag1_amplitude`, `lag1_volume_ratio`.
- Momentum: `lag1_ret_20d_mean`, fallback `lag1_ret_60d_mean`, `lag1_ret_20d`, `lag1_ret_5d_mean`, `lag1_ret_5d`.

Top residual RankIC features:

| Feature | Neutralized RankIC | RankIC t-stat | Interpretation |
| --- | ---: | ---: | --- |
| `lag1_limit_touch_up__neutral` | 0.02111 | 4.80 | Strong residual signal, but should be risk/tradability-gated before alpha use. |
| `lag1_near_limit_up_2pct__neutral` | 0.02021 | 4.48 | Strong residual signal, same caution as above. |
| `lag1_bollinger_z_20d__neutral` | -0.01664 | -7.74 | Residual short-term reversal / overextension signal. |
| `lag1_ret_5d_mean__neutral` | -0.01653 | -7.50 | Residual short-horizon reversal signal. |
| `lag1_excess_ret_5d_mean__neutral` | -0.01640 | -7.46 | Residual benchmark-relative reversal signal. |
| `lag1_ret_5d__neutral` | -0.01538 | -7.01 | Residual short-horizon reversal signal. |
| `lag1_amount_rank_pct__neutral` | -0.01360 | -6.83 | Still carries residual information, but belongs to liquidity/risk-control review. |
| `lag1_price_to_ma20__neutral` | -0.01352 | -6.18 | Residual overextension signal. |
| `lag1_excess_ret_10d_mean__neutral` | -0.01134 | -5.31 | Medium-short residual reversal signal. |
| `lag1_amount_log__neutral` | -0.00974 | -4.90 | Liquidity/amount residual remains, should be constrained. |

Important decay observations:

- Raw turnover proxies were among the strongest raw RankIC signals:
  - `lag1_turnover_cost_proxy`: raw RankIC -0.04365.
  - `lag1_turnover_rate`: raw RankIC -0.04356.
  - `lag1_turnover_rate_f`: raw RankIC -0.04279.
- After neutralization, the strongest residuals shift away from pure turnover and toward:
  - limit-state / near-limit signals,
  - short-term reversal,
  - amount/liquidity residuals,
  - selected money-flow strength features.
- `lag1_month__neutral` has high absolute mean RankIC but weak t-stat, so it should not be promoted as a robust alpha feature.

Neutralization decay table:

Status: completed.

Output:

```text
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/factor_neutralization_decay.csv
```

The table compares raw IC/RankIC with neutralized IC/RankIC and includes:

- raw and neutralized IC mean,
- raw and neutralized RankIC mean,
- absolute decay,
- retention ratio,
- raw and neutralized t-stat,
- positive RankIC ratio,
- residual signal class.

Residual signal class counts:

| Class | Count | Meaning |
| --- | ---: | --- |
| `strong_residual` | 6 | Residual RankIC remains strong after neutralization. |
| `moderate_residual` | 5 | Residual signal remains usable but should be reviewed with role constraints. |
| `review` | 14 | Mixed evidence or unstable statistics. |
| `mostly_style_or_weak` | 34 | Mostly explained by neutralized exposures or too weak after controls. |

Top decay-table rows by residual RankIC strength:

| Feature | Raw RankIC | Neutralized RankIC | Retention | Class |
| --- | ---: | ---: | ---: | --- |
| `lag1_limit_touch_up` | -0.02068 | 0.02111 | 1.021 | `strong_residual` |
| `lag1_near_limit_up_2pct` | -0.01873 | 0.02021 | 1.079 | `strong_residual` |
| `lag1_month` | 0.02189 | -0.01932 | 0.883 | `review` |
| `lag1_bollinger_z_20d` | -0.02552 | -0.01664 | 0.652 | `strong_residual` |
| `lag1_ret_5d_mean` | -0.02776 | -0.01653 | 0.596 | `strong_residual` |
| `lag1_excess_ret_5d_mean` | -0.02775 | -0.01640 | 0.591 | `strong_residual` |
| `lag1_ret_5d` | -0.02655 | -0.01538 | 0.579 | `strong_residual` |
| `lag1_amount_rank_pct` | -0.02659 | -0.01360 | 0.512 | `moderate_residual` |
| `lag1_price_to_ma20` | -0.02541 | -0.01352 | 0.532 | `moderate_residual` |
| `lag1_excess_ret_10d_mean` | -0.02649 | -0.01134 | 0.428 | `review` |

### 2. Turnover / Amount / Size / Volatility Residualization

Status: partially completed.

Completed:

- Residualized all evaluated advanced features against industry plus size, liquidity, volatility, and momentum exposures.
- The neutralized IC table provides the first residual-alpha evidence layer.

Still required:

- Create explicit residualized feature columns for training, not just validation-time residual IC.
- Split raw style-control features into:
  - alpha-eligible residualized variants,
  - risk/tradability controls,
  - excluded raw style exposures.
- Decide whether direct exposure columns such as turnover, amount, size, and volatility should be excluded from GRU inputs or retained only as controls.

### 2A. Feature Role Tags

Status: completed for `advanced_sequence_fixed`.

Role table outputs:

```text
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/feature_role_tags_advanced_sequence_fixed.csv
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/feature_role_tags_advanced_sequence_fixed.csv
```

Role distribution:

| Role | Count | Intended use |
| --- | ---: | --- |
| `alpha` | 16 | Candidate model input features after collinearity pruning. |
| `risk_control` | 13 | Exposure controls, residualization inputs, diagnostics, or portfolio constraints. |
| `tradability_control` | 23 | Executability, liquidity, limit-lock, and stock-pool filtering controls. |
| `exclude` | 10 | Weak, unstable, redundant, or low-economic-justification features. |

Alpha candidates:

```text
lag1_net_mf_strength_20d_mean
lag1_net_mf_strength_60d_mean
lag1_close_position
lag1_excess_ret_10d_mean
lag1_excess_ret_1d
lag1_excess_ret_5d_mean
lag1_industry_neutral_ret_1d
lag1_ret_1d
lag1_ret_20d
lag1_ret_5d
lag1_ret_5d_mean
lag1_bollinger_z_20d
lag1_ma_ratio_20_60
lag1_macd_hist
lag1_price_to_ma20
lag1_rsi_14d
```

Risk controls:

```text
lag1_amplitude
lag1_beta_20d
lag1_beta_60d
lag1_industry_pb_rank
lag1_pb_winsor
lag1_pe_ttm_winsor
lag1_ps_ttm_winsor
lag1_ret_20d_std
lag1_ret_60d_std
lag1_vol_log
lag1_industry_mv_rank
lag1_log_circ_mv
lag1_log_total_mv
```

Tradability controls:

```text
lag1_amount_20d_mean
lag1_amount_5d_mean
lag1_amount_log
lag1_amount_rank_pct
lag1_gap_open
lag1_industry_amount_rank
lag1_industry_turnover_rank
lag1_limit_position
lag1_limit_touch_down
lag1_limit_touch_up
lag1_near_limit_down_2pct
lag1_near_limit_up_2pct
lag1_net_mf_amount_to_amount
lag1_turnover_10d_mean
lag1_turnover_10d_std
lag1_turnover_20d_mean
lag1_turnover_20d_std
lag1_turnover_5d_mean
lag1_turnover_60d_mean
lag1_turnover_60d_std
lag1_turnover_cost_proxy
lag1_turnover_rate
lag1_turnover_rate_f
```

Excluded:

```text
lag1_month
lag1_weekday
lag1_large_order_imbalance
lag1_main_mf_strength
lag1_net_mf_strength_10d_mean
lag1_net_mf_strength_5d_mean
lag1_excess_ret_20d_mean
lag1_industry_neutral_ret_20d
lag1_ma_ratio_5_20
lag1_macd_diff
```

Interpretation:

- The alpha list is intentionally narrow and still requires collinearity pruning.
- Limit-touch and near-limit features remain strong residual signals, but are assigned to `tradability_control` because execution feasibility and limit-lock handling matter more than raw IC.
- Turnover and amount features are mostly removed from alpha input status and retained as liquidity/tradability controls.
- Size, beta, volatility, valuation, and amplitude features are treated as risk controls.

### 3. Collinearity Deletion / Merge

Status: completed for alpha candidates.

Generated outputs:

```text
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/alpha_collinearity_pruning_proposal.csv
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/alpha_features_after_collinearity_pruning.csv
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/alpha_collinearity_pruning_proposal.csv
```

Pruning scope:

- Only `alpha` features from `feature_role_tags_advanced_sequence_fixed.csv` were pruned.
- `risk_control` and `tradability_control` features were intentionally excluded from alpha pruning because they will be used as controls, filters, residualization inputs, or portfolio constraints.
- Spearman correlation threshold: 0.85.

Result:

| Action | Count |
| --- | ---: |
| `keep` | 13 |
| `drop_collinear` | 3 |

Dropped alpha features:

| Dropped | Kept representative | Reason |
| --- | --- | --- |
| `lag1_ret_5d` | `lag1_ret_5d_mean` | Spearman corr 0.9897; kept feature has stronger residual score. |
| `lag1_price_to_ma20` | `lag1_bollinger_z_20d` | Spearman corr 0.9420; both describe 20-day overextension. |
| `lag1_rsi_14d` | `lag1_bollinger_z_20d` | Spearman corr 0.8626 with `bollinger_z_20d` and 0.8806 with `price_to_ma20`. |

Kept alpha features after pruning:

```text
lag1_net_mf_strength_20d_mean
lag1_net_mf_strength_60d_mean
lag1_close_position
lag1_excess_ret_10d_mean
lag1_excess_ret_1d
lag1_excess_ret_5d_mean
lag1_industry_neutral_ret_1d
lag1_ret_1d
lag1_ret_20d
lag1_ret_5d_mean
lag1_bollinger_z_20d
lag1_ma_ratio_20_60
lag1_macd_hist
```

Interpretation:

- The alpha candidate set shrank from 16 to 13.
- The largest redundancy was in short-horizon return/reversal and technical overextension features.
- This is now the preferred alpha-only candidate list for the next cleaned feature-set draft.

Existing evidence indicates high redundancy among:

- Turnover family: `turnover_rate`, `turnover_rate_f`, `turnover_cost_proxy`, rolling turnover means/stds.
- Amount family: `amount_log`, `amount_rank_pct`, rolling amount means.
- Momentum / reversal family: `ret_5d`, `ret_5d_mean`, `excess_ret_5d_mean`, moving-average distance features.
- Technical family: MACD variants, Bollinger/MA ratio/price-to-MA.

Next action:

- Combine the 13 pruned alpha features with residualized style families and external control masks.
- Generate a new candidate feature set, likely `advanced_sequence_residual_v1`.

### 3A. Cleaned Feature Set Draft

Status: completed as a draft config.

Generated:

```text
configs/feature_sets/advanced_sequence_clean_v1.yaml
outputs/factor_validation/advanced_sequence_fixed/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/advanced_sequence_clean_v1_summary.json
```

Registered in:

```text
configs/features.yaml
```

Validation command:

```bash
conda run -n dl_env python scripts/validate_clean_feature_set.py
```

Validation result:

| Group | Count |
| --- | ---: |
| `alpha_features` | 13 |
| `risk_controls` | 13 |
| `tradability_controls` | 23 |
| `excluded_features` | 10 |

Checks:

- `selected_features` in `configs/features.yaml` matches `alpha_features`.
- No overlaps across alpha, risk, tradability, and excluded groups.
- All configured features exist in `data/mart/datasets/dataset_v20260526.parquet`.

Important implementation detail:

- `advanced_sequence_clean_v1.selected_features` currently contains only the 13 pruned alpha features.
- Risk controls and tradability controls are intentionally not included in the raw GRU tensor by default.
- Residualized style features are declared as planned under `residualized_style_features`, but the actual residualized columns have not yet been materialized into a dataset.

Current clean-v1 alpha tensor:

```text
lag1_net_mf_strength_20d_mean
lag1_net_mf_strength_60d_mean
lag1_close_position
lag1_excess_ret_10d_mean
lag1_excess_ret_1d
lag1_excess_ret_5d_mean
lag1_industry_neutral_ret_1d
lag1_ret_1d
lag1_ret_20d
lag1_ret_5d_mean
lag1_bollinger_z_20d
lag1_ma_ratio_20_60
lag1_macd_hist
```

Next action:

- Implement residualized feature generation for selected turnover/amount/style families.
- Implement strict tradable mask generation before training this cleaned feature set as the next GRU dataset.

### 4. Tradability / Risk Controls

Status: design decided, implementation pending.

The following should be moved away from raw alpha status:

- `lag1_dist_to_limit_up`
- `lag1_dist_to_limit_down`
- `lag1_near_limit_up_2pct`
- `lag1_near_limit_down_2pct`
- `lag1_limit_touch_up`
- `lag1_limit_touch_down`
- `lag1_limit_position`
- liquidity and amount proxies

Reason:

- Some limit-state features show strong residual IC, but they are tightly coupled to execution feasibility and limit-lock risk.
- They should be used for stock-pool filtering, post-score gating, or portfolio constraints before being trusted as alpha drivers.

Next action:

- Add feature-role metadata: `alpha`, `risk_control`, `tradability_control`, `exclude`.
- Update dataset-building logic so risk controls can be retained for filtering/reporting without automatically entering the GRU alpha input tensor.

### 5. Strict Stock-Pool Filtering

Status: not completed.

Required filters:

- Remove ST and `*ST`.
- Remove long suspensions.
- Remove very low liquidity names, for example average daily amount below a chosen threshold.
- Remove bottom market-cap toxic microcaps.
- Remove locked limit-up / limit-down samples from executable trading labels or portfolio selection.

Current available infrastructure:

- `data/lake/state/security_daily_state.parquet` is configured as the daily state source.
- `chinext_pool_scd2.parquet` provides pool membership and industry.
- Limit-state features already exist in the mart feature table.

Next action:

- Build a strict tradable mask table by `trade_date, ts_code`.
- Apply it to factor validation first, then to sequence dataset construction.
- Record filtering counts by date and reason.

## Recommended Task Order

### P0 - Evidence And Governance

1. Add a neutralization decay report:
   - raw RankIC,
   - neutralized RankIC,
   - absolute decay,
   - retention ratio,
   - t-stat change.
2. Classify advanced features into alpha/risk/tradability/exclude roles.
3. Produce a collinearity pruning proposal using the advanced fixed feature set.

### P1 - Dataset Construction

4. Implement residualized feature generation for turnover, amount, size, and volatility families.
5. Create `advanced_sequence_residual_v1`.
6. Build strict tradable universe masks and filter logs.

### P2 - Model And Portfolio Recheck

7. Train the frozen LeakyReLU slope 0.005 GRU on the cleaned/residualized dataset.
8. Recompute IC, neutralized IC for model `pred_score`, Top-K, and long-short backtest.
9. Compare against the frozen pre-cleaning GRU baseline and document whether residual alpha survives.

## Current Verdict

The project has completed the first residual-alpha audit layer. The result is constructive but not yet sufficient for a tradable claim.

The main conclusion is:

```text
Raw style-heavy signals weaken after neutralization, but several residual effects remain, especially near-limit state and short-term reversal features.
```

The next milestone is not another model tuning pass. It is feature governance: residualized feature construction, collinearity pruning, role separation, and strict tradable-universe filtering.

## 2026-05-29 Update - Clean Dataset Builder

Status: initial builder completed.

New independent entrypoints:

- `pipelines/mart/clean_dataset.py`
- `scripts/build_clean_model_datasets.py`
- `configs/feature_sets/advanced_sequence_clean_v1.yaml`

Design decision:

- The original dataset builder is kept unchanged as the legacy/baseline path.
- The clean builder reads role metadata from `advanced_sequence_clean_v1`.
- Model tensors are separated from controls:
  - `alpha_only`: 13 pruned alpha features only.
  - `alpha_plus_residual_style`: 13 alpha features plus 5 initial residualized style features.
  - risk and tradability controls are exported to sidecar parquet rather than fed directly into the GRU tensor.

Generated datasets:

- `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026.npz`
  - X shape: `(196138, 20, 13)`
  - y shape: `(196138,)`
  - split counts: train 124527, validation 42759, test 28852
- `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026.npz`
  - X shape: `(196138, 20, 18)`
  - y shape: `(196138,)`
  - split counts: train 124527, validation 42759, test 28852

Residualized style features currently included:

- `lag1_turnover_cost_proxy__resid_style`
- `lag1_turnover_20d_std__resid_style`
- `lag1_turnover_60d_std__resid_style`
- `lag1_amount_rank_pct__resid_style`
- `lag1_amount_log__resid_style`

Validation commands:

```bash
conda run -n dl_env python scripts/validate_clean_feature_set.py
conda run -n dl_env python scripts/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
conda run -n dl_env python scripts/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

Remaining before final feature-pool freeze:

- Train frozen-head GRU on both clean datasets.
- Compare `alpha_only` vs `alpha_plus_residual_style` using IC, neutralized IC, RankIC, and Top-K/backtest metrics.
- Audit strict tradable mask thresholds and sensitivity.

## 2026-05-29 Update - Strict Tradable Mask

Status: initial mask implemented in the clean dataset builder.

Policy:

- Strict tradable mask is a sample filter and execution-feasibility control.
- Mask/state columns are not appended to the model input tensor.
- Filter decisions are recorded in manifest and filter-log CSV files.

Current filters:

- Require matching row in `data/lake/state/security_daily_state.parquet`.
- Require `is_tradable`, valid price, valid volume.
- Remove ST / suspended rows through state flags.
- Remove locked limit-up / limit-down execution samples.
- Remove low-liquidity rows using `lag1_amount_20d_mean < 70000` and bottom 5% by date.
- Remove microcap rows using bottom 5% of `lag1_log_circ_mv` by date.

Filter result before sequence windowing:

- Input panel rows: 241643
- Kept rows: 211978
- Dropped rows: 29665
- Drop rate: 12.28%

Reason counts:

- `mask_locked_limit`: 2023
- `mask_low_amount`: 20253
- `mask_microcap`: 12827
- `mask_state_missing`: 0
- `mask_state_not_tradable`: 0
- `mask_st`: 0
- `mask_suspended`: 0
- `mask_price_invalid`: 0
- `mask_volume_invalid`: 0

Outputs:

- `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv`
- `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026_filter_log.csv`

Verification:

- `alpha_only` model tensor remains 13 features.
- `alpha_resid_style` model tensor remains 18 features.
- Sidecar contains `strict_tradable` and mask reason columns.
- Final sidecar rows are all `strict_tradable=True`.

