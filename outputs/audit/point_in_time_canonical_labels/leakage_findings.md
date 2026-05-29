# Point-In-Time Audit Findings

Verdict: `PASS_WITH_WARNINGS`

- Blockers: 0
- Warnings: 1
- Pass checks: 10

## Findings

### DATA003 - WARNING - style_or_microstructure_features_detected

- Component: `mart_dataset`
- Evidence: Suspicious feature count=42. Examples=['lag1_amount_log', 'lag1_vol_log', 'lag1_log_total_mv', 'lag1_log_circ_mv', 'lag1_turnover_rate', 'lag1_turnover_rate_f', 'lag1_volume_ratio', 'lag1_net_mf_amount_to_amount', 'lag1_gap_open', 'lag1_intraday_return', 'lag1_benchmark_ret_1d', 'lag1_industry_turnover_rank', 'lag1_industry_amount_rank', 'lag1_is_limit_up', 'lag1_is_limit_down', 'lag1_has_price_limit', 'lag1_limit_ratio', 'lag1_dist_to_limit_up', 'lag1_dist_to_limit_down', 'lag1_limit_position', 'lag1_near_limit_up_2pct', 'lag1_near_limit_down_2pct', 'lag1_limit_touch_up', 'lag1_limit_touch_down', 'lag1_amount_rank_pct', 'lag1_turnover_cost_proxy', 'lag1_turnover_acceleration', 'lag1_weekday', 'lag1_month', 'lag1_is_month_end', 'lag1_amount_5d_mean', 'lag1_amount_10d_mean', 'lag1_amount_20d_mean', 'lag1_amount_60d_mean', 'lag1_turnover_5d_mean', 'lag1_turnover_10d_mean', 'lag1_turnover_20d_mean', 'lag1_turnover_60d_mean', 'lag1_turnover_5d_std', 'lag1_turnover_10d_std', 'lag1_turnover_20d_std', 'lag1_turnover_60d_std']
- Recommendation: These fields are not necessarily leaked, but they require style, liquidity, and execution attribution before production use.

### CFG002 - INFO - label_policy_declared

- Component: `labels_config`
- Evidence: configs\labels.yaml declares default_horizon=5, label_mode=relative_return.
- Recommendation: Document exact signal timestamp and execution timestamp for this horizon.

### CODE002 - INFO - close_to_close_forward_label

- Component: `label_construction`
- Evidence: Labels are constructed as future close-to-close returns: stock close.shift(-horizon)/close - 1 and benchmark close.shift(-horizon)/close - 1.
- Recommendation: This is acceptable as a supervised target when production-like evaluation uses canonical execution labels and T+1 fill simulation.

### MSK002 - INFO - same_day_limit_filter_documented

- Component: `strict_mask`
- Evidence: filter_log contains mask_locked_limit. In current clean_dataset implementation, this is derived from same trade_date state. Canonical labels now provide next-session buy/sell executable flags for production-like backtests.
- Recommendation: Use strict mask as a conservative sample filter only; use canonical T+1 fields for execution.

### CFG001 - PASS - lag_policy_declared

- Component: `features_config`
- Evidence: configs\features.yaml declares future_shift_allowed=false, dataset_requires_lagged_features_only=true, feature_availability='lag1_close_to_next_session'.
- Recommendation: Keep this policy and verify it against actual dataset columns.

### CFG003 - PASS - controls_separated

- Component: `clean_feature_config`
- Evidence: configs\feature_sets\advanced_sequence_clean_v1.yaml separates raw controls from model tensor and enables strict tradable mask.
- Recommendation: Keep controls out of model input unless a specific ablation opts in.

### CODE001 - PASS - negative_shift_scan_clean

- Component: `source_code`
- Evidence: No unapproved negative shift found in audited mart/backtest scripts.
- Recommendation: Continue scanning new feature scripts before accepting results.

### CODE003 - PASS - lagged_feature_shift_found

- Component: `feature_construction`
- Evidence: add_lagged_features uses grouped[column].shift(1) for lag1_ features.
- Recommendation: Keep raw same-day features out of model-ready datasets.

### DATA001 - PASS - all_feature_columns_lagged

- Component: `mart_dataset`
- Evidence: All 95 feature columns use lag1_ prefix.
- Recommendation: Still verify that lag1_ columns are created with shift(1), not just named lag1_.

### DATA002 - PASS - unique_trade_date_ts_code

- Component: `mart_dataset`
- Evidence: Rows=241643, date_range=20160104..20260518, duplicate_keys=0.
- Recommendation: Keep key uniqueness checks in CI for every dataset rebuild.

### LBL001 - PASS - unique_label_keys

- Component: `labels`
- Evidence: Rows=242938, date_range=20160104..20260525, null_counts={'future_return': 1295, 'benchmark_future_return': 500, 'label_rel_return': 1295}.
- Recommendation: Keep label key uniqueness checks for every rebuild.

### LBL002 - PASS - relative_label_identity_pass

- Component: `labels`
- Evidence: max_abs(future_return - benchmark_future_return - label_rel_return)=0.
- Recommendation: Identity check passes; execution timing still needs separate audit.

### LBL003 - PASS - canonical_execution_labels_present

- Component: `labels`
- Evidence: Canonical execution fields are present. Coverage={'next_open_return_5d': 0.997941861709572, 'next_vwap_return_5d': 0.997941861709572, 'execution_excess_open_to_close5': 0.997941861709572, 'buy_executable_rate': 0.9946941194872766, 'sell_executable_rate': 0.9963159324601338, 'limit_up_rate': 0.004890136578056953, 'limit_down_rate': 0.0032724398817805364}.
- Recommendation: Use this upgraded label table for PIT audit, T+1 fill simulation, and production-like evaluation.

### MSK001 - PASS - filter_keys_unique

- Component: `strict_mask`
- Evidence: Rows=241643, keep_rate=0.8772, mask_rates={'mask_state_missing': 0.0, 'mask_state_not_tradable': 0.0, 'mask_st': 0.0, 'mask_suspended': 0.0, 'mask_price_invalid': 0.0, 'mask_volume_invalid': 0.0, 'mask_locked_limit': 0.008371854347115373, 'mask_low_amount': 0.08381372520619261, 'mask_microcap': 0.05308243979755259}.
- Recommendation: Keep filter-key uniqueness checks for every rebuild.

## Generated Files

- `field_audit.csv`
- `feature_column_audit.csv`
- `negative_shift_audit.csv`
- `suspect_features.txt`
