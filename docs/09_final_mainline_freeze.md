# Final Mainline Freeze / 最终主线冻结记录

## English

Freeze date: 2026-05-31.

This document records the final submitted mainline, evidence chain, and interpretation constraints. Use the root `README.md` for reproduction commands.

### Frozen Mainline

```text
model: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
stop_reason: metric_early_stop:checkpoint_score
```

Final optimizer:

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
```

### Single Entry Points

```text
configs/portfolio/final_mainline_optimizer.yaml
scripts/portfolio/run_final_mainline_optimizer.py
```

### Evidence Chain

| Evidence | Path |
| --- | --- |
| model config | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/config.yaml` |
| model metrics | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/metrics.json` |
| checkpoint | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/model.pt` |
| predictions | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet` |
| optimizer grid | `outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/` |
| final optimizer | `outputs/backtest/optimizer/final_mainline_ckptscore_e12/` |
| closed-loop summary | `outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/` |

These files are generated artifacts and are not tracked by Git.

### Final Results

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

### Interpretation

- Epoch 12 was selected by validation `checkpoint_score`.
- The optimizer improves validation absolute return relative to raw T+1 rows, but it still does not produce positive executable-universe excess in both validation and test.
- `risk_control=none` means `style_penalty=0.1` does not imply active style exposure control in the final optimizer.
- The final result is a reproducible research submission, not a production-ready stable alpha.

## 中文

冻结日期：2026-05-31。

本文记录最终提交主线、证据链和解释边界。复现命令请以根目录 `README.md` 为准。

### 冻结主线

```text
model: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
stop_reason: metric_early_stop:checkpoint_score
```

最终 optimizer：

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
```

### 单一入口

```text
configs/portfolio/final_mainline_optimizer.yaml
scripts/portfolio/run_final_mainline_optimizer.py
```

### 证据链

| 证据 | 路径 |
| --- | --- |
| 模型配置 | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/config.yaml` |
| 模型指标 | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/metrics.json` |
| checkpoint | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/model.pt` |
| predictions | `outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet` |
| optimizer grid | `outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/` |
| final optimizer | `outputs/backtest/optimizer/final_mainline_ckptscore_e12/` |
| 闭环摘要 | `outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/` |

这些文件均为本地生成物，不由 Git 跟踪。

### 最终结果

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

### 解释边界

- epoch 12 由 validation `checkpoint_score` 选择。
- optimizer 相比裸 T+1 行改善了 validation 绝对收益，但仍未在 validation 和 test 上同时取得相对可执行域正超额。
- `risk_control=none` 意味着 `style_penalty=0.1` 在最终 optimizer 中并不代表主动风格暴露控制。
- 最终结果是可复现研究提交，不是生产就绪的稳定 alpha。
