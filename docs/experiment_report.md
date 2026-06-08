# Experiment Report / 实验报告

## English

### Overview

This project studies GRU-based stock selection on the ChiNext universe. The final submitted pipeline builds a point-in-time clean dataset, trains a feature-style interaction GRU, evaluates T+1 execution, and freezes a portfolio optimizer selected on validation evidence.

The root `README.md` is the only reproduction guide. This report records the experiment design and final findings.

### Data And Universe

| Item | Setting |
| --- | --- |
| universe | ChiNext index universe, `399006.SZ` |
| data version | `v20260526` |
| historical range | `20160104` to `20260525` |
| split protocol | purged walk-forward |
| label horizon | 5 trading days |
| main dataset | `clean_dataset v20260526` |

Data files are not submitted with the source code.

### Clean Dataset

The clean dataset separates alpha features from execution and risk-control fields. The final model uses:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

| Item | Value |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

### Final Model

```text
model: feature_style_interaction_gru
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
training objective: topk_band_margin_ic
```

Model scale:

| Item | Value |
| --- | ---: |
| train samples | 114,946 |
| validation samples | 40,624 |
| test samples | 26,678 |
| validation dates | 468 |
| prediction rows | 67,302 |
| batch size | 384 |
| batch mode | date |

### Prediction Diagnostics

| split | rows | dates | score_mean | score_std | label_mean | label_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 40,624 | 468 | 0.000809 | 0.010875 | -0.001073 | 0.054722 |
| test | 26,678 | 311 | -0.002766 | 0.012977 | -0.002935 | 0.067726 |

The model output is a ranking score, not a calibrated return forecast.

### T+1 Execution Evidence

Better validation rows remained negative:

| validation setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_10_keep_1.5x | -0.147331 | -0.073154 | -0.364982 | -0.013408 |
| top_10_keep_2x | -0.147857 | -0.071794 | -0.368843 | -0.012154 |
| top_30_keep_1.5x | -0.152696 | -0.061664 | -0.386409 | -0.003156 |

Test T+1 showed stronger absolute returns:

| test setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.052450 |
| top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.021582 |
| top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | -0.036879 |

This indicates test-period stock-selection value, but it does not remove validation weakness.

### Final Optimizer

Frozen optimizer settings:

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
participation_cap: 0.03
portfolio_nav: 10000000
cost_bps: 10
slippage_bps: 5
```

Final optimizer result:

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

Solver health:

| split | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.978723 |
| test | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.984127 |

### Conclusion

The final mainline is a complete reproducible research loop:

```text
clean data -> model selection -> prediction diagnostics -> T+1 execution -> optimizer -> audit evidence
```

It should be described as a reproducible final research submission, not as a production-ready stable alpha strategy.

## 中文

### 概览

本项目研究基于 GRU 的创业板股票选择。最终提交流程构建 point-in-time clean dataset，训练 feature-style interaction GRU，进行 T+1 执行评估，并冻结一个基于 validation 证据选择的组合 optimizer。

根目录 `README.md` 是唯一复现指南。本文记录实验设计和最终结论。

### 数据与股票池

| 项目 | 设置 |
| --- | --- |
| 股票池 | 创业板指数成分，`399006.SZ` |
| 数据版本 | `v20260526` |
| 历史区间 | `20160104` 至 `20260525` |
| 切分协议 | purged walk-forward |
| 标签 horizon | 5 个交易日 |
| 主数据集 | `clean_dataset v20260526` |

数据文件不随源代码提交。

### Clean Dataset

clean dataset 将 alpha 特征与执行、风控字段分离。最终模型使用：

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

| 项目 | 数值 |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

### 最终模型

```text
model: feature_style_interaction_gru
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
training objective: topk_band_margin_ic
```

模型规模：

| 项目 | 数值 |
| --- | ---: |
| train samples | 114,946 |
| validation samples | 40,624 |
| test samples | 26,678 |
| validation dates | 468 |
| prediction rows | 67,302 |
| batch size | 384 |
| batch mode | date |

### 预测诊断

| split | rows | dates | score_mean | score_std | label_mean | label_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 40,624 | 468 | 0.000809 | 0.010875 | -0.001073 | 0.054722 |
| test | 26,678 | 311 | -0.002766 | 0.012977 | -0.002935 | 0.067726 |

模型输出是排序分数，不是校准后的收益幅度预测。

### T+1 执行证据

较好的 validation 行仍为负：

| validation setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_10_keep_1.5x | -0.147331 | -0.073154 | -0.364982 | -0.013408 |
| top_10_keep_2x | -0.147857 | -0.071794 | -0.368843 | -0.012154 |
| top_30_keep_1.5x | -0.152696 | -0.061664 | -0.386409 | -0.003156 |

test T+1 绝对收益较强：

| test setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.052450 |
| top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.021582 |
| top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | -0.036879 |

这说明 test 期存在选股价值，但不能消除 validation 偏弱的问题。

### 最终 Optimizer

冻结 optimizer 设置：

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
participation_cap: 0.03
portfolio_nav: 10000000
cost_bps: 10
slippage_bps: 5
```

最终 optimizer 结果：

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

solver 健康度：

| split | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.978723 |
| test | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.984127 |

### 结论

最终主线是完整可复现研究闭环：

```text
clean data -> model selection -> prediction diagnostics -> T+1 execution -> optimizer -> audit evidence
```

它应描述为可复现最终研究提交，而不是生产就绪的稳定 alpha 策略。
