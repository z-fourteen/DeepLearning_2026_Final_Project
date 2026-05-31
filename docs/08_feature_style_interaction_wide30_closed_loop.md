# 特征-风格交互 GRU Wide30 闭环记录

本文件记录的是 L20 wide30 历史候选，不是最终冻结主线。

历史候选运行名称：

```text
feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean
```

本文记录训练后的可执行闭环：预测输出合同、T+1 成交仿真、soft optimizer 网格以及分析交接。

## 产物

训练输出：

```text
outputs/runs/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/
```

执行输出：

```text
outputs/backtest/t1_fill_sim/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/
```

Soft optimizer 输出：

```text
outputs/backtest/optimizer/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean_core80/
```

闭环分析：

```text
outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/
```

主读数：

```text
outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/closed_loop_findings.md
```

## 复现

T+1 开盘成交仿真：

```powershell
conda run -n dl_env python scripts/backtest/backtest_t1_fill_sim.py `
  --predictions outputs/runs/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet `
  --output-dir outputs/backtest/t1_fill_sim/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean `
  --k '10,20,30' `
  --keep-multiplier '1,1.5,2,3' `
  --portfolio-nav 10000000 `
  --participation-cap 0.03 `
  --rebalance-stride 5 `
  --min-daily-count 20
```

Soft optimizer core80 网格：

```powershell
conda run -n dl_env python scripts/portfolio/run_soft_optimizer_grid.py `
  --predictions outputs/runs/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet `
  --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean_core80 `
  --risk-control 'none,industry_size_liquidity_vol_mom' `
  --k '10,20,30' `
  --style-penalty '0,0.1' `
  --turnover-penalty '0,0.02' `
  --exposure-cap '0.15' `
  --min-invested '0.8' `
  --turnover-cap '0.5' `
  --participation-cap '0.03' `
  --single-name-cap '0.1' `
  --exposure-slack-penalty '100' `
  --buy-capacity-slack-penalty '1000' `
  --cash-penalty '0.1' `
  --min-invested-shortfall-penalty '0' `
  --portfolio-nav 10000000 `
  --cost-bps 10 `
  --slippage-bps 5 `
  --rebalance-stride 5 `
  --min-daily-count 40 `
  --solver CLARABEL
```

闭环摘要：

```powershell
conda run -n dl_env python scripts/analysis/summarize_model_closed_loop.py
```

## 关键数值

训练按 validation `rank_ic_mean` 选择 epoch 10。

| 指标 | 数值 |
| --- | ---: |
| validation rank IC mean | 0.040001 |
| validation rank ICIR | 0.261220 |
| validation IC mean | 0.020936 |
| prediction rows | 68,681 |

最佳 T+1 成交仿真行：

| Split | 设置 | 净年化 | 净 IR | 最大回撤 | 相对执行域超额 |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | top_10_keep_1x | -0.112992 | -0.037093 | -0.380032 | 0.042897 |
| test | top_20_keep_1x | 0.547856 | 0.211174 | -0.216987 | 0.058175 |

最佳 soft optimizer core80 行：

| Split | 风险控制 | k | 风格惩罚 | 换手惩罚 | 净年化 | 净 IR | 相对执行域超额 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | industry_size_liquidity_vol_mom | 20 | 0.1 | 0.0 | -0.162305 | -0.056938 | -0.115352 |
| test | none | 10 | 0.0 | 0.02 | 0.516370 | 0.269538 | 0.158297 |

## 解释

训练目标在排序指标层面有效，且更宽的 TopK band 相比更窄的特征-风格交互 run 提升了测试期可执行绝对收益。最佳 test T+1 行为 `top_20_keep_1x`，净年化收益为 54.8%，回撤小于此前窄 topk10 clean run。

主要阻碍是 validation/test 分化。validation 执行收益在 T+1 和 soft-optimizer 两条路径上均为负，而 test 收益较强。因此该结果应被视为有潜力的研究候选，而不是面向生产的晋级版本。测试期绝对收益也仍然部分受到市场 regime 辅助：最佳 T+1 绝对收益行在年化超额收益上仍弱于基准。

## 决策

不将该 run 提升为生产主线。

保留它作为候选架构，并用它指导下一轮受控实验。下一次 run 应直接针对 validation 对齐：降低容量、加强正则化，或在模型选择中显式加入 benchmark/executable-universe excess gate。
