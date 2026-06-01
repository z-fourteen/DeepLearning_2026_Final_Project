# 回测与执行

最终主线执行入口：

```powershell
conda run -n dl_env python scripts/portfolio/run_final_mainline_optimizer.py
```

冻结参数来自：

```text
configs/portfolio/final_mainline_optimizer.yaml
```

最终 optimizer：

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
```

核心执行产物：

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution/
```

optimizer 会评估 T+1 买卖可执行性、容量约束、交易成本、滑点、换手率、现金权重、最低仓位规则以及相对基准/可执行域收益。
