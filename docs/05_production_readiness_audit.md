# 生产就绪审计

主审计入口：

```powershell
python scripts/audit/audit_point_in_time.py
python scripts/audit/audit_barra_lite_residual_alpha.py
conda run -n dl_env python scripts/portfolio/run_final_mainline_optimizer.py
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution --split validation --top-n 6
```

审计流程重点关注：

- point-in-time 特征构建
- alpha 特征与风险控制的严格分离
- 风格控制后的 residual alpha 存续性
- T+1 执行可行性和容量敏感性
- epoch 12 `checkpoint_score` 选择是否与最终 optimizer 证据一致

最终冻结配置：

```text
configs/portfolio/final_mainline_optimizer.yaml
```

长版审计路线图已归档在：

```text
docs/archive/production_readiness_audit_roadmap.md
```
