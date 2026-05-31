# 生产就绪审计

主审计入口：

```powershell
python scripts/audit/audit_point_in_time.py
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/audit/audit_clean_resid_mainline.py
```

审计流程重点关注：

- point-in-time 特征构建
- alpha 特征与风险控制的严格分离
- 风格控制后的 residual alpha 存续性
- T+1 执行可行性和容量敏感性

长版审计路线图已归档在：

```text
docs/archive/production_readiness_audit_roadmap.md
```
