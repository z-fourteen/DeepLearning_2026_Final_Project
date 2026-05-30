# Production Readiness Audit

Main audit entry points:

```powershell
python scripts/audit/audit_point_in_time.py
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/audit/audit_clean_resid_mainline.py
```

The audit process emphasizes:

- point-in-time feature construction
- strict separation between alpha features and risk controls
- residual alpha survival after style controls
- T+1 execution feasibility and capacity sensitivity

The long-form audit roadmap is archived in:

```text
docs/archive/production_readiness_audit_roadmap.md
```
