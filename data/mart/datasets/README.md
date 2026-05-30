# Mart Dataset Layout

This directory keeps the active `clean_dataset` mart artifacts physically separated from legacy experiment outputs.

- `core/`: base point-in-time mart table, generated locally and ignored by Git.
- `clean_purged_wf/`: active clean GRU tensor artifacts for the purged walk-forward split. Only manifest JSON files are tracked.
- Historical LGBM, full62, single-holdout, and ablation datasets live under `legacy/legacy_full62_v1/data_manifests/datasets/`. Their binary artifacts are local only; manifest JSON files remain tracked for auditability.
