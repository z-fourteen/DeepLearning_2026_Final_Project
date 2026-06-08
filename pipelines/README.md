# Pipelines

This directory contains core pipeline implementations called by scripts. Users should run the command-line entry points under `scripts/`; pipeline modules are maintained as library code.

## Layout

| Path | Purpose |
| --- | --- |
| `pipelines/ingest/` | Raw data ingestion into the local lake. |
| `pipelines/pool/` | ChiNext universe and pool construction. |
| `pipelines/state/` | Security daily state, tradability, listing, suspension, and price-limit logic. |
| `pipelines/mart/` | Feature, label, clean dataset, and tensor builders. |

The root `README.md` documents the end-to-end reproduction commands.
