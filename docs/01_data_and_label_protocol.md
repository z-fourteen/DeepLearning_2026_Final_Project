# Data And Label Protocol

Canonical labels and execution labels live under `data/mart/labels/`.

The clean pipeline separates three concepts:

- model target labels for supervised GRU training
- strict tradable masks for tensor construction
- T+1 execution labels for realistic fill simulation

The key engineering rule is that model inputs must be point-in-time, while
execution feasibility is evaluated after prediction through sidecar labels and
the T+1 fill simulator.

The main split contract is `configs/data/splits.yaml`. It uses a purged
walk-forward protocol with a 5-trading-day label horizon, 5 purge days, and a
20-trading-day embargo. The active production fold is declared explicitly in
the config so regenerated tensors record the selected fold in their manifests.
