# Data And Label Protocol

Canonical labels and execution labels live under `data/mart/labels/`.

The clean pipeline separates three concepts:

- model target labels for supervised GRU training
- strict tradable masks for tensor construction
- T+1 execution labels for realistic fill simulation

The key engineering rule is that model inputs must be point-in-time, while
execution feasibility is evaluated after prediction through sidecar labels and
the T+1 fill simulator.
