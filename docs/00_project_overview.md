# Project Overview

This project implements a GRU-based time-series stock selection workflow for
ChiNext equities. The repository has been sanitized around the current
`clean_dataset` mainline.

Final mainline:

1. Build point-in-time clean tensors from `advanced_sequence_clean_v1`.
2. Train GRU sequence models on strict tradable samples.
3. Evaluate predictions with T+1 open-fill simulation.
4. Attach capacity and residual-alpha audits before portfolio deployment.

Historical `full62` experiments are frozen under `legacy/legacy_full62_v1/`.
They remain useful as research evidence, but they are not production entry
points.
