import pandas as pd
pred = pd.read_parquet("outputs/runs/transformer_l20_clean_alpha_only_purgedwf/predictions.parquet")
labels = pd.read_parquet("data/mart/labels/execution_labels_v20260526.parquet")

print("Pred dates:", pred["trade_date"].min(), "-", pred["trade_date"].max(), "n:", pred["trade_date"].nunique())
print("Label dates:", labels["trade_date"].min(), "-", labels["trade_date"].max(), "n:", labels["trade_date"].nunique())

bm = labels[["trade_date", "benchmark_next_open_to_exit_close_return"]].dropna()
print("BM dates:", bm["trade_date"].min(), "-", bm["trade_date"].max(), "n:", bm["trade_date"].nunique())

pred_d = set(pred["trade_date"].astype(str).unique())
bm_d = set(bm["trade_date"].astype(str).unique())
print("Overlap:", len(pred_d & bm_d), "Missing BM:", len(pred_d - bm_d))
print("Sample missing:", sorted(pred_d - bm_d)[:10])
