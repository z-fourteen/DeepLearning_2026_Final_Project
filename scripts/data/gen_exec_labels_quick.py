"""Quick execution labels generator - v2 with benchmark fix."""
import pandas as pd
import numpy as np
import glob, os, json
from pathlib import Path


def main():
    daily_root = Path("A股数据/daily")
    mart_path = Path("data/mart/datasets/core/dataset_v20260526.parquet")
    output = Path("data/mart/labels/execution_labels_v20260526.parquet")
    holding_days = 5

    # Read daily
    print("Reading daily CSV...")
    files = sorted(glob.glob(str(daily_root / "*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["trade_date"] = os.path.basename(f).replace(".csv", "")
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily[
        ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "vwap"]
    ].copy()
    for c in ["open", "high", "low", "close", "vol", "amount", "vwap"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily = (
        daily.drop_duplicates(["trade_date", "ts_code"], keep="last")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    print(f"Daily: {len(daily)} rows")

    # Get benchmark returns from mart dataset
    print("Loading benchmark from mart...")
    mart = pd.read_parquet(mart_path)
    bm_data = (
        mart[["trade_date", "ts_code", "benchmark_future_return"]]
        .drop_duplicates(["trade_date", "ts_code"], keep="last")
    )
    # Each stock has same benchmark per date; just take first
    bm_map = (
        bm_data.sort_values(["trade_date", "ts_code"])
        .groupby("trade_date")["benchmark_future_return"]
        .first()
    )
    print(f"Benchmark dates: {len(bm_map)}, range: {bm_map.index.min()}-{bm_map.index.max()}")

    # Build forward columns
    print("Building execution labels...")
    result = daily.sort_values(["ts_code", "trade_date"]).copy()
    grouped = result.groupby("ts_code", group_keys=False)
    result["next_trade_date"] = grouped["trade_date"].shift(-1)
    result["exit_trade_date"] = grouped["trade_date"].shift(-holding_days)
    result["next_open"] = grouped["open"].shift(-1)
    result["next_amount"] = grouped["amount"].shift(-1)
    result["exit_close"] = grouped["close"].shift(-holding_days)
    result["execution_return_open_to_close5"] = (
        result["exit_close"] / result["next_open"] - 1.0
    )

    # Finalize
    labels = result[
        [
            "trade_date",
            "ts_code",
            "next_trade_date",
            "exit_trade_date",
            "next_open",
            "next_amount",
            "execution_return_open_to_close5",
        ]
    ].copy()
    labels["buy_executable_t1_open"] = True
    labels["sell_executable_t1_open"] = True
    labels["benchmark_next_open_to_exit_close_return"] = (
        labels["trade_date"].astype(str).map(bm_map)
    )
    labels = labels.dropna(
        subset=["next_open", "execution_return_open_to_close5"]
    )
    print(f"Labels: {len(labels)} rows")
    print(f'BM coverage: {labels["benchmark_next_open_to_exit_close_return"].notna().sum()}')

    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output, index=False)

    manifest = {
        "output": str(output),
        "rows": len(labels),
        "trade_dates": int(labels["trade_date"].nunique()),
        "stocks": int(labels["ts_code"].nunique()),
        "date_min": str(labels["trade_date"].min()),
        "date_max": str(labels["trade_date"].max()),
        "holding_days": holding_days,
        "method": "quick_csv_v2_with_mart_bm",
    }
    (output.parent / "execution_labels_v20260526_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
