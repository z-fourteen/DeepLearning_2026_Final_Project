from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter an existing prediction parquet by the strict tradable mask."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet",
        help="Input predictions.parquet.",
    )
    parser.add_argument(
        "--filter-log",
        default=(
            "data/mart/datasets/"
            "dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv"
        ),
        help="Strict tradable filter log with trade_date, ts_code, split, strict_tradable.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay",
        help="Output directory for filtered predictions and manifest.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    filter_log_path = Path(args.filter_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(predictions_path).copy()
    filter_log = pd.read_csv(filter_log_path)

    for frame in (predictions, filter_log):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["split"] = frame["split"].astype(str)

    strict_keys = filter_log.loc[
        filter_log["strict_tradable"].astype(bool),
        ["trade_date", "ts_code", "split"],
    ].drop_duplicates()

    filtered = predictions.merge(
        strict_keys,
        on=["trade_date", "ts_code", "split"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["split", "trade_date", "ts_code"])

    output_path = output_dir / "predictions.parquet"
    filtered.to_parquet(output_path, index=False)

    manifest = {
        "method": "strictmask_prediction_overlay",
        "input_predictions": str(predictions_path),
        "filter_log": str(filter_log_path),
        "output_predictions": str(output_path),
        "input_rows": int(len(predictions)),
        "strict_key_rows": int(len(strict_keys)),
        "output_rows": int(len(filtered)),
        "input_split_counts": predictions["split"].value_counts().to_dict(),
        "output_split_counts": filtered["split"].value_counts().to_dict(),
        "date_counts": filtered.groupby("split")["trade_date"].nunique().to_dict(),
    }
    (output_dir / "overlay_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
