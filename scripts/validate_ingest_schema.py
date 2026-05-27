from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.ingest.agent import discover_sources, load_yaml, read_csv, validate_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw CSV schemas before ingestion.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--sample-per-dataset", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    sources = discover_sources(config, PROJECT_ROOT)
    validated_counts: dict[str, int] = {}

    for source in sources:
        count = validated_counts.get(source.dataset, 0)
        if args.sample_per_dataset >= 0 and count >= args.sample_per_dataset:
            continue
        df = read_csv(source, config["source"].get("csv_encoding", "utf-8"))
        validate_schema(df, source, config)
        validated_counts[source.dataset] = count + 1
        print(f"{source.dataset}: OK -> {source.path.name}")

    total = sum(validated_counts.values())
    print(f"validated_files={total}")
    print(f"validated_datasets={len(validated_counts)}")


if __name__ == "__main__":
    main()
