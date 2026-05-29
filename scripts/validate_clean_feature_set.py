from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def overlaps(groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = list(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            common = sorted(set(groups[left_name]) & set(groups[right_name]))
            if common:
                rows.append({"left": left_name, "right": right_name, "features": common, "count": len(common)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cleaned feature-set role config.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/feature_sets/advanced_sequence_clean_v1.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--feature-set", default="advanced_sequence_clean_v1")
    parser.add_argument("--data-version", default="v20260526")
    parser.add_argument(
        "--output",
        default=(
            "outputs/factor_validation/advanced_sequence_fixed/"
            "label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/"
            "advanced_sequence_clean_v1_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    role_config = load_yaml(project_root / args.config)
    features_config = load_yaml(project_root / args.features_config)
    feature_set_config = features_config.get("feature_sets", {}).get(args.feature_set, {})
    selected_features = list(feature_set_config.get("selected_features", []))

    groups = {
        "alpha_features": list(role_config.get("alpha_features", [])),
        "risk_controls": list(role_config.get("risk_controls", [])),
        "tradability_controls": list(role_config.get("tradability_controls", [])),
        "excluded_features": list(role_config.get("excluded_features", [])),
    }
    mart_path = project_root / "data" / "mart" / "datasets" / f"dataset_{args.data_version}.parquet"
    mart_columns = set(pd.read_parquet(mart_path).columns)
    all_config_features = sorted(set().union(*(set(values) for values in groups.values())))
    missing_in_mart = sorted(feature for feature in all_config_features if feature not in mart_columns)
    selected_mismatch = sorted(set(selected_features) ^ set(groups["alpha_features"]))
    group_overlaps = overlaps(groups)
    summary = {
        "name": role_config.get("name"),
        "status": role_config.get("status"),
        "feature_set": args.feature_set,
        "data_version": args.data_version,
        "counts": {name: len(values) for name, values in groups.items()},
        "selected_features_count": len(selected_features),
        "selected_features_match_alpha": not selected_mismatch,
        "selected_mismatch": selected_mismatch,
        "group_overlaps": group_overlaps,
        "missing_in_mart": missing_in_mart,
        "valid": not selected_mismatch and not group_overlaps and not missing_in_mart,
    }
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
