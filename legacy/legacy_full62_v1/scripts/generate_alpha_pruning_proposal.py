from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(
    "outputs/factor_validation/advanced_sequence_fixed/"
    "label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on"
)
DEFAULT_ROLE_PATH = BASE_DIR / "feature_role_tags_advanced_sequence_fixed.csv"
DEFAULT_CORR_PATH = BASE_DIR / "feature_correlation_top.csv"
DEFAULT_DECAY_PATH = BASE_DIR / "factor_neutralization_decay.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "alpha_collinearity_pruning_proposal.csv"
DEFAULT_KEEP_OUTPUT_PATH = BASE_DIR / "alpha_features_after_collinearity_pruning.csv"


def score_feature(row: pd.Series) -> float:
    neutral_rank_ic = abs(float(row.get("neutral_rank_ic_mean", 0) or 0))
    neutral_t = abs(float(row.get("neutral_rank_ic_t_stat", 0) or 0))
    retention = float(row.get("rank_ic_abs_retention", 0) or 0)
    signal_class = str(row.get("residual_signal_class", ""))
    class_bonus = {"strong_residual": 3.0, "moderate_residual": 1.5, "review": 0.5}.get(signal_class, 0.0)
    return neutral_rank_ic * 1000 + neutral_t + retention + class_bonus


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)

    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def build_pruning_table(
    role_table: pd.DataFrame,
    corr_table: pd.DataFrame,
    decay_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    alpha_features = role_table[role_table["role"].eq("alpha")]["feature"].tolist()
    decay = decay_table.set_index("feature") if not decay_table.empty else pd.DataFrame()
    alpha_set = set(alpha_features)

    corr_edges: list[tuple[str, str]] = []
    corr_lookup: dict[tuple[str, str], float] = {}
    for row in corr_table.itertuples(index=False):
        left = str(getattr(row, "feature_left"))
        right = str(getattr(row, "feature_right"))
        corr = float(getattr(row, "spearman_corr"))
        corr_lookup[(left, right)] = corr
        corr_lookup[(right, left)] = corr
        if left in alpha_set and right in alpha_set and abs(corr) >= threshold:
            corr_edges.append((left, right))

    components = connected_components(alpha_features, corr_edges)
    role_meta = role_table.set_index("feature")
    rows: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        scored: list[tuple[str, float]] = []
        for feature in component:
            decay_row = decay.loc[feature] if feature in decay.index else pd.Series(dtype="float64")
            scored.append((feature, score_feature(decay_row)))
        scored = sorted(scored, key=lambda item: item[1], reverse=True)
        keep_feature = scored[0][0]
        for feature, score in scored:
            correlated_with = []
            max_abs_corr = 0.0
            for other in component:
                if other == feature:
                    continue
                corr = corr_lookup.get((feature, other))
                if corr is None:
                    continue
                if abs(corr) >= threshold:
                    correlated_with.append(f"{other}:{corr:.4f}")
                    max_abs_corr = max(max_abs_corr, abs(corr))
            decay_row = decay.loc[feature] if feature in decay.index else pd.Series(dtype="float64")
            action = "keep" if feature == keep_feature else "drop_collinear"
            reason = (
                "highest residual score in correlated alpha group"
                if action == "keep"
                else f"collinear with kept feature {keep_feature}"
            )
            rows.append(
                {
                    "group_id": index,
                    "feature": feature,
                    "action": action,
                    "keep_feature": keep_feature,
                    "category": role_meta.loc[feature, "category"] if feature in role_meta.index else "",
                    "residual_signal_class": decay_row.get("residual_signal_class", ""),
                    "raw_rank_ic_mean": decay_row.get("raw_rank_ic_mean"),
                    "neutral_rank_ic_mean": decay_row.get("neutral_rank_ic_mean"),
                    "neutral_rank_ic_t_stat": decay_row.get("neutral_rank_ic_t_stat"),
                    "rank_ic_abs_retention": decay_row.get("rank_ic_abs_retention"),
                    "score": score,
                    "group_size": len(component),
                    "max_abs_corr_in_group": max_abs_corr,
                    "correlated_with": ";".join(correlated_with),
                    "reason": reason,
                }
            )
    table = pd.DataFrame(rows)
    action_order = {"keep": 0, "drop_collinear": 1}
    table["action_order"] = table["action"].map(action_order)
    return table.sort_values(["group_id", "action_order", "score"], ascending=[True, True, False]).drop(columns=["action_order"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate alpha collinearity pruning proposal.")
    parser.add_argument("--role-path", default=str(DEFAULT_ROLE_PATH))
    parser.add_argument("--corr-path", default=str(DEFAULT_CORR_PATH))
    parser.add_argument("--decay-path", default=str(DEFAULT_DECAY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--keep-output", default=str(DEFAULT_KEEP_OUTPUT_PATH))
    parser.add_argument("--threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    role_table = pd.read_csv(args.role_path)
    corr_table = pd.read_csv(args.corr_path)
    decay_table = pd.read_csv(args.decay_path)
    proposal = build_pruning_table(role_table, corr_table, decay_table, args.threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposal.to_csv(output_path, index=False, encoding="utf-8-sig")
    keep_output_path = Path(args.keep_output)
    keep_output_path.parent.mkdir(parents=True, exist_ok=True)
    proposal[proposal["action"].eq("keep")][["feature", "category", "residual_signal_class", "neutral_rank_ic_mean"]].to_csv(
        keep_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(proposal["action"].value_counts().to_string())
    print(output_path)
    print(keep_output_path)


if __name__ == "__main__":
    main()
