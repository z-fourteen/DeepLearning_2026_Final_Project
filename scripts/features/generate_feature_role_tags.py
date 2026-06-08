from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_DECAY_PATH = Path(
    "outputs/factor_validation/advanced_sequence_fixed/"
    "label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/"
    "factor_neutralization_decay.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/factor_validation/advanced_sequence_fixed/"
    "label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_off_ext_off_neutral_on/"
    "feature_role_tags_advanced_sequence_fixed.csv"
)


def load_advanced_features(config_path: Path, feature_set: str) -> list[str]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    features = config.get("feature_sets", {}).get(feature_set, {}).get("selected_features", [])
    if not features:
        raise ValueError(f"No selected_features found for feature set: {feature_set}")
    return list(features)


def categorize_feature(feature: str) -> str:
    if any(token in feature for token in ["limit", "amount", "turnover", "volume_ratio", "gap_open"]):
        return "tradability"
    if any(token in feature for token in ["log_circ_mv", "log_total_mv", "industry_mv_rank"]):
        return "size"
    if any(token in feature for token in ["ret_20d_std", "ret_60d_std", "amplitude", "vol_log", "beta_"]):
        return "risk"
    if any(token in feature for token in ["pe_ttm", "pb_winsor", "ps_ttm", "industry_pb_rank"]):
        return "risk"
    if any(token in feature for token in ["weekday", "month"]):
        return "calendar"
    if any(token in feature for token in ["macd_", "rsi_", "ma_ratio", "price_to_ma", "bollinger"]):
        return "technical"
    if any(token in feature for token in ["ret_", "excess_ret", "close_position"]):
        return "return_reversal"
    if any(token in feature for token in ["net_mf", "main_mf", "large_order"]):
        return "moneyflow"
    if feature.startswith("lag1_industry_"):
        return "industry_relative"
    return "other"


def role_from_feature(feature: str, decay: dict[str, Any]) -> tuple[str, str, str]:
    category = categorize_feature(feature)
    signal_class = str(decay.get("residual_signal_class", "missing"))
    neutral_rank_ic = decay.get("neutral_rank_ic_mean")
    neutral_t = decay.get("neutral_rank_ic_t_stat")
    retention = decay.get("rank_ic_abs_retention")

    if category == "tradability":
        return (
            "tradability_control",
            category,
            "Use for executable-universe filtering, limit-lock handling, liquidity gating, or portfolio constraints; do not feed as raw alpha.",
        )
    if category in {"size", "risk"}:
        return (
            "risk_control",
            category,
            "Use as exposure control or residualization input; raw value should not be treated as standalone alpha.",
        )
    if category == "calendar":
        return (
            "exclude",
            category,
            "Calendar effect is weak/unstable after neutralization and has limited economic justification.",
        )
    if category == "industry_relative" and signal_class in {"mostly_style_or_weak", "missing"}:
        return (
            "risk_control",
            category,
            "Industry-relative descriptor is better used for exposure diagnostics unless residual evidence improves.",
        )
    if category == "moneyflow" and feature in {"lag1_main_mf_strength", "lag1_large_order_imbalance"}:
        return (
            "exclude",
            category,
            "Highly collinear weak money-flow proxy; reserve for pruning rather than current alpha set.",
        )
    if signal_class in {"strong_residual", "moderate_residual"}:
        return (
            "alpha",
            category,
            f"Residual signal survives neutralization: neutral_rank_ic={neutral_rank_ic}, t={neutral_t}, retention={retention}.",
        )
    if category in {"technical", "return_reversal", "moneyflow"} and signal_class == "review":
        return (
            "alpha",
            category,
            "Keep as alpha candidate for collinearity pruning; residual evidence is mixed but economically interpretable.",
        )
    return (
        "exclude",
        category,
        "Mostly style-explained, weak after neutralization, or redundant with stronger candidate features.",
    )


def build_role_table(features: list[str], decay_table: pd.DataFrame) -> pd.DataFrame:
    decay_by_feature = decay_table.set_index("feature").to_dict(orient="index") if not decay_table.empty else {}
    rows: list[dict[str, Any]] = []
    for feature in features:
        decay = decay_by_feature.get(feature, {})
        role, category, reason = role_from_feature(feature, decay)
        rows.append(
            {
                "feature": feature,
                "role": role,
                "category": category,
                "residual_signal_class": decay.get("residual_signal_class", "missing"),
                "raw_rank_ic_mean": decay.get("raw_rank_ic_mean"),
                "neutral_rank_ic_mean": decay.get("neutral_rank_ic_mean"),
                "rank_ic_abs_decay": decay.get("rank_ic_abs_decay"),
                "rank_ic_abs_retention": decay.get("rank_ic_abs_retention"),
                "neutral_rank_ic_t_stat": decay.get("neutral_rank_ic_t_stat"),
                "reason": reason,
            }
        )
    role_order = {"alpha": 0, "risk_control": 1, "tradability_control": 2, "exclude": 3}
    table = pd.DataFrame(rows)
    table["role_order"] = table["role"].map(role_order).fillna(9)
    return table.sort_values(["role_order", "category", "feature"]).drop(columns=["role_order"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature role tags for a configured feature set.")
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--feature-set", default="advanced_sequence_fixed")
    parser.add_argument("--decay-path", default=str(DEFAULT_DECAY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_advanced_features(Path(args.config), args.feature_set)
    decay_path = Path(args.decay_path)
    decay_table = pd.read_csv(decay_path) if decay_path.exists() else pd.DataFrame()
    role_table = build_role_table(features, decay_table)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    role_table.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(role_table["role"].value_counts().to_string())
    print(output_path)


if __name__ == "__main__":
    main()
