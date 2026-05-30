"""
快速启动：3个高潜力交叉特征的单因子 IC 验证

新特征：
  1. risk_adj_momentum_20d = excess_ret_20d_mean / ret_60d_std   (风险调整动量)
  2. volume_breakout        = ret_5d_mean * turnover_acceleration  (放量突破)
  3. relative_strength      = industry_neutral_ret_20d * ret_20d_mean (行业内相对强度)

数据来源：NPZ 序列数据集 (X[:, -1, :] 取最后时间步平铺)

用法：
  conda run -n dl_env python scripts/validate_cross_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# NPZ 数据路径
NPZ_PATH = (
    PROJECT_ROOT
    / "data"
    / "mart"
    / "datasets"
    / "dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz"
)

# ── 第二轮候选交叉特征：非线性/差分/条件型 (避免简单乘除冗余) ──
CROSS_FEATURES: dict[str, str | callable] = {
    # ── Round 1 (已验证: 简单乘除 → 高度冗余) ──
    "lag1_risk_adj_momentum_20d": "lag1_excess_ret_20d_mean / lag1_ret_60d_std",
    "lag1_volume_breakout": "lag1_ret_5d_mean * lag1_turnover_5d_mean",
    "lag1_relative_strength": "lag1_industry_neutral_ret_20d * lag1_ret_20d",

    # ── Round 2 (新增: 非线性/差分/条件) ──
    # 4. 动量分歧: 短期超额 - 长期超额 (差分比乘法更具辨识度)
    "lag1_momentum_divergence": "lag1_excess_ret_5d_mean - lag1_excess_ret_20d_mean",

    # 5. 波动率不对称: 高波动 + 正收益 vs 高波动 + 负收益
    "lag1_volatility_asymmetry": lambda d: (
        np.where(d["lag1_ret_5d"] > 0, d["lag1_ret_60d_std"], -d["lag1_ret_60d_std"])
    ),

    # 6. 资金流-价格背离: 主力方向与收益方向是否一致
    "lag1_flow_price_divergence": lambda d: (
        (np.sign(d["lag1_main_mf_strength"].fillna(0)) * np.abs(d["lag1_excess_ret_5d_mean"]))
    ),

    # 7. 技术面综合信号: 多指标同向计数 (0~4分)
    "lag1_tech_consensus": lambda d: (
        (d["lag1_rsi_14d"] > 60).astype(float)
        + (d["lag1_macd_diff"] > 0).astype(float)
        + (d["lag1_price_to_ma20"] > 1.0).astype(float)
        + (d["lag1_ma_ratio_5_20"] > 1.0).astype(float)
    ),

    # 8. 流动性调整振幅: 单位换手的价格波幅
    "lag1_liquidity_adjusted_amp": "lag1_amplitude / (lag1_turnover_rate + 0.001)",

    # 9. 换手异常: 当日换手偏离自身20日均线的程度
    "lag1_turnover_anomaly": "lag1_turnover_rate / (lag1_turnover_20d_mean + 0.001)",
}

MIN_CROSS_SECTION = 20


def load_npz_as_dataframe(npz_path: Path) -> pd.DataFrame:
    """从 NPZ 加载并转为 DataFrame (取 X[:, -1, :] 作为信号日特征)."""
    print(f"  Loading NPZ: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]           # [N, T, F]
    y = data["y"]           # [N]
    trade_dates = data["trade_date"]
    ts_codes = data["ts_code"]
    splits = data["split"]
    feature_names = list(data["feature_names"])

    print(f"  Shape: X={X.shape}, samples={len(y)}, features={len(feature_names)}")

    # 取最后一个时间步 (T-1) 的特征 -> 对应信号日可用的 lag1 特征状态
    df = pd.DataFrame(X[:, -1, :], columns=feature_names)
    df["label_rel_return"] = y.astype(float)
    df["trade_date"] = trade_dates
    df["ts_code"] = ts_codes
    df["split"] = splits

    return df


def safe_eval_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    """Simple arithmetic via pd.eval."""
    result = df.eval(expr)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_cross_feature(df: pd.DataFrame, name: str, expr_or_fn: str | callable) -> pd.Series:
    """Compute cross feature: simple expr via eval, complex via callable."""
    if callable(expr_or_fn):
        result = expr_or_fn(df)
    else:
        result = safe_eval_expr(df, expr_or_fn)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    return pd.Series(result, index=df.index).replace([np.inf, -np.inf], np.nan)


def compute_daily_ic(
    df: pd.DataFrame, feature: str, label: str = "label_rel_return"
) -> dict[str, float]:
    sub = df[["trade_date", feature, label]].dropna()
    if sub.empty:
        return {k: np.nan for k in [
            "ic_mean", "ic_std", "icir", "rank_ic_mean", "rank_ic_std",
            "rank_icir", "pos_ratio", "days",
        ]}

    counts = sub.groupby("trade_date").size()
    valid_dates = counts[counts >= MIN_CROSS_SECTION].index
    sub = sub[sub["trade_date"].isin(valid_dates)]
    if sub.empty:
        return {k: np.nan for k in [
            "ic_mean", "ic_std", "icir", "rank_ic_mean", "rank_ic_std",
            "rank_icir", "pos_ratio", "days",
        ]}

    daily_ic = (
        sub.groupby("trade_date")[[feature, label]]
        .corr(method="pearson").xs(feature, level=1)[label]
    )
    ranked = sub.copy()
    for col in [feature, label]:
        ranked[col] = ranked.groupby("trade_date")[col].rank(method="average")
    daily_rank_ic = (
        ranked.groupby("trade_date")[[feature, label]]
        .corr(method="pearson").xs(feature, level=1)[label]
    )

    ic_valid = daily_ic.dropna()
    ric_valid = daily_rank_ic.dropna()

    def _safe_icir(mean_val, std_val):
        return float(mean_val / std_val) if std_val and std_val > 0 else np.nan

    return {
        "ic_mean": float(ic_valid.mean()),
        "ic_std": float(ic_valid.std(ddof=1)),
        "icir": _safe_icir(ic_valid.mean(), ic_valid.std(ddof=1)),
        "rank_ic_mean": float(ric_valid.mean()),
        "rank_ic_std": float(ric_valid.std(ddof=1)),
        "rank_icir": _safe_icir(ric_valid.mean(), ric_valid.std(ddof=1)),
        "pos_ratio": float((ric_valid > 0).mean()),
        "days": int(ric_valid.count()),
    }


def check_correlation_with_existing(
    df: pd.DataFrame, new_feature: str,
    existing_features: list[str], threshold: float = 0.85,
) -> list[dict]:
    records = []
    for feat in existing_features:
        if feat not in df.columns or feat == new_feature:
            continue
        corr = df[[new_feature, feat]].corr(method="spearman").iloc[0, 1]
        if abs(corr) >= threshold:
            records.append({"existing_feature": feat, "spearman_corr": float(corr)})
    return sorted(records, key=lambda x: abs(x["spearman_corr"]), reverse=True)


def main() -> None:
    print("=" * 70)
    print("Cross-Feature Quick Validation")
    print("=" * 70)

    # ── 1. Load NPZ ──
    if not NPZ_PATH.exists():
        print(f"[ERROR] NPZ not found: {NPZ_PATH}")
        sys.exit(1)

    print(f"\n[1/5] Load NPZ dataset")
    df = load_npz_as_dataframe(NPZ_PATH)
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  Feature names ({len(df.columns) - 5}): {list(df.columns)[:5]} ...")

    # ── 2. Train split ──
    train_end = "20221231"
    df_train = df[df["trade_date"].astype(str) <= train_end].copy()
    print(f"\n[2/5] Train split: <= {train_end}, n={len(df_train):,}")

    # Check required base features
    required_base = set()
    for expr_or_fn in CROSS_FEATURES.values():
        if isinstance(expr_or_fn, str):
            for token in expr_or_fn.replace("/", " ").replace("*", " ").split():
                token = token.strip()
                if token.startswith("lag1_"):
                    required_base.add(token)
        # callables: skip — we assume they access existing columns

    missing = required_base - set(df_train.columns)
    if missing:
        print(f"[ERROR] Missing base features: {missing}")
        print(f"  Available sample: {[c for c in df_train.columns if 'ret' in c.lower()][:10]}")
        sys.exit(1)
    print(f"  Base features OK ({len(required_base)} found)")

    # ── 3. Compute cross features ──
    print(f"\n[3/5] Compute {len(CROSS_FEATURES)} cross features:")
    for name, expr_or_fn in CROSS_FEATURES.items():
        df_train[name] = compute_cross_feature(df_train, name, expr_or_fn)
        non_nan = df_train[name].notna().sum()
        expr_str = expr_or_fn if isinstance(expr_or_fn, str) else f"<lambda:{name}>"
        print(f"  + {name}: expr='{expr_str}', valid={non_nan:,}/{len(df_train):,}")

    # ── 4. Single-factor IC ──
    print(f"\n[4/5] Train-period IC analysis (min_cs={MIN_CROSS_SECTION}):")
    print("-" * 70)

    results = []
    for name, expr_or_fn in CROSS_FEATURES.items():
        stats = compute_daily_ic(df_train, name, "label_rel_return")
        stats["feature"] = name
        stats["expression"] = expr_or_fn if isinstance(expr_or_fn, str) else f"<lambda:{name}>"
        results.append(stats)

        flag = "PASS" if abs(stats["rank_ic_mean"]) >= 0.02 and abs(stats["rank_icir"]) >= 0.3 else "WEAK"
        print(f"\n  [{flag}] {name}")
        print(f"    expr   : {stats['expression']}")
        print(f"    IC     : mean={stats['ic_mean']:.4f}  std={stats['ic_std']:.4f}  ICIR={stats['icir']:.3f}")
        print(f"    RankIC : mean={stats['rank_ic_mean']:.4f}  std={stats['rank_ic_std']:.4f}  ICIR={stats['rank_icir']:.3f}")
        print(f"    pos%   : {stats['pos_ratio']:.1%}  days={stats['days']}")

    # ── 5. Correlation check ──
    print(f"\n{'=' * 70}")
    print("[5/5] Correlation check (|Spearman| >= 0.85):")
    print("-" * 70)

    existing_62 = [c for c in df_train.columns if c.startswith("lag1_") and c not in CROSS_FEATURES]

    for name in CROSS_FEATURES:
        high_corr = check_correlation_with_existing(df_train, name, existing_62, threshold=0.85)
        if high_corr:
            print(f"\n  ! HIGH CORR: {name}")
            for hc in high_corr[:5]:
                print(f"      x {hc['existing_feature']:45s}  r={hc['spearman_corr']:.4f}")
        else:
            print(f"\n  OK: {name} (no |r| >= 0.85)")

    # ── 6. Final verdict ──
    print(f"\n{'=' * 70}")
    print("FINAL VERDICT")
    print("=" * 70)

    summary = []
    for r in results:
        verdict = "RECOMMEND"
        reasons = []

        if abs(r["rank_ic_mean"]) < 0.01:
            verdict = "WEAK"; reasons.append("RankIC too low (< 0.01)")
        elif abs(r["rank_ic_mean"]) < 0.02:
            if verdict == "RECOMMEND":
                verdict = "MARGINAL"
            reasons.append("RankIC marginal (< 0.02)")

        if abs(r["rank_icir"]) < 0.3:
            if verdict == "RECOMMEND":
                verdict = "MARGINAL"
            elif verdict != "WEAK":
                pass
            reasons.append("ICIR unstable (< 0.3)")

        high_corr = check_correlation_with_existing(df_train, r["feature"], existing_62, 0.85)
        if high_corr:
            verdict = "REDUNDANT"
            reasons.append(f"redundant w/ {high_corr[0]['existing_feature']}")

        summary.append({
            "feature": r["feature"],
            "expression": r["expression"],
            "verdict": verdict,
            "rank_ic_mean": round(r["rank_ic_mean"], 4),
            "rank_icir": round(r["rank_icir"], 3),
            "icir": round(r["icir"], 3),
            "pos_ratio": round(r["pos_ratio"], 3),
            "days": r["days"],
            "reasons": "; ".join(reasons) if reasons else "OK",
        })

        icon = {"RECOMMEND": "+", "MARGINAL": "~", "WEAK": "-", "REDUNDANT": "x"}
        print(f"\n  [{icon.get(verdict, '?')}] {r['feature']}")
        print(f"      RankIC={r['rank_ic_mean']:.4f}  ICIR={r['icir']:.3f}  RankICIR={r['rank_icir']:.3f}")
        print(f"      pos%={r['pos_ratio']:.1%}  days={r['days']}")
        print(f"      => {verdict}: {'; '.join(reasons) if reasons else 'All checks passed'}")

    # Save
    out_path = PROJECT_ROOT / "outputs" / "cross_feature_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
