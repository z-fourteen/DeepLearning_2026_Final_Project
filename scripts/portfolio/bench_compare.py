"""Quick benchmark comparison for optimizer results. Output to file to avoid encoding issues."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

periods_path = "outputs/backtest/optimizer/transformer_l20_clean_alpha_only_purgedwf/optimizer_periods.csv"
summary_path = "outputs/backtest/optimizer/transformer_l20_clean_alpha_only_purgedwf/optimizer_summary.csv"
out_path = "outputs/backtest/optimizer/transformer_l20_clean_alpha_only_purgedwf/bench_compare.txt"

periods = pd.read_csv(periods_path)
summary = pd.read_csv(summary_path)

lines = []
def p(s=""):
    lines.append(s)

PPY = 252.0 / 5.0

p("=" * 110)
p("Transformer l20 clean_alpha_only purgedwf - Portfolio Optimizer vs Benchmark (Test Set)")
p("=" * 110)

test = periods[periods["split"] == "test"].copy()
test_summary = summary[summary["split"] == "test"].copy()

bench = test[["trade_date","benchmark_return"]].drop_duplicates("trade_date").set_index("trade_date")["benchmark_return"]
univ = test[["trade_date","executable_universe_return"]].drop_duplicates("trade_date").set_index("trade_date")["executable_universe_return"]

def ann_stats(ser, name):
    n = len(ser)
    cum = float((1 + ser).prod() - 1)
    ann = float((1 + cum) ** (PPY / n) - 1)
    std = float(ser.std(ddof=1))
    ir = float(ser.mean() / std) if std > 1e-12 else float("nan")
    eq = (1 + ser).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    p(f"  {name:40s}: ann={ann:+.4%}  IR={ir:+.4f}  MaxDD={dd:.4%}  cum={cum:+.4%}  periods={n}")

p("")
p("--- Benchmark Reference ---")
ann_stats(bench, "Benchmark (ChiNext 399006.SZ)")
ann_stats(univ, "Executable Universe (equal-weight tradable pool)")

p("")
p(f"{'config':58s} {'net_ann':>8s} {'IR':>6s} {'MaxDD':>8s} {'ex_bench':>12s} {'ex_univ':>12s} {'hit%':>7s}")
p("-" * 120)

for _, row in test_summary.sort_values("net_ann", ascending=False).iterrows():
    cfg = f"{row['risk_control']:28s} sp={row['style_penalty']:.2f} tp={row['turnover_penalty']:.2f}"
    sub = test[
        (test.risk_control == row.risk_control)
        & (test.style_penalty == row.style_penalty)
        & (test.turnover_penalty == row.turnover_penalty)
    ]
    hit = float((sub.excess_vs_benchmark > 0).mean())
    p(
        f"{cfg:58s} {row['net_ann']:+7.2%} {row['net_ir']:+5.3f} "
        f"{row['net_max_drawdown']:+7.2%} {row['excess_benchmark_ann']:+11.2%} "
        f"{row['excess_exec_universe_ann']:+11.2%} {hit:6.1%}"
    )

# Best config detail
best = test_summary.sort_values("net_ann", ascending=False).iloc[0]
b = test[
    (test.risk_control == best.risk_control)
    & (test.style_penalty == best.style_penalty)
    & (test.turnover_penalty == best.turnover_penalty)
].copy().sort_values("trade_date")

p(f"")
p(f"--- Best config: {best.risk_control}, style={best.style_penalty:.2f}, turnover={best.turnover_penalty:.2f} ---")
hit_bench = float((b.excess_vs_benchmark > 0).mean())
hit_univ = float((b.excess_vs_executable_universe > 0).mean())
p(f"  Win rate vs Benchmark: {hit_bench:.1%}")
p(f"  Win rate vs ExecUniv:  {hit_univ:.1%}")
p(f"  Avg excess vs Bench:   {float(b.excess_vs_benchmark.mean()):+.4f}")
p(f"  Avg excess vs Univ:    {float(b.excess_vs_executable_universe.mean()):+.4f}")

b_idx = b.set_index("trade_date")
port_cum = (1 + b_idx["net_return"]).cumprod()
bm_cum = (1 + b_idx["benchmark_return"]).cumprod()
uv_cum = (1 + b_idx["executable_universe_return"]).cumprod()

p(f"")
p(f"{'date':12s} {'Portfolio':>10s} {'Benchmark':>10s} {'ExecUniv':>10s} {'vsBench':>11s} {'vsUniv':>11s}")
dates_show = list(port_cum.index[::10]) + [port_cum.index[-1]]
for d in dates_show:
    pc = float(port_cum.loc[d])
    bmc = float(bm_cum.loc[d])
    uvc = float(uv_cum.loc[d])
    p(f"{str(d):12s} {pc:9.4f} {bmc:9.4f} {uvc:9.4f} {pc-bmc:+10.4f} {pc-uvc:+10.4f}")

p(f"")
p(f"Final cumulative: Port={port_cum.iloc[-1]:.4f}, Bench={bm_cum.iloc[-1]:.4f}, Univ={uv_cum.iloc[-1]:.4f}")
p(f"Final excess: vsBench={float(port_cum.iloc[-1]-bm_cum.iloc[-1]):+.4f}, vsUniv={float(port_cum.iloc[-1]-uv_cum.iloc[-1]):+.4f}")

# Rolling windows
for w in [10, 20, 40]:
    if len(b) >= w:
        re = b["net_return"].rolling(w).mean() - b["benchmark_return"].rolling(w).mean()
        p(f"  Rolling {w:2d}-period avg_excess={float(re.mean()):+.4f}  std={float(re.std()):.4f}")

# Monthly
b["ym"] = b["trade_date"].astype(str).str[:6]
monthly = b.groupby("ym").agg(excess=("excess_vs_benchmark", "sum"))
me = monthly["excess"]
pos = int((me > 0).sum())
total = len(me)
p(f"")
p(f"--- Monthly Excess Distribution ---")
p(f"  Positive months: {pos}/{total} ({pos/max(total,1):.1%})")
p(f"  Mean={float(me.mean()):+.4f}  Median={float(me.median()):+.4f}  Std={float(me.std()):.4f}")
p(f"  Best month={float(me.max()):+.4f}  Worst month={float(me.min()):+.4f}")

# Write
Path(out_path).write_text("\n".join(lines), encoding="utf-8")
print(f"Output written to: {out_path}")
print("\n".join(lines))
