import json, os

backtest_dir = "outputs/backtest"
results = []

for d in sorted(os.listdir(backtest_dir)):
    if not d.startswith("rolling_"): continue
    path = os.path.join(backtest_dir, d, "rolling_window_metrics.json")
    if not os.path.exists(path): continue
    with open(path) as f:
        data = json.load(f)
    
    # Get model name from directory
    model_name = d.replace("rolling_", "")
    
    # Extract test k20 metrics (primary comparison metric)
    split = "test"
    k = "k_20"
    if split in data and k in data[split]:
        m = data[split][k]
        results.append({
            "model": model_name,
            "excess_10d": m.get("mean_excess_vs_bench_10d", None),
            "winrate": m.get("mean_win_rate_vs_bench", None),
            "poswin": m.get("pct_positive_windows", None),
            "ir_med": m.get("median_ir", None),
            "comp": m.get("mean_compliance_rate", None),
            "nwindows": m.get("total_windows", "?"),
        })

# Sort by test excess return descending
results.sort(key=lambda x: x["excess_10d"] or -999, reverse=True)

print(f"{'Model':<48} {'Excess10d':>9} {'WinRt':>6} {'PosWin':>6} {'IRmed':>7} {'Comp':>5}")
print("-" * 90)
for r in results:
    e = f"{r['excess_10d']*100:+.2f}%" if r['excess_10d'] is not None else "N/A"
    wr = f"{r['winrate']:.1%}" if r['winrate'] is not None else "?"
    pw = f"{r['poswin']:.1%}" if r['poswin'] is not None else "?"
    ir = f"{r['ir_med']:+.3f}" if r['ir_med'] is not None else "?"
    co = f"{r['comp']:.0%}" if r['comp'] is not None else "?"
    print(f"{r['model']:<48} {e:>9} {wr:>6} {pw:>6} {ir:>7} {co:>5}")

print("\n--- Also check validation set ---")
for d in sorted(os.listdir(backtest_dir)):
    if not d.startswith("rolling_"): continue
    path = os.path.join(backtest_dir, d, "rolling_window_metrics.json")
    if not os.path.exists(path): continue
    with open(path) as f:
        data = json.load(f)
    model_name = d.replace("rolling_", "")
    if "validation" in data and "k_20" in data["validation"]:
        m = data["validation"]["k_20"]
        e = m.get("mean_excess_vs_bench_10d")
        wr = m.get("mean_win_rate_vs_bench")
        print(f"  {model_name:<48} val_k20: {e*100:+.2f}%  Win={wr:.1%}")
