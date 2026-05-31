import json
import os

base = "outputs/backtest/rolling_EnhancedTF-l60-cls-mse-f13-label5"
path = os.path.join(base, "rolling_window_metrics.json")
with open(path) as f:
    d = json.load(f)

for split in ['test', 'validation']:
    if split not in d: continue
    print(f"\n=== {split.upper()} ===")
    for k in ['k_10', 'k_20', 'k_30']:
        if k not in d[split]: continue
        m = d[split][k]
        print(f"  {k}: Excess10d={m.get('mean_excess_vs_bench_10d','?'):+.4f} ({m['mean_excess_vs_bench_10d']*100:+.2f}%)  WinRate={m.get('mean_win_rate_vs_bench','?'):.1%}  Comp={m.get('mean_compliance_rate','?'):.0%}  PosWin={m.get('pct_positive_windows','?'):.1%}  IR_med={m.get('median_ir','?'):+.3f}")
        est = m.get('est_annualized_return_pct', None)
        if est is not None:
            print(f"        Est.AnnRet={est:.1f}%  Nwins={m.get('total_windows', '?')}")
