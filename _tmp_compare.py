import json
with open('outputs/runs/EnhancedTF-l60-cls-mse-f13-label5/metrics.json') as f:
    d = json.load(f)
best = int(d['best_epoch'])
print(f'Best epoch: {best}')
for h in d['history']:
    if h['epoch'] == best:
        print(f"  Val RankIC: {h['rank_ic_mean']:.4f} | IC: {h['ic_mean']:.4f} | ICIR: {h['icir']:.3f}")
        print(f"  pred_std: {h['pred_std']:.6f} | pred_range: [{h['pred_min']:.4f}, {h['pred_max']:.4f}]")
        print(f"  target_std: {h['target_std']:.4f} (label*5 applied)")
        ratio = h['pred_std'] / h['target_std'] if h['target_std'] > 0 else float('inf')
        print(f"  pred/target std ratio: {ratio:.2%}")
        break

# Also compare with previous models
print("\n=== COMPARISON ===")
models = [
    ("EnhancedTF-l60-Huber (orig)", "outputs/runs/EnhancedTF-l60-cls-huber-f13/metrics.json"),
    ("EnhancedTF-l60-Huber*10", "outputs/runs/EnhancedTF-l60-cls-huber-f13-label10/metrics.json"),
    ("StdTF-l60-MSE", "outputs/runs/StdTF-l60-cls-mse-f13/metrics.json"),
    ("NEW EnhancedTF-l60-MSE*5", "outputs/runs/EnhancedTF-l60-cls-mse-f13-label5/metrics.json"),
]
for name, path in models:
    try:
        with open(path) as f:
            d2 = json.load(f)
        be = int(d2['best_epoch'])
        for h in d2['history']:
            if h['epoch'] == be:
                ts = h.get('target_std', 0)
                ps = h.get('pred_std', 0)
                r = ps/ts if ts > 0 else 0
                print(f"  {name}: ep{be} RankIC={h['rank_ic_mean']:.4f} pred_std={ps:.6f} ratio={r:.2%}")
                break
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
