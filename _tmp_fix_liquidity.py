"""Fix liquidity for NAV=1M: ensure participation_cap doesn't bottleneck full investment."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRADE_DATE = "20260601"
NAV = 1_000_000.0
PARTICIPATION_CAP = 0.03
SINGLE_NAME_CAP = 0.10

# Need: PARTICIPATION_CAP * amount / NAV >> SINGLE_NAME_CAP
# => amount > SINGLE_NAME_CAP * NAV / PARTICIPATION_CAP = 0.10 * 1M / 0.03 ≈ 3.33M
panel_path = PROJECT_ROOT / f"data/live/features/features_{TRADE_DATE}.parquet"
panel = pd.read_parquet(panel_path)
panel["next_amount"] = 20_000_000.0  # 20M -> buy_cap = 60% >> 10% cap
panel.to_parquet(panel_path, index=False)

cap_per_stock = PARTICIPATION_CAP * 20_000_000 / NAV
print(f"Updated {panel_path}")
print(f"  NAV={NAV:,.0f}  next_amount=20M")
print(f"  buy_capacity per stock = {cap_per_stock:.1%} (>> single_name_cap {SINGLE_NAME_CAP:.0%})")
