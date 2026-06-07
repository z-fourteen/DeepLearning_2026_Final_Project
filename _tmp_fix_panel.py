"""Build feature panel: map last data date to signal date for live trading."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRADE_DATE = "20260601"
END_DATE = "20260525"

df = pd.read_parquet(PROJECT_ROOT / "data/mart/features_daily/features_daily_v20260526.parquet")
df["trade_date"] = df["trade_date"].astype(str)

# Map END_DATE -> TRADE_DATE so the inference script can find "today's" data
df.loc[df["trade_date"] == END_DATE, "trade_date"] = TRADE_DATE

out_path = PROJECT_ROOT / f"data/live/features/features_{TRADE_DATE}.parquet"
df.to_parquet(out_path, index=False)

# Verify
check = pd.read_parquet(out_path)
today = check[check["trade_date"] == TRADE_DATE]
print(f"Saved: {out_path}")
print(f"  total={len(check)}  signal_day_rows={len(today)}  codes_on_signal_day={today['ts_code'].nunique()}")
