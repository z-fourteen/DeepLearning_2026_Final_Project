"""Prepare live trading data: build sequence NPZ from features_daily parquet."""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Config ---
TRADE_DATE = "20260601"       # signal day (Monday)
END_DATE = "20260525"          # last available data date (Friday)
LOOKBACK = 60
FEATURES = [
    "lag1_net_mf_strength_20d_mean",
    "lag1_net_mf_strength_60d_mean",
    "lag1_close_position",
    "lag1_excess_ret_10d_mean",
    "lag1_excess_ret_1d",
    "lag1_excess_ret_5d_mean",
    "lag1_industry_neutral_ret_1d",
    "lag1_ret_1d",
    "lag1_ret_20d",
    "lag1_ret_5d_mean",
    "lag1_bollinger_z_20d",
    "lag1_ma_ratio_20_60",
    "lag1_macd_hist",
]

# --- Load feature panel ---
parquet_path = PROJECT_ROOT / "data/mart/features_daily/features_daily_v20260526.parquet"
print(f"Loading {parquet_path} ...")
df = pd.read_parquet(parquet_path)
df["trade_date"] = df["trade_date"].astype(str)
print(f"  shape={df.shape}  date_range=[{df['trade_date'].min()}, {df['trade_date'].max()}]")

# Filter data up to END_DATE
df = df[df["trade_date"] <= END_DATE].copy()
print(f"After filtering <={END_DATE}: shape={df.shape}  nunique_codes={df['ts_code'].nunique()}")

# --- Build sequences per stock ---
sequences = []
codes = []
for code, group in df.groupby("ts_code", sort=True):
    group = group.sort_values("trade_date")
    if group["trade_date"].iloc[-1] != END_DATE:
        continue
    tail = group.tail(LOOKBACK)
    if len(tail) != LOOKBACK:
        continue
    values = tail[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        continue
    sequences.append(values)
    codes.append(str(code))

if not sequences:
    raise SystemExit("ERROR: no valid sequences after lookback/NaN filtering!")

X = np.stack(sequences, axis=0)
print(f"\nLive sequence built:")
print(f"  X.shape = {X.shape}")
print(f"  n_stocks = {len(codes)}")
print(f"  lookback = {LOOKBACK}, features = {len(FEATURES)}")

# --- Save NPZ ---
out_dir = PROJECT_ROOT / "data/live/features"
out_dir.mkdir(parents=True, exist_ok=True)

npz_path = out_dir / f"live_sequence_{TRADE_DATE}.npz"
np.savez_compressed(
    npz_path,
    X=X,
    ts_code=np.array(codes, dtype=object),
    feature_names=np.array(FEATURES, dtype=object),
    trade_date=np.array([TRADE_DATE] * len(codes), dtype=object),
)
print(f"\nSaved: {npz_path}")

# Also save as parquet for liquidity/feature panel use in step 02
panel_out = out_dir / f"features_{TRADE_DATE}.parquet"
latest_rows = df[df["trade_date"] == END_DATE][["trade_date", "ts_code", *FEATURES]].copy()
latest_rows.to_parquet(panel_out, index=False)
print(f"Saved feature panel: {panel_out} ({len(latest_rows)} rows)")

# --- Create placeholder account files (first day: empty positions) ---
acct_dir = PROJECT_ROOT / "data/live/account"
acct_dir.mkdir(parents=True, exist_ok=True)

# Empty positions for first day (all cash)
empty_pos = pd.DataFrame({"ts_code": [], "weight": [], "volume": [], "market_value": []})
pos_path = acct_dir / f"positions_{TRADE_DATE}.csv"
empty_pos.to_csv(pos_path, index=False)
print(f"Saved empty positions: {pos_path}")

# Previous close positions (also empty for first day)
prev_date = "20250529"  # placeholder before competition
prev_pos_path = acct_dir / f"close_positions_{prev_date}.csv"
empty_pos.to_csv(prev_pos_path, index=False)
print(f"Saved prev close positions: {prev_pos_path}")

# Price snapshot placeholder (use pre_close from latest data)
mkt_dir = PROJECT_ROOT / "data/live/market"
mkt_dir.mkdir(parents=True, exist_ok=True)
if "pre_close" in df.columns or "close" in df.columns:
    price_col = "pre_close" if "pre_close" in df.columns else "close"
    price_snap = df[df["trade_date"] == END_DATE][["ts_code", price_col]].copy()
    price_snap.columns = ["ts_code", "price"]
else:
    # fallback: use a dummy price
    price_snap = pd.DataFrame({"ts_code": codes, "price": 10.0})
quote_path = mkt_dir / f"quotes_{TRADE_DATE}_0920.csv"
price_snap.to_csv(quote_path, index=False)
print(f"Saved price snapshot: {quote_path} ({len(price_snap)} rows)")

print("\n=== Live data preparation complete ===")
