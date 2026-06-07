"""Prepare live trading data: features_daily -> signal date.

Auto-reads feature list + lookback from model YAML configs.
Generates per-model NPZ with l{lookback} suffix.
Auto-finds latest features_daily parquet.

Usage:
    python _tmp_prep_live_v2.py --trade-date 20260602
"""
import argparse, glob, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "live"
MART_DIR = PROJECT_ROOT / "data" / "mart" / "features_daily"


def find_parquet(version=None):
    if version:
        p = MART_DIR / f"features_daily_{version}.parquet"
        if p.exists():
            return p
    files = sorted(glob.glob(str(MART_DIR / "features_daily_*.parquet")), reverse=True)
    if not files:
        print(f"ERROR: No parquet in {MART_DIR}"); sys.exit(1)
    return Path(files[0])


def load_config(name):
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = cfg.get("model", {})
    lookback = int(model.get("lookback", 60))
    features = list(model.get("expected_features", []))
    return {"name": name, "lookback": lookback, "features": features}


def build_sequences(df_hist, end_date, features, lookback):
    sequences, codes = [], []
    for code, group in df_hist.groupby("ts_code", sort=True):
        g = group.sort_values("trade_date")
        if str(g["trade_date"].iloc[-1]) != str(end_date):
            continue
        tail = g.tail(lookback)
        if len(tail) != lookback:
            continue
        vals = tail[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        if not np.isfinite(vals).all():
            continue
        sequences.append(vals); codes.append(str(code))
    if not sequences:
        return None, []
    return np.stack(sequences, axis=0), codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-date", required=True)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Config names under configs/live/ (without .yaml)")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--data-version", default=None)
    args = ap.parse_args()
    trade_date = args.trade_date

    # Load source parquet
    pq_path = find_parquet(args.data_version)
    print(f"Source: {pq_path}")
    df = pd.read_parquet(pq_path); df["trade_date"] = df["trade_date"].astype(str)
    end_date = args.end_date or str(df["trade_date"].max())
    print(f"  rows={len(df)} max_date={df['trade_date'].max()} using_end={end_date}")
    df_hist = df[df["trade_date"] <= end_date].copy()

    # Discover configs
    if args.configs:
        config_names = args.configs
    else:
        config_names = [f.stem for f in sorted(CONFIGS_DIR.glob("live_trading*.yaml"))
                        if f.stem != "live_trading"]

    models = {}
    for name in config_names:
        c = load_config(name)
        if c and c["features"]:
            models[name] = c

    if not models:
        print("ERROR: No valid configs found."); sys.exit(1)

    print(f"Models: {list(models.keys())}")
    out_dir = PROJECT_ROOT / "data" / "live" / "features"; out_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-model NPZ
    all_codes = set()
    for name, mc in models.items():
        lb = mc["lookback"]; feats = mc["features"]
        X, codes = build_sequences(df_hist, end_date, feats, lb)
        if X is None:
            print(f"  SKIP {name} (l{lb}): no valid sequences"); continue
        npz = out_dir / f"live_sequence_{trade_date}_l{lb}.npz"
        np.savez_compressed(npz, X=X, ts_code=np.array(codes, dtype=object),
                            feature_names=np.array(feats, dtype=object),
                            trade_date=np.array([trade_date]*len(codes), dtype=object))
        print(f"  OK {name} l{lb}: {npz} shape={X.shape}")
        all_codes.update(codes)

    # Feature panel (latest date mapped to trade_date)
    if models:
        sample_feats = list(models.values())[0]["features"]
    else:
        sample_feats = []
    latest = df_hist[df_hist["trade_date"] == end_date].copy()
    if not latest.empty and sample_feats:
        cols = ["trade_date", "ts_code"] + [c for c in sample_feats if c in latest.columns]
        panel = latest[cols].copy(); panel["trade_date"] = trade_date
        panel.to_parquet(out_dir / f"features_{trade_date}.parquet", index=False)
        print(f"  Panel: {out_dir / f'features_{trade_date}.parquet'} ({len(panel)} rows)")

    # Positions placeholder
    acct = PROJECT_ROOT / "data" / "live" / "account"; acct.mkdir(parents=True, exist_ok=True)
    pos = acct / f"positions_{trade_date}.csv"
    if not pos.exists():
        prev = acct / f"positions_{str(int(trade_date)-1).zfill(8)}.csv"
        if prev.exists(): shutil.copy(prev, pos); print(f"  Copied positions from prev")
        else:
            pd.DataFrame({"ts_code":[],"weight":[],"volume":[],"market_value":[]}).to_csv(pos, index=False)
            print(f"  Empty positions created")

    # Price snapshot
    mkt = PROJECT_ROOT / "data" / "live" / "market"; mkt.mkdir(parents=True, exist_ok=True)
    quote = mkt / f"quotes_{trade_date}_0920.csv"
    if not quote.exists() and not latest.empty:
        pc = "close" if "close" in latest.columns else None
        if pc:
            latest[["ts_code", pc]].rename(columns={pc:"price"}).to_csv(quote, index=False)
            print(f"  Price snapshot: {quote}")
    print(f"\nDone. trade_date={trade_date}")


if __name__ == "__main__":
    main()
