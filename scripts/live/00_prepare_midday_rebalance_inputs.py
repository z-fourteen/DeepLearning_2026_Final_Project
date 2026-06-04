from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    assert_universe,
    die,
    format_path,
    load_yaml,
    normalize_code_column,
    price_column,
    previous_trading_day,
    resolve_path,
    today_yyyymmdd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare intraday account and price snapshots for a midday rebalance."
    )
    parser.add_argument("--config", default="configs/live/live_trading.yaml")
    parser.add_argument("--trade-date", default=today_yyyymmdd())
    parser.add_argument(
        "--midday-prices",
        required=True,
        help="Manual 11:30 snapshot CSV with ts_code/code and price/last_price/close.",
    )
    parser.add_argument(
        "--base-price-snapshot",
        help="Optional full-market fallback quote CSV. Defaults to the configured 09:20 snapshot for the same trade date.",
    )
    parser.add_argument(
        "--portfolio-state",
        default="outputs/live/portfolio_state.json",
        help="Current portfolio_state.json after the morning execution.",
    )
    parser.add_argument("--tag", default="midday")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def write_csv(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        die(f"output already exists; pass --overwrite to replace: {path}")
    frame.to_csv(path, index=False)
    print(f"wrote: {path}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        die(f"missing portfolio state: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def holdings_frame(state: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, holding in (state.get("holdings") or {}).items():
        shares = int(holding.get("shares", 0) or 0)
        if shares <= 0:
            continue
        rows.append({"ts_code": str(code), "volume": shares})
    if not rows:
        die("portfolio_state.json has no positive-share holdings")
    return pd.DataFrame(rows)


def normalized_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"missing price snapshot: {path}")
    frame = normalize_code_column(pd.read_csv(path))
    px_col = price_column(frame)
    frame["price"] = pd.to_numeric(frame[px_col], errors="coerce")
    return frame[["ts_code", "price"]].drop_duplicates("ts_code")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    trade_date = str(args.trade_date)
    tag = safe_tag(args.tag)
    prev_trade_date = previous_trading_day(config, trade_date)

    state = load_state(resolve_path(args.portfolio_state))
    holdings = holdings_frame(state)
    assert_universe(holdings, config, "midday holdings")

    manual_prices = normalized_prices(resolve_path(args.midday_prices))
    base_price_path = (
        resolve_path(args.base_price_snapshot)
        if args.base_price_snapshot
        else format_path(
            config["live_inputs"]["price_snapshot"],
            trade_date=trade_date,
            prev_trade_date=prev_trade_date,
        )
    )
    if base_price_path.exists():
        base_prices = normalized_prices(base_price_path)
        prices = pd.concat([base_prices, manual_prices], ignore_index=True)
        prices = prices.drop_duplicates("ts_code", keep="last")
    else:
        prices = manual_prices
    assert_universe(prices, config, "midday prices")
    bad_prices = prices["price"].isna() | prices["price"].le(0)
    if bad_prices.any():
        sample = prices.loc[bad_prices, "ts_code"].astype(str).head(10).tolist()
        die(f"midday prices contain missing/non-positive values; sample={sample}")

    missing = sorted(set(holdings["ts_code"].astype(str)) - set(prices["ts_code"].astype(str)))
    if missing:
        die(f"midday price snapshot missing current holdings; sample={missing[:10]}")

    cash = float(state.get("cash", 0.0) or 0.0)
    current = holdings.merge(prices[["ts_code", "price"]], on="ts_code", how="left")
    current["market_value"] = current["volume"].astype(float) * current["price"].astype(float)
    nav = float(current["market_value"].sum() + cash)
    if nav <= 0:
        die(f"midday NAV must be positive, got {nav}")
    current["weight"] = current["market_value"] / nav
    current["cash"] = cash
    current["nav"] = nav
    positions = current[
        ["ts_code", "weight", "volume", "market_value", "cash", "nav"]
    ].sort_values("ts_code")

    quotes = prices[["ts_code", "price"]].drop_duplicates("ts_code").copy()
    quotes = quotes.rename(columns={"ts_code": "code"}).sort_values("code")

    out_account = resolve_path(Path("data") / "live" / "account" / f"positions_{trade_date}_{tag}.csv")
    out_quotes = resolve_path(Path("data") / "live" / "market" / f"quotes_{trade_date}_{tag}.csv")
    write_csv(out_account, positions, args.overwrite)
    write_csv(out_quotes, quotes, args.overwrite)

    print(
        "midday snapshot: "
        f"trade_date={trade_date} tag={tag} nav={nav:,.2f} cash={cash:,.2f} "
        f"position_value={float(current['market_value'].sum()):,.2f} "
        f"invested={float(positions['weight'].sum()):.2%} holdings={len(positions)} "
        f"manual_prices={len(manual_prices)} quote_rows={len(quotes)}"
    )


if __name__ == "__main__":
    main()
