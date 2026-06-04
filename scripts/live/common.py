from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import FeatureStyleInteractionGRUStockModel, GRUStockModel, RegimeGatedGRUStockModel  # noqa: E402


def alarm(message: str) -> None:
    # BEL 字符会让多数终端发出声音；同时打印醒目的错误块，避免实盘静默失败。
    print("\a", file=sys.stderr)
    print("=" * 88, file=sys.stderr)
    print(f"LIVE TRADING GUARD FAILED: {message}", file=sys.stderr)
    print("=" * 88, file=sys.stderr)


def die(message: str) -> None:
    alarm(message)
    raise SystemExit(2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.exists():
        die(f"missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_path(template: str | Path, **values: Any) -> Path:
    return resolve_path(str(template).format(**values))


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_code_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "ts_code" not in frame.columns:
        if "code" in frame.columns:
            frame = frame.rename(columns={"code": "ts_code"})
        elif "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "ts_code"})
    if "ts_code" not in frame.columns:
        die("input file must contain ts_code/code/symbol column")
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "trade_date" not in frame.columns:
        if "date" in frame.columns:
            frame = frame.rename(columns={"date": "trade_date"})
        else:
            die("input file must contain trade_date/date column")
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    return frame


def ensure_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        die(f"{label} missing columns: {missing}")


def universe_mask(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if "ts_code" not in frame.columns:
        die("universe filter requires ts_code column")
    universe = config.get("universe", {}) or {}
    prefixes = [str(item) for item in universe.get("allowed_prefixes", [])]
    suffix = universe.get("allowed_suffix")
    codes = frame["ts_code"].astype(str)
    mask = pd.Series(True, index=frame.index)
    if prefixes:
        mask &= codes.str[:3].isin(prefixes)
    if suffix:
        mask &= codes.str.endswith(str(suffix))
    return mask


def apply_universe_filter(frame: pd.DataFrame, config: dict[str, Any], label: str) -> pd.DataFrame:
    before_rows = len(frame)
    before_stocks = int(frame["ts_code"].nunique()) if "ts_code" in frame.columns else 0
    filtered = frame.loc[universe_mask(frame, config)].copy()
    after_rows = len(filtered)
    after_stocks = int(filtered["ts_code"].nunique()) if "ts_code" in filtered.columns else 0
    name = (config.get("universe", {}) or {}).get("name", "configured")
    print(
        f"{label} universe={name}: rows {before_rows}->{after_rows}, "
        f"stocks {before_stocks}->{after_stocks}"
    )
    if filtered.empty:
        die(f"{label} has no rows after universe={name} filter")
    return filtered


def assert_universe(frame: pd.DataFrame, config: dict[str, Any], label: str) -> None:
    bad = frame.loc[~universe_mask(frame, config), "ts_code"].astype(str).drop_duplicates()
    if not bad.empty:
        sample = bad.head(10).tolist()
        name = (config.get("universe", {}) or {}).get("name", "configured")
        die(f"{label} contains non-{name} symbols; sample={sample}")


def trading_days(config: dict[str, Any]) -> list[str]:
    days = [str(day) for day in config.get("competition", {}).get("trading_days", [])]
    if not days:
        die("configs/live/live_trading.yaml must define competition.trading_days")
    return days


def previous_trading_day(config: dict[str, Any], trade_date: str) -> str:
    days = trading_days(config)
    if trade_date not in days:
        die(f"trade_date={trade_date} is outside configured competition trading days: {days}")
    idx = days.index(trade_date)
    if idx == 0:
        # 比赛首日也必须继承上一交易日真实收盘持仓，默认使用上一个工作日 20260529。
        return "20260529"
    return days[idx - 1]


def competition_progress(config: dict[str, Any], trade_date: str) -> float:
    days = trading_days(config)
    idx = days.index(trade_date)
    return idx / max(1, len(days) - 1)


def dynamic_shortfall_penalty(config: dict[str, Any], trade_date: str) -> float:
    # 时间步动态短缺惩罚：越接近比赛后段，低于 80% 仓位的惩罚越重。
    # 首日若 3% participation cap 导致 80% 硬约束不可达，soft shortfall 变量可避免优化器死锁；
    # 后续交易日惩罚逐步抬升，推动组合尽快回到 min_invested 规则内。
    opt = config["optimizer"]
    base = float(opt.get("shortfall_penalty_base", 500.0))
    max_penalty = float(opt.get("shortfall_penalty_max", 5000.0))
    progress = competition_progress(config, trade_date)
    return float(base + (max_penalty - base) * progress * progress)


def assert_market_coverage(frame: pd.DataFrame, config: dict[str, Any], label: str) -> None:
    guards = config.get("guards", {})
    expected = int(guards.get("expected_universe_size", 5000))
    ratio = float(guards.get("min_market_coverage_ratio", 0.8))
    unique_codes = int(frame["ts_code"].nunique())
    required = math.ceil(expected * ratio)
    if unique_codes < required:
        die(
            f"{label} market coverage too low: unique_codes={unique_codes}, "
            f"required>={required} ({ratio:.0%} of expected_universe_size={expected})"
        )


def load_positions(path: str | Path, label: str = "positions") -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        die(f"missing {label} file: {path}")
    frame = pd.read_csv(path)
    frame = normalize_code_column(frame)
    if "weight" not in frame.columns:
        if "old_w" in frame.columns:
            frame = frame.rename(columns={"old_w": "weight"})
        elif "market_value" in frame.columns:
            total = pd.to_numeric(frame["market_value"], errors="coerce").sum()
            if total <= 0:
                die(f"{label} cannot derive weight because market_value sum <= 0")
            frame["weight"] = pd.to_numeric(frame["market_value"], errors="coerce") / total
        else:
            die(f"{label} must contain weight/old_w or market_value")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
    return frame


def assert_position_inheritance(
    current: pd.DataFrame,
    previous_close: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    # 实盘状态不能断裂：每日 old_w 必须与上一交易日真实成交收盘持仓完全对齐。
    guards = config.get("guards", {})
    weight_tol = float(guards.get("position_weight_tolerance", 1e-6))
    volume_tol = int(guards.get("position_volume_tolerance", 0))

    cur = current[["ts_code", "weight", *(['volume'] if "volume" in current.columns else [])]].copy()
    prev = previous_close[["ts_code", "weight", *(['volume'] if "volume" in previous_close.columns else [])]].copy()
    merged = cur.merge(prev, on="ts_code", how="outer", suffixes=("_current", "_prev")).fillna(0)
    merged["weight_diff"] = (merged["weight_current"] - merged["weight_prev"]).abs()
    bad = merged[merged["weight_diff"] > weight_tol]
    if "volume_current" in merged.columns and "volume_prev" in merged.columns:
        merged["volume_diff"] = (merged["volume_current"] - merged["volume_prev"]).abs()
        bad = pd.concat([bad, merged[merged["volume_diff"] > volume_tol]], ignore_index=True).drop_duplicates("ts_code")
    if not bad.empty:
        sample = bad.head(10).to_dict(orient="records")
        die(f"position inheritance check failed; sample mismatches={sample}")


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_frozen_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config_path = resolve_path(config["model"]["config"])
    model_checkpoint_path = resolve_path(config["model"]["checkpoint"])
    model_yaml = load_yaml(model_config_path)
    model_cfg = model_yaml["model"]
    name = str(model_cfg.get("name"))
    num_features = int(model_cfg.get("num_features", len(config["model"]["expected_features"])))
    if name == "feature_style_interaction_gru":
        model = FeatureStyleInteractionGRUStockModel(num_features=num_features, config=model_cfg)
    elif name == "gru_baseline":
        model = GRUStockModel(num_features=num_features, config=model_cfg)
    elif name == "regime_gated_gru":
        model = RegimeGatedGRUStockModel(num_features=num_features, config=model_cfg)
    else:
        die(f"unsupported model name in frozen config: {name}")
    if not model_checkpoint_path.exists():
        die(f"missing frozen checkpoint: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def price_column(frame: pd.DataFrame) -> str:
    for column in ["price", "last_price", "open", "pre_close", "close"]:
        if column in frame.columns:
            return column
    die("price snapshot must contain one of: price, last_price, open, pre_close, close")
    raise AssertionError("unreachable")


def resolve_portfolio_nav(
    config: dict[str, Any],
    trade_date: str,
    prev_trade_date: str,
    *,
    positions: pd.DataFrame | None = None,
    portfolio_state_path: str | Path | None = None,
    explicit_nav: float | None = None,
) -> tuple[float, str]:
    if explicit_nav is not None:
        nav = float(explicit_nav)
        if nav <= 0:
            die(f"--portfolio-nav must be positive, got {nav}")
        return nav, "explicit"

    if positions is not None and "nav" in positions.columns:
        nav_values = pd.to_numeric(positions["nav"], errors="coerce").dropna()
        nav_values = nav_values[nav_values > 0]
        if not nav_values.empty:
            return float(nav_values.iloc[0]), "positions.nav"

    state_path = (
        resolve_path(portfolio_state_path)
        if portfolio_state_path
        else PROJECT_ROOT / "outputs" / "live" / "portfolio_state.json"
    )
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        last_valuation = state.get("last_valuation") or {}
        if str(last_valuation.get("trade_date")) == str(prev_trade_date):
            nav = last_valuation.get("nav")
            if nav is not None and float(nav) > 0:
                return float(nav), f"portfolio_state.last_valuation[{prev_trade_date}]"

    valuation_path = (
        PROJECT_ROOT
        / "outputs"
        / "live"
        / "valuations"
        / f"valuation_{prev_trade_date}.json"
    )
    if valuation_path.exists():
        valuation = json.loads(valuation_path.read_text(encoding="utf-8"))
        nav = (valuation.get("summary") or {}).get("nav")
        if nav is not None and float(nav) > 0:
            return float(nav), f"valuation_{prev_trade_date}.json"

    nav = float((config.get("optimizer") or {}).get("portfolio_nav", 0.0))
    if nav <= 0:
        die("optimizer.portfolio_nav must be positive when no live NAV source is available")
    return nav, "optimizer.portfolio_nav"


def current_git_branch() -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    branch = completed.stdout.strip()
    return branch or "HEAD"


def git_commit_and_push(
    commit_msg: str,
    paths: list[str | Path],
    branch: str | None = None,
    push: bool = True,
) -> None:
    resolved_paths = [str(resolve_path(path).relative_to(PROJECT_ROOT)) for path in paths]
    subprocess.run(["git", "add", *resolved_paths], cwd=PROJECT_ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--short", "--", *resolved_paths],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not status.stdout.strip():
        print("  No live state changes to commit.")
    else:
        commit = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if commit.returncode != 0:
            print(f"  Git commit skipped/failed: {(commit.stderr or commit.stdout).strip()}")
            return
        print(f"  Git commit created: {commit_msg}")

    if push:
        target_branch = branch or current_git_branch()
        pushed = subprocess.run(
            ["git", "push", "origin", target_branch],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if pushed.returncode != 0:
            print(f"  Git push failed: {(pushed.stderr or pushed.stdout).strip()}")
            return
        print(f"  Git pushed to origin/{target_branch}.")
