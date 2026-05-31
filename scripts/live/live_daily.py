"""实盘每日交易脚本 — 交互式订单执行 + 自动 Git 推送

═══════════════════════════════════════════════════════════════
  每日操作步骤
═══════════════════════════════════════════════════════════════

  第1步 (数据更新):  从「A股数据」文件夹读取最新原始CSV，
                     经过 ingest→pool→state→mart 流水线处理，
                     生成模型可用的特征矩阵 (parquet)

                     ┌─ 首次运行: 全量构建所有步骤
                     └─ 日常运行: 增量模式
                       · ingest_raw:  MD5检测变更文件，只处理新增/修改的CSV
                       · build_pool:  跳过 (成分股池半年才调整一次)
                       · build_state: 只构建缺失日期的分区
                       · build_mart:  全量重建 (因rolling/EMA特征依赖完整历史)

  第2步 (模型推理):  加载训练好的 Transformer 模型，
                     用最近60个交易日的特征序列做前向传播，
                     得到每只股票的预测得分 (pred_score)

  第3步 (选股策略):  按预测得分降序排列，选 Top-20 只股票。
                     根据当前持仓状态决定操作类型：
                       - 建仓（首次）：买入 Top-20，等权各5%
                       - 调仓（每3天）：卖出最差1/3，买入新的 Top-K
                       - 持仓/补仓：仓位<80%时补仓，否则不动

  第4步 (交互执行):  逐笔显示买卖指令，你手动输入实际成交情况：
                       - 成交价（实际买入/卖出的价格）
                       - 成交量（实际买入/卖出的股数）
                     脚本自动更新持仓记录和现金余额

  第5步 (保存推送):  自动将今日的交易记录和持仓快照
                     commit + push 到 GitHub

═══════════════════════════════════════════════════════════════
  使用方法
═══════════════════════════════════════════════════════════════

  第1天 (首次建仓，自动全量构建):
    cd Final-OXX2
    python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260529 --reset

  第2天起 (自动增量更新):
    python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260601

  强制全量重建 (成分股池调整后):
    python scripts/live/live_daily.py --run-dag --full-dag --data-version v20260526 --end-date 20260601

  只推理不更新数据 (数据已是最新的):
    python scripts/live/live_daily.py --skip-dag

═══════════════════════════════════════════════════════════════
  前置条件
═══════════════════════════════════════════════════════════════

  1. 已训练模型: outputs/runs/StdTF-l60-cls-mse-f13/model.pt
     (如未训练，先运行: python scripts/modeling/train_sequence.py
      --config configs/models/StdTF-l60-cls-mse-f13.yaml)

  2. A股数据文件夹中有最新的原始CSV文件 (daily/, metric/, 等)

  3. conda 环境已激活 (包含 torch, pandas, pyarrow, numpy, pyyaml)

  4. git 已登录 GitHub (用于自动推送)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# ── 项目根目录 ────────────────────────────────────────────────
# scripts/live/live_daily.py → parents[2] = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 下避免 OpenMP 重复加载警告
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.models import TransformerStockModel  # noqa: E402

# ════════════════════════════════════════════════════════════════
#  常量配置
# ════════════════════════════════════════════════════════════════

# 模型使用的13个alpha特征（与训练时完全一致）
ALPHA_FEATURES: list[str] = [
    "lag1_net_mf_strength_20d_mean",   # 过去20日资金流向强度均值
    "lag1_net_mf_strength_60d_mean",   # 过去60日资金流向强度均值
    "lag1_close_position",             # 收盘价在当日高低范围内的相对位置
    "lag1_excess_ret_10d_mean",        # 过去10日超额收益均值
    "lag1_excess_ret_1d",              # 前一日超额收益
    "lag1_excess_ret_5d_mean",         # 过去5日超额收益均值
    "lag1_industry_neutral_ret_1d",    # 行业中性化后的前一日收益
    "lag1_ret_1d",                     # 前一日收益率
    "lag1_ret_20d",                    # 过去20日收益率
    "lag1_ret_5d_mean",               # 过去5日收益率均值
    "lag1_bollinger_z_20d",           # 20日布林带Z值（偏离均值的程度）
    "lag1_ma_ratio_20_60",            # 20日均线 / 60日均线
    "lag1_macd_hist",                  # MACD柱状图值
]

LOOKBACK = 60        # 序列回看天数：用过去60个交易日的特征
NUM_FEATURES = 13    # 特征数量

# 默认路径
DEFAULT_MODEL_CONFIG = "configs/models/StdTF-l60-cls-mse-f13.yaml"
DEFAULT_MODEL_PATH = "outputs/runs/StdTF-l60-cls-mse-f13/model.pt"
DEFAULT_MART_PATH = "data/mart/features_daily/features_daily_v20260526.parquet"
DEFAULT_POOL_PATH = "data/lake/core/chinext_pool/chinext_pool_scd2.parquet"
DEFAULT_STATE_PATH = "data/lake/state/security_daily_state.parquet"
DEFAULT_PORTFOLIO_STATE = "outputs/live/portfolio_state.json"
DEFAULT_ORDERS_DIR = "outputs/live/orders"

# 可交易性过滤参数（剔除不合适的股票）
LIQUIDITY_COL = "lag1_amount_20d_mean"  # 流动性指标：20日均成交额
SIZE_COL = "lag1_log_circ_mv"          # 市值指标：流通市值的对数
MIN_AMOUNT = 70_000.0                   # 最低日均成交额：7万元
BOTTOM_Q_LIQ = 0.05                    # 按日期剔除成交量最低5%的股票
BOTTOM_Q_SIZE = 0.05                    # 按日期剔除市值最低5%的股票

# 策略参数
DEFAULT_K = 20                          # 持仓股票数量：选得分最高的20只
DEFAULT_REBALANCE_STRIDE = 3            # 每隔几个交易日调仓一次
DEFAULT_REBALANCE_FRACTION = 0.33       # 每次调仓替换33%（约7只）
DEFAULT_MIN_POSITION_RATIO = 0.80       # 最低仓位要求：80%
DEFAULT_NAV = 10_000_000.0             # 默认初始资金：1000万
LOT_SIZE = 100                           # A股最小买入单位：100股（一手）


# ════════════════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════════════════

@dataclass
class Order:
    """一笔交易指令"""
    ts_code: str            # 股票代码，如 301001.SZ
    action: str            # "BUY" 买入 / "SELL" 卖出
    target_shares: int     # 目标买入/卖出股数（100的整数倍）
    target_weight: float   # 目标权重（占总资金的比例）
    close_price: float | None = None  # 参考价格（收盘价）
    target_value: float = 0.0        # 目标金额 = 股数 × 价格
    reason: str = ""       # 产生这笔指令的原因


@dataclass
class ExecutionResult:
    """一笔订单的实际执行结果"""
    ts_code: str
    action: str
    target_shares: int      # 计划股数
    actual_shares: int      # 实际成交股数
    actual_price: float     # 实际成交均价
    actual_value: float     # 实际成交金额
    status: str             # "filled"全量成交 / "partial"部分成交 / "failed"失败 / "skipped"跳过
    reason: str = ""


@dataclass
class Holding:
    """单只股票的持仓记录"""
    shares: int = 0         # 持有股数
    avg_cost: float = 0.0   # 平均买入成本
    weight_at_entry: float = 0.0  # 买入时的目标权重

    def market_value(self, price: float) -> float:
        """当前市值 = 股数 × 当前价格"""
        return self.shares * price


@dataclass
class PortfolioState:
    """整个投资组合的状态（持久化到 JSON 文件）"""
    last_signal_date: str = ""    # 上一次产生信号的日期
    day_index: int = 0             # 已经运行了多少个交易日
    cash: float = DEFAULT_NAV      # 当前现金余额
    initial_nav: float = DEFAULT_NAV  # 初始资金（用于计算收益率）
    holdings: dict[str, dict] = field(default_factory=dict)
    # holdings = {"301001.SZ": {"shares": 1000, "avg_cost": 25.30, "weight_at_entry": 0.05}}
    pending_orders: list[dict[str, Any]] = field(default_factory=list)

    def to_holdings(self) -> dict[str, Holding]:
        return {code: Holding(**vals) for code, vals in self.holdings.items()}

    def set_holdings(self, h: dict[str, Holding]) -> None:
        self.holdings = {code: asdict(v) for code, v in h.items()}

    def total_position_value(self, prices: dict[str, float]) -> float:
        """持仓总市值 = 所有持仓股票的 股数×价格 之和"""
        return sum(v["shares"] * prices.get(code, v.get("avg_cost", 0))
                   for code, v in self.holdings.items())

    def total_nav(self, prices: dict[str, float]) -> float:
        """总资产 = 现金 + 持仓市值"""
        return self.cash + self.total_position_value(prices)

    def position_ratio(self, prices: dict[str, float]) -> float:
        """仓位比例 = 持仓市值 / 总资产"""
        nav = self.total_nav(prices)
        return self.total_position_value(prices) / nav if nav > 0 else 0.0

    def load(self, path: Path) -> bool:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.last_signal_date = data.get("last_signal_date", "")
            self.day_index = data.get("day_index", 0)
            self.cash = data.get("cash", DEFAULT_NAV)
            self.initial_nav = data.get("initial_nav", DEFAULT_NAV)
            self.holdings = data.get("holdings", {})
            self.pending_orders = data.get("pending_orders", [])
            return True
        return False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "last_signal_date": self.last_signal_date,
                "day_index": self.day_index,
                "cash": self.cash,
                "initial_nav": self.initial_nav,
                "holdings": self.holdings,
                "pending_orders": self.pending_orders,
                "updated_at": datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _calc_shares(value: float, price: float | None) -> int:
    """计算可买入股数，向下取整到100股（一手）"""
    if not price or price <= 0 or value <= 0:
        return 0
    return int(value / price / LOT_SIZE) * LOT_SIZE


def _ask_input(prompt: str, default: str = "") -> str:
    """从终端读取用户输入，支持默认值"""
    if default:
        full_prompt = f"    {prompt} [默认: {default}]: "
    else:
        full_prompt = f"    {prompt}: "
    try:
        val = input(full_prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ════════════════════════════════════════════════════════════════
#  第1步：数据流水线 (DAG)
# ════════════════════════════════════════════════════════════════

def run_data_dag(data_version: str, end_date: str, incremental: bool = True) -> None:
    """运行数据流水线：原始CSV → 特征矩阵

    流水线步骤：
      1. ingest_raw:   读取 A股数据/ 下的CSV文件，写入 data/lake/raw/
                     (自动增量: 通过 MD5 检测，只处理新增/修改的文件)
      2. build_pool:   筛选创业板(399006.SZ)成分股，写入 data/lake/core/
                     (增量模式跳过: 成分股池很少变化，约每半年调整一次)
      3. build_state:  计算每只股票每日的状态（是否ST、涨跌停、停牌等）
                     (自动增量: 只构建缺失或源数据更新的日期分区)
      4. validate:     检查数据完整性
      5. build_mart:   计算所有技术特征（收益率、资金流向、布林带等）
                     (全量重建: 因 rolling/EMA 特征依赖完整历史序列)

    增量模式(--incremental) vs 全量模式:
      - 增量模式: 跳过 pool，其余步骤正常执行。适用于每日更新。
      - 全量模式: 所有步骤都执行。适用于首次构建或数据结构变化后。

    首次运行时必须使用全量模式 (incremental=False)。
    """
    print_header("第1步：更新数据 (运行数据流水线)")
    print("  正在从 A股数据/ 文件夹读取最新数据...")
    print(f"  数据版本: {data_version}, 截止日期: {end_date}")

    if incremental:
        print("  模式: 增量更新")
        print("    - ingest_raw: 自动检测变更文件 (MD5)")
        print("    - build_pool:  跳过 (成分股池稳定)")
        print("    - build_state: 只构建新增日期分区")
        print("    - build_mart:  全量重建 (rolling特征需要完整历史)")
    else:
        print("  模式: 全量构建 (所有步骤都执行)")
    print()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_daily_dag.py"),
        "--data-version", data_version,
        "--end-date", end_date,
    ]
    if incremental:
        cmd.append("--incremental")
    print(f"  执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")

    if result.returncode != 0:
        print(f"\n  [错误] 数据流水线失败！")
        print(f"  输出: {result.stdout[-500:]}")
        print(f"  错误: {result.stderr[-500:]}")
        raise RuntimeError("数据流水线执行失败")

    stdout = result.stdout.strip()
    json_start = stdout.rfind("{")
    if json_start >= 0:
        try:
            summary = json.loads(stdout[json_start:])
            steps = summary.get("steps", [])
            dag_mode = summary.get("dag_mode", "full")
            print(f"  流水线模式: {dag_mode}")
            for step in steps:
                name = step.get("step", "?")
                r = step.get("result", {})
                if r.get("skipped"):
                    reason = r.get("reason", "")
                    print(f"  [跳过] {name} {f'({reason})' if reason else ''}")
                else:
                    print(f"  [完成] {name}")
        except json.JSONDecodeError:
            print(f"  [输出] {stdout[-200:]}")

    print("\n  数据更新完成，特征矩阵已就绪。")


# ════════════════════════════════════════════════════════════════
#  第2步：加载并准备数据
# ════════════════════════════════════════════════════════════════

def load_mart_dataset(mart_path: Path) -> pd.DataFrame:
    """加载特征矩阵 (mart parquet文件)

    这个文件包含所有股票、所有交易日的特征数据。
    每行 = 一只股票在某一天的所有特征值。
    """
    if not mart_path.exists():
        raise FileNotFoundError(
            f"特征矩阵文件不存在: {mart_path}\n"
            f"请先运行数据流水线: python scripts/run_daily_dag.py --data-version <version>"
        )
    df = pd.read_parquet(mart_path)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    print(f"  特征矩阵加载成功:")
    print(f"    - 总行数: {df.shape[0]:,}")
    print(f"    - 特征列数: {df.shape[1]}")
    print(f"    - 日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    return df


def add_pool_filter(df: pd.DataFrame, pool_path: Path, latest_date: str) -> pd.DataFrame:
    """过滤：只保留创业板成分股

    pool文件记录了哪些股票在什么时间段内是创业板指数的成分股。
    我们只在这些成分股中进行选股。
    """
    if not pool_path.exists():
        print("  [跳过] 成分股文件不存在，使用全部股票")
        return df

    pool = pd.read_parquet(pool_path)
    pool["ts_code"] = pool["ts_code"].astype(str)
    pool["effective_from"] = pool["effective_from"].astype(str)
    pool["effective_to"] = pool["effective_to"].fillna("99991231").astype(str)

    active = pool[
        (pool["effective_from"] <= latest_date) &
        (pool["effective_to"] >= latest_date)
    ]["ts_code"].unique()

    n_before = len(df[df["trade_date"] == latest_date])
    df = df[df["ts_code"].isin(active)]
    n_after = len(df[df["trade_date"] == latest_date])

    print(f"  成分股过滤: 最新日期 {latest_date} 有 {n_before} 只股票 → 筛选后 {n_after} 只")
    return df


def add_tradable_mask(df: pd.DataFrame, state_path: Path) -> pd.DataFrame:
    """过滤：剔除不可交易的股票

    剔除条件：
      1. ST/*ST 股票（退市风险警示）
      2. 停牌股票
      3. 涨停/跌停（封板无法买入/卖出）
      4. 价格/成交量数据异常
      5. 流动性太差（日均成交额 < 7万 或排在当天最低5%）
      6. 市值太小（排在当天最低5%）
    """
    if not state_path.exists():
        print("  [跳过] 状态文件不存在，不做可交易性过滤")
        df["strict_tradable"] = True
        return df

    state = pd.read_parquet(state_path, columns=[
        "trade_date", "ts_code", "is_st", "is_suspended",
        "is_limit_up", "is_limit_down", "is_tradable",
        "price_valid", "volume_valid",
    ])
    state["trade_date"] = state["trade_date"].astype(str)
    state["ts_code"] = state["ts_code"].astype(str)
    state = state.drop_duplicates(["trade_date", "ts_code"], keep="last")

    n_before = len(df)
    df = df.merge(state, on=["trade_date", "ts_code"], how="left")

    # 各类过滤条件
    mask_st = df.get("is_st", False).fillna(False).astype(bool)
    mask_suspended = df.get("is_suspended", False).fillna(False).astype(bool)
    mask_limit = (
        df.get("is_limit_up", False).fillna(False).astype(bool) |
        df.get("is_limit_down", False).fillna(False).astype(bool)
    )
    mask_price_invalid = ~df.get("price_valid", True).fillna(True).astype(bool)
    mask_volume_invalid = ~df.get("volume_valid", True).fillna(True).astype(bool)

    # 流动性过滤
    if LIQUIDITY_COL in df.columns:
        amt = pd.to_numeric(df[LIQUIDITY_COL], errors="coerce")
        mask_low_amt = amt.isna() | amt.lt(MIN_AMOUNT)
        q_thresh = df.groupby("trade_date", sort=False)[LIQUIDITY_COL].transform(
            lambda x: pd.to_numeric(x, errors="coerce").quantile(BOTTOM_Q_LIQ)
        )
        mask_low_amt = mask_low_amt | amt.lt(q_thresh)
    else:
        mask_low_amt = pd.Series(False, index=df.index)

    # 市值过滤
    if SIZE_COL in df.columns:
        sz = pd.to_numeric(df[SIZE_COL], errors="coerce")
        q_sz = df.groupby("trade_date", sort=False)[SIZE_COL].transform(
            lambda x: pd.to_numeric(x, errors="coerce").quantile(BOTTOM_Q_SIZE)
        )
        mask_microcap = sz.lt(q_sz)
    else:
        mask_microcap = pd.Series(False, index=df.index)

    # 综合过滤
    df["strict_tradable"] = ~(mask_st | mask_suspended | mask_limit |
                               mask_price_invalid | mask_volume_invalid |
                               mask_low_amt | mask_microcap)

    removed = (~df["strict_tradable"]).sum()
    df = df[df["strict_tradable"]].reset_index(drop=True)
    print(f"  可交易性过滤: 移除 {removed} 条记录，剩余 {len(df)} 条")

    reasons = {
        "ST股": mask_st.sum(),
        "停牌": mask_suspended.sum(),
        "涨跌停": mask_limit.sum(),
        "价格异常": mask_price_invalid.sum(),
        "成交量异常": mask_volume_invalid.sum(),
    }
    for reason, count in reasons.items():
        if count > 0:
            print(f"    - {reason}: {count} 条")
    print(f"    - 流动性/市值过低: 约 {removed - sum(reasons.values())} 条")

    return df


# ════════════════════════════════════════════════════════════════
#  第3步：构建序列 + 模型推理
# ════════════════════════════════════════════════════════════════

def build_live_sequences(
    df: pd.DataFrame,
    latest_date: str,
    features: list[str],
    lookback: int,
) -> tuple[np.ndarray, list[str]]:
    """为最新日期的每只股票构建60天的特征序列

    模型输入格式: [股票数量, 60天, 13个特征]
    每只股票取最近60个交易日的13个特征值组成一个序列。

    如果某只股票上市不足60天或数据有缺失，则跳过该股票。
    """
    required = ["trade_date", "ts_code"] + features
    avail = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [警告] 缺少特征列: {missing}")

    panel = df[avail].copy()
    for f in features:
        if f in panel.columns:
            panel[f] = pd.to_numeric(panel[f], errors="coerce")
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    sequences = []
    stock_codes = []

    for ts_code, group in panel.groupby("ts_code", sort=True):
        group_data = group[features].to_numpy(np.float32)
        dates = group["trade_date"].values
        latest_idx = np.where(dates == latest_date)[0]
        if len(latest_idx) == 0:
            continue
        latest_i = latest_idx[0]
        start_i = latest_i - lookback + 1
        if start_i < 0:
            continue  # 历史数据不足60天
        win = group_data[start_i:latest_i + 1]
        if np.isnan(win).any():
            continue  # 有缺失值
        sequences.append(win)
        stock_codes.append(ts_code)

    if not sequences:
        return np.empty((0, lookback, len(features)), np.float32), []

    X = np.stack(sequences).astype(np.float32)
    print(f"  序列构建完成: {X.shape[0]} 只股票 × {lookback} 天 × {len(features)} 个特征")
    return X, stock_codes


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    """加载训练好的 Transformer 模型"""
    if not config_path.exists():
        raise FileNotFoundError(
            f"模型配置文件不存在: {config_path}\n"
            f"请确认 configs/models/ 目录下有对应的 YAML 文件"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"模型权重文件不存在: {checkpoint_path}\n"
            f"请先训练模型:\n"
            f"  python scripts/modeling/train_sequence.py \\\n"
            f"    --config configs/models/StdTF-l60-cls-mse-f13.yaml"
        )

    config = _load_yaml(config_path)
    model_config = config.get("model", {})
    model_name = model_config.get("name", "transformer_encoder")

    if model_name == "transformer_encoder":
        model = TransformerStockModel(num_features=NUM_FEATURES, config=model_config)
    elif model_name == "transformer_enhanced":
        from src.models import EnhancedTransformerModel
        model = EnhancedTransformerModel(num_features=NUM_FEATURES, config=model_config)
    elif model_name == "gru_baseline":
        from src.models import GRUStockModel
        model = GRUStockModel(num_features=NUM_FEATURES, config=model_config)
    else:
        raise ValueError(f"未知模型类型: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"  模型加载成功:")
    print(f"    - 类型: {model_name}")
    print(f"    - 来源: {checkpoint_path}")
    print(f"    - 设备: {device}")
    return model


@torch.no_grad()
def run_inference(model, X: np.ndarray, stock_codes: list[str], device: torch.device) -> pd.DataFrame:
    """运行模型前向传播，得到每只股票的预测得分"""
    if len(stock_codes) == 0:
        return pd.DataFrame(columns=["ts_code", "pred_score"])

    tensor = torch.from_numpy(X).to(device, non_blocking=True)
    pred = model(tensor).detach().cpu().numpy().flatten()

    result = pd.DataFrame({
        "ts_code": stock_codes,
        "pred_score": pred,
    }).sort_values("pred_score", ascending=False).reset_index(drop=True)

    print(f"  推理完成:")
    print(f"    - 参与评分的股票数: {len(result)}")
    print(f"    - 得分均值: {result['pred_score'].mean():.6f}")
    print(f"    - 得分标准差: {result['pred_score'].std():.6f}")
    print(f"    - 最高分: {result['pred_score'].iloc[0]:.6f} ({result['ts_code'].iloc[0]})")
    print(f"    - 最低分: {result['pred_score'].iloc[-1]:.6f} ({result['ts_code'].iloc[-1]})")
    return result


# ════════════════════════════════════════════════════════════════
#  第4步：选股策略 + 订单生成
# ════════════════════════════════════════════════════════════════

class LivePortfolioManager:
    """投资组合管理器：根据预测得分和策略生成买卖指令"""

    def __init__(
        self,
        k: int = DEFAULT_K,
        rebalance_stride: int = DEFAULT_REBALANCE_STRIDE,
        rebalance_fraction: float = DEFAULT_REBALANCE_FRACTION,
        min_position_ratio: float = DEFAULT_MIN_POSITION_RATIO,
    ):
        self.k = k
        self.rebalance_stride = rebalance_stride
        self.rebalance_fraction = rebalance_fraction
        self.min_position_ratio = min_position_ratio
        self.state = PortfolioState()

    def load_state(self, path: Path) -> None:
        loaded = self.state.load(path)
        if loaded:
            n_stocks = len(self.state.holdings)
            print(f"  持仓状态加载成功:")
            print(f"    - 已运行交易日数: {self.state.day_index}")
            print(f"    - 当前持仓: {n_stocks} 只股票")
            print(f"    - 现金余额: ¥{self.state.cash:,.0f}")
            print(f"    - 上次信号日期: {self.state.last_signal_date}")
        else:
            print("  未找到持仓状态文件，将作为首次运行（建仓模式）")

    def save_state(self, path: Path) -> None:
        self.state.save(path)

    def generate_orders(
        self,
        predictions: pd.DataFrame,
        latest_date: str,
        prices: dict[str, float],
    ) -> list[Order]:
        """根据预测得分和策略生成交易指令

        决定今天的操作类型：
          - 如果没有任何持仓 → 建仓：买入 Top-20，每只5%仓位
          - 如果是调仓日(每3天) → 调仓：卖出得分最低的7只，买入新进入Top-20的
          - 如果仓位 < 80% → 补仓：买入Top-K中未持仓的股票
          - 否则 → 持仓不动
        """
        day_idx = self.state.day_index
        is_rebalance_day = (day_idx % self.rebalance_stride == 0)
        h = self.state.to_holdings()
        current_nav = self.state.total_nav(prices)
        orders: list[Order] = []

        pos_val = sum(v.market_value(prices.get(code, v.avg_cost))
                      for code, v in h.items())
        pos_ratio = pos_val / current_nav if current_nav > 0 else 0.0

        print_header("第4步：生成交易指令")
        print(f"  当前状态:")
        print(f"    - 交易日序号: 第 {day_idx} 天 (从0开始计数)")
        print(f"    - 总资产(NAV): ¥{current_nav:,.0f}")
        print(f"    - 持仓市值: ¥{pos_val:,.0f} ({pos_ratio:.1%})")
        print(f"    - 现金余额: ¥{self.state.cash:,.0f} ({1 - pos_ratio:.1%})")
        print(f"    - 持仓股票数: {len(h)} / {self.k}")
        print(f"    - 是否调仓日: {'是' if is_rebalance_day else '否'} "
              f"(每{self.rebalance_stride}天调仓一次，下一次调仓: 第{day_idx + (self.rebalance_stride - day_idx % self.rebalance_stride) if day_idx % self.rebalance_stride != 0 else 0}天)")

        if not h:
            # ─── 建仓 ───
            print(f"\n  [操作类型] 首次建仓")
            print(f"  策略: 买入预测得分最高的 {self.k} 只股票，每只占总资金的 {1/self.k:.0%}")
            print()

            top_k = predictions.head(self.k)
            for _, row in top_k.iterrows():
                code = row["ts_code"]
                price = prices.get(code)
                target_weight = 1.0 / self.k
                target_value = current_nav * target_weight
                shares = _calc_shares(target_value, price)
                orders.append(Order(
                    ts_code=code, action="BUY",
                    target_shares=shares, target_weight=target_weight,
                    close_price=price, target_value=target_value,
                    reason="首次建仓",
                ))

        elif is_rebalance_day:
            # ─── 调仓 ───
            n_hold = len(h)
            n_replace = max(1, int(n_hold * self.rebalance_fraction))
            print(f"\n  [操作类型] 定期调仓 (第{day_idx}天)")
            print(f"  策略: 卖出当前持仓中预测得分最低的 {n_replace} 只，"
                  f"用释放的资金买入新的 Top-{self.k} 股票")
            print()

            # 按得分从低到高排列持仓
            pred_by_code = dict(zip(predictions["ts_code"], predictions["pred_score"]))
            held_sorted = sorted(h.items(), key=lambda x: pred_by_code.get(x[0], -999.0))

            # 卖出得分最低的N只
            to_sell = [code for code, _ in held_sorted[:n_replace]]
            sell_value_freed = 0.0
            for code in to_sell:
                holding = h[code]
                price = prices.get(code, holding.avg_cost)
                shares = holding.shares
                value = shares * price
                rank = list(pred_by_code.keys()).index(code) + 1 if code in pred_by_code else "?"
                print(f"    卖出 {code} — {shares}股 × ¥{price:.2f} = ¥{value:,.0f} "
                      f"(排名第{rank}, 得分={pred_by_code.get(code, 0):.6f})")
                orders.append(Order(
                    ts_code=code, action="SELL",
                    target_shares=shares, target_weight=0.0,
                    close_price=price, target_value=value,
                    reason=f"调仓卖出(得分最低{n_replace}只)",
                ))
                sell_value_freed += value

            # 从Top-K中买入不在持仓中的股票
            top_codes = set(predictions.head(self.k)["ts_code"].tolist())
            candidates = [c for c in predictions["ts_code"]
                          if c in top_codes and c not in h]
            n_new = max(1, self.k - len(h))

            print(f"\n    释放资金: ¥{sell_value_freed:,.0f}")
            print(f"    可买新股票: {n_new} 只")
            print(f"    候选股票: {[c for c in candidates[:n_new]]}")
            print()

            for code in candidates:
                if sell_value_freed <= 1e-6 or len(h) >= self.k:
                    break
                w = min(sell_value_freed / n_new, sell_value_freed)
                price = prices.get(code)
                shares = _calc_shares(current_nav * w, price)
                print(f"    买入 {code} — {shares}股 × ¥{price:.2f} = ¥{current_nav * w:,.0f} "
                      f"(得分={pred_by_code.get(code, 0):.6f})")
                orders.append(Order(
                    ts_code=code, action="BUY",
                    target_shares=shares, target_weight=w,
                    close_price=price, target_value=current_nav * w,
                    reason="调仓买入",
                ))
                sell_value_freed -= current_nav * w

        else:
            # ─── 持仓/补仓 ───
            if pos_ratio < self.min_position_ratio:
                deficit = (self.min_position_ratio - pos_ratio) * current_nav
                print(f"\n  [操作类型] 紧急补仓")
                print(f"  原因: 当前仓位 {pos_ratio:.1%} 低于最低要求 {self.min_position_ratio:.0%}")
                print(f"  需要补仓金额: ¥{deficit:,.0f}")
                print()

                top_codes = set(predictions.head(self.k)["ts_code"].tolist())
                candidates = [c for c in predictions["ts_code"]
                              if c in top_codes and c not in h]
                n_cand = max(1, len(candidates))
                per_stock = deficit / n_cand

                for code in candidates:
                    if self.state.cash < per_stock * 0.5:
                        break
                    price = prices.get(code)
                    shares = _calc_shares(per_stock, price)
                    print(f"    买入 {code} — {shares}股 × ¥{price:.2f} = ¥{per_stock:,.0f}")
                    orders.append(Order(
                        ts_code=code, action="BUY",
                        target_shares=shares,
                        target_weight=per_stock / current_nav,
                        close_price=price, target_value=per_stock,
                        reason="紧急补仓",
                    ))
            else:
                print(f"\n  [操作类型] 持仓不动")
                print(f"  原因: 当前仓位 {pos_ratio:.1%} 满足要求 (>= {self.min_position_ratio:.0%})，"
                      f"且今天不是调仓日")

        print()
        return orders


def extract_prices(df: pd.DataFrame, latest_date: str) -> dict[str, float]:
    """从特征矩阵中提取最新日期的收盘价"""
    prices = {}
    day_df = df[df["trade_date"] == latest_date]
    if "close" in day_df.columns:
        for _, row in day_df.iterrows():
            p = pd.to_numeric(row.get("close"), errors="coerce")
            if pd.notna(p) and p > 0:
                prices[str(row["ts_code"])] = float(p)
    if prices:
        print(f"  提取了 {len(prices)} 只股票的收盘价 (作为参考价)")
    else:
        print(f"  [注意] 特征矩阵中没有收盘价列，订单中的价格将显示为空")
        print(f"         你需要在执行时手动输入实际价格")
    return prices


def print_top_k(predictions: pd.DataFrame, k: int, prices: dict[str, float]) -> None:
    """打印 Top-K 选股排名"""
    print(f"\n  ┌──────┬──────────────┬────────────┬──────────┐")
    print(f"  │ 排名 │   股票代码    │  预测得分   │ 参考收盘价│")
    print(f"  ├──────┼──────────────┼────────────┼──────────┤")
    for i, (_, row) in enumerate(predictions.head(k).iterrows(), 1):
        code = row["ts_code"]
        score = row["pred_score"]
        price = prices.get(code, 0)
        price_str = f"¥{price:.2f}" if price > 0 else "  N/A  "
        in_top = " ◄" if i <= 20 else ""
        print(f"  │ {i:>4} │ {code:<12} │ {score:>10.6f} │{price_str:>9} │{in_top}")
    print(f"  └──────┴──────────────┴────────────┴──────────┘")
    print(f"\n  (◄ = 进入 Top-20 持仓候选)")


# ════════════════════════════════════════════════════════════════
#  第5步：交互式订单执行
# ════════════════════════════════════════════════════════════════

def interactive_execute_orders(
    orders: list[Order],
    state: PortfolioState,
    prices: dict[str, float],
) -> list[ExecutionResult]:
    """逐笔展示订单，由用户确认实际成交情况

    对于每笔订单，用户需要输入：
      - 实际成交价（可能与参考价不同）
      - 实际成交股数（可能因涨停/资金不足等原因无法全量成交）

    可用操作：
      y = 按建议数量执行
      n = 跳过这笔订单
      p = 部分成交（自定义股数）
      c = 完全自定义（自定义价格和股数）
    """
    h = state.to_holdings()
    results: list[ExecutionResult] = []

    if not orders:
        print_header("第5步：执行交易指令")
        print("  今天没有任何交易指令。所有持仓保持不变。")
        return results

    sell_orders = [o for o in orders if o.action == "SELL"]
    buy_orders = [o for o in orders if o.action == "BUY"]

    print_header("第5步：执行交易指令（逐笔确认）")
    print(f"  共 {len(sell_orders)} 笔卖出 + {len(buy_orders)} 笔买入")
    print(f"  当前现金: ¥{state.cash:,.0f}")
    print()
    print("  操作说明:")
    print("    y = 确认执行 (按建议数量)")
    print("    n = 跳过这笔订单")
    print("    p = 部分成交 (手动输入实际股数)")
    print("    c = 完全自定义 (手动输入价格和股数)")
    print()

    # ── 先处理卖出 ──
    if sell_orders:
        print(f"  ╔{'═' * 60}╗")
        print(f"  ║  卖出订单 ({len(sell_orders)} 笔)                            ║")
        print(f"  ╠{'═' * 60}╣")

        for i, o in enumerate(sell_orders, 1):
            price_str = f"¥{o.close_price:.2f}" if o.close_price else "N/A"
            print(f"  ║")
            print(f"  ║  [{i}/{len(sell_orders)}] 卖出 {o.ts_code}")
            print(f"  ║    计划: {o.target_shares} 股 @ {price_str} = ¥{o.target_value:,.0f}")
            print(f"  ║    原因: {o.reason}")

            choice = _ask_input("执行?", "y").lower()

            if choice == "n":
                results.append(ExecutionResult(
                    ts_code=o.ts_code, action="SELL",
                    target_shares=o.target_shares, actual_shares=0,
                    actual_price=0, actual_value=0,
                    status="skipped", reason="手动跳过",
                ))
                print(f"  ║    → 已跳过")
                continue

            # 输入实际成交价
            default_price = f"{o.close_price:.2f}" if o.close_price else ""
            price_input = _ask_input("实际成交价", default_price)
            try:
                actual_price = float(price_input)
            except ValueError:
                actual_price = o.close_price or 0
                print(f"  ║    [提示] 价格输入无效，使用参考价 ¥{actual_price:.2f}")

            # 输入实际成交股数
            if choice == "c":
                shares_input = _ask_input("实际卖出股数 (100的倍数)", str(o.target_shares))
            elif choice == "p":
                shares_input = _ask_input("实际卖出股数 (100的倍数)", str(o.target_shares))
            else:
                shares_input = str(o.target_shares)

            try:
                actual_shares = int(shares_input)
                actual_shares = (actual_shares // LOT_SIZE) * LOT_SIZE
            except ValueError:
                actual_shares = o.target_shares
                print(f"  ║    [提示] 股数输入无效，使用计划数量 {actual_shares}")

            actual_value = actual_shares * actual_price

            # 更新持仓
            if o.ts_code in h:
                old = h[o.ts_code]
                old.shares -= actual_shares
                if old.shares <= 0:
                    del h[o.ts_code]
                else:
                    h[o.ts_code] = old
                state.set_holdings(h)

            state.cash += actual_value

            status = "filled" if actual_shares >= o.target_shares else "partial"
            results.append(ExecutionResult(
                ts_code=o.ts_code, action="SELL",
                target_shares=o.target_shares, actual_shares=actual_shares,
                actual_price=actual_price, actual_value=actual_value,
                status=status,
            ))

            status_cn = "全部成交" if status == "filled" else "部分成交"
            print(f"  ║    → {status_cn}: {actual_shares}股 × ¥{actual_price:.2f} = ¥{actual_value:,.0f}")
            print(f"  ║    → 现金余额: ¥{state.cash:,.0f}")
            print(f"  ║")

        print(f"  ╚{'═' * 60}╝\n")

    # ── 再处理买入 ──
    if buy_orders:
        print(f"  ╔{'═' * 60}╗")
        print(f"  ║  买入订单 ({len(buy_orders)} 笔)                            ║")
        print(f"  ╠{'═' * 60}╣")

        for i, o in enumerate(buy_orders, 1):
            price_str = f"¥{o.close_price:.2f}" if o.close_price else "N/A"
            print(f"  ║")
            print(f"  ║  [{i}/{len(buy_orders)}] 买入 {o.ts_code}")
            print(f"  ║    计划: {o.target_shares} 股 @ {price_str} ≈ ¥{o.target_value:,.0f}")
            print(f"  ║    原因: {o.reason}")

            choice = _ask_input("执行?", "y").lower()

            if choice == "n":
                results.append(ExecutionResult(
                    ts_code=o.ts_code, action="BUY",
                    target_shares=o.target_shares, actual_shares=0,
                    actual_price=0, actual_value=0,
                    status="skipped", reason="手动跳过",
                ))
                print(f"  ║    → 已跳过")
                continue

            # 输入实际成交价
            default_price = f"{o.close_price:.2f}" if o.close_price else ""
            price_input = _ask_input("实际成交价", default_price)
            try:
                actual_price = float(price_input)
            except ValueError:
                actual_price = o.close_price or 0
                print(f"  ║    [提示] 价格输入无效，使用参考价 ¥{actual_price:.2f}")

            # 输入实际成交股数
            if choice in ("p", "c"):
                shares_input = _ask_input("实际买入股数 (100的倍数)", str(o.target_shares))
                try:
                    actual_shares = int(shares_input)
                    actual_shares = (actual_shares // LOT_SIZE) * LOT_SIZE
                except ValueError:
                    actual_shares = o.target_shares
            else:
                actual_shares = o.target_shares

            actual_value = actual_shares * actual_price

            # 检查资金是否充足
            if actual_value > state.cash:
                affordable_shares = int(state.cash / actual_price / LOT_SIZE) * LOT_SIZE
                if affordable_shares <= 0:
                    print(f"  ║    → 资金不足！需要 ¥{actual_value:,.0f}，现金仅 ¥{state.cash:,.0f}")
                    results.append(ExecutionResult(
                        ts_code=o.ts_code, action="BUY",
                        target_shares=o.target_shares, actual_shares=0,
                        actual_price=actual_price, actual_value=0,
                        status="failed", reason="资金不足",
                    ))
                    print(f"  ║    → 买入失败")
                    continue
                print(f"  ║    → 资金不足，自动调整为 {affordable_shares} 股 "
                      f"(¥{affordable_shares * actual_price:,.0f})")
                actual_shares = affordable_shares
                actual_value = actual_shares * actual_price

            # 更新持仓和现金
            state.cash -= actual_value
            if o.ts_code in h:
                old = h[o.ts_code]
                total_cost = old.avg_cost * old.shares + actual_price * actual_shares
                old.shares += actual_shares
                old.avg_cost = total_cost / old.shares if old.shares > 0 else 0
                h[o.ts_code] = old
            else:
                h[o.ts_code] = Holding(
                    shares=actual_shares,
                    avg_cost=actual_price,
                    weight_at_entry=o.target_weight,
                )
            state.set_holdings(h)

            status = "filled" if actual_shares >= o.target_shares else "partial"
            results.append(ExecutionResult(
                ts_code=o.ts_code, action="BUY",
                target_shares=o.target_shares, actual_shares=actual_shares,
                actual_price=actual_price, actual_value=actual_value,
                status=status,
            ))

            status_cn = "全部成交" if status == "filled" else "部分成交"
            print(f"  ║    → {status_cn}: {actual_shares}股 × ¥{actual_price:.2f} = ¥{actual_value:,.0f}")
            print(f"  ║    → 现金余额: ¥{state.cash:,.0f}")
            print(f"  ║")

        print(f"  ╚{'═' * 60}╝\n")

    return results


# ════════════════════════════════════════════════════════════════
#  持仓汇总 & 执行日志
# ════════════════════════════════════════════════════════════════

def print_portfolio_summary(state: PortfolioState, prices: dict[str, float]) -> None:
    """打印当前持仓汇总"""
    h = state.to_holdings()
    nav = state.total_nav(prices)
    pos_val = state.total_position_value(prices)
    pos_ratio = pos_val / nav if nav > 0 else 0
    total_return = (nav - state.initial_nav) / state.initial_nav if state.initial_nav > 0 else 0

    print_header("当前持仓汇总")
    print(f"  总资产(NAV):     ¥{nav:>14,.0f}")
    print(f"  现金余额:        ¥{state.cash:>14,.0f} ({state.cash / nav:.1%})")
    print(f"  持仓市值:        ¥{pos_val:>14,.0f} ({pos_ratio:.1%})")
    print(f"  累计收益率:      {total_return:>+13.2%}")
    print(f"  持仓股票数:      {len(h)} 只")
    print(f"  交易日序号:      第 {state.day_index} 天")

    if h:
        print(f"\n  ┌──────────────┬────────┬──────────┬──────────┬──────────────┬──────────┐")
        print(f"  │   股票代码    │  持仓股数 │  均买入价 │  参考现价 │    市值(元)  │  浮动盈亏 │")
        print(f"  ├──────────────┼────────┼──────────┼──────────┼──────────────┼──────────┤")

        total_pnl = 0
        for code, holding in sorted(h.items()):
            cur_price = prices.get(code, holding.avg_cost)
            mv = holding.market_value(cur_price)
            cost = holding.shares * holding.avg_cost
            pnl = mv - cost
            pnl_pct = pnl / cost if cost > 0 else 0
            total_pnl += pnl
            pnl_sign = "+" if pnl >= 0 else " "
            print(f"  │ {code:<12} │ {holding.shares:>6} │ ¥{holding.avg_cost:>7.2f} │ ¥{cur_price:>7.2f} │ ¥{mv:>11,.0f} │{pnl_sign}{pnl_pct:>8.1%} │")

        print(f"  ├──────────────┴────────┴──────────┴──────────┴──────────────┼──────────┤")
        total_pnl_pct = total_pnl / (nav - state.cash) if (nav - state.cash) > 0 else 0
        pnl_sign = "+" if total_pnl >= 0 else " "
        print(f"  │ 浮动盈亏合计: ¥{total_pnl:>+11,.0f}                           │{pnl_sign}{total_pnl_pct:>8.1%} │")
        print(f"  └──────────────────────────────────────────────────────────────┴──────────┘")

    print()


def save_execution_log(
    results: list[ExecutionResult],
    orders_dir: Path,
    latest_date: str,
    state: PortfolioState,
) -> None:
    """保存执行日志到 JSON 文件"""
    orders_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "signal_date": latest_date,
        "executions": [
            {
                "ts_code": r.ts_code,
                "action": r.action,
                "target_shares": r.target_shares,
                "actual_shares": r.actual_shares,
                "actual_price": r.actual_price,
                "actual_value": r.actual_value,
                "status": r.status,
                "reason": r.reason,
            }
            for r in results
        ],
        "summary": {
            "buy_filled": sum(1 for r in results if r.action == "BUY" and r.status == "filled"),
            "buy_partial": sum(1 for r in results if r.action == "BUY" and r.status == "partial"),
            "buy_failed": sum(1 for r in results if r.action == "BUY" and r.status in ("skipped", "failed")),
            "sell_filled": sum(1 for r in results if r.action == "SELL" and r.status == "filled"),
            "sell_partial": sum(1 for r in results if r.action == "SELL" and r.status == "partial"),
            "sell_failed": sum(1 for r in results if r.action == "SELL" and r.status in ("skipped", "failed")),
            "total_buy_value": sum(r.actual_value for r in results if r.action == "BUY"),
            "total_sell_value": sum(r.actual_value for r in results if r.action == "SELL"),
            "post_cash": state.cash,
        },
        "logged_at": datetime.now().isoformat(),
    }
    path = orders_dir / f"execution_{latest_date}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  执行日志已保存: {path}")


# ════════════════════════════════════════════════════════════════
#  第6步：Git 提交 & 推送
# ════════════════════════════════════════════════════════════════

def git_commit_and_push(state: PortfolioState, latest_date: str) -> None:
    """将今天的交易记录和持仓状态提交到 Git 并推送到 GitHub"""
    print_header("第6步：提交到 GitHub")
    print("  将以下文件添加到 Git 并推送:")
    print("    - outputs/live/portfolio_state.json  (持仓状态)")
    print("    - outputs/live/orders/              (执行日志)")
    print()

    try:
        # 查看要提交的文件
        check_result = subprocess.run(
            ["git", "status", "--short", "outputs/live/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        if check_result.stdout.strip():
            print(f"  待提交文件:")
            for line in check_result.stdout.strip().split("\n"):
                print(f"    {line}")
        else:
            print("  没有新的变更需要提交")

            # 仍然尝试推送（可能之前有未推送的提交）
            push_result = subprocess.run(
                ["git", "push", "origin", "OXX2"],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            if push_result.returncode == 0:
                print("  已推送到远程仓库")
            else:
                print(f"  推送结果: {push_result.stdout.strip() or push_result.stderr.strip()}")
            return

        print()

        # git add
        add_result = subprocess.run(
            ["git", "add", "outputs/live/portfolio_state.json",
             "outputs/live/orders/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )

        # git commit
        commit_msg = f"live: 交易日{latest_date} 执行记录 (day {state.day_index})"
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )

        if commit_result.returncode != 0:
            print(f"  [提示] 提交失败: {commit_result.stderr.strip()}")
            print(f"  你可以手动执行:")
            print(f"    cd {PROJECT_ROOT}")
            print(f'    git add outputs/live/ && git commit -m "{commit_msg}"')
            print(f"    git push origin OXX2")
            return

        print(f"  提交成功: {commit_msg}")

        # git push
        print(f"  正在推送到 GitHub...")
        push_result = subprocess.run(
            ["git", "push", "origin", "OXX2"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )

        if push_result.returncode == 0:
            print(f"  推送成功!")
        else:
            print(f"  [注意] 推送失败: {push_result.stderr.strip()}")
            print(f"  请稍后手动执行: git push origin OXX2")

    except Exception as e:
        print(f"  [错误] Git 操作失败: {e}")
        print(f"  你可以手动执行:")
        print(f"    cd {PROJECT_ROOT}")
        print(f"    git add outputs/live/")
        print(f"    git commit -m 'live: 交易日{latest_date}'")
        print(f"    git push origin OXX2")


# ════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="实盘每日交易脚本 — 交互式执行 + 自动 Git 推送",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 第1天（首次建仓，自动全量构建流水线）:
  python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260529 --reset

  # 第2天起（自动增量更新）:
  python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260601

  # 强制全量重建流水线（成分股池调整后使用）:
  python scripts/live/live_daily.py --run-dag --full-dag --data-version v20260526 --end-date 20260601

  # 跳过数据更新:
  python scripts/live/live_daily.py --skip-dag

  # 不推送Git:
  python scripts/live/live_daily.py --skip-dag --no-push
        """,
    )
    p.add_argument("--run-dag", action="store_true",
                   help="运行数据流水线（更新特征矩阵）")
    p.add_argument("--skip-dag", action="store_true",
                   help="跳过数据流水线（数据已是最新的）")
    p.add_argument("--full-dag", action="store_true",
                   help="强制全量构建流水线（不使用增量模式）")
    p.add_argument("--data-version", default="v20260526",
                   help="数据版本号 (默认: v20260526)")
    p.add_argument("--end-date", default="",
                   help="数据截止日期 YYYYMMDD (运行DAG时使用)")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--mart-path", default=DEFAULT_MART_PATH)
    p.add_argument("--pool-path", default=DEFAULT_POOL_PATH)
    p.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    p.add_argument("--portfolio-state", default=DEFAULT_PORTFOLIO_STATE)
    p.add_argument("--orders-dir", default=DEFAULT_ORDERS_DIR)
    p.add_argument("--initial-nav", type=float, default=DEFAULT_NAV,
                   help=f"初始资金 (默认: ¥{DEFAULT_NAV:,.0f})")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"持仓股票数 (默认: {DEFAULT_K})")
    p.add_argument("--rebalance-stride", type=int, default=DEFAULT_REBALANCE_STRIDE,
                   help=f"调仓间隔天数 (默认: {DEFAULT_REBALANCE_STRIDE})")
    p.add_argument("--rebalance-fraction", type=float, default=DEFAULT_REBALANCE_FRACTION,
                   help=f"每次调仓替换比例 (默认: {DEFAULT_REBALANCE_FRACTION})")
    p.add_argument("--min-position-ratio", type=float, default=DEFAULT_MIN_POSITION_RATIO,
                   help=f"最低仓位比例 (默认: {DEFAULT_MIN_POSITION_RATIO})")
    p.add_argument("--reset", action="store_true",
                   help="重置持仓状态（从零开始建仓）")
    p.add_argument("--no-push", action="store_true",
                   help="不自动推送到 GitHub")
    p.add_argument("--device", default="auto",
                   help="计算设备: auto/cpu/cuda (默认: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print_header(f"实盘交易系统 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目录: {PROJECT_ROOT}")
    print(f"  模型: StdTF-l60-cls-mse-f13")
    print(f"  策略: Top-{args.k} 等权, 每{args.rebalance_stride}天调仓{args.rebalance_fraction:.0%}")

    # 路径解析
    model_config_path = PROJECT_ROOT / args.model_config
    model_path = PROJECT_ROOT / args.model_path
    mart_path = PROJECT_ROOT / args.mart_path
    pool_path = PROJECT_ROOT / args.pool_path
    state_path = PROJECT_ROOT / args.state_path
    portfolio_state_path = PROJECT_ROOT / args.portfolio_state
    orders_dir = PROJECT_ROOT / args.orders_dir

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  计算设备: {device}")

    # ── 第1步：数据更新 ──
    if args.run_dag:
        end_date = args.end_date or datetime.now().strftime("%Y%m%d")
        # 首次运行(--reset)时使用全量模式，后续使用增量模式
        use_incremental = not args.full_dag and not args.reset
        run_data_dag(args.data_version, end_date, incremental=use_incremental)
    elif not args.skip_dag:
        print("\n  [提示] 未指定 --run-dag 或 --skip-dag，将直接使用已有数据")

    # ── 第2步：加载数据 ──
    print_header("第2步：加载特征数据")
    df = load_mart_dataset(mart_path)
    latest_date = df["trade_date"].max()
    print(f"  数据中最新日期: {latest_date}")
    print()

    print("  过滤股票池 (只保留创业板成分股)...")
    df = add_pool_filter(df, pool_path, latest_date)

    print("  过滤不可交易的股票 (ST、停牌、涨跌停、流动性差等)...")
    df = add_tradable_mask(df, state_path)
    print()

    # ── 第3步：序列构建 + 模型推理 ──
    print_header("第3步：模型推理")
    print(f"  为最新日期 {latest_date} 构建过去{LOOKBACK}天的特征序列...")
    X, stock_codes = build_live_sequences(df, latest_date, ALPHA_FEATURES, LOOKBACK)
    if len(stock_codes) == 0:
        print("  [错误] 没有股票可以构建有效序列！请检查数据。")
        sys.exit(1)
    print()

    print("  加载训练好的 Transformer 模型...")
    model = load_model(model_config_path, model_path, device)
    print()

    print("  运行模型前向传播 (对所有候选股票打分)...")
    predictions = run_inference(model, X, stock_codes, device)

    prices = extract_prices(df, latest_date)
    print_top_k(predictions, args.k, prices)

    # ── 第4步：生成订单 ──
    manager = LivePortfolioManager(
        k=args.k,
        rebalance_stride=args.rebalance_stride,
        rebalance_fraction=args.rebalance_fraction,
        min_position_ratio=args.min_position_ratio,
    )

    if args.reset:
        print_header("重置持仓状态")
        print(f"  初始资金: ¥{args.initial_nav:,.0f}")
        if portfolio_state_path.exists():
            portfolio_state_path.unlink()
        manager.state.cash = args.initial_nav
        manager.state.initial_nav = args.initial_nav
        print()

    print("  加载持仓状态...")
    manager.load_state(portfolio_state_path)
    print()

    orders = manager.generate_orders(predictions, latest_date, prices)

    # ── 第5步：交互执行 ──
    results = interactive_execute_orders(orders, manager.state, prices)

    # 保存状态
    manager.state.last_signal_date = latest_date
    manager.state.day_index += 1
    manager.save_state(portfolio_state_path)
    print(f"  持仓状态已保存: {portfolio_state_path}")

    # 打印汇总
    print_portfolio_summary(manager.state, prices)

    # 保存执行日志
    save_execution_log(results, orders_dir, latest_date, manager.state)

    # ── 第6步：Git 提交推送 ──
    if not args.no_push:
        git_commit_and_push(manager.state, latest_date)
    else:
        print("\n  [跳过] 未推送到 GitHub (--no-push)")

    # 下次运行提示
    print_header("运行完毕")
    next_rebalance = manager.state.day_index
    while next_rebalance % args.rebalance_stride != 0:
        next_rebalance += 1
    days_until = next_rebalance - manager.state.day_index
    print(f"  下次操作:")
    print(f"    - 下一个调仓日: 第 {next_rebalance} 天 ({days_until} 个交易日后)")
    print(f"    - 下次运行命令:")
    print(f"      python scripts/live/live_daily.py --run-dag --data-version {args.data_version} --end-date <最新日期>")
    print()


if __name__ == "__main__":
    main()
