# 全历史特征验证效率瓶颈诊断报告

报告日期：2026-05-26  
分析对象：`pipelines/mart/validation.py`、`pipelines/mart/agent.py`、全历史 `data/mart/datasets/dataset_v20260526.parquet`  
当前任务：全历史 Agent4 Mart 因子验证效率诊断

## 1. 统一总述

当前瓶颈主要不在 Mart 构建本身，而在全历史因子验证 `validation`。

当前全历史 Mart 规模约为：

```text
241643 rows
2516 trade_dates
259 stocks
95 lag1_ features
```

这个量级对工业级单机特征验证来说并不算大，Pandas 在合理设计下也应可以处理。当前无法稳定跑完整套验证的根本原因，是验证器仍然带有明显的研究脚本式结构：大量按 `feature -> trade_date -> groupby/qcut/corr/apply` 的重复计算，且缺少 checkpoint、profile、分阶段落盘和并行调度。

一句话概括：

> 数据量不是根本问题，根本问题是验证器把同一份 24 万行数据反复按特征、日期、分层、窗口重复切分和重算，并且全程单进程 Pandas 黑盒运行，没有缓存、没有检查点、没有并行化。

## 2. 痛点排查列表

| 维度 | 当前低效点 | 工业化标准对标 | 优化方向 |
| --- | --- | --- | --- |
| 数据读取 | `read_raw_dataset()` 每次按 registry 读取大量 parquet 小文件再 concat | 数据湖应支持按日期/列裁剪读取，避免小文件风暴 | 使用 partitioned parquet + column pruning；Mart validation 尽量只读 dataset 文件 |
| 数据复制 | validation 中频繁 `df.copy()`、`replace()`、`dropna()`、`apply(pd.to_numeric)` | 大表应减少全量复制，类型在入口统一 | 在 dataset 生成时固化 numeric dtype；validation 入口一次性转换 |
| 内存结构 | 每个 feature 单独构造 `working = df[[...]].copy()` | 特征矩阵应一次性抽取为二维矩阵或 Arrow/NumPy block | 将 features 转成 `float32` matrix，按日期 index 分块计算 |
| IC 计算 | 已从逐日循环优化为 per-feature `groupby.corr`，但仍是 feature loop | 全特征 IC 可按日期块矩阵化 | 对每个 trade_date 取 `X` 矩阵和 `y` 向量，一次算所有 Pearson/RankIC |
| RankIC 计算 | 每个 feature 单独 rank | 横截面 rank 可一次性对所有特征 rank | `df.groupby(trade_date)[features].rank()` 一次生成 rank matrix |
| 分层回测 | `compute_quantile_table()` 对每个 feature 逐日 `qcut` | 分位数应批量化或分批并行 | 对 feature chunk 并行；或先只对 Top-K 因子分层 |
| 扩展分层 | yearly/regime quantile 会重复调用完整 quantile 逻辑 | regime/year 应复用已计算的 daily quantile return | 先产出 daily factor quantile return 明细，再聚合 year/regime |
| 中性化 IC | `build_neutralized_dataset()` 对每个 feature 做 `groupby.apply` 横截面回归 | 横截面回归应矩阵化或批量线性代数 | 每日一次性构造 exposure matrix，对所有 feature 做批量残差化 |
| 并行度 | 当前基本单进程 Pandas | 特征级任务天然可并行 | 用 `joblib` / multiprocessing 按 feature chunk 并行；或 Polars lazy/group_by |
| GIL 问题 | Python 层循环过多，CPU 利用率可能很低 | 重计算应落到 NumPy/Arrow/Cython/Polars | 减少 Python loop，使用 NumPy 矩阵计算或 Polars |
| 重复计算 | IC、RankIC、Long-Short、yearly、regime、holdout 各自重新切分 | 研究验证应有中间层缓存 | 产出 `daily_ic.parquet`、`daily_quantile_return.parquet`，后续只聚合 |
| 时间轴复用 | rolling 特征已在 Mart 算好，但 validation 不复用验证中间结果 | validation 应分 stage | `stage=ic`、`stage=quantile`、`stage=neutralized` 分开跑 |
| Checkpoint | 中断后全量重跑 | 工业任务必须可恢复 | 每个 report 单独落盘，存在则可 `--resume` 跳过 |
| Profiling | 当前靠超时判断瓶颈 | 应有耗时、内存、行数日志 | 给每个阶段打 timer，记录 rows/features/dates/memory |
| 输出覆盖 | 同一路径 `outputs/factor_validation/v20260526/` 被不同模式覆盖 | 产物应按 mode/version 分目录 | `outputs/factor_validation/v20260526/full_all_lag1/`、`baseline/` |
| 研究协议 | all_lag1 一次性跑所有报告 | 工业验证应先筛选，再重型验证 | 先跑全特征 IC/相关性，再对 Top-K 跑 Long-Short/neutralized |
| 扩展性 | 当前适合 24 万行，不适合百万/千万级 | 应面向全市场、多 horizon、多池子扩展 | 抽象 validation engine，支持 feature chunk、date partition、parallel |

## 3. 核心低效原因

### 3.1 Validation 是特征循环驱动，不是矩阵或分块驱动

当前验证逻辑大致是：

```text
for feature in features:
    copy feature + label
    groupby trade_date
    qcut / corr / rank / residual
```

95 个特征时已经明显吃力；如果未来扩展到 300、500、1000 个特征，耗时会近似线性爆炸。

更工业化的模式应该是：

```text
for trade_date block:
    X = all_features_matrix
    y = label_vector
    一次性计算所有 IC / RankIC / quantile bins
```

### 3.2 Long-Short 分层是当前最大热点

IC 已经部分优化，但 `compute_quantile_table()` 仍然对每个特征重复：

```text
copy
dropna
groupby trade_date
qcut
groupby quantile
pivot
long-short
```

这一步在全历史全特征上非常重。

更合理的流程是：

```text
1. 全特征只跑 IC / RankIC / quality / correlation
2. 选出 Top-K，例如 20 或 30 个
3. 只对 Top-K 跑 Long-Short、yearly、regime、neutralized
```

### 3.3 中性化 IC 是当前最不工业化的部分

当前中性化逻辑是：

```text
for feature:
    groupby(trade_date).apply(OLS residual)
```

这会导致大量 Python 回调，非常慢。

工业化做法应该是每日一次矩阵回归：

```text
X_exposure = [1, size, liquidity]
Y_features = all feature matrix
Beta = inv(X'X) X'Y
Residual = Y - X Beta
```

也就是一次性对所有特征做横截面残差化。

### 3.4 缺少 Checkpoint，导致超时成本极高

当前一个完整 validation 可能包含：

```text
IC
RankIC
Long-Short
quality
correlation
regime IC
yearly quantile
regime quantile
neutralized IC
holdout quantile
recommendation
```

但它们被绑在一个长命令里。任何一步慢或中断，前面结果都浪费。

工业化应拆成独立阶段：

```text
factor_ic_rankic.parquet
factor_quantile_daily.parquet
factor_neutralized_ic.parquet
feature_recommendations.parquet
```

每一步都能 resume。

## 4. 四个维度深度诊断

### 4.1 内存管理与数据架构

当前问题：

- 频繁构造临时 DataFrame。
- 多处 `copy()`、`replace()`、`dropna()`、`apply(pd.to_numeric)` 触发隐式复制。
- validation 阶段没有统一 dtype 管理。
- 输出路径没有按 validation mode 隔离，容易覆盖不同实验产物。

优化方向：

- 在 Mart 写出时统一数值列 dtype，例如 `float32`。
- validation 入口一次性提取 feature matrix 和 label vector。
- 按报告阶段落盘中间结果。
- 输出路径按模式隔离，例如：

```text
outputs/factor_validation/v20260526/
  all_lag1_ic/
  baseline_full/
  baseline_holdout/
  topk_neutralized/
  topk_regime/
```

### 4.2 计算并行度与 CPU 利用率

当前问题：

- 多数重计算仍由 Python 层循环驱动。
- 分层回测、中性化回归、特征循环天然可以并行，但当前没有并行调度。
- Pandas `groupby.apply` 和逐特征 `qcut` 使 CPU 利用率不稳定。

优化方向：

- 使用 feature chunk 并行。
- 将 RankIC、Pearson IC 变成矩阵计算。
- 将中性化回归改成每日批量线性代数。
- 后续可考虑 Polars lazy、Numba 或 joblib。

### 4.3 穿越嫌疑与回测特有逻辑

当前已修复：

- 特征输出统一 `lag1_`。
- label horizon 已参数化。
- dataset 中无裸同日特征列。

当前仍需优化：

- 验证阶段重复切分时间轴，没有复用 daily IC / daily quantile 中间结果。
- regime 分析已经改为历史可得 benchmark 指标，但扩展分层仍应独立缓存。
- train-only 推荐协议已经初步实现，但仍需与 staged validation 深度结合。

优化方向：

- 先生成 daily factor validation 明细，再做年度、regime、holdout 聚合。
- 严格区分 train recommendation 与 holdout evaluation。
- 所有验证报告记录所使用的 train/eval 日期边界。

### 4.4 工业化规范与工程健壮性

当前问题：

- 缺少 stage 级别 checkpoint。
- 缺少 `--resume`。
- 缺少 profiling。
- 命令超时后无法知道卡在哪个阶段。

优化方向：

- 增加：

```text
--stage ic
--stage quantile
--stage neutralized
--stage regime
--stage recommendation
--resume
```

- 每个阶段记录：

```text
start_time
end_time
elapsed_seconds
rows
features
trade_dates
memory_mb
output_path
```

## 5. 下一步行动优先级

### Priority 1：先做 Profiling 和 Stage 化

不要继续盲目加大等待时间。应先把 validation 拆成阶段：

```text
--stage ic
--stage quantile
--stage neutralized
--stage regime
--stage recommendation
--resume
```

这是最高优先级，因为当前最大问题是验证任务处于黑盒状态。

### Priority 2：先跑全特征轻量筛选，再跑 Top-K 重型验证

建议流程：

```text
1. all_lag1 IC / RankIC / quality / correlation
2. 选 Top-K features
3. Top-K Long-Short
4. Top-K neutralized IC
5. Top-K yearly/regime
```

不要再对 95 个特征一次性跑所有重型报告。

### Priority 3：优化 quantile 与 neutralized 计算

具体优化：

- `compute_quantile_table()` 改为输出 daily quantile return 明细并缓存。
- `yearly/regime quantile` 从明细聚合，不重新 qcut。
- `neutralized IC` 改为每日矩阵回归，而不是 feature-level `groupby.apply`。

### Priority 4：输出目录按验证模式隔离

当前 `v20260526` 会被不同 validation 命令覆盖。建议改成：

```text
outputs/factor_validation/v20260526/
  all_lag1_ic/
  baseline_full/
  baseline_holdout/
  topk_neutralized/
  topk_regime/
```

## 6. 建议结论

下一步不建议继续硬跑完整 `all_lag1 validation`。

更合理的是先改 validation engine：

```text
1. 增加 --stage
2. 增加 --resume
3. 增加 profiling 日志
4. 先实现 stage=ic 全特征快速落盘
5. 根据 IC/RankIC/相关性选 Top-K
6. 再对 Top-K 跑 Long-Short、中性化、regime
```

这会比“单命令全量跑到超时”更接近工业级量化研究平台。
