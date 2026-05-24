# 创业板股票池金融时间序列预测实验数据分析与特征工程执行手册

## 0. 手册定位

本手册面向课程实验团队，目标是使用项目内已有 `A股数据/` 构建一个完整的金融时间序列预测与选股研究流程。实验对象建议定义为：

```text
股票池：创业板指 399006.SZ 历史动态成分股
频率：日频
任务：预测未来收益或相对收益，并在股票池内排序选股
核心数据：daily + metric + moneyflow + index_weight + trade_cal + stock_st + market
推荐主线：未来 5 日相对收益预测 + 横截面排序 + Top-K 回测
```

这不是单只股票价格预测任务，而是多股票、日频、面板时间序列任务。模型每天面对的是一个股票横截面，输出每只股票未来一段时间的收益预期或排序分数，再据此形成组合。

## 1. 数据目录与实验产物

### 1.1 原始数据

| 路径 | 作用 | 典型字段 |
| --- | --- | --- |
| `A股数据/basic.csv` | 股票基础信息、行业、上市日期、板块过滤 | `ts_code`, `name`, `industry`, `market`, `list_date` |
| `A股数据/trade_cal.csv` | 交易日历、开闭市判断、窗口构造 | `cal_date`, `is_open`, `pretrade_date` |
| `A股数据/daily/` | 日频行情，每个文件是一日全市场截面 | `open`, `high`, `low`, `close`, `pct_chg`, `vol`, `amount`, `vwap` |
| `A股数据/metric/` | 估值、市值、换手率等每日指标 | `turnover_rate`, `pe_ttm`, `pb`, `total_mv`, `circ_mv` |
| `A股数据/moneyflow/` | 主动资金流向 | `buy_lg_amount`, `sell_lg_amount`, `net_mf_amount` |
| `A股数据/index_weight/` | 指数成分股与权重 | `index_code`, `con_code`, `trade_date`, `weight` |
| `A股数据/market/399006.SZ.csv` | 创业板指行情，作为 benchmark | `close`, `pct_chg`, `vol`, `amount` |
| `A股数据/stock_st/` | 每日 ST 列表 | `ts_code`, `type_name` |
| `A股数据/news/` | 每日新闻文本，高阶可选 | `datetime`, `title`, `content` |

### 1.2 建议输出目录

```text
data/interim/
  chinext_pool_daily.parquet
  panel_raw.parquet

data/processed/
  features_daily.parquet
  labels_daily.parquet
  dataset_lgbm.parquet
  dataset_sequence.npz

outputs/
  runs/{run_id}/config.yaml
  runs/{run_id}/metrics.json
  runs/{run_id}/predictions.parquet
  runs/{run_id}/backtest_report.md
  figures/
```

## 第一阶段：任务定义

### 目标

明确实验到底预测什么、如何交易、如何评价，防止后续把任务写成泛泛的“预测股价”。

### 输入

- 课程作业要求。
- 当前可用 A 股数据。
- 团队算力和时间限制。

### 输出

- 实验任务说明。
- 标签方案。
- 模型路线。
- 回测约束。

### 执行步骤

1. 将研究范围限定为 `399006.SZ` 创业板指历史成分股。
2. 将预测对象定义为股票在未来 `h` 个交易日的收益、超额收益或横截面排名。
3. 将交易规则定义为：在 `t` 日收盘后生成信号，在 `t+1` 日按开盘价或 VWAP 近似成交。
4. 将评估拆成两部分：预测评价和投资评价。
5. 明确 baseline：LightGBM 回归预测未来 5 日相对收益。

### 推荐工具

- Python。
- pandas / numpy。
- scikit-learn。
- LightGBM。
- PyTorch。

### 注意事项

- 不建议把任务写成“预测明天收盘价”，价格本身非平稳且跨股票不可比。
- 对金融实验而言，是否避免未来信息泄露比模型复杂度更重要。
- 课程报告需要展示完整流程，而不只是模型结果。

### 常见错误

- 用随机划分训练集和测试集。
- 用全市场股票训练，却声称是创业板股票池实验。
- 只报告 MSE，不报告收益、回撤和 IC。

### 最佳实践

- 用一句话固定主任务：预测创业板成分股未来 5 日相对创业板指收益，并按预测值做 Top-K 选股回测。
- 所有中间表保留 `trade_date` 和 `ts_code`，方便审计。

## 第二阶段：数据获取与股票池构建

### 目标

从现有文件中构建每日可交易股票池，保证每个交易日只使用当时应当已知的成分股。

### 输入

```text
A股数据/index_weight/*_399006.SZ.csv
A股数据/basic.csv
A股数据/trade_cal.csv
A股数据/stock_st/*.csv
```

### 输出

```text
data/interim/chinext_pool_daily.parquet
```

字段建议：

```text
trade_date, ts_code, index_code, index_weight, name, industry,
market, list_date, is_st, listed_days, is_tradable
```

### 执行步骤

1. 读取所有文件名匹配 `*_399006.SZ.csv` 的指数权重文件。
2. 将 `con_code` 统一命名为 `ts_code`，将 `weight` 命名为 `index_weight`。
3. 用权重文件中的 `trade_date` 作为该批成分股的生效信息日期。
4. 按交易日历把最近一次已知成分股向后填充到后续交易日，直到下一次权重文件更新。
5. 合并 `basic.csv`，补充股票名称、行业、市场和上市日期。
6. 合并 `stock_st/`，对每日 ST 股票打标并过滤。
7. 计算 `listed_days`，剔除上市时间过短的样本，建议阈值为 120 个交易日。
8. 保存每日股票池。

### 推荐工具

- pandas 的 `merge_asof` 或按月份/生效日手工扩展。
- pyarrow / fastparquet 保存 parquet。

### 注意事项

- 如果某个权重文件只有表头，应跳过该文件。
- 不能用 2026 年成分股反向定义 2018 年股票池。
- ST 过滤必须按日进行，不能只看最新名称。

### 常见错误

- 只按股票代码 `300xxx.SZ` 过滤，把所有创业板上市公司当成创业板指成分股。
- 忽略指数成分股调整，导致幸存者偏差。
- 把 `index_weight` 当成未来可知的高频权重，在权重发布日期前使用。

### 最佳实践

- 在报告中说明“使用历史动态股票池”，并画出每日股票池数量曲线。
- 保留被过滤股票数量统计：ST、停牌/无行情、上市不足样本。

## 第三阶段：数据清洗

### 目标

把分散在日文件中的行情、估值、资金流和指数数据整理成可建模的面板数据，并处理缺失、异常和不可交易样本。

### 输入

```text
A股数据/daily/*.csv
A股数据/metric/*.csv
A股数据/moneyflow/*.csv
A股数据/market/399006.SZ.csv
data/interim/chinext_pool_daily.parquet
```

### 输出

```text
data/interim/panel_raw.parquet
data/interim/panel_clean.parquet
```

主键：

```text
trade_date, ts_code
```

### 执行步骤

1. 读取 `daily/`，拼接为长表，按 `trade_date, ts_code` 去重。
2. 读取 `metric/`，保留估值、市值、换手率字段。
3. 读取 `moneyflow/`，保留各类买卖金额和净流入字段。
4. 按 `trade_date, ts_code` 左连接三类个股数据。
5. 与每日创业板股票池内连接，只保留当日股票池成员。
6. 与 `market/399006.SZ.csv` 按 `trade_date` 合并，加入指数收益。
7. 过滤明显错误样本：`close <= 0`、`high < low`、`volume/vol < 0`、关键行情字段缺失。
8. 对可保留缺失值做分组填充或缺失标记，估值类指标不建议跨股票均值硬填。
9. 对极端数值进行 winsorize，建议按日期横截面 1% 和 99% 分位裁剪。
10. 保存清洗前后样本量日志。

### 推荐工具

- pandas / polars。
- numpy。
- scipy.stats.mstats.winsorize 或自写分位裁剪。
- Great Expectations 或简单断言脚本。

### 注意事项

- A 股停牌、涨跌停、ST 状态会影响标签和回测，应在清洗阶段打标。
- `vol` 是成交量，用户需求中的 `volume` 可统一映射为 `vol`，建模字段可命名为 `volume`。
- `amount` 单位可能与指数数据不同，建模前应使用相对变化、对数或横截面标准化。

### 常见错误

- 用全样本均值填充时间序列缺失值。
- 在清洗时删除所有含缺失估值的股票，导致样本偏向成熟公司。
- 不记录被删除样本，后续无法解释结果。

### 最佳实践

- 清洗规则写成配置文件，例如 `min_listed_days=120`、`winsor_q=0.01`。
- 先保留缺失指示特征，如 `pe_ttm_isna`，再决定是否填充。
- 每次清洗输出数据质量表：行数、股票数、日期数、缺失率、异常率。

## 第四阶段：特征工程

### 目标

把原始量价、估值、资金流和时间信息转化为可学习、可比较、低泄露风险的特征。

### 输入

```text
data/interim/panel_clean.parquet
A股数据/trade_cal.csv
```

### 输出

```text
data/processed/features_daily.parquet
```

### 数据字段设计

#### 基础行情字段

| 字段 | 含义 | 用法建议 | 适用层级 | 泄露风险 |
| --- | --- | --- | --- | --- |
| `open` | 开盘价 | 不直接跨股票输入，构造开盘跳空、隔夜收益 | baseline 可用衍生项 | 若用 `t+1 open` 做 `t` 特征则泄露 |
| `high` | 最高价 | 构造振幅、真实波幅 | baseline 可用衍生项 | 当日收盘前交易不能使用当日最高价 |
| `low` | 最低价 | 构造振幅、下影线 | baseline 可用衍生项 | 同上 |
| `close` | 收盘价 | 构造收益率和技术指标，不建议直接输入价格 | baseline 可用衍生项 | 用未来 close 构造特征会泄露 |
| `volume` / `vol` | 成交量 | 构造量比、成交量变化、流动性 | baseline | 直接量级差异大 |
| `amount` | 成交额 | 构造成交额变化、资金活跃度 | baseline | 需要对数化或标准化 |
| `turnover_rate` | 换手率 | 衡量交易活跃度 | baseline | 当日收盘后可用；盘中预测需谨慎 |
| `amplitude` | 振幅 | `(high-low)/pre_close` | baseline | 同日高低价对盘前预测不可用 |
| `pct_chg` | 当日涨跌幅 | 动量、反转特征基础 | baseline | 标签构造必须 shift 到未来 |

#### 复权相关

| 字段/概念 | 含义 | 用法建议 |
| --- | --- | --- |
| `qfq` | 前复权价格口径 | 适合画图和训练连续价格衍生特征；当前数据未直接提供，应谨慎声明 |
| `hfq` | 后复权价格口径 | 适合长期收益计算；当前数据未直接提供 |
| `pre_close` | 除权昨收口径 | 当前数据中最重要的复权相关字段，可用来构造 `pct_chg` 和收益 |
| `pct_chg` | 基于除权昨收的收益字段 | 推荐作为标签和收益特征的基础 |

如果没有明确复权因子，不要自行声称使用了完整前复权或后复权价格。课程实验中推荐以 `pct_chg` 为收益率口径。

#### 技术指标

| 指标 | 构造方式 | 推荐用途 | 适用层级 | 泄露风险 |
| --- | --- | --- | --- | --- |
| MA | `close` 的滚动均线，如 5/10/20 日 | 趋势强弱 | baseline | rolling 必须只用过去和当日 |
| EMA | 指数移动均线 | 趋势和动量 | baseline | 不得用 centered rolling |
| MACD | EMA12、EMA26、DEA、DIF | 中短期趋势 | 中级 | 参数多，易过拟合 |
| RSI | 上涨/下跌幅滚动比 | 超买超卖 | 中级 | 窗口不足样本需删除 |
| KDJ | RSV、K、D、J | 短期反转 | 中级 | 使用当日高低价时需明确交易时点 |
| ATR | True Range 均值 | 波动和风险 | 中级 | 高频交易口径需谨慎 |
| Bollinger Band | MA ± k 倍标准差 | 趋势突破和波动 | 中级 | 标准差只能用过去窗口 |

#### 市场结构特征

| 字段 | 来源/构造 | 推荐用途 | 适用层级 | 泄露风险 |
| --- | --- | --- | --- | --- |
| 行业 | `basic.csv` 的 `industry` | 行业哑变量、行业中性化 | baseline | 行业变更较少，风险低 |
| 市值 | `metric.total_mv`, `circ_mv` | 风格控制、小盘效应 | baseline | 市值是当日收盘后数据，交易时点要滞后一日 |
| 换手率 | `turnover_rate`, `turnover_rate_f` | 流动性、热度 | baseline | 同日可得性取决于信号时点 |
| 波动率 | 收益率滚动标准差 | 风险、仓位控制 | baseline | rolling 不可看未来 |
| Beta | 个股收益对创业板指收益滚动回归 | 市场暴露 | 进阶 | 回归窗口必须只用历史 |
| Alpha | 个股收益减 Beta 暴露后的残差 | 选股信号分析 | 进阶/高级 | 用未来收益算 alpha 标签可以，但不能作为当期特征 |

#### 时间特征

| 字段 | 构造 | 推荐用途 | 适用层级 | 泄露风险 |
| --- | --- | --- | --- | --- |
| `weekday` | 交易日星期几 | 周内效应 | baseline | 低 |
| `month` | 月份 | 月度季节性 | baseline | 低 |
| `quarter` | 季度 | 财报季、风格切换 | baseline | 低 |
| `holiday_effect` | 节前/节后第 N 个交易日 | 节假日前后效应 | 进阶 | 使用交易日历即可，风险低 |

### 执行步骤

1. 将价格转为收益特征：`ret_1 = pct_chg / 100`，并构造 `ret_5_mean`、`ret_10_sum`。
2. 构造量价形态：振幅、实体比例、收盘位置、跳空幅度、成交量变化率、成交额变化率。
3. 构造滚动窗口特征：5/10/20/60 日动量、波动率、最大回撤、成交活跃度。
4. 构造技术指标：MA、EMA、MACD、RSI、KDJ、ATR、Bollinger Band。
5. 构造估值与规模特征：`log_total_mv`、`log_circ_mv`、`pe_ttm`、`pb`、`ps_ttm`。
6. 构造资金流特征：净流入占成交额比例、大单净流入比例、特大单净流入比例。
7. 构造市场相对特征：个股收益减创业板指收益、滚动 beta、滚动 alpha。
8. 构造行业相对特征：个股特征减当日同行业均值。
9. 对连续特征按交易日横截面标准化：`z = (x - median) / MAD` 或均值标准差。
10. 所有用于 `t` 日预测的特征最终统一 `shift(1)`，确保交易在下一日执行时不使用未来。

### 推荐工具

- pandas `groupby("ts_code").rolling()`。
- ta-lib 或 pandas-ta。
- statsmodels 用于 rolling beta。
- sklearn.preprocessing。

### 注意事项

- 特征工程必须按股票分组后再 rolling。
- 标准化必须在每个交易日横截面内做，不能用测试集未来分布。
- 深度学习序列样本中，窗口长度建议从 20 或 60 个交易日开始。

### 常见错误

- rolling 后没有按 `ts_code` 分组，导致不同股票之间串线。
- 把未来收益混入技术指标。
- 在全数据上 fit 标准化器，再划分训练测试。

### 最佳实践

- baseline 使用 30 到 80 个稳定特征即可，不要一开始堆几百个。
- 每个特征记录：来源、窗口、是否滞后、是否标准化。
- 对高缺失估值特征保留缺失标记。

## 第五阶段：标签构建

### 目标

构建适合金融预测的回归、分类和排序标签，并明确不同难度实验的推荐方案。

### 输入

```text
data/interim/panel_clean.parquet
A股数据/market/399006.SZ.csv
```

### 输出

```text
data/processed/labels_daily.parquet
```

字段建议：

```text
trade_date, ts_code,
y_ret_1d, y_ret_5d, y_ret_10d,
y_excess_5d,
y_up_5d, y_excess_up_5d,
y_rank_5d, y_topk_5d
```

### 回归任务

预测未来：

| 标签 | 公式示例 | 说明 |
| --- | --- | --- |
| 1 日收益率 | `close[t+1] / close[t] - 1` 或下一日 `pct_chg` | 噪声最大，适合演示但不稳定 |
| 5 日收益率 | `close[t+5] / close[t] - 1` | 推荐 baseline，噪声和样本量较平衡 |
| 10 日收益率 | `close[t+10] / close[t] - 1` | 更平滑，但换手下降，标签重叠更明显 |
| 5 日超额收益 | 个股未来 5 日收益 - 创业板指未来 5 日收益 | 推荐主标签，更贴近选股 |

### 分类任务

预测：

| 标签 | 构造 | 说明 |
| --- | --- | --- |
| 涨跌 | `future_ret > 0` | 简单但受市场整体涨跌影响大 |
| 超额收益 | `future_excess_ret > 0` | 比涨跌更适合相对选股 |
| Top-K 排名 | 每日按未来收益排序，前 K% 为 1 | 更贴近组合构建 |

### 标签问题分析

#### 标签平滑

1 日标签极其跳动，容易让模型学习噪声。5 日或 10 日累计收益能平滑部分噪声，但会导致相邻样本标签重叠。建议 baseline 用 5 日标签；若使用 10 日标签，回测调仓周期也相应拉长。

#### 噪声

金融市场短期收益受公告、情绪、资金冲击和随机交易影响，信噪比很低。模型预测值不应被解释为精确收益，而应被看作排序分数。

#### 极端值

涨跌停、重大事件和复牌可能产生极端收益。训练标签建议按日期横截面进行 1%/99% 裁剪，或使用 rank 标签降低极端值影响。

#### 类别不平衡

牛市中上涨样本多，熊市中下跌样本多。涨跌分类容易随市场状态偏移。超额收益分类和 Top-K 标签能减少市场方向带来的类别漂移。

#### horizon 影响

| horizon | 优点 | 缺点 | 建议 |
| --- | --- | --- | --- |
| 1 日 | 样本多、反馈快 | 噪声极大、交易成本敏感 | 只作辅助 |
| 5 日 | 平衡噪声与交易频率 | 标签有重叠 | baseline 首选 |
| 10 日 | 更平滑 | 样本有效独立性下降 | 进阶对比 |

### 推荐标签

#### baseline 推荐标签

```text
y_excess_5d = future_5d_stock_return - future_5d_chinext_index_return
```

模型用回归方式预测，评估 IC、RankIC 和 Top-K 收益。

#### 中级实验推荐标签

```text
y_rank_5d = 每个交易日内 y_excess_5d 的横截面分位排名
y_top20_5d = y_rank_5d >= 0.8
```

适合排序学习、分类模型和组合选股。

#### 高级实验推荐标签

```text
行业中性未来收益
风格中性未来收益
风险调整收益 = future_excess_return / future_volatility
```

适合体现量化研究深度，但实现和解释成本更高。

### 为什么金融预测难

- 收益序列信噪比低，许多可观测特征只能解释很小部分未来波动。
- 市场结构会变化，过去有效的规律可能失效。
- 股票之间相关性强，市场整体方向会掩盖个股 alpha。
- 交易成本、冲击成本和涨跌停限制会吃掉预测收益。

### 为什么短期预测噪声极大

短期价格包含大量订单流、情绪、新闻和流动性冲击。日频数据只能看到收盘后的压缩结果，无法完整解释日内发生了什么，因此 1 日收益往往接近随机扰动。

### 为什么排序任务通常优于直接价格预测

选股只需要判断相对强弱，不需要精确预测价格点位。横截面排序可以抵消一部分市场共同涨跌，也更贴近“每天选出更好的股票”这一投资目标。

### 执行步骤

1. 按 `ts_code` 排序后计算未来收益，使用 `shift(-h)`。
2. 计算创业板指未来收益。
3. 构造超额收益。
4. 对标签按日期进行极端值裁剪。
5. 构造每日横截面 rank。
6. 删除无法得到完整未来窗口的尾部样本。
7. 保存标签表并记录样本数。

### 推荐工具

- pandas groupby shift。
- numpy。
- scipy / sklearn。

### 注意事项

- 标签允许使用未来收益，因为它是监督学习目标；但标签不能参与特征。
- 回测成交日必须晚于信号日。
- 标签 horizon 应与调仓周期一致或可解释。

### 常见错误

- `shift(5)` 和 `shift(-5)` 写反。
- 用未来 5 日最高价构造标签，造成不可交易收益。
- 标签和特征在同一天没有错位。

### 最佳实践

- 同时保存原始收益标签、超额收益标签和 rank 标签，便于对比。
- 报告中用标签分布图说明噪声和极端值。

## 第六阶段：数据集组织

### 目标

把特征和标签组织成传统机器学习表格数据与深度学习序列数据，并按时间正确划分。

### 输入

```text
data/processed/features_daily.parquet
data/processed/labels_daily.parquet
```

### 输出

```text
data/processed/dataset_lgbm.parquet
data/processed/dataset_sequence.npz
data/processed/split_config.yaml
```

### 时间序列正确划分

禁止：

```text
train_test_split(..., shuffle=True)
随机抽样划分 train/test
先全样本标准化再划分
```

必须：

```text
按 trade_date 先后划分
训练集只早于验证集
验证集只早于测试集
```

推荐示例：

```text
train:      2016-01-04 到 2022-12-31
validation: 2023-01-01 到 2024-12-31
test:       2025-01-01 到 2026-05-22
```

具体日期应以实际数据覆盖和课程提交时间为准。

### walk-forward validation

把时间轴切成多个滚动评估段。例如：

```text
第 1 折：2016-2020 训练，2021 验证
第 2 折：2016-2021 训练，2022 验证
第 3 折：2016-2022 训练，2023 验证
```

适合检验模型在不同市场状态下是否稳定。

### expanding window

训练窗口不断扩张，越往后使用越多历史数据。优点是训练样本多；缺点是旧市场状态可能影响当前模型。

### rolling window

固定最近 N 年训练。例如每次只用过去 3 年训练。优点是更贴近当前市场；缺点是样本少，深度模型可能不稳定。

### 为什么金融任务中数据泄露严重

金融数据有强时间依赖，未来的市场分布、股票池变化、ST 状态、退市风险和标准化参数都可能隐含未来信息。只要测试期信息参与了训练期特征处理，回测收益就会虚高。

### 执行步骤

1. 合并特征与标签。
2. 删除标签缺失和关键特征缺失样本。
3. 根据 `trade_date` 添加 `split` 字段。
4. 表格模型：每一行是一个 `trade_date, ts_code` 样本。
5. 序列模型：对每只股票取过去 `lookback` 日特征作为输入，预测未来标签。
6. 对训练集拟合需要跨时间的 scaler，对验证和测试只 transform。
7. 保存数据集和切分配置。

### 推荐工具

- pandas。
- sklearn Pipeline。
- PyTorch Dataset / DataLoader。
- yaml 保存配置。

### 注意事项

- 横截面标准化可在每日内部做；时间标准化必须只 fit 训练集。
- 序列样本窗口不能跨股票。
- 新上市股票历史不足窗口长度时应跳过。

### 常见错误

- 随机切分导致同一天横截面同时出现在训练和测试。
- LSTM 序列窗口跨越股票边界。
- 用测试集日期调参。

### 最佳实践

- 固定一个公开的 split 配置，团队成员共用。
- 每个实验只改一个主要因素，便于归因。

## 第七阶段：模型训练

### 目标

按由浅入深的路线训练模型，先建立可信 baseline，再尝试深度学习和 Transformer。

### 输入

```text
data/processed/dataset_lgbm.parquet
data/processed/dataset_sequence.npz
data/processed/split_config.yaml
```

### 输出

```text
outputs/runs/{run_id}/model.pkl
outputs/runs/{run_id}/predictions.parquet
outputs/runs/{run_id}/metrics.json
```

### 模型路线图

#### 第一层：传统机器学习

| 模型 | 数据形式 | 输入输出 | 优点 | 缺点 | 训练难点 | 课程适配 |
| --- | --- | --- | --- | --- | --- | --- |
| Linear/Ridge/Lasso | 表格特征 | 当日特征到未来收益 | 可解释、快 | 非线性弱 | 特征尺度敏感 | 适合作为最简单 baseline |
| Random Forest | 表格特征 | 特征到收益/分类 | 抗异常 | 时间外推弱、慢 | 参数和样本量 | 可做对比 |
| XGBoost/LightGBM | 表格特征 | 特征到收益/rank | 效果稳定、调参友好 | 不直接建模序列 | 防过拟合 | 最推荐 baseline |
| Logistic Regression | 表格特征 | 特征到涨跌/Top-K | 简洁 | 表达力有限 | 类别不平衡 | 适合分类对照 |

#### 第二层：时序深度学习

| 模型 | 数据形式 | 输入输出 | 优点 | 缺点 | 训练难点 | 课程适配 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 拼接窗口特征 | `[lookback * features] -> y` | 实现简单 | 时序归纳弱 | 过拟合 | 适合过渡 |
| RNN/GRU | 序列 | `[B,T,F] -> y` | 能处理时间依赖 | 长依赖有限 | 样本组织 | 推荐进阶 |
| LSTM | 序列 | `[B,T,F] -> y` | 经典时序模型 | 参数多 | 训练慢 | 适合课程展示 |
| TCN | 序列 | 卷积序列到 y | 并行、稳定 | 参数设计较多 | 感受野设置 | 容易做出效果 |

#### 第三层：Transformer 体系

| 模型 | 数据形式 | 输入输出 | 优点 | 缺点 | 训练难点 | 课程适配 |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla Transformer Encoder | 序列 | `[B,T,F] -> y` | 表达力强 | 小数据易过拟合 | 正则化、位置编码 | 可做高阶尝试 |
| Informer/Autoformer 思路 | 长序列 | 多步预测 | 适合长序列 | 实现复杂 | 数据规模 | 不建议作为唯一主线 |
| Cross-sectional Transformer | 同日股票截面 | `[N,F] -> score` | 建模股票间关系 | 实现复杂 | padding/mask | 高级亮点 |

#### 第四层：金融专用方向

| 方向 | 数据形式 | 输出 | 优点 | 缺点 | 课程适配 |
| --- | --- | --- | --- | --- | --- |
| Learning to Rank | 横截面样本 | 排序分数 | 贴近选股 | 评估复杂 | 强烈推荐进阶 |
| 多任务学习 | 收益 + 分类 + 波动 | 多输出 | 稳定表示 | 调参复杂 | 高级 |
| 行业/风格中性模型 | 中性化特征或标签 | alpha 分数 | 更量化 | 解释门槛高 | 报告亮点 |
| 图神经网络 | 股票关系图 | 分数 | 可表达行业/相关性 | 构图困难 | 不建议短期主线 |

### 推荐方案

#### 最推荐 baseline

```text
LightGBM 回归
输入：横截面标准化后的量价、技术、估值、市值、资金流特征
输出：未来 5 日超额收益
评价：IC、RankIC、Top-K 收益、Sharpe、Max Drawdown
```

#### 最推荐进阶模型

```text
GRU 或 TCN
输入：过去 20/60 日特征序列
输出：未来 5 日超额收益或 rank
```

#### 最推荐“容易做出效果”的方案

```text
LightGBM + y_excess_5d + 横截面 rank 评价 + 每周 Top-20 等权调仓
```

原因：实现成本低、对小样本友好、与金融排序任务匹配。

### 执行步骤

1. 先训练线性模型，得到最低基准。
2. 训练 LightGBM，使用验证集 early stopping。
3. 输出每日每股预测分数。
4. 训练 GRU/TCN，使用同一 split 和同一标签。
5. 所有模型统一输出 `trade_date, ts_code, pred_score`。
6. 只用验证集调参，测试集只评估一次。

### 推荐工具

- scikit-learn。
- lightgbm。
- pytorch。
- optuna 可选。

### 注意事项

- 金融数据中验证集表现波动大，不能只看单次最优。
- 深度模型必须有 dropout、weight decay、early stopping。
- 类别标签要处理不平衡，排序标签要按日构造 batch 或 group。

### 常见错误

- 测试集反复调参。
- 训练损失下降就认为策略有效。
- 深度模型参数过大，记住股票代码和历史噪声。

### 最佳实践

- 同一套预测文件接入同一个回测器，公平比较模型。
- 报告中展示传统模型和深度模型的收益/风险差异，而非只比较 loss。

## 第八阶段：回测与评估

### 目标

从投资角度验证预测信号是否有经济意义，不只看预测误差。

### 输入

```text
outputs/runs/{run_id}/predictions.parquet
data/interim/panel_clean.parquet
A股数据/market/399006.SZ.csv
```

### 输出

```text
outputs/runs/{run_id}/backtest_report.md
outputs/runs/{run_id}/equity_curve.parquet
outputs/figures/
```

### 预测指标

| 指标 | 含义 | 用法 |
| --- | --- | --- |
| MSE/MAE | 回归误差 | 辅助指标，不作为最终结论 |
| Accuracy/AUC | 分类质量 | 适合涨跌或 Top-K 分类 |
| IC | 每日预测值与未来收益的相关系数 | 衡量线性相关 |
| RankIC | 每日预测排名与未来收益排名的相关系数 | 选股最重要指标之一 |
| ICIR | IC 均值 / IC 标准差 | 衡量稳定性 |

### 投资指标

| 指标 | 含义 | 重要性 |
| --- | --- | --- |
| 收益率 | 策略累计收益、年化收益 | 核心结果 |
| Sharpe Ratio | 单位波动收益 | 衡量风险调整收益 |
| Max Drawdown | 最大回撤 | 衡量极端风险 |
| 超额收益 | 策略相对创业板指收益 | 判断是否跑赢 benchmark |
| 换手率 | 调仓变化程度 | 影响交易成本 |
| 胜率 | 正收益周期比例 | 辅助解释 |

### 回测规则建议

```text
信号日：t 日收盘后
买入日：t+1 日
持有期：5 个交易日或每周调仓
选股：预测分数 Top-K，例如 Top 10/20/30
权重：等权；进阶可按预测分数归一化
交易成本：单边 0.1% 到 0.2%
benchmark：399006.SZ 创业板指
风险过滤：剔除 ST、无行情、涨跌停不可成交样本
```

### 执行步骤

1. 对每个交易日按 `pred_score` 排序。
2. 选择 Top-K 股票。
3. 将持仓收益对齐到未来可成交区间。
4. 扣除交易成本和换手成本。
5. 计算每日组合收益、累计净值、benchmark 净值。
6. 计算 Sharpe、Max Drawdown、年化收益、IC、RankIC。
7. 分年度、分市场状态、分行业分析表现。
8. 输出图表和回测报告。

### 推荐工具

- pandas。
- empyrical 或 quantstats。
- matplotlib / seaborn / plotly。

### 注意事项

- 回测要明确成交价格假设：下一日开盘价、下一日 VWAP 或下一日收盘价。
- 若用未来 5 日收益标签，回测调仓周期最好与 5 日一致，或使用持仓重叠组合并解释。
- 创业板涨跌停约束会影响真实可交易性，应至少在风险章节说明。

### 常见错误

- 用 `t` 日收盘价买入 `t` 日收盘后生成的信号。
- 不扣交易成本。
- 只展示最优 Top-K，不展示参数敏感性。

### 最佳实践

- 同时报告 Top 10、Top 20、Top 30 的结果。
- 报告 IC 时间序列和累计 IC。
- 与创业板指、等权股票池、随机选股做对比。

## 第九阶段：实验管理

### 目标

让团队成员可以复现实验、比较结果、追踪参数和避免文件混乱。

### 输入

- 数据版本。
- 特征配置。
- 标签配置。
- 模型配置。
- 回测配置。

### 输出

```text
configs/{experiment_name}.yaml
outputs/runs/{run_id}/
docs/experiment_log.md
```

### 执行步骤

1. 为每次实验创建唯一 `run_id`，例如 `20260524_lgbm_excess5d_v1`。
2. 用 YAML 保存数据范围、特征列表、标签、模型参数和回测参数。
3. 保存训练日志、验证指标、测试指标、预测文件和图表。
4. 用固定脚本生成结果表，避免手工复制错误。
5. 团队分工：数据、特征、模型、回测、报告分别负责，但共用统一接口。
6. 每次实验只改变少数变量，并记录变更原因。

### 推荐工具

- Git。
- YAML / JSON。
- MLflow、Weights & Biases 或简单 CSV 日志。
- Makefile / invoke / argparse 脚本。

### 注意事项

- 原始数据不要改动，只写入 `data/interim` 和 `data/processed`。
- 大文件不要随意提交到 Git。
- 测试集结果应少看，避免人为过拟合。

### 常见错误

- 预测文件没有保存，导致无法复查回测。
- 只记录最终收益，不记录参数。
- 多人各自改脚本，结果不可复现。

### 最佳实践

- 定义统一命令：

```text
python scripts/build_dataset.py --config configs/data_chinext.yaml
python scripts/train.py --config configs/lgbm_excess5d.yaml
python scripts/backtest.py --run-id 20260524_lgbm_excess5d_v1
```

- 每个 run 文件夹保存 `config.yaml`、`metrics.json`、`predictions.parquet`、`figures/`。

## 第十阶段：报告与可视化

### 目标

把实验过程、数据分析、模型结果和回测结论整理成课程报告基础材料。

### 输入

```text
outputs/runs/{run_id}/metrics.json
outputs/runs/{run_id}/backtest_report.md
outputs/figures/
data/interim/data_quality_report.csv
```

### 输出

```text
docs/final_experiment_report_draft.md
outputs/figures/*.png
```

### 执行步骤

1. 数据概览：展示时间范围、股票数量、样本数量、缺失率。
2. 股票池分析：每日成分股数量、行业分布、市值分布。
3. 标签分析：未来收益分布、极端值、不同 horizon 对比。
4. 特征分析：相关性、重要性、行业/风格暴露。
5. 模型结果：训练/验证/测试指标，baseline 与进阶模型对比。
6. 回测结果：净值曲线、超额收益、回撤、年度收益、Top-K 敏感性。
7. 风险分析：交易成本、换手率、涨跌停、过拟合、数据泄露控制。
8. 结论：哪些方法有效，哪些无效，原因是什么。

### 推荐图表

- 样本数量随时间变化。
- 每日股票池数量。
- 行业分布柱状图。
- 标签分布直方图。
- IC / RankIC 时间序列。
- 累计净值曲线。
- 回撤曲线。
- 特征重要性图。
- 不同 Top-K 和交易成本敏感性热力图。

### 推荐工具

- matplotlib。
- seaborn。
- plotly。
- pandas profiling 可选。

### 注意事项

- 图表标题要说明数据范围和标签 horizon。
- 不要只展示最好看的模型，必须保留 baseline。
- 结论要区分预测能力和交易能力。

### 常见错误

- 报告只写模型结构，没有数据泄露控制。
- 只展示累计收益，不展示回撤。
- 没有说明交易成本和 benchmark。

### 最佳实践

- 用“实验设置 - 数据处理 - 特征工程 - 模型 - 回测 - 风险 - 结论”的报告结构。
- 把失败实验也简要记录，体现研究过程。

## 风险控制清单

| 风险 | 处理方式 |
| --- | --- |
| 未来信息泄露 | 所有特征 shift，按时间划分，标准化只用训练期或当日截面 |
| 幸存者偏差 | 使用动态指数成分股，不用最新成分股回填历史 |
| ST 和不可交易 | 按日过滤 ST、缺行情、明显异常样本 |
| 极端收益 | 标签 winsorize 或 rank 化 |
| 过拟合 | 验证集 early stopping、walk-forward、多年度分析 |
| 交易成本 | 回测扣除单边成本，分析换手率 |
| 市场风格偏移 | 分年度、分牛熊、分行业评估 |
| 流动性风险 | 剔除成交额过低股票，限制单票权重 |

## 最小可执行 baseline

如果团队时间有限，优先完成以下闭环：

```text
1. 构建 399006.SZ 动态股票池
2. 拼接 daily + metric + moneyflow
3. 构造 5/10/20 日动量、波动率、换手率、市值、资金流比例
4. 构造 y_excess_5d
5. 按时间划分 train/validation/test
6. 训练 LightGBM 回归
7. 计算 IC、RankIC
8. 用 Top-20 等权组合做周频回测
9. 输出净值、Sharpe、Max Drawdown 和报告图表
```

## 最终交付物清单

| 类型 | 文件 |
| --- | --- |
| 数据 | `data/processed/features_daily.parquet`, `labels_daily.parquet` |
| 配置 | `configs/data_chinext.yaml`, `configs/lgbm_excess5d.yaml` |
| 模型输出 | `outputs/runs/{run_id}/predictions.parquet` |
| 评估 | `outputs/runs/{run_id}/metrics.json` |
| 回测 | `outputs/runs/{run_id}/backtest_report.md` |
| 图表 | `outputs/figures/*.png` |
| 报告 | `docs/final_experiment_report_draft.md` |

