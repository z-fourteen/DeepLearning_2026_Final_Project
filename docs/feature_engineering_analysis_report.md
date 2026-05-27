# Agent 4 Data Mart 特征工程体系复审报告

报告日期：2026-05-26  
审计对象：`pipelines/mart/agent.py`、`pipelines/mart/validation.py`、`configs/features.yaml`、`configs/labels.yaml`、`data/mart/*_v20260526.parquet`、`outputs/factor_validation/v20260526/*`  
任务语境：创业板指 `399006.SZ` 历史动态成分股，日频，预测未来 5 日相对创业板指数的超额收益，并用于横截面排序选股。  
复审结论摘要：Agent 4 已从“最小可运行 baseline”升级为“具备创业板微观结构意识的研究数据集市”。第一次审计指出的两个最致命问题，即特征未统一滞后、benchmark horizon 固定为 5 日，已经在工程上完成修正。但新版本也引入了更复杂的共线性、验证口径和样本覆盖风险，需要继续治理。

## 1. 当前特征工程体系总览与特征树

### 1.1 系统梳理

当前 Agent 4 的数据流已经变为：

```text
raw daily + metric + moneyflow + basic
        -> SCD2 创业板动态股票池过滤
        -> Agent 3 security_daily_state 注入
        -> 只保留 tradable 样本
        -> 市场指数当日收益合并
        -> 价格/波动/流动性/资金流/估值/市场相对/行业相对/技术指标/涨跌停/时间特征
        -> add_lagged_features(): 全部模型特征统一 lag1_
        -> horizon 参数化 benchmark future return
        -> labels 与 dataset 落盘
        -> 因子 IC/RankIC/分层/相关性/推荐闭环
```

当前落盘验证产物说明：本次 `v20260526` mart 文件是短窗口 smoke-test / 修正验证产物，窗口来自 `20250501-20260525`，并不代表原始数据或正式训练集只有一年。原始 raw/state 层覆盖 `20160104-20260525` 全历史，正式开跑时应使用全历史窗口重建 mart 并重跑 factor validation。

```text
features_daily_v20260526.parquet: 25259 rows, 83 columns, 81 个 lag1_ 特征
dataset_v20260526.parquet: 24679 rows, 86 columns
current mart window: 20250501-20260525 smoke-test
full-history expected window: 20160104-20260525
raw_same_day feature columns: []
feature_columns in audit: 81
feature_availability: lag1_close_to_next_session
label_horizon: 5
future_leakage_check: PASS
```

这说明当前模型输入已经不再暴露裸当日特征，而是统一以 `lag1_` 前缀输出，显著提升了时间边界可信度。251 个交易日、约 2.47 万行只反映当前短窗口验证产物；正式实验若用全历史重建，样本规模应接近 `2521 个交易日 × 每日约 100 只成分股`，约 25 万行量级。

### 1.2 特征树

```text
Feature Tree
├─ A. 价格动量类
│  ├─ lag1_ret_1d
│  ├─ lag1_ret_5d
│  ├─ lag1_ret_20d
│  ├─ lag1_ret_{5,10,20,60}d_mean
│  └─ lag1_excess_ret_{5,10,20,60}d_mean
├─ B. 波动率类
│  ├─ lag1_ret_{5,10,20,60}d_std
│  ├─ lag1_amplitude
│  └─ lag1_bollinger_z_20d
├─ C. 成交量/流动性类
│  ├─ lag1_amount_log
│  ├─ lag1_vol_log
│  ├─ lag1_volume_ratio
│  ├─ lag1_turnover_rate / lag1_turnover_rate_f
│  ├─ lag1_amount_{5,10,20,60}d_mean
│  └─ lag1_turnover_{5,10,20,60}d_mean/std
├─ D. 资金流类
│  ├─ lag1_net_mf_amount_to_amount
│  ├─ lag1_large_order_imbalance
│  ├─ lag1_main_mf_strength
│  └─ lag1_net_mf_strength_{5,10,20,60}d_mean
├─ E. 估值类
│  ├─ lag1_pe_ttm_winsor
│  ├─ lag1_pb_winsor
│  └─ lag1_ps_ttm_winsor
├─ F. 市值/规模类
│  ├─ lag1_log_total_mv
│  └─ lag1_log_circ_mv
├─ G. 市场相对类
│  ├─ lag1_benchmark_ret_1d
│  ├─ lag1_excess_ret_1d
│  ├─ lag1_beta_20d / lag1_beta_60d
│  └─ lag1_residual_ret_20d / lag1_residual_ret_60d
├─ H. 行业相对类
│  ├─ lag1_industry_ret_1d_mean
│  ├─ lag1_industry_neutral_ret_1d / lag1_industry_neutral_ret_20d
│  ├─ lag1_industry_turnover_rank
│  ├─ lag1_industry_amount_rank
│  ├─ lag1_industry_pb_rank
│  └─ lag1_industry_mv_rank
├─ I. 时间类
│  ├─ lag1_weekday
│  ├─ lag1_month
│  └─ lag1_is_month_end
├─ J. 技术指标类
│  ├─ lag1_rsi_14d
│  ├─ lag1_macd_diff / dea / hist
│  ├─ lag1_ma_ratio_5_20 / ma_ratio_20_60
│  ├─ lag1_price_to_ma20
│  └─ lag1_bollinger_z_20d
└─ K. 创业板涨跌停与状态类
   ├─ lag1_is_limit_up / lag1_is_limit_down
   ├─ lag1_has_price_limit
   ├─ lag1_limit_ratio
   ├─ lag1_dist_to_limit_up
   ├─ lag1_dist_to_limit_down
   └─ lag1_listed_trading_days
```

### 1.3 九大维度映射与底层拆解

| 维度 | 当前覆盖 | 金融意义 | 微观行为 | Alpha 假设 | 尾部风险 |
| --- | --- | --- | --- | --- | --- |
| 价格动量 | 强 | 趋势延续与短期反转 | 游资接力、散户追涨、题材扩散 | 过去收益路径包含未来 5 日排序信息 | 高度拥挤后退潮，涨停不可买 |
| 波动率 | 中强 | 情绪强度、风险预算、冲击扩散 | 波动聚集、冲高回落、换手释放 | 高波动与量价/资金流交互后可转 Alpha | 单独高波动可能只是崩盘前兆 |
| 成交量/流动性 | 强 | 热度、拥挤、可交易性 | 高换手、量价共振、流动性踩踏 | 活跃度变化领先价格扩散 | 交易成本和冲击成本未进入特征验证 |
| 资金流 | 强 | 主动买卖压力 | 大单行为、资金接力 | 净流入强度和大单不平衡具有短期信息 | moneyflow 盘后口径需严格 T+1 使用 |
| 估值 | 中 | 成长预期与风格暴露 | 高估值容忍、风格切换 | winsor 后可做风格控制 | 创业板 PE/PB 在短周期中经常钝化 |
| 市场相对 | 强 | 剥离指数 Beta | 创业板系统性共振 | residual momentum 更贴近选股 Alpha | beta 估计窗口短，当前短窗口验证下稳定性需复核 |
| 行业相对 | 中强 | 剥离题材共同冲击 | 行业轮动、板块扩散 | 行业内 rank 更有截面单调性 | 行业分类静态且可能粗糙 |
| 时间 | 弱中 | 日历效应、调仓结构 | 月末/周内资金行为 | 作为状态变量辅助 | `lag1_month`、`lag1_weekday` 的滞后语义不自然 |
| 技术指标 | 中强 | 趋势、超买超卖、突破 | 动量与反转共存 | RSI/MACD/均线状态可辅助 5 日收益 | 与收益均值、均线、Bollinger 高共线 |

## 2. 创业板对比主板的微观结构与风格差异

### 2.1 高波动性

创业板的高波动不是单纯风险惩罚项，而是信息冲击、题材扩散和流动性再定价的载体。新系统加入 `lag1_amplitude`、`lag1_ret_*d_std`、`lag1_bollinger_z_20d`，比旧版只靠 rolling std 更贴近创业板生态。

审计判断：波动率特征已具备研究价值，但仍缺 ATR、最大回撤、上行/下行波动拆分。当前 `lag1_ret_60d_std` 被推荐为 baseline，说明样本期内长期波动或风格活跃度具有较强分层能力；但这类因子在熊市可能反向，需要分状态验证后再进入最终模型。

### 2.2 高换手与情绪驱动

新系统显著强化了高换手表达：`turnover_rate`、`turnover_rate_f`、`turnover_{w}d_mean/std`、`volume_ratio`、`amount_log`、`amount_{w}d_mean` 均已落盘。因子验证中，成交额和换手相关特征在 long-short 中表现最强：

```text
lag1_amount_log long_short_mean_return: 0.01548
lag1_turnover_rate_f long_short_mean_return: 0.01362
lag1_turnover_rate long_short_mean_return: 0.01320
```

这与创业板“资金热度驱动短周期收益”的经验一致。但这里也埋着危险：成交额/换手/市值强相关，可能代表规模和流动性风格，而非纯 Alpha。

### 2.3 小市值风格与盘口脆弱性

旧版使用裸 `total_mv/circ_mv`，新版改为 `log_total_mv/log_circ_mv`，并加入行业内市值 rank，方向正确。验证显示 `lag1_log_circ_mv`、`lag1_log_total_mv` 有显著 long-short 表现，但最大回撤分别达到约 -42% 和 -44%，说明规模因子虽有收益差异，却伴随强风格切换风险。

结论：规模特征应作为风格控制和非线性分裂变量，而不应在最终组合中被模型过度依赖。

### 2.4 动量与反转共存机理

新系统已覆盖 `ret_1d`、`ret_5d`、`ret_20d`、多窗口均值、RSI、MACD、均线比、Bollinger。它终于能表达创业板“短期反转、中期动量”的撕裂结构：

- `lag1_ret_1d`：隔日情绪冲击，适合识别短反转。
- `lag1_ret_5d_mean`：短期过热与趋势混合。
- `lag1_ret_20d_mean`、`lag1_ret_60d_mean`：中期题材扩散和风格动量。
- `lag1_rsi_14d`：超买超卖状态。
- `lag1_macd_hist`：趋势加速度。

但相关性表显示：`lag1_ret_5d` 与 `lag1_ret_5d_mean` 相关 0.992，`lag1_ret_20d` 与 `lag1_ret_20d_mean` 相关 0.986。技术指标体系必须剪枝，否则深度模型会在冗余趋势信号上过拟合。

## 3. 核心特征大类有效性白盒审计

### 3.1 价格动量类

1. 金融逻辑：捕捉反应不足、过度反应、题材延续。
2. 创业板适配性：强；20% 涨跌停强化趋势和反转共存。
3. 时间稳定性：牛市中期动量强，震荡市短反转强，熊市高动量易崩。
4. 横截面有效性：`ret_5d_mean`、`ret_10d_mean`、`ret_20d_mean` 被推荐为 baseline，但 RankIC 绝对值并不高，更多依赖分层收益。
5. 标签一致性：与未来 5 日超额收益匹配。
6. 泄露体检：已统一 `lag1_`，dataset 中无裸当日特征，核心泄露风险显著降低。
7. LightGBM 适配性：适合阈值和交互，但同源窗口过多。
8. GRU/Transformer 适配性：比旧版更好，但如果序列输入仍使用 lag1 摘要而不是原始逐日路径，会损失时序表达。

### 3.2 波动率类

1. 金融逻辑：风险溢价、事件冲击、情绪强度。
2. 创业板适配性：强，但方向非单调。
3. 时间稳定性：需分牛熊震荡验证；当前 regime 分析是事后归因，不宜当实时状态结论。
4. 横截面有效性：`lag1_ret_60d_std` 与 `lag1_ret_20d_std` 被 baseline 推荐。
5. 标签一致性：合理。
6. 泄露体检：rolling 使用历史窗口并 lag1，基本合格。
7. LightGBM 适配性：强。
8. GRU/Transformer 适配性：建议加入原始日收益序列与波动状态，而不只用 rolling std。

### 3.3 成交量/流动性类

1. 金融逻辑：关注度、资金热度、拥挤度和可交易性。
2. 创业板适配性：极强，是当前最有效的一类。
3. 时间稳定性：牛市强正向，熊市可能变成流动性出逃。
4. 横截面有效性：验证结果强，但与市值、资金流共线严重。
5. 标签一致性：对 5 日 horizon 合理。
6. 泄露体检：已 lag1；盘后统计口径通过 T+1 使用缓解。
7. LightGBM 适配性：强，但 `amount_log` 与 `amount_*d_mean` 高相关。
8. GRU/Transformer 适配性：适合建模题材生命周期。

### 3.4 资金流类

1. 金融逻辑：主动买卖压力与大单行为。
2. 创业板适配性：极强。
3. 时间稳定性：强依赖市场情绪状态。
4. 横截面有效性：`net_mf_strength_*d_mean`、`net_mf_amount_to_amount` 表现进入 baseline。
5. 标签一致性：适合未来 5 日短周期。
6. 泄露体检：已 lag1，较旧版质变。
7. LightGBM 适配性：比例化后更稳定。
8. GRU/Transformer 适配性：资金流连续性适合序列模型。

### 3.5 估值类

1. 金融逻辑：成长预期和风格暴露。
2. 创业板适配性：中等，被高成长叙事钝化。
3. 时间稳定性：熊市估值约束增强。
4. 横截面有效性：`industry_pb_rank`、`pb_winsor` 有验证信号，但最大回撤较大。
5. 标签一致性：对 5 日收益偏弱，更适合风格控制。
6. 泄露体检：winsor 按当日截面后再 lag1，时间边界可接受。
7. LightGBM 适配性：winsor 后更稳。
8. GRU/Transformer 适配性：慢变量，应作为静态或低频特征。

### 3.6 市值/规模类

1. 金融逻辑：小市值弹性与流动性风险补偿。
2. 创业板适配性：强但非线性。
3. 时间稳定性：风格切换风险高。
4. 横截面有效性：`log_circ_mv` 进入 baseline。
5. 标签一致性：更多是风格暴露。
6. 泄露体检：lag1 后可接受。
7. LightGBM 适配性：log 化后可用。
8. GRU/Transformer 适配性：不宜在日序列中重复灌入高权重。

### 3.7 市场相对类

1. 金融逻辑：剥离创业板系统性 Beta。
2. 创业板适配性：必需。
3. 时间稳定性：各市场状态都重要。
4. 横截面有效性：`excess_ret_*d_mean` 已进入 baseline；`beta_*d` 被归入 advanced。
5. 标签一致性：非常一致。
6. 泄露体检：市场当日收益合并后统一 lag1，合格。
7. LightGBM 适配性：提升泛化。
8. GRU/Transformer 适配性：适合作公共上下文。

### 3.8 行业相对类

1. 金融逻辑：剥离行业轮动，捕捉行业内强弱。
2. 创业板适配性：强。
3. 时间稳定性：需跨年度验证。
4. 横截面有效性：`industry_pb_rank`、`industry_turnover_rank`、`industry_neutral_ret_20d` 被推荐。
5. 标签一致性：合理。
6. 泄露体检：当日行业截面计算后 lag1，可接受。
7. LightGBM 适配性：有助于减少押行业。
8. GRU/Transformer 适配性：建议未来加入行业 embedding。

### 3.9 时间类

1. 金融逻辑：周内、月内、节假日前后效应。
2. 创业板适配性：中等。
3. 时间稳定性：弱。
4. 横截面有效性：时间特征在同一天横截面内通常常数，对 Cross-Sectional 排序帮助弱。
5. 标签一致性：对 5 日收益可能有状态调节作用。
6. 泄露体检：低风险，但 `lag1_weekday/month/is_month_end` 的语义是“上一交易日的日历”，对预测 T 日未来收益未必优于当前日历。
7. LightGBM 适配性：可作为状态变量，但不应进入横截面 baseline 核心。
8. GRU/Transformer 适配性：更适合做位置/日历 embedding。

### 3.10 技术指标类

1. 金融逻辑：趋势、超买超卖、突破。
2. 创业板适配性：强。
3. 时间稳定性：依赖行情状态。
4. 横截面有效性：多被归入 advanced。
5. 标签一致性：与 5 日收益匹配。
6. 泄露体检：EMA/rolling 分组计算后 lag1，基本合格。
7. LightGBM 适配性：强但冗余大。
8. GRU/Transformer 适配性：若序列模型已有原始量价，技术指标应剪枝。

### 3.11 涨跌停状态与距离类

1. 金融逻辑：20% 涨跌停边界刻画情绪压力、不可交易风险和资金封板强度。
2. 创业板适配性：极强，是区别主板的重要制度因子。
3. 时间稳定性：题材行情中有效，退潮期风险极高。
4. 横截面有效性：`is_limit_up/down` 稀疏，被验证逻辑 drop；`dist_to_limit_up/down` 保留为 advanced。
5. 标签一致性：与未来 5 日收益有合理关系。
6. 泄露体检：由 Agent 3 状态字段合并，派生后 lag1，合格。
7. LightGBM 适配性：距离连续变量比稀疏布尔变量更友好。
8. GRU/Transformer 适配性：适合捕捉涨停边界附近的路径依赖。

关键批判：相关性表显示 `lag1_ret_1d` 与 `lag1_dist_to_limit_up` 相关 -0.99998，与 `lag1_dist_to_limit_down` 相关 0.99997。这不是普通共线，而是几乎同一变量的重参数化。原因是创业板普通股票涨跌停价格近似由 `pre_close * (1 ± limit_ratio)` 决定，`dist_to_limit_up = limit_up_price / close - 1` 基本是当日收益的单调函数。距离特征仍有制度解释价值，但在建模中必须与 `ret_1d` 剪枝或改造为更有增量信息的 `limit_position`、`is_near_limit_up`、`distance_bucket`、`touch/open_break`。

## 4. 特征冗余、多重共线性与因子拥挤度分析

### 4.1 已观测的强共线结构

当前 `feature_correlation_top.csv` 暴露了多组高度相关特征：

```text
lag1_ret_1d vs lag1_dist_to_limit_up: -0.99998
lag1_ret_1d vs lag1_dist_to_limit_down: 0.99997
lag1_large_order_imbalance vs lag1_main_mf_strength: 0.99405
lag1_ret_5d vs lag1_ret_5d_mean: 0.99205
lag1_ret_20d vs lag1_ret_20d_mean: 0.98629
lag1_amount_5d_mean vs lag1_amount_10d_mean: 0.98557
lag1_price_to_ma20 vs lag1_bollinger_z_20d: 0.93985
lag1_macd_diff vs lag1_macd_dea: 0.92486
```

这是新系统的主要代价：覆盖面大幅增强后，同源特征也显著堆叠。

### 4.2 模型抵抗力测试

LightGBM 可以通过特征子采样部分缓解共线性，但不能解决解释层和因子拥挤问题。相关特征会互相替代，导致 feature importance 不稳定，且容易把同一 Alpha 重复计权。

GRU/Transformer 对冗余更敏感。当前短窗口验证产物只有 251 个交易日、116 只股票、约 2.47 万行，因此不能用它直接评价深度模型的最终容量上限。正式开跑若用 `20160104-20260525` 全历史重建，样本规模会提升到约 25 万行量级；但 81 个高共线特征直接输入仍然需要剪枝、分组 dropout 或低维投影，问题核心是共线和泛化，而不是原始数据覆盖不足。

### 4.3 因子剪枝决策

建议删除或只保留一个：

```text
lag1_dist_to_limit_up / lag1_dist_to_limit_down 与 lag1_ret_1d 不可同时高权重使用
lag1_large_order_imbalance / lag1_main_mf_strength 二选一
lag1_ret_5d / lag1_ret_5d_mean 二选一
lag1_ret_20d / lag1_ret_20d_mean 二选一
amount_5d/10d/20d/60d_mean 保留 20d 或 60d，并改为 log 或 z-score
macd_diff / macd_dea / macd_hist 最多保留 hist 或 diff
price_to_ma20 / bollinger_z_20d 二选一
```

建议保留：

```text
lag1_ret_1d
lag1_ret_20d_mean 或 lag1_excess_ret_20d_mean
lag1_ret_20d_std / lag1_ret_60d_std
lag1_turnover_5d_mean / lag1_turnover_60d_mean
lag1_net_mf_strength_20d_mean
lag1_log_circ_mv
lag1_industry_neutral_ret_20d
lag1_industry_turnover_rank
lag1_pb_winsor 或 lag1_industry_pb_rank
```

建议改造：

```text
dist_to_limit_* -> near_limit flags / limit_position / distance_bucket
amount_*d_mean -> amount_log_*d_mean 或 amount_z_*d
regime analysis -> 用历史指数收益/波动定义市场状态，而非未来标签均值
```

## 5. 创业板特有 Alpha 漏失与深度挖掘

### 5.1 当前覆盖度

当前已经覆盖：

- 20% 涨跌停制度状态：`is_limit_up/down`、`has_price_limit`、`limit_ratio`。
- 涨跌停距离：`dist_to_limit_up/down`。
- 高换手持续性：`turnover_*d_mean/std`。
- 资金流强度：`net_mf_amount_to_amount`、`main_mf_strength`、`net_mf_strength_*d_mean`。
- 题材/行业轮动代理：行业收益均值、行业中性收益、行业内 rank。
- 波动/情绪冲击：amplitude、shadow、close_position、Bollinger。
- 技术动量反转：RSI、MACD、均线比。

### 5.2 仍然缺失的关键 Alpha

当前仍缺少：

```text
limit_up_touch / limit_down_touch：盘中触及涨跌停
limit_open_break：炸板或打开涨跌停
consecutive_limit_up：连板/连续极端涨幅
near_limit_up_bucket：接近涨停但未封板的分桶状态
limit_position = (close - limit_down_price) / (limit_up_price - limit_down_price)
max_drawdown_20d / new_high_20d：趋势位置
turnover_acceleration：换手加速度
industry_moneyflow_strength：行业资金流强度
transaction_cost_proxy：成交额约束下的冲击成本代理
```

如果原始 daily 只有 OHLCV，无法真实判断盘中触及涨停但未收涨停时的封单/炸板细节；但 `high >= limit_up_price` 可以构造粗粒度 touch 代理。

## 6. 特征系统致命问题与冷酷批判

### 6.1 已修复的致命漏洞

第一次审计中最严厉的两个问题已经修复：

1. 特征统一 lag1：dataset 中除 ID 和 label 外没有裸当日特征列。
2. benchmark future return 已用 `horizon` 参数化，不再固定 5 日。

这两点使当前 mart 从“不能直接信任的 baseline”提升为“可以进入因子研究和 baseline 训练”的状态。

### 6.2 新暴露的问题

第一，涨跌停距离与 `ret_1d` 近乎完全共线。当前距离因子有制度解释，但信息增量极低，直接进入模型会造成重复计权。

第二，因子验证的 regime 划分使用 `label_rel_return` 的日均值做代理。这是典型事后分层，适合解释“这些因子在最终表现好的日子/差的日子怎样”，但不能宣称为可实时识别的牛熊震荡市场状态。生产化 regime 必须用 T 日以前可得的指数收益、波动率、成交额和宽基状态定义。

第三，rolling feature cache 仍未真正启用。`features.yaml` 有 `cache_dir`，但 `agent.py` 仍在当前输入窗口内重算 rolling。若增量窗口不足 60 日，长窗口特征在边界会漂移。

第四，当前验证产物是短窗口 smoke-test。251 个交易日只说明本次 mart 构建窗口较短，不是数据中台能力缺陷。正式实验前必须以 `20160104-20260525` 全历史窗口重建 mart，并在全历史上重跑 IC、RankIC、分层和 regime 稳健性；在此之前，当前 RankIC/long-short 只能视为修正验证结果，不能作为最终因子有效性结论。

第五，特征推荐逻辑偏向“样本内分层收益”。例如 `lag1_amount_60d_mean`、`lag1_turnover_60d_mean` 强势进入 baseline，但它们可能是规模、流动性和市场阶段的混合暴露，需要行业/市值中性后的二次验证。

### 6.3 回测净值虚高风险剩余项

统一 lag1 已经消除了最大时间边界风险，但后续回测仍必须防止：

- 使用不可买入的涨停股作为 T+1 成交对象。
- 不扣交易成本和冲击成本。
- 用同一测试期反复调特征推荐阈值。
- 用 `outputs/factor_validation` 的全样本推荐结果反向决定训练期特征，再报告同一时期测试表现。

## 7. 增量流水线高优先级改进提案

### Priority A：必须立刻重构

1. 引入 rolling feature cache  
金融逻辑：保证增量更新与全量回放特征一致。  
工程成本：中。  
预期收益：极高，解决 20/60 日窗口边界漂移。  
LightGBM 帮助：稳定特征分布。  
Transformer 帮助：序列样本右边界一致。  
课程展示吸睛度：高，体现工业级增量中台。

2. 改造涨跌停距离特征  
金融逻辑：从“收益重参数化”变成“制度边界状态”。  
工程成本：低。  
预期收益：高。  
建议新增：`limit_position`、`near_limit_up_2pct`、`near_limit_down_2pct`、`limit_touch_up`、`limit_touch_down`。  
LightGBM 帮助：减少共线重复切分。  
Transformer 帮助：更好表达边界状态。

3. 修正 regime 定义  
金融逻辑：市场状态必须由历史可得信息定义。  
工程成本：低到中。  
预期收益：高。  
建议：用创业板指数过去 20/60 日收益、20 日波动率、成交额分位划分 bull/bear/sideways/high-vol。  
课程展示吸睛度：高，能避免“事后解释伪装预测”。

### Priority B：强烈建议拓展

1. 行业资金流强度：行业内资金流均值、rank、行业资金扩散。  
2. 交易成本代理：成交额分位、预期换手成本、不可成交状态。  
3. 特征中性化验证：对规模、行业、流动性做残差化后再算 IC。  
4. 训练/验证/测试分期推荐：只用训练期做 feature recommendation，再在验证/测试期评价。

### Priority C：探索性研究增强

1. 新闻文本情绪与题材热度，但必须按发布时间截断。  
2. Cross-sectional Transformer 建模同日股票关系。  
3. 多任务标签：收益、上涨概率、极端回撤风险联合学习。

## 8. 面向最终实验的工业级特征推荐方案

### 8.1 Baseline 集合：适用于 LightGBM

建议使用“稳健、低冗余、可解释”的子集：

```text
Momentum:
  lag1_ret_1d
  lag1_excess_ret_20d_mean
  lag1_ret_60d_mean

Volatility:
  lag1_ret_20d_std
  lag1_ret_60d_std
  lag1_amplitude

Liquidity:
  lag1_volume_ratio
  lag1_turnover_5d_mean
  lag1_turnover_60d_mean
  lag1_vol_log

Moneyflow:
  lag1_net_mf_amount_to_amount
  lag1_net_mf_strength_20d_mean

Valuation/Size:
  lag1_log_circ_mv
  lag1_pb_winsor
  lag1_industry_pb_rank

Industry Relative:
  lag1_industry_turnover_rank
  lag1_industry_neutral_ret_20d

Limit/State:
  lag1_listed_trading_days
  near-limit 改造后特征，而不是直接同时放入 ret_1d 与 dist_to_limit_*
```

### 8.2 Advanced 集合：适用于 GRU / Transformer

建议把 Advanced 分成“逐日序列变量”和“静态/慢变量”：

```text
Per-day sequence:
  lag1_ret_1d
  lag1_excess_ret_1d
  lag1_amplitude
  lag1_close_position
  lag1_gap_open
  lag1_turnover_rate
  lag1_volume_ratio
  lag1_net_mf_amount_to_amount
  lag1_large_order_imbalance
  lag1_rsi_14d
  lag1_macd_hist
  lag1_bollinger_z_20d
  lag1_limit_position or near-limit bucket

Static/context:
  lag1_log_circ_mv
  lag1_pb_winsor
  industry_id embedding
  lag1_beta_20d
  lag1_listed_trading_days
```

### 8.3 最终裁决组合

明确保留：

```text
lag1_ 统一特征机制
资金流比例化特征
log 市值
winsor 估值
市场相对收益与 beta/residual
行业中性和行业 rank
高换手持续性
K 线结构
RSI/MACD/Bollinger 的精选摘要
```

明确删除或降级：

```text
lag1_is_limit_up / lag1_is_limit_down：低变异，当前验证已 drop
lag1_has_price_limit / lag1_limit_ratio：对创业板池几乎常数，当前验证已 drop
lag1_is_month_end：低变异，当前验证已 drop
dist_to_limit_up/down 与 ret_1d 同时使用：必须二选一或改造
重复 amount 多窗口：保留一个长窗口和一个短窗口即可
```

明确新增：

```text
rolling cache
limit_position
limit_touch_up/down
near_limit_up/down buckets
历史可得 regime definition
行业资金流强度
交易成本/流动性冲击代理
train-only feature recommendation protocol
```

## 9. 因子研究实验设计

### 9.1 横截面 IC 与 RankIC

当前 `pipelines/mart/validation.py` 已实现：

```text
daily IC
daily RankIC
IC mean/std/t-stat
RankIC mean/std/t-stat
positive_rank_ic_ratio
avg_cross_section
```

这满足基础闭环。下一步必须加：

```text
训练期 IC、验证期 IC、测试期 IC 分离
行业中性后 IC
市值中性后 IC
流动性中性后 IC
```

### 9.2 分层绩效

当前已实现 5 分位 long-short，并输出收益、t-stat、胜率、最大回撤。该设计合理，但需补充：

```text
Top minus Bottom 的交易成本扣减
分位单调性检验
分年度/分市场状态分层收益
换手率估计
不可交易股票剔除后的真实可买性
```

### 9.3 稳健性检验

当前已实现 regime IC，但 regime 定义使用未来标签均值，不适合作为预测可得市场状态。应改为：

```text
benchmark_ret_20d > 上分位：bull
benchmark_ret_20d < 下分位：bear
abs(benchmark_ret_20d) 低且 vol 高：sideways/high-vol
benchmark_vol_20d 高分位：high_vol
```

所有 regime 变量必须在 T 日或 T-1 日可得。

### 9.4 模型黑盒解密

建议在 LightGBM 训练后加入：

```text
gain/split/permutation importance
SHAP summary
按年份/月份的重要性稳定性
对 Top-K 入选股票做 SHAP waterfall
资金流 × 换手、动量 × 涨跌停距离的 SHAP interaction
```

若 SHAP 显示 `amount_*`、`turnover_*`、`log_circ_mv` 长期垄断贡献，则模型可能是在交易流动性风格，而不是纯选股 Alpha。

## 10. 深度总结：中台系统价值判定

当前 Agent 4 已经具备严肃研究价值。它不再只是把行情、估值、资金流拼在一起，而是将创业板关键生态变量系统性注入：高换手、资金流强度、小市值、市场相对收益、行业相对强弱、技术趋势反转、20% 涨跌停状态与距离。

从量化架构看，统一 `lag1_` 输出、horizon 参数化、状态层注入、因子验证闭环，是非常关键的工业化进步。尤其是 dataset 中已经没有裸当日特征列，这一点使后续 LightGBM baseline 的可信度大幅提高。

但从顶级量化生产标准看，系统还不能宣布完成。它现在的主要问题已经从“硬泄露风险”转向“复杂特征系统治理”：高共线、当前验证产物尚未全历史重建、验证推荐可能样本内过拟合、regime 口径事后化、rolling cache 未启用、涨跌停距离信息增量不足。

最终判定：

```text
量化架构价值：高。Agent 4 已形成可审计数据集市和因子验证闭环。
金融微观结构适配：中高。创业板高换手、资金流、涨跌停、行业轮动已有覆盖。
机器学习泛化性：中。特征丰富但共线严重；当前短窗口验证不能替代正式全历史验证。
风险控制严谨性：中高。lag1 和 horizon 已修复，但 regime 和增量 rolling 仍需治理。
生产价值：可作为课程最终 baseline 和 LightGBM 训练输入；暂不建议作为无修剪的实盘特征池。
研究价值：已经具备严肃因子研究价值，但必须先全历史重建 mart，再追加 train-only 推荐、跨期稳定性和中性化检验。
```

一句冷酷结论：这次修正把 Agent 4 从“可能制造漂亮回测的危险 baseline”推进到了“值得认真研究的创业板特征中台”；下一步不要继续堆特征，而要做剪枝、中性化、增量一致性和真实交易约束。
