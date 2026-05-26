# Agent 4 Data Mart 特征工程体系白盒审计与深度分析报告

报告日期：2026-05-26  
审计对象：`pipelines/mart/agent.py`、`configs/features.yaml`、`configs/labels.yaml`、`docs/experiment_manual.md`、`docs/20260526_progress_agent1_agent2_agent3_report.md`  
任务语境：创业板指 `399006.SZ` 历史动态成分股，日频，预测未来 5 日相对创业板指数的超额收益，并用于横截面排序选股。

## 1. 当前特征工程体系总览与特征树

### 1.1 系统梳理

当前 Agent 4 已实现一个极窄但可运行的 baseline 数据集市。其数据流为：

```text
raw daily + raw metric + raw moneyflow
        -> SCD2 创业板动态股票池过滤
        -> 基础面板 panel
        -> add_features()
        -> read_benchmark()
        -> add_labels()
        -> security_daily_state tradable_only 过滤
        -> features_daily / labels / dataset
```

当前实际特征来自 `pipelines/mart/agent.py`：

- 基础收益：`ret_1d = close / pre_close - 1`。
- 对数成交：`amount_log = log1p(amount)`、`vol_log = log1p(vol)`。
- 流动性与估值原始字段：`turnover_rate`、`turnover_rate_f`、`volume_ratio`、`pe_ttm`、`pb`、`ps_ttm`、`total_mv`、`circ_mv`。
- 资金流原始字段：`net_mf_amount`、`net_mf_vol`、`buy_lg_amount`、`sell_lg_amount`。
- 滚动窗口特征：对 `[5, 10, 20, 60]` 窗口构造 `ret_{w}d_mean`、`ret_{w}d_std`、`amount_{w}d_mean`。
- 标签：`future_return = close.shift(-horizon) / close - 1`，`label_rel_return = future_return - benchmark_future_return`。

阶段报告显示当前产物：

```text
features_daily_v20260526.parquet: 25259 rows
labels_v20260526.parquet: 25259 rows
dataset_v20260526.parquet: 24679 rows
date range: 20250506 至 20260518
```

### 1.2 特征树

```text
Feature Tree
├─ A. 价格动量类
│  ├─ ret_1d
│  ├─ ret_5d_mean
│  ├─ ret_10d_mean
│  ├─ ret_20d_mean
│  └─ ret_60d_mean
├─ B. 波动率类
│  ├─ ret_5d_std
│  ├─ ret_10d_std
│  ├─ ret_20d_std
│  └─ ret_60d_std
├─ C. 成交量/流动性类
│  ├─ amount_log
│  ├─ vol_log
│  ├─ turnover_rate
│  ├─ turnover_rate_f
│  ├─ volume_ratio
│  ├─ amount_5d_mean
│  ├─ amount_10d_mean
│  ├─ amount_20d_mean
│  └─ amount_60d_mean
├─ D. 资金流类
│  ├─ net_mf_amount
│  ├─ net_mf_vol
│  ├─ buy_lg_amount
│  └─ sell_lg_amount
├─ E. 估值类
│  ├─ pe_ttm
│  ├─ pb
│  └─ ps_ttm
├─ F. 市值/规模类
│  ├─ total_mv
│  └─ circ_mv
├─ G. 市场相对类
│  └─ label 端使用 benchmark_future_return；特征端尚无相对指数收益、beta、alpha
├─ H. 行业相对类
│  └─ 尚无行业哑变量、行业中性化、行业相对估值/动量/资金流
├─ I. 时间类
│  └─ 尚无 weekday、month、month-end、holiday effect
└─ J. 技术指标类
   └─ 尚无 MA、EMA、MACD、RSI、KDJ、ATR、Bollinger Band
```

### 1.3 九大维度映射与底层拆解

| 维度            | 当前覆盖           | 金融意义                         | 市场微观行为                     | Alpha 核心假设                          | 隐藏尾部风险                                 |
| --------------- | ------------------ | -------------------------------- | -------------------------------- | --------------------------------------- | -------------------------------------------- |
| 价格动量类      | 弱覆盖             | 捕捉短中期价格延续或反转         | 散户追涨杀跌、游资接力、题材惯性 | 过去收益含有未来 5 日排序信息           | 涨停后不可买、退潮时高动量坍塌               |
| 波动率类        | 基础覆盖           | 衡量不确定性、情绪强度与风险预算 | 波动聚集、日内冲击放大           | 创业板高波动既是风险也是事件 Alpha 载体 | 低波失效，高波可能只是崩盘前兆               |
| 成交量/流动性类 | 中等覆盖           | 衡量交易拥挤度、关注度与可交易性 | 高换手、资金接力、散户聚集       | 活跃度变化领先价格变化                  | 高换手衰竭、流动性踩踏、交易成本吞噬         |
| 资金流类        | 粗糙覆盖           | 衡量主动买卖力量                 | 大单/主力/游资净流入流出         | 净流入与大单买入具有短期信息优势        | 原始金额强受市值影响，不标准化会学到规模噪声 |
| 估值类          | 基础覆盖           | 衡量成长预期和价格相对基本面     | 高成长公司估值容忍度高           | 估值极端反映错误定价或风格溢价          | 创业板估值长时间失锚，PE 缺失或负值严重      |
| 市场相对类      | 标签覆盖，特征缺失 | 将个股收益剥离指数 Beta          | 创业板指数系统性涨跌强           | 相对收益更适合选股                      | 标签端正确不代表特征端能识别 Beta 暴露       |
| 行业相对类      | 缺失               | 剥离题材/行业共同冲击            | 医药、新能源、AI 等轮动极快      | 行业内排序比全市场排序更纯净            | 缺失后模型可能只是在押行业风格               |
| 时间类          | 缺失               | 捕捉日历效应与调仓结构           | 月末资金、节前避险、财报季       | 时间状态影响风险偏好                    | 时间特征若粗糙，易成为不可泛化噪声           |
| 技术指标类      | 基本缺失           | 识别趋势、超买超卖、突破         | 动量与反转共存                   | 非线性技术状态可辅助 5 日收益           | 指标高度共线、参数挖掘过拟合                 |

结论：当前特征工程完成的是“最小可运行面板”，不是“创业板适配型 Alpha 特征系统”。其最大优点是链路清晰、股票池和可交易状态已有基础；最大缺陷是特征表达贫瘠，且关键时间边界控制不足。

## 2. 创业板对比主板的微观结构与风格差异

### 2.1 高波动性

创业板股票普遍小市值、高成长、高估值，信息披露与预期修正对价格的弹性大于沪深 300 等主板大盘股。传统 Low-Vol 因子在主板常被解释为行为偏差和风险预算约束带来的稳定溢价，但在创业板容易失效，原因有三点：

1. 低波股票往往是低关注、低弹性、题材真空股票，在创业板横截面中缺乏资金推动。
2. 创业板收益分布具有明显尖峰厚尾，波动率上升不总是坏信号，也可能是题材启动、涨停打开、资金换手的前兆。
3. 20% 涨跌停制度使趋势推进更剧烈，波动率可能由价格发现产生，而非纯粹风险。

因此，`ret_5d_std`、`ret_10d_std`、`ret_20d_std` 在创业板不能简单作为负向风险因子使用。更合理的用法是与动量、成交量、资金流交互：高波动 + 放量 + 净流入可能是趋势确认；高波动 + 缩量 + 净流出则可能是流动性风险释放。

### 2.2 高换手与情绪驱动

创业板的高换手源于散户参与、短线资金偏好、主题交易和涨跌停制度共同作用。资金流因子在创业板的边际效用通常高于主板，因为中小盘股票对边际订单更敏感，且大单/超大单更容易改变盘口供需。

当前系统保留了 `turnover_rate`、`turnover_rate_f`、`volume_ratio`、`net_mf_amount`、`buy_lg_amount`、`sell_lg_amount`，方向正确，但表达方式不够工业化：

- 原始资金流金额没有除以成交额、市值或自由流通市值，模型会混淆“资金强度”和“公司规模”。
- 缺少高换手持续性，例如 `turnover_rate_5d_mean`、`turnover_rate_5d_z`、`turnover_acceleration`。
- 缺少量价共振，例如 `ret_1d * volume_ratio`、`ret_5d_mean * amount_5d_z`。

### 2.3 小市值风格与盘口脆弱性

创业板成分股虽已由指数规则筛选，但仍显著偏向成长、小中市值、流动性分层明显。`total_mv` 与 `circ_mv` 捕捉了规模暴露，但当前使用原始市值而非 `log_total_mv`、`log_circ_mv`，存在两个问题：

1. 量级巨大，容易让树模型优先按规模粗暴切分。
2. 与成交额、资金流原始金额共线，导致模型将资金流 Alpha 误读为规模因子。

市值因子在创业板具有非线性：过大市值可能弹性不足，过小市值可能流动性脆弱且冲击成本极高，中间市值叠加题材和流动性改善往往更具交易价值。因此建议使用分位数、截面 z-score、行业内规模分位，而非裸 `total_mv`。

### 2.4 动量与反转共存机理

创业板常呈现“短期反转、中期动量”的撕裂结构。行为金融上，短期反转来自散户过度反应、涨跌停附近的获利兑现和隔夜情绪回摆；中期动量来自题材扩散、机构调仓滞后和游资接力。

数学上可将 `ret_1`、`ret_5`、`ret_20`、`RSI`、`MACD` 视为不同时间尺度的状态变量：

- `ret_1d`：极短期冲击项，若涨幅过大且无资金继续流入，未来 5 日可能反转。
- `ret_5d_mean`：短期趋势和交易拥挤度的混合变量，在创业板既可能代表趋势，也可能代表过热。
- `ret_20d_mean`：更接近题材扩散和中期动量，若伴随成交额抬升，可能更稳定。
- `RSI`：可将涨跌幅分解为上行动能与下行动能，适合识别短期超买超卖。
- `MACD`：通过 EMA12、EMA26 和 DEA 捕捉趋势斜率与拐点，适合刻画中短期动量状态。

当前系统没有 `RSI`、`MACD`，只能通过滚动均值和标准差粗略表达价格状态。对于未来 5 日超额收益，缺少 `ret_1d` 与 `ret_20d_mean` 的交叉、`RSI` 超买回落、`MACD` DIF/DEA 金叉强度等关键状态，因此难以刻画创业板动量与反转共存的核心结构。

## 3. 核心特征大类有效性白盒审计

### 3.1 价格动量类

1. 金融逻辑：捕获投资者反应不足导致的趋势延续，以及过度反应后的短期反转。
2. 创业板适配性：被显著放大。题材传播快、涨停制度强、散户追涨明显。
3. 时间稳定性：牛市中 `ret_20d_mean` 更可能有效；熊市中高动量容易成为流动性陷阱；震荡市中 `ret_1d` 反转更强。
4. 横截面有效性：潜在有效，但必须区分短期冲击和中期趋势。
5. 标签一致性：与未来 5 日超额收益合理相关，但需避免使用当日收盘后才可得数据去假设当日收盘成交。
6. 泄露风险：当前特征没有统一 `shift(1)`。若信号定义为 T 日收盘后、T+1 交易，则 T 日完整行情可用于 T+1；但若 dataset 被模型或回测误解为 T 日可交易信号，`ret_1d` 与 rolling 当日值会形成交易时点泄露。手册明确要求最终统一 `shift(1)`，当前未执行。
7. LightGBM 适配性：树模型能捕捉非线性阈值，如高 `ret_5d_mean` 后反转；但窗口均值之间共线较强。
8. GRU/Transformer 适配性：当前把序列压缩为均值，丢失路径信息，不适合作为高级时序模型主输入。

### 3.2 波动率类

1. 金融逻辑：捕获风险溢价、信息冲击和情绪强度。
2. 创业板适配性：高适配，但方向非单调。
3. 时间稳定性：牛市高波动可伴随 Alpha；熊市高波动多为风险暴露；震荡市高波动后可能反转。
4. 横截面有效性：单独排序不稳定，需与成交量、涨跌停、资金流交互。
5. 标签一致性：未来 5 日收益对短期波动状态敏感，合理。
6. 泄露风险：rolling 按 `ts_code` 分组且只向后滚动，没有 centered rolling；但同样缺少统一信号滞后。
7. LightGBM 适配性：适合阈值分裂，如 `ret_20d_std` 高低分组。
8. GRU/Transformer 适配性：滚动 std 是二阶摘要，平稳性优于价格，但会抹掉波动聚集路径。

### 3.3 成交量/流动性类

1. 金融逻辑：捕获关注度、换手拥挤度、流动性改善或衰竭。
2. 创业板适配性：强放大。高换手是创业板最重要的微观状态之一。
3. 时间稳定性：牛市放量更偏正向，熊市放量可能是出逃，震荡市放量冲高后易回落。
4. 横截面有效性：`turnover_rate` 与 `volume_ratio` 有潜在单调性，但裸 `amount` 均值偏向大市值。
5. 标签一致性：未来 5 日超额收益对短期热度非常敏感，逻辑一致。
6. 泄露风险：当日成交数据能否用于 T 日信号取决于信号生成时点；当前没有在字段元数据中声明可得性。
7. LightGBM 适配性：强，能处理换手率非线性和缺失。
8. GRU/Transformer 适配性：成交量序列对题材生命周期有价值，但当前仅输出 rolling mean，不足。

### 3.4 资金流类

1. 金融逻辑：捕获主动买卖压力和大单资金行为。
2. 创业板适配性：非常强。游资和大单冲击在小中市值股票中影响更大。
3. 时间稳定性：牛市净流入可延续；熊市净流入可能只是护盘或诱多；震荡市需结合价格位置。
4. 横截面有效性：原始金额单调性弱，标准化后更强。
5. 标签一致性：对 5 日收益高度相关，尤其适合短周期选股。
6. 泄露风险：如果 moneyflow 数据为盘后统计，必须只作为收盘后信号输入，且交易从 T+1 开始。
7. LightGBM 适配性：原始大额异常值会诱导树模型过度切分，需 winsorize 与比例化。
8. GRU/Transformer 适配性：资金流序列若标准化为净流入强度、连续流入天数，适合注意力捕捉资金接力。

### 3.5 估值类

1. 金融逻辑：捕获成长预期、估值修复和价值错杀。
2. 创业板适配性：被钝化。创业板高成长、高研发、盈利波动大，传统 PE/PB 有时解释力弱。
3. 时间稳定性：熊市估值约束增强；牛市估值容忍度提高；震荡市估值修复可能有效。
4. 横截面有效性：PE 缺失、负值、极端值会破坏单调性。
5. 标签一致性：对 5 日收益偏弱，更像风格控制变量。
6. 泄露风险：若 metric 为盘后字段，需滞后；当前没有统一滞后。
7. LightGBM 适配性：能处理非线性，但对极端 PE/PB 需裁剪。
8. GRU/Transformer 适配性：日频估值变化慢，作为静态风格嵌入可用，不应高权重输入。

### 3.6 市值/规模类

1. 金融逻辑：捕获小市值溢价、流动性风险补偿和弹性。
2. 创业板适配性：强，但非线性。
3. 时间稳定性：小盘牛市强，熊市流动性踩踏严重，震荡市受题材影响。
4. 横截面有效性：有排序价值，但必须行业中性和流动性约束。
5. 标签一致性：对 5 日收益可能有效，但更适合作风格控制。
6. 泄露风险：市值是收盘后价格与股本派生字段，交易时点必须滞后。
7. LightGBM 适配性：裸值量级过大，建议 log + 截面标准化。
8. GRU/Transformer 适配性：慢变量，不适合长序列高频更新；可作为静态特征拼接。

### 3.7 市场相对类

1. 金融逻辑：剥离创业板指数系统性涨跌，聚焦选股 Alpha。
2. 创业板适配性：必需。创业板 beta 共振强。
3. 时间稳定性：各阶段都重要。
4. 横截面有效性：相对指数收益、rolling beta、residual momentum 能改善排序纯度。
5. 标签一致性：目标就是未来 5 日超额收益，因此特征端缺失相对收益是不完整的。
6. 泄露风险：rolling beta 只能用历史窗口；不能用未来收益残差。
7. LightGBM 适配性：相对收益特征更平稳，有利于树模型泛化。
8. GRU/Transformer 适配性：指数状态序列可作为公共上下文，但当前未实现。

### 3.8 行业相对类

1. 金融逻辑：剥离行业共同因子，捕捉行业内个股强弱。
2. 创业板适配性：极强。题材轮动高度行业化。
3. 时间稳定性：牛市行业扩散强，熊市行业拥挤退潮明显，震荡市轮动频繁。
4. 横截面有效性：行业内 rank 往往比全市场 rank 更稳定。
5. 标签一致性：未来 5 日超额收益经常来自行业轮动和行业内扩散。
6. 泄露风险：行业静态字段低风险；但行业当日均值计算只能用当日截面已知特征。
7. LightGBM 适配性：行业中性特征能降低模型押注行业。
8. GRU/Transformer 适配性：可用行业 embedding 或行业内相对序列。

### 3.9 时间类

1. 金融逻辑：捕捉日历效应、财报季、节前风险偏好和月末再平衡。
2. 创业板适配性：中等。情绪市场对节假日前后风险偏好敏感。
3. 时间稳定性：弱到中等，需验证。
4. 横截面有效性：单独横截面排序弱，更多作为状态变量。
5. 标签一致性：对 5 日 horizon 有一定帮助。
6. 泄露风险：低。
7. LightGBM 适配性：适合离散分裂。
8. GRU/Transformer 适配性：适合作位置/日历 embedding。

### 3.10 技术指标类

1. 金融逻辑：用非线性方式表达趋势、超买超卖、突破和波动通道。
2. 创业板适配性：强，但必须防过拟合。
3. 时间稳定性：RSI 在震荡市更有效，MACD 在趋势市更有效，ATR 在高波动期更重要。
4. 横截面有效性：单一指标不稳定，组合状态更有效。
5. 标签一致性：与 5 日收益高度匹配，尤其 RSI、MACD、ATR。
6. 泄露风险：rolling 与 EMA 必须按个股历史计算；当前未实现。
7. LightGBM 适配性：强，但冗余大。
8. GRU/Transformer 适配性：若已有原始序列，过多技术指标反而重复；可少量保留状态摘要。

## 4. 特征冗余、多重共线性与因子拥挤度分析

### 4.1 经典冗余剖析

当前系统尚未实现 `MA5`、`MA10`、`EMA`、`MACD`，但实验手册要求后续构造这些技术指标。必须提前警惕：

- `MA5`、`MA10`、`MA20` 本质是价格低通滤波，彼此高度相关。
- `EMA12`、`EMA26` 与 MA 特征共用价格趋势信息。
- `MACD DIF = EMA12 - EMA26`，DEA 又是 DIF 的 EMA，MACD 柱继续派生，三者内部强共线。
- `ret_5d_mean` 与 `close / MA5 - 1` 高度同源。

危害是：线性模型系数不稳定；树模型重要性被同源特征分摊，解释失真；深度学习模型会在有限样本下记忆技术指标噪声。

### 4.2 模型抵抗力测试

LightGBM 的 feature subsampling 可以缓解部分共线性，但不能消除共线因子的虚假稳定性。若同一信号被复制成多个窗口和指标，模型在验证集可能表现更稳，却只是押注同一历史模式。

GRU/Transformer 更危险。深度模型对冗余输入的容量更大，若训练样本只覆盖 20250506 至 20260518，模型很容易记住 2025-2026 的局部市场风格，尤其是题材行情、资金流异常、特定行业周期。

### 4.3 因子剪枝决策

建议删除或暂不进入 baseline：

```text
total_mv 原始值
circ_mv 原始值
net_mf_amount 原始值
buy_lg_amount 原始值
sell_lg_amount 原始值
amount_{w}d_mean 原始值
高度重复的 MA/EMA 全套裸指标
```

建议保留但必须改造：

```text
ret_1d
ret_5d_mean / ret_20d_mean / ret_60d_mean
ret_5d_std / ret_20d_std / ret_60d_std
turnover_rate / turnover_rate_f
volume_ratio
pe_ttm / pb / ps_ttm
```

建议通过比例化、标准化或降维合并：

```text
log_total_mv = log1p(total_mv)
log_circ_mv = log1p(circ_mv)
net_mf_amount_to_amount = net_mf_amount / amount
large_order_imbalance = (buy_lg_amount - sell_lg_amount) / amount
amount_z_20d = amount / rolling_mean(amount,20) - 1
trend_pca_1 = PCA(ret_5d_mean, ret_10d_mean, ret_20d_mean, close/MA20-1)
vol_pca_1 = PCA(ret_5d_std, ret_10d_std, ret_20d_std, ATR)
```

## 5. 创业板特有 Alpha 漏失与深度挖掘

### 5.1 隐藏 Alpha 盘点

当前系统覆盖度较低：

- 情绪 Alpha：仅通过收益、波动、量能间接覆盖；缺少涨跌停、连板、长上影、冲高回落、炸板等情绪结构。
- 资金流 Alpha：有原始 moneyflow 字段，但缺少比例化、持续性和异常值处理。
- 波动率 Alpha：有 rolling std，但缺少 ATR、振幅、真实波幅、波动突破和波动收缩后扩张。
- 小市值 Alpha：有市值字段，但未 log、未中性化、未分位化。
- 20% 涨跌停 Alpha：状态层已经识别 `is_limit_up`、`is_limit_down`，但 Agent 4 未将其注入特征。
- 高换手持续性：缺失。
- 题材轮动：缺少行业相对特征、行业动量、行业资金流强度。

### 5.2 致命盲区

当前缺失以下创业板特有因子：

```text
limit_up_touch / limit_down_touch：是否触及涨跌停边界
limit_up_close：是否收于涨停
limit_open_break：涨停打开或炸板代理
distance_to_limit_up / distance_to_limit_down：距离涨跌停边界
turnover_persistence_3d/5d：高换手持续性
moneyflow_strength：净流入占成交额
large_order_imbalance：大单买卖不平衡
ret_minus_benchmark_1d/5d/20d：相对创业板指数动量
rolling_beta_20d/60d：创业板 beta 暴露
industry_ret_rank / industry_turnover_rank：行业内相对强弱
amplitude、upper_shadow、lower_shadow、close_position：K 线结构
gap_open：隔夜跳空
drawdown_20d、new_high_20d：趋势位置
```

盘口订单流不平衡、逐笔成交方向、买卖队列深度、委托撤单率等更高频微观结构 Alpha 当前数据源未提供，但报告中应明确为未来生产化方向。

## 6. 特征系统致命问题与冷酷批判

### 6.1 看似有效但当前是噪声的特征

`net_mf_amount`、`buy_lg_amount`、`sell_lg_amount` 以原始金额进入模型，极可能主要表达市值与成交额规模，而不是资金流强度。对于创业板选股，原始金额越大并不等于资金行为越强；大市值权重股天然金额大，小市值题材股金额小但边际冲击强。

`total_mv`、`circ_mv` 原始值同样危险。它们会让 LightGBM 轻易切出规模桶，产生看似稳定的风格收益，但一旦市场从小盘风格切换到大盘成长，回测净值会失真。

### 6.2 增量更新中极易过拟合的特征

当前 `configs/features.yaml` 声明了 `cache_dir`，但 Agent 4 未真正使用 rolling feature cache。每次在输入窗口上重算 rolling，且起始窗口可能截断历史，导致边界特征随 start_date 改变。如果生产中按增量窗口构建，`ret_60d_mean` 和 `ret_60d_std` 在窗口开头会因 `min_periods=30` 和历史不足而不稳定。

### 6.3 回测净值虚高的元凶

最危险的是时间边界语义不清。实验手册明确要求所有用于 T 日预测的特征最终统一 `shift(1)`。当前 Agent 4 的 `ret_1d`、rolling return、成交额、资金流、估值、市值均使用当日字段直接输出。如果后续模型或回测将 `trade_date = T` 的特征解释为 T 日开盘前可用，回测会使用 T 日收盘后信息预测 T 至 T+5 的收益，形成严重泄露。

其次，`validate_no_future_leakage()` 只是用正则扫描 `.shift(-数字)`，且允许包含 `future_return` 的上下文。它不能检测：

- 特征是否统一滞后。
- 当日盘后字段是否被误用于盘前交易。
- benchmark 是否与 horizon 一致。
- 截面标准化是否使用未来全局分布。
- rolling 是否因截断历史造成边界偏差。

该检查当前返回 PASS 不能证明无未来函数。

第三，`read_benchmark()` 中 `benchmark_future_return` 固定使用 `shift(-5)`，而 `add_labels()` 的个股收益使用配置 `horizon`。如果未来把 `default_horizon` 改为 1、10 或 20，标签会变成“个股 h 日收益减创业板 5 日收益”，这是静默错误。

## 7. 增量流水线高优先级改进提案

### Priority A：必须立刻重构

1. 统一特征可得性与滞后机制  
金融逻辑：保证 T 日特征只用于 T+1 之后交易。  
工程成本：低到中，按 `ts_code` 对所有特征列执行配置化 lag。  
预期收益：极高，直接消除回测骗局风险。  
LightGBM 帮助：提升验证可信度。  
Transformer 帮助：明确序列窗口右边界。  
课程展示吸睛度：高，可展示严谨防泄露设计。

2. 修正 benchmark horizon  
金融逻辑：相对收益必须同 horizon。  
工程成本：低，将 `read_benchmark(..., horizon)` 参数化。  
预期收益：高，避免未来实验静默错标。  
LightGBM/Transformer 帮助：标签一致性提升。  
课程展示吸睛度：中，但非常专业。

3. 资金流比例化与市值 log 化  
金融逻辑：把资金行为从规模噪声中剥离。  
工程成本：低。  
预期收益：高。  
LightGBM 帮助：减少粗暴规模切分。  
Transformer 帮助：输入更平稳。  
课程展示吸睛度：高，贴合创业板资金驱动。

4. 注入状态层涨跌停特征  
金融逻辑：20% 涨跌停是创业板最关键制度变量。  
工程成本：中，读取 state 层字段并合并。  
预期收益：高。  
LightGBM 帮助：识别不可交易和情绪极端状态。  
Transformer 帮助：序列中可学习涨停后的扩散或衰竭。  
课程展示吸睛度：极高。

### Priority B：强烈建议拓展

1. 行业相对特征：行业内收益 rank、行业内换手 rank、行业内估值 z-score。  
2. 市场相对特征：`ret_1d - benchmark_ret_1d`、`ret_5d_mean - benchmark_ret_5d_mean`、rolling beta。  
3. K 线结构特征：amplitude、close_position、upper_shadow、lower_shadow、gap_open。  
4. 高换手持续性：`turnover_5d_mean`、`turnover_5d_std`、`turnover_acceleration`。  
5. 缺失值指示器与 winsorize：尤其 PE、PB、资金流。

### Priority C：探索性研究增强

1. 题材轮动代理：行业动量扩散、行业资金流强度、行业拥挤度。  
2. 新闻文本情绪：利用 `news/` 作为高阶特征，但须严格按发布时间截断。  
3. Cross-sectional Transformer：同日股票间注意力建模行业和资金扩散。  
4. 风险调整标签：`future_excess_return / future_volatility`，用于稳健排序。

## 8. 面向最终实验的工业级特征推荐方案

### 8.1 Baseline 集合：适用于 LightGBM

目标：截面单调性、高鲁棒性、抗过拟合、低泄露风险。

```text
ID:
  trade_date, ts_code

Momentum:
  lag1_ret_1d
  lag1_ret_5d_mean
  lag1_ret_20d_mean
  lag1_ret_60d_mean
  lag1_excess_ret_1d
  lag1_excess_ret_5d

Volatility:
  lag1_ret_5d_std
  lag1_ret_20d_std
  lag1_ret_60d_std
  lag1_amplitude
  lag1_atr_14

Liquidity:
  lag1_turnover_rate
  lag1_turnover_rate_f
  lag1_volume_ratio
  lag1_amount_log
  lag1_amount_z_20d
  lag1_turnover_5d_mean

Moneyflow:
  lag1_net_mf_amount_to_amount
  lag1_net_mf_vol_to_vol
  lag1_large_order_imbalance
  lag1_moneyflow_5d_sum_to_amount

Valuation/Size:
  lag1_log_total_mv
  lag1_log_circ_mv
  lag1_pe_ttm_winsor
  lag1_pb_winsor
  lag1_ps_ttm_winsor

Market/Industry Relative:
  lag1_beta_20d
  lag1_residual_ret_20d
  lag1_industry_ret_rank_5d
  lag1_industry_turnover_rank
  lag1_industry_valuation_z

Limit/Tradability:
  lag1_is_limit_up
  lag1_is_limit_down
  lag1_distance_to_limit_up
  lag1_distance_to_limit_down

Time:
  weekday
  month
  is_month_end_window
```

### 8.2 Advanced 集合：适用于 GRU / Transformer

目标：时序平稳性、长程依赖表达、微观量价路径。

序列长度建议：20 或 60 个交易日。

```text
Per-day sequence features:
  ret_1d
  excess_ret_1d
  intraday_range
  close_position
  gap_open
  turnover_rate
  volume_ratio
  amount_log_z
  net_mf_amount_to_amount
  large_order_imbalance
  is_limit_up
  is_limit_down
  distance_to_limit_up
  distance_to_limit_down
  benchmark_ret_1d
  industry_ret_1d
  industry_moneyflow_strength

Static/context features:
  log_total_mv
  log_circ_mv
  industry_id
  market_board
  listed_days_bucket
```

### 8.3 最终裁决组合

保留：

```text
ret_1d
ret_5d_mean / ret_20d_mean / ret_60d_mean
ret_5d_std / ret_20d_std / ret_60d_std
amount_log
vol_log
turnover_rate
turnover_rate_f
volume_ratio
pe_ttm / pb / ps_ttm
```

删除或替换：

```text
total_mv -> log_total_mv
circ_mv -> log_circ_mv
net_mf_amount -> net_mf_amount_to_amount
net_mf_vol -> net_mf_vol_to_vol
buy_lg_amount / sell_lg_amount -> large_order_imbalance
amount_{w}d_mean -> amount_z_{w}d or amount_log_{w}d_mean
```

新增：

```text
feature_lag_1d 统一机制
benchmark horizon 参数化
market relative return
rolling beta / residual momentum
industry relative rank
limit-up/limit-down state features
K-line morphology
turnover persistence
moneyflow persistence
technical summary: RSI14, MACD_hist, ATR14, Bollinger_position
```

## 9. 因子研究实验设计

### 9.1 横截面 IC 与 RankIC

对每个特征 `f`，在每个交易日 `t` 的创业板可交易股票池内计算：

```text
IC_t(f) = corr(f_{i,t}, y_{i,t})
RankIC_t(f) = spearman_rank_corr(f_{i,t}, y_{i,t})
```

其中 `y` 使用未来 5 日超额收益 `label_rel_return`。输出：

```text
IC mean
IC std
ICIR = mean / std
IC positive ratio
RankIC mean
RankIC t-stat
monthly/yearly IC breakdown
```

必须按时间滚动评估，不允许随机打散。

### 9.2 分层绩效

每日按特征值排序分 5 或 10 组：

```text
Q1: lowest feature group
...
Q5/Q10: highest feature group
Long-Short = top quantile - bottom quantile
```

输出：

```text
各分位平均未来 5 日超额收益
分位收益单调性
Long-Short 年化收益
Long-Short Sharpe
Long-Short Max Drawdown
换手率
交易成本敏感性
```

对于资金流、动量、波动率，需额外做方向分组。例如高波动不预设正负，而是分位观察。

### 9.3 稳健性检验

时间分段：

```text
牛市或上行段：指数 60 日收益显著为正
熊市或下行段：指数 60 日收益显著为负
震荡市：指数收益绝对值低但波动高
```

每段分别计算 IC、RankIC、分层收益和回撤。若某因子只在单一市场状态有效，必须标注为状态条件因子。

行业中性化：

```text
f_neutral = f - industry_mean(f)
或对 f 回归 industry dummies 后取 residual
```

比较中性化前后 IC。如果 IC 大幅下降，说明因子主要押注行业轮动；如果中性化后仍有效，说明具备个股 Alpha。

### 9.4 模型黑盒解密

LightGBM 训练后输出：

```text
gain importance
split importance
permutation importance
by-year importance stability
```

SHAP 归因实验：

```text
全样本 SHAP summary
牛/熊/震荡分段 SHAP
Top-K 入选股票 SHAP waterfall
资金流、换手、涨跌停特征的 SHAP interaction
```

若 `total_mv`、`amount`、`net_mf_amount` 等原始规模特征长期占据最高重要性，应判定模型可能在学习规模和流动性风格，而非真实资金流 Alpha。

## 10. 深度总结：中台系统价值判定

当前 Agent 4 的工程价值高于研究价值。它已经把 raw lake、SCD2 动态股票池、security state 和数据集市串成闭环，说明项目正在从脚本实验走向可审计流水线，这是很重要的生产化基础。

但是，从创业板量化选股研究角度看，当前特征体系还远未达到“面向 399006.SZ 微观生态适配”的标准。它缺少创业板最关键的 20% 涨跌停结构、资金流强度、高换手持续性、题材行业轮动、市场相对 beta、行业相对排序和 K 线冲击结构。现有特征多为通用日频字段，无法充分表达创业板高波动、高换手、游资主导、情绪驱动和动量反转共存的核心机制。

更严厉地说，当前系统最危险的问题不是特征少，而是“看起来通过了防泄露检查”。`future_leakage_check: PASS` 只能证明代码中没有未授权的 `shift(-n)` 字面模式，不能证明特征在交易时间边界上可用。若后续回测没有严格定义 T 日收盘后出信号、T+1 成交，当前未滞后的特征会直接制造虚高净值。

最终判定：

```text
量化架构价值：中高，数据链路和动态股票池基础正确。
金融微观结构适配：偏低，创业板专属 Alpha 覆盖不足。
机器学习泛化性：中低，特征少且原始规模噪声重。
风险控制严谨性：中低，状态层强，但特征层时间边界和标签一致性仍有硬伤。
生产价值：可作为 baseline mart，不可直接作为最终研究特征体系。
研究价值：需要完成 Priority A 和 Priority B 后，才具备严肃因子研究价值。
```

本报告建议把 Agent 4 的下一阶段目标从“跑通 dataset”升级为“可证明无泄露、可解释、创业板特化、可做 IC 闭环验证的研究级特征中台”。只有这样，LightGBM 的结果才不会是规模噪声和时间泄露的幻觉，GRU/Transformer 的展示也才不会沦为复杂模型记忆短样本市场风格。
