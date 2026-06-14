# 大作业报告推进说明

本文档整理当前对话中已经完成的工作、报告撰写思路、实验现状与下一步执行计划。路径以 `Final` 为当前工作目录。

## 1. 报告总体思路

报告建议围绕三条主线组织：

1. **模型构建与内部消融**
   - 分别说明 GRU 与 Transformer 的数据处理、模型结构、训练目标和最终冻结模型选择依据。
   - 对 Transformer，当前重点比较六个候选模型：
     - `EnhancedTF-l60-cls-mse-f13`
     - `StdTF-l20-cls-mse-f13`
     - `StdTF-l60-attn-mse-f13`
     - `StdTF-l60-cls-huber-f13`
     - `StdTF-l60-cls-mse-f18`
     - `StdTF-l60-cls-mse-f13`
   - 评价证据包括验证集 IC/RankIC、历史 5 日调仓回测、历史每日换手回测。

2. **冻结模型实盘理论比较**
   - 最终冻结模型暂定为：
     - Transformer: `StdTF-l60-cls-mse-f13`
     - GRU: `gru-final` 或 A 同学最终确认的冻结 GRU 模型
   - 对两个冻结模型分别做 2026-06-01 至 2026-06-12 实盘理论测试：
     - GRU 五日调仓
     - GRU 每日调仓
     - Transformer 五日调仓
     - Transformer 每日调仓

3. **实盘一致性总结**
   - 加入 2026-06-01 至 2026-06-12 每日实际持仓、调仓、收益。
   - 比较实际持仓与模型推荐名单的一致性，例如 TopK 重合率、主要买入/卖出是否来自模型高分名单、收益差异来源。

## 2. 已新增的重要文件

### 2.1 Transformer 六模型历史 5 日调仓批量回测

文件：

```text
scripts/backtest/run_transformer_t1_5d_backtests.ps1
```

作用：

- 批量调用已有的 `scripts/backtest/backtest_t1_fill_sim.py`。
- 对六个 Transformer 候选模型运行统一口径的历史 5 日调仓 T+1 fill simulation。
- 使用 `Final-YXR/data/mart/labels/execution_labels_v20260526.parquet` 中的历史 execution labels。

输出目录：

```text
../Final-YXR/outputs/backtest/unified_t1_5d_transformer/<model_name>/
```

每个模型输出：

```text
t1_fill_periods.csv
t1_fill_metrics.json
```

说明：

- 该脚本用于历史测试集回测，适合放入“模型内部消融”部分。
- 它不是 2026-06-01 至 2026-06-12 的实盘理论测试。

### 2.2 Transformer 六模型历史每日换手回测

文件：

```text
scripts/backtest/backtest_daily_rebalance.py
scripts/backtest/run_transformer_daily_rebalance_backtests.ps1
```

作用：

- `backtest_daily_rebalance.py` 是新写的每日换手历史回测脚本。
- `run_transformer_daily_rebalance_backtests.ps1` 用于批量运行六个 Transformer 候选模型。
- 逻辑为：使用历史 `predictions.parquet` 的 t 日预测，在下一个交易日开盘调仓，计算隔夜收益、调仓成本、日内收益和汇总指标。

输出目录：

```text
../Final-YXR/outputs/backtest/unified_daily_rebalance_transformer/<model_name>/full_topk/
```

每个模型输出：

```text
daily_rebalance_periods.csv
daily_rebalance_positions.csv
daily_rebalance_summary.csv
daily_rebalance_metrics.json
```

说明：

- 该脚本也属于历史测试集回测，适合放入“模型内部消融”部分。
- 当前默认跑 `test` split。

### 2.3 实盘同日评测脚本

文件：

```text
scripts/backtest/backtest_live_same_day_rebalance.py
```

作用：

- 专门用于 2026-06-01 至 2026-06-12 的实盘理论测试。
- 读取 `Final-YXR/outputs/live_predictions/predictions_YYYYMMDD.csv`。
- 读取 `Final-OXX2/A股数据/daily/YYYYMMDD.csv`。
- 按实盘逻辑处理：预测日即执行日，即盘前预测、当天开盘调仓。
- 可同时输出每日调仓与 5 日调仓两类结果。

输出文件：

```text
live_same_day_periods.csv
live_same_day_positions.csv
live_same_day_summary.csv
live_same_day_metrics.json
```

说明：

- 这是后续写“冻结模型实盘理论比较”部分的核心脚本。
- 当前已经用已有的 2026-06-01 至 2026-06-04 Transformer live predictions 做过局部运行验证，脚本可以跑通。

### 2.4 README 补充

文件：

```text
scripts/backtest/README.md
```

作用：

- 补充了 Transformer 历史 5 日调仓与历史每日换手回测命令。
- 后续可继续补充实盘同日评测命令。

## 3. 当前实验结果与判断

### 3.1 历史回测日期范围

以最终 Transformer 候选模型 `StdTF-l60-cls-mse-f13` 为例：

```text
历史 5 日调仓 test 信号日：20250205 - 20260518
历史每日换手 test 信号日：20250205 - 20260518
历史每日换手执行日：20250206 - 20260519
```

因此，已经跑完的六模型历史回测可以用于“模型内部消融”，但不能直接作为 2026-06-01 至 2026-06-12 的实盘理论结果。

### 3.2 当前推荐的写作口径

当前已经完成了 Transformer 模型内部消融部分的两类历史回测：

- 5 日调仓历史回测：原计划中的主要交易评价口径，用于评估模型在中周期持有下的稳定性。
- 每日调仓历史回测：原计划中不一定必须放在模型内部消融中，但它对解释最终冻结模型选择非常有帮助。

若只看 5 日调仓历史回测，`StdTF-l60-attn-mse-f13` 是最好的模型；若综合 5 日调仓与每日调仓两种口径，`StdTF-l60-cls-mse-f13` 更适合作为最终冻结模型。报告中可以这样解释：

> 模型选择阶段采用 IC、RankIC、ICIR 与统一历史回测进行评价，并以 5 日持有回测评估模型的中周期稳定性。正式模拟交易阶段，由于比赛要求每日满仓、每日可操作，我们将模型输出作为每日横截面排序信号，并采用“每日评分 + 有限换手”的执行策略。由于历史 5 日标签与比赛每日操作频率不同，我们额外补充 5 日调仓与每日调仓两种口径的对照实验，以区分模型预测能力和交易执行效果。

不建议在正式分析中直接删除 `StdTF-l60-attn-mse-f13`。它是一个很好的强对照：它在 5 日调仓中最优，但在每日调仓中弱于 `StdTF-l60-cls-mse-f13`，恰好说明最终模型选择不是为了“挑表格最高值”，而是为了匹配最终实盘执行频率。若篇幅有限，可以在正文保留结论，在附录或简表中给出完整六模型对比。

### 3.3 Transformer 模型层指标

按各模型 best epoch 的验证集 RankIC 排序：

| 模型 | RankIC | IC | 说明 |
| --- | ---: | ---: | --- |
| `StdTF-l60-cls-mse-f18` | 0.0665 | 0.0507 | 验证 RankIC 最高 |
| `EnhancedTF-l60-cls-mse-f13` | 0.0617 | 0.0390 | 模型层指标较高 |
| `StdTF-l60-cls-huber-f13` | 0.0603 | 0.0408 | Huber 损失版本 |
| `StdTF-l60-attn-mse-f13` | 0.0591 | 0.0462 | attention pooling 版本 |
| `StdTF-l60-cls-mse-f13` | 0.0550 | 0.0451 | 最终选择的冻结 Transformer |
| `StdTF-l20-cls-mse-f13` | 0.0362 | 0.0272 | 短 lookback 表现较弱 |

初步结论：

- 单看验证 RankIC，`StdTF-l60-cls-mse-f18` 最好。
- 但模型选择不能只看 RankIC，还要看交易回测表现。

### 3.4 历史 5 日调仓回测结果

test split、Top20、统一 5 日调仓口径：

| 模型 | 累计收益 | Sharpe-like | 说明 |
| --- | ---: | ---: | --- |
| `StdTF-l60-attn-mse-f13` | 70.20% | 1.837 | 5 日调仓最强 |
| `StdTF-l60-cls-mse-f13` | 65.98% | 1.593 | 稳定第二 |
| `StdTF-l60-cls-huber-f13` | 36.96% | 0.837 | 中等 |
| `StdTF-l60-cls-mse-f18` | 32.49% | 1.103 | RankIC 高但交易收益一般 |
| `EnhancedTF-l60-cls-mse-f13` | 32.30% | 0.725 | 回测不突出 |
| `StdTF-l20-cls-mse-f13` | 28.69% | 0.748 | 较弱 |

### 3.5 历史每日换手回测结果

test split、Top20、每日换手口径：

| 模型 | 累计收益 | Sharpe-like | 说明 |
| --- | ---: | ---: | --- |
| `StdTF-l60-cls-mse-f13` | 58.93% | 1.535 | 每日换手最强 |
| `StdTF-l60-attn-mse-f13` | 39.56% | 1.095 | 低于最终模型 |
| `StdTF-l60-cls-huber-f13` | 34.57% | 1.071 | 中等 |
| `StdTF-l60-cls-mse-f18` | 33.38% | 1.059 | RankIC 高但交易收益不强 |
| `StdTF-l20-cls-mse-f13` | 32.37% | 0.926 | 一般 |
| `EnhancedTF-l60-cls-mse-f13` | 21.28% | 0.724 | 最弱 |

### 3.6 Transformer 冻结模型选择判断

当前决定选择：

```text
StdTF-l60-cls-mse-f13
```

选择理由：

- 虽然验证 RankIC 不是最高，但在每日换手交易口径下表现最好。
- 在历史 5 日调仓回测中也排名第二，仅略低于 `StdTF-l60-attn-mse-f13`。
- 相比 `StdTF-l60-cls-mse-f18`，说明增加特征数并不必然提升交易收益。
- 相比 `StdTF-l60-attn-mse-f13`，说明 attention pooling 在 5 日调仓中更强，但在更贴近实盘每日调仓的口径下不如 cls pooling 稳定。

这组证据可以支撑报告中对最终 Transformer 模型的选择。

## 4. 当前实盘评测现状

### 4.1 A 股日线数据

已确认：

```text
../Final-OXX2/A股数据/daily/
```

中已有：

```text
20260601.csv
20260602.csv
20260603.csv
20260604.csv
20260605.csv
20260608.csv
20260609.csv
20260610.csv
20260611.csv
20260612.csv
```

因此，实盘组合收益计算所需的股票日线数据已经具备。

### 4.2 Transformer live predictions

当前已有：

```text
../Final-YXR/outputs/live_predictions/predictions_20260601.csv
../Final-YXR/outputs/live_predictions/predictions_20260602.csv
../Final-YXR/outputs/live_predictions/predictions_20260603.csv
../Final-YXR/outputs/live_predictions/predictions_20260604.csv
```

当前缺少：

```text
predictions_20260605.csv
predictions_20260608.csv
predictions_20260609.csv
predictions_20260610.csv
predictions_20260611.csv
predictions_20260612.csv
```

因此，Transformer 6.1-6.12 完整实盘理论评测尚未完成。

### 4.3 live feature/sequence 文件

当前在 `Final-YXR/data/live/features` 中只看到早期文件，例如：

```text
features_20260602.parquet
features_20260603.parquet
live_sequence_20260601.npz
live_sequence_20260602.npz
live_sequence_20260602_l60.npz
live_sequence_20260603_l60.npz
```

未看到 20260605 至 20260612 的 live feature 或 live sequence 文件。若运行 live inference 报缺失，需要先补构造这些特征/序列文件。

### 4.4 指数 benchmark 数据

当前 `Final-OXX2/A股数据/market` 中的指数文件只看到更新到：

```text
20260529
```

因此：

- 组合本身的收益可以计算。
- 若报告需要相对创业板指、沪深300、上证指数的超额收益，需要把 market 指数文件也更新到 20260612。

## 5. 下一步执行计划

### Step 1：补齐 Transformer 6.1-6.12 live predictions

在 `Final-YXR` 目录运行：

```powershell
cd ..\Final-YXR

$dates = @("20260605","20260608","20260609","20260610","20260611","20260612")
foreach ($d in $dates) {
  conda run --no-capture-output -n dl_env python scripts/live/01_live_inference.py `
    --config configs/live/live_trading_StdTF.yaml `
    --trade-date $d
}
```

如果报缺少 feature/npz 文件，需要先补构造：

```text
data/live/features/live_sequence_YYYYMMDD_l60.npz
```

或：

```text
data/live/features/features_YYYYMMDD.parquet
```

目标是生成完整的：

```text
outputs/live_predictions/predictions_20260601.csv
...
outputs/live_predictions/predictions_20260612.csv
```

### Step 2：更新指数 benchmark 数据

如果报告需要超额收益，需要将以下文件更新到 20260612：

```text
../Final-OXX2/A股数据/market/000001.SH.csv
../Final-OXX2/A股数据/market/000300.SH.csv
../Final-OXX2/A股数据/market/399006.SZ.csv
```

若暂时不能更新，也可以先在报告中只展示组合绝对收益，并注明 benchmark 数据暂缺。

### Step 3：运行 Transformer 实盘同日评测

在 `Final` 目录运行：

```powershell
cd ..\Final

python .\scripts\backtest\backtest_live_same_day_rebalance.py `
  --predictions-dir ..\Final-YXR\outputs\live_predictions `
  --daily-root ..\Final-OXX2\A股数据\daily `
  --market-root ..\Final-OXX2\A股数据\market `
  --output-dir ..\Final-YXR\outputs\backtest\live_same_day_StdTF-l60-cls-mse-f13_20260601_20260612 `
  --k 20 `
  --strategy both
```

输出：

```text
../Final-YXR/outputs/backtest/live_same_day_StdTF-l60-cls-mse-f13_20260601_20260612/live_same_day_periods.csv
../Final-YXR/outputs/backtest/live_same_day_StdTF-l60-cls-mse-f13_20260601_20260612/live_same_day_positions.csv
../Final-YXR/outputs/backtest/live_same_day_StdTF-l60-cls-mse-f13_20260601_20260612/live_same_day_summary.csv
../Final-YXR/outputs/backtest/live_same_day_StdTF-l60-cls-mse-f13_20260601_20260612/live_same_day_metrics.json
```

报告中主要使用：

- `live_same_day_summary.csv`：五日调仓与每日调仓总收益、回撤、胜率、换手。
- `live_same_day_periods.csv`：逐日收益与调仓日解释。
- `live_same_day_positions.csv`：每日理论持仓，用于和实际持仓做一致性比较。

### Step 4：补齐 GRU 对照实验

还需要 A 同学侧确认或补齐：

1. GRU 最终冻结模型路径。
2. GRU 历史 `predictions.parquet`。
3. GRU 训练/验证 metrics，尤其 IC、RankIC。
4. GRU 6.1-6.12 live predictions。

然后按同样口径运行：

- 历史 5 日调仓回测。
- 历史每日换手回测。
- 6.1-6.12 实盘五日调仓理论评测。
- 6.1-6.12 实盘每日调仓理论评测。

这样报告中 GRU 与 Transformer 的比较才具有统一口径。

### Step 5：整理实际交易一致性

需要收集或整理：

```text
20260601 - 20260612 每日实际持仓
20260601 - 20260612 每日实际买入/卖出
20260601 - 20260612 每日实际组合收益
```

然后和理论模型输出比较：

- 实际持仓与 Transformer Top20 的重合率。
- 实际持仓与 GRU Top20 的重合率。
- 实际买入是否来自模型高分池。
- 实际卖出是否对应模型低分或跌出 TopK。
- 实际收益与理论每日调仓、理论五日调仓之间的差异。

## 6. 建议写入报告的结论框架

### 6.1 Transformer 消融结论

可以写成：

- 模型选择阶段首先采用 IC、RankIC、ICIR 衡量横截面预测能力，再使用统一历史回测检验预测信号能否转化为组合收益。
- 5 日调仓历史回测是模型内部消融的主口径，用于评估模型在中周期持有下的稳定性；在该口径下，`StdTF-l60-attn-mse-f13` 表现最好，`StdTF-l60-cls-mse-f13` 排名第二。
- 考虑到最终比赛操作更接近每日重新评分、每日可调仓，因此补充每日调仓历史回测作为执行口径敏感性分析；在该口径下，`StdTF-l60-cls-mse-f13` 表现最好。
- `StdTF-l60-cls-mse-f18` 验证 RankIC 最高但回测收益一般，说明模型层相关性指标和可交易收益并不完全一致。
- `StdTF-l20-cls-mse-f13` 整体较弱，说明 60 日 lookback 比 20 日 lookback 更能捕捉有效时序信息。

### 6.2 冻结模型选择结论

可以写成：

- 最终选择 `StdTF-l60-cls-mse-f13`，因为它在 5 日调仓回测中接近最优，同时在每日调仓回测中取得最佳表现，能够兼顾历史标签对应的中周期预测能力与比赛阶段的每日执行需求。
- 该选择不是单纯追求单一指标最高，也不是只看 5 日回测最高值，而是综合考虑 IC、RankIC、累计收益、回撤、换手成本与不同调仓频率下的稳定性。
- `StdTF-l60-attn-mse-f13` 可以作为强对照保留在消融表中：它证明 attention pooling 在 5 日持有下有优势，但并不一定适合每日调仓执行口径。

### 6.3 实盘理论比较结论

待完成四组结果后补写：

| 模型 | 每日调仓收益 | 五日调仓收益 | 回撤 | 换手 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| GRU-final | 待补 | 待补 | 待补 | 待补 | 待补 |
| StdTF-l60-cls-mse-f13 | 待补 | 待补 | 待补 | 待补 | 待补 |

### 6.4 实盘一致性结论

待实际持仓整理后补写：

- 实际操作是否更接近每日调仓还是五日调仓。
- 实际交易中偏离模型推荐的原因，例如流动性、涨跌停、人工风控、已有持仓约束。
- 最终收益差异来自选股信号、调仓频率还是执行约束。

## 7. 当前最优先事项

优先级从高到低：

1. 补齐 Transformer 20260605 至 20260612 live predictions。
2. 跑完整 Transformer 6.1-6.12 实盘五日/每日理论评测。
3. 更新指数 benchmark 到 20260612，若报告要写超额收益。
4. 确认 GRU-final 的模型、预测文件和 metrics。
5. 用统一脚本跑 GRU 的历史消融回测和实盘理论评测。
6. 整理实际交易持仓、调仓和收益，完成一致性分析。
