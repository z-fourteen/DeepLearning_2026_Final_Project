# 增量式量化数据流水线阶段性进展报告 - Agent 1 / Agent 2 / Agent 3

报告日期：2026-05-26

## 1. 报告背景

当前项目正在从 `docs/experiment_manual.md` 中描述的实验型流程，升级为可审计、可增量、可扩展的量化数据流水线。

实验手册中的核心研究目标是：

- 股票池：创业板指 `399006.SZ` 的历史动态成分股。
- 数据频率：日频。
- 任务目标：预测未来收益或相对收益，并在横截面上对股票排序。
- 核心数据：`daily`、`metric`、`moneyflow`、`index_weight`、`trade_cal`、`stock_st`、`market`、`basic`。
- 关键约束：防止未来函数、防止幸存者偏差、保留 `trade_date` 与 `ts_code` 以便审计。

截至目前，已经完成并验证了前三个数据层 Agent：

```text
Agent 1：Ingestion Agent，数据接入层
Agent 2：Canonical Pool Agent，标准股票池层
Agent 3：Market State Agent，市场状态层
```

Agent 4，即面向模型训练的数据集市层，尚未开始。

## 2. 当前项目目录和工程基础

当前已经建立标准化工程目录：

```text
configs/          配置中心
data/
  lake/
    raw/          原始数据湖
    core/         标准核心资产
    state/        市场状态资产
    audit/        数据湖审计区
  mart/           后续模型数据集市
  cache/          后续滚动特征缓存
meta/             元数据
logs/audit/       审计日志
pipelines/        Pipeline Agent 代码
scripts/          执行入口脚本
outputs/          后续模型和回测产出
worklog/          工作留痕
```

已配置 Git 忽略生成型数据产物，避免把 parquet、审计 JSON 和本地工作日志误提交：

```text
data/lake/raw/**/*.parquet
data/lake/core/**/*.parquet
meta/file_registry.parquet
meta/data_versions.parquet
logs/audit/*.json
worklog/
```

## 3. Agent 1：数据接入层完成情况

### 3.1 职责定位

Agent 1 是当前系统中唯一允许直接读取原始 CSV 的组件。

它负责：

- 扫描 `A股数据/` 下的源文件。
- 计算文件指纹。
- 识别新增或修改文件。
- 执行 schema 校验。
- 将通过校验的 CSV 写入不可变 raw lake。
- 生成 `meta/file_registry.parquet`。
- 生成数据版本和审计日志。

### 3.2 已实现文件

```text
pipelines/ingest/agent.py
scripts/run_ingest_raw.py
scripts/init_pipeline_dirs.py
scripts/validate_ingest_schema.py
configs/data.yaml
meta/schema_registry.yaml
```

### 3.3 Schema Registry 与校验

已在 `configs/data.yaml` 中定义原始数据流标准 schema，并同步登记到 `meta/schema_registry.yaml`。

目前覆盖：

```text
basic
trade_cal
daily
stock_st
index_weight
```

校验逻辑包括：

- 必需列检查。
- 数值列检查。
- 整数列检查。
- `YYYYMMDD` 日期格式检查。
- 对 pandas 读取空值日期时产生的 `20261230.0` 这类值进行兼容处理。

如果校验不通过，Agent 1 会直接抛出 `ValueError`，阻止脏数据进入 raw lake。

### 3.4 首次真实 Ingest 结果

首次真实接入命令：

```bash
conda run -n dl_env python scripts/run_ingest_raw.py --data-version v20260526
```

执行结果：

```text
data_version: v20260526
discovered_files: 5119
ingested_files: 5119
modified_files: 0
new_trade_dates: 2522
ingested_rows: 11133551
```

生成资产：

```text
data/lake/raw/
meta/file_registry.parquet
meta/data_versions.parquet
logs/audit/v20260526_audit.json
```

### 3.5 增量机制验证

首次接入完成后，第二次 dry-run 返回：

```text
changed_files: 0
new_files: 0
modified_files: 0
ingested_rows: 0
```

这说明文件注册表已经生效，系统不会重复接入历史文件。

## 4. Agent 2：标准股票池层完成情况

### 4.1 职责定位

Agent 2 负责构建创业板指 `399006.SZ` 的历史动态股票池。

这直接对应实验手册中的要求：

> 使用创业板指历史动态成分股，而不是用最新成分股反向定义历史股票池。

核心设计采用 SCD Type 2 时间版本化结构：

```text
ts_code
index_code
effective_from
effective_to
```

这使回测或训练时可以通过如下逻辑获取当日有效股票池：

```sql
trade_date BETWEEN effective_from AND effective_to
```

从而避免幸存者偏差。

### 4.2 已实现文件

```text
pipelines/pool/agent.py
pipelines/pool/__init__.py
scripts/run_build_pool.py
configs/data.yaml
meta/schema_registry.yaml
```

### 4.3 输入数据

Agent 2 只读取 Agent 1 生成并登记过的 raw lake 数据：

```text
data/lake/raw/index_weight/
data/lake/raw/basic/
meta/file_registry.parquet
```

默认指数：

```text
399006.SZ
```

### 4.4 SCD2 输出

构建命令：

```bash
conda run -n dl_env python scripts/run_build_pool.py --data-version v20260526 --overwrite
```

输出资产：

```text
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
```

输出字段：

```text
ts_code
index_code
effective_from
effective_to
index_weight
source_trade_date
name
industry
market
list_date
pool_version
is_active
created_at
```

### 4.5 校验结果

Agent 2 校验已通过：

```text
survivorship_bias_check: PASS
pool_intervals: 293
pool_active_intervals: 100
pool_closed_intervals: 193
pool_unique_stocks: 259
min_effective_from: 20160104
closed_max_effective_to: 20251128
```

已完成的检查：

- `effective_from <= effective_to`。
- 同一 `index_code + ts_code` 不允许区间重叠。
- 当前仍有效的成分股使用 `effective_to = 99991231` 表示开放区间。

### 4.6 审计状态

`logs/audit/v20260526_audit.json` 已包含 Agent 2 指标：

```text
pool_agent: PASS
survivorship_bias_check: PASS
pool_intervals: 293
pool_active_intervals: 100
pool_closed_intervals: 193
pool_unique_stocks: 259
```

### 4.7 增量能力

Agent 2 已补充增量注册表：

```text
meta/pool_registry.parquet
```

当前行为：

```text
默认增量模式：对比 file_registry 中 index_weight 指纹和 pool_registry 中已处理指纹
无新增或变更：跳过重建
显式 backfill：使用 --backfill --overwrite 重建全量 SCD2 股票池
```

验证结果：

```text
pool_registry_rows: 125
index_codes: 399006.SZ
默认增量运行：skipped=true, pool_changed_files=0
```

## 5. Agent 3：市场状态层完成情况

### 5.1 职责定位

Agent 3 负责构建全市场统一的日度股票状态矩阵。

该状态层是后续特征工程、标签构建、模型训练和回测统一使用的过滤依据，避免各模块重复编写过滤逻辑。

目标输出字段为：

```text
trade_date
ts_code
is_st
is_suspended
is_limit_up
is_limit_down
is_tradable
listed_days
volume_valid
price_valid
state_version
created_at
```

### 5.2 已实现文件

```text
pipelines/state/agent.py
pipelines/state/__init__.py
scripts/run_build_market_state.py
```

### 5.3 输入数据

Agent 3 读取 Agent 1 生成的 raw lake：

```text
data/lake/raw/daily/
data/lake/raw/basic/
data/lake/raw/stock_st/
data/lake/raw/trade_cal/
```

### 5.4 状态字段计算逻辑

当前实现包括：

- `is_st`：当日出现在 `stock_st` 中。
- `price_valid`：`open/high/low/close/pre_close` 非空且 `close > 0`。
- `volume_valid`：`vol` 非空且 `vol > 0`。
- `is_suspended`：当前实现中等价于 `volume_valid == false`。
- `is_limit_up`：基于板块、日期、上市交易日、ST 状态和 `pre_close` 推导涨停价后判断。
- `is_limit_down`：基于板块、日期、上市交易日、ST 状态和 `pre_close` 推导跌停价后判断。
- `listed_days`：优先基于交易日历计算上市交易日天数；交易日历不可用时退化为自然日。
- `is_tradable`：非 ST、非停牌、价格有效、成交量有效、上市天数满足阈值。
- `state_version`：使用数据版本号，例如 `v20260526`。

涨跌停规则已配置化，当前重点覆盖创业板：

```text
创业板 2020-08-24 前：普通股票 10%，ST/*ST 5%
创业板 2020-08-24 起：普通股票 20%，ST/*ST 20%
新股上市前 5 个交易日：不设涨跌幅限制
```

状态层新增规则解释字段：

```text
market_board
listed_trading_days
has_price_limit
limit_ratio
limit_up_price
limit_down_price
limit_rule_id
limit_rule_reason
```

### 5.5 当前状态层输出

当前已生成状态层资产：

```text
data/lake/state/security_daily_state.parquet/
```

目前已经完成全历史 backfill，覆盖 raw daily 中的全部交易日：

```text
start_date: 20160104
end_date: 20260525
state_rows: 10719660
state_trade_dates: 2521
state_version: v20260526
st_filtered: 298231
invalid_price_rows: 168
suspended_rows: 0
state_partitions: 2521
limit_rule_coverage: PASS
```

### 5.6 Agent 3 审计

Agent 3 当前同时写入独立审计文件和每日主审计文件：

```text
logs/audit/v20260526_state_audit.json
logs/audit/v20260526_audit.json
```

内容摘要：

```text
state_agent: PASS
state_rows: 10719660
state_trade_dates: 2521
st_filtered: 298231
invalid_price_rows: 168
suspended_rows: 0
```

### 5.7 当前限制

Agent 3 已完成全历史状态矩阵构建。当前仍需继续细化的是规则层和查询层。

当前状态：

```text
已完成：20160104 至 20260525 全历史状态层
已完成：创业板涨跌停规则细化和可扩展规则接口
已完成：统一状态层查询接口
待强化：训练区间覆盖校验接入 DAG
```

## 6. 当前数据资产总览

### 6.1 元数据

```text
meta/file_registry.parquet
  rows: 5119
  datasets: basic, daily, index_weight, stock_st, trade_cal

meta/data_versions.parquet
  rows: 1
  version: v20260526
```

### 6.2 Raw Lake

```text
data/lake/raw/
  basic
  trade_cal
  daily
  stock_st
  index_weight
```

### 6.3 Core Lake

```text
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
  rows: 293
  unique stocks: 259
  active stocks: 100
```

### 6.4 State Lake

```text
data/lake/state/security_daily_state.parquet/
  partitions: 2521
  rows: 10719660
  state version: v20260526
```

### 6.5 Audit

```text
logs/audit/v20260526_audit.json
logs/audit/v20260526_state_audit.json
```

## 7. 与实验手册的对应关系

当前已完成的工程能力，覆盖了实验手册中的数据基础部分：

- 已使用 `A股数据/` 作为统一源数据。
- 已把 CSV 原始数据接入 raw lake。
- 已建立文件指纹系统，避免重复处理历史文件。
- 已用 `index_weight` 构建创业板指历史动态成分股池。
- 已使用 SCD Type 2 股票池区间，防止幸存者偏差。
- 已建立全历史日度市场状态层，统一 ST、停牌、价格有效性、成交量有效性和可交易性判断。
- 已细化创业板涨跌停规则，并为其它板块预留可配置扩展接口。
- 已保留 `trade_date` 和 `ts_code`，便于后续特征、标签、训练集和回测统一关联。

尚未完成的实验手册内容：

- `metric`、`moneyflow`、`market` 等更多 raw 数据流接入。
- 特征工程。
- 标签构建。
- 模型训练数据集。
- LightGBM / 深度学习模型训练。
- Top-K 回测和投资评估。

## 8. 当前风险与待修正点

### 8.1 Agent 2 增量能力仍需强化

Agent 2 已完成第一版增量注册表能力，能够在无 `index_weight` 变更时跳过重建。

后续仍可继续强化为真正的局部区间更新：

```text
当前：检测到 index_weight 变化后仍重建确定性 SCD2 表
下一步增强：只切开、闭合、追加受影响的 SCD2 区间
```

### 8.2 状态层规则仍需继续扩展

当前已完成创业板关键规则细化，但后续仍可继续扩展其它市场和特殊交易场景。

后续可以进一步区分：

- 主板。
- ST 股票。
- 注册制前后规则差异。
- 科创板。
- 北交所。
- 退市整理期。

### 8.3 状态层查询接口已补齐

已经新增标准查询接口：

```text
pipelines/state/query.py
scripts/query_market_state.py
```

该接口支持：

```text
按日期区间读取
按 state_version 读取
可交易过滤：--tradable-only
价格有效过滤：--require-price-valid
成交量有效过滤：--require-volume-valid
指定股票代码过滤：--ts-code
SCD2 股票池过滤：--pool-path
默认执行状态层覆盖校验
```

已验证与 Agent 2 的创业板 SCD2 股票池联动：

```text
query_market_state.py
  --data-version v20260526
  --start-date 20260521
  --end-date 20260525
  --tradable-only
  --pool-path data/lake/core/chinext_pool/chinext_pool_scd2.parquet

rows: 300
trade_dates: 3
ts_codes: 100
```

## 9. 下一阶段建议

Agent 3 的查询接口和规则细化已完成，下一阶段可以进入 Agent 4 的前置数据准备。

推荐顺序：

1. 将状态层覆盖校验接入 DAG。
2. 接入 `metric`、`moneyflow`、`market` 等后续特征所需 raw 数据流。
3. 开始 Agent 4：Data Mart Agent。
4. 构建 `features_daily`、`labels` 和训练集。
5. 后续按需要将 Agent 2 从“变更后确定性重建”升级为“局部 SCD2 区间更新”。

当前系统已经具备两个关键防偏差基础：

```text
Agent 2：历史成分股区间，解决幸存者偏差
Agent 3：日度可交易状态，统一过滤逻辑并降低未来函数风险
```
