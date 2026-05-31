# OXX2 Live Trading - 更新说明

## 本次变更概览

新增了**实盘模拟交易系统** (`scripts/live/live_daily.py`)，基于训练好的 Transformer 模型，
对创业板股票进行每日评分、选股、生成交易指令，并支持交互式确认和持仓管理。

### 新增文件

| 文件 | 说明 |
|---|---|
| `scripts/live/live_daily.py` | 实盘模拟交易主脚本（选股 + 下单 + 持仓管理） |
| `outputs/runs/StdTF-l60-cls-mse-f13/model.pt` | 训练好的 Transformer 模型权重 |
| `outputs/runs/StdTF-l60-cls-mse-f13/config.yaml` | 模型配置（d_model=64, 2层Encoder, CLS池化） |
| `outputs/runs/StdTF-l60-cls-mse-f13/metrics.json` | 模型训练指标 |
| `data/mart/features_daily/features_daily_v20260526.parquet` | 预构建的特征矩阵（到最新交易日） |
| `data/lake/core/chinext_pool/chinext_pool_scd2.parquet` | 创业板成分股池（SCD2 缓慢变化维） |

### 修改文件

| 文件 | 变更内容 |
|---|---|
| `scripts/run_daily_dag.py` | 新增 `--incremental` 增量模式，跳过不变的 build_pool 步骤 |
| `.gitignore` | 解除模型、特征、成分股池等必要文件的忽略 |
| `.gitattributes` | 补充 LFS 追踪规则 |

---

## 环境要求

```
Python >= 3.10
PyTorch >= 2.0
pandas, pyarrow, numpy, pyyaml, scikit-learn
```

依赖安装：
```bash
pip install torch pandas pyarrow numpy pyyaml scikit-learn
```

---

## 快速开始

### 第1步：获取原始数据

将 A 股行情 CSV 文件放入 `A股数据/` 目录，目录结构如下：

```
A股数据/
├── daily/          # 每日行情（open, high, low, close, vol, amount）
├── daily_open/     # 开盘价数据
├── metric/         # 个股指标（换手率、流通市值等）
├── moneyflow/      # 资金流向
├── market/         # 大盘指数行情
├── stock_st/       # ST 标记
└── index_weight/   # 指数成分股权重
```

> 每个 CSV 文件以交易日命名，如 `20260529.csv`。

### 第2步：首次运行（全量构建 + 建仓）

```bash
python scripts/live/live_daily.py \
  --run-dag \
  --data-version v20260526 \
  --end-date 20260529 \
  --reset
```

参数说明：
- `--run-dag`: 运行完整的数据流水线（ingest → pool → state → validate → mart）
- `--data-version v20260526`: 数据版本号
- `--end-date 20260529`: 数据截止日期（最新交易日）
- `--reset`: 首次建仓，清空旧持仓，从 1000 万现金开始

> 首次运行预计 **3-8 分钟**（全量构建流水线）。

### 第3步：日常运行（增量更新 + 调仓）

每天只需更新数据并执行调仓：

```bash
python scripts/live/live_daily.py \
  --run-dag \
  --data-version v20260526 \
  --end-date <最新交易日>
```

增量模式下：
- `ingest_raw`: 自动检测变更的 CSV 文件（MD5 比对），只处理新增/修改的文件
- `build_pool`: **跳过**（成分股池约半年调整一次）
- `build_state`: 只构建缺失日期的分区
- `build_mart`: 全量重建（rolling/EMA 特征依赖完整历史序列）
- `validate`: 校验数据完整性

> 日常增量运行预计 **2-6 分钟**。

### 其他常用命令

```bash
# 跳过数据更新（数据已是最新的）：
python scripts/live/live_daily.py --skip-dag

# 强制全量重建流水线（成分股池调整后使用）：
python scripts/live/live_daily.py --run-dag --full-dag --end-date 20260601

# 修改初始资金和持仓数量：
python scripts/live/live_daily.py --skip-dag --initial-nav 5000000 --k 10

# 不自动推送到 Git：
python scripts/live/live_daily.py --skip-dag --no-push
```

---

## 脚本执行流程

```
第1步: 数据更新 (--run-dag)
  ├─ ingest_raw:  A股数据/ CSV → data/lake/raw/ (parquet)
  ├─ build_pool:  筛选创业板成分股 → data/lake/core/
  ├─ build_state: 计算市场状态（ST、涨跌停等）→ data/lake/state/
  ├─ validate:     校验数据完整性
  └─ build_mart:  计算技术特征 → data/mart/features_daily/

第2步: 加载特征数据
  ├─ 从 features_daily parquet 加载特征矩阵
  ├─ 过滤创业板成分股
  └─ 过滤不可交易的股票（ST、涨跌停、流动性差）

第3步: 模型推理
  ├─ 构建最近 60 个交易日的特征序列
  ├─ Transformer 模型前向传播 → 每只股票的预测得分
  └─ 按得分排序，选出 Top-K

第4步: 生成交易指令
  ├─ 与当前持仓比较，确定买入/卖出标的
  ├─ 按等权重分配资金
  └─ 计算具体买入股数（100股取整）

第5步: 交互执行
  ├─ 逐笔弹出交易指令，等待确认
  ├─ 记录实际执行价格和数量
  └─ 保存持仓状态和执行日志

第6步: Git 推送（可选）
  └─ 自动 commit + push 到远程仓库
```

---

## 持仓状态管理

持仓状态保存在 `outputs/live/portfolio_state.json`，包含：
- 现金余额
- 当前持仓（股票代码、股数、成本价）
- 历史交易记录
- 上次信号日期

> **重要**: 不要手动修改此文件。如需重置，使用 `--reset` 参数。

---

## 注意事项

1. `A股数据/` 目录不会被 Git 追踪（太大且包含私有数据），需要用户自行准备
2. 大文件（parquet、模型权重）通过 **Git LFS** 管理，clone 时需要安装 LFS：
   ```bash
   git lfs install
   ```
3. 模型使用的是 `StdTF-l60-cls-mse-f13`（Standard Transformer, 60天回看, CLS池化, MSE损失, 13特征）
4. 默认策略参数：Top-20 持仓，每 3 个交易日调仓一次，每次替换 33%
