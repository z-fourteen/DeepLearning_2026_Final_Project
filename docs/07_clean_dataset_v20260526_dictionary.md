# clean_dataset v20260526: Universe Specs And Feature Dictionary

本文档是 `clean_dataset` 主线数据资产的内部技术说明，覆盖当前 production tensor、严格可交易样本域、13/18 特征合同，以及信号到 CVXPY 执行优化器的传导机制。所有可复核口径均来自本地落盘资产：

- `configs/data/splits.yaml`
- `configs/features/advanced_sequence_clean_v1.yaml`
- `data/mart/datasets/clean_purged_wf/*_manifest.json`
- `outputs/runs/gru_l20_clean_*_purgedwf_strictmask_leaky0005/metrics.json`
- `outputs/backtest/clean_dataset_execution_stack_purgedwf_both_cash_buffer/**`

标准复核命令统一使用项目环境：

```powershell
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
conda run -n dl_env python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
```

## 1. Universe Specs

### 1.1 时序切分

当前生产折为 `chinext_purged_walk_forward_v1 / final_2025_2026`。切分采用 purged walk-forward，监督标签 horizon 为 5 个交易日，训练/验证与验证/测试之间分别设置 5 天 purge 与 20 天 embargo，避免标签窗口和调仓信息跨区间泄露。

| Split | Date Span | Tensor Samples | Daily Cross-Sections | Execution Rebalance Dates | Regime Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 20160104-20221231 | 124,527 | 1,650 | - | 模型估计区间，不用于最终模型选择 |
| Validation | 20230201-20241231 | 41,393 | 468 | 94 | 偏弱/压力 regime；T+1 执行基准在 rebalance grid 上累计约 -19.3% |
| Test | 20250201-20260525 | 27,288 | 311 | 63 | 锁定 holdout；T+1 执行基准在 rebalance grid 上累计约 +91.2% |

说明：`Daily Cross-Sections` 来自训练日志中的 date-batched evaluation 步数；`Execution Rebalance Dates` 来自执行栈 `rebalance_stride=5` 后的周频调仓日期。因此模型样本域和组合路径域不是同一种行粒度。

### 1.2 数据截面与落盘行数

模型 tensor 是股票日序列样本：

| Key | Shape / Count | Meaning |
| --- | ---: | --- |
| `X` | `[193208, 20, num_features]` | 20 日 lookback 的序列输入 |
| `y` | `[193208]` | `label_rel_return` 监督目标 |
| `trade_date` | `[193208]` | 信号日 |
| `ts_code` | `[193208]` | 股票代码 |
| `split` | `[193208]` | `train` / `validation` / `test` |
| `feature_names` | 13 或 18 | 有序模型输入合同 |

执行优化器的 `optimizer_periods.csv` 不是股票日样本，而是组合路径行。以 `clean_dataset_execution_stack_purgedwf_both_cash_buffer` 为例，单个模型分支有 15,072 行：

| Dimension | Cardinality | Values |
| --- | ---: | --- |
| Rebalance dates | 157 | validation 94 + test 63 |
| `risk_control` | 4 | `none`, `industry_proxy`, `industry_size`, `industry_size_liquidity_vol_mom` |
| `k` | 3 | 10, 20, 30 |
| `style_penalty` | 4 | 0, 0.05, 0.10, 0.20 |
| `turnover_penalty` | 2 | 0, 0.02 |
| Total rows | 15,072 | `157 * 4 * 3 * 4 * 2` |

因此，15,072 行是组合参数路径的 period-level accounting，不是“交易日乘股票数”。每一行代表一个 split、一个 rebalance date、一个风控/惩罚参数组合下的完整组合状态、成交约束、现金权重、交易成本和收益归因。

### 1.3 Purged WF, Strict Mask And Executable Universe

`purgedwf` 负责时间上的 point-in-time 纯净性；`strictmask` 负责截面上的可交易纯净性；T+1 execution stack 负责成交可达性和容量约束。三者共同定义可进入建模和回测的 **Executable Universe**。

| Layer | Mechanism | Current Production Contract |
| --- | --- | --- |
| Purged WF | 5 日标签 horizon；5 日 purge；20 日 embargo；test 锁定 | 阻断训练、验证、测试之间的标签窗口重叠与调仓信息泄露 |
| State mask | `require_is_tradable`, remove ST/*ST, remove suspended, require valid price/volume | 停牌、价格/成交量无效、交易状态异常不得进入 tensor |
| Limit-lock mask | `remove_locked_limit_up_or_down_execution_samples` | 剔除 T+1 买入涨停不可买、卖出跌停不可卖一类硬锁定样本 |
| Liquidity mask | `lag1_amount_20d_mean >= 70000` and bottom 5% by date removed | 剔除极低成交额尾部 |
| Size mask | bottom 5% by date on `lag1_log_circ_mv` removed | 剔除微盘尾部 |
| Participation cap | `participation_cap = 0.03` in T+1 fill / CVXPY execution | 单票买入/卖出成交量不超过次日成交额的 3%，作为容量硬约束/软松弛边界 |

Strict mask 落盘审计口径：

| Metric | Count / Rate |
| --- | ---: |
| Raw candidate rows before strict mask | 238,253 |
| Rows kept after strict mask | 208,966 |
| Rows dropped | 29,287 |
| Drop rate | 12.29% |
| Locked-limit drops | 2,021 |
| Low-amount drops | 19,982 |
| Microcap drops | 12,657 |

注意：drop reason 可以重叠，因此单项 reason count 之和大于 dropped rows。`participation_cap=0.03` 不进入模型 `X`，而在执行层约束买卖成交容量；这保证模型学习的是信号，不是交易规则本身。

## 2. Feature Matrix Dictionary

当前 `advanced_sequence_clean_v1` 有两套生产 tensor：

| Build Mode | Feature Count | Tensor Path |
| --- | ---: | --- |
| `alpha_only` | 13 | `data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_purged_walk_forward.npz` |
| `alpha_plus_residual_style` | 18 | `data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz` |

### 2.1 `alpha_only`: Pure But Thin

13 个 alpha 输入均为 lag-1 可得的时序/高频/技术类信号，风险控制、交易状态和原始流动性控制不直接喂给模型。其核心物理性质是 **Pure but Thin**：剥离风格切换噪声后，信号更接近 residual alpha；但在弱势市场或快速 regime 切换时，可表达的状态空间较瘦，可能无法充分学习市值、流动性、波动率与行业拥挤度对未来收益的非线性调制。

| # | Feature Code | Feature Family | Quant Meaning | Financial Interpretation |
| ---: | --- | --- | --- | --- |
| 1 | `lag1_net_mf_strength_20d_mean` | Money flow | 20 日净资金强度均值，lag-1 可得 | 中频资金流入/流出压力 |
| 2 | `lag1_net_mf_strength_60d_mean` | Money flow | 60 日净资金强度均值 | 更慢速的资金偏好和筹码迁移 |
| 3 | `lag1_close_position` | Intraday position | 收盘价在当日高低区间中的位置 | 日内收盘强弱、尾盘资金行为 |
| 4 | `lag1_excess_ret_10d_mean` | Relative return | 相对基准/截面超额收益 10 日均值 | 短中期相对强弱 |
| 5 | `lag1_excess_ret_1d` | Relative return | 1 日相对超额收益 | 最近一日冲击/反转/延续 |
| 6 | `lag1_excess_ret_5d_mean` | Relative return | 5 日相对超额收益均值 | 一周维度相对强弱 |
| 7 | `lag1_industry_neutral_ret_1d` | Industry-neutral return | 行业中性后的 1 日收益 | 剔除行业 beta 后的个股 idiosyncratic move |
| 8 | `lag1_ret_1d` | Raw return | 1 日收益 | 最近价格冲击 |
| 9 | `lag1_ret_20d` | Raw return | 20 日累计/窗口收益 | 月度 momentum / reversal 原料 |
| 10 | `lag1_ret_5d_mean` | Raw return | 5 日收益均值 | 周频 trend pressure |
| 11 | `lag1_bollinger_z_20d` | Technical | 20 日布林 z-score | 价格相对短期波动带的位置 |
| 12 | `lag1_ma_ratio_20_60` | Technical | 20 日均线相对 60 日均线 | 中短均线结构和趋势斜率 |
| 13 | `lag1_macd_hist` | Technical | MACD histogram | 趋势动量的二阶变化 |

**组合含义**：这套特征强调 alpha 的点预测纯度。它牺牲了 raw style carry，使模型更不容易把“买小盘、买低流动性、买高波动”误学成 alpha；但当 market regime 本身主要通过这些风格暴露传递时，模型会变得过度保守，后端优化器只能在较少的高置信 alpha 上部署风险预算。

### 2.2 `alpha_resid_style`: Aggressive But Poisonous

18 特征版本 = 13 个 alpha + 5 个 residualized style carry。需要特别说明：当前 manifest 中新增的 5 列不是直接输入 raw Size/Vol/Mom/Industry 暴露，而是从流动性/成交活跃度相关变量中提取、并对 industry 与 style exposure 线性中性化后的 residual style 信息。其作用是把风格非线性红利的一部分带回模型，同时避免直接把风险控制列混入 alpha tensor。

| # | Feature Code | Carrier Axis | Residualization Meaning | Risk |
| ---: | --- | --- | --- | --- |
| 14 | `lag1_turnover_cost_proxy__resid_style` | Liquidity / turnover | 成交摩擦代理在行业、规模、流动性、波动、动量控制后的残差 | 长尾流动性陷阱，容量约束敏感 |
| 15 | `lag1_turnover_20d_std__resid_style` | Liquidity / activity volatility | 20 日换手波动残差 | 活跃度突然变化可能映射拥挤交易 |
| 16 | `lag1_turnover_60d_std__resid_style` | Liquidity / activity volatility | 60 日换手波动残差 | 慢速拥挤度和流动性状态 |
| 17 | `lag1_amount_rank_pct__resid_style` | Liquidity / size proxy | 成交额截面分位残差 | 与市值/成交容量高度耦合 |
| 18 | `lag1_amount_log__resid_style` | Liquidity / size proxy | 成交额 log 残差 | 容量红利与微盘尾部风险并存 |

这些 residualized style columns 的中性化控制轴如下：

| Style Axis | Representative Controls | Use In Pipeline |
| --- | --- | --- |
| Industry proxy | `industry`, `lag1_industry_turnover_rank`, `lag1_industry_amount_rank`, `lag1_industry_pb_rank`, `lag1_industry_mv_rank` | residualization、optimizer risk constraints |
| Size | `lag1_log_circ_mv`, `lag1_log_total_mv`, `lag1_industry_mv_rank` | residualization、optimizer `industry_size` |
| Liquidity | `lag1_amount_log`, `lag1_amount_rank_pct`, `lag1_turnover_rate_f`, `lag1_turnover_20d_mean` | residualization、capacity and risk diagnostics |
| Volatility | `lag1_ret_20d_std`, `lag1_ret_60d_std`, `lag1_amplitude`, `lag1_vol_log` | residualization、optimizer full style risk set |
| Momentum | `lag1_ret_20d_mean`, `lag1_ret_60d_mean`, `lag1_ret_20d`, `lag1_ret_5d_mean` | residualization、regime/style interaction diagnostics |

**组合含义**：18 特征是 **Aggressive but Poisonous**。它带回了 residual style 的火种，能帮助模型捕捉非线性风格红利和 regime-sensitive ranking；但这些信号在长尾个股上与容量、行业、规模、成交额高度纠缠。若 CVXPY 后端没有足够强的软约束、容量松弛和风格对冲机制，模型分数会把优化器推向约束边界，表现为 infeasible、fallback、或者为满足容量约束而持有大量现金。

## 3. Transmission To The CVXPY Optimizer

### 3.1 从模型分数到组合权重

执行栈使用模型输出 `pred_score` 做截面排序，按每个 rebalance date 构造候选池。CVXPY 决策变量为个股权重 `w`，目标大致由 alpha 分数、现金/持仓激励、换手惩罚、风格暴露惩罚和容量松弛惩罚共同决定：

```text
maximize  alpha_z @ w
          + cash_penalty * sum(w)
          - turnover_penalty * sum(buys + sells)
          - style_penalty * exposure_penalty
          - exposure_slack_penalty * exposure_slack
          - buy_capacity_slack_penalty * buy_capacity_slack
```

关键硬约束包括：

| Constraint | Meaning |
| --- | --- |
| `sum(w) <= 1` | 多头现金账户，不允许杠杆 |
| `w_i <= single_name_cap` | 单票权重上限，默认约等于 `1/k` |
| `sum(buys + sells) <= turnover_cap` | 调仓换手上限 |
| `buys <= 0.03 * next_amount / portfolio_nav + slack` | 单日 3% participation cap |
| `sells <= 0.03 * next_amount / portfolio_nav` | 卖出容量约束 |
| risk exposure cap | 对行业、市值、流动性、波动率、动量代理做 z-score 暴露限制 |

### 3.2 为什么 13 特征会“现金躺平”

`alpha_only` 的预测空间更纯，但在弱 validation regime 下 signal support 较窄。优化器看到的是一个“可解释但瘦”的 alpha vector：高分股票数量不足、可交易容量有限、且在加入行业/市值/风格约束后可部署权重下降。结果是优化器倾向保留现金，而不是为了低置信 alpha 强行填满仓位。

本地可复核 evidence：

| Scope | Avg Cash | Infeasible / Fallback |
| --- | ---: | ---: |
| `clean_alpha_only`, both_cash_buffer all grid | 40.8% | 547 / 15,072 |
| `clean_alpha_only`, validation all grid | 45.5% | 344 / 9,024 |
| `clean_alpha_only`, test `industry_size,k=10,turnover=0.02` | 47.7% | 2 / 63 |

团队沟通中提到的“现金躺平 47.3%”应理解为特定 optimizer slice 的诊断现象，而不是所有路径的简单均值；当前落盘全网格已经可以复现同量级的高现金状态。

### 3.3 为什么 18 特征更容易撞墙

18 特征版本把 residual style carry 带回模型，通常能提高 test ranking 的进攻性。例如在 `industry_proxy,k=20,turnover=0.02` 路径上，test 年化从 `alpha_only` 的约 24.0% 提升到 `alpha_resid_style` 的约 31.2%。但这种收益来源并非免费午餐：模型更偏向长尾流动性、行业拥挤和规模异常组合，CVXPY 在同时满足容量、单票上限、风格暴露和换手约束时更频繁触边。

本地可复核 evidence：

| Scope | Avg Cash | Infeasible / Fallback |
| --- | ---: | ---: |
| `clean_alpha_resid_style`, both_cash_buffer all grid | 33.2% | 952 / 15,072 |
| `clean_alpha_resid_style`, validation all grid | 36.8% | 682 / 9,024 |
| `industry_size_liquidity_vol_mom,k=20,style=0,turnover=0`, all split | 18.4% | 103 / 157 |

因此，用户侧诊断中的“Infeasible 撞击 580 次”应被视为某一 promoted/sliced grid 的中间口径；当前本地 `both_cash_buffer` 全网格显示更严厉的 952 次 infeasible，validation 子集为 682 次。结论方向一致：18 特征的 alpha 更强，但对组合优化器的风格/容量冲击也更强。

### 3.4 TopKMarginICLoss(k=10) 的代数聚焦

后续模型改造的目标不是让模型在全截面均匀降低 MSE，而是让它在真实开仓域内排序正确。`TopKMarginICLoss(k=10)` 做了三件事：

| Component | Algebraic Focus | Portfolio Meaning |
| --- | --- | --- |
| True top-k positives | `true_top_idx = topk(target, k)` | 把真实未来收益最高的 10 只股票作为多头 oracle |
| Hard negatives | `topk(pred, k + negative_multiplier*k)` excluding true positives | 惩罚模型高分但真实收益不佳的 false positives |
| Margin pairwise loss | `softplus((margin - pred_pos + pred_neg) / temperature)` | 要求真赢家的分数显著高于高分伪赢家 |
| Soft portfolio return term | `relu(mean(target_true_top) - softmax(pred/T) @ target)` | 直接对齐可交易多头组合收益 |
| IC/MSE auxiliary | `ranking_loss + ic_alpha * IC_loss + mse_alpha * MSE` | 保留日度截面相关性，避免纯 top-k 过拟合 |

这解释了为什么“换模型/换 loss”是主线，而不是只调优化器参数。优化器只能在给定 `pred_score` 上做可行域投影；如果模型分数本身没有在 TopK 开仓域聚焦，CVXPY 要么现金防守，要么被风格和容量约束反复打回。

## 4. Recommended Operating Contract

| Use Case | Recommended Tensor | Rationale |
| --- | --- | --- |
| Baseline / audit / PM sanity check | `alpha_only` | 最干净、最容易解释 residual alpha 是否存在 |
| Regime-aware model development | `alpha_resid_style` + gated architecture | 允许模型条件性使用 residual style carry |
| Production optimizer research | 18 特征模型 + explicit soft exposure controls | 必须把 industry/size/liquidity/vol/mom 对冲和 capacity slack 一起调 |
| Final test reporting | Locked test only after validation decision | 避免把 2025-2026 holdout 反复用于选择模型 |

最低复核 checklist：

```powershell
conda run -n dl_env python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --dry-run --device cpu
conda run -n dl_env python scripts/audit/audit_point_in_time.py --out-dir outputs/audit/_pathcheck_clean_model
conda run -n dl_env python scripts/audit/audit_barra_lite_residual_alpha.py
conda run -n dl_env python scripts/audit/audit_clean_resid_mainline.py
```

核心结论：`clean_dataset v20260526` 已经从“能训练模型的数据集”升级为“可执行、可审计、可解释的组合研究资产”。13 特征提供干净 residual alpha 基线；18 特征提供进攻性 residual style carry。真正的研究焦点不再是继续堆原始特征，而是让模型在 TopK 多头开仓域学会何时使用风格火种、何时抑制长尾毒性，并让 CVXPY 后端以显式约束吸收而非放大这些暴露。
