# ============================================================
#  Transformer 全流程脚本：训练 → 回测 → 组合优化
#  用法:  .\run_transformer_all.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$PROJECT = $PSScriptRoot

# ---- 4 组配置 ----
$CONFIGS = @(
    "configs/models/transformer_l20_clean_alpha_resid_style.yaml"           # CLS + Huber
    "configs/models/transformer_l20_clean_alpha_resid_style_mseic.yaml"     # CLS + MSE_IC
    "configs/models/transformer_l20_clean_alpha_resid_style_attn_pool.yaml" # Attention + Huber
    "configs/models/transformer_l20_clean_alpha_resid_style_attn_mseic.yaml" # Attention + MSE_IC
)

# ============================================================
#  Phase 1: 训练
# ============================================================
Write-Host "`n========== Phase 1: Transformer 训练 (4 组) ==========" -ForegroundColor Cyan

foreach ($cfg in $CONFIGS) {
    # 从配置文件提取 run_name
    $yaml = Get-Content "$PROJECT/$cfg" -Raw
    if ($yaml -match 'name:\s*"([^"]+)"') {
        $runName = $Matches[1]
    } else {
        Write-Host "[WARN] Cannot parse run name from $cfg, skipping" -ForegroundColor Yellow
        continue
    }
    $predPath = "$PROJECT/outputs/runs/$runName/predictions.parquet"

    if (Test-Path $predPath) {
        Write-Host "[SKIP] $runName — predictions.parquet already exists" -ForegroundColor Green
    } else {
        Write-Host "`n[TRAIN] $runName" -ForegroundColor White
        python "$PROJECT/scripts/modeling/train_sequence_transformer.py" `
            --config "$PROJECT/$cfg" `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Training failed for $runName, skipping downstream" -ForegroundColor Red
            continue
        }
    }
}

# ============================================================
#  Phase 2: T+1 回测
# ============================================================
Write-Host "`n========== Phase 2: T+1 回测 ==========" -ForegroundColor Cyan

$EXEC_LABELS = "$PROJECT/data/mart/labels/execution_labels_v20260526.parquet"

foreach ($cfg in $CONFIGS) {
    $yaml = Get-Content "$PROJECT/$cfg" -Raw
    if ($yaml -match 'name:\s*"([^"]+)"') {
        $runName = $Matches[1]
    } else { continue }
    $predPath = "$PROJECT/outputs/runs/$runName/predictions.parquet"
    $btDir = "$PROJECT/outputs/backtest/${runName}_t1"

    if (-not (Test-Path $predPath)) {
        Write-Host "[SKIP] $runName — no predictions, skipping backtest" -ForegroundColor Yellow
        continue
    }
    if (Test-Path "$btDir/t1_fill_metrics.json") {
        Write-Host "[SKIP] $runName — backtest already exists" -ForegroundColor Green
        continue
    }

    Write-Host "`n[BACKTEST] $runName (k=20, keep=2)" -ForegroundColor White
    python "$PROJECT/scripts/backtest/backtest_t1_fill_sim.py" `
        --predictions $predPath `
        --execution-labels $EXEC_LABELS `
        --output-dir $btDir `
        --k 20 `
        --keep-multiplier 2 `
        --portfolio-nav 10000000 `
        --participation-cap 0.03 `
        --cost-bps 10 `
        --slippage-bps 5 `
        --rebalance-stride 5 `
        --min-daily-count 20
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Backtest failed for $runName" -ForegroundColor Red
        continue
    }
}

# ============================================================
#  Phase 3: LP 组合优化
# ============================================================
Write-Host "`n========== Phase 3: LP 组合优化 ==========" -ForegroundColor Cyan

$MART = "$PROJECT/data/mart/datasets/core/dataset_v20260526.parquet"

foreach ($cfg in $CONFIGS) {
    $yaml = Get-Content "$PROJECT/$cfg" -Raw
    if ($yaml -match 'name:\s*"([^"]+)"') {
        $runName = $Matches[1]
    } else { continue }
    $predPath = "$PROJECT/outputs/runs/$runName/predictions.parquet"
    $optDir = "$PROJECT/outputs/backtest/${runName}_optimizer"

    if (-not (Test-Path $predPath)) {
        Write-Host "[SKIP] $runName — no predictions, skipping optimizer" -ForegroundColor Yellow
        continue
    }
    if (Test-Path "$optDir/t1_fill_metrics.json") {
        Write-Host "[SKIP] $runName — optimizer already exists" -ForegroundColor Green
        continue
    }

    Write-Host "`n[OPTIMIZE] $runName" -ForegroundColor White
    python "$PROJECT/scripts/portfolio/optimize_feasible_cash_buffer.py" `
        --predictions $predPath `
        --mart $MART `
        --labels $EXEC_LABELS `
        --output-dir $optDir `
        --k 20 `
        --style-penalty 0,0.05,0.10,0.20 `
        --turnover-penalty 0,0.02 `
        --exposure-cap 0.35 `
        --min-invested 0.80 `
        --turnover-cap 1.0 `
        --participation-cap 0.03 `
        --single-name-cap 0.0 `
        --exposure-slack-penalty 25.0 `
        --buy-capacity-slack-penalty 1000 `
        --cash-penalty 0.0 `
        --min-invested-shortfall-penalty 0.0 `
        --portfolio-nav 10000000 `
        --cost-bps 10 `
        --slippage-bps 5 `
        --rebalance-stride 5 `
        --min-daily-count 40 `
        --risk-control none,industry_proxy,industry_size,industry_size_liquidity_vol_mom
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Optimizer failed for $runName" -ForegroundColor Red
        continue
    }
}

# ============================================================
#  Summary
# ============================================================
Write-Host "`n========== 完成！输出汇总 ==========" -ForegroundColor Cyan

foreach ($cfg in $CONFIGS) {
    $yaml = Get-Content "$PROJECT/$cfg" -Raw
    if ($yaml -match 'name:\s*"([^"]+)"') {
        $runName = $Matches[1]
    } else { continue }

    $hasPred   = (Test-Path "$PROJECT/outputs/runs/$runName/predictions.parquet")
    $hasBT     = (Test-Path "$PROJECT/outputs/backtest/${runName}_t1/t1_fill_metrics.json")
    $hasOpt    = (Test-Path "$PROJECT/outputs/backtest/${runName}_optimizer/t1_fill_metrics.json")

    $status = if ($hasPred) { "OK" } else { "NO" }
    $btStatus = if ($hasBT) { "OK" } else { "NO" }
    $optStatus = if ($hasOpt) { "OK" } else { "NO" }

    Write-Host "  $runName"
    Write-Host "    训练:    $status  outputs/runs/$runName/"
    Write-Host "    回测:    $btStatus  outputs/backtest/${runName}_t1/"
    Write-Host "    优化:    $optStatus  outputs/backtest/${runName}_optimizer/"
}
