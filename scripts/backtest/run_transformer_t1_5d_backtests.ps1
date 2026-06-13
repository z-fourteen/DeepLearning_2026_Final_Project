param(
  [string]$Python = "python",
  [string]$FinalYxrRoot = "",
  [string]$OutputRoot = "",
  [string]$K = "10,20,30",
  [string]$KeepMultiplier = "1",
  [double]$PortfolioNav = 10000000,
  [double]$ParticipationCap = 0.03,
  [double]$CostBps = 10.0,
  [double]$SlippageBps = 5.0,
  [int]$RebalanceStride = 5,
  [int]$MinDailyCount = 40
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if ([string]::IsNullOrWhiteSpace($FinalYxrRoot)) {
  $FinalYxrRoot = Resolve-Path (Join-Path $ProjectRoot "..\Final-YXR")
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $FinalYxrRoot "outputs\backtest\unified_t1_5d_transformer"
}

$BacktestScript = Join-Path $ProjectRoot "scripts\backtest\backtest_t1_fill_sim.py"
$ExecutionLabels = Join-Path $FinalYxrRoot "data\mart\labels\execution_labels_v20260526.parquet"

$Runs = @(
  "EnhancedTF-l60-cls-mse-f13",
  "StdTF-l20-cls-mse-f13",
  "StdTF-l60-attn-mse-f13",
  "StdTF-l60-cls-huber-f13",
  "StdTF-l60-cls-mse-f18",
  "StdTF-l60-cls-mse-f13"
)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($Run in $Runs) {
  $Predictions = Join-Path $FinalYxrRoot "outputs\runs\$Run\predictions.parquet"
  $RunOutput = Join-Path $OutputRoot $Run
  if (-not (Test-Path $Predictions)) {
    Write-Warning "Skip $Run because predictions file is missing: $Predictions"
    continue
  }

  Write-Host ""
  Write-Host "=== Running 5-day T+1 fill backtest: $Run ==="
  New-Item -ItemType Directory -Force -Path $RunOutput | Out-Null
  & $Python $BacktestScript `
    --predictions $Predictions `
    --execution-labels $ExecutionLabels `
    --output-dir $RunOutput `
    --k $K `
    --keep-multiplier $KeepMultiplier `
    --portfolio-nav $PortfolioNav `
    --participation-cap $ParticipationCap `
    --cost-bps $CostBps `
    --slippage-bps $SlippageBps `
    --rebalance-stride $RebalanceStride `
    --min-daily-count $MinDailyCount
}

Write-Host ""
Write-Host "Done. Outputs are under:"
Write-Host $OutputRoot
Write-Host ""
Write-Host "Each model directory contains:"
Write-Host "  t1_fill_periods.csv"
Write-Host "  t1_fill_metrics.json"
