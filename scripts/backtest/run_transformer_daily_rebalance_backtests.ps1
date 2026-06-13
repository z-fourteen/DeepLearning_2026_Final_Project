param(
  [string]$Python = "python",
  [string]$FinalYxrRoot = "",
  [string]$FinalOxx2Root = "",
  [string]$OutputRoot = "",
  [string]$K = "10,20,30",
  [string]$Mode = "full_topk",
  [int]$ReplaceCount = 3,
  [string]$PredictionSplit = "test",
  [string]$StartDate = "",
  [string]$EndDate = "",
  [double]$CostBps = 10.0,
  [double]$SlippageBps = 5.0,
  [int]$MinDailyCount = 40,
  [string]$BenchmarkCode = "399006.SZ"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if ([string]::IsNullOrWhiteSpace($FinalYxrRoot)) {
  $FinalYxrRoot = Resolve-Path (Join-Path $ProjectRoot "..\Final-YXR")
}
if ([string]::IsNullOrWhiteSpace($FinalOxx2Root)) {
  $FinalOxx2Root = Resolve-Path (Join-Path $ProjectRoot "..\Final-OXX2")
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $FinalYxrRoot "outputs\backtest\unified_daily_rebalance_transformer"
}

$BacktestScript = Join-Path $ProjectRoot "scripts\backtest\backtest_daily_rebalance.py"
if ((Test-Path (Join-Path $FinalOxx2Root "daily")) -and (Test-Path (Join-Path $FinalOxx2Root "market"))) {
  $AshareRoot = Get-Item $FinalOxx2Root
} else {
  $AshareRoot = Get-ChildItem -Path $FinalOxx2Root -Directory |
    Where-Object {
      (Test-Path (Join-Path $_.FullName "daily")) -and
      (Test-Path (Join-Path $_.FullName "market"))
    } |
    Select-Object -First 1
}
if ($null -eq $AshareRoot) {
  throw "Could not find data root under $FinalOxx2Root. Expected a directory containing daily and market subfolders."
}
$DailyRoot = Join-Path $AshareRoot.FullName "daily"
$MarketRoot = Join-Path $AshareRoot.FullName "market"

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
  $RunOutput = Join-Path $OutputRoot "$Run\$Mode"
  if (-not (Test-Path $Predictions)) {
    Write-Warning "Skip $Run because predictions file is missing: $Predictions"
    continue
  }

  Write-Host ""
  Write-Host "=== Running daily rebalance backtest: $Run / $Mode ==="
  New-Item -ItemType Directory -Force -Path $RunOutput | Out-Null

  $ScriptArgs = @(
    $BacktestScript,
    "--predictions", $Predictions,
    "--daily-root", $DailyRoot,
    "--market-root", $MarketRoot,
    "--benchmark-code", $BenchmarkCode,
    "--output-dir", $RunOutput,
    "--k", $K,
    "--mode", $Mode,
    "--replace-count", $ReplaceCount,
    "--split", $PredictionSplit,
    "--cost-bps", $CostBps,
    "--slippage-bps", $SlippageBps,
    "--min-daily-count", $MinDailyCount
  )
  if (-not [string]::IsNullOrWhiteSpace($StartDate)) {
    $ScriptArgs += @("--start-date", $StartDate)
  }
  if (-not [string]::IsNullOrWhiteSpace($EndDate)) {
    $ScriptArgs += @("--end-date", $EndDate)
  }
  & $Python @ScriptArgs
}

Write-Host ""
Write-Host "Done. Outputs are under:"
Write-Host $OutputRoot
Write-Host ""
Write-Host "Each model/mode directory contains:"
Write-Host "  daily_rebalance_periods.csv"
Write-Host "  daily_rebalance_positions.csv"
Write-Host "  daily_rebalance_summary.csv"
Write-Host "  daily_rebalance_metrics.json"
