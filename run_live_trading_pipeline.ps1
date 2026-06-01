param(
  [string]$TradeDate = (Get-Date -Format "yyyyMMdd"),
  [string]$Config = "configs/live/live_trading.yaml",
  [switch]$WaitForSchedule,
  [switch]$RunIntradayMonitor
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:CONDA_NO_PLUGINS = "true"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

function Wait-UntilClock {
  param([string]$Clock)
  if (-not $WaitForSchedule) { return }
  $target = [datetime]::ParseExact("$TradeDate $Clock", "yyyyMMdd HH:mm", $null)
  while ((Get-Date) -lt $target) {
    $remain = [int](($target - (Get-Date)).TotalSeconds)
    Write-Host "Waiting until $Clock, remaining ${remain}s ..."
    Start-Sleep -Seconds ([Math]::Min(60, [Math]::Max(1, $remain)))
  }
}

function Invoke-LiveStage {
  param(
    [string]$Name,
    [string[]]$Args
  )
  Write-Host ""
  Write-Host "================================================================================"
  Write-Host "LIVE STAGE: $Name"
  Write-Host "================================================================================"
  conda run --no-capture-output -n dl_env python @Args
  if ($LASTEXITCODE -ne 0) {
    [console]::beep(1200, 800)
    throw "Live stage failed: $Name"
  }
}

Write-Host "Live trading pipeline trade_date=$TradeDate config=$Config"
Write-Host "Use -WaitForSchedule to enforce 08:30/09:00/09:15 wall-clock gates."

Wait-UntilClock "08:30"
Invoke-LiveStage "08:30-09:00 inference" @(
  "scripts/live/01_live_inference.py",
  "--config", $Config,
  "--trade-date", $TradeDate
)

Wait-UntilClock "09:00"
Invoke-LiveStage "09:00-09:15 optimization" @(
  "scripts/live/02_live_optimization.py",
  "--config", $Config,
  "--trade-date", $TradeDate
)

Wait-UntilClock "09:15"
Invoke-LiveStage "09:15-09:25 target orders" @(
  "scripts/live/03_generate_target_orders.py",
  "--config", $Config,
  "--trade-date", $TradeDate
)

if ($RunIntradayMonitor) {
  Wait-UntilClock "09:30"
  Invoke-LiveStage "09:30-15:00 intraday execution monitor" @(
    "scripts/live/04_intraday_execution_monitor.py",
    "--config", $Config,
    "--trade-date", $TradeDate,
    "--loop"
  )
}

Write-Host ""
Write-Host "Live pipeline completed. Orders:"
Write-Host "outputs/live_orders/orders_$TradeDate.csv"
