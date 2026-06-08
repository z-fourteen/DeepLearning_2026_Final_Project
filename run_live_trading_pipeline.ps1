param(
  [string]$TradeDate = (Get-Date -Format "yyyyMMdd"),
  [string]$Config = "configs/live/live_trading.yaml",
  [string]$DataVersion = "v20260526",
  [string]$StartDate = "20160104",
  [string]$EndDate = "",
  [string]$FeatureDate = "",
  [switch]$RunDag,
  [switch]$SkipDag,
  [switch]$FullDag,
  [switch]$SkipPrepareFeatures,
  [switch]$SkipPrepareAccountInputs,
  [switch]$AllowRawFeatureFallback,
  [switch]$Execute,
  [switch]$Reset,
  [switch]$NoPush,
  [switch]$WaitForSchedule
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
if (-not $FeatureDate) {
  $FeatureDate = if ($EndDate) { $EndDate } else { $TradeDate }
}
Write-Host "Feature/data end date=$FeatureDate"
Write-Host "Use -WaitForSchedule to enforce 08:30/09:00/09:15 wall-clock gates."

if ($RunDag) {
  $DagArgs = @(
    "scripts/run_daily_dag.py",
    "--data-version", $DataVersion,
    "--start-date", $StartDate,
    "--end-date", $FeatureDate
  )
  if (-not $FullDag -and -not $Reset) {
    $DagArgs += "--incremental"
  }
  Invoke-LiveStage "data DAG" $DagArgs
} elseif ($SkipDag) {
  Write-Host "Skipping data DAG by request."
} else {
  Write-Host "Data DAG not requested. Use -RunDag for full/incremental refresh or -SkipDag to make this explicit."
}

$PrepareInputsArgs = @(
  "scripts/live/00_prepare_live_inputs.py",
  "--config", $Config,
  "--data-version", $DataVersion,
  "--trade-date", $TradeDate,
  "--feature-date", $FeatureDate,
  "--features-parquet", "data/live/features/features_$FeatureDate.parquet"
)
if ($SkipPrepareFeatures) {
  $PrepareInputsArgs += "--skip-prepare-features"
}
if ($SkipPrepareAccountInputs) {
  $PrepareInputsArgs += "--skip-prepare-account-inputs"
}
if ($AllowRawFeatureFallback) {
  $PrepareInputsArgs += "--allow-raw-fallback"
}
if ($Reset) {
  $PrepareInputsArgs += "--overwrite"
}
if ($SkipPrepareFeatures -and $SkipPrepareAccountInputs) {
  Write-Host "Skipping live input preparation by request."
} else {
  Invoke-LiveStage "prepare live inputs" $PrepareInputsArgs
}

Wait-UntilClock "08:30"
Invoke-LiveStage "08:30-09:00 inference" @(
  "scripts/live/01_live_inference.py",
  "--config", $Config,
  "--trade-date", $TradeDate,
  "--feature-date", $FeatureDate,
  "--features-parquet", "data/live/features/features_$FeatureDate.parquet"
)

Wait-UntilClock "09:00"
Invoke-LiveStage "09:00-09:15 optimization" @(
  "scripts/live/02_live_optimization.py",
  "--config", $Config,
  "--trade-date", $TradeDate,
  "--feature-date", $FeatureDate,
  "--liquidity-parquet", "data/live/features/features_$FeatureDate.parquet"
)

Wait-UntilClock "09:15"
Invoke-LiveStage "09:15-09:25 target orders" @(
  "scripts/live/03_generate_target_orders.py",
  "--config", $Config,
  "--trade-date", $TradeDate
)

if ($Execute) {
  $ExecutionArgs = @(
    "scripts/live/05_interactive_execution.py",
    "--config", $Config,
    "--trade-date", $TradeDate
  )
  if ($Reset) {
    $ExecutionArgs += "--reset"
  }
  if ($NoPush) {
    $ExecutionArgs += "--no-push"
  }
  Invoke-LiveStage "interactive execution" $ExecutionArgs
}

Write-Host ""
Write-Host "Live pipeline completed. Orders:"
Write-Host "outputs/live_orders/orders_$TradeDate.csv"
Write-Host ""
Write-Host "One-command forms:"
Write-Host "  Full build:        .\run_live_trading_pipeline.ps1 -RunDag -FullDag -DataVersion $DataVersion -EndDate $FeatureDate -TradeDate $TradeDate"
Write-Host "  Auto incremental:  .\run_live_trading_pipeline.ps1 -RunDag -DataVersion $DataVersion -EndDate $FeatureDate -TradeDate $TradeDate"
