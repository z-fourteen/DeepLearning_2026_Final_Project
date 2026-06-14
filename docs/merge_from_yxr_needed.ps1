param(
  [string]$FinalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$YxrRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\Final-YXR")).Path,
  [switch]$Apply
)

$ErrorActionPreference = "Stop"

$Items = @(
  "src\models\transformer.py",
  "src\models\transformer_enhanced.py",
  "src\data\__init__.py",
  "src\data\sequence_npz_dataset.py",
  "src\data\samplers.py",
  "configs\models\StdTF-l20-cls-mse-f13.yaml",
  "configs\models\StdTF-l60-cls-mse-f13.yaml",
  "configs\models\StdTF-l60-cls-mse-f18.yaml",
  "configs\models\StdTF-l60-cls-huber-f13.yaml",
  "configs\models\StdTF-l60-attn-mse-f13.yaml",
  "configs\models\EnhancedTF-l60-cls-mse-f13.yaml",
  "configs\live\live_trading_StdTF.yaml",
  "configs\live\live_trading_EnhancedTF.yaml",
  "configs\data\data.yaml",
  "configs\data\experiment.yaml",
  "configs\data\labels.yaml",
  "configs\data\splits.yaml"
)

$Directories = @(
  "scripts\data"
)

Write-Host "Final root: $FinalRoot"
Write-Host "YXR root:   $YxrRoot"
Write-Host ""

foreach ($Item in $Items) {
  $Source = Join-Path $YxrRoot $Item
  $Target = Join-Path $FinalRoot $Item
  if (-not (Test-Path $Source)) {
    Write-Warning "Missing in YXR: $Item"
    continue
  }
  if ($Apply) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target -Force
    Write-Host "Copied file: $Item"
  } else {
    Write-Host "Need file:   $Item"
  }
}

foreach ($Directory in $Directories) {
  $Source = Join-Path $YxrRoot $Directory
  $Target = Join-Path $FinalRoot $Directory
  if (-not (Test-Path $Source)) {
    Write-Warning "Missing in YXR: $Directory"
    continue
  }
  if ($Apply) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Target -Recurse -Force
    Write-Host "Copied dir:  $Directory"
  } else {
    Write-Host "Need dir:    $Directory"
  }
}

Write-Host ""
Write-Host "Manual follow-up after copying:"
Write-Host "  1. Add TransformerStockModel and EnhancedTransformerModel exports to src\models\__init__.py."
Write-Host "  2. Register transformer_encoder and transformer_enhanced in scripts\modeling\train_sequence.py."
Write-Host "  3. Keep data/, outputs/, model.pt, and predictions.parquet out of the source submission unless explicitly required."
