$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m pytest tests -q
python -m under_extinction --config configs/smoke.yaml smoke
python -m under_extinction --config configs/smoke.yaml dry-run
