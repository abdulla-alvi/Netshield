<#
Rebuild DB schema (idempotent), register edge_bert_quantized.pt as active model, verify environment.
Usage: .\run_everything.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Error "Missing venv. Run: python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt"
}

& .\venv\Scripts\Activate.ps1

Write-Host "--- init_db.py ---"
python init_db.py

Write-Host "--- register_bundled_model.py ---"
python scripts\register_bundled_model.py

Write-Host "--- check_setup.py --mysql ---"
python scripts\check_setup.py --mysql

Write-Host "--- done ---"
