Write-Host "Starting Plate Tampering Dashboard on http://localhost:8501"
Write-Host "Make sure the API is already running in another terminal."
Write-Host ""

Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
python run_dashboard.py
