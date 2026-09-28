Write-Host "Starting Plate Tampering API on http://127.0.0.1:8000"
Write-Host "KEEP THIS WINDOW OPEN while using the dashboard."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
python run_api.py
