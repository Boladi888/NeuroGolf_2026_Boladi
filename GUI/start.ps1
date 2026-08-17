# NeuroGolf Lab launcher (Windows / PowerShell)
# Opens the server in its own visible console window. Closing that window
# kills the server (in-process uvicorn + KILL_ON_JOB_CLOSE JobObject).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from template."
}

if (-not (Test-Path "client\dist")) {
    Write-Host "client/dist missing - building the web client (one-time)..."
    Push-Location client
    npm install
    npm run build
    Pop-Location
}

Start-Process cmd -ArgumentList "/k", "python", "launch.py"
Write-Host "NeuroGolf Lab window opened (default http://127.0.0.1:8082)."
Write-Host "Close that window to stop the server."
