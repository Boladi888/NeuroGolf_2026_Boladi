param(
    [int]$IntervalSeconds = 3,
    [switch]$NoClear
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

try {
    $Host.UI.RawUI.WindowTitle = "NG Gate Queue Watch -- CLOSE THIS WINDOW TO STOP"
} catch {}

$py = $env:NEUROGOLF_PYTHON
if (-not $py -and (Test-Path ".venv\Scripts\python.exe")) {
    $py = (Resolve-Path ".venv\Scripts\python.exe").Path
}
if (-not $py) {
    $py = "python"
}

Write-Host "NG Gate Queue Watch is running in THIS window."
Write-Host "Close this window or press Ctrl+C to stop the watcher only."
Write-Host "Repo: $Root"
Write-Host "Python: $py"
Write-Host ""

$argsList = @("tools\ng_gate_queue_snapshot.py", "--watch", "--interval", "$IntervalSeconds")
if ($NoClear) {
    $argsList += "--no-clear"
}

& $py @argsList

