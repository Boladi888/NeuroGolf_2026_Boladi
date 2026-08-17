[CmdletBinding()]
param(
    [ValidateSet('Codex','Codex2','Claude')]
    [string]$Lane = 'Codex',
    [string]$Batch = '',
    [string]$Reason = 'supervisor_stop'
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $PSCommandPath
$handoffRoot = Resolve-Path (Join-Path $scriptDir '..')
$laneLc = $Lane.ToLower()
if (-not $Batch) { throw 'Pass -Batch so the stop is scoped.' }
$pidFile = Join-Path $handoffRoot ("private/rescue_verify_{0}_p{1}.pid.json" -f $laneLc, $Batch)

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "no rescue pid file at $pidFile"
    return
}

$info = $null
try { $info = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json } catch { }
if (-not $info -or -not $info.pid) {
    Write-Warning "rescue pid file is unreadable or has no pid: $pidFile"
    return
}

$rescuePid = [int]$info.pid
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$rescuePid" -ErrorAction SilentlyContinue
if ($proc) {
    if ((($info.PSObject.Properties.Name -contains 'running' -and -not [bool]$info.running) -or [string]$info.state -eq 'stopped') -and $proc.CommandLine -notmatch 'rescue_verify\.py|Start-HandoffRescue') {
        Write-Host "rescue pid file is already stopped; pid=$rescuePid now belongs to another process, not killing"
    } elseif ($proc.CommandLine -match 'rescue_verify\.py|Start-HandoffRescue') {
        & taskkill /PID $rescuePid /T /F 2>&1 | Out-String | Write-Host
        Write-Host "killed rescue verifier tree pid=$rescuePid lane=$Lane batch=$Batch"
    } else {
        throw "Refusing to kill pid=$rescuePid because it is not a rescue verifier: $($proc.CommandLine)"
    }
} else {
    Write-Host "rescue pid=$rescuePid not alive"
}

$stamp = [ordered]@{}
if ($info) {
    foreach ($prop in $info.PSObject.Properties) { $stamp[$prop.Name] = $prop.Value }
}
$stamp['pid'] = $rescuePid
$stamp['lane'] = $Lane
$stamp['batch'] = $Batch
$stamp['state'] = 'stopped'
$stamp['running'] = $false
$stamp['stopped_utc'] = (Get-Date).ToUniversalTime().ToString('o')
$stamp['exit_reason'] = "Stop-HandoffRescue: $Reason"
$stamp | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $pidFile -Encoding UTF8
Write-Host "rescue pid file stamped stopped: $pidFile"
