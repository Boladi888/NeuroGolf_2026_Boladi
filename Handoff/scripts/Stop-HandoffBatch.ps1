[CmdletBinding()]
<#
.SYNOPSIS
  Safe, surgical abort of a HandOff runner batch (proposal codex_p38 P8 / claude_p37 close-out).
.DESCRIPTION
  Kills ONLY the lane's runner-loop process tree (never a blanket claude.exe/codex kill — that would
  take out your own session and unrelated agents), stamps the pid file state=stopped with an explicit
  exit_reason (a force-kill skips the runner's own graceful stamp), optionally stops the lane monitor,
  and writes a close-out note with the task-state counts at abort. Stale-pid guarded: it refuses to kill
  a pid whose command line is not a HandoffRunner (protects against pid reuse).
.EXAMPLE
  .\Handoff\scripts\Stop-HandoffBatch.ps1 -Lane Codex -Batch 38 -Reason stuck_task369 -StopMonitor
#>
param(
    [ValidateSet('Codex','Codex2','Claude')]
    [string]$Lane = 'Claude',
    [string]$Batch = '',
    [string]$Reason = 'supervisor_manual_abort',
    [switch]$StopMonitor,
    [switch]$KeepRescue
)

$ErrorActionPreference = 'Stop'
$scriptDir   = Split-Path -Parent $PSCommandPath
$handoffRoot = Resolve-Path (Join-Path $scriptDir '..')
$repoRoot    = Resolve-Path (Join-Path $handoffRoot '..')
$laneLc      = $Lane.ToLower()
$pidFile     = Join-Path $handoffRoot ("private/runner_loop.{0}.pid.json" -f $laneLc)

function Test-PidAlive([int]$ProcId) {
    try {
        $null = Get-Process -Id $ProcId -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Stop-Tree([int]$ProcId, [string]$what) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        $out = & taskkill /PID $ProcId /T /F 2>&1 | Out-String
        $out | Write-Host
        Start-Sleep -Milliseconds 500
        if (-not (Test-PidAlive $ProcId)) {
            Write-Host "  killed $what tree pid=$ProcId attempt=$attempt verified"
            return
        }
        Write-Warning "  $what tree pid=$ProcId still alive after taskkill attempt=$attempt"
    }
    throw "failed to stop $what tree pid=$ProcId after verified taskkill attempts"
}

# --- read runner pid (tolerate a glitchy/partial read) ---
$info = $null
if (Test-Path -LiteralPath $pidFile) { try { $info = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json } catch { } }
else { Write-Warning "no pid file at $pidFile" }
$runnerPid = if ($info -and $info.pid) { [int]$info.pid } else { 0 }

# --- kill ONLY the runner loop tree, stale-pid guarded ---
if ($runnerPid -gt 0) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$runnerPid" -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Host "runner pid=$runnerPid not alive (nothing to kill)"
    } elseif ($proc.CommandLine -match 'Start-HandoffRunner') {
        Stop-Tree $runnerPid "runner (lane=$Lane)"
    } else {
        Write-Warning "pid $runnerPid is NOT a HandoffRunner (possible pid reuse); NOT killing. cmd: $($proc.CommandLine)"
    }
} else {
    Write-Host "no runner pid recorded; skipping kill"
}

# --- optionally stop the lane monitor loop (matched by command line + lane) ---
if ($StopMonitor) {
    $mon = Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $cmd = [string]$_.CommandLine
            $laneMatch = $cmd -match ("(?i)-Lane\s+{0}\b|--lane\s+{0}\b|_{0}_" -f $laneLc)
            $batchMatch = (-not $Batch) -or $cmd -match ("(?i)-Batch\s+{0}\b|--batch\s+{0}\b|batch{0}\b|_p{0}\b" -f [regex]::Escape($Batch))
            $cmd -match 'handoff_monitor|Start-HandoffMonitor' -and $laneMatch -and $batchMatch
        }
    if ($mon) { foreach ($m in $mon) { Stop-Tree ([int]$m.ProcessId) "monitor" } } else { Write-Host "no matching $Lane monitor process found" }
}

# --- stop the managed rescue verifier too, unless explicitly left alone ---
$rescueStopped = $false
if (-not $KeepRescue -and $Batch) {
    $stopRescue = Join-Path $scriptDir 'Stop-HandoffRescue.ps1'
    if (Test-Path -LiteralPath $stopRescue) {
        try {
            & $stopRescue -Lane $Lane -Batch $Batch -Reason "Stop-HandoffBatch: $Reason"
            $rescueStopped = $true
        } catch {
            Write-Warning "rescue stop failed: $($_.Exception.Message)"
        }
    }
}

# --- stamp the pid file stopped (preserve original fields where present) ---
$stamp = [ordered]@{ pid = $runnerPid; lane = $Lane; batch = $Batch }
if ($info) { foreach ($k in 'started_utc','command','config','poll_seconds','role','exit_when_drained') {
    if ($info.PSObject.Properties.Name -contains $k) { $stamp[$k] = $info.$k } } }
$stamp['state']       = 'stopped'
$stamp['running']     = $false
$stamp['stopped_utc'] = (Get-Date).ToUniversalTime().ToString('o')
$stamp['exit_reason'] = "Stop-HandoffBatch: $Reason"
$stamp | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $pidFile -Encoding UTF8
Write-Host "pid file stamped stopped: $pidFile"

# --- close-out note with task-state counts (best-effort, from the config's state file) ---
$counts = '(state file not resolved)'
if ($info -and $info.config) {
    $cfgFull = Join-Path $repoRoot $info.config
    if (Test-Path -LiteralPath $cfgFull) {
        try {
            $cfg = Get-Content -LiteralPath $cfgFull -Raw | ConvertFrom-Json
            $statePath = Join-Path $repoRoot $cfg.paths.private_state
            if (Test-Path -LiteralPath $statePath) {
                $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                $counts = ($state.tasks.PSObject.Properties.Value | Group-Object status | Sort-Object Name |
                           ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ', '
            }
        } catch { $counts = '(state unreadable)' }
    }
}
$notePath = Join-Path $handoffRoot ("logs/{0}_p{1}_aborted.md" -f $laneLc, $Batch)
@"
# $Lane batch $Batch — manual abort (Stop-HandoffBatch)

- Aborted (UTC): $((Get-Date).ToUniversalTime().ToString('o'))
- Reason: $Reason
- Runner pid killed: $runnerPid (process tree only; no blanket claude.exe/codex kill)
- Monitor stopped: $([bool]$StopMonitor)
- Managed rescue stopped: $rescueStopped (pass -KeepRescue to leave it running)
- Task state at abort: $counts
- pid file: $pidFile (state=stopped)

Non-terminal tasks were left as-is (worker_running / todo). If any should be parked,
edit the state file deliberately; this script does not mutate task states.
"@ | Set-Content -LiteralPath $notePath -Encoding UTF8
Write-Host "close-out note: $notePath"
