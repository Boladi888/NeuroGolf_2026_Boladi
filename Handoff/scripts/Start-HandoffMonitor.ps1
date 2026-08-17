[CmdletBinding()]
param(
    [switch]$Once,
    [switch]$Loop,
    [int]$IntervalMinutes = 40,
    [string]$Batch = '31',
    [ValidateSet('Codex','Codex2','Claude')]
    [string]$Lane = 'Codex',
    [string]$ConfigPath,
    [string]$OperationLog,
    [string]$StatePath,
    [switch]$ExitOnTerminal,
    [switch]$CheckBoard
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $scriptDir '..\..')
$monitor = Join-Path $repoRoot 'Handoff\runner\handoff_monitor.py'
$laneLc = $Lane.ToLower()

if (-not $ConfigPath) {
    $ConfigPath = if ($Lane -eq 'Claude') { 'Handoff/config/runner_claude.json' } else { 'Handoff/config/runner.json' }
}

# Default the watched state file to whatever the lane's runner config points at,
# so the monitor follows the ACTIVE batch instead of a hard-coded path.
if (-not $StatePath) {
    $cfgFull = Join-Path $repoRoot $ConfigPath
    if (Test-Path -LiteralPath $cfgFull) {
        $cfg = Get-Content -LiteralPath $cfgFull -Raw | ConvertFrom-Json
        $StatePath = $cfg.paths.private_state
        if (-not $OperationLog -and $cfg.paths.operation_log) { $OperationLog = $cfg.paths.operation_log }
    }
}

$pidFile = "Handoff/private/runner_loop.$laneLc.pid.json"
$argsList = @(
    $monitor,
    '--batch', $Batch,
    '--lane', $Lane,
    '--runner-pid-file', $pidFile,
    '--interval-seconds', ($IntervalMinutes * 60)
)
if ($StatePath) {
    $argsList += @(
        '--state', $StatePath,
        '--log', "Handoff/logs/monitor_${laneLc}_batch$Batch.jsonl",
        '--latest', "Handoff/logs/monitor_${laneLc}_batch$Batch.latest.txt"
    )
}
if ($OperationLog) { $argsList += @('--operation-log', $OperationLog) }
if ($ExitOnTerminal) { $argsList += '--exit-on-terminal' }
# Pass the lane config so --check-board can resolve the board project id from
# its project block (the monitor has no other source for it).
if ($CheckBoard) { $argsList += @('--check-board', '--config', $ConfigPath) }
if ($Loop) { $argsList += '--loop' }

Push-Location $repoRoot
try {
    & python @argsList
} finally {
    Pop-Location
}


