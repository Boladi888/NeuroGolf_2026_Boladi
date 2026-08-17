[CmdletBinding()]
param(
    [switch]$Once,
    [switch]$Loop,
    [switch]$DryRun,
    [ValidateSet('Codex','Codex2','Claude')]
    [string]$Lane = 'Codex',
    [switch]$Run,
    [switch]$RunCodex,   # back-compat alias: -Lane Codex -Run
    [switch]$RunClaude,  # alias: -Lane Claude -Run
    [switch]$SyncHandoff,
    [switch]$AllowKaggleSubmit,
    [switch]$InitState,
    [string]$ConfigPath,
    [ValidateSet('all','worker','reviewer','captain','orchestrator')]
    [string]$Role = 'all',
    [int]$PollSeconds = 90,
    [string]$Batch = '31',
    [switch]$ExitWhenDrained,
    [int]$MaxConsecutiveFailures = 5
)

$ErrorActionPreference = 'Stop'

if ($RunCodex)  {
    if (-not $PSBoundParameters.ContainsKey('Lane')) { $Lane = 'Codex' }
    $Run = $true
}
if ($RunClaude) { $Lane = 'Claude'; $Run = $true }
$realRun = $Run -and -not $DryRun

$scriptDir = Split-Path -Parent $PSCommandPath
$handoffRoot = Resolve-Path (Join-Path $scriptDir '..')
$repoRoot = Resolve-Path (Join-Path $handoffRoot '..')
$privateDir = Join-Path $handoffRoot 'private'

if ($Lane -eq 'Claude') {
    $runner = Join-Path $handoffRoot 'runner\handoff_runner_claude.py'
    $realFlag = '--run-claude'
} else {
    $runner = Join-Path $handoffRoot 'runner\handoff_runner.py'
    $realFlag = '--run-codex'
}
$pidFile = Join-Path $privateDir ("runner_loop.{0}.pid.json" -f $Lane.ToLower())
if (-not $env:NEUROGOLF_HEAVY_GATE_N) {
    $env:NEUROGOLF_HEAVY_GATE_N = '8000'
}
if (-not $env:NEUROGOLF_HEAVY_GATE_K) {
    $env:NEUROGOLF_HEAVY_GATE_K = '5'
}

# Owner standing rule: the Claude lane runs on SUBSCRIPTION auth, never the API
# key (it hard-failed on low balance and violates the cost rule). Strip the key
# from this process env so every spawned `claude -p` worker/reviewer inherits a
# key-less env. Codex lane uses its own ChatGPT-account auth and is untouched.
if ($Lane -eq 'Claude') {
    Remove-Item Env:\ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing runner script at $runner"
}
New-Item -ItemType Directory -Force -Path $privateDir | Out-Null

if (-not $Once -and -not $Loop) {
    $Once = $true
}

function Resolve-ConfigFullPath {
    $cfgPath = $ConfigPath
    if (-not $cfgPath) {
        $cfgPath = if ($Lane -eq 'Claude') { 'Handoff/config/runner_claude.json' } else { 'Handoff/config/runner.json' }
    }
    if ([System.IO.Path]::IsPathRooted($cfgPath)) {
        $full = $cfgPath
    } else {
        $full = Join-Path $repoRoot $cfgPath
    }
    try {
        if (Test-Path -LiteralPath $full) { return (Resolve-Path -LiteralPath $full).Path }
        return [System.IO.Path]::GetFullPath($full)
    } catch {
        return [string]$full
    }
}

function Test-PidAlive {
    param([object]$PidValue)
    try {
        if (-not $PidValue) { return $false }
        $null = Get-Process -Id ([int]$PidValue) -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Get-PidCommandLine {
    param([object]$PidValue)
    try {
        if (-not $PidValue) { return '' }
        $p = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f ([int]$PidValue)) -ErrorAction Stop
        return [string]$p.CommandLine
    } catch {
        return ''
    }
}

function Assert-SingleRunner {
    if (-not $realRun -or -not (Test-Path -LiteralPath $pidFile)) { return }
    try {
        $existing = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    } catch {
        Write-Warning "Existing runner pid file is unreadable; treating as stale: $pidFile"
        return
    }
    if (-not $existing.running -or [string]$existing.state -ne 'running') { return }
    $alive = Test-PidAlive $existing.pid
    $cmd = Get-PidCommandLine $existing.pid
    if (-not $alive -or $cmd -notmatch 'Start-HandoffRunner\.ps1') {
        Write-Warning "Recovering stale runner pid file for $Lane batch $Batch (pid=$($existing.pid), alive=$alive)."
        return
    }

    $requestedConfig = Resolve-ConfigFullPath
    $existingConfig = [string]$existing.config_full_path
    if (-not $existingConfig -and $existing.config) {
        try {
            $existingConfig = if ([System.IO.Path]::IsPathRooted([string]$existing.config)) {
                [System.IO.Path]::GetFullPath([string]$existing.config)
            } else {
                [System.IO.Path]::GetFullPath((Join-Path $repoRoot ([string]$existing.config)))
            }
        } catch {
            $existingConfig = [string]$existing.config
        }
    }
    $sameLane = ([string]$existing.lane -eq [string]$Lane)
    $sameBatch = ([string]$existing.batch -eq [string]$Batch)
    $sameConfig = ([string]$existingConfig -eq [string]$requestedConfig)
    if ($sameLane -and $sameBatch -and $sameConfig) {
        throw "Runner already active for lane=$Lane batch=$Batch config=$requestedConfig pid=$($existing.pid). Refusing duplicate start."
    }
    throw "Runner already active for lane=$($existing.lane) batch=$($existing.batch) config=$existingConfig pid=$($existing.pid). Refusing to overwrite lane pid file for requested lane=$Lane batch=$Batch config=$requestedConfig."
}

function Invoke-RunnerOnce {
    $runnerArgs = @($runner, '--role', $Role)
    if ($ConfigPath) { $runnerArgs += @('--config', $ConfigPath) }
    if (-not $realRun) { $runnerArgs += '--dry-run' }
    if ($realRun) { $runnerArgs += $realFlag }
    if ($SyncHandoff) { $runnerArgs += '--sync-handoff' }
    if ($AllowKaggleSubmit) { $runnerArgs += '--allow-kaggle-submit' }
    if ($InitState) { $runnerArgs += '--init-state' }

    Push-Location $repoRoot
    try {
        & python @runnerArgs
    } finally {
        Pop-Location
    }
}

function Get-RunnerStatePath {
    $cfgPath = $ConfigPath
    if (-not $cfgPath) {
        $cfgPath = if ($Lane -eq 'Claude') { 'Handoff/config/runner_claude.json' } else { 'Handoff/config/runner.json' }
    }
    $cfgFull = Join-Path $repoRoot $cfgPath
    if (-not (Test-Path -LiteralPath $cfgFull)) { return $null }
    $cfg = Get-Content -LiteralPath $cfgFull -Raw | ConvertFrom-Json
    if (-not $cfg.paths.private_state) { return $null }
    return Join-Path $repoRoot $cfg.paths.private_state
}

function Test-BatchDrained {
    $statePath = Get-RunnerStatePath
    if (-not $statePath -or -not (Test-Path -LiteralPath $statePath)) { return $false }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if (-not $state.tasks) { return $false }
    $cfgPath = $ConfigPath
    if (-not $cfgPath) {
        $cfgPath = if ($Lane -eq 'Claude') { 'Handoff/config/runner_claude.json' } else { 'Handoff/config/runner.json' }
    }
    $cfgFull = Join-Path $repoRoot $cfgPath
    if (Test-Path -LiteralPath $cfgFull) {
        $cfg = Get-Content -LiteralPath $cfgFull -Raw | ConvertFrom-Json
        $configuredTasks = @($cfg.tasks | ForEach-Object { [string]$_.task })
        $stateTaskNames = @($state.tasks.PSObject.Properties.Name)
        foreach ($taskName in $configuredTasks) {
            if ($stateTaskNames -notcontains $taskName) { return $false }
        }
    }
    $terminal = @(
        'ship_packeted',
        'captain_staged',
        'drop',
        'worker_failed',
        'review_failed',
        'review_parse_failed',
        'needs_packet_fix'
    )
    $tasks = @($state.tasks.PSObject.Properties.Value)
    if ($tasks.Count -eq 0) { return $false }
    foreach ($task in $tasks) {
        if ($terminal -notcontains [string]$task.status) { return $false }
    }
    return $true
}

# Records loop lifecycle to the lane's pid file the monitor reads, so a crashed/
# stopped loop is detectable (the old loop wrote nothing on exit -> stale
# "running"). Lane-specific so Codex + Claude loops can run concurrently.
function Write-RunnerPid {
    param([string]$State, [string]$Reason = '')
    $info = [ordered]@{
        pid          = $PID
        lane         = $Lane
        started_utc  = $script:StartedUtc
        command      = $MyInvocation.Line
        batch        = $Batch
        role         = $Role
        config       = $ConfigPath
        config_full_path = (Resolve-ConfigFullPath)
        poll_seconds = $PollSeconds
        exit_when_drained = [bool]$ExitWhenDrained
        state        = $State
        running      = ($State -eq 'running')
    }
    if ($State -ne 'running') {
        $info['stopped_utc'] = (Get-Date).ToUniversalTime().ToString('o')
        $info['exit_reason'] = $Reason
    }
    $info | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $pidFile -Encoding UTF8
}

$script:StartedUtc = (Get-Date).ToUniversalTime().ToString('o')
Assert-SingleRunner

if ($Once) {
    Invoke-RunnerOnce
    exit $LASTEXITCODE
}

Write-RunnerPid -State 'running'
$exitReason = 'stopped'
$consecutiveFailures = 0
try {
    while ($true) {
        Invoke-RunnerOnce
        if ($LASTEXITCODE -ne 0) {
            $consecutiveFailures++
            Write-Warning "Runner ($Lane) exited $LASTEXITCODE (consecutive failure $consecutiveFailures/$MaxConsecutiveFailures)"
            if ($consecutiveFailures -ge $MaxConsecutiveFailures) {
                # A single transient failure no longer kills the loop; only a
                # persistent run of failures does.
                $exitReason = "too_many_failures(exit=$LASTEXITCODE)"
                exit $LASTEXITCODE
            }
            Start-Sleep -Seconds ([Math]::Min($PollSeconds * $consecutiveFailures, 600))
            continue
        }
        $consecutiveFailures = 0
        if ($ExitWhenDrained -and (Test-BatchDrained)) {
            $exitReason = 'complete'
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    Write-RunnerPid -State 'stopped' -Reason $exitReason
}


