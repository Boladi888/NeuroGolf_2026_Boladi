[CmdletBinding()]
param(
    [ValidateSet('Codex','Codex2','Claude')]
    [string]$Lane = 'Codex',
    [string]$Batch = '',
    [string]$ConfigPath = '',
    [string[]]$Task = @(),
    [int]$HeavyN = 8000,
    [int]$CappedN = 5000,
    [switch]$Foreground,
    [switch]$ForceRestart
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $PSCommandPath
$handoffRoot = Resolve-Path (Join-Path $scriptDir '..')
$repoRoot = Resolve-Path (Join-Path $handoffRoot '..')
$laneLc = $Lane.ToLower()
$privateDir = Join-Path $handoffRoot 'private'
$logDir = Join-Path $handoffRoot 'logs'
New-Item -ItemType Directory -Force -Path $privateDir, $logDir | Out-Null

if (-not $Batch -and -not $ConfigPath) {
    throw 'Pass -Batch or -ConfigPath so the rescue controller can be scoped.'
}
if (-not $ConfigPath) {
    $configStem = if ($Lane -eq 'Codex2') { 'runner_codex2' } elseif ($Lane -eq 'Codex') { 'runner_codex' } else { 'runner_claude' }
    $ConfigPath = "Handoff/config/${configStem}_p${Batch}.json"
}
$configFull = Resolve-Path (Join-Path $repoRoot $ConfigPath)
if (-not $Batch) {
    if ($ConfigPath -match '_p(\d+)\.json$') { $Batch = $Matches[1] }
    else { throw 'Could not infer -Batch from -ConfigPath; pass -Batch explicitly.' }
}

$pidFile = Join-Path $privateDir ("rescue_verify_{0}_p{1}.pid.json" -f $laneLc, $Batch)

function Stop-ExistingRescue {
    if (-not (Test-Path -LiteralPath $pidFile)) { return }
    $info = $null
    try { $info = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json } catch { return }
    if (-not $info -or -not $info.pid) { return }
    if (($info.PSObject.Properties.Name -contains 'running' -and -not [bool]$info.running) -or [string]$info.state -eq 'stopped') {
        return
    }
    $pid = [int]$info.pid
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pid" -ErrorAction SilentlyContinue
    if (-not $proc) { return }
    if ($proc.CommandLine -notmatch 'rescue_verify\.py|Start-HandoffRescue') {
        throw "Refusing to touch pid=$pid because it is not a rescue verifier: $($proc.CommandLine)"
    }
    if (-not $ForceRestart) {
        throw "Managed rescue verifier already appears live at pid=$pid. Re-run with -ForceRestart to replace it."
    }
    & taskkill /PID $pid /T /F | Out-Host
}

Stop-ExistingRescue

if (-not $Task -or $Task.Count -eq 0) {
    $cfg = Get-Content -LiteralPath $configFull -Raw | ConvertFrom-Json
    $Task = @($cfg.tasks | ForEach-Object { [string]$_.task })
}
$Task = @($Task | Where-Object { $_ } | ForEach-Object {
    $t = [string]$_
    if ($t -match '^task') { $t } else { 'task' + ([int]$t).ToString('000') }
} | Sort-Object -Unique)
if ($Task.Count -eq 0) {
    throw 'No tasks selected for rescue verification.'
}

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$prefix = "rescue_verify_${laneLc}_p${Batch}_${stamp}"
$report = "Handoff/logs/${prefix}.jsonl"
$plan = "Handoff/logs/${prefix}.plan.json"
$stdout = Join-Path $logDir "${prefix}.stdout.log"
$stderr = Join-Path $logDir "${prefix}.stderr.log"

$rescueArgs = @(
    'Agenters/Submit_Captain/rescue_verify.py',
    '--verify',
    '--write-packets',
    '--heavy-n', [string]$HeavyN,
    '--capped-n', [string]$CappedN,
    '--report', $report,
    '--plan-json', $plan,
    '--pid-file', ("Handoff/private/rescue_verify_{0}_p{1}.pid.json" -f $laneLc, $Batch),
    '--pid-lane', $Lane,
    '--batch', $Batch
)
foreach ($t in $Task) { $rescueArgs += @('--task', $t) }

function Quote-PSArg([string]$Value) {
    return "'" + ($Value -replace "'", "''") + "'"
}

if ($Foreground) {
    Push-Location $repoRoot
    try {
        & python @rescueArgs
        if ($LASTEXITCODE -ne 0) {
            throw "rescue verifier exited with code $LASTEXITCODE"
        }
        return
    } finally {
        Pop-Location
    }
}

$argText = ($rescueArgs | ForEach-Object { Quote-PSArg $_ }) -join ' '
$cmd = @"
`$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath '$(($repoRoot.Path) -replace "'", "''")'
& python $argText
exit `$LASTEXITCODE
"@

$proc = Start-Process -FilePath powershell.exe `
    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[ordered]@{
    lane = $Lane
    batch = $Batch
    wrapper_pid = $proc.Id
    pid_file = $pidFile
    task_count = $Task.Count
    heavy_n = $HeavyN
    capped_n = $CappedN
    report = $report
    plan = $plan
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json -Depth 5
