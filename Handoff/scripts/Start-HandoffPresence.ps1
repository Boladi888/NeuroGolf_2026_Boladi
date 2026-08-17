param(
    [int]$PollSeconds = 60,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $PSCommandPath
$handoffRoot = Resolve-Path (Join-Path $scriptDir '..')
$repoRoot = Resolve-Path (Join-Path $handoffRoot '..')
$pythonScript = Join-Path $handoffRoot 'runner\handoff_presence.py'
$privateDir = Join-Path $handoffRoot 'private'
$logsDir = Join-Path $handoffRoot 'logs'
New-Item -ItemType Directory -Force -Path $privateDir, $logsDir | Out-Null

if ($Once) {
    & python $pythonScript --once --poll-seconds $PollSeconds
    exit $LASTEXITCODE
}

$stdoutPath = Join-Path $logsDir 'presence.stdout.log'
$stderrPath = Join-Path $logsDir 'presence.stderr.log'
$argsList = @($pythonScript, '--poll-seconds', [string]$PollSeconds)
$proc = Start-Process -FilePath 'python' -ArgumentList $argsList -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$pidInfo = [ordered]@{
    pid = $proc.Id
    started_utc = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    poll_seconds = $PollSeconds
    script = $pythonScript
    stdout = $stdoutPath
    stderr = $stderrPath
}
$pidInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $privateDir 'presence.pid.json') -Encoding UTF8
$pidInfo | ConvertTo-Json -Depth 5


