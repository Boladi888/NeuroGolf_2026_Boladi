. "$PSScriptRoot\HandoffLib.ps1"

$env = Read-HandoffEnv
$agentId = $env['HANDOFF_AGENT_ID']
$handoffRoot = Get-HandoffRoot
$exportDir = Join-Path $handoffRoot 'export\sanitized'
New-Item -ItemType Directory -Force -Path $exportDir | Out-Null

$me = Invoke-HandoffJson -Path '/auth/me' -NoLog
$jobs = Invoke-HandoffJson -Path '/jobs' -NoLog
$teams = Invoke-HandoffJson -Path '/teams' -NoLog
$settlement = Invoke-HandoffJson -Path '/config/settlement' -NoLog

$projectRows = @()
foreach ($job in @($jobs.jobs)) {
    $req = $job.request
    $tasks = @($job.tasks)
    $involved = $false
    if ($req.title -match 'NeuroGolf|K-NeuroGolf|ARC-AGI|ONNX') { $involved = $true }
    if ($req.requester -eq $agentId) { $involved = $true }
    if (@($tasks | Where-Object { $_.assignee -eq $agentId }).Count -gt 0) { $involved = $true }
    if (-not $involved) { continue }

    $projectRows += [ordered]@{
        id = $req.id
        title = $req.title
        requester = $req.requester
        status = $req.status
        created_at = $req.created_at
        goals = $req.goals
        tasks = $tasks
        phase = $job.phase
    }
}

$teamRows = @()
foreach ($team in @($teams.teams)) {
    if (($projectRows | ForEach-Object { $_.id }) -contains $team.request_id) {
        $teamRows += $team
    }
}

$snapshot = [ordered]@{
    schema = 'neurogolf.handoff.sanitized_snapshot.v1'
    exported_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    testnet_only = $true
    agent_id = $agentId
    user_id = $me.user.id
    settlement = [ordered]@{
        default_network = $settlement.defaultNetwork
        networks = @($settlement.networks | ForEach-Object {
            [ordered]@{ name = $_.name; label = $_.label; chainId = $_.chainId; testnet = $_.testnet }
        })
    }
    projects = $projectRows
    teams = $teamRows
}

$safe = Redact-HandoffValue $snapshot
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$path = Join-Path $exportDir "handoff_snapshot_$stamp.json"
$safe | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $path -Encoding UTF8

Write-HandoffLog -Kind 'export' -Summary 'Wrote sanitized HandOff snapshot' -Data @{
    path = $path
    project_count = @($projectRows).Count
    team_count = @($teamRows).Count
}

[pscustomobject]@{
    success = $true
    path = $path
    project_count = @($projectRows).Count
    team_count = @($teamRows).Count
} | ConvertTo-Json -Depth 4
