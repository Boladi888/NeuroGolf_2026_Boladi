# Posts the consolidated runner/platform feedback as a TASK on the HandOff "handoff" project.
# You must authenticate as the project REQUESTER (handoff-claude) — boladi_a is not authorized (403).
# The secret is read from an env var you set; it is never printed.
#
# Usage (PowerShell), set ONE of:
#   $env:HANDOFF_CLAUDE_BEARER    = '<handoff-claude bearer token>'
#   # or
#   $env:HANDOFF_CLAUDE_AGENT_KEY = '<handoff-claude X-Agent-Key>'
# then:  pwsh -File Handoff/_proposals/post_feedback_task.ps1

$ErrorActionPreference = 'Stop'
$proj = '963632e8-7ddd-4236-ae71-7b566302b59b'
$base = 'https://handoff.socnet.lol/api/v1'
$docPath = Join-Path $PSScriptRoot 'combined_handoff_platform_feedback_2026-07-02.md'
$doc = Get-Content -Raw -LiteralPath $docPath

$headers = @{ 'Content-Type' = 'application/json' }
if ($env:HANDOFF_CLAUDE_BEARER)    { $headers['Authorization'] = "Bearer $($env:HANDOFF_CLAUDE_BEARER)" }
if ($env:HANDOFF_CLAUDE_AGENT_KEY) { $headers['X-Agent-Key'] = $env:HANDOFF_CLAUDE_AGENT_KEY; $headers['X-Agent-Id'] = 'handoff-claude' }
if (-not $headers.ContainsKey('Authorization') -and -not $headers.ContainsKey('X-Agent-Key')) {
    throw 'Set $env:HANDOFF_CLAUDE_BEARER or $env:HANDOFF_CLAUDE_AGENT_KEY first (handoff-claude requester auth).'
}

# 1) create-or-resolve a feedback goal (idempotent via external_key)
$goalBody = @{ title = 'Runner/platform feedback - NeuroGolf lanes'; external_key = 'handoff:feedback:neurogolf-runner-lanes' } | ConvertTo-Json
$goal = Invoke-RestMethod -Method Post -Uri "$base/requests/$proj/goals" -Headers $headers -Body $goalBody
$goalId = if ($goal.goal_id) { $goal.goal_id } elseif ($goal.id) { $goal.id } else { $goal.goal.id }
Write-Host "goal_id = $goalId"

# 2) create-or-resolve the task under that goal, with the combined doc as the prompt/description
$taskBody = @{
    title        = 'Consolidated HandOff runner/platform feedback (Codex/Codex2 lanes p38/p39/p42/p43)'
    goal_id      = $goalId
    external_key = 'handoff:feedback:neurogolf-runner-lanes:consolidated'
    prompt       = $doc
    description  = $doc
} | ConvertTo-Json -Depth 6
$task = Invoke-RestMethod -Method Post -Uri "$base/requests/$proj/tasks" -Headers $headers -Body $taskBody
$taskId = if ($task.id) { $task.id } elseif ($task.task_id) { $task.task_id } else { $task.task.id }
Write-Host "CREATED task_id = $taskId  (goal $goalId, project $proj)"
