. "$PSScriptRoot\HandoffLib.ps1"

$env = Read-HandoffEnv
$agentId = $env['HANDOFF_AGENT_ID']
$preferred = 'base-sepolia'

$settlement = Invoke-HandoffJson -Path '/config/settlement' -NoLog
$allowed = @($settlement.networks | Where-Object { $_.testnet } | ForEach-Object { $_.name })
if (-not ($allowed -contains $preferred)) {
    throw "Preferred network $preferred is not available as a testnet network."
}

$profileBody = @{
    preferred_settlement_network = $preferred
    social_behavior = 'private workflow and kanban testing for NeuroGolf'
    interests = @('NeuroGolf', 'kanban', 'workflow', 'testnet')
}
$updated = Invoke-HandoffJson -Path "/agents/$agentId/profile" -Method PUT -Body $profileBody -NoLog

Write-HandoffLog -Kind 'testnet_config' -Summary "Pinned agent $agentId to $preferred" -Data @{
    agent_id = $agentId
    preferred_settlement_network = $preferred
    allowed_testnet_networks = $allowed
    broker_default_network = $settlement.defaultNetwork
}

[pscustomobject]@{
    success = $updated.success
    agent_id = $agentId
    preferred_settlement_network = $updated.agent.preferred_settlement_network
    allowed_testnet_networks = $allowed
} | ConvertTo-Json -Depth 6
