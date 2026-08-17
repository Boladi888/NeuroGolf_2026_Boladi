. "$PSScriptRoot\HandoffLib.ps1"

$env = Read-HandoffEnv
$agentId = $env['HANDOFF_AGENT_ID']

$me = Invoke-HandoffJson -Path '/auth/me' -NoLog
$settlement = Invoke-HandoffJson -Path '/config/settlement' -NoLog
$agent = @($me.agents | Where-Object { $_.agent_id -eq $agentId })[0]
if (-not $agent) {
    throw "Authenticated account does not own configured agent: $agentId"
}

$nonTestnet = @($settlement.networks | Where-Object { -not $_.testnet } | ForEach-Object { $_.name })
$ok = $me.success -and $agent -and $agent.preferred_settlement_network -eq 'base-sepolia' -and $nonTestnet.Count -eq 0

$result = [ordered]@{
    success = [bool]$ok
    user_id = $me.user.id
    agent_id = $agentId
    owns_agent = [bool]$agent
    delivery_mode = $agent.delivery_mode
    preferred_settlement_network = $agent.preferred_settlement_network
    broker_default_network = $settlement.defaultNetwork
    broker_networks = @($settlement.networks | ForEach-Object {
        [ordered]@{ name = $_.name; testnet = $_.testnet; chainId = $_.chainId }
    })
    non_testnet_networks_seen = $nonTestnet
}

Write-HandoffLog -Kind 'setup_check' -Summary "Verified HandOff setup for $agentId" -Data $result
$result | ConvertTo-Json -Depth 8
