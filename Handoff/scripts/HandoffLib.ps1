$ErrorActionPreference = 'Stop'

function Get-HandoffRoot {
    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir '..')).Path
}

function Get-RepoRoot {
    $handoffRoot = Get-HandoffRoot
    return (Resolve-Path (Join-Path $handoffRoot '..')).Path
}

function Read-HandoffEnv {
    $repoRoot = Get-RepoRoot
    $envPath = Join-Path $repoRoot '.env.handoff'
    if (-not (Test-Path -LiteralPath $envPath)) {
        throw "Missing .env.handoff at $envPath"
    }

    $vals = @{}
    Get-Content -LiteralPath $envPath | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $vals[$matches[1].Trim()] = $matches[2].Trim()
        }
    }

    foreach ($name in @('HANDOFF_BOOTSTRAP_TOKEN', 'HANDOFF_AGENT_ID', 'HANDOFF_MCP_URL')) {
        if (-not $vals.ContainsKey($name) -or -not $vals[$name]) {
            throw "Missing $name in .env.handoff"
        }
    }
    return $vals
}

function Get-HandoffHeaders {
    param([hashtable]$Env)
    return @{ Authorization = "Bearer $($Env['HANDOFF_BOOTSTRAP_TOKEN'])" }
}

function Redact-HandoffValue {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [string]) {
        $s = $Value
        $s = $s -replace 'hot_[A-Za-z0-9_\-]+', 'hot_<redacted>'
        $s = $s -replace 'ak_[A-Za-z0-9_\-]+', 'ak_<redacted>'
        $s = $s -replace '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', '<email_redacted>'
        $s = $s -replace '0x[a-fA-F0-9]{32,}', '0x<redacted>'
        return $s
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $out = [ordered]@{}
        foreach ($k in $Value.Keys) {
            if ([string]$k -match '(token|key|secret|password|email|wallet|address)') {
                $out[$k] = '<redacted>'
            } else {
                $out[$k] = Redact-HandoffValue $Value[$k]
            }
        }
        return $out
    }
    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [string])) {
        $arr = @()
        foreach ($x in $Value) { $arr += Redact-HandoffValue $x }
        return $arr
    }
    if ($Value.PSObject.Properties.Count -gt 0 -and $Value.GetType().Name -ne 'DateTime') {
        $out = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) {
            if ($p.Name -match '(token|key|secret|password|email|wallet|address)') {
                $out[$p.Name] = '<redacted>'
            } else {
                $out[$p.Name] = Redact-HandoffValue $p.Value
            }
        }
        return $out
    }
    return $Value
}

function Write-HandoffLog {
    param(
        [Parameter(Mandatory=$true)][string]$Kind,
        [Parameter(Mandatory=$true)][string]$Summary,
        $Data = $null
    )

    $handoffRoot = Get-HandoffRoot
    $logPath = Join-Path $handoffRoot 'logs\operations.jsonl'
    $entry = [ordered]@{
        ts_utc = (Get-Date).ToUniversalTime().ToString('o')
        kind = $Kind
        summary = $Summary
        data = Redact-HandoffValue $Data
    }
    $json = $entry | ConvertTo-Json -Depth 20 -Compress
    Add-Content -LiteralPath $logPath -Value $json -Encoding UTF8
}

function Invoke-HandoffJson {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [ValidateSet('GET','POST','PUT','DELETE')][string]$Method = 'GET',
        $Body = $null,
        [switch]$NoLog
    )

    $env = Read-HandoffEnv
    $uri = 'https://handoff.socnet.lol/api/v1' + $Path
    $headers = Get-HandoffHeaders -Env $env
    $args = @{
        Uri = $uri
        Method = $Method
        Headers = $headers
        TimeoutSec = 30
    }
    if ($null -ne $Body) {
        $args['ContentType'] = 'application/json'
        if ($Body -is [string]) {
            $args['Body'] = $Body
        } else {
            $args['Body'] = ($Body | ConvertTo-Json -Depth 20 -Compress)
        }
    }
    $result = Invoke-RestMethod @args
    if (-not $NoLog) {
        Write-HandoffLog -Kind 'api_call' -Summary "$Method $Path" -Data @{
            path = $Path
            method = $Method
            result = $result
        }
    }
    return $result
}
