param(
    [Parameter(Mandatory=$true)][string]$Kind,
    [Parameter(Mandatory=$true)][string]$Summary,
    [string]$ProjectId = '',
    [string]$TaskId = '',
    [string]$DataJson = '{}'
)

. "$PSScriptRoot\HandoffLib.ps1"

$data = [ordered]@{
    project_id = $ProjectId
    task_id = $TaskId
}
if ($DataJson -and $DataJson.Trim() -ne '{}') {
    $data['extra'] = $DataJson | ConvertFrom-Json
}

Write-HandoffLog -Kind $Kind -Summary $Summary -Data $data
[pscustomobject]@{ success = $true; kind = $Kind; summary = $Summary } | ConvertTo-Json -Depth 4
