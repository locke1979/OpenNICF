function Invoke-Rollback {
    param([string]$ReleaseName)
    Write-Warning "rolling back deployment"
    Restore-Release -Name $ReleaseName
}

function Invoke-Deploy {
    param([string]$ReleaseName)
    Write-Information "deploying release"
    Publish-Release -Name $ReleaseName
}
