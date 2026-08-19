param(
    [Parameter(Mandatory = $true)][string]$InstanceId,
    [Parameter(Mandatory = $true)][datetime]$HardDeadlineUtc,
    [switch]$Arm
)

$ErrorActionPreference = "Stop"
if (-not $Arm) {
    Write-Output "DRY RUN: would terminate instance $InstanceId at $($HardDeadlineUtc.ToUniversalTime().ToString('o')). Re-run with -Arm."
    exit 0
}
if (-not $env:LAMBDA_API_KEY) {
    throw "Set LAMBDA_API_KEY only on this local machine; never copy it to the GPU instance."
}

$deadline = $HardDeadlineUtc.ToUniversalTime()
while ((Get-Date).ToUniversalTime() -lt $deadline) {
    $remaining = $deadline - (Get-Date).ToUniversalTime()
    Write-Output "Watchdog armed for $InstanceId; $([math]::Round($remaining.TotalMinutes, 1)) minutes remain."
    Start-Sleep -Seconds ([math]::Min(60, [math]::Max(1, $remaining.TotalSeconds)))
}

$headers = @{
    Authorization = "Bearer $env:LAMBDA_API_KEY"
    Accept = "application/json"
}
$body = @{ instance_ids = @($InstanceId) } | ConvertTo-Json -Compress
$request = @{
    Method = "Post"
    Uri = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    Headers = $headers
    ContentType = "application/json"
    Body = $body
    TimeoutSec = 20
}
$terminated = $false
for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        $response = Invoke-RestMethod @request
        $matched = @($response.data.terminated_instances | Where-Object { $_.id -eq $InstanceId })
        if ($matched.Count -eq 0) {
            throw "Termination response did not contain target instance $InstanceId."
        }
        $response | ConvertTo-Json -Depth 8
        $terminated = $true
        break
    }
    catch {
        Write-Warning "Termination attempt $attempt failed: $($_.Exception.Message)"
        if ($attempt -lt 5) { Start-Sleep -Seconds ([math]::Min(30, 2 * $attempt)) }
    }
}
if (-not $terminated) {
    throw "All termination attempts failed. Terminate $InstanceId immediately in the Lambda console."
}

$statusRequest = @{
    Method = "Get"
    Uri = "https://cloud.lambda.ai/api/v1/instances/$InstanceId"
    Headers = $headers
    TimeoutSec = 20
}
for ($poll = 1; $poll -le 12; $poll++) {
    try {
        $statusResponse = Invoke-RestMethod @statusRequest
        $status = $statusResponse.data.status
        Write-Output "Instance $InstanceId status: $status"
        if ($status -in @("terminating", "terminated", "preempted")) { exit 0 }
    }
    catch {
        if ($_.Exception.Response.StatusCode.value__ -eq 404) {
            Write-Output "Instance $InstanceId is no longer retrievable; treating termination as complete."
            exit 0
        }
        Write-Warning "Status poll $poll failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 5
}
throw "Termination was accepted but not confirmed. Check the Lambda console now."
