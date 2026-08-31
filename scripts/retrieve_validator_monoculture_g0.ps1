[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$UserName,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$KeyPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RemoteRoot,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DestinationParent
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-SafeIdentityComponent {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ($Value -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
        throw "$Label contains characters that are unsafe for an SSH destination"
    }
}

function Assert-NoReparseAncestor {
    param([Parameter(Mandatory = $true)][System.IO.FileSystemInfo]$Item)

    $cursor = $Item
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing reparse-point path component: $($cursor.FullName)"
        }
        $cursor = $cursor.Parent
    }
}

function Get-VerifiedSha256 {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo]$File)

    if (($File.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to hash reparse-point file: $($File.FullName)"
    }
    return (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}

Assert-SafeIdentityComponent -Value $HostName -Label "HostName"
Assert-SafeIdentityComponent -Value $UserName -Label "UserName"

$RemoteRoot = $RemoteRoot.TrimEnd('/')
if (-not $RemoteRoot.StartsWith('/', [System.StringComparison]::Ordinal)) {
    throw "RemoteRoot must be an absolute POSIX path"
}
if ($RemoteRoot -eq '/') {
    throw "RemoteRoot may not be the remote filesystem root"
}
if ($RemoteRoot -cnotmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteRoot.Contains('//')) {
    throw "RemoteRoot contains unsupported or unsafe characters"
}
if ($RemoteRoot -cmatch '(^|/)\.\.?(/|$)') {
    throw "RemoteRoot may not contain dot or parent-directory segments"
}

$keyItem = Get-Item -LiteralPath $KeyPath -Force
if ($keyItem.PSIsContainer -or
    (($keyItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
    throw "KeyPath must be a regular non-reparse-point file"
}

if (-not [System.IO.Path]::IsPathRooted($DestinationParent)) {
    throw "DestinationParent must be an absolute local path"
}
$destinationItem = Get-Item -LiteralPath $DestinationParent -Force
if (-not $destinationItem.PSIsContainer) {
    throw "DestinationParent must be an existing directory"
}
Assert-NoReparseAncestor -Item $destinationItem
$destinationFull = $destinationItem.FullName.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)

$ssh = Get-Command ssh.exe -CommandType Application
$scp = Get-Command scp.exe -CommandType Application
$remoteIdentity = "$UserName@$HostName"
$remoteManifest = "$RemoteRoot/COMPLETION_MANIFEST.json"

# These are read-only remote checks. Capture the manifest hash before transfer
# so an in-flight or accidental remote change cannot be hidden by a matching
# locally copied manifest.
$sshArguments = @(
    "-i", $keyItem.FullName,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    $remoteIdentity,
    "test", "-d", $RemoteRoot, "-a", "-f", $remoteManifest
)
& $ssh.Source @sshArguments
if ($LASTEXITCODE -ne 0) {
    throw "Remote evidence root or completion manifest is unavailable"
}
$hashArguments = @(
    "-i", $keyItem.FullName,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    $remoteIdentity,
    "sha256sum", "--", $remoteManifest
)
$remoteHashLine = (& $ssh.Source @hashArguments | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $remoteHashLine -cnotmatch '^([0-9a-f]{64})\s+') {
    throw "Could not obtain the remote completion-manifest SHA-256"
}
$remoteManifestSha256 = $Matches[1]

$timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$nonce = [Guid]::NewGuid().ToString('N').Substring(0, 12)
$targetName = "validator_monoculture_g0_${timestamp}_${nonce}"
$retrievalContainer = Join-Path -Path $destinationFull -ChildPath $targetName
$containerFull = [System.IO.Path]::GetFullPath($retrievalContainer)
$destinationPrefix = $destinationFull + [System.IO.Path]::DirectorySeparatorChar
if (-not $containerFull.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Generated retrieval target escapes DestinationParent"
}
if (Test-Path -LiteralPath $containerFull) {
    throw "Refusing existing exact retrieval target: $containerFull"
}

$createdContainer = New-Item -ItemType Directory -Path $containerFull
Assert-NoReparseAncestor -Item $createdContainer

# scp copies the complete remote directory, including dotfiles.  No command in
# this script deletes, renames, changes permissions on, or otherwise mutates the
# remote evidence root.
$remoteSpec = "${remoteIdentity}:$RemoteRoot"
$scpArguments = @(
    "-i", $keyItem.FullName,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-r",
    $remoteSpec,
    $containerFull
)
& $scp.Source @scpArguments
if ($LASTEXITCODE -ne 0) {
    throw "scp retrieval failed; partial local evidence is preserved at $containerFull"
}

$remoteLeaf = ($RemoteRoot -split '/')[-1]
if ([string]::IsNullOrWhiteSpace($remoteLeaf)) {
    throw "Could not derive the remote evidence-root name"
}
$localRoot = Join-Path -Path $containerFull -ChildPath $remoteLeaf
$localRootItem = Get-Item -LiteralPath $localRoot -Force
if (-not $localRootItem.PSIsContainer) {
    throw "scp did not produce the expected local evidence root: $localRoot"
}
Assert-NoReparseAncestor -Item $localRootItem

$manifestPath = Join-Path -Path $localRootItem.FullName -ChildPath 'COMPLETION_MANIFEST.json'
$manifestItem = Get-Item -LiteralPath $manifestPath -Force
if ($manifestItem.PSIsContainer -or
    (($manifestItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
    throw "COMPLETION_MANIFEST.json is not a regular file"
}
$localManifestSha256 = Get-VerifiedSha256 -File $manifestItem
if ($localManifestSha256 -cne $remoteManifestSha256) {
    throw "Completion manifest changed during retrieval"
}
$manifest = Get-Content -LiteralPath $manifestItem.FullName -Raw | ConvertFrom-Json
if ($manifest.kind -ne 'validator_monoculture_g0_orchestration' -or
    $manifest.status -ne 'generation_complete__offline_analysis_pending') {
    throw "Completion manifest kind or status is not a completed validator-monoculture run"
}
if ($null -eq $manifest.artifacts_sha256) {
    throw "Completion manifest does not contain artifacts_sha256"
}
if ([string]$manifest.git_commit -cnotmatch '^[0-9a-f]{40}$' -or
    [string]$manifest.code_tree_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.run_binding_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.evidence_root_sha256 -cnotmatch '^[0-9a-f]{64}$') {
    throw "Completion manifest lacks valid Git, code, run-binding, or evidence commitments"
}

$rootFull = [System.IO.Path]::GetFullPath($localRootItem.FullName).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$rootPrefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
$listedFiles = @{}

foreach ($property in $manifest.artifacts_sha256.PSObject.Properties) {
    $relative = [string]$property.Name
    $expected = ([string]$property.Value).ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($relative) -or
        [System.IO.Path]::IsPathRooted($relative) -or
        $relative.Contains('\') -or
        $relative -cmatch '(^|/)\.\.?(/|$)') {
        throw "Manifest contains an unsafe relative artifact path: $relative"
    }
    if ($expected -cnotmatch '^[0-9a-f]{64}$') {
        throw "Manifest contains an invalid SHA-256 for $relative"
    }
    if ($listedFiles.ContainsKey($relative)) {
        throw "Manifest repeats artifact path: $relative"
    }

    $candidate = $rootFull
    foreach ($segment in ($relative -split '/')) {
        if ([string]::IsNullOrEmpty($segment)) {
            throw "Manifest contains an empty path segment: $relative"
        }
        $candidate = Join-Path -Path $candidate -ChildPath $segment
    }
    $candidateFull = [System.IO.Path]::GetFullPath($candidate)
    if (-not $candidateFull.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest artifact escapes the retrieved root: $relative"
    }
    $artifact = Get-Item -LiteralPath $candidateFull -Force
    if ($artifact.PSIsContainer) {
        throw "Manifest artifact is not a file: $relative"
    }
    $observed = Get-VerifiedSha256 -File $artifact
    if ($observed -cne $expected) {
        throw "SHA-256 mismatch for ${relative}: expected $expected, observed $observed"
    }
    $listedFiles[$relative] = $true
}

if ($listedFiles.Count -eq 0) {
    throw "Completion manifest lists no artifacts"
}

# Reject reparse points and unlisted files anywhere in the retrieved evidence.
foreach ($entry in (Get-ChildItem -LiteralPath $rootFull -Recurse -Force)) {
    if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Retrieved evidence contains a reparse point: $($entry.FullName)"
    }
    if (-not $entry.PSIsContainer -and $entry.FullName -cne $manifestItem.FullName) {
        $relative = $entry.FullName.Substring($rootPrefix.Length).Replace('\', '/')
        if (-not $listedFiles.ContainsKey($relative)) {
            throw "Retrieved evidence contains an unlisted file: $relative"
        }
    }
}

$receipt = [ordered]@{
    kind = 'validator_monoculture_g0_retrieval_receipt'
    local_root = $rootFull
    remote_completion_manifest_sha256 = $remoteManifestSha256
    evidence_root_sha256 = [string]$manifest.evidence_root_sha256
    run_binding_sha256 = [string]$manifest.run_binding_sha256
    code_tree_sha256 = [string]$manifest.code_tree_sha256
    git_commit = [string]$manifest.git_commit
}
$receiptPath = Join-Path -Path $containerFull -ChildPath 'RETRIEVAL_RECEIPT.json'
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
Write-Output $rootFull
