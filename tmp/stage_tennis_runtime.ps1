param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [Parameter(Mandatory = $true)]
    [string]$Runtime
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $Runtime) {
    throw "Runtime staging target already exists: $Runtime"
}

$baseId = "84b24640925c2a8213249ca5d6c3b9ce239529ed812d1799a2e2a4847a8e6cef"
$operational = "snapshot-84b24640925c2a8-v1"
$captureId = "a13a3a8c0ec6908cfd31465c0c2ef37ed8d5e736d2b74a2d216ba7ea313cc301"

$copies = @(
    @(
        (Join-Path $Repo "artifacts\current-usopen-2026\$baseId"),
        (Join-Path $Runtime "artifacts\current-usopen-2026\$baseId")
    ),
    @(
        (Join-Path $Repo "artifacts\live-usopen-2026\$operational"),
        (Join-Path $Runtime "artifacts\live-usopen-2026\$operational")
    ),
    @(
        (Join-Path $Repo "artifacts\live-usopen-2026\five-first-courts-2026-08-31-adaptive-v1\source-captures\$captureId"),
        (Join-Path $Runtime "source-captures\$captureId")
    )
)

foreach ($pair in $copies) {
    New-Item -ItemType Directory -Path $pair[1] -Force | Out-Null
    & robocopy $pair[0] $pair[1] /E /B /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "Backup-mode copy failed with robocopy exit code $LASTEXITCODE for $($pair[0])"
    }
}

$escapedRepo = $Repo.Replace("\", "\\")
$escapedRuntime = $Runtime.Replace("\", "\\")
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($tour in @("atp", "wta")) {
    $snapshot = Join-Path $Runtime "artifacts\live-usopen-2026\$operational\model_snapshot_$tour.json"
    $content = [System.IO.File]::ReadAllText($snapshot)
    $updated = $content.Replace($escapedRepo, $escapedRuntime)
    if ($updated -eq $content) {
        throw "No runtime paths replaced in $snapshot"
    }
    [System.IO.File]::WriteAllText($snapshot, $updated, $utf8)
}
