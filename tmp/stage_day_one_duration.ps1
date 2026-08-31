param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [Parameter(Mandatory = $true)]
    [string]$Runtime
)

$ErrorActionPreference = "Stop"
$baseId = "2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664"
$durationPrefix = "2edefbc0b1c8522b"
$durationRun = "4c9d944a03931055df58b5ec8405eb22"
$operational = "day-one-prop-bundle-v1"

$copies = @(
    @(
        (Join-Path $Repo "artifacts\current-usopen-2026\$baseId"),
        (Join-Path $Runtime "artifacts\current-usopen-2026\$baseId")
    ),
    @(
        (Join-Path $Repo "artifacts\duration-usopen-2026\$durationPrefix\$durationRun"),
        (Join-Path $Runtime "artifacts\duration-usopen-2026\$durationPrefix\$durationRun")
    ),
    @(
        (Join-Path $Repo "artifacts\live-usopen-2026\official-2117-v1"),
        (Join-Path $Runtime "artifacts\live-usopen-2026\$operational")
    )
)

foreach ($pair in $copies) {
    New-Item -ItemType Directory -Path $pair[1] -Force | Out-Null
    & robocopy $pair[0] $pair[1] /E /B /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "Backup-mode copy failed with robocopy exit code $LASTEXITCODE for $($pair[0])"
    }
}

$durationRoot = Join-Path $Runtime "artifacts\duration-usopen-2026\$durationPrefix\$durationRun"
$operationalRoot = Join-Path $Runtime "artifacts\live-usopen-2026\$operational"
$escapedRepo = $Repo.Replace("\", "\\")
$escapedRuntime = $Runtime.Replace("\", "\\")
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($tour in @("atp", "wta")) {
    $source = Join-Path $durationRoot "model_snapshot_${tour}_v3.json"
    $target = Join-Path $operationalRoot "model_snapshot_${tour}.json"
    $content = [System.IO.File]::ReadAllText($source)
    $updated = $content.Replace($escapedRepo, $escapedRuntime)
    if ($updated -eq $content) {
        throw "No runtime paths replaced in $source"
    }
    [System.IO.File]::WriteAllText($target, $updated, $utf8)
}
