param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [Parameter(Mandatory = $true)]
    [string]$Runtime
)

$ErrorActionPreference = "Stop"

foreach ($relative in @("data\raw", "data\processed")) {
    $source = Join-Path $Repo $relative
    if (-not (Test-Path -LiteralPath $source)) {
        continue
    }
    $destination = Join-Path $Runtime $relative
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    & robocopy $source $destination /E /B /COPY:DAT /DCOPY:DAT /R:1 /W:1 /XJ /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "Backup-mode copy failed with robocopy exit code $LASTEXITCODE for $source"
    }
}
