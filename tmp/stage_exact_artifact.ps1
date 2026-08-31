param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
& robocopy $Source $Destination /E /B /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -ge 8) {
    throw "Backup-mode copy failed with robocopy exit code $LASTEXITCODE for $Source"
}
