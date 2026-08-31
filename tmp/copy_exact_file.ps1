param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Path $parent -Force | Out-Null
[System.IO.File]::Copy($Source, $Destination, $true)
if (-not (Test-Path -LiteralPath $Destination)) {
    throw "File copy did not publish $Destination"
}
