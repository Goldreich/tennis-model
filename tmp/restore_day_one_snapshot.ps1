$ErrorActionPreference = 'Stop'
$transcript = 'C:\Users\orgol\OneDrive\Documents\Independent-Research\tennis-model\tmp\restore-day-one.transcript.txt'
Start-Transcript -LiteralPath $transcript -Force | Out-Null

$snapshotId = '2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664'
$artifactId = 'e1320c41fee70190e2e4f52c99fd6d77'
$repo = 'C:\Users\orgol\OneDrive\Documents\Independent-Research\tennis-model'
$existingRuntime = 'C:\Users\orgol\AppData\Local\tennis-model-runtime\usopen-2026-08-31-five-v3'
$runtime = 'C:\Users\orgol\AppData\Local\tm-usopen'
$sourceRoot = Join-Path $repo "artifacts\current-usopen-2026\$snapshotId"
$destinationRoot = Join-Path $runtime "artifacts\current-usopen-2026\$snapshotId"
$relativeArtifact = "retirement_fits\atp\20260830T121353Z\$artifactId\retirement-fit.json"
$sourceArtifact = Join-Path $sourceRoot $relativeArtifact
$destinationArtifact = Join-Path $destinationRoot $relativeArtifact
$officialSource = Join-Path $repo 'artifacts\live-usopen-2026\official-2117-v1'
$officialDestination = Join-Path $runtime 'artifacts\live-usopen-2026\official-2117-duration-v1'

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Day-one snapshot directory does not exist: $sourceRoot"
}

[System.IO.Directory]::CreateDirectory($runtime) | Out-Null
& "$env:SystemRoot\System32\robocopy.exe" $existingRuntime $runtime /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "runtime staging failed with exit code $LASTEXITCODE"
}

$account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
& "$env:SystemRoot\System32\takeown.exe" /F $sourceRoot /R /D Y | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "takeown failed with exit code $LASTEXITCODE"
}

& "$env:SystemRoot\System32\icacls.exe" $sourceRoot /grant:r "${account}:(OI)(CI)F" /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "icacls failed with exit code $LASTEXITCODE"
}

& "$env:SystemRoot\System32\takeown.exe" /F $officialSource /R /D Y | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "official bundle takeown failed with exit code $LASTEXITCODE"
}
& "$env:SystemRoot\System32\icacls.exe" $officialSource /grant:r "${account}:(OI)(CI)F" /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "official bundle icacls failed with exit code $LASTEXITCODE"
}
[System.IO.Directory]::CreateDirectory($officialDestination) | Out-Null
& "$env:SystemRoot\System32\robocopy.exe" $officialSource $officialDestination /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "official bundle staging failed with exit code $LASTEXITCODE"
}

[System.IO.Directory]::CreateDirectory($destinationRoot) | Out-Null
& "$env:SystemRoot\System32\robocopy.exe" $sourceRoot $destinationRoot /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $sourceArtifact)) {
    throw "Required source retirement artifact is still unavailable: $sourceArtifact"
}
[System.IO.Directory]::CreateDirectory((Split-Path -Parent $destinationArtifact)) | Out-Null
$longSourceArtifact = '\\?\' + $sourceArtifact
[System.IO.File]::Copy($longSourceArtifact, $destinationArtifact, $true)
if (-not (Test-Path -LiteralPath $destinationArtifact)) {
    throw "Required staged retirement artifact is missing: $destinationArtifact"
}

$sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $longSourceArtifact).Hash
$destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationArtifact).Hash
if ($sourceHash -ne $destinationHash) {
    throw 'Staged retirement artifact failed SHA-256 verification.'
}

$legacyDestinationArtifact = Join-Path $existingRuntime "artifacts\current-usopen-2026\$snapshotId\$relativeArtifact"
$longLegacyDestinationArtifact = '\\?\' + $legacyDestinationArtifact
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($longLegacyDestinationArtifact)) | Out-Null
[System.IO.File]::Copy($destinationArtifact, $longLegacyDestinationArtifact, $true)
$legacyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $longLegacyDestinationArtifact).Hash
if ($sourceHash -ne $legacyHash) {
    throw 'Absolute-reference retirement artifact failed SHA-256 verification.'
}

Write-Output "RESTORED $destinationArtifact $destinationHash"
