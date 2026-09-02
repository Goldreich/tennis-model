$ErrorActionPreference = "Stop"

$ledger = (Get-Content -Raw -LiteralPath "tmp/audit_forecast_ledger.json" |
    ConvertFrom-Json).matches
$wanted = @{}
foreach ($forecast in $ledger) {
    $wanted[[string] $forecast.official_match_id] = $forecast
}

$reportFiles = @(
    Get-ChildItem -LiteralPath "C:/Users/orgol/AppData/Local/tm-usopen/outputs" `
        -Recurse -File -Filter "batch-report-*.json" -ErrorAction SilentlyContinue
)
$candidates = @()
$failures = @()
foreach ($file in $reportFiles) {
    try {
        $report = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
    } catch {
        $failures += $file.FullName
        continue
    }
    foreach ($match in @($report.matches)) {
        $officialId = [string] $match.official_match_id
        if (-not $wanted.ContainsKey($officialId)) {
            continue
        }
        $candidates += [pscustomobject] @{
            id = $officialId
            match = [string] $match.match
            paths = [int] $match.paths
            cutoff = [datetime] $report.batch_information_cutoff_utc
            report_path = $file.FullName
            match_record = $match
        }
    }
}

$selected = @()
foreach ($group in ($candidates | Group-Object id)) {
    $forecast = $wanted[$group.Name]
    $target = ($forecast.win_target -replace "[^a-zA-Z-]", "").ToLowerInvariant()
    $compatible = @(
        $group.Group | Where-Object {
            (($_.match -replace "[^a-zA-Z-]", "").ToLowerInvariant()).Contains($target)
        }
    )
    $pool = if ($compatible.Count -gt 0) { $compatible } else { @($group.Group) }
    $selected += @($pool | Sort-Object cutoff, paths | Select-Object -Last 1)
}

$rows = @()
foreach ($item in $selected) {
    $match = $item.match_record
    $lockPath = Join-Path (Split-Path -Parent $match.card_path) "lock.json"
    $lock = $null
    if (Test-Path -LiteralPath $lockPath) {
        try {
            $lock = (Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json).lock
        } catch {
            $lock = $null
        }
    }
    $rows += [pscustomobject] [ordered] @{
        official_match_id = $item.id
        match = $match.match
        paths = $match.paths
        cutoff = $item.cutoff.ToString("o")
        report_path = $item.report_path
        lock_path = $lockPath
        warnings = @($match.warnings)
        players = @($match.players)
        match_win_probability = $match.match_win_probability
        parameter_summaries = if ($null -ne $lock) {
            @($lock.parameter_summaries)
        } else {
            @()
        }
    }
}

[pscustomobject] @{
    generated_at_utc = [datetime]::UtcNow.ToString("o")
    report_files_scanned = $reportFiles.Count
    parse_failures = $failures
    selected = $rows
} | ConvertTo-Json -Depth 14 |
    Set-Content -LiteralPath "tmp/audit_retained_lock_summaries.json" -Encoding utf8

[pscustomobject] @{
    report_files_scanned = $reportFiles.Count
    candidates = $candidates.Count
    selected = $rows.Count
    with_parameter_summaries = @(
        $rows | Where-Object { $_.parameter_summaries.Count -eq 2 }
    ).Count
    parse_failures = $failures.Count
    rows = @(
        $rows | ForEach-Object {
            [pscustomobject] @{
                id = $_.official_match_id
                match = $_.match
                paths = $_.paths
                warnings = $_.warnings
                directions = @(
                    $_.parameter_summaries | ForEach-Object {
                        $direction = $_
                        [pscustomobject] @{
                            server_id = $direction.server_id
                            receiver_id = $direction.receiver_id
                            service_point_win = $direction.service_point_win
                            hold = $direction.analytic_hold_probability
                            F = ($direction.primitives | Where-Object component -eq "F").map_mean
                            A = ($direction.primitives | Where-Object component -eq "A").map_mean
                            Q1 = ($direction.primitives | Where-Object component -eq "Q1").map_mean
                            D = ($direction.primitives | Where-Object component -eq "D").map_mean
                            Q2 = ($direction.primitives | Where-Object component -eq "Q2").map_mean
                            Q1_kappa = (
                                $direction.primitives | Where-Object component -eq "Q1"
                            ).predictive_concentration
                            Q2_kappa = (
                                $direction.primitives | Where-Object component -eq "Q2"
                            ).predictive_concentration
                        }
                    }
                )
            }
        }
    )
} | ConvertTo-Json -Depth 9 -Compress
