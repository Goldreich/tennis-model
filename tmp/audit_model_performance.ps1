$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Normalize-Name([string] $Value) {
    if ($null -eq $Value) {
        return ""
    }
    $decomposed = $Value.Normalize([Text.NormalizationForm]::FormD)
    $characters = @(
        $decomposed.ToCharArray() | Where-Object {
            [Globalization.CharUnicodeInfo]::GetUnicodeCategory($_) -ne
                [Globalization.UnicodeCategory]::NonSpacingMark
        }
    )
    return ((-join $characters).ToLowerInvariant() -replace "[^a-z0-9]+", " ").Trim()
}

function Resolve-TargetSide($Target, $Match) {
    $targetName = Normalize-Name $Target
    $team1Name = Normalize-Name (
        "$($Match.team1.firstNameA) $($Match.team1.lastNameA) $($Match.team1.displayNameA)"
    )
    $team2Name = Normalize-Name (
        "$($Match.team2.firstNameA) $($Match.team2.lastNameA) $($Match.team2.displayNameA)"
    )
    $isTeam1 = $team1Name.Contains($targetName)
    $isTeam2 = $team2Name.Contains($targetName)
    if ($isTeam1 -and -not $isTeam2) {
        return 1
    }
    if ($isTeam2 -and -not $isTeam1) {
        return 2
    }
    throw "Could not map $Target in $($Match.team1.displayNameA) v $($Match.team2.displayNameA)"
}

function Get-Mean($Values) {
    $items = @($Values)
    if ($items.Count -eq 0) {
        return $null
    }
    return [double] (($items | Measure-Object -Average).Average)
}

function Get-Summary($Rows) {
    $items = @($Rows)
    if ($items.Count -eq 0) {
        return $null
    }
    $briers = @($items | ForEach-Object { [math]::Pow($_.probability - $_.outcome, 2) })
    $logLosses = @(
        $items | ForEach-Object {
            $probability = [math]::Min(0.999999, [math]::Max(0.000001, $_.probability))
            -($_.outcome * [math]::Log($probability) +
                (1 - $_.outcome) * [math]::Log(1 - $probability))
        }
    )
    $outcomeProbabilities = @(
        $items | ForEach-Object {
            if ($_.outcome -eq 1) { $_.probability } else { 1 - $_.probability }
        }
    )
    $decisive = @($items | Where-Object { $_.probability -ne 0.5 })
    $correct = @(
        $decisive | Where-Object {
            (($_.probability -gt 0.5) -and ($_.outcome -eq 1)) -or
            (($_.probability -lt 0.5) -and ($_.outcome -eq 0))
        }
    ).Count
    $meanBrier = Get-Mean $briers
    $brierVariance = if ($items.Count -gt 1) {
        [double] ((
            $briers | ForEach-Object { [math]::Pow($_ - $meanBrier, 2) } |
                Measure-Object -Sum
        ).Sum) / ($items.Count - 1)
    } else {
        0.0
    }
    $meanProbability = Get-Mean @($items.probability)
    $observedRate = Get-Mean @($items.outcome)
    return [pscustomobject] [ordered] @{
        n = $items.Count
        mean_probability = [math]::Round($meanProbability, 4)
        observed_yes_rate = [math]::Round($observedRate, 4)
        calibration_bias = [math]::Round($meanProbability - $observedRate, 4)
        brier = [math]::Round($meanBrier, 4)
        brier_se = [math]::Round([math]::Sqrt($brierVariance / $items.Count), 4)
        brier_skill_vs_50 = [math]::Round(1 - $meanBrier / 0.25, 4)
        log_loss = [math]::Round((Get-Mean $logLosses), 4)
        mean_probability_assigned_to_outcome = [math]::Round(
            (Get-Mean $outcomeProbabilities), 4
        )
        decisive_n = $decisive.Count
        decisive_accuracy = if ($decisive.Count -gt 0) {
            [math]::Round($correct / $decisive.Count, 4)
        } else {
            $null
        }
    }
}

function Get-DifferenceSummary($Values) {
    $items = @($Values)
    $mean = Get-Mean $items
    $variance = [double] ((
        $items | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum
    ).Sum) / ($items.Count - 1)
    $standardError = [math]::Sqrt($variance / $items.Count)
    return [pscustomobject] [ordered] @{
        n = $items.Count
        mean = [math]::Round($mean, 4)
        se = [math]::Round($standardError, 4)
        ci95_low = [math]::Round($mean - 1.96 * $standardError, 4)
        ci95_high = [math]::Round($mean + 1.96 * $standardError, 4)
    }
}

$ledger = (Get-Content -Raw -LiteralPath "tmp/audit_forecast_ledger.json" |
    ConvertFrom-Json).matches
$scored = @()
$excluded = @()

foreach ($forecast in $ledger) {
    $resultPath = "tmp/audit_official_results/$($forecast.official_match_id).json"
    if (-not (Test-Path -LiteralPath $resultPath)) {
        $excluded += [pscustomobject] @{
            id = $forecast.official_match_id
            match = $forecast.match
            reason = "official complete feed unavailable"
        }
        continue
    }
    $rawResult = Get-Content -Raw -LiteralPath $resultPath
    $match = ($rawResult | ConvertFrom-Json).matches[0]
    $officialMatchName = "$($match.team1.displayNameA) v $($match.team2.displayNameA)"
    if ($match.statusCode -ne "D" -or $match.status -ne "Completed") {
        $excluded += [pscustomobject] @{
            id = $forecast.official_match_id
            match = $officialMatchName
            reason = "nonterminal status $($match.statusCode) $($match.status)"
        }
        continue
    }
    if ($rawResult -match "(?i)walkover|retired|retirement|cancelled|canceled|withdraw") {
        $excluded += [pscustomobject] @{
            id = $forecast.official_match_id
            match = $officialMatchName
            reason = "retirement/walkover/cancellation signal"
        }
        continue
    }

    $team1Aces = $match.base_stats.match.team_1.t_ace
    $team2Aces = $match.base_stats.match.team_2.t_ace
    $team1DoubleFaults = $match.base_stats.match.team_1.df
    $team2DoubleFaults = $match.base_stats.match.team_2.df
    if (
        $null -eq $team1Aces -or $null -eq $team2Aces -or
        $null -eq $team1DoubleFaults -or $null -eq $team2DoubleFaults
    ) {
        $excluded += [pscustomobject] @{
            id = $forecast.official_match_id
            match = $officialMatchName
            reason = "completed but official ace/double-fault totals missing"
        }
        continue
    }

    $tour = if ($match.eventCode -eq "MS") { "ATP" } else { "WTA" }
    $winner = if ($match.team1.won) {
        $match.team1.displayNameA
    } else {
        $match.team2.displayNameA
    }
    $definitions = @(
        [pscustomobject] @{ kind = "aces"; target = $forecast.aces_target;
            probability = [double] $forecast.aces_probability / 100 },
        [pscustomobject] @{ kind = "double_faults"; target = $forecast.df_target;
            probability = [double] $forecast.df_probability / 100 },
        [pscustomobject] @{ kind = "match_win"; target = $forecast.win_target;
            probability = [double] $forecast.win_probability / 100 }
    )

    foreach ($definition in $definitions) {
        $side = Resolve-TargetSide $definition.target $match
        $outcome = if ($definition.kind -eq "aces") {
            if ($side -eq 1) {
                [int] ($team1Aces -gt $team2Aces)
            } else {
                [int] ($team2Aces -gt $team1Aces)
            }
        } elseif ($definition.kind -eq "double_faults") {
            if ($side -eq 1) {
                [int] ($team1DoubleFaults -gt $team2DoubleFaults)
            } else {
                [int] ($team2DoubleFaults -gt $team1DoubleFaults)
            }
        } else {
            if ($side -eq 1) { [int] [bool] $match.team1.won }
            else { [int] [bool] $match.team2.won }
        }
        $probability = [double] $definition.probability
        $scored += [pscustomobject] [ordered] @{
            official_match_id = [string] $forecast.official_match_id
            tour = $tour
            match = $officialMatchName
            kind = $definition.kind
            target = $definition.target
            submitted_percent = [int] [math]::Round(100 * $probability)
            probability = $probability
            outcome = $outcome
            probability_assigned_to_outcome = if ($outcome -eq 1) {
                $probability
            } else {
                1 - $probability
            }
            brier = [math]::Pow($probability - $outcome, 2)
            winner = $winner
            team1_aces = [int] $team1Aces
            team2_aces = [int] $team2Aces
            team1_double_faults = [int] $team1DoubleFaults
            team2_double_faults = [int] $team2DoubleFaults
        }
    }
}

$summaries = [ordered] @{}
foreach ($kind in @("aces", "double_faults", "match_win")) {
    $summaries[$kind] = Get-Summary @($scored | Where-Object { $_.kind -eq $kind })
    foreach ($tour in @("ATP", "WTA")) {
        $summaries["${kind}_${tour}"] = Get-Summary @(
            $scored | Where-Object { $_.kind -eq $kind -and $_.tour -eq $tour }
        )
    }
}

$binDefinitions = @(
    [pscustomobject] @{ name = "00-20"; low = 0.0; high = 0.2; includeHigh = $false },
    [pscustomobject] @{ name = "20-40"; low = 0.2; high = 0.4; includeHigh = $false },
    [pscustomobject] @{ name = "40-60"; low = 0.4; high = 0.6; includeHigh = $false },
    [pscustomobject] @{ name = "60-80"; low = 0.6; high = 0.8; includeHigh = $false },
    [pscustomobject] @{ name = "80-100"; low = 0.8; high = 1.0; includeHigh = $true }
)
$calibrationBins = @()
foreach ($kind in @("aces", "double_faults", "match_win")) {
    foreach ($bin in $binDefinitions) {
        $rows = @(
            $scored | Where-Object {
                $_.kind -eq $kind -and $_.probability -ge $bin.low -and
                ($_.probability -lt $bin.high -or
                    ($bin.includeHigh -and $_.probability -le $bin.high))
            }
        )
        if ($rows.Count -gt 0) {
            $calibrationBins += [pscustomobject] @{
                kind = $kind
                bin = $bin.name
                n = $rows.Count
                mean_probability = [math]::Round((Get-Mean @($rows.probability)), 3)
                observed_yes_rate = [math]::Round((Get-Mean @($rows.outcome)), 3)
                brier = [math]::Round((Get-Mean @($rows.brier)), 3)
            }
        }
    }
}

$paired = @()
$completedIds = @($scored.official_match_id | Sort-Object -Unique)
foreach ($officialId in $completedIds) {
    $rows = @($scored | Where-Object { $_.official_match_id -eq $officialId })
    if ($rows.Count -eq 3) {
        $winBrier = ($rows | Where-Object { $_.kind -eq "match_win" }).brier
        $aceBrier = ($rows | Where-Object { $_.kind -eq "aces" }).brier
        $dfBrier = ($rows | Where-Object { $_.kind -eq "double_faults" }).brier
        $paired += [pscustomobject] @{
            id = $officialId
            win_minus_aces = $winBrier - $aceBrier
            win_minus_double_faults = $winBrier - $dfBrier
        }
    }
}

$matchDiagnostics = @()
foreach ($officialId in $completedIds) {
    $rows = @($scored | Where-Object { $_.official_match_id -eq $officialId })
    $ace = $rows | Where-Object { $_.kind -eq "aces" }
    $df = $rows | Where-Object { $_.kind -eq "double_faults" }
    $win = $rows | Where-Object { $_.kind -eq "match_win" }
    $serveAdvantage = ($ace.probability + (1 - $df.probability)) / 2
    $matchDiagnostics += [pscustomobject] @{
        official_match_id = $officialId
        match = $win.match
        target = $win.target
        win_probability = $win.probability
        win_outcome = $win.outcome
        serve_prop_advantage = $serveAdvantage
        ace_probability = $ace.probability
        ace_outcome = $ace.outcome
        double_fault_probability = $df.probability
        double_fault_outcome = $df.outcome
        win_brier = $win.brier
    }
}

$result = [pscustomobject] [ordered] @{
    schema_version = "usopen-prop-performance-audit/v1"
    generated_at_utc = [datetime]::UtcNow.ToString("o")
    completed_matches = $completedIds.Count
    excluded = $excluded
    summaries = $summaries
    calibration_bins = $calibrationBins
    paired_brier_differences = [pscustomobject] @{
        win_minus_aces = Get-DifferenceSummary @($paired.win_minus_aces)
        win_minus_double_faults = Get-DifferenceSummary @($paired.win_minus_double_faults)
    }
    largest_errors = @($scored | Sort-Object brier -Descending | Select-Object -First 18)
    largest_win_errors = @(
        $scored | Where-Object { $_.kind -eq "match_win" } |
            Sort-Object brier -Descending | Select-Object -First 15
    )
    match_diagnostics = $matchDiagnostics
    scored = $scored
}

$result | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath "tmp/audit_scored_predictions.json" -Encoding utf8
$scored | Export-Csv -NoTypeInformation -LiteralPath "tmp/audit_scored_predictions.csv" -Encoding utf8

[pscustomobject] @{
    completed_matches = $result.completed_matches
    excluded_count = $excluded.Count
    excluded = $excluded
    summaries = $summaries
    paired_brier_differences = $result.paired_brier_differences
    calibration_bins = $calibrationBins
    largest_win_errors = $result.largest_win_errors
    match_diagnostics = $matchDiagnostics
} | ConvertTo-Json -Depth 8 -Compress
