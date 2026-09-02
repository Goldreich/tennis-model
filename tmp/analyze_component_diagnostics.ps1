$ErrorActionPreference = "Stop"

function Mean($Values) {
    $items = @($Values)
    if ($items.Count -eq 0) { return $null }
    return [double] (($items | Measure-Object -Average).Average)
}

function Correlation($X, $Y) {
    $left = @($X)
    $right = @($Y)
    if ($left.Count -ne $right.Count -or $left.Count -lt 2) { return $null }
    $leftMean = Mean $left
    $rightMean = Mean $right
    $cross = 0.0
    $leftSquare = 0.0
    $rightSquare = 0.0
    for ($index = 0; $index -lt $left.Count; $index++) {
        $leftDelta = [double] $left[$index] - $leftMean
        $rightDelta = [double] $right[$index] - $rightMean
        $cross += $leftDelta * $rightDelta
        $leftSquare += $leftDelta * $leftDelta
        $rightSquare += $rightDelta * $rightDelta
    }
    if ($leftSquare -eq 0.0 -or $rightSquare -eq 0.0) { return $null }
    return $cross / [math]::Sqrt($leftSquare * $rightSquare)
}

function Error-Summary($Rows) {
    $items = @($Rows)
    $errors = @($items | ForEach-Object { $_.predicted - $_.actual })
    return [pscustomobject] [ordered] @{
        n = $items.Count
        bias = [math]::Round((Mean $errors), 4)
        mae = [math]::Round((Mean @($errors | ForEach-Object { [math]::Abs($_) })), 4)
        rmse = [math]::Round(
            [math]::Sqrt((Mean @($errors | ForEach-Object { $_ * $_ }))), 4
        )
        correlation = [math]::Round(
            (Correlation @($items.predicted) @($items.actual)), 4
        )
    }
}

$retained = (Get-Content -Raw -LiteralPath "tmp/audit_retained_lock_summaries.json" |
    ConvertFrom-Json).selected
$audit = Get-Content -Raw -LiteralPath "tmp/audit_scored_predictions.json" |
    ConvertFrom-Json
$scored = @($audit.scored)
$completedIds = @($scored.official_match_id | Sort-Object -Unique)

$componentRows = @()
$serviceRows = @()
$matchRows = @()

foreach ($record in $retained) {
    $officialId = [string] $record.official_match_id
    if ($officialId -notin $completedIds) { continue }
    $result = (Get-Content -Raw -LiteralPath (
        "tmp/audit_official_results/$officialId.json"
    ) | ConvertFrom-Json).matches[0]
    $directions = @($record.parameter_summaries)
    if ($directions.Count -ne 2) { continue }

    $actualStats = @($result.base_stats.match.team_1, $result.base_stats.match.team_2)
    $playerNames = @($record.players[0].name, $record.players[1].name)
    $actualServiceWins = @()
    for ($side = 0; $side -lt 2; $side++) {
        $stats = $actualStats[$side]
        $direction = $directions[$side]
        $total = [double] $stats.t_f_srv
        $firstIn = [double] $stats.t_f_srv_in
        $firstWon = [double] $stats.t_f_srv_w
        $secondWon = [double] $stats.t_s_srv_w
        $aces = [double] $stats.t_ace
        $doubleFaults = [double] $stats.df
        $secondOpportunities = $total - $firstIn
        $actuals = [ordered] @{
            F = $firstIn / $total
            A = $aces / $firstIn
            Q1 = ($firstWon - $aces) / ($firstIn - $aces)
            D = $doubleFaults / $secondOpportunities
            Q2 = $secondWon / ($secondOpportunities - $doubleFaults)
        }
        foreach ($component in @("F", "A", "Q1", "D", "Q2")) {
            $primitive = $direction.primitives | Where-Object component -eq $component
            $componentRows += [pscustomobject] @{
                official_match_id = $officialId
                tour = if ($officialId.StartsWith("1")) { "ATP" } else { "WTA" }
                match = $record.match
                player = $playerNames[$side]
                component = $component
                predicted = [double] $primitive.map_mean
                actual = [double] $actuals[$component]
                error = [double] $primitive.map_mean - [double] $actuals[$component]
                sparse_match = "SPARSE_PLAYER_COMPONENT_HISTORY" -in @($record.warnings)
                predictive_concentration = [double] $primitive.predictive_concentration
            }
        }
        $actualServiceWin = ($firstWon + $secondWon) / $total
        $actualServiceWins += $actualServiceWin
        $serviceRows += [pscustomobject] @{
            official_match_id = $officialId
            tour = if ($officialId.StartsWith("1")) { "ATP" } else { "WTA" }
            match = $record.match
            player = $playerNames[$side]
            predicted = [double] $direction.service_point_win
            actual = $actualServiceWin
            error = [double] $direction.service_point_win - $actualServiceWin
            sparse_match = "SPARSE_PLAYER_COMPONENT_HISTORY" -in @($record.warnings)
        }
    }

    $win = $scored | Where-Object {
        $_.official_match_id -eq $officialId -and $_.kind -eq "match_win"
    }
    $modelEdge = [double] $directions[0].service_point_win -
        [double] $directions[1].service_point_win
    $actualEdge = [double] $actualServiceWins[0] - [double] $actualServiceWins[1]
    $callCorrect = if ($win.probability -eq 0.5) {
        $null
    } else {
        [bool] ((($win.probability -gt 0.5) -and ($win.outcome -eq 1)) -or
            (($win.probability -lt 0.5) -and ($win.outcome -eq 0)))
    }
    $matchRows += [pscustomobject] [ordered] @{
        official_match_id = $officialId
        tour = if ($officialId.StartsWith("1")) { "ATP" } else { "WTA" }
        match = $record.match
        target = $win.target
        target_win_probability = $win.probability
        target_won = $win.outcome
        call_correct = $callCorrect
        win_brier = $win.brier
        sparse_match = "SPARSE_PLAYER_COMPONENT_HISTORY" -in @($record.warnings)
        model_service_edge_team1 = $modelEdge
        actual_service_edge_team1 = $actualEdge
        absolute_edge_error = [math]::Abs($modelEdge - $actualEdge)
        model_team1_service_win = [double] $directions[0].service_point_win
        model_team2_service_win = [double] $directions[1].service_point_win
        actual_team1_service_win = [double] $actualServiceWins[0]
        actual_team2_service_win = [double] $actualServiceWins[1]
        model_team1_hold = [double] $directions[0].analytic_hold_probability
        model_team2_hold = [double] $directions[1].analytic_hold_probability
    }
}

$componentSummaries = [ordered] @{}
foreach ($component in @("F", "A", "Q1", "D", "Q2")) {
    $componentSummaries[$component] = Error-Summary @(
        $componentRows | Where-Object component -eq $component
    )
}

$tourWinSummaries = [ordered] @{}
foreach ($tour in @("ATP", "WTA")) {
    $rows = @($scored | Where-Object {
        $_.kind -eq "match_win" -and
        (if ($_.official_match_id.StartsWith("1")) { "ATP" } else { "WTA" }) -eq $tour
    })
    $tourWinSummaries[$tour] = [pscustomobject] @{
        n = $rows.Count
        brier = [math]::Round((Mean @($rows.brier)), 4)
        mean_probability_assigned_to_outcome = [math]::Round(
            (Mean @($rows.probability_assigned_to_outcome)), 4
        )
        decisive_accuracy = [math]::Round(
            @($rows | Where-Object {
                $_.probability -ne 0.5 -and
                ((($_.probability -gt 0.5) -and ($_.outcome -eq 1)) -or
                    (($_.probability -lt 0.5) -and ($_.outcome -eq 0)))
            }).Count / @($rows | Where-Object probability -ne 0.5).Count,
            4
        )
    }
}

$sparseRows = @($matchRows | Where-Object sparse_match)
$nonSparseRows = @($matchRows | Where-Object { -not $_.sparse_match })
$wrongRows = @($matchRows | Where-Object { $_.call_correct -eq $false })
$correctRows = @($matchRows | Where-Object { $_.call_correct -eq $true })
$winRows = @($scored | Where-Object kind -eq "match_win")
$withoutWorst = @($winRows | Sort-Object brier -Descending | Select-Object -Skip 1)
$withoutTwoWorst = @($winRows | Sort-Object brier -Descending | Select-Object -Skip 2)

$result = [pscustomobject] [ordered] @{
    schema_version = "usopen-component-performance-diagnostic/v1"
    completed_matches = $matchRows.Count
    component_realization_errors = $componentSummaries
    service_point_realization_error = Error-Summary $serviceRows
    service_point_edge_correlation = [math]::Round(
        (Correlation @($matchRows.model_service_edge_team1) `
            @($matchRows.actual_service_edge_team1)), 4
    )
    win_brier_correlation_with_absolute_service_edge_error = [math]::Round(
        (Correlation @($matchRows.win_brier) @($matchRows.absolute_edge_error)), 4
    )
    mean_absolute_service_edge_error_correct_calls = [math]::Round(
        (Mean @($correctRows.absolute_edge_error)), 4
    )
    mean_absolute_service_edge_error_wrong_calls = [math]::Round(
        (Mean @($wrongRows.absolute_edge_error)), 4
    )
    win_performance_by_tour = $tourWinSummaries
    sparse_match_performance = [pscustomobject] @{
        n = $sparseRows.Count
        brier = [math]::Round((Mean @($sparseRows.win_brier)), 4)
        wrong_calls = @($sparseRows | Where-Object call_correct -eq $false).Count
    }
    non_sparse_match_performance = [pscustomobject] @{
        n = $nonSparseRows.Count
        brier = [math]::Round((Mean @($nonSparseRows.win_brier)), 4)
        wrong_calls = @($nonSparseRows | Where-Object call_correct -eq $false).Count
    }
    win_brier_all = [math]::Round((Mean @($winRows.brier)), 4)
    win_brier_without_worst = [math]::Round((Mean @($withoutWorst.brier)), 4)
    win_brier_without_two_worst = [math]::Round((Mean @($withoutTwoWorst.brier)), 4)
    q1_concentration = [pscustomobject] @{
        min = ($componentRows | Where-Object component -eq "Q1" |
            Measure-Object predictive_concentration -Minimum).Minimum
        max = ($componentRows | Where-Object component -eq "Q1" |
            Measure-Object predictive_concentration -Maximum).Maximum
    }
    q2_concentration = [pscustomobject] @{
        min = ($componentRows | Where-Object component -eq "Q2" |
            Measure-Object predictive_concentration -Minimum).Minimum
        max = ($componentRows | Where-Object component -eq "Q2" |
            Measure-Object predictive_concentration -Maximum).Maximum
    }
    wrong_win_calls = @($wrongRows | Sort-Object win_brier -Descending)
    largest_service_edge_errors = @(
        $matchRows | Sort-Object absolute_edge_error -Descending | Select-Object -First 12
    )
    largest_component_errors = @(
        $componentRows | Sort-Object { [math]::Abs($_.error) } -Descending |
            Select-Object -First 20
    )
    matches = $matchRows
}

$result | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath "tmp/audit_component_diagnostics.json" -Encoding utf8
$result | ConvertTo-Json -Depth 8 -Compress
