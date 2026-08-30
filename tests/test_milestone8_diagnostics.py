from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import betabinom

from tennis_model.calibration.metrics import reliability_table
from tennis_model.estimation.dependence_diagnostics import (
    CandidateAdoptionEvidence,
    OOFComponentPrediction,
    candidate_adoption_conclusion,
    candidate_one_factor_spec,
    fit_strictly_nested_candidate_loadings,
    run_dependence_randomization_sensitivity,
    run_residual_dependence_diagnostic,
    sample_candidate_beta_marginals,
)
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.locking.path_counts import (
    FROZEN_PATH_COUNT_POLICY,
    PathCountPolicy,
    escalation_reasons,
)
from tennis_model.schemas import Tour
from tennis_model.simulation import MATCH_WIN, PropEstimate
from tennis_model.simulation.match import _simulate_one_path
from tennis_model.simulation.monotonicity import coupled_primitive_monotonicity
from tennis_model.simulation.point import ServePerformanceDraw


def _oof_predictions(*, correlated: bool, seed: int) -> tuple[OOFComponentPrediction, ...]:
    rng = np.random.default_rng(seed)
    records = []
    for fold in range(1, 5):
        fold_start = datetime(2021 + fold, 1, 1, tzinfo=UTC)
        for row_index in range(100):
            shared = float(rng.standard_normal())
            for component_index, component in enumerate(ServeComponent):
                latent = (
                    0.92 * shared + np.sqrt(1.0 - 0.92**2) * float(rng.standard_normal())
                    if correlated
                    else float(rng.standard_normal())
                )
                quantile = float(ndtr(latent))
                alpha = 18.0 + component_index
                beta_shape = 22.0 + component_index
                trials = 60
                successes = int(betabinom.ppf(quantile, trials, alpha, beta_shape))
                match_start = fold_start + timedelta(days=row_index)
                records.append(
                    OOFComponentPrediction(
                        row_id=f"f{fold}-r{row_index}",
                        event_id=f"f{fold}-event{row_index // 10}",
                        player_id=f"player-{row_index % 20}",
                        tour=Tour.ATP,
                        chronological_fold=fold,
                        component=component,
                        successes=successes,
                        trials=trials,
                        beta_alpha=alpha,
                        beta_beta=beta_shape,
                        prediction_cutoff_utc=match_start - timedelta(days=1),
                        match_start_utc=match_start,
                    )
                )
    return tuple(records)


def test_independent_residuals_do_not_activate_live_v1_dependence() -> None:
    report = run_residual_dependence_diagnostic(
        _oof_predictions(correlated=False, seed=91),
        tour=Tour.ATP,
        randomization_seed=92,
        bootstrap_seed=93,
        bootstrap_replicates=150,
    )
    assert not report.candidate_gate_passed
    assert report.live_v1_mode == "independent"
    with pytest.raises(ValueError, match="blocked"):
        candidate_one_factor_spec(
            report,
            tuple((component, 0.1) for component in ServeComponent),
        )


def test_correlated_residuals_pass_research_gate_and_candidate_preserves_marginals() -> None:
    report = run_residual_dependence_diagnostic(
        _oof_predictions(correlated=True, seed=101),
        tour=Tour.ATP,
        randomization_seed=102,
        bootstrap_seed=103,
        bootstrap_replicates=200,
    )
    assert report.candidate_gate_passed
    assert any(item.candidate_gate_passed for item in report.pairs)
    assert all(item.event_block_interval.replicates > 0 for item in report.pairs)
    spec = candidate_one_factor_spec(
        report,
        tuple((component, 0.35) for component in ServeComponent),
    )
    shapes = tuple((component, 10.0, 15.0) for component in ServeComponent)
    draws = sample_candidate_beta_marginals(shapes, spec, n_draws=20_000, seed=104)
    assert draws.shape == (20_000, 5)
    assert np.allclose(draws.mean(axis=0), 0.4, atol=0.012)

    supported = candidate_adoption_conclusion(
        CandidateAdoptionEvidence(
            joint_log_density_improvement=0.02,
            joint_log_density_interval_lower=0.001,
            improves_core_prop_metric=True,
            core_prop_brier_changes=(("ACES", -0.001), ("MATCH_SET", 0.0008)),
        )
    )
    assert supported == "V1_1_CANDIDATE_SUPPORTED"
    rejected = candidate_adoption_conclusion(
        CandidateAdoptionEvidence(
            joint_log_density_improvement=0.02,
            joint_log_density_interval_lower=-0.001,
            improves_core_prop_metric=True,
            core_prop_brier_changes=(("ACES", -0.001),),
        )
    )
    assert rejected == "NO_EVIDENCE_FOR_V1_1_DEPENDENCE"


def _estimate(probability: float, *, settled: int, total: int, se: float) -> PropEstimate:
    yes = round(probability * settled)
    return PropEstimate(
        prop=MATCH_WIN("A"),
        probability_raw=probability,
        probability_settled=settled / total,
        yes_paths=yes,
        no_paths=settled - yes,
        void_paths=total - settled,
        unresolved_paths=0,
        settled_paths=settled,
        total_paths=total,
        mc_standard_error=se,
    )


def test_frozen_path_count_escalation_triggers_are_deterministic() -> None:
    assert FROZEN_PATH_COUNT_POLICY.standard_paths == 100_000
    assert FROZEN_PATH_COUNT_POLICY.escalated_paths == 400_000
    assert escalation_reasons((_estimate(0.02, settled=100_000, total=100_000, se=0.001),)) == (
        "PROBABILITY_BELOW_3_PERCENT",
    )
    assert "FEWER_THAN_50000_SETTLED_PATHS" in escalation_reasons(
        (_estimate(0.5, settled=49_999, total=100_000, se=0.002),)
    )
    assert "INTEGER_SUBMISSION_MC_SENSITIVITY" in escalation_reasons(
        (_estimate(0.625, settled=100_000, total=100_000, se=0.001),)
    )
    quiet_policy = PathCountPolicy(integer_boundary_standard_errors=1.0)
    assert (
        escalation_reasons(
            (_estimate(0.60, settled=100_000, total=100_000, se=0.001),),
            quiet_policy,
        )
        == ()
    )


def test_reliability_bins_use_fixed_tenths_and_ignore_void_rows() -> None:
    rows = cast(
        tuple[Any, ...],
        (
            SimpleNamespace(
                resolution_status="yes",
                probability_raw=0.05,
                outcome_binary=1,
                brier_raw_model=0.9025,
            ),
            SimpleNamespace(
                resolution_status="no",
                probability_raw=0.15,
                outcome_binary=0,
                brier_raw_model=0.0225,
            ),
            SimpleNamespace(
                resolution_status="void",
                probability_raw=0.05,
                outcome_binary=None,
                brier_raw_model=None,
            ),
            SimpleNamespace(
                resolution_status="yes",
                probability_raw=1.0,
                outcome_binary=1,
                brier_raw_model=0.0,
            ),
        ),
    )
    bins = reliability_table(rows)
    assert [item.count for item in bins] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert bins[0].empirical_frequency == 1.0
    assert bins[1].empirical_frequency == 0.0
    assert bins[9].upper_inclusive


def test_ace_df_service_and_hold_monotonicity_use_preallocated_uniforms() -> None:
    opponent = ServePerformanceDraw(0.62, 0.08, 0.58, 0.08, 0.52)
    baseline = ServePerformanceDraw(0.64, 0.04, 0.56, 0.06, 0.52)
    high_ace = ServePerformanceDraw(0.64, 0.36, 0.56, 0.06, 0.52)
    high_df = ServePerformanceDraw(0.64, 0.04, 0.56, 0.46, 0.52)

    ace_diagnostics = coupled_primitive_monotonicity(
        baseline,
        target="ace",
        low_probability=baseline.ace_given_first_in,
        high_probability=high_ace.ace_given_first_in,
        n_draws=10_000,
        seed=707,
    )
    df_diagnostics = coupled_primitive_monotonicity(
        baseline,
        target="double_fault",
        low_probability=baseline.double_fault_given_second_opp,
        high_probability=high_df.double_fault_given_second_opp,
        n_draws=10_000,
        seed=808,
    )
    assert all(item.passed for item in (*ace_diagnostics, *df_diagnostics))
    assert ace_diagnostics[0].evidence == "PATHWISE_PREALLOCATED_UNIFORMS"
    assert df_diagnostics[0].evidence == "PATHWISE_PREALLOCATED_UNIFORMS"
    assert ace_diagnostics[1].evidence == "ANALYTIC"
    assert ace_diagnostics[2].evidence == "ANALYTIC"

    def outcomes(
        performance: ServePerformanceDraw, *, seed: int, paths: int
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        wins = 0
        holds = 0
        service_games = 0
        for _ in range(paths):
            path = _simulate_one_path(
                "A",
                "B",
                best_of=3,
                first_server_id="A",
                player_a_performance=performance,
                player_b_performance=opponent,
                rng=rng,
                trace_points=False,
            )
            wins += path.winner_id == "A"
            stats = path.player_stats["A"]
            holds += stats.service_games_held
            service_games += stats.service_games_played
        return wins / paths, holds / service_games

    # High-level match evidence is deliberately independent and distributional;
    # it does not claim pathwise coupling from identical root seeds.
    paths = 1_000
    baseline_win, _ = outcomes(baseline, seed=1001, paths=paths)
    ace_win, _ = outcomes(high_ace, seed=2002, paths=paths)
    df_win, _ = outcomes(high_df, seed=3003, paths=paths)
    ace_se = np.sqrt(baseline_win * (1 - baseline_win) / paths + ace_win * (1 - ace_win) / paths)
    df_se = np.sqrt(baseline_win * (1 - baseline_win) / paths + df_win * (1 - df_win) / paths)
    assert ace_win + 5 * ace_se >= baseline_win
    assert df_win <= baseline_win + 5 * df_se


def test_dependence_randomization_sensitivity_and_nested_loading_fit() -> None:
    predictions = _oof_predictions(correlated=True, seed=919)
    sensitivity = run_dependence_randomization_sensitivity(
        predictions,
        tour=Tour.ATP,
        randomization_seeds=(11, 22, 33),
        bootstrap_seed=44,
        bootstrap_replicates=30,
        minimum_partition_records=150,
    )
    assert sensitivity.label == "CANDIDATE_V1_1_ONLY"
    assert sensitivity.live_v1_mode == "independent"
    assert len(sensitivity.reports) == 3
    assert {item.status for item in sensitivity.stability_views} <= {
        "DESCRIPTIVE",
        "UNDERPOWERED",
    }
    assert any(item.partition_type == "surface" for item in sensitivity.stability_views)
    assert any(item.partition_type == "context" for item in sensitivity.stability_views)
    assert all(len(item.pair_correlations) == 10 for item in sensitivity.stability_views)
    assert any(
        pair.central_correlation is not None
        for item in sensitivity.stability_views
        for pair in item.pair_correlations
    )

    nested = fit_strictly_nested_candidate_loadings(
        predictions,
        tour=Tour.ATP,
        randomization_seed=55,
        minimum_complete_training_rows=20,
    )
    assert tuple(item.evaluation_fold for item in nested) == (2, 3, 4)
    assert all(item.label == "CANDIDATE_V1_1_ONLY" for item in nested)
    assert all(
        item.training_end_utc is not None and item.training_end_utc < item.evaluation_start_utc
        for item in nested
    )
    assert all(item.status == "FROZEN_FROM_EARLIER_DATA" for item in nested)
