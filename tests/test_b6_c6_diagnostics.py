from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

import pytest

from tennis_model.calibration.b6_c6_diagnostics import (
    DiagnosticStatus,
    DiscretePredictiveMass,
    InactivityServeExclusion,
    InactivityServeObservation,
    PredictiveCountInterval,
    RetirementIncidenceObservation,
    RetirementTimingObservation,
    ServeCalibrationMetric,
    SettlementFrequencyObservation,
    randomized_discrete_pit,
    summarize_inactivity_serve,
    summarize_retirement_incidence,
    summarize_retirement_timing,
    summarize_settlement_frequency,
)
from tennis_model.estimation.inactivity import InactivityBand
from tennis_model.estimation.retirement import RetirementHistoryBand
from tennis_model.schemas import Tour

_CUTOFF = datetime(2026, 1, 1, 12, tzinfo=UTC)
_START = datetime(2026, 1, 2, 12, tzinfo=UTC)
_REVEAL = datetime(2026, 1, 3, 12, tzinfo=UTC)


def _incidence(
    observation_id: str,
    *,
    event_id: str,
    tour: Tour,
    best_of: Literal[3, 5],
    history_band: RetirementHistoryBand,
    predicted: float,
    observed: Literal[0, 1],
) -> RetirementIncidenceObservation:
    return RetirementIncidenceObservation(
        observation_id=observation_id,
        match_id=f"match-{observation_id}",
        event_id=event_id,
        player_id=f"player-{observation_id}",
        tour=tour,
        best_of=best_of,
        history_band=history_band,
        predicted_retirement_probability=predicted,
        observed_retirement=observed,
        prediction_cutoff_utc=_CUTOFF,
        match_start_utc=_START,
        outcome_available_at_utc=_REVEAL,
    )


def test_retirement_incidence_groups_are_exact_isolated_and_reproducible() -> None:
    atp = (
        _incidence(
            "atp-a",
            event_id="event-1",
            tour=Tour.ATP,
            best_of=3,
            history_band=RetirementHistoryBand.NO_HISTORY,
            predicted=0.1,
            observed=1,
        ),
        _incidence(
            "atp-b",
            event_id="event-1",
            tour=Tour.ATP,
            best_of=3,
            history_band=RetirementHistoryBand.SPARSE,
            predicted=0.2,
            observed=0,
        ),
        _incidence(
            "atp-c",
            event_id="event-2",
            tour=Tour.ATP,
            best_of=5,
            history_band=RetirementHistoryBand.SUBSTANTIAL,
            predicted=0.3,
            observed=1,
        ),
    )
    wta = _incidence(
        "wta-a",
        event_id="event-3",
        tour=Tour.WTA,
        best_of=3,
        history_band=RetirementHistoryBand.INTERMEDIATE,
        predicted=0.4,
        observed=0,
    )

    report = summarize_retirement_incidence(
        (*atp, wta), bootstrap_seed=20260830, bootstrap_replicates=128
    )
    reordered = summarize_retirement_incidence(
        (wta, *reversed(atp)), bootstrap_seed=20260830, bootstrap_replicates=128
    )
    assert report == reordered

    by_tour = {group.tour: group for group in report.by_tour}
    atp_overall = by_tour[Tour.ATP]
    assert atp_overall.player_starts == 3
    assert atp_overall.observed_retirements == 2
    assert atp_overall.expected_retirements == pytest.approx(0.6)
    assert atp_overall.observed_rate == pytest.approx(2 / 3)
    assert atp_overall.predicted_rate == pytest.approx(0.2)
    assert atp_overall.rate_difference == pytest.approx(2 / 3 - 0.2)
    assert atp_overall.z_score == pytest.approx(1.4 / math.sqrt(0.46))
    assert atp_overall.status is DiagnosticStatus.NOT_FLAGGED
    assert atp_overall.event_block_interval is not None
    assert atp_overall.event_block_interval.replicates == 128
    assert atp_overall.event_block_interval.level == 0.99

    assert by_tour[Tour.WTA].player_starts == 1
    assert by_tour[Tour.WTA].expected_retirements == pytest.approx(0.4)
    assert by_tour[Tour.WTA].observed_retirements == 0

    formats = {(group.tour, group.group): group for group in report.by_format}
    assert formats[Tour.ATP, "BO3"].player_starts == 2
    assert formats[Tour.ATP, "BO3"].expected_retirements == pytest.approx(0.3)
    assert formats[Tour.ATP, "BO5"].observed_retirements == 1
    assert formats[Tour.WTA, "BO3"].expected_retirements == pytest.approx(0.4)
    assert all(group.status is DiagnosticStatus.UNDERPOWERED for group in report.by_format)

    history = {(group.tour, group.group): group for group in report.by_history_band}
    assert history[Tour.ATP, RetirementHistoryBand.NO_HISTORY.value].observed_retirements == 1
    assert history[Tour.ATP, RetirementHistoryBand.SPARSE.value].observed_retirements == 0
    assert history[Tour.ATP, RetirementHistoryBand.SUBSTANTIAL.value].player_starts == 1
    assert history[Tour.WTA, RetirementHistoryBand.INTERMEDIATE.value].player_starts == 1
    assert all(group.status is DiagnosticStatus.UNDERPOWERED for group in report.by_history_band)

    atp_only = summarize_retirement_incidence(
        atp, bootstrap_seed=20260830, bootstrap_replicates=128
    )
    assert atp_only.by_tour == (atp_overall,)
    assert tuple(group for group in report.by_format if group.tour is Tour.ATP) == (
        atp_only.by_format
    )
    assert tuple(group for group in report.by_history_band if group.tour is Tour.ATP) == (
        atp_only.by_history_band
    )


def _timing(
    index: int,
    *,
    observed: int | None,
    missing_reason: str | None = None,
) -> RetirementTimingObservation:
    return RetirementTimingObservation(
        observation_id=f"timing-{index:03d}",
        match_id=f"match-{index:03d}",
        event_id=f"event-{index // 5:03d}",
        tour=Tour.ATP,
        observed_completed_game=observed,
        missing_reason=missing_reason,
        predictive_mass=DiscretePredictiveMass(
            support=(0, 1),
            probabilities=(0.4, 0.6),
        ),
    )


def test_retirement_timing_minimum_missingness_and_seeded_pit_ks() -> None:
    reliable = tuple(_timing(index, observed=index % 2) for index in range(50))
    missing = _timing(50, observed=None, missing_reason="official timing unavailable")

    under_minimum = summarize_retirement_timing(
        (*reliable[:49], missing),
        randomization_seed=101,
        bootstrap_seed=202,
        bootstrap_replicates=64,
    ).by_tour[0]
    assert under_minimum.status is DiagnosticStatus.UNAVAILABLE
    assert under_minimum.reliable_timings == 49
    assert under_minimum.missing_timings == 1
    assert under_minimum.missing_reasons == (("official timing unavailable", 1),)
    assert under_minimum.randomized_pit == ()
    assert under_minimum.ks_statistic is None
    assert under_minimum.bootstrap_p_value is None
    assert under_minimum.bootstrap_replicates == 0

    available = summarize_retirement_timing(
        (*reliable, missing),
        randomization_seed=101,
        bootstrap_seed=202,
        bootstrap_replicates=64,
    ).by_tour[0]
    reordered = summarize_retirement_timing(
        tuple(reversed((*reliable, missing))),
        randomization_seed=101,
        bootstrap_seed=202,
        bootstrap_replicates=64,
    ).by_tour[0]
    assert available == reordered
    assert available.reliable_timings == 50
    assert available.missing_timings == 1
    assert len(available.randomized_pit) == 50
    assert all(0.0 <= value <= 1.0 for value in available.randomized_pit)
    assert available.ks_statistic is not None
    assert 0.0 <= available.ks_statistic <= 1.0
    assert available.bootstrap_p_value is not None
    assert 0.0 < available.bootstrap_p_value <= 1.0
    assert available.bootstrap_replicates == 64
    assert available.status in {DiagnosticStatus.NOT_FLAGGED, DiagnosticStatus.FLAGGED}

    assert randomized_discrete_pit(
        DiscretePredictiveMass(support=(0, 1), probabilities=(0.25, 0.75)),
        observed=1,
        randomizer=0.4,
    ) == pytest.approx(0.55)


def test_settlement_frequency_keeps_missing_states_out_of_observed_rates() -> None:
    rows = (
        SettlementFrequencyObservation(
            match_id="m1",
            prop_id="aces-1",
            prop_family="ACES",
            tour=Tour.ATP,
            predicted_settled_probability=0.8,
            predicted_void_probability=0.1,
            predicted_unresolved_probability=0.1,
            observed_state="settled",
        ),
        SettlementFrequencyObservation(
            match_id="m2",
            prop_id="aces-2",
            prop_family="ACES",
            tour=Tour.ATP,
            predicted_settled_probability=0.6,
            predicted_void_probability=0.3,
            predicted_unresolved_probability=0.1,
            observed_state="void",
        ),
        SettlementFrequencyObservation(
            match_id="m3",
            prop_id="aces-3",
            prop_family="ACES",
            tour=Tour.ATP,
            predicted_settled_probability=0.5,
            predicted_void_probability=0.2,
            predicted_unresolved_probability=0.3,
            observed_state="unavailable",
            missing_reason="source result absent",
        ),
        SettlementFrequencyObservation(
            match_id="m4",
            prop_id="games-1",
            prop_family="GAMES",
            tour=Tour.WTA,
            predicted_settled_probability=0.7,
            predicted_void_probability=0.1,
            predicted_unresolved_probability=0.2,
            observed_state="unresolved",
            missing_reason="unsupported settlement case",
        ),
    )

    summaries = {
        (group.tour, group.prop_family): group for group in summarize_settlement_frequency(rows)
    }
    aces = summaries[Tour.ATP, "ACES"]
    assert aces.rows == 3
    assert aces.observed_eligible == 2
    assert aces.unavailable_or_unresolved == 1
    assert aces.predicted_settled_rate == pytest.approx(1.9 / 3)
    assert aces.observed_settled_rate == 0.5
    assert aces.settled_rate_difference == pytest.approx(0.5 - 1.9 / 3)
    assert aces.predicted_void_rate == pytest.approx(0.2)
    assert aces.observed_void_rate == 0.5
    assert aces.void_rate_difference == pytest.approx(0.3)
    assert aces.predicted_unresolved_rate == pytest.approx(1 / 6)

    games = summaries[Tour.WTA, "GAMES"]
    assert games.observed_eligible == 0
    assert games.unavailable_or_unresolved == 1
    assert games.observed_settled_rate is None
    assert games.observed_void_rate is None
    assert games.settled_rate_difference is None
    assert games.void_rate_difference is None


def _serve_observation(
    observation_id: str,
    *,
    tour: Tour,
    band: InactivityBand,
    metric: ServeCalibrationMetric,
    observed: int,
    probabilities: tuple[float, ...] = (0.25, 0.5, 0.25),
    interval: tuple[int, int] = (0, 1),
) -> InactivityServeObservation:
    trials = len(probabilities) - 1
    return InactivityServeObservation(
        observation_id=observation_id,
        event_id=f"event-{observation_id}",
        player_id=f"player-{observation_id}",
        tour=tour,
        inactivity_band=band,
        metric=metric,
        observed_count=observed,
        trials=trials,
        predictive_mass=DiscretePredictiveMass(
            support=tuple(range(trials + 1)),
            probabilities=probabilities,
        ),
        interval=PredictiveCountInterval(
            lower=interval[0],
            upper=interval[1],
            nominal_coverage=0.8,
            policy_version="central-count-interval/v1",
        ),
        prediction_cutoff_utc=_CUTOFF,
        match_start_utc=_START,
        outcome_available_at_utc=_REVEAL,
    )


def test_inactivity_serve_calibration_covers_all_metrics_bands_and_missingness() -> None:
    observations = (
        _serve_observation(
            "f-1",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_91_180,
            metric=ServeCalibrationMetric.F,
            observed=0,
        ),
        _serve_observation(
            "f-2",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_91_180,
            metric=ServeCalibrationMetric.F,
            observed=2,
            probabilities=(0.2, 0.3, 0.5),
        ),
        _serve_observation(
            "a",
            tour=Tour.ATP,
            band=InactivityBand.ACTIVE_DAYS_0_90,
            metric=ServeCalibrationMetric.A,
            observed=1,
        ),
        _serve_observation(
            "q1",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_181_365,
            metric=ServeCalibrationMetric.Q1,
            observed=1,
        ),
        _serve_observation(
            "d",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_OVER_365,
            metric=ServeCalibrationMetric.D,
            observed=1,
        ),
        _serve_observation(
            "q2",
            tour=Tour.WTA,
            band=InactivityBand.DAYS_91_180,
            metric=ServeCalibrationMetric.Q2,
            observed=1,
        ),
        _serve_observation(
            "w1",
            tour=Tour.ATP,
            band=InactivityBand.ACTIVE_DAYS_0_90,
            metric=ServeCalibrationMetric.W1,
            observed=1,
        ),
        _serve_observation(
            "w2-zero-mass",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_181_365,
            metric=ServeCalibrationMetric.W2,
            observed=0,
            probabilities=(0.0, 0.5, 0.5),
            interval=(0, 0),
        ),
        _serve_observation(
            "p-srv",
            tour=Tour.ATP,
            band=InactivityBand.DAYS_OVER_365,
            metric=ServeCalibrationMetric.P_SRV,
            observed=2,
        ),
        _serve_observation(
            "cold-start",
            tour=Tour.WTA,
            band=InactivityBand.COLD_START,
            metric=ServeCalibrationMetric.F,
            observed=1,
        ),
    )
    exclusions = (
        InactivityServeExclusion(
            observation_id="excluded-1",
            tour=Tour.ATP,
            inactivity_band=InactivityBand.DAYS_91_180,
            metric=ServeCalibrationMetric.F,
            reason="missing denominator",
        ),
        InactivityServeExclusion(
            observation_id="excluded-2",
            tour=Tour.WTA,
            inactivity_band=None,
            metric=None,
            reason="quarantined raw anomaly",
        ),
        InactivityServeExclusion(
            observation_id="excluded-3",
            tour=Tour.WTA,
            inactivity_band=InactivityBand.COLD_START,
            metric=ServeCalibrationMetric.Q1,
            reason="missing denominator",
        ),
    )

    report = summarize_inactivity_serve(
        observations, exclusions=exclusions, randomization_seed=20260830
    )
    reordered = summarize_inactivity_serve(
        tuple(reversed(observations)),
        exclusions=tuple(reversed(exclusions)),
        randomization_seed=20260830,
    )
    assert report == reordered
    assert {group.metric for group in report.primitive} == {
        ServeCalibrationMetric.F,
        ServeCalibrationMetric.A,
        ServeCalibrationMetric.Q1,
        ServeCalibrationMetric.D,
        ServeCalibrationMetric.Q2,
    }
    assert {group.metric for group in report.derived} == {
        ServeCalibrationMetric.W1,
        ServeCalibrationMetric.W2,
        ServeCalibrationMetric.P_SRV,
    }
    assert {group.inactivity_band for group in (*report.primitive, *report.derived)} == {
        InactivityBand.ACTIVE_DAYS_0_90,
        InactivityBand.DAYS_91_180,
        InactivityBand.DAYS_181_365,
        InactivityBand.DAYS_OVER_365,
    }
    assert report.cold_start_rows == 1
    assert report.exclusions_by_reason == (
        ("missing denominator", 2),
        ("quarantined raw anomaly", 1),
    )

    groups = {
        (group.tour, group.inactivity_band, group.metric): group
        for group in (*report.primitive, *report.derived)
    }
    f_group = groups[Tour.ATP, InactivityBand.DAYS_91_180, ServeCalibrationMetric.F]
    assert f_group.family == "primitive"
    assert f_group.rows == 2
    assert f_group.total_trials == 4
    assert f_group.observed_count == 2
    assert f_group.expected_count == pytest.approx(2.3)
    assert f_group.observed_rate == pytest.approx(0.5)
    assert f_group.predicted_rate == pytest.approx(0.575)
    assert f_group.count_difference == pytest.approx(-0.3)
    assert math.isfinite(f_group.randomized_quantile_residual_mean)
    assert f_group.randomized_quantile_residual_variance > 0.0
    assert f_group.mean_log_predictive_density == pytest.approx(
        (math.log(0.25) + math.log(0.5)) / 2
    )
    assert f_group.zero_predictive_mass_observations == 0
    assert f_group.interval_covered == 1
    assert f_group.interval_coverage == 0.5
    assert f_group.nominal_interval_coverage == 0.8
    assert f_group.interval_policy_version == "central-count-interval/v1"

    w2_group = groups[
        Tour.ATP,
        InactivityBand.DAYS_181_365,
        ServeCalibrationMetric.W2,
    ]
    assert w2_group.family == "derived"
    assert w2_group.zero_predictive_mass_observations == 1
    assert w2_group.mean_log_predictive_density is None
    assert math.isfinite(w2_group.randomized_quantile_residual_mean)
    assert w2_group.interval_coverage == 1.0

    assert (
        groups[
            Tour.WTA,
            InactivityBand.DAYS_91_180,
            ServeCalibrationMetric.Q2,
        ].rows
        == 1
    )
    assert (
        Tour.ATP,
        InactivityBand.DAYS_91_180,
        ServeCalibrationMetric.Q2,
    ) not in groups
