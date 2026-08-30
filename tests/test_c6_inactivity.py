from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise

import numpy as np
import pytest

from tennis_model.estimation.inactivity import (
    CompetitionClass,
    DuplicateInactivityAdjustmentError,
    InactivityAdjustmentState,
    InactivityCoordinateReference,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityMatchCandidate,
    InactivityRecord,
    InactivityTerminalStatus,
    InactivityUnavailableError,
    PlayedPointEvidence,
    affected_coordinate_roles,
    apply_inactivity_adjustment,
    build_inactivity_record,
    inactivity_factors,
)
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.schemas import Tour

HASH = "a" * 64
CUTOFF = datetime(2026, 8, 20, 12, tzinfo=UTC)
START = date(2026, 8, 25)


def _coverage(
    state: InactivityCoverageState = InactivityCoverageState.VERIFIED_COMPLETE,
) -> InactivityCoverageAssertion:
    return InactivityCoverageAssertion(
        state=state,
        source_manifest_id="manifest-v1",
        source_manifest_sha256=HASH,
        canonical_player_id=("p1" if state is InactivityCoverageState.VERIFIED_COMPLETE else None),
        asserted_at_utc=datetime(2026, 8, 19, tzinfo=UTC),
        reason=None if state is InactivityCoverageState.VERIFIED_COMPLETE else "not resolved",
    )


def _candidate(
    *,
    days: int,
    available_at_utc: datetime | None = None,
    match_id: str | None = None,
    terminal_status: InactivityTerminalStatus = InactivityTerminalStatus.STARTED_RETIREMENT,
    started_evidence: tuple[PlayedPointEvidence, ...] = (
        PlayedPointEvidence.POSITIVE_POINT_STAT_COUNT,
    ),
) -> InactivityMatchCandidate:
    return InactivityMatchCandidate(
        player_id="p1",
        identity_resolved=True,
        tour=Tour.ATP,
        match_id=match_id or f"match-{days}",
        match_date_local=START.fromordinal(START.toordinal() - days),
        discipline="singles",
        competition_class=CompetitionClass.MAIN_DRAW,
        terminal_status=terminal_status,
        started_evidence=started_evidence,
        source_manifest_id="manifest-v1",
        source_pin="source-row-1",
        source_sha256=HASH,
        available_at_utc=available_at_utc or datetime(2026, 8, 18, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("days", "gap", "multiplier", "inflation"),
    (
        (0, 0, 1.0, 1.0),
        (90, 0, 1.0, 1.0),
        (91, 1, 0.9961565872205752, 1.0076720537370566),
        (270, 180, 0.5, 1.75),
        (450, 360, 0.25, 1.9375),
        (810, 720, 0.0625, 1.99609375),
    ),
)
def test_frozen_inactivity_table(days: int, gap: int, multiplier: float, inflation: float) -> None:
    observed = inactivity_factors(days)
    assert observed[0] == gap
    assert observed[1] == pytest.approx(multiplier, abs=1e-12)
    assert observed[2] == pytest.approx(inflation, abs=1e-12)


def test_negative_days_fail_instead_of_clipping() -> None:
    with pytest.raises(ValueError, match="negative inactivity"):
        inactivity_factors(-1)


def test_last_started_match_is_cutoff_safe_and_retirement_is_eligible() -> None:
    record = build_inactivity_record(
        player_id="p1",
        tour=Tour.ATP,
        scheduled_start_local_date=START,
        information_cutoff_utc=CUTOFF,
        coverage=_coverage(),
        candidates=(
            _candidate(days=270),
            _candidate(days=1, available_at_utc=CUTOFF),
        ),
    )
    assert record.last_eligible_match is not None
    assert record.last_eligible_match.match_id == "match-270"
    assert record.inactivity_days == 270
    assert record.hard_deviation_multiplier == 0.5
    assert record.variance_inflation_factor == 1.75


def test_cutoff_visible_correction_replaces_earlier_started_match_revision() -> None:
    record = build_inactivity_record(
        player_id="p1",
        tour=Tour.ATP,
        scheduled_start_local_date=START,
        information_cutoff_utc=CUTOFF,
        coverage=_coverage(),
        candidates=(
            _candidate(
                days=10,
                match_id="corrected-match",
                available_at_utc=datetime(2026, 8, 18, tzinfo=UTC),
            ),
            _candidate(
                days=10,
                match_id="corrected-match",
                terminal_status=InactivityTerminalStatus.WALKOVER,
                started_evidence=(),
                available_at_utc=datetime(2026, 8, 19, tzinfo=UTC),
            ),
            _candidate(days=270, match_id="older-started-match"),
        ),
    )
    assert record.last_eligible_match is not None
    assert record.last_eligible_match.match_id == "older-started-match"
    assert record.inactivity_days == 270


def test_verified_no_history_is_cold_start_but_missing_coverage_blocks() -> None:
    record = build_inactivity_record(
        player_id="p1",
        tour=Tour.ATP,
        scheduled_start_local_date=START,
        information_cutoff_utc=CUTOFF,
        coverage=_coverage(),
        candidates=(),
    )
    assert record.cold_start
    assert record.inactivity_days is None
    assert record.post_threshold_days is None
    assert record.hard_deviation_multiplier == 0.0
    assert record.variance_inflation_factor == 1.0
    with pytest.raises(InactivityUnavailableError, match="coverage is unavailable"):
        build_inactivity_record(
            player_id="p1",
            tour=Tour.ATP,
            scheduled_start_local_date=START,
            information_cutoff_utc=CUTOFF,
            coverage=_coverage(InactivityCoverageState.INCOMPLETE_SOURCE),
            candidates=(),
        )


def _record(days: int) -> InactivityRecord:
    return build_inactivity_record(
        player_id="p1",
        tour=Tour.ATP,
        scheduled_start_local_date=START,
        information_cutoff_utc=CUTOFF,
        coverage=_coverage(),
        candidates=(_candidate(days=days),),
    )


def _coordinate_fixture() -> tuple[
    tuple[str, ...], np.ndarray, np.ndarray, tuple[InactivityCoordinateReference, ...]
]:
    roles = affected_coordinate_roles(ServeComponent.A)
    ids = (*tuple(f"A:p1:{role.value}" for role in roles), "A:context:intercept")
    mean = np.asarray((0.40, 0.20, -0.30, -0.10, 1.20), dtype=np.float64)
    root = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.2, 0.9, 0.0, 0.0, 0.0),
            (-0.1, 0.1, 0.8, 0.0, 0.0),
            (0.3, -0.2, 0.1, 0.7, 0.0),
            (0.2, 0.1, -0.1, 0.2, 0.6),
        ),
        dtype=np.float64,
    )
    covariance = root @ root.T
    catalog = tuple(
        InactivityCoordinateReference(
            component=ServeComponent.A,
            player_id="p1",
            role=role,
            coordinate_id=ids[index],
            index=index,
        )
        for index, role in enumerate(roles)
    )
    return ids, mean, covariance, catalog


def test_mean_covariance_congruence_preserves_globals_correlations_psd_and_kappa() -> None:
    ids, mean, covariance, catalog = _coordinate_fixture()
    adjusted = apply_inactivity_adjustment(
        component=ServeComponent.A,
        coordinate_ids=ids,
        mean=mean,
        covariance=covariance,
        kappa=31.0,
        inactivity_records=(_record(270),),
        coordinate_catalog=catalog,
    )
    assert adjusted.mean[[0, 2, 4]] == pytest.approx(mean[[0, 2, 4]], abs=0.0)
    assert adjusted.mean[[1, 3]] == pytest.approx(0.5 * mean[[1, 3]], abs=0.0)
    assert np.diag(adjusted.covariance)[:4] == pytest.approx(1.75 * np.diag(covariance)[:4])
    assert adjusted.covariance[4, 4] == covariance[4, 4]
    before_corr = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    after_corr = adjusted.covariance / np.sqrt(
        np.outer(np.diag(adjusted.covariance), np.diag(adjusted.covariance))
    )
    assert after_corr == pytest.approx(before_corr, abs=1e-14)
    assert np.min(np.linalg.eigvalsh(adjusted.covariance)) >= -1e-12
    assert adjusted.kappa == 31.0
    with pytest.raises(DuplicateInactivityAdjustmentError):
        apply_inactivity_adjustment(
            component=ServeComponent.A,
            coordinate_ids=ids,
            mean=mean,
            covariance=covariance,
            kappa=31.0,
            inactivity_records=(_record(270),),
            coordinate_catalog=catalog,
            adjustment_state=InactivityAdjustmentState.C6_APPLIED,
        )


def test_active_player_transform_is_bitwise_identity() -> None:
    ids, mean, covariance, catalog = _coordinate_fixture()
    adjusted = apply_inactivity_adjustment(
        component=ServeComponent.A,
        coordinate_ids=ids,
        mean=mean,
        covariance=covariance,
        kappa=31.0,
        inactivity_records=(_record(90),),
        coordinate_catalog=catalog,
    )
    assert np.array_equal(adjusted.mean, mean)
    assert np.array_equal(adjusted.covariance, covariance)


def test_frozen_affected_roles_and_returner_sign_are_exact() -> None:
    expected = {
        ServeComponent.F: ("server_global", "server_hard_deviation"),
        ServeComponent.A: (
            "server_global",
            "server_hard_deviation",
            "returner_global",
            "returner_hard_deviation",
        ),
        ServeComponent.Q1: (
            "server_global",
            "server_hard_deviation",
            "returner_global",
            "returner_hard_deviation",
        ),
        ServeComponent.D: ("server_global", "server_hard_deviation"),
        ServeComponent.Q2: (
            "server_global",
            "server_hard_deviation",
            "returner_global",
            "returner_hard_deviation",
        ),
    }
    for component, roles in expected.items():
        assert tuple(item.value for item in affected_coordinate_roles(component)) == roles

    global_effect = -0.30
    hard_deviation = -0.10
    multiplier = _record(270).hard_deviation_multiplier
    adjusted_effect = global_effect + multiplier * hard_deviation
    assert adjusted_effect == pytest.approx(-0.35)
    assert -adjusted_effect == pytest.approx(0.35)  # returner effects retain subtraction sign


def test_cold_start_zeros_hard_means_without_extra_variance_inflation() -> None:
    ids, mean, covariance, catalog = _coordinate_fixture()
    cold = build_inactivity_record(
        player_id="p1",
        tour=Tour.ATP,
        scheduled_start_local_date=START,
        information_cutoff_utc=CUTOFF,
        coverage=_coverage(),
        candidates=(),
    )
    adjusted = apply_inactivity_adjustment(
        component=ServeComponent.A,
        coordinate_ids=ids,
        mean=mean,
        covariance=covariance,
        kappa=31.0,
        inactivity_records=(cold,),
        coordinate_catalog=catalog,
    )
    assert adjusted.mean[[0, 2, 4]] == pytest.approx(mean[[0, 2, 4]], abs=0.0)
    assert adjusted.mean[[1, 3]] == pytest.approx((0.0, 0.0), abs=0.0)
    assert np.array_equal(adjusted.covariance, covariance)
    assert adjusted.kappa == 31.0


def test_inactivity_mechanics_are_monotone_without_global_rust_penalty() -> None:
    days = (90, 91, 270, 450, 810, 10_000)
    factors = tuple(inactivity_factors(item) for item in days)
    multiplier = tuple(item[1] for item in factors)
    inflation = tuple(item[2] for item in factors)
    assert all(left >= right for left, right in pairwise(multiplier))
    assert all(left <= right for left, right in pairwise(inflation))
    global_effect = 0.4
    hard_deviation = -0.2
    adjusted_hard = tuple(global_effect + item * hard_deviation for item in multiplier)
    assert all(abs(item * hard_deviation) <= abs(hard_deviation) for item in multiplier)
    assert all(global_effect == 0.4 for _item in adjusted_hard)


@pytest.mark.slow_statistical
def test_one_million_adjusted_gaussian_draws_recover_mean_and_variance() -> None:
    draws = 1_000_000
    ids, mean, covariance, catalog = _coordinate_fixture()
    adjusted = apply_inactivity_adjustment(
        component=ServeComponent.A,
        coordinate_ids=ids,
        mean=mean,
        covariance=covariance,
        kappa=31.0,
        inactivity_records=(_record(270),),
        coordinate_catalog=catalog,
    )
    adjusted_mean = adjusted.mean
    adjusted_covariance = adjusted.covariance
    samples = np.random.default_rng(20260830).multivariate_normal(
        adjusted_mean, adjusted_covariance, size=draws
    )
    empirical_mean = samples.mean(axis=0)
    empirical_variance = samples.var(axis=0, ddof=1)
    mean_se = np.sqrt(np.diag(adjusted_covariance) / draws)
    variance = np.diag(adjusted_covariance)
    fourth_central = 3.0 * variance**2
    variance_se = np.sqrt((fourth_central - ((draws - 3) / (draws - 1)) * variance**2) / draws)
    assert np.all(np.abs(empirical_mean[:4] - adjusted_mean[:4]) <= 5.0 * mean_se[:4])
    assert np.all(np.abs(empirical_variance[:4] - variance[:4]) <= 5.0 * variance_se[:4])


@pytest.mark.slow_statistical
def test_one_million_beta_performance_draws_keep_the_same_kappa_law() -> None:
    draws = 1_000_000
    mean = 0.63
    kappa = 37.0
    samples = np.random.default_rng(314159).beta(kappa * mean, kappa * (1.0 - mean), size=draws)
    expected_variance = mean * (1.0 - mean) / (kappa + 1.0)
    mean_se = np.sqrt(expected_variance / draws)
    alpha = kappa * mean
    beta = kappa * (1.0 - mean)
    excess_kurtosis = (
        6.0
        * ((alpha - beta) ** 2 * (kappa + 1.0) - alpha * beta * (kappa + 2.0))
        / (alpha * beta * (kappa + 2.0) * (kappa + 3.0))
    )
    fourth_central = (excess_kurtosis + 3.0) * expected_variance**2
    variance_se = np.sqrt(
        (fourth_central - ((draws - 3) / (draws - 1)) * expected_variance**2) / draws
    )
    assert abs(float(samples.mean()) - mean) <= 5.0 * mean_se
    assert abs(float(samples.var(ddof=1)) - expected_variance) <= 5.0 * variance_se
