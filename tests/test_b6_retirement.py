from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from tennis_model.estimation.retirement import (
    RETIREMENT_HALF_LIFE_DAYS,
    RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS,
    RETIREMENT_PRIOR_EFFECTIVE_STARTS,
    RETIREMENT_REFERENCE_GAMES,
    RETIREMENT_WINDOW_DAYS,
    CompetingRetirementOutcome,
    HistoricalTerminalClass,
    HistoricalTerminationInput,
    OfficialTerminalStatus,
    PlayerRetirementPosterior,
    RetirementArtifactIntegrityError,
    RetirementCoverageError,
    RetirementFitArtifact,
    RetirementHistoryBand,
    RetirementModelError,
    RetirementScenario,
    RetirementScenarioMixture,
    RetirementSourceCoverage,
    StartedEvidence,
    build_retirement_observations,
    competing_retirement_probabilities,
    draw_competing_retirement,
    draw_mixture_player_retirement_path,
    draw_player_retirement_path,
    draw_retirement_scenario,
    fit_retirement_artifact,
    load_retirement_fit_artifact,
    normalize_historical_termination,
    normalize_historical_terminations_before_cutoff,
    player_retirement_posterior,
    reissue_retirement_fit_eligibility_metadata,
    retirement_history_band,
    retirement_probability_to_intensity,
    retirement_recency_weight,
    write_retirement_fit_artifact,
)
from tennis_model.schemas import Tour

_CUTOFF = datetime(2026, 8, 30, 12, tzinfo=UTC)
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64
_HASH_E = "e" * 64


def _termination_input(
    *,
    match_id: str,
    status: OfficialTerminalStatus,
    age_days: int = 10,
    tour: Tour = Tour.ATP,
    started: bool = True,
    retiree: str | None = None,
    winner: str | None = "player-b",
    retirement_completed_games: int | None = None,
    available_at: datetime | None = None,
    anomaly: str | None = None,
) -> HistoricalTerminationInput:
    evidence = (StartedEvidence.LEGAL_SCORE_COMPLETED_GAME,) if started else ()
    return HistoricalTerminationInput(
        match_id=match_id,
        tour=tour,
        player_a_id=f"{match_id}-player-a",
        player_b_id=f"{match_id}-player-b",
        match_date=_CUTOFF.date() - timedelta(days=age_days),
        official_status=status,
        started_evidence=evidence,
        retiring_player_id=(f"{match_id}-{retiree}" if retiree is not None else None),
        advancing_winner_id=(f"{match_id}-{winner}" if winner is not None else None),
        retirement_completed_games=retirement_completed_games,
        source_id="official-fixture",
        source_sha256=_HASH_A,
        available_at_utc=available_at or (_CUTOFF - timedelta(seconds=1)),
        upstream_anomaly_code=anomaly,
    )


@pytest.mark.parametrize(
    ("status", "started", "retiree", "expected_class", "eligible"),
    [
        (
            OfficialTerminalStatus.NORMAL_COMPLETION,
            True,
            None,
            HistoricalTerminalClass.NORMAL_COMPLETION,
            True,
        ),
        (
            OfficialTerminalStatus.RETIREMENT,
            True,
            "player-a",
            HistoricalTerminalClass.STARTED_RETIREMENT,
            True,
        ),
        (
            OfficialTerminalStatus.WALKOVER_OR_PRESTART_WITHDRAWAL,
            False,
            None,
            HistoricalTerminalClass.WALKOVER,
            False,
        ),
        (
            OfficialTerminalStatus.DEFAULT_DISQUALIFICATION_OR_MISCONDUCT,
            True,
            None,
            HistoricalTerminalClass.ADMINISTRATIVE_TERMINATION,
            False,
        ),
        (
            OfficialTerminalStatus.ABANDONMENT,
            True,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
        (
            OfficialTerminalStatus.CANCELLATION,
            False,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
        (
            OfficialTerminalStatus.NO_CONTEST,
            True,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
        (
            OfficialTerminalStatus.SUSPENDED_UNRESOLVED,
            True,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
        (
            OfficialTerminalStatus.CONFLICTING,
            True,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
        (
            OfficialTerminalStatus.OTHER_AMBIGUOUS,
            True,
            None,
            HistoricalTerminalClass.AMBIGUOUS,
            False,
        ),
    ],
)
def test_terminal_status_classes_are_exhaustive_and_exact(
    status: OfficialTerminalStatus,
    started: bool,
    retiree: str | None,
    expected_class: HistoricalTerminalClass,
    eligible: bool,
) -> None:
    record = normalize_historical_termination(
        _termination_input(
            match_id=f"m-{status.value}",
            status=status,
            started=started,
            retiree=retiree,
            anomaly="UPSTREAM_AMBIGUITY"
            if status is OfficialTerminalStatus.OTHER_AMBIGUOUS
            else None,
        )
    )
    assert record.terminal_class is expected_class
    assert record.incidence_eligible is eligible
    assert record.play_started is started
    assert (record.anomaly_code is not None) is (
        expected_class is HistoricalTerminalClass.AMBIGUOUS
    )


def test_started_retirement_emits_one_and_zero_even_without_timing() -> None:
    classified = normalize_historical_termination(
        _termination_input(
            match_id="ret",
            status=OfficialTerminalStatus.RETIREMENT,
            retiree="player-a",
            retirement_completed_games=None,
        )
    )
    batch = build_retirement_observations((classified,), _CUTOFF)
    responses = {row.player_id: row.response for row in batch.observations}
    assert responses == {"ret-player-a": 1, "ret-player-b": 0}
    assert batch.observations[0].retirement_completed_games is None
    assert not classified.timing_available


@pytest.mark.parametrize(
    ("record", "anomaly"),
    [
        (
            _termination_input(
                match_id="missing-retiree",
                status=OfficialTerminalStatus.RETIREMENT,
                retiree=None,
            ),
            "RETIRING_PLAYER_NOT_RELIABLY_IDENTIFIED",
        ),
        (
            _termination_input(
                match_id="retiree-wins",
                status=OfficialTerminalStatus.RETIREMENT,
                retiree="player-b",
                winner="player-b",
            ),
            "RETIRING_PLAYER_RECORDED_AS_ADVANCING_WINNER",
        ),
        (
            _termination_input(
                match_id="normal-plus-retirement",
                status=OfficialTerminalStatus.NORMAL_COMPLETION,
                retiree="player-a",
            ),
            "NORMAL_COMPLETION_WITH_RETIREMENT_FACTS",
        ),
        (
            _termination_input(
                match_id="retirement-no-start",
                status=OfficialTerminalStatus.RETIREMENT,
                started=False,
                retiree="player-a",
            ),
            "RETIREMENT_WITHOUT_STARTED_EVIDENCE",
        ),
    ],
)
def test_conflicting_or_unresolved_retirements_are_quarantined(
    record: HistoricalTerminationInput,
    anomaly: str,
) -> None:
    classified = normalize_historical_termination(record)
    assert classified.terminal_class is HistoricalTerminalClass.AMBIGUOUS
    assert classified.anomaly_code == anomaly
    batch = build_retirement_observations((classified,), _CUTOFF)
    assert not batch.observations
    assert batch.exclusions[0].anomaly_code == anomaly


@pytest.mark.parametrize(
    "status",
    (OfficialTerminalStatus.NORMAL_COMPLETION, OfficialTerminalStatus.RETIREMENT),
)
def test_upstream_anomaly_fails_closed_for_otherwise_eligible_statuses(
    status: OfficialTerminalStatus,
) -> None:
    record = _termination_input(
        match_id=f"upstream-{status.value}",
        status=status,
        retiree="player-a" if status is OfficialTerminalStatus.RETIREMENT else None,
        anomaly="SOURCE_STATUS_CONFLICT",
    )
    classified = normalize_historical_termination(record)
    assert classified.terminal_class is HistoricalTerminalClass.AMBIGUOUS
    assert classified.anomaly_code == "SOURCE_STATUS_CONFLICT"
    batch = build_retirement_observations((classified,), _CUTOFF)
    assert not batch.observations
    exclusion = batch.exclusions[0]
    assert exclusion.exclusion_reason == (
        f"EXCLUDED:AMBIGUOUS:{status.value.upper()}:SOURCE_STATUS_CONFLICT"
    )
    assert exclusion.recency_weight is not None


def test_window_weights_and_strict_cutoff_are_exact() -> None:
    ages = (0, 1, 730, 1_826, 1_827)
    classified = tuple(
        normalize_historical_termination(
            _termination_input(
                match_id=f"age-{age}",
                status=OfficialTerminalStatus.NORMAL_COMPLETION,
                age_days=age,
            )
        )
        for age in ages
    )
    batch = build_retirement_observations(classified, _CUTOFF)
    weights = {
        row.match_id: row.recency_weight
        for row in batch.observations
        if row.player_id.endswith("player-a")
    }
    assert weights["age-0"] == 1.0
    assert weights["age-1"] == pytest.approx(2 ** (-1 / 730), abs=1e-15)
    assert weights["age-730"] == 0.5
    assert weights["age-1826"] == pytest.approx(2 ** (-1826 / 730), abs=1e-15)
    assert "age-1827" not in weights
    assert batch.exclusions[0].exclusion_reason == "OUTSIDE_1826_DAY_WINDOW"
    assert batch.exclusions[0].recency_weight == 0.0

    at_cutoff = normalize_historical_termination(
        _termination_input(
            match_id="at-cutoff",
            status=OfficialTerminalStatus.NORMAL_COMPLETION,
            available_at=_CUTOFF,
        )
    )
    cutoff_batch = build_retirement_observations((at_cutoff,), _CUTOFF)
    assert not cutoff_batch.observations
    assert cutoff_batch.exclusions[0].exclusion_reason == "AT_OR_AFTER_INFORMATION_CUTOFF"
    assert cutoff_batch.exclusions[0].recency_weight is None


def test_append_only_terminal_corrections_obey_strict_fit_cutoffs() -> None:
    original = _termination_input(
        match_id="corrected",
        status=OfficialTerminalStatus.NORMAL_COMPLETION,
        available_at=_CUTOFF - timedelta(hours=3),
    )
    correction = _termination_input(
        match_id="corrected",
        status=OfficialTerminalStatus.RETIREMENT,
        retiree="player-a",
        available_at=_CUTOFF - timedelta(hours=2),
    )
    later_correction = _termination_input(
        match_id="corrected",
        status=OfficialTerminalStatus.WALKOVER_OR_PRESTART_WITHDRAWAL,
        started=False,
        winner=None,
        available_at=_CUTOFF + timedelta(hours=1),
    )

    before_correction = normalize_historical_terminations_before_cutoff(
        (original, correction, later_correction),
        _CUTOFF - timedelta(hours=2, minutes=30),
    )
    assert before_correction[0].terminal_class is HistoricalTerminalClass.NORMAL_COMPLETION

    after_correction = normalize_historical_terminations_before_cutoff(
        (original, correction, later_correction), _CUTOFF
    )
    assert after_correction[0].terminal_class is HistoricalTerminalClass.STARTED_RETIREMENT
    batch = build_retirement_observations(after_correction, _CUTOFF)
    assert sorted(item.response for item in batch.observations) == [0, 1]


def test_same_timestamp_conflicting_corrections_fail_closed() -> None:
    first = _termination_input(
        match_id="conflicting-correction",
        status=OfficialTerminalStatus.NORMAL_COMPLETION,
    )
    second = _termination_input(
        match_id="conflicting-correction",
        status=OfficialTerminalStatus.RETIREMENT,
        retiree="player-a",
    )
    with pytest.raises(RetirementModelError, match="conflicting terminal corrections"):
        normalize_historical_terminations_before_cutoff((first, second), _CUTOFF)


def test_recency_weight_rejects_noninteger_and_out_of_window_ages() -> None:
    assert RETIREMENT_WINDOW_DAYS == 1_826
    assert RETIREMENT_HALF_LIFE_DAYS == 730
    with pytest.raises(TypeError):
        retirement_recency_weight(True)
    with pytest.raises(TypeError):
        retirement_recency_weight(1.5)  # type: ignore[arg-type]
    with pytest.raises(RetirementModelError):
        retirement_recency_weight(-1)
    with pytest.raises(RetirementModelError):
        retirement_recency_weight(1_827)


def _coverage(tour: Tour, *, complete: bool = True) -> RetirementSourceCoverage:
    return RetirementSourceCoverage(
        tour=tour,
        complete=complete,
        assertion_id=f"coverage-{tour.value.lower()}",
        verified_at_utc=_CUTOFF - timedelta(days=1),
        details="Pinned fixture coverage independently verified.",
    )


def _fit(
    records: tuple[HistoricalTerminationInput, ...],
    *,
    tour: Tour,
    coverage_complete: bool = True,
) -> RetirementFitArtifact:
    batch = build_retirement_observations(
        tuple(normalize_historical_termination(item) for item in records), _CUTOFF
    )
    return fit_retirement_artifact(
        batch,
        tour=tour,
        source_manifest_id="fixture-manifest-v1",
        source_manifest_sha256=_HASH_A,
        source_coverage=_coverage(tour, complete=coverage_complete),
        fitted_at_utc=_CUTOFF + timedelta(hours=1),
        software_version="tennis-model-test",
        config_sha256=_HASH_B,
        data_sha256=_HASH_C,
        code_sha256=_HASH_D,
        deterministic_test_result_sha256=_HASH_E,
    )


def _coverage_fixture_records(
    tour: Tour, *, retire_first: bool = False
) -> tuple[HistoricalTerminationInput, ...]:
    records: list[HistoricalTerminationInput] = []
    for index in range(250):
        retirement = retire_first and index == 0
        records.append(
            _termination_input(
                match_id=f"{tour.value.lower()}-{index:03d}",
                status=(
                    OfficialTerminalStatus.RETIREMENT
                    if retirement
                    else OfficialTerminalStatus.NORMAL_COMPLETION
                ),
                age_days=0,
                tour=tour,
                retiree="player-a" if retirement else None,
            )
        )
    return tuple(records)


def test_tour_fits_are_separate_and_posterior_formulas_are_exact() -> None:
    atp = _fit(_coverage_fixture_records(Tour.ATP, retire_first=True), tour=Tour.ATP)
    wta = _fit(_coverage_fixture_records(Tour.WTA), tour=Tour.WTA)
    assert atp.tour_starts_n == wta.tour_starts_n == 500.0
    assert atp.tour_retirements_y == 1.0
    assert wta.tour_retirements_y == 0.0
    assert atp.tour_baseline_rho == pytest.approx(1.5 / 501, rel=1e-12)
    assert wta.tour_baseline_rho == pytest.approx(0.5 / 501, rel=1e-12)
    assert atp.production_eligible and wta.production_eligible

    retiring = player_retirement_posterior(atp, "atp-000-player-a")
    assert retiring.retirements_y == 1.0
    assert retiring.starts_n == 1.0
    assert retiring.alpha == pytest.approx(
        RETIREMENT_PRIOR_EFFECTIVE_STARTS * atp.tour_baseline_rho + 1.0,
        rel=1e-12,
    )
    assert retiring.beta == pytest.approx(
        RETIREMENT_PRIOR_EFFECTIVE_STARTS * (1 - atp.tour_baseline_rho),
        rel=1e-12,
    )
    assert retiring.mean_rho == pytest.approx(retiring.alpha / (retiring.alpha + retiring.beta))

    zero_retirements = player_retirement_posterior(wta, "wta-000-player-a")
    assert zero_retirements.starts_n == 1.0
    assert zero_retirements.retirements_y == 0.0
    assert zero_retirements.mean_rho > 0.0

    unseen = player_retirement_posterior(atp, "unseen-player")
    assert unseen.history_band is RetirementHistoryBand.NO_HISTORY
    assert unseen.starts_n == unseen.retirements_y == 0.0
    assert unseen.mean_rho == pytest.approx(atp.tour_baseline_rho, rel=1e-12)


def test_retirement_artifact_preserves_specific_exclusion_counts() -> None:
    anomalous = _termination_input(
        match_id="source-conflict",
        status=OfficialTerminalStatus.NORMAL_COMPLETION,
        anomaly="SOURCE_STATUS_CONFLICT",
    )
    artifact = _fit(
        (*_coverage_fixture_records(Tour.ATP), anomalous),
        tour=Tour.ATP,
    )
    reason = "EXCLUDED:AMBIGUOUS:NORMAL_COMPLETION:SOURCE_STATUS_CONFLICT"
    count = next(item for item in artifact.excluded_counts if item.reason == reason)
    weight = retirement_recency_weight(10)
    assert count.match_count == 1
    assert count.match_weight == pytest.approx(weight, abs=1e-15)
    assert count.player_start_count == 2
    assert count.player_start_weight == pytest.approx(2 * weight, abs=1e-15)


def test_coverage_gate_fails_closed_without_cross_tour_or_emergency_rate() -> None:
    small = _fit(
        (
            _termination_input(
                match_id="small",
                status=OfficialTerminalStatus.NORMAL_COMPLETION,
            ),
        ),
        tour=Tour.ATP,
    )
    assert small.tour_starts_n == 2 * retirement_recency_weight(10)
    assert not small.weighted_start_coverage_gate_passed
    assert not small.production_eligible
    with pytest.raises(RetirementCoverageError, match="500"):
        small.require_production_coverage()
    with pytest.raises(RetirementCoverageError):
        player_retirement_posterior(small, "small-player-a")

    incomplete = _fit(
        _coverage_fixture_records(Tour.ATP),
        tour=Tour.ATP,
        coverage_complete=False,
    )
    assert incomplete.tour_starts_n >= RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS
    assert incomplete.weighted_start_coverage_gate_passed
    assert not incomplete.production_eligible
    with pytest.raises(RetirementCoverageError, match="coverage"):
        player_retirement_posterior(
            incomplete,
            "never-seen",
            require_production_coverage=False,
        )


def test_current_exact_dated_fit_is_eligible_but_cannot_invent_no_history() -> None:
    original = _fit(
        _coverage_fixture_records(Tour.ATP, retire_first=True),
        tour=Tour.ATP,
        coverage_complete=False,
    )
    coverage = RetirementSourceCoverage(
        tour=Tour.ATP,
        complete=False,
        assertion_id="current-exact-dated-atp",
        verified_at_utc=_CUTOFF - timedelta(days=1),
        details="All admitted rows are exact-dated; retained undated rows remain excluded.",
        fit_input_date_eligibility_verified=True,
        historical_exact_date_coverage_complete=False,
        included_exact_dated_matches=250,
        excluded_undated_matches=17,
        included_exact_dated_player_starts=500,
        excluded_undated_player_starts=34,
        source_sha256s=(_HASH_A,),
        crosswalk_sha256s=(_HASH_B,),
        eligibility_rule_version="exact-dated-fit-inputs/v1",
    )
    amended = reissue_retirement_fit_eligibility_metadata(original, coverage)
    assert amended.production_eligible
    assert amended.tour_starts_n == original.tour_starts_n
    assert amended.tour_retirements_y == original.tour_retirements_y
    assert amended.tour_baseline_rho == original.tour_baseline_rho
    assert amended.player_statistics == original.player_statistics
    amended.require_production_coverage()
    known = player_retirement_posterior(amended, "atp-000-player-a")
    assert known.history_band is RetirementHistoryBand.SPARSE
    with pytest.raises(RetirementCoverageError, match="no-history"):
        player_retirement_posterior(amended, "unseen-player")


@pytest.mark.parametrize(
    ("weighted_starts", "expected"),
    [
        (0.0, RetirementHistoryBand.NO_HISTORY),
        (1e-12, RetirementHistoryBand.SPARSE),
        (24.999, RetirementHistoryBand.SPARSE),
        (25.0, RetirementHistoryBand.INTERMEDIATE),
        (99.999, RetirementHistoryBand.INTERMEDIATE),
        (100.0, RetirementHistoryBand.SUBSTANTIAL),
    ],
)
def test_history_bands_do_not_change_the_continuous_estimator(
    weighted_starts: float,
    expected: RetirementHistoryBand,
) -> None:
    assert retirement_history_band(weighted_starts) is expected


def test_retirement_artifact_round_trip_is_canonical_and_tamper_evident(
    tmp_path: Path,
) -> None:
    artifact = _fit(_coverage_fixture_records(Tour.ATP), tour=Tour.ATP)
    persisted = write_retirement_fit_artifact(artifact, tmp_path)
    assert persisted.artifact == artifact
    assert write_retirement_fit_artifact(artifact, tmp_path) == persisted
    raw = persisted.artifact_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["artifact_id"] == artifact.artifact_id
    parsed["software_version"] = "tampered"
    persisted.artifact_path.write_text(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RetirementArtifactIntegrityError):
        load_retirement_fit_artifact(persisted.directory)


@pytest.mark.parametrize("rho", [1e-6, 0.001, 0.01, 0.10, 0.50])
def test_22_game_intensity_mapping_identity(rho: float) -> None:
    mapped = retirement_probability_to_intensity(rho)
    assert RETIREMENT_REFERENCE_GAMES == 22
    reconstructed = 1.0 - (1.0 - mapped.discrete_hazard) ** 22
    assert reconstructed == pytest.approx(rho, abs=1e-12)
    assert -math.expm1(-22 * mapped.intensity_lambda) == pytest.approx(rho, abs=1e-12)


def test_intensity_edge_and_invalid_values() -> None:
    zero = retirement_probability_to_intensity(0.0)
    assert zero.intensity_lambda == zero.discrete_hazard == 0.0
    one = retirement_probability_to_intensity(1.0)
    assert one.rho == math.nextafter(1.0, 0.0)
    assert math.isfinite(one.intensity_lambda)
    for invalid in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises((RetirementModelError, ValueError)):
            retirement_probability_to_intensity(invalid)


def _scenario(
    *,
    scenario_id: str,
    eta: float,
    weight: float | None,
    publication_at: datetime | None = None,
) -> RetirementScenario:
    return RetirementScenario(
        scenario_id=scenario_id,
        scenario_version="health-scenarios/v1",
        named_state=scenario_id,
        player_id="p1",
        central=scenario_id == "central",
        log_hazard_ratio=eta,
        weight=weight,
        source_id="official-statement",
        source_sha256=_HASH_A,
        observation_at_utc=_CUTOFF - timedelta(hours=3),
        publication_at_utc=publication_at or (_CUTOFF - timedelta(hours=2)),
        authoring_method="Predeclared evidence translation.",
    )


def test_versioned_scenario_mixture_is_cutoff_safe_and_drawn_once() -> None:
    central = _scenario(scenario_id="central", eta=0.0, weight=0.75)
    limited = _scenario(scenario_id="limited", eta=math.log(2), weight=0.25)
    mixture = RetirementScenarioMixture(
        mixture_id="p1-health-mixture-v1",
        player_id="p1",
        information_cutoff_utc=_CUTOFF,
        scenarios=(central, limited),
    )
    first_rng = default_rng(418)
    second_rng = default_rng(418)
    assert draw_retirement_scenario(mixture, first_rng) == draw_retirement_scenario(
        mixture, second_rng
    )
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    posterior = PlayerRetirementPosterior(
        player_id="p1",
        tour=Tour.ATP,
        retirements_y=0.0,
        starts_n=0.0,
        alpha=2.0,
        beta=98.0,
        mean_rho=0.02,
        history_band=RetirementHistoryBand.NO_HISTORY,
        artifact_id=_HASH_A,
    )
    manual_rng = default_rng(991)
    selected = draw_retirement_scenario(mixture, manual_rng)
    manual = draw_player_retirement_path(posterior, selected, manual_rng)
    combined_rng = default_rng(991)
    combined = draw_mixture_player_retirement_path(posterior, mixture, combined_rng)
    assert combined == manual
    assert combined_rng.bit_generator.state == manual_rng.bit_generator.state
    with pytest.raises(ValidationError, match="strictly before"):
        RetirementScenarioMixture(
            mixture_id="future-evidence",
            player_id="p1",
            information_cutoff_utc=_CUTOFF,
            scenarios=(
                _scenario(
                    scenario_id="future",
                    eta=0.1,
                    weight=1.0,
                    publication_at=_CUTOFF,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="sum to 1"):
        RetirementScenarioMixture(
            mixture_id="bad-weights",
            player_id="p1",
            information_cutoff_utc=_CUTOFF,
            scenarios=(
                _scenario(scenario_id="a", eta=0.0, weight=0.4),
                _scenario(scenario_id="b", eta=0.0, weight=0.4),
            ),
        )


def test_player_path_draw_uses_beta_posterior_and_only_named_multiplier() -> None:
    posterior = PlayerRetirementPosterior(
        player_id="p1",
        tour=Tour.ATP,
        retirements_y=0.0,
        starts_n=0.0,
        alpha=2.0,
        beta=98.0,
        mean_rho=0.02,
        history_band=RetirementHistoryBand.NO_HISTORY,
        artifact_id=_HASH_A,
    )
    scenario = _scenario(scenario_id="limited", eta=math.log(1.5), weight=None)
    first = draw_player_retirement_path(posterior, scenario, default_rng(9182))
    second = draw_player_retirement_path(posterior, scenario, default_rng(9182))
    assert first == second
    assert first.adjusted_intensity_lambda == pytest.approx(
        1.5 * first.base_intensity_lambda, rel=1e-12
    )
    assert first.adjusted_discrete_hazard == pytest.approx(
        -math.expm1(-first.adjusted_intensity_lambda), abs=1e-15
    )


def test_scenarios_reject_nan_infinite_underflow_and_overflow() -> None:
    for eta in (math.nan, math.inf, -math.inf, -1_000.0, 1_000.0):
        with pytest.raises(ValidationError):
            _scenario(scenario_id="invalid", eta=eta, weight=None)
    with pytest.raises(ValidationError, match="central"):
        RetirementScenario(
            scenario_id="central-invalid",
            scenario_version="health-scenarios/v1",
            named_state="central",
            player_id="p1",
            central=True,
            log_hazard_ratio=0.1,
            source_id="frozen-config",
            source_sha256=_HASH_A,
            observation_at_utc=_CUTOFF - timedelta(hours=2),
            publication_at_utc=_CUTOFF - timedelta(hours=1),
            authoring_method="Frozen central scenario.",
        )


def test_competing_risk_grid_mass_and_swap_symmetry() -> None:
    grid = (0.0, 1e-12, 1e-6, 0.001, 0.1, 10.0, 100.0)
    for intensity_a in grid:
        for intensity_b in grid:
            probabilities = competing_retirement_probabilities(intensity_a, intensity_b)
            values = (
                probabilities.p_no_retirement,
                probabilities.p_player_a_retires,
                probabilities.p_player_b_retires,
            )
            assert all(0.0 <= item <= 1.0 for item in values)
            assert math.fsum(values) == pytest.approx(1.0, abs=1e-15)
            swapped = competing_retirement_probabilities(intensity_b, intensity_a)
            assert swapped.p_no_retirement == pytest.approx(
                probabilities.p_no_retirement, abs=1e-15
            )
            assert swapped.p_player_a_retires == pytest.approx(
                probabilities.p_player_b_retires, abs=1e-15
            )
            assert swapped.p_player_b_retires == pytest.approx(
                probabilities.p_player_a_retires, abs=1e-15
            )


def test_competing_draw_zero_hazard_does_not_advance_rng() -> None:
    rng = default_rng(722)
    before = copy.deepcopy(rng.bit_generator.state)
    result = draw_competing_retirement(0.0, 0.0, rng)
    assert result.outcome is CompetingRetirementOutcome.NO_RETIREMENT
    assert rng.bit_generator.state == before

    nonzero = draw_competing_retirement(0.0, 0.02, rng)
    assert nonzero.outcome in {
        CompetingRetirementOutcome.NO_RETIREMENT,
        CompetingRetirementOutcome.PLAYER_B_RETIRES,
    }
    assert rng.bit_generator.state != before


def test_competing_risk_rejects_invalid_or_overflowed_intensities() -> None:
    for pair in ((-0.1, 0.2), (math.nan, 0.2), (math.inf, 0.2), (1e308, 1e308)):
        with pytest.raises((RetirementModelError, ValueError)):
            competing_retirement_probabilities(*pair)


@pytest.mark.slow_statistical
def test_statistical_one_million_22_boundary_exposures_reproduce_rho() -> None:
    rho = 0.01
    n_draws = 1_000_000
    chunk_size = 100_000
    hazard = retirement_probability_to_intensity(rho).discrete_hazard
    rng = default_rng(2026083001)
    retired = 0
    for _ in range(n_draws // chunk_size):
        retired += int(
            np.any(
                rng.random((chunk_size, RETIREMENT_REFERENCE_GAMES)) < hazard,
                axis=1,
            ).sum()
        )
    empirical = retired / n_draws
    tolerance = 5 * math.sqrt(rho * (1 - rho) / n_draws) + 1e-6
    assert abs(empirical - rho) <= tolerance


@pytest.mark.slow_statistical
def test_statistical_one_million_competing_draws_match_analytic_mass() -> None:
    n_draws = 1_000_000
    probabilities = competing_retirement_probabilities(0.003, 0.007)
    expected = np.array(
        [
            probabilities.p_no_retirement,
            probabilities.p_player_a_retires,
            probabilities.p_player_b_retires,
        ]
    )
    uniforms = default_rng(2026083002).random(n_draws)
    first_threshold = expected[0]
    second_threshold = expected[0] + expected[1]
    observed = (
        np.array(
            [
                np.count_nonzero(uniforms < first_threshold),
                np.count_nonzero((uniforms >= first_threshold) & (uniforms < second_threshold)),
                np.count_nonzero(uniforms >= second_threshold),
            ]
        )
        / n_draws
    )
    tolerances = 5 * np.sqrt(expected * (1 - expected) / n_draws) + 1e-6
    assert np.all(np.abs(observed - expected) <= tolerances)


@pytest.mark.slow_statistical
def test_statistical_one_million_beta_draws_match_mean_and_variance() -> None:
    alpha = 2.75
    beta = 97.25
    n_draws = 1_000_000
    draws = default_rng(2026083003).beta(alpha, beta, size=n_draws)
    total = alpha + beta
    analytic_mean = alpha / total
    analytic_variance = alpha * beta / (total**2 * (total + 1))
    excess_kurtosis = (
        6
        * ((alpha - beta) ** 2 * (total + 1) - alpha * beta * (total + 2))
        / (alpha * beta * (total + 2) * (total + 3))
    )
    fourth_central = (excess_kurtosis + 3.0) * analytic_variance**2
    mean_standard_error = math.sqrt(analytic_variance / n_draws)
    variance_standard_error = math.sqrt(
        (fourth_central - ((n_draws - 3) / (n_draws - 1)) * analytic_variance**2) / n_draws
    )
    assert abs(float(np.mean(draws)) - analytic_mean) <= 5 * mean_standard_error
    assert abs(float(np.var(draws, ddof=1)) - analytic_variance) <= 5 * variance_standard_error
