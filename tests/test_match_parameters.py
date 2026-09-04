from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from serve_model_test_helpers import (
    TEST_CUTOFF,
    make_model_config,
    make_provenance,
    synthetic_component_counts,
)

from tennis_model.calibration import (
    CalibrationLedger,
    LedgerError,
    OfficialHistoricalOutcome,
    ledger_rows_from_settlement,
    settle_historical_lock,
)
from tennis_model.estimation.artifacts import FitArtifact, write_fit_artifact
from tennis_model.estimation.duration_model import (
    FLOOR_DURATION_DISPLAY_POLICY,
    UNRESOLVED_DURATION_DISPLAY_POLICY,
    DurationCoefficient,
    DurationDraw,
    DurationFitArtifact,
    DurationFitDiagnostics,
    DurationPathExposure,
    PersistedDurationFitArtifact,
)
from tennis_model.estimation.inactivity import (
    InactivityAdjustmentState,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityRecord,
    InactivityTerminalStatus,
    LastEligibleMatch,
    PlayedPointEvidence,
    create_inactivity_configuration_artifact,
    inactivity_factors,
)
from tennis_model.estimation.retirement import (
    HistoricalTerminationInput,
    OfficialTerminalStatus,
    RetirementSourceCoverage,
    StartedEvidence,
    build_retirement_observations,
    fit_retirement_artifact,
    normalize_historical_termination,
    write_retirement_fit_artifact,
)
from tennis_model.estimation.serve_components import (
    FittedServeComponent,
    FutureMatchContext,
    ModelDataError,
    PosteriorApproximation,
    ServeComponent,
    fit_all_serve_components,
    project_component_parameters,
)
from tennis_model.estimation.snapshot import (
    DurationArtifactReference,
    ModelSnapshot,
    ModelSnapshotError,
    create_model_snapshot,
    load_snapshot_duration_artifact,
    load_snapshot_fits,
)
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.card import render_locked_match_card
from tennis_model.locking.models import (
    CodeProvenance,
    InformationBundle,
    LockRevisionReason,
    RetainedArtifactRecord,
    SourceManifestProvenance,
)
from tennis_model.locking.path_counts import PathCountPolicy
from tennis_model.locking.service import (
    LockCreationError,
    create_prediction_lock,
    reproduce_prediction_lock,
)
from tennis_model.locking.store import LockIntegrityError, LockStore
from tennis_model.props import CANONICAL_SETTLEMENT_POLICY, ComparisonOperator
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RowDateSemantics,
    SourceManifest,
    Tour,
    TourCoverage,
)
from tennis_model.simulation.match import (
    DURATION_MIN,
    MATCH_WIN,
    _simulate_one_path,
    simulate_matches,
)
from tennis_model.simulation.parallel import simulate_matches_parallel
from tennis_model.simulation.parameters import (
    BetaComponentParameters,
    MatchCondition,
    MatchContext,
    MatchParameterDistribution,
    MatchParameterError,
    MatchParameterRecord,
    PerformanceDependenceSpec,
    SeedReference,
    ServingDirectionParameterDraw,
    derive_match_seed_plan,
    estimate_match,
    generator_from_seed_reference,
    restore_match_parameter_distribution,
    sample_beta_probability,
    sample_match_performance,
    sample_matchup_parameters,
    sample_posterior_parameters,
    sample_serve_performance,
)
from tennis_model.simulation.point import generate_service_point

_INTERCEPTS = {
    ServeComponent.F: 0.55,
    ServeComponent.A: -2.0,
    ServeComponent.Q1: 0.65,
    ServeComponent.D: -2.3,
    ServeComponent.Q2: 0.2,
}


@dataclass(frozen=True, slots=True)
class _MatchFixture:
    fits: dict[ServeComponent, FittedServeComponent]
    artifacts: dict[ServeComponent, FitArtifact]
    snapshot: ModelSnapshot
    context: MatchContext
    distribution: MatchParameterDistribution


@pytest.fixture(scope="module")
def match_fixture(tmp_path_factory: pytest.TempPathFactory) -> _MatchFixture:
    frames = []
    for index, component in enumerate(ServeComponent):
        frame = synthetic_component_counts(
            component,
            repetitions=4,
            trials=120,
            intercept=_INTERCEPTS[component],
            server_effects={"p0": 0.8, "p1": -0.65, "p2": 0.3, "p3": -0.25},
            returner_effects={"p0": 0.65, "p1": -0.7, "p2": 0.4, "p3": -0.35},
            hard_deviations={"p0": 0.45, "p1": -0.4, "p2": 0.2, "p3": -0.15},
            kappa=80.0,
            seed=6100 + index,
        )
        frame.loc[frame.index % 2 == 1, "event"] = "Other Open"
        frames.append(frame)
    config = make_model_config(
        include_indoor_hard=True,
        event_components=(ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2),
    )
    fits = fit_all_serve_components(
        pd.concat(frames, ignore_index=True),
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
        provenance=make_provenance("milestone-5"),
    )
    artifact_root = tmp_path_factory.mktemp("m5-fit-artifacts")
    artifacts = {
        component: write_fit_artifact(fits[component], artifact_root)
        for component in ServeComponent
    }
    snapshot = create_model_snapshot(artifacts)
    context = MatchContext(
        player_a_id="p0",
        player_b_id="p1",
        tour=Tour.ATP,
        event="Synthetic Open",
        round="R128",
        scheduled_start_utc=TEST_CUTOFF + timedelta(days=10),
        best_of=3,
        indoor=False,
        conditions=(MatchCondition(name="weather_scenario", value="dry"),),
        information_cutoff_utc=TEST_CUTOFF + timedelta(days=1),
        information_scenario_id="historical-central",
    )
    return _MatchFixture(
        fits=fits,
        artifacts=artifacts,
        snapshot=snapshot,
        context=context,
        distribution=estimate_match(snapshot, context),
    )


def _future_context(
    fixture: _MatchFixture,
    *,
    server: str = "p0",
    receiver: str = "p1",
    surface: str = "Hard",
    indoor: bool = False,
    best_of: int = 3,
    event: str = "Synthetic Open",
    event_year: int = 2026,
) -> FutureMatchContext:
    return FutureMatchContext(
        serving_player_id=server,
        returning_player_id=receiver,
        tour=Tour.ATP,
        surface=surface,
        indoor=indoor,
        event=event,
        event_year=event_year,
        match_date_utc=fixture.context.scheduled_start_utc,
        information_cutoff_utc=fixture.snapshot.data_cutoff_utc,
        best_of=best_of,  # type: ignore[arg-type]
    )


def _component_means(direction: object) -> tuple[float, ...]:
    summary = direction.map_distribution  # type: ignore[attr-defined]
    return (
        summary.first_serve_in.map_mean,
        summary.ace_given_first_in.map_mean,
        summary.returnable_first_win.map_mean,
        summary.double_fault_given_second_opp.map_mean,
        summary.playable_second_win.map_mean,
    )


def _performance_values(draw: object) -> tuple[float, ...]:
    return (
        draw.first_serve_in,  # type: ignore[attr-defined]
        draw.ace_given_first_in,  # type: ignore[attr-defined]
        draw.returnable_first_win,  # type: ignore[attr-defined]
        draw.double_fault_given_second_opp,  # type: ignore[attr-defined]
        draw.playable_second_win,  # type: ignore[attr-defined]
    )


def _inactivity_record(
    fixture: _MatchFixture,
    *,
    player_id: str,
    days: int,
    source_manifest_id: str = "fixture-manifest-v1",
    source_manifest_sha256: str = "a" * 64,
) -> InactivityRecord:
    context = fixture.context
    assert context.scheduled_start_local_date is not None
    gap, multiplier, inflation = inactivity_factors(days)
    return InactivityRecord(
        player_id=player_id,
        tour=context.tour,
        scheduled_start_local_date=context.scheduled_start_local_date,
        information_cutoff_utc=context.information_cutoff_utc,
        coverage=InactivityCoverageAssertion(
            state=InactivityCoverageState.VERIFIED_COMPLETE,
            source_manifest_id=source_manifest_id,
            source_manifest_sha256=source_manifest_sha256,
            canonical_player_id=player_id,
            asserted_at_utc=context.information_cutoff_utc - timedelta(hours=2),
        ),
        last_eligible_match=LastEligibleMatch(
            match_id=f"last-{player_id}-{days}",
            match_date_local=date.fromordinal(
                context.scheduled_start_local_date.toordinal() - days
            ),
            terminal_status=InactivityTerminalStatus.NORMAL_COMPLETION,
            started_evidence=(PlayedPointEvidence.LEGAL_SCORE_WITH_COMPLETED_GAME_OR_TIEBREAK,),
            source_pin=f"fixture-row-{player_id}",
            source_sha256="a" * 64,
            available_at_utc=context.information_cutoff_utc - timedelta(hours=2),
        ),
        inactivity_days=days,
        post_threshold_days=gap,
        hard_deviation_multiplier=multiplier,
        variance_inflation_factor=inflation,
        cold_start=False,
    )


def _amended_fixture(
    fixture: _MatchFixture,
    tmp_path: Path,
    *,
    days_a: int,
    days_b: int,
    source_manifest_id: str = "fixture-manifest-v1",
    source_manifest_sha256: str = "a" * 64,
) -> tuple[ModelSnapshot, MatchContext, MatchParameterDistribution]:
    cutoff = fixture.snapshot.data_cutoff_utc
    rows = tuple(
        HistoricalTerminationInput(
            match_id=f"retirement-fit-{index:03d}",
            tour=Tour.ATP,
            player_a_id="p0",
            player_b_id="p1",
            match_date=cutoff.date(),
            official_status=OfficialTerminalStatus.NORMAL_COMPLETION,
            started_evidence=(StartedEvidence.LEGAL_SCORE_COMPLETED_GAME,),
            advancing_winner_id="p0",
            source_id="fixture-source",
            source_sha256="a" * 64,
            available_at_utc=cutoff - timedelta(microseconds=1),
        )
        for index in range(250)
    )
    batch = build_retirement_observations(
        tuple(normalize_historical_termination(item) for item in rows), cutoff
    )
    retirement = fit_retirement_artifact(
        batch,
        tour=Tour.ATP,
        source_manifest_id=source_manifest_id,
        source_manifest_sha256=source_manifest_sha256,
        source_coverage=RetirementSourceCoverage(
            tour=Tour.ATP,
            complete=True,
            assertion_id="fixture-coverage",
            verified_at_utc=cutoff - timedelta(days=1),
            details="Complete synthetic ATP coverage.",
        ),
        fitted_at_utc=cutoff + timedelta(hours=12),
        software_version="test",
        config_sha256="b" * 64,
        data_sha256="c" * 64,
        code_sha256="d" * 64,
        deterministic_test_result_sha256="e" * 64,
    )
    persisted = write_retirement_fit_artifact(retirement, tmp_path / "retirement")
    inactivity_config = create_inactivity_configuration_artifact(
        config_sha256="f" * 64,
        code_sha256="d" * 64,
    )
    snapshot = create_model_snapshot(
        fixture.artifacts,
        retirement_artifact=persisted,
        inactivity_configuration=inactivity_config,
    )
    context = MatchContext.model_validate(
        fixture.context.model_dump(mode="python")
        | {"scheduled_start_local_date": fixture.context.scheduled_start_utc.date()}
    )
    amended_fixture = _MatchFixture(
        fits=fixture.fits,
        artifacts=fixture.artifacts,
        snapshot=snapshot,
        context=context,
        distribution=fixture.distribution,
    )
    records = (
        _inactivity_record(
            amended_fixture,
            player_id="p0",
            days=days_a,
            source_manifest_id=source_manifest_id,
            source_manifest_sha256=source_manifest_sha256,
        ),
        _inactivity_record(
            amended_fixture,
            player_id="p1",
            days=days_b,
            source_manifest_id=source_manifest_id,
            source_manifest_sha256=source_manifest_sha256,
        ),
    )
    return snapshot, context, estimate_match(snapshot, context, inactivity_records=records)


def test_match_simulator_consumes_real_match_parameter_distribution(
    match_fixture: _MatchFixture,
) -> None:
    seed = np.random.SeedSequence(7601)

    first = simulate_matches(
        match_fixture.distribution,
        n_paths=3,
        seed=seed,
        first_server_id=match_fixture.context.player_a_id,
    )
    second = simulate_matches(
        match_fixture.distribution,
        n_paths=3,
        seed=seed,
        first_server_id=match_fixture.context.player_a_id,
    )

    assert first.paths == second.paths
    assert seed.n_children_spawned == 0
    assert all(path.first_server_id == match_fixture.context.player_a_id for path in first.paths)
    assert all(path.point_trace is None for path in first.paths)
    assert first.provenance["snapshot_id"] == match_fixture.snapshot.snapshot_id


def test_match_simulator_path_ranges_reproduce_one_full_seed_prefix(
    match_fixture: _MatchFixture,
) -> None:
    full = simulate_matches(match_fixture.distribution, n_paths=8, seed=7602)
    first = simulate_matches(match_fixture.distribution, n_paths=3, seed=7602)
    extension = simulate_matches(
        match_fixture.distribution,
        n_paths=5,
        seed=7602,
        path_start=3,
    )

    assert first.paths + extension.paths == full.paths
    assert extension.provenance["path_start"] == 3


def test_checkpointed_parallel_simulation_reproduces_serial_paths_and_resumes(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    serial = simulate_matches(match_fixture.distribution, n_paths=8, seed=7603)
    checkpoint_dir = tmp_path / "parallel-checkpoints"
    parallel = simulate_matches_parallel(
        match_fixture.distribution,
        n_paths=8,
        seed=7603,
        workers=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_paths=2,
        duration_display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
    )
    resumed = simulate_matches_parallel(
        match_fixture.distribution,
        n_paths=8,
        seed=7603,
        workers=2,
        checkpoint_dir=checkpoint_dir,
        checkpoint_paths=2,
        duration_display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
    )

    assert parallel.paths == serial.paths
    assert resumed.paths == serial.paths
    assert parallel.seed_id == serial.seed_id
    assert parallel.provenance["parallel_workers"] == 2
    assert parallel.provenance["checkpoint_paths"] == 2
    assert len(tuple(checkpoint_dir.glob("chunk-*.pickle"))) == 4


def test_c6_active_records_preserve_parameter_and_performance_draws_bit_for_bit(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    _snapshot, _context, amended = _amended_fixture(
        match_fixture,
        tmp_path,
        days_a=90,
        days_b=90,
    )
    legacy = sample_match_performance(match_fixture.distribution, np.random.SeedSequence(20260830))
    active = sample_match_performance(amended, np.random.SeedSequence(20260830))
    assert active.player_a_serving == legacy.player_a_serving
    assert active.player_b_serving == legacy.player_b_serving
    assert active.matchup_parameters.components == legacy.matchup_parameters.components
    assert _component_means(amended.player_a_serving) == _component_means(
        match_fixture.distribution.player_a_serving
    )
    assert _component_means(amended.player_b_serving) == _component_means(
        match_fixture.distribution.player_b_serving
    )
    assert active.retirement_draws
    assert amended.inactivity is not None
    assert all(
        adjustment.unadjusted_mean_covariance_sha256 == adjustment.adjusted_mean_covariance_sha256
        for adjustment in amended.inactivity.component_adjustments
    )
    restored = restore_match_parameter_distribution(amended.canonical_json())
    assert restored.to_record() == amended.to_record()
    legacy_path = _simulate_one_path(
        "p0",
        "p1",
        best_of=3,
        first_server_id="p0",
        player_a_performance=legacy.player_a_serving,
        player_b_performance=legacy.player_b_serving,
        rng=np.random.default_rng(8080),
    )
    active_path = _simulate_one_path(
        "p0",
        "p1",
        best_of=3,
        first_server_id="p0",
        player_a_performance=active.player_a_serving,
        player_b_performance=active.player_b_serving,
        rng=np.random.default_rng(8080),
    )
    assert active_path == legacy_path


def test_v3_duration_snapshot_and_match_record_round_trip(
    match_fixture: _MatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_v2, context, amended = _amended_fixture(
        match_fixture,
        tmp_path,
        days_a=90,
        days_b=90,
    )
    duration = DurationFitArtifact.model_construct(
        schema_version="duration-fit-artifact/v1",
        framework_version="v1.0",
        artifact_id="9" * 64,
        tour=context.tour,
        information_cutoff_utc=snapshot_v2.data_cutoff_utc,
        fit_cutoff_utc=snapshot_v2.data_cutoff_utc,
        fitted_at_utc=snapshot_v2.fitted_at_utc,
        diagnostics=DurationFitDiagnostics.model_construct(converged=True),
    )
    reference = DurationArtifactReference(
        artifact_id=duration.artifact_id,
        directory=(tmp_path / "duration-artifact").resolve(),
        tour=duration.tour,
        information_cutoff_utc=duration.information_cutoff_utc,
        fitted_at_utc=duration.fitted_at_utc,
    )
    snapshot_v3 = ModelSnapshot.model_validate(
        snapshot_v2.model_dump(mode="python")
        | {
            "schema_version": "serve-model-snapshot/v3",
            "duration_artifact": reference,
            "duration_schema_version": duration.schema_version,
        }
    )
    monkeypatch.setattr(
        "tennis_model.simulation.parameters.load_snapshot_duration_artifact",
        lambda snapshot: duration,
    )

    distribution = estimate_match(
        snapshot_v3,
        context,
        inactivity_records=amended.inactivity.records if amended.inactivity else (),
    )
    record = distribution.to_record()
    restored = restore_match_parameter_distribution(distribution.canonical_json())

    assert snapshot_v3.duration_complete
    assert snapshot_v3.snapshot_id != snapshot_v2.snapshot_id
    assert distribution.duration is duration
    assert distribution.provenance.duration_artifact_id == duration.artifact_id
    assert record.schema_version == "match-parameter-distribution/v3"
    assert record.duration is not None
    assert record.duration.artifact_id == duration.artifact_id
    assert record.duration.player_ids == (context.player_a_id, context.player_b_id)
    assert restored.to_record() == record

    exposures: list[DurationPathExposure] = []

    prepared_marker = object()
    prepare_calls = 0

    def prepare_target_parameters(
        artifact: DurationFitArtifact,
        player_ids: tuple[str, str],
    ) -> object:
        nonlocal prepare_calls
        prepare_calls += 1
        assert artifact is duration
        assert player_ids == (context.player_a_id, context.player_b_id)
        return prepared_marker

    def target_parameters(
        prepared: object,
        rng: np.random.Generator,
    ) -> object:
        assert prepared is prepared_marker
        assert isinstance(rng, np.random.Generator)
        return object()

    def duration_draw(
        parameters: object,
        exposure: DurationPathExposure,
        residual_rng: np.random.Generator,
        *,
        display_policy: object,
        partial: bool,
    ) -> DurationDraw:
        del parameters
        assert display_policy == UNRESOLVED_DURATION_DISPLAY_POLICY
        exposures.append(exposure)
        latent = 80.0 + exposure.total_points / 100.0 + float(residual_rng.random())
        floor_value = int(latent)
        candidates = tuple(sorted({floor_value, int(latent + 0.5)}))
        return DurationDraw(
            artifact_id=duration.artifact_id,
            latent_minutes=latent,
            official_minutes=None,
            candidate_official_minutes=candidates,
            partial=partial,
            center_minutes=latent,
            scale_minutes=1.0,
            standardized_residual=0.0,
            display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
        )

    monkeypatch.setattr(
        "tennis_model.simulation.match.prepare_duration_parameter_sampler",
        prepare_target_parameters,
    )
    monkeypatch.setattr(
        "tennis_model.simulation.match.sample_prepared_duration_parameters",
        target_parameters,
    )
    monkeypatch.setattr("tennis_model.simulation.match.draw_duration", duration_draw)
    seed = np.random.SeedSequence(20260830)
    baseline = simulate_matches(
        amended,
        n_paths=4,
        seed=seed,
        first_server_id=context.player_a_id,
    )
    duration_batch = simulate_matches(
        distribution,
        n_paths=4,
        seed=seed,
        first_server_id=context.player_a_id,
    )
    repeated = simulate_matches(
        distribution,
        n_paths=4,
        seed=seed,
        first_server_id=context.player_a_id,
    )
    assert duration_batch.paths == repeated.paths
    assert prepare_calls == 2
    for duration_path, baseline_path in zip(duration_batch.paths, baseline.paths, strict=True):
        assert (
            replace(
                duration_path,
                duration_latent=None,
                duration_official=None,
                duration_partial=False,
                duration_display_policy_version=None,
                duration_display_candidates=(),
            )
            == baseline_path
        )

    def player_a_retires(
        intensity_a: float,
        intensity_b: float,
        rng: np.random.Generator,
    ) -> object:
        del intensity_a, intensity_b, rng
        from tennis_model.estimation.retirement import (
            CompetingRetirementOutcome,
            CompetingRiskProbabilities,
            RetirementBoundaryDraw,
        )

        return RetirementBoundaryDraw(
            outcome=CompetingRetirementOutcome.PLAYER_A_RETIRES,
            probabilities=CompetingRiskProbabilities(
                p_no_retirement=0.0,
                p_player_a_retires=1.0,
                p_player_b_retires=0.0,
            ),
        )

    monkeypatch.setattr(
        "tennis_model.simulation.match.draw_competing_retirement", player_a_retires
    )
    exposures.clear()
    retired_batch = simulate_matches(
        distribution,
        n_paths=1,
        seed=717,
        first_server_id=context.player_a_id,
    )
    retired_path = retired_batch.paths[0]
    assert retired_path.retired_player_id == context.player_a_id
    assert retired_path.duration_partial
    assert len(exposures) == 1
    assert exposures[0].official_games == retired_path.total_games
    assert exposures[0].total_points == sum(
        item.service_points for item in retired_path.player_stats.values()
    )
    assert exposures[0].sets == retired_path.sets_started
    assert exposures[0].tiebreaks == retired_path.tiebreak_count

    with pytest.raises(ValidationError, match="v2 snapshots cannot"):
        ModelSnapshot.model_validate(
            snapshot_v2.model_dump(mode="python")
            | {
                "duration_artifact": reference,
                "duration_schema_version": duration.schema_version,
            }
        )

    failed_duration = duration.model_copy(
        update={"diagnostics": DurationFitDiagnostics.model_construct(converged=False)}
    )
    failed_persisted = PersistedDurationFitArtifact.model_construct(
        directory=reference.directory,
        artifact=failed_duration,
    )
    monkeypatch.setattr(
        "tennis_model.estimation.snapshot.load_duration_fit_artifact",
        lambda directory: failed_persisted,
    )
    with pytest.raises(ModelSnapshotError, match="did not converge"):
        load_snapshot_duration_artifact(snapshot_v3)


def test_seen_inactive_player_uses_sampled_direct_effect_gaussian(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    snapshot, context, amended = _amended_fixture(
        match_fixture,
        tmp_path,
        days_a=270,
        days_b=90,
    )
    assert amended.inactivity is not None
    assert amended.inactivity.posterior_coordinate_version == ("canonical-player-effect-laplace/v1")
    assert amended.inactivity.records[0].hard_deviation_multiplier == 0.5
    assert amended.inactivity.records[0].variance_inflation_factor == 1.75
    assert all(
        adjustment.state_after is InactivityAdjustmentState.C6_APPLIED
        for adjustment in amended.inactivity.component_adjustments
    )
    restored = restore_match_parameter_distribution(amended.canonical_json())
    assert restored.to_record() == amended.to_record()
    assert restored.context == context

    legacy = sample_match_performance(match_fixture.distribution, np.random.SeedSequence(731))
    inactive = sample_match_performance(amended, np.random.SeedSequence(731))
    replay = sample_match_performance(amended, np.random.SeedSequence(731))
    assert inactive == replay
    assert inactive.player_a_serving != legacy.player_a_serving
    assert inactive.matchup_parameters.components != legacy.matchup_parameters.components
    assert any(item.c6_adjusted_coordinates for item in inactive.matchup_parameters.components)
    assert tuple(
        item.predictive_concentration for item in inactive.matchup_parameters.components
    ) == tuple(item.predictive_concentration for item in legacy.matchup_parameters.components)

    active_records = (
        _inactivity_record(
            _MatchFixture(
                fits=match_fixture.fits,
                artifacts=match_fixture.artifacts,
                snapshot=snapshot,
                context=context,
                distribution=amended,
            ),
            player_id="p0",
            days=90,
        ),
        amended.inactivity.records[1],
    )
    active_c6 = estimate_match(snapshot, context, inactivity_records=active_records)
    active_draw = sample_match_performance(active_c6, np.random.SeedSequence(731))
    assert active_draw.retirement_draws == inactive.retirement_draws


def test_b6_joint_simulation_is_fixed_seed_reproducible(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    _snapshot, _context, amended = _amended_fixture(
        match_fixture,
        tmp_path,
        days_a=90,
        days_b=90,
    )
    first = simulate_matches(amended, n_paths=30, seed=9191, first_server_id="p0")
    second = simulate_matches(amended, n_paths=30, seed=9191, first_server_id="p0")
    assert first == second
    assert first.provenance["retirement_enabled"] is True
    assert all(path.retirement_intensities is not None for path in first.paths)
    assert all(path.retirement_scenario_ids == ("central", "central") for path in first.paths)


def test_b6_c6_lock_is_machine_complete_and_replayable(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    coverage = CoverageRange(
        first_match_date=date(2023, 1, 1),
        last_match_date=match_fixture.snapshot.data_cutoff_utc.date(),
        verified_at_utc=match_fixture.snapshot.data_cutoff_utc,
    )
    manifest = SourceManifest(
        manifest_version="fixture-manifest-v2",
        sources=(
            PinnedSource(
                source_id="fixture-history",
                identity_namespace="fixture-history",
                tour=Tour.ATP,
                upstream_attribution="synthetic B6/C6 integration fixture",
                locator="fixture://history",
                object_identifier="fixture-v2",
                sha256="1" * 64,
                schema_version="fixture/v1",
                stated_license="test-only",
                retrieved_at_utc=match_fixture.snapshot.data_cutoff_utc,
                verified_coverage=coverage,
                row_date_semantics=RowDateSemantics.MATCH_DATE,
                availability_lag_days=1,
            ),
        ),
        coverage_by_tour=TourCoverage(atp=coverage, wta=None),
    )
    manifest_hash = SourceManifestProvenance.from_manifest(manifest).manifest_sha256
    snapshot, context, distribution = _amended_fixture(
        match_fixture,
        tmp_path,
        days_a=90,
        days_b=90,
        source_manifest_id=manifest.manifest_version,
        source_manifest_sha256=manifest_hash,
    )
    assert distribution.inactivity is not None
    policy = PathCountPolicy(
        standard_paths=20,
        escalated_paths=40,
        minimum_settled_paths=0,
        extreme_probability=0.001,
        integer_boundary_window=0.0,
        integer_boundary_standard_errors=0.0,
    )
    lock = create_prediction_lock(
        snapshot,
        context,
        InformationBundle(
            bundle_id="b6-c6-lock-info",
            scenario_id=context.information_scenario_id,
            information_cutoff_utc=context.information_cutoff_utc,
        ),
        (MATCH_WIN("p0"),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=CodeProvenance(commit="test", dirty=False, diff_sha256=None),
        seed=20260830,
        n_paths=policy.standard_paths,
        execution_mode="test",
        path_count_policy=policy,
        created_at_utc=context.information_cutoff_utc,
        inactivity_records=distribution.inactivity.records,
    )
    assert lock.schema_version == "prediction-lock/v2"
    assert lock.retirement_model_artifact_id == distribution.retirement.artifact_id  # type: ignore[union-attr]
    assert lock.inactivity_configuration_artifact_id == (
        distribution.inactivity.configuration_artifact_id
    )
    assert lock.match_summary.retirement_probability is not None
    player_retirement = tuple(item.retirement_probability for item in lock.match_summary.players)
    assert all(item is not None for item in player_retirement)
    assert sum(item for item in player_retirement if item is not None) == pytest.approx(
        lock.match_summary.retirement_probability,
        abs=1e-15,
    )
    assert distribution.retirement is not None
    assert tuple(item[0] for item in distribution.retirement.central_intensity_summaries) == (
        "p0",
        "p1",
    )
    assert distribution.retirement.deterministic_test_result_sha256 == "e" * 64
    assert lock.simulation.retirement_rng_stream_version == (
        "seedsequence-retirement-parameters-boundaries/v1"
    )
    assert lock.simulation.ordinary_termination_before_retirement_version == (
        "ordinary-terminal-bypass-before-b6/v1"
    )
    assert reproduce_prediction_lock(lock).reproduced
    card = render_locked_match_card(lock)
    assert "B6 retirement and C6 inactivity" in card
    assert "hard multiplier" in card

    retired_at = context.scheduled_start_utc + timedelta(hours=2)
    missing_timing = OfficialHistoricalOutcome(
        match_id="historical-b6-c6-retirement",
        player_a_id=context.player_a_id,
        player_b_id=context.player_b_id,
        best_of=context.best_of,
        started=True,
        completed=False,
        winner_id=context.player_a_id,
        retired_player_id=context.player_b_id,
        sets_started=1,
        official_source_id="official-result",
        official_source_sha256="9" * 64,
        official_source_locator="fixture://official-b6-c6-retirement",
        available_at_utc=retired_at,
        retrieved_at_utc=retired_at + timedelta(minutes=5),
    )
    missing_timing_settlement = settle_historical_lock(lock, missing_timing)
    with pytest.raises(LedgerError, match="explicit timing-missing reason"):
        ledger_rows_from_settlement(lock, missing_timing, missing_timing_settlement)

    outcome = missing_timing.model_copy(
        update={
            "retirement_timing_missing_reason": (
                "official result identifies retirement but has no exact game boundary"
            )
        }
    )
    settlement = settle_historical_lock(lock, outcome)
    rows = ledger_rows_from_settlement(
        lock,
        outcome,
        settlement,
        created_at_utc=outcome.retrieved_at_utc,
        backtest_run_id="b6-c6-ledger-round-trip",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.schema_version == "calibration-ledger/v2"
    assert row.b6_c6_provenance is not None
    assert row.b6_c6_provenance.retirement == lock.match_parameters.retirement
    assert row.b6_c6_provenance.inactivity == lock.match_parameters.inactivity
    assert row.b6_c6_provenance.match_retirement_probability == (
        lock.match_summary.retirement_probability
    )
    assert row.b6_c6_provenance.player_retirement_probabilities == tuple(
        (item.player_id, item.retirement_probability) for item in lock.match_summary.players
    )
    assert row.b6_c6_provenance.retired_player_id == context.player_b_id
    assert row.b6_c6_provenance.retirement_completed_games is None
    assert row.b6_c6_provenance.retirement_timing_missing_reason == (
        "official result identifies retirement but has no exact game boundary"
    )
    ledger = CalibrationLedger(tmp_path / "b6-c6-ledger.sqlite3")
    ledger.append(row)
    assert ledger.read() == rows

    with pytest.raises(LockCreationError, match="both player inactivity records"):
        create_prediction_lock(
            snapshot,
            context,
            lock.information,
            (MATCH_WIN("p0"),),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=manifest,
            code=lock.code,
            seed=1,
            n_paths=policy.standard_paths,
            execution_mode="test",
            path_count_policy=policy,
        )


def test_operational_v3_identity_revisions_and_retained_artifact_verification(
    match_fixture: _MatchFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coverage = CoverageRange(
        first_match_date=date(2023, 1, 1),
        last_match_date=match_fixture.snapshot.data_cutoff_utc.date(),
        verified_at_utc=datetime.now(UTC),
    )
    manifest = SourceManifest(
        manifest_version="fixture-operational-v3",
        sources=(
            PinnedSource(
                source_id="fixture-history-v3",
                identity_namespace="fixture-history",
                tour=Tour.ATP,
                upstream_attribution="synthetic operational fixture",
                locator="fixture://history-v3",
                object_identifier="fixture-v3",
                sha256="1" * 64,
                schema_version="fixture/v1",
                stated_license="test-only",
                retrieved_at_utc=datetime.now(UTC),
                verified_coverage=coverage,
                row_date_semantics=RowDateSemantics.MATCH_DATE,
                availability_lag_days=1,
                source_effective_at_utc=match_fixture.snapshot.data_cutoff_utc,
                source_available_at_utc=match_fixture.snapshot.data_cutoff_utc,
            ),
        ),
        coverage_by_tour=TourCoverage(atp=coverage, wta=None),
    )
    manifest_hash = SourceManifestProvenance.from_manifest(manifest).manifest_sha256
    snapshot, context, distribution = _amended_fixture(
        match_fixture,
        tmp_path / "operational-fit",
        days_a=90,
        days_b=90,
        source_manifest_id=manifest.manifest_version,
        source_manifest_sha256=manifest_hash,
    )
    assert distribution.inactivity is not None
    identity = CanonicalMatchIdentity.from_official_id(
        source_namespace="fixture-history",
        tour=Tour.ATP,
        official_match_id="official-match-7788",
        participant_ids=(context.player_a_id, context.player_b_id),
        source_id="fixture-history-v3",
        source_sha256="1" * 64,
        source_locator="fixture://history-v3#official-match-7788",
        resolved_at_utc=datetime.now(UTC),
    )
    artifact_root = tmp_path / "retained"
    artifact_root.mkdir()
    retained = []
    for kind in (
        "source_snapshot",
        "normalized_snapshot",
        "component_counts",
        "component_fit",
        "retirement_fit",
        "inactivity_config",
        "model_config",
        "settlement_policy",
        "code_archive",
    ):
        path = artifact_root / f"{kind}.artifact"
        payload = f"immutable-{kind}\n".encode()
        path.write_bytes(payload)
        retained.append(
            RetainedArtifactRecord.model_validate(
                {
                    "kind": kind,
                    "artifact_id": f"fixture-{kind}",
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )
    empty_digest = hashlib.sha256(b"").hexdigest()
    code = CodeProvenance(
        commit="test-operational",
        dirty=False,
        diff_sha256=None,
        fingerprint_version="complete-git-state/v2",
        staged_sha256=empty_digest,
        unstaged_sha256=empty_digest,
        untracked_sha256=empty_digest,
    )
    policy = PathCountPolicy(
        standard_paths=20,
        escalated_paths=40,
        minimum_settled_paths=0,
        extreme_probability=0.001,
        integer_boundary_window=0.0,
        integer_boundary_standard_errors=0.0,
    )
    information = InformationBundle(
        bundle_id="operational-v3-info",
        scenario_id=context.information_scenario_id,
        information_cutoff_utc=context.information_cutoff_utc,
    )
    store = LockStore(tmp_path / "operational-locks")
    lock = create_prediction_lock(
        snapshot,
        context,
        information,
        (MATCH_WIN(context.player_a_id),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=code,
        seed=88001,
        store=store,
        n_paths=policy.standard_paths,
        execution_mode="test",
        path_count_policy=policy,
        inactivity_records=distribution.inactivity.records,
        canonical_match_identity=identity,
        retained_artifacts=tuple(retained),
    )
    assert lock.schema_version == "prediction-lock/v3"
    assert lock.base_lock_id == identity.base_lock_id
    assert lock.historical_time is not None
    assert lock.historical_time.artifact_created_at_utc > context.information_cutoff_utc
    assert lock.historical_time.sources[0].retrieved_at_utc > context.information_cutoff_utc
    assert lock.historical_time.sources[0].source_available_at_utc < (
        context.information_cutoff_utc
    )
    assert store.load(lock.base_lock_id, 1).lock == lock
    legacy_payload = lock.model_dump(mode="json")
    legacy_payload.pop("duration_model_artifact_id")
    legacy_payload["simulation"].pop("duration_rng_stream_version")
    legacy_payload["simulation"].pop("duration_display_policy_version")
    legacy_payload["simulation"].pop("inspected_path_counts")
    legacy_payload["simulation"].pop("platform_submission_policy_version")
    legacy_payload["match_summary"].pop("duration")
    legacy_payload["match_parameters"].pop("duration")
    legacy_payload["match_parameters"]["provenance"].pop("duration_artifact_id")
    legacy_payload["match_parameters"]["snapshot"].pop("duration_artifact")
    legacy_payload["match_parameters"]["snapshot"].pop("duration_schema_version")
    for estimate in legacy_payload["prop_estimates"]:
        estimate.pop("sensitivity_low")
        estimate.pop("sensitivity_high")
        estimate.pop("display_policy_version")
        for field_name in (
            "model_probability_raw",
            "model_probability_integer",
            "platform_submission_integer",
            "platform_submission_policy_version",
            "model_rounding_policy_version",
            "mc_policy_version",
            "mc_confidence_level",
            "mc_confidence_sequence_method",
            "mc_confidence_sequence_lower",
            "mc_confidence_sequence_upper",
            "mc_stopping_status",
            "final_cumulative_path_count",
        ):
            estimate.pop(field_name)
    from tennis_model.locking._json import sha256_json
    from tennis_model.locking.models import PredictionSnapshot

    restored_legacy = PredictionSnapshot.model_validate(legacy_payload)
    assert restored_legacy.content_sha256 == sha256_json(legacy_payload)
    assert restored_legacy.content_sha256 == lock.content_sha256

    rescheduled = context.model_copy(
        update={"scheduled_start_utc": context.scheduled_start_utc + timedelta(hours=2)}
    )
    revision = create_prediction_lock(
        snapshot,
        rescheduled,
        information,
        (MATCH_WIN(context.player_a_id),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=code,
        seed=88002,
        store=store,
        n_paths=policy.standard_paths,
        execution_mode="test",
        path_count_policy=policy,
        inactivity_records=distribution.inactivity.records,
        canonical_match_identity=identity,
        retained_artifacts=tuple(retained),
        parent=lock,
        revision_reason=LockRevisionReason(
            category="schedule", summary="official start moved two hours"
        ),
    )
    assert revision.base_lock_id == lock.base_lock_id
    assert revision.revision == 2
    assert revision.parent_content_sha256 == lock.content_sha256

    duration = DurationFitArtifact.model_construct(
        schema_version="duration-fit-artifact/v1",
        framework_version="v1.0",
        artifact_id="9" * 64,
        tour=rescheduled.tour,
        source_manifest_id=manifest.manifest_version,
        source_manifest_sha256=manifest_hash,
        information_cutoff_utc=snapshot.data_cutoff_utc,
        fit_cutoff_utc=snapshot.data_cutoff_utc,
        fitted_at_utc=snapshot.fitted_at_utc,
        current_event=rescheduled.event,
        current_event_year=rescheduled.scheduled_start_utc.year,
        coefficients=(
            DurationCoefficient(
                name="current_usopen_2026", value=0.5, standard_error=0.1
            ),
        ),
    )
    duration_reference = DurationArtifactReference(
        artifact_id=duration.artifact_id,
        directory=(tmp_path / "operational-duration-fit").resolve(),
        tour=duration.tour,
        information_cutoff_utc=duration.information_cutoff_utc,
        fitted_at_utc=duration.fitted_at_utc,
    )
    duration_snapshot = ModelSnapshot.model_validate(
        snapshot.model_dump(mode="python")
        | {
            "schema_version": "serve-model-snapshot/v3",
            "duration_artifact": duration_reference,
            "duration_schema_version": duration.schema_version,
        }
    )
    monkeypatch.setattr(
        "tennis_model.simulation.parameters.load_snapshot_duration_artifact",
        lambda _snapshot: duration,
    )

    duration_prepared_marker = object()

    def prepare_target_duration_parameters(
        artifact: DurationFitArtifact,
        player_ids: tuple[str, str],
    ) -> object:
        assert artifact is duration
        assert player_ids == (rescheduled.player_a_id, rescheduled.player_b_id)
        return duration_prepared_marker

    def target_duration_parameters(
        prepared: object,
        rng: np.random.Generator,
    ) -> object:
        assert prepared is duration_prepared_marker
        assert isinstance(rng, np.random.Generator)
        return object()

    def attached_duration(
        parameters: object,
        exposure: DurationPathExposure,
        residual_rng: np.random.Generator,
        *,
        display_policy: object,
        partial: bool,
    ) -> DurationDraw:
        del parameters
        assert display_policy == FLOOR_DURATION_DISPLAY_POLICY
        latent = 70.0 + 0.4 * exposure.total_points + float(residual_rng.random())
        official = int(latent)
        return DurationDraw(
            artifact_id=duration.artifact_id,
            latent_minutes=latent,
            official_minutes=official,
            candidate_official_minutes=(official,),
            partial=partial,
            center_minutes=latent,
            scale_minutes=1.0,
            standardized_residual=0.0,
            display_policy=FLOOR_DURATION_DISPLAY_POLICY,
        )

    monkeypatch.setattr(
        "tennis_model.simulation.match.prepare_duration_parameter_sampler",
        prepare_target_duration_parameters,
    )
    monkeypatch.setattr(
        "tennis_model.simulation.match.sample_prepared_duration_parameters",
        target_duration_parameters,
    )
    monkeypatch.setattr("tennis_model.simulation.match.draw_duration", attached_duration)
    duration_path = artifact_root / "duration_fit.artifact"
    duration_payload = b"immutable-duration-fit\n"
    duration_path.write_bytes(duration_payload)
    duration_retained = RetainedArtifactRecord(
        kind="duration_fit",
        artifact_id=duration.artifact_id,
        path=str(duration_path),
        sha256=hashlib.sha256(duration_payload).hexdigest(),
    )
    duration_lock = create_prediction_lock(
        duration_snapshot,
        rescheduled,
        information,
        (
            DURATION_MIN(
                ComparisonOperator.MORE_THAN,
                120,
                display_conversion_version=FLOOR_DURATION_DISPLAY_POLICY.policy_version,
            ),
        ),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=code,
        seed=88003,
        store=store,
        n_paths=policy.standard_paths,
        execution_mode="test",
        path_count_policy=policy,
        inactivity_records=distribution.inactivity.records,
        canonical_match_identity=identity,
        retained_artifacts=(*retained, duration_retained),
        parent=revision,
        revision_reason=LockRevisionReason(
            category="model_refresh", summary="attach frozen duration artifact"
        ),
        duration_display_policy=FLOOR_DURATION_DISPLAY_POLICY,
    )
    assert duration_lock.schema_version == "prediction-lock/v4"
    assert duration_lock.revision == 3
    assert duration_lock.duration_model_artifact_id == duration.artifact_id
    assert duration_lock.match_summary.duration is not None
    assert duration_lock.match_summary.duration.data_grade == "B"
    assert duration_lock.prop_estimates[0].data_grade == "B"
    assert duration_lock.simulation.duration_rng_stream_version == (
        "seedsequence-duration-parameters-residual/v1"
    )
    assert reproduce_prediction_lock(duration_lock).reproduced
    duration_card = render_locked_match_card(duration_lock)
    assert "Match duration" in duration_card
    assert duration.artifact_id in duration_card
    assert store.load(duration_lock.base_lock_id, 3).lock == duration_lock
    with pytest.raises(
        LockCreationError, match="production duration locks require the unresolved"
    ):
        create_prediction_lock(
            duration_snapshot,
            rescheduled,
            information,
            (
                DURATION_MIN(
                    ComparisonOperator.MORE_THAN,
                    120,
                    display_conversion_version=FLOOR_DURATION_DISPLAY_POLICY.policy_version,
                ),
            ),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=manifest,
            code=code,
            seed=88004,
            store=store,
            execution_mode="production",
            inactivity_records=distribution.inactivity.records,
            canonical_match_identity=identity,
            retained_artifacts=(*retained, duration_retained),
            duration_display_policy=FLOOR_DURATION_DISPLAY_POLICY,
        )

    first_path = Path(retained[0].path)
    original = first_path.read_bytes()
    first_path.write_bytes(original + b"tampered")
    with pytest.raises(LockIntegrityError, match="retained artifact"):
        store.load(lock.base_lock_id, 1)
    first_path.write_bytes(original)
    assert store.verify(lock.base_lock_id, 1) == lock.content_sha256
    first_path.unlink()
    with pytest.raises(LockIntegrityError, match="retained artifact is missing"):
        store.load(lock.base_lock_id, 1)


def test_snapshot_loads_exact_five_artifacts_and_round_trips(
    match_fixture: _MatchFixture,
) -> None:
    snapshot = match_fixture.snapshot
    loaded = load_snapshot_fits(snapshot)

    assert tuple(loaded) == tuple(ServeComponent)
    assert loaded == match_fixture.fits
    assert snapshot.component_artifact_ids == {
        component: match_fixture.artifacts[component].artifact_id for component in ServeComponent
    }
    restored = ModelSnapshot.from_json(snapshot.canonical_json())
    assert restored == snapshot
    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.canonical_json() == snapshot.canonical_json()


def test_snapshot_identity_is_location_independent(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    relocated_artifacts = {
        component: write_fit_artifact(match_fixture.fits[component], tmp_path / "relocated")
        for component in ServeComponent
    }
    relocated = create_model_snapshot(relocated_artifacts)

    assert relocated.snapshot_id == match_fixture.snapshot.snapshot_id
    assert relocated.component_artifact_ids == match_fixture.snapshot.component_artifact_ids
    assert relocated.component_artifacts != match_fixture.snapshot.component_artifacts


def test_snapshot_identity_normalizes_equivalent_timestamp_offsets(
    match_fixture: _MatchFixture,
) -> None:
    payload = match_fixture.snapshot.model_dump(mode="python")
    central = timezone(timedelta(hours=-6))
    payload["fitted_at_utc"] = match_fixture.snapshot.fitted_at_utc.astimezone(central)
    payload["data_cutoff_utc"] = match_fixture.snapshot.data_cutoff_utc.astimezone(central)
    equivalent = ModelSnapshot.model_validate(payload)

    assert equivalent.fitted_at_utc.tzinfo is UTC
    assert equivalent.data_cutoff_utc.tzinfo is UTC
    assert equivalent.snapshot_id == match_fixture.snapshot.snapshot_id
    assert equivalent.canonical_json() == match_fixture.snapshot.canonical_json()


def test_snapshot_rejects_missing_components_and_missing_referenced_artifact(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    incomplete = dict(match_fixture.artifacts)
    incomplete.pop(ServeComponent.Q2)
    with pytest.raises(ModelSnapshotError, match="exactly the five"):
        create_model_snapshot(incomplete)

    references = list(match_fixture.snapshot.component_artifacts)
    references[0] = references[0].model_copy(
        update={"directory": (tmp_path / "missing-fit-artifact").resolve()}
    )
    missing = ModelSnapshot.model_validate(
        match_fixture.snapshot.model_dump(mode="python")
        | {"component_artifacts": tuple(references)}
    )
    assert missing.snapshot_id == match_fixture.snapshot.snapshot_id
    with pytest.raises(MatchParameterError, match="cannot load model snapshot"):
        estimate_match(missing, match_fixture.context)


def test_snapshot_rejects_mixed_provenance_and_wrong_component_routing(
    match_fixture: _MatchFixture,
    tmp_path: Path,
) -> None:
    mixed = dict(match_fixture.artifacts)
    changed_fit = match_fixture.fits[ServeComponent.A].model_copy(
        update={"code_commit": "different-fit-commit"}
    )
    mixed[ServeComponent.A] = write_fit_artifact(changed_fit, tmp_path / "mixed")
    with pytest.raises(ModelSnapshotError, match="coherent snapshot"):
        create_model_snapshot(mixed)

    misrouted = dict(match_fixture.artifacts)
    misrouted[ServeComponent.F] = match_fixture.artifacts[ServeComponent.D]
    with pytest.raises(ModelSnapshotError, match="wrong component"):
        create_model_snapshot(misrouted)


def test_match_estimation_rejects_wrong_tour_and_future_data_but_allows_later_construction(
    match_fixture: _MatchFixture,
) -> None:
    wrong_tour = ModelSnapshot.model_validate(
        match_fixture.snapshot.model_dump(mode="python") | {"tour": Tour.WTA}
    )
    with pytest.raises(MatchParameterError, match="tour"):
        estimate_match(wrong_tour, match_fixture.context)

    earlier_cutoff = match_fixture.context.model_copy(
        update={"information_cutoff_utc": TEST_CUTOFF - timedelta(seconds=1)}
    )
    earlier_cutoff = MatchContext.model_validate(earlier_cutoff.model_dump(mode="python"))
    with pytest.raises(MatchParameterError, match="after the match cutoff"):
        estimate_match(match_fixture.snapshot, earlier_cutoff)

    after_data_but_before_fit = MatchContext.model_validate(
        match_fixture.context.model_copy(
            update={"information_cutoff_utc": TEST_CUTOFF + timedelta(hours=12)}
        ).model_dump(mode="python")
    )
    retrospective = estimate_match(match_fixture.snapshot, after_data_but_before_fit)
    assert retrospective.snapshot.fitted_at_utc > retrospective.context.information_cutoff_utc
    assert retrospective.snapshot.data_cutoff_utc <= retrospective.context.information_cutoff_utc

    assert (
        match_fixture.distribution.provenance.data_cutoff_utc
        < match_fixture.distribution.provenance.match_information_cutoff_utc
    )


def test_match_context_rejects_ambiguous_or_impossible_identity_and_time() -> None:
    base = {
        "player_a_id": "p0",
        "player_b_id": "p1",
        "tour": Tour.ATP,
        "event": "Synthetic Open",
        "round": "R128",
        "scheduled_start_utc": TEST_CUTOFF + timedelta(days=2),
        "best_of": 3,
        "indoor": False,
        "information_cutoff_utc": TEST_CUTOFF + timedelta(days=1),
    }
    with pytest.raises(ValidationError, match="distinct"):
        MatchContext(**(base | {"player_b_id": "p0"}))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="timezone-aware"):
        MatchContext(  # type: ignore[arg-type]
            **(base | {"information_cutoff_utc": TEST_CUTOFF.replace(tzinfo=None)})
        )
    with pytest.raises(ValidationError, match="cannot precede"):
        MatchContext(  # type: ignore[arg-type]
            **(
                base
                | {
                    "scheduled_start_utc": TEST_CUTOFF,
                    "information_cutoff_utc": TEST_CUTOFF + timedelta(days=1),
                }
            )
        )


def test_matchup_builds_distinct_explicit_serving_directions(
    match_fixture: _MatchFixture,
) -> None:
    distribution = match_fixture.distribution
    assert (distribution.player_a_serving.server_id, distribution.player_a_serving.receiver_id) == (
        "p0",
        "p1",
    )
    assert (distribution.player_b_serving.server_id, distribution.player_b_serving.receiver_id) == (
        "p1",
        "p0",
    )
    assert tuple(distribution.player_a_serving.by_component) == tuple(ServeComponent)
    assert tuple(distribution.player_b_serving.by_component) == tuple(ServeComponent)
    assert all(0.0 < value < 1.0 for value in _component_means(distribution.player_a_serving))
    assert all(0.0 < value < 1.0 for value in _component_means(distribution.player_b_serving))
    assert _component_means(distribution.player_a_serving) != _component_means(
        distribution.player_b_serving
    )


def test_f_and_d_are_server_led_while_opponent_components_use_returner(
    match_fixture: _MatchFixture,
) -> None:
    receiver_one = _future_context(match_fixture, receiver="p1")
    receiver_two = _future_context(match_fixture, receiver="p2")

    for component in (ServeComponent.F, ServeComponent.D):
        fit = match_fixture.fits[component]
        theta = fit.posterior.map_parameters
        first = project_component_parameters(fit, receiver_one, theta)
        second = project_component_parameters(fit, receiver_two, theta)
        assert first.base_linear_predictor == second.base_linear_predictor
        assert all(not block.role.startswith("returner") for block in fit.effect_blocks)

    for component in (ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2):
        fit = match_fixture.fits[component]
        theta = list(fit.posterior.map_parameters)
        block = next(item for item in fit.effect_blocks if item.role == "returner_global")
        p1_position = block.levels.index("p1")
        p2_position = block.levels.index("p2")
        theta[block.scale_parameter_index] = 0.0
        theta[block.free_parameter_indices[p1_position]] = 1.25
        theta[block.free_parameter_indices[p2_position]] = -1.25
        first = project_component_parameters(fit, receiver_one, theta)
        second = project_component_parameters(fit, receiver_two, theta)
        assert first.base_linear_predictor < second.base_linear_predictor


def test_surface_and_approved_context_terms_enter_once(
    match_fixture: _MatchFixture,
) -> None:
    fit = match_fixture.fits[ServeComponent.F]
    baseline = list(fit.posterior.map_parameters)
    hard_block = next(
        block
        for block in fit.effect_blocks
        if block.role == "server_surface" and block.surface == "Hard"
    )
    p0_position = hard_block.levels.index("p0")
    changed = baseline.copy()
    changed[hard_block.free_parameter_indices[p0_position]] += 0.8

    hard_before = project_component_parameters(
        fit, _future_context(match_fixture, surface="Hard"), baseline
    )
    hard_after = project_component_parameters(
        fit, _future_context(match_fixture, surface="Hard"), changed
    )
    clay_before = project_component_parameters(
        fit, _future_context(match_fixture, surface="Clay"), baseline
    )
    clay_after = project_component_parameters(
        fit, _future_context(match_fixture, surface="Clay"), changed
    )
    assert hard_after.base_linear_predictor != hard_before.base_linear_predictor
    assert clay_after.base_linear_predictor == clay_before.base_linear_predictor

    indoor_index = fit.fixed_parameter_indices["indoor_hard"]
    context_theta = baseline.copy()
    context_theta[indoor_index] = 0.37
    outdoor = project_component_parameters(
        fit, _future_context(match_fixture, indoor=False), context_theta
    )
    indoor = project_component_parameters(
        fit, _future_context(match_fixture, indoor=True), context_theta
    )
    assert indoor.base_linear_predictor - outdoor.base_linear_predictor == pytest.approx(0.37)


def test_event_year_effects_are_only_active_for_approved_components(
    match_fixture: _MatchFixture,
) -> None:
    first_event = _future_context(match_fixture, event="Other Open", event_year=2025)
    second_event = _future_context(match_fixture, event="Synthetic Open", event_year=2025)
    for component in (ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2):
        fit = match_fixture.fits[component]
        theta = list(fit.posterior.map_parameters)
        block = next(item for item in fit.effect_blocks if item.role == "event_year")
        theta[block.scale_parameter_index] = 0.0
        theta[block.free_parameter_indices[0]] = 0.9
        first = project_component_parameters(fit, first_event, theta)
        second = project_component_parameters(fit, second_event, theta)
        assert first.base_linear_predictor != second.base_linear_predictor

    for component in (ServeComponent.F, ServeComponent.D):
        fit = match_fixture.fits[component]
        assert all(block.role != "event_year" for block in fit.effect_blocks)
        first = project_component_parameters(fit, first_event, fit.posterior.map_parameters)
        second = project_component_parameters(fit, second_event, fit.posterior.map_parameters)
        assert first.base_linear_predictor == second.base_linear_predictor


def test_explicit_parameter_projection_matches_milestone_three_map_summary(
    match_fixture: _MatchFixture,
) -> None:
    direction = match_fixture.distribution.player_a_serving
    for component in ServeComponent:
        item = direction.by_component[component]
        projection = project_component_parameters(
            item.fit,
            direction.context,
            item.fit.posterior.map_parameters,
        )
        assert projection.base_linear_predictor == pytest.approx(
            item.map_prediction.linear_predictor_map
        )
        assert projection.predictive_concentration == pytest.approx(
            item.map_prediction.predictive_concentration
        )
        assert sum(effect.standard_deviation**2 for effect in projection.unseen_effects) == (
            pytest.approx(item.map_prediction.unseen_effect_variance)
        )


@pytest.mark.parametrize("sampled_log_scale", [-1000.0, 1000.0])
def test_sampled_effect_scale_numerical_failure_is_not_clipped(
    match_fixture: _MatchFixture,
    sampled_log_scale: float,
) -> None:
    fit = match_fixture.fits[ServeComponent.F]
    theta = list(fit.posterior.map_parameters)
    theta[fit.effect_blocks[0].scale_parameter_index] = sampled_log_scale
    with pytest.raises(ModelDataError, match="effect scale"):
        project_component_parameters(fit, _future_context(match_fixture), theta)


def test_inactive_match_metadata_does_not_change_primitive_means(
    match_fixture: _MatchFixture,
) -> None:
    fit = match_fixture.fits[ServeComponent.Q1]
    theta = fit.posterior.map_parameters
    best_of_three = project_component_parameters(
        fit, _future_context(match_fixture, best_of=3), theta
    )
    best_of_five = project_component_parameters(
        fit, _future_context(match_fixture, best_of=5), theta
    )
    assert best_of_three == best_of_five

    metadata_only = match_fixture.context.model_copy(
        update={
            "round": "Final",
            "conditions": (MatchCondition(name="unsupported_note", value="ignore"),),
            "information_scenario_id": "metadata-only",
        }
    )
    reconstructed = estimate_match(match_fixture.snapshot, metadata_only)
    assert _component_means(reconstructed.player_a_serving) == _component_means(
        match_fixture.distribution.player_a_serving
    )
    assert _component_means(reconstructed.player_b_serving) == _component_means(
        match_fixture.distribution.player_b_serving
    )


def _controlled_correlated_fit(fit: FittedServeComponent) -> FittedServeComponent:
    size = len(fit.posterior.parameter_names)
    covariance = np.eye(size, dtype=np.float64) * 0.04
    covariance[1, 1] = 0.09
    covariance[0, 1] = covariance[1, 0] = 0.03
    hessian = np.linalg.inv(covariance)
    posterior = PosteriorApproximation(
        parameter_names=fit.posterior.parameter_names,
        map_parameters=fit.posterior.map_parameters,
        curvature_kind="full",
        hessian=tuple(tuple(float(value) for value in row) for row in hessian),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        variance_diagonal=tuple(float(value) for value in np.diag(covariance)),
        raw_min_eigenvalue=float(np.min(np.linalg.eigvalsh(hessian))),
        regularization_added=0.0,
        condition_number=float(np.linalg.cond(hessian)),
    )
    return FittedServeComponent.model_validate(
        fit.model_copy(update={"posterior": posterior}).model_dump(mode="python")
    )


def test_posterior_draws_recover_known_mean_covariance_and_stable_indexing(
    match_fixture: _MatchFixture,
) -> None:
    fit = _controlled_correlated_fit(match_fixture.fits[ServeComponent.F])
    rng = np.random.default_rng(9401)
    draws = np.asarray(
        [sample_posterior_parameters(fit, rng).values for _ in range(12_000)],
        dtype=np.float64,
    )
    target_mean = np.asarray(fit.posterior.map_parameters)
    empirical_mean = np.mean(draws, axis=0)
    empirical_covariance = np.cov(draws[:, :2], rowvar=False, ddof=1)

    np.testing.assert_allclose(empirical_mean[0], target_mean[0], atol=0.012)
    np.testing.assert_allclose(empirical_mean[1], target_mean[1], atol=0.018)
    np.testing.assert_allclose(empirical_covariance, [[0.04, 0.03], [0.03, 0.09]], atol=0.004)
    exact_seed = 22
    expected = np.asarray(fit.posterior.map_parameters) + np.linalg.cholesky(
        fit.posterior.covariance_array()
    ) @ np.random.default_rng(exact_seed).standard_normal(len(fit.posterior.parameter_names))
    first = sample_posterior_parameters(fit, np.random.default_rng(exact_seed))
    second = sample_posterior_parameters(fit, np.random.default_rng(exact_seed))
    np.testing.assert_array_equal(first.values_array(), expected)
    assert first == second
    assert first.parameter_names == fit.posterior.parameter_names


def test_explicit_diagonal_posterior_fallback_is_sampled_diagonally(
    match_fixture: _MatchFixture,
) -> None:
    fit = match_fixture.fits[ServeComponent.D]
    variance = tuple(0.01 + 0.002 * index for index in range(len(fit.posterior.parameter_names)))
    posterior = PosteriorApproximation(
        parameter_names=fit.posterior.parameter_names,
        map_parameters=fit.posterior.map_parameters,
        curvature_kind="diagonal",
        hessian=None,
        covariance=None,
        variance_diagonal=variance,
        raw_min_eigenvalue=1.0,
        regularization_added=0.0,
        condition_number=1.0,
    )
    diagonal_fit = FittedServeComponent.model_validate(
        fit.model_copy(update={"posterior": posterior}).model_dump(mode="python")
    )
    seed = 812
    expected_standard = np.random.default_rng(seed).standard_normal(len(variance))
    expected = np.asarray(posterior.map_parameters) + np.sqrt(variance) * expected_standard
    observed = sample_posterior_parameters(diagonal_fit, np.random.default_rng(seed))
    np.testing.assert_array_equal(observed.values_array(), expected)


@pytest.mark.parametrize(
    ("mean", "concentration"),
    [(0.12, 5.0), (0.35, 25.0), (0.65, 80.0), (0.88, 500.0)],
)
def test_beta_predictive_draws_recover_theoretical_moments(
    mean: float,
    concentration: float,
) -> None:
    sample_size = 25_000
    rng = np.random.default_rng(5021 + int(concentration))
    draws = np.fromiter(
        (sample_beta_probability(mean, concentration, rng) for _ in range(sample_size)),
        dtype=np.float64,
        count=sample_size,
    )
    target_variance = mean * (1.0 - mean) / (concentration + 1.0)
    mean_standard_error = np.sqrt(target_variance / sample_size)

    assert np.all((draws > 0.0) & (draws < 1.0))
    assert abs(float(np.mean(draws)) - mean) < 6.0 * mean_standard_error
    assert float(np.var(draws, ddof=1)) == pytest.approx(target_variance, rel=0.08)


def test_beta_numerical_validation_and_concentration_behavior() -> None:
    for invalid_mean in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(MatchParameterError, match="mean"):
            sample_beta_probability(invalid_mean, 20.0, np.random.default_rng(1))
    for invalid_concentration in (0.0, -1.0, float("inf")):
        with pytest.raises(MatchParameterError, match="concentration"):
            sample_beta_probability(0.65, invalid_concentration, np.random.default_rng(1))
    with pytest.raises(TypeError, match="explicit"):
        sample_beta_probability(0.65, 80.0, None)  # type: ignore[arg-type]

    small_rng = np.random.default_rng(199)
    large_rng = np.random.default_rng(200)
    small = np.asarray([sample_beta_probability(0.65, 5.0, small_rng) for _ in range(10_000)])
    large = np.asarray([sample_beta_probability(0.65, 500.0, large_rng) for _ in range(10_000)])
    assert np.var(small) > 50.0 * np.var(large)


def test_beta_sampler_stabilizes_only_rounded_floating_point_endpoints() -> None:
    result = sample_beta_probability(
        0.9999991216640421,
        452.90871610020355,
        np.random.default_rng(0),
    )

    assert result == float(np.nextafter(1.0, 0.0))


def test_parameter_only_and_performance_only_uncertainty_are_separate(
    match_fixture: _MatchFixture,
) -> None:
    parameter_rng = np.random.default_rng(308)
    parameter_means = np.asarray(
        [
            sample_matchup_parameters(match_fixture.distribution, parameter_rng)
            .player_a_serving.by_component[ServeComponent.F]
            .mean
            for _ in range(50)
        ]
    )
    assert float(np.var(parameter_means)) > 0.0

    fixed_parameters = sample_matchup_parameters(
        match_fixture.distribution, np.random.default_rng(309)
    ).player_a_serving
    fixed_f = fixed_parameters.by_component[ServeComponent.F]
    performance_rng = np.random.default_rng(310)
    performance_draws = np.asarray(
        [
            sample_serve_performance(fixed_parameters, performance_rng).first_serve_in
            for _ in range(8000)
        ]
    )
    expected_variance = fixed_f.mean * (1.0 - fixed_f.mean) / (fixed_f.concentration + 1.0)
    assert float(np.mean(performance_draws)) == pytest.approx(fixed_f.mean, abs=0.02)
    assert float(np.var(performance_draws, ddof=1)) == pytest.approx(expected_variance, rel=0.1)


def test_matchup_sampling_does_not_revalidate_loaded_fits(
    match_fixture: _MatchFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_revalidation(_fitted: FittedServeComponent) -> FittedServeComponent:
        raise AssertionError("simulation hot path revalidated an already loaded fit")

    monkeypatch.setattr(
        "tennis_model.estimation.serve_components._revalidate_fit_for_prediction",
        unexpected_revalidation,
    )

    sample_matchup_parameters(match_fixture.distribution, np.random.default_rng(441))


def test_one_component_parameter_draw_is_replayed_for_both_directions(
    match_fixture: _MatchFixture,
) -> None:
    sampled = sample_matchup_parameters(match_fixture.distribution, np.random.default_rng(442))
    for joint in sampled.components:
        unseen = {effect.key: effect.value for effect in joint.unseen_effects}
        for direction, expected_mean in (
            (match_fixture.distribution.player_a_serving, joint.player_a_serving_mean),
            (match_fixture.distribution.player_b_serving, joint.player_b_serving_mean),
        ):
            fit = direction.by_component[joint.component].fit
            projection = project_component_parameters(
                fit,
                direction.context,
                joint.posterior.values,
            )
            eta = projection.base_linear_predictor + sum(
                effect.coefficient * unseen[effect.key] for effect in projection.unseen_effects
            )
            assert 1.0 / (1.0 + exp(-eta)) == expected_mean
            assert projection.predictive_concentration == joint.predictive_concentration


def test_component_beta_draws_are_conditionally_independent() -> None:
    fixed = ServingDirectionParameterDraw(
        server_id="p0",
        receiver_id="p1",
        components=tuple(
            BetaComponentParameters(component=component, mean=0.5, concentration=20.0)
            for component in ServeComponent
        ),
    )
    rng = np.random.default_rng(735)
    draws = np.asarray(
        [_performance_values(sample_serve_performance(fixed, rng)) for _ in range(6000)]
    )
    off_diagonal = np.corrcoef(draws, rowvar=False) - np.eye(5)
    assert float(np.max(np.abs(off_diagonal))) < 0.07


def test_full_sampler_is_exactly_seeded_and_keeps_direction_streams_separate(
    match_fixture: _MatchFixture,
) -> None:
    first = sample_match_performance(match_fixture.distribution, np.random.SeedSequence(4567))
    second = sample_match_performance(match_fixture.distribution, np.random.SeedSequence(4567))
    different = sample_match_performance(match_fixture.distribution, np.random.SeedSequence(4568))
    assert first == second
    assert first != different
    assert first.matchup_parameters.player_a_serving != first.matchup_parameters.player_b_serving
    assert first.player_a_serving != first.player_b_serving

    plan = derive_match_seed_plan(np.random.SeedSequence(7654))
    parameters = sample_matchup_parameters(
        match_fixture.distribution, generator_from_seed_reference(plan.parameter_draws)
    )
    expected_b = sample_serve_performance(
        parameters.player_b_serving,
        generator_from_seed_reference(plan.player_b_performance),
    )
    a_rng = generator_from_seed_reference(plan.player_a_performance)
    for _ in range(5):
        sample_serve_performance(parameters.player_a_serving, a_rng)
    observed_b = sample_serve_performance(
        parameters.player_b_serving,
        generator_from_seed_reference(plan.player_b_performance),
    )
    assert observed_b == expected_b


def test_seed_plan_is_stable_serializable_and_does_not_use_global_randomness(
    match_fixture: _MatchFixture,
) -> None:
    caller_seed = np.random.SeedSequence(np.uint64(1234))
    children_before = caller_seed.n_children_spawned
    plan = derive_match_seed_plan(caller_seed)
    assert caller_seed.n_children_spawned == children_before
    assert plan == derive_match_seed_plan(plan.root.to_seed_sequence())
    assert type(plan).model_validate_json(plan.model_dump_json()) == plan
    assert plan.bit_generator == "PCG64"
    assert (
        len(
            {
                plan.parameter_draws.spawn_key,
                plan.player_a_performance.spawn_key,
                plan.player_b_performance.spawn_key,
                plan.point_path.spawn_key,
                plan.retirement.spawn_key,
                plan.duration.spawn_key,
            }
        )
        == 6
    )
    assert plan.parameter_draws.spawn_key == (0,)
    assert plan.player_a_performance.spawn_key == (1,)
    assert plan.player_b_performance.spawn_key == (2,)
    assert plan.point_path.spawn_key == (3,)
    assert plan.retirement.spawn_key == (4,)
    assert plan.duration.spawn_key == (5,)
    assert plan.retirement_parameters.spawn_key == (4, 0)
    assert plan.retirement_boundaries.spawn_key == (4, 1)
    assert plan.duration_parameters.spawn_key == (5, 0)
    assert plan.duration_residual.spawn_key == (5, 1)
    with pytest.raises(ValidationError, match="greater than or equal to 4"):
        SeedReference(entropy=1, spawn_key=(), pool_size=1)
    with pytest.raises(ValidationError, match="nonnegative"):
        SeedReference(entropy=-1, spawn_key=(), pool_size=4)

    previously_used = np.random.SeedSequence(88)
    previously_used.spawn(2)
    continued = derive_match_seed_plan(previously_used)
    assert continued.root.n_children_spawned == 2
    assert continued.parameter_draws.spawn_key == (2,)
    assert continued.player_a_performance.spawn_key == (3,)
    assert continued.player_b_performance.spawn_key == (4,)
    assert continued.point_path.spawn_key == (5,)
    assert continued.retirement.spawn_key == (6,)
    assert continued.duration.spawn_key == (7,)
    assert continued.duration_parameters.spawn_key == (7, 0)
    assert continued.duration_residual.spawn_key == (7, 1)
    assert previously_used.n_children_spawned == 2

    before = np.random.get_state()
    sample_match_performance(match_fixture.distribution, np.random.SeedSequence(9921))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_v1_dependence_configuration_is_inert_and_non_independent_modes_fail() -> None:
    assert PerformanceDependenceSpec(loadings={}).loadings == ()
    with pytest.raises(ValidationError, match="cannot have loadings"):
        PerformanceDependenceSpec(loadings={ServeComponent.F: 0.2})
    with pytest.raises(ValidationError, match="independent"):
        PerformanceDependenceSpec.model_validate(
            {"mode": "one_factor_beta_copula", "loadings": {}, "validation_artifact_id": None}
        )


def test_match_parameter_record_is_stable_minimal_and_reconstructable(
    match_fixture: _MatchFixture,
) -> None:
    distribution = match_fixture.distribution
    payload = distribution.canonical_json()
    restored = restore_match_parameter_distribution(payload)

    assert payload == distribution.canonical_json()
    assert restored.to_record() == distribution.to_record()
    assert restored.canonical_json() == payload
    assert distribution.provenance.snapshot_id == distribution.snapshot.snapshot_id
    assert distribution.provenance.implementation_version == "match-parameters-laplace-beta/v1"
    assert distribution.provenance.fit_code_commit == "test-fixture-commit"
    assert tuple(item[0] for item in distribution.provenance.component_artifact_ids) == tuple(
        ServeComponent
    )
    assert '"covariance"' not in payload
    assert '"map_parameters"' not in payload
    assert '"dependence_mode":"independent"' in payload

    contradictory = json.loads(payload)
    contradictory["provenance"]["data_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="provenance contradicts"):
        MatchParameterRecord.model_validate(contradictory)


def test_fixed_match_performance_draw_feeds_many_milestone_four_points(
    match_fixture: _MatchFixture,
) -> None:
    full_draw = sample_match_performance(
        match_fixture.distribution, np.random.SeedSequence(20260829)
    )
    performance = full_draw.player_a_serving
    values_before = _performance_values(performance)
    point_rng = generator_from_seed_reference(full_draw.seed_plan.point_path)
    points = [
        generate_service_point(
            performance,
            point_rng,
            server_id="p0",
            receiver_id="p1",
        )
        for _ in range(400)
    ]

    assert _performance_values(performance) == values_before
    assert len({point.branch for point in points}) > 1
    assert len({point.server_won for point in points}) == 2
    assert all(point.server_id == "p0" and point.receiver_id == "p1" for point in points)


def test_synthetic_historical_cutoff_demonstration_is_reproducible(
    match_fixture: _MatchFixture,
) -> None:
    map_means = (
        _component_means(match_fixture.distribution.player_a_serving),
        _component_means(match_fixture.distribution.player_b_serving),
    )
    seeded = []
    for seed in (20260829, 20260830, 20260831):
        draw = sample_match_performance(
            match_fixture.distribution,
            np.random.SeedSequence(seed),
        )
        seeded.append(
            (
                seed,
                _performance_values(draw.player_a_serving),
                _performance_values(draw.player_b_serving),
            )
        )
    expected_map = (
        (0.805894413965, 0.441566354723, 0.935366166835, 0.231844303545, 0.882171870061),
        (0.325788335682, 0.026587166411, 0.263664229700, 0.049045271714, 0.128879385111),
    )
    expected_seeded = (
        (
            20260829,
            (0.738639177440, 0.538344324126, 0.922614208456, 0.241483738166, 0.897049733998),
            (0.270676072045, 0.029863988195, 0.370897941007, 0.043889183991, 0.076991669539),
        ),
        (
            20260830,
            (0.205181670882, 0.650040867734, 0.875479405414, 0.101384905283, 0.814465725134),
            (0.361692374132, 0.064812437134, 0.290591872874, 0.168052309644, 0.097394607784),
        ),
        (
            20260831,
            (0.914998277869, 0.209165960264, 0.946545187016, 0.286659735107, 0.897198176322),
            (0.346606227787, 0.044792706501, 0.302843338176, 0.028156614277, 0.106083796268),
        ),
    )
    np.testing.assert_allclose(map_means, expected_map, rtol=0.0, atol=1e-6)
    for observed, expected in zip(seeded, expected_seeded, strict=True):
        assert observed[0] == expected[0]
        np.testing.assert_allclose(observed[1:], expected[1:], rtol=0.0, atol=1e-6)
    assert match_fixture.snapshot.data_cutoff_utc == TEST_CUTOFF
    assert match_fixture.context.scheduled_start_utc > TEST_CUTOFF
