"""Prediction-lock orchestration over the frozen estimator, simulator, and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from tennis_model.data.historical_validation import (
    POINT_IN_TIME_VINTAGE_POLICY,
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
)
from tennis_model.estimation.duration_model import (
    FLOOR_DURATION_DISPLAY_POLICY,
    NEAREST_DURATION_DISPLAY_POLICY,
    UNRESOLVED_DURATION_DISPLAY_POLICY,
    DurationDisplayPolicy,
    DurationFitArtifact,
)
from tennis_model.estimation.inactivity import InactivityRecord, build_inactivity_record
from tennis_model.estimation.retirement import RetirementScenarioMixture
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.estimation.snapshot import ModelSnapshot
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.models import (
    AdaptiveMCPolicyRecord,
    CodeProvenance,
    DurationSimulationSummary,
    ExactScoreProbability,
    HistoricalTimeProvenance,
    HistoricalTrainingEligibilityProvenance,
    InformationBundle,
    LockRevisionReason,
    MatchSimulationSummary,
    PathCountPolicyRecord,
    PlayerSimulationSummary,
    PredictionSnapshot,
    PrimitiveSummary,
    PropEstimateRecord,
    PropGateRecord,
    PropSupportDecision,
    PropSupportStatus,
    ReplayLevel,
    RetainedArtifactRecord,
    RuntimeFingerprint,
    ServingDirectionSummary,
    SettlementPolicyRecord,
    SimulationProvenance,
    SourceManifestProvenance,
    SourceTimeProvenance,
    serialize_prop,
)
from tennis_model.locking.path_counts import (
    ADAPTIVE_MC_CS_V1_POLICY,
    FIXED_50K_V1_POLICY,
    AdaptiveMCPolicy,
    AdaptivePropDiagnostics,
    MCStoppingStatus,
    PathCountPolicy,
    adaptive_prop_diagnostics,
    escalation_reasons,
)
from tennis_model.locking.provenance import capture_runtime_fingerprint, enforce_dirty_tree_policy
from tennis_model.locking.store import LockStore
from tennis_model.props.policy import assess_prop_support, prop_generation_available
from tennis_model.props.rounding import (
    SPORTSPREDICT_SUBMISSION_POLICY,
    PlatformSubmissionPolicy,
)
from tennis_model.props.settlement import SettlementPolicy
from tennis_model.schemas import SourceManifest
from tennis_model.simulation.match import (
    BooleanCompositeSpec,
    MatchPath,
    PropEstimate,
    PropSpec,
    SimulationBatch,
    evaluate_settlement,
    simulate_matches,
)
from tennis_model.simulation.parameters import (
    MatchContext,
    MatchParameterDistribution,
    SeedReference,
    ServingDirectionDistribution,
    estimate_match,
    restore_match_parameter_distribution,
)


class LockCreationError(RuntimeError):
    pass


CORE_CAPABILITY_BLOCKERS: tuple[str, ...] = ()


_KNOWN_DURATION_DISPLAY_POLICIES = {
    policy.policy_version: policy
    for policy in (
        UNRESOLVED_DURATION_DISPLAY_POLICY,
        FLOOR_DURATION_DISPLAY_POLICY,
        NEAREST_DURATION_DISPLAY_POLICY,
    )
}


@dataclass(frozen=True, slots=True)
class ReproductionReport:
    reproduced: bool
    original_content_sha256: str
    prop_counts_match: bool
    summaries_match: bool
    replay_level: ReplayLevel = ReplayLevel.SAME_RUNTIME_EXACT
    runtime_matches: bool | None = None
    semantic_max_probability_error: float | None = None


def _base_lock_id(
    context: MatchContext,
    canonical_identity: CanonicalMatchIdentity | None,
) -> str:
    if canonical_identity is not None:
        return canonical_identity.base_lock_id
    from tennis_model.locking._json import sha256_json

    identity = {
        "framework": "v1.0",
        "tour": context.tour.value,
        "event": context.event,
        "round": context.round,
        "scheduled_start_utc": context.scheduled_start_utc.isoformat(),
        "players": sorted((context.player_a_id, context.player_b_id)),
    }
    return f"TMV1-{context.tour.value}-{sha256_json(identity)[:16]}"


def _analytic_hold_probability(point_probability: float) -> float:
    p = point_probability
    q = 1.0 - p
    return p**4 * (1.0 + 4.0 * q + 10.0 * q**2) + (20.0 * p**3 * q**3 * p**2 / (p**2 + q**2))


def _primitive_information(
    direction: ServingDirectionDistribution,
    component: ServeComponent,
) -> tuple[float | None, float | None, float | None, bool]:
    item = direction.by_component[component]
    information = next(
        (
            record
            for record in item.fit.diagnostics.player_information
            if record.player_id == direction.server_id
        ),
        None,
    )
    if information is None:
        return None, None, None, True
    return (
        information.weighted_trials,
        information.effective_matches,
        information.information_equivalent_trials,
        information.sparse_warning,
    )


def _direction_summary(direction: ServingDirectionDistribution) -> ServingDirectionSummary:
    distribution = direction.map_distribution
    predictions = (
        distribution.first_serve_in,
        distribution.ace_given_first_in,
        distribution.returnable_first_win,
        distribution.double_fault_given_second_opp,
        distribution.playable_second_win,
    )
    primitives = []
    for component, prediction in zip(ServeComponent, predictions, strict=True):
        weighted, matches, equivalent, sparse = _primitive_information(direction, component)
        primitives.append(
            PrimitiveSummary(
                component=component,
                map_mean=prediction.map_mean,
                linear_predictor_sd=prediction.linear_predictor_sd,
                predictive_concentration=prediction.predictive_concentration,
                weighted_trials=weighted,
                effective_matches=matches,
                information_equivalent_trials=equivalent,
                sparse_warning=sparse,
            )
        )
    f, a, q1, d, q2 = (item.map_mean for item in predictions)
    first = a + (1.0 - a) * q1
    second = (1.0 - d) * q2
    service = f * first + (1.0 - f) * second
    return ServingDirectionSummary(
        server_id=direction.server_id,
        receiver_id=direction.receiver_id,
        primitives=tuple(primitives),
        first_serve_win=first,
        second_serve_win=second,
        service_point_win=service,
        analytic_hold_probability=_analytic_hold_probability(service),
        ace_rate_per_service_point=f * a,
        double_fault_rate_per_service_point=(1.0 - f) * d,
    )


def summarize_simulations(
    batch: SimulationBatch,
    *,
    duration_artifact: DurationFitArtifact | None = None,
    duration_boundary_sensitive: bool = False,
) -> MatchSimulationSummary:
    """Aggregate only stored path outcomes; retirement remains unavailable if not generated."""

    if not batch.paths:
        raise ValueError("a prediction lock requires at least one simulated path")
    players = (batch.context.player_a_id, batch.context.player_b_id)
    started = tuple(path for path in batch.paths if path.started and not path.walkover)
    if not started:
        raise ValueError("a prediction lock requires started simulated paths")
    completed = tuple(path for path in started if path.completed)

    def mean_stat(player: str, field: str) -> float:
        values = [float(getattr(path.player_stats[player], field)) for path in started]
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    player_values = tuple(
        PlayerSimulationSummary(
            player_id=player,
            match_win_probability=sum(path.advancing_player_id == player for path in started)
            / len(started),
            expected_service_games=mean_stat(player, "service_games_played"),
            expected_breaks=mean_stat(player, "breaks_achieved"),
            expected_breaks_conceded=mean_stat(player, "breaks_conceded"),
            expected_aces=mean_stat(player, "aces"),
            expected_double_faults=mean_stat(player, "double_faults"),
            retirement_probability=(
                sum(path.retired_player_id == player for path in started) / len(started)
                if bool(batch.provenance.get("retirement_enabled", False))
                else None
            ),
        )
        for player in players
    )
    player_rows = (player_values[0], player_values[1])
    exact_counts: dict[tuple[str, int, int], int] = {}
    for path in completed:
        wins = (
            sum(result.winner_index == 0 for result in path.sets),
            sum(result.winner_index == 1 for result in path.sets),
        )
        winner = path.winner_id
        if winner is None:
            raise ValueError("completed path has no winner")
        index = 0 if winner == players[0] else 1
        key = (winner, wins[index], wins[1 - index])
        exact_counts[key] = exact_counts.get(key, 0) + 1
    exact_scores = (
        tuple(
            ExactScoreProbability(
                winner_id=winner,
                winner_sets=winner_sets,
                loser_sets=loser_sets,
                probability=count / len(completed),
            )
            for (winner, winner_sets, loser_sets), count in sorted(exact_counts.items())
        )
        if completed
        else ()
    )
    total_games = np.asarray([path.total_games for path in started], dtype=np.float64)
    quantile_values = np.quantile(total_games, (0.1, 0.5, 0.9), method="linear")
    quantiles = (
        float(quantile_values[0]),
        float(quantile_values[1]),
        float(quantile_values[2]),
    )
    retirement_generated = bool(batch.provenance.get("retirement_enabled", False))
    retirement_probability = (
        sum(path.retired_player_id is not None for path in batch.paths) / len(batch.paths)
        if retirement_generated
        else None
    )
    duration_summary: DurationSimulationSummary | None = None
    duration_paths = tuple(path for path in started if path.duration_latent is not None)
    if duration_artifact is None:
        if duration_paths:
            raise ValueError("duration paths require their fitted artifact for lock summary")
    else:
        if len(duration_paths) != len(started):
            raise ValueError("duration-enabled simulation omitted one or more path draws")
        duration_values = np.asarray(
            [path.duration_latent for path in duration_paths], dtype=np.float64
        )
        duration_quantiles = np.quantile(duration_values, (0.1, 0.5, 0.9), method="linear")
        event_effect = None
        if (
            batch.context.event == duration_artifact.current_event
            and batch.context.scheduled_start_utc.year == duration_artifact.current_event_year
        ):
            event_effect = next(
                (
                    item.value
                    for item in duration_artifact.coefficients
                    if item.name == "current_usopen_2026"
                ),
                None,
            )
        display_versions = {
            path.duration_display_policy_version for path in duration_paths
        }
        if len(display_versions) != 1 or None in display_versions:
            raise ValueError("duration paths do not share one versioned display policy")
        display_version = next(iter(display_versions))
        assert display_version is not None
        duration_summary = DurationSimulationSummary(
            expected_minutes=float(np.mean(duration_values)),
            quantiles=(
                float(duration_quantiles[0]),
                float(duration_quantiles[1]),
                float(duration_quantiles[2]),
            ),
            data_grade="B",
            artifact_id=duration_artifact.artifact_id,
            current_event_effect_minutes=event_effect,
            display_policy_version=display_version,
            display_boundary_sensitive=duration_boundary_sensitive,
        )
    return MatchSimulationSummary(
        players=player_rows,
        exact_scores=exact_scores,
        any_tiebreak_probability=sum(path.tiebreak_count > 0 for path in started) / len(started),
        deciding_set_probability=sum(path.sets_started == path.best_of for path in started)
        / len(started),
        expected_total_games=float(np.mean(total_games)),
        total_games_quantiles=quantiles,
        expected_total_breaks=sum(row.expected_breaks for row in player_rows),
        completion_probability=len(completed) / len(started),
        retirement_probability=retirement_probability,
        walkover_probability=sum(path.walkover for path in batch.paths) / len(batch.paths),
        duration=duration_summary,
    )


def _validate_summary(summary: MatchSimulationSummary) -> tuple[str, ...]:
    if abs(sum(row.match_win_probability for row in summary.players) - 1.0) > 1e-12:
        raise LockCreationError("match-winner probabilities do not sum to one")
    if (
        summary.completion_probability
        and abs(sum(cell.probability for cell in summary.exact_scores) - 1.0) > 1e-12
    ):
        raise LockCreationError("exact-score probabilities do not sum to one on completed paths")
    checks = [
        "MATCH_WIN_PROBABILITIES_SUM_TO_ONE",
        "EXACT_SCORE_PROBABILITIES_SUM_TO_ONE_CONDITIONAL_ON_COMPLETION",
        "FROZEN_V1_PERFORMANCE_DRAWS_INDEPENDENT",
        "ALL_PROP_ESTIMATES_SHARE_ONE_SIMULATION_BATCH",
    ]
    if summary.duration is not None:
        checks.extend(
            (
                "DURATION_CONDITIONAL_ON_REALIZED_JOINT_PATH",
                "DURATION_DRAW_CANNOT_ALTER_SCORE_OR_STATISTICS",
            )
        )
    return tuple(checks)


def _settlement_record(policy: SettlementPolicy) -> SettlementPolicyRecord:
    return SettlementPolicyRecord(
        version=policy.version,
        comparison_tie_is_no=policy.comparison_tie_is_no,
        walkover_voids_all=policy.walkover_voids_all,
        allow_policy_blocked=policy.allow_policy_blocked,
        description=policy.description,
    )


def _path_count_record(
    policy: PathCountPolicy | AdaptiveMCPolicy,
    *,
    execution_mode: Literal["production", "development", "test"],
) -> PathCountPolicyRecord | AdaptiveMCPolicyRecord:
    if isinstance(policy, AdaptiveMCPolicy):
        return AdaptiveMCPolicyRecord(
            checkpoints=policy.checkpoints,
            confidence_level=policy.confidence_level,
            beta_prior_a=policy.beta_prior_a,
            beta_prior_b=policy.beta_prior_b,
        )
    return PathCountPolicyRecord(
        version=(
            "fixed-50k/v1"
            if policy == FIXED_50K_V1_POLICY
            else "frozen-v1.0"
            if execution_mode == "production"
            else "explicit-development-test"
        ),
        standard_paths=policy.standard_paths,
        escalated_paths=policy.escalated_paths,
        minimum_settled_paths=policy.minimum_settled_paths,
        extreme_probability=policy.extreme_probability,
        integer_boundary_window=policy.integer_boundary_window,
        integer_boundary_standard_errors=policy.integer_boundary_standard_errors,
    )


def _duration_policy_for_version(version: str | None) -> DurationDisplayPolicy:
    if version is None:
        return UNRESOLVED_DURATION_DISPLAY_POLICY
    try:
        return _KNOWN_DURATION_DISPLAY_POLICIES[version]
    except KeyError as exc:
        raise LockCreationError(
            f"cannot replay unknown duration display policy {version!r}"
        ) from exc


def _prop_contains_duration(prop: PropSpec | BooleanCompositeSpec) -> bool:
    if isinstance(prop, PropSpec):
        return prop.kind == "DURATION_MIN"
    return any(_prop_contains_duration(child) for child in prop.exprs)


def _run(
    distribution: MatchParameterDistribution,
    props: tuple[PropSpec | BooleanCompositeSpec, ...],
    policy: SettlementPolicy,
    *,
    n_paths: int,
    seed: int | np.random.SeedSequence,
    first_server_id: str | None,
    trace_level: Literal["summary", "points"],
    duration_display_policy: DurationDisplayPolicy,
    path_start: int = 0,
) -> tuple[SimulationBatch, tuple[PropEstimate, ...]]:
    batch = simulate_matches(
        distribution,
        n_paths=n_paths,
        seed=seed,
        trace_level=trace_level,
        first_server_id=first_server_id,
        duration_display_policy=duration_display_policy,
        path_start=path_start,
    )
    return batch, tuple(evaluate_settlement(prop, batch, policy) for prop in props)


def _run_adaptive(
    distribution: MatchParameterDistribution,
    props: tuple[PropSpec | BooleanCompositeSpec, ...],
    settlement_policy: SettlementPolicy,
    adaptive_policy: AdaptiveMCPolicy,
    *,
    seed: int | np.random.SeedSequence,
    first_server_id: str | None,
    trace_level: Literal["summary", "points"],
    duration_display_policy: DurationDisplayPolicy,
) -> tuple[
    SimulationBatch,
    tuple[PropEstimate, ...],
    tuple[AdaptivePropDiagnostics, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    """Extend one deterministic path prefix until all props are integer-stable."""

    accumulated_paths: tuple[MatchPath, ...] = ()
    inspected: list[int] = []
    continuation_reasons: list[str] = []
    final_batch: SimulationBatch | None = None
    final_estimates: tuple[PropEstimate, ...] = ()
    final_diagnostics: tuple[AdaptivePropDiagnostics, ...] = ()
    prior_checkpoint = 0
    for checkpoint in adaptive_policy.checkpoints:
        chunk, _unused_chunk_estimates = _run(
            distribution,
            (),
            settlement_policy,
            n_paths=checkpoint - prior_checkpoint,
            seed=seed,
            first_server_id=first_server_id,
            trace_level=trace_level,
            duration_display_policy=duration_display_policy,
            path_start=prior_checkpoint,
        )
        accumulated_paths += chunk.paths
        final_batch = SimulationBatch(
            context=chunk.context,
            n_paths=checkpoint,
            seed_id=chunk.seed_id,
            paths=accumulated_paths,
            provenance={
                **chunk.provenance,
                "path_start": 0,
                "adaptive_mc_policy_version": adaptive_policy.version,
                "inspected_path_counts": tuple((*inspected, checkpoint)),
            },
        )
        final_estimates = tuple(
            evaluate_settlement(prop, final_batch, settlement_policy) for prop in props
        )
        terminal = checkpoint == adaptive_policy.maximum_paths
        final_diagnostics = tuple(
            adaptive_prop_diagnostics(estimate, adaptive_policy, terminal=terminal)
            for estimate in final_estimates
        )
        inspected.append(checkpoint)
        if all(
            item.stopping_status is MCStoppingStatus.INTEGER_STABLE
            for item in final_diagnostics
        ):
            break
        if not terminal:
            continuation_reasons.append(f"INTEGER_UNSTABLE_AFTER_{checkpoint}_PATHS")
        prior_checkpoint = checkpoint
    if final_batch is None:
        raise RuntimeError("adaptive policy contains no checkpoints")
    return (
        final_batch,
        final_estimates,
        final_diagnostics,
        tuple(inspected),
        tuple(continuation_reasons),
    )


def _resolve_inactivity_records(
    information: InformationBundle,
    context: MatchContext,
    supplied: tuple[InactivityRecord, ...],
) -> tuple[InactivityRecord, ...]:
    """Resolve target-only C6 facts carried by the current-match bundle."""

    if not information.player_inactivity:
        return supplied
    if supplied:
        raise LockCreationError(
            "C6 records must come from either the information bundle or the legacy argument, "
            "not both"
        )
    players = (context.player_a_id, context.player_b_id)
    by_player = {item.player_id: item for item in information.player_inactivity}
    if set(by_player) != set(players):
        raise LockCreationError(
            "current-match C6 information must contain exactly the two forecast players"
        )
    try:
        return tuple(
            build_inactivity_record(
                player_id=player_id,
                tour=context.tour,
                scheduled_start_local_date=context.scheduled_start_local_date,
                information_cutoff_utc=context.information_cutoff_utc,
                coverage=by_player[player_id].coverage,
                candidates=by_player[player_id].candidates,
            )
            for player_id in players
        )
    except (ValueError, RuntimeError) as exc:
        raise LockCreationError(f"cannot resolve current-match C6 information: {exc}") from exc


def create_prediction_lock(
    snapshot: ModelSnapshot,
    context: MatchContext,
    information: InformationBundle,
    props: tuple[PropSpec | BooleanCompositeSpec, ...],
    policy: SettlementPolicy,
    *,
    source_manifest: SourceManifest,
    code: CodeProvenance,
    seed: int | np.random.SeedSequence,
    store: LockStore | None = None,
    n_paths: int | None = None,
    first_server_id: str | None = None,
    trace_level: Literal["summary", "points"] = "summary",
    execution_mode: Literal["production", "development", "test"] = "production",
    path_count_policy: PathCountPolicy | AdaptiveMCPolicy = FIXED_50K_V1_POLICY,
    platform_submission_policy: PlatformSubmissionPolicy | None = None,
    allow_dirty: bool = False,
    created_at_utc: datetime | None = None,
    parent: PredictionSnapshot | None = None,
    revision_reason: LockRevisionReason | None = None,
    inactivity_records: tuple[InactivityRecord, ...] = (),
    retirement_scenario_mixtures: tuple[RetirementScenarioMixture, ...] = (),
    canonical_match_identity: CanonicalMatchIdentity | None = None,
    retained_artifacts: tuple[RetainedArtifactRecord, ...] = (),
    runtime_fingerprint: RuntimeFingerprint | None = None,
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
    training_eligibility: HistoricalTrainingEligibilityProvenance | None = None,
    duration_display_policy: DurationDisplayPolicy = UNRESOLVED_DURATION_DISPLAY_POLICY,
) -> PredictionSnapshot:
    """Estimate, simulate, evaluate, summarize, and optionally persist one lock.

    Pre-amendment development artifacts remain usable but are visibly incomplete.
    Production requires a v2 snapshot and cutoff-safe records for both B6 and C6.
    """

    enforce_dirty_tree_policy(code, allow_dirty=allow_dirty)
    if (
        platform_submission_policy is not None
        and platform_submission_policy != SPORTSPREDICT_SUBMISSION_POLICY
    ):
        raise LockCreationError("unknown external platform submission policy")
    if platform_submission_policy is not None and not isinstance(
        path_count_policy, AdaptiveMCPolicy
    ):
        raise LockCreationError("external platform transforms require adaptive Monte Carlo")
    if information.information_cutoff_utc != context.information_cutoff_utc:
        raise LockCreationError("information bundle and match context cutoffs differ")
    if information.scenario_id != context.information_scenario_id:
        raise LockCreationError("information bundle and match context scenarios differ")
    if not props:
        raise LockCreationError("prediction lock requires at least one prop")
    resolved_inactivity_records = _resolve_inactivity_records(
        information, context, inactivity_records
    )
    relevant_sources = source_manifest.sources_for_tour(snapshot.tour)
    if not relevant_sources:
        raise LockCreationError("source manifest has no pinned source for the model tour")
    late_finalized_sources = any(
        source.source_available_at_utc is not None
        and source.source_available_at_utc >= context.information_cutoff_utc
        for source in relevant_sources
    )
    if (
        late_finalized_sources
        and historical_validation_policy.mode
        is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE
    ):
        raise LockCreationError(
            "source manifest contains probability-relevant information unavailable at cutoff"
        )
    if (
        historical_validation_policy.mode
        is HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED
        and not historical_validation_policy.exact_date_history_complete
    ):
        raise LockCreationError(
            "retrospective-finalized lock blocked: exact-date history is incomplete"
        )
    if execution_mode == "production":
        if path_count_policy != FIXED_50K_V1_POLICY:
            raise LockCreationError("production locks require fixed-50k/v1")
        if not snapshot.b6_c6_complete:
            raise LockCreationError(
                "production lock blocked: snapshot lacks the required B6 retirement generator "
                "artifact and C6 inactivity configuration"
            )
        if canonical_match_identity is None:
            raise LockCreationError("production locks require a stable canonical match identity")
        if code.fingerprint_version != "complete-git-state/v2":
            raise LockCreationError("production locks require complete dirty-state provenance")
        if not retained_artifacts:
            raise LockCreationError("production locks require retained immutable artifacts")
        if store is None:
            raise LockCreationError("production locks require atomic publication to a lock store")
    if isinstance(path_count_policy, AdaptiveMCPolicy):
        if n_paths is not None:
            raise LockCreationError("adaptive path counts come only from the policy checkpoints")
        requested = path_count_policy.maximum_paths
    else:
        requested = path_count_policy.standard_paths if n_paths is None else n_paths
        if requested not in {path_count_policy.standard_paths, path_count_policy.escalated_paths}:
            raise LockCreationError(
                "path count must equal the declared standard or escalated count"
            )

    try:
        distribution = estimate_match(
            snapshot,
            context,
            inactivity_records=resolved_inactivity_records,
            retirement_scenario_mixtures=retirement_scenario_mixtures,
        )
    except (ValueError, RuntimeError) as exc:
        raise LockCreationError(f"cannot construct B6/C6 match parameters: {exc}") from exc
    source_manifest_provenance = SourceManifestProvenance.from_manifest(source_manifest)
    if distribution.retirement is not None and (
        distribution.retirement.source_manifest_id != source_manifest.manifest_version
        or distribution.retirement.source_manifest_sha256
        != source_manifest_provenance.manifest_sha256
    ):
        raise LockCreationError("B6 artifact source manifest differs from the lock manifest")
    if distribution.inactivity is not None and any(
        record.coverage.source_manifest_id != source_manifest.manifest_version
        or record.coverage.source_manifest_sha256 != source_manifest_provenance.manifest_sha256
        for record in distribution.inactivity.records
    ):
        raise LockCreationError("C6 coverage assertion differs from the lock source manifest")
    if distribution.duration is not None:
        if (
            distribution.duration.source_manifest_id != source_manifest.manifest_version
            or distribution.duration.source_manifest_sha256
            != source_manifest_provenance.manifest_sha256
        ):
            raise LockCreationError(
                "duration artifact source manifest differs from the lock manifest"
            )
        if distribution.duration.information_cutoff_utc > context.information_cutoff_utc:
            raise LockCreationError("duration artifact contains information after the match cutoff")
    if distribution.retirement is not None:
        coverage = distribution.retirement.source_coverage
        if coverage.fit_input_date_eligibility_verified:
            if training_eligibility is None:
                raise LockCreationError(
                    "current exact-dated B6 eligibility requires retained training provenance"
                )
            if training_eligibility.tour is not snapshot.tour:
                raise LockCreationError("training eligibility belongs to another tour")
            if (
                training_eligibility.source_manifest_sha256
                != source_manifest_provenance.manifest_sha256
            ):
                raise LockCreationError("training eligibility source manifest differs")
            if (
                training_eligibility.source_sha256s != coverage.source_sha256s
                or training_eligibility.crosswalk_sha256s != coverage.crosswalk_sha256s
            ):
                raise LockCreationError("B6 and lock training evidence hashes differ")
            b6_record = training_eligibility.records[-1]
            if (
                b6_record.component != "B6"
                or b6_record.included_exact_dated_rows
                != coverage.included_exact_dated_matches
                or b6_record.excluded_undated_candidate_rows
                != coverage.excluded_undated_matches
                or b6_record.included_unweighted_player_starts
                != coverage.included_exact_dated_player_starts
                or b6_record.excluded_undated_player_starts
                != coverage.excluded_undated_player_starts
            ):
                raise LockCreationError("B6 exposure differs from lock training provenance")
    if execution_mode == "production" and distribution.retirement is not None:
        permitted_evidence_hashes = {
            *(source.sha256 for source in source_manifest.sources),
            *(item.source_sha256 for item in information.items),
        }
        if any(
            not scenario.central and scenario.source_sha256 not in permitted_evidence_hashes
            for mixture in distribution.retirement.scenario_mixtures
            for scenario in mixture.scenarios
        ):
            raise LockCreationError(
                "noncentral B6 scenario evidence is not pinned in lock information"
            )
    duration_available = distribution.duration is not None
    if duration_available and canonical_match_identity is None:
        raise LockCreationError(
            "duration-enabled immutable locks require canonical identity and prediction-lock/v4"
        )
    if duration_available:
        known_policy = _KNOWN_DURATION_DISPLAY_POLICIES.get(
            duration_display_policy.policy_version
        )
        if known_policy != duration_display_policy:
            raise LockCreationError(
                "duration simulation requires one of the frozen, explicitly known display policies"
            )
        if (
            execution_mode == "production"
            and duration_display_policy != UNRESOLVED_DURATION_DISPLAY_POLICY
        ):
            raise LockCreationError(
                "production duration locks require the unresolved official-minute display "
                "policy until an explicit conversion rule is frozen"
            )
    requested_support = tuple(
        assess_prop_support(prop, duration_available=duration_available) for prop in props
    )
    generated_props = tuple(
        prop
        for prop in props
        if prop_generation_available(prop, duration_available=duration_available)
    )
    generated_support = tuple(
        decision
        for prop, decision in zip(props, requested_support, strict=True)
        if prop_generation_available(prop, duration_available=duration_available)
    )
    gated_props = tuple(
        PropGateRecord(
            prop_id=serialized.prop_id,
            prop=serialized,
            support_status=decision.status,
            reason_code=decision.reason_code or "NOT_IMPLEMENTED",
            detail=decision.detail or "prop generation is not implemented",
        )
        for prop, decision in zip(props, requested_support, strict=True)
        if not prop_generation_available(prop, duration_available=duration_available)
        for serialized in (serialize_prop(prop),)
    )
    adaptive_diagnostics: tuple[AdaptivePropDiagnostics, ...] | None = None
    inspected_path_counts: tuple[int, ...] = ()
    if isinstance(path_count_policy, AdaptiveMCPolicy):
        (
            batch,
            estimates,
            adaptive_diagnostics,
            inspected_path_counts,
            reasons,
        ) = _run_adaptive(
            distribution,
            generated_props,
            policy,
            path_count_policy,
            seed=seed,
            first_server_id=first_server_id,
            trace_level=trace_level,
            duration_display_policy=duration_display_policy,
        )
        escalated = batch.n_paths > path_count_policy.checkpoints[0]
    else:
        batch, estimates = _run(
            distribution,
            generated_props,
            policy,
            n_paths=requested,
            seed=seed,
            first_server_id=first_server_id,
            trace_level=trace_level,
            duration_display_policy=duration_display_policy,
        )
        reasons = escalation_reasons(estimates, path_count_policy)
        escalated = False
        if (
            requested == path_count_policy.standard_paths
            and path_count_policy.escalated_paths > path_count_policy.standard_paths
            and reasons
        ):
            batch, estimates = _run(
                distribution,
                generated_props,
                policy,
                n_paths=path_count_policy.escalated_paths,
                seed=seed,
                first_server_id=first_server_id,
                trace_level=trace_level,
                duration_display_policy=duration_display_policy,
            )
            escalated = True

    support_decisions: list[PropSupportDecision] = []
    for index, (decision, estimate) in enumerate(
        zip(generated_support, estimates, strict=True)
    ):
        if decision.status is PropSupportStatus.SUPPORTED and estimate.unresolved_paths:
            if estimate.display_policy_version is not None:
                decision = PropSupportDecision(
                    status=PropSupportStatus.SETTLEMENT_BLOCKED,
                    reason_code="DURATION_DISPLAY_POLICY_BOUNDARY_SENSITIVE",
                    detail=(
                        "one or more paths differ under the preserved official-minute "
                        "display-policy candidates"
                    ),
                )
            else:
                decision = PropSupportDecision(
                    status=PropSupportStatus.SETTLEMENT_BLOCKED,
                    reason_code="RETIREMENT_SETTLEMENT_EDGE_UNRESOLVED",
                    detail="one or more retirement paths require an unresolved settlement ruling",
                )
        if (
            decision.status is PropSupportStatus.SUPPORTED
            and adaptive_diagnostics is not None
            and adaptive_diagnostics[index].stopping_status is MCStoppingStatus.UNAVAILABLE
        ):
            decision = PropSupportDecision(
                status=PropSupportStatus.DATA_UNAVAILABLE,
                reason_code="ZERO_SETTLED_PATHS",
                detail="no simulated path settled this prop by the final adaptive checkpoint",
            )
        support_decisions.append(decision)
    duration_boundary_sensitive = any(
        estimate.display_policy_version is not None and estimate.unresolved_paths > 0
        for estimate in estimates
    )
    summary = summarize_simulations(
        batch,
        duration_artifact=distribution.duration,
        duration_boundary_sensitive=duration_boundary_sensitive,
    )
    warnings: list[str] = []
    if distribution.retirement is None:
        warnings.extend(
            (
                "RETIREMENT_MODEL_UNAVAILABLE: pre-amendment development/test lock",
                "INACTIVITY_UNCERTAINTY_RULE_UNAVAILABLE: pre-amendment development/test lock",
            )
        )
    if code.dirty:
        warnings.append(f"DIRTY_CODE_TREE_RECORDED:{code.diff_sha256}")
    if training_eligibility is not None and training_eligibility.warning is not None:
        warnings.append(training_eligibility.warning)
    if context.indoor is None:
        warnings.append("UNKNOWN_INDOOR_ROOF_STATE")
    if duration_boundary_sensitive:
        warnings.append("DURATION_OFFICIAL_MINUTE_CONVERSION_BOUNDARY_SENSITIVE")
    warnings.extend(
        f"MISSING_CURRENT_CONDITION:{item}" for item in information.missing_current_conditions
    )
    warnings.extend(f"PROP_{gate.support_status.value}:{gate.reason_code}" for gate in gated_props)
    direction_summaries = (
        _direction_summary(distribution.player_a_serving),
        _direction_summary(distribution.player_b_serving),
    )
    if any(
        primitive.sparse_warning
        for direction in direction_summaries
        for primitive in direction.primitives
    ):
        warnings.append("SPARSE_PLAYER_COMPONENT_HISTORY")
    for estimate in estimates:
        if estimate.unresolved_paths:
            warnings.append(f"POLICY_BLOCKED_PROP:{type(estimate.prop).__name__}")
        if estimate.probability_settled < 0.5:
            warnings.append("LOW_SETTLEMENT_PROBABILITY")
        if estimate.mc_standard_error > 0.005:
            warnings.append("HIGH_MONTE_CARLO_UNCERTAINTY")
    if adaptive_diagnostics is not None:
        if any(
            item.stopping_status is MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE
            for item in adaptive_diagnostics
        ):
            warnings.append("INTEGER_BOUNDARY_SENSITIVE_AT_70000_PATHS")
        if any(
            item.stopping_status is MCStoppingStatus.UNAVAILABLE
            for item in adaptive_diagnostics
        ):
            warnings.append("ZERO_SETTLED_PATHS_MC_ESTIMATE_UNAVAILABLE")
    warnings.extend(
        f"PROP_{decision.status.value}:{decision.reason_code}"
        for decision in support_decisions
        if decision.status is not PropSupportStatus.SUPPORTED
    )
    warnings = list(dict.fromkeys(warnings))

    settlement_record = _settlement_record(policy)
    path_count_record = _path_count_record(
        path_count_policy,
        execution_mode=execution_mode,
    )
    from tennis_model.locking._json import sha256_json

    lock_configuration_sha256 = sha256_json(
        {
            "model_config_sha256": snapshot.config_hash,
            "settlement_policy": settlement_record.model_dump(mode="json"),
            "information_scenario_id": information.scenario_id,
            "trace_level": trace_level,
            "first_server_id": first_server_id,
            "path_count_mode": execution_mode,
            "path_count_policy": path_count_record.model_dump(mode="json"),
            "performance_dependence_mode": "independent",
            "historical_validation_policy": historical_validation_policy.model_dump(mode="json"),
            **(
                {}
                if platform_submission_policy is None
                else {
                    "platform_submission_policy_version": platform_submission_policy.version
                }
            ),
            **(
                {}
                if distribution.retirement is None or distribution.inactivity is None
                else {
                    "retirement_artifact_id": distribution.retirement.artifact_id,
                    "retirement_scenario_mixtures": [
                        item.model_dump(mode="json")
                        for item in distribution.retirement.scenario_mixtures
                    ],
                    "inactivity_configuration_artifact_id": (
                        distribution.inactivity.configuration_artifact_id
                    ),
                    "inactivity_record_hashes": [
                        item.sha256 for item in distribution.inactivity.records
                    ],
                    "ordinary_termination_before_retirement_version": (
                        "ordinary-terminal-bypass-before-b6/v1"
                    ),
                }
            ),
            **(
                {}
                if training_eligibility is None
                else {"training_eligibility": training_eligibility.model_dump(mode="json")}
            ),
            **(
                {}
                if distribution.duration is None
                else {
                    "duration_artifact_id": distribution.duration.artifact_id,
                    "duration_display_policy_version": duration_display_policy.policy_version,
                    "duration_rng_stream_version": (
                        "seedsequence-duration-parameters-residual/v1"
                    ),
                }
            ),
        }
    )

    base_id = _base_lock_id(context, canonical_match_identity)
    if parent is None:
        revision = 1
        parent_revision = None
        parent_hash = None
        reason = revision_reason or LockRevisionReason(category="initial", summary="Initial lock")
    else:
        if parent.base_lock_id != base_id:
            raise LockCreationError("parent lock belongs to a different scheduled matchup")
        revision = parent.revision + 1
        parent_revision = parent.revision
        parent_hash = parent.content_sha256
        if revision_reason is None or revision_reason.category == "initial":
            raise LockCreationError("a lock revision requires a structured non-initial reason")
        reason = revision_reason
    created = datetime.now(UTC) if created_at_utc is None else created_at_utc
    runtime = runtime_fingerprint or capture_runtime_fingerprint(
        simulator_algorithm_version=(
            "joint-match-simulator-duration/v2"
            if distribution.duration is not None
            else "joint-match-simulator/v1"
        ),
        chunk_size=batch.n_paths,
        thread_count=1,
        process_count=1,
    )
    historical_time = HistoricalTimeProvenance(
        information_cutoff_utc=context.information_cutoff_utc,
        training_data_cutoff_utc=snapshot.data_cutoff_utc,
        artifact_created_at_utc=created,
        sources=tuple(
            SourceTimeProvenance(
                source_id=source.source_id,
                source_effective_through=min(
                    source.verified_coverage.last_match_date,
                    snapshot.data_cutoff_utc.date(),
                ),
                source_effective_at_utc=source.source_effective_at_utc,
                source_available_at_utc=source.source_available_at_utc,
                information_availability_rule=(
                    f"{source.row_date_semantics.value}+{source.availability_lag_days}d"
                ),
                retrieved_at_utc=source.retrieved_at_utc,
                verified_at_utc=source.verified_coverage.verified_at_utc,
            )
            for source in relevant_sources
        ),
        historical_validation_policy=historical_validation_policy,
    )
    lock = PredictionSnapshot(
        schema_version=(
            "prediction-lock/v4"
            if canonical_match_identity is not None and distribution.duration is not None
            else "prediction-lock/v3"
            if canonical_match_identity is not None
            else "prediction-lock/v2"
            if distribution.retirement is not None
            else "prediction-lock/v1"
        ),
        identity_schema_version=(
            "canonical-match-identity/v2"
            if canonical_match_identity is not None
            else "legacy-forecast-state/v1"
        ),
        base_lock_id=base_id,
        canonical_match_identity=canonical_match_identity,
        revision=revision,
        created_at_utc=created,
        parent_revision=parent_revision,
        parent_content_sha256=parent_hash,
        revision_reason=reason,
        framework_version=snapshot.framework_version,
        settlement_policy=settlement_record,
        context=context,
        information=information,
        source_manifest=source_manifest_provenance,
        code=code,
        historical_validation_policy=historical_validation_policy,
        historical_time=(historical_time if canonical_match_identity is not None else None),
        training_eligibility=training_eligibility,
        runtime=(runtime if canonical_match_identity is not None else None),
        retained_artifacts=retained_artifacts,
        lock_configuration_sha256=lock_configuration_sha256,
        match_parameters=distribution.to_record(),
        parameter_summaries=direction_summaries,
        simulation=SimulationProvenance(
            seed_id=batch.seed_id,
            requested_paths=requested,
            actual_paths=batch.n_paths,
            trace_level=trace_level,
            first_server_id=first_server_id,
            path_count_mode=execution_mode,
            escalated=escalated,
            escalation_reasons=reasons,
            path_count_policy=path_count_record,
            inspected_path_counts=inspected_path_counts,
            platform_submission_policy_version=(
                None
                if platform_submission_policy is None
                else platform_submission_policy.version
            ),
            retirement_rng_stream_version=(
                None
                if distribution.retirement is None
                else "seedsequence-retirement-parameters-boundaries/v1"
            ),
            ordinary_termination_before_retirement_version=(
                None if distribution.retirement is None else "ordinary-terminal-bypass-before-b6/v1"
            ),
            duration_rng_stream_version=(
                None
                if distribution.duration is None
                else "seedsequence-duration-parameters-residual/v1"
            ),
            duration_display_policy_version=(
                None
                if distribution.duration is None
                else duration_display_policy.policy_version
            ),
            seed_policy_version="production-seed-policy/v1",
            chunk_size=batch.n_paths,
            thread_count=1,
            process_count=1,
        ),
        match_summary=summary,
        prop_estimates=tuple(
            PropEstimateRecord.from_estimate(
                estimate,
                decision,
                data_grade=("B" if _prop_contains_duration(estimate.prop) else "A"),
                adaptive_diagnostics=diagnostics,
                platform_submission_policy=platform_submission_policy,
            )
            for estimate, decision, diagnostics in zip(
                estimates,
                support_decisions,
                (
                    adaptive_diagnostics
                    if adaptive_diagnostics is not None
                    else tuple(None for _ in estimates)
                ),
                strict=True,
            )
        ),
        prop_gates=gated_props,
        retirement_model_artifact_id=(
            None if distribution.retirement is None else distribution.retirement.artifact_id
        ),
        inactivity_configuration_artifact_id=(
            None
            if distribution.inactivity is None
            else distribution.inactivity.configuration_artifact_id
        ),
        duration_model_artifact_id=(
            None if distribution.duration is None else distribution.duration.artifact_id
        ),
        warnings=tuple(warnings),
        validation_checks=_validate_summary(summary),
    )
    if store is not None:
        store.write(lock)
    return lock


def reproduce_prediction_lock(
    lock: PredictionSnapshot,
    *,
    replay_level: ReplayLevel = ReplayLevel.SAME_RUNTIME_EXACT,
    semantic_tolerance: float = 1e-12,
    current_runtime: RuntimeFingerprint | None = None,
) -> ReproductionReport:
    """Verify or rerun a lock under one explicitly named replay guarantee."""

    if replay_level is ReplayLevel.HASH_VERIFICATION:
        return ReproductionReport(
            reproduced=True,
            original_content_sha256=lock.content_sha256,
            prop_counts_match=True,
            summaries_match=True,
            replay_level=replay_level,
            runtime_matches=None,
        )
    if semantic_tolerance < 0:
        raise ValueError("semantic replay tolerance cannot be negative")

    from tennis_model.locking.models import deserialize_prop

    distribution = restore_match_parameter_distribution(lock.match_parameters.canonical_json())
    seed = SeedReference.model_validate_json(lock.simulation.seed_id).to_seed_sequence()
    policy = SettlementPolicy(
        version=lock.settlement_policy.version,
        comparison_tie_is_no=lock.settlement_policy.comparison_tie_is_no,
        walkover_voids_all=lock.settlement_policy.walkover_voids_all,
        allow_policy_blocked=lock.settlement_policy.allow_policy_blocked,
        description=lock.settlement_policy.description,
    )
    props = tuple(deserialize_prop(item.prop) for item in lock.prop_estimates)
    duration_display_policy = _duration_policy_for_version(
        lock.simulation.duration_display_policy_version
    )
    batch, estimates = _run(
        distribution,
        props,
        policy,
        n_paths=lock.simulation.actual_paths,
        seed=seed,
        first_server_id=lock.simulation.first_server_id,
        trace_level=lock.simulation.trace_level,
        duration_display_policy=duration_display_policy,
    )
    replay_adaptive_diagnostics: tuple[AdaptivePropDiagnostics, ...] | None = None
    replay_platform_policy: PlatformSubmissionPolicy | None = None
    if isinstance(lock.simulation.path_count_policy, AdaptiveMCPolicyRecord):
        adaptive_policy = AdaptiveMCPolicy(
            checkpoints=lock.simulation.path_count_policy.checkpoints,
            confidence_level=lock.simulation.path_count_policy.confidence_level,
            beta_prior_a=lock.simulation.path_count_policy.beta_prior_a,
            beta_prior_b=lock.simulation.path_count_policy.beta_prior_b,
            confidence_sequence_method=(
                lock.simulation.path_count_policy.confidence_sequence_method
            ),
            model_rounding_policy_version=(
                lock.simulation.path_count_policy.model_rounding_policy_version
            ),
        )
        replay_adaptive_diagnostics = tuple(
            adaptive_prop_diagnostics(estimate, adaptive_policy, terminal=True)
            for estimate in estimates
        )
        platform_versions = {
            item.platform_submission_policy_version
            for item in lock.prop_estimates
            if item.platform_submission_policy_version is not None
        }
        if platform_versions:
            if platform_versions != {SPORTSPREDICT_SUBMISSION_POLICY.version}:
                raise LockCreationError("cannot replay unknown external platform submission policy")
            replay_platform_policy = SPORTSPREDICT_SUBMISSION_POLICY
    support = tuple(
        PropSupportDecision(
            status=item.support_status,
            reason_code=item.support_reason_code,
            detail=item.support_detail,
        )
        for item in lock.prop_estimates
    )
    records = tuple(
        PropEstimateRecord.from_estimate(
            estimate,
            decision,
            data_grade=("B" if _prop_contains_duration(estimate.prop) else "A"),
            adaptive_diagnostics=diagnostics,
            platform_submission_policy=replay_platform_policy,
        )
        for estimate, decision, diagnostics in zip(
            estimates,
            support,
            (
                replay_adaptive_diagnostics
                if replay_adaptive_diagnostics is not None
                else tuple(None for _ in estimates)
            ),
            strict=True,
        )
    )
    summary = summarize_simulations(
        batch,
        duration_artifact=distribution.duration,
        duration_boundary_sensitive=any(
            estimate.display_policy_version is not None and estimate.unresolved_paths > 0
            for estimate in estimates
        ),
    )
    prop_match = records == lock.prop_estimates
    summary_match = summary == lock.match_summary
    runtime = current_runtime or capture_runtime_fingerprint(
        simulator_algorithm_version=(
            "joint-match-simulator-duration/v2"
            if distribution.duration is not None
            else "joint-match-simulator/v1"
        ),
        chunk_size=lock.simulation.actual_paths,
        thread_count=lock.simulation.thread_count,
        process_count=lock.simulation.process_count,
    )
    runtime_matches = lock.runtime is None or runtime == lock.runtime
    probability_errors = []
    for observed, expected in zip(records, lock.prop_estimates, strict=True):
        if observed.probability_raw is None or expected.probability_raw is None:
            probability_errors.append(
                0.0 if observed.probability_raw is expected.probability_raw else float("inf")
            )
        else:
            probability_errors.append(abs(observed.probability_raw - expected.probability_raw))
    probability_error = max(probability_errors, default=0.0)
    reproduced = (
        prop_match and summary_match and runtime_matches
        if replay_level is ReplayLevel.SAME_RUNTIME_EXACT
        else probability_error <= semantic_tolerance
    )
    return ReproductionReport(
        reproduced=reproduced,
        original_content_sha256=lock.content_sha256,
        prop_counts_match=prop_match,
        summaries_match=summary_match,
        replay_level=replay_level,
        runtime_matches=runtime_matches,
        semantic_max_probability_error=probability_error,
    )
