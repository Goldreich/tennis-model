"""Frozen I3 validation infrastructure; comparators are evaluation-only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import log
from typing import Literal

import numpy as np
from scipy.special import expit, ndtri  # type: ignore[import-untyped]
from scipy.stats import kstest  # type: ignore[import-untyped]

from tennis_model.calibration.ledger import CalibrationLedgerRow
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.locking.models import PropSupportStatus
from tennis_model.schemas import Tour
from tennis_model.simulation.match import SimulationBatch


class ComparatorKind(StrEnum):
    TOUR_EVENT_BASELINE = "TOUR_EVENT_BASELINE"
    RAW_LOGIT_BASELINE = "RAW_LOGIT_BASELINE"
    SURFACE_ELO_BENCHMARK = "SURFACE_ELO_BENCHMARK"
    STATIONARY_POINT_MODEL = "STATIONARY_POINT_MODEL"
    TENNIS_MODEL_V1 = "TENNIS_MODEL_V1"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"
    UNDERPOWERED = "UNDERPOWERED"


@dataclass(frozen=True, slots=True)
class ComparatorForecast:
    comparator: ComparatorKind
    target_id: str
    event_id: str
    tour: Tour
    forecast_cutoff_utc: datetime
    target_start_utc: datetime
    artifact_created_at_utc: datetime
    probability: float

    def __post_init__(self) -> None:
        for name, value in (
            ("forecast_cutoff_utc", self.forecast_cutoff_utc),
            ("target_start_utc", self.target_start_utc),
            ("artifact_created_at_utc", self.artifact_created_at_utc),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.forecast_cutoff_utc.astimezone(UTC) >= self.target_start_utc.astimezone(UTC):
            raise ValueError("comparator forecast cutoff must precede the historical target")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("comparator probability must lie in [0, 1]")

    @property
    def evaluation_only(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ComparatorResolution:
    forecast: ComparatorForecast
    outcome: int
    outcome_available_at_utc: datetime

    def __post_init__(self) -> None:
        if self.outcome not in {0, 1}:
            raise ValueError("comparator outcome must be zero or one")
        if (
            self.outcome_available_at_utc.tzinfo is None
            or self.outcome_available_at_utc.utcoffset() is None
        ):
            raise ValueError("outcome availability must be timezone-aware")
        if self.outcome_available_at_utc <= self.forecast.forecast_cutoff_utc:
            raise ValueError("comparator outcome crossed the historical forecast cutoff")


@dataclass(frozen=True, slots=True)
class CalibrationRegression:
    intercept: float | None
    slope: float | None
    intercept_standard_error: float | None
    slope_standard_error: float | None
    observations: int
    probability_clip: float
    status: GateStatus


@dataclass(frozen=True, slots=True)
class ComparatorMetricRow:
    comparator: ComparatorKind
    scope: Literal["OVERALL", "TOUR", "EVENT"]
    tour: Tour | None
    event_id: str | None
    observations: int
    brier: float | None
    brier_interval: BootstrapInterval
    brier_skill: float | None
    brier_skill_baseline: ComparatorKind | None
    brier_skill_observations: int
    log_loss: float | None
    calibration: CalibrationRegression
    calibration_bootstrap: CalibrationBootstrap


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float | None
    upper: float | None
    replicates: int
    seed: int


@dataclass(frozen=True, slots=True)
class CalibrationBootstrap:
    intercept: BootstrapInterval
    slope: BootstrapInterval


@dataclass(frozen=True, slots=True)
class ReliabilityDiagnosticBin:
    lower: float
    upper: float
    upper_inclusive: bool
    observations: int
    mean_probability: float | None
    observed_rate: float | None
    observed_rate_interval: BootstrapInterval
    mean_brier: float | None
    mean_brier_interval: BootstrapInterval


@dataclass(frozen=True, slots=True)
class OpportunityObservation:
    kind: Literal["HOLD", "BREAK"]
    event_id: str
    tour: Tour
    predicted_probability: float
    observed: int

    def __post_init__(self) -> None:
        if not 0 <= self.predicted_probability <= 1 or self.observed not in {0, 1}:
            raise ValueError("opportunity prediction/outcome is invalid")


@dataclass(frozen=True, slots=True)
class OpportunityDecile:
    kind: Literal["HOLD", "BREAK"]
    tour: Tour
    decile: int
    lower: float
    upper: float
    opportunities: int
    predicted_rate: float
    observed_rate: float
    observed_rate_interval: BootstrapInterval


@dataclass(frozen=True, slots=True)
class DiscretePredictiveObservation:
    observation_id: str
    event_id: str
    observed_value: int
    mass: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        values = tuple(value for value, _ in self.mass)
        probabilities = tuple(probability for _, probability in self.mass)
        if not self.mass or values != tuple(sorted(set(values))):
            raise ValueError("discrete support must be unique and ordered")
        if any(probability < 0 for probability in probabilities) or not np.isclose(
            sum(probabilities), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("discrete predictive mass must be nonnegative and sum to one")
        if self.observed_value not in values:
            raise ValueError("observed value lies outside predictive support")


@dataclass(frozen=True, slots=True)
class DiscretePITReport:
    seed: int
    observations: int
    values: tuple[float, ...]
    ks_statistic: float | None
    ks_p_value: float | None


@dataclass(frozen=True, slots=True)
class PrimitivePredictiveObservation:
    observation_id: str
    event_id: str
    tour: Tour
    component: ServeComponent
    observed: int
    predictive_mass: tuple[float, ...]
    interval_lower: int
    interval_upper: int

    def __post_init__(self) -> None:
        if self.observed < 0 or self.observed >= len(self.predictive_mass):
            raise ValueError("primitive observation lies outside its predictive mass")
        if any(value < 0 for value in self.predictive_mass) or not np.isclose(
            sum(self.predictive_mass), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("primitive predictive mass must sum to one")
        if not 0 <= self.interval_lower <= self.interval_upper < len(self.predictive_mass):
            raise ValueError("primitive predictive interval is invalid")


@dataclass(frozen=True, slots=True)
class PrimitiveDistributionDiagnostic:
    tour: Tour
    component: ServeComponent
    observations: int
    mean_randomized_quantile_residual: float | None
    interval_coverage: float | None
    mean_log_predictive_density: float | None
    seed: int


@dataclass(frozen=True, slots=True)
class ExactScoreCoherenceReport:
    best_of: int
    completed_paths: int
    match_win_probability: tuple[tuple[str, float], ...]
    exact_score_win_probability: tuple[tuple[str, float], ...]
    exact_score_distribution: tuple[tuple[str, int, int, float], ...]
    deciding_set_probability: float | None
    deciding_set_probability_from_scores: float | None
    legal_support: bool
    coherent: bool


@dataclass(frozen=True, slots=True)
class PropFamilyValidation:
    family: str
    scope: Literal["OVERALL", "TOUR", "EVENT"]
    tour: Tour | None
    event_id: str | None
    targets: int
    settled: int
    void: int
    unavailable: int
    policy_disabled: int
    raw_brier: float | None
    submitted_brier: float | None
    quantization_loss: float | None
    mean_probability: float | None
    observed_frequency: float | None
    calibration: CalibrationRegression
    reliability: tuple[ReliabilityDiagnosticBin, ...]


@dataclass(frozen=True, slots=True)
class CorrectnessGate:
    name: str
    status: GateStatus
    detail: str


@dataclass(frozen=True, slots=True)
class StatisticalDiagnostic:
    name: str
    status: GateStatus
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_version: Literal["validation-report/v1"]
    hard_correctness_gates: tuple[CorrectnessGate, ...]
    statistical_diagnostics: tuple[StatisticalDiagnostic, ...]
    comparator_rows: tuple[ComparatorMetricRow, ...]
    comparator_rows_by_tour: tuple[ComparatorMetricRow, ...]
    comparator_rows_by_event: tuple[ComparatorMetricRow, ...]
    prop_families: tuple[PropFamilyValidation, ...]
    prop_families_by_tour: tuple[PropFamilyValidation, ...]
    prop_families_by_event: tuple[PropFamilyValidation, ...]
    hold_break_deciles: tuple[OpportunityDecile, ...]
    game_distribution_pit: DiscretePITReport | None
    primitive_diagnostics: tuple[PrimitiveDistributionDiagnostic, ...]
    exact_score_coherence: tuple[ExactScoreCoherenceReport, ...]
    genuine_historical_validation: bool

    @property
    def operational_correctness_passed(self) -> bool:
        return bool(self.hard_correctness_gates) and all(
            item.status is GateStatus.PASS for item in self.hard_correctness_gates
        )


def brier_skill(model_brier: float, baseline_brier: float) -> float | None:
    if model_brier < 0 or baseline_brier < 0:
        raise ValueError("Brier scores cannot be negative")
    if baseline_brier == 0:
        return None
    return 1.0 - model_brier / baseline_brier


def calibration_regression(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    probability_clip: float = 1e-6,
) -> CalibrationRegression:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if probabilities.ndim != 1 or outcomes.shape != probabilities.shape:
        raise ValueError("calibration arrays must be aligned one-dimensional vectors")
    if np.any((probabilities < 0) | (probabilities > 1)) or np.any(
        (outcomes != 0) & (outcomes != 1)
    ):
        raise ValueError("calibration values are outside their supports")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability clip must lie in (0, 0.5)")
    if probabilities.size < 10 or np.unique(outcomes).size < 2:
        return CalibrationRegression(
            None,
            None,
            None,
            None,
            int(probabilities.size),
            probability_clip,
            GateStatus.UNDERPOWERED,
        )
    clipped = np.clip(probabilities, probability_clip, 1.0 - probability_clip)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(probabilities.size), logit))
    beta = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(100):
        fitted = expit(design @ beta)
        weights = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        score = design.T @ (outcomes - fitted)
        information = design.T @ (weights[:, None] * design)
        step = np.linalg.pinv(information) @ score
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    fitted = expit(design @ beta)
    weights = np.clip(fitted * (1.0 - fitted), 1e-12, None)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return CalibrationRegression(
        float(beta[0]),
        float(beta[1]),
        float(standard_errors[0]),
        float(standard_errors[1]),
        int(probabilities.size),
        probability_clip,
        GateStatus.PASS,
    )


def comparator_metrics(
    rows: tuple[ComparatorResolution, ...],
    *,
    skill_baseline: ComparatorKind = ComparatorKind.TOUR_EVENT_BASELINE,
    scope_tour: Tour | None = None,
    scope_event: str | None = None,
    bootstrap_replicates: int = 2_000,
    seed: int = 20260830,
) -> tuple[ComparatorMetricRow, ...]:
    if scope_tour is not None:
        rows = tuple(item for item in rows if item.forecast.tour is scope_tour)
    if scope_event is not None:
        rows = tuple(item for item in rows if item.forecast.event_id == scope_event)
    identities = tuple((item.forecast.comparator, item.forecast.target_id) for item in rows)
    if len(identities) != len(set(identities)):
        raise ValueError("comparator rows must be unique by comparator and target")
    outcomes_by_target: dict[str, int] = {}
    for item in rows:
        prior = outcomes_by_target.setdefault(item.forecast.target_id, item.outcome)
        if prior != item.outcome:
            raise ValueError("comparators disagree on the target outcome")
    scope: Literal["OVERALL", "TOUR", "EVENT"] = (
        "EVENT" if scope_event is not None else "TOUR" if scope_tour is not None else "OVERALL"
    )
    grouped: dict[ComparatorKind, list[ComparatorResolution]] = {}
    for row in rows:
        grouped.setdefault(row.forecast.comparator, []).append(row)
    briers = {
        kind: float(np.mean([(item.forecast.probability - item.outcome) ** 2 for item in members]))
        for kind, members in grouped.items()
    }
    baseline_by_target = {item.forecast.target_id: item for item in grouped.get(skill_baseline, [])}
    root = np.random.SeedSequence(seed)
    result = []
    for kind, child in zip(ComparatorKind, root.spawn(len(ComparatorKind)), strict=True):
        members = grouped.get(kind, [])
        probabilities = np.asarray(
            [item.forecast.probability for item in members], dtype=np.float64
        )
        outcomes = np.asarray([item.outcome for item in members], dtype=np.float64)
        brier = briers.get(kind)
        losses = [
            -(
                item.outcome * log(max(item.forecast.probability, 1e-15))
                + (1 - item.outcome) * log(max(1 - item.forecast.probability, 1e-15))
            )
            for item in members
        ]
        paired = [
            (item, baseline_by_target[item.forecast.target_id])
            for item in members
            if item.forecast.target_id in baseline_by_target
        ]
        paired_model_brier = (
            None
            if not paired
            else float(
                np.mean([(item.forecast.probability - item.outcome) ** 2 for item, _ in paired])
            )
        )
        paired_baseline_brier = (
            None
            if not paired
            else float(
                np.mean(
                    [
                        (baseline.forecast.probability - baseline.outcome) ** 2
                        for _, baseline in paired
                    ]
                )
            )
        )
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        brier_interval, calibration_bootstrap = _comparator_event_bootstrap(
            members,
            replicates=bootstrap_replicates,
            seed=child_seed,
        )
        result.append(
            ComparatorMetricRow(
                comparator=kind,
                scope=scope,
                tour=scope_tour,
                event_id=scope_event,
                observations=len(members),
                brier=brier,
                brier_interval=brier_interval,
                brier_skill=(
                    None
                    if paired_model_brier is None or paired_baseline_brier is None
                    else brier_skill(paired_model_brier, paired_baseline_brier)
                ),
                brier_skill_baseline=(skill_baseline if paired else None),
                brier_skill_observations=len(paired),
                log_loss=(None if not losses else float(np.mean(losses))),
                calibration=calibration_regression(probabilities, outcomes),
                calibration_bootstrap=calibration_bootstrap,
            )
        )
    return tuple(result)


def _interval_from_estimates(
    estimates: list[float], *, replicates: int, seed: int
) -> BootstrapInterval:
    if not estimates:
        return BootstrapInterval(None, None, 0, seed)
    lower, upper = np.quantile(np.asarray(estimates), (0.025, 0.975), method="linear")
    return BootstrapInterval(float(lower), float(upper), replicates, seed)


def _comparator_event_bootstrap(
    members: list[ComparatorResolution],
    *,
    replicates: int,
    seed: int,
) -> tuple[BootstrapInterval, CalibrationBootstrap]:
    by_event: dict[str, list[ComparatorResolution]] = {}
    for item in members:
        by_event.setdefault(item.forecast.event_id, []).append(item)
    events = sorted(by_event)
    empty = BootstrapInterval(None, None, 0, seed)
    if len(events) < 2 or replicates <= 0:
        return empty, CalibrationBootstrap(empty, empty)
    rng = np.random.default_rng(seed)
    briers: list[float] = []
    intercepts: list[float] = []
    slopes: list[float] = []
    for _ in range(replicates):
        selected = rng.choice(events, len(events), replace=True)
        sample = [item for event in selected for item in by_event[str(event)]]
        probabilities = np.asarray([item.forecast.probability for item in sample], dtype=np.float64)
        outcomes = np.asarray([item.outcome for item in sample], dtype=np.float64)
        briers.append(float(np.mean((probabilities - outcomes) ** 2)))
        regression = calibration_regression(probabilities, outcomes)
        if regression.intercept is not None and regression.slope is not None:
            intercepts.append(regression.intercept)
            slopes.append(regression.slope)
    return (
        _interval_from_estimates(briers, replicates=replicates, seed=seed),
        CalibrationBootstrap(
            _interval_from_estimates(intercepts, replicates=replicates, seed=seed),
            _interval_from_estimates(slopes, replicates=replicates, seed=seed),
        ),
    )


def _event_bootstrap_interval(
    event_values: dict[str, list[float]],
    *,
    replicates: int,
    seed: int,
) -> BootstrapInterval:
    events = sorted(event_values)
    if len(events) < 2 or replicates <= 0:
        return BootstrapInterval(None, None, 0, seed)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        selected = rng.choice(events, len(events), replace=True)
        values = [value for event in selected for value in event_values[str(event)]]
        estimates.append(float(np.mean(values)))
    lower, upper = np.quantile(np.asarray(estimates), (0.025, 0.975), method="linear")
    return BootstrapInterval(float(lower), float(upper), replicates, seed)


def reliability_with_event_bootstrap(
    rows: tuple[ComparatorResolution, ...],
    *,
    replicates: int,
    seed: int,
) -> tuple[ReliabilityDiagnosticBin, ...]:
    bins: list[list[ComparatorResolution]] = [[] for _ in range(10)]
    for row in rows:
        bins[min(int(row.forecast.probability * 10), 9)].append(row)
    result = []
    root = np.random.SeedSequence(seed)
    for index, (members, child) in enumerate(zip(bins, root.spawn(10), strict=True)):
        event_values: dict[str, list[float]] = {}
        event_briers: dict[str, list[float]] = {}
        for item in members:
            event_values.setdefault(item.forecast.event_id, []).append(float(item.outcome))
            event_briers.setdefault(item.forecast.event_id, []).append(
                (item.forecast.probability - item.outcome) ** 2
            )
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        brier_seed = child_seed ^ 0x9E3779B97F4A7C15
        result.append(
            ReliabilityDiagnosticBin(
                lower=index / 10,
                upper=(index + 1) / 10,
                upper_inclusive=index == 9,
                observations=len(members),
                mean_probability=(
                    None
                    if not members
                    else float(np.mean([item.forecast.probability for item in members]))
                ),
                observed_rate=(
                    None if not members else float(np.mean([item.outcome for item in members]))
                ),
                observed_rate_interval=_event_bootstrap_interval(
                    event_values, replicates=replicates, seed=child_seed
                ),
                mean_brier=(
                    None
                    if not members
                    else float(
                        np.mean(
                            [(item.forecast.probability - item.outcome) ** 2 for item in members]
                        )
                    )
                ),
                mean_brier_interval=_event_bootstrap_interval(
                    event_briers,
                    replicates=replicates,
                    seed=brier_seed,
                ),
            )
        )
    return tuple(result)


def hold_break_deciles(
    observations: tuple[OpportunityObservation, ...],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[OpportunityDecile, ...]:
    reports = []
    root = np.random.SeedSequence(seed)
    groups = [
        (
            kind,
            tour,
            tuple(item for item in observations if item.kind == kind and item.tour is tour),
        )
        for kind in ("HOLD", "BREAK")
        for tour in Tour
    ]
    for (kind, tour, members), group_seed in zip(groups, root.spawn(len(groups)), strict=True):
        if not members:
            continue
        probabilities = np.asarray([item.predicted_probability for item in members])
        boundaries = np.quantile(probabilities, np.linspace(0, 1, 11), method="linear")
        child_seeds = group_seed.spawn(10)
        for decile in range(10):
            lower, upper = float(boundaries[decile]), float(boundaries[decile + 1])
            selected = tuple(
                item
                for item in members
                if item.predicted_probability >= lower
                and (
                    item.predicted_probability <= upper
                    if decile == 9
                    else item.predicted_probability < upper
                )
            )
            if not selected:
                continue
            by_event: dict[str, list[float]] = {}
            for item in selected:
                by_event.setdefault(item.event_id, []).append(float(item.observed))
            child_seed = int(child_seeds[decile].generate_state(1, dtype=np.uint64)[0])
            reports.append(
                OpportunityDecile(
                    kind=kind,  # type: ignore[arg-type]
                    tour=tour,
                    decile=decile + 1,
                    lower=lower,
                    upper=upper,
                    opportunities=len(selected),
                    predicted_rate=float(
                        np.mean([item.predicted_probability for item in selected])
                    ),
                    observed_rate=float(np.mean([item.observed for item in selected])),
                    observed_rate_interval=_event_bootstrap_interval(
                        by_event, replicates=bootstrap_replicates, seed=child_seed
                    ),
                )
            )
    return tuple(reports)


def randomized_discrete_pit(
    observations: tuple[DiscretePredictiveObservation, ...],
    *,
    seed: int,
) -> DiscretePITReport:
    rng = np.random.default_rng(seed)
    values = []
    for item in sorted(observations, key=lambda row: row.observation_id):
        lower = sum(probability for value, probability in item.mass if value < item.observed_value)
        at = next(probability for value, probability in item.mass if value == item.observed_value)
        values.append(lower + float(rng.random()) * at)
    if not values:
        return DiscretePITReport(seed, 0, (), None, None)
    test = kstest(np.asarray(values), "uniform")
    return DiscretePITReport(
        seed,
        len(values),
        tuple(values),
        float(test.statistic),
        float(test.pvalue),
    )


def primitive_distribution_diagnostics(
    observations: tuple[PrimitivePredictiveObservation, ...],
    *,
    seed: int,
) -> tuple[PrimitiveDistributionDiagnostic, ...]:
    root = np.random.SeedSequence(seed)
    groups = [
        (
            tour,
            component,
            tuple(
                item for item in observations if item.tour is tour and item.component is component
            ),
        )
        for tour in Tour
        for component in ServeComponent
    ]
    reports = []
    for (tour, component, members), child in zip(groups, root.spawn(len(groups)), strict=True):
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(child_seed)
        residuals = []
        coverage = []
        densities = []
        for item in sorted(members, key=lambda row: row.observation_id):
            lower = sum(item.predictive_mass[: item.observed])
            at = item.predictive_mass[item.observed]
            probability = np.clip(
                lower + float(rng.random()) * at,
                np.nextafter(0.0, 1.0),
                np.nextafter(1.0, 0.0),
            )
            residuals.append(float(ndtri(probability)))
            coverage.append(item.interval_lower <= item.observed <= item.interval_upper)
            densities.append(log(max(at, 1e-300)))
        reports.append(
            PrimitiveDistributionDiagnostic(
                tour=tour,
                component=component,
                observations=len(members),
                mean_randomized_quantile_residual=(
                    None if not residuals else float(np.mean(residuals))
                ),
                interval_coverage=(None if not coverage else float(np.mean(coverage))),
                mean_log_predictive_density=(None if not densities else float(np.mean(densities))),
                seed=child_seed,
            )
        )
    return tuple(reports)


def exact_score_coherence(batch: SimulationBatch) -> ExactScoreCoherenceReport:
    completed = tuple(path for path in batch.paths if path.completed)
    players = (batch.context.player_a_id, batch.context.player_b_id)
    required = batch.context.best_of // 2 + 1
    exact_counts = {player: 0 for player in players}
    score_counts: dict[tuple[str, int, int], int] = {}
    legal = True
    deciding = 0
    for path in completed:
        wins = (
            sum(result.winner_index == 0 for result in path.sets),
            sum(result.winner_index == 1 for result in path.sets),
        )
        winner_sets = max(wins)
        loser_sets = min(wins)
        if winner_sets != required or loser_sets >= required:
            legal = False
        if path.winner_id is None:
            legal = False
        else:
            winner_index = players.index(path.winner_id)
            if wins[winner_index] != required or wins[1 - winner_index] != loser_sets:
                legal = False
            exact_counts[path.winner_id] += 1
            score_key = (path.winner_id, winner_sets, loser_sets)
            score_counts[score_key] = score_counts.get(score_key, 0) + 1
        deciding += int(path.sets_started == batch.context.best_of)
    denominator = len(completed)
    win = tuple(
        (player, 0.0 if not denominator else exact_counts[player] / denominator)
        for player in players
    )
    distribution = tuple(
        (
            winner,
            winner_sets,
            loser_sets,
            0.0 if not denominator else count / denominator,
        )
        for (winner, winner_sets, loser_sets), count in sorted(score_counts.items())
    )
    exact = tuple(
        (
            player,
            sum(probability for winner, _, _, probability in distribution if winner == player),
        )
        for player in players
    )
    deciding_from_scores = (
        None
        if not denominator
        else sum(
            probability
            for _, _, loser_sets, probability in distribution
            if loser_sets == required - 1
        )
    )
    deciding_probability = None if not denominator else deciding / denominator
    coherent = bool(
        legal
        and all(
            np.isclose(left[1], right[1], rtol=0.0, atol=0.0)
            for left, right in zip(win, exact, strict=True)
        )
        and (
            deciding_probability is None
            or (
                deciding_from_scores is not None
                and np.isclose(
                    deciding_probability,
                    deciding_from_scores,
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    )
    return ExactScoreCoherenceReport(
        best_of=batch.context.best_of,
        completed_paths=denominator,
        match_win_probability=win,
        exact_score_win_probability=exact,
        exact_score_distribution=distribution,
        deciding_set_probability=deciding_probability,
        deciding_set_probability_from_scores=deciding_from_scores,
        legal_support=legal,
        coherent=coherent,
    )


def _ledger_reliability(
    rows: list[CalibrationLedgerRow],
    *,
    replicates: int,
    seed: int,
) -> tuple[ReliabilityDiagnosticBin, ...]:
    bins: list[list[CalibrationLedgerRow]] = [[] for _ in range(10)]
    for row in rows:
        if row.outcome_binary is None or row.probability_raw is None:
            continue
        bins[min(int(row.probability_raw * 10), 9)].append(row)
    root = np.random.SeedSequence(seed)
    result = []
    for index, (members, child) in enumerate(zip(bins, root.spawn(10), strict=True)):
        rates: dict[str, list[float]] = {}
        briers: dict[str, list[float]] = {}
        for item in members:
            assert item.outcome_binary is not None
            assert item.brier_raw_model is not None
            rates.setdefault(item.event, []).append(float(item.outcome_binary))
            briers.setdefault(item.event, []).append(item.brier_raw_model)
        rate_seed, brier_seed = (
            int(item.generate_state(1, dtype=np.uint64)[0]) for item in child.spawn(2)
        )
        result.append(
            ReliabilityDiagnosticBin(
                lower=index / 10,
                upper=(index + 1) / 10,
                upper_inclusive=index == 9,
                observations=len(members),
                mean_probability=(
                    None
                    if not members
                    else float(
                        np.mean(
                            [
                                item.probability_raw
                                for item in members
                                if item.probability_raw is not None
                            ]
                        )
                    )
                ),
                observed_rate=(
                    None
                    if not members
                    else float(
                        np.mean(
                            [
                                item.outcome_binary
                                for item in members
                                if item.outcome_binary is not None
                            ]
                        )
                    )
                ),
                observed_rate_interval=_event_bootstrap_interval(
                    rates, replicates=replicates, seed=rate_seed
                ),
                mean_brier=(
                    None
                    if not members
                    else float(
                        np.mean(
                            [
                                item.brier_raw_model
                                for item in members
                                if item.brier_raw_model is not None
                            ]
                        )
                    )
                ),
                mean_brier_interval=_event_bootstrap_interval(
                    briers, replicates=replicates, seed=brier_seed
                ),
            )
        )
    return tuple(result)


def prop_family_report(
    rows: tuple[CalibrationLedgerRow, ...],
    *,
    scope_tour: Tour | None = None,
    scope_event: str | None = None,
    bootstrap_replicates: int = 2_000,
    seed: int = 20260830,
) -> tuple[PropFamilyValidation, ...]:
    if scope_tour is not None:
        rows = tuple(item for item in rows if item.tour is scope_tour)
    if scope_event is not None:
        rows = tuple(item for item in rows if item.event == scope_event)
    scope: Literal["OVERALL", "TOUR", "EVENT"] = (
        "EVENT" if scope_event is not None else "TOUR" if scope_tour is not None else "OVERALL"
    )
    grouped: dict[str, list[CalibrationLedgerRow]] = {}
    for row in rows:
        grouped.setdefault(row.prop_family, []).append(row)
    result = []
    root = np.random.SeedSequence(seed)
    grouped_items = sorted(grouped.items())
    for (family, members), child in zip(grouped_items, root.spawn(len(grouped_items)), strict=True):
        settled = [item for item in members if item.outcome_binary is not None]
        probabilities = np.asarray([item.probability_raw for item in settled], dtype=np.float64)
        outcomes = np.asarray([item.outcome_binary for item in settled], dtype=np.float64)
        result.append(
            PropFamilyValidation(
                family=family,
                scope=scope,
                tour=scope_tour,
                event_id=scope_event,
                targets=len(members),
                settled=len(settled),
                void=sum(item.resolution_status == "void" for item in members),
                unavailable=sum(item.resolution_status == "unavailable" for item in members),
                policy_disabled=sum(
                    item.support_status is PropSupportStatus.POLICY_DISABLED for item in members
                ),
                raw_brier=(
                    None
                    if not settled
                    else float(np.mean([item.brier_raw_model for item in settled]))
                ),
                submitted_brier=(
                    None
                    if not any(item.brier_submitted is not None for item in settled)
                    else float(
                        np.mean(
                            [
                                item.brier_submitted
                                for item in settled
                                if item.brier_submitted is not None
                            ]
                        )
                    )
                ),
                quantization_loss=(
                    None
                    if not any(item.quantization_loss is not None for item in settled)
                    else float(
                        np.mean(
                            [
                                item.quantization_loss
                                for item in settled
                                if item.quantization_loss is not None
                            ]
                        )
                    )
                ),
                mean_probability=(None if not settled else float(np.mean(probabilities))),
                observed_frequency=(None if not settled else float(np.mean(outcomes))),
                calibration=calibration_regression(probabilities, outcomes),
                reliability=_ledger_reliability(
                    members,
                    replicates=bootstrap_replicates,
                    seed=int(child.generate_state(1, dtype=np.uint64)[0]),
                ),
            )
        )
    return tuple(result)


def build_validation_report(
    *,
    hard_correctness_gates: tuple[CorrectnessGate, ...],
    comparator_resolutions: tuple[ComparatorResolution, ...],
    ledger_rows: tuple[CalibrationLedgerRow, ...],
    opportunity_observations: tuple[OpportunityObservation, ...] = (),
    game_observations: tuple[DiscretePredictiveObservation, ...] = (),
    primitive_observations: tuple[PrimitivePredictiveObservation, ...] = (),
    simulation_batches: tuple[SimulationBatch, ...] = (),
    additional_statistical_diagnostics: tuple[StatisticalDiagnostic, ...] = (),
    randomization_seed: int = 20260830,
    bootstrap_replicates: int = 2_000,
    genuine_historical_validation: bool = False,
) -> ValidationReport:
    """Assemble correctness gates separately from empirical diagnostics."""

    comparator_rows = comparator_metrics(
        comparator_resolutions,
        bootstrap_replicates=bootstrap_replicates,
        seed=randomization_seed,
    )
    comparator_rows_by_tour = tuple(
        row
        for index, tour in enumerate(Tour)
        for row in comparator_metrics(
            comparator_resolutions,
            scope_tour=tour,
            bootstrap_replicates=bootstrap_replicates,
            seed=randomization_seed + index + 1,
        )
        if row.observations
    )
    event_keys = sorted(
        {(item.forecast.tour, item.forecast.event_id) for item in comparator_resolutions},
        key=lambda item: (item[0].value, item[1]),
    )
    comparator_rows_by_event = tuple(
        row
        for index, (tour, event_id) in enumerate(event_keys)
        for row in comparator_metrics(
            comparator_resolutions,
            scope_tour=tour,
            scope_event=event_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=randomization_seed + len(Tour) + index + 1,
        )
        if row.observations
    )
    deciles = hold_break_deciles(
        opportunity_observations,
        bootstrap_replicates=bootstrap_replicates,
        seed=randomization_seed,
    )
    pit = (
        None
        if not game_observations
        else randomized_discrete_pit(game_observations, seed=randomization_seed)
    )
    primitives = primitive_distribution_diagnostics(primitive_observations, seed=randomization_seed)
    coherence = tuple(exact_score_coherence(batch) for batch in simulation_batches)
    prop_families = prop_family_report(
        ledger_rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=randomization_seed,
    )
    prop_families_by_tour = tuple(
        row
        for index, tour in enumerate(Tour)
        for row in prop_family_report(
            ledger_rows,
            scope_tour=tour,
            bootstrap_replicates=bootstrap_replicates,
            seed=randomization_seed + 10_000 + index,
        )
    )
    ledger_event_keys = sorted(
        {(item.tour, item.event) for item in ledger_rows},
        key=lambda item: (item[0].value, item[1]),
    )
    prop_families_by_event = tuple(
        row
        for index, (tour, event_id) in enumerate(ledger_event_keys)
        for row in prop_family_report(
            ledger_rows,
            scope_tour=tour,
            scope_event=event_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=randomization_seed + 20_000 + index,
        )
    )
    coherence_gates = tuple(
        CorrectnessGate(
            name=f"EXACT_SCORE_COHERENCE_BO{item.best_of}_{index}",
            status=GateStatus.PASS if item.coherent else GateStatus.FAIL,
            detail=(f"{item.completed_paths} completed joint paths; legal={item.legal_support}"),
        )
        for index, item in enumerate(coherence)
    )
    diagnostics = (
        StatisticalDiagnostic(
            "BASELINE_COMPARATORS",
            GateStatus.PASS if comparator_resolutions else GateStatus.UNAVAILABLE,
            "evaluation-only comparator forecasts"
            if comparator_resolutions
            else "no genuine historical comparator forecasts supplied",
        ),
        StatisticalDiagnostic(
            "HOLD_BREAK_DECILES",
            GateStatus.PASS if deciles else GateStatus.UNAVAILABLE,
            f"{len(deciles)} tour/kind decile rows",
        ),
        StatisticalDiagnostic(
            "DISCRETE_GAME_PIT",
            GateStatus.PASS if pit is not None else GateStatus.UNAVAILABLE,
            "fixed-seed randomized discrete PIT"
            if pit is not None
            else "no historical game distributions supplied",
        ),
        StatisticalDiagnostic(
            "PRIMITIVE_DISTRIBUTIONS",
            GateStatus.PASS if primitive_observations else GateStatus.UNAVAILABLE,
            "F/A/Q1/D/Q2 predictive residual and coverage scaffolding",
        ),
        *additional_statistical_diagnostics,
    )
    return ValidationReport(
        schema_version="validation-report/v1",
        hard_correctness_gates=hard_correctness_gates + coherence_gates,
        statistical_diagnostics=diagnostics,
        comparator_rows=comparator_rows,
        comparator_rows_by_tour=comparator_rows_by_tour,
        comparator_rows_by_event=comparator_rows_by_event,
        prop_families=prop_families,
        prop_families_by_tour=prop_families_by_tour,
        prop_families_by_event=prop_families_by_event,
        hold_break_deciles=deciles,
        game_distribution_pit=pit,
        primitive_diagnostics=primitives,
        exact_score_coherence=coherence,
        genuine_historical_validation=genuine_historical_validation,
    )


__all__ = [
    "BootstrapInterval",
    "CalibrationBootstrap",
    "CalibrationRegression",
    "ComparatorForecast",
    "ComparatorKind",
    "ComparatorMetricRow",
    "ComparatorResolution",
    "CorrectnessGate",
    "DiscretePITReport",
    "DiscretePredictiveObservation",
    "ExactScoreCoherenceReport",
    "GateStatus",
    "OpportunityDecile",
    "OpportunityObservation",
    "PrimitiveDistributionDiagnostic",
    "PrimitivePredictiveObservation",
    "PropFamilyValidation",
    "ReliabilityDiagnosticBin",
    "StatisticalDiagnostic",
    "ValidationReport",
    "brier_skill",
    "build_validation_report",
    "calibration_regression",
    "comparator_metrics",
    "exact_score_coherence",
    "hold_break_deciles",
    "primitive_distribution_diagnostics",
    "prop_family_report",
    "randomized_discrete_pit",
    "reliability_with_event_bootstrap",
]
