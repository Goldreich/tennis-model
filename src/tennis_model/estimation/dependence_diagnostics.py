"""Strict out-of-fold residual-dependence diagnostics for possible v1.1 research.

Nothing in this module is imported by the live match parameter or simulation path.
Frozen v1.0 therefore remains conditionally independent regardless of its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from math import isfinite
from typing import Literal

import numpy as np
from scipy.special import ndtr, ndtri  # type: ignore[import-untyped]
from scipy.stats import beta, betabinom, rankdata  # type: ignore[import-untyped]

from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.schemas import Tour

DIAGNOSTIC_LABEL: Literal["CANDIDATE_V1_1_ONLY"] = "CANDIDATE_V1_1_ONLY"


@dataclass(frozen=True, slots=True)
class OOFComponentPrediction:
    row_id: str
    event_id: str
    player_id: str
    tour: Tour
    chronological_fold: int
    component: ServeComponent
    successes: int
    trials: int
    beta_alpha: float
    beta_beta: float
    prediction_cutoff_utc: datetime
    match_start_utc: datetime
    surface: str = "hard"
    context_partition: str = "default"

    def __post_init__(self) -> None:
        if not self.row_id or not self.event_id or not self.player_id:
            raise ValueError("OOF residual records require row, event, and player identities")
        if self.chronological_fold not in {1, 2, 3, 4}:
            raise ValueError("chronological_fold must be one of 1, 2, 3, 4")
        if self.trials <= 0 or not 0 <= self.successes <= self.trials:
            raise ValueError("OOF component counts require 0 <= successes <= positive trials")
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("OOF beta predictive shapes must be positive")
        for field, value in (
            ("prediction_cutoff_utc", self.prediction_cutoff_utc),
            ("match_start_utc", self.match_start_utc),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.prediction_cutoff_utc.astimezone(UTC) >= self.match_start_utc.astimezone(UTC):
            raise ValueError("OOF prediction cutoff must strictly precede the match")
        if not self.surface.strip() or not self.context_partition.strip():
            raise ValueError("OOF surface and context partition must be nonempty")


@dataclass(frozen=True, slots=True)
class ResidualRecord:
    row_id: str
    event_id: str
    player_id: str
    tour: Tour
    chronological_fold: int
    component: ServeComponent
    randomized_quantile_residual: float
    surface: str
    context_partition: str


@dataclass(frozen=True, slots=True)
class CorrelationInterval:
    lower: float | None
    upper: float | None
    replicates: int

    @property
    def excludes_zero(self) -> bool:
        return (
            self.lower is not None and self.upper is not None and (self.lower > 0 or self.upper < 0)
        )


@dataclass(frozen=True, slots=True)
class ComponentPairDiagnostic:
    left: ServeComponent
    right: ServeComponent
    observations: int
    pearson: float | None
    rank_correlation: float | None
    event_block_interval: CorrelationInterval
    player_cluster_interval: CorrelationInterval
    fold_pearson: tuple[float | None, float | None, float | None, float | None]
    candidate_gate_passed: bool


@dataclass(frozen=True, slots=True)
class ResidualDependenceReport:
    label: Literal["CANDIDATE_V1_1_ONLY"]
    tour: Tour
    randomization_seed: int
    bootstrap_seed: int
    residuals: tuple[ResidualRecord, ...]
    pairs: tuple[ComponentPairDiagnostic, ...]
    candidate_gate_passed: bool
    live_v1_mode: Literal["independent"] = "independent"


@dataclass(frozen=True, slots=True)
class DependenceStabilityView:
    partition_type: Literal["surface", "context", "fold", "tour"]
    partition: str
    observations: int
    status: Literal["DESCRIPTIVE", "UNDERPOWERED"]
    pair_correlations: tuple[PairRandomizationStability, ...]


@dataclass(frozen=True, slots=True)
class PairRandomizationStability:
    left: ServeComponent
    right: ServeComponent
    randomizations_available: int
    central_correlation: float | None
    minimum_correlation: float | None
    maximum_correlation: float | None


@dataclass(frozen=True, slots=True)
class DependenceRandomizationSensitivity:
    label: Literal["CANDIDATE_V1_1_ONLY"]
    tour: Tour
    randomization_seeds: tuple[int, ...]
    reports: tuple[ResidualDependenceReport, ...]
    gate_stable: bool
    stability_views: tuple[DependenceStabilityView, ...]
    live_v1_mode: Literal["independent"] = "independent"


@dataclass(frozen=True, slots=True)
class NestedCandidateFold:
    label: Literal["CANDIDATE_V1_1_ONLY"]
    evaluation_fold: int
    training_rows: int
    evaluation_rows: int
    training_end_utc: datetime | None
    evaluation_start_utc: datetime
    loadings: tuple[tuple[ServeComponent, float], ...] | None
    status: Literal["FROZEN_FROM_EARLIER_DATA", "UNDERPOWERED"]


def _randomized_residual(record: OOFComponentPrediction, uniform: float) -> float:
    lower = float(
        betabinom.cdf(
            record.successes - 1,
            record.trials,
            record.beta_alpha,
            record.beta_beta,
        )
    )
    upper = float(
        betabinom.cdf(
            record.successes,
            record.trials,
            record.beta_alpha,
            record.beta_beta,
        )
    )
    probability = lower + uniform * (upper - lower)
    probability = min(
        float(np.nextafter(1.0, 0.0)),
        max(float(np.nextafter(0.0, 1.0)), probability),
    )
    return float(ndtri(probability))


def randomized_quantile_residuals(
    predictions: tuple[OOFComponentPrediction, ...],
    *,
    seed: int,
) -> tuple[ResidualRecord, ...]:
    identities = tuple((item.row_id, item.component) for item in predictions)
    if len(identities) != len(set(identities)):
        raise ValueError("OOF predictions must contain one record per row/component")
    ordered = tuple(sorted(predictions, key=lambda item: (item.row_id, item.component.value)))
    rng = np.random.default_rng(seed)
    return tuple(
        ResidualRecord(
            row_id=item.row_id,
            event_id=item.event_id,
            player_id=item.player_id,
            tour=item.tour,
            chronological_fold=item.chronological_fold,
            component=item.component,
            randomized_quantile_residual=_randomized_residual(item, float(rng.random())),
            surface=item.surface,
            context_partition=item.context_partition,
        )
        for item in ordered
    )


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if rank:
        left = np.asarray(rankdata(left), dtype=np.float64)
        right = np.asarray(rankdata(right), dtype=np.float64)
    value = float(np.corrcoef(left, right)[0, 1])
    return value if isfinite(value) else None


def _paired(
    residuals: tuple[ResidualRecord, ...],
    left: ServeComponent,
    right: ServeComponent,
) -> list[tuple[str, str, int, float, float]]:
    values: dict[str, dict[ServeComponent, ResidualRecord]] = {}
    for item in residuals:
        values.setdefault(item.row_id, {})[item.component] = item
    result = []
    for _row_id, components in values.items():
        if left not in components or right not in components:
            continue
        a, b = components[left], components[right]
        if (
            a.event_id,
            a.player_id,
            a.chronological_fold,
            a.tour,
            a.surface,
            a.context_partition,
        ) != (
            b.event_id,
            b.player_id,
            b.chronological_fold,
            b.tour,
            b.surface,
            b.context_partition,
        ):
            raise ValueError("paired OOF component metadata disagree")
        result.append(
            (
                a.event_id,
                a.player_id,
                a.chronological_fold,
                a.randomized_quantile_residual,
                b.randomized_quantile_residual,
            )
        )
    return result


def _block_interval(
    rows: list[tuple[str, str, int, float, float]],
    *,
    block_index: int,
    replicates: int,
    rng: np.random.Generator,
) -> CorrelationInterval:
    blocks: dict[str, list[tuple[str, str, int, float, float]]] = {}
    for row in rows:
        blocks.setdefault(str(row[block_index]), []).append(row)
    labels = sorted(blocks)
    if len(labels) < 2 or replicates <= 0:
        return CorrelationInterval(None, None, 0)
    estimates = []
    for _ in range(replicates):
        selected = rng.choice(labels, size=len(labels), replace=True)
        sample = [row for label in selected for row in blocks[str(label)]]
        value = _correlation(
            np.asarray([row[3] for row in sample]),
            np.asarray([row[4] for row in sample]),
        )
        if value is not None:
            estimates.append(value)
    if not estimates:
        return CorrelationInterval(None, None, 0)
    lower, upper = np.quantile(np.asarray(estimates), (0.025, 0.975), method="linear")
    return CorrelationInterval(float(lower), float(upper), len(estimates))


def run_residual_dependence_diagnostic(
    predictions: tuple[OOFComponentPrediction, ...],
    *,
    tour: Tour,
    randomization_seed: int,
    bootstrap_seed: int,
    bootstrap_replicates: int = 2_000,
) -> ResidualDependenceReport:
    selected = tuple(item for item in predictions if item.tour is tour)
    if not selected:
        raise ValueError(f"no OOF predictions are available for {tour.value}")
    # Require genuinely chronological folds instead of accepting arbitrary labels.
    bounds = {
        fold: (
            min(item.match_start_utc for item in selected if item.chronological_fold == fold),
            max(item.match_start_utc for item in selected if item.chronological_fold == fold),
        )
        for fold in range(1, 5)
        if any(item.chronological_fold == fold for item in selected)
    }
    if set(bounds) != {1, 2, 3, 4}:
        raise ValueError("dependence diagnostic requires all four chronological folds")
    if any(bounds[fold][1] >= bounds[fold + 1][0] for fold in range(1, 4)):
        raise ValueError("chronological folds overlap or are out of order")

    residuals = randomized_quantile_residuals(selected, seed=randomization_seed)
    root = np.random.SeedSequence(bootstrap_seed)
    children = iter(root.spawn(20))
    diagnostics = []
    for left, right in combinations(ServeComponent, 2):
        paired = _paired(residuals, left, right)
        left_values = np.asarray([row[3] for row in paired], dtype=np.float64)
        right_values = np.asarray([row[4] for row in paired], dtype=np.float64)
        pearson = _correlation(left_values, right_values)
        rank = _correlation(left_values, right_values, rank=True)
        event_interval = _block_interval(
            paired,
            block_index=0,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(next(children)),
        )
        player_interval = _block_interval(
            paired,
            block_index=1,
            replicates=bootstrap_replicates,
            rng=np.random.default_rng(next(children)),
        )
        fold_values = tuple(
            _correlation(
                np.asarray([row[3] for row in paired if row[2] == fold]),
                np.asarray([row[4] for row in paired if row[2] == fold]),
            )
            for fold in range(1, 5)
        )
        stable = 0
        if pearson is not None and pearson != 0:
            sign = 1 if pearson > 0 else -1
            stable = sum(value is not None and value * sign >= 0.05 for value in fold_values)
        passed = (
            pearson is not None
            and abs(pearson) >= 0.10
            and event_interval.excludes_zero
            and stable >= 3
        )
        diagnostics.append(
            ComponentPairDiagnostic(
                left=left,
                right=right,
                observations=len(paired),
                pearson=pearson,
                rank_correlation=rank,
                event_block_interval=event_interval,
                player_cluster_interval=player_interval,
                fold_pearson=fold_values,  # type: ignore[arg-type]
                candidate_gate_passed=passed,
            )
        )
    return ResidualDependenceReport(
        label=DIAGNOSTIC_LABEL,
        tour=tour,
        randomization_seed=randomization_seed,
        bootstrap_seed=bootstrap_seed,
        residuals=residuals,
        pairs=tuple(diagnostics),
        candidate_gate_passed=any(item.candidate_gate_passed for item in diagnostics),
    )


def run_dependence_randomization_sensitivity(
    predictions: tuple[OOFComponentPrediction, ...],
    *,
    tour: Tour,
    randomization_seeds: tuple[int, ...],
    bootstrap_seed: int,
    bootstrap_replicates: int = 2_000,
    minimum_partition_records: int = 100,
) -> DependenceRandomizationSensitivity:
    """Repeat randomized residual diagnostics and expose fixed stability views."""

    if len(randomization_seeds) < 2 or len(set(randomization_seeds)) != len(randomization_seeds):
        raise ValueError("dependence sensitivity requires at least two unique fixed seeds")
    root = np.random.SeedSequence(bootstrap_seed)
    reports = tuple(
        run_residual_dependence_diagnostic(
            predictions,
            tour=tour,
            randomization_seed=seed,
            bootstrap_seed=int(child.generate_state(1, dtype=np.uint64)[0]),
            bootstrap_replicates=bootstrap_replicates,
        )
        for seed, child in zip(
            randomization_seeds,
            root.spawn(len(randomization_seeds)),
            strict=True,
        )
    )
    selected = tuple(item for item in predictions if item.tour is tour)
    partitions: list[tuple[Literal["surface", "context", "fold", "tour"], str, int]] = []
    for surface in sorted({item.surface for item in selected}):
        partitions.append(("surface", surface, sum(item.surface == surface for item in selected)))
    for context in sorted({item.context_partition for item in selected}):
        partitions.append(
            (
                "context",
                context,
                sum(item.context_partition == context for item in selected),
            )
        )
    for fold in range(1, 5):
        partitions.append(
            ("fold", str(fold), sum(item.chronological_fold == fold for item in selected))
        )
    partitions.append(("tour", tour.value, len(selected)))

    def partition_residuals(
        report: ResidualDependenceReport,
        kind: Literal["surface", "context", "fold", "tour"],
        label: str,
    ) -> tuple[ResidualRecord, ...]:
        if kind == "surface":
            return tuple(item for item in report.residuals if item.surface == label)
        if kind == "context":
            return tuple(item for item in report.residuals if item.context_partition == label)
        if kind == "fold":
            return tuple(item for item in report.residuals if item.chronological_fold == int(label))
        return report.residuals

    def pair_stability(
        kind: Literal["surface", "context", "fold", "tour"], label: str
    ) -> tuple[PairRandomizationStability, ...]:
        values: dict[tuple[ServeComponent, ServeComponent], list[float]] = {
            pair: [] for pair in combinations(ServeComponent, 2)
        }
        for report in reports:
            residuals = partition_residuals(report, kind, label)
            for pair in values:
                paired = _paired(residuals, *pair)
                correlation = _correlation(
                    np.asarray([item[3] for item in paired]),
                    np.asarray([item[4] for item in paired]),
                )
                if correlation is not None:
                    values[pair].append(correlation)
        return tuple(
            PairRandomizationStability(
                left=pair[0],
                right=pair[1],
                randomizations_available=len(correlations),
                central_correlation=(None if not correlations else float(np.median(correlations))),
                minimum_correlation=(None if not correlations else min(correlations)),
                maximum_correlation=(None if not correlations else max(correlations)),
            )
            for pair, correlations in values.items()
        )

    views = tuple(
        DependenceStabilityView(
            partition_type=kind,
            partition=label,
            observations=count,
            status=("DESCRIPTIVE" if count >= minimum_partition_records else "UNDERPOWERED"),
            pair_correlations=pair_stability(kind, label),
        )
        for kind, label, count in partitions
    )
    conclusions = tuple(report.candidate_gate_passed for report in reports)
    return DependenceRandomizationSensitivity(
        label=DIAGNOSTIC_LABEL,
        tour=tour,
        randomization_seeds=randomization_seeds,
        reports=reports,
        gate_stable=len(set(conclusions)) == 1,
        stability_views=views,
    )


def fit_strictly_nested_candidate_loadings(
    predictions: tuple[OOFComponentPrediction, ...],
    *,
    tour: Tour,
    randomization_seed: int,
    minimum_complete_training_rows: int = 20,
) -> tuple[NestedCandidateFold, ...]:
    """Fit diagnostic loadings only on folds earlier than each evaluation fold."""

    selected = tuple(item for item in predictions if item.tour is tour)
    residuals = randomized_quantile_residuals(selected, seed=randomization_seed)
    residual_by_identity = {
        (item.row_id, item.component): item.randomized_quantile_residual for item in residuals
    }
    result = []
    for evaluation_fold in range(2, 5):
        training_predictions = tuple(
            item for item in selected if item.chronological_fold < evaluation_fold
        )
        evaluation_predictions = tuple(
            item for item in selected if item.chronological_fold == evaluation_fold
        )
        if not evaluation_predictions:
            raise ValueError(f"evaluation fold {evaluation_fold} has no records")
        training_end = max(
            (item.match_start_utc for item in training_predictions),
            default=None,
        )
        evaluation_start = min(item.match_start_utc for item in evaluation_predictions)
        if training_end is not None and training_end >= evaluation_start:
            raise ValueError("candidate loading training overlaps its evaluation fold")
        row_ids = sorted({item.row_id for item in training_predictions})
        complete = [
            row_id
            for row_id in row_ids
            if all((row_id, component) in residual_by_identity for component in ServeComponent)
        ]
        loadings: tuple[tuple[ServeComponent, float], ...] | None = None
        status: Literal["FROZEN_FROM_EARLIER_DATA", "UNDERPOWERED"] = "UNDERPOWERED"
        if len(complete) >= minimum_complete_training_rows:
            matrix = np.asarray(
                [
                    [residual_by_identity[(row_id, component)] for component in ServeComponent]
                    for row_id in complete
                ],
                dtype=np.float64,
            )
            correlation = np.corrcoef(matrix, rowvar=False)
            values, vectors = np.linalg.eigh(correlation)
            principal = vectors[:, int(np.argmax(values))]
            if principal[0] < 0:
                principal = -principal
            scale = min(1.0, float(np.sqrt(max(values[-1] - 1.0, 0.0) / len(ServeComponent))))
            loadings = tuple(
                (component, float(np.clip(principal[index] * scale, -1.0, 1.0)))
                for index, component in enumerate(ServeComponent)
            )
            status = "FROZEN_FROM_EARLIER_DATA"
        result.append(
            NestedCandidateFold(
                label=DIAGNOSTIC_LABEL,
                evaluation_fold=evaluation_fold,
                training_rows=len(complete),
                evaluation_rows=len({item.row_id for item in evaluation_predictions}),
                training_end_utc=training_end,
                evaluation_start_utc=evaluation_start,
                loadings=loadings,
                status=status,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CandidateOneFactorSpec:
    label: Literal["CANDIDATE_V1_1_ONLY"]
    loadings: tuple[tuple[ServeComponent, float], ...]

    def __post_init__(self) -> None:
        if tuple(component for component, _ in self.loadings) != tuple(ServeComponent):
            raise ValueError("candidate loadings must contain F/A/Q1/D/Q2 in order")
        if any(abs(value) > 1 for _, value in self.loadings):
            raise ValueError("candidate loadings must lie in [-1, 1]")
        if self.loadings[0][1] < 0:
            raise ValueError("F loading sign is fixed nonnegative for identification")


def candidate_one_factor_spec(
    report: ResidualDependenceReport,
    loadings: tuple[tuple[ServeComponent, float], ...],
) -> CandidateOneFactorSpec:
    if not report.candidate_gate_passed:
        raise ValueError(
            "candidate one-factor evaluation is blocked until the residual gate passes"
        )
    return CandidateOneFactorSpec(label=DIAGNOSTIC_LABEL, loadings=loadings)


def sample_candidate_beta_marginals(
    shapes: tuple[tuple[ServeComponent, float, float], ...],
    spec: CandidateOneFactorSpec,
    *,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Diagnostic-only one-factor beta copula preserving every beta marginal."""

    if tuple(component for component, _, _ in shapes) != tuple(ServeComponent):
        raise ValueError("candidate beta shapes must contain F/A/Q1/D/Q2 in order")
    if n_draws <= 0 or any(alpha <= 0 or beta_shape <= 0 for _, alpha, beta_shape in shapes):
        raise ValueError("candidate draw count and beta shapes must be positive")
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n_draws)
    result = np.empty((n_draws, len(shapes)), dtype=np.float64)
    loadings = dict(spec.loadings)
    for index, (component, alpha, beta_shape) in enumerate(shapes):
        loading = loadings[component]
        latent = loading * shared + np.sqrt(1.0 - loading**2) * rng.standard_normal(n_draws)
        uniforms = np.clip(
            ndtr(latent),
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        result[:, index] = beta.ppf(uniforms, alpha, beta_shape)
    return result


@dataclass(frozen=True, slots=True)
class CandidateAdoptionEvidence:
    joint_log_density_improvement: float
    joint_log_density_interval_lower: float
    improves_core_prop_metric: bool
    core_prop_brier_changes: tuple[tuple[str, float], ...]


def candidate_adoption_conclusion(
    evidence: CandidateAdoptionEvidence,
) -> Literal["V1_1_CANDIDATE_SUPPORTED", "NO_EVIDENCE_FOR_V1_1_DEPENDENCE"]:
    supported = (
        evidence.joint_log_density_improvement > 0
        and evidence.joint_log_density_interval_lower > 0
        and evidence.improves_core_prop_metric
        and all(change <= 0.001 for _, change in evidence.core_prop_brier_changes)
    )
    return "V1_1_CANDIDATE_SUPPORTED" if supported else "NO_EVIDENCE_FOR_V1_1_DEPENDENCE"


__all__ = [
    "DIAGNOSTIC_LABEL",
    "CandidateAdoptionEvidence",
    "CandidateOneFactorSpec",
    "ComponentPairDiagnostic",
    "DependenceRandomizationSensitivity",
    "DependenceStabilityView",
    "NestedCandidateFold",
    "OOFComponentPrediction",
    "PairRandomizationStability",
    "ResidualDependenceReport",
    "candidate_adoption_conclusion",
    "candidate_one_factor_spec",
    "fit_strictly_nested_candidate_loadings",
    "randomized_quantile_residuals",
    "run_dependence_randomization_sensitivity",
    "run_residual_dependence_diagnostic",
    "sample_candidate_beta_marginals",
]
