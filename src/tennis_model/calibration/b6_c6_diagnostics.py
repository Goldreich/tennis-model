"""Predeclared B6/C6 rolling-origin diagnostic machinery.

These functions summarize already locked, subsequently revealed observations.
Synthetic unit fixtures validate the machinery only; they are not historical
evidence and none of these diagnostics tunes the frozen v1.0 constants.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.special import ndtri  # type: ignore[import-untyped]

from tennis_model.calibration.outcomes import HistoricalLockSettlement
from tennis_model.estimation.inactivity import InactivityBand
from tennis_model.estimation.retirement import RetirementHistoryBand
from tennis_model.locking.models import PredictionSnapshot
from tennis_model.schemas import FrozenModel, Tour

DIAGNOSTIC_VERSION = "b6-c6-historical-diagnostics/v1"
INCIDENCE_BOOTSTRAP_LEVEL = 0.99
RETIREMENT_TIMING_MINIMUM = 50
RETIREMENT_TIMING_FLAG_P = 0.01
RETIREMENT_SUBGROUP_MINIMUM_EVENTS = 20


class DiagnosticStatus(StrEnum):
    NOT_FLAGGED = "NOT_FLAGGED"
    FLAGGED = "FLAGGED"
    UNDERPOWERED = "UNDERPOWERED"
    UNAVAILABLE = "UNAVAILABLE"


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _derived_seed(root_seed: int, *parts: object) -> int:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("diagnostic seeds must be nonnegative integers")
    payload = "\0".join((str(root_seed), DIAGNOSTIC_VERSION, *(str(item) for item in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


class DiscretePredictiveMass(FrozenModel):
    """Finite discrete posterior-predictive mass without an assumed family."""

    support: tuple[int, ...]
    probabilities: tuple[float, ...]

    @field_validator("probabilities")
    @classmethod
    def probabilities_are_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(_finite(item, field="predictive probability") for item in values)

    @model_validator(mode="after")
    def mass_is_coherent(self) -> Self:
        if not self.support or len(self.support) != len(self.probabilities):
            raise ValueError("predictive support and probabilities must be nonempty and aligned")
        if any(isinstance(item, bool) for item in self.support):
            raise ValueError("predictive support must contain integers, not booleans")
        if any(left >= right for left, right in zip(self.support, self.support[1:], strict=False)):
            raise ValueError("predictive support must be strictly increasing")
        if any(item < 0.0 for item in self.probabilities):
            raise ValueError("predictive probabilities must be nonnegative")
        if not math.isclose(math.fsum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("predictive probabilities must sum to one")
        return self

    def index(self, observed: int) -> int:
        try:
            return self.support.index(observed)
        except ValueError as exc:
            raise ValueError("observed value lies outside predictive support") from exc


def randomized_discrete_pit(
    distribution: DiscretePredictiveMass,
    observed: int,
    randomizer: float,
) -> float:
    """Return ``F(y-) + v P(y)`` for an explicit ``v`` in ``[0,1]``."""

    value = _finite(randomizer, field="PIT randomizer")
    if not 0.0 <= value <= 1.0:
        raise ValueError("PIT randomizer must lie in [0, 1]")
    index = distribution.index(observed)
    pit = math.fsum(distribution.probabilities[:index]) + value * distribution.probabilities[index]
    return min(1.0, max(0.0, pit))


class RetirementIncidenceObservation(FrozenModel):
    observation_id: str
    match_id: str
    event_id: str
    player_id: str
    tour: Tour
    best_of: Literal[3, 5]
    history_band: RetirementHistoryBand
    predicted_retirement_probability: Annotated[float, Field(ge=0, le=1)]
    observed_retirement: Literal[0, 1]
    prediction_cutoff_utc: datetime
    match_start_utc: datetime
    outcome_available_at_utc: datetime

    @field_validator("prediction_cutoff_utc", "match_start_utc", "outcome_available_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: object) -> datetime:
        return _utc(value, field=str(getattr(info, "field_name", "timestamp")))

    @model_validator(mode="after")
    def observation_is_rolling_origin(self) -> Self:
        if not all(
            item.strip()
            for item in (self.observation_id, self.match_id, self.event_id, self.player_id)
        ):
            raise ValueError("retirement incidence identities must not be empty")
        if not self.prediction_cutoff_utc < self.match_start_utc < self.outcome_available_at_utc:
            raise ValueError("retirement incidence must be forecast before start and result reveal")
        return self


class EventBlockInterval(FrozenModel):
    level: float = Field(default=INCIDENCE_BOOTSTRAP_LEVEL)
    lower: float
    upper: float
    replicates: Annotated[int, Field(gt=0)]
    seed: Annotated[int, Field(ge=0)]

    @field_validator("level")
    @classmethod
    def level_is_frozen(cls, value: float) -> float:
        if value != INCIDENCE_BOOTSTRAP_LEVEL:
            raise ValueError("retirement incidence bootstrap level must equal 0.99")
        return value


class RetirementIncidenceGroup(FrozenModel):
    group_kind: Literal["tour", "format", "history_band"]
    tour: Tour
    group: str
    player_starts: Annotated[int, Field(ge=0)]
    observed_retirements: Annotated[int, Field(ge=0)]
    expected_retirements: Annotated[float, Field(ge=0)]
    observed_rate: Annotated[float | None, Field(ge=0, le=1)]
    predicted_rate: Annotated[float | None, Field(ge=0, le=1)]
    rate_difference: float | None
    z_score: float | None
    event_block_interval: EventBlockInterval | None
    status: DiagnosticStatus


class RetirementIncidenceReport(FrozenModel):
    schema_version: Literal["retirement-incidence-diagnostics/v1"] = (
        "retirement-incidence-diagnostics/v1"
    )
    by_tour: tuple[RetirementIncidenceGroup, ...]
    by_format: tuple[RetirementIncidenceGroup, ...]
    by_history_band: tuple[RetirementIncidenceGroup, ...]


def _event_block_interval(
    rows: tuple[RetirementIncidenceObservation, ...],
    *,
    seed: int,
    replicates: int,
) -> EventBlockInterval:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    blocks: dict[str, list[RetirementIncidenceObservation]] = defaultdict(list)
    for row in rows:
        blocks[row.event_id].append(row)
    ordered = tuple(tuple(blocks[key]) for key in sorted(blocks))
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(ordered), size=len(ordered))
        residual = 0.0
        count = 0
        for block_index in selected:
            block = ordered[int(block_index)]
            residual += math.fsum(
                row.observed_retirement - row.predicted_retirement_probability for row in block
            )
            count += len(block)
        values[index] = residual / count
    lower, upper = np.quantile(values, (0.005, 0.995), method="linear")
    return EventBlockInterval(
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
    )


def _incidence_group(
    rows: tuple[RetirementIncidenceObservation, ...],
    *,
    kind: Literal["tour", "format", "history_band"],
    tour: Tour,
    group: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> RetirementIncidenceGroup:
    observed = sum(item.observed_retirement for item in rows)
    expected = math.fsum(item.predicted_retirement_probability for item in rows)
    count = len(rows)
    predicted_rate = None if count == 0 else expected / count
    observed_rate = None if count == 0 else observed / count
    difference = (
        None if observed_rate is None or predicted_rate is None else observed_rate - predicted_rate
    )
    variance = math.fsum(
        item.predicted_retirement_probability * (1.0 - item.predicted_retirement_probability)
        for item in rows
    )
    interval = (
        None
        if not rows
        else _event_block_interval(
            rows,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
    )
    z_score = None if variance <= 0.0 else (observed - expected) / math.sqrt(variance)
    underpowered = kind != "tour" and observed < RETIREMENT_SUBGROUP_MINIMUM_EVENTS
    excludes_zero = interval is not None and (interval.lower > 0.0 or interval.upper < 0.0)
    status = (
        DiagnosticStatus.UNAVAILABLE
        if not rows or z_score is None
        else DiagnosticStatus.UNDERPOWERED
        if underpowered
        else DiagnosticStatus.FLAGGED
        if abs(z_score) > 3.0 and excludes_zero
        else DiagnosticStatus.NOT_FLAGGED
    )
    return RetirementIncidenceGroup(
        group_kind=kind,
        tour=tour,
        group=group,
        player_starts=count,
        observed_retirements=observed,
        expected_retirements=expected,
        observed_rate=observed_rate,
        predicted_rate=predicted_rate,
        rate_difference=difference,
        z_score=z_score,
        event_block_interval=interval,
        status=status,
    )


def summarize_retirement_incidence(
    observations: tuple[RetirementIncidenceObservation, ...],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> RetirementIncidenceReport:
    """Report frozen ``Z_T`` groups with a 99% event-block bootstrap interval."""

    identities = tuple(item.observation_id for item in observations)
    player_starts = tuple((item.match_id, item.player_id) for item in observations)
    if len(identities) != len(set(identities)) or len(player_starts) != len(set(player_starts)):
        raise ValueError("retirement incidence observations must be unique player-starts")
    ordered = tuple(sorted(observations, key=lambda item: item.observation_id))
    reports: dict[str, list[RetirementIncidenceGroup]] = {
        "tour": [],
        "format": [],
        "history_band": [],
    }
    for tour in Tour:
        tour_rows = tuple(item for item in ordered if item.tour is tour)
        if not tour_rows:
            continue
        reports["tour"].append(
            _incidence_group(
                tour_rows,
                kind="tour",
                tour=tour,
                group=tour.value,
                bootstrap_seed=_derived_seed(bootstrap_seed, "incidence", tour.value),
                bootstrap_replicates=bootstrap_replicates,
            )
        )
        for best_of in (3, 5):
            rows = tuple(item for item in tour_rows if item.best_of == best_of)
            if rows:
                reports["format"].append(
                    _incidence_group(
                        rows,
                        kind="format",
                        tour=tour,
                        group=f"BO{best_of}",
                        bootstrap_seed=_derived_seed(
                            bootstrap_seed, "incidence", tour.value, f"BO{best_of}"
                        ),
                        bootstrap_replicates=bootstrap_replicates,
                    )
                )
        for band in RetirementHistoryBand:
            rows = tuple(item for item in tour_rows if item.history_band is band)
            if rows:
                reports["history_band"].append(
                    _incidence_group(
                        rows,
                        kind="history_band",
                        tour=tour,
                        group=band.value,
                        bootstrap_seed=_derived_seed(
                            bootstrap_seed, "incidence", tour.value, band.value
                        ),
                        bootstrap_replicates=bootstrap_replicates,
                    )
                )
    return RetirementIncidenceReport(
        by_tour=tuple(reports["tour"]),
        by_format=tuple(reports["format"]),
        by_history_band=tuple(reports["history_band"]),
    )


class RetirementTimingObservation(FrozenModel):
    observation_id: str
    match_id: str
    event_id: str
    tour: Tour
    observed_completed_game: Annotated[int | None, Field(ge=0)]
    missing_reason: str | None = None
    predictive_mass: DiscretePredictiveMass

    @model_validator(mode="after")
    def timing_or_reason_is_present(self) -> Self:
        if not all(item.strip() for item in (self.observation_id, self.match_id, self.event_id)):
            raise ValueError("retirement timing identities must not be empty")
        if (self.observed_completed_game is None) != (self.missing_reason is not None):
            raise ValueError("missing timing requires exactly one explicit reason")
        if self.observed_completed_game is not None:
            self.predictive_mass.index(self.observed_completed_game)
        return self


class RetirementTimingGroup(FrozenModel):
    tour: Tour
    reliable_timings: Annotated[int, Field(ge=0)]
    missing_timings: Annotated[int, Field(ge=0)]
    missing_reasons: tuple[tuple[str, int], ...]
    randomized_pit: tuple[float, ...]
    ks_statistic: Annotated[float | None, Field(ge=0, le=1)]
    bootstrap_p_value: Annotated[float | None, Field(ge=0, le=1)]
    bootstrap_replicates: Annotated[int, Field(ge=0)]
    status: DiagnosticStatus


class RetirementTimingReport(FrozenModel):
    schema_version: Literal["retirement-timing-diagnostics/v1"] = "retirement-timing-diagnostics/v1"
    by_tour: tuple[RetirementTimingGroup, ...]


def _ks_uniform(values: np.ndarray) -> float:
    ordered = np.sort(values)
    count = len(ordered)
    ranks = np.arange(1, count + 1, dtype=np.float64)
    return float(max(np.max(ranks / count - ordered), np.max(ordered - (ranks - 1.0) / count)))


def _sample_discrete(distribution: DiscretePredictiveMass, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(np.asarray(distribution.probabilities, dtype=np.float64))
    index = min(
        int(np.searchsorted(cumulative, float(rng.random()), side="right")),
        len(cumulative) - 1,
    )
    return distribution.support[index]


def summarize_retirement_timing(
    observations: tuple[RetirementTimingObservation, ...],
    *,
    randomization_seed: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> RetirementTimingReport:
    identities = tuple(item.observation_id for item in observations)
    if len(identities) != len(set(identities)):
        raise ValueError("retirement timing observation IDs must be unique")
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    groups: list[RetirementTimingGroup] = []
    for tour in Tour:
        rows = tuple(
            sorted(
                (item for item in observations if item.tour is tour),
                key=lambda item: item.observation_id,
            )
        )
        if not rows:
            continue
        reliable = tuple(item for item in rows if item.observed_completed_game is not None)
        missing = tuple(item for item in rows if item.observed_completed_game is None)
        reasons = tuple(sorted(Counter(cast(str, item.missing_reason) for item in missing).items()))
        if len(reliable) < RETIREMENT_TIMING_MINIMUM:
            groups.append(
                RetirementTimingGroup(
                    tour=tour,
                    reliable_timings=len(reliable),
                    missing_timings=len(missing),
                    missing_reasons=reasons,
                    randomized_pit=(),
                    ks_statistic=None,
                    bootstrap_p_value=None,
                    bootstrap_replicates=0,
                    status=DiagnosticStatus.UNAVAILABLE,
                )
            )
            continue
        pit_rng = np.random.default_rng(_derived_seed(randomization_seed, "timing", tour.value))
        pits = np.asarray(
            [
                randomized_discrete_pit(
                    item.predictive_mass,
                    cast(int, item.observed_completed_game),
                    float(pit_rng.random()),
                )
                for item in reliable
            ],
            dtype=np.float64,
        )
        statistic = _ks_uniform(pits)
        bootstrap_rng = np.random.default_rng(_derived_seed(bootstrap_seed, "timing", tour.value))
        exceedances = 0
        for _ in range(bootstrap_replicates):
            simulated = np.asarray(
                [
                    randomized_discrete_pit(
                        item.predictive_mass,
                        _sample_discrete(item.predictive_mass, bootstrap_rng),
                        float(bootstrap_rng.random()),
                    )
                    for item in reliable
                ],
                dtype=np.float64,
            )
            exceedances += _ks_uniform(simulated) >= statistic
        p_value = (1 + exceedances) / (bootstrap_replicates + 1)
        groups.append(
            RetirementTimingGroup(
                tour=tour,
                reliable_timings=len(reliable),
                missing_timings=len(missing),
                missing_reasons=reasons,
                randomized_pit=tuple(float(item) for item in pits),
                ks_statistic=statistic,
                bootstrap_p_value=p_value,
                bootstrap_replicates=bootstrap_replicates,
                status=(
                    DiagnosticStatus.FLAGGED
                    if p_value < RETIREMENT_TIMING_FLAG_P
                    else DiagnosticStatus.NOT_FLAGGED
                ),
            )
        )
    return RetirementTimingReport(by_tour=tuple(groups))


class SettlementFrequencyObservation(FrozenModel):
    match_id: str
    prop_id: str
    prop_family: str
    tour: Tour
    predicted_settled_probability: Annotated[float, Field(ge=0, le=1)]
    predicted_void_probability: Annotated[float, Field(ge=0, le=1)]
    predicted_unresolved_probability: Annotated[float, Field(ge=0, le=1)]
    observed_state: Literal["settled", "void", "unavailable", "unresolved"]
    missing_reason: str | None = None

    @model_validator(mode="after")
    def probabilities_and_missingness_are_coherent(self) -> Self:
        if not math.isclose(
            self.predicted_settled_probability
            + self.predicted_void_probability
            + self.predicted_unresolved_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("settlement probabilities must sum to one")
        missing = self.observed_state in {"unavailable", "unresolved"}
        if missing != (self.missing_reason is not None):
            raise ValueError("unavailable settlement states require an explicit reason")
        return self


class SettlementFrequencyGroup(FrozenModel):
    tour: Tour
    prop_family: str
    rows: Annotated[int, Field(ge=0)]
    observed_eligible: Annotated[int, Field(ge=0)]
    unavailable_or_unresolved: Annotated[int, Field(ge=0)]
    predicted_settled_rate: Annotated[float | None, Field(ge=0, le=1)]
    observed_settled_rate: Annotated[float | None, Field(ge=0, le=1)]
    settled_rate_difference: float | None
    predicted_void_rate: Annotated[float | None, Field(ge=0, le=1)]
    observed_void_rate: Annotated[float | None, Field(ge=0, le=1)]
    void_rate_difference: float | None
    predicted_unresolved_rate: Annotated[float | None, Field(ge=0, le=1)]


def _prop_family(kind: str) -> str:
    if kind in {"MATCH_WIN", "EXACT_SCORE", "STRAIGHT_SETS"}:
        return "MATCH_SET"
    if "ACE" in kind:
        return "ACES"
    if kind in {"PLAYER_DF", "TOTAL_DF", "DF_COMPARE"}:
        return "DOUBLE_FAULTS"
    if "BREAK" in kind:
        return "BREAKS"
    if "GAME" in kind:
        return "GAMES"
    if "TIEBREAK" in kind or kind == "DECIDING_SET":
        return "TIEBREAK"
    return kind


def settlement_frequency_observations(
    lock: PredictionSnapshot,
    settlement: HistoricalLockSettlement,
) -> tuple[SettlementFrequencyObservation, ...]:
    """Convert stored path partitions and post-lock resolutions without imputation."""

    if settlement.lock_id != lock.lock_id:
        raise ValueError("settlement differs from prediction lock")
    resolutions = {item.prop_id: item for item in settlement.resolutions}
    rows = []
    for forecast in lock.prop_estimates:
        resolution = resolutions[forecast.prop_id]
        if forecast.total_paths <= 0:
            raise ValueError("settlement-frequency forecast has no simulated paths")
        state: Literal["settled", "void", "unavailable", "unresolved"]
        if resolution.state in {"yes", "no"}:
            state = "settled"
        elif resolution.state == "void":
            state = "void"
        elif resolution.state == "unavailable":
            state = "unavailable"
        else:
            state = "unresolved"
        rows.append(
            SettlementFrequencyObservation(
                match_id=settlement.outcome_match_id,
                prop_id=forecast.prop_id,
                prop_family=_prop_family(forecast.prop.kind),
                tour=lock.context.tour,
                predicted_settled_probability=forecast.settled_paths / forecast.total_paths,
                predicted_void_probability=forecast.void_paths / forecast.total_paths,
                predicted_unresolved_probability=forecast.unresolved_paths / forecast.total_paths,
                observed_state=state,
                missing_reason=(
                    resolution.unavailable_reason
                    if state == "unavailable"
                    else forecast.policy_issue
                    if state == "unresolved"
                    else None
                ),
            )
        )
    return tuple(rows)


def summarize_settlement_frequency(
    observations: tuple[SettlementFrequencyObservation, ...],
) -> tuple[SettlementFrequencyGroup, ...]:
    grouped: dict[tuple[Tour, str], list[SettlementFrequencyObservation]] = defaultdict(list)
    for row in observations:
        grouped[(row.tour, row.prop_family)].append(row)
    result = []
    for (tour, family), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        eligible = [item for item in rows if item.observed_state in {"settled", "void"}]
        predicted_settled = float(np.mean([item.predicted_settled_probability for item in rows]))
        predicted_void = float(np.mean([item.predicted_void_probability for item in rows]))
        predicted_unresolved = float(
            np.mean([item.predicted_unresolved_probability for item in rows])
        )
        observed_settled = (
            None
            if not eligible
            else sum(item.observed_state == "settled" for item in eligible) / len(eligible)
        )
        observed_void = None if not eligible else 1.0 - cast(float, observed_settled)
        result.append(
            SettlementFrequencyGroup(
                tour=tour,
                prop_family=family,
                rows=len(rows),
                observed_eligible=len(eligible),
                unavailable_or_unresolved=len(rows) - len(eligible),
                predicted_settled_rate=predicted_settled,
                observed_settled_rate=observed_settled,
                settled_rate_difference=(
                    None if observed_settled is None else observed_settled - predicted_settled
                ),
                predicted_void_rate=predicted_void,
                observed_void_rate=observed_void,
                void_rate_difference=(
                    None if observed_void is None else observed_void - predicted_void
                ),
                predicted_unresolved_rate=predicted_unresolved,
            )
        )
    return tuple(result)


class ServeCalibrationMetric(StrEnum):
    F = "F"
    A = "A"
    Q1 = "Q1"
    D = "D"
    Q2 = "Q2"
    W1 = "W1"
    W2 = "W2"
    P_SRV = "P_SRV"

    @property
    def family(self) -> Literal["primitive", "derived"]:
        return "primitive" if self in {self.F, self.A, self.Q1, self.D, self.Q2} else "derived"


class PredictiveCountInterval(FrozenModel):
    lower: Annotated[int, Field(ge=0)]
    upper: Annotated[int, Field(ge=0)]
    nominal_coverage: Annotated[float, Field(gt=0, lt=1)]
    policy_version: str

    @model_validator(mode="after")
    def interval_is_ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("predictive count interval bounds are reversed")
        if not self.policy_version.strip():
            raise ValueError("predictive interval policy must be versioned")
        return self


class InactivityServeObservation(FrozenModel):
    observation_id: str
    event_id: str
    player_id: str
    tour: Tour
    inactivity_band: InactivityBand
    metric: ServeCalibrationMetric
    observed_count: Annotated[int, Field(ge=0)]
    trials: Annotated[int, Field(gt=0)]
    predictive_mass: DiscretePredictiveMass
    interval: PredictiveCountInterval
    prediction_cutoff_utc: datetime
    match_start_utc: datetime
    outcome_available_at_utc: datetime

    @field_validator("prediction_cutoff_utc", "match_start_utc", "outcome_available_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: object) -> datetime:
        return _utc(value, field=str(getattr(info, "field_name", "timestamp")))

    @model_validator(mode="after")
    def observation_is_valid(self) -> Self:
        if not all(item.strip() for item in (self.observation_id, self.event_id, self.player_id)):
            raise ValueError("serve diagnostic identities must not be empty")
        if self.observed_count > self.trials or self.interval.upper > self.trials:
            raise ValueError("serve diagnostic count exceeds its denominator")
        if self.predictive_mass.support != tuple(range(self.trials + 1)):
            raise ValueError("count predictive support must be exactly 0..trials")
        if not self.prediction_cutoff_utc < self.match_start_utc < self.outcome_available_at_utc:
            raise ValueError("serve calibration must be forecast before start and result reveal")
        return self


class InactivityServeExclusion(FrozenModel):
    observation_id: str
    tour: Tour
    inactivity_band: InactivityBand | None
    metric: ServeCalibrationMetric | None
    reason: str


class InactivityServeGroup(FrozenModel):
    tour: Tour
    inactivity_band: InactivityBand
    metric: ServeCalibrationMetric
    family: Literal["primitive", "derived"]
    rows: Annotated[int, Field(gt=0)]
    total_trials: Annotated[int, Field(gt=0)]
    observed_count: Annotated[int, Field(ge=0)]
    expected_count: Annotated[float, Field(ge=0)]
    observed_rate: Annotated[float, Field(ge=0, le=1)]
    predicted_rate: Annotated[float, Field(ge=0, le=1)]
    count_difference: float
    randomized_quantile_residual_mean: float
    randomized_quantile_residual_variance: Annotated[float, Field(ge=0)]
    mean_log_predictive_density: float | None
    zero_predictive_mass_observations: Annotated[int, Field(ge=0)]
    interval_covered: Annotated[int, Field(ge=0)]
    interval_coverage: Annotated[float, Field(ge=0, le=1)]
    nominal_interval_coverage: Annotated[float, Field(gt=0, lt=1)]
    interval_policy_version: str


class InactivityServeReport(FrozenModel):
    schema_version: Literal["inactivity-serve-diagnostics/v1"] = "inactivity-serve-diagnostics/v1"
    primitive: tuple[InactivityServeGroup, ...]
    derived: tuple[InactivityServeGroup, ...]
    cold_start_rows: Annotated[int, Field(ge=0)]
    exclusions_by_reason: tuple[tuple[str, int], ...]


_KNOWN_INACTIVITY_BANDS = {
    InactivityBand.ACTIVE_DAYS_0_90,
    InactivityBand.DAYS_91_180,
    InactivityBand.DAYS_181_365,
    InactivityBand.DAYS_OVER_365,
}


def summarize_inactivity_serve(
    observations: tuple[InactivityServeObservation, ...],
    *,
    exclusions: tuple[InactivityServeExclusion, ...] = (),
    randomization_seed: int,
) -> InactivityServeReport:
    identities = tuple(item.observation_id for item in observations)
    if len(identities) != len(set(identities)):
        raise ValueError("serve diagnostic observation IDs must be unique")
    exclusion_ids = tuple(item.observation_id for item in exclusions)
    if len(exclusion_ids) != len(set(exclusion_ids)) or set(identities).intersection(exclusion_ids):
        raise ValueError("serve diagnostic exclusions must be unique and disjoint")
    known = tuple(item for item in observations if item.inactivity_band in _KNOWN_INACTIVITY_BANDS)
    grouped: dict[
        tuple[Tour, InactivityBand, ServeCalibrationMetric], list[InactivityServeObservation]
    ] = defaultdict(list)
    for row in known:
        grouped[(row.tour, row.inactivity_band, row.metric)].append(row)
    summaries: list[InactivityServeGroup] = []
    for (tour, band, metric), unordered in sorted(
        grouped.items(), key=lambda item: (item[0][0].value, item[0][1].value, item[0][2].value)
    ):
        rows = tuple(sorted(unordered, key=lambda item: item.observation_id))
        policies = {(item.interval.nominal_coverage, item.interval.policy_version) for item in rows}
        if len(policies) != 1:
            raise ValueError("one serve diagnostic group cannot mix interval policies")
        nominal, policy = next(iter(policies))
        rng = np.random.default_rng(
            _derived_seed(randomization_seed, "serve", tour.value, band.value, metric.value)
        )
        pits = [
            randomized_discrete_pit(
                item.predictive_mass,
                item.observed_count,
                float(rng.random()),
            )
            for item in rows
        ]
        interior = np.clip(
            np.asarray(pits, dtype=np.float64),
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        residuals = np.asarray(ndtri(interior), dtype=np.float64)
        expected = math.fsum(
            math.fsum(
                value * probability
                for value, probability in zip(
                    item.predictive_mass.support, item.predictive_mass.probabilities, strict=True
                )
            )
            for item in rows
        )
        observed = sum(item.observed_count for item in rows)
        trials = sum(item.trials for item in rows)
        observed_masses = [
            item.predictive_mass.probabilities[item.predictive_mass.index(item.observed_count)]
            for item in rows
        ]
        zero_masses = sum(item == 0.0 for item in observed_masses)
        log_density = (
            None
            if zero_masses
            else math.fsum(math.log(item) for item in observed_masses) / len(rows)
        )
        covered = sum(
            item.interval.lower <= item.observed_count <= item.interval.upper for item in rows
        )
        summaries.append(
            InactivityServeGroup(
                tour=tour,
                inactivity_band=band,
                metric=metric,
                family=metric.family,
                rows=len(rows),
                total_trials=trials,
                observed_count=observed,
                expected_count=expected,
                observed_rate=observed / trials,
                predicted_rate=expected / trials,
                count_difference=observed - expected,
                randomized_quantile_residual_mean=float(np.mean(residuals)),
                randomized_quantile_residual_variance=float(np.var(residuals, ddof=0)),
                mean_log_predictive_density=log_density,
                zero_predictive_mass_observations=zero_masses,
                interval_covered=covered,
                interval_coverage=covered / len(rows),
                nominal_interval_coverage=nominal,
                interval_policy_version=policy,
            )
        )
    return InactivityServeReport(
        primitive=tuple(item for item in summaries if item.family == "primitive"),
        derived=tuple(item for item in summaries if item.family == "derived"),
        cold_start_rows=sum(
            item.inactivity_band is InactivityBand.COLD_START for item in observations
        ),
        exclusions_by_reason=tuple(sorted(Counter(item.reason for item in exclusions).items())),
    )


__all__ = [
    "DIAGNOSTIC_VERSION",
    "DiagnosticStatus",
    "DiscretePredictiveMass",
    "EventBlockInterval",
    "InactivityServeExclusion",
    "InactivityServeGroup",
    "InactivityServeObservation",
    "InactivityServeReport",
    "PredictiveCountInterval",
    "RetirementIncidenceGroup",
    "RetirementIncidenceObservation",
    "RetirementIncidenceReport",
    "RetirementTimingGroup",
    "RetirementTimingObservation",
    "RetirementTimingReport",
    "ServeCalibrationMetric",
    "SettlementFrequencyGroup",
    "SettlementFrequencyObservation",
    "randomized_discrete_pit",
    "settlement_frequency_observations",
    "summarize_inactivity_serve",
    "summarize_retirement_incidence",
    "summarize_retirement_timing",
    "summarize_settlement_frequency",
]
