"""Cutoff-safe match-parameter construction and two-stage uncertainty draws.

One posterior parameter vector is sampled per primitive component and reused for
both serving directions.  Conditional on those shared parameter draws, the two
directions receive independent beta match-performance draws.  Point-path
randomness remains the responsibility of :mod:`tennis_model.simulation.point`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from math import isfinite, sqrt
from numbers import Integral, Real
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator
from scipy.special import expit  # type: ignore[import-untyped]

from tennis_model.estimation.duration_model import DurationFitArtifact
from tennis_model.estimation.inactivity import (
    InactivityAdjustment,
    InactivityAdjustmentState,
    InactivityCoordinateReference,
    InactivityCoordinateRole,
    InactivityRecord,
    affected_coordinate_roles,
    apply_inactivity_adjustment,
)
from tennis_model.estimation.retirement import (
    RETIREMENT_COMPETING_RISK_VERSION,
    PlayerRetirementPosterior,
    RetirementFitArtifact,
    RetirementIntensity,
    RetirementPathDraw,
    RetirementScenario,
    RetirementScenarioMixture,
    RetirementSourceCoverage,
    draw_player_retirement_path,
    draw_retirement_scenario,
    player_retirement_posterior,
    retirement_probability_to_intensity,
)
from tennis_model.estimation.serve_components import (
    ComponentParameterProjection,
    ComponentPrediction,
    EffectBlock,
    FittedServeComponent,
    FutureMatchContext,
    ModelDataError,
    PlayerEffectCoordinateProjection,
    ServeComponent,
    ServePerformanceDistribution,
    _project_validated_component_parameters,
    effect_block_player_contribution,
    predict_serve_performance,
    project_component_linear_coordinate,
    project_player_effect_coordinate,
)
from tennis_model.estimation.snapshot import (
    ModelSnapshot,
    ModelSnapshotError,
    load_snapshot_duration_artifact,
    load_snapshot_fits,
    load_snapshot_retirement_artifact,
)
from tennis_model.schemas import FrozenModel, Tour
from tennis_model.simulation.point import ServePerformanceDraw

_COMPONENT_ORDER = tuple(ServeComponent)
MATCH_PARAMETER_IMPLEMENTATION_VERSION: Literal["match-parameters-laplace-beta/v1"] = (
    "match-parameters-laplace-beta/v1"
)
MATCH_RNG_BIT_GENERATOR: Literal["PCG64"] = "PCG64"
C6_DIRECT_EFFECT_POSTERIOR_VERSION: Literal["canonical-player-effect-laplace/v1"] = (
    "canonical-player-effect-laplace/v1"
)


class MatchParameterError(ValueError):
    """Match parameter construction or sampling violates the frozen contract."""


class _ParameterModel(FrozenModel):
    model_config = ConfigDict(allow_inf_nan=False)


type ConditionValue = str | int | float | bool | None
type SeedEntropy = int | tuple[int, ...]


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _nonempty(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _sha256(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must contain exactly 64 hexadecimal characters")
    return normalized


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


class MatchCondition(_ParameterModel):
    """Lock/reporting metadata that is not automatically a fitted predictor."""

    name: str
    value: ConditionValue

    @field_validator("name")
    @classmethod
    def name_is_present(cls, value: str) -> str:
        return _nonempty(value, field="condition name")


class MatchContext(_ParameterModel):
    """Frozen US Open matchup metadata and explicit information cutoff."""

    player_a_id: str
    player_b_id: str
    tour: Tour
    event: str
    round: str
    scheduled_start_utc: datetime
    scheduled_start_local_date: date | None = None
    best_of: Literal[3, 5]
    surface: Literal["hard"] = "hard"
    indoor: bool | None
    conditions: tuple[MatchCondition, ...] = ()
    information_cutoff_utc: datetime
    information_scenario_id: str = "central"

    @field_validator("player_a_id", "player_b_id", "event", "round", "information_scenario_id")
    @classmethod
    def text_is_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)

    @field_validator("scheduled_start_utc", "information_cutoff_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def matchup_is_coherent(self) -> Self:
        if self.player_a_id == self.player_b_id:
            raise ValueError("match players must be distinct")
        if self.scheduled_start_utc < self.information_cutoff_utc:
            raise ValueError("scheduled start cannot precede the information cutoff")
        names = tuple(condition.name for condition in self.conditions)
        if len(names) != len(set(names)):
            raise ValueError("condition metadata names must be unique")
        return self


def _fitted_direction_context(
    snapshot: ModelSnapshot,
    context: MatchContext,
    *,
    server: str,
    receiver: str,
) -> FutureMatchContext:
    return FutureMatchContext(
        serving_player_id=server,
        returning_player_id=receiver,
        tour=context.tour,
        surface="Hard",
        indoor=context.indoor,
        event=context.event,
        event_year=context.scheduled_start_utc.year,
        match_date_utc=context.scheduled_start_utc,
        information_cutoff_utc=snapshot.data_cutoff_utc,
        best_of=context.best_of,
    )


class PerformanceDependenceSpec(_ParameterModel):
    """Frozen-v1.0 dependence configuration; the future copula is inert."""

    mode: Literal["independent"] = "independent"
    loadings: tuple[tuple[ServeComponent, float], ...] = ()
    validation_artifact_id: None = None

    @field_validator("loadings", mode="before")
    @classmethod
    def empty_mapping_is_the_independent_form(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(value.items())
        return value

    @model_validator(mode="after")
    def v1_is_independent(self) -> Self:
        if self.loadings:
            raise ValueError("v1.0 independent performance draws cannot have loadings")
        return self


class RetirementMatchParameters(_ParameterModel):
    """Cutoff-safe B6 inputs needed to reproduce all path-local draws."""

    schema_version: Literal["retirement-match-parameters/v1"] = "retirement-match-parameters/v1"
    artifact_id: str
    artifact_schema_version: str
    response_coding_version: str
    intensity_transform_version: str
    source_manifest_id: str
    source_manifest_sha256: str
    source_coverage: RetirementSourceCoverage
    data_sha256: str
    config_sha256: str
    code_sha256: str
    deterministic_test_result_sha256: str
    software_version: str
    tour_retirements_y: Annotated[float, Field(ge=0)]
    tour_starts_n: Annotated[float, Field(ge=0)]
    tour_baseline_rho: Annotated[float, Field(gt=0, lt=1)]
    weighted_start_coverage_gate_passed: bool
    production_eligible: bool
    reference_games: Literal[22] = 22
    competing_risk_version: str = RETIREMENT_COMPETING_RISK_VERSION
    player_posteriors: tuple[PlayerRetirementPosterior, PlayerRetirementPosterior]
    central_intensity_summaries: tuple[
        tuple[str, RetirementIntensity], tuple[str, RetirementIntensity]
    ]
    scenario_mixtures: tuple[RetirementScenarioMixture, RetirementScenarioMixture]

    @field_validator(
        "artifact_id",
        "source_manifest_sha256",
        "data_sha256",
        "config_sha256",
        "code_sha256",
        "deterministic_test_result_sha256",
    )
    @classmethod
    def artifact_id_is_valid(cls, value: str) -> str:
        return _sha256(value, field="retirement artifact hash")

    @model_validator(mode="after")
    def players_and_artifact_are_coherent(self) -> Self:
        _nonempty(self.source_manifest_id, field="retirement source_manifest_id")
        _nonempty(self.software_version, field="retirement software_version")
        if self.competing_risk_version != RETIREMENT_COMPETING_RISK_VERSION:
            raise ValueError("unsupported B6 competing-risk version")
        posterior_players = tuple(item.player_id for item in self.player_posteriors)
        mixture_players = tuple(item.player_id for item in self.scenario_mixtures)
        if posterior_players != mixture_players or len(set(posterior_players)) != 2:
            raise ValueError("B6 posteriors and scenario mixtures must identify both players")
        if any(item.artifact_id != self.artifact_id for item in self.player_posteriors):
            raise ValueError("B6 player posterior references another artifact")
        if tuple(item[0] for item in self.central_intensity_summaries) != posterior_players:
            raise ValueError("B6 central intensity summaries differ from player ordering")
        if any(
            item[1] != retirement_probability_to_intensity(posterior.mean_rho)
            for item, posterior in zip(
                self.central_intensity_summaries, self.player_posteriors, strict=True
            )
        ):
            raise ValueError("B6 central intensity summary differs from posterior mean")
        if (
            not self.source_coverage.production_fit_inputs_eligible
            or not self.weighted_start_coverage_gate_passed
        ):
            raise ValueError("match parameters require production-eligible B6 coverage")
        if not self.production_eligible:
            raise ValueError("match parameters cannot contain a non-production B6 artifact")
        return self


class DurationMatchParameters(_ParameterModel):
    """Compact reference to the verified B5 artifact used by one matchup."""

    schema_version: Literal["duration-match-parameters/v1"] = (
        "duration-match-parameters/v1"
    )
    artifact_id: str
    artifact_schema_version: str
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime
    player_ids: tuple[str, str]

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_valid(cls, value: str) -> str:
        return _sha256(value, field="duration artifact ID")

    @field_validator("artifact_schema_version")
    @classmethod
    def artifact_schema_is_present(cls, value: str) -> str:
        return _nonempty(value, field="duration artifact schema version")

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def duration_timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def duration_reference_is_coherent(self) -> Self:
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("duration fit cannot precede its information cutoff")
        if len(set(self.player_ids)) != 2 or any(not item.strip() for item in self.player_ids):
            raise ValueError("duration match parameters require two distinct player IDs")
        return self


class InactivityMatchParameters(_ParameterModel):
    """C6 inputs and per-component congruence-transform audit records."""

    schema_version: Literal["inactivity-match-parameters/v1"] = "inactivity-match-parameters/v1"
    configuration_artifact_id: str
    adjustment_version: Literal["c6-post-90-day/v1"] = "c6-post-90-day/v1"
    posterior_coordinate_version: Literal["canonical-player-effect-laplace/v1"] = (
        C6_DIRECT_EFFECT_POSTERIOR_VERSION
    )
    records: tuple[InactivityRecord, InactivityRecord]
    component_adjustments: tuple[InactivityAdjustment, ...]

    @field_validator("configuration_artifact_id")
    @classmethod
    def configuration_id_is_valid(cls, value: str) -> str:
        return _sha256(value, field="inactivity configuration artifact ID")

    @model_validator(mode="after")
    def components_and_players_are_complete(self) -> Self:
        if len({item.player_id for item in self.records}) != 2:
            raise ValueError("C6 requires exactly two player records")
        if tuple(item.component for item in self.component_adjustments) != _COMPONENT_ORDER:
            raise ValueError("C6 adjustments must contain F/A/Q1/D/Q2 in canonical order")
        return self


@dataclass(frozen=True, slots=True)
class _C6EffectPlan:
    """Live route from one logical C6 coordinate to its fitted effect block."""

    component: ServeComponent
    player_id: str
    role: InactivityCoordinateRole
    block: EffectBlock
    projection: PlayerEffectCoordinateProjection
    map_parameters: tuple[float, ...]
    hard_multiplier: float
    covariance_scale: float

    @property
    def changes_draw(self) -> bool:
        return self.covariance_scale != 1.0 or (
            self.role.is_hard_deviation
            and self.hard_multiplier != 1.0
            and self.projection.map_contribution != 0.0
        )


@dataclass(frozen=True, slots=True)
class _C6ComponentPlan:
    """One explicitly sampled canonical direct-effect Gaussian posterior."""

    component: ServeComponent
    coordinate_ids: tuple[str, ...]
    raw_size: int
    adjusted_mean: np.ndarray[Any, np.dtype[np.float64]]
    adjusted_factor: np.ndarray[Any, np.dtype[np.float64]]
    effect_plans: tuple[_C6EffectPlan, ...]

    def __post_init__(self) -> None:
        if self.adjusted_mean.shape != (len(self.coordinate_ids),):
            raise MatchParameterError("C6 direct-effect mean has invalid dimensions")
        if self.adjusted_factor.shape[0] != len(self.coordinate_ids):
            raise MatchParameterError("C6 direct-effect factor has invalid dimensions")
        if not 0 < self.raw_size < len(self.coordinate_ids):
            raise MatchParameterError("C6 direct-effect raw-coordinate boundary is invalid")
        if len(self.effect_plans) != len(self.coordinate_ids) - self.raw_size:
            raise MatchParameterError("C6 direct-effect plans do not match logical coordinates")
        self.adjusted_mean.setflags(write=False)
        self.adjusted_factor.setflags(write=False)

    @property
    def changes_draw(self) -> bool:
        return any(plan.changes_draw for plan in self.effect_plans)


class MatchParameterProvenance(_ParameterModel):
    framework_version: Literal["v1.0"]
    implementation_version: Literal["match-parameters-laplace-beta/v1"]
    snapshot_id: str
    component_artifact_ids: tuple[tuple[ServeComponent, str], ...]
    data_cutoff_utc: datetime
    match_information_cutoff_utc: datetime
    data_hash: str
    config_hash: str
    fit_code_commit: str
    information_scenario_id: str
    dependence_mode: Literal["independent"]
    retirement_artifact_id: str | None = None
    inactivity_configuration_artifact_id: str | None = None
    duration_artifact_id: str | None = None

    @field_validator("snapshot_id", "data_hash", "config_hash")
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field=info.field_name)

    @field_validator("data_cutoff_utc", "match_information_cutoff_utc")
    @classmethod
    def cutoffs_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def identities_are_complete(self) -> Self:
        if tuple(component for component, _artifact_id in self.component_artifact_ids) != (
            _COMPONENT_ORDER
        ):
            raise ValueError("provenance must identify F/A/Q1/D/Q2 in canonical order")
        for component, artifact_id in self.component_artifact_ids:
            _sha256(artifact_id, field=f"{component.value} artifact ID")
        if self.data_cutoff_utc > self.match_information_cutoff_utc:
            raise ValueError("snapshot data cutoff cannot follow match information cutoff")
        _nonempty(self.fit_code_commit, field="fit_code_commit")
        _nonempty(self.information_scenario_id, field="information_scenario_id")
        contract_ids = (
            self.retirement_artifact_id,
            self.inactivity_configuration_artifact_id,
        )
        if (contract_ids[0] is None) != (contract_ids[1] is None):
            raise ValueError("B6/C6 provenance IDs must be present together")
        for label, value in zip(
            ("retirement_artifact_id", "inactivity_configuration_artifact_id"),
            contract_ids,
            strict=True,
        ):
            if value is not None:
                _sha256(value, field=label)
        if self.duration_artifact_id is not None:
            _sha256(self.duration_artifact_id, field="duration_artifact_id")
            if self.retirement_artifact_id is None:
                raise ValueError("duration provenance requires complete B6/C6 provenance")
        return self


@dataclass(frozen=True, slots=True)
class DirectionComponentDistribution:
    """One loaded fitted component and its MAP summary for one direction."""

    artifact_id: str
    fit: FittedServeComponent
    map_prediction: ComponentPrediction

    def __post_init__(self) -> None:
        _sha256(self.artifact_id, field="artifact_id")
        if self.fit.component is not self.map_prediction.component:
            raise MatchParameterError("direction component fit and MAP summary differ")


@dataclass(frozen=True, slots=True)
class ServingDirectionDistribution:
    """Loaded F/A/Q1/D/Q2 distributions for one explicit serving direction."""

    server_id: str
    receiver_id: str
    context: FutureMatchContext
    map_distribution: ServePerformanceDistribution
    components: tuple[DirectionComponentDistribution, ...]

    def __post_init__(self) -> None:
        if not self.server_id.strip() or not self.receiver_id.strip():
            raise MatchParameterError("serving-direction player IDs must not be empty")
        if self.server_id == self.receiver_id:
            raise MatchParameterError("serving-direction player IDs must be distinct")
        if self.context.serving_player_id != self.server_id:
            raise MatchParameterError("direction context has the wrong server")
        if self.context.returning_player_id != self.receiver_id:
            raise MatchParameterError("direction context has the wrong receiver")
        if self.map_distribution.context != self.context:
            raise MatchParameterError("direction MAP summary has a different context")
        observed = tuple(item.fit.component for item in self.components)
        if observed != _COMPONENT_ORDER:
            raise MatchParameterError("direction must contain F/A/Q1/D/Q2 in canonical order")
        if tuple(item.map_prediction.component for item in self.components) != observed:
            raise MatchParameterError("direction MAP summaries are routed incorrectly")
        summaries = (
            self.map_distribution.first_serve_in,
            self.map_distribution.ace_given_first_in,
            self.map_distribution.returnable_first_win,
            self.map_distribution.double_fault_given_second_opp,
            self.map_distribution.playable_second_win,
        )
        if tuple(item.map_prediction for item in self.components) != summaries:
            raise MatchParameterError("direction component summaries differ from their bundle")

    @property
    def by_component(self) -> Mapping[ServeComponent, DirectionComponentDistribution]:
        return MappingProxyType({item.fit.component: item for item in self.components})


class ServingDirectionRecord(_ParameterModel):
    server_id: str
    receiver_id: str
    artifact_ids: tuple[tuple[ServeComponent, str], ...]
    map_distribution: ServePerformanceDistribution

    @model_validator(mode="after")
    def components_are_complete(self) -> Self:
        if tuple(component for component, _artifact_id in self.artifact_ids) != _COMPONENT_ORDER:
            raise ValueError("direction record must identify F/A/Q1/D/Q2")
        for component, artifact_id in self.artifact_ids:
            _sha256(artifact_id, field=f"{component.value} artifact ID")
        if self.map_distribution.context.serving_player_id != self.server_id:
            raise ValueError("direction record has the wrong server context")
        if self.map_distribution.context.returning_player_id != self.receiver_id:
            raise ValueError("direction record has the wrong receiver context")
        return self


class MatchParameterRecord(_ParameterModel):
    """Stable audit record without copied posterior matrices or live RNGs."""

    schema_version: Literal[
        "match-parameter-distribution/v1",
        "match-parameter-distribution/v2",
        "match-parameter-distribution/v3",
    ] = "match-parameter-distribution/v1"
    snapshot_id: str
    snapshot: ModelSnapshot
    context: MatchContext
    player_a_serving: ServingDirectionRecord
    player_b_serving: ServingDirectionRecord
    performance_dependence: PerformanceDependenceSpec
    provenance: MatchParameterProvenance
    retirement: RetirementMatchParameters | None = None
    inactivity: InactivityMatchParameters | None = None
    duration: DurationMatchParameters | None = None

    @field_validator("snapshot_id")
    @classmethod
    def snapshot_id_is_valid(cls, value: str) -> str:
        return _sha256(value, field="snapshot_id")

    @model_validator(mode="after")
    def record_is_coherent(self) -> Self:
        if self.snapshot_id != self.snapshot.snapshot_id:
            raise ValueError("record snapshot ID does not match snapshot content")
        if self.provenance.snapshot_id != self.snapshot_id:
            raise ValueError("record provenance references another snapshot")
        if self.schema_version == "match-parameter-distribution/v1":
            if any(
                item is not None for item in (self.retirement, self.inactivity, self.duration)
            ):
                raise ValueError("v1 match parameters cannot contain B6/C6 or duration inputs")
        elif self.retirement is None or self.inactivity is None:
            raise ValueError("v2/v3 match parameters require complete B6/C6 inputs")
        elif self.schema_version == "match-parameter-distribution/v2":
            if self.duration is not None:
                raise ValueError("v2 match parameters cannot silently add duration inputs")
        elif self.duration is None:
            raise ValueError("v3 match parameters require the B5 duration artifact")
        expected_record_schema = {
            "serve-model-snapshot/v1": "match-parameter-distribution/v1",
            "serve-model-snapshot/v2": "match-parameter-distribution/v2",
            "serve-model-snapshot/v3": "match-parameter-distribution/v3",
        }[self.snapshot.schema_version]
        if self.schema_version != expected_record_schema:
            raise ValueError("match-parameter schema differs from its model snapshot schema")
        expected_artifacts = tuple(
            (reference.component, reference.artifact_id)
            for reference in self.snapshot.component_artifacts
        )
        if (
            self.player_a_serving.artifact_ids != expected_artifacts
            or self.player_b_serving.artifact_ids != expected_artifacts
            or self.provenance.component_artifact_ids != expected_artifacts
        ):
            raise ValueError("record artifact references differ from its snapshot")
        if self.snapshot.tour is not self.context.tour:
            raise ValueError("record snapshot tour differs from match context")
        if self.snapshot.data_cutoff_utc > self.context.information_cutoff_utc:
            raise ValueError("record snapshot contains data after the match cutoff")
        expected_a_context = _fitted_direction_context(
            self.snapshot,
            self.context,
            server=self.context.player_a_id,
            receiver=self.context.player_b_id,
        )
        expected_b_context = _fitted_direction_context(
            self.snapshot,
            self.context,
            server=self.context.player_b_id,
            receiver=self.context.player_a_id,
        )
        if (
            self.player_a_serving.map_distribution.context != expected_a_context
            or self.player_b_serving.map_distribution.context != expected_b_context
        ):
            raise ValueError("record serving contexts differ from snapshot and match context")
        provenance_identity = (
            self.provenance.framework_version,
            self.provenance.data_cutoff_utc,
            self.provenance.match_information_cutoff_utc,
            self.provenance.data_hash,
            self.provenance.config_hash,
            self.provenance.fit_code_commit,
            self.provenance.information_scenario_id,
            self.provenance.dependence_mode,
        )
        expected_identity = (
            self.snapshot.framework_version,
            self.snapshot.data_cutoff_utc,
            self.context.information_cutoff_utc,
            self.snapshot.data_hash,
            self.snapshot.config_hash,
            self.snapshot.code_commit,
            self.context.information_scenario_id,
            self.performance_dependence.mode,
        )
        if provenance_identity != expected_identity:
            raise ValueError("record provenance contradicts its snapshot or context")
        if (
            self.player_a_serving.server_id,
            self.player_a_serving.receiver_id,
            self.player_b_serving.server_id,
            self.player_b_serving.receiver_id,
        ) != (
            self.context.player_a_id,
            self.context.player_b_id,
            self.context.player_b_id,
            self.context.player_a_id,
        ):
            raise ValueError("serialized serving directions do not match the matchup")
        if self.retirement is not None and self.inactivity is not None:
            expected_players = (self.context.player_a_id, self.context.player_b_id)
            if tuple(item.player_id for item in self.retirement.player_posteriors) != (
                expected_players
            ):
                raise ValueError("B6 player ordering differs from match context")
            if tuple(item.player_id for item in self.inactivity.records) != expected_players:
                raise ValueError("C6 player ordering differs from match context")
            if self.context.scheduled_start_local_date is None:
                raise ValueError("B6/C6 match parameters require the official local start date")
            if any(
                item.scheduled_start_local_date != self.context.scheduled_start_local_date
                or item.information_cutoff_utc != self.context.information_cutoff_utc
                or item.tour is not self.context.tour
                for item in self.inactivity.records
            ):
                raise ValueError("C6 records differ from the match date, tour, or cutoff")
            if (
                self.provenance.retirement_artifact_id != self.retirement.artifact_id
                or self.provenance.inactivity_configuration_artifact_id
                != self.inactivity.configuration_artifact_id
            ):
                raise ValueError("B6/C6 record provenance is inconsistent")
        if self.duration is not None:
            if self.snapshot.schema_version != "serve-model-snapshot/v3":
                raise ValueError("duration match parameters require a v3 model snapshot")
            if self.snapshot.duration_artifact is None:
                raise ValueError("duration match parameters lost their snapshot reference")
            if (
                self.duration.artifact_id != self.snapshot.duration_artifact.artifact_id
                or self.duration.artifact_schema_version
                != self.snapshot.duration_schema_version
                or self.duration.tour is not self.context.tour
                or self.duration.information_cutoff_utc != self.snapshot.data_cutoff_utc
                or self.duration.player_ids
                != (self.context.player_a_id, self.context.player_b_id)
                or self.provenance.duration_artifact_id != self.duration.artifact_id
            ):
                raise ValueError("duration record provenance is inconsistent")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class MatchParameterDistribution:
    """Loaded two-direction distribution ready for repeated match-level draws."""

    snapshot: ModelSnapshot
    context: MatchContext
    player_a_serving: ServingDirectionDistribution
    player_b_serving: ServingDirectionDistribution
    performance_dependence: PerformanceDependenceSpec
    provenance: MatchParameterProvenance
    retirement: RetirementMatchParameters | None = None
    inactivity: InactivityMatchParameters | None = None
    duration: DurationFitArtifact | None = None
    c6_component_plans: tuple[_C6ComponentPlan, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.player_a_serving.server_id,
            self.player_a_serving.receiver_id,
            self.player_b_serving.server_id,
            self.player_b_serving.receiver_id,
        ) != (
            self.context.player_a_id,
            self.context.player_b_id,
            self.context.player_b_id,
            self.context.player_a_id,
        ):
            raise MatchParameterError("serving directions do not match the requested players")
        if self.provenance.snapshot_id != self.snapshot.snapshot_id:
            raise MatchParameterError("parameter provenance references another snapshot")
        if self.snapshot.tour is not self.context.tour:
            raise MatchParameterError("parameter snapshot tour differs from match context")
        if self.snapshot.data_cutoff_utc > self.context.information_cutoff_utc:
            raise MatchParameterError("parameter snapshot contains post-cutoff data")
        expected_a_context = _fitted_direction_context(
            self.snapshot,
            self.context,
            server=self.context.player_a_id,
            receiver=self.context.player_b_id,
        )
        expected_b_context = _fitted_direction_context(
            self.snapshot,
            self.context,
            server=self.context.player_b_id,
            receiver=self.context.player_a_id,
        )
        if (
            self.player_a_serving.context != expected_a_context
            or self.player_b_serving.context != expected_b_context
        ):
            raise MatchParameterError("serving contexts differ from snapshot and match context")
        expected_artifacts = tuple(
            (reference.component, reference.artifact_id)
            for reference in self.snapshot.component_artifacts
        )
        if self.provenance.component_artifact_ids != expected_artifacts:
            raise MatchParameterError("parameter provenance has different component artifacts")
        provenance_identity = (
            self.provenance.framework_version,
            self.provenance.data_cutoff_utc,
            self.provenance.match_information_cutoff_utc,
            self.provenance.data_hash,
            self.provenance.config_hash,
            self.provenance.fit_code_commit,
            self.provenance.information_scenario_id,
            self.provenance.dependence_mode,
        )
        expected_identity = (
            self.snapshot.framework_version,
            self.snapshot.data_cutoff_utc,
            self.context.information_cutoff_utc,
            self.snapshot.data_hash,
            self.snapshot.config_hash,
            self.snapshot.code_commit,
            self.context.information_scenario_id,
            self.performance_dependence.mode,
        )
        if provenance_identity != expected_identity:
            raise MatchParameterError("parameter provenance contradicts snapshot or context")
        for component in _COMPONENT_ORDER:
            left = self.player_a_serving.by_component[component]
            right = self.player_b_serving.by_component[component]
            if left.artifact_id != right.artifact_id or left.fit != right.fit:
                raise MatchParameterError(
                    "both serving directions must share each fitted component artifact"
                )
            if left.artifact_id != self.snapshot.component_artifact_ids[component]:
                raise MatchParameterError("direction artifact differs from the model snapshot")
        if (self.retirement is None) != (self.inactivity is None):
            raise MatchParameterError("B6 and C6 match parameters must be present together")
        if self.snapshot.b6_c6_complete != (self.retirement is not None):
            raise MatchParameterError("match probability contracts differ from the snapshot")
        if self.inactivity is None:
            if self.c6_component_plans:
                raise MatchParameterError("legacy match parameters cannot contain C6 plans")
        elif tuple(plan.component for plan in self.c6_component_plans) != _COMPONENT_ORDER:
            raise MatchParameterError("C6 component plans must contain F/A/Q1/D/Q2")
        if (
            self.retirement is not None
            and self.inactivity is not None
            and (
                self.provenance.retirement_artifact_id != self.retirement.artifact_id
                or self.provenance.inactivity_configuration_artifact_id
                != self.inactivity.configuration_artifact_id
            )
        ):
            raise MatchParameterError("B6/C6 provenance differs from match parameters")
        if self.snapshot.duration_complete != (self.duration is not None):
            raise MatchParameterError("duration parameters differ from the model snapshot schema")
        if self.duration is not None:
            reference = self.snapshot.duration_artifact
            if reference is None:
                raise MatchParameterError("duration-enabled snapshot lost its artifact reference")
            if (
                self.duration.artifact_id != reference.artifact_id
                or self.duration.schema_version != self.snapshot.duration_schema_version
                or self.duration.tour is not self.context.tour
                or self.duration.information_cutoff_utc != self.snapshot.data_cutoff_utc
                or self.provenance.duration_artifact_id != self.duration.artifact_id
            ):
                raise MatchParameterError("duration artifact or provenance is inconsistent")

    def to_record(self) -> MatchParameterRecord:
        def direction_record(
            direction: ServingDirectionDistribution,
        ) -> ServingDirectionRecord:
            return ServingDirectionRecord(
                server_id=direction.server_id,
                receiver_id=direction.receiver_id,
                artifact_ids=tuple(
                    (item.fit.component, item.artifact_id) for item in direction.components
                ),
                map_distribution=direction.map_distribution,
            )

        return MatchParameterRecord(
            schema_version=(
                "match-parameter-distribution/v3"
                if self.duration is not None
                else "match-parameter-distribution/v2"
                if self.retirement is not None
                else "match-parameter-distribution/v1"
            ),
            snapshot_id=self.snapshot.snapshot_id,
            snapshot=self.snapshot,
            context=self.context,
            player_a_serving=direction_record(self.player_a_serving),
            player_b_serving=direction_record(self.player_b_serving),
            performance_dependence=self.performance_dependence,
            provenance=self.provenance,
            retirement=self.retirement,
            inactivity=self.inactivity,
            duration=(
                None
                if self.duration is None
                else DurationMatchParameters(
                    artifact_id=self.duration.artifact_id,
                    artifact_schema_version=self.duration.schema_version,
                    tour=self.duration.tour,
                    information_cutoff_utc=self.duration.information_cutoff_utc,
                    fitted_at_utc=self.duration.fitted_at_utc,
                    player_ids=(self.context.player_a_id, self.context.player_b_id),
                )
            ),
        )

    def canonical_json(self) -> str:
        return self.to_record().canonical_json()


class PosteriorParameterDraw(_ParameterModel):
    """One ordered full-vector draw from a component Laplace approximation."""

    component: ServeComponent
    parameter_names: tuple[str, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def dimensions_match(self) -> Self:
        if not self.parameter_names or len(self.parameter_names) != len(self.values):
            raise ValueError("posterior parameter draw dimensions do not match")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("posterior parameter names must be unique")
        if any(not name.strip() for name in self.parameter_names):
            raise ValueError("posterior parameter names must not be blank")
        return self

    def values_array(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.asarray(self.values, dtype=np.float64).copy()


class UnseenEffectDraw(_ParameterModel):
    block_name: str
    level: str
    standard_deviation: Annotated[float, Field(gt=0)]
    value: float

    @field_validator("block_name", "level")
    @classmethod
    def identity_is_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)

    @property
    def key(self) -> tuple[str, str]:
        return self.block_name, self.level


class C6AdjustedCoordinateDraw(_ParameterModel):
    """One sampled logical player coordinate from the adjusted Gaussian."""

    component: ServeComponent
    coordinate_id: str
    player_id: str
    role: InactivityCoordinateRole
    value: float
    adjustment_state: Literal[InactivityAdjustmentState.C6_APPLIED] = (
        InactivityAdjustmentState.C6_APPLIED
    )

    @field_validator("coordinate_id", "player_id")
    @classmethod
    def identity_is_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)


class JointComponentParameterDraw(_ParameterModel):
    """One component theta shared across two directional matchup evaluations."""

    component: ServeComponent
    posterior: PosteriorParameterDraw
    unseen_effects: tuple[UnseenEffectDraw, ...]
    c6_adjusted_coordinates: tuple[C6AdjustedCoordinateDraw, ...] = ()
    player_a_serving_mean: Annotated[float, Field(gt=0, lt=1)]
    player_b_serving_mean: Annotated[float, Field(gt=0, lt=1)]
    predictive_concentration: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def component_and_effects_are_coherent(self) -> Self:
        if self.posterior.component is not self.component:
            raise ValueError("posterior draw belongs to another component")
        keys = tuple(item.key for item in self.unseen_effects)
        if len(keys) != len(set(keys)):
            raise ValueError("unseen effects must have unique keys")
        coordinate_ids = tuple(item.coordinate_id for item in self.c6_adjusted_coordinates)
        if len(coordinate_ids) != len(set(coordinate_ids)):
            raise ValueError("C6 adjusted coordinate draws must be unique")
        if any(item.component is not self.component for item in self.c6_adjusted_coordinates):
            raise ValueError("C6 adjusted coordinate draw belongs to another component")
        return self


class BetaComponentParameters(_ParameterModel):
    component: ServeComponent
    mean: Annotated[float, Field(gt=0, lt=1)]
    concentration: Annotated[float, Field(gt=0)]


class ServingDirectionParameterDraw(_ParameterModel):
    server_id: str
    receiver_id: str
    components: tuple[BetaComponentParameters, ...]

    @field_validator("server_id", "receiver_id")
    @classmethod
    def player_id_is_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)

    @model_validator(mode="after")
    def primitives_are_complete(self) -> Self:
        if tuple(item.component for item in self.components) != _COMPONENT_ORDER:
            raise ValueError("direction parameter draw must contain F/A/Q1/D/Q2")
        if self.server_id == self.receiver_id:
            raise ValueError("direction parameter players must be distinct")
        return self

    @property
    def by_component(self) -> Mapping[ServeComponent, BetaComponentParameters]:
        return MappingProxyType({item.component: item for item in self.components})


class MatchupParameterDraw(_ParameterModel):
    """Parameter-only stage: directional means before beta performance draws."""

    snapshot_id: str
    player_a_serving: ServingDirectionParameterDraw
    player_b_serving: ServingDirectionParameterDraw
    components: tuple[JointComponentParameterDraw, ...]

    @model_validator(mode="after")
    def shared_components_match_directions(self) -> Self:
        _sha256(self.snapshot_id, field="snapshot_id")
        if (
            self.player_a_serving.server_id,
            self.player_a_serving.receiver_id,
        ) != (
            self.player_b_serving.receiver_id,
            self.player_b_serving.server_id,
        ):
            raise ValueError("matchup parameter directions must be exact reversals")
        if tuple(item.component for item in self.components) != _COMPONENT_ORDER:
            raise ValueError("matchup parameter draw must contain F/A/Q1/D/Q2")
        for item in self.components:
            left = self.player_a_serving.by_component[item.component]
            right = self.player_b_serving.by_component[item.component]
            if left.mean != item.player_a_serving_mean:
                raise ValueError("player-A direction mean differs from joint component draw")
            if right.mean != item.player_b_serving_mean:
                raise ValueError("player-B direction mean differs from joint component draw")
            if (
                left.concentration != item.predictive_concentration
                or right.concentration != item.predictive_concentration
            ):
                raise ValueError("direction concentrations differ from shared parameter draw")
        return self


class SeedReference(_ParameterModel):
    """Serializable reconstruction data for one NumPy SeedSequence node."""

    entropy: SeedEntropy
    spawn_key: tuple[Annotated[int, Field(ge=0)], ...]
    pool_size: Annotated[int, Field(ge=4)]
    n_children_spawned: Annotated[int, Field(ge=0)] = 0

    @field_validator("entropy")
    @classmethod
    def entropy_is_nonnegative(cls, value: SeedEntropy) -> SeedEntropy:
        values = (value,) if isinstance(value, int) else value
        if any(item < 0 for item in values):
            raise ValueError("SeedSequence entropy must be nonnegative")
        return value

    @classmethod
    def from_seed_sequence(cls, seed: np.random.SeedSequence) -> SeedReference:
        raw_entropy = seed.entropy
        entropy: SeedEntropy
        if isinstance(raw_entropy, Integral):
            entropy = int(raw_entropy)
        elif raw_entropy is None:
            raise MatchParameterError("SeedSequence entropy must be explicit")
        else:
            values = cast(Sequence[int], raw_entropy)
            entropy = tuple(int(value) for value in values)
        return cls(
            entropy=entropy,
            spawn_key=tuple(int(value) for value in seed.spawn_key),
            pool_size=int(seed.pool_size),
            n_children_spawned=int(seed.n_children_spawned),
        )

    def to_seed_sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            self.entropy,
            spawn_key=self.spawn_key,
            pool_size=self.pool_size,
            n_children_spawned=self.n_children_spawned,
        )


class MatchSeedPlan(_ParameterModel):
    """Independent child streams for every stochastic match stage.

    Duration was appended as the sixth root child.  The original parameter,
    performance, point-path, and retirement children therefore retain their
    pre-duration spawn keys exactly.
    """

    root: SeedReference
    parameter_draws: SeedReference
    player_a_performance: SeedReference
    player_b_performance: SeedReference
    point_path: SeedReference
    retirement: SeedReference
    retirement_parameters: SeedReference
    retirement_boundaries: SeedReference
    duration: SeedReference
    duration_parameters: SeedReference
    duration_residual: SeedReference
    bit_generator: Literal["PCG64"] = MATCH_RNG_BIT_GENERATOR

    @model_validator(mode="after")
    def children_are_distinct(self) -> Self:
        children = (
            self.parameter_draws.spawn_key,
            self.player_a_performance.spawn_key,
            self.player_b_performance.spawn_key,
            self.point_path.spawn_key,
            self.retirement.spawn_key,
            self.retirement_parameters.spawn_key,
            self.retirement_boundaries.spawn_key,
            self.duration.spawn_key,
            self.duration_parameters.spawn_key,
            self.duration_residual.spawn_key,
        )
        if len(set(children)) != len(children):
            raise ValueError("match seed child streams must be distinct")
        return self


@dataclass(frozen=True, slots=True)
class MatchPerformanceDraw:
    """Both fixed serving-performance vectors for one future match path."""

    matchup_parameters: MatchupParameterDraw
    player_a_serving: ServePerformanceDraw
    player_b_serving: ServePerformanceDraw
    seed_plan: MatchSeedPlan
    retirement_draws: tuple[RetirementPathDraw, ...] = ()


def _component_prediction(
    distribution: ServePerformanceDistribution,
    component: ServeComponent,
) -> ComponentPrediction:
    return {
        ServeComponent.F: distribution.first_serve_in,
        ServeComponent.A: distribution.ace_given_first_in,
        ServeComponent.Q1: distribution.returnable_first_win,
        ServeComponent.D: distribution.double_fault_given_second_opp,
        ServeComponent.Q2: distribution.playable_second_win,
    }[component]


def _direction_distribution(
    *,
    server_id: str,
    receiver_id: str,
    context: FutureMatchContext,
    fits: Mapping[ServeComponent, FittedServeComponent],
    artifact_ids: Mapping[ServeComponent, str],
) -> ServingDirectionDistribution:
    summary = predict_serve_performance(fits, context)
    return ServingDirectionDistribution(
        server_id=server_id,
        receiver_id=receiver_id,
        context=context,
        map_distribution=summary,
        components=tuple(
            DirectionComponentDistribution(
                artifact_id=artifact_ids[component],
                fit=fits[component],
                map_prediction=_component_prediction(summary, component),
            )
            for component in _COMPONENT_ORDER
        ),
    )


def _plan_direction_coefficient(
    plan: _C6EffectPlan,
    context: FutureMatchContext,
) -> float:
    if (
        plan.role
        in {
            InactivityCoordinateRole.SERVER_GLOBAL,
            InactivityCoordinateRole.SERVER_HARD_DEVIATION,
        }
        and plan.player_id == context.serving_player_id
    ):
        return 1.0
    if (
        plan.role
        in {
            InactivityCoordinateRole.RETURNER_GLOBAL,
            InactivityCoordinateRole.RETURNER_HARD_DEVIATION,
        }
        and plan.player_id == context.returning_player_id
    ):
        return -1.0
    return 0.0


def _adjust_direction_map_distribution(
    direction: ServingDirectionDistribution,
    plans: Sequence[_C6EffectPlan],
) -> ServingDirectionDistribution:
    """Apply C6 to stored MAP summaries without recomputing it in the card."""

    changing = tuple(plan for plan in plans if plan.changes_draw)
    if not changing:
        return direction
    prediction_updates: dict[ServeComponent, ComponentPrediction] = {}
    for item in direction.components:
        component_plans = tuple(plan for plan in changing if plan.component is item.fit.component)
        if not component_plans:
            prediction_updates[item.fit.component] = item.map_prediction
            continue
        linear = project_component_linear_coordinate(item.fit, direction.context)
        eta = linear.linear_predictor_map
        gradient = np.asarray(linear.gradient, dtype=np.float64)
        unseen_scales: dict[tuple[str, str], float] = {}
        for plan in component_plans:
            coefficient = _plan_direction_coefficient(plan, direction.context)
            if coefficient == 0.0:
                continue
            center = plan.projection.map_contribution
            if plan.role.is_hard_deviation:
                eta += coefficient * (plan.hard_multiplier - 1.0) * center
            if plan.projection.seen:
                gradient += (
                    coefficient
                    * (plan.covariance_scale - 1.0)
                    * np.asarray(plan.projection.map_gradient, dtype=np.float64)
                )
            elif plan.covariance_scale != 1.0:
                unseen_scales[(plan.projection.block_name, plan.player_id)] = plan.covariance_scale
        covariance = item.fit.posterior.covariance_array()
        unseen_variance = sum(
            (requirement.standard_deviation * unseen_scales.get(requirement.key, 1.0)) ** 2
            for requirement in linear.unseen_effects
        )
        variance = float(gradient @ covariance @ gradient) + unseen_variance
        if variance < -1e-10 or not isfinite(variance):
            raise MatchParameterError("C6 adjusted linear-predictor variance is invalid")
        mean = _mean_from_linear_predictor(eta)
        concentration = item.map_prediction.predictive_concentration
        prediction_updates[item.fit.component] = item.map_prediction.model_copy(
            update={
                "map_mean": mean,
                "linear_predictor_map": eta,
                "linear_predictor_sd": max(0.0, variance) ** 0.5,
                "unseen_effect_variance": unseen_variance,
                "beta_alpha_at_map": concentration * mean,
                "beta_beta_at_map": concentration * (1.0 - mean),
            }
        )
    updated_summary = direction.map_distribution.model_copy(
        update={
            "first_serve_in": prediction_updates[ServeComponent.F],
            "ace_given_first_in": prediction_updates[ServeComponent.A],
            "returnable_first_win": prediction_updates[ServeComponent.Q1],
            "double_fault_given_second_opp": prediction_updates[ServeComponent.D],
            "playable_second_win": prediction_updates[ServeComponent.Q2],
        }
    )
    return replace(
        direction,
        map_distribution=updated_summary,
        components=tuple(
            replace(item, map_prediction=prediction_updates[item.fit.component])
            for item in direction.components
        ),
    )


_C6_ROLE_BLOCK: dict[InactivityCoordinateRole, tuple[str, str | None]] = {
    InactivityCoordinateRole.SERVER_GLOBAL: ("server_global", None),
    InactivityCoordinateRole.SERVER_HARD_DEVIATION: ("server_surface", "Hard"),
    InactivityCoordinateRole.RETURNER_GLOBAL: ("returner_global", None),
    InactivityCoordinateRole.RETURNER_HARD_DEVIATION: ("returner_surface", "Hard"),
}


def _effect_block(
    fitted: FittedServeComponent,
    role: InactivityCoordinateRole,
) -> EffectBlock:
    block_role, surface = _C6_ROLE_BLOCK[role]
    block = next(
        (
            item
            for item in fitted.effect_blocks
            if item.role == block_role and item.surface == surface
        ),
        None,
    )
    if block is None:
        raise MatchParameterError(f"{fitted.component.value} lacks frozen C6 block {role.value}")
    return block


def _build_inactivity_parameters(
    fits: Mapping[ServeComponent, FittedServeComponent],
    records: tuple[InactivityRecord, InactivityRecord],
    *,
    configuration_artifact_id: str,
) -> tuple[InactivityMatchParameters, tuple[_C6ComponentPlan, ...]]:
    """Build the five canonical direct-effect Gaussian posteriors for C6.

    The stored optimizer coordinates remain the immutable Milestone 3 artifact.
    For C6, the Laplace Gaussian is re-expressed once in a versioned canonical
    coordinate system containing the centered player effects.  The Jacobian at
    the MAP defines that Gaussian coordinate representation; the adjusted mean
    and congruence-scaled factor below are then sampled directly.
    """

    component_plans: list[_C6ComponentPlan] = []
    adjustments: list[InactivityAdjustment] = []
    for component in _COMPONENT_ORDER:
        fitted = fits[component]
        covariance = fitted.posterior.covariance_array()
        logical: list[
            tuple[
                InactivityRecord,
                InactivityCoordinateRole,
                EffectBlock,
                PlayerEffectCoordinateProjection,
            ]
        ] = []
        for record in records:
            for role in affected_coordinate_roles(component):
                block_role, surface = _C6_ROLE_BLOCK[role]
                projection = project_player_effect_coordinate(
                    fitted,
                    player_id=record.player_id,
                    role=cast(Any, block_role),
                    surface=surface,
                )
                logical.append((record, role, _effect_block(fitted, role), projection))

        raw_coordinate_ids = tuple(
            f"engineering:{component.value}:{name}" for name in fitted.posterior.parameter_names
        )
        raw_mean = np.asarray(fitted.posterior.map_parameters, dtype=np.float64)
        logical_coordinate_ids = tuple(item[3].coordinate_id for item in logical)
        logical_mean = np.asarray([item[3].map_contribution for item in logical], dtype=np.float64)
        logical_covariance = np.zeros((len(logical), len(logical)), dtype=np.float64)
        gradients = [np.asarray(item[3].map_gradient, dtype=np.float64) for item in logical]
        for left_index, left in enumerate(logical):
            left_projection = left[3]
            if not left_projection.seen:
                assert left_projection.unseen_standard_deviation is not None
                logical_covariance[left_index, left_index] = (
                    left_projection.unseen_standard_deviation**2
                )
                continue
            for right_index in range(left_index, len(logical)):
                right_projection = logical[right_index][3]
                if not right_projection.seen:
                    continue
                value = float(gradients[left_index] @ covariance @ gradients[right_index])
                logical_covariance[left_index, right_index] = value
                logical_covariance[right_index, left_index] = value
        raw_logical_covariance = covariance @ np.stack(gradients, axis=1)
        for index, item in enumerate(logical):
            if not item[3].seen:
                raw_logical_covariance[:, index] = 0.0
        augmented_mean = np.concatenate((raw_mean, logical_mean))
        augmented_covariance = np.block(
            [
                [covariance, raw_logical_covariance],
                [raw_logical_covariance.T, logical_covariance],
            ]
        )
        coordinate_ids = raw_coordinate_ids + logical_coordinate_ids
        raw_size = len(raw_coordinate_ids)
        catalog = tuple(
            InactivityCoordinateReference(
                component=component,
                player_id=item[0].player_id,
                role=item[1],
                coordinate_id=item[3].coordinate_id,
                index=raw_size + index,
            )
            for index, item in enumerate(logical)
        )
        adjusted = apply_inactivity_adjustment(
            component=component,
            coordinate_ids=coordinate_ids,
            mean=augmented_mean,
            covariance=augmented_covariance,
            kappa=fitted.kappa,
            inactivity_records=records,
            coordinate_catalog=catalog,
            adjustment_state=InactivityAdjustmentState.UNADJUSTED,
        )
        if adjusted.kappa != fitted.kappa:
            raise MatchParameterError("C6 must not change fitted beta concentration")
        adjustments.append(adjusted.adjustment)
        effect_plans = tuple(
            _C6EffectPlan(
                component=component,
                player_id=record.player_id,
                role=role,
                block=block,
                projection=projection,
                map_parameters=fitted.posterior.map_parameters,
                hard_multiplier=(
                    record.hard_deviation_multiplier if role.is_hard_deviation else 1.0
                ),
                covariance_scale=sqrt(record.variance_inflation_factor),
            )
            for record, role, block, projection in logical
        )

        raw_factor = (
            np.linalg.cholesky(covariance)
            if fitted.posterior.curvature_kind == "full"
            else np.diag(np.sqrt(np.diag(covariance)))
        )
        unseen_count = sum(not item[3].seen for item in logical)
        base_factor = np.zeros(
            (len(coordinate_ids), raw_size + unseen_count),
            dtype=np.float64,
        )
        base_factor[:raw_size, :raw_size] = raw_factor
        unseen_index = 0
        coordinate_scales = np.ones(len(coordinate_ids), dtype=np.float64)
        for index, (record, _role, _block, projection) in enumerate(logical):
            row = raw_size + index
            coordinate_scales[row] = sqrt(record.variance_inflation_factor)
            if projection.seen:
                gradient = np.asarray(projection.map_gradient, dtype=np.float64)
                base_factor[row, :raw_size] = gradient @ raw_factor
            else:
                assert projection.unseen_standard_deviation is not None
                base_factor[row, raw_size + unseen_index] = projection.unseen_standard_deviation
                unseen_index += 1
        adjusted_factor = coordinate_scales[:, np.newaxis] * base_factor
        reconstructed_covariance = adjusted_factor @ adjusted_factor.T
        if not np.allclose(
            reconstructed_covariance,
            adjusted.covariance,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise MatchParameterError(
                "C6 direct-effect sampling factor differs from adjusted covariance"
            )
        component_plans.append(
            _C6ComponentPlan(
                component=component,
                coordinate_ids=coordinate_ids,
                raw_size=raw_size,
                adjusted_mean=adjusted.mean.copy(),
                adjusted_factor=adjusted_factor,
                effect_plans=effect_plans,
            )
        )
    return (
        InactivityMatchParameters(
            configuration_artifact_id=configuration_artifact_id,
            records=records,
            component_adjustments=tuple(adjustments),
        ),
        tuple(component_plans),
    )


def _central_retirement_mixture(
    artifact: RetirementFitArtifact,
    player_id: str,
    *,
    information_cutoff_utc: datetime,
) -> RetirementScenarioMixture:
    evidence_time = min(
        artifact.information_cutoff_utc,
        information_cutoff_utc,
    ) - timedelta(microseconds=1)
    scenario = RetirementScenario(
        scenario_id="central",
        scenario_version="b6-central/v1",
        named_state="central",
        central=True,
        player_id=player_id,
        log_hazard_ratio=0.0,
        weight=1.0,
        source_id="frozen-b6-central-scenario",
        source_sha256=artifact.config_sha256,
        observation_at_utc=evidence_time,
        publication_at_utc=evidence_time,
        authoring_method="frozen B6 central scenario (eta=0)",
    )
    return RetirementScenarioMixture(
        mixture_id=f"central:{artifact.artifact_id}:{player_id}",
        player_id=player_id,
        information_cutoff_utc=information_cutoff_utc,
        scenarios=(scenario,),
    )


def _build_retirement_parameters(
    artifact: RetirementFitArtifact,
    context: MatchContext,
    supplied_mixtures: tuple[RetirementScenarioMixture, ...],
) -> RetirementMatchParameters:
    artifact.require_production_coverage()
    players = (context.player_a_id, context.player_b_id)
    by_player = {item.player_id: item for item in supplied_mixtures}
    if len(by_player) != len(supplied_mixtures) or any(
        player not in players for player in by_player
    ):
        raise MatchParameterError("B6 scenario mixtures must be unique and match the players")
    mixtures = tuple(
        by_player.get(player)
        or _central_retirement_mixture(
            artifact,
            player,
            information_cutoff_utc=context.information_cutoff_utc,
        )
        for player in players
    )
    if any(item.information_cutoff_utc != context.information_cutoff_utc for item in mixtures):
        raise MatchParameterError("B6 scenario mixture cutoff differs from the match cutoff")
    posteriors = tuple(player_retirement_posterior(artifact, player) for player in players)
    return RetirementMatchParameters(
        artifact_id=artifact.artifact_id,
        artifact_schema_version=artifact.schema_version,
        response_coding_version=artifact.response_coding_version,
        intensity_transform_version=artifact.intensity_transform_version,
        source_manifest_id=artifact.source_manifest_id,
        source_manifest_sha256=artifact.source_manifest_sha256,
        source_coverage=artifact.source_coverage,
        data_sha256=artifact.data_sha256,
        config_sha256=artifact.config_sha256,
        code_sha256=artifact.code_sha256,
        deterministic_test_result_sha256=artifact.deterministic_test_result_sha256,
        software_version=artifact.software_version,
        tour_retirements_y=artifact.tour_retirements_y,
        tour_starts_n=artifact.tour_starts_n,
        tour_baseline_rho=artifact.tour_baseline_rho,
        weighted_start_coverage_gate_passed=artifact.weighted_start_coverage_gate_passed,
        production_eligible=artifact.production_eligible,
        player_posteriors=cast(
            tuple[PlayerRetirementPosterior, PlayerRetirementPosterior], posteriors
        ),
        central_intensity_summaries=cast(
            tuple[tuple[str, RetirementIntensity], tuple[str, RetirementIntensity]],
            tuple(
                (posterior.player_id, retirement_probability_to_intensity(posterior.mean_rho))
                for posterior in posteriors
            ),
        ),
        scenario_mixtures=cast(
            tuple[RetirementScenarioMixture, RetirementScenarioMixture], mixtures
        ),
    )


def estimate_match(
    snapshot: ModelSnapshot,
    context: MatchContext,
    *,
    inactivity_records: Sequence[InactivityRecord] = (),
    retirement_scenario_mixtures: Sequence[RetirementScenarioMixture] = (),
) -> MatchParameterDistribution:
    """Load one explicit snapshot and construct both directional distributions.

    Probability-relevant training data must precede the requested lock cutoff.
    The physical fit/artifact timestamp may be later during a truthful
    retrospective reconstruction and remains explicit in provenance.
    """

    snapshot = ModelSnapshot.model_validate(snapshot.model_dump(mode="python"))
    context = MatchContext.model_validate(context.model_dump(mode="python"))
    if snapshot.tour is not context.tour:
        raise MatchParameterError("match tour does not match the model snapshot")
    if snapshot.data_cutoff_utc > context.information_cutoff_utc:
        raise MatchParameterError("model snapshot contains data after the match cutoff")
    try:
        fits = load_snapshot_fits(snapshot)
    except (ModelSnapshotError, ModelDataError) as exc:
        raise MatchParameterError(f"cannot load model snapshot: {exc}") from exc

    retirement_parameters: RetirementMatchParameters | None = None
    inactivity_parameters: InactivityMatchParameters | None = None
    duration_parameters: DurationFitArtifact | None = None
    c6_component_plans: tuple[_C6ComponentPlan, ...] = ()
    if snapshot.b6_c6_complete:
        if context.scheduled_start_local_date is None:
            raise MatchParameterError(
                "B6/C6 estimation requires the official scheduled-start local date"
            )
        if len(inactivity_records) != 2:
            raise MatchParameterError("B6/C6 estimation requires both player inactivity records")
        records_by_player = {item.player_id: item for item in inactivity_records}
        if len(records_by_player) != 2:
            raise MatchParameterError("C6 inactivity records must identify distinct players")
        try:
            ordered_records = cast(
                tuple[InactivityRecord, InactivityRecord],
                tuple(
                    records_by_player[player]
                    for player in (context.player_a_id, context.player_b_id)
                ),
            )
        except KeyError as exc:
            raise MatchParameterError("C6 inactivity records do not match the matchup") from exc
        if any(
            item.tour is not context.tour
            or item.information_cutoff_utc != context.information_cutoff_utc
            or item.scheduled_start_local_date != context.scheduled_start_local_date
            for item in ordered_records
        ):
            raise MatchParameterError("C6 records differ from match tour, cutoff, or local date")
        try:
            retirement_artifact = load_snapshot_retirement_artifact(snapshot)
        except ModelSnapshotError as exc:
            raise MatchParameterError(f"cannot load B6 artifact: {exc}") from exc
        if retirement_artifact.information_cutoff_utc > context.information_cutoff_utc:
            raise MatchParameterError("B6 artifact contains information after the match cutoff")
        retirement_parameters = _build_retirement_parameters(
            retirement_artifact,
            context,
            tuple(retirement_scenario_mixtures),
        )
        assert snapshot.inactivity_configuration is not None
        inactivity_parameters, c6_component_plans = _build_inactivity_parameters(
            fits,
            ordered_records,
            configuration_artifact_id=snapshot.inactivity_configuration.artifact_id,
        )
    elif inactivity_records or retirement_scenario_mixtures:
        raise MatchParameterError("pre-amendment snapshots cannot accept B6/C6 inputs")

    if snapshot.duration_complete:
        try:
            duration_parameters = load_snapshot_duration_artifact(snapshot)
        except ModelSnapshotError as exc:
            raise MatchParameterError(f"cannot load B5 duration artifact: {exc}") from exc
        if duration_parameters.information_cutoff_utc > context.information_cutoff_utc:
            raise MatchParameterError(
                "duration artifact contains information after the match cutoff"
            )

    c6_effect_plans = tuple(
        effect_plan
        for component_plan in c6_component_plans
        for effect_plan in component_plan.effect_plans
    )

    def direction(server: str, receiver: str) -> ServingDirectionDistribution:
        fitted_context = _fitted_direction_context(
            snapshot,
            context,
            server=server,
            receiver=receiver,
        )
        base = _direction_distribution(
            server_id=server,
            receiver_id=receiver,
            context=fitted_context,
            fits=fits,
            artifact_ids=snapshot.component_artifact_ids,
        )
        return _adjust_direction_map_distribution(base, c6_effect_plans)

    dependence = PerformanceDependenceSpec()
    provenance = MatchParameterProvenance(
        framework_version=snapshot.framework_version,
        implementation_version=MATCH_PARAMETER_IMPLEMENTATION_VERSION,
        snapshot_id=snapshot.snapshot_id,
        component_artifact_ids=tuple(
            (reference.component, reference.artifact_id)
            for reference in snapshot.component_artifacts
        ),
        data_cutoff_utc=snapshot.data_cutoff_utc,
        match_information_cutoff_utc=context.information_cutoff_utc,
        data_hash=snapshot.data_hash,
        config_hash=snapshot.config_hash,
        fit_code_commit=snapshot.code_commit,
        information_scenario_id=context.information_scenario_id,
        dependence_mode=dependence.mode,
        retirement_artifact_id=(
            None if retirement_parameters is None else retirement_parameters.artifact_id
        ),
        inactivity_configuration_artifact_id=(
            None
            if inactivity_parameters is None
            else inactivity_parameters.configuration_artifact_id
        ),
        duration_artifact_id=(
            None if duration_parameters is None else duration_parameters.artifact_id
        ),
    )
    return MatchParameterDistribution(
        snapshot=snapshot,
        context=context,
        player_a_serving=direction(context.player_a_id, context.player_b_id),
        player_b_serving=direction(context.player_b_id, context.player_a_id),
        performance_dependence=dependence,
        provenance=provenance,
        retirement=retirement_parameters,
        inactivity=inactivity_parameters,
        duration=duration_parameters,
        c6_component_plans=c6_component_plans,
    )


def restore_match_parameter_distribution(
    payload: str | bytes,
) -> MatchParameterDistribution:
    """Reload referenced artifacts and verify a serialized matchup record."""

    record = MatchParameterRecord.model_validate_json(payload)
    distribution = estimate_match(
        record.snapshot,
        record.context,
        inactivity_records=() if record.inactivity is None else record.inactivity.records,
        retirement_scenario_mixtures=(
            () if record.retirement is None else record.retirement.scenario_mixtures
        ),
    )
    if distribution.to_record() != record:
        raise MatchParameterError("reconstructed matchup differs from its serialized record")
    return distribution


def _rng(value: np.random.Generator) -> np.random.Generator:
    if not isinstance(value, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    return value


def sample_posterior_parameters(
    fitted: FittedServeComponent,
    rng: np.random.Generator,
) -> PosteriorParameterDraw:
    """Draw the full ordered theta vector from its stored Laplace approximation."""

    generator = _rng(rng)
    if not isinstance(fitted, FittedServeComponent):
        raise TypeError("fitted must be a FittedServeComponent")
    mean = np.asarray(fitted.posterior.map_parameters, dtype=np.float64)
    standard = np.asarray(generator.standard_normal(len(mean)), dtype=np.float64)
    if fitted.posterior.curvature_kind == "full":
        covariance = fitted.posterior.covariance_array()
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise MatchParameterError("validated full posterior covariance is not usable") from exc
        values = mean + factor @ standard
    else:
        standard_deviations = np.sqrt(
            np.asarray(fitted.posterior.variance_diagonal, dtype=np.float64)
        )
        values = mean + standard_deviations * standard
    if not np.all(np.isfinite(values)):
        raise MatchParameterError("posterior parameter draw is not finite")
    return PosteriorParameterDraw(
        component=fitted.component,
        parameter_names=fitted.posterior.parameter_names,
        values=tuple(float(value) for value in values),
    )


def sample_beta_probability(
    mean: float,
    concentration: float,
    rng: np.random.Generator,
) -> float:
    """Draw one beta match-performance probability after explicit shape checks."""

    generator = _rng(rng)
    if isinstance(mean, bool) or not isinstance(mean, Real):
        raise TypeError("beta mean must be numeric")
    if isinstance(concentration, bool) or not isinstance(concentration, Real):
        raise TypeError("beta concentration must be numeric")
    canonical_mean = float(mean)
    canonical_concentration = float(concentration)
    if not isfinite(canonical_mean) or not 0.0 < canonical_mean < 1.0:
        raise MatchParameterError("beta mean must be finite and strictly inside (0, 1)")
    if not isfinite(canonical_concentration) or canonical_concentration <= 0.0:
        raise MatchParameterError("beta concentration must be finite and positive")
    alpha = canonical_concentration * canonical_mean
    beta = canonical_concentration * (1.0 - canonical_mean)
    if not isfinite(alpha) or not isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        raise MatchParameterError("beta predictive shapes must be finite and positive")
    result = float(generator.beta(alpha, beta))
    if result == 0.0:
        # A mathematically interior beta variate can round to a floating-point
        # endpoint for very small shape parameters. Move only that representational
        # endpoint to the nearest interior float; do not clip invalid inputs.
        return float(np.nextafter(0.0, 1.0))
    if result == 1.0:
        return float(np.nextafter(1.0, 0.0))
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise MatchParameterError("beta generator returned a non-interior probability")
    return result


def _mean_from_linear_predictor(value: float) -> float:
    if not isfinite(value):
        raise MatchParameterError("sampled matchup linear predictor is not finite")
    mean = float(expit(value))
    if not 0.0 < mean < 1.0:
        raise MatchParameterError(
            "inverse-logit rounded to a probability boundary; refusing to clip"
        )
    return mean


def _shared_unseen_effects(
    left: ComponentParameterProjection,
    right: ComponentParameterProjection,
    rng: np.random.Generator,
    *,
    c6_scales: Mapping[tuple[str, str], float] | None = None,
    preset_values: Mapping[tuple[str, str], float] | None = None,
) -> tuple[tuple[UnseenEffectDraw, ...], Mapping[tuple[str, str], float]]:
    requirements: dict[tuple[str, str], float] = {}
    for projection in (left, right):
        for requirement in projection.unseen_effects:
            existing = requirements.get(requirement.key)
            if existing is not None and not np.isclose(
                existing,
                requirement.standard_deviation,
                rtol=1e-12,
                atol=0.0,
            ):
                raise MatchParameterError("shared unseen effect has inconsistent scales")
            requirements[requirement.key] = requirement.standard_deviation
    draws: list[UnseenEffectDraw] = []
    values = dict(preset_values or {})
    for block_name, level in sorted(requirements):
        if (block_name, level) in values:
            continue
        standard_deviation = requirements[(block_name, level)]
        if c6_scales is not None:
            standard_deviation *= c6_scales.get((block_name, level), 1.0)
        value = float(rng.normal(0.0, standard_deviation))
        if not isfinite(value):
            raise MatchParameterError("unseen-effect draw is not finite")
        values[(block_name, level)] = value
        draws.append(
            UnseenEffectDraw(
                block_name=block_name,
                level=level,
                standard_deviation=standard_deviation,
                value=value,
            )
        )
    return tuple(draws), MappingProxyType(values)


def _c6_projection_adjustment(
    *,
    fitted_context: FutureMatchContext,
    parameters: Sequence[float],
    plans: Sequence[_C6EffectPlan],
    adjusted_coordinates: Mapping[str, float],
) -> float:
    """Return the pathwise affine C6 delta for one directional predictor.

    The stored fit uses reference-coded ``z`` values and a shared log scale as
    engineering coordinates.  The frozen C6 coordinates are the centered
    player effects themselves.  Their Laplace draw is therefore the exact
    Gaussian first-order projection ``center + gradient @ (theta - theta_map)``;
    applying the affine C6 transform to the nonlinear ``sigma*z`` evaluation
    would not be a draw from the adjusted Gaussian recorded in the lock.
    """

    delta = 0.0
    for plan in plans:
        if not plan.changes_draw or not plan.projection.seen:
            continue
        coefficient = 0.0
        if (
            plan.role
            in {
                InactivityCoordinateRole.SERVER_GLOBAL,
                InactivityCoordinateRole.SERVER_HARD_DEVIATION,
            }
            and plan.player_id == fitted_context.serving_player_id
        ):
            coefficient = 1.0
        elif (
            plan.role
            in {
                InactivityCoordinateRole.RETURNER_GLOBAL,
                InactivityCoordinateRole.RETURNER_HARD_DEVIATION,
            }
            and plan.player_id == fitted_context.returning_player_id
        ):
            coefficient = -1.0
        if coefficient == 0.0:
            continue
        sampled_exact, seen = effect_block_player_contribution(
            plan.block,
            plan.player_id,
            parameters,
        )
        if not seen:
            raise MatchParameterError("C6 logical coordinate unexpectedly became unseen")
        try:
            adjusted = adjusted_coordinates[plan.projection.coordinate_id]
        except KeyError as exc:
            raise MatchParameterError("C6 adjusted Gaussian coordinate is missing") from exc
        delta += coefficient * (adjusted - sampled_exact)
    return delta


def _sample_c6_component_posterior(
    fitted: FittedServeComponent,
    plan: _C6ComponentPlan,
    rng: np.random.Generator,
) -> tuple[
    PosteriorParameterDraw,
    tuple[UnseenEffectDraw, ...],
    Mapping[tuple[str, str], float],
    tuple[C6AdjustedCoordinateDraw, ...],
]:
    """Sample the stored adjusted direct-effect Gaussian, not a post-draw patch."""

    if fitted.component is not plan.component or not plan.changes_draw:
        raise MatchParameterError("C6 direct-effect sampler requires one changed component")
    standard = np.asarray(
        rng.standard_normal(plan.adjusted_factor.shape[1]),
        dtype=np.float64,
    )
    sampled = plan.adjusted_mean + plan.adjusted_factor @ standard
    if not np.all(np.isfinite(sampled)):
        raise MatchParameterError("C6 direct-effect posterior draw is not finite")
    raw_values = sampled[: plan.raw_size]
    posterior = PosteriorParameterDraw(
        component=fitted.component,
        parameter_names=fitted.posterior.parameter_names,
        values=tuple(float(value) for value in raw_values),
    )

    logical_values = sampled[plan.raw_size :]
    adjusted_draws: list[C6AdjustedCoordinateDraw] = []
    unseen_by_key: dict[tuple[str, str], tuple[float, float]] = {}
    for index, effect_plan in enumerate(plan.effect_plans):
        value = float(logical_values[index])
        row = plan.raw_size + index
        if not effect_plan.projection.seen:
            key = (effect_plan.projection.block_name, effect_plan.player_id)
            standard_deviation = float(np.linalg.norm(plan.adjusted_factor[row]))
            prior = unseen_by_key.get(key)
            if prior is not None and prior != (standard_deviation, value):
                raise MatchParameterError("C6 unseen direct-effect coordinate is duplicated")
            unseen_by_key[key] = (standard_deviation, value)
        if effect_plan.changes_draw:
            adjusted_draws.append(
                C6AdjustedCoordinateDraw(
                    component=effect_plan.component,
                    coordinate_id=effect_plan.projection.coordinate_id,
                    player_id=effect_plan.player_id,
                    role=effect_plan.role,
                    value=value,
                )
            )
    unseen_draws = tuple(
        UnseenEffectDraw(
            block_name=block_name,
            level=level,
            standard_deviation=unseen_by_key[(block_name, level)][0],
            value=unseen_by_key[(block_name, level)][1],
        )
        for block_name, level in sorted(unseen_by_key)
    )
    unseen_values = MappingProxyType(
        {key: value for key, (_standard_deviation, value) in unseen_by_key.items()}
    )
    return posterior, unseen_draws, unseen_values, tuple(adjusted_draws)


def _realized_linear_predictor(
    projection: ComponentParameterProjection,
    unseen_values: Mapping[tuple[str, str], float],
) -> float:
    return projection.base_linear_predictor + sum(
        requirement.coefficient * unseen_values[requirement.key]
        for requirement in projection.unseen_effects
    )


def sample_matchup_parameters(
    distribution: MatchParameterDistribution,
    rng: np.random.Generator,
) -> MatchupParameterDraw:
    """Sample parameter uncertainty once and evaluate both serving directions."""

    generator = _rng(rng)
    joint: list[JointComponentParameterDraw] = []
    c6_by_component = {plan.component: plan for plan in distribution.c6_component_plans}
    if len(c6_by_component) != len(distribution.c6_component_plans):
        raise MatchParameterError("C6 component plans must be unique")
    for component in _COMPONENT_ORDER:
        left = distribution.player_a_serving.by_component[component]
        right = distribution.player_b_serving.by_component[component]
        c6_component = c6_by_component.get(component)
        component_plans = () if c6_component is None else c6_component.effect_plans
        if c6_component is not None and c6_component.changes_draw:
            posterior, unseen_draws, unseen_values, c6_coordinates = _sample_c6_component_posterior(
                left.fit, c6_component, generator
            )
        else:
            posterior = sample_posterior_parameters(left.fit, generator)
            unseen_draws = ()
            unseen_values = MappingProxyType({})
            c6_coordinates = ()
        try:
            left_projection = _project_validated_component_parameters(
                left.fit,
                distribution.player_a_serving.context,
                posterior.values,
            )
            right_projection = _project_validated_component_parameters(
                right.fit,
                distribution.player_b_serving.context,
                posterior.values,
            )
        except ModelDataError as exc:
            raise MatchParameterError(
                f"cannot evaluate sampled {component.value} parameters: {exc}"
            ) from exc
        if left_projection.predictive_concentration != (right_projection.predictive_concentration):
            raise MatchParameterError("shared component theta produced inconsistent kappa")
        if c6_component is None or not c6_component.changes_draw:
            unseen_draws, unseen_values = _shared_unseen_effects(
                left_projection,
                right_projection,
                generator,
            )
        else:
            additional_draws, unseen_values = _shared_unseen_effects(
                left_projection,
                right_projection,
                generator,
                preset_values=unseen_values,
            )
            unseen_draws += additional_draws
            required_unseen = {
                requirement.key
                for projection in (left_projection, right_projection)
                for requirement in projection.unseen_effects
            }
            if not required_unseen.issubset(unseen_values):
                raise MatchParameterError(
                    "C6 direct-effect draw does not match unseen matchup requirements"
                )
        adjusted_values = MappingProxyType(
            {item.coordinate_id: item.value for item in c6_coordinates}
        )
        left_delta = _c6_projection_adjustment(
            fitted_context=distribution.player_a_serving.context,
            parameters=posterior.values,
            plans=component_plans,
            adjusted_coordinates=adjusted_values,
        )
        right_delta = _c6_projection_adjustment(
            fitted_context=distribution.player_b_serving.context,
            parameters=posterior.values,
            plans=component_plans,
            adjusted_coordinates=adjusted_values,
        )
        joint.append(
            JointComponentParameterDraw(
                component=component,
                posterior=posterior,
                unseen_effects=unseen_draws,
                c6_adjusted_coordinates=c6_coordinates,
                player_a_serving_mean=_mean_from_linear_predictor(
                    _realized_linear_predictor(left_projection, unseen_values) + left_delta
                ),
                player_b_serving_mean=_mean_from_linear_predictor(
                    _realized_linear_predictor(right_projection, unseen_values) + right_delta
                ),
                predictive_concentration=left_projection.predictive_concentration,
            )
        )

    def direction(
        server_id: str,
        receiver_id: str,
        mean_field: Literal["player_a_serving_mean", "player_b_serving_mean"],
    ) -> ServingDirectionParameterDraw:
        return ServingDirectionParameterDraw(
            server_id=server_id,
            receiver_id=receiver_id,
            components=tuple(
                BetaComponentParameters(
                    component=item.component,
                    mean=getattr(item, mean_field),
                    concentration=item.predictive_concentration,
                )
                for item in joint
            ),
        )

    return MatchupParameterDraw(
        snapshot_id=distribution.snapshot.snapshot_id,
        player_a_serving=direction(
            distribution.context.player_a_id,
            distribution.context.player_b_id,
            "player_a_serving_mean",
        ),
        player_b_serving=direction(
            distribution.context.player_b_id,
            distribution.context.player_a_id,
            "player_b_serving_mean",
        ),
        components=tuple(joint),
    )


def sample_serve_performance(
    direction: ServingDirectionParameterDraw,
    rng: np.random.Generator,
) -> ServePerformanceDraw:
    """Apply beta match-performance variation to fixed directional means once."""

    generator = _rng(rng)
    values = {
        item.component: sample_beta_probability(item.mean, item.concentration, generator)
        for item in direction.components
    }
    return ServePerformanceDraw(
        first_serve_in=values[ServeComponent.F],
        ace_given_first_in=values[ServeComponent.A],
        returnable_first_win=values[ServeComponent.Q1],
        double_fault_given_second_opp=values[ServeComponent.D],
        playable_second_win=values[ServeComponent.Q2],
    )


def derive_match_seed_plan(seed: np.random.SeedSequence) -> MatchSeedPlan:
    """Derive stable independent stage streams without mutating the caller's seed."""

    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError("seed must be an explicit numpy.random.SeedSequence")
    root_reference = SeedReference.from_seed_sequence(seed)
    root = root_reference.to_seed_sequence()
    parameter, player_a, player_b, point_path, retirement, duration = root.spawn(6)
    retirement_parameters, retirement_boundaries = retirement.spawn(2)
    duration_parameters, duration_residual = duration.spawn(2)
    return MatchSeedPlan(
        root=root_reference,
        parameter_draws=SeedReference.from_seed_sequence(parameter),
        player_a_performance=SeedReference.from_seed_sequence(player_a),
        player_b_performance=SeedReference.from_seed_sequence(player_b),
        point_path=SeedReference.from_seed_sequence(point_path),
        retirement=SeedReference.from_seed_sequence(retirement),
        retirement_parameters=SeedReference.from_seed_sequence(retirement_parameters),
        retirement_boundaries=SeedReference.from_seed_sequence(retirement_boundaries),
        duration=SeedReference.from_seed_sequence(duration),
        duration_parameters=SeedReference.from_seed_sequence(duration_parameters),
        duration_residual=SeedReference.from_seed_sequence(duration_residual),
    )


def generator_from_seed_reference(seed: SeedReference) -> np.random.Generator:
    """Reconstruct the explicitly recorded PCG64 child generator."""

    if not isinstance(seed, SeedReference):
        raise TypeError("seed must be a SeedReference")
    return np.random.Generator(np.random.PCG64(seed.to_seed_sequence()))


def sample_match_performance(
    distribution: MatchParameterDistribution,
    seed: np.random.SeedSequence,
) -> MatchPerformanceDraw:
    """Run the full parameter-then-performance stages for one match path."""

    plan = derive_match_seed_plan(seed)
    parameters = sample_matchup_parameters(
        distribution,
        generator_from_seed_reference(plan.parameter_draws),
    )
    player_a = sample_serve_performance(
        parameters.player_a_serving,
        generator_from_seed_reference(plan.player_a_performance),
    )
    player_b = sample_serve_performance(
        parameters.player_b_serving,
        generator_from_seed_reference(plan.player_b_performance),
    )
    retirement_draws: tuple[RetirementPathDraw, ...] = ()
    if distribution.retirement is not None:
        retirement_rng = generator_from_seed_reference(plan.retirement_parameters)
        # All path-level scenarios are selected before either beta posterior is
        # drawn.  A degenerate one-state scenario needs no categorical RNG.
        selected_scenarios = tuple(
            mixture.scenarios[0]
            if len(mixture.scenarios) == 1
            else draw_retirement_scenario(mixture, retirement_rng)
            for mixture in distribution.retirement.scenario_mixtures
        )
        retirement_draws = cast(
            tuple[RetirementPathDraw, RetirementPathDraw],
            tuple(
                draw_player_retirement_path(posterior, scenario, retirement_rng)
                for posterior, scenario in zip(
                    distribution.retirement.player_posteriors,
                    selected_scenarios,
                    strict=True,
                )
            ),
        )
    return MatchPerformanceDraw(
        matchup_parameters=parameters,
        player_a_serving=player_a,
        player_b_serving=player_b,
        seed_plan=plan,
        retirement_draws=retirement_draws,
    )


__all__ = [
    "C6_DIRECT_EFFECT_POSTERIOR_VERSION",
    "MATCH_PARAMETER_IMPLEMENTATION_VERSION",
    "MATCH_RNG_BIT_GENERATOR",
    "BetaComponentParameters",
    "C6AdjustedCoordinateDraw",
    "DirectionComponentDistribution",
    "DurationMatchParameters",
    "JointComponentParameterDraw",
    "MatchCondition",
    "MatchContext",
    "MatchParameterDistribution",
    "MatchParameterError",
    "MatchParameterProvenance",
    "MatchParameterRecord",
    "MatchPerformanceDraw",
    "MatchSeedPlan",
    "MatchupParameterDraw",
    "PerformanceDependenceSpec",
    "PosteriorParameterDraw",
    "SeedReference",
    "ServingDirectionDistribution",
    "ServingDirectionParameterDraw",
    "ServingDirectionRecord",
    "UnseenEffectDraw",
    "derive_match_seed_plan",
    "estimate_match",
    "generator_from_seed_reference",
    "restore_match_parameter_distribution",
    "sample_beta_probability",
    "sample_match_performance",
    "sample_matchup_parameters",
    "sample_posterior_parameters",
    "sample_serve_performance",
]
