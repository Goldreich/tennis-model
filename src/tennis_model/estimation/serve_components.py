"""Five frozen primitive serve-component models for Tennis Model v1.0.

Each fit consumes the long-form Milestone 1 component-count table directly.
Rows enter one beta-binomial likelihood exactly once; surface pooling is an
explicit joint global-plus-surface parameterization, never a two-stage prior.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from math import exp, isfinite, log
from numbers import Integral, Real
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator
from scipy.optimize import OptimizeResult, minimize  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.special import expit, logit  # type: ignore[import-untyped]

from tennis_model.data.artifacts import (
    ProcessedArtifactBundle,
    read_processed_table,
    verify_processed_bundle,
)
from tennis_model.data.component_counts import (
    TRANSFORMATION_VERSION,
    ComponentCountTable,
    ComponentStatus,
)
from tennis_model.estimation.weighted_glmm import (
    FROZEN_WINDOW_DAYS,
    CurvatureResult,
    beta_binomial_scores,
    laplace_curvature,
    time_weight,
)
from tennis_model.schemas import FrozenModel, Tour

FloatArray = NDArray[np.float64]
FRAMEWORK_VERSION: Final[Literal["v1.0"]] = "v1.0"
MODEL_IMPLEMENTATION_VERSION: Final[Literal["serve-components-map-laplace/v1"]] = (
    "serve-components-map-laplace/v1"
)
FIT_INPUT_SET_VERSION: Final[Literal["fit-input-set/v1"]] = "fit-input-set/v1"

type EffectRole = Literal[
    "server_global",
    "returner_global",
    "server_surface",
    "returner_surface",
    "event_year",
]
type ScaleGroup = Literal["server_global", "returner_global", "surface", "event_year"]


class _EstimationModel(FrozenModel):
    """Frozen artifact record that also rejects NaN and infinite numbers."""

    model_config = ConfigDict(allow_inf_nan=False)


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("artifact hashes must be 64 lowercase hexadecimal characters")
    return normalized


class ServeComponent(StrEnum):
    """The five primitive, and only primitive, serve components."""

    F = "F"
    A = "A"
    Q1 = "Q1"
    D = "D"
    Q2 = "Q2"


_OPPONENT_COMPONENTS = frozenset({ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2})
_MISSING_SURFACE = "__missing_surface__"
_MISSING_EVENT_YEAR = "__missing_event_year__"


class ModelDataError(ValueError):
    """Count rows or contexts violate the frozen fitting contract."""


class FitConvergenceError(RuntimeError):
    """MAP optimization failed and no fitted artifact may be emitted."""

    def __init__(self, component: ServeComponent, result: OptimizeResult) -> None:
        self.component = component
        self.status = int(result.status)
        self.message = str(result.message)
        self.iterations = int(getattr(result, "nit", 0))
        super().__init__(
            f"{component.value} MAP fit failed: status={self.status}, "
            f"iterations={self.iterations}, message={self.message}"
        )


class PriorConfig(_EstimationModel):
    """Explicit probability-affecting priors absent as numbers from the prose spec."""

    intercept_sd: Annotated[float, Field(gt=0)]
    context_coefficient_sd: Annotated[float, Field(gt=0)]
    log_shrinkage_scale_mean: float
    log_shrinkage_scale_sd: Annotated[float, Field(gt=0)]

    @field_validator("*", mode="before")
    @classmethod
    def prior_values_are_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("prior values must be numeric YAML scalars")
        return value


class OptimizerConfig(_EstimationModel):
    """Auditable numerical settings; these affect computation, not model features."""

    method: Literal["L-BFGS-B"]
    max_iterations: Annotated[int, Field(ge=1)]
    max_line_search_steps: Annotated[int, Field(ge=1)]
    gradient_tolerance: Annotated[float, Field(gt=0)]
    function_tolerance: Annotated[float, Field(gt=0)]
    initial_kappa: Annotated[float, Field(gt=0)]
    kappa_lower: Annotated[float, Field(gt=0)]
    kappa_upper: Annotated[float, Field(gt=0)]
    initial_shrinkage_scale: Annotated[float, Field(gt=0)]
    shrinkage_scale_lower: Annotated[float, Field(gt=0)]
    shrinkage_scale_upper: Annotated[float, Field(gt=0)]
    laplace_max_full_parameters: Annotated[int, Field(ge=1)]
    laplace_relative_step: Annotated[float, Field(gt=0)]
    laplace_eigenvalue_floor: Annotated[float, Field(gt=0)]

    @field_validator(
        "max_iterations",
        "max_line_search_steps",
        "laplace_max_full_parameters",
        mode="before",
    )
    @classmethod
    def integer_options_are_exact(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("integer optimizer options must be numeric YAML integers")
        return value

    @field_validator(
        "gradient_tolerance",
        "function_tolerance",
        "initial_kappa",
        "kappa_lower",
        "kappa_upper",
        "initial_shrinkage_scale",
        "shrinkage_scale_lower",
        "shrinkage_scale_upper",
        "laplace_relative_step",
        "laplace_eigenvalue_floor",
        mode="before",
    )
    @classmethod
    def float_options_are_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("floating optimizer options must be numeric YAML scalars")
        return value

    @model_validator(mode="after")
    def ordered_bounds_and_initial_values(self) -> Self:
        if not self.kappa_lower < self.initial_kappa < self.kappa_upper:
            raise ValueError("initial_kappa must lie strictly inside its bounds")
        if not (
            self.shrinkage_scale_lower < self.initial_shrinkage_scale < self.shrinkage_scale_upper
        ):
            raise ValueError("initial_shrinkage_scale must lie strictly inside its bounds")
        return self


class ContextConfig(_EstimationModel):
    """Only preapproved low-dimensional context switches."""

    include_indoor_hard: bool
    event_year_effect_components: tuple[ServeComponent, ...]

    @field_validator("include_indoor_hard", mode="before")
    @classmethod
    def context_switch_is_boolean(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("context switches must be YAML booleans")
        return value

    @model_validator(mode="after")
    def components_are_unique(self) -> Self:
        if len(self.event_year_effect_components) != len(set(self.event_year_effect_components)):
            raise ValueError("event_year_effect_components must be unique")
        return self


class DiagnosticConfig(_EstimationModel):
    sparse_weighted_trials_warning: Annotated[float, Field(gt=0)]

    @field_validator("sparse_weighted_trials_warning", mode="before")
    @classmethod
    def warning_threshold_is_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("diagnostic thresholds must be numeric YAML scalars")
        return value


class ServeModelConfig(_EstimationModel):
    """Complete, hashable Milestone 3 model and optimizer configuration."""

    framework_version: Literal["v1.0"]
    prior: PriorConfig
    optimizer: OptimizerConfig
    context: ContextConfig
    diagnostics: DiagnosticConfig

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class FitProvenance(_EstimationModel):
    """Caller-supplied immutable identities for one reproducible fit."""

    data_snapshot_sha256: str
    component_count_artifact_sha256: str
    code_commit: str
    fitted_at_utc: datetime

    @field_validator("data_snapshot_sha256", "component_count_artifact_sha256")
    @classmethod
    def sha256_is_valid(cls, value: str) -> str:
        return _normalize_sha256(value)

    @field_validator("code_commit")
    @classmethod
    def commit_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code_commit must not be empty; use an explicit unavailable marker")
        return value.strip()

    @field_validator("fitted_at_utc")
    @classmethod
    def fitted_at_is_aware(cls, value: datetime) -> datetime:
        return _utc_datetime(value, field="fitted_at_utc")


class EffectBlock(_EstimationModel):
    """One reference-coded block with a centered Gaussian prior."""

    name: str
    role: EffectRole
    scale_group: ScaleGroup
    surface: str | None
    levels: tuple[str, ...]
    free_parameter_indices: tuple[int, ...]
    scale_parameter_index: int

    @model_validator(mode="after")
    def dimensions_match(self) -> Self:
        if not self.levels:
            raise ValueError("effect block must contain at least one observed level")
        expected = max(0, len(self.levels) - 1)
        if len(self.free_parameter_indices) != expected:
            raise ValueError("effect block must reference one parameter per non-reference level")
        if len(set(self.levels)) != len(self.levels) or tuple(sorted(self.levels)) != self.levels:
            raise ValueError("effect block levels must be unique and sorted")
        return self


class PosteriorApproximation(_EstimationModel):
    """Reproducible MAP vector and regularized Laplace approximation."""

    parameter_names: tuple[str, ...]
    map_parameters: tuple[float, ...]
    curvature_kind: Literal["full", "diagonal"]
    hessian: tuple[tuple[float, ...], ...] | None
    covariance: tuple[tuple[float, ...], ...] | None
    variance_diagonal: tuple[float, ...]
    raw_min_eigenvalue: float
    regularization_added: Annotated[float, Field(ge=0)]
    condition_number: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def dimensions_are_consistent(self) -> Self:
        size = len(self.parameter_names)
        if len(self.map_parameters) != size or len(self.variance_diagonal) != size:
            raise ValueError("posterior vector dimensions do not match parameter names")
        if len(set(self.parameter_names)) != size:
            raise ValueError("posterior parameter names must be unique")
        if self.curvature_kind == "full":
            if self.hessian is None or self.covariance is None:
                raise ValueError("full curvature requires Hessian and covariance matrices")
            if any(len(row) != size for row in self.hessian) or len(self.hessian) != size:
                raise ValueError("Hessian dimensions do not match parameter vector")
            if any(len(row) != size for row in self.covariance) or len(self.covariance) != size:
                raise ValueError("covariance dimensions do not match parameter vector")
            hessian = np.asarray(self.hessian, dtype=np.float64)
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
                raise ValueError("Hessian must be symmetric")
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise ValueError("covariance must be symmetric")
            if not np.allclose(
                np.diag(covariance),
                np.asarray(self.variance_diagonal),
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError("covariance diagonal differs from variance_diagonal")
            if np.min(np.linalg.eigvalsh(hessian)) <= 0.0:
                raise ValueError("regularized Hessian must be positive definite")
            if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
                raise ValueError("Laplace covariance must be positive definite")
            if not np.allclose(
                hessian @ covariance,
                np.eye(size, dtype=np.float64),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError("stored covariance is not the inverse Hessian")
        elif self.hessian is not None or self.covariance is not None:
            raise ValueError("diagonal curvature must not contain dense matrices")
        if not np.all(np.isfinite(np.asarray(self.map_parameters, dtype=np.float64))):
            raise ValueError("MAP parameters must all be finite")
        variances = np.asarray(self.variance_diagonal, dtype=np.float64)
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("Laplace variances must all be finite and positive")
        return self

    def covariance_array(self) -> FloatArray:
        if self.covariance is None:
            return np.diag(np.asarray(self.variance_diagonal, dtype=np.float64))
        return np.asarray(self.covariance, dtype=np.float64)


class ExclusionCount(_EstimationModel):
    reason: str
    rows: Annotated[int, Field(ge=0)]


class PlayerInformation(_EstimationModel):
    player_id: str
    weighted_trials: Annotated[float, Field(ge=0)]
    effective_matches: Annotated[float, Field(ge=0)]
    information_equivalent_trials: Annotated[float, Field(ge=0)]
    sparse_warning: bool


class CoefficientEstimate(_EstimationModel):
    name: str
    kind: str
    value: float


class NamedIndex(_EstimationModel):
    name: str
    index: Annotated[int, Field(ge=0)]


class NamedScale(_EstimationModel):
    name: str
    value: Annotated[float, Field(gt=0)]


class FitDiagnostics(_EstimationModel):
    usable_rows: Annotated[int, Field(ge=0)]
    raw_trials: Annotated[int, Field(ge=0)]
    weighted_trials: Annotated[float, Field(ge=0)]
    unique_servers: Annotated[int, Field(ge=0)]
    unique_returners: Annotated[int, Field(ge=0)]
    excluded_rows: Annotated[int, Field(ge=0)]
    missing_rows: Annotated[int, Field(ge=0)]
    quarantined_rows: Annotated[int, Field(ge=0)]
    quarantined_anomaly_count: Annotated[int, Field(ge=0)]
    date_fallback_rows: Annotated[int, Field(ge=0)]
    exclusion_counts: tuple[ExclusionCount, ...]
    kappa: Annotated[float, Field(gt=0)]
    shrinkage_scale_estimates: tuple[NamedScale, ...]
    converged: bool
    objective_value: float
    iterations: Annotated[int, Field(ge=0)]
    function_evaluations: Annotated[int, Field(ge=0)]
    gradient_max_abs: Annotated[float, Field(ge=0)]
    optimizer_status: int
    optimizer_message: str
    observed_rate: Annotated[float, Field(ge=0, le=1)]
    fitted_rate: Annotated[float, Field(ge=0, le=1)]
    weighted_row_rmse: Annotated[float, Field(ge=0)]
    player_information: tuple[PlayerInformation, ...]
    warnings: tuple[str, ...]

    @model_validator(mode="after")
    def scale_names_are_unique(self) -> Self:
        names = tuple(item.name for item in self.shrinkage_scale_estimates)
        if not names or len(set(names)) != len(names):
            raise ValueError("fitted shrinkage-scale names must be nonempty and unique")
        reasons = tuple(item.reason for item in self.exclusion_counts)
        if len(set(reasons)) != len(reasons):
            raise ValueError("fit exclusion reasons must be unique")
        if sum(item.rows for item in self.exclusion_counts) != self.excluded_rows:
            raise ValueError("fit exclusion counts do not sum to excluded_rows")
        players = tuple(item.player_id for item in self.player_information)
        if len(players) != self.unique_servers or len(set(players)) != len(players):
            raise ValueError("player information must contain every unique server exactly once")
        if self.usable_rows == 0 or self.raw_trials == 0 or self.weighted_trials <= 0.0:
            raise ValueError("a converged fit must contain positive usable exposure")
        return self

    @property
    def shrinkage_scales(self) -> Mapping[str, float]:
        return MappingProxyType({item.name: item.value for item in self.shrinkage_scale_estimates})


class FittedServeComponent(_EstimationModel):
    """One converged tour/component fit and everything needed for prediction."""

    implementation_version: Literal["serve-components-map-laplace/v1"]
    framework_version: Literal["v1.0"]
    tour: Tour
    component: ServeComponent
    data_cutoff_utc: datetime
    training_window_start_utc: datetime
    fitted_at_utc: datetime
    data_snapshot_sha256: str
    component_count_artifact_sha256: str
    training_data_sha256: str
    model_config_sha256: str
    code_commit: str
    config: ServeModelConfig
    fixed_parameters: tuple[NamedIndex, ...]
    effect_blocks: tuple[EffectBlock, ...]
    posterior: PosteriorApproximation
    diagnostics: FitDiagnostics
    coefficient_summary: tuple[CoefficientEstimate, ...]

    @property
    def fixed_parameter_indices(self) -> Mapping[str, int]:
        return MappingProxyType({item.name: item.index for item in self.fixed_parameters})

    @field_validator("data_cutoff_utc", "training_window_start_utc", "fitted_at_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: Any) -> datetime:
        return _utc_datetime(value, field=info.field_name)

    @field_validator(
        "data_snapshot_sha256",
        "component_count_artifact_sha256",
        "training_data_sha256",
        "model_config_sha256",
    )
    @classmethod
    def hashes_are_valid(cls, value: str) -> str:
        return _normalize_sha256(value)

    @field_validator("code_commit")
    @classmethod
    def code_commit_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code_commit must not be empty; use an explicit unavailable marker")
        return value.strip()

    @model_validator(mode="after")
    def fit_is_coherent(self) -> Self:
        if self.framework_version != self.config.framework_version:
            raise ValueError("fit framework and configuration framework differ")
        if self.model_config_sha256 != self.config.sha256:
            raise ValueError("stored model configuration hash is inconsistent")
        if not self.diagnostics.converged:
            raise ValueError("a failed optimizer result cannot be a fitted model")
        if self.training_window_start_utc != self.data_cutoff_utc - timedelta(
            days=FROZEN_WINDOW_DAYS
        ):
            raise ValueError("training window does not use the frozen 1,095-day width")
        if self.fitted_at_utc < self.data_cutoff_utc:
            raise ValueError("fit creation time cannot precede its information cutoff")
        parameter_count = len(self.posterior.parameter_names)
        fixed_names = tuple(item.name for item in self.fixed_parameters)
        if not fixed_names or len(set(fixed_names)) != len(fixed_names):
            raise ValueError("fixed-effect names must be nonempty and unique")
        fixed_indices = tuple(self.fixed_parameter_indices.values())
        if len(set(fixed_indices)) != len(fixed_indices) or any(
            index < 0 or index >= parameter_count for index in fixed_indices
        ):
            raise ValueError("fixed-effect indices are invalid or duplicated")
        for name, index in self.fixed_parameter_indices.items():
            if self.posterior.parameter_names[index] != name:
                raise ValueError("fixed-effect index does not match posterior parameter name")
        free_indices: list[int] = []
        scale_indices: set[int] = set()
        for block in self.effect_blocks:
            indices = (*block.free_parameter_indices, block.scale_parameter_index)
            if any(index < 0 or index >= parameter_count for index in indices):
                raise ValueError("effect block references an invalid parameter index")
            free_indices.extend(block.free_parameter_indices)
            scale_indices.add(block.scale_parameter_index)
            if self.posterior.parameter_names[block.scale_parameter_index] != (
                f"log_scale:{block.scale_group}"
            ):
                raise ValueError("effect block scale index does not match its parameter name")
            for index in block.free_parameter_indices:
                if not self.posterior.parameter_names[index].startswith(f"z:{block.name}["):
                    raise ValueError("effect block index does not match its parameter name")
        log_kappa_index = self.posterior.parameter_names.index("log_kappa")
        if len(set(free_indices)) != len(free_indices):
            raise ValueError("random-effect parameter indices are duplicated")
        fixed_set = set(fixed_indices)
        free_set = set(free_indices)
        if fixed_set & free_set or fixed_set & scale_indices or free_set & scale_indices:
            raise ValueError("fixed, random-effect, and scale parameter indices overlap")
        all_indices = set(fixed_indices) | set(free_indices) | scale_indices | {log_kappa_index}
        if all_indices != set(range(parameter_count)):
            raise ValueError("posterior contains unreferenced or multiply purposed parameters")
        roles = [block.role for block in self.effect_blocks]
        if roles.count("server_global") != 1:
            raise ValueError("every component requires exactly one server-global block")
        if self.component in _OPPONENT_COMPONENTS:
            if roles.count("returner_global") != 1:
                raise ValueError("opponent-adjusted components require one returner-global block")
        elif any(role.startswith("returner") for role in roles):
            raise ValueError("F and D must not contain returner effects")
        expected_scale_groups = {block.scale_group for block in self.effect_blocks}
        if set(self.diagnostics.shrinkage_scales) != expected_scale_groups:
            raise ValueError("diagnostic shrinkage scales do not match effect blocks")
        coefficient_names = tuple(item.name for item in self.coefficient_summary)
        if len(set(coefficient_names)) != len(coefficient_names):
            raise ValueError("coefficient summary names must be unique")
        if not np.isclose(
            exp(self.posterior.map_parameters[log_kappa_index]),
            self.diagnostics.kappa,
            rtol=1e-10,
            atol=0.0,
        ):
            raise ValueError("posterior and diagnostic kappa values differ")
        return self

    @property
    def kappa(self) -> float:
        index = self.posterior.parameter_names.index("log_kappa")
        return exp(self.posterior.map_parameters[index])


class FutureMatchContext(_EstimationModel):
    serving_player_id: str
    returning_player_id: str
    tour: Tour
    surface: str
    indoor: bool | None
    event: str | None
    event_year: int | None
    match_date_utc: datetime
    information_cutoff_utc: datetime
    best_of: Literal[3, 5] | None = None
    serving_hand: str | None = None
    returning_hand: str | None = None

    @field_validator("match_date_utc", "information_cutoff_utc")
    @classmethod
    def dates_are_aware(cls, value: datetime, info: Any) -> datetime:
        return _utc_datetime(value, field=info.field_name)

    @field_validator("serving_player_id", "returning_player_id", "surface")
    @classmethod
    def required_text_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("player identifiers and surface must not be empty")
        return value.strip()

    @field_validator("event")
    @classmethod
    def optional_event_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("event must be nonempty when supplied")
        return None if value is None else value.strip()

    @model_validator(mode="after")
    def match_is_not_before_its_information_cutoff(self) -> Self:
        if self.match_date_utc < self.information_cutoff_utc:
            raise ValueError("future match date must not precede its information cutoff")
        if self.serving_player_id == self.returning_player_id:
            raise ValueError("serving and returning players must be distinct")
        return self


class ComponentPrediction(_EstimationModel):
    component: ServeComponent
    map_mean: Annotated[float, Field(gt=0, lt=1)]
    linear_predictor_map: float
    linear_predictor_sd: Annotated[float, Field(ge=0)]
    unseen_effect_variance: Annotated[float, Field(ge=0)]
    predictive_concentration: Annotated[float, Field(gt=0)]
    beta_alpha_at_map: Annotated[float, Field(gt=0)]
    beta_beta_at_map: Annotated[float, Field(gt=0)]
    serving_player_seen: bool
    returning_player_seen: bool | None
    surface_seen: bool
    event_year_seen: bool | None

    @model_validator(mode="after")
    def beta_shape_is_coherent(self) -> Self:
        if not np.isclose(
            self.beta_alpha_at_map + self.beta_beta_at_map,
            self.predictive_concentration,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("beta shape parameters do not sum to concentration")
        if not np.isclose(
            self.beta_alpha_at_map / self.predictive_concentration,
            self.map_mean,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("beta shape parameters do not reproduce the MAP mean")
        return self


class UnseenEffectRequirement(_EstimationModel):
    """One zero-centered latent effect absent from a fitted parameter vector."""

    block_name: str
    level: str
    standard_deviation: Annotated[float, Field(gt=0)]
    coefficient: Literal[-1, 1]

    @field_validator("block_name", "level")
    @classmethod
    def identity_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("unseen-effect identities must not be empty")
        return value.strip()

    @property
    def key(self) -> tuple[str, str]:
        """Stable identity used to share one latent draw across projections."""

        return self.block_name, self.level


class ComponentParameterProjection(_EstimationModel):
    """A matchup linear predictor evaluated at one ordered parameter vector."""

    component: ServeComponent
    parameter_names: tuple[str, ...]
    base_linear_predictor: float
    predictive_concentration: Annotated[float, Field(gt=0)]
    unseen_effects: tuple[UnseenEffectRequirement, ...]
    serving_player_seen: bool
    returning_player_seen: bool | None
    surface_seen: bool
    event_year_seen: bool | None

    @model_validator(mode="after")
    def requirements_are_unique(self) -> Self:
        keys = tuple(item.key for item in self.unseen_effects)
        if len(keys) != len(set(keys)):
            raise ValueError("unseen-effect requirements must be unique")
        return self


class PlayerEffectCoordinateProjection(_EstimationModel):
    """Canonical direct player-effect coordinate for the Laplace posterior.

    Fitted artifacts retain their original reference-coded coordinates.  This
    descriptor exposes the centered player coefficient (including the reference
    player) and the MAP Jacobian that defines its versioned canonical Gaussian
    coordinate in the Laplace approximation.  This keeps reference coding and
    scale parameterization internal while allowing C6 to sample its adjusted
    direct-effect Gaussian explicitly.
    """

    component: ServeComponent
    player_id: str
    block_name: str
    role: EffectRole
    surface: str | None
    coordinate_id: str
    seen: bool
    map_contribution: float
    map_gradient: tuple[float, ...]
    unseen_standard_deviation: Annotated[float, Field(gt=0)] | None

    @model_validator(mode="after")
    def state_is_coherent(self) -> Self:
        if not self.player_id.strip() or not self.coordinate_id.strip():
            raise ValueError("logical player-effect identities must not be empty")
        if self.seen and self.unseen_standard_deviation is not None:
            raise ValueError("seen player effects cannot have an unseen-effect scale")
        if not self.seen and self.unseen_standard_deviation is None:
            raise ValueError("unseen player effects require their fitted prior scale")
        return self


class ComponentLinearCoordinateProjection(_EstimationModel):
    """MAP linear predictor and full theta gradient for auditable transforms."""

    component: ServeComponent
    linear_predictor_map: float
    gradient: tuple[float, ...]
    unseen_effects: tuple[UnseenEffectRequirement, ...]


class ServeFitBundleIdentity(_EstimationModel):
    framework_version: Literal["v1.0"]
    implementation_version: Literal["serve-components-map-laplace/v1"]
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime
    data_snapshot_sha256: str
    component_count_artifact_sha256: str
    model_config_sha256: str
    code_commit: str

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def dates_are_aware(cls, value: datetime, info: Any) -> datetime:
        return _utc_datetime(value, field=info.field_name)

    @field_validator(
        "data_snapshot_sha256",
        "component_count_artifact_sha256",
        "model_config_sha256",
    )
    @classmethod
    def hashes_are_valid(cls, value: str) -> str:
        return _normalize_sha256(value)


class ServePerformanceDistribution(_EstimationModel):
    """Primitive posterior/predictive information, not a match realization."""

    fit_identity: ServeFitBundleIdentity
    context: FutureMatchContext
    first_serve_in: ComponentPrediction
    ace_given_first_in: ComponentPrediction
    returnable_first_win: ComponentPrediction
    double_fault_given_second_opp: ComponentPrediction
    playable_second_win: ComponentPrediction

    @model_validator(mode="after")
    def components_are_exactly_routed(self) -> Self:
        observed = (
            self.first_serve_in.component,
            self.ace_given_first_in.component,
            self.returnable_first_win.component,
            self.double_fault_given_second_opp.component,
            self.playable_second_win.component,
        )
        expected = (
            ServeComponent.F,
            ServeComponent.A,
            ServeComponent.Q1,
            ServeComponent.D,
            ServeComponent.Q2,
        )
        if observed != expected:
            raise ValueError("serve-performance distribution must contain F/A/Q1/D/Q2")
        return self


@dataclass(frozen=True, slots=True)
class PreparedComponentRows:
    frame: pd.DataFrame
    successes: FloatArray
    trials: FloatArray
    weights: FloatArray
    data_sha256: str
    input_rows: int
    excluded: Counter[str]
    missing_rows: int
    quarantined_rows: int
    quarantined_anomaly_count: int
    date_fallback_rows: int


@dataclass(frozen=True, slots=True)
class _BlockDesign:
    metadata: EffectBlock
    matrix: csr_matrix
    active_mask: FloatArray


@dataclass(frozen=True, slots=True)
class _ModelDesign:
    fixed_matrix: FloatArray
    fixed_names: tuple[str, ...]
    fixed_indices: tuple[int, ...]
    fixed_prior_sds: FloatArray
    blocks: tuple[_BlockDesign, ...]
    scale_indices: dict[str, int]
    log_kappa_index: int
    parameter_names: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _utc_datetime(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _cutoff(value: datetime) -> datetime:
    return _utc_datetime(value, field="cutoff")


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_available_at(value: Any) -> datetime | None:
    if _missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise ModelDataError(f"available_at_utc is not a datetime: {value!r}")
    return _utc_datetime(value, field="available_at_utc")


def _as_observation_datetime(row: pd.Series) -> tuple[datetime | None, bool]:
    for column, fallback in (("match_date", False), ("source_date", True)):
        value = row.get(column)
        if _missing(value):
            continue
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return _utc_datetime(value, field=column) if value.tzinfo else value.replace(
                tzinfo=UTC
            ), fallback
        if isinstance(value, date):
            return datetime.combine(value, time.min, tzinfo=UTC), fallback
        try:
            parsed = pd.Timestamp(cast(Any, value))
        except (TypeError, ValueError) as exc:
            raise ModelDataError(f"{column} is not a date: {value!r}") from exc
        if pd.isna(parsed):
            continue
        py_value = parsed.to_pydatetime()
        return py_value.replace(tzinfo=UTC) if py_value.tzinfo is None else py_value.astimezone(
            UTC
        ), fallback
    return None, False


def _component_value(value: ServeComponent | str) -> ServeComponent:
    try:
        return value if isinstance(value, ServeComponent) else ServeComponent(str(value).upper())
    except ValueError as exc:
        raise ModelDataError(f"unknown primitive serve component: {value!r}") from exc


def _tour_value(value: Tour | str) -> Tour:
    try:
        return value if isinstance(value, Tour) else Tour(str(value).upper())
    except ValueError as exc:
        raise ModelDataError(f"unknown tour: {value!r}") from exc


def _surface(value: Any) -> str:
    if _missing(value):
        return _MISSING_SURFACE
    text = str(value).strip()
    return text.title() if text else _MISSING_SURFACE


def _event_year(row: pd.Series) -> str:
    event = row.get("event")
    year = row.get("event_year")
    if _missing(event) or _missing(year):
        return _MISSING_EVENT_YEAR
    return f"{str(event).strip()}|{int(cast(Any, year))}"


def _exact_count(value: Any, *, field: str) -> int:
    if _missing(value) or isinstance(value, (bool, np.bool_)):
        raise ModelDataError(f"eligible {field} must be an exact integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelDataError(f"eligible {field} must be an exact integer") from exc
    if not isfinite(numeric) or numeric != float(int(numeric)):
        raise ModelDataError(f"eligible {field} must be an exact integer")
    return int(numeric)


def _indoor_indicator(value: Any) -> float | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "indoor", "i"}:
        return 1.0
    if text in {"0", "false", "no", "n", "outdoor", "o"}:
        return 0.0
    raise ModelDataError(f"unrecognized indoor/outdoor token: {value!r}")


def _stable_training_hash(frame: pd.DataFrame) -> str:
    records: list[dict[str, Any]] = []
    columns = (
        "snapshot_id",
        "match_id",
        "player_id",
        "opponent_id",
        "component",
        "successes",
        "trials",
        "available_at_utc",
        "_observation_at_utc",
        "_surface",
        "_event_year",
        "_indoor_hard",
        "_weight",
    )
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for column in columns:
            value = row.get(column)
            if _missing(value):
                record[column] = None
            elif isinstance(value, (datetime, pd.Timestamp)):
                parsed = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
                record[column] = parsed.astimezone(UTC).isoformat()
            elif isinstance(value, np.generic):
                record[column] = value.item()
            else:
                record[column] = value
        records.append(record)
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def prepare_component_rows(
    counts: ComponentCountTable | pd.DataFrame,
    *,
    component: ServeComponent | str,
    tour: Tour | str,
    cutoff: datetime,
    config: ServeModelConfig,
) -> PreparedComponentRows:
    """Select one primitive likelihood under strict tour, cutoff, and status gates."""

    selected_component = _component_value(component)
    selected_tour = _tour_value(tour)
    cutoff_utc = _cutoff(cutoff)
    frame = (
        counts.counts.copy(deep=True)
        if isinstance(counts, ComponentCountTable)
        else counts.copy(deep=True)
    )
    required = {
        "match_id",
        "tour",
        "component",
        "player_id",
        "opponent_id",
        "successes",
        "trials",
        "status",
        "eligible_for_likelihood",
        "available_at_utc",
        "snapshot_sha256",
        "transformation_version",
        "surface",
        "indoor",
        "event",
        "event_year",
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ModelDataError(
            "component counts lack Milestone 3 columns: " + ", ".join(sorted(missing_columns))
        )
    frame = frame.loc[
        (frame["component"].astype("string") == selected_component.value)
        & (frame["tour"].astype("string").str.upper() == selected_tour.value)
    ].copy()
    input_rows = len(frame)
    if frame.empty:
        raise ModelDataError(
            f"no {selected_tour.value} rows exist for component {selected_component.value}"
        )

    observed_versions = set(frame["transformation_version"].dropna().astype(str))
    if observed_versions != {TRANSFORMATION_VERSION}:
        raise ModelDataError(
            "component rows must use the supported Milestone 1 transformation version "
            f"{TRANSFORMATION_VERSION}"
        )

    duplicate_key = ["match_id", "player_id", "component"]
    if frame.duplicated(duplicate_key, keep=False).any():
        raise ModelDataError("component counts contain duplicate player-match likelihood rows")

    excluded: Counter[str] = Counter()
    usable_indices: list[Any] = []
    observation_times: dict[Any, datetime] = {}
    weights: dict[Any, float] = {}
    surfaces: dict[Any, str] = {}
    events: dict[Any, str] = {}
    indoor_values: dict[Any, float] = {}
    date_fallback_rows = 0
    status_strings = frame["status"].astype("string")
    missing_rows = int((status_strings == ComponentStatus.MISSING_INPUT.value).sum())
    quarantined_rows = int((status_strings == ComponentStatus.QUARANTINED.value).sum())
    anomaly_count = quarantined_rows
    if isinstance(counts, ComponentCountTable) and not counts.anomalies.empty:
        anomalies = counts.anomalies
        if {"component", "tour"}.issubset(anomalies.columns):
            anomaly_count = len(
                anomalies.loc[
                    (anomalies["component"].astype("string") == selected_component.value)
                    & (anomalies["tour"].astype("string").str.upper() == selected_tour.value)
                ]
            )

    for index, row in frame.iterrows():
        status = str(row["status"])
        eligible = row["eligible_for_likelihood"]
        if _missing(eligible):
            raise ModelDataError("eligible_for_likelihood must never be missing")
        if not isinstance(eligible, (bool, np.bool_)):
            raise ModelDataError("eligible_for_likelihood must be an exact boolean")
        eligible_bool = bool(eligible)
        if eligible_bool != (status == ComponentStatus.VALID.value):
            raise ModelDataError("component status and likelihood eligibility disagree")
        if not eligible_bool:
            excluded[status] += 1
            continue

        available_at = _as_available_at(row["available_at_utc"])
        if available_at is None:
            excluded["unknown_availability"] += 1
            continue
        if available_at >= cutoff_utc:
            excluded["at_or_after_cutoff"] += 1
            continue

        observation_at, used_fallback = _as_observation_datetime(row)
        if observation_at is None:
            excluded["missing_observation_date"] += 1
            continue
        age_days = (cutoff_utc - observation_at).total_seconds() / 86400.0
        if age_days < 0.0:
            raise ModelDataError("an observation date lies after the information cutoff")
        weight = time_weight(age_days)
        if weight == 0.0:
            excluded["outside_1095_day_window"] += 1
            continue

        successes = _exact_count(row["successes"], field="successes")
        trials = _exact_count(row["trials"], field="trials")
        if successes < 0 or trials <= 0 or successes > trials:
            raise ModelDataError("eligible component counts violate 0 <= successes <= trials")

        surface = _surface(row["surface"])
        if surface == _MISSING_SURFACE:
            excluded["missing_surface_context"] += 1
            continue
        indoor = _indoor_indicator(row["indoor"])
        if config.context.include_indoor_hard and surface == "Hard" and indoor is None:
            excluded["missing_indoor_hard_context"] += 1
            continue
        usable_indices.append(index)
        observation_times[index] = observation_at
        weights[index] = weight
        surfaces[index] = surface
        events[index] = _event_year(row)
        indoor_values[index] = 0.0 if indoor is None or surface != "Hard" else indoor
        date_fallback_rows += int(used_fallback)

    usable = frame.loc[usable_indices].copy()
    if usable.empty:
        raise ModelDataError(
            f"no cutoff-safe positive-denominator rows remain for {selected_component.value}"
        )
    usable["_observation_at_utc"] = pd.Series(observation_times).loc[usable.index]
    usable["_weight"] = pd.Series(weights).loc[usable.index].astype(float)
    usable["_surface"] = pd.Series(surfaces).loc[usable.index].astype("string")
    usable["_event_year"] = pd.Series(events).loc[usable.index].astype("string")
    usable["_indoor_hard"] = pd.Series(indoor_values).loc[usable.index].astype(float)
    usable = usable.sort_values(
        ["_observation_at_utc", "match_id", "player_id", "opponent_id"],
        kind="stable",
    ).reset_index(drop=True)
    y = np.asarray(
        usable["successes"].astype("int64").to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    n = np.asarray(
        usable["trials"].astype("int64").to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    omega = usable["_weight"].to_numpy(dtype=np.float64)
    return PreparedComponentRows(
        frame=usable,
        successes=y,
        trials=n,
        weights=omega,
        data_sha256=_stable_training_hash(usable),
        input_rows=input_rows,
        excluded=excluded,
        missing_rows=missing_rows,
        quarantined_rows=quarantined_rows,
        quarantined_anomaly_count=anomaly_count,
        date_fallback_rows=date_fallback_rows,
    )


def _reference_matrix(row_levels: Sequence[str], levels: tuple[str, ...]) -> csr_matrix:
    free_levels = levels[:-1]
    lookup = {level: index for index, level in enumerate(free_levels)}
    rows: list[int] = []
    columns: list[int] = []
    for row_index, level in enumerate(row_levels):
        column = lookup.get(str(level))
        if column is not None:
            rows.append(row_index)
            columns.append(column)
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, columns)), shape=(len(row_levels), len(free_levels)))


def _active_mask(row_levels: Sequence[str], levels: tuple[str, ...]) -> FloatArray:
    known = frozenset(levels)
    return np.asarray([float(str(level) in known) for level in row_levels], dtype=np.float64)


def _build_design(
    prepared: PreparedComponentRows,
    component: ServeComponent,
    config: ServeModelConfig,
) -> _ModelDesign:
    frame = prepared.frame
    fixed_names = ["intercept"]
    surfaces = tuple(
        sorted(
            surface for surface in set(frame["_surface"].astype(str)) if surface != _MISSING_SURFACE
        )
    )
    for surface in surfaces:
        if surface != "Hard":
            fixed_names.append(f"surface_fixed:{surface}")
    if config.context.include_indoor_hard:
        fixed_names.append("indoor_hard")
    fixed_matrix = np.zeros((len(frame), len(fixed_names)), dtype=np.float64)
    fixed_matrix[:, 0] = 1.0
    for column, name in enumerate(fixed_names[1:], start=1):
        if name == "indoor_hard":
            fixed_matrix[:, column] = frame["_indoor_hard"].to_numpy(dtype=np.float64)
        else:
            surface = name.split(":", 1)[1]
            fixed_matrix[:, column] = (frame["_surface"].astype(str) == surface).astype(float)

    parameter_names = list(fixed_names)
    fixed_indices = tuple(range(len(fixed_names)))
    fixed_prior_sds = np.asarray(
        [config.prior.intercept_sd]
        + [config.prior.context_coefficient_sd] * (len(fixed_names) - 1),
        dtype=np.float64,
    )
    pending: list[
        tuple[str, EffectRole, ScaleGroup, str | None, tuple[str, ...], csr_matrix, FloatArray]
    ] = []

    server_levels = tuple(sorted(set(frame["player_id"].astype(str))))
    pending.append(
        (
            "server_global",
            "server_global",
            "server_global",
            None,
            server_levels,
            _reference_matrix(frame["player_id"].astype(str).tolist(), server_levels),
            _active_mask(frame["player_id"].astype(str).tolist(), server_levels),
        )
    )
    if component in _OPPONENT_COMPONENTS:
        returner_levels = tuple(sorted(set(frame["opponent_id"].astype(str))))
        pending.append(
            (
                "returner_global",
                "returner_global",
                "returner_global",
                None,
                returner_levels,
                _reference_matrix(frame["opponent_id"].astype(str).tolist(), returner_levels),
                _active_mask(frame["opponent_id"].astype(str).tolist(), returner_levels),
            )
        )

    for surface in surfaces:
        mask = frame["_surface"].astype(str) == surface
        server_surface_levels = tuple(sorted(set(frame.loc[mask, "player_id"].astype(str))))
        server_row_levels = [
            str(player) if bool(on_surface) else "__not_in_block__"
            for player, on_surface in zip(frame["player_id"], mask, strict=True)
        ]
        pending.append(
            (
                f"server_surface:{surface}",
                "server_surface",
                "surface",
                surface,
                server_surface_levels,
                _reference_matrix(server_row_levels, server_surface_levels),
                _active_mask(server_row_levels, server_surface_levels),
            )
        )
        if component in _OPPONENT_COMPONENTS:
            return_surface_levels = tuple(sorted(set(frame.loc[mask, "opponent_id"].astype(str))))
            return_row_levels = [
                str(player) if bool(on_surface) else "__not_in_block__"
                for player, on_surface in zip(frame["opponent_id"], mask, strict=True)
            ]
            pending.append(
                (
                    f"returner_surface:{surface}",
                    "returner_surface",
                    "surface",
                    surface,
                    return_surface_levels,
                    _reference_matrix(return_row_levels, return_surface_levels),
                    _active_mask(return_row_levels, return_surface_levels),
                )
            )

    if component in config.context.event_year_effect_components:
        event_rows = frame["_event_year"].astype(str).tolist()
        event_levels = tuple(
            sorted(level for level in set(event_rows) if level != _MISSING_EVENT_YEAR)
        )
        if event_levels:
            pending.append(
                (
                    "event_year",
                    "event_year",
                    "event_year",
                    None,
                    event_levels,
                    _reference_matrix(event_rows, event_levels),
                    _active_mask(event_rows, event_levels),
                )
            )

    block_parameter_indices: list[tuple[int, ...]] = []
    for name, _role, _scale_group, _surface_name, levels, _matrix, _active in pending:
        indices = tuple(range(len(parameter_names), len(parameter_names) + max(0, len(levels) - 1)))
        block_parameter_indices.append(indices)
        for level in levels[:-1]:
            parameter_names.append(f"z:{name}[{level}]")

    scale_groups = tuple(dict.fromkeys(item[2] for item in pending))
    scale_indices: dict[str, int] = {}
    for group in scale_groups:
        scale_indices[group] = len(parameter_names)
        parameter_names.append(f"log_scale:{group}")
    log_kappa_index = len(parameter_names)
    parameter_names.append("log_kappa")

    blocks: list[_BlockDesign] = []
    for pending_item, indices in zip(pending, block_parameter_indices, strict=True):
        name, role, scale_group, surface_name, levels, matrix, active_mask = pending_item
        metadata = EffectBlock(
            name=name,
            role=role,
            scale_group=scale_group,
            surface=surface_name,
            levels=levels,
            free_parameter_indices=indices,
            scale_parameter_index=scale_indices[scale_group],
        )
        blocks.append(_BlockDesign(metadata=metadata, matrix=matrix, active_mask=active_mask))
    return _ModelDesign(
        fixed_matrix=fixed_matrix,
        fixed_names=tuple(fixed_names),
        fixed_indices=fixed_indices,
        fixed_prior_sds=fixed_prior_sds,
        blocks=tuple(blocks),
        scale_indices=scale_indices,
        log_kappa_index=log_kappa_index,
        parameter_names=tuple(parameter_names),
    )


def _centered_z(theta: FloatArray, block: EffectBlock) -> tuple[FloatArray, float]:
    free = np.asarray([theta[index] for index in block.free_parameter_indices], dtype=np.float64)
    full = np.concatenate((free, np.zeros(1, dtype=np.float64)))
    mean = float(np.mean(full))
    return full - mean, mean


def _block_sign(block: EffectBlock) -> float:
    return -1.0 if block.role in {"returner_global", "returner_surface"} else 1.0


def _eta(theta: FloatArray, design: _ModelDesign) -> FloatArray:
    eta = design.fixed_matrix @ theta[np.asarray(design.fixed_indices, dtype=np.int64)]
    for block in design.blocks:
        if not block.metadata.free_parameter_indices:
            continue
        sigma = exp(float(theta[block.metadata.scale_parameter_index]))
        _centered, mean = _centered_z(theta, block.metadata)
        free = theta[np.asarray(block.metadata.free_parameter_indices, dtype=np.int64)]
        contribution = np.asarray(block.matrix @ free, dtype=np.float64) - (
            block.active_mask * mean
        )
        eta = eta + _block_sign(block.metadata) * sigma * np.asarray(
            contribution,
            dtype=np.float64,
        )
    return np.asarray(eta, dtype=np.float64)


def _objective_and_gradient(
    theta: FloatArray,
    prepared: PreparedComponentRows,
    design: _ModelDesign,
    config: ServeModelConfig,
) -> tuple[float, FloatArray]:
    linear = _eta(theta, design)
    kappa = exp(float(theta[design.log_kappa_index]))
    logpmf, score_eta, score_log_kappa = beta_binomial_scores(
        prepared.successes,
        prepared.trials,
        linear,
        kappa,
    )
    weighted_score_eta = prepared.weights * score_eta
    objective = -float(np.dot(prepared.weights, logpmf))
    gradient = np.zeros_like(theta)

    fixed_values = theta[np.asarray(design.fixed_indices, dtype=np.int64)]
    objective += 0.5 * float(np.sum((fixed_values / design.fixed_prior_sds) ** 2))
    fixed_gradient = -np.asarray(design.fixed_matrix.T @ weighted_score_eta, dtype=np.float64)
    fixed_gradient += fixed_values / (design.fixed_prior_sds**2)
    gradient[np.asarray(design.fixed_indices, dtype=np.int64)] = fixed_gradient

    scale_likelihood_gradients = {group: 0.0 for group in design.scale_indices}
    for block in design.blocks:
        metadata = block.metadata
        centered, mean = _centered_z(theta, metadata)
        objective += 0.5 * float(np.dot(centered, centered))
        if metadata.free_parameter_indices:
            indices = np.asarray(metadata.free_parameter_indices, dtype=np.int64)
            sigma = exp(float(theta[metadata.scale_parameter_index]))
            free = theta[indices]
            contribution = np.asarray(block.matrix @ free, dtype=np.float64) - (
                block.active_mask * mean
            )
            sign = _block_sign(metadata)
            likelihood_block_score = np.asarray(
                block.matrix.T @ weighted_score_eta,
                dtype=np.float64,
            ) - float(np.dot(block.active_mask, weighted_score_eta)) / len(metadata.levels)
            likelihood_gradient = -sign * sigma * likelihood_block_score
            gradient[indices] += likelihood_gradient + centered[:-1]
            scale_likelihood_gradients[metadata.scale_group] -= (
                sign * sigma * float(np.dot(contribution, weighted_score_eta))
            )

    for group, index in design.scale_indices.items():
        log_scale = float(theta[index])
        standardized = (
            log_scale - config.prior.log_shrinkage_scale_mean
        ) / config.prior.log_shrinkage_scale_sd
        objective += 0.5 * standardized**2
        gradient[index] = scale_likelihood_gradients[group] + standardized / (
            config.prior.log_shrinkage_scale_sd
        )

    gradient[design.log_kappa_index] = -float(np.dot(prepared.weights, score_log_kappa))
    if not isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("nonfinite weighted log-posterior or gradient")
    return objective, gradient


def _initial_parameters(
    prepared: PreparedComponentRows,
    design: _ModelDesign,
    config: ServeModelConfig,
) -> FloatArray:
    theta = np.zeros(len(design.parameter_names), dtype=np.float64)
    pooled = float(
        np.dot(prepared.weights, prepared.successes) / np.dot(prepared.weights, prepared.trials)
    )
    pooled = min(max(pooled, 1e-6), 1.0 - 1e-6)
    theta[design.fixed_indices[0]] = float(logit(pooled))
    for index in design.scale_indices.values():
        theta[index] = log(config.optimizer.initial_shrinkage_scale)
    theta[design.log_kappa_index] = log(config.optimizer.initial_kappa)
    return theta


def _bounds(
    design: _ModelDesign, config: ServeModelConfig
) -> list[tuple[float | None, float | None]]:
    result: list[tuple[float | None, float | None]] = [(None, None) for _ in design.parameter_names]
    scale_bounds = (
        log(config.optimizer.shrinkage_scale_lower),
        log(config.optimizer.shrinkage_scale_upper),
    )
    for index in design.scale_indices.values():
        result[index] = scale_bounds
    result[design.log_kappa_index] = (
        log(config.optimizer.kappa_lower),
        log(config.optimizer.kappa_upper),
    )
    return result


def _minimize_map(
    objective: Any,
    initial: FloatArray,
    bounds: list[tuple[float | None, float | None]],
    config: ServeModelConfig,
) -> OptimizeResult:
    """Run frozen L-BFGS-B, with one identical warm-start continuation at its cap.

    The continuation changes neither the objective nor any configured numerical
    setting.  It handles the operational case where a full MAP update reaches
    the per-call iteration cap immediately before SciPy's convergence test.
    """

    options = {
        "maxiter": config.optimizer.max_iterations,
        "maxls": config.optimizer.max_line_search_steps,
        "gtol": config.optimizer.gradient_tolerance,
        "ftol": config.optimizer.function_tolerance,
    }
    result = minimize(
        objective,
        initial,
        method=config.optimizer.method,
        jac=True,
        bounds=bounds,
        options=options,
    )
    capped = (
        not bool(result.success)
        and int(result.status) == 1
        and int(getattr(result, "nit", 0)) >= config.optimizer.max_iterations
        and config.optimizer.max_iterations > 1
    )
    if not capped:
        return result
    continued = minimize(
        objective,
        np.asarray(result.x, dtype=np.float64),
        method=config.optimizer.method,
        jac=True,
        bounds=bounds,
        options=options,
    )
    continued.nit = int(getattr(result, "nit", 0)) + int(getattr(continued, "nit", 0))
    continued.nfev = int(getattr(result, "nfev", 0)) + int(getattr(continued, "nfev", 0))
    continued.message = "DETERMINISTIC_WARM_START_CONTINUATION: " + str(continued.message)
    return continued


def _curvature_model(
    result: CurvatureResult, parameter_names: tuple[str, ...], theta: FloatArray
) -> PosteriorApproximation:
    hessian = (
        None
        if result.hessian is None
        else tuple(tuple(float(value) for value in row) for row in result.hessian)
    )
    covariance = (
        None
        if result.covariance is None
        else tuple(tuple(float(value) for value in row) for row in result.covariance)
    )
    return PosteriorApproximation(
        parameter_names=parameter_names,
        map_parameters=tuple(float(value) for value in theta),
        curvature_kind=cast(Literal["full", "diagonal"], result.kind),
        hessian=hessian,
        covariance=covariance,
        variance_diagonal=tuple(float(value) for value in result.variance_diagonal),
        raw_min_eigenvalue=result.raw_min_eigenvalue,
        regularization_added=result.regularization_added,
        condition_number=result.condition_number,
    )


def _block_effects(theta: FloatArray, block: EffectBlock) -> list[CoefficientEstimate]:
    centered_z, _mean = _centered_z(theta, block)
    sigma = _positive_exp(
        float(theta[block.scale_parameter_index]),
        field=f"effect scale {block.scale_group}",
    )
    return [
        CoefficientEstimate(
            name=f"{block.name}[{level}]",
            kind=block.role,
            value=float(sigma * centered_z[index]),
        )
        for index, level in enumerate(block.levels)
    ]


def _player_information(
    prepared: PreparedComponentRows,
    kappa: float,
    warning_threshold: float,
) -> tuple[PlayerInformation, ...]:
    frame = prepared.frame
    rho = 1.0 / (kappa + 1.0)
    result: list[PlayerInformation] = []
    for player_id, group in frame.groupby("player_id", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        weighted_opportunities = prepared.weights[indices] * prepared.trials[indices]
        weighted_trials = float(np.sum(weighted_opportunities))
        denominator = float(np.sum(weighted_opportunities**2))
        effective_matches = 0.0 if denominator == 0.0 else weighted_trials**2 / denominator
        information = float(
            np.sum(
                prepared.weights[indices]
                * prepared.trials[indices]
                / (1.0 + (prepared.trials[indices] - 1.0) * rho)
            )
        )
        result.append(
            PlayerInformation(
                player_id=str(player_id),
                weighted_trials=weighted_trials,
                effective_matches=effective_matches,
                information_equivalent_trials=information,
                sparse_warning=weighted_trials < warning_threshold,
            )
        )
    return tuple(result)


def _diagnostics(
    prepared: PreparedComponentRows,
    design: _ModelDesign,
    theta: FloatArray,
    optimize_result: OptimizeResult,
    component: ServeComponent,
    config: ServeModelConfig,
) -> FitDiagnostics:
    linear = _eta(theta, design)
    fitted = np.asarray(expit(linear), dtype=np.float64)
    kappa = exp(float(theta[design.log_kappa_index]))
    weighted_trials = float(np.dot(prepared.weights, prepared.trials))
    observed_rate = float(np.dot(prepared.weights, prepared.successes) / weighted_trials)
    fitted_rate = float(np.dot(prepared.weights * prepared.trials, fitted) / weighted_trials)
    row_rates = prepared.successes / prepared.trials
    rmse = float(
        np.sqrt(np.average((row_rates - fitted) ** 2, weights=prepared.weights * prepared.trials))
    )
    jacobian = np.asarray(optimize_result.jac, dtype=np.float64)
    scales = {group: exp(float(theta[index])) for group, index in design.scale_indices.items()}
    warning_values: list[str] = []
    tolerance = 1e-5
    if abs(kappa - config.optimizer.kappa_lower) <= tolerance * config.optimizer.kappa_lower:
        warning_values.append("kappa_at_lower_bound")
    if abs(kappa - config.optimizer.kappa_upper) <= tolerance * config.optimizer.kappa_upper:
        warning_values.append("kappa_at_upper_bound")
    for group, scale in scales.items():
        if abs(scale - config.optimizer.shrinkage_scale_lower) <= tolerance * max(
            1.0, config.optimizer.shrinkage_scale_lower
        ):
            warning_values.append(f"{group}_scale_at_lower_bound")
        if abs(scale - config.optimizer.shrinkage_scale_upper) <= tolerance * max(
            1.0, config.optimizer.shrinkage_scale_upper
        ):
            warning_values.append(f"{group}_scale_at_upper_bound")
    player_information = _player_information(
        prepared,
        kappa,
        config.diagnostics.sparse_weighted_trials_warning,
    )
    if any(item.sparse_warning for item in player_information):
        warning_values.append("sparse_player_exposure")
    return FitDiagnostics(
        usable_rows=len(prepared.frame),
        raw_trials=int(np.sum(prepared.trials)),
        weighted_trials=weighted_trials,
        unique_servers=int(prepared.frame["player_id"].nunique()),
        unique_returners=(
            int(prepared.frame["opponent_id"].nunique()) if component in _OPPONENT_COMPONENTS else 0
        ),
        excluded_rows=prepared.input_rows - len(prepared.frame),
        missing_rows=prepared.missing_rows,
        quarantined_rows=prepared.quarantined_rows,
        quarantined_anomaly_count=prepared.quarantined_anomaly_count,
        date_fallback_rows=prepared.date_fallback_rows,
        exclusion_counts=tuple(
            ExclusionCount(reason=reason, rows=count)
            for reason, count in sorted(prepared.excluded.items())
        ),
        kappa=kappa,
        shrinkage_scale_estimates=tuple(
            NamedScale(name=name, value=value) for name, value in sorted(scales.items())
        ),
        converged=bool(optimize_result.success),
        objective_value=float(optimize_result.fun),
        iterations=int(getattr(optimize_result, "nit", 0)),
        function_evaluations=int(getattr(optimize_result, "nfev", 0)),
        gradient_max_abs=float(np.max(np.abs(jacobian))),
        optimizer_status=int(optimize_result.status),
        optimizer_message=str(optimize_result.message),
        observed_rate=observed_rate,
        fitted_rate=fitted_rate,
        weighted_row_rmse=rmse,
        player_information=player_information,
        warnings=tuple(warning_values),
    )


def fit_serve_component(
    counts: ComponentCountTable | pd.DataFrame,
    *,
    component: ServeComponent | str,
    tour: Tour | str,
    cutoff: datetime,
    config: ServeModelConfig,
    provenance: FitProvenance,
) -> FittedServeComponent:
    """Fit one frozen primitive component by weighted MAP and Laplace approximation."""

    selected_component = _component_value(component)
    selected_tour = _tour_value(tour)
    cutoff_utc = _cutoff(cutoff)
    prepared = prepare_component_rows(
        counts,
        component=selected_component,
        tour=selected_tour,
        cutoff=cutoff_utc,
        config=config,
    )
    _validate_prepared_provenance(prepared, provenance)
    design = _build_design(prepared, selected_component, config)
    initial = _initial_parameters(prepared, design, config)

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        return _objective_and_gradient(theta, prepared, design, config)

    result = _minimize_map(objective, initial, _bounds(design, config), config)
    if not bool(result.success):
        raise FitConvergenceError(selected_component, result)
    theta = np.asarray(result.x, dtype=np.float64)

    def gradient(point: FloatArray) -> FloatArray:
        return _objective_and_gradient(point, prepared, design, config)[1]

    curvature = laplace_curvature(
        gradient,
        theta,
        max_full_parameters=config.optimizer.laplace_max_full_parameters,
        relative_step=config.optimizer.laplace_relative_step,
        eigenvalue_floor=config.optimizer.laplace_eigenvalue_floor,
    )
    posterior = _curvature_model(curvature, design.parameter_names, theta)
    diagnostics = _diagnostics(
        prepared,
        design,
        theta,
        result,
        selected_component,
        config,
    )
    coefficients: list[CoefficientEstimate] = []
    for name, index in zip(design.fixed_names, design.fixed_indices, strict=True):
        coefficients.append(CoefficientEstimate(name=name, kind="fixed", value=float(theta[index])))
    for block in design.blocks:
        coefficients.extend(_block_effects(theta, block.metadata))
    for group, index in design.scale_indices.items():
        coefficients.append(
            CoefficientEstimate(
                name=f"shrinkage_scale:{group}",
                kind="scale",
                value=exp(float(theta[index])),
            )
        )
    coefficients.append(
        CoefficientEstimate(name="kappa", kind="concentration", value=diagnostics.kappa)
    )
    return FittedServeComponent(
        implementation_version=MODEL_IMPLEMENTATION_VERSION,
        framework_version=FRAMEWORK_VERSION,
        tour=selected_tour,
        component=selected_component,
        data_cutoff_utc=cutoff_utc,
        training_window_start_utc=cutoff_utc - timedelta(days=FROZEN_WINDOW_DAYS),
        fitted_at_utc=provenance.fitted_at_utc,
        data_snapshot_sha256=provenance.data_snapshot_sha256,
        component_count_artifact_sha256=provenance.component_count_artifact_sha256,
        training_data_sha256=prepared.data_sha256,
        model_config_sha256=config.sha256,
        code_commit=provenance.code_commit,
        config=config,
        fixed_parameters=tuple(
            NamedIndex(name=name, index=index)
            for name, index in zip(design.fixed_names, design.fixed_indices, strict=True)
        ),
        effect_blocks=tuple(block.metadata for block in design.blocks),
        posterior=posterior,
        diagnostics=diagnostics,
        coefficient_summary=tuple(coefficients),
    )


def fit_all_serve_components(
    counts: ComponentCountTable | pd.DataFrame,
    *,
    tour: Tour | str,
    cutoff: datetime,
    config: ServeModelConfig,
    provenance: FitProvenance,
) -> dict[ServeComponent, FittedServeComponent]:
    """Fit F/A/Q1/D/Q2 independently for exactly one tour."""

    return {
        component: fit_serve_component(
            counts,
            component=component,
            tour=tour,
            cutoff=cutoff,
            config=config,
            provenance=provenance,
        )
        for component in ServeComponent
    }


def _validate_prepared_provenance(
    prepared: PreparedComponentRows,
    provenance: FitProvenance,
) -> None:
    snapshot_values = prepared.frame["snapshot_sha256"]
    if snapshot_values.map(_missing).any():
        raise ModelDataError("usable component rows must retain their source snapshot hash")
    try:
        snapshots = {_normalize_sha256(str(value)) for value in snapshot_values}
    except ValueError as exc:
        raise ModelDataError("component rows contain an invalid source snapshot hash") from exc
    if fit_input_set_sha256("source_snapshots", snapshots) != provenance.data_snapshot_sha256:
        raise ModelDataError("fit provenance does not match the component-row snapshot hash")
    artifact_column = "component_count_artifact_sha256"
    if artifact_column in prepared.frame:
        artifact_values = prepared.frame[artifact_column]
        if artifact_values.map(_missing).any():
            raise ModelDataError("component-count artifact identity is partially missing")
        try:
            artifacts = {_normalize_sha256(str(value)) for value in artifact_values}
        except ValueError as exc:
            raise ModelDataError("component rows contain an invalid count-artifact hash") from exc
        if (
            fit_input_set_sha256("component_count_artifacts", artifacts)
            != provenance.component_count_artifact_sha256
        ):
            raise ModelDataError("fit provenance does not match the component-count artifact hash")


def fit_input_set_sha256(kind: str, digests: Iterable[str]) -> str:
    """Hash an ordered-independent set of immutable fit inputs.

    A singleton retains its historical digest so existing fitted artifacts remain
    byte-compatible. Multiple yearly inputs receive a schema-tagged composite
    digest without overwriting the per-row source snapshot identities.
    """

    if not kind.strip():
        raise ValueError("fit input kind must not be empty")
    normalized = tuple(sorted({_normalize_sha256(str(value)) for value in digests}))
    if not normalized:
        raise ValueError("fit input set must not be empty")
    if len(normalized) == 1:
        return normalized[0]
    payload = {
        "schema_version": FIT_INPUT_SET_VERSION,
        "kind": kind.strip(),
        "sha256": normalized,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def fit_all_serve_components_from_bundle(
    bundle: ProcessedArtifactBundle,
    *,
    tour: Tour | str,
    cutoff: datetime,
    config: ServeModelConfig,
    code_commit: str,
    fitted_at_utc: datetime,
) -> dict[ServeComponent, FittedServeComponent]:
    """Fit all primitives from a verified Milestone 1 artifact with bound provenance."""

    verify_processed_bundle(bundle)
    selected_tour = _tour_value(tour)
    if bundle.manifest.source.tour is not selected_tour:
        raise ModelDataError("processed artifact tour does not match the requested fit tour")
    receipt = bundle.manifest.receipt_for("component_counts")
    counts = read_processed_table(bundle, "component_counts")
    counts["component_count_artifact_sha256"] = receipt.sha256
    provenance = FitProvenance(
        data_snapshot_sha256=bundle.manifest.snapshot_sha256,
        component_count_artifact_sha256=receipt.sha256,
        code_commit=code_commit,
        fitted_at_utc=fitted_at_utc,
    )
    return fit_all_serve_components(
        counts,
        tour=selected_tour,
        cutoff=cutoff,
        config=config,
        provenance=provenance,
    )


def fit_all_serve_components_from_bundles(
    bundles: Sequence[ProcessedArtifactBundle],
    *,
    tour: Tour | str,
    cutoff: datetime,
    config: ServeModelConfig,
    code_commit: str,
    fitted_at_utc: datetime,
) -> dict[ServeComponent, FittedServeComponent]:
    """Fit one tour from multiple verified, nonoverlapping source objects."""

    if not bundles:
        raise ModelDataError("a multi-source fit requires at least one processed bundle")
    selected_tour = _tour_value(tour)
    frames: list[pd.DataFrame] = []
    snapshot_hashes: list[str] = []
    artifact_hashes: list[str] = []
    for bundle in bundles:
        verify_processed_bundle(bundle)
        if bundle.manifest.source.tour is not selected_tour:
            raise ModelDataError("processed artifact tour does not match the requested fit tour")
        receipt = bundle.manifest.receipt_for("component_counts")
        counts = read_processed_table(bundle, "component_counts")
        counts["component_count_artifact_sha256"] = receipt.sha256
        frames.append(counts)
        snapshot_hashes.append(bundle.manifest.snapshot_sha256)
        artifact_hashes.append(receipt.sha256)
    combined = pd.concat(frames, ignore_index=True)
    identity_columns = ("match_id", "player_id", "opponent_id", "component")
    duplicated = combined.duplicated(list(identity_columns), keep=False)
    if bool(duplicated.any()):
        raise ModelDataError(
            "multi-source component bundles overlap on match/player/component identity"
        )
    provenance = FitProvenance(
        data_snapshot_sha256=fit_input_set_sha256("source_snapshots", snapshot_hashes),
        component_count_artifact_sha256=fit_input_set_sha256(
            "component_count_artifacts", artifact_hashes
        ),
        code_commit=code_commit,
        fitted_at_utc=fitted_at_utc,
    )
    return fit_all_serve_components(
        combined,
        tour=selected_tour,
        cutoff=cutoff,
        config=config,
        provenance=provenance,
    )


def _block_for(
    fitted: FittedServeComponent,
    role: str,
    *,
    surface: str | None = None,
) -> EffectBlock | None:
    for block in fitted.effect_blocks:
        if block.role == role and block.surface == surface:
            return block
    return None


def _block_raw_contribution_and_gradient(
    theta: FloatArray,
    block: EffectBlock,
    level: str,
) -> tuple[float, FloatArray, bool, float]:
    gradient = np.zeros_like(theta)
    sigma = _positive_exp(
        float(theta[block.scale_parameter_index]),
        field=f"effect scale {block.scale_group}",
    )
    free_indices = np.asarray(block.free_parameter_indices, dtype=np.int64)
    free = theta[free_indices] if len(free_indices) else np.empty(0, dtype=np.float64)
    full = np.concatenate((free, np.zeros(1, dtype=np.float64)))
    mean_z = float(np.mean(full))
    seen = level in block.levels
    if not seen:
        return 0.0, gradient, False, sigma**2
    raw_z = 0.0
    if level != block.levels[-1]:
        position = block.levels.index(level)
        raw_z = float(free[position])
        gradient[free_indices[position]] = sigma
    if len(free_indices):
        gradient[free_indices] -= sigma / len(block.levels)
    contribution = sigma * (raw_z - mean_z)
    gradient[block.scale_parameter_index] += contribution
    return contribution, gradient, seen, 0.0


def effect_block_player_contribution(
    block: EffectBlock,
    player_id: str,
    parameters: Sequence[float],
) -> tuple[float, bool]:
    """Evaluate one centered logical player coefficient at explicit parameters.

    This is the allocation-free path used by the C6 match sampler.  It keeps the
    original fitted reference coding internal while making the actual centered
    player coefficient explicit, including for the reference level.
    """

    try:
        theta = np.asarray(parameters, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelDataError("effect parameters must be a finite numeric vector") from exc
    if theta.ndim != 1 or not np.all(np.isfinite(theta)):
        raise ModelDataError("effect parameters must be a finite one-dimensional vector")
    if block.scale_parameter_index >= len(theta) or any(
        index >= len(theta) for index in block.free_parameter_indices
    ):
        raise ModelDataError("effect block indices exceed the parameter vector")
    sigma = _positive_exp(
        float(theta[block.scale_parameter_index]),
        field=f"effect scale {block.scale_group}",
    )
    if player_id not in block.levels:
        return 0.0, False
    free = theta[np.asarray(block.free_parameter_indices, dtype=np.int64)]
    mean_z = float(np.sum(free) / len(block.levels))
    raw_z = 0.0
    if player_id != block.levels[-1]:
        raw_z = float(free[block.levels.index(player_id)])
    return sigma * (raw_z - mean_z), True


def project_player_effect_coordinate(
    fitted: FittedServeComponent,
    *,
    player_id: str,
    role: EffectRole,
    surface: str | None = None,
) -> PlayerEffectCoordinateProjection:
    """Expose one canonical logical player coordinate at the stored MAP.

    The gradient is with respect to the complete stored Laplace vector, so a
    reference-level player is handled exactly and covariance with all raw
    coordinates remains auditable.
    """

    fitted = _revalidate_fit_for_prediction(fitted)
    block = _block_for(fitted, role, surface=surface)
    if block is None:
        target = role if surface is None else f"{role}:{surface}"
        raise ModelDataError(f"fitted component lacks required effect block {target}")
    theta = np.asarray(fitted.posterior.map_parameters, dtype=np.float64)
    contribution, gradient, seen, unseen_variance = _block_raw_contribution_and_gradient(
        theta,
        block,
        player_id,
    )
    surface_label = "global" if surface is None else surface
    return PlayerEffectCoordinateProjection(
        component=fitted.component,
        player_id=player_id,
        block_name=block.name,
        role=role,
        surface=surface,
        coordinate_id=(f"{fitted.component.value}:{role}:{surface_label}:player:{player_id}"),
        seen=seen,
        map_contribution=contribution,
        map_gradient=tuple(float(value) for value in gradient),
        unseen_standard_deviation=(None if seen else float(max(0.0, unseen_variance) ** 0.5)),
    )


def _positive_exp(value: float, *, field: str) -> float:
    try:
        result = exp(value)
    except OverflowError as exc:
        raise ModelDataError(f"{field} is not finite after exponentiation") from exc
    if not isfinite(result) or result <= 0.0:
        raise ModelDataError(f"{field} must be finite and positive")
    return result


def _unseen_requirement(
    theta: FloatArray,
    block: EffectBlock,
    level: str,
    *,
    coefficient: Literal[-1, 1],
    block_name: str | None = None,
) -> UnseenEffectRequirement:
    return UnseenEffectRequirement(
        block_name=block.name if block_name is None else block_name,
        level=level,
        standard_deviation=_positive_exp(
            float(theta[block.scale_parameter_index]),
            field=f"unseen-effect scale {block.scale_group}",
        ),
        coefficient=coefficient,
    )


def _first_block_with_role(
    fitted: FittedServeComponent,
    role: Literal["server_surface", "returner_surface"],
) -> EffectBlock:
    for block in fitted.effect_blocks:
        if block.role == role:
            return block
    raise ModelDataError(f"fitted component lacks required {role} scale")


@dataclass(frozen=True, slots=True)
class _LinearTerms:
    linear_predictor: float
    gradient: FloatArray
    unseen_effects: tuple[UnseenEffectRequirement, ...]
    serving_player_seen: bool
    returning_player_seen: bool | None
    surface_seen: bool
    event_year_seen: bool | None


def _linear_terms_at_parameters(
    fitted: FittedServeComponent,
    context: FutureMatchContext,
    theta: FloatArray,
) -> _LinearTerms:
    if fitted.tour is not context.tour:
        raise ModelDataError("prediction context tour does not match fitted tour")
    if context.information_cutoff_utc != fitted.data_cutoff_utc:
        raise ModelDataError("prediction cutoff must exactly match the fitted artifact cutoff")
    gradient = np.zeros_like(theta)
    unseen_effects: list[UnseenEffectRequirement] = []
    intercept_index = fitted.fixed_parameter_indices["intercept"]
    eta = float(theta[intercept_index])
    gradient[intercept_index] = 1.0
    surface = _surface(context.surface)
    surface_name = f"surface_fixed:{surface}"
    if surface_name in fitted.fixed_parameter_indices:
        index = fitted.fixed_parameter_indices[surface_name]
        eta += float(theta[index])
        gradient[index] = 1.0
    if fitted.config.context.include_indoor_hard and surface == "Hard":
        if context.indoor is None:
            raise ModelDataError("indoor/outdoor hard context is required by this fitted model")
        index = fitted.fixed_parameter_indices["indoor_hard"]
        indicator = float(context.indoor)
        eta += indicator * float(theta[index])
        gradient[index] = indicator

    server_global = _block_for(fitted, "server_global")
    if server_global is None:
        raise AssertionError("fitted component lacks server-global block")
    contribution, derivative, server_seen, new_variance = _block_raw_contribution_and_gradient(
        theta,
        server_global,
        context.serving_player_id,
    )
    eta += contribution
    gradient += derivative
    if new_variance > 0.0:
        unseen_effects.append(
            _unseen_requirement(
                theta,
                server_global,
                context.serving_player_id,
                coefficient=1,
            )
        )
    server_surface = _block_for(fitted, "server_surface", surface=surface)
    surface_seen = server_surface is not None
    if server_surface is not None:
        contribution, derivative, surface_server_seen, new_variance = (
            _block_raw_contribution_and_gradient(
                theta,
                server_surface,
                context.serving_player_id,
            )
        )
        eta += contribution
        gradient += derivative
        if not surface_server_seen:
            unseen_effects.append(
                _unseen_requirement(
                    theta,
                    server_surface,
                    context.serving_player_id,
                    coefficient=1,
                )
            )
    else:
        unseen_effects.append(
            _unseen_requirement(
                theta,
                _first_block_with_role(fitted, "server_surface"),
                context.serving_player_id,
                coefficient=1,
                block_name=f"server_surface:{surface}",
            )
        )

    returner_seen: bool | None = None
    if fitted.component in _OPPONENT_COMPONENTS:
        returner_global = _block_for(fitted, "returner_global")
        if returner_global is None:
            raise AssertionError("opponent-adjusted component lacks returner block")
        contribution, derivative, returner_seen, new_variance = (
            _block_raw_contribution_and_gradient(
                theta,
                returner_global,
                context.returning_player_id,
            )
        )
        eta -= contribution
        gradient -= derivative
        if new_variance > 0.0:
            unseen_effects.append(
                _unseen_requirement(
                    theta,
                    returner_global,
                    context.returning_player_id,
                    coefficient=-1,
                )
            )
        returner_surface = _block_for(fitted, "returner_surface", surface=surface)
        if returner_surface is not None:
            contribution, derivative, surface_returner_seen, new_variance = (
                _block_raw_contribution_and_gradient(
                    theta,
                    returner_surface,
                    context.returning_player_id,
                )
            )
            eta -= contribution
            gradient -= derivative
            if not surface_returner_seen:
                unseen_effects.append(
                    _unseen_requirement(
                        theta,
                        returner_surface,
                        context.returning_player_id,
                        coefficient=-1,
                    )
                )
        else:
            unseen_effects.append(
                _unseen_requirement(
                    theta,
                    _first_block_with_role(fitted, "returner_surface"),
                    context.returning_player_id,
                    coefficient=-1,
                    block_name=f"returner_surface:{surface}",
                )
            )

    event_seen: bool | None = None
    event_block = _block_for(fitted, "event_year")
    if event_block is not None:
        if context.event is None or context.event_year is None:
            event_seen = None
        else:
            event_level = f"{context.event.strip()}|{context.event_year}"
            contribution, derivative, event_seen, new_variance = (
                _block_raw_contribution_and_gradient(
                    theta,
                    event_block,
                    event_level,
                )
            )
            eta += contribution
            gradient += derivative
            if new_variance > 0.0:
                unseen_effects.append(
                    _unseen_requirement(
                        theta,
                        event_block,
                        event_level,
                        coefficient=1,
                    )
                )

    if not isfinite(eta):
        raise ModelDataError("component linear predictor must be finite")
    return _LinearTerms(
        linear_predictor=eta,
        gradient=gradient,
        unseen_effects=tuple(unseen_effects),
        serving_player_seen=server_seen,
        returning_player_seen=returner_seen,
        surface_seen=surface_seen,
        event_year_seen=event_seen,
    )


def project_component_parameters(
    fitted: FittedServeComponent,
    context: FutureMatchContext,
    parameters: Sequence[float],
) -> ComponentParameterProjection:
    """Evaluate one component at an explicit, stably indexed parameter vector.

    Unseen player/surface/event effects remain zero-centered in the returned base
    predictor and are represented as named Gaussian requirements.  A match-level
    sampler must realize those requirements once and share matching keys across
    both serving directions.
    """

    fitted = _revalidate_fit_for_prediction(fitted)
    try:
        theta = np.asarray(tuple(parameters), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelDataError("component parameters must be a finite numeric vector") from exc
    expected_size = len(fitted.posterior.parameter_names)
    if theta.ndim != 1 or len(theta) != expected_size:
        raise ModelDataError("component parameter vector has the wrong dimension")
    if not np.all(np.isfinite(theta)):
        raise ModelDataError("component parameter vector must be finite")
    terms = _linear_terms_at_parameters(fitted, context, theta)
    log_kappa_index = fitted.posterior.parameter_names.index("log_kappa")
    kappa = _positive_exp(float(theta[log_kappa_index]), field="predictive concentration")
    return ComponentParameterProjection(
        component=fitted.component,
        parameter_names=fitted.posterior.parameter_names,
        base_linear_predictor=terms.linear_predictor,
        predictive_concentration=kappa,
        unseen_effects=terms.unseen_effects,
        serving_player_seen=terms.serving_player_seen,
        returning_player_seen=terms.returning_player_seen,
        surface_seen=terms.surface_seen,
        event_year_seen=terms.event_year_seen,
    )


def project_component_linear_coordinate(
    fitted: FittedServeComponent,
    context: FutureMatchContext,
) -> ComponentLinearCoordinateProjection:
    """Return the stored-MAP predictor gradient without changing the fit."""

    fitted = _revalidate_fit_for_prediction(fitted)
    theta = np.asarray(fitted.posterior.map_parameters, dtype=np.float64)
    terms = _linear_terms_at_parameters(fitted, context, theta)
    return ComponentLinearCoordinateProjection(
        component=fitted.component,
        linear_predictor_map=terms.linear_predictor,
        gradient=tuple(float(value) for value in terms.gradient),
        unseen_effects=terms.unseen_effects,
    )


def _posterior_linear_prediction(
    fitted: FittedServeComponent,
    context: FutureMatchContext,
) -> tuple[float, float, float, bool, bool | None, bool, bool | None]:
    theta = np.asarray(fitted.posterior.map_parameters, dtype=np.float64)
    terms = _linear_terms_at_parameters(fitted, context, theta)
    unseen_effect_variance = sum(item.standard_deviation**2 for item in terms.unseen_effects)

    covariance = fitted.posterior.covariance_array()
    variance = float(terms.gradient @ covariance @ terms.gradient) + unseen_effect_variance
    if variance < -1e-10 or not isfinite(variance):
        raise ModelDataError("posterior linear-predictor variance is invalid")
    return (
        terms.linear_predictor,
        max(0.0, variance) ** 0.5,
        unseen_effect_variance,
        terms.serving_player_seen,
        terms.returning_player_seen,
        terms.surface_seen,
        terms.event_year_seen,
    )


def predict_component(
    fitted: FittedServeComponent,
    context: FutureMatchContext,
) -> ComponentPrediction:
    """Return MAP mean, posterior linear uncertainty, and beta predictive shape."""

    fitted = _revalidate_fit_for_prediction(fitted)
    eta, eta_sd, unseen_variance, server_seen, returner_seen, surface_seen, event_seen = (
        _posterior_linear_prediction(fitted, context)
    )
    mean = float(expit(eta))
    kappa = fitted.kappa
    return ComponentPrediction(
        component=fitted.component,
        map_mean=mean,
        linear_predictor_map=eta,
        linear_predictor_sd=eta_sd,
        unseen_effect_variance=unseen_variance,
        predictive_concentration=kappa,
        beta_alpha_at_map=kappa * mean,
        beta_beta_at_map=kappa * (1.0 - mean),
        serving_player_seen=server_seen,
        returning_player_seen=returner_seen,
        surface_seen=surface_seen,
        event_year_seen=event_seen,
    )


def _validated_serve_fit_bundle(
    fitted: Mapping[ServeComponent, FittedServeComponent],
) -> tuple[dict[ServeComponent, FittedServeComponent], ServeFitBundleIdentity]:
    if set(fitted) != set(ServeComponent):
        raise ModelDataError("prediction requires exactly the five F/A/Q1/D/Q2 fits")
    validated = {
        component: _revalidate_fit_for_prediction(fit) for component, fit in fitted.items()
    }
    if any(fit.component is not component for component, fit in validated.items()):
        raise ModelDataError("component-fit mapping keys do not match fitted components")
    identity_values = {
        (
            fit.framework_version,
            fit.implementation_version,
            fit.tour,
            fit.data_cutoff_utc,
            fit.fitted_at_utc,
            fit.data_snapshot_sha256,
            fit.component_count_artifact_sha256,
            fit.model_config_sha256,
            fit.code_commit,
        )
        for fit in validated.values()
    }
    if len(identity_values) != 1:
        raise ModelDataError(
            "the five component fits must share framework, tour, cutoff, data, "
            "config, and code provenance"
        )
    (
        framework_version,
        implementation_version,
        tour,
        cutoff,
        fitted_at,
        snapshot_hash,
        count_hash,
        config_hash,
        code_commit,
    ) = next(iter(identity_values))
    identity = ServeFitBundleIdentity(
        framework_version=framework_version,
        implementation_version=implementation_version,
        tour=tour,
        information_cutoff_utc=cutoff,
        fitted_at_utc=fitted_at,
        data_snapshot_sha256=snapshot_hash,
        component_count_artifact_sha256=count_hash,
        model_config_sha256=config_hash,
        code_commit=code_commit,
    )
    return validated, identity


def validate_serve_fit_bundle(
    fitted: Mapping[ServeComponent, FittedServeComponent],
) -> ServeFitBundleIdentity:
    """Validate a coherent five-component fitted bundle without predicting."""

    _validated, identity = _validated_serve_fit_bundle(fitted)
    return identity


def predict_serve_performance(
    fitted: Mapping[ServeComponent, FittedServeComponent],
    context: FutureMatchContext,
) -> ServePerformanceDistribution:
    """Construct the five primitive posterior/predictive summaries for one direction."""

    validated, identity = _validated_serve_fit_bundle(fitted)
    predictions = {
        component: predict_component(validated[component], context) for component in ServeComponent
    }
    return ServePerformanceDistribution(
        fit_identity=identity,
        context=context,
        first_serve_in=predictions[ServeComponent.F],
        ace_given_first_in=predictions[ServeComponent.A],
        returnable_first_win=predictions[ServeComponent.Q1],
        double_fault_given_second_opp=predictions[ServeComponent.D],
        playable_second_win=predictions[ServeComponent.Q2],
    )


def _revalidate_fit_for_prediction(fitted: FittedServeComponent) -> FittedServeComponent:
    try:
        return FittedServeComponent.model_validate(fitted.model_dump(mode="python"))
    except ValidationError as exc:
        raise ModelDataError(f"fitted component is internally invalid: {exc}") from exc


__all__ = [
    "FIT_INPUT_SET_VERSION",
    "FRAMEWORK_VERSION",
    "MODEL_IMPLEMENTATION_VERSION",
    "CoefficientEstimate",
    "ComponentLinearCoordinateProjection",
    "ComponentParameterProjection",
    "ComponentPrediction",
    "ContextConfig",
    "DiagnosticConfig",
    "EffectBlock",
    "FitConvergenceError",
    "FitDiagnostics",
    "FitProvenance",
    "FittedServeComponent",
    "FutureMatchContext",
    "ModelDataError",
    "OptimizerConfig",
    "PlayerEffectCoordinateProjection",
    "PlayerInformation",
    "PosteriorApproximation",
    "PreparedComponentRows",
    "PriorConfig",
    "ServeComponent",
    "ServeFitBundleIdentity",
    "ServeModelConfig",
    "ServePerformanceDistribution",
    "UnseenEffectRequirement",
    "effect_block_player_contribution",
    "fit_all_serve_components",
    "fit_all_serve_components_from_bundle",
    "fit_all_serve_components_from_bundles",
    "fit_input_set_sha256",
    "fit_serve_component",
    "predict_component",
    "predict_serve_performance",
    "prepare_component_rows",
    "project_component_linear_coordinate",
    "project_component_parameters",
    "project_player_effect_coordinate",
    "validate_serve_fit_bundle",
]
