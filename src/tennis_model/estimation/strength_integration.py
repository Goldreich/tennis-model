"""Cross-fitted strength integration and coherent Q1/Q2 tilting for v1.1."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, log, sqrt
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator
from scipy.optimize import brentq, minimize  # type: ignore[import-untyped]

from tennis_model.estimation.strength import StrengthPrediction
from tennis_model.exact_probability import exact_match_win_probability
from tennis_model.schemas import FrozenModel, Tour
from tennis_model.serve import PrimitiveServeMeans


class StrengthIntegrationError(ValueError):
    """Cross-fitting or path integration violates the v1.1 contract."""


class _IntegrationModel(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _logistic(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _logit(value: float, bound: float) -> float:
    p = min(1.0 - bound, max(bound, float(value)))
    return log(p / (1.0 - p))


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return exp(value)
    return log(1.0 + exp(value))


class StrengthIntegrationConfig(_IntegrationModel):
    schema_version: Literal["strength-integration-config/v1"]
    l2_penalty: Annotated[float, Field(ge=0.0)]
    reliability_prior_logit: float
    maximum_absolute_tilt: Annotated[float, Field(gt=0.0)]
    root_tolerance: Annotated[float, Field(gt=0.0)]
    probability_bound: Annotated[float, Field(gt=0.0, lt=0.01)]
    coefficient_draws_for_summary: Annotated[int, Field(ge=32)]
    q1_weight: Annotated[float, Field(gt=0.0)]
    q2_weight: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def q_weights_are_identified(self) -> Self:
        if abs((self.q1_weight + self.q2_weight) / 2.0 - 1.0) > 1e-12:
            raise ValueError("Q1/Q2 integration weights must have mean one")
        return self


class CrossFittedStrengthRecord(_IntegrationModel):
    match_id: str
    tour: Tour
    player_a_id: str
    player_b_id: str
    forecast_cutoff_utc: datetime
    scheduled_start_utc: datetime
    outcome_available_at_utc: datetime
    component_logit: float
    anchor_logit: float
    component_variance: Annotated[float, Field(gt=0.0)]
    anchor_variance: Annotated[float, Field(gt=0.0)]
    component_instability: Annotated[float, Field(ge=0.0, le=1.0)]
    component_sparsity: Annotated[float, Field(ge=0.0, le=1.0)]
    player_a_won: bool

    @field_validator(
        "forecast_cutoff_utc", "scheduled_start_utc", "outcome_available_at_utc"
    )
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def chronology_is_safe(self) -> Self:
        if self.forecast_cutoff_utc >= self.scheduled_start_utc:
            raise ValueError("cross-fitted forecast cutoff must precede match start")
        if self.outcome_available_at_utc <= self.forecast_cutoff_utc:
            raise ValueError("outcome must be unavailable when its forecast is generated")
        if self.player_a_id == self.player_b_id:
            raise ValueError("cross-fitted players must be distinct")
        return self


class StrengthIntegrationDiagnostics(_IntegrationModel):
    rows: int = Field(ge=0)
    converged: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int = Field(ge=0)
    mean_brier: float
    mean_log_loss: float


class StrengthIntegrationFit(_IntegrationModel):
    schema_version: Literal["strength-integration-fit/v1"] = "strength-integration-fit/v1"
    framework_version: Literal["v1.1-candidate", "v1.1"] = "v1.1"
    tour: Tour
    training_cutoff_utc: datetime
    fitted_at_utc: datetime
    config: StrengthIntegrationConfig
    records_sha256: str
    code_commit: str
    raw_parameter_names: tuple[str, ...]
    raw_parameters: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    diagnostics: StrengthIntegrationDiagnostics

    @field_validator("training_cutoff_utc", "fitted_at_utc")
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def dimensions_are_coherent(self) -> Self:
        size = len(self.raw_parameter_names)
        if size != 5 or len(self.raw_parameters) != size:
            raise ValueError("integration fit requires five ordered raw parameters")
        if len(self.covariance) != size or any(len(row) != size for row in self.covariance):
            raise ValueError("integration covariance dimensions do not match")
        if self.fitted_at_utc < self.training_cutoff_utc:
            raise ValueError("integration fit time cannot precede its training cutoff")
        return self


class FixedStrengthIntegrationFit(_IntegrationModel):
    """Prevalidated fixed-logit blend used by the selected surface Elo anchor."""

    schema_version: Literal["fixed-strength-integration-fit/v1"] = (
        "fixed-strength-integration-fit/v1"
    )
    framework_version: Literal["v1.1-candidate", "v1.1"] = "v1.1"
    tour: Tour
    training_cutoff_utc: datetime
    fitted_at_utc: datetime
    config: StrengthIntegrationConfig
    anchor_weight: Annotated[float, Field(gt=0.0, lt=1.0)]
    selection_reference: str
    code_commit: str

    @field_validator("training_cutoff_utc", "fitted_at_utc")
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @field_validator("selection_reference")
    @classmethod
    def selection_is_present(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("fixed integration selection reference must not be empty")
        return normalized

    @model_validator(mode="after")
    def fit_is_coherent(self) -> Self:
        if self.fitted_at_utc < self.training_cutoff_utc:
            raise ValueError("fixed integration fit time cannot precede its cutoff")
        return self


StrengthIntegrationArtifactFit = StrengthIntegrationFit | FixedStrengthIntegrationFit


class StrengthMatchParameters(_IntegrationModel):
    schema_version: Literal["strength-match-parameters/v1"] = (
        "strength-match-parameters/v1"
    )
    anchor_artifact_id: str
    integration_artifact_id: str
    player_a_id: str
    player_b_id: str
    anchor_logit_mean: float
    anchor_logit_variance: Annotated[float, Field(gt=0.0)]
    anchor_probability: Annotated[float, Field(gt=0.0, lt=1.0)]
    component_logit: float
    component_variance: Annotated[float, Field(gt=0.0)]
    component_probability: Annotated[float, Field(gt=0.0, lt=1.0)]
    component_instability: Annotated[float, Field(ge=0.0, le=1.0)]
    component_sparsity: Annotated[float, Field(ge=0.0, le=1.0)]
    reliability_weight: Annotated[float, Field(gt=0.0, lt=1.0)]
    target_logit: float
    target_probability: Annotated[float, Field(gt=0.0, lt=1.0)]
    q_tilt_mean: float
    q_tilt_sd: Annotated[float, Field(ge=0.0)]
    target_attained_logit: float
    tilt_saturated: bool
    sign_disagreement: bool
    logit_disagreement: Annotated[float, Field(ge=0.0)]
    player_a_ace_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    player_b_ace_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    player_a_df_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    player_b_df_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    graph_weakly_connected: bool
    rng_stream_version: Literal["seedsequence-strength-integration/v1"] = (
        "seedsequence-strength-integration/v1"
    )

    @field_validator("anchor_artifact_id", "integration_artifact_id")
    @classmethod
    def artifact_ids_are_valid(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("strength artifact IDs must be SHA-256 digests")
        return normalized


class StrengthIntegrationDraw(_IntegrationModel):
    component_logit: float
    anchor_logit: float
    target_logit: float
    reliability_weight: Annotated[float, Field(gt=0.0, lt=1.0)]
    q_tilt: float
    attained_logit: float
    saturated: bool


@dataclass(frozen=True, slots=True)
class PersistedStrengthIntegrationArtifact:
    directory: Path
    artifact_id: str
    fit: StrengthIntegrationArtifactFit


def create_fixed_strength_integration(
    *,
    tour: Tour,
    training_cutoff_utc: datetime,
    fitted_at_utc: datetime,
    config: StrengthIntegrationConfig,
    anchor_weight: float,
    selection_reference: str,
    code_commit: str,
) -> FixedStrengthIntegrationFit:
    return FixedStrengthIntegrationFit(
        tour=tour,
        training_cutoff_utc=_utc(training_cutoff_utc, field="training_cutoff_utc"),
        fitted_at_utc=_utc(fitted_at_utc, field="fitted_at_utc"),
        config=config,
        anchor_weight=anchor_weight,
        selection_reference=selection_reference,
        code_commit=code_commit.strip() or "unknown",
    )


def _decode(
    raw: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float, np.ndarray[Any, np.dtype[np.float64]]]:
    beta = exp(float(raw[0]))
    eta = np.asarray(
        [
            float(raw[1]),
            _softplus(float(raw[2])),
            _softplus(float(raw[3])),
            _softplus(float(raw[4])),
        ],
        dtype=np.float64,
    )
    return beta, eta


def reliability_weight(
    raw_parameters: tuple[float, ...] | np.ndarray[Any, np.dtype[np.float64]],
    *,
    component_variance: float,
    anchor_variance: float,
    component_instability: float,
    component_sparsity: float,
) -> float:
    raw = np.asarray(raw_parameters, dtype=np.float64)
    _beta, eta = _decode(raw)
    features = np.asarray(
        [
            1.0,
            log((component_variance + 1e-12) / (anchor_variance + 1e-12)),
            component_instability,
            component_sparsity,
        ],
        dtype=np.float64,
    )
    return min(1.0 - 1e-12, max(1e-12, _logistic(float(eta @ features))))


def fit_strength_integration(
    records: tuple[CrossFittedStrengthRecord, ...],
    *,
    tour: Tour,
    training_cutoff_utc: datetime,
    fitted_at_utc: datetime,
    config: StrengthIntegrationConfig,
    code_commit: str,
) -> StrengthIntegrationFit:
    cutoff = _utc(training_cutoff_utc, field="training_cutoff_utc")
    fitted_at = _utc(fitted_at_utc, field="fitted_at_utc")
    rows = tuple(
        row
        for row in (CrossFittedStrengthRecord.model_validate(item) for item in records)
        if row.tour is tour and row.outcome_available_at_utc < cutoff
    )
    if len(rows) < 8:
        raise StrengthIntegrationError("integration fit requires at least eight settled rows")
    initial = np.asarray(
        [0.0, config.reliability_prior_logit, -2.0, -2.0, -2.0], dtype=np.float64
    )

    def objective(raw: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        beta, _eta = _decode(raw)
        loss = 0.0
        for row in rows:
            gate = reliability_weight(
                raw,
                component_variance=row.component_variance,
                anchor_variance=row.anchor_variance,
                component_instability=row.component_instability,
                component_sparsity=row.component_sparsity,
            )
            target = beta * row.component_logit + gate * (
                row.anchor_logit - row.component_logit
            )
            p = min(1.0 - 1e-12, max(1e-12, _logistic(target)))
            y = float(row.player_a_won)
            loss -= y * log(p) + (1.0 - y) * log(1.0 - p)
        penalty = config.l2_penalty * float(
            (raw[0] ** 2)
            + ((raw[1] - config.reliability_prior_logit) ** 2)
            + np.sum(raw[2:] ** 2)
        )
        return loss + penalty

    result = minimize(objective, initial, method="BFGS", options={"maxiter": 1000, "gtol": 1e-8})
    if not np.all(np.isfinite(result.x)):
        raise StrengthIntegrationError("integration optimizer returned non-finite parameters")
    try:
        covariance = np.asarray(result.hess_inv, dtype=np.float64)
    except Exception:
        covariance = np.eye(5, dtype=np.float64) * 0.01
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    probabilities: list[float] = []
    for row in rows:
        beta, _eta = _decode(np.asarray(result.x, dtype=np.float64))
        gate = reliability_weight(
            result.x,
            component_variance=row.component_variance,
            anchor_variance=row.anchor_variance,
            component_instability=row.component_instability,
            component_sparsity=row.component_sparsity,
        )
        probabilities.append(
            _logistic(
                beta * row.component_logit
                + gate * (row.anchor_logit - row.component_logit)
            )
        )
    outcomes = [float(row.player_a_won) for row in rows]
    records_hash = _digest(_canonical_bytes([row.model_dump(mode="json") for row in rows]))
    return StrengthIntegrationFit(
        tour=tour,
        training_cutoff_utc=cutoff,
        fitted_at_utc=fitted_at,
        config=config,
        records_sha256=records_hash,
        code_commit=code_commit.strip() or "unknown",
        raw_parameter_names=(
            "log_beta_component",
            "eta_intercept",
            "raw_eta_relative_variance",
            "raw_eta_instability",
            "raw_eta_sparsity",
        ),
        raw_parameters=tuple(float(value) for value in result.x),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        diagnostics=StrengthIntegrationDiagnostics(
            rows=len(rows),
            converged=bool(result.success),
            optimizer_status=int(result.status),
            optimizer_message=str(result.message),
            iterations=int(getattr(result, "nit", 0)),
            mean_brier=(
                sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True))
                / len(rows)
            ),
            mean_log_loss=sum(
                -(y * log(max(1e-12, p)) + (1.0 - y) * log(max(1e-12, 1.0 - p)))
                for p, y in zip(probabilities, outcomes, strict=True)
            )
            / len(rows),
        ),
    )


def _service_point(draw: PrimitiveServeMeans) -> float:
    return draw.service_point_win


def _tilt_draw(
    draw: PrimitiveServeMeans,
    delta: float,
    *,
    direction: float,
    config: StrengthIntegrationConfig,
) -> PrimitiveServeMeans:
    def shifted(value: float, weight: float) -> float:
        return _logistic(_logit(value, config.probability_bound) + direction * delta * weight / 2.0)

    return PrimitiveServeMeans(
        first_serve_in=draw.first_serve_in,
        ace_given_first_in=draw.ace_given_first_in,
        returnable_first_win=shifted(draw.returnable_first_win, config.q1_weight),
        double_fault_given_second_opp=draw.double_fault_given_second_opp,
        playable_second_win=shifted(draw.playable_second_win, config.q2_weight),
    )


def solve_q_tilt(
    player_a: PrimitiveServeMeans,
    player_b: PrimitiveServeMeans,
    *,
    target_logit: float,
    best_of: Literal[3, 5],
    config: StrengthIntegrationConfig,
) -> tuple[float, float, bool]:
    def attained(delta: float) -> float:
        a = _tilt_draw(player_a, delta, direction=1.0, config=config)
        b = _tilt_draw(player_b, delta, direction=-1.0, config=config)
        probability = exact_match_win_probability(
            _service_point(a), _service_point(b), best_of=best_of
        )
        return _logit(probability, config.probability_bound)

    lower = -config.maximum_absolute_tilt
    upper = config.maximum_absolute_tilt
    lower_value = attained(lower)
    upper_value = attained(upper)
    if target_logit <= lower_value:
        return lower, lower_value, True
    if target_logit >= upper_value:
        return upper, upper_value, True

    def residual(delta: float) -> float:
        return attained(delta) - target_logit

    # Brent's method preserves the specification's deterministic bracketed-root
    # contract while avoiding roughly thirty full exact-scoring evaluations per
    # ordinary draw. Verify the configured attained-logit tolerance explicitly;
    # the fixed bisection fallback retains the prior numerical behavior if a
    # particularly flat mapping defeats the x-space stopping criterion.
    root = float(
        brentq(
            residual,
            lower,
            upper,
            xtol=max(5e-324, config.root_tolerance * 0.01),
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=100,
        )
    )
    root_value = attained(root)
    if abs(root_value - target_logit) <= config.root_tolerance:
        return root, root_value, False

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        value = attained(midpoint)
        if abs(value - target_logit) <= config.root_tolerance:
            return midpoint, value, False
        if value < target_logit:
            lower = midpoint
        else:
            upper = midpoint
    midpoint = (lower + upper) / 2.0
    value = attained(midpoint)
    return midpoint, value, abs(value - target_logit) > config.root_tolerance


def prepare_strength_match_parameters(
    *,
    anchor_artifact_id: str,
    integration_artifact_id: str,
    anchor: StrengthPrediction,
    integration: StrengthIntegrationArtifactFit,
    player_a: PrimitiveServeMeans,
    player_b: PrimitiveServeMeans,
    best_of: Literal[3, 5],
    component_variance: float,
    component_instability: float,
    component_sparsity: float,
) -> StrengthMatchParameters:
    component_probability = exact_match_win_probability(
        player_a.service_point_win, player_b.service_point_win, best_of=best_of
    )
    component_logit = _logit(component_probability, integration.config.probability_bound)
    if isinstance(integration, FixedStrengthIntegrationFit):
        beta = 1.0
        gate = integration.anchor_weight
    else:
        beta, _eta = _decode(np.asarray(integration.raw_parameters, dtype=np.float64))
        gate = reliability_weight(
            integration.raw_parameters,
            component_variance=component_variance,
            anchor_variance=anchor.variance_logit,
            component_instability=component_instability,
            component_sparsity=component_sparsity,
        )
    target = beta * component_logit + gate * (anchor.mean_logit - component_logit)
    tilt, attained, saturated = solve_q_tilt(
        player_a,
        player_b,
        target_logit=target,
        best_of=best_of,
        config=integration.config,
    )
    step = 1e-4
    _plus_tilt, _plus, _ = solve_q_tilt(
        player_a,
        player_b,
        target_logit=attained + step,
        best_of=best_of,
        config=integration.config,
    )
    derivative = step / max(1e-8, abs(_plus_tilt - tilt))
    target_variance = (
        (1.0 - gate) ** 2 * component_variance
        + gate**2 * anchor.variance_logit
    )
    tilt_sd = min(
        integration.config.maximum_absolute_tilt,
        sqrt(target_variance) / max(1e-8, derivative),
    )
    return StrengthMatchParameters(
        anchor_artifact_id=anchor_artifact_id,
        integration_artifact_id=integration_artifact_id,
        player_a_id=anchor.player_a_id,
        player_b_id=anchor.player_b_id,
        anchor_logit_mean=anchor.mean_logit,
        anchor_logit_variance=anchor.variance_logit,
        anchor_probability=anchor.probability,
        component_logit=component_logit,
        component_variance=max(1e-12, component_variance),
        component_probability=component_probability,
        component_instability=component_instability,
        component_sparsity=component_sparsity,
        reliability_weight=gate,
        target_logit=target,
        target_probability=_logistic(target),
        q_tilt_mean=tilt,
        q_tilt_sd=max(0.0, tilt_sd),
        target_attained_logit=attained,
        tilt_saturated=saturated,
        sign_disagreement=component_logit * anchor.mean_logit < 0.0,
        logit_disagreement=abs(anchor.mean_logit - component_logit),
        player_a_ace_rate=player_a.ace_rate,
        player_b_ace_rate=player_b.ace_rate,
        player_a_df_rate=player_a.double_fault_rate,
        player_b_df_rate=player_b.double_fault_rate,
        graph_weakly_connected=anchor.weakly_connected,
    )


def integrate_serve_performance(
    player_a: PrimitiveServeMeans,
    player_b: PrimitiveServeMeans,
    *,
    parameters: StrengthMatchParameters,
    integration: StrengthIntegrationArtifactFit,
    best_of: Literal[3, 5],
    rng: np.random.Generator,
) -> tuple[PrimitiveServeMeans, PrimitiveServeMeans, StrengthIntegrationDraw]:
    anchor_logit = float(
        rng.normal(parameters.anchor_logit_mean, sqrt(parameters.anchor_logit_variance))
    )
    component_probability = exact_match_win_probability(
        _service_point(player_a), _service_point(player_b), best_of=best_of
    )
    component_logit = _logit(component_probability, integration.config.probability_bound)
    if isinstance(integration, FixedStrengthIntegrationFit):
        beta = 1.0
        gate = integration.anchor_weight
    else:
        raw_mean = np.asarray(integration.raw_parameters, dtype=np.float64)
        covariance = np.asarray(integration.covariance, dtype=np.float64)
        raw = np.asarray(rng.multivariate_normal(raw_mean, covariance), dtype=np.float64)
        beta, _eta = _decode(raw)
        gate = reliability_weight(
            raw,
            component_variance=parameters.component_variance,
            anchor_variance=parameters.anchor_logit_variance,
            component_instability=parameters.component_instability,
            component_sparsity=parameters.component_sparsity,
        )
    target = beta * component_logit + gate * (anchor_logit - component_logit)
    tilt, attained, saturated = solve_q_tilt(
        player_a,
        player_b,
        target_logit=target,
        best_of=best_of,
        config=integration.config,
    )
    adjusted_a = _tilt_draw(player_a, tilt, direction=1.0, config=integration.config)
    adjusted_b = _tilt_draw(player_b, tilt, direction=-1.0, config=integration.config)
    return adjusted_a, adjusted_b, StrengthIntegrationDraw(
        component_logit=component_logit,
        anchor_logit=anchor_logit,
        target_logit=target,
        reliability_weight=gate,
        q_tilt=tilt,
        attained_logit=attained,
        saturated=saturated,
    )


def write_strength_integration_artifact(
    fit: StrengthIntegrationArtifactFit,
    artifact_root: str | Path,
) -> PersistedStrengthIntegrationArtifact:
    payload = _canonical_bytes(fit)
    artifact_id = _digest(payload)
    parent = (
        Path(artifact_root).resolve()
        / fit.tour.value.lower()
        / fit.training_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    )
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact_id[:32]
    if target.exists():
        return load_strength_integration_artifact(target)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        path = staging / "integration.json"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        diagnostics = [
            f"framework: {fit.framework_version}",
            f"tour: {fit.tour.value}",
            f"training cutoff: {fit.training_cutoff_utc.isoformat()}",
        ]
        if isinstance(fit, FixedStrengthIntegrationFit):
            diagnostics.extend(
                (
                    "mode: fixed",
                    f"anchor weight: {fit.anchor_weight:.8f}",
                    f"selection: {fit.selection_reference}",
                )
            )
        else:
            diagnostics.extend(
                (
                    f"rows: {fit.diagnostics.rows}",
                    f"converged: {fit.diagnostics.converged}",
                    f"mean Brier: {fit.diagnostics.mean_brier:.8f}",
                )
            )
        (staging / "diagnostics.txt").write_text(
            "\n".join(diagnostics) + "\n", encoding="ascii"
        )
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_strength_integration_artifact(target)


def load_strength_integration_artifact(
    directory: str | Path,
) -> PersistedStrengthIntegrationArtifact:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise StrengthIntegrationError("integration artifact directory is invalid")
    path = root / "integration.json"
    if path.is_symlink() or not path.is_file():
        raise StrengthIntegrationError("integration artifact payload is missing")
    payload = path.read_bytes()
    schema_version = json.loads(payload).get("schema_version")
    fit: StrengthIntegrationArtifactFit
    if schema_version == "fixed-strength-integration-fit/v1":
        fit = FixedStrengthIntegrationFit.model_validate_json(payload)
    else:
        fit = StrengthIntegrationFit.model_validate_json(payload)
    if _canonical_bytes(fit) != payload:
        raise StrengthIntegrationError("integration artifact is not canonical")
    artifact_id = _digest(payload)
    if root.name != artifact_id[:32]:
        raise StrengthIntegrationError("integration artifact directory does not match content")
    return PersistedStrengthIntegrationArtifact(root.resolve(), artifact_id, fit)


__all__ = [
    "CrossFittedStrengthRecord",
    "FixedStrengthIntegrationFit",
    "PersistedStrengthIntegrationArtifact",
    "StrengthIntegrationArtifactFit",
    "StrengthIntegrationConfig",
    "StrengthIntegrationDraw",
    "StrengthIntegrationError",
    "StrengthIntegrationFit",
    "StrengthMatchParameters",
    "create_fixed_strength_integration",
    "fit_strength_integration",
    "integrate_serve_performance",
    "load_strength_integration_artifact",
    "prepare_strength_match_parameters",
    "reliability_weight",
    "solve_q_tilt",
    "write_strength_integration_artifact",
]
