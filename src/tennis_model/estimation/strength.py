"""Cutoff-safe dynamic Bradley-Terry strength anchor for v1.1-candidate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from tennis_model.schemas import FrozenModel, Tour


class StrengthModelError(ValueError):
    """Strength inputs, fitting, or artifact provenance are invalid."""


class _StrengthModel(FrozenModel):
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _logistic(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


class DynamicStrengthConfig(_StrengthModel):
    schema_version: Literal["dynamic-strength-config/v1"]
    decay_days: Annotated[float, Field(gt=0.0)]
    process_sd: Annotated[float, Field(gt=0.0)]
    initial_sd: Annotated[float, Field(gt=0.0)]
    surface_sd: Annotated[float, Field(gt=0.0)]
    observation_scale_best_of_3: Annotated[float, Field(gt=0.0)]
    observation_scale_best_of_5: Annotated[float, Field(gt=0.0)]
    variance_floor: Annotated[float, Field(gt=0.0)]
    ranking_sparse_prior_enabled: bool
    ranking_intercept: float
    ranking_log_slope: float
    ranking_prior_sd: Annotated[float, Field(gt=0.0)]
    head_to_head_enabled: bool
    head_to_head_prior_sd: Annotated[float, Field(gt=0.0)]

    def format_scale(self, best_of: int) -> float:
        if best_of == 3:
            return self.observation_scale_best_of_3
        if best_of == 5:
            return self.observation_scale_best_of_5
        raise StrengthModelError("best_of must be 3 or 5")


class StrengthOutcomeRecord(_StrengthModel):
    match_id: str
    tour: Tour
    player_a_id: str
    player_b_id: str
    winner_id: str | None
    start_utc: datetime
    available_at_utc: datetime
    surface: str
    best_of: Literal[3, 5]
    started: bool = True
    completed: bool = True
    retired: bool = False
    walkover: bool = False
    player_a_rank: int | None = Field(default=None, ge=1)
    player_b_rank: int | None = Field(default=None, ge=1)

    @field_validator("match_id", "player_a_id", "player_b_id", "surface")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strength record text fields must not be empty")
        return normalized

    @field_validator("start_utc", "available_at_utc")
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def record_is_coherent(self) -> Self:
        if self.player_a_id == self.player_b_id:
            raise ValueError("strength record players must be distinct")
        if self.winner_id is not None and self.winner_id not in {
            self.player_a_id,
            self.player_b_id,
        }:
            raise ValueError("winner must identify one of the players")
        if self.available_at_utc <= self.start_utc:
            raise ValueError("result availability must follow match start")
        return self

    @property
    def eligible_outcome(self) -> bool:
        return (
            self.started
            and self.completed
            and not self.retired
            and not self.walkover
            and self.winner_id is not None
        )


class SurfaceStrengthState(_StrengthModel):
    surface: str
    mean: float
    variance: Annotated[float, Field(gt=0.0)]


class PlayerStrengthState(_StrengthModel):
    player_id: str
    global_mean: float
    global_variance: Annotated[float, Field(gt=0.0)]
    surfaces: tuple[SurfaceStrengthState, ...]
    last_played_utc: datetime | None
    eligible_matches: int = Field(ge=0)
    graph_component: int = Field(ge=0)

    @field_validator("last_played_utc")
    @classmethod
    def last_played_is_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value, field="last_played_utc")


class HeadToHeadState(_StrengthModel):
    player_low_id: str
    player_high_id: str
    low_orientation_mean: float
    variance: Annotated[float, Field(gt=0.0)]
    observations: int = Field(ge=0)


class StrengthFitDiagnostics(_StrengthModel):
    supplied_rows: int = Field(ge=0)
    eligible_rows: int = Field(ge=0)
    future_rows_excluded: int = Field(ge=0)
    nonstandard_rows_excluded: int = Field(ge=0)
    players: int = Field(ge=0)
    graph_components: int = Field(ge=0)
    mean_log_loss: float | None
    mean_brier: float | None


class DynamicStrengthFit(_StrengthModel):
    schema_version: Literal["dynamic-bradley-terry-fit/v1"] = (
        "dynamic-bradley-terry-fit/v1"
    )
    framework_version: Literal["v1.1-candidate"] = "v1.1-candidate"
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime
    config: DynamicStrengthConfig
    records_sha256: str
    code_commit: str
    player_states: tuple[PlayerStrengthState, ...]
    head_to_head_states: tuple[HeadToHeadState, ...] = ()
    diagnostics: StrengthFitDiagnostics

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @field_validator("records_sha256")
    @classmethod
    def digest_is_valid(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("records_sha256 must be a SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def fit_is_coherent(self) -> Self:
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("strength fit timestamp cannot precede its cutoff")
        if len({item.player_id for item in self.player_states}) != len(self.player_states):
            raise ValueError("strength fit contains duplicate player states")
        return self


class StrengthPrediction(_StrengthModel):
    player_a_id: str
    player_b_id: str
    surface: str
    best_of: Literal[3, 5]
    scheduled_start_utc: datetime
    mean_logit: float
    variance_logit: Annotated[float, Field(gt=0.0)]
    probability: Annotated[float, Field(gt=0.0, lt=1.0)]
    player_a_graph_component: int | None
    player_b_graph_component: int | None
    weakly_connected: bool
    player_a_known: bool
    player_b_known: bool

    @field_validator("scheduled_start_utc")
    @classmethod
    def start_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="scheduled_start_utc")


@dataclass(frozen=True, slots=True)
class PersistedStrengthArtifact:
    directory: Path
    artifact_id: str
    fit: DynamicStrengthFit


@dataclass(slots=True)
class _MutablePlayerState:
    mean: float
    variance: float
    surfaces: dict[str, tuple[float, float]]
    last_played: datetime | None
    matches: int


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        root = self.parent[value]
        while root != self.parent[root]:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _prior_mean(config: DynamicStrengthConfig, rank: int | None) -> float:
    if not config.ranking_sparse_prior_enabled or rank is None:
        return 0.0
    return config.ranking_intercept - config.ranking_log_slope * log(float(rank))


def _advance(
    state: _MutablePlayerState,
    when: datetime,
    config: DynamicStrengthConfig,
) -> None:
    if state.last_played is None:
        return
    days = max(0.0, (when - state.last_played).total_seconds() / 86400.0)
    rho = exp(-days / config.decay_days)
    process_variance = config.process_sd**2
    state.mean *= rho
    state.variance = max(
        config.variance_floor,
        rho * rho * state.variance + process_variance * (1.0 - rho * rho),
    )


def fit_dynamic_strength(
    records: tuple[StrengthOutcomeRecord, ...],
    *,
    tour: Tour,
    cutoff_utc: datetime,
    fitted_at_utc: datetime,
    config: DynamicStrengthConfig,
    code_commit: str,
) -> DynamicStrengthFit:
    """Fit a deterministic diagonal-Laplace dynamic Bradley-Terry filter."""

    cutoff = _utc(cutoff_utc, field="cutoff_utc")
    fitted_at = _utc(fitted_at_utc, field="fitted_at_utc")
    if fitted_at < cutoff:
        raise StrengthModelError("fitted_at_utc cannot precede cutoff_utc")
    supplied = tuple(StrengthOutcomeRecord.model_validate(item) for item in records)
    future = tuple(item for item in supplied if item.available_at_utc >= cutoff)
    candidates = tuple(
        item
        for item in supplied
        if item.tour is tour and item.available_at_utc < cutoff and item.start_utc < cutoff
    )
    eligible = tuple(item for item in candidates if item.eligible_outcome)
    ordered = tuple(sorted(eligible, key=lambda item: (item.start_utc, item.match_id)))
    if not ordered:
        raise StrengthModelError("strength fit has no cutoff-safe eligible outcomes")

    states: dict[str, _MutablePlayerState] = {}
    h2h: dict[tuple[str, str], tuple[float, float, int]] = {}
    graph = _UnionFind()
    log_losses: list[float] = []
    briers: list[float] = []
    eps = 1e-12

    for row in ordered:
        rank_by_player = {
            row.player_a_id: row.player_a_rank,
            row.player_b_id: row.player_b_rank,
        }
        for player in (row.player_a_id, row.player_b_id):
            if player not in states:
                states[player] = _MutablePlayerState(
                    mean=_prior_mean(config, rank_by_player[player]),
                    variance=(
                        config.ranking_prior_sd**2
                        if config.ranking_sparse_prior_enabled
                        and rank_by_player[player] is not None
                        else config.initial_sd**2
                    ),
                    surfaces={},
                    last_played=None,
                    matches=0,
                )
            _advance(states[player], row.start_utc, config)
            states[player].surfaces.setdefault(
                row.surface.casefold(), (0.0, config.surface_sd**2)
            )

        left, right = states[row.player_a_id], states[row.player_b_id]
        surface = row.surface.casefold()
        left_surface_mean, left_surface_var = left.surfaces[surface]
        right_surface_mean, right_surface_var = right.surfaces[surface]
        pair = tuple(sorted((row.player_a_id, row.player_b_id)))
        h_mean, h_var, h_count = h2h.get(
            pair, (0.0, config.head_to_head_prior_sd**2, 0)
        )
        orientation = 1.0 if row.player_a_id == pair[0] else -1.0
        h_effect = orientation * h_mean if config.head_to_head_enabled else 0.0
        scale = config.format_scale(row.best_of)
        linear = scale * (
            left.mean + left_surface_mean - right.mean - right_surface_mean + h_effect
        )
        probability = min(1.0 - eps, max(eps, _logistic(linear)))
        outcome = 1.0 if row.winner_id == row.player_a_id else 0.0
        residual = outcome - probability
        curvature = probability * (1.0 - probability)
        variances = [left.variance, left_surface_var, right.variance, right_surface_var]
        if config.head_to_head_enabled:
            variances.append(h_var)
        total_variance = sum(variances)
        denominator = 1.0 + curvature * scale * scale * total_variance

        def update(mean: float, variance: float, sign: float) -> tuple[float, float]:
            gain = sign * variance * scale / denominator
            updated_mean = mean + gain * residual
            updated_variance = variance - (
                variance * variance * scale * scale * curvature / denominator
            )
            return updated_mean, max(config.variance_floor, updated_variance)

        left.mean, left.variance = update(left.mean, left.variance, 1.0)
        left_surface_mean, left_surface_var = update(
            left_surface_mean, left_surface_var, 1.0
        )
        right.mean, right.variance = update(right.mean, right.variance, -1.0)
        right_surface_mean, right_surface_var = update(
            right_surface_mean, right_surface_var, -1.0
        )
        left.surfaces[surface] = (left_surface_mean, left_surface_var)
        right.surfaces[surface] = (right_surface_mean, right_surface_var)
        if config.head_to_head_enabled:
            h_mean, h_var = update(h_mean, h_var, orientation)
            h2h[pair] = (h_mean, h_var, h_count + 1)
        left.last_played = right.last_played = row.start_utc
        left.matches += 1
        right.matches += 1
        graph.union(row.player_a_id, row.player_b_id)
        log_losses.append(-(outcome * log(probability) + (1.0 - outcome) * log(1.0 - probability)))
        briers.append((probability - outcome) ** 2)

    roots = sorted({graph.find(player) for player in graph.parent})
    component_number = {root: index for index, root in enumerate(roots)}
    player_states = tuple(
        PlayerStrengthState(
            player_id=player,
            global_mean=state.mean,
            global_variance=state.variance,
            surfaces=tuple(
                SurfaceStrengthState(surface=name, mean=value[0], variance=value[1])
                for name, value in sorted(state.surfaces.items())
            ),
            last_played_utc=state.last_played,
            eligible_matches=state.matches,
            graph_component=component_number[graph.find(player)],
        )
        for player, state in sorted(states.items())
    )
    h2h_states = tuple(
        HeadToHeadState(
            player_low_id=pair[0],
            player_high_id=pair[1],
            low_orientation_mean=value[0],
            variance=value[1],
            observations=value[2],
        )
        for pair, value in sorted(h2h.items())
    )
    records_hash = _sha256(
        _canonical_bytes([item.model_dump(mode="json") for item in ordered])
    )
    return DynamicStrengthFit(
        tour=tour,
        information_cutoff_utc=cutoff,
        fitted_at_utc=fitted_at,
        config=config,
        records_sha256=records_hash,
        code_commit=code_commit.strip() or "unknown",
        player_states=player_states,
        head_to_head_states=h2h_states,
        diagnostics=StrengthFitDiagnostics(
            supplied_rows=len(supplied),
            eligible_rows=len(ordered),
            future_rows_excluded=len(future),
            nonstandard_rows_excluded=len(candidates) - len(eligible),
            players=len(player_states),
            graph_components=len(roots),
            mean_log_loss=sum(log_losses) / len(log_losses),
            mean_brier=sum(briers) / len(briers),
        ),
    )


def _advanced_state(
    state: PlayerStrengthState | None,
    *,
    surface: str,
    scheduled_start_utc: datetime,
    config: DynamicStrengthConfig,
    ranking: int | None,
) -> tuple[float, float, int | None, bool]:
    if state is None:
        return (
            _prior_mean(config, ranking),
            (
                config.ranking_prior_sd**2
                if config.ranking_sparse_prior_enabled and ranking is not None
                else config.initial_sd**2 + config.surface_sd**2
            ),
            None,
            False,
        )
    mean = state.global_mean
    variance = state.global_variance
    if state.last_played_utc is not None:
        days = max(
            0.0,
            (scheduled_start_utc - state.last_played_utc).total_seconds() / 86400.0,
        )
        rho = exp(-days / config.decay_days)
        mean *= rho
        variance = rho * rho * variance + config.process_sd**2 * (1.0 - rho * rho)
    surface_state = next(
        (item for item in state.surfaces if item.surface == surface.casefold()), None
    )
    if surface_state is None:
        variance += config.surface_sd**2
    else:
        mean += surface_state.mean
        variance += surface_state.variance
    return mean, max(config.variance_floor, variance), state.graph_component, True


def predict_strength(
    fit: DynamicStrengthFit,
    *,
    player_a_id: str,
    player_b_id: str,
    surface: str,
    best_of: Literal[3, 5],
    scheduled_start_utc: datetime,
    player_a_rank: int | None = None,
    player_b_rank: int | None = None,
) -> StrengthPrediction:
    start = _utc(scheduled_start_utc, field="scheduled_start_utc")
    if player_a_id == player_b_id:
        raise StrengthModelError("strength prediction requires distinct players")
    states = {item.player_id: item for item in fit.player_states}
    a = _advanced_state(
        states.get(player_a_id),
        surface=surface,
        scheduled_start_utc=start,
        config=fit.config,
        ranking=player_a_rank,
    )
    b = _advanced_state(
        states.get(player_b_id),
        surface=surface,
        scheduled_start_utc=start,
        config=fit.config,
        ranking=player_b_rank,
    )
    pair = tuple(sorted((player_a_id, player_b_id)))
    h_state = next(
        (
            item
            for item in fit.head_to_head_states
            if (item.player_low_id, item.player_high_id) == pair
        ),
        None,
    )
    h_mean = 0.0
    h_variance = 0.0
    if fit.config.head_to_head_enabled:
        if h_state is None:
            h_variance = fit.config.head_to_head_prior_sd**2
        else:
            orientation = 1.0 if player_a_id == pair[0] else -1.0
            h_mean = orientation * h_state.low_orientation_mean
            h_variance = h_state.variance
    scale = fit.config.format_scale(best_of)
    mean_logit = scale * (a[0] - b[0] + h_mean)
    variance_logit = max(
        fit.config.variance_floor,
        scale * scale * (a[1] + b[1] + h_variance),
    )
    # Logistic-normal mean approximation, symmetric under player reversal.
    adjusted = mean_logit / sqrt(1.0 + 3.141592653589793 * variance_logit / 8.0)
    probability = min(1.0 - 1e-12, max(1e-12, _logistic(adjusted)))
    weak = a[2] is None or b[2] is None or a[2] != b[2]
    return StrengthPrediction(
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        surface=surface.casefold(),
        best_of=best_of,
        scheduled_start_utc=start,
        mean_logit=mean_logit,
        variance_logit=variance_logit,
        probability=probability,
        player_a_graph_component=a[2],
        player_b_graph_component=b[2],
        weakly_connected=weak,
        player_a_known=a[3],
        player_b_known=b[3],
    )


def write_strength_artifact(
    fit: DynamicStrengthFit,
    artifact_root: str | Path,
) -> PersistedStrengthArtifact:
    payload = _canonical_bytes(fit)
    artifact_id = _sha256(payload)
    parent = (
        Path(artifact_root).resolve()
        / fit.tour.value.lower()
        / fit.information_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    )
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact_id[:32]
    if target.exists():
        return load_strength_artifact(target)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        path = staging / "strength.json"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        (staging / "diagnostics.txt").write_text(
            "\n".join(
                (
                    f"framework: {fit.framework_version}",
                    f"tour: {fit.tour.value}",
                    f"cutoff: {fit.information_cutoff_utc.isoformat()}",
                    f"eligible rows: {fit.diagnostics.eligible_rows}",
                    f"players: {fit.diagnostics.players}",
                    f"graph components: {fit.diagnostics.graph_components}",
                )
            )
            + "\n",
            encoding="ascii",
        )
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_strength_artifact(target)


def load_strength_artifact(directory: str | Path) -> PersistedStrengthArtifact:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise StrengthModelError("strength artifact directory is invalid")
    path = root / "strength.json"
    if path.is_symlink() or not path.is_file():
        raise StrengthModelError("strength artifact payload is missing")
    payload = path.read_bytes()
    fit = DynamicStrengthFit.model_validate_json(payload)
    if _canonical_bytes(fit) != payload:
        raise StrengthModelError("strength artifact is not canonical")
    artifact_id = _sha256(payload)
    if root.name != artifact_id[:32]:
        raise StrengthModelError("strength artifact directory does not match content")
    return PersistedStrengthArtifact(root.resolve(), artifact_id, fit)


__all__ = [
    "DynamicStrengthConfig",
    "DynamicStrengthFit",
    "PersistedStrengthArtifact",
    "StrengthModelError",
    "StrengthOutcomeRecord",
    "StrengthPrediction",
    "fit_dynamic_strength",
    "load_strength_artifact",
    "predict_strength",
    "write_strength_artifact",
]
