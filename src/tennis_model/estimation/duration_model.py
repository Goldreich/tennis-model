"""Frozen B5 match-duration model and content-addressed fit artifacts.

The module is intentionally independent of tennis point generation.  It fits the
specified duration layer to completed, legal, exact-dated match records and later
maps a simulated match's realised exposure into a latent duration draw.  Official
minute display is a separate, versioned policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import yaml
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.linalg import helmert  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import digamma, gammaln  # type: ignore[import-untyped]
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from tennis_model.data.validate import ScoreValidationResult, SetScoreKind, validate_score
from tennis_model.estimation.weighted_glmm import CurvatureError, laplace_curvature
from tennis_model.schemas import FrozenModel, NonEmptyStr, Tour

DURATION_CONFIG_SCHEMA_VERSION = "duration-model-config/v1"
DURATION_ARTIFACT_SCHEMA_VERSION = "duration-fit-artifact/v1"
DURATION_TRAINING_SCHEMA_VERSION = "duration-training-batch/v1"
DURATION_DISPLAY_POLICY_SCHEMA_VERSION = "duration-display-policy/v1"
DURATION_DRAW_SCHEMA_VERSION = "duration-draw/v1"

_ARTIFACT_FILENAME = "duration-fit.json"
_SHA256_CHARS = frozenset("0123456789abcdef")
_CORE_COEFFICIENTS = ("theta0", "points", "games", "sets", "tiebreaks")
_CONTEXT_COEFFICIENTS = (
    "indoor",
    "temperature_c",
    "night_session",
    "current_usopen_2026",
)


class DurationModelError(ValueError):
    """A duration input or fitted quantity violates the B5 contract."""


class DurationArtifactError(RuntimeError):
    """A duration artifact cannot be written or loaded."""


class DurationArtifactIntegrityError(DurationArtifactError):
    """Persisted duration artifact content or identity is invalid."""


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in _SHA256_CHARS for character in normalized):
        raise ValueError(f"{field_name} must contain 64 hexadecimal characters")
    return normalized


def _finite(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


class DurationRidgeConfig(FrozenModel):
    """Hash-pinned Gaussian prior scales for the B5 linear predictor."""

    intercept_sd: Annotated[float, Field(gt=0)]
    points_sd: Annotated[float, Field(gt=0)]
    games_sd: Annotated[float, Field(gt=0)]
    sets_sd: Annotated[float, Field(gt=0)]
    tiebreaks_sd: Annotated[float, Field(gt=0)]
    player_pace_sd: Annotated[float, Field(gt=0)]
    indoor_sd: Annotated[float, Field(gt=0)]
    temperature_sd: Annotated[float, Field(gt=0)]
    night_session_sd: Annotated[float, Field(gt=0)]
    current_event_sd: Annotated[float, Field(gt=0)]

    @field_validator("*")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationResidualConfig(FrozenModel):
    """Pinned initialization and weak regularization for the Student-t scale."""

    initial_sigma0: Annotated[float, Field(gt=0)]
    initial_sigma1: Annotated[float, Field(gt=0)]
    initial_nu: Annotated[float, Field(gt=2)]
    log_sigma0_prior_sd: Annotated[float, Field(gt=0)]
    log_sigma1_prior_sd: Annotated[float, Field(gt=0)]
    log_nu_minus_two_prior_sd: Annotated[float, Field(gt=0)]

    @field_validator("*")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationOptimizerConfig(FrozenModel):
    """Deterministic numerical choices included in the configuration hash."""

    method: Literal["L-BFGS-B"]
    max_iterations: Annotated[int, Field(gt=0)]
    max_line_search_steps: Annotated[int, Field(gt=0)]
    gradient_tolerance: Annotated[float, Field(gt=0)]
    function_tolerance: Annotated[float, Field(gt=0)]
    covariance_relative_step: Annotated[float, Field(gt=0)]
    covariance_eigenvalue_floor: Annotated[float, Field(gt=0)]
    full_covariance_max_parameters: Annotated[int, Field(gt=0)]

    @field_validator(
        "gradient_tolerance",
        "function_tolerance",
        "covariance_relative_step",
        "covariance_eigenvalue_floor",
    )
    @classmethod
    def numerical_values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationContextConfig(FrozenModel):
    """Predeclared optional contexts; unavailable terms remain inactive."""

    supported_terms: tuple[
        Literal["indoor", "temperature_c", "night_session", "current_usopen_2026"], ...
    ]
    temperature_reference_c: float
    current_event: NonEmptyStr
    current_event_year: Annotated[int, Field(ge=2000)]
    minimum_level_rows: Annotated[int, Field(gt=0)]

    @field_validator("temperature_reference_c")
    @classmethod
    def temperature_is_finite(cls, value: float) -> float:
        return _finite(value, field_name="temperature_reference_c")

    @field_validator("supported_terms")
    @classmethod
    def terms_are_unique_and_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("supported_terms must not contain duplicates")
        expected = tuple(term for term in _CONTEXT_COEFFICIENTS if term in value)
        if value != expected:
            raise ValueError("supported_terms must follow the frozen canonical order")
        return value


class DurationModelConfig(FrozenModel):
    """Complete, immutable B5 fit configuration."""

    schema_version: Literal["duration-model-config/v1"]
    framework_version: Literal["v1.0"]
    window_days: Annotated[int, Field(gt=0)]
    half_life_days: Annotated[float, Field(gt=0)]
    ridge: DurationRidgeConfig
    residual: DurationResidualConfig
    optimizer: DurationOptimizerConfig
    context: DurationContextConfig

    @field_validator("half_life_days")
    @classmethod
    def half_life_is_finite(cls, value: float) -> float:
        return _finite(value, field_name="half_life_days")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_duration_model_config(path: str | Path) -> DurationModelConfig:
    """Read one complete duration configuration with no fallback defaults."""

    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DurationModelError(
            f"cannot read duration configuration {config_path}: {exc}"
        ) from exc
    try:
        value = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise DurationModelError(
            f"invalid YAML in duration configuration {config_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DurationModelError("duration configuration root must be a mapping")
    try:
        return DurationModelConfig.model_validate(value)
    except Exception as exc:
        raise DurationModelError(f"invalid duration configuration: {exc}") from exc


class DurationConditions(FrozenModel):
    """Only the predeclared B5 contexts, with unknown values kept missing."""

    indoor: bool | None = None
    temperature_c: float | None = None
    night_session: bool | None = None
    event: str | None = None
    event_year: int | None = None

    @field_validator("temperature_c")
    @classmethod
    def optional_temperature_is_finite(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value, field_name="temperature_c")

    @model_validator(mode="after")
    def event_pair_is_coherent(self) -> Self:
        if (self.event is None) != (self.event_year is None):
            raise ValueError("event and event_year must be supplied together")
        return self


class DurationTrainingObservation(FrozenModel):
    """One legal completed match, constructed from exactly two reciprocal rows."""

    schema_version: Literal["duration-training-observation/v1"] = "duration-training-observation/v1"
    match_id: NonEmptyStr
    tour: Tour
    match_date: date
    available_at_utc: datetime
    player_a_id: NonEmptyStr
    player_b_id: NonEmptyStr
    duration_minutes: Annotated[float, Field(gt=0)]
    total_points: Annotated[int, Field(gt=0)]
    official_games: Annotated[int, Field(gt=0)]
    sets: Annotated[int, Field(gt=0)]
    tiebreaks: Annotated[int, Field(ge=0)]
    match_tiebreak_format: bool
    conditions: DurationConditions
    recency_weight: Annotated[float, Field(gt=0, le=1)]
    source_ids: tuple[NonEmptyStr, ...]
    source_sha256s: tuple[str, ...]
    crosswalk_sha256s: tuple[str, ...] = ()

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="available_at_utc")

    @field_validator("duration_minutes", "recency_weight")
    @classmethod
    def numeric_values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("source_ids must be nonempty, unique, and sorted")
        return value

    @field_validator("source_sha256s", "crosswalk_sha256s")
    @classmethod
    def hashes_are_canonical(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        normalized = tuple(_sha256(item, field_name=info.field_name) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError(f"{info.field_name} must be unique and sorted")
        if info.field_name == "source_sha256s" and not normalized:
            raise ValueError("source_sha256s must be nonempty")
        return normalized

    @model_validator(mode="after")
    def players_are_distinct(self) -> Self:
        if self.player_a_id >= self.player_b_id:
            raise ValueError("players must be distinct and stored in canonical order")
        return self


class DurationExclusion(FrozenModel):
    """A preserved match-level rejection with explicit anomaly detail."""

    match_id: NonEmptyStr
    tour: Tour | None
    reason: NonEmptyStr
    details: NonEmptyStr
    match_date: date | None = None
    source_ids: tuple[str, ...] = ()
    source_sha256s: tuple[str, ...] = ()


class DurationReasonCount(FrozenModel):
    reason: NonEmptyStr
    match_count: Annotated[int, Field(ge=0)]


class DurationTrainingAudit(FrozenModel):
    """Complete included/excluded accounting for a duration training build."""

    input_row_count: Annotated[int, Field(ge=0)]
    candidate_match_count: Annotated[int, Field(ge=0)]
    included_match_count: Annotated[int, Field(ge=0)]
    excluded_match_count: Annotated[int, Field(ge=0)]
    retirement_match_count: Annotated[int, Field(ge=0)]
    anomaly_match_count: Annotated[int, Field(ge=0)]
    match_tiebreak_format_included_count: Annotated[int, Field(ge=0)]
    reason_counts: tuple[DurationReasonCount, ...]

    @model_validator(mode="after")
    def counts_reconcile(self) -> Self:
        if self.candidate_match_count != self.included_match_count + self.excluded_match_count:
            raise ValueError("included and excluded match counts do not reconcile")
        if sum(item.match_count for item in self.reason_counts) != self.excluded_match_count:
            raise ValueError("reason counts do not reconcile to exclusions")
        return self


class DurationTrainingBatch(FrozenModel):
    """ATP- or WTA-specific B5 observations and the full rejection audit."""

    schema_version: Literal["duration-training-batch/v1"] = "duration-training-batch/v1"
    tour: Tour
    information_cutoff_utc: datetime
    config_sha256: str
    observations: tuple[DurationTrainingObservation, ...]
    exclusions: tuple[DurationExclusion, ...]
    audit: DurationTrainingAudit
    data_sha256: str

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="information_cutoff_utc")

    @field_validator("config_sha256", "data_sha256")
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def batch_is_coherent(self) -> Self:
        if any(item.tour is not self.tour for item in self.observations):
            raise ValueError("duration observations may not cross tours")
        keys = tuple(item.match_id for item in self.observations)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("observations must be unique and sorted by match_id")
        exclusion_keys = tuple((item.match_id, item.reason) for item in self.exclusions)
        if exclusion_keys != tuple(sorted(exclusion_keys)):
            raise ValueError("exclusions must be sorted by match_id and reason")
        if self.audit.included_match_count != len(self.observations):
            raise ValueError("audit included count differs from observations")
        if self.audit.excluded_match_count != len(self.exclusions):
            raise ValueError("audit excluded count differs from exclusions")
        payload = self.model_dump(mode="json", exclude={"data_sha256"})
        expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if self.data_sha256 != expected:
            raise ValueError("data_sha256 does not match duration batch content")
        return self


def _records(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        values = rows.to_dict(orient="records")
    else:
        values = list(rows)
    output: list[dict[str, Any]] = []
    for row in values:
        if isinstance(row, BaseModel):
            output.append(row.model_dump(mode="python"))
        elif isinstance(row, Mapping):
            output.append(dict(row))
        else:
            raise DurationModelError("duration input rows must be mappings or Pydantic models")
    return output


def _missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _parse_date(value: Any) -> date | None:
    if _missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if _missing(value):
        return None
    if isinstance(value, datetime):
        try:
            return _aware_utc(value, field_name="available_at_utc")
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return _aware_utc(parsed, field_name="available_at_utc")
    except ValueError:
        return None


def _parse_bool(value: Any) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "t", "yes", "y", "1", "indoor", "night"}:
        return True
    if text in {"false", "f", "no", "n", "0", "outdoor", "day"}:
        return False
    return None


def _reverse_score(score: str) -> str:
    """Reverse score orientation without changing status or tiebreak annotations."""

    output: list[str] = []
    for token in score.split():
        if token.upper() in {"RET", "RET.", "W/O", "WO", "DEF", "ABN", "ABD"}:
            output.append(token)
            continue
        bracketed = token.startswith("[") and token.endswith("]")
        body = token[1:-1] if bracketed else token
        annotation = ""
        if "(" in body and body.endswith(")"):
            body, annotation_tail = body.split("(", 1)
            annotation = f"({annotation_tail}"
        if "-" not in body:
            output.append(token)
            continue
        first, second = body.split("-", 1)
        reversed_token = f"{second}-{first}{annotation}"
        output.append(f"[{reversed_token}]" if bracketed else reversed_token)
    return " ".join(output)


def _validate_either_orientation(score: str | None, best_of: int) -> ScoreValidationResult:
    result = validate_score(score, best_of=best_of)
    if result.valid and result.completed:
        return result
    if score:
        reversed_result = validate_score(_reverse_score(score), best_of=best_of)
        if reversed_result.valid and reversed_result.completed:
            return reversed_result
    return result


def _first_consistent(rows: Sequence[dict[str, Any]], field: str) -> Any:
    values = [row.get(field) for row in rows]
    return _first_consistent_value(values, field=field)


def _first_consistent_value(values: Sequence[Any], *, field: str) -> Any:
    """Return one reciprocal match value or fail closed on disagreement."""

    if not values:
        raise DurationModelError(f"cannot resolve {field} from an empty match")
    first = values[0]
    if any(
        (_missing(item) != _missing(first)) or (not _missing(item) and item != first)
        for item in values[1:]
    ):
        raise DurationModelError(f"reciprocal rows disagree on {field}")
    return first


def _match_sources(
    rows: Sequence[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    source_ids: set[str] = set()
    for row in rows:
        raw_source_id = row.get("source_id")
        if _missing(raw_source_id) or not str(raw_source_id).strip():
            raise DurationModelError("included match has no truthful source_id")
        source_ids.add(str(raw_source_id).strip())
    ids = tuple(sorted(source_ids))
    hashes: set[str] = set()
    crosswalks: set[str] = set()
    for row in rows:
        raw_hash = row.get("snapshot_sha256") or row.get("source_sha256")
        if not _missing(raw_hash):
            hashes.add(_sha256(str(raw_hash), field_name="snapshot_sha256"))
        match_date_hash = row.get("match_date_source_sha256")
        if not _missing(match_date_hash):
            normalized = _sha256(str(match_date_hash), field_name="match_date_source_sha256")
            if normalized not in hashes:
                crosswalks.add(normalized)
        for field in (
            "exact_date_crosswalk_sha256",
            "match_date_crosswalk_sha256",
            "crosswalk_sha256",
        ):
            value = row.get(field)
            if not _missing(value):
                crosswalks.add(_sha256(str(value), field_name=field))
    if not hashes:
        raise DurationModelError("included match has no source snapshot hash")
    return ids, tuple(sorted(hashes)), tuple(sorted(crosswalks))


def _exclusion(
    match_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    tour: Tour | None,
    reason: str,
    details: str,
    match_date: date | None = None,
) -> DurationExclusion:
    ids = tuple(
        sorted({str(row.get("source_id")) for row in rows if not _missing(row.get("source_id"))})
    )
    hashes: list[str] = []
    for row in rows:
        value = row.get("snapshot_sha256") or row.get("source_sha256")
        if not _missing(value):
            with suppress(ValueError):
                hashes.append(_sha256(str(value), field_name="source hash"))
    return DurationExclusion(
        match_id=match_id,
        tour=tour,
        reason=reason,
        details=details,
        match_date=match_date,
        source_ids=ids,
        source_sha256s=tuple(sorted(set(hashes))),
    )


def build_duration_training_batch(
    rows: Any,
    *,
    tour: Tour,
    information_cutoff_utc: datetime,
    config: DurationModelConfig,
) -> DurationTrainingBatch:
    """Construct B5 matches from reciprocal normalized player rows.

    Excluded records are never repaired, clipped, imputed, or converted to zero.
    The latest accepted availability timestamp must be strictly before the cutoff.
    """

    cutoff = _aware_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    records = _records(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[str(row.get("match_id") or f"missing-match-id:{index}")].append(row)

    observations: list[DurationTrainingObservation] = []
    exclusions: list[DurationExclusion] = []
    retirement_count = 0
    anomaly_count = 0
    match_tiebreak_count = 0
    for match_id, match_rows in sorted(grouped.items()):
        row_tours: set[Tour] = set()
        for row in match_rows:
            with suppress(ValueError):
                row_tours.add(Tour(str(row.get("tour"))))
        row_tour = next(iter(row_tours)) if len(row_tours) == 1 else None
        if row_tour is not tour:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=row_tour,
                    reason="WRONG_OR_AMBIGUOUS_TOUR",
                    details="match does not belong exclusively to requested tour",
                )
            )
            anomaly_count += 1
            continue
        if len(match_rows) != 2:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="RECIPROCAL_ROW_COUNT",
                    details=f"expected two player rows; found {len(match_rows)}",
                )
            )
            anomaly_count += 1
            continue
        player_ids = [str(row.get("player_id") or "") for row in match_rows]
        opponent_ids = [str(row.get("opponent_id") or "") for row in match_rows]
        if (
            not all(player_ids)
            or len(set(player_ids)) != 2
            or set(player_ids) != set(opponent_ids)
            or any(row.get("player_id") == row.get("opponent_id") for row in match_rows)
        ):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="NONRECIPROCAL_PLAYERS",
                    details="player/opponent identities are not reciprocal",
                )
            )
            anomaly_count += 1
            continue

        match_date = _parse_date(match_rows[0].get("match_date"))
        if match_date is None or any(
            _parse_date(row.get("match_date")) != match_date for row in match_rows
        ):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="MISSING_OR_INCONSISTENT_EXACT_DATE",
                    details="verified exact match date is absent or inconsistent",
                )
            )
            continue
        availabilities = [_parse_datetime(row.get("available_at_utc")) for row in match_rows]
        if any(value is None for value in availabilities):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="MISSING_OR_INVALID_AVAILABILITY",
                    details="availability timestamp is absent or invalid",
                    match_date=match_date,
                )
            )
            continue
        latest_available = max(value for value in availabilities if value is not None)
        if latest_available >= cutoff:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="NOT_AVAILABLE_BEFORE_CUTOFF",
                    details="source record was not available strictly before cutoff",
                    match_date=match_date,
                )
            )
            continue
        age_days = (cutoff.date() - match_date).days
        if age_days < 0:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="FUTURE_MATCH_DATE",
                    details="match date follows information cutoff",
                    match_date=match_date,
                )
            )
            continue
        if age_days > config.window_days:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="OUTSIDE_FIT_WINDOW",
                    details=f"age {age_days} days exceeds {config.window_days}",
                    match_date=match_date,
                )
            )
            continue

        if any(bool(_parse_bool(row.get("retirement"))) for row in match_rows):
            retirement_count += 1
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="RETIREMENT_EXCLUDED_FROM_ORDINARY_FIT",
                    details="retirement retained for explicit audit, not ordinary duration fit",
                    match_date=match_date,
                )
            )
            continue
        if any(bool(_parse_bool(row.get("walkover"))) for row in match_rows):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="WALKOVER_OR_PRESTART",
                    details="walkover/prestart termination is not a completed match",
                    match_date=match_date,
                )
            )
            continue
        if any(_parse_bool(row.get("completed")) is not True for row in match_rows):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="NOT_NORMAL_COMPLETION",
                    details="ordinary fit requires both rows marked completed",
                    match_date=match_date,
                )
            )
            continue

        try:
            best_of = int(_first_consistent(match_rows, "best_of"))
            raw_score = _first_consistent(match_rows, "score")
            raw_duration = _first_consistent(match_rows, "duration_minutes")
        except (DurationModelError, TypeError, ValueError) as exc:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="INCONSISTENT_MATCH_FIELDS",
                    details=str(exc),
                    match_date=match_date,
                )
            )
            anomaly_count += 1
            continue
        score = _validate_either_orientation(
            None if _missing(raw_score) else str(raw_score), best_of
        )
        if not score.valid or not score.completed or score.official_games is None:
            details = ",".join(score.anomaly_codes) or "score not a legal completed match"
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="ILLEGAL_OR_INCOMPLETE_SCORE",
                    details=details,
                    match_date=match_date,
                )
            )
            anomaly_count += 1
            continue
        try:
            duration = float(raw_duration)
            service_points = [int(str(row.get("service_points"))) for row in match_rows]
        except (TypeError, ValueError):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="MISSING_OR_NONPOSITIVE_EXPOSURE",
                    details="duration or service points is missing/non-numeric",
                    match_date=match_date,
                )
            )
            continue
        total_points = sum(service_points)
        regular_sets = sum(item.kind is SetScoreKind.REGULAR for item in score.sets)
        has_match_tiebreak = any(item.kind is SetScoreKind.MATCH_TIEBREAK for item in score.sets)
        sets = len(score.sets)
        tiebreaks = sum(
            (item.kind is SetScoreKind.MATCH_TIEBREAK)
            or (
                item.kind is SetScoreKind.REGULAR
                and {item.player_score, item.opponent_score} == {6, 7}
            )
            for item in score.sets
        )
        if (
            not math.isfinite(duration)
            or duration <= 0
            or total_points <= 0
            or score.official_games <= 0
            or sets <= 0
            or regular_sets <= 0
        ):
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="MISSING_OR_NONPOSITIVE_EXPOSURE",
                    details="T, P, G, and S must all be strictly positive",
                    match_date=match_date,
                )
            )
            continue
        try:
            source_ids, source_hashes, crosswalk_hashes = _match_sources(match_rows)
        except (DurationModelError, ValueError) as exc:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="INVALID_OR_MISSING_PROVENANCE",
                    details=str(exc),
                    match_date=match_date,
                )
            )
            anomaly_count += 1
            continue

        try:
            raw_indoor = _first_consistent(match_rows, "indoor")
            indoor = None if _missing(raw_indoor) else _parse_bool(raw_indoor)
            if not _missing(raw_indoor) and indoor is None:
                raise DurationModelError("indoor has an invalid boolean value")

            raw_temperature = _first_consistent(match_rows, "temperature_c")
            temperature: float | None = None
            if not _missing(raw_temperature):
                temperature = float(raw_temperature)
                if not math.isfinite(temperature):
                    raise DurationModelError("temperature_c must be finite when supplied")

            session_values: list[Any] = []
            for row in match_rows:
                raw_session = row.get("night_session")
                if _missing(raw_session):
                    raw_session = row.get("session")
                session_values.append(raw_session)
            raw_session = _first_consistent_value(session_values, field="night_session/session")
            night = None if _missing(raw_session) else _parse_bool(raw_session)
            if not _missing(raw_session) and night is None:
                raise DurationModelError("night_session/session has an invalid value")

            raw_event = _first_consistent(match_rows, "event")
            raw_event_year = _first_consistent(match_rows, "event_year")
            if _missing(raw_event) != _missing(raw_event_year):
                raise DurationModelError("event and event_year must be supplied together")
            event: str | None = None
            event_year: int | None = None
            if not _missing(raw_event):
                event = str(raw_event)
                if not event.strip():
                    raise DurationModelError("event must be nonblank when supplied")
                if isinstance(raw_event_year, (bool, np.bool_)):
                    raise DurationModelError("event_year must be an integer")
                numeric_event_year = float(raw_event_year)
                if not math.isfinite(numeric_event_year) or not numeric_event_year.is_integer():
                    raise DurationModelError("event_year must be a finite integer")
                event_year = int(numeric_event_year)
        except (DurationModelError, TypeError, ValueError) as exc:
            exclusions.append(
                _exclusion(
                    match_id,
                    match_rows,
                    tour=tour,
                    reason="INCONSISTENT_OR_INVALID_CONTEXT",
                    details=str(exc),
                    match_date=match_date,
                )
            )
            anomaly_count += 1
            continue
        player_a, player_b = sorted(player_ids)
        weight = 2.0 ** (-age_days / config.half_life_days)
        observation = DurationTrainingObservation(
            match_id=match_id,
            tour=tour,
            match_date=match_date,
            available_at_utc=latest_available,
            player_a_id=player_a,
            player_b_id=player_b,
            duration_minutes=duration,
            total_points=total_points,
            official_games=score.official_games,
            sets=sets,
            tiebreaks=tiebreaks,
            match_tiebreak_format=has_match_tiebreak,
            conditions=DurationConditions(
                indoor=indoor,
                temperature_c=temperature,
                night_session=night,
                event=event,
                event_year=event_year,
            ),
            recency_weight=weight,
            source_ids=source_ids,
            source_sha256s=source_hashes,
            crosswalk_sha256s=crosswalk_hashes,
        )
        observations.append(observation)
        match_tiebreak_count += int(has_match_tiebreak)

    observations.sort(key=lambda item: item.match_id)
    exclusions.sort(key=lambda item: (item.match_id, item.reason))
    reason_counts = tuple(
        DurationReasonCount(reason=reason, match_count=count)
        for reason, count in sorted(Counter(item.reason for item in exclusions).items())
    )
    audit = DurationTrainingAudit(
        input_row_count=len(records),
        candidate_match_count=len(grouped),
        included_match_count=len(observations),
        excluded_match_count=len(exclusions),
        retirement_match_count=retirement_count,
        anomaly_match_count=anomaly_count,
        match_tiebreak_format_included_count=match_tiebreak_count,
        reason_counts=reason_counts,
    )
    payload: dict[str, Any] = {
        "schema_version": DURATION_TRAINING_SCHEMA_VERSION,
        "tour": tour,
        "information_cutoff_utc": cutoff,
        "config_sha256": config.sha256,
        "observations": tuple(observations),
        "exclusions": tuple(exclusions),
        "audit": audit,
    }
    provisional = DurationTrainingBatch.model_construct(data_sha256="0" * 64, **payload)
    digest = hashlib.sha256(
        _canonical_json_bytes(provisional.model_dump(mode="json", exclude={"data_sha256"}))
    ).hexdigest()
    return DurationTrainingBatch(data_sha256=digest, **payload)


class DurationPathExposure(FrozenModel):
    """Realised exposure from one simulated match path."""

    tour: Tour
    player_a_id: NonEmptyStr
    player_b_id: NonEmptyStr
    total_points: Annotated[int, Field(gt=0)]
    official_games: Annotated[int, Field(gt=0)]
    sets: Annotated[int, Field(gt=0)]
    tiebreaks: Annotated[int, Field(ge=0)]
    conditions: DurationConditions = DurationConditions()

    @model_validator(mode="after")
    def players_are_distinct(self) -> Self:
        if self.player_a_id == self.player_b_id:
            raise ValueError("duration exposure players must differ")
        return self


class DurationCoefficient(FrozenModel):
    name: NonEmptyStr
    value: float
    standard_error: Annotated[float, Field(ge=0)]

    @field_validator("value", "standard_error")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationPaceEffect(FrozenModel):
    player_id: NonEmptyStr
    value: float
    standard_error: Annotated[float, Field(ge=0)]
    weighted_matches: Annotated[float, Field(ge=0)]

    @field_validator("value", "standard_error", "weighted_matches")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationContextStatus(FrozenModel):
    name: NonEmptyStr
    active: bool
    reason: NonEmptyStr


class DurationPosteriorApproximation(FrozenModel):
    """Audited MAP-Laplace Gaussian approximation for the stored duration vector."""

    method: Literal["finite-difference-map-gradient-laplace/v1"]
    curvature_parameterization: Literal["beta-helmert-pace-log-residual/v1"]
    curvature_parameter_count: Annotated[int, Field(gt=0)]
    curvature_relative_step: Annotated[float, Field(gt=0)]
    curvature_eigenvalue_floor: Annotated[float, Field(gt=0)]
    curvature_full_max_parameters: Annotated[int, Field(gt=0)]
    raw_min_eigenvalue: float
    regularization_added: Annotated[float, Field(ge=0)]
    condition_number: Annotated[float, Field(gt=0)]
    parameter_names: tuple[NonEmptyStr, ...]
    map_values: tuple[float, ...]
    covariance_mode: Literal["full", "diagonal"]
    covariance: tuple[tuple[float, ...], ...]

    @field_validator("map_values")
    @classmethod
    def map_is_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("posterior map_values must be finite")
        return value

    @field_validator(
        "curvature_relative_step",
        "curvature_eigenvalue_floor",
        "raw_min_eigenvalue",
        "regularization_added",
        "condition_number",
    )
    @classmethod
    def curvature_values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def dimensions_are_coherent(self) -> Self:
        n = len(self.parameter_names)
        if n == 0 or len(set(self.parameter_names)) != n or len(self.map_values) != n:
            raise ValueError("posterior parameter names and values are inconsistent")
        if self.curvature_parameter_count != n - 1:
            raise ValueError(
                "centered pace coordinates must remove exactly one curvature parameter"
            )
        expected_mode = (
            "full"
            if self.curvature_parameter_count <= self.curvature_full_max_parameters
            else "diagonal"
        )
        if self.covariance_mode != expected_mode:
            raise ValueError("posterior covariance mode disagrees with curvature threshold")
        if self.covariance_mode == "full":
            if len(self.covariance) != n or any(len(row) != n for row in self.covariance):
                raise ValueError("full posterior covariance has invalid dimensions")
        elif len(self.covariance) != n or any(len(row) != 1 for row in self.covariance):
            raise ValueError("diagonal posterior covariance must store one variance per row")
        flat = [item for row in self.covariance for item in row]
        if not all(math.isfinite(item) for item in flat):
            raise ValueError("posterior covariance must be finite")
        matrix = np.asarray(self.covariance, dtype=float)
        if self.covariance_mode == "full":
            if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
                raise ValueError("posterior covariance must be symmetric")
            if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-8:
                raise ValueError("posterior covariance must be positive semidefinite")
        elif np.any(matrix[:, 0] < 0):
            raise ValueError("posterior variances must be nonnegative")
        return self


class DurationFitDiagnostics(FrozenModel):
    objective_name: Literal["recency-weighted-heteroskedastic-student-t-map/v1"]
    converged: bool
    optimizer_status: int
    optimizer_message: NonEmptyStr
    iterations: Annotated[int, Field(ge=0)]
    objective_value: float
    gradient_infinity_norm: Annotated[float, Field(ge=0)]
    observation_count: Annotated[int, Field(gt=0)]
    effective_observation_weight: Annotated[float, Field(gt=0)]
    regularized_design_condition_number: Annotated[float, Field(gt=0)]

    @field_validator(
        "objective_value",
        "gradient_infinity_norm",
        "effective_observation_weight",
        "regularized_design_condition_number",
    )
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class DurationFitArtifact(FrozenModel):
    """Complete content-addressed ATP- or WTA-specific B5 fit."""

    artifact_id: str
    schema_version: Literal["duration-fit-artifact/v1"] = "duration-fit-artifact/v1"
    framework_version: Literal["v1.0"] = "v1.0"
    tour: Tour
    source_manifest_id: NonEmptyStr
    source_manifest_sha256: str
    information_cutoff_utc: datetime
    fit_cutoff_utc: datetime
    fitted_at_utc: datetime
    training_start_date: date
    training_end_date: date
    window_days: Annotated[int, Field(gt=0)]
    half_life_days: Annotated[float, Field(gt=0)]
    ridge: DurationRidgeConfig
    residual_config: DurationResidualConfig
    coefficients: tuple[DurationCoefficient, ...]
    pace_effects: tuple[DurationPaceEffect, ...]
    context_status: tuple[DurationContextStatus, ...]
    temperature_reference_c: float
    current_event: NonEmptyStr
    current_event_year: Annotated[int, Field(ge=2000)]
    sigma0: Annotated[float, Field(gt=0)]
    sigma1: Annotated[float, Field(ge=0)]
    nu: Annotated[float, Field(gt=2)]
    posterior: DurationPosteriorApproximation
    diagnostics: DurationFitDiagnostics
    training_audit: DurationTrainingAudit
    source_sha256s: tuple[str, ...]
    crosswalk_sha256s: tuple[str, ...]
    software_version: NonEmptyStr
    config_sha256: str
    data_sha256: str
    code_sha256: str
    deterministic_test_result_sha256: str

    @field_validator(
        "artifact_id",
        "source_manifest_sha256",
        "config_sha256",
        "data_sha256",
        "code_sha256",
        "deterministic_test_result_sha256",
    )
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field_name=info.field_name)

    @field_validator("source_sha256s", "crosswalk_sha256s")
    @classmethod
    def hash_lists_are_valid(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        normalized = tuple(_sha256(item, field_name=info.field_name) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError(f"{info.field_name} must be unique and sorted")
        if info.field_name == "source_sha256s" and not normalized:
            raise ValueError("source_sha256s must be nonempty")
        return normalized

    @field_validator("information_cutoff_utc", "fit_cutoff_utc", "fitted_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _aware_utc(value, field_name=info.field_name)

    @field_validator("half_life_days", "temperature_reference_c", "sigma0", "sigma1", "nu")
    @classmethod
    def numeric_values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def contract_and_hash_are_valid(self) -> Self:
        if self.fit_cutoff_utc != self.information_cutoff_utc:
            raise ValueError("duration fit cutoff must equal information cutoff")
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("fitted_at_utc must not precede information cutoff")
        if self.training_start_date > self.training_end_date:
            raise ValueError("duration training date range is reversed")
        if self.training_end_date > self.information_cutoff_utc.date():
            raise ValueError("duration training end date follows its cutoff")
        if (self.information_cutoff_utc.date() - self.training_start_date).days > self.window_days:
            raise ValueError("duration training start date falls outside the configured window")
        if self.diagnostics.observation_count != self.training_audit.included_match_count:
            raise ValueError("duration diagnostics and training audit counts do not reconcile")
        names = tuple(item.name for item in self.coefficients)
        if names[: len(_CORE_COEFFICIENTS)] != _CORE_COEFFICIENTS or len(names) != len(set(names)):
            raise ValueError("duration coefficients must begin with the frozen core order")
        players = tuple(item.player_id for item in self.pace_effects)
        if players != tuple(sorted(players)) or len(players) != len(set(players)):
            raise ValueError("pace effects must be unique and sorted")
        if self.pace_effects and not math.isclose(
            math.fsum(item.value for item in self.pace_effects), 0.0, abs_tol=1e-8
        ):
            raise ValueError("player pace effects must be centered")
        contexts = tuple(item.name for item in self.context_status)
        if contexts != tuple(name for name in _CONTEXT_COEFFICIENTS if name in contexts):
            raise ValueError("context status must follow canonical order")
        active_contexts = tuple(item.name for item in self.context_status if item.active)
        if names[len(_CORE_COEFFICIENTS) :] != active_contexts:
            raise ValueError("active duration contexts do not agree with coefficient vector")
        if any((item.reason == "active") != item.active for item in self.context_status):
            raise ValueError("duration context status reason is internally inconsistent")
        expected_posterior_names = (
            tuple(f"coefficient:{item.name}" for item in self.coefficients)
            + tuple(f"pace:{item.player_id}" for item in self.pace_effects)
            + ("log_sigma0", "log_sigma1", "log_nu_minus_two")
        )
        if self.posterior.parameter_names != expected_posterior_names:
            raise ValueError("duration posterior parameter names do not match fitted parameters")
        if self.sigma1 <= 0:
            raise ValueError("log-scale posterior representation requires sigma1 > 0")
        expected_map = (
            tuple(item.value for item in self.coefficients)
            + tuple(item.value for item in self.pace_effects)
            + (math.log(self.sigma0), math.log(self.sigma1), math.log(self.nu - 2.0))
        )
        if any(
            not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10)
            for observed, expected in zip(self.posterior.map_values, expected_map, strict=True)
        ):
            raise ValueError("duration posterior MAP does not match stored fitted parameters")
        expected = hashlib.sha256(
            _canonical_json_bytes(self.model_dump(mode="json", exclude={"artifact_id"}))
        ).hexdigest()
        if self.artifact_id != expected:
            raise ValueError("artifact_id does not match duration fit content")
        return self


def _context_activation(
    observations: Sequence[DurationTrainingObservation], config: DurationModelConfig
) -> tuple[tuple[str, ...], tuple[DurationContextStatus, ...]]:
    active: list[str] = []
    statuses: list[DurationContextStatus] = []
    minimum = config.context.minimum_level_rows
    for name in config.context.supported_terms:
        values: list[float | None] = []
        for item in observations:
            conditions = item.conditions
            if name == "indoor":
                values.append(None if conditions.indoor is None else float(conditions.indoor))
            elif name == "temperature_c":
                values.append(
                    None
                    if conditions.temperature_c is None
                    else conditions.temperature_c - config.context.temperature_reference_c
                )
            elif name == "night_session":
                values.append(
                    None if conditions.night_session is None else float(conditions.night_session)
                )
            else:
                values.append(
                    float(
                        conditions.event == config.context.current_event
                        and conditions.event_year == config.context.current_event_year
                    )
                    if conditions.event is not None
                    else None
                )
        numeric = [float(value) for value in values if value is not None]
        unique = sorted(set(numeric))
        if not numeric:
            reason = "inactive: context has no known values"
        elif len(unique) < 2:
            reason = "inactive: known context values have no variation"
        elif name in {"temperature_c", "current_usopen_2026"}:
            # Temperature uses its configured reference as the zero contribution.
            # The current-event indicator has a deliberately strong fixed prior,
            # so even one genuine current row and one known baseline row identify
            # a shrunk update. Missing source values remain missing in the audit;
            # they merely contribute the explicit zero design baseline below.
            reason = "active"
            active.append(name)
        else:
            counts = Counter(numeric)
            if min(counts.values()) < minimum:
                reason = f"inactive: a known level has fewer than {minimum} matches"
            else:
                reason = "active"
                active.append(name)
        statuses.append(DurationContextStatus(name=name, active=name in active, reason=reason))
    return tuple(active), tuple(statuses)


def _design(
    observations: Sequence[DurationTrainingObservation],
    active_contexts: tuple[str, ...],
    config: DurationModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names = _CORE_COEFFICIENTS + active_contexts
    rows: list[list[float]] = []
    for item in observations:
        values = [
            1.0,
            float(item.total_points),
            float(item.official_games),
            float(item.sets),
            float(item.tiebreaks),
        ]
        for name in active_contexts:
            if name == "indoor":
                values.append(float(bool(item.conditions.indoor)))
            elif name == "temperature_c":
                values.append(
                    0.0
                    if item.conditions.temperature_c is None
                    else item.conditions.temperature_c - config.context.temperature_reference_c
                )
            elif name == "night_session":
                values.append(float(bool(item.conditions.night_session)))
            else:
                values.append(
                    float(
                        item.conditions.event == config.context.current_event
                        and item.conditions.event_year == config.context.current_event_year
                    )
                )
        rows.append(values)
    return (
        np.asarray(rows, dtype=float),
        np.asarray([item.duration_minutes for item in observations], dtype=float),
        np.asarray([item.recency_weight for item in observations], dtype=float),
        names,
    )


def _prior_sds(names: tuple[str, ...], config: DurationModelConfig) -> np.ndarray:
    mapping = {
        "theta0": config.ridge.intercept_sd,
        "points": config.ridge.points_sd,
        "games": config.ridge.games_sd,
        "sets": config.ridge.sets_sd,
        "tiebreaks": config.ridge.tiebreaks_sd,
        "indoor": config.ridge.indoor_sd,
        "temperature_c": config.ridge.temperature_sd,
        "night_session": config.ridge.night_session_sd,
        "current_usopen_2026": config.ridge.current_event_sd,
    }
    return np.asarray([mapping[name] for name in names], dtype=float)


def _student_objective(
    parameters: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    incidence: np.ndarray,
    contrast: np.ndarray,
    beta_sds: np.ndarray,
    config: DurationModelConfig,
) -> tuple[float, np.ndarray]:
    k = x.shape[1]
    z_count = contrast.shape[1]
    beta = parameters[:k]
    z_effect = parameters[k : k + z_count]
    residual_parameters = parameters[k + z_count :]
    log_sigma0, log_sigma1, log_nu_minus_two = residual_parameters
    sigma0 = math.exp(float(np.clip(log_sigma0, -20.0, 20.0)))
    sigma1 = math.exp(float(np.clip(log_sigma1, -20.0, 20.0)))
    nu_minus_two = math.exp(float(np.clip(log_nu_minus_two, -20.0, 20.0)))
    nu = 2.0 + nu_minus_two
    pace = contrast @ z_effect if z_count else np.zeros(incidence.shape[1])
    mu = x @ beta + incidence @ pace
    scale = sigma0 + sigma1 * np.sqrt(np.asarray([max(value, 1.0) for value in x[:, 1]]))
    error = y - mu
    z2 = np.square(error / scale)
    per_row = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - np.log(scale)
        - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    )
    nll = -float(np.dot(weights, per_row))
    nll += 0.5 * float(np.sum(np.square(beta / beta_sds)))
    if z_count:
        nll += 0.5 * float(np.sum(np.square(z_effect / config.ridge.player_pace_sd)))
    residual_centers = np.log(
        np.asarray(
            [
                config.residual.initial_sigma0,
                config.residual.initial_sigma1,
                config.residual.initial_nu - 2.0,
            ]
        )
    )
    residual_sds = np.asarray(
        [
            config.residual.log_sigma0_prior_sd,
            config.residual.log_sigma1_prior_sd,
            config.residual.log_nu_minus_two_prior_sd,
        ]
    )
    nll += 0.5 * float(np.sum(np.square((residual_parameters - residual_centers) / residual_sds)))

    denominator = nu * np.square(scale) + np.square(error)
    grad_mu = -(nu + 1.0) * error / denominator * weights
    gradient_beta = x.T @ grad_mu + beta / np.square(beta_sds)
    gradient_pace = incidence.T @ grad_mu
    gradient_z = (
        contrast.T @ gradient_pace + z_effect / config.ridge.player_pace_sd**2
        if z_count
        else np.empty(0)
    )
    grad_scale = weights * (1.0 - (nu + 1.0) * np.square(error) / denominator) / scale
    gradient_log_sigma0 = float(np.sum(grad_scale) * sigma0)
    gradient_log_sigma1 = float(np.dot(grad_scale, np.sqrt(x[:, 1])) * sigma1)
    derivative_nu = (
        -0.5 * digamma((nu + 1.0) / 2.0)
        + 0.5 * digamma(nu / 2.0)
        + 0.5 / nu
        + 0.5 * np.log1p(z2 / nu)
        - (nu + 1.0) * z2 / (2.0 * nu * (nu + z2))
    )
    gradient_log_nu = float(np.dot(weights, derivative_nu) * nu_minus_two)
    residual_gradient = np.asarray(
        [
            gradient_log_sigma0,
            gradient_log_sigma1,
            gradient_log_nu,
        ]
    ) + (residual_parameters - residual_centers) / np.square(residual_sds)
    return nll, np.concatenate((gradient_beta, gradient_z, residual_gradient))


def _positive_covariance(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def fit_duration_model(
    batch: DurationTrainingBatch,
    *,
    config: DurationModelConfig,
    source_manifest_id: str,
    source_manifest_sha256: str,
    fitted_at_utc: datetime,
    software_version: str,
    code_sha256: str,
    deterministic_test_result_sha256: str,
) -> DurationFitArtifact:
    """Fit the centered, heteroskedastic Student-t B5 model for one tour."""

    fitted_at = _aware_utc(fitted_at_utc, field_name="fitted_at_utc")
    if fitted_at < batch.information_cutoff_utc:
        raise DurationModelError("fitted_at_utc must not precede information cutoff")
    if batch.config_sha256 != config.sha256:
        raise DurationModelError("training batch was built under a different configuration")
    if not batch.observations:
        raise DurationModelError("duration model requires at least one eligible match")
    observations = batch.observations
    active_contexts, context_status = _context_activation(observations, config)
    x, y, weights, coefficient_names = _design(observations, active_contexts, config)
    players = tuple(
        sorted(
            {item.player_a_id for item in observations}
            | {item.player_b_id for item in observations}
        )
    )
    player_index = {player: index for index, player in enumerate(players)}
    incidence = np.zeros((len(observations), len(players)), dtype=float)
    weighted_matches = np.zeros(len(players), dtype=float)
    for row_index, item in enumerate(observations):
        for player in (item.player_a_id, item.player_b_id):
            incidence[row_index, player_index[player]] = 1.0
            weighted_matches[player_index[player]] += item.recency_weight
    contrast = helmert(len(players), full=False).T if len(players) > 1 else np.empty((1, 0))
    beta_sds = _prior_sds(coefficient_names, config)
    ridge = np.diag(1.0 / np.square(beta_sds))
    initial_beta = np.linalg.solve(x.T @ (weights[:, None] * x) + ridge, x.T @ (weights * y))
    initial = np.concatenate(
        (
            initial_beta,
            np.zeros(contrast.shape[1]),
            np.log(
                np.asarray(
                    [
                        config.residual.initial_sigma0,
                        config.residual.initial_sigma1,
                        config.residual.initial_nu - 2.0,
                    ]
                )
            ),
        )
    )

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        return _student_objective(
            value,
            x=x,
            y=y,
            weights=weights,
            incidence=incidence,
            contrast=contrast,
            beta_sds=beta_sds,
            config=config,
        )

    result = minimize(
        objective,
        initial,
        jac=True,
        method=config.optimizer.method,
        options={
            "maxiter": config.optimizer.max_iterations,
            "maxls": config.optimizer.max_line_search_steps,
            "gtol": config.optimizer.gradient_tolerance,
            "ftol": config.optimizer.function_tolerance,
        },
    )
    if not np.all(np.isfinite(result.x)):
        raise DurationModelError("duration optimizer returned non-finite parameters")
    if not bool(result.success):
        raise DurationModelError(
            "duration MAP optimization did not converge: "
            f"status={int(result.status)} message={result.message!s}"
        )
    map_unconstrained = np.asarray(result.x, dtype=float)
    k = len(coefficient_names)
    z_count = contrast.shape[1]
    beta = map_unconstrained[:k]
    pace = contrast @ map_unconstrained[k : k + z_count] if z_count else np.zeros(len(players))
    log_sigma0, log_sigma1, log_nu = map_unconstrained[k + z_count :]
    sigma0 = math.exp(log_sigma0)
    sigma1 = math.exp(log_sigma1)
    nu = 2.0 + math.exp(log_nu)

    def gradient(value: np.ndarray) -> np.ndarray:
        return objective(value)[1]

    try:
        curvature = laplace_curvature(
            gradient,
            map_unconstrained,
            max_full_parameters=config.optimizer.full_covariance_max_parameters,
            relative_step=config.optimizer.covariance_relative_step,
            eigenvalue_floor=config.optimizer.covariance_eigenvalue_floor,
        )
    except CurvatureError as exc:
        raise DurationModelError(f"duration posterior curvature failed: {exc}") from exc
    if curvature.covariance is None:
        raw_covariance = np.diag(curvature.variance_diagonal)
    else:
        raw_covariance = np.asarray(curvature.covariance, dtype=float)
    transform = np.zeros((k + len(players) + 3, len(map_unconstrained)), dtype=float)
    transform[:k, :k] = np.eye(k)
    if z_count:
        transform[k : k + len(players), k : k + z_count] = contrast
    transform[k + len(players) :, k + z_count :] = np.eye(3)
    # The Helmert transform makes the centered pace block deliberately singular.
    # Keep that identifying constraint in the published Gaussian approximation;
    # only negative numerical eigenvalues are removed here.
    covariance = _positive_covariance(transform @ raw_covariance @ transform.T, 0.0)
    posterior_names = (
        tuple(f"coefficient:{name}" for name in coefficient_names)
        + tuple(f"pace:{player}" for player in players)
        + ("log_sigma0", "log_sigma1", "log_nu_minus_two")
    )
    posterior_map = tuple(beta) + tuple(pace) + (log_sigma0, log_sigma1, log_nu)
    if curvature.kind == "full":
        covariance_mode: Literal["full", "diagonal"] = "full"
        stored_covariance = tuple(tuple(float(value) for value in row) for row in covariance)
    else:
        covariance_mode = "diagonal"
        stored_covariance = tuple((float(value),) for value in np.diag(covariance))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    coefficients = tuple(
        DurationCoefficient(
            name=name, value=float(value), standard_error=float(standard_errors[index])
        )
        for index, (name, value) in enumerate(zip(coefficient_names, beta, strict=True))
    )
    pace_effects = tuple(
        DurationPaceEffect(
            player_id=player,
            value=float(pace[index]),
            standard_error=float(standard_errors[k + index]),
            weighted_matches=float(weighted_matches[index]),
        )
        for index, player in enumerate(players)
    )
    _, final_gradient = objective(map_unconstrained)
    diagnostics = DurationFitDiagnostics(
        objective_name="recency-weighted-heteroskedastic-student-t-map/v1",
        converged=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message) or "optimizer returned no message",
        iterations=int(getattr(result, "nit", 0)),
        objective_value=float(result.fun),
        gradient_infinity_norm=float(np.linalg.norm(final_gradient, ord=np.inf)),
        observation_count=len(observations),
        effective_observation_weight=float(np.sum(weights)),
        regularized_design_condition_number=float(
            np.linalg.cond(x.T @ (weights[:, None] * x) + ridge)
        ),
    )
    source_hashes = tuple(sorted({value for item in observations for value in item.source_sha256s}))
    crosswalk_hashes = tuple(
        sorted({value for item in observations for value in item.crosswalk_sha256s})
    )
    payload: dict[str, Any] = {
        "schema_version": DURATION_ARTIFACT_SCHEMA_VERSION,
        "framework_version": "v1.0",
        "tour": batch.tour,
        "source_manifest_id": source_manifest_id,
        "source_manifest_sha256": _sha256(
            source_manifest_sha256, field_name="source_manifest_sha256"
        ),
        "information_cutoff_utc": batch.information_cutoff_utc,
        "fit_cutoff_utc": batch.information_cutoff_utc,
        "fitted_at_utc": fitted_at,
        "training_start_date": min(item.match_date for item in observations),
        "training_end_date": max(item.match_date for item in observations),
        "window_days": config.window_days,
        "half_life_days": config.half_life_days,
        "ridge": config.ridge,
        "residual_config": config.residual,
        "coefficients": coefficients,
        "pace_effects": pace_effects,
        "context_status": context_status,
        "temperature_reference_c": config.context.temperature_reference_c,
        "current_event": config.context.current_event,
        "current_event_year": config.context.current_event_year,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "nu": nu,
        "posterior": DurationPosteriorApproximation(
            method="finite-difference-map-gradient-laplace/v1",
            curvature_parameterization="beta-helmert-pace-log-residual/v1",
            curvature_parameter_count=len(map_unconstrained),
            curvature_relative_step=config.optimizer.covariance_relative_step,
            curvature_eigenvalue_floor=config.optimizer.covariance_eigenvalue_floor,
            curvature_full_max_parameters=config.optimizer.full_covariance_max_parameters,
            raw_min_eigenvalue=curvature.raw_min_eigenvalue,
            regularization_added=curvature.regularization_added,
            condition_number=curvature.condition_number,
            parameter_names=posterior_names,
            map_values=tuple(float(value) for value in posterior_map),
            covariance_mode=covariance_mode,
            covariance=stored_covariance,
        ),
        "diagnostics": diagnostics,
        "training_audit": batch.audit,
        "source_sha256s": source_hashes,
        "crosswalk_sha256s": crosswalk_hashes,
        "software_version": software_version,
        "config_sha256": config.sha256,
        "data_sha256": batch.data_sha256,
        "code_sha256": _sha256(code_sha256, field_name="code_sha256"),
        "deterministic_test_result_sha256": _sha256(
            deterministic_test_result_sha256, field_name="deterministic_test_result_sha256"
        ),
    }
    provisional = DurationFitArtifact.model_construct(artifact_id="0" * 64, **payload)
    artifact_id = hashlib.sha256(
        _canonical_json_bytes(provisional.model_dump(mode="json", exclude={"artifact_id"}))
    ).hexdigest()
    return DurationFitArtifact(artifact_id=artifact_id, **payload)


class PersistedDurationFitArtifact(FrozenModel):
    """Verified directory and parsed B5 artifact."""

    directory: Path
    artifact: DurationFitArtifact

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def artifact_path(self) -> Path:
        return self.directory / _ARTIFACT_FILENAME


def _artifact_bytes(artifact: DurationFitArtifact) -> bytes:
    return _canonical_json_bytes(artifact.model_dump(mode="json"))


def write_duration_fit_artifact(
    artifact: DurationFitArtifact, artifact_root: str | Path
) -> PersistedDurationFitArtifact:
    """Atomically publish an immutable duration fit without overwriting."""

    artifact = DurationFitArtifact.model_validate(artifact.model_dump(mode="python"))
    cutoff_segment = artifact.information_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    parent = Path(artifact_root).resolve() / artifact.tour.value.lower() / cutoff_segment
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact.artifact_id[:32]
    if target.exists():
        existing = load_duration_fit_artifact(target)
        if existing.artifact != artifact:
            raise DurationArtifactIntegrityError(
                "existing duration artifact path has conflicting content"
            )
        return existing
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        path = staging / _ARTIFACT_FILENAME
        try:
            with path.open("xb") as handle:
                handle.write(_artifact_bytes(artifact))
                handle.flush()
                os.fsync(handle.fileno())
            staging.rename(target)
        except OSError as exc:
            if not target.exists():
                raise DurationArtifactError(
                    f"cannot publish duration artifact {target}: {exc}"
                ) from exc
            existing = load_duration_fit_artifact(target)
            if existing.artifact != artifact:
                raise DurationArtifactIntegrityError(
                    "concurrent duration artifact publication conflicted"
                ) from exc
            return existing
        return load_duration_fit_artifact(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_duration_fit_artifact(directory: str | Path) -> PersistedDurationFitArtifact:
    """Load only canonical, schema-valid duration content from its addressed path."""

    artifact_directory = Path(directory)
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise DurationArtifactIntegrityError(
            f"duration artifact is not a regular directory: {directory}"
        )
    artifact_path = artifact_directory / _ARTIFACT_FILENAME
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise DurationArtifactIntegrityError(f"duration artifact JSON is missing: {artifact_path}")
    try:
        raw = artifact_path.read_bytes()
        artifact = DurationFitArtifact.model_validate_json(raw)
    except Exception as exc:
        raise DurationArtifactIntegrityError(f"duration artifact JSON is invalid: {exc}") from exc
    if raw != _artifact_bytes(artifact):
        raise DurationArtifactIntegrityError("duration artifact JSON is not canonical")
    if artifact_directory.name != artifact.artifact_id[:32]:
        raise DurationArtifactIntegrityError(
            "duration artifact directory does not match content identity"
        )
    return PersistedDurationFitArtifact(directory=artifact_directory, artifact=artifact)


class DurationParameterDraw(FrozenModel):
    """One explicit Gaussian parameter draw; residual randomness is separate."""

    artifact_id: str
    tour: Tour
    coefficients: tuple[DurationCoefficient, ...]
    pace_effects: tuple[DurationPaceEffect, ...]
    temperature_reference_c: float
    current_event: NonEmptyStr
    current_event_year: Annotated[int, Field(ge=2000)]
    sigma0: Annotated[float, Field(gt=0)]
    sigma1: Annotated[float, Field(ge=0)]
    nu: Annotated[float, Field(gt=2)]

    @field_validator("artifact_id")
    @classmethod
    def artifact_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="artifact_id")


def map_duration_parameters(artifact: DurationFitArtifact) -> DurationParameterDraw:
    """Return the deterministic MAP parameter vector without sampling."""

    return DurationParameterDraw(
        artifact_id=artifact.artifact_id,
        tour=artifact.tour,
        coefficients=artifact.coefficients,
        pace_effects=artifact.pace_effects,
        temperature_reference_c=artifact.temperature_reference_c,
        current_event=artifact.current_event,
        current_event_year=artifact.current_event_year,
        sigma0=artifact.sigma0,
        sigma1=artifact.sigma1,
        nu=artifact.nu,
    )


def sample_duration_parameters(
    artifact: DurationFitArtifact, rng: Generator
) -> DurationParameterDraw:
    """Draw model parameters using only the caller-owned RNG."""

    return sample_duration_parameters_for_players(
        artifact,
        tuple(item.player_id for item in artifact.pace_effects),
        rng,
    )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDurationParameterSampler:
    """One matchup's exact stored Gaussian marginal, prepared once for many paths."""

    artifact_id: str
    tour: Tour
    coefficient_templates: tuple[DurationCoefficient, ...]
    pace_templates: tuple[DurationPaceEffect, ...]
    temperature_reference_c: float
    current_event: str
    current_event_year: int
    mean: NDArray[np.float64]
    covariance_mode: Literal["full", "diagonal"]
    covariance_or_variance: NDArray[np.float64]

    def __post_init__(self) -> None:
        _sha256(self.artifact_id, field_name="artifact_id")
        expected = len(self.coefficient_templates) + len(self.pace_templates) + 3
        if self.mean.shape != (expected,) or not np.all(np.isfinite(self.mean)):
            raise DurationModelError("prepared duration mean has invalid shape or values")
        expected_shape = (expected, expected) if self.covariance_mode == "full" else (expected,)
        if self.covariance_or_variance.shape != expected_shape or not np.all(
            np.isfinite(self.covariance_or_variance)
        ):
            raise DurationModelError("prepared duration uncertainty has invalid shape or values")
        self.mean.setflags(write=False)
        self.covariance_or_variance.setflags(write=False)


def prepare_duration_parameter_sampler(
    artifact: DurationFitArtifact,
    player_ids: Sequence[str],
) -> PreparedDurationParameterSampler:
    """Prepare the exact target-player marginal without consuming randomness.

    Only core/active-context coefficients, known requested pace effects, and
    residual parameters are retained. Unknown target players keep the frozen
    zero pace effect. Full stored covariance materialization occurs exactly once
    per prepared matchup rather than once per simulated path.
    """

    posterior = artifact.posterior
    requested = set(player_ids)
    known_pace = tuple(item for item in artifact.pace_effects if item.player_id in requested)
    selected_names = (
        tuple(f"coefficient:{item.name}" for item in artifact.coefficients)
        + tuple(f"pace:{item.player_id}" for item in known_pace)
        + ("log_sigma0", "log_sigma1", "log_nu_minus_two")
    )
    all_index = {name: index for index, name in enumerate(posterior.parameter_names)}
    try:
        selected_indices = np.asarray([all_index[name] for name in selected_names], dtype=int)
    except KeyError as exc:
        raise DurationModelError(
            f"duration posterior omits required parameter {exc.args[0]}"
        ) from exc
    all_mean = np.asarray(posterior.map_values, dtype=float)
    mean = np.asarray(all_mean[selected_indices], dtype=np.float64).copy()
    if posterior.covariance_mode == "full":
        full_covariance = np.asarray(posterior.covariance, dtype=float)
        uncertainty = np.asarray(
            full_covariance[np.ix_(selected_indices, selected_indices)],
            dtype=np.float64,
        ).copy()
    else:
        all_variance = np.asarray([row[0] for row in posterior.covariance], dtype=float)
        uncertainty = np.asarray(all_variance[selected_indices], dtype=np.float64).copy()
    return PreparedDurationParameterSampler(
        artifact_id=artifact.artifact_id,
        tour=artifact.tour,
        coefficient_templates=artifact.coefficients,
        pace_templates=known_pace,
        temperature_reference_c=artifact.temperature_reference_c,
        current_event=artifact.current_event,
        current_event_year=artifact.current_event_year,
        mean=mean,
        covariance_mode=posterior.covariance_mode,
        covariance_or_variance=uncertainty,
    )


def sample_prepared_duration_parameters(
    prepared: PreparedDurationParameterSampler,
    rng: Generator,
) -> DurationParameterDraw:
    """Sample one prepared exact Gaussian marginal using only the supplied RNG."""

    if prepared.covariance_mode == "full":
        sampled = rng.multivariate_normal(
            prepared.mean,
            prepared.covariance_or_variance,
            check_valid="raise",
        )
    else:
        sampled = prepared.mean + rng.normal(size=len(prepared.mean)) * np.sqrt(
            np.maximum(prepared.covariance_or_variance, 0.0)
        )
    coefficient_count = len(prepared.coefficient_templates)
    pace_count = len(prepared.pace_templates)
    coefficients = tuple(
        DurationCoefficient(
            name=item.name,
            value=float(sampled[index]),
            standard_error=item.standard_error,
        )
        for index, item in enumerate(prepared.coefficient_templates)
    )
    pace_effects = tuple(
        DurationPaceEffect(
            player_id=item.player_id,
            value=float(sampled[coefficient_count + index]),
            standard_error=item.standard_error,
            weighted_matches=item.weighted_matches,
        )
        for index, item in enumerate(prepared.pace_templates)
    )
    tail = coefficient_count + pace_count
    return DurationParameterDraw(
        artifact_id=prepared.artifact_id,
        tour=prepared.tour,
        coefficients=coefficients,
        pace_effects=pace_effects,
        temperature_reference_c=prepared.temperature_reference_c,
        current_event=prepared.current_event,
        current_event_year=prepared.current_event_year,
        sigma0=math.exp(float(np.clip(sampled[tail], -20.0, 20.0))),
        sigma1=math.exp(float(np.clip(sampled[tail + 1], -20.0, 20.0))),
        nu=2.0 + math.exp(float(np.clip(sampled[tail + 2], -20.0, 20.0))),
    )


def sample_duration_parameters_for_players(
    artifact: DurationFitArtifact,
    player_ids: Sequence[str],
    rng: Generator,
) -> DurationParameterDraw:
    """Prepare and sample one exact target-player Gaussian marginal."""

    return sample_prepared_duration_parameters(
        prepare_duration_parameter_sampler(artifact, player_ids),
        rng,
    )


def _coefficient_map(value: DurationFitArtifact | DurationParameterDraw) -> dict[str, float]:
    return {item.name: item.value for item in value.coefficients}


def duration_predictor(
    parameters: DurationFitArtifact | DurationParameterDraw,
    exposure: DurationPathExposure,
) -> float:
    """Evaluate theta0 + thetaP P + thetaG G + thetaS S + thetaB B + q_i + q_j."""

    if parameters.tour is not exposure.tour:
        raise DurationModelError("duration parameters cannot be borrowed across tours")
    coefficients = _coefficient_map(parameters)
    required = set(_CORE_COEFFICIENTS)
    if not required.issubset(coefficients):
        raise DurationModelError("duration parameters omit a frozen core coefficient")
    center = (
        coefficients["theta0"]
        + coefficients["points"] * exposure.total_points
        + coefficients["games"] * exposure.official_games
        + coefficients["sets"] * exposure.sets
        + coefficients["tiebreaks"] * exposure.tiebreaks
    )
    pace = {item.player_id: item.value for item in parameters.pace_effects}
    center += pace.get(exposure.player_a_id, 0.0) + pace.get(exposure.player_b_id, 0.0)
    conditions = exposure.conditions
    if "indoor" in coefficients and conditions.indoor is not None:
        center += coefficients["indoor"] * float(conditions.indoor)
    if "temperature_c" in coefficients and conditions.temperature_c is not None:
        center += coefficients["temperature_c"] * (
            conditions.temperature_c - parameters.temperature_reference_c
        )
    if "night_session" in coefficients and conditions.night_session is not None:
        center += coefficients["night_session"] * float(conditions.night_session)
    if (
        "current_usopen_2026" in coefficients
        and conditions.event == parameters.current_event
        and conditions.event_year == parameters.current_event_year
    ):
        center += coefficients["current_usopen_2026"]
    return _finite(center, field_name="duration predictor")


def duration_scale(
    parameters: DurationFitArtifact | DurationParameterDraw, total_points: int
) -> float:
    """Evaluate the frozen positive heteroskedastic scale sigma0+sigma1 sqrt(P)."""

    if total_points <= 0:
        raise DurationModelError("total_points must be strictly positive")
    return _finite(
        parameters.sigma0 + parameters.sigma1 * math.sqrt(total_points),
        field_name="duration scale",
    )


class DurationDisplayMode(StrEnum):
    UNRESOLVED = "unresolved"
    FLOOR = "floor"
    NEAREST_HALF_UP = "nearest_half_up"


class DurationDisplayPolicy(FrozenModel):
    """Versioned mapping from continuous latent minutes to an official integer."""

    schema_version: Literal["duration-display-policy/v1"] = "duration-display-policy/v1"
    policy_version: NonEmptyStr
    mode: DurationDisplayMode


UNRESOLVED_DURATION_DISPLAY_POLICY = DurationDisplayPolicy(
    policy_version="duration-display-unresolved/v1", mode=DurationDisplayMode.UNRESOLVED
)
FLOOR_DURATION_DISPLAY_POLICY = DurationDisplayPolicy(
    policy_version="duration-display-floor/v1", mode=DurationDisplayMode.FLOOR
)
NEAREST_DURATION_DISPLAY_POLICY = DurationDisplayPolicy(
    policy_version="duration-display-nearest-half-up/v1",
    mode=DurationDisplayMode.NEAREST_HALF_UP,
)


class DurationDraw(FrozenModel):
    """Duration result preserving latent and candidate official-minute values."""

    schema_version: Literal["duration-draw/v1"] = "duration-draw/v1"
    artifact_id: str
    latent_minutes: Annotated[float, Field(gt=0)]
    official_minutes: Annotated[int | None, Field(gt=0)]
    candidate_official_minutes: tuple[Annotated[int, Field(gt=0)], ...]
    partial: bool
    center_minutes: float
    scale_minutes: Annotated[float, Field(gt=0)]
    standardized_residual: float
    display_policy: DurationDisplayPolicy

    @field_validator("artifact_id")
    @classmethod
    def artifact_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="artifact_id")

    @field_validator("latent_minutes", "center_minutes", "scale_minutes", "standardized_residual")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def display_is_coherent(self) -> Self:
        if self.candidate_official_minutes != tuple(sorted(set(self.candidate_official_minutes))):
            raise ValueError("candidate official minutes must be unique and sorted")
        if self.display_policy.mode is DurationDisplayMode.UNRESOLVED:
            if self.official_minutes is not None or len(self.candidate_official_minutes) not in {
                1,
                2,
            }:
                raise ValueError(
                    "unresolved display must preserve candidates and no official value"
                )
        elif self.official_minutes is None or self.candidate_official_minutes != (
            self.official_minutes,
        ):
            raise ValueError("resolved display must expose exactly its official minute")
        return self


def draw_duration(
    parameter_draw: DurationParameterDraw,
    exposure: DurationPathExposure,
    residual_rng: Generator,
    *,
    display_policy: DurationDisplayPolicy = UNRESOLVED_DURATION_DISPLAY_POLICY,
    partial: bool = False,
) -> DurationDraw:
    """Draw one centered Student-t residual with an explicit caller-owned RNG."""

    center = duration_predictor(parameter_draw, exposure)
    scale = duration_scale(parameter_draw, exposure.total_points)
    residual = float(residual_rng.standard_t(parameter_draw.nu))
    latent = max(1.0, center + scale * residual)
    floor_value = max(1, math.floor(latent))
    ceiling_value = max(1, math.ceil(latent))
    nearest_value = max(1, math.floor(latent + 0.5))
    official: int | None
    candidates: tuple[int, ...]
    if display_policy.mode is DurationDisplayMode.FLOOR:
        official = floor_value
        candidates = (official,)
    elif display_policy.mode is DurationDisplayMode.NEAREST_HALF_UP:
        official = nearest_value
        candidates = (official,)
    else:
        official = None
        candidates = tuple(sorted({floor_value, ceiling_value}))
    return DurationDraw(
        artifact_id=parameter_draw.artifact_id,
        latent_minutes=latent,
        official_minutes=official,
        candidate_official_minutes=candidates,
        partial=partial,
        center_minutes=center,
        scale_minutes=scale,
        standardized_residual=residual,
        display_policy=display_policy,
    )


__all__ = [
    "DURATION_ARTIFACT_SCHEMA_VERSION",
    "DURATION_CONFIG_SCHEMA_VERSION",
    "DURATION_DISPLAY_POLICY_SCHEMA_VERSION",
    "DURATION_DRAW_SCHEMA_VERSION",
    "DURATION_TRAINING_SCHEMA_VERSION",
    "FLOOR_DURATION_DISPLAY_POLICY",
    "NEAREST_DURATION_DISPLAY_POLICY",
    "UNRESOLVED_DURATION_DISPLAY_POLICY",
    "DurationArtifactError",
    "DurationArtifactIntegrityError",
    "DurationCoefficient",
    "DurationConditions",
    "DurationContextStatus",
    "DurationDisplayMode",
    "DurationDisplayPolicy",
    "DurationDraw",
    "DurationExclusion",
    "DurationFitArtifact",
    "DurationFitDiagnostics",
    "DurationModelConfig",
    "DurationModelError",
    "DurationPaceEffect",
    "DurationParameterDraw",
    "DurationPathExposure",
    "DurationPosteriorApproximation",
    "DurationReasonCount",
    "DurationTrainingAudit",
    "DurationTrainingBatch",
    "DurationTrainingObservation",
    "PersistedDurationFitArtifact",
    "PreparedDurationParameterSampler",
    "build_duration_training_batch",
    "draw_duration",
    "duration_predictor",
    "duration_scale",
    "fit_duration_model",
    "load_duration_fit_artifact",
    "load_duration_model_config",
    "map_duration_parameters",
    "prepare_duration_parameter_sampler",
    "sample_duration_parameters",
    "sample_duration_parameters_for_players",
    "sample_prepared_duration_parameters",
    "write_duration_fit_artifact",
]
