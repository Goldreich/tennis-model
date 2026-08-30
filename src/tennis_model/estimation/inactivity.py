"""Frozen C6 post-90-day inactivity records and posterior transforms.

This module implements only the deterministic C6 contract.  It deliberately
does not infer health, retirement, or a generic "rust" penalty from inactivity.
Callers must supply explicit player-effect coordinates: the fitted serve model's
reference-coded ``z`` representation is not itself one coordinate per player.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, field_validator, model_validator

from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.schemas import FrozenModel, Tour

INACTIVITY_THRESHOLD_DAYS: Literal[90] = 90
INACTIVITY_HALF_LIFE_DAYS: Literal[180] = 180
INACTIVITY_VARIANCE_LIMIT: Literal[2] = 2
INACTIVITY_RECORD_SCHEMA_VERSION: Literal["inactivity-record/v1"] = "inactivity-record/v1"
INACTIVITY_CONFIG_SCHEMA_VERSION: Literal["inactivity-config/v1"] = "inactivity-config/v1"
INACTIVITY_ADJUSTMENT_SCHEMA_VERSION: Literal["inactivity-adjustment/v1"] = (
    "inactivity-adjustment/v1"
)
INACTIVITY_ADJUSTMENT_VERSION: Literal["c6-post-90-day/v1"] = "c6-post-90-day/v1"

type FloatArray = NDArray[np.float64]


class InactivityError(ValueError):
    """The C6 record or transform violates the frozen contract."""


class InactivityUnavailableError(InactivityError):
    """Cutoff-safe inactivity information is insufficient for production use."""


class DuplicateInactivityAdjustmentError(InactivityError):
    """A component posterior has already received the C6 transform."""


class InactivityCoverageState(StrEnum):
    """Whether no-history and last-match conclusions are supported."""

    VERIFIED_COMPLETE = "VERIFIED_COMPLETE"
    INCOMPLETE_SOURCE = "INCOMPLETE_SOURCE"
    UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"


class PlayedPointEvidence(StrEnum):
    """Permitted evidence that a competitive singles match started."""

    POSITIVE_POINT_STAT_COUNT = "POSITIVE_POINT_STAT_COUNT"
    LEGAL_SCORE_WITH_COMPLETED_GAME_OR_TIEBREAK = "LEGAL_SCORE_WITH_COMPLETED_GAME_OR_TIEBREAK"
    OFFICIAL_STARTED_OR_IN_PLAY_MARKER = "OFFICIAL_STARTED_OR_IN_PLAY_MARKER"


class InactivityTerminalStatus(StrEnum):
    """Terminal category relevant to last-match eligibility, not B6 fitting."""

    NORMAL_COMPLETION = "NORMAL_COMPLETION"
    STARTED_RETIREMENT = "STARTED_RETIREMENT"
    WALKOVER = "WALKOVER"
    PRE_START_WITHDRAWAL = "PRE_START_WITHDRAWAL"
    DEFAULT_OR_DISQUALIFICATION = "DEFAULT_OR_DISQUALIFICATION"
    ABANDONED = "ABANDONED"
    OTHER = "OTHER"


class CompetitionClass(StrEnum):
    """Frozen competitive-singles universe classifications."""

    MAIN_DRAW = "MAIN_DRAW"
    QUALIFYING = "QUALIFYING"
    TEAM = "TEAM"
    OLYMPIC = "OLYMPIC"
    OTHER_COMPETITIVE = "OTHER_COMPETITIVE"
    EXHIBITION = "EXHIBITION"


class InactivityEligibility(StrEnum):
    """Auditable result of applying the last-match eligibility rules."""

    ELIGIBLE = "ELIGIBLE"
    AT_OR_AFTER_INFORMATION_CUTOFF = "AT_OR_AFTER_INFORMATION_CUTOFF"
    PLAYER_MISMATCH = "PLAYER_MISMATCH"
    TOUR_MISMATCH = "TOUR_MISMATCH"
    SOURCE_NOT_IN_COVERAGE_ASSERTION = "SOURCE_NOT_IN_COVERAGE_ASSERTION"
    UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"
    DOUBLES = "DOUBLES"
    EXHIBITION = "EXHIBITION"
    WALKOVER_OR_PRE_START_WITHDRAWAL = "WALKOVER_OR_PRE_START_WITHDRAWAL"
    NO_STARTED_MATCH_EVIDENCE = "NO_STARTED_MATCH_EVIDENCE"


class InactivityBand(StrEnum):
    """Predeclared C6 reporting bands; these never change the transform."""

    ACTIVE_DAYS_0_90 = "D_LE_90"
    DAYS_91_180 = "D_91_180"
    DAYS_181_365 = "D_181_365"
    DAYS_OVER_365 = "D_GT_365"
    COLD_START = "COLD_START"


class InactivityCoordinateRole(StrEnum):
    """Canonical player-effect coordinates affected by C6."""

    SERVER_GLOBAL = "server_global"
    SERVER_HARD_DEVIATION = "server_hard_deviation"
    RETURNER_GLOBAL = "returner_global"
    RETURNER_HARD_DEVIATION = "returner_hard_deviation"

    @property
    def is_hard_deviation(self) -> bool:
        return self in {
            InactivityCoordinateRole.SERVER_HARD_DEVIATION,
            InactivityCoordinateRole.RETURNER_HARD_DEVIATION,
        }


class InactivityAdjustmentState(StrEnum):
    """Exact-once guard for a component posterior."""

    UNADJUSTED = "UNADJUSTED"
    C6_APPLIED = "C6_APPLIED"


class _FiniteFrozenModel(FrozenModel):
    model_config = ConfigDict(allow_inf_nan=False)


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


def _canonical_json_bytes(value: Any) -> bytes:
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


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class InactivityCoverageAssertion(_FiniteFrozenModel):
    """Versioned assertion distinguishing verified no-history from missing data."""

    schema_version: Literal["inactivity-coverage/v1"] = "inactivity-coverage/v1"
    state: InactivityCoverageState
    source_manifest_id: str
    source_manifest_sha256: str
    canonical_player_id: str | None
    asserted_at_utc: datetime
    reason: str | None = None

    @field_validator("source_manifest_id")
    @classmethod
    def manifest_id_is_present(cls, value: str) -> str:
        return _nonempty(value, field="source_manifest_id")

    @field_validator("source_manifest_sha256")
    @classmethod
    def manifest_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field="source_manifest_sha256")

    @field_validator("canonical_player_id")
    @classmethod
    def optional_player_id_is_present(cls, value: str | None) -> str | None:
        return None if value is None else _nonempty(value, field="canonical_player_id")

    @field_validator("asserted_at_utc")
    @classmethod
    def assertion_time_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="asserted_at_utc")

    @field_validator("reason")
    @classmethod
    def optional_reason_is_present(cls, value: str | None) -> str | None:
        return None if value is None else _nonempty(value, field="reason")

    @model_validator(mode="after")
    def coverage_and_identity_are_coherent(self) -> Self:
        if self.state is InactivityCoverageState.VERIFIED_COMPLETE:
            if self.canonical_player_id is None:
                raise ValueError("verified coverage requires a resolved canonical player")
        elif self.reason is None:
            raise ValueError("unavailable coverage states require an explicit reason")
        if (
            self.state is InactivityCoverageState.UNRESOLVED_IDENTITY
            and self.canonical_player_id is not None
        ):
            raise ValueError("unresolved identity cannot name a canonical player")
        return self

    @property
    def sha256(self) -> str:
        return _content_hash(self.model_dump(mode="json"))


class InactivityMatchCandidate(_FiniteFrozenModel):
    """One normalized, source-pinned candidate for a player's last match."""

    player_id: str
    identity_resolved: bool
    tour: Tour
    match_id: str
    match_date_local: date
    discipline: Literal["singles", "doubles"]
    competition_class: CompetitionClass
    terminal_status: InactivityTerminalStatus
    started_evidence: tuple[PlayedPointEvidence, ...]
    source_manifest_id: str
    source_pin: str
    source_sha256: str
    available_at_utc: datetime

    @field_validator("player_id", "match_id", "source_manifest_id", "source_pin")
    @classmethod
    def identities_are_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field="source_sha256")

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="available_at_utc")

    @model_validator(mode="after")
    def evidence_is_unique(self) -> Self:
        if len(self.started_evidence) != len(set(self.started_evidence)):
            raise ValueError("started-match evidence values must be unique")
        return self


class LastEligibleMatch(_FiniteFrozenModel):
    """Cutoff-safe source identity of the match defining ``L_i``."""

    match_id: str
    match_date_local: date
    terminal_status: InactivityTerminalStatus
    started_evidence: tuple[PlayedPointEvidence, ...]
    source_pin: str
    source_sha256: str
    available_at_utc: datetime

    @field_validator("match_id", "source_pin")
    @classmethod
    def identities_are_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field="source_sha256")

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="available_at_utc")


class InactivityRecord(_FiniteFrozenModel):
    """Immutable cutoff-safe C6 input and exact deterministic factors."""

    schema_version: Literal["inactivity-record/v1"] = INACTIVITY_RECORD_SCHEMA_VERSION
    player_id: str
    tour: Tour
    scheduled_start_local_date: date
    information_cutoff_utc: datetime
    coverage: InactivityCoverageAssertion
    last_eligible_match: LastEligibleMatch | None
    inactivity_days: Annotated[int, Field(ge=0)] | None
    post_threshold_days: Annotated[int, Field(ge=0)] | None
    hard_deviation_multiplier: Annotated[float, Field(ge=0, le=1)]
    variance_inflation_factor: Annotated[float, Field(ge=1, le=2)]
    cold_start: bool
    adjustment_version: Literal["c6-post-90-day/v1"] = INACTIVITY_ADJUSTMENT_VERSION

    @field_validator("player_id")
    @classmethod
    def player_id_is_present(cls, value: str) -> str:
        return _nonempty(value, field="player_id")

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="information_cutoff_utc")

    @model_validator(mode="after")
    def record_is_exact_and_cutoff_safe(self) -> Self:
        if self.coverage.state is not InactivityCoverageState.VERIFIED_COMPLETE:
            raise ValueError("an inactivity record requires verified complete source coverage")
        if self.coverage.canonical_player_id != self.player_id:
            raise ValueError("coverage assertion resolves a different player")
        if self.coverage.asserted_at_utc >= self.information_cutoff_utc:
            raise ValueError("coverage assertion must be available strictly before the cutoff")
        if self.cold_start:
            if self.last_eligible_match is not None:
                raise ValueError("cold start cannot contain a last eligible match")
            if self.inactivity_days is not None or self.post_threshold_days is not None:
                raise ValueError("cold-start D and g must be null")
            if self.hard_deviation_multiplier != 0.0:
                raise ValueError("cold-start m must equal zero")
            if self.variance_inflation_factor != 1.0:
                raise ValueError("cold-start c must equal one")
            return self

        if self.last_eligible_match is None:
            raise ValueError("known-history inactivity requires a last eligible match")
        if self.last_eligible_match.available_at_utc >= self.information_cutoff_utc:
            raise ValueError("last-match evidence must be available strictly before the cutoff")
        expected_days = (
            self.scheduled_start_local_date - self.last_eligible_match.match_date_local
        ).days
        if expected_days < 0:
            raise ValueError("last eligible match follows the scheduled start date")
        expected_gap, expected_multiplier, expected_inflation = inactivity_factors(expected_days)
        if self.inactivity_days != expected_days or self.post_threshold_days != expected_gap:
            raise ValueError("stored inactivity day counts do not match the source dates")
        if self.hard_deviation_multiplier != expected_multiplier:
            raise ValueError("stored inactivity multiplier does not match the frozen formula")
        if self.variance_inflation_factor != expected_inflation:
            raise ValueError("stored variance factor does not match the frozen formula")
        return self

    @property
    def sha256(self) -> str:
        return _content_hash(self.model_dump(mode="json"))

    @property
    def band(self) -> InactivityBand:
        return inactivity_band(self.inactivity_days)


def inactivity_factors(inactivity_days: int) -> tuple[int, float, float]:
    """Return exact frozen ``(g, m, c)`` values for known inactivity ``D``."""

    if isinstance(inactivity_days, bool) or not isinstance(inactivity_days, Integral):
        raise TypeError("inactivity_days must be an integer number of calendar days")
    days = int(inactivity_days)
    if days < 0:
        raise InactivityError("negative inactivity is invalid and must not be clipped")
    gap = max(0, days - INACTIVITY_THRESHOLD_DAYS)
    if days <= INACTIVITY_THRESHOLD_DAYS:
        return gap, 1.0, 1.0
    # These bounds remove floating-point overshoot only.  There is no positive
    # lower cap: natural underflow to zero validly produces the limiting c=2.
    multiplier = min(1.0, max(0.0, 2.0 ** (-gap / INACTIVITY_HALF_LIFE_DAYS)))
    inflation = min(
        float(INACTIVITY_VARIANCE_LIMIT),
        max(1.0, float(INACTIVITY_VARIANCE_LIMIT) - multiplier * multiplier),
    )
    return gap, multiplier, inflation


def inactivity_band(inactivity_days: int | None) -> InactivityBand:
    """Map known ``D`` or verified cold start to the fixed diagnostic bands."""

    if inactivity_days is None:
        return InactivityBand.COLD_START
    if isinstance(inactivity_days, bool) or not isinstance(inactivity_days, Integral):
        raise TypeError("inactivity_days must be an integer or null")
    days = int(inactivity_days)
    if days < 0:
        raise InactivityError("negative inactivity has no diagnostic band")
    if days <= 90:
        return InactivityBand.ACTIVE_DAYS_0_90
    if days <= 180:
        return InactivityBand.DAYS_91_180
    if days <= 365:
        return InactivityBand.DAYS_181_365
    return InactivityBand.DAYS_OVER_365


def inactivity_match_eligibility(
    candidate: InactivityMatchCandidate,
    *,
    player_id: str,
    tour: Tour,
    information_cutoff_utc: datetime,
    source_manifest_id: str,
) -> InactivityEligibility:
    """Classify one candidate without inferring start from its status string."""

    cutoff = _utc(information_cutoff_utc, field="information_cutoff_utc")
    if candidate.player_id != player_id:
        return InactivityEligibility.PLAYER_MISMATCH
    if candidate.tour is not tour:
        return InactivityEligibility.TOUR_MISMATCH
    if candidate.source_manifest_id != source_manifest_id:
        return InactivityEligibility.SOURCE_NOT_IN_COVERAGE_ASSERTION
    if candidate.available_at_utc >= cutoff:
        return InactivityEligibility.AT_OR_AFTER_INFORMATION_CUTOFF
    if not candidate.identity_resolved:
        return InactivityEligibility.UNRESOLVED_IDENTITY
    if candidate.discipline != "singles":
        return InactivityEligibility.DOUBLES
    if candidate.competition_class is CompetitionClass.EXHIBITION:
        return InactivityEligibility.EXHIBITION
    if candidate.terminal_status in {
        InactivityTerminalStatus.WALKOVER,
        InactivityTerminalStatus.PRE_START_WITHDRAWAL,
    }:
        return InactivityEligibility.WALKOVER_OR_PRE_START_WITHDRAWAL
    if not candidate.started_evidence:
        return InactivityEligibility.NO_STARTED_MATCH_EVIDENCE
    return InactivityEligibility.ELIGIBLE


def build_inactivity_record(
    *,
    player_id: str,
    tour: Tour,
    scheduled_start_local_date: date | None,
    information_cutoff_utc: datetime,
    coverage: InactivityCoverageAssertion,
    candidates: Sequence[InactivityMatchCandidate],
) -> InactivityRecord:
    """Select cutoff-safe ``L_i`` and construct the exact C6 record.

    Candidate records unavailable at the cutoff are ignored.  Missing coverage
    and unresolved identity fail rather than entering the cold-start branch.
    """

    canonical_player = _nonempty(player_id, field="player_id")
    cutoff = _utc(information_cutoff_utc, field="information_cutoff_utc")
    if scheduled_start_local_date is None:
        raise InactivityUnavailableError("official scheduled-start local date is required")
    if coverage.state is not InactivityCoverageState.VERIFIED_COMPLETE:
        raise InactivityUnavailableError(
            f"inactivity coverage is unavailable: {coverage.state.value}"
        )
    if coverage.canonical_player_id != canonical_player:
        raise InactivityUnavailableError("coverage assertion does not resolve the target player")
    if coverage.asserted_at_utc >= cutoff:
        raise InactivityUnavailableError(
            "coverage assertion was not available strictly before the cutoff"
        )

    # Append-only corrections share a match identity.  Select the latest
    # pre-cutoff revision before applying eligibility, so a later walkover or
    # no-play correction cannot leave an earlier started row active.
    revisions: dict[tuple[Tour, str, str], list[InactivityMatchCandidate]] = {}
    for candidate in candidates:
        revisions.setdefault((candidate.tour, candidate.match_id, candidate.player_id), []).append(
            candidate
        )
    cutoff_candidates: list[InactivityMatchCandidate] = []
    for identity in sorted(revisions, key=lambda item: (item[0].value, item[1], item[2])):
        visible = [item for item in revisions[identity] if item.available_at_utc < cutoff]
        if not visible:
            continue
        latest_time = max(item.available_at_utc for item in visible)
        latest_revisions = [item for item in visible if item.available_at_utc == latest_time]
        if any(item != latest_revisions[0] for item in latest_revisions[1:]):
            raise InactivityUnavailableError(
                f"conflicting last-match corrections share an availability time: {identity}"
            )
        cutoff_candidates.append(latest_revisions[0])

    eligible: list[InactivityMatchCandidate] = []
    for candidate in cutoff_candidates:
        status = inactivity_match_eligibility(
            candidate,
            player_id=canonical_player,
            tour=tour,
            information_cutoff_utc=cutoff,
            source_manifest_id=coverage.source_manifest_id,
        )
        if status is InactivityEligibility.UNRESOLVED_IDENTITY:
            raise InactivityUnavailableError(
                "candidate history contains unresolved player identity"
            )
        if status is InactivityEligibility.ELIGIBLE:
            eligible.append(candidate)

    if not eligible:
        return InactivityRecord(
            player_id=canonical_player,
            tour=tour,
            scheduled_start_local_date=scheduled_start_local_date,
            information_cutoff_utc=cutoff,
            coverage=coverage,
            last_eligible_match=None,
            inactivity_days=None,
            post_threshold_days=None,
            hard_deviation_multiplier=0.0,
            variance_inflation_factor=1.0,
            cold_start=True,
        )

    # Same-date matches have identical probability consequences.  Availability
    # and match ID provide a stable provenance-only tie break.
    latest_match = max(
        eligible,
        key=lambda item: (item.match_date_local, item.available_at_utc, item.match_id),
    )
    days = (scheduled_start_local_date - latest_match.match_date_local).days
    if days < 0:
        raise InactivityUnavailableError(
            "latest eligible match follows the official scheduled-start date"
        )
    gap, multiplier, inflation = inactivity_factors(days)
    return InactivityRecord(
        player_id=canonical_player,
        tour=tour,
        scheduled_start_local_date=scheduled_start_local_date,
        information_cutoff_utc=cutoff,
        coverage=coverage,
        last_eligible_match=LastEligibleMatch(
            match_id=latest_match.match_id,
            match_date_local=latest_match.match_date_local,
            terminal_status=latest_match.terminal_status,
            started_evidence=latest_match.started_evidence,
            source_pin=latest_match.source_pin,
            source_sha256=latest_match.source_sha256,
            available_at_utc=latest_match.available_at_utc,
        ),
        inactivity_days=days,
        post_threshold_days=gap,
        hard_deviation_multiplier=multiplier,
        variance_inflation_factor=inflation,
        cold_start=False,
    )


_SERVER_ROLES = (
    InactivityCoordinateRole.SERVER_GLOBAL,
    InactivityCoordinateRole.SERVER_HARD_DEVIATION,
)
_SERVER_AND_RETURNER_ROLES = (
    *_SERVER_ROLES,
    InactivityCoordinateRole.RETURNER_GLOBAL,
    InactivityCoordinateRole.RETURNER_HARD_DEVIATION,
)
_AFFECTED_ROLES: dict[ServeComponent, tuple[InactivityCoordinateRole, ...]] = {
    ServeComponent.F: _SERVER_ROLES,
    ServeComponent.A: _SERVER_AND_RETURNER_ROLES,
    ServeComponent.Q1: _SERVER_AND_RETURNER_ROLES,
    ServeComponent.D: _SERVER_ROLES,
    ServeComponent.Q2: _SERVER_AND_RETURNER_ROLES,
}


def affected_coordinate_roles(component: ServeComponent) -> tuple[InactivityCoordinateRole, ...]:
    """Return the exact frozen C6 role set for one primitive component."""

    return _AFFECTED_ROLES[component]


def coordinate_role_for_effect_block(
    role: str,
    *,
    surface: str | None,
) -> InactivityCoordinateRole | None:
    """Map an existing named effect block to a C6 role, excluding other context."""

    if role == "server_global" and surface is None:
        return InactivityCoordinateRole.SERVER_GLOBAL
    if role == "returner_global" and surface is None:
        return InactivityCoordinateRole.RETURNER_GLOBAL
    if role == "server_surface" and surface == "Hard":
        return InactivityCoordinateRole.SERVER_HARD_DEVIATION
    if role == "returner_surface" and surface == "Hard":
        return InactivityCoordinateRole.RETURNER_HARD_DEVIATION
    return None


class InactivityCoordinateReference(_FiniteFrozenModel):
    """One explicit canonical player-effect coordinate in a component posterior."""

    component: ServeComponent
    player_id: str
    role: InactivityCoordinateRole
    coordinate_id: str
    index: Annotated[int, Field(ge=0)]

    @field_validator("player_id", "coordinate_id")
    @classmethod
    def identities_are_present(cls, value: str, info: Any) -> str:
        return _nonempty(value, field=info.field_name)


def select_affected_coordinates(
    component: ServeComponent,
    player_id: str,
    catalog: Sequence[InactivityCoordinateReference],
) -> tuple[InactivityCoordinateReference, ...]:
    """Select and validate every required player/role coordinate exactly once."""

    expected = affected_coordinate_roles(component)
    selected = [
        item for item in catalog if item.component is component and item.player_id == player_id
    ]
    by_role: dict[InactivityCoordinateRole, InactivityCoordinateReference] = {}
    for item in selected:
        if item.role in by_role:
            raise InactivityError(
                f"duplicate {component.value}/{player_id}/{item.role.value} coordinate"
            )
        by_role[item.role] = item
    if set(by_role) != set(expected):
        missing = sorted(role.value for role in set(expected) - set(by_role))
        unexpected = sorted(role.value for role in set(by_role) - set(expected))
        raise InactivityError(
            f"incomplete C6 coordinate catalog for {component.value}/{player_id}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(by_role[role] for role in expected)


class InactivityAdjustment(_FiniteFrozenModel):
    """Hash-addressed provenance for one exact C6 component transform."""

    schema_version: Literal["inactivity-adjustment/v1"] = INACTIVITY_ADJUSTMENT_SCHEMA_VERSION
    component: ServeComponent
    adjustment_version: Literal["c6-post-90-day/v1"] = INACTIVITY_ADJUSTMENT_VERSION
    state_before: Literal[InactivityAdjustmentState.UNADJUSTED] = (
        InactivityAdjustmentState.UNADJUSTED
    )
    state_after: Literal[InactivityAdjustmentState.C6_APPLIED] = (
        InactivityAdjustmentState.C6_APPLIED
    )
    affected_coordinates: tuple[InactivityCoordinateReference, ...]
    player_record_hashes: tuple[tuple[str, str], ...]
    unadjusted_mean_covariance_sha256: str
    adjusted_mean_covariance_sha256: str

    @field_validator("unadjusted_mean_covariance_sha256", "adjusted_mean_covariance_sha256")
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field=info.field_name)

    @model_validator(mode="after")
    def references_are_unique(self) -> Self:
        coordinate_ids = tuple(item.coordinate_id for item in self.affected_coordinates)
        if len(coordinate_ids) != len(set(coordinate_ids)):
            raise ValueError("affected C6 coordinate IDs must be unique")
        player_ids = tuple(item[0] for item in self.player_record_hashes)
        if len(player_ids) != len(set(player_ids)) or tuple(sorted(player_ids)) != player_ids:
            raise ValueError("player record hashes must be unique and sorted")
        for player_id, digest in self.player_record_hashes:
            _nonempty(player_id, field="player_record_hash player ID")
            _sha256(digest, field="player_record_hash")
        return self


@dataclass(frozen=True, slots=True)
class AdjustedComponentPosterior:
    """Read-only numerical posterior plus the authoritative C6 audit record."""

    component: ServeComponent
    coordinate_ids: tuple[str, ...]
    mean: FloatArray
    covariance: FloatArray
    kappa: float
    adjustment_state: InactivityAdjustmentState
    adjustment: InactivityAdjustment

    def __post_init__(self) -> None:
        self.mean.setflags(write=False)
        self.covariance.setflags(write=False)


def posterior_mean_covariance_sha256(
    component: ServeComponent,
    coordinate_ids: Sequence[str],
    mean: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
) -> str:
    """Hash named mean/covariance contents without platform-specific array bytes."""

    mean_array = np.asarray(mean, dtype=np.float64)
    covariance_array = np.asarray(covariance, dtype=np.float64)
    return _content_hash(
        {
            "component": component.value,
            "coordinate_ids": list(coordinate_ids),
            "mean": mean_array.tolist(),
            "covariance": covariance_array.tolist(),
        }
    )


def _validated_posterior_arrays(
    coordinate_ids: Sequence[str],
    mean: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
) -> tuple[tuple[str, ...], FloatArray, FloatArray]:
    ids = tuple(_nonempty(item, field="coordinate_id") for item in coordinate_ids)
    if len(ids) != len(set(ids)):
        raise InactivityError("posterior coordinate IDs must be unique")
    try:
        mean_array = np.asarray(mean, dtype=np.float64)
        covariance_array = np.asarray(covariance, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InactivityError("posterior mean and covariance must be numeric") from exc
    size = len(ids)
    if mean_array.shape != (size,) or covariance_array.shape != (size, size):
        raise InactivityError("posterior mean/covariance dimensions do not match coordinates")
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(covariance_array)):
        raise InactivityError("posterior mean/covariance must be finite")
    if not np.array_equal(covariance_array, covariance_array.T):
        if not np.allclose(covariance_array, covariance_array.T, rtol=0.0, atol=1e-14):
            raise InactivityError("posterior covariance must be symmetric")
        covariance_array = (covariance_array + covariance_array.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(covariance_array))) < -1e-12:
        raise InactivityError("posterior covariance is not positive semidefinite")
    return ids, mean_array.copy(), covariance_array.copy()


def apply_inactivity_adjustment(
    *,
    component: ServeComponent,
    coordinate_ids: Sequence[str],
    mean: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
    kappa: float,
    inactivity_records: Sequence[InactivityRecord],
    coordinate_catalog: Sequence[InactivityCoordinateReference],
    adjustment_state: InactivityAdjustmentState = InactivityAdjustmentState.UNADJUSTED,
) -> AdjustedComponentPosterior:
    """Apply the exact mean and covariance transform once.

    Global player means remain unchanged.  Hard-deviation means receive ``m``;
    both kinds of affected player coordinate receive covariance scale
    ``sqrt(c)``.  No cross-component object or random factor is introduced.
    """

    try:
        normalized_state = InactivityAdjustmentState(adjustment_state)
    except ValueError as exc:
        raise InactivityError("unknown inactivity adjustment state") from exc
    if normalized_state is not InactivityAdjustmentState.UNADJUSTED:
        raise DuplicateInactivityAdjustmentError(
            f"C6 transform requires UNADJUSTED state, got {normalized_state.value}"
        )
    if isinstance(kappa, bool) or not isinstance(kappa, Real):
        raise TypeError("kappa must be a finite positive number")
    unchanged_kappa = float(kappa)
    if not isfinite(unchanged_kappa) or unchanged_kappa <= 0.0:
        raise InactivityError("kappa must be finite and positive")
    if not inactivity_records:
        raise InactivityError("at least one player inactivity record is required")
    records_by_player = {item.player_id: item for item in inactivity_records}
    if len(records_by_player) != len(inactivity_records):
        raise InactivityError("player inactivity records must be unique")

    ids, original_mean, original_covariance = _validated_posterior_arrays(
        coordinate_ids, mean, covariance
    )
    adjusted_mean = original_mean.copy()
    scales = np.ones(len(ids), dtype=np.float64)
    affected: list[InactivityCoordinateReference] = []
    used_indices: set[int] = set()
    for player_id in sorted(records_by_player):
        record = records_by_player[player_id]
        references = select_affected_coordinates(component, player_id, coordinate_catalog)
        for reference in references:
            if reference.index >= len(ids):
                raise InactivityError(f"C6 coordinate index is out of range: {reference.index}")
            if ids[reference.index] != reference.coordinate_id:
                raise InactivityError("C6 coordinate ID does not match its posterior index")
            if reference.index in used_indices:
                raise InactivityError(
                    "distinct C6 player roles cannot share a posterior coordinate"
                )
            used_indices.add(reference.index)
            if reference.role.is_hard_deviation:
                if record.cold_start:
                    adjusted_mean[reference.index] = 0.0
                elif record.hard_deviation_multiplier != 1.0:
                    adjusted_mean[reference.index] *= record.hard_deviation_multiplier
            if record.variance_inflation_factor != 1.0:
                scales[reference.index] = sqrt(record.variance_inflation_factor)
            affected.append(reference)

    if np.array_equal(scales, np.ones(len(ids), dtype=np.float64)):
        adjusted_covariance = original_covariance.copy()
    else:
        adjusted_covariance = scales[:, np.newaxis] * original_covariance * scales[np.newaxis, :]
        adjusted_covariance = (adjusted_covariance + adjusted_covariance.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(adjusted_covariance))) < -1e-12:
        raise InactivityError("adjusted posterior covariance is not positive semidefinite")

    unadjusted_hash = posterior_mean_covariance_sha256(
        component, ids, original_mean, original_covariance
    )
    adjusted_hash = posterior_mean_covariance_sha256(
        component, ids, adjusted_mean, adjusted_covariance
    )
    adjustment = InactivityAdjustment(
        component=component,
        affected_coordinates=tuple(affected),
        player_record_hashes=tuple(
            (player_id, records_by_player[player_id].sha256)
            for player_id in sorted(records_by_player)
        ),
        unadjusted_mean_covariance_sha256=unadjusted_hash,
        adjusted_mean_covariance_sha256=adjusted_hash,
    )
    return AdjustedComponentPosterior(
        component=component,
        coordinate_ids=ids,
        mean=adjusted_mean,
        covariance=adjusted_covariance,
        kappa=unchanged_kappa,
        adjustment_state=InactivityAdjustmentState.C6_APPLIED,
        adjustment=adjustment,
    )


class InactivityAffectedRoleSpec(_FiniteFrozenModel):
    component: ServeComponent
    roles: tuple[InactivityCoordinateRole, ...]

    @model_validator(mode="after")
    def roles_match_frozen_contract(self) -> Self:
        if self.roles != affected_coordinate_roles(self.component):
            raise ValueError(f"{self.component.value} affected roles differ from frozen C6")
        return self


class InactivityConfigurationArtifact(_FiniteFrozenModel):
    """Content-addressed immutable representation of all C6 constants."""

    schema_version: Literal["inactivity-config/v1"] = INACTIVITY_CONFIG_SCHEMA_VERSION
    threshold_days: Literal[90] = INACTIVITY_THRESHOLD_DAYS
    post_threshold_half_life_days: Literal[180] = INACTIVITY_HALF_LIFE_DAYS
    variance_formula: Literal["c=2-m^2"] = "c=2-m^2"
    limiting_variance_factor: Literal[2] = INACTIVITY_VARIANCE_LIMIT
    cold_start_rule_version: Literal["verified-coverage-cold-start/v1"] = (
        "verified-coverage-cold-start/v1"
    )
    eligible_match_date_rule_version: Literal["cutoff-safe-started-singles-local-date/v1"] = (
        "cutoff-safe-started-singles-local-date/v1"
    )
    source_coverage_rule_version: Literal["verified-complete-or-block/v1"] = (
        "verified-complete-or-block/v1"
    )
    affected_roles: tuple[InactivityAffectedRoleSpec, ...]
    transform_version: Literal["c6-post-90-day/v1"] = INACTIVITY_ADJUSTMENT_VERSION
    config_sha256: str
    code_sha256: str

    @field_validator("config_sha256", "code_sha256")
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field=info.field_name)

    @model_validator(mode="after")
    def every_component_is_exactly_declared(self) -> Self:
        if tuple(item.component for item in self.affected_roles) != tuple(ServeComponent):
            raise ValueError("inactivity config must declare F/A/Q1/D/Q2 in canonical order")
        return self

    @property
    def artifact_id(self) -> str:
        return _content_hash(self.model_dump(mode="json"))


def create_inactivity_configuration_artifact(
    *,
    config_sha256: str,
    code_sha256: str,
) -> InactivityConfigurationArtifact:
    """Create the sole frozen v1.0 C6 configuration artifact."""

    return InactivityConfigurationArtifact(
        affected_roles=tuple(
            InactivityAffectedRoleSpec(
                component=component,
                roles=affected_coordinate_roles(component),
            )
            for component in ServeComponent
        ),
        config_sha256=config_sha256,
        code_sha256=code_sha256,
    )


__all__ = [
    "INACTIVITY_ADJUSTMENT_SCHEMA_VERSION",
    "INACTIVITY_ADJUSTMENT_VERSION",
    "INACTIVITY_CONFIG_SCHEMA_VERSION",
    "INACTIVITY_HALF_LIFE_DAYS",
    "INACTIVITY_RECORD_SCHEMA_VERSION",
    "INACTIVITY_THRESHOLD_DAYS",
    "INACTIVITY_VARIANCE_LIMIT",
    "AdjustedComponentPosterior",
    "CompetitionClass",
    "DuplicateInactivityAdjustmentError",
    "InactivityAdjustment",
    "InactivityAdjustmentState",
    "InactivityAffectedRoleSpec",
    "InactivityBand",
    "InactivityConfigurationArtifact",
    "InactivityCoordinateReference",
    "InactivityCoordinateRole",
    "InactivityCoverageAssertion",
    "InactivityCoverageState",
    "InactivityEligibility",
    "InactivityError",
    "InactivityMatchCandidate",
    "InactivityRecord",
    "InactivityTerminalStatus",
    "InactivityUnavailableError",
    "LastEligibleMatch",
    "PlayedPointEvidence",
    "affected_coordinate_roles",
    "apply_inactivity_adjustment",
    "build_inactivity_record",
    "coordinate_role_for_effect_block",
    "create_inactivity_configuration_artifact",
    "inactivity_band",
    "inactivity_factors",
    "inactivity_match_eligibility",
    "posterior_mean_covariance_sha256",
    "select_affected_coordinates",
]
