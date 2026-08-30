"""Frozen B6 retirement incidence, artifacts, scenarios, and competing risks.

This module deliberately stops at the probability primitives.  The scoring
engine decides *when* a completed-game boundary is eligible for a check; prop
settlement remains a separate consumer of the resulting terminal match path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.random import Generator
from pydantic import Field, field_validator, model_validator

from tennis_model.schemas import FrozenModel, NonEmptyStr, Tour

RETIREMENT_RESPONSE_CODING_VERSION = "retirement-response-coding/v1"
RETIREMENT_ARTIFACT_SCHEMA_VERSION = "retirement-fit-artifact/v2"
RETIREMENT_LEGACY_ARTIFACT_SCHEMA_VERSION = "retirement-fit-artifact/v1"
RETIREMENT_INTENSITY_TRANSFORM_VERSION = "retirement-rho-to-game-intensity/v1"
RETIREMENT_SCENARIO_SCHEMA_VERSION = "retirement-scenario/v1"
RETIREMENT_COMPETING_RISK_VERSION = "retirement-competing-risk/v1"

RETIREMENT_WINDOW_DAYS = 1_826
RETIREMENT_HALF_LIFE_DAYS = 730
RETIREMENT_PRIOR_EFFECTIVE_STARTS = 100.0
RETIREMENT_REFERENCE_GAMES = 22
RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS = 500.0

_ARTIFACT_FILENAME = "retirement-fit.json"
_SHA256_CHARS = frozenset("0123456789abcdef")


class RetirementModelError(ValueError):
    """A B6 input or derived quantity violates the frozen specification."""


class RetirementCoverageError(RetirementModelError):
    """A retirement fit cannot be used for a production forecast."""


class RetirementArtifactError(RuntimeError):
    """A retirement artifact cannot be published or loaded."""


class RetirementArtifactIntegrityError(RetirementArtifactError):
    """Persisted retirement artifact content or identity is invalid."""


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
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


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


class OfficialTerminalStatus(StrEnum):
    """Source-normalized official terminal fact before B6 eligibility coding."""

    NORMAL_COMPLETION = "normal_completion"
    RETIREMENT = "retirement"
    WALKOVER_OR_PRESTART_WITHDRAWAL = "walkover_or_prestart_withdrawal"
    DEFAULT_DISQUALIFICATION_OR_MISCONDUCT = "default_disqualification_or_misconduct"
    ABANDONMENT = "abandonment"
    CANCELLATION = "cancellation"
    NO_CONTEST = "no_contest"
    SUSPENDED_UNRESOLVED = "suspended_unresolved"
    CONFLICTING = "conflicting"
    OTHER_AMBIGUOUS = "other_ambiguous"


class StartedEvidence(StrEnum):
    """Evidence admitted by B6.1 to establish that at least one point was played."""

    POSITIVE_POINT_STAT_COUNT = "positive_point_stat_count"
    LEGAL_SCORE_COMPLETED_GAME = "legal_score_completed_game"
    LEGAL_SCORE_TIEBREAK = "legal_score_tiebreak"
    EXPLICIT_OFFICIAL_STARTED_OR_IN_PLAY = "explicit_official_started_or_in_play"


class HistoricalTerminalClass(StrEnum):
    """Exhaustive B6 terminal classes after normalization."""

    NORMAL_COMPLETION = "normal_completion"
    STARTED_RETIREMENT = "started_retirement"
    WALKOVER = "walkover"
    ADMINISTRATIVE_TERMINATION = "administrative_termination"
    AMBIGUOUS = "ambiguous"


class HistoricalTerminationInput(FrozenModel):
    """Typed official facts consumed by the source-independent B6 normalizer."""

    match_id: NonEmptyStr
    tour: Tour
    player_a_id: NonEmptyStr
    player_b_id: NonEmptyStr
    match_date: date
    official_status: OfficialTerminalStatus
    started_evidence: tuple[StartedEvidence, ...] = ()
    retiring_player_id: NonEmptyStr | None = None
    advancing_winner_id: NonEmptyStr | None = None
    retirement_completed_games: Annotated[int | None, Field(ge=0)] = None
    source_id: NonEmptyStr
    source_sha256: str
    available_at_utc: datetime
    upstream_anomaly_code: NonEmptyStr | None = None

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="available_at_utc")

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="source_sha256")

    @field_validator("started_evidence")
    @classmethod
    def evidence_is_unique_and_canonical(
        cls, value: tuple[StartedEvidence, ...]
    ) -> tuple[StartedEvidence, ...]:
        if len(value) != len(set(value)):
            raise ValueError("started_evidence must not contain duplicates")
        return tuple(sorted(value, key=lambda item: item.value))

    @model_validator(mode="after")
    def players_are_distinct(self) -> Self:
        if self.player_a_id == self.player_b_id:
            raise ValueError("player_a_id and player_b_id must differ")
        return self


class HistoricalTerminationClass(FrozenModel):
    """B6 classification with evidence and anomaly details retained for audit."""

    match_id: NonEmptyStr
    tour: Tour
    player_a_id: NonEmptyStr
    player_b_id: NonEmptyStr
    match_date: date
    official_status: OfficialTerminalStatus
    terminal_class: HistoricalTerminalClass
    started_evidence: tuple[StartedEvidence, ...]
    play_started: bool
    retiring_player_id: NonEmptyStr | None
    advancing_winner_id: NonEmptyStr | None
    retirement_completed_games: Annotated[int | None, Field(ge=0)]
    timing_available: bool
    incidence_eligible: bool
    source_id: NonEmptyStr
    source_sha256: str
    available_at_utc: datetime
    anomaly_code: NonEmptyStr | None
    response_coding_version: str = RETIREMENT_RESPONSE_CODING_VERSION

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="available_at_utc")

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="source_sha256")

    @model_validator(mode="after")
    def classification_is_coherent(self) -> Self:
        if self.response_coding_version != RETIREMENT_RESPONSE_CODING_VERSION:
            raise ValueError("unsupported retirement response-coding version")
        if self.player_a_id == self.player_b_id:
            raise ValueError("player_a_id and player_b_id must differ")
        if self.play_started != bool(self.started_evidence):
            raise ValueError("play_started must agree with started_evidence")
        if self.timing_available != (self.retirement_completed_games is not None):
            raise ValueError("timing_available must agree with retirement_completed_games")
        eligible = self.terminal_class in {
            HistoricalTerminalClass.NORMAL_COMPLETION,
            HistoricalTerminalClass.STARTED_RETIREMENT,
        }
        if self.incidence_eligible != eligible:
            raise ValueError("incidence_eligible is inconsistent with terminal_class")
        if eligible and not self.play_started:
            raise ValueError("an incidence-eligible match must have started evidence")
        if self.terminal_class is HistoricalTerminalClass.STARTED_RETIREMENT:
            if self.retiring_player_id not in {self.player_a_id, self.player_b_id}:
                raise ValueError("started retirement requires a resolved retiring player")
            if self.retiring_player_id == self.advancing_winner_id:
                raise ValueError("the retiring player cannot be the advancing winner")
        elif (
            self.retiring_player_id is not None
            and self.terminal_class is not HistoricalTerminalClass.AMBIGUOUS
        ):
            raise ValueError("only a retirement or ambiguous record may retain a retiree")
        if self.terminal_class is HistoricalTerminalClass.AMBIGUOUS and self.anomaly_code is None:
            raise ValueError("ambiguous termination requires a specific anomaly_code")
        return self


def normalize_historical_termination(
    record: HistoricalTerminationInput,
) -> HistoricalTerminationClass:
    """Apply the exhaustive B6.1 coding without repairing conflicting facts."""

    players = {record.player_a_id, record.player_b_id}
    started = bool(record.started_evidence)
    terminal_class: HistoricalTerminalClass
    anomaly: str | None = None

    invalid_identity = (
        record.retiring_player_id is not None and record.retiring_player_id not in players
    ) or (record.advancing_winner_id is not None and record.advancing_winner_id not in players)
    if invalid_identity:
        terminal_class = HistoricalTerminalClass.AMBIGUOUS
        anomaly = "UNRESOLVED_PLAYER_IDENTITY"
    elif record.upstream_anomaly_code is not None:
        terminal_class = HistoricalTerminalClass.AMBIGUOUS
        anomaly = record.upstream_anomaly_code
    elif record.official_status is OfficialTerminalStatus.NORMAL_COMPLETION:
        if record.retiring_player_id is not None or record.retirement_completed_games is not None:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "NORMAL_COMPLETION_WITH_RETIREMENT_FACTS"
        elif not started:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "NORMAL_COMPLETION_WITHOUT_STARTED_EVIDENCE"
        else:
            terminal_class = HistoricalTerminalClass.NORMAL_COMPLETION
    elif record.official_status is OfficialTerminalStatus.RETIREMENT:
        if not started:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "RETIREMENT_WITHOUT_STARTED_EVIDENCE"
        elif record.retiring_player_id is None:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "RETIRING_PLAYER_NOT_RELIABLY_IDENTIFIED"
        elif record.retiring_player_id == record.advancing_winner_id:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "RETIRING_PLAYER_RECORDED_AS_ADVANCING_WINNER"
        else:
            terminal_class = HistoricalTerminalClass.STARTED_RETIREMENT
    elif record.official_status is OfficialTerminalStatus.WALKOVER_OR_PRESTART_WITHDRAWAL:
        if (
            started
            or record.retiring_player_id is not None
            or record.retirement_completed_games is not None
        ):
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "WALKOVER_WITH_STARTED_OR_RETIREMENT_FACTS"
        else:
            terminal_class = HistoricalTerminalClass.WALKOVER
    elif record.official_status is OfficialTerminalStatus.DEFAULT_DISQUALIFICATION_OR_MISCONDUCT:
        if record.retiring_player_id is not None or record.retirement_completed_games is not None:
            terminal_class = HistoricalTerminalClass.AMBIGUOUS
            anomaly = "ADMINISTRATIVE_TERMINATION_WITH_RETIREMENT_FACTS"
        else:
            terminal_class = HistoricalTerminalClass.ADMINISTRATIVE_TERMINATION
    else:
        terminal_class = HistoricalTerminalClass.AMBIGUOUS
        anomaly = record.upstream_anomaly_code or record.official_status.value.upper()

    eligible = terminal_class in {
        HistoricalTerminalClass.NORMAL_COMPLETION,
        HistoricalTerminalClass.STARTED_RETIREMENT,
    }
    return HistoricalTerminationClass(
        match_id=record.match_id,
        tour=record.tour,
        player_a_id=record.player_a_id,
        player_b_id=record.player_b_id,
        match_date=record.match_date,
        official_status=record.official_status,
        terminal_class=terminal_class,
        started_evidence=record.started_evidence,
        play_started=started,
        retiring_player_id=record.retiring_player_id,
        advancing_winner_id=record.advancing_winner_id,
        retirement_completed_games=record.retirement_completed_games,
        timing_available=record.retirement_completed_games is not None,
        incidence_eligible=eligible,
        source_id=record.source_id,
        source_sha256=record.source_sha256,
        available_at_utc=record.available_at_utc,
        anomaly_code=anomaly,
    )


def normalize_historical_terminations_before_cutoff(
    records: tuple[HistoricalTerminationInput, ...],
    information_cutoff_utc: datetime,
) -> tuple[HistoricalTerminationClass, ...]:
    """Select the latest available append-only label, then normalize it.

    A later official correction replaces an earlier label only for fits whose
    strict information cutoff follows that correction's availability time.
    Conflicting records with the same latest availability time are rejected
    instead of being resolved by input order.
    """

    cutoff = _aware_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    grouped: dict[tuple[Tour, str], list[HistoricalTerminationInput]] = defaultdict(list)
    for record in records:
        grouped[(record.tour, record.match_id)].append(record)
    selected: list[HistoricalTerminationClass] = []
    for key in sorted(grouped, key=lambda item: (item[0].value, item[1])):
        visible = [item for item in grouped[key] if item.available_at_utc < cutoff]
        if not visible:
            continue
        latest_available = max(item.available_at_utc for item in visible)
        latest = [item for item in visible if item.available_at_utc == latest_available]
        first = latest[0]
        if any(item != first for item in latest[1:]):
            raise RetirementModelError(
                f"conflicting terminal corrections share the latest availability time: {key}"
            )
        selected.append(normalize_historical_termination(first))
    return tuple(selected)


class RetirementObservation(FrozenModel):
    """One cutoff-safe player-start response admitted to the B6 estimator."""

    tour: Tour
    player_id: NonEmptyStr
    opponent_id: NonEmptyStr
    match_id: NonEmptyStr
    match_date: date
    response: Annotated[int, Field(ge=0, le=1)]
    recency_weight: Annotated[float, Field(gt=0, le=1)]
    age_days: Annotated[int, Field(ge=0, le=RETIREMENT_WINDOW_DAYS)]
    terminal_class: HistoricalTerminalClass
    retiring_player_id: NonEmptyStr | None
    retirement_completed_games: Annotated[int | None, Field(ge=0)]
    information_cutoff_utc: datetime
    available_at_utc: datetime
    source_id: NonEmptyStr
    source_sha256: str
    response_coding_version: str = RETIREMENT_RESPONSE_CODING_VERSION

    @field_validator("response", mode="before")
    @classmethod
    def response_is_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("response must be integer 0 or 1, not bool")
        return value

    @field_validator("recency_weight")
    @classmethod
    def weight_is_finite(cls, value: float) -> float:
        return _finite(value, field_name="recency_weight")

    @field_validator("information_cutoff_utc", "available_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _aware_utc(value, field_name=info.field_name)

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="source_sha256")

    @model_validator(mode="after")
    def observation_is_coherent(self) -> Self:
        if self.player_id == self.opponent_id:
            raise ValueError("player_id and opponent_id must differ")
        if self.available_at_utc >= self.information_cutoff_utc:
            raise ValueError("retirement observation is not strictly before cutoff")
        if self.response_coding_version != RETIREMENT_RESPONSE_CODING_VERSION:
            raise ValueError("unsupported retirement response-coding version")
        if self.terminal_class not in {
            HistoricalTerminalClass.NORMAL_COMPLETION,
            HistoricalTerminalClass.STARTED_RETIREMENT,
        }:
            raise ValueError("retirement observation must come from an eligible terminal class")
        if self.response == 1 and self.player_id != self.retiring_player_id:
            raise ValueError("only the reliably identified retiree may have response 1")
        if self.terminal_class is HistoricalTerminalClass.NORMAL_COMPLETION and self.response != 0:
            raise ValueError("normal completion must have response 0")
        expected_weight = retirement_recency_weight(self.age_days)
        if not math.isclose(self.recency_weight, expected_weight, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("recency_weight does not match the frozen 730-day half-life")
        return self


class RetirementExclusion(FrozenModel):
    """Preserved match-level reason for excluding a terminal record from B6."""

    tour: Tour
    match_id: NonEmptyStr
    match_date: date
    terminal_class: HistoricalTerminalClass
    exclusion_reason: NonEmptyStr
    anomaly_code: NonEmptyStr | None
    recency_weight: Annotated[float | None, Field(ge=0, le=1)]
    source_id: NonEmptyStr
    source_sha256: str
    available_at_utc: datetime

    @field_validator("available_at_utc")
    @classmethod
    def availability_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="available_at_utc")

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="source_sha256")

    @field_validator("recency_weight")
    @classmethod
    def optional_weight_is_finite(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value, field_name="recency_weight")


class RetirementObservationBatch(FrozenModel):
    """Cutoff-safe response rows and preserved match-level exclusions."""

    information_cutoff_utc: datetime
    response_coding_version: str = RETIREMENT_RESPONSE_CODING_VERSION
    observations: tuple[RetirementObservation, ...]
    exclusions: tuple[RetirementExclusion, ...]

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="information_cutoff_utc")

    @model_validator(mode="after")
    def match_pairs_are_complete(self) -> Self:
        if self.response_coding_version != RETIREMENT_RESPONSE_CODING_VERSION:
            raise ValueError("unsupported retirement response-coding version")
        grouped: dict[tuple[Tour, str], list[RetirementObservation]] = defaultdict(list)
        for observation in self.observations:
            if observation.information_cutoff_utc != self.information_cutoff_utc:
                raise ValueError("observation cutoff differs from batch cutoff")
            grouped[(observation.tour, observation.match_id)].append(observation)
        excluded_keys = {(item.tour, item.match_id) for item in self.exclusions}
        if len(excluded_keys) != len(self.exclusions):
            raise ValueError("retirement exclusions must be unique by tour and match")
        if excluded_keys.intersection(grouped):
            raise ValueError("a match cannot be both included and excluded")
        for match_rows in grouped.values():
            if len(match_rows) != 2:
                raise ValueError("every included started match must emit exactly two player rows")
            first, second = match_rows
            if {first.player_id, second.player_id} != {first.opponent_id, second.opponent_id}:
                raise ValueError("retirement observation players/opponents are not symmetric")
            shared = (
                first.match_date == second.match_date
                and first.recency_weight == second.recency_weight
                and first.age_days == second.age_days
                and first.terminal_class is second.terminal_class
                and first.retiring_player_id == second.retiring_player_id
                and first.source_id == second.source_id
                and first.source_sha256 == second.source_sha256
                and first.available_at_utc == second.available_at_utc
            )
            if not shared:
                raise ValueError("the two player rows disagree on match-level retirement facts")
            expected_sum = (
                1 if first.terminal_class is HistoricalTerminalClass.STARTED_RETIREMENT else 0
            )
            if first.response + second.response != expected_sum:
                raise ValueError("included match responses do not match terminal class")
        return self


def retirement_recency_weight(age_days: int) -> float:
    """Return the fixed B6 weight for an in-window whole-calendar-day age."""

    if isinstance(age_days, bool) or not isinstance(age_days, int):
        raise TypeError("age_days must be an integer")
    if age_days < 0 or age_days > RETIREMENT_WINDOW_DAYS:
        raise RetirementModelError("age_days is outside the inclusive 0..1826 window")
    return math.exp2(-age_days / RETIREMENT_HALF_LIFE_DAYS)


def build_retirement_observations(
    records: tuple[HistoricalTerminationClass, ...],
    information_cutoff_utc: datetime,
) -> RetirementObservationBatch:
    """Create two player-start rows per eligible match using only pre-cutoff facts."""

    cutoff = _aware_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    observations: list[RetirementObservation] = []
    exclusions: list[RetirementExclusion] = []
    seen: set[tuple[Tour, str]] = set()
    for record in records:
        key = (record.tour, record.match_id)
        if key in seen:
            raise RetirementModelError(f"duplicate retirement terminal record: {key}")
        seen.add(key)

        age_days = (cutoff.date() - record.match_date).days
        weight: float | None = None
        reason: str | None = None
        if record.available_at_utc >= cutoff:
            reason = "AT_OR_AFTER_INFORMATION_CUTOFF"
        elif age_days < 0:
            reason = "MATCH_DATE_AFTER_CUTOFF_DATE"
        elif age_days > RETIREMENT_WINDOW_DAYS:
            reason = "OUTSIDE_1826_DAY_WINDOW"
            weight = 0.0
        else:
            weight = retirement_recency_weight(age_days)
            if not record.incidence_eligible:
                reason_parts = (
                    "EXCLUDED",
                    record.terminal_class.value.upper(),
                    record.official_status.value.upper(),
                    record.anomaly_code,
                )
                reason = ":".join(item for item in reason_parts if item is not None)

        if reason is not None:
            exclusions.append(
                RetirementExclusion(
                    tour=record.tour,
                    match_id=record.match_id,
                    match_date=record.match_date,
                    terminal_class=record.terminal_class,
                    exclusion_reason=reason,
                    anomaly_code=record.anomaly_code,
                    recency_weight=weight,
                    source_id=record.source_id,
                    source_sha256=record.source_sha256,
                    available_at_utc=record.available_at_utc,
                )
            )
            continue

        assert weight is not None
        for player_id, opponent_id in (
            (record.player_a_id, record.player_b_id),
            (record.player_b_id, record.player_a_id),
        ):
            response = int(
                record.terminal_class is HistoricalTerminalClass.STARTED_RETIREMENT
                and player_id == record.retiring_player_id
            )
            observations.append(
                RetirementObservation(
                    tour=record.tour,
                    player_id=player_id,
                    opponent_id=opponent_id,
                    match_id=record.match_id,
                    match_date=record.match_date,
                    response=response,
                    recency_weight=weight,
                    age_days=age_days,
                    terminal_class=record.terminal_class,
                    retiring_player_id=record.retiring_player_id,
                    retirement_completed_games=(
                        record.retirement_completed_games if response == 1 else None
                    ),
                    information_cutoff_utc=cutoff,
                    available_at_utc=record.available_at_utc,
                    source_id=record.source_id,
                    source_sha256=record.source_sha256,
                )
            )
    return RetirementObservationBatch(
        information_cutoff_utc=cutoff,
        observations=tuple(observations),
        exclusions=tuple(exclusions),
    )


class RetirementSourceCoverage(FrozenModel):
    """Separate global history coverage from exact-dated fit-input eligibility.

    ``complete`` retains its original, strong meaning: the source history is
    complete enough to prove a true no-history player state.  A current fit may
    instead be production-eligible when every admitted row has an exact date,
    while honestly retaining excluded undated rows.  That narrower assertion
    never authorizes a no-history posterior.
    """

    tour: Tour
    complete: bool
    assertion_id: NonEmptyStr
    verified_at_utc: datetime
    details: NonEmptyStr
    fit_input_date_eligibility_verified: bool = False
    historical_exact_date_coverage_complete: bool | None = None
    included_exact_dated_matches: Annotated[int | None, Field(ge=0)] = None
    excluded_undated_matches: Annotated[int, Field(ge=0)] = 0
    included_exact_dated_player_starts: Annotated[int | None, Field(ge=0)] = None
    excluded_undated_player_starts: Annotated[int, Field(ge=0)] = 0
    source_sha256s: tuple[str, ...] = ()
    crosswalk_sha256s: tuple[str, ...] = ()
    eligibility_rule_version: Literal[
        "global-source-completeness/v1", "exact-dated-fit-inputs/v1"
    ] = "global-source-completeness/v1"

    @field_validator("verified_at_utc")
    @classmethod
    def verified_time_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="verified_at_utc")

    @field_validator("source_sha256s", "crosswalk_sha256s")
    @classmethod
    def coverage_hashes_are_canonical(
        cls, value: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        normalized = tuple(_sha256(item, field_name=info.field_name) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError(f"{info.field_name} must be unique and sorted")
        return normalized

    @model_validator(mode="after")
    def exact_date_assertion_is_coherent(self) -> Self:
        if (
            self.historical_exact_date_coverage_complete is not None
            and self.historical_exact_date_coverage_complete != self.complete
        ):
            raise ValueError("global coverage flags disagree")
        if self.fit_input_date_eligibility_verified:
            if self.eligibility_rule_version != "exact-dated-fit-inputs/v1":
                raise ValueError("verified exact-dated inputs require the current rule version")
            if self.included_exact_dated_matches is None:
                raise ValueError("verified exact-dated inputs require an included match count")
            if self.included_exact_dated_player_starts is None:
                raise ValueError("verified exact-dated inputs require included player-starts")
            if not self.source_sha256s or not self.crosswalk_sha256s:
                raise ValueError("verified exact-dated inputs require source and crosswalk hashes")
        if self.historical_exact_date_coverage_complete is False:
            if self.excluded_undated_matches == 0:
                raise ValueError("incomplete exact-date coverage requires excluded matches")
            if self.excluded_undated_player_starts == 0:
                raise ValueError("incomplete exact-date coverage requires excluded player-starts")
        return self

    @property
    def production_fit_inputs_eligible(self) -> bool:
        """Whether B6 sufficient statistics use only eligibility-verified rows."""

        return self.complete or self.fit_input_date_eligibility_verified


class RetirementReasonCount(FrozenModel):
    """Weighted and unweighted audit counts for one terminal/exclusion reason."""

    reason: NonEmptyStr
    match_count: Annotated[int, Field(ge=0)]
    match_weight: Annotated[float, Field(ge=0)]
    player_start_count: Annotated[int, Field(ge=0)]
    player_start_weight: Annotated[float, Field(ge=0)]

    @field_validator("match_weight", "player_start_weight")
    @classmethod
    def weights_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)


class RetirementPlayerSufficientStatistics(FrozenModel):
    """Weighted player starts and the fixed 100-start shrunk beta posterior."""

    player_id: NonEmptyStr
    retirements_y: Annotated[float, Field(ge=0)]
    starts_n: Annotated[float, Field(ge=0)]
    alpha: Annotated[float, Field(gt=0)]
    beta: Annotated[float, Field(gt=0)]

    @field_validator("retirements_y", "starts_n", "alpha", "beta")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def successes_do_not_exceed_starts(self) -> Self:
        if self.retirements_y > self.starts_n:
            raise ValueError("player weighted retirements exceed weighted starts")
        return self


class RetirementFitArtifact(FrozenModel):
    """Complete content-addressed ATP- or WTA-specific B6 fit."""

    artifact_id: str
    schema_version: str = RETIREMENT_ARTIFACT_SCHEMA_VERSION
    response_coding_version: str = RETIREMENT_RESPONSE_CODING_VERSION
    tour: Tour
    source_manifest_id: NonEmptyStr
    source_manifest_sha256: str
    source_coverage: RetirementSourceCoverage
    information_cutoff_utc: datetime
    fit_cutoff_utc: datetime
    fitted_at_utc: datetime
    window_days: int = RETIREMENT_WINDOW_DAYS
    half_life_days: int = RETIREMENT_HALF_LIFE_DAYS
    prior_effective_starts: float = RETIREMENT_PRIOR_EFFECTIVE_STARTS
    reference_games: int = RETIREMENT_REFERENCE_GAMES
    intensity_transform_version: str = RETIREMENT_INTENSITY_TRANSFORM_VERSION
    included_counts: tuple[RetirementReasonCount, ...]
    excluded_counts: tuple[RetirementReasonCount, ...]
    tour_retirements_y: Annotated[float, Field(ge=0)]
    tour_starts_n: Annotated[float, Field(ge=0)]
    tour_baseline_rho: Annotated[float, Field(gt=0, lt=1)]
    tour_alpha: Annotated[float, Field(gt=0)]
    tour_beta: Annotated[float, Field(gt=0)]
    player_statistics: tuple[RetirementPlayerSufficientStatistics, ...]
    weighted_start_coverage_gate_passed: bool
    production_eligible: bool
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

    @field_validator("information_cutoff_utc", "fit_cutoff_utc", "fitted_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _aware_utc(value, field_name=info.field_name)

    @field_validator(
        "prior_effective_starts",
        "tour_retirements_y",
        "tour_starts_n",
        "tour_baseline_rho",
        "tour_alpha",
        "tour_beta",
    )
    @classmethod
    def numeric_values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def frozen_contract_and_hash_are_valid(self) -> Self:
        if self.schema_version not in {
            RETIREMENT_LEGACY_ARTIFACT_SCHEMA_VERSION,
            RETIREMENT_ARTIFACT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported retirement artifact schema version")
        if self.response_coding_version != RETIREMENT_RESPONSE_CODING_VERSION:
            raise ValueError("unsupported retirement response-coding version")
        if self.intensity_transform_version != RETIREMENT_INTENSITY_TRANSFORM_VERSION:
            raise ValueError("unsupported retirement intensity transform version")
        constants = (
            self.window_days == RETIREMENT_WINDOW_DAYS
            and self.half_life_days == RETIREMENT_HALF_LIFE_DAYS
            and self.prior_effective_starts == RETIREMENT_PRIOR_EFFECTIVE_STARTS
            and self.reference_games == RETIREMENT_REFERENCE_GAMES
        )
        if not constants:
            raise ValueError("retirement artifact constants differ from frozen B6")
        if self.source_coverage.tour is not self.tour:
            raise ValueError("source coverage tour differs from artifact tour")
        if self.source_coverage.verified_at_utc >= self.information_cutoff_utc:
            raise ValueError("source coverage assertion must be available before fit cutoff")
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("fitted_at_utc must not precede information cutoff")
        if self.fit_cutoff_utc != self.information_cutoff_utc:
            raise ValueError("B6 fit cutoff must equal its information cutoff")
        if self.tour_retirements_y > self.tour_starts_n:
            raise ValueError("tour weighted retirements exceed weighted starts")
        expected_alpha = self.tour_retirements_y + 0.5
        expected_beta = self.tour_starts_n - self.tour_retirements_y + 0.5
        expected_rho = expected_alpha / (self.tour_starts_n + 1.0)
        for observed, expected, label in (
            (self.tour_alpha, expected_alpha, "tour_alpha"),
            (self.tour_beta, expected_beta, "tour_beta"),
            (self.tour_baseline_rho, expected_rho, "tour_baseline_rho"),
        ):
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"{label} does not match frozen B6 formula")
        player_ids = tuple(item.player_id for item in self.player_statistics)
        if player_ids != tuple(sorted(player_ids)) or len(player_ids) != len(set(player_ids)):
            raise ValueError("player_statistics must be unique and sorted by player_id")
        for item in self.player_statistics:
            expected_player_alpha = (
                RETIREMENT_PRIOR_EFFECTIVE_STARTS * self.tour_baseline_rho + item.retirements_y
            )
            expected_player_beta = (
                RETIREMENT_PRIOR_EFFECTIVE_STARTS * (1.0 - self.tour_baseline_rho)
                + item.starts_n
                - item.retirements_y
            )
            if not math.isclose(item.alpha, expected_player_alpha, rel_tol=1e-12):
                raise ValueError("player alpha does not match frozen shrinkage formula")
            if not math.isclose(item.beta, expected_player_beta, rel_tol=1e-12):
                raise ValueError("player beta does not match frozen shrinkage formula")
        player_y = math.fsum(item.retirements_y for item in self.player_statistics)
        player_n = math.fsum(item.starts_n for item in self.player_statistics)
        if not math.isclose(player_y, self.tour_retirements_y, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("player retirement totals do not reconcile to tour Y")
        if not math.isclose(player_n, self.tour_starts_n, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("player start totals do not reconcile to tour N")
        expected_gate = self.tour_starts_n >= RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS
        if self.weighted_start_coverage_gate_passed != expected_gate:
            raise ValueError("weighted-start coverage gate is inconsistent")
        expected_production = (
            expected_gate and self.source_coverage.production_fit_inputs_eligible
        )
        if self.production_eligible != expected_production:
            raise ValueError("production_eligible is inconsistent with coverage gates")
        for counts in (self.included_counts, self.excluded_counts):
            reasons = tuple(item.reason for item in counts)
            if reasons != tuple(sorted(reasons)) or len(reasons) != len(set(reasons)):
                raise ValueError("reason counts must be unique and sorted")
        payload = _retirement_artifact_payload(self)
        expected_artifact_id = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if self.artifact_id != expected_artifact_id:
            raise ValueError("artifact_id does not match retirement fit content")
        return self

    def require_production_coverage(self) -> None:
        """Fail closed instead of borrowing another tour or an emergency rate."""

        if not self.source_coverage.production_fit_inputs_eligible:
            raise RetirementCoverageError(
                f"{self.tour.value} retirement fit-input eligibility is unverified"
            )
        if not self.weighted_start_coverage_gate_passed:
            raise RetirementCoverageError(
                f"{self.tour.value} retirement fit has {self.tour_starts_n:.12g} weighted "
                f"player-starts; {RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS:g} are required"
            )


def _reason_counts(
    observations: tuple[RetirementObservation, ...],
    exclusions: tuple[RetirementExclusion, ...],
) -> tuple[tuple[RetirementReasonCount, ...], tuple[RetirementReasonCount, ...]]:
    included_groups: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"matches": 0, "match_weight": 0.0, "starts": 0, "start_weight": 0.0}
    )
    match_rows: dict[str, list[RetirementObservation]] = defaultdict(list)
    for observation in observations:
        match_rows[observation.match_id].append(observation)
    for rows in match_rows.values():
        first = rows[0]
        group = included_groups[first.terminal_class.value]
        group["matches"] = int(group["matches"]) + 1
        group["match_weight"] = float(group["match_weight"]) + first.recency_weight
        group["starts"] = int(group["starts"]) + len(rows)
        group["start_weight"] = float(group["start_weight"]) + sum(
            item.recency_weight for item in rows
        )
    included = tuple(
        RetirementReasonCount(
            reason=reason,
            match_count=int(values["matches"]),
            match_weight=float(values["match_weight"]),
            player_start_count=int(values["starts"]),
            player_start_weight=float(values["start_weight"]),
        )
        for reason, values in sorted(included_groups.items())
    )

    excluded_groups: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"matches": 0, "match_weight": 0.0}
    )
    for exclusion in exclusions:
        group = excluded_groups[exclusion.exclusion_reason]
        group["matches"] = int(group["matches"]) + 1
        group["match_weight"] = float(group["match_weight"]) + (exclusion.recency_weight or 0.0)
        group["starts"] = int(group.get("starts", 0)) + 2
        group["start_weight"] = float(group.get("start_weight", 0.0)) + 2.0 * (
            exclusion.recency_weight or 0.0
        )
    excluded = tuple(
        RetirementReasonCount(
            reason=reason,
            match_count=int(values["matches"]),
            match_weight=float(values["match_weight"]),
            player_start_count=int(values.get("starts", 0)),
            player_start_weight=float(values.get("start_weight", 0.0)),
        )
        for reason, values in sorted(excluded_groups.items())
    )
    return included, excluded


def fit_retirement_artifact(
    batch: RetirementObservationBatch,
    *,
    tour: Tour,
    source_manifest_id: str,
    source_manifest_sha256: str,
    source_coverage: RetirementSourceCoverage,
    fitted_at_utc: datetime,
    software_version: str,
    config_sha256: str,
    data_sha256: str,
    code_sha256: str,
    deterministic_test_result_sha256: str,
) -> RetirementFitArtifact:
    """Fit the fixed weighted beta estimator for exactly one tour."""

    fitted_at = _aware_utc(fitted_at_utc, field_name="fitted_at_utc")
    if fitted_at < batch.information_cutoff_utc:
        raise RetirementModelError("fitted_at_utc must not precede information cutoff")
    if source_coverage.tour is not tour:
        raise RetirementModelError("source coverage cannot be borrowed across tours")
    observations = tuple(item for item in batch.observations if item.tour is tour)
    exclusions = tuple(item for item in batch.exclusions if item.tour is tour)
    y_tour = math.fsum(item.recency_weight * item.response for item in observations)
    n_tour = math.fsum(item.recency_weight for item in observations)
    if not math.isfinite(y_tour) or not math.isfinite(n_tour) or not 0 <= y_tour <= n_tour:
        raise RetirementModelError("invalid tour retirement sufficient statistics")
    rho_tour = (y_tour + 0.5) / (n_tour + 1.0)

    player_rows: dict[str, list[RetirementObservation]] = defaultdict(list)
    for observation in observations:
        player_rows[observation.player_id].append(observation)
    player_statistics: list[RetirementPlayerSufficientStatistics] = []
    for player_id, rows in sorted(player_rows.items()):
        y_player = math.fsum(item.recency_weight * item.response for item in rows)
        n_player = math.fsum(item.recency_weight for item in rows)
        if (
            not math.isfinite(y_player)
            or not math.isfinite(n_player)
            or not 0 <= y_player <= n_player
        ):
            raise RetirementModelError(f"invalid retirement statistics for player {player_id}")
        player_statistics.append(
            RetirementPlayerSufficientStatistics(
                player_id=player_id,
                retirements_y=y_player,
                starts_n=n_player,
                alpha=RETIREMENT_PRIOR_EFFECTIVE_STARTS * rho_tour + y_player,
                beta=(RETIREMENT_PRIOR_EFFECTIVE_STARTS * (1.0 - rho_tour) + n_player - y_player),
            )
        )
    included_counts, excluded_counts = _reason_counts(observations, exclusions)
    payload: dict[str, Any] = {
        "schema_version": RETIREMENT_ARTIFACT_SCHEMA_VERSION,
        "response_coding_version": RETIREMENT_RESPONSE_CODING_VERSION,
        "tour": tour,
        "source_manifest_id": source_manifest_id,
        "source_manifest_sha256": source_manifest_sha256,
        "source_coverage": source_coverage,
        "information_cutoff_utc": batch.information_cutoff_utc,
        "fit_cutoff_utc": batch.information_cutoff_utc,
        "fitted_at_utc": fitted_at,
        "window_days": RETIREMENT_WINDOW_DAYS,
        "half_life_days": RETIREMENT_HALF_LIFE_DAYS,
        "prior_effective_starts": RETIREMENT_PRIOR_EFFECTIVE_STARTS,
        "reference_games": RETIREMENT_REFERENCE_GAMES,
        "intensity_transform_version": RETIREMENT_INTENSITY_TRANSFORM_VERSION,
        "included_counts": included_counts,
        "excluded_counts": excluded_counts,
        "tour_retirements_y": y_tour,
        "tour_starts_n": n_tour,
        "tour_baseline_rho": rho_tour,
        "tour_alpha": y_tour + 0.5,
        "tour_beta": n_tour - y_tour + 0.5,
        "player_statistics": tuple(player_statistics),
        "weighted_start_coverage_gate_passed": (n_tour >= RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS),
        "production_eligible": (
            source_coverage.production_fit_inputs_eligible
            and n_tour >= RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS
        ),
        "software_version": software_version,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "code_sha256": code_sha256,
        "deterministic_test_result_sha256": deterministic_test_result_sha256,
    }
    json_payload = RetirementFitArtifact.model_construct(
        artifact_id="0" * 64, **payload
    ).model_dump(mode="json", exclude={"artifact_id"})
    artifact_id = hashlib.sha256(_canonical_json_bytes(json_payload)).hexdigest()
    return RetirementFitArtifact(artifact_id=artifact_id, **payload)


class PersistedRetirementFitArtifact(FrozenModel):
    """Verified location and parsed retirement artifact."""

    directory: Path
    artifact: RetirementFitArtifact

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def artifact_path(self) -> Path:
        return self.directory / _ARTIFACT_FILENAME


def _retirement_artifact_payload(artifact: RetirementFitArtifact) -> dict[str, Any]:
    """Return the hash payload while preserving canonical legacy-v1 bytes."""

    payload = artifact.model_dump(mode="json", exclude={"artifact_id"})
    if artifact.schema_version == RETIREMENT_LEGACY_ARTIFACT_SCHEMA_VERSION:
        coverage = payload["source_coverage"]
        payload["source_coverage"] = {
            key: coverage[key]
            for key in ("tour", "complete", "assertion_id", "verified_at_utc", "details")
        }
    return payload


def _artifact_bytes(artifact: RetirementFitArtifact) -> bytes:
    payload = _retirement_artifact_payload(artifact)
    return _canonical_json_bytes({"artifact_id": artifact.artifact_id, **payload})


def reissue_retirement_fit_eligibility_metadata(
    artifact: RetirementFitArtifact,
    source_coverage: RetirementSourceCoverage,
) -> RetirementFitArtifact:
    """Reissue only B6 eligibility metadata without recomputing sufficient statistics."""

    if source_coverage.tour is not artifact.tour:
        raise RetirementModelError("source coverage cannot be borrowed across tours")
    payload = {
        name: getattr(artifact, name)
        for name in type(artifact).model_fields
        if name not in {"artifact_id", "source_coverage"}
    }
    payload["schema_version"] = RETIREMENT_ARTIFACT_SCHEMA_VERSION
    payload["source_coverage"] = source_coverage
    payload["production_eligible"] = (
        source_coverage.production_fit_inputs_eligible
        and artifact.weighted_start_coverage_gate_passed
    )
    provisional = RetirementFitArtifact.model_construct(artifact_id="0" * 64, **payload)
    artifact_id = hashlib.sha256(
        _canonical_json_bytes(_retirement_artifact_payload(provisional))
    ).hexdigest()
    return RetirementFitArtifact(artifact_id=artifact_id, **payload)


def write_retirement_fit_artifact(
    artifact: RetirementFitArtifact,
    artifact_root: str | Path,
) -> PersistedRetirementFitArtifact:
    """Atomically publish an immutable retirement fit without overwriting."""

    artifact = RetirementFitArtifact.model_validate(artifact.model_dump(mode="python"))
    cutoff_segment = artifact.information_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    parent = Path(artifact_root).resolve() / artifact.tour.value.lower() / cutoff_segment
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact.artifact_id[:32]
    if target.exists():
        existing = load_retirement_fit_artifact(target)
        if existing.artifact != artifact:
            raise RetirementArtifactIntegrityError(
                "existing retirement artifact path contains conflicting content"
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
                raise RetirementArtifactError(
                    f"cannot publish retirement artifact {target}: {exc}"
                ) from exc
            existing = load_retirement_fit_artifact(target)
            if existing.artifact != artifact:
                raise RetirementArtifactIntegrityError(
                    "concurrent retirement artifact publication conflicted"
                ) from exc
            return existing
        return load_retirement_fit_artifact(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_retirement_fit_artifact(
    directory: str | Path,
) -> PersistedRetirementFitArtifact:
    """Load only canonical, schema-valid content at its content-addressed path."""

    artifact_directory = Path(directory)
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise RetirementArtifactIntegrityError(
            f"retirement artifact is not a regular directory: {directory}"
        )
    artifact_path = artifact_directory / _ARTIFACT_FILENAME
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise RetirementArtifactIntegrityError(
            f"retirement artifact JSON is missing: {artifact_path}"
        )
    try:
        raw = artifact_path.read_bytes()
        artifact = RetirementFitArtifact.model_validate_json(raw)
    except Exception as exc:
        raise RetirementArtifactIntegrityError(
            f"retirement artifact JSON is invalid: {exc}"
        ) from exc
    if raw != _artifact_bytes(artifact):
        raise RetirementArtifactIntegrityError("retirement artifact JSON is not canonical")
    if artifact_directory.name != artifact.artifact_id[:32]:
        raise RetirementArtifactIntegrityError(
            "retirement artifact directory does not match its content identity"
        )
    return PersistedRetirementFitArtifact(directory=artifact_directory, artifact=artifact)


class RetirementHistoryBand(StrEnum):
    """Diagnostic-only weighted-history bands; the estimator never branches."""

    NO_HISTORY = "no_history"
    SPARSE = "sparse"
    INTERMEDIATE = "intermediate"
    SUBSTANTIAL = "substantial"


def retirement_history_band(weighted_starts: float) -> RetirementHistoryBand:
    weighted_starts = _finite(weighted_starts, field_name="weighted_starts")
    if weighted_starts < 0:
        raise RetirementModelError("weighted_starts must be nonnegative")
    if weighted_starts == 0:
        return RetirementHistoryBand.NO_HISTORY
    if weighted_starts < 25:
        return RetirementHistoryBand.SPARSE
    if weighted_starts < 100:
        return RetirementHistoryBand.INTERMEDIATE
    return RetirementHistoryBand.SUBSTANTIAL


class PlayerRetirementPosterior(FrozenModel):
    """Player beta posterior retrieved from one tour-specific fit artifact."""

    player_id: NonEmptyStr
    tour: Tour
    retirements_y: Annotated[float, Field(ge=0)]
    starts_n: Annotated[float, Field(ge=0)]
    alpha: Annotated[float, Field(gt=0)]
    beta: Annotated[float, Field(gt=0)]
    mean_rho: Annotated[float, Field(gt=0, lt=1)]
    history_band: RetirementHistoryBand
    artifact_id: str

    @field_validator("artifact_id")
    @classmethod
    def artifact_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="artifact_id")

    @field_validator("retirements_y", "starts_n", "alpha", "beta", "mean_rho")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def posterior_is_coherent(self) -> Self:
        if self.retirements_y > self.starts_n:
            raise ValueError("weighted retirements exceed starts")
        if self.history_band is not retirement_history_band(self.starts_n):
            raise ValueError("history band does not match weighted starts")
        expected_mean = self.alpha / (self.alpha + self.beta)
        if not math.isclose(self.mean_rho, expected_mean, rel_tol=1e-12):
            raise ValueError("mean_rho does not equal beta posterior mean")
        return self


def player_retirement_posterior(
    artifact: RetirementFitArtifact,
    player_id: str,
    *,
    require_production_coverage: bool = True,
) -> PlayerRetirementPosterior:
    """Retrieve a player's fixed posterior, using the tour prior only for true no-history."""

    if not player_id.strip():
        raise RetirementModelError("player_id must be nonempty")
    statistics = next(
        (item for item in artifact.player_statistics if item.player_id == player_id), None
    )
    if statistics is None and not artifact.source_coverage.complete:
        raise RetirementCoverageError(
            "no-history posterior is unavailable without complete source coverage"
        )
    if require_production_coverage:
        artifact.require_production_coverage()
    if statistics is None:
        y_player = 0.0
        n_player = 0.0
        alpha = RETIREMENT_PRIOR_EFFECTIVE_STARTS * artifact.tour_baseline_rho
        beta = RETIREMENT_PRIOR_EFFECTIVE_STARTS * (1.0 - artifact.tour_baseline_rho)
    else:
        y_player = statistics.retirements_y
        n_player = statistics.starts_n
        alpha = statistics.alpha
        beta = statistics.beta
    return PlayerRetirementPosterior(
        player_id=player_id,
        tour=artifact.tour,
        retirements_y=y_player,
        starts_n=n_player,
        alpha=alpha,
        beta=beta,
        mean_rho=alpha / (alpha + beta),
        history_band=retirement_history_band(n_player),
        artifact_id=artifact.artifact_id,
    )


class RetirementIntensity(FrozenModel):
    """Stable conversion of a 22-game match incidence to a game-boundary hazard."""

    rho: Annotated[float, Field(ge=0, le=1)]
    intensity_lambda: Annotated[float, Field(ge=0)]
    discrete_hazard: Annotated[float, Field(ge=0, le=1)]
    reference_games: int = RETIREMENT_REFERENCE_GAMES
    transform_version: str = RETIREMENT_INTENSITY_TRANSFORM_VERSION

    @field_validator("rho", "intensity_lambda", "discrete_hazard")
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def mapping_is_exact(self) -> Self:
        if self.reference_games != RETIREMENT_REFERENCE_GAMES:
            raise ValueError("retirement reference exposure must be 22 games")
        if self.transform_version != RETIREMENT_INTENSITY_TRANSFORM_VERSION:
            raise ValueError("unsupported retirement intensity transform")
        mapped_rho = -math.expm1(-self.reference_games * self.intensity_lambda)
        if not math.isclose(mapped_rho, self.rho, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("intensity does not reproduce rho at 22 game boundaries")
        expected_hazard = -math.expm1(-self.intensity_lambda)
        if not math.isclose(expected_hazard, self.discrete_hazard, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("discrete hazard does not match intensity")
        return self


def retirement_probability_to_intensity(rho: float) -> RetirementIntensity:
    """Map ``rho`` with stable log1p/expm1 operations and the exact rho=1 edge rule."""

    rho = _finite(rho, field_name="rho")
    if not 0.0 <= rho <= 1.0:
        raise RetirementModelError("rho must lie in [0, 1]")
    adjusted_rho = math.nextafter(1.0, 0.0) if rho == 1.0 else rho
    intensity = -math.log1p(-adjusted_rho) / RETIREMENT_REFERENCE_GAMES
    hazard = -math.expm1(-intensity)
    return RetirementIntensity(
        rho=adjusted_rho,
        intensity_lambda=intensity,
        discrete_hazard=hazard,
    )


class RetirementScenario(FrozenModel):
    """Versioned evidence record for one player's named log-hazard multiplier."""

    schema_version: str = RETIREMENT_SCENARIO_SCHEMA_VERSION
    scenario_id: NonEmptyStr
    scenario_version: NonEmptyStr
    named_state: NonEmptyStr
    player_id: NonEmptyStr
    central: bool = False
    log_hazard_ratio: float
    weight: Annotated[float | None, Field(ge=0, le=1)] = None
    source_id: NonEmptyStr
    source_sha256: str
    observation_at_utc: datetime
    publication_at_utc: datetime
    authoring_method: NonEmptyStr

    @field_validator("log_hazard_ratio")
    @classmethod
    def log_ratio_is_finite_and_exponentiable(cls, value: float) -> float:
        value = _finite(value, field_name="log_hazard_ratio")
        try:
            multiplier = math.exp(value)
        except OverflowError as exc:
            raise ValueError("log_hazard_ratio overflows its multiplier") from exc
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("log_hazard_ratio underflows or overflows its positive multiplier")
        return value

    @field_validator("weight")
    @classmethod
    def optional_weight_is_finite(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value, field_name="weight")

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="source_sha256")

    @field_validator("observation_at_utc", "publication_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _aware_utc(value, field_name=info.field_name)

    @model_validator(mode="after")
    def record_is_versioned_and_ordered(self) -> Self:
        if self.schema_version != RETIREMENT_SCENARIO_SCHEMA_VERSION:
            raise ValueError("unsupported retirement scenario schema")
        if self.publication_at_utc < self.observation_at_utc:
            raise ValueError("publication time cannot precede observation time")
        if self.central and self.log_hazard_ratio != 0.0:
            raise ValueError("the central retirement scenario must have log-hazard ratio zero")
        return self

    @property
    def hazard_multiplier(self) -> float:
        return math.exp(self.log_hazard_ratio)


class RetirementScenarioMixture(FrozenModel):
    """Pre-cutoff fixed scenario weights drawn once per simulated path."""

    mixture_id: NonEmptyStr
    player_id: NonEmptyStr
    information_cutoff_utc: datetime
    scenarios: tuple[RetirementScenario, ...]

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return _aware_utc(value, field_name="information_cutoff_utc")

    @model_validator(mode="after")
    def scenarios_are_valid(self) -> Self:
        if not self.scenarios:
            raise ValueError("scenario mixture must not be empty")
        ids = tuple(item.scenario_id for item in self.scenarios)
        if len(ids) != len(set(ids)):
            raise ValueError("scenario IDs must be unique within a mixture")
        if any(item.player_id != self.player_id for item in self.scenarios):
            raise ValueError("all mixture scenarios must affect the declared player")
        if any(item.weight is None for item in self.scenarios):
            raise ValueError("every blended scenario must have an explicit weight")
        total = math.fsum(item.weight or 0.0 for item in self.scenarios)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("scenario weights must sum to 1")
        for item in self.scenarios:
            if (
                item.observation_at_utc >= self.information_cutoff_utc
                or item.publication_at_utc >= self.information_cutoff_utc
            ):
                raise ValueError("scenario evidence must be strictly before information cutoff")
        return self


def draw_retirement_scenario(
    mixture: RetirementScenarioMixture,
    rng: Generator,
) -> RetirementScenario:
    """Use one explicit RNG draw to select a predeclared path-level scenario."""

    draw = float(rng.random())
    cumulative = 0.0
    for scenario in mixture.scenarios:
        assert scenario.weight is not None
        cumulative += scenario.weight
        if draw < cumulative:
            return scenario
    return mixture.scenarios[-1]


class RetirementPathDraw(FrozenModel):
    """Path-local beta draw and its base/scenario-adjusted game intensity."""

    player_id: NonEmptyStr
    artifact_id: str
    scenario_id: NonEmptyStr
    posterior_rho_draw: Annotated[float, Field(ge=0, lt=1)]
    base_intensity_lambda: Annotated[float, Field(ge=0)]
    scenario_log_hazard_ratio: float
    adjusted_intensity_lambda: Annotated[float, Field(ge=0)]
    adjusted_discrete_hazard: Annotated[float, Field(ge=0, le=1)]

    @field_validator("artifact_id")
    @classmethod
    def artifact_hash_is_valid(cls, value: str) -> str:
        return _sha256(value, field_name="artifact_id")

    @field_validator(
        "posterior_rho_draw",
        "base_intensity_lambda",
        "scenario_log_hazard_ratio",
        "adjusted_intensity_lambda",
        "adjusted_discrete_hazard",
    )
    @classmethod
    def values_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def intensities_are_coherent(self) -> Self:
        try:
            multiplier = math.exp(self.scenario_log_hazard_ratio)
        except OverflowError as exc:
            raise ValueError("scenario multiplier overflows") from exc
        expected = self.base_intensity_lambda * multiplier
        if not math.isfinite(expected) or not math.isclose(
            expected, self.adjusted_intensity_lambda, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("adjusted intensity does not match the scenario multiplier")
        expected_hazard = -math.expm1(-self.adjusted_intensity_lambda)
        if not math.isclose(
            expected_hazard, self.adjusted_discrete_hazard, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("adjusted discrete hazard does not match intensity")
        return self


def draw_player_retirement_path(
    posterior: PlayerRetirementPosterior,
    scenario: RetirementScenario,
    rng: Generator,
) -> RetirementPathDraw:
    """Draw one player's beta incidence and apply only the selected named scenario."""

    if scenario.player_id != posterior.player_id:
        raise RetirementModelError("retirement scenario targets a different player")
    rho = float(rng.beta(posterior.alpha, posterior.beta))
    base = retirement_probability_to_intensity(rho)
    adjusted = base.intensity_lambda * scenario.hazard_multiplier
    if not math.isfinite(adjusted) or adjusted < 0:
        raise RetirementModelError("scenario-adjusted retirement intensity is invalid")
    return RetirementPathDraw(
        player_id=posterior.player_id,
        artifact_id=posterior.artifact_id,
        scenario_id=scenario.scenario_id,
        posterior_rho_draw=base.rho,
        base_intensity_lambda=base.intensity_lambda,
        scenario_log_hazard_ratio=scenario.log_hazard_ratio,
        adjusted_intensity_lambda=adjusted,
        adjusted_discrete_hazard=-math.expm1(-adjusted),
    )


def draw_mixture_player_retirement_path(
    posterior: PlayerRetirementPosterior,
    mixture: RetirementScenarioMixture,
    rng: Generator,
) -> RetirementPathDraw:
    """Enforce the required path order: scenario selection, then beta draw."""

    if mixture.player_id != posterior.player_id:
        raise RetirementModelError("retirement scenario mixture targets a different player")
    scenario = draw_retirement_scenario(mixture, rng)
    return draw_player_retirement_path(posterior, scenario, rng)


class CompetingRetirementOutcome(StrEnum):
    """The only three legal outcomes at one eligible completed-game boundary."""

    NO_RETIREMENT = "no_retirement"
    PLAYER_A_RETIRES = "player_a_retires"
    PLAYER_B_RETIRES = "player_b_retires"


class CompetingRiskProbabilities(FrozenModel):
    """Exact two-player categorical mass at one eligible game boundary."""

    p_no_retirement: Annotated[float, Field(ge=0, le=1)]
    p_player_a_retires: Annotated[float, Field(ge=0, le=1)]
    p_player_b_retires: Annotated[float, Field(ge=0, le=1)]
    competing_risk_version: str = RETIREMENT_COMPETING_RISK_VERSION

    @field_validator("p_no_retirement", "p_player_a_retires", "p_player_b_retires")
    @classmethod
    def probabilities_are_finite(cls, value: float, info: Any) -> float:
        return _finite(value, field_name=info.field_name)

    @model_validator(mode="after")
    def mass_sums_to_one(self) -> Self:
        if self.competing_risk_version != RETIREMENT_COMPETING_RISK_VERSION:
            raise ValueError("unsupported retirement competing-risk version")
        total = self.p_no_retirement + self.p_player_a_retires + self.p_player_b_retires
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("competing-risk probabilities do not sum to 1")
        return self


def competing_retirement_probabilities(
    intensity_a: float,
    intensity_b: float,
) -> CompetingRiskProbabilities:
    """Form exact stable B6.7 probabilities from finite nonnegative intensities."""

    intensity_a = _finite(intensity_a, field_name="intensity_a")
    intensity_b = _finite(intensity_b, field_name="intensity_b")
    if intensity_a < 0 or intensity_b < 0:
        raise RetirementModelError("retirement intensities must be nonnegative")
    total = intensity_a + intensity_b
    if not math.isfinite(total):
        raise RetirementModelError("combined retirement intensity must be finite")
    if total == 0:
        return CompetingRiskProbabilities(
            p_no_retirement=1.0,
            p_player_a_retires=0.0,
            p_player_b_retires=0.0,
        )
    p_no = math.exp(-total)
    retirement_mass = -math.expm1(-total)
    p_a = retirement_mass * intensity_a / total
    p_b = retirement_mass * intensity_b / total
    return CompetingRiskProbabilities(
        p_no_retirement=p_no,
        p_player_a_retires=p_a,
        p_player_b_retires=p_b,
    )


class RetirementBoundaryDraw(FrozenModel):
    """One categorical draw and the exact probabilities that generated it."""

    outcome: CompetingRetirementOutcome
    probabilities: CompetingRiskProbabilities


def draw_competing_retirement(
    intensity_a: float,
    intensity_b: float,
    rng: Generator,
) -> RetirementBoundaryDraw:
    """Draw exactly once, except that a zero-total hazard must not advance the RNG."""

    probabilities = competing_retirement_probabilities(intensity_a, intensity_b)
    if intensity_a == 0 and intensity_b == 0:
        return RetirementBoundaryDraw(
            outcome=CompetingRetirementOutcome.NO_RETIREMENT,
            probabilities=probabilities,
        )
    draw = float(rng.random())
    if draw < probabilities.p_no_retirement:
        outcome = CompetingRetirementOutcome.NO_RETIREMENT
    elif draw < probabilities.p_no_retirement + probabilities.p_player_a_retires:
        outcome = CompetingRetirementOutcome.PLAYER_A_RETIRES
    else:
        outcome = CompetingRetirementOutcome.PLAYER_B_RETIRES
    return RetirementBoundaryDraw(outcome=outcome, probabilities=probabilities)


__all__ = [
    "RETIREMENT_ARTIFACT_SCHEMA_VERSION",
    "RETIREMENT_COMPETING_RISK_VERSION",
    "RETIREMENT_HALF_LIFE_DAYS",
    "RETIREMENT_INTENSITY_TRANSFORM_VERSION",
    "RETIREMENT_LEGACY_ARTIFACT_SCHEMA_VERSION",
    "RETIREMENT_MINIMUM_WEIGHTED_TOUR_STARTS",
    "RETIREMENT_PRIOR_EFFECTIVE_STARTS",
    "RETIREMENT_REFERENCE_GAMES",
    "RETIREMENT_RESPONSE_CODING_VERSION",
    "RETIREMENT_SCENARIO_SCHEMA_VERSION",
    "RETIREMENT_WINDOW_DAYS",
    "CompetingRetirementOutcome",
    "CompetingRiskProbabilities",
    "HistoricalTerminalClass",
    "HistoricalTerminationClass",
    "HistoricalTerminationInput",
    "OfficialTerminalStatus",
    "PersistedRetirementFitArtifact",
    "PlayerRetirementPosterior",
    "RetirementArtifactError",
    "RetirementArtifactIntegrityError",
    "RetirementBoundaryDraw",
    "RetirementCoverageError",
    "RetirementExclusion",
    "RetirementFitArtifact",
    "RetirementHistoryBand",
    "RetirementIntensity",
    "RetirementModelError",
    "RetirementObservation",
    "RetirementObservationBatch",
    "RetirementPathDraw",
    "RetirementPlayerSufficientStatistics",
    "RetirementReasonCount",
    "RetirementScenario",
    "RetirementScenarioMixture",
    "RetirementSourceCoverage",
    "StartedEvidence",
    "build_retirement_observations",
    "competing_retirement_probabilities",
    "draw_competing_retirement",
    "draw_mixture_player_retirement_path",
    "draw_player_retirement_path",
    "draw_retirement_scenario",
    "fit_retirement_artifact",
    "load_retirement_fit_artifact",
    "normalize_historical_termination",
    "normalize_historical_terminations_before_cutoff",
    "player_retirement_posterior",
    "reissue_retirement_fit_eligibility_metadata",
    "retirement_history_band",
    "retirement_probability_to_intensity",
    "retirement_recency_weight",
    "write_retirement_fit_artifact",
]
