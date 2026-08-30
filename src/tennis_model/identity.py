"""Canonical identities for source-backed tennis records.

Names are deliberately excluded from canonical identifiers.  A player identity is
anchored to a stable external identifier within a source and tour namespace.  A
match identity is anchored to the source's own tournament and match keys; the
participants are retained for validation and duplicate/collision detection, but
do not change the match identifier.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID, uuid5

from pydantic import field_validator, model_validator

from tennis_model.schemas import FrozenModel, Tour

# Fixed, project-owned UUID namespace.  Changing this value is an identity-schema
# change because every generated identifier would change.
_IDENTITY_NAMESPACE = UUID("97397fa1-d73f-5c08-8ba7-145f2cfe0e3a")


class IdentityAnomalyCode(StrEnum):
    """Stable machine-readable identity and duplicate anomaly codes."""

    MISSING_SOURCE_NAMESPACE = "missing_source_namespace"
    MISSING_TOUR = "missing_tour"
    INVALID_TOUR = "invalid_tour"
    MISSING_PLAYER_EXTERNAL_ID = "missing_player_external_id"
    MISSING_TOURNEY_ID = "missing_tourney_id"
    MISSING_MATCH_NUM = "missing_match_num"
    MISSING_PLAYER_ID = "missing_player_id"
    SELF_MATCH = "self_match"
    MISSING_SOURCE_ROW_REF = "missing_source_row_ref"
    MISSING_RECORD_FINGERPRINT = "missing_record_fingerprint"
    DUPLICATE_MATCH = "duplicate_match"
    CONFLICTING_MATCH_RECORDS = "conflicting_match_records"
    MATCH_KEY_COLLISION = "match_key_collision"


class IdentityValidationError(ValueError):
    """A rejected identity carrying a stable typed anomaly code."""

    def __init__(
        self,
        code: IdentityAnomalyCode,
        *,
        field: str,
        value: object | None = None,
    ) -> None:
        self.code = code
        self.field = field
        self.value = value
        super().__init__(f"{code.value}: invalid or missing {field}")


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """Canonical player identity plus non-identifying display aliases."""

    player_id: str
    source_namespace: str
    tour: Tour
    external_id: str
    display_aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchKey:
    """The complete source key used to generate a canonical match ID."""

    source_namespace: str
    tour: Tour
    tourney_id: str
    match_num: str


@dataclass(frozen=True, slots=True)
class MatchIdentity:
    """Canonical match identity with a validated, unordered player pair."""

    match_id: str
    key: MatchKey
    participant_ids: tuple[str, str]


class CanonicalMatchIdentity(FrozenModel):
    """Stable scheduled-match identity used by operational prediction locks.

    The identity key is either an official/source match identifier or an
    event-edition draw slot.  Schedule, cutoff, model, framework, code, and
    settlement state are deliberately absent.  ``resolved_at_utc`` and source
    provenance document the resolution but do not participate in the ID.
    """

    schema_version: Literal["canonical-match-identity/v2"] = "canonical-match-identity/v2"
    canonical_match_id: str
    resolution_method: Literal["official_source_id", "event_edition_draw_slot"]
    source_namespace: str
    tour: Tour
    official_match_id: str | None = None
    event_edition_id: str | None = None
    draw_slot: str | None = None
    participant_ids: tuple[str, str]
    source_id: str
    source_sha256: str
    source_locator: str
    resolved_at_utc: datetime

    @field_validator(
        "canonical_match_id",
        "source_namespace",
        "official_match_id",
        "event_edition_id",
        "draw_slot",
        "source_id",
        "source_locator",
    )
    @classmethod
    def text_is_present(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @field_validator("source_sha256")
    @classmethod
    def source_digest_is_valid(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("source_sha256 must contain 64 hexadecimal characters")
        return normalized

    @field_validator("resolved_at_utc")
    @classmethod
    def resolution_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("resolved_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def stable_key_is_coherent(self) -> Self:
        if len(set(self.participant_ids)) != 2 or any(
            not item.strip() for item in self.participant_ids
        ):
            raise ValueError("canonical match identity requires two distinct participants")
        official = self.official_match_id is not None
        draw = self.event_edition_id is not None and self.draw_slot is not None
        if self.resolution_method == "official_source_id":
            if not official or self.event_edition_id is not None or self.draw_slot is not None:
                raise ValueError("official identity requires only official_match_id")
        elif not draw or official:
            raise ValueError("draw-slot identity requires event_edition_id and draw_slot only")
        if self.canonical_match_id != self.expected_match_id:
            raise ValueError("canonical_match_id does not match the stable source key")
        return self

    @property
    def expected_match_id(self) -> str:
        key: tuple[str, ...]
        if self.resolution_method == "official_source_id":
            key = (
                self.source_namespace.casefold(),
                self.tour.value,
                self.official_match_id or "",
            )
        else:
            key = (
                self.source_namespace.casefold(),
                self.tour.value,
                self.event_edition_id or "",
                self.draw_slot or "",
            )
        payload = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return f"match_{hashlib.sha256(payload).hexdigest()}"

    @property
    def base_lock_id(self) -> str:
        return f"TMV1-{self.tour.value}-{self.canonical_match_id.removeprefix('match_')[:20]}"

    @classmethod
    def from_official_id(
        cls,
        *,
        source_namespace: str,
        tour: Tour,
        official_match_id: str,
        participant_ids: tuple[str, str],
        source_id: str,
        source_sha256: str,
        source_locator: str,
        resolved_at_utc: datetime,
    ) -> CanonicalMatchIdentity:
        key = (source_namespace.casefold(), tour.value, official_match_id.strip())
        payload = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return cls(
            canonical_match_id=f"match_{hashlib.sha256(payload).hexdigest()}",
            resolution_method="official_source_id",
            source_namespace=source_namespace,
            tour=tour,
            official_match_id=official_match_id,
            participant_ids=participant_ids,
            source_id=source_id,
            source_sha256=source_sha256,
            source_locator=source_locator,
            resolved_at_utc=resolved_at_utc,
        )

    @classmethod
    def from_draw_slot(
        cls,
        *,
        source_namespace: str,
        tour: Tour,
        event_edition_id: str,
        draw_slot: str,
        participant_ids: tuple[str, str],
        source_id: str,
        source_sha256: str,
        source_locator: str,
        resolved_at_utc: datetime,
    ) -> CanonicalMatchIdentity:
        key = (
            source_namespace.casefold(),
            tour.value,
            event_edition_id.strip(),
            draw_slot.strip(),
        )
        payload = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return cls(
            canonical_match_id=f"match_{hashlib.sha256(payload).hexdigest()}",
            resolution_method="event_edition_draw_slot",
            source_namespace=source_namespace,
            tour=tour,
            event_edition_id=event_edition_id,
            draw_slot=draw_slot,
            participant_ids=participant_ids,
            source_id=source_id,
            source_sha256=source_sha256,
            source_locator=source_locator,
            resolved_at_utc=resolved_at_utc,
        )


@dataclass(frozen=True, slots=True)
class MatchRecordIdentity:
    """Lineage needed to identify duplicate and conflicting source records.

    ``record_fingerprint`` should be a deterministic digest of the complete raw
    source record (not just the identity columns).  The identity layer treats it
    as opaque so ingestion can choose its canonical byte/field representation.
    """

    match: MatchIdentity
    source_row_ref: str
    record_fingerprint: str


class MatchDuplicateKind(StrEnum):
    """Classification of a group sharing one generated match ID."""

    EXACT_DUPLICATE = "exact_duplicate"
    CONFLICTING_RECORDS = "conflicting_records"
    MATCH_KEY_COLLISION = "match_key_collision"


@dataclass(frozen=True, slots=True)
class MatchDuplicateGroup:
    """All records in a duplicate/collision group; no record is selected."""

    match_id: str
    kind: MatchDuplicateKind
    anomaly_code: IdentityAnomalyCode
    records: tuple[MatchRecordIdentity, ...]


def _required_text(
    value: object | None,
    *,
    field: str,
    code: IdentityAnomalyCode,
) -> str:
    if value is None:
        raise IdentityValidationError(code, field=field, value=value)
    normalized = str(value).strip()
    if not normalized:
        raise IdentityValidationError(code, field=field, value=value)
    return normalized


def _source_namespace(value: object | None) -> str:
    # Source namespaces are controlled identifiers, not source-provided values;
    # case-folding avoids accidental parallel namespaces such as Sackmann/sackmann.
    return _required_text(
        value,
        field="source_namespace",
        code=IdentityAnomalyCode.MISSING_SOURCE_NAMESPACE,
    ).casefold()


def _tour(value: Tour | str | None) -> Tour:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise IdentityValidationError(
            IdentityAnomalyCode.MISSING_TOUR,
            field="tour",
            value=value,
        )
    if isinstance(value, Tour):
        return value
    try:
        return Tour(str(value).strip().upper())
    except ValueError as exc:
        raise IdentityValidationError(
            IdentityAnomalyCode.INVALID_TOUR,
            field="tour",
            value=value,
        ) from exc


def _uuid_id(kind: str, parts: Iterable[str]) -> str:
    canonical_name = "\x1f".join(("tennis-model-identity-v1", kind, *parts))
    prefix = "player" if kind == "player" else "match"
    return f"{prefix}_{uuid5(_IDENTITY_NAMESPACE, canonical_name)}"


def canonical_player_id(
    source_namespace: object,
    tour: Tour | str,
    external_id: object,
) -> str:
    """Return a deterministic ID from source, tour, and stable external ID.

    Player names are intentionally not accepted by this function.
    """

    source = _source_namespace(source_namespace)
    canonical_tour = _tour(tour)
    source_player_id = _required_text(
        external_id,
        field="external_id",
        code=IdentityAnomalyCode.MISSING_PLAYER_EXTERNAL_ID,
    )
    return _uuid_id("player", (source, canonical_tour.value, source_player_id))


def make_player_identity(
    source_namespace: object,
    tour: Tour | str,
    external_id: object,
    *,
    display_name: object | None = None,
) -> PlayerIdentity:
    """Build a canonical player identity without using a name as identity."""

    source = _source_namespace(source_namespace)
    canonical_tour = _tour(tour)
    source_player_id = _required_text(
        external_id,
        field="external_id",
        code=IdentityAnomalyCode.MISSING_PLAYER_EXTERNAL_ID,
    )
    display_aliases: tuple[str, ...] = ()
    if display_name is not None and str(display_name).strip():
        display_aliases = (str(display_name).strip(),)
    return PlayerIdentity(
        player_id=_uuid_id("player", (source, canonical_tour.value, source_player_id)),
        source_namespace=source,
        tour=canonical_tour,
        external_id=source_player_id,
        display_aliases=display_aliases,
    )


def make_match_key(
    source_namespace: object,
    tour: Tour | str,
    tourney_id: object,
    match_num: object,
) -> MatchKey:
    """Validate and normalize a source match key."""

    return MatchKey(
        source_namespace=_source_namespace(source_namespace),
        tour=_tour(tour),
        tourney_id=_required_text(
            tourney_id,
            field="tourney_id",
            code=IdentityAnomalyCode.MISSING_TOURNEY_ID,
        ),
        match_num=_required_text(
            match_num,
            field="match_num",
            code=IdentityAnomalyCode.MISSING_MATCH_NUM,
        ),
    )


def canonical_match_id(
    source_namespace: object,
    tour: Tour | str,
    tourney_id: object,
    match_num: object,
) -> str:
    """Return an ID based only on the stable source match key."""

    key = make_match_key(source_namespace, tour, tourney_id, match_num)
    return _uuid_id(
        "match",
        (
            key.source_namespace,
            key.tour.value,
            key.tourney_id,
            key.match_num,
        ),
    )


def make_match_identity(
    source_namespace: object,
    tour: Tour | str,
    tourney_id: object,
    match_num: object,
    *,
    player_a_id: object,
    player_b_id: object,
) -> MatchIdentity:
    """Build a canonical match identity and reject missing/self participants."""

    key = make_match_key(source_namespace, tour, tourney_id, match_num)
    player_a = _required_text(
        player_a_id,
        field="player_a_id",
        code=IdentityAnomalyCode.MISSING_PLAYER_ID,
    )
    player_b = _required_text(
        player_b_id,
        field="player_b_id",
        code=IdentityAnomalyCode.MISSING_PLAYER_ID,
    )
    if player_a == player_b:
        raise IdentityValidationError(
            IdentityAnomalyCode.SELF_MATCH,
            field="participant_ids",
            value=(player_a, player_b),
        )
    participants = tuple(sorted((player_a, player_b)))
    return MatchIdentity(
        match_id=_uuid_id(
            "match",
            (
                key.source_namespace,
                key.tour.value,
                key.tourney_id,
                key.match_num,
            ),
        ),
        key=key,
        participant_ids=(participants[0], participants[1]),
    )


def make_match_record_identity(
    match: MatchIdentity,
    *,
    source_row_ref: object,
    record_fingerprint: object,
) -> MatchRecordIdentity:
    """Attach raw-row lineage for duplicate/correction detection."""

    return MatchRecordIdentity(
        match=match,
        source_row_ref=_required_text(
            source_row_ref,
            field="source_row_ref",
            code=IdentityAnomalyCode.MISSING_SOURCE_ROW_REF,
        ),
        record_fingerprint=_required_text(
            record_fingerprint,
            field="record_fingerprint",
            code=IdentityAnomalyCode.MISSING_RECORD_FINGERPRINT,
        ),
    )


def detect_match_duplicate_groups(
    records: Iterable[MatchRecordIdentity],
) -> tuple[MatchDuplicateGroup, ...]:
    """Classify every match-ID group containing more than one raw record.

    Conflicting records are never resolved here.  The returned group contains
    every input record so ingestion can quarantine it or apply a separately
    versioned source-precedence policy.
    """

    records_by_id: dict[str, list[MatchRecordIdentity]] = defaultdict(list)
    for record in records:
        records_by_id[record.match.match_id].append(record)

    groups: list[MatchDuplicateGroup] = []
    for match_id in sorted(records_by_id):
        group_records = records_by_id[match_id]
        if len(group_records) < 2:
            continue

        ordered_records = tuple(
            sorted(
                group_records,
                key=lambda record: (
                    record.source_row_ref,
                    record.record_fingerprint,
                    record.match.participant_ids,
                ),
            )
        )
        keys = {record.match.key for record in ordered_records}
        participant_pairs = {record.match.participant_ids for record in ordered_records}
        fingerprints = {record.record_fingerprint for record in ordered_records}

        if len(keys) != 1 or len(participant_pairs) != 1:
            kind = MatchDuplicateKind.MATCH_KEY_COLLISION
            anomaly_code = IdentityAnomalyCode.MATCH_KEY_COLLISION
        elif len(fingerprints) == 1:
            kind = MatchDuplicateKind.EXACT_DUPLICATE
            anomaly_code = IdentityAnomalyCode.DUPLICATE_MATCH
        else:
            kind = MatchDuplicateKind.CONFLICTING_RECORDS
            anomaly_code = IdentityAnomalyCode.CONFLICTING_MATCH_RECORDS

        groups.append(
            MatchDuplicateGroup(
                match_id=match_id,
                kind=kind,
                anomaly_code=anomaly_code,
                records=ordered_records,
            )
        )

    return tuple(groups)
