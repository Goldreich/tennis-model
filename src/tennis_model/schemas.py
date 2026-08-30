"""Frozen, provenance-oriented schemas shared by the data layer."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SourceId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FrozenModel(BaseModel):
    """Base class for immutable, strict provenance records."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Tour(StrEnum):
    """Tour-specific models and data must never share an implicit namespace."""

    ATP = "ATP"
    WTA = "WTA"


class RowDateSemantics(StrEnum):
    """Meaning of a source row's date before availability-lag handling."""

    MATCH_DATE = "match_date"
    TOURNAMENT_START_DATE = "tournament_start_date"


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class CoverageRange(FrozenModel):
    """Verified inclusive match-date coverage for exactly one tour."""

    first_match_date: date
    last_match_date: date
    verified_at_utc: datetime

    @field_validator("verified_at_utc")
    @classmethod
    def normalize_verified_at(cls, value: datetime) -> datetime:
        return _as_utc(value, field_name="verified_at_utc")

    @model_validator(mode="after")
    def dates_are_ordered(self) -> Self:
        if self.first_match_date > self.last_match_date:
            raise ValueError("first_match_date must not be after last_match_date")
        return self


class TourCoverage(FrozenModel):
    """Coverage is represented independently for ATP and WTA."""

    atp: CoverageRange | None
    wta: CoverageRange | None

    @model_validator(mode="after")
    def at_least_one_tour_is_verified(self) -> Self:
        if self.atp is None and self.wta is None:
            raise ValueError("at least one tour must have verified coverage")
        return self

    def for_tour(self, tour: Tour) -> CoverageRange | None:
        """Return only the explicitly verified range for ``tour``."""

        return self.atp if tour is Tour.ATP else self.wta

    def verified_tours(self) -> frozenset[Tour]:
        """Return tours with non-null, independently verified coverage."""

        tours: set[Tour] = set()
        if self.atp is not None:
            tours.add(Tour.ATP)
        if self.wta is not None:
            tours.add(Tour.WTA)
        return frozenset(tours)


class PinnedSource(FrozenModel):
    """Exact identity and intake semantics for one immutable source object."""

    source_id: SourceId
    identity_namespace: SourceId
    tour: Tour
    upstream_attribution: NonEmptyStr
    locator: NonEmptyStr
    archive_identifier: NonEmptyStr | None = None
    object_identifier: NonEmptyStr | None = None
    sha256: str
    schema_version: NonEmptyStr
    stated_license: NonEmptyStr
    retrieved_at_utc: datetime
    verified_coverage: CoverageRange
    row_date_semantics: RowDateSemantics
    # Both supported semantics are date-only. A row cannot conservatively be
    # treated as known at the opening instant of its source date.
    availability_lag_days: Annotated[int, Field(ge=1)]
    source_effective_at_utc: datetime | None = None
    source_available_at_utc: datetime | None = None

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return normalized

    @field_validator("retrieved_at_utc", "source_effective_at_utc", "source_available_at_utc")
    @classmethod
    def normalize_source_times(
        cls, value: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if value is None:
            return None
        return _as_utc(value, field_name=info.field_name or "source timestamp")

    @model_validator(mode="after")
    def source_information_times_are_coherent(self) -> Self:
        if (self.source_effective_at_utc is None) != (self.source_available_at_utc is None):
            raise ValueError("source effective and available timestamps must be supplied together")
        if (
            self.source_effective_at_utc is not None
            and self.source_available_at_utc is not None
            and self.source_effective_at_utc > self.source_available_at_utc
        ):
            raise ValueError("source information cannot be available before it is effective")
        return self


class SourceManifest(FrozenModel):
    """Versioned source registry with independently verified tour coverage."""

    manifest_version: NonEmptyStr
    sources: tuple[PinnedSource, ...]
    coverage_by_tour: TourCoverage

    @model_validator(mode="after")
    def sources_are_unique_and_covered(self) -> Self:
        if not self.sources:
            raise ValueError("sources must not be empty")

        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")

        source_tours = {source.tour for source in self.sources}
        coverage_tours = set(self.coverage_by_tour.verified_tours())
        if source_tours != coverage_tours:
            raise ValueError(
                "verified coverage tours must exactly match the tours represented by sources"
            )

        for tour in source_tours:
            aggregate = self.coverage_by_tour.for_tour(tour)
            if aggregate is None:  # Defensive; equality above makes this unreachable.
                raise ValueError(f"missing aggregate verified coverage for {tour.value}")
            source_ranges = [
                source.verified_coverage for source in self.sources if source.tour is tour
            ]
            first = min(item.first_match_date for item in source_ranges)
            last = max(item.last_match_date for item in source_ranges)
            if first != aggregate.first_match_date or last != aggregate.last_match_date:
                raise ValueError(
                    f"aggregate {tour.value} coverage must equal the bounds of its source objects"
                )
        return self

    def sources_for_tour(self, tour: Tour) -> tuple[PinnedSource, ...]:
        """Return pinned objects for one tour without cross-tour fallback."""

        return tuple(source for source in self.sources if source.tour is tour)


class RawSourceSnapshot(FrozenModel):
    """Materialized content-addressed source and its persisted provenance."""

    source: PinnedSource
    payload_path: Path
    provenance_path: Path
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return normalized

    @model_validator(mode="after")
    def digest_matches_source(self) -> Self:
        if self.sha256 != self.source.sha256:
            raise ValueError("snapshot sha256 must match its pinned source")
        return self
