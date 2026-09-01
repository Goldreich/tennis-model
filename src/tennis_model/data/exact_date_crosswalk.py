"""Deterministic Sackmann-to-Tennis-Data exact match-date augmentation.

Only identity, date, event structure, and score fields are admitted from the
augmentation source.  Bookmaker/market columns are never copied into the
crosswalk or returned to model code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from collections.abc import Mapping
from typing import Any, Self

import pandas as pd
from pydantic import Field, field_validator, model_validator

from tennis_model.schemas import FrozenModel, Tour

EXACT_DATE_CROSSWALK_SCHEMA_VERSION = "exact-match-date-crosswalk/v1"
EXACT_DATE_MATCHING_ALGORITHM_VERSION = "sackmann-tennis-data-exact-date/v1"
ALIASED_EXACT_DATE_CROSSWALK_SCHEMA_VERSION = "exact-match-date-crosswalk/v2"
ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION = "sackmann-tennis-data-exact-date/v2"


class ExactDateJoinStatus(StrEnum):
    MATCHED = "MATCHED"
    UNMATCHED = "UNMATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    REUSED_AUGMENTATION_ROW = "REUSED_AUGMENTATION_ROW"
    DUPLICATE_SOURCE_MATCH = "DUPLICATE_SOURCE_MATCH"
    STRUCTURAL_CONFLICT = "STRUCTURAL_CONFLICT"


class ExactDateSourcePin(FrozenModel):
    """Immutable receipt for one exact-date augmentation object."""

    source_id: str
    tour: Tour
    year: int = Field(ge=1900, le=2200)
    locator: str
    sha256: str
    size_bytes: int = Field(ge=0)
    retrieved_at_utc: datetime
    source_last_modified_at_utc: datetime | None = None
    schema_version: str = "tennis-data-results-workbook/v1"

    @field_validator("sha256")
    @classmethod
    def digest_is_valid(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("source SHA-256 must be lowercase hexadecimal")
        return normalized

    @field_validator("retrieved_at_utc", "source_last_modified_at_utc")
    @classmethod
    def time_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source timestamps must be timezone-aware")
        return value.astimezone(UTC)


class ExactDateCrosswalkManifest(FrozenModel):
    """Content receipt and conservative sufficiency decision for one crosswalk."""

    schema_version: str = EXACT_DATE_CROSSWALK_SCHEMA_VERSION
    algorithm_version: str = EXACT_DATE_MATCHING_ALGORITHM_VERSION
    crosswalk_id: str
    sackmann_source_id: str
    sackmann_source_sha256: str
    augmentation_source: ExactDateSourcePin
    source_rows: int = Field(ge=0)
    augmentation_rows: int = Field(ge=0)
    matched_rows: int = Field(ge=0)
    residual_rows: int = Field(ge=0)
    status_counts: tuple[tuple[ExactDateJoinStatus, int], ...]
    detail_sha256: str
    complete_for_b6_c6_history: bool
    sackmann_name_key_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @field_validator("sackmann_source_sha256", "detail_sha256")
    @classmethod
    def hashes_are_valid(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("crosswalk hashes must be lowercase SHA-256")
        return normalized

    @model_validator(mode="after")
    def identity_and_counts_are_coherent(self) -> Self:
        supported = {
            EXACT_DATE_CROSSWALK_SCHEMA_VERSION: EXACT_DATE_MATCHING_ALGORITHM_VERSION,
            ALIASED_EXACT_DATE_CROSSWALK_SCHEMA_VERSION: (
                ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION
            ),
        }
        if self.schema_version not in supported:
            raise ValueError("unsupported crosswalk schema version")
        if self.algorithm_version != supported[self.schema_version]:
            raise ValueError("unsupported exact-date matching algorithm")
        if (
            self.schema_version == EXACT_DATE_CROSSWALK_SCHEMA_VERSION
            and self.sackmann_name_key_aliases
        ):
            raise ValueError("v1 crosswalks cannot declare name-key aliases")
        if self.matched_rows + self.residual_rows != self.source_rows:
            raise ValueError("crosswalk row counts do not reconcile")
        if sum(count for _, count in self.status_counts) != self.source_rows:
            raise ValueError("crosswalk status counts do not reconcile")
        if self.complete_for_b6_c6_history != (self.residual_rows == 0):
            raise ValueError("B6/C6 completeness must require zero residual rows")
        excluded = {"crosswalk_id"}
        if self.schema_version == EXACT_DATE_CROSSWALK_SCHEMA_VERSION:
            excluded.add("sackmann_name_key_aliases")
        payload = self.model_dump(mode="json", exclude=excluded)
        if self.crosswalk_id != _sha256_json(payload):
            raise ValueError("crosswalk ID does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class ExactDateCrosswalkResult:
    detail: pd.DataFrame
    manifest: ExactDateCrosswalkManifest

    @property
    def coverage(self) -> float | None:
        if self.manifest.source_rows == 0:
            return None
        return self.manifest.matched_rows / self.manifest.source_rows


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tokens(value: Any) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_value.casefold())


def _sackmann_name_keys(value: Any) -> set[str]:
    parts = _tokens(value)
    if len(parts) < 2:
        return set()
    return {
        f"{''.join(parts[-width:])}|{parts[0][0]}"
        for width in range(1, min(4, len(parts) - 1) + 1)
    }


def _tennis_data_name_key(value: Any) -> str | None:
    raw = str(value).strip()
    if " " not in raw:
        return None
    surname, initials = raw.rsplit(maxsplit=1)
    surname_parts, initial_parts = _tokens(surname), _tokens(initials)
    if not surname_parts or not initial_parts:
        return None
    return f"{''.join(surname_parts)}|{initial_parts[0][0]}"


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _surface(value: Any) -> str | None:
    if _missing(value):
        return None
    parts = _tokens(value)
    return parts[0] if parts else None


def _td_round(value: Any) -> str:
    text = " ".join(_tokens(value))
    aliases = {
        "the final": "F",
        "final": "F",
        "semifinals": "SF",
        "semi finals": "SF",
        "quarterfinals": "QF",
        "quarter finals": "QF",
        "4th round": "R4",
        "fourth round": "R4",
        "3rd round": "R3",
        "third round": "R3",
        "2nd round": "R2",
        "second round": "R2",
        "1st round": "R1",
        "first round": "R1",
        "round robin": "RR",
    }
    return aliases.get(text, text.upper())


def _expected_round(round_value: Any, draw_size: Any) -> str | None:
    code = str(round_value).upper()
    if code in {"F", "SF", "QF", "RR"}:
        return code
    remaining = {"R128": 128, "R64": 64, "R32": 32, "R16": 16}.get(code)
    if remaining is None or _missing(draw_size):
        return None
    try:
        draw = int(draw_size)
    except (TypeError, ValueError):
        return None
    container = 1 << math.ceil(math.log2(max(draw, 2)))
    if remaining > container:
        return None
    return f"R{int(math.log2(container) - math.log2(remaining) + 1)}"


def _sackmann_score(value: Any) -> tuple[tuple[int, int], ...] | None:
    if _missing(value):
        return None
    sets = [
        (int(match.group(1)), int(match.group(2)))
        for token in str(value).split()
        if (match := re.match(r"^(\d+)-(\d+)", token))
    ]
    return tuple(sets) or None


def _td_score(row: pd.Series) -> tuple[tuple[int, int], ...] | None:
    sets: list[tuple[int, int]] = []
    for number in range(1, 6):
        winner, loser = row.get(f"W{number}"), row.get(f"L{number}")
        if pd.isna(winner) or pd.isna(loser):
            continue
        sets.append((int(winner), int(loser)))
    return tuple(sets) or None


def _compatible(left: Any, right: Any) -> bool:
    if _missing(left) or _missing(right):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _detail_digest(detail: pd.DataFrame) -> str:
    records = detail.where(pd.notna(detail), None).to_dict(orient="records")
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def build_exact_date_crosswalk(
    sackmann: pd.DataFrame,
    tennis_data: pd.DataFrame,
    *,
    sackmann_source_id: str,
    sackmann_source_sha256: str,
    augmentation_source: ExactDateSourcePin,
    sackmann_name_key_aliases: Mapping[str, tuple[str, ...]] | None = None,
    algorithm_version: str = EXACT_DATE_MATCHING_ALGORITHM_VERSION,
) -> ExactDateCrosswalkResult:
    """Build a one-to-one, non-fuzzy exact-date crosswalk for one tour-year."""

    if algorithm_version not in {
        EXACT_DATE_MATCHING_ALGORITHM_VERSION,
        ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION,
    }:
        raise ValueError("unsupported requested exact-date matching algorithm")
    alias_items = tuple(
        sorted(
            (
                str(name),
                tuple(sorted(set(str(key) for key in keys))),
            )
            for name, keys in (sackmann_name_key_aliases or {}).items()
        )
    )
    if algorithm_version == EXACT_DATE_MATCHING_ALGORITHM_VERSION and alias_items:
        raise ValueError("name-key aliases require exact-date matching algorithm v2")
    aliases = dict(alias_items)

    required_sackmann = {
        "winner_name",
        "loser_name",
        "tourney_date",
        "tourney_id",
        "round",
        "surface",
        "best_of",
        "score",
    }
    required_td = {"Date", "Winner", "Loser", "Surface", "Round", "Best of"}
    missing_sackmann = required_sackmann.difference(sackmann.columns)
    missing_td = required_td.difference(tennis_data.columns)
    if missing_sackmann or missing_td:
        raise ValueError(
            "crosswalk source columns missing: "
            + ", ".join(sorted(missing_sackmann | missing_td))
        )
    td = tennis_data.loc[:, [
        column
        for column in (
            "Date", "Tournament", "Winner", "Loser", "Surface", "Round", "Best of",
            "W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5",
        )
        if column in tennis_data.columns
    ]].copy()
    td["_date"] = pd.to_datetime(td["Date"], errors="coerce").dt.date
    td["_winner_key"] = td["Winner"].map(_tennis_data_name_key)
    td["_loser_key"] = td["Loser"].map(_tennis_data_name_key)
    td["_surface"] = td["Surface"].map(_surface)
    td["_round"] = td["Round"].map(_td_round)
    td["_score"] = pd.Series(
        [_td_score(row) for _, row in td.iterrows()], index=td.index, dtype="object"
    )
    pair_index: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for index, row in td.iterrows():
        if row["_winner_key"] is not None and row["_loser_key"] is not None:
            pair_index[(row["_winner_key"], row["_loser_key"])].append(index)

    records: list[dict[str, Any]] = []
    for ordinal, (source_index, row) in enumerate(sackmann.iterrows(), start=2):
        source_row_number = int(row.get("_source_row_number", ordinal))
        winner_keys = _sackmann_name_keys(row["winner_name"])
        loser_keys = _sackmann_name_keys(row["loser_name"])
        winner_keys.update(aliases.get(str(row["winner_name"]), ()))
        loser_keys.update(aliases.get(str(row["loser_name"]), ()))
        candidate_ids: set[Any] = set()
        for winner_key in winner_keys:
            for loser_key in loser_keys:
                candidate_ids.update(pair_index.get((winner_key, loser_key), ()))
        parsed_start = pd.to_datetime(str(row["tourney_date"]), format="%Y%m%d", errors="coerce")
        start: date | None = None if pd.isna(parsed_start) else parsed_start.date()
        dated = []
        if start is not None:
            for candidate_id in sorted(candidate_ids, key=str):
                raw_candidate_date = td.at[candidate_id, "_date"]
                candidate_date = (
                    raw_candidate_date.date()
                    if isinstance(raw_candidate_date, datetime)
                    else raw_candidate_date
                    if isinstance(raw_candidate_date, date)
                    else None
                )
                if (
                    candidate_date is not None
                    and -3 <= (candidate_date - start).days <= 21
                ):
                    dated.append(candidate_id)
        expected_round = _expected_round(row.get("round"), row.get("draw_size"))
        surface = _surface(row.get("surface"))
        score = _sackmann_score(row.get("score"))

        def structurally_matches(
            candidate_id: Any,
            *,
            strict: bool,
            source_surface: str | None = surface,
            source_row: pd.Series = row,
            source_round: str | None = expected_round,
            source_score: tuple[tuple[int, int], ...] | None = score,
        ) -> bool:
            candidate = td.loc[candidate_id]
            base = (
                source_surface is None
                or candidate["_surface"] is None
                or source_surface == candidate["_surface"]
            ) and _compatible(candidate.get("Best of"), source_row.get("best_of"))
            return bool(
                base
                and (not strict or source_round is None or candidate["_round"] == source_round)
                and (
                    not strict
                    or source_score is None
                    or candidate["_score"] is None
                    or candidate["_score"] == source_score
                )
            )

        strict_candidates = [item for item in dated if structurally_matches(item, strict=True)]
        base_candidates = [item for item in dated if structurally_matches(item, strict=False)]
        selected: Any | None = None
        if len(strict_candidates) == 1:
            status = ExactDateJoinStatus.MATCHED
            selected = strict_candidates[0]
        elif len(strict_candidates) > 1 or len(base_candidates) > 1:
            status = ExactDateJoinStatus.AMBIGUOUS
        elif len(base_candidates) == 1:
            status = ExactDateJoinStatus.STRUCTURAL_CONFLICT
        else:
            status = ExactDateJoinStatus.UNMATCHED
        selected_row = None if selected is None else td.loc[selected]
        records.append(
            {
                "source_row_number": source_row_number,
                "source_frame_index": str(source_index),
                "tourney_id": row.get("tourney_id"),
                "tourney_name": row.get("tourney_name"),
                "tourney_level": row.get("tourney_level"),
                "tourney_date": None if start is None else start.isoformat(),
                "match_num": row.get("match_num"),
                "round": row.get("round"),
                "winner_name": row.get("winner_name"),
                "loser_name": row.get("loser_name"),
                "status": status.value,
                "candidate_count": len(dated),
                "strict_candidate_count": len(strict_candidates),
                "augmentation_row_number": None if selected is None else int(selected) + 2,
                "exact_match_date": (
                    None if selected_row is None else selected_row["_date"].isoformat()
                ),
                "augmentation_tournament": (
                    None if selected_row is None else selected_row.get("Tournament")
                ),
                "match_date_source_id": augmentation_source.source_id,
                "match_date_source_sha256": augmentation_source.sha256,
            }
        )
    detail = pd.DataFrame.from_records(records)
    duplicate_key = ["tourney_id", "winner_name", "loser_name", "round"]
    duplicates = detail.duplicated(duplicate_key, keep=False)
    detail.loc[duplicates, "status"] = ExactDateJoinStatus.DUPLICATE_SOURCE_MATCH.value
    matched = detail["status"].eq(ExactDateJoinStatus.MATCHED.value)
    reused = detail.loc[matched, "augmentation_row_number"].duplicated(keep=False)
    detail.loc[detail.loc[matched].index[reused], "status"] = (
        ExactDateJoinStatus.REUSED_AUGMENTATION_ROW.value
    )
    detail.loc[~detail["status"].eq(ExactDateJoinStatus.MATCHED.value), "exact_match_date"] = None
    counts = {
        status: int(detail["status"].eq(status.value).sum()) for status in ExactDateJoinStatus
    }
    matched_rows = counts[ExactDateJoinStatus.MATCHED]
    schema_version = (
        ALIASED_EXACT_DATE_CROSSWALK_SCHEMA_VERSION
        if algorithm_version == ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION
        else EXACT_DATE_CROSSWALK_SCHEMA_VERSION
    )
    manifest_payload = {
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "sackmann_source_id": sackmann_source_id,
        "sackmann_source_sha256": sackmann_source_sha256,
        "augmentation_source": augmentation_source.model_dump(mode="json"),
        "source_rows": len(detail),
        "augmentation_rows": len(td),
        "matched_rows": matched_rows,
        "residual_rows": len(detail) - matched_rows,
        "status_counts": tuple((status.value, counts[status]) for status in ExactDateJoinStatus),
        "detail_sha256": _detail_digest(detail),
        "complete_for_b6_c6_history": matched_rows == len(detail),
    }
    if schema_version == ALIASED_EXACT_DATE_CROSSWALK_SCHEMA_VERSION:
        manifest_payload["sackmann_name_key_aliases"] = alias_items
    manifest = ExactDateCrosswalkManifest.model_validate(
        {"crosswalk_id": _sha256_json(manifest_payload), **manifest_payload}
    )
    return ExactDateCrosswalkResult(detail=detail, manifest=manifest)


def apply_exact_match_dates(
    raw: pd.DataFrame, crosswalk: ExactDateCrosswalkResult
) -> pd.DataFrame:
    """Attach only uniquely matched dates and immutable provenance to raw rows."""

    if crosswalk.manifest.source_rows != len(raw):
        raise ValueError("crosswalk does not cover the supplied raw table cardinality")
    mapping = crosswalk.detail.set_index("source_row_number")
    if not mapping.index.is_unique:
        raise ValueError("crosswalk contains duplicate source row numbers")
    enriched = raw.copy(deep=True)
    row_numbers = enriched.get(
        "_source_row_number", pd.Series(range(2, len(enriched) + 2), index=enriched.index)
    )
    missing = set(int(value) for value in row_numbers).difference(mapping.index)
    if missing:
        raise ValueError("crosswalk is missing source row numbers")
    enriched["_exact_match_date"] = [
        mapping.at[int(number), "exact_match_date"] for number in row_numbers
    ]
    enriched["_exact_date_join_status"] = [
        mapping.at[int(number), "status"] for number in row_numbers
    ]
    enriched["_match_date_source_id"] = crosswalk.manifest.augmentation_source.source_id
    enriched["_match_date_source_sha256"] = crosswalk.manifest.augmentation_source.sha256
    enriched["_match_date_crosswalk_id"] = crosswalk.manifest.crosswalk_id
    return enriched


__all__ = [
    "ALIASED_EXACT_DATE_CROSSWALK_SCHEMA_VERSION",
    "ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION",
    "EXACT_DATE_CROSSWALK_SCHEMA_VERSION",
    "EXACT_DATE_MATCHING_ALGORITHM_VERSION",
    "ExactDateCrosswalkManifest",
    "ExactDateCrosswalkResult",
    "ExactDateJoinStatus",
    "ExactDateSourcePin",
    "apply_exact_match_dates",
    "build_exact_date_crosswalk",
]
