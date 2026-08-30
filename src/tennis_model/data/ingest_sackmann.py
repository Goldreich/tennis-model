"""Cutoff-safe ingestion for pinned Jeff Sackmann-style ATP/WTA CSV files."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from tennis_model.data.component_counts import (
    ComponentCountTable,
    build_serve_component_counts,
)
from tennis_model.data.cutoff import (
    CutoffSelection,
    InformationCutoff,
    availability_from_source_date,
)
from tennis_model.data.exact_date_crosswalk import (
    ExactDateCrosswalkManifest,
    ExactDateCrosswalkResult,
    apply_exact_match_dates,
)
from tennis_model.data.historical_validation import (
    POINT_IN_TIME_VINTAGE_POLICY,
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
    conservative_match_completion_utc,
    select_historical_training_rows,
)
from tennis_model.data.normalize import (
    NormalizedServiceRows,
    SackmannSchemaError,
    combine_anomaly_tables,
    normalize_player_service_rows,
    validate_sackmann_columns,
)
from tennis_model.data.snapshot import read_snapshot_bytes
from tennis_model.schemas import RawSourceSnapshot, Tour

SACKMANN_SCHEMA_VERSION = "sackmann-match-csv/v1"


class SackmannIngestionError(RuntimeError):
    """A verified snapshot cannot be parsed as the declared source schema."""


class CoverageValidationError(SackmannIngestionError):
    """Observed row dates contradict the pin's verified coverage claim."""


@dataclass(frozen=True, slots=True)
class RawCutoffSelection:
    """Raw source rows admitted before parsing probability-affecting fields."""

    rows: pd.DataFrame
    excluded: pd.DataFrame


@dataclass(frozen=True, slots=True)
class HistoricalIngestionResult:
    """A cutoff-safe Milestone 1 dataset and its audit tables."""

    snapshot: RawSourceSnapshot
    cutoff: InformationCutoff
    normalized: NormalizedServiceRows
    component_counts: ComponentCountTable
    anomalies: pd.DataFrame
    cutoff_exclusions: pd.DataFrame
    raw_row_count: int
    selected_raw_row_count: int
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY
    exact_date_crosswalk_manifest: ExactDateCrosswalkManifest | None = None

    @property
    def service_rows(self) -> pd.DataFrame:
        return self.normalized.rows.copy(deep=True)

    @property
    def counts(self) -> pd.DataFrame:
        return self.component_counts.counts.copy(deep=True)


def _decode_csv(payload: bytes) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SackmannIngestionError("snapshot payload is not valid UTF-8/UTF-8-BOM text") from exc


def _validate_header(text: str) -> None:
    try:
        header = next(csv.reader(io.StringIO(text, newline="")))
    except (StopIteration, csv.Error) as exc:
        raise SackmannIngestionError("snapshot has no readable CSV header") from exc
    if not header or any(not name.strip() for name in header):
        raise SackmannIngestionError("snapshot contains a blank CSV column name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise SackmannIngestionError(
            "snapshot contains duplicate CSV columns: " + ", ".join(duplicates)
        )


def read_sackmann_snapshot(snapshot: RawSourceSnapshot) -> pd.DataFrame:
    """Verify and parse exact snapshot bytes without pandas NA coercion."""

    if snapshot.source.schema_version != SACKMANN_SCHEMA_VERSION:
        raise SackmannIngestionError(
            "unsupported Sackmann schema version: "
            f"{snapshot.source.schema_version!r}; expected {SACKMANN_SCHEMA_VERSION!r}"
        )
    payload = read_snapshot_bytes(snapshot)
    text = _decode_csv(payload)
    _validate_header(text)
    try:
        raw = pd.read_csv(
            io.StringIO(text, newline=""),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            on_bad_lines="error",
        )
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise SackmannIngestionError(f"cannot parse Sackmann CSV: {exc}") from exc
    try:
        validate_sackmann_columns(raw)
    except SackmannSchemaError as exc:
        raise SackmannIngestionError(str(exc)) from exc

    raw = raw.copy(deep=True)
    raw.insert(0, "_source_row_number", range(2, len(raw) + 2))
    original_columns = [column for column in raw.columns if not column.startswith("_")]
    raw.insert(
        1,
        "_raw_record_json",
        [
            json.dumps(
                {column: str(row[column]) for column in original_columns},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for _, row in raw.iterrows()
        ],
    )
    return raw


def _parse_source_date(value: Any) -> datetime | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip()
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _validate_declared_coverage(raw: pd.DataFrame, snapshot: RawSourceSnapshot) -> None:
    coverage = snapshot.source.verified_coverage
    observed = [
        parsed.date()
        for value in raw["tourney_date"]
        if (parsed := _parse_source_date(value)) is not None
    ]
    outside = [
        value
        for value in observed
        if value < coverage.first_match_date or value > coverage.last_match_date
    ]
    if outside:
        raise CoverageValidationError(
            f"{len(outside)} source rows fall outside declared "
            f"{snapshot.source.tour.value} coverage "
            f"[{coverage.first_match_date}, {coverage.last_match_date}]"
        )


def select_raw_before_cutoff(
    raw: pd.DataFrame,
    *,
    snapshot: RawSourceSnapshot,
    cutoff: InformationCutoff,
) -> RawCutoffSelection:
    """Fail closed before any future row's statistics are normalized."""

    selected_indices: list[Any] = []
    excluded_records: list[dict[str, Any]] = []
    for index, row in raw.iterrows():
        parsed = _parse_source_date(row.get("tourney_date"))
        if parsed is None:
            available_at = None
            code = "UNKNOWN_AVAILABILITY"
        else:
            derived_available_at = availability_from_source_date(
                parsed.date(),
                availability_lag_days=snapshot.source.availability_lag_days,
            )
            available_at = max(
                derived_available_at,
                snapshot.source.source_available_at_utc or derived_available_at,
            )
            code = "AT_OR_AFTER_INFORMATION_CUTOFF" if available_at >= cutoff.at_utc else ""
        if not code:
            selected_indices.append(index)
            continue
        excluded_records.append(
            {
                "snapshot_id": snapshot.source.sha256,
                "snapshot_sha256": snapshot.sha256,
                "source_id": snapshot.source.source_id,
                "source_row_number": int(row["_source_row_number"]),
                "available_at_utc": available_at,
                "information_cutoff_utc": cutoff.at_utc,
                "cutoff_exclusion_code": code,
            }
        )
    selected = raw.loc[selected_indices].copy(deep=True)
    excluded = pd.DataFrame.from_records(
        excluded_records,
        columns=(
            "snapshot_id",
            "snapshot_sha256",
            "source_id",
            "source_row_number",
            "available_at_utc",
            "information_cutoff_utc",
            "cutoff_exclusion_code",
        ),
    )
    return RawCutoffSelection(rows=selected, excluded=excluded)


def select_finalized_raw_before_cutoff(
    raw: pd.DataFrame,
    *,
    snapshot: RawSourceSnapshot,
    cutoff: InformationCutoff,
) -> RawCutoffSelection:
    """Select exact-dated finalized training records without parsing their statistics."""

    if "_exact_match_date" not in raw:
        raise SackmannIngestionError("retrospective-finalized intake requires an exact crosswalk")
    selected_indices: list[Any] = []
    excluded_records: list[dict[str, Any]] = []
    for index, row in raw.iterrows():
        parsed = _parse_source_date(row.get("_exact_match_date"))
        status = str(row.get("_exact_date_join_status", ""))
        if parsed is None or status != "MATCHED":
            completion = None
            code = "MISSING_EXACT_MATCH_DATE"
        else:
            completion = conservative_match_completion_utc(parsed.date())
            code = "AT_OR_AFTER_FORECAST_CUTOFF" if completion > cutoff.at_utc else ""
        if not code:
            selected_indices.append(index)
            continue
        excluded_records.append(
            {
                "snapshot_id": snapshot.source.sha256,
                "snapshot_sha256": snapshot.sha256,
                "source_id": snapshot.source.source_id,
                "source_row_number": int(row["_source_row_number"]),
                "available_at_utc": completion,
                "information_cutoff_utc": cutoff.at_utc,
                "cutoff_exclusion_code": code,
            }
        )
    columns = (
        "snapshot_id",
        "snapshot_sha256",
        "source_id",
        "source_row_number",
        "available_at_utc",
        "information_cutoff_utc",
        "cutoff_exclusion_code",
    )
    return RawCutoffSelection(
        rows=raw.loc[selected_indices].copy(deep=True),
        excluded=pd.DataFrame.from_records(excluded_records, columns=columns),
    )


def _cutoff_anomalies(excluded: pd.DataFrame) -> pd.DataFrame:
    if excluded.empty:
        return pd.DataFrame()
    result = excluded.rename(columns={"cutoff_exclusion_code": "code"}).copy()
    result["scope"] = "cutoff"
    result["severity"] = "info"
    return result


def ingest_sackmann_snapshot(
    snapshot: RawSourceSnapshot,
    *,
    cutoff: InformationCutoff,
    tour: Tour | str | None = None,
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
    exact_date_crosswalk: ExactDateCrosswalkResult | None = None,
) -> HistoricalIngestionResult:
    """Run the complete Milestone 1 transform under a mandatory cutoff."""

    if tour is not None:
        requested_tour = tour if isinstance(tour, Tour) else Tour(str(tour).upper())
        if requested_tour is not snapshot.source.tour:
            raise SackmannIngestionError(
                f"requested tour {requested_tour.value} does not match pinned "
                f"source tour {snapshot.source.tour.value}"
            )

    raw = read_sackmann_snapshot(snapshot)
    _validate_declared_coverage(raw, snapshot)
    if historical_validation_policy.mode is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE:
        if exact_date_crosswalk is not None:
            raise SackmannIngestionError("strict vintage intake cannot use a finalized crosswalk")
        raw_selection = select_raw_before_cutoff(raw, snapshot=snapshot, cutoff=cutoff)
    else:
        if exact_date_crosswalk is None:
            raise SackmannIngestionError(
                "retrospective-finalized intake requires an exact-date crosswalk"
            )
        if exact_date_crosswalk.manifest.crosswalk_id not in (
            historical_validation_policy.exact_date_member_crosswalk_ids
        ):
            raise SackmannIngestionError("historical policy does not pin the supplied crosswalk")
        if exact_date_crosswalk.manifest.sackmann_source_sha256 != snapshot.sha256:
            raise SackmannIngestionError("crosswalk was built for a different Sackmann snapshot")
        if (
            historical_validation_policy.exact_date_history_complete
            != exact_date_crosswalk.manifest.complete_for_b6_c6_history
        ):
            raise SackmannIngestionError("historical policy misstates exact-date coverage")
        raw = apply_exact_match_dates(raw, exact_date_crosswalk)
        raw_selection = select_finalized_raw_before_cutoff(
            raw, snapshot=snapshot, cutoff=cutoff
        )
    normalized = normalize_player_service_rows(raw_selection.rows, snapshot=snapshot)

    # Defense in depth: public historical output must satisfy the same strict
    # guard even though raw rows were already filtered before normalization.
    selected_normalized: CutoffSelection = select_historical_training_rows(
        normalized.rows,
        cutoff,
        policy=historical_validation_policy,
    )
    if not selected_normalized.excluded.empty:
        raise SackmannIngestionError(
            "normalized cutoff guard rejected rows admitted by the raw guard"
        )
    cutoff_normalized = NormalizedServiceRows(
        rows=selected_normalized.rows,
        anomalies=normalized.anomalies,
        raw_row_count=normalized.raw_row_count,
        accepted_match_count=normalized.accepted_match_count,
        normalization_version=normalized.normalization_version,
    )
    counts = build_serve_component_counts(cutoff_normalized.rows)
    from tennis_model.data.historical_validation import assert_historical_training_safe

    assert_historical_training_safe(
        cutoff_normalized.rows, cutoff, policy=historical_validation_policy
    )
    assert_historical_training_safe(counts.counts, cutoff, policy=historical_validation_policy)
    anomalies = combine_anomaly_tables(
        cutoff_normalized.anomalies,
        counts.anomalies,
        _cutoff_anomalies(raw_selection.excluded),
    )
    return HistoricalIngestionResult(
        snapshot=snapshot,
        cutoff=cutoff,
        normalized=cutoff_normalized,
        component_counts=counts,
        anomalies=anomalies,
        cutoff_exclusions=raw_selection.excluded,
        raw_row_count=len(raw),
        selected_raw_row_count=len(raw_selection.rows),
        historical_validation_policy=historical_validation_policy,
        exact_date_crosswalk_manifest=(
            None if exact_date_crosswalk is None else exact_date_crosswalk.manifest
        ),
    )


__all__ = [
    "SACKMANN_SCHEMA_VERSION",
    "CoverageValidationError",
    "HistoricalIngestionResult",
    "RawCutoffSelection",
    "SackmannIngestionError",
    "ingest_sackmann_snapshot",
    "read_sackmann_snapshot",
    "select_finalized_raw_before_cutoff",
    "select_raw_before_cutoff",
]
