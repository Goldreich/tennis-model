"""Immutable, content-addressed Parquet bundles for Milestone 1 outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Self

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, field_validator, model_validator

from tennis_model.data.component_counts import TRANSFORMATION_VERSION
from tennis_model.data.exact_date_crosswalk import ExactDateCrosswalkManifest
from tennis_model.data.historical_validation import (
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
    assert_historical_training_safe,
)
from tennis_model.data.ingest_sackmann import HistoricalIngestionResult
from tennis_model.data.snapshot import verify_snapshot
from tennis_model.schemas import FrozenModel, PinnedSource

PROCESSED_ARTIFACT_SCHEMA_VERSION = "historical-ingestion-bundle/v2"
_MANIFEST_FILENAME = "manifest.json"
_TABLE_FILENAMES = {
    "service_rows": "service_rows.parquet",
    "component_counts": "component_counts.parquet",
    "anomalies": "anomalies.parquet",
    "cutoff_exclusions": "cutoff_exclusions.parquet",
}
_JSON_SEQUENCE_COLUMNS = frozenset({"invalid_stat_fields", "anomaly_codes"})


class ProcessedArtifactError(RuntimeError):
    """Base failure for processed artifact publication or verification."""


class ProcessedArtifactIntegrityError(ProcessedArtifactError):
    """A processed bundle is incomplete, inconsistent, or was modified."""


class TableReceipt(FrozenModel):
    """Checksum and logical schema for one persisted Parquet table."""

    name: str
    filename: str
    sha256: str
    size_bytes: Annotated[int, Field(ge=0)]
    row_count: Annotated[int, Field(ge=0)]
    columns: tuple[str, ...]
    json_encoded_columns: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def known_name(cls, value: str) -> str:
        if value not in _TABLE_FILENAMES:
            raise ValueError(f"unknown processed table name: {value}")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        return normalized

    @model_validator(mode="after")
    def filename_and_encoded_columns_are_safe(self) -> Self:
        if self.filename != _TABLE_FILENAMES[self.name]:
            raise ValueError(f"unexpected filename for table {self.name}")
        unknown = set(self.json_encoded_columns).difference(self.columns)
        if unknown:
            raise ValueError(
                "JSON-encoded columns are absent from table schema: " + ", ".join(sorted(unknown))
            )
        return self


class ProcessedArtifactManifest(FrozenModel):
    """Deterministic provenance receipt for one cutoff-safe data bundle."""

    artifact_schema_version: str
    snapshot_sha256: str
    snapshot_size_bytes: Annotated[int, Field(ge=0)]
    source: PinnedSource
    information_cutoff_utc: datetime
    normalization_version: str
    component_transformation_version: str
    historical_validation_policy: HistoricalValidationPolicy = HistoricalValidationPolicy()
    exact_date_crosswalk_manifest: ExactDateCrosswalkManifest | None = None
    raw_row_count: Annotated[int, Field(ge=0)]
    selected_raw_row_count: Annotated[int, Field(ge=0)]
    accepted_match_count: Annotated[int, Field(ge=0)]
    tables: tuple[TableReceipt, ...]

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("information_cutoff_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def provenance_is_complete(self) -> Self:
        if self.artifact_schema_version not in {
            "historical-ingestion-bundle/v1",
            PROCESSED_ARTIFACT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported processed artifact schema version")
        if self.snapshot_sha256 != self.source.sha256:
            raise ValueError("snapshot checksum does not match source pin")
        if self.component_transformation_version != TRANSFORMATION_VERSION:
            raise ValueError("unsupported component-count transformation version")
        retrospective = (
            self.historical_validation_policy.mode
            is HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED
        )
        if retrospective != (self.exact_date_crosswalk_manifest is not None):
            raise ValueError("retrospective bundles must contain their exact-date manifest")
        if (
            self.artifact_schema_version == "historical-ingestion-bundle/v1"
            and self.historical_validation_policy.mode
            is not HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE
        ):
            raise ValueError("v1 processed bundles can only represent strict vintage data")
        if (
            self.exact_date_crosswalk_manifest is not None
            and self.exact_date_crosswalk_manifest.crosswalk_id
            not in self.historical_validation_policy.exact_date_member_crosswalk_ids
        ):
            raise ValueError("processed policy and crosswalk manifest differ")
        names = tuple(receipt.name for receipt in self.tables)
        if names != tuple(_TABLE_FILENAMES):
            raise ValueError("processed manifest must contain every table in order")
        return self

    def receipt_for(self, name: str) -> TableReceipt:
        for receipt in self.tables:
            if receipt.name == name:
                return receipt
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class ProcessedArtifactBundle:
    """A verified location and parsed immutable processed manifest."""

    directory: Path
    bundle_id: str
    manifest: ProcessedArtifactManifest

    @property
    def manifest_path(self) -> Path:
        return self.directory / _MANIFEST_FILENAME

    def table_path(self, name: str) -> Path:
        return self.directory / self.manifest.receipt_for(name).filename


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _manifest_bytes(manifest: ProcessedArtifactManifest) -> bytes:
    value = manifest.model_dump(mode="json")
    if manifest.artifact_schema_version == "historical-ingestion-bundle/v1":
        value.pop("historical_validation_policy")
        value.pop("exact_date_crosswalk_manifest")
    return _canonical_json_bytes(value)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ProcessedArtifactIntegrityError(f"cannot read artifact {path}: {exc}") from exc
    return digest.hexdigest(), size


def _json_sequence(value: Any) -> str:
    if value is None or value is pd.NA:
        items: list[Any] = []
    elif isinstance(value, str):
        items = [value]
    else:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            items = []
        else:
            try:
                items = list(value)
            except TypeError:
                items = [value]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _parquet_ready(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if frame.columns.has_duplicates:
        raise ProcessedArtifactError("processed table contains duplicate columns")
    ready = frame.copy(deep=True)
    encoded: list[str] = []
    for column in ready.columns:
        if column in _JSON_SEQUENCE_COLUMNS:
            ready[column] = ready[column].map(_json_sequence).astype("string")
            encoded.append(column)
    return ready, tuple(encoded)


def _write_table(name: str, frame: pd.DataFrame, directory: Path) -> TableReceipt:
    ready, encoded = _parquet_ready(frame)
    filename = _TABLE_FILENAMES[name]
    path = directory / filename
    try:
        ready.to_parquet(
            path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
    except Exception as exc:
        raise ProcessedArtifactError(f"cannot write processed table {name}: {exc}") from exc
    digest, size = _file_digest(path)
    return TableReceipt(
        name=name,
        filename=filename,
        sha256=digest,
        size_bytes=size,
        row_count=len(ready),
        columns=tuple(str(column) for column in ready.columns),
        json_encoded_columns=encoded,
    )


def _table_frames(result: HistoricalIngestionResult) -> dict[str, pd.DataFrame]:
    return {
        "service_rows": result.service_rows,
        "component_counts": result.counts,
        "anomalies": result.anomalies.copy(deep=True),
        "cutoff_exclusions": result.cutoff_exclusions.copy(deep=True),
    }


def write_processed_bundle(
    result: HistoricalIngestionResult, processed_root: str | Path
) -> ProcessedArtifactBundle:
    """Atomically publish or idempotently reuse one processed Parquet bundle."""

    verify_snapshot(result.snapshot)
    assert_historical_training_safe(
        result.normalized.rows,
        result.cutoff,
        policy=result.historical_validation_policy,
    )
    assert_historical_training_safe(
        result.component_counts.counts,
        result.cutoff,
        policy=result.historical_validation_policy,
    )

    root = Path(processed_root).resolve()
    parent = (
        root
        / result.snapshot.source.tour.value.lower()
        / result.snapshot.source.source_id
        / result.snapshot.sha256[:16]
    )
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        frames = _table_frames(result)
        receipts = tuple(_write_table(name, frames[name], staging) for name in _TABLE_FILENAMES)
        manifest = ProcessedArtifactManifest(
            artifact_schema_version=PROCESSED_ARTIFACT_SCHEMA_VERSION,
            snapshot_sha256=result.snapshot.sha256,
            snapshot_size_bytes=result.snapshot.size_bytes,
            source=result.snapshot.source,
            information_cutoff_utc=result.cutoff.at_utc,
            normalization_version=result.normalized.normalization_version,
            component_transformation_version=TRANSFORMATION_VERSION,
            historical_validation_policy=result.historical_validation_policy,
            exact_date_crosswalk_manifest=result.exact_date_crosswalk_manifest,
            raw_row_count=result.raw_row_count,
            selected_raw_row_count=result.selected_raw_row_count,
            accepted_match_count=result.normalized.accepted_match_count,
            tables=receipts,
        )
        manifest_bytes = _manifest_bytes(manifest)
        bundle_id = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = staging / _MANIFEST_FILENAME
        try:
            with manifest_path.open("xb") as handle:
                handle.write(manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ProcessedArtifactError(f"cannot write processed manifest: {exc}") from exc

        # Short digest directory names avoid legacy Windows MAX_PATH failures;
        # the full digests remain in, and are verified from, the manifest.
        target = parent / bundle_id[:32]
        try:
            staging.rename(target)
        except OSError as exc:
            if not target.exists():
                raise
            existing = load_processed_bundle(target)
            if existing.manifest != manifest:
                raise ProcessedArtifactIntegrityError(
                    f"existing bundle {bundle_id} has conflicting provenance"
                ) from exc
            return existing
        return load_processed_bundle(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_processed_bundle(directory: str | Path) -> ProcessedArtifactBundle:
    """Load and fully verify a content-addressed processed bundle."""

    bundle_directory = Path(directory)
    if bundle_directory.is_symlink() or not bundle_directory.is_dir():
        raise ProcessedArtifactIntegrityError(
            f"processed bundle is missing or not a regular directory: {bundle_directory}"
        )
    manifest_path = bundle_directory / _MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ProcessedArtifactIntegrityError(
            f"processed manifest is missing or not a regular file: {manifest_path}"
        )
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = ProcessedArtifactManifest.model_validate_json(raw_manifest)
    except Exception as exc:
        raise ProcessedArtifactIntegrityError(
            f"invalid processed manifest {manifest_path}: {exc}"
        ) from exc
    canonical = _manifest_bytes(manifest)
    if raw_manifest != canonical:
        raise ProcessedArtifactIntegrityError("processed manifest is not canonical")
    bundle_id = hashlib.sha256(canonical).hexdigest()
    if bundle_directory.name != bundle_id[:32]:
        raise ProcessedArtifactIntegrityError(
            f"processed bundle directory does not match manifest hash prefix {bundle_id[:32]}"
        )
    bundle = ProcessedArtifactBundle(
        directory=bundle_directory,
        bundle_id=bundle_id,
        manifest=manifest,
    )
    verify_processed_bundle(bundle)
    return bundle


def verify_processed_bundle(bundle: ProcessedArtifactBundle) -> None:
    """Rehash every table and validate its Parquet metadata."""

    for receipt in bundle.manifest.tables:
        path = bundle.directory / receipt.filename
        if path.is_symlink() or not path.is_file():
            raise ProcessedArtifactIntegrityError(
                f"processed table is missing or not a regular file: {path}"
            )
        digest, size = _file_digest(path)
        if digest != receipt.sha256 or size != receipt.size_bytes:
            raise ProcessedArtifactIntegrityError(
                f"processed table checksum/size mismatch: {receipt.name}"
            )
        try:
            metadata = pq.read_metadata(path)
        except Exception as exc:
            raise ProcessedArtifactIntegrityError(
                f"cannot read Parquet metadata for {receipt.name}: {exc}"
            ) from exc
        if metadata.num_rows != receipt.row_count:
            raise ProcessedArtifactIntegrityError(
                f"processed table row-count mismatch: {receipt.name}"
            )
        schema_columns = tuple(metadata.schema.names)
        if schema_columns != receipt.columns:
            raise ProcessedArtifactIntegrityError(
                f"processed table schema mismatch: {receipt.name}"
            )


def read_processed_table(bundle: ProcessedArtifactBundle, name: str) -> pd.DataFrame:
    """Verify the bundle and restore deterministic sequence-valued columns."""

    verify_processed_bundle(bundle)
    receipt = bundle.manifest.receipt_for(name)
    try:
        frame: pd.DataFrame = pd.read_parquet(
            bundle.table_path(name),
            engine="pyarrow",
            to_pandas_kwargs={},
        )
    except TypeError:
        frame = pd.read_parquet(  # type: ignore[call-overload]
            bundle.table_path(name),
            engine="pyarrow",
        )
    for column in receipt.json_encoded_columns:
        frame[column] = frame[column].map(lambda value: tuple(json.loads(value)))
    return frame


__all__ = [
    "PROCESSED_ARTIFACT_SCHEMA_VERSION",
    "ProcessedArtifactBundle",
    "ProcessedArtifactError",
    "ProcessedArtifactIntegrityError",
    "ProcessedArtifactManifest",
    "TableReceipt",
    "load_processed_bundle",
    "read_processed_table",
    "verify_processed_bundle",
    "write_processed_bundle",
]
