from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_model.data.snapshot import (
    SnapshotChecksumMismatch,
    SnapshotIntegrityError,
    materialize_snapshot,
    read_snapshot_bytes,
    verified_snapshot_path,
    verify_snapshot,
)
from tennis_model.data.source_manifest import (
    dump_source_manifest,
    load_source_manifest,
    manifest_sha256,
)
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RowDateSemantics,
    SourceManifest,
    Tour,
    TourCoverage,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(
    tour: Tour,
    payload: bytes,
    *,
    source_id: str | None = None,
    coverage: CoverageRange | None = None,
) -> PinnedSource:
    slug = tour.value.lower()
    return PinnedSource(
        source_id=source_id or f"sackmann-{slug}-2025",
        identity_namespace="jeff-sackmann",
        tour=tour,
        upstream_attribution="Jeff Sackmann tennis_atp/tennis_wta project",
        locator=f"https://archive.example/{slug}/matches-2025.csv",
        archive_identifier=f"archive:{slug}:2025",
        object_identifier=f"commit:{'a' * 40}",
        sha256=_digest(payload),
        schema_version="sackmann-match-csv/v1",
        stated_license="CC BY-NC-SA 4.0",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        verified_coverage=coverage
        or CoverageRange(
            first_match_date=date(2025, 1, 1),
            last_match_date=date(2025, 12, 31),
            verified_at_utc=datetime(2026, 8, 28, 13, tzinfo=UTC),
        ),
        row_date_semantics=RowDateSemantics.TOURNAMENT_START_DATE,
        availability_lag_days=21,
    )


def _range(first: date, last: date) -> CoverageRange:
    return CoverageRange(
        first_match_date=first,
        last_match_date=last,
        verified_at_utc=datetime(2026, 8, 28, 13, tzinfo=UTC),
    )


def _manifest(atp_payload: bytes = b"atp", wta_payload: bytes = b"wta") -> SourceManifest:
    atp_coverage = _range(date(2022, 1, 1), date(2025, 12, 31))
    wta_coverage = _range(date(2023, 1, 2), date(2025, 12, 30))
    return SourceManifest(
        manifest_version="tennis-source-manifest/v1",
        sources=(
            _source(Tour.ATP, atp_payload, coverage=atp_coverage),
            _source(Tour.WTA, wta_payload, coverage=wta_coverage),
        ),
        coverage_by_tour=TourCoverage(
            atp=atp_coverage,
            wta=wta_coverage,
        ),
    )


def test_manifest_preserves_separate_atp_and_wta_coverage() -> None:
    manifest = _manifest()

    assert manifest.coverage_by_tour.for_tour(Tour.ATP) == _range(
        date(2022, 1, 1), date(2025, 12, 31)
    )
    assert manifest.coverage_by_tour.for_tour(Tour.WTA) == _range(
        date(2023, 1, 2), date(2025, 12, 30)
    )
    assert len(manifest.sources_for_tour(Tour.ATP)) == 1
    assert len(manifest.sources_for_tour(Tour.WTA)) == 1


def test_manifest_requires_coverage_for_exactly_its_source_tours() -> None:
    atp = _source(Tour.ATP, b"atp")

    with pytest.raises(ValidationError, match="exactly match"):
        SourceManifest(
            manifest_version="tennis-source-manifest/v1",
            sources=(atp,),
            coverage_by_tour=TourCoverage(
                atp=_range(date(2025, 1, 1), date(2025, 12, 31)),
                wta=_range(date(2025, 1, 1), date(2025, 12, 31)),
            ),
        )


def test_pinned_source_requires_explicit_cutoff_semantics_and_exact_provenance() -> None:
    source = _source(Tour.ATP, b"payload")

    assert source.row_date_semantics is RowDateSemantics.TOURNAMENT_START_DATE
    assert source.availability_lag_days == 21
    assert source.identity_namespace == "jeff-sackmann"
    assert source.object_identifier.startswith("commit:")
    assert source.retrieved_at_utc.tzinfo is UTC
    assert source.verified_coverage.first_match_date == date(2025, 1, 1)
    with pytest.raises(ValidationError):
        source.availability_lag_days = 2

    value = source.model_dump(mode="python")
    del value["row_date_semantics"]
    with pytest.raises(ValidationError, match="row_date_semantics"):
        PinnedSource.model_validate(value)

    value = source.model_dump(mode="python")
    del value["upstream_attribution"]
    with pytest.raises(ValidationError, match="upstream_attribution"):
        PinnedSource.model_validate(value)


def test_identity_namespace_is_stable_across_yearly_source_objects() -> None:
    source_2024 = _source(Tour.ATP, b"2024", source_id="sackmann-atp-2024")
    source_2025 = _source(Tour.ATP, b"2025", source_id="sackmann-atp-2025")

    assert source_2024.source_id != source_2025.source_id
    assert source_2024.identity_namespace == source_2025.identity_namespace


def test_archive_and_object_identifiers_are_optional_when_locator_is_exact() -> None:
    value = _source(Tour.ATP, b"payload").model_dump(mode="python")
    value["archive_identifier"] = None
    value["object_identifier"] = None

    source = PinnedSource.model_validate(value)

    assert source.locator == "https://archive.example/atp/matches-2025.csv"
    assert source.archive_identifier is None
    assert source.object_identifier is None


def test_manifest_rejects_aggregate_coverage_not_backed_by_source_objects() -> None:
    source = _source(Tour.ATP, b"atp")

    with pytest.raises(ValidationError, match="must equal the bounds"):
        SourceManifest(
            manifest_version="tennis-source-manifest/v1",
            sources=(source,),
            coverage_by_tour=TourCoverage(
                atp=_range(date(2024, 1, 1), date(2025, 12, 31)),
                wta=None,
            ),
        )


@pytest.mark.parametrize("lag", [0, -1, -100])
def test_date_only_availability_lag_must_be_positive(lag: int) -> None:
    value = _source(Tour.ATP, b"payload").model_dump(mode="python")
    value["availability_lag_days"] = lag

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PinnedSource.model_validate(value)


def test_all_provenance_datetimes_must_be_timezone_aware() -> None:
    source_value = _source(Tour.ATP, b"payload").model_dump(mode="python")
    source_value["retrieved_at_utc"] = datetime(2026, 8, 28, 12)
    with pytest.raises(ValidationError, match="timezone-aware"):
        PinnedSource.model_validate(source_value)

    with pytest.raises(ValidationError, match="timezone-aware"):
        CoverageRange(
            first_match_date=date(2025, 1, 1),
            last_match_date=date(2025, 12, 31),
            verified_at_utc=datetime(2026, 8, 28, 12),
        )


def test_utc_timestamps_are_normalized_and_coverage_is_ordered() -> None:
    source = _source(Tour.ATP, b"payload").model_copy(
        update={
            "retrieved_at_utc": datetime(
                2026,
                8,
                28,
                7,
                tzinfo=timezone(timedelta(hours=-5)),
            )
        }
    )
    # model_copy is intentionally not a validation boundary; round-trip through
    # validation to assert persisted timestamps become canonical UTC.
    source = PinnedSource.model_validate(source.model_dump(mode="python"))
    assert source.retrieved_at_utc == datetime(2026, 8, 28, 12, tzinfo=UTC)

    with pytest.raises(ValidationError, match="must not be after"):
        _range(date(2025, 12, 31), date(2025, 1, 1))


def test_manifest_yaml_round_trip_and_fingerprint_are_deterministic(tmp_path: Path) -> None:
    manifest = _manifest()
    first_hash = manifest_sha256(manifest)
    manifest_path = tmp_path / "sources.yaml"
    manifest_path.write_text(dump_source_manifest(manifest), encoding="utf-8")

    loaded = load_source_manifest(manifest_path)

    assert loaded == manifest
    assert manifest_sha256(loaded) == first_hash
    assert len(first_hash) == 64


def test_malformed_checksum_is_rejected() -> None:
    value = _source(Tour.ATP, b"payload").model_dump(mode="python")
    value["sha256"] = "not-a-sha256"

    with pytest.raises(ValidationError, match="64 hexadecimal"):
        PinnedSource.model_validate(value)


def test_snapshot_creation_is_content_addressed_verified_and_idempotent(tmp_path: Path) -> None:
    payload = b"winner_name,loser_name,w_svpt,l_svpt\nA,B,50,48\n"
    source = _source(Tour.ATP, payload)
    incoming = tmp_path / "incoming.csv"
    incoming.write_bytes(payload)
    raw_root = tmp_path / "raw"

    first = materialize_snapshot(source, incoming, raw_root)
    first_payload_mtime = first.payload_path.stat().st_mtime_ns
    second = materialize_snapshot(source, incoming, raw_root)

    expected_directory = raw_root.resolve() / "atp" / source.source_id / source.sha256
    assert first == second
    assert first.payload_path == expected_directory / "payload"
    assert first.provenance_path == expected_directory / "source.json"
    assert first.payload_path.stat().st_mtime_ns == first_payload_mtime
    assert read_snapshot_bytes(first) == payload
    persisted_source = PinnedSource.model_validate_json(first.provenance_path.read_bytes())
    assert persisted_source.verified_coverage == source.verified_coverage
    assert verified_snapshot_path(first) == first.payload_path
    verify_snapshot(first)
    assert [path.name for path in expected_directory.parent.iterdir()] == [source.sha256]


def test_snapshot_refuses_input_with_wrong_declared_checksum(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming.csv"
    incoming.write_bytes(b"actual bytes")
    source_value = _source(Tour.ATP, b"different bytes").model_dump(mode="python")
    source = PinnedSource.model_validate(source_value)
    raw_root = tmp_path / "raw"

    with pytest.raises(SnapshotChecksumMismatch, match="source checksum mismatch"):
        materialize_snapshot(source, incoming, raw_root)

    assert not raw_root.exists()


def test_snapshot_payload_tampering_fails_loudly_and_is_not_repaired(tmp_path: Path) -> None:
    payload = b"original"
    source = _source(Tour.WTA, payload)
    incoming = tmp_path / "incoming.csv"
    incoming.write_bytes(payload)
    snapshot = materialize_snapshot(source, incoming, tmp_path / "raw")
    snapshot.payload_path.write_bytes(b"tampered")

    with pytest.raises(SnapshotIntegrityError, match="checksum mismatch"):
        verify_snapshot(snapshot)
    with pytest.raises(SnapshotIntegrityError, match="checksum mismatch"):
        materialize_snapshot(source, incoming, tmp_path / "raw")
    assert snapshot.payload_path.read_bytes() == b"tampered"


def test_snapshot_provenance_tampering_fails_loudly(tmp_path: Path) -> None:
    payload = b"original"
    source = _source(Tour.ATP, payload)
    incoming = tmp_path / "incoming.csv"
    incoming.write_bytes(payload)
    snapshot = materialize_snapshot(source, incoming, tmp_path / "raw")
    snapshot.provenance_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="invalid snapshot provenance"):
        read_snapshot_bytes(snapshot)
