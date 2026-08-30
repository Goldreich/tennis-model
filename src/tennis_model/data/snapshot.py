"""Local, content-addressed storage for immutable raw-source snapshots."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from tennis_model.data.source_manifest import canonical_source_bytes
from tennis_model.schemas import PinnedSource, RawSourceSnapshot

_PAYLOAD_FILENAME = "payload"
_PROVENANCE_FILENAME = "source.json"
_COPY_CHUNK_BYTES = 1024 * 1024


class SnapshotError(RuntimeError):
    """Base class for snapshot creation and verification failures."""


class SnapshotChecksumMismatch(SnapshotError):
    """Input bytes do not match the checksum declared by the pinned source."""


class SnapshotIntegrityError(SnapshotError):
    """A materialized snapshot is incomplete, inconsistent, or tampered with."""


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise SnapshotError(f"cannot read source payload {path}: {exc}") from exc
    return digest.hexdigest(), size


def _snapshot_directory(source: PinnedSource, raw_root: Path) -> Path:
    return raw_root / source.tour.value.lower() / source.source_id / source.sha256


def _snapshot_record(source: PinnedSource, directory: Path, size_bytes: int) -> RawSourceSnapshot:
    return RawSourceSnapshot(
        source=source,
        payload_path=directory / _PAYLOAD_FILENAME,
        provenance_path=directory / _PROVENANCE_FILENAME,
        size_bytes=size_bytes,
        sha256=source.sha256,
    )


def _read_persisted_source(path: Path) -> PinnedSource:
    if path.is_symlink() or not path.is_file():
        raise SnapshotIntegrityError(
            f"snapshot provenance is missing or not a regular file: {path}"
        )
    try:
        return PinnedSource.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SnapshotIntegrityError(f"invalid snapshot provenance at {path}: {exc}") from exc


def _verify_provenance(snapshot: RawSourceSnapshot) -> None:
    persisted = _read_persisted_source(snapshot.provenance_path)
    if persisted != snapshot.source:
        raise SnapshotIntegrityError(
            f"snapshot provenance does not match pinned source {snapshot.source.source_id}"
        )


def open_verified_snapshot(snapshot: RawSourceSnapshot) -> BinaryIO:
    """Open a snapshot only after verifying its metadata, size, and SHA-256.

    The returned binary handle is rewound to byte zero and must be closed by the
    caller. Verification and later reads use the same open handle.
    """

    _verify_provenance(snapshot)
    payload = snapshot.payload_path
    if payload.is_symlink() or not payload.is_file():
        raise SnapshotIntegrityError(
            f"snapshot payload is missing or not a regular file: {payload}"
        )

    try:
        handle = payload.open("rb")
    except OSError as exc:
        raise SnapshotIntegrityError(f"cannot open snapshot payload {payload}: {exc}") from exc

    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != snapshot.sha256:
            raise SnapshotIntegrityError(
                f"snapshot checksum mismatch for {payload}: "
                f"expected {snapshot.sha256}, got {actual_sha256}"
            )
        if size != snapshot.size_bytes:
            raise SnapshotIntegrityError(
                f"snapshot size mismatch for {payload}: expected {snapshot.size_bytes}, got {size}"
            )
        handle.seek(0)
        return handle
    except Exception:
        handle.close()
        raise


def verify_snapshot(snapshot: RawSourceSnapshot) -> None:
    """Verify persisted provenance and payload without returning its bytes."""

    with open_verified_snapshot(snapshot):
        pass


def read_snapshot_bytes(snapshot: RawSourceSnapshot) -> bytes:
    """Read a small snapshot through the mandatory integrity check."""

    with open_verified_snapshot(snapshot) as handle:
        return handle.read()


def verified_snapshot_path(snapshot: RawSourceSnapshot) -> Path:
    """Return the local payload path only after an integrity check."""

    verify_snapshot(snapshot)
    return snapshot.payload_path


def _load_existing_snapshot(source: PinnedSource, directory: Path) -> RawSourceSnapshot:
    if directory.is_symlink() or not directory.is_dir():
        raise SnapshotIntegrityError(
            f"content-addressed snapshot path is not a regular directory: {directory}"
        )
    payload = directory / _PAYLOAD_FILENAME
    try:
        size_bytes = payload.stat().st_size
    except OSError as exc:
        raise SnapshotIntegrityError(f"incomplete snapshot at {directory}: {exc}") from exc
    snapshot = _snapshot_record(source, directory, size_bytes)
    verify_snapshot(snapshot)
    return snapshot


def _copy_payload(source_path: Path, destination: Path) -> None:
    try:
        with source_path.open("rb") as source_handle, destination.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=_COPY_CHUNK_BYTES)
            target_handle.flush()
            os.fsync(target_handle.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot stage snapshot payload from {source_path}: {exc}") from exc


def _write_provenance(source: PinnedSource, destination: Path) -> None:
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_source_bytes(source))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot stage snapshot provenance at {destination}: {exc}") from exc


def materialize_snapshot(
    source: PinnedSource,
    source_path: str | Path,
    raw_root: str | Path,
) -> RawSourceSnapshot:
    """Create or reuse a verified, content-addressed local source snapshot.

    ``source_path`` must already be local. The source manifest may record an
    HTTPS locator, but this function deliberately has no downloader and performs
    no network I/O. The completed directory becomes visible through one rename.
    Existing content is verified and never overwritten or repaired silently.
    """

    input_path = Path(source_path)
    if input_path.is_symlink() or not input_path.is_file():
        raise SnapshotError(f"source payload is missing or not a regular file: {input_path}")

    actual_sha256, input_size = _digest_file(input_path)
    if actual_sha256 != source.sha256:
        raise SnapshotChecksumMismatch(
            f"source checksum mismatch for {input_path}: "
            f"expected {source.sha256}, got {actual_sha256}"
        )

    root = Path(raw_root).resolve()
    target = _snapshot_directory(source, root)
    if target.exists() or target.is_symlink():
        return _load_existing_snapshot(source, target)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.sha256}.partial-", dir=parent))
    try:
        staged_payload = temporary / _PAYLOAD_FILENAME
        _copy_payload(input_path, staged_payload)
        staged_sha256, staged_size = _digest_file(staged_payload)
        if staged_sha256 != source.sha256 or staged_size != input_size:
            raise SnapshotIntegrityError(
                f"staged snapshot changed while copying {input_path}: "
                f"expected {source.sha256}/{input_size}, got {staged_sha256}/{staged_size}"
            )
        _write_provenance(source, temporary / _PROVENANCE_FILENAME)

        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            # Another writer may have won the race. It must be byte-for-byte
            # and provenance-identical; otherwise verification fails loudly.
            return _load_existing_snapshot(source, target)

        snapshot = _snapshot_record(source, target, staged_size)
        verify_snapshot(snapshot)
        return snapshot
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
