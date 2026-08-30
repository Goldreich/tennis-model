"""Atomic, content-addressed, non-overwriting prediction-lock storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from tennis_model.locking._json import canonical_json_bytes
from tennis_model.locking.card import render_locked_match_card
from tennis_model.locking.models import PredictionSnapshot, StoredPredictionLock


class LockStoreError(RuntimeError):
    pass


class LockAlreadyExistsError(LockStoreError):
    pass


class LockReservationConflict(LockStoreError):
    pass


class LockPublicationInterrupted(LockStoreError):
    pass


class LockIntegrityError(LockStoreError):
    pass


@dataclass(frozen=True, slots=True)
class IncompletePublication:
    base_lock_id: str
    revision: int
    reservation_path: Path
    temporary_paths: tuple[Path, ...]
    age_seconds: float


def _fsync_file(path: Path) -> None:
    # Windows requires a writable file descriptor for ``os.fsync``.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_retained_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise LockIntegrityError(f"retained artifact is missing: {path}")
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


class LockStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def revision_directory(self, base_lock_id: str, revision: int) -> Path:
        return self.root / base_lock_id / f"L{revision}"

    @staticmethod
    def verify_retained_artifacts(lock: PredictionSnapshot) -> None:
        for artifact in lock.retained_artifacts:
            observed = _hash_retained_path(Path(artifact.path))
            if observed != artifact.sha256:
                raise LockIntegrityError(
                    f"retained artifact failed hash verification: {artifact.artifact_id}"
                )

    def _reservation_path(self, base_lock_id: str, revision: int) -> Path:
        return self.root / base_lock_id / ".reservations" / f"L{revision}.reserve"

    def _reserve(self, lock: PredictionSnapshot, token: str) -> Path:
        target = self.revision_directory(lock.base_lock_id, lock.revision)
        if target.exists():
            raise LockAlreadyExistsError(f"lock revision already exists: {lock.lock_id}")
        reservation = self._reservation_path(lock.base_lock_id, lock.revision)
        reservation.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(
            {
                "schema_version": "lock-revision-reservation/v1",
                "base_lock_id": lock.base_lock_id,
                "revision": lock.revision,
                "content_sha256": lock.content_sha256,
                "token": token,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
            }
        )
        try:
            descriptor = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LockReservationConflict(
                f"lock revision is already reserved: {lock.lock_id}"
            ) from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(reservation.parent)
        return reservation

    @staticmethod
    def _verify_directory(directory: Path) -> StoredPredictionLock:
        try:
            manifest_bytes = (directory / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest.get("schema_version") not in {
                "prediction-lock-files/v1",
                "prediction-lock-files/v2",
            }:
                raise LockIntegrityError("unsupported lock-file manifest schema")
            if canonical_json_bytes(manifest) != manifest_bytes:
                raise LockIntegrityError("lock-file manifest is not canonical JSON")
            for filename, expected in manifest["files"].items():
                observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                if observed != expected:
                    raise LockIntegrityError(f"lock artifact was modified: {filename}")
            envelope = StoredPredictionLock.model_validate_json(
                (directory / "lock.json").read_bytes()
            )
        except LockIntegrityError:
            raise
        except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise LockIntegrityError(f"cannot verify lock artifact: {exc}") from exc
        if envelope.content_sha256 != manifest.get("content_sha256"):
            raise LockIntegrityError("file manifest and lock content hashes differ")
        if (
            render_locked_match_card(envelope.lock).encode("utf-8")
            != (directory / "card.md").read_bytes()
        ):
            raise LockIntegrityError("stored card is not a pure rendering of lock content")
        return envelope

    def write(
        self,
        lock: PredictionSnapshot,
        *,
        interrupt_at: Literal["before_manifest", "after_payload", "after_verification"]
        | None = None,
    ) -> Path:
        """Reserve, verify, fsync, and atomically publish one immutable revision."""

        if lock.schema_version in {"prediction-lock/v3", "prediction-lock/v4"}:
            self.verify_retained_artifacts(lock)
        target = self.revision_directory(lock.base_lock_id, lock.revision)
        token = uuid4().hex
        reservation = self._reserve(lock, token)
        parent = target.parent
        temporary = Path(tempfile.mkdtemp(prefix=f".partial-L{lock.revision}-{token}-", dir=parent))
        envelope = StoredPredictionLock(content_sha256=lock.content_sha256, lock=lock)
        lock_bytes = canonical_json_bytes(envelope.model_dump(mode="json"))
        card_bytes = render_locked_match_card(lock).encode("utf-8")
        manifest = {
            "schema_version": "prediction-lock-files/v2",
            "content_sha256": lock.content_sha256,
            "files": {
                "lock.json": hashlib.sha256(lock_bytes).hexdigest(),
                "card.md": hashlib.sha256(card_bytes).hexdigest(),
            },
        }
        try:
            (temporary / "lock.json").write_bytes(lock_bytes)
            (temporary / "card.md").write_bytes(card_bytes)
            _fsync_file(temporary / "lock.json")
            _fsync_file(temporary / "card.md")
            if interrupt_at == "before_manifest":
                raise LockPublicationInterrupted("publication interrupted before manifest write")
            (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            _fsync_file(temporary / "manifest.json")
            _fsync_directory(temporary)
            if interrupt_at == "after_payload":
                raise LockPublicationInterrupted("publication interrupted before verification")
            verified = self._verify_directory(temporary)
            if verified.content_sha256 != lock.content_sha256:
                raise LockIntegrityError("temporary lock verification returned a different hash")
            if interrupt_at == "after_verification":
                raise LockPublicationInterrupted("publication interrupted before atomic publish")
            os.rename(temporary, target)
            _fsync_directory(parent)
            reservation.unlink()
            _fsync_directory(reservation.parent)
        except (LockPublicationInterrupted, LockIntegrityError):
            raise
        except FileExistsError as exc:
            raise LockAlreadyExistsError(f"lock revision already exists: {lock.lock_id}") from exc
        except OSError as exc:
            raise LockStoreError(f"cannot publish lock {lock.lock_id}: {exc}") from exc
        self.load(lock.base_lock_id, lock.revision)
        return target

    def incomplete_publications(
        self,
        *,
        now_utc: datetime | None = None,
    ) -> tuple[IncompletePublication, ...]:
        now = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
        records: list[IncompletePublication] = []
        if not self.root.exists():
            return ()
        for reservation in sorted(self.root.glob("*/.reservations/L*.reserve")):
            base = reservation.parent.parent.name
            revision = int(reservation.stem.removeprefix("L"))
            parent = reservation.parent.parent
            temporary = tuple(sorted(parent.glob(f".partial-L{revision}-*")))
            modified = datetime.fromtimestamp(reservation.stat().st_mtime, tz=UTC)
            records.append(
                IncompletePublication(
                    base_lock_id=base,
                    revision=revision,
                    reservation_path=reservation,
                    temporary_paths=temporary,
                    age_seconds=max(0.0, (now - modified).total_seconds()),
                )
            )
        return tuple(records)

    def quarantine_stale_publication(
        self,
        base_lock_id: str,
        revision: int,
        *,
        minimum_age: timedelta,
        now_utc: datetime | None = None,
    ) -> tuple[Path, ...]:
        matching = next(
            (
                item
                for item in self.incomplete_publications(now_utc=now_utc)
                if item.base_lock_id == base_lock_id and item.revision == revision
            ),
            None,
        )
        if matching is None:
            raise LockStoreError("no incomplete publication exists for that revision")
        if matching.age_seconds < minimum_age.total_seconds():
            raise LockReservationConflict("reservation is not old enough to classify as stale")
        token = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        moved: list[Path] = []
        for path in (*matching.temporary_paths, matching.reservation_path):
            destination = path.with_name(f".failed-{token}-{path.name.lstrip('.')}")
            path.rename(destination)
            moved.append(destination)
        return tuple(moved)

    def load(self, base_lock_id: str, revision: int) -> StoredPredictionLock:
        directory = self.revision_directory(base_lock_id, revision)
        envelope = self._verify_directory(directory)
        if envelope.lock.schema_version in {"prediction-lock/v3", "prediction-lock/v4"}:
            self.verify_retained_artifacts(envelope.lock)
        return envelope

    def verify(self, base_lock_id: str, revision: int) -> str:
        return self.load(base_lock_id, revision).content_sha256


__all__ = [
    "IncompletePublication",
    "LockAlreadyExistsError",
    "LockIntegrityError",
    "LockPublicationInterrupted",
    "LockReservationConflict",
    "LockStore",
    "LockStoreError",
]
