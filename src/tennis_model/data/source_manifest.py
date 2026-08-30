"""Load and fingerprint strict, pinned historical-source manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tennis_model.schemas import PinnedSource, SourceManifest


class SourceManifestError(ValueError):
    """A source manifest is unreadable or violates its frozen schema."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def canonical_source_bytes(source: PinnedSource) -> bytes:
    """Return deterministic UTF-8 provenance bytes for a pinned source."""

    return _canonical_json_bytes(source.model_dump(mode="json"))


def canonical_manifest_bytes(manifest: SourceManifest) -> bytes:
    """Return deterministic UTF-8 bytes suitable for provenance hashing."""

    return _canonical_json_bytes(manifest.model_dump(mode="json"))


def manifest_sha256(manifest: SourceManifest) -> str:
    """Fingerprint all manifest fields, including coverage and cutoff semantics."""

    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def parse_source_manifest(value: Mapping[str, Any]) -> SourceManifest:
    """Validate an already-decoded manifest mapping."""

    try:
        return SourceManifest.model_validate(value)
    except ValidationError as exc:
        raise SourceManifestError(f"invalid source manifest: {exc}") from exc


def load_source_manifest(path: str | Path) -> SourceManifest:
    """Load YAML from a local path; this function never performs network I/O."""

    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceManifestError(f"cannot read source manifest {manifest_path}: {exc}") from exc

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SourceManifestError(
            f"invalid YAML in source manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(value, Mapping):
        raise SourceManifestError("source manifest root must be a mapping")
    return parse_source_manifest(value)


def dump_source_manifest(manifest: SourceManifest) -> str:
    """Serialize a validated manifest to human-readable YAML without writing it."""

    value = manifest.model_dump(mode="json")
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
