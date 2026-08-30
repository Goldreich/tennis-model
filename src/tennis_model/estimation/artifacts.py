"""Append-only, content-addressed fitted-component artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tennis_model.estimation.serve_components import FittedServeComponent

_FIT_FILENAME = "fit.json"
_DIAGNOSTICS_FILENAME = "diagnostics.txt"


class FitArtifactError(RuntimeError):
    """A fit artifact cannot be published or verified."""


class FitArtifactIntegrityError(FitArtifactError):
    """A persisted fit differs from its content-addressed identity."""


@dataclass(frozen=True, slots=True)
class FitArtifact:
    directory: Path
    artifact_id: str
    fit: FittedServeComponent

    @property
    def fit_path(self) -> Path:
        return self.directory / _FIT_FILENAME

    @property
    def diagnostics_path(self) -> Path:
        return self.directory / _DIAGNOSTICS_FILENAME


def _canonical_json_bytes(fit: FittedServeComponent) -> bytes:
    value = fit.model_dump(mode="json")
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


def _diagnostics_bytes(fit: FittedServeComponent) -> bytes:
    diagnostics = fit.diagnostics
    scales = ", ".join(
        f"{name}={value:.6g}" for name, value in sorted(diagnostics.shrinkage_scales.items())
    )
    warnings = ", ".join(diagnostics.warnings) if diagnostics.warnings else "none"
    lines = (
        f"framework: {fit.framework_version}",
        f"tour/component: {fit.tour.value}/{fit.component.value}",
        f"cutoff: {fit.data_cutoff_utc.isoformat()}",
        f"usable rows: {diagnostics.usable_rows}",
        f"raw/weighted trials: {diagnostics.raw_trials}/{diagnostics.weighted_trials:.6f}",
        f"servers/returners: {diagnostics.unique_servers}/{diagnostics.unique_returners}",
        f"excluded/missing/quarantined: {diagnostics.excluded_rows}/"
        f"{diagnostics.missing_rows}/{diagnostics.quarantined_rows}",
        f"kappa: {diagnostics.kappa:.8g}",
        f"shrinkage scales: {scales}",
        f"observed/fitted rate: {diagnostics.observed_rate:.8f}/{diagnostics.fitted_rate:.8f}",
        f"optimizer: converged={diagnostics.converged}, status={diagnostics.optimizer_status}, "
        f"iterations={diagnostics.iterations}, gradient_max_abs="
        f"{diagnostics.gradient_max_abs:.6g}",
        f"curvature: {fit.posterior.curvature_kind}, condition="
        f"{fit.posterior.condition_number:.6g}, regularization="
        f"{fit.posterior.regularization_added:.6g}",
        f"warnings: {warnings}",
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FitArtifactError(f"cannot write fit artifact file {path}: {exc}") from exc


def write_fit_artifact(
    fit: FittedServeComponent,
    artifact_root: str | Path,
) -> FitArtifact:
    """Atomically publish a fit without overwriting any prior artifact."""

    try:
        fit = FittedServeComponent.model_validate(fit.model_dump(mode="python"))
    except ValidationError as exc:
        raise FitArtifactError(f"refusing to persist an invalid fitted component: {exc}") from exc
    fit_bytes = _canonical_json_bytes(fit)
    artifact_id = hashlib.sha256(fit_bytes).hexdigest()
    cutoff_segment = fit.data_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    parent = (
        Path(artifact_root).resolve()
        / fit.tour.value.lower()
        / fit.component.value.lower()
        / cutoff_segment
    )
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact_id[:32]
    if target.exists():
        existing = load_fit_artifact(target)
        if existing.artifact_id != artifact_id or existing.fit != fit:
            raise FitArtifactIntegrityError("existing artifact path contains a conflicting fit")
        return existing

    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        _write_new(staging / _FIT_FILENAME, fit_bytes)
        _write_new(staging / _DIAGNOSTICS_FILENAME, _diagnostics_bytes(fit))
        try:
            staging.rename(target)
        except OSError as exc:
            if not target.exists():
                raise FitArtifactError(f"cannot publish fit artifact {target}: {exc}") from exc
            existing = load_fit_artifact(target)
            if existing.artifact_id != artifact_id or existing.fit != fit:
                raise FitArtifactIntegrityError(
                    "concurrent artifact publication produced conflicting content"
                ) from exc
            return existing
        return load_fit_artifact(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_fit_artifact(directory: str | Path) -> FitArtifact:
    """Load a fitted artifact only after canonical-content verification."""

    artifact_directory = Path(directory)
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise FitArtifactIntegrityError(f"fit artifact is not a regular directory: {directory}")
    fit_path = artifact_directory / _FIT_FILENAME
    diagnostics_path = artifact_directory / _DIAGNOSTICS_FILENAME
    if fit_path.is_symlink() or not fit_path.is_file():
        raise FitArtifactIntegrityError(f"fit JSON is missing: {fit_path}")
    if diagnostics_path.is_symlink() or not diagnostics_path.is_file():
        raise FitArtifactIntegrityError(f"fit diagnostics are missing: {diagnostics_path}")
    try:
        raw_fit = fit_path.read_bytes()
        fit = FittedServeComponent.model_validate_json(raw_fit)
    except Exception as exc:
        raise FitArtifactIntegrityError(f"fit JSON is invalid: {exc}") from exc
    canonical = _canonical_json_bytes(fit)
    if raw_fit != canonical:
        raise FitArtifactIntegrityError("fit JSON is not canonical")
    artifact_id = hashlib.sha256(canonical).hexdigest()
    if artifact_directory.name != artifact_id[:32]:
        raise FitArtifactIntegrityError("artifact directory does not match fit content hash")
    try:
        observed_diagnostics = diagnostics_path.read_bytes()
    except OSError as exc:
        raise FitArtifactIntegrityError(f"cannot read diagnostics: {exc}") from exc
    if observed_diagnostics != _diagnostics_bytes(fit):
        raise FitArtifactIntegrityError("human-readable diagnostics differ from fitted metadata")
    return FitArtifact(directory=artifact_directory, artifact_id=artifact_id, fit=fit)


__all__ = [
    "FitArtifact",
    "FitArtifactError",
    "FitArtifactIntegrityError",
    "load_fit_artifact",
    "write_fit_artifact",
]
