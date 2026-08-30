"""Immutable, cutoff-identified fitted-model bundles.

Snapshot v1 contains the five serve components, v2 adds the frozen B6/C6
contracts, and v3 adds the independently fitted B5 duration artifact.  Earlier
schemas remain loadable and retain their original content identities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import field_validator, model_validator

from tennis_model.estimation.artifacts import (
    FitArtifact,
    FitArtifactError,
    load_fit_artifact,
)
from tennis_model.estimation.duration_model import (
    DurationArtifactError,
    DurationFitArtifact,
    PersistedDurationFitArtifact,
    load_duration_fit_artifact,
)
from tennis_model.estimation.inactivity import InactivityConfigurationArtifact
from tennis_model.estimation.retirement import (
    PersistedRetirementFitArtifact,
    RetirementArtifactError,
    RetirementFitArtifact,
    load_retirement_fit_artifact,
)
from tennis_model.estimation.serve_components import (
    FittedServeComponent,
    ModelDataError,
    ServeComponent,
    validate_serve_fit_bundle,
)
from tennis_model.schemas import FrozenModel, Tour

_COMPONENT_ORDER = tuple(ServeComponent)


class ModelSnapshotError(RuntimeError):
    """A fitted-model snapshot is incomplete, inconsistent, or unreadable."""


def _sha256(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must contain exactly 64 hexadecimal characters")
    return normalized


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


class ComponentArtifactReference(FrozenModel):
    """Content identity and explicit location of one fitted component."""

    component: ServeComponent
    artifact_id: str
    directory: Path

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_sha256(cls, value: str) -> str:
        return _sha256(value, field="artifact_id")

    @field_validator("directory")
    @classmethod
    def directory_is_absolute(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("component artifact directory must be absolute")
        return path


class RetirementArtifactReference(FrozenModel):
    """Content identity and immutable location of the fitted B6 artifact."""

    artifact_id: str
    directory: Path
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_sha256(cls, value: str) -> str:
        return _sha256(value, field="retirement artifact_id")

    @field_validator("directory")
    @classmethod
    def directory_is_absolute(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("retirement artifact directory must be absolute")
        return path

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def fit_follows_cutoff(self) -> Self:
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("retirement fit time cannot precede its data cutoff")
        return self


class DurationArtifactReference(FrozenModel):
    """Content identity and immutable location of the fitted B5 artifact."""

    artifact_id: str
    directory: Path
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_sha256(cls, value: str) -> str:
        return _sha256(value, field="duration artifact_id")

    @field_validator("directory")
    @classmethod
    def directory_is_absolute(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("duration artifact directory must be absolute")
        return path

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def fit_follows_cutoff(self) -> Self:
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("duration fit time cannot precede its data cutoff")
        return self


class ModelSnapshot(FrozenModel):
    """Stable references to one coherent, immutable fitted-model bundle."""

    schema_version: Literal[
        "serve-model-snapshot/v1",
        "serve-model-snapshot/v2",
        "serve-model-snapshot/v3",
    ] = "serve-model-snapshot/v1"
    framework_version: Literal["v1.0"]
    implementation_version: Literal["serve-components-map-laplace/v1"]
    tour: Tour
    fitted_at_utc: datetime
    data_cutoff_utc: datetime
    component_artifacts: tuple[ComponentArtifactReference, ...]
    data_hash: str
    component_count_artifact_hash: str
    config_hash: str
    code_commit: str
    retirement_artifact: RetirementArtifactReference | None = None
    retirement_schema_version: str | None = None
    inactivity_configuration: InactivityConfigurationArtifact | None = None
    inactivity_schema_version: str | None = None
    duration_artifact: DurationArtifactReference | None = None
    duration_schema_version: str | None = None

    @field_validator("fitted_at_utc", "data_cutoff_utc")
    @classmethod
    def timestamps_are_aware(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("data_hash", "component_count_artifact_hash", "config_hash")
    @classmethod
    def hashes_are_sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, field=info.field_name)

    @field_validator("code_commit")
    @classmethod
    def code_commit_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code_commit must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def bundle_is_complete_and_ordered(self) -> Self:
        observed = tuple(reference.component for reference in self.component_artifacts)
        if observed != _COMPONENT_ORDER:
            raise ValueError("component artifacts must contain F/A/Q1/D/Q2 in canonical order")
        if self.fitted_at_utc < self.data_cutoff_utc:
            raise ValueError("snapshot fit time cannot precede its data cutoff")
        contract_fields = (
            self.retirement_artifact,
            self.retirement_schema_version,
            self.inactivity_configuration,
            self.inactivity_schema_version,
        )
        if self.schema_version == "serve-model-snapshot/v1":
            if any(item is not None for item in contract_fields) or any(
                item is not None
                for item in (self.duration_artifact, self.duration_schema_version)
            ):
                raise ValueError(
                    "pre-amendment snapshots cannot contain B6/C6 or duration references"
                )
        elif any(item is None for item in contract_fields):
            raise ValueError("v2/v3 snapshots require complete B6 and C6 references")
        elif (
            self.retirement_schema_version is None
            or self.inactivity_schema_version is None
            or self.inactivity_configuration is None
        ):
            raise AssertionError("validated v2 snapshot lost required probability contracts")
        elif self.inactivity_schema_version != self.inactivity_configuration.schema_version:
            raise ValueError("snapshot C6 schema differs from its configuration artifact")
        elif self.retirement_artifact is not None and (
            self.retirement_artifact.tour is not self.tour
            or self.retirement_artifact.information_cutoff_utc != self.data_cutoff_utc
        ):
            raise ValueError("snapshot B6 reference differs from its tour or data cutoff")
        if self.schema_version == "serve-model-snapshot/v2" and any(
            item is not None for item in (self.duration_artifact, self.duration_schema_version)
        ):
            raise ValueError("v2 snapshots cannot silently add a duration artifact")
        if self.schema_version == "serve-model-snapshot/v3":
            if self.duration_artifact is None or self.duration_schema_version is None:
                raise ValueError("v3 snapshots require the B5 duration artifact")
            if (
                self.duration_artifact.tour is not self.tour
                or self.duration_artifact.information_cutoff_utc != self.data_cutoff_utc
            ):
                raise ValueError("snapshot duration reference differs from its tour or cutoff")
        return self

    @property
    def b6_c6_complete(self) -> bool:
        return self.schema_version in {
            "serve-model-snapshot/v2",
            "serve-model-snapshot/v3",
        }

    @property
    def duration_complete(self) -> bool:
        return self.schema_version == "serve-model-snapshot/v3"

    @property
    def component_artifact_ids(self) -> Mapping[ServeComponent, str]:
        return MappingProxyType(
            {reference.component: reference.artifact_id for reference in self.component_artifacts}
        )

    @property
    def snapshot_id(self) -> str:
        """Location-independent identity of the fitted statistical snapshot."""

        identity: dict[str, Any] = {
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "implementation_version": self.implementation_version,
            "tour": self.tour.value,
            "fitted_at_utc": self.fitted_at_utc.isoformat(),
            "data_cutoff_utc": self.data_cutoff_utc.isoformat(),
            "component_artifact_ids": {
                reference.component.value: reference.artifact_id
                for reference in self.component_artifacts
            },
            "data_hash": self.data_hash,
            "component_count_artifact_hash": self.component_count_artifact_hash,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
        }
        if self.b6_c6_complete:
            assert self.retirement_artifact is not None
            assert self.inactivity_configuration is not None
            identity["retirement_artifact_id"] = self.retirement_artifact.artifact_id
            identity["retirement_schema_version"] = self.retirement_schema_version
            identity["inactivity_configuration_id"] = self.inactivity_configuration.artifact_id
            identity["inactivity_schema_version"] = self.inactivity_schema_version
        if self.duration_complete:
            assert self.duration_artifact is not None
            identity["duration_artifact_id"] = self.duration_artifact.artifact_id
            identity["duration_schema_version"] = self.duration_schema_version
        return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        """Return a stable machine-readable record; no fitted matrices are copied."""

        return _canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> ModelSnapshot:
        """Parse snapshot references without loading or refitting components."""

        return cls.model_validate_json(payload)


def create_model_snapshot(
    artifacts: Mapping[ServeComponent, FitArtifact],
    *,
    retirement_artifact: PersistedRetirementFitArtifact | None = None,
    inactivity_configuration: InactivityConfigurationArtifact | None = None,
    duration_artifact: PersistedDurationFitArtifact | None = None,
) -> ModelSnapshot:
    """Create a snapshot from five already-persisted, content-verified fits."""

    if set(artifacts) != set(ServeComponent):
        raise ModelSnapshotError("snapshot requires exactly the five F/A/Q1/D/Q2 artifacts")

    verified: dict[ServeComponent, FitArtifact] = {}
    for component in _COMPONENT_ORDER:
        artifact = artifacts[component]
        if not isinstance(artifact, FitArtifact):
            raise ModelSnapshotError("snapshot inputs must be verified FitArtifact values")
        try:
            loaded = load_fit_artifact(artifact.directory)
        except FitArtifactError as exc:
            raise ModelSnapshotError(f"cannot verify {component.value} artifact: {exc}") from exc
        if loaded.artifact_id != artifact.artifact_id or loaded.fit != artifact.fit:
            raise ModelSnapshotError(f"{component.value} artifact reference is inconsistent")
        if loaded.fit.component is not component:
            raise ModelSnapshotError(f"{component.value} artifact contains the wrong component")
        verified[component] = loaded

    try:
        identity = validate_serve_fit_bundle(
            {component: artifact.fit for component, artifact in verified.items()}
        )
    except ModelDataError as exc:
        raise ModelSnapshotError(
            f"component artifacts are not one coherent snapshot: {exc}"
        ) from exc
    if (retirement_artifact is None) != (inactivity_configuration is None):
        raise ModelSnapshotError("B6 and C6 snapshot references must be supplied together")
    if duration_artifact is not None and retirement_artifact is None:
        raise ModelSnapshotError("B5 snapshot integration requires the complete B6/C6 snapshot")
    verified_retirement: PersistedRetirementFitArtifact | None = None
    if retirement_artifact is not None:
        try:
            verified_retirement = load_retirement_fit_artifact(retirement_artifact.directory)
        except RetirementArtifactError as exc:
            raise ModelSnapshotError(f"cannot verify retirement artifact: {exc}") from exc
        if (
            verified_retirement.artifact_id != retirement_artifact.artifact_id
            or verified_retirement.artifact != retirement_artifact.artifact
        ):
            raise ModelSnapshotError("retirement artifact reference is inconsistent")
        if verified_retirement.artifact.tour is not identity.tour:
            raise ModelSnapshotError("retirement artifact belongs to another tour")
        if verified_retirement.artifact.information_cutoff_utc != identity.information_cutoff_utc:
            raise ModelSnapshotError("retirement and serve artifacts must share a data cutoff")
    verified_duration: PersistedDurationFitArtifact | None = None
    if duration_artifact is not None:
        try:
            verified_duration = load_duration_fit_artifact(duration_artifact.directory)
        except DurationArtifactError as exc:
            raise ModelSnapshotError(f"cannot verify duration artifact: {exc}") from exc
        if (
            verified_duration.artifact_id != duration_artifact.artifact_id
            or verified_duration.artifact != duration_artifact.artifact
        ):
            raise ModelSnapshotError("duration artifact reference is inconsistent")
        if verified_duration.artifact.tour is not identity.tour:
            raise ModelSnapshotError("duration artifact belongs to another tour")
        if verified_duration.artifact.information_cutoff_utc != identity.information_cutoff_utc:
            raise ModelSnapshotError("duration and serve artifacts must share a data cutoff")
        if not verified_duration.artifact.diagnostics.converged:
            raise ModelSnapshotError("duration artifact optimizer did not converge")
    return ModelSnapshot(
        schema_version=(
            "serve-model-snapshot/v3"
            if verified_duration is not None
            else "serve-model-snapshot/v2"
            if verified_retirement is not None
            else "serve-model-snapshot/v1"
        ),
        framework_version=identity.framework_version,
        implementation_version=identity.implementation_version,
        tour=identity.tour,
        fitted_at_utc=identity.fitted_at_utc,
        data_cutoff_utc=identity.information_cutoff_utc,
        component_artifacts=tuple(
            ComponentArtifactReference(
                component=component,
                artifact_id=verified[component].artifact_id,
                directory=verified[component].directory.resolve(),
            )
            for component in _COMPONENT_ORDER
        ),
        data_hash=identity.data_snapshot_sha256,
        component_count_artifact_hash=identity.component_count_artifact_sha256,
        config_hash=identity.model_config_sha256,
        code_commit=identity.code_commit,
        retirement_artifact=(
            None
            if verified_retirement is None
            else RetirementArtifactReference(
                artifact_id=verified_retirement.artifact_id,
                directory=verified_retirement.directory.resolve(),
                tour=verified_retirement.artifact.tour,
                information_cutoff_utc=verified_retirement.artifact.information_cutoff_utc,
                fitted_at_utc=verified_retirement.artifact.fitted_at_utc,
            )
        ),
        retirement_schema_version=(
            None if verified_retirement is None else verified_retirement.artifact.schema_version
        ),
        inactivity_configuration=inactivity_configuration,
        inactivity_schema_version=(
            None if inactivity_configuration is None else inactivity_configuration.schema_version
        ),
        duration_artifact=(
            None
            if verified_duration is None
            else DurationArtifactReference(
                artifact_id=verified_duration.artifact_id,
                directory=verified_duration.directory.resolve(),
                tour=verified_duration.artifact.tour,
                information_cutoff_utc=verified_duration.artifact.information_cutoff_utc,
                fitted_at_utc=verified_duration.artifact.fitted_at_utc,
            )
        ),
        duration_schema_version=(
            None if verified_duration is None else verified_duration.artifact.schema_version
        ),
    )


def load_snapshot_retirement_artifact(snapshot: ModelSnapshot) -> RetirementFitArtifact:
    """Load and content-verify the B6 artifact referenced by a v2/v3 snapshot."""

    if snapshot.retirement_artifact is None:
        raise ModelSnapshotError("snapshot has no B6 retirement artifact")
    try:
        persisted = load_retirement_fit_artifact(snapshot.retirement_artifact.directory)
    except RetirementArtifactError as exc:
        raise ModelSnapshotError(f"cannot load snapshot retirement artifact: {exc}") from exc
    if persisted.artifact_id != snapshot.retirement_artifact.artifact_id:
        raise ModelSnapshotError("loaded retirement artifact does not match snapshot identity")
    if persisted.artifact.tour is not snapshot.tour:
        raise ModelSnapshotError("loaded retirement artifact tour differs from snapshot")
    if persisted.artifact.schema_version != snapshot.retirement_schema_version:
        raise ModelSnapshotError("loaded retirement artifact schema differs from snapshot")
    if (
        persisted.artifact.information_cutoff_utc
        != snapshot.retirement_artifact.information_cutoff_utc
        or persisted.artifact.fitted_at_utc != snapshot.retirement_artifact.fitted_at_utc
    ):
        raise ModelSnapshotError("loaded retirement artifact timestamps differ from snapshot")
    return persisted.artifact


def load_snapshot_duration_artifact(snapshot: ModelSnapshot) -> DurationFitArtifact:
    """Load and content-verify the B5 artifact referenced by a v3 snapshot."""

    if snapshot.duration_artifact is None:
        raise ModelSnapshotError("snapshot has no B5 duration artifact")
    try:
        persisted = load_duration_fit_artifact(snapshot.duration_artifact.directory)
    except DurationArtifactError as exc:
        raise ModelSnapshotError(f"cannot load snapshot duration artifact: {exc}") from exc
    if persisted.artifact_id != snapshot.duration_artifact.artifact_id:
        raise ModelSnapshotError("loaded duration artifact does not match snapshot identity")
    if persisted.artifact.tour is not snapshot.tour:
        raise ModelSnapshotError("loaded duration artifact tour differs from snapshot")
    if persisted.artifact.schema_version != snapshot.duration_schema_version:
        raise ModelSnapshotError("loaded duration artifact schema differs from snapshot")
    if not persisted.artifact.diagnostics.converged:
        raise ModelSnapshotError("loaded duration artifact optimizer did not converge")
    if (
        persisted.artifact.information_cutoff_utc
        != snapshot.duration_artifact.information_cutoff_utc
        or persisted.artifact.fitted_at_utc != snapshot.duration_artifact.fitted_at_utc
    ):
        raise ModelSnapshotError("loaded duration artifact timestamps differ from snapshot")
    return persisted.artifact


def load_snapshot_fits(
    snapshot: ModelSnapshot,
) -> Mapping[ServeComponent, FittedServeComponent]:
    """Load exactly the referenced fits, verify all identities, and never refit."""

    snapshot = ModelSnapshot.model_validate(snapshot.model_dump(mode="python"))
    loaded: dict[ServeComponent, FittedServeComponent] = {}
    for reference in snapshot.component_artifacts:
        try:
            artifact = load_fit_artifact(reference.directory)
        except FitArtifactError as exc:
            raise ModelSnapshotError(
                f"cannot load {reference.component.value} snapshot artifact: {exc}"
            ) from exc
        if artifact.artifact_id != reference.artifact_id:
            raise ModelSnapshotError(
                f"{reference.component.value} artifact ID does not match the snapshot"
            )
        if artifact.fit.component is not reference.component:
            raise ModelSnapshotError(
                f"{reference.component.value} snapshot reference contains the wrong fit"
            )
        loaded[reference.component] = artifact.fit

    try:
        identity = validate_serve_fit_bundle(loaded)
    except ModelDataError as exc:
        raise ModelSnapshotError(f"loaded component fits are not coherent: {exc}") from exc
    observed = (
        identity.framework_version,
        identity.implementation_version,
        identity.tour,
        identity.fitted_at_utc,
        identity.information_cutoff_utc,
        identity.data_snapshot_sha256,
        identity.component_count_artifact_sha256,
        identity.model_config_sha256,
        identity.code_commit,
    )
    expected = (
        snapshot.framework_version,
        snapshot.implementation_version,
        snapshot.tour,
        snapshot.fitted_at_utc,
        snapshot.data_cutoff_utc,
        snapshot.data_hash,
        snapshot.component_count_artifact_hash,
        snapshot.config_hash,
        snapshot.code_commit,
    )
    if observed != expected:
        raise ModelSnapshotError("loaded component fits do not match snapshot provenance")
    return MappingProxyType(loaded)


__all__ = [
    "ComponentArtifactReference",
    "DurationArtifactReference",
    "ModelSnapshot",
    "ModelSnapshotError",
    "RetirementArtifactReference",
    "create_model_snapshot",
    "load_snapshot_duration_artifact",
    "load_snapshot_fits",
    "load_snapshot_retirement_artifact",
]
