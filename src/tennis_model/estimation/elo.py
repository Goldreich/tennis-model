"""Deterministic surface-aware Elo strength artifacts for production v1.1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, log
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from tennis_model.estimation.strength import StrengthPrediction
from tennis_model.schemas import FrozenModel, Tour


class SurfaceEloError(ValueError):
    """Surface Elo input, identity, or artifact provenance is invalid."""


class _EloModel(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return normalized


def _normalized_identity(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.strip())
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _logistic(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


class SurfaceEloConfig(_EloModel):
    schema_version: Literal["surface-elo-config/v1"]
    initial_rating: Annotated[float, Field(gt=0.0)] = 1500.0
    k_factor: Annotated[float, Field(gt=0.0)] = 16.0
    surface_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    rating_scale: Annotated[float, Field(gt=0.0)] = 400.0
    deterministic_logit_variance_floor: Annotated[float, Field(gt=0.0)] = 1e-12


class SurfaceEloRating(_EloModel):
    surface: str
    rating: float

    @field_validator("surface")
    @classmethod
    def surface_is_present(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not normalized:
            raise ValueError("surface must not be empty")
        return normalized


class SurfaceEloPlayerState(_EloModel):
    player_id: str
    player_name: str
    aliases: tuple[str, ...] = ()
    global_rating: float
    surfaces: tuple[SurfaceEloRating, ...]
    eligible_matches: int = Field(ge=0)
    last_played_utc: datetime | None = None

    @field_validator("player_id", "player_name")
    @classmethod
    def text_is_present(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @field_validator("aliases")
    @classmethod
    def aliases_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value if item.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("surface Elo aliases must be unique")
        return normalized

    @field_validator("last_played_utc")
    @classmethod
    def last_played_is_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value, field="last_played_utc")

    @model_validator(mode="after")
    def surfaces_are_unique(self) -> Self:
        names = tuple(item.surface for item in self.surfaces)
        if len(names) != len(set(names)):
            raise ValueError("surface Elo player state contains duplicate surfaces")
        return self


class SurfaceEloDiagnostics(_EloModel):
    players: int = Field(ge=0)
    eligible_matches: int = Field(ge=0)
    initialized_players: int = Field(ge=0)


class SurfaceEloFit(_EloModel):
    schema_version: Literal["surface-elo-fit/v1"] = "surface-elo-fit/v1"
    framework_version: Literal["v1.1-candidate", "v1.1"] = "v1.1"
    tour: Tour
    information_cutoff_utc: datetime
    fitted_at_utc: datetime
    config: SurfaceEloConfig
    ratings_sha256: str
    source_manifest_sha256: str
    code_commit: str
    player_states: tuple[SurfaceEloPlayerState, ...]
    diagnostics: SurfaceEloDiagnostics

    @field_validator("information_cutoff_utc", "fitted_at_utc")
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @field_validator("ratings_sha256", "source_manifest_sha256")
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return _digest(value, field=info.field_name)

    @model_validator(mode="after")
    def fit_is_coherent(self) -> Self:
        if self.fitted_at_utc < self.information_cutoff_utc:
            raise ValueError("surface Elo fit timestamp cannot precede its cutoff")
        ids = tuple(item.player_id for item in self.player_states)
        if len(ids) != len(set(ids)):
            raise ValueError("surface Elo fit contains duplicate player IDs")
        if self.diagnostics.players != len(self.player_states):
            raise ValueError("surface Elo player diagnostic differs from stored states")
        return self


@dataclass(frozen=True, slots=True)
class PersistedSurfaceEloArtifact:
    directory: Path
    artifact_id: str
    fit: SurfaceEloFit


def import_surface_elo_csv(
    ratings_path: str | Path,
    source_manifest_path: str | Path,
    *,
    tour: Tour,
    cutoff_utc: datetime,
    fitted_at_utc: datetime,
    config: SurfaceEloConfig,
    code_commit: str,
) -> SurfaceEloFit:
    """Import a computed Elo table without recomputing or changing its ratings."""

    ratings_payload = Path(ratings_path).read_bytes()
    manifest_payload = Path(source_manifest_path).read_bytes()
    reader = csv.DictReader(ratings_payload.decode("utf-8-sig").splitlines())
    states: list[SurfaceEloPlayerState] = []
    initialized = 0
    for row in reader:
        if str(row.get("tour", "")).strip().upper() != tour.value:
            continue
        player_id = str(row.get("player_id") or row.get("player_key") or "").strip()
        player_name = str(row.get("player_name") or "").strip()
        if not player_id or not player_name:
            raise SurfaceEloError("rating rows require player_id/player_key and player_name")
        player_key = str(row.get("player_key") or "").strip()
        aliases = tuple(
            sorted(
                {
                    item
                    for item in (player_id, player_key, _normalized_identity(player_name))
                    if item
                }
            )
        )
        matches = int(float(str(row.get("matches") or 0)))
        if matches == 0:
            initialized += 1
        last_value = re.sub(r"\D", "", str(row.get("last_event_start") or ""))
        last_played = None
        if len(last_value) >= 8 and int(last_value[:8]) > 0:
            last_played = datetime.strptime(last_value[:8], "%Y%m%d").replace(tzinfo=UTC)
        surfaces = tuple(
            SurfaceEloRating(
                surface=surface,
                rating=float(str(row.get(f"{surface}_elo") or config.initial_rating)),
            )
            for surface in ("hard", "clay", "grass", "carpet")
        )
        states.append(
            SurfaceEloPlayerState(
                player_id=player_id,
                player_name=player_name,
                aliases=aliases,
                global_rating=float(str(row["global_elo"])),
                surfaces=surfaces,
                eligible_matches=matches,
                last_played_utc=last_played,
            )
        )
    if not states:
        raise SurfaceEloError(f"ratings table contains no {tour.value} players")
    states.sort(key=lambda item: item.player_id)
    return SurfaceEloFit(
        tour=tour,
        information_cutoff_utc=_utc(cutoff_utc, field="cutoff_utc"),
        fitted_at_utc=_utc(fitted_at_utc, field="fitted_at_utc"),
        config=config,
        ratings_sha256=_sha256(ratings_payload),
        source_manifest_sha256=_sha256(manifest_payload),
        code_commit=code_commit.strip() or "unknown",
        player_states=tuple(states),
        diagnostics=SurfaceEloDiagnostics(
            players=len(states),
            eligible_matches=sum(item.eligible_matches for item in states) // 2,
            initialized_players=initialized,
        ),
    )


def _resolve_player(fit: SurfaceEloFit, identity: str) -> SurfaceEloPlayerState | None:
    exact = {item.player_id: item for item in fit.player_states}
    if identity in exact:
        return exact[identity]
    wanted = _normalized_identity(identity)
    matches = {
        item.player_id: item
        for item in fit.player_states
        if wanted == _normalized_identity(item.player_name)
        or any(wanted == _normalized_identity(alias) for alias in item.aliases)
    }
    return next(iter(matches.values())) if len(matches) == 1 else None


def predict_surface_elo(
    fit: SurfaceEloFit,
    *,
    player_a_id: str,
    player_b_id: str,
    surface: str,
    best_of: Literal[3, 5],
    scheduled_start_utc: datetime,
) -> StrengthPrediction:
    start = _utc(scheduled_start_utc, field="scheduled_start_utc")
    if player_a_id == player_b_id:
        raise SurfaceEloError("surface Elo prediction requires distinct players")
    a = _resolve_player(fit, player_a_id)
    b = _resolve_player(fit, player_b_id)
    surface_key = surface.strip().casefold()

    def rating(state: SurfaceEloPlayerState | None) -> float:
        if state is None:
            return fit.config.initial_rating
        by_surface = {item.surface: item.rating for item in state.surfaces}
        surface_rating = by_surface.get(surface_key, fit.config.initial_rating)
        return (
            (1.0 - fit.config.surface_weight) * state.global_rating
            + fit.config.surface_weight * surface_rating
        )

    difference = rating(a) - rating(b)
    mean_logit = log(10.0) * difference / fit.config.rating_scale
    probability = min(1.0 - 1e-12, max(1e-12, _logistic(mean_logit)))
    return StrengthPrediction(
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        surface=surface_key,
        best_of=best_of,
        scheduled_start_utc=start,
        mean_logit=mean_logit,
        variance_logit=fit.config.deterministic_logit_variance_floor,
        probability=probability,
        player_a_graph_component=0 if a is not None else None,
        player_b_graph_component=0 if b is not None else None,
        weakly_connected=a is None or b is None,
        player_a_known=a is not None,
        player_b_known=b is not None,
    )


def write_surface_elo_artifact(
    fit: SurfaceEloFit, artifact_root: str | Path
) -> PersistedSurfaceEloArtifact:
    payload = _canonical_bytes(fit)
    artifact_id = _sha256(payload)
    parent = (
        Path(artifact_root).resolve()
        / fit.tour.value.lower()
        / fit.information_cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    )
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / artifact_id[:32]
    if target.exists():
        return load_surface_elo_artifact(target)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        path = staging / "surface_elo.json"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        (staging / "diagnostics.txt").write_text(
            "\n".join(
                (
                    f"framework: {fit.framework_version}",
                    f"tour: {fit.tour.value}",
                    f"cutoff: {fit.information_cutoff_utc.isoformat()}",
                    f"players: {fit.diagnostics.players}",
                    f"eligible matches: {fit.diagnostics.eligible_matches}",
                )
            )
            + "\n",
            encoding="ascii",
        )
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_surface_elo_artifact(target)


def load_surface_elo_artifact(directory: str | Path) -> PersistedSurfaceEloArtifact:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise SurfaceEloError("surface Elo artifact directory is invalid")
    path = root / "surface_elo.json"
    if path.is_symlink() or not path.is_file():
        raise SurfaceEloError("surface Elo artifact payload is missing")
    payload = path.read_bytes()
    fit = SurfaceEloFit.model_validate_json(payload)
    if _canonical_bytes(fit) != payload:
        raise SurfaceEloError("surface Elo artifact is not canonical")
    artifact_id = _sha256(payload)
    if root.name != artifact_id[:32]:
        raise SurfaceEloError("surface Elo artifact directory does not match content")
    return PersistedSurfaceEloArtifact(root.resolve(), artifact_id, fit)


__all__ = [
    "PersistedSurfaceEloArtifact",
    "SurfaceEloConfig",
    "SurfaceEloDiagnostics",
    "SurfaceEloError",
    "SurfaceEloFit",
    "SurfaceEloPlayerState",
    "SurfaceEloRating",
    "import_surface_elo_csv",
    "load_surface_elo_artifact",
    "predict_surface_elo",
    "write_surface_elo_artifact",
]
