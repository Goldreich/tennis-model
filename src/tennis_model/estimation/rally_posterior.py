"""Sequential Bayesian posterior state for the v1.1 rally-termination layer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from tennis_model.estimation.rally_termination import (
    ACCOUNTING_CONVENTION,
    RallyFitRow,
    RallyTerminationArtifact,
    canonical_player_key,
)


POSTERIOR_SCHEMA_VERSION = "rally-termination-posterior/v2"
POINTER_SCHEMA_VERSION = "rally-termination-posterior-pointer/v1"
UPDATE_METHOD_VERSION = "sequential-laplace-multinomial-logit/v1"
EFFECT_NAMES = ("aggression", "allows_winners", "pressure", "error_tendency")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("posterior timestamps must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class MatchupEffectPosterior:
    """Joint posterior for A's four effects followed by B's four effects."""

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SequentialRallyPosteriorArtifact:
    artifact_id: str
    manifest_path: Path
    schema_version: str
    tour: str
    updated_at_utc: datetime
    data_cutoff_utc: datetime
    parent_artifact_id: str
    initialization_artifact_id: str
    accounting_convention: str
    update_method_version: str
    baseline_winner_logit: float
    baseline_ue_logit: float
    concentration: float
    prior_sd: float
    players: tuple[str, ...]
    aliases: Mapping[str, str]
    seen_matches: Mapping[str, str]
    update_sequence: int
    last_update: Mapping[str, Any] | None
    posterior_mean: np.ndarray
    posterior_precision: sparse.csc_matrix
    state_sha256: str

    def __post_init__(self) -> None:
        expected = 4 * len(self.players)
        if self.posterior_mean.shape != (expected,):
            raise ValueError("posterior mean has the wrong dimension")
        if self.posterior_precision.shape != (expected, expected):
            raise ValueError("posterior precision has the wrong dimension")
        if self.accounting_convention != ACCOUNTING_CONVENTION:
            raise ValueError("posterior uses an unsupported accounting convention")

    @property
    def player_effects(self) -> dict[str, dict[str, float]]:
        return {
            player: {
                effect: float(self.posterior_mean[4 * index + offset])
                for offset, effect in enumerate(EFFECT_NAMES)
            }
            for index, player in enumerate(self.players)
        }

    def resolve_player(self, value: str) -> str:
        key = canonical_player_key(value)
        return self.aliases.get(key, key)

    def _player_index(self, value: str) -> int | None:
        key = self.resolve_player(value)
        try:
            return self.players.index(key)
        except ValueError:
            return None

    def _effect(self, player: str, offset: int) -> float:
        index = self._player_index(player)
        return 0.0 if index is None else float(self.posterior_mean[4 * index + offset])

    def direction(self, point_winner: str, point_loser: str) -> Any:
        from tennis_model.estimation.rally_termination import (
            DirectionalRallyProbabilities,
        )

        winner_logit = (
            self.baseline_winner_logit
            + self._effect(point_winner, 0)
            + self._effect(point_loser, 1)
        )
        ue_logit = (
            self.baseline_ue_logit
            + self._effect(point_winner, 2)
            + self._effect(point_loser, 3)
        )
        maximum = max(0.0, winner_logit, ue_logit)
        weights = np.exp(np.asarray((winner_logit, ue_logit, 0.0)) - maximum)
        values = weights / float(weights.sum())
        return DirectionalRallyProbabilities(*map(float, values))

    def matchup_posterior(self, player_a: str, player_b: str) -> MatchupEffectPosterior:
        requested = (player_a, player_b)
        global_indices: list[int | None] = []
        for player in requested:
            player_index = self._player_index(player)
            global_indices.extend(
                [None] * 4
                if player_index is None
                else [4 * player_index + offset for offset in range(4)]
            )
        mean = np.zeros(8, dtype=np.float64)
        known_positions = [i for i, index in enumerate(global_indices) if index is not None]
        known_indices = [int(global_indices[i]) for i in known_positions]
        for local, global_index in zip(known_positions, known_indices, strict=True):
            mean[local] = self.posterior_mean[global_index]
        covariance = np.zeros((8, 8), dtype=np.float64)
        if known_indices:
            factor = splu(self.posterior_precision)
            selectors = np.zeros((self.posterior_precision.shape[0], len(known_indices)))
            selectors[known_indices, np.arange(len(known_indices))] = 1.0
            solutions = factor.solve(selectors)
            marginal = solutions[np.asarray(known_indices), :]
            covariance[np.ix_(known_positions, known_positions)] = marginal
        for local, global_index in enumerate(global_indices):
            if global_index is None:
                covariance[local, local] = self.prior_sd * self.prior_sd
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
        return MatchupEffectPosterior(
            mean=tuple(map(float, mean)),
            covariance=tuple(tuple(map(float, row)) for row in covariance),
        )


def _state_payload(mean: np.ndarray, precision: sparse.csc_matrix, path: Path) -> str:
    matrix = precision.tocsc()
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            posterior_mean=np.asarray(mean, dtype=np.float64),
            precision_data=np.asarray(matrix.data, dtype=np.float64),
            precision_indices=np.asarray(matrix.indices, dtype=np.int64),
            precision_indptr=np.asarray(matrix.indptr, dtype=np.int64),
            precision_shape=np.asarray(matrix.shape, dtype=np.int64),
        )
    return _sha256(path.read_bytes())


def _load_state(path: Path, expected_sha256: str) -> tuple[np.ndarray, sparse.csc_matrix]:
    payload = path.read_bytes()
    observed = _sha256(payload)
    if observed != expected_sha256:
        raise ValueError("rally posterior state hash mismatch")
    with np.load(path, allow_pickle=False) as state:
        shape = tuple(int(value) for value in state["precision_shape"])
        precision = sparse.csc_matrix(
            (
                state["precision_data"],
                state["precision_indices"],
                state["precision_indptr"],
            ),
            shape=shape,
        )
        mean = np.asarray(state["posterior_mean"], dtype=np.float64)
    return mean, precision


def load_posterior_artifact(path: str | Path) -> SequentialRallyPosteriorArtifact:
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = str(raw.pop("artifact_id"))
    observed = _sha256(_canonical_bytes(raw))
    if claimed != observed:
        raise ValueError("rally posterior manifest identity mismatch")
    if raw.get("schema_version") != POSTERIOR_SCHEMA_VERSION:
        raise ValueError("unsupported rally posterior schema")
    state_path = manifest_path.parent / str(raw["state_file"])
    mean, precision = _load_state(state_path, str(raw["state_sha256"]))
    return SequentialRallyPosteriorArtifact(
        artifact_id=claimed,
        manifest_path=manifest_path,
        schema_version=str(raw["schema_version"]),
        tour=str(raw["tour"]),
        updated_at_utc=datetime.fromisoformat(str(raw["updated_at_utc"])).astimezone(UTC),
        data_cutoff_utc=datetime.fromisoformat(str(raw["data_cutoff_utc"])).astimezone(UTC),
        parent_artifact_id=str(raw["parent_artifact_id"]),
        initialization_artifact_id=str(raw["initialization_artifact_id"]),
        accounting_convention=str(raw["accounting_convention"]),
        update_method_version=str(raw["update_method_version"]),
        baseline_winner_logit=float(raw["baseline_winner_logit"]),
        baseline_ue_logit=float(raw["baseline_ue_logit"]),
        concentration=float(raw["concentration"]),
        prior_sd=float(raw["prior_sd"]),
        players=tuple(str(value) for value in raw["players"]),
        aliases={str(k): str(v) for k, v in raw["aliases"].items()},
        seen_matches={str(k): str(v) for k, v in raw["seen_matches"].items()},
        update_sequence=int(raw["update_sequence"]),
        last_update=None if raw.get("last_update") is None else dict(raw["last_update"]),
        posterior_mean=mean,
        posterior_precision=precision,
        state_sha256=str(raw["state_sha256"]),
    )


def _write_artifact(
    *,
    artifact_root: Path,
    tour: str,
    updated_at_utc: datetime,
    data_cutoff_utc: datetime,
    parent_artifact_id: str,
    initialization_artifact_id: str,
    baseline_winner_logit: float,
    baseline_ue_logit: float,
    concentration: float,
    prior_sd: float,
    players: Sequence[str],
    aliases: Mapping[str, str],
    seen_matches: Mapping[str, str],
    update_sequence: int,
    last_update: Mapping[str, Any] | None,
    posterior_mean: np.ndarray,
    posterior_precision: sparse.csc_matrix,
) -> SequentialRallyPosteriorArtifact:
    artifact_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".rally-state-", suffix=".npz", dir=artifact_root, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        state_sha256 = _state_payload(posterior_mean, posterior_precision, temporary_path)
        body = {
            "schema_version": POSTERIOR_SCHEMA_VERSION,
            "tour": tour.upper(),
            "updated_at_utc": _utc(updated_at_utc).isoformat(),
            "data_cutoff_utc": _utc(data_cutoff_utc).isoformat(),
            "parent_artifact_id": parent_artifact_id,
            "initialization_artifact_id": initialization_artifact_id,
            "accounting_convention": ACCOUNTING_CONVENTION,
            "update_method_version": UPDATE_METHOD_VERSION,
            "baseline_winner_logit": float(baseline_winner_logit),
            "baseline_ue_logit": float(baseline_ue_logit),
            "concentration": float(concentration),
            "prior_sd": float(prior_sd),
            "players": list(players),
            "aliases": dict(sorted(aliases.items())),
            "seen_matches": dict(sorted(seen_matches.items())),
            "update_sequence": int(update_sequence),
            "last_update": None if last_update is None else dict(last_update),
            "state_file": "state.npz",
            "state_sha256": state_sha256,
        }
        artifact_id = _sha256(_canonical_bytes(body))
        target = artifact_root / tour.casefold() / artifact_id
        manifest_path = target / "manifest.json"
        manifest_payload = _canonical_bytes({"artifact_id": artifact_id, **body})
        if target.exists():
            if (
                not manifest_path.exists()
                or manifest_path.read_bytes() != manifest_payload
                or _sha256((target / "state.npz").read_bytes()) != state_sha256
            ):
                raise RuntimeError("existing immutable posterior artifact conflicts")
        else:
            target.mkdir(parents=True)
            temporary_path.replace(target / "state.npz")
            manifest_path.write_bytes(manifest_payload)
        return load_posterior_artifact(manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _index(players: Sequence[str]) -> dict[str, int]:
    return {player: 4 * index for index, player in enumerate(players)}


def _resolved_row(row: RallyFitRow, aliases: Mapping[str, str]) -> RallyFitRow:
    winner = aliases.get(canonical_player_key(row.point_winner), canonical_player_key(row.point_winner))
    loser = aliases.get(canonical_player_key(row.point_loser), canonical_player_key(row.point_loser))
    return RallyFitRow(
        winner,
        loser,
        row.rally_winners,
        row.opponent_unforced_errors,
        row.other_or_forced,
    )


def _row_terms(
    theta: np.ndarray,
    row: RallyFitRow,
    player_index: Mapping[str, int],
    baseline_winner_logit: float,
    baseline_ue_logit: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    actor = player_index[row.point_winner]
    opponent = player_index[row.point_loser]
    winner_indices = (actor, opponent + 1)
    ue_indices = (actor + 2, opponent + 3)
    eta_w = baseline_winner_logit + sum(theta[index] for index in winner_indices)
    eta_u = baseline_ue_logit + sum(theta[index] for index in ue_indices)
    maximum = max(0.0, float(eta_w), float(eta_u))
    weights = np.exp(np.asarray((eta_w, eta_u, 0.0)) - maximum)
    probabilities = weights / float(weights.sum())
    counts = np.asarray(
        (row.rally_winners, row.opponent_unforced_errors, row.other_or_forced),
        dtype=np.float64,
    )
    return probabilities, counts, winner_indices, ue_indices


def _add_row_hessian(
    matrix: sparse.lil_matrix,
    theta: np.ndarray,
    row: RallyFitRow,
    player_index: Mapping[str, int],
    baseline_winner_logit: float,
    baseline_ue_logit: float,
) -> None:
    probabilities, counts, winner_indices, ue_indices = _row_terms(
        theta, row, player_index, baseline_winner_logit, baseline_ue_logit
    )
    total = float(counts.sum())
    p_w, p_u = map(float, probabilities[:2])
    information = total * np.asarray(
        ((p_w * (1.0 - p_w), -p_w * p_u), (-p_w * p_u, p_u * (1.0 - p_u)))
    )
    groups = (winner_indices, ue_indices)
    for left_group, left_indices in enumerate(groups):
        for right_group, right_indices in enumerate(groups):
            value = float(information[left_group, right_group])
            for left in left_indices:
                for right in right_indices:
                    matrix[left, right] += value


def initialize_posterior(
    base: RallyTerminationArtifact,
    rows: Sequence[RallyFitRow],
    *,
    seen_matches: Mapping[str, str],
    artifact_root: str | Path,
    updated_at_utc: datetime,
) -> SequentialRallyPosteriorArtifact:
    players = tuple(sorted(base.player_effects))
    player_index = _index(players)
    mean = np.asarray(
        [
            float(base.player_effects[player][effect])
            for player in players
            for effect in EFFECT_NAMES
        ],
        dtype=np.float64,
    )
    precision = sparse.eye(
        len(mean), format="lil", dtype=np.float64
    ) * (1.0 / (base.prior_sd * base.prior_sd))
    for raw_row in rows:
        row = _resolved_row(raw_row, base.aliases)
        if row.point_winner not in player_index or row.point_loser not in player_index:
            continue
        _add_row_hessian(
            precision,
            mean,
            row,
            player_index,
            base.baseline_winner_logit,
            base.baseline_ue_logit,
        )
    return _write_artifact(
        artifact_root=Path(artifact_root),
        tour=base.tour,
        updated_at_utc=updated_at_utc,
        data_cutoff_utc=base.data_cutoff_utc,
        parent_artifact_id=base.artifact_id,
        initialization_artifact_id=base.artifact_id,
        baseline_winner_logit=base.baseline_winner_logit,
        baseline_ue_logit=base.baseline_ue_logit,
        concentration=base.concentration,
        prior_sd=base.prior_sd,
        players=players,
        aliases=base.aliases,
        seen_matches=seen_matches,
        update_sequence=0,
        last_update={
            "kind": "initialization",
            "source_snapshot_id": base.source_snapshot_id,
            "source_manifest_sha256": base.source_manifest_sha256,
            "directional_rows": len(rows),
        },
        posterior_mean=mean,
        posterior_precision=precision.tocsc(),
    )


def _extend_state(
    artifact: SequentialRallyPosteriorArtifact,
    rows: Sequence[RallyFitRow],
    aliases: Mapping[str, str],
) -> tuple[tuple[str, ...], np.ndarray, sparse.csc_matrix, tuple[RallyFitRow, ...]]:
    combined_aliases = {**artifact.aliases, **aliases}
    resolved = tuple(_resolved_row(row, combined_aliases) for row in rows)
    new_players = sorted(
        ({row.point_winner for row in resolved} | {row.point_loser for row in resolved})
        - set(artifact.players)
    )
    if not new_players:
        return artifact.players, artifact.posterior_mean.copy(), artifact.posterior_precision.copy(), resolved
    players = (*artifact.players, *new_players)
    mean = np.concatenate((artifact.posterior_mean, np.zeros(4 * len(new_players))))
    added_precision = sparse.eye(
        4 * len(new_players), format="csc"
    ) * (1.0 / (artifact.prior_sd * artifact.prior_sd))
    precision = sparse.block_diag(
        (artifact.posterior_precision, added_precision), format="csc"
    )
    return tuple(players), mean, precision, resolved


def update_posterior(
    artifact: SequentialRallyPosteriorArtifact,
    rows: Sequence[RallyFitRow],
    *,
    match_id: str,
    source_sha256: str,
    available_at_utc: datetime,
    aliases: Mapping[str, str],
    artifact_root: str | Path,
    updated_at_utc: datetime,
) -> SequentialRallyPosteriorArtifact:
    if match_id in artifact.seen_matches:
        if artifact.seen_matches[match_id] != source_sha256:
            raise ValueError("pre1iously observed match ID has different immutable source content")
        return artifact
    if len(rows) != 2:
        raise ValueError("one completed singles match must provide exactly two directional rows")
    players, old_mean, old_precision, resolved_rows = _extend_state(
        artifact, rows, aliases
    )
    player_index = _index(players)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        delta = theta - old_mean
        gradient = np.asarray(old_precision @ delta).reshape(-1)
        value = 0.5 * float(delta @ gradient)
        for row in resolved_rows:
            probabilities, counts, winner_indices, ue_indices = _row_terms(
                theta,
                row,
                player_index,
                artifact.baseline_winner_logit,
                artifact.baseline_ue_logit,
            )
            value -= float(np.dot(counts, np.log(probabilities)))
            residual = counts.sum() * probabilities[:2] - counts[:2]
            for index in winner_indices:
                gradient[index] += residual[0]
            for index in ue_indices:
                gradient[index] += residual[1]
        return value, gradient

    fitted = minimize(
        objective,
        old_mean,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError(f"sequential rally posterior update failed: {fitted.message}")
    new_mean = np.asarray(fitted.x, dtype=np.float64)
    new_precision = old_precision.tolil(copy=True)
    for row in resolved_rows:
        _add_row_hessian(
            new_precision,
            new_mean,
            row,
            player_index,
            artifact.baseline_winner_logit,
            artifact.baseline_ue_logit,
        )
    cutoff = max(
        artifact.data_cutoff_utc,
        _utc(available_at_utc),
    )
    combined_aliases = {**artifact.aliases, **aliases}
    seen_matches = {**artifact.seen_matches, match_id: source_sha256}
    return _write_artifact(
        artifact_root=Path(artifact_root),
        tour=artifact.tour,
        updated_at_utc=updated_at_utc,
        data_cutoff_utc=cutoff,
        parent_artifact_id=artifact.artifact_id,
        initialization_artifact_id=artifact.initialization_artifact_id,
        baseline_winner_logit=artifact.baseline_winner_logit,
        baseline_ue_logit=artifact.baseline_ue_logit,
        concentration=artifact.concentration,
        prior_sd=artifact.prior_sd,
        players=players,
        aliases=combined_aliases,
        seen_matches=seen_matches,
        update_sequence=artifact.update_sequence + 1,
        last_update={
            "kind": "completed_match",
            "match_id": match_id,
            "source_sha256": source_sha256,
            "available_at_utc": _utc(available_at_utc).isoformat(),
            "directional_counts": [
                {
                    "point_winner": row.point_winner,
                    "point_loser": row.point_loser,
                    "rally_winners": row.rally_winners,
                    "opponent_unforced_errors": row.opponent_unforced_errors,
                    "other_or_forced": row.other_or_forced,
                }
                for row in resolved_rows
            ],
            "optimizer_iterations": int(fitted.nit),
        },
        posterior_mean=new_mean,
        posterior_precision=new_precision.tocsc(),
    )


def pointer_path(production_root: str | Path, tour: str) -> Path:
    return Path(production_root) / f"rally_termination_{tour.casefold()}_current.json"


def activate_posterior(
    artifact: SequentialRallyPosteriorArtifact,
    production_root: str | Path,
) -> Path:
    root = Path(production_root).resolve()
    target = pointer_path(root, artifact.tour)
    relative_manifest = artifact.manifest_path.relative_to(root).as_posix()
    manifest_sha256 = _sha256(artifact.manifest_path.read_bytes())
    payload = _canonical_bytes(
        {
            "schema_version": POINTER_SCHEMA_VERSION,
            "tour": artifact.tour,
            "artifact_id": artifact.artifact_id,
            "manifest_path": relative_manifest,
            "manifest_sha256": manifest_sha256,
            "activated_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


def load_active_posterior(
    production_root: str | Path,
    tour: str,
) -> SequentialRallyPosteriorArtifact:
    root = Path(production_root).resolve()
    pointer = json.loads(pointer_path(root, tour).read_text(encoding="utf-8"))
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ValueError("unsupported rally posterior pointer schema")
    manifest = root / str(pointer["manifest_path"])
    if _sha256(manifest.read_bytes()) != pointer["manifest_sha256"]:
        raise ValueError("active rally posterior manifest hash mismatch")
    artifact = load_posterior_artifact(manifest)
    if artifact.artifact_id != pointer["artifact_id"]:
        raise ValueError("active rally posterior ID mismatch")
    return artifact


def load_parent(
    artifact: SequentialRallyPosteriorArtifact,
    production_root: str | Path,
) -> SequentialRallyPosteriorArtifact | None:
    candidate = (
        Path(production_root)
        / "rally-posterior"
        / artifact.tour.casefold()
        / artifact.parent_artifact_id
        / "manifest.json"
    )
    return load_posterior_artifact(candidate) if candidate.exists() else None


__all__ = [
    "EFFECT_NAMES",
    "MatchupEffectPosterior",
    "POINTER_SCHEMA_VERSION",
    "POSTERIOR_SCHEMA_VERSION",
    "SequentialRallyPosteriorArtifact",
    "UPDATE_METHOD_VERSION",
    "activate_posterior",
    "initialize_posterior",
    "load_active_posterior",
    "load_parent",
    "load_posterior_artifact",
    "pointer_path",
    "update_posterior",
]
