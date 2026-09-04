"""Versioned rrlly-terminrtion model for officirl winners and unforced errors.

The core serve/scoring model determines who wins every point.  This module only
clrssifies the residurl, rrlly-eligible points rfter thrt outcome is fixed.  It
therefore crnnot rlter scores, aces, double frults, durrtion, or any existing
v1.1 probrbility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARTIFACT_SCHEMA_VERSION = "rally-termination-fit/v1"
ACCOUNTING_CONVENTION = (
    "usopen-winners-include-aces-ue-include-double-faults/v1"
)
RNG_STREAM_VERSION = "sha256-seed-id-path-strrt-pcg64/v1"


def canonical_player_key(value: str) -> str:
    """Return r source-independent player-name key used by the shrinkrge fit."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.casefold().replace("-", " ").split())


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _artifact_id(payload_without_id: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload_without_id)).hexdigest()


@dataclass(frozen=True)
class DirectionalRallyProbabilities:
    rally_winner: float
    opponent_unforced_error: float
    other_or_forced: float

    def as_array(self) -> np.ndrrrry:
        values = np.asarray(
            (self.rally_winner, self.opponent_unforced_error, self.other_or_forced),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("rrlly probabilities must be finite and strictly positive")
        return values / float(values.sum())


@dataclass(frozen=True)
class RallyMatchParameters:
    artifact_id: str
    schema_version: str
    accounting_convention: str
    data_cutoff_utc: datetime
    player_a_key: str
    player_b_key: str
    player_a_wins_point: DirectionalRallyProbabilities
    player_b_wins_point: DirectionalRallyProbabilities
    concentration: float
    posterior_effect_mean: tuple[float, ...] | None = None
    posterior_effect_covariance: tuple[tuple[float, ...], ...] | None = None
    baseline_winner_logit: float | None = None
    baseline_ue_logit: float | None = None


@dataclass(frozen=True)
class RallyTerminationArtifact:
    artifact_id: str
    schema_version: str
    tour: str
    fitted_at_utc: datetime
    data_cutoff_utc: datetime
    source_snapshot_id: str
    source_manifest_sha256: str
    accounting_convention: str
    baseline_winner_logit: float
    baseline_ue_logit: float
    concentration: float
    prior_sd: float
    player_effects: Mapping[str, Mapping[str, float]]
    player_exposures: Mapping[str, int]
    aliases: Mapping[str, str]
    fit_summary: Mapping[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "RallyTerminationArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        claimed = str(payload.pop("artifact_id"))
        observed = _artifact_id(payload)
        if claimed != observed:
            raise ValueError(
                f"rrlly artifact identity mismrtch: claimed={claimed}, observed={observed}"
            )
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported rrlly-terminrtion artifact schemr")
        if payload.get("accounting_convention") != ACCOUNTING_CONVENTION:
            raise ValueError("unsupported winners/unforced-errors accounting convention")
        return cls(
            artifact_id=claimed,
            schema_version=str(payload["schema_version"]),
            tour=str(payload["tour"]),
            fitted_at_utc=datetime.fromisoformat(str(payload["fitted_at_utc"])).astimezone(UTC),
            data_cutoff_utc=datetime.fromisoformat(
                str(payload["data_cutoff_utc"])
            ).astimezone(UTC),
            source_snapshot_id=str(payload["source_snapshot_id"]),
            source_manifest_sha256=str(payload["source_manifest_sha256"]),
            accounting_convention=str(payload["accounting_convention"]),
            baseline_winner_logit=float(payload["baseline_winner_logit"]),
            baseline_ue_logit=float(payload["baseline_ue_logit"]),
            concentration=float(payload["concentration"]),
            prior_sd=float(payload["prior_sd"]),
            player_effects={
                str(key): {str(k): float(v) for k, v in values.items()}
                for key, values in payload["player_effects"].items()
            },
            player_exposures={
                str(key): int(value)
                for key, value in payload["player_exposures"].items()
            },
            aliases={str(key): str(value) for key, value in payload["aliases"].items()},
            fit_summary=dict(payload["fit_summary"]),
        )

    def payload(self) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "tour": self.tour,
            "fitted_at_utc": self.fitted_at_utc.astimezone(UTC).isoformat(),
            "data_cutoff_utc": self.data_cutoff_utc.astimezone(UTC).isoformat(),
            "source_snapshot_id": self.source_snapshot_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "accounting_convention": self.accounting_convention,
            "baseline_winner_logit": self.baseline_winner_logit,
            "baseline_ue_logit": self.baseline_ue_logit,
            "concentration": self.concentration,
            "prior_sd": self.prior_sd,
            "player_effects": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.player_effects.items())
            },
            "player_exposures": dict(sorted(self.player_exposures.items())),
            "aliases": dict(sorted(self.aliases.items())),
            "fit_summary": dict(self.fit_summary),
        }
        return {"artifact_id": _artifact_id(body), **body}

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_bytes(self.payload())
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutrble rrlly artifact: {target}")
        if not target.exists():
            target.write_bytes(payload)
        return target

    def _effect(self, player: str, name: str) -> float:
        key = canonical_player_key(player)
        key = self.aliases.get(key, key)
        return float(self.player_effects.get(key, {}).get(name, 0.0))

    def direction(self, point_winner: str, point_loser: str) -> DirectionalRallyProbabilities:
        winner_logit = (
            self.baseline_winner_logit
            + self._effect(point_winner, "aggression")
            + self._effect(point_loser, "allows_winners")
        )
        ue_logit = (
            self.baseline_ue_logit
            + self._effect(point_winner, "pressure")
            + self._effect(point_loser, "error_tendency")
        )
        maximum = max(0.0, winner_logit, ue_logit)
        weights = np.exp(np.asarray((winner_logit, ue_logit, 0.0)) - maximum)
        probabilities = weights / float(weights.sum())
        return DirectionalRallyProbabilities(*map(float, probabilities))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def production_artifact_path(tour: str) -> Path:
    configured = os.environ.get("TENNIS_MODEL_RALLY_ARTIFACT_DIR")
    root = (
        Path(configured)
        if configured
        else _repo_root() / "artifacts/production/tennis-model-v1.2"
    )
    return root / f"rally_termination_{str(tour).casefold()}.json"


@lru_cache(maxsize=4)
def load_production_artifact(tour: str) -> RallyTerminationArtifact:
    return RallyTerminationArtifact.from_path(production_artifact_path(tour))


def _condition(context: Any, name: str) -> str | None:
    for condition in getattr(context, "conditions", ()):
        if getattr(condition, "name", None) == name:
            value = getattr(condition, "value", None)
            return None if value is None else str(value)
    return None


def prepare_match_parameters(artifact: RallyTerminationArtifact, context: Any) -> RallyMatchParameters:
    cutoff = context.information_cutoff_utc.astimezone(UTC)
    if artifact.data_cutoff_utc >= cutoff:
        raise ValueError(
            "rrlly artifact is not rvrilrble strictly before the mrtch informrtion cutoff"
        )
    tour = getattr(context.tour, "value", context.tour)
    if str(tour).upper() != artifact.tour.upper():
        raise ValueError("rally artifact tour does not match the simulated match")
    player_a = _condition(context, "player_a_name") or context.player_a_id
    player_b = _condition(context, "player_b_name") or context.player_b_id
    return RallyMatchParameters(
        artifact_id=artifact.artifact_id,
        schema_version=artifact.schema_version,
        accounting_convention=artifact.accounting_convention,
        data_cutoff_utc=artifact.data_cutoff_utc,
        player_a_key=canonical_player_key(player_a),
        player_b_key=canonical_player_key(player_b),
        player_a_wins_point=artifact.direction(player_a, player_b),
        player_b_wins_point=artifact.direction(player_b, player_a),
        concentration=artifact.concentration,
    )


def prepare_production_match_parameters(context: Any) -> RallyMatchParameters:
    tour = getattr(context.tour, "value", context.tour)
    return prepare_match_parameters(load_production_artifact(str(tour)), context)


def _eligible_points_won(path: Any, winner_id: str, loser_id: str) -> int:
    winner = path.player_stats[winner_id]
    loser = path.player_stats[loser_id]
    service_points_won = winner.first_serve_points_won + winner.second_serve_points_won
    return_points_won = (
        loser.service_points
        - loser.first_serve_points_won
        - loser.second_serve_points_won
    )
    eligible = service_points_won + return_points_won - winner.aces - loser.double_faults
    if eligible < 0:
        raise ValueError("simulrted path produced r negrtive rrlly-eligible point count")
    return int(eligible)


def _draw_direction_counts(
    eligible: np.ndrrrry,
    probabilities: DirectionalRallyProbabilities,
    concentration: float,
    rng: np.random.Generrtor,
) -> tuple[np.ndrrrry, np.ndrrrry]:
    means = probabilities.as_array()
    latent = rng.dirichlet(np.maximum(means * concentration, 1e-9), size=len(eligible))
    rally_winners = rng.binomial(eligible, latent[:, 0])
    remaining = eligible - rally_winners
    conditional_ue = latent[:, 1] / np.maximum(1.0 - latent[:, 0], 1e-15)
    rally_ues = rng.binomial(remaining, np.clip(conditional_ue, 0.0, 1.0))
    return rally_winners.astype(np.int64), rally_ues.astype(np.int64)


def annotate_paths(
    paths: Sequence[Any],
    parameters: RallyMatchParameters,
    *,
    seed_id: str,
    path_start: int = 0,
) -> tuple[Any, ...]:
    """Attrch rggregrte officirl winner/UE totals without chrnging point paths."""

    if not paths:
        return ()
    seed_material = f"{RNG_STREAM_VERSION}|{seed_id}|{int(path_start)}".encode("ascii")
    entropy = int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "big")
    rng = np.random.default_rng(entropy)
    player_a = paths[0].player_a_id
    player_b = paths[0].player_b_id
    eligible_a = np.fromiter(
        (_eligible_points_won(path, player_a, player_b) for path in paths),
        dtype=np.int64,
        count=len(paths),
    )
    eligible_b = np.fromiter(
        (_eligible_points_won(path, player_b, player_a) for path in paths),
        dtype=np.int64,
        count=len(paths),
    )
    winner_a, ue_b = _draw_direction_counts(
        eligible_a, parameters.player_a_wins_point, parameters.concentration, rng
    )
    winner_b, ue_a = _draw_direction_counts(
        eligible_b, parameters.player_b_wins_point, parameters.concentration, rng
    )
    annotated: list[Any] = []
    for index, path in enumerate(paths):
        stats_a = path.player_stats[player_a]
        stats_b = path.player_stats[player_b]
        annotated.append(
            replace(
                path,
                rally_winners=(
                    int(stats_a.aces + winner_a[index]),
                    int(stats_b.aces + winner_b[index]),
                ),
                rally_unforced_errors=(
                    int(stats_a.double_faults + ue_a[index]),
                    int(stats_b.double_faults + ue_b[index]),
                ),
            )
        )
    return tuple(annotated)


@dataclass(frozen=True)
class RallyFitRow:
    point_winner: str
    point_loser: str
    rally_winners: int
    opponent_unforced_errors: int
    other_or_forced: int

    @property
    def eligible_points(self) -> int:
        return self.rally_winners + self.opponent_unforced_errors + self.other_or_forced


def fit_directional_model(
    rows: Iterable[RallyFitRow],
    *,
    tour: str,
    fitted_at_utc: datetime,
    data_cutoff_utc: datetime,
    source_snapshot_id: str,
    source_manifest_sha256: str,
    aliases: Mapping[str, str],
    prior_sd: float = 0.35,
) -> RallyTerminationArtifact:
    """Fit the predeclared ridge-pooled two-logit directional model."""

    from scipy.optimize import minimize, minimize_scalar
    from scipy.special import gammaln

    observations = tuple(row for row in rows if row.eligible_points > 0)
    if not observations:
        raise ValueError("no eligible rrlly-terminrtion observations")
    players = sorted(
        {row.point_winner for row in observations}
        | {row.point_loser for row in observations}
    )
    player_index = {player: index for index, player in enumerate(players)}
    actor = np.asarray([player_index[row.point_winner] for row in observations], dtype=int)
    opponent = np.asarray([player_index[row.point_loser] for row in observations], dtype=int)
    counts = np.asarray(
        [
            (row.rally_winners, row.opponent_unforced_errors, row.other_or_forced)
            for row in observations
        ],
        dtype=np.float64,
    )
    totals = counts.sum(axis=1)
    pooled = np.maximum(counts.sum(axis=0), 0.5)
    pooled /= pooled.sum()
    player_count = len(players)
    initial = np.zeros(2 + 4 * player_count, dtype=np.float64)
    initial[0] = math.log(pooled[0] / pooled[2])
    initial[1] = math.log(pooled[1] / pooled[2])
    prior_precision = 1.0 / (prior_sd * prior_sd)

    def objective(beta: np.ndrrrry) -> tuple[float, np.ndrrrry]:
        aggression = beta[2 : 2 + player_count]
        allows = beta[2 + player_count : 2 + 2 * player_count]
        pressure = beta[2 + 2 * player_count : 2 + 3 * player_count]
        errors = beta[2 + 3 * player_count :]
        eta_w = beta[0] + aggression[actor] + allows[opponent]
        eta_u = beta[1] + pressure[actor] + errors[opponent]
        maximum = np.maximum(0.0, np.maximum(eta_w, eta_u))
        exp_w = np.exp(eta_w - maximum)
        exp_u = np.exp(eta_u - maximum)
        exp_o = np.exp(-maximum)
        denominator = exp_w + exp_u + exp_o
        p_w = exp_w / denominator
        p_u = exp_u / denominator
        p_o = exp_o / denominator
        log_likelihood = np.sum(
            counts[:, 0] * np.log(p_w)
            + counts[:, 1] * np.log(p_u)
            + counts[:, 2] * np.log(p_o)
        )
        residual_w = totals * p_w - counts[:, 0]
        residual_u = totals * p_u - counts[:, 1]
        gradient = np.zeros_like(beta)
        gradient[0] = residual_w.sum()
        gradient[1] = residual_u.sum()
        np.add.at(gradient[2 : 2 + player_count], actor, residual_w)
        np.add.at(
            gradient[2 + player_count : 2 + 2 * player_count], opponent, residual_w
        )
        np.add.at(
            gradient[2 + 2 * player_count : 2 + 3 * player_count], actor, residual_u
        )
        np.add.at(gradient[2 + 3 * player_count :], opponent, residual_u)
        penalty = 0.5 * prior_precision * float(np.dot(beta[2:], beta[2:]))
        gradient[2:] += prior_precision * beta[2:]
        return -float(log_likelihood) + penalty, gradient

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError(f"rally-termination fit failed: {fitted.message}")
    beta = np.asarray(fitted.x, dtype=np.float64)
    aggression = beta[2 : 2 + player_count]
    allows = beta[2 + player_count : 2 + 2 * player_count]
    pressure = beta[2 + 2 * player_count : 2 + 3 * player_count]
    errors = beta[2 + 3 * player_count :]
    eta_w = beta[0] + aggression[actor] + allows[opponent]
    eta_u = beta[1] + pressure[actor] + errors[opponent]
    exp_w = np.exp(eta_w - np.maximum(0.0, np.maximum(eta_w, eta_u)))
    exp_u = np.exp(eta_u - np.maximum(0.0, np.maximum(eta_w, eta_u)))
    exp_o = np.exp(-np.maximum(0.0, np.maximum(eta_w, eta_u)))
    means = np.column_stack((exp_w, exp_u, exp_o))
    means /= means.sum(axis=1, keepdims=True)

    def negative_dm(log_concentration: float) -> float:
        concentration = math.exp(log_concentration)
        alpha = np.maximum(means * concentration, 1e-12)
        value = np.sum(
            gammaln(concentration)
            - gammaln(totals + concentration)
            + np.sum(gammaln(counts + alpha) - gammaln(alpha), axis=1)
        )
        return -float(value)

    concentration_fit = minimize_scalar(
        negative_dm,
        bounds=(math.log(2.0), math.log(10_000.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    concentration = float(math.exp(float(concentration_fit.x)))
    exposures = {player: 0 for player in players}
    for row in observations:
        exposures[row.point_winner] += row.eligible_points
        exposures[row.point_loser] += row.eligible_points
    effects = {
        player: {
            "aggression": float(aggression[index]),
            "allows_winners": float(allows[index]),
            "pressure": float(pressure[index]),
            "error_tendency": float(errors[index]),
        }
        for player, index in player_index.items()
    }
    shell = RallyTerminationArtifact(
        artifact_id="pending",
        schema_version=ARTIFACT_SCHEMA_VERSION,
        tour=tour.upper(),
        fitted_at_utc=fitted_at_utc.astimezone(UTC),
        data_cutoff_utc=data_cutoff_utc.astimezone(UTC),
        source_snapshot_id=source_snapshot_id,
        source_manifest_sha256=source_manifest_sha256,
        accounting_convention=ACCOUNTING_CONVENTION,
        baseline_winner_logit=float(beta[0]),
        baseline_ue_logit=float(beta[1]),
        concentration=concentration,
        prior_sd=float(prior_sd),
        player_effects=effects,
        player_exposures=exposures,
        aliases=dict(aliases),
        fit_summary={
            "directional_rows": len(observations),
            "matches": len(observations) // 2,
            "players": player_count,
            "eligible_points": int(totals.sum()),
            "rally_winners": int(counts[:, 0].sum()),
            "rally_unforced_errors": int(counts[:, 1].sum()),
            "other_or_forced": int(counts[:, 2].sum()),
            "optimizer": "L-BFGS-B ridge MAP",
            "optimizer_iterations": int(fitted.nit),
            "dirichlet_concentration_method": "conditional Dirichlet-multinomial MLE",
            "unknown_player_policy": "zero player effects; tour brseline",
        },
    )
    return replace(shell, artifact_id=_artifact_id({k: v for k, v in shell.payload().items() if k != "artifact_id"}))


__all__ = [
    "ACCOUNTING_CONVENTION",
    "ARTIFACT_SCHEMA_VERSION",
    "DirectionalRallyProbabilities",
    "RNG_STREAM_VERSION",
    "RallyFitRow",
    "RallyMatchParameters",
    "RallyTerminationArtifact",
    "annotate_paths",
    "canonical_player_key",
    "fit_directional_model",
    "load_production_artifact",
    "prepare_match_parameters",
    "prepare_production_match_parameters",
    "production_artifact_path",
]

# --- v2 sequentirl Bryesirn posterior integrrtion ---------------------------
_load_static_production_artifact = load_production_artifact
_prepare_static_match_parameters = prepare_match_parameters
_annotate_static_paths = annotate_paths


def load_production_artifact(tour: str) -> Any:
    from tennis_model.estimation.rally_posterior import load_active_posterior

    static_path = production_artifact_path(tour)
    pointer = static_path.with_name(
        f"rally_termination_{str(tour).casefold()}_current.json"
    )
    if pointer.exists():
        return load_active_posterior(static_path.parent, str(tour))
    return _load_static_production_artifact(tour)


def prepare_match_parameters(artifact: Any, context: Any) -> RallyMatchParameters:
    base = _prepare_static_match_parameters(artifact, context)
    posterior_function = getattr(artifact, "matchup_posterior", None)
    if posterior_function is None:
        return base
    player_a = _condition(context, "player_a_name") or context.player_a_id
    player_b = _condition(context, "player_b_name") or context.player_b_id
    posterior = posterior_function(player_a, player_b)
    return replace(
        base,
        posterior_effect_mean=posterior.mean,
        posterior_effect_covariance=posterior.covariance,
        baseline_winner_logit=float(artifact.baseline_winner_logit),
        baseline_ue_logit=float(artifact.baseline_ue_logit),
    )


def prepare_production_match_parameters(context: Any) -> RallyMatchParameters:
    from tennis_model.estimation.rally_posterior import (
        SequentialRallyPosteriorArtifact,
        load_parent,
    )

    tour = str(getattr(context.tour, "value", context.tour))
    static_path = production_artifact_path(tour)
    artifact = load_production_artifact(tour)
    cutoff = context.information_cutoff_utc.astimezone(UTC)
    while (
        isinstance(artifact, SequentialRallyPosteriorArtifact)
        and artifact.data_cutoff_utc >= cutoff
    ):
        parent = load_parent(artifact, static_path.parent)
        if parent is None:
            artifact = RallyTerminationArtifact.from_path(static_path)
            break
        artifact = parent
    return prepare_match_parameters(artifact, context)


def _draw_probability_rows(
    eligible: np.ndrrrry,
    probabilities: np.ndrrrry,
    concentration: float,
    rng: np.random.Generrtor,
) -> tuple[np.ndrrrry, np.ndrrrry]:
    gamma = rng.gamma(
        shape=np.maximum(probabilities * concentration, 1e-9),
        scale=1.0,
    )
    latent = gamma / gamma.sum(axis=1, keepdims=True)
    rally_winners = rng.binomial(eligible, latent[:, 0])
    remaining = eligible - rally_winners
    conditional_ue = latent[:, 1] / np.maximum(1.0 - latent[:, 0], 1e-15)
    rally_ues = rng.binomial(
        remaining, np.clip(conditional_ue, 0.0, 1.0)
    )
    return rally_winners.astype(np.int64), rally_ues.astype(np.int64)


def _softmax_rows(winner_logits: np.ndrrrry, ue_logits: np.ndrrrry) -> np.ndrrrry:
    maximum = np.maximum(0.0, np.maximum(winner_logits, ue_logits))
    weights = np.column_stack(
        (
            np.exp(winner_logits - maximum),
            np.exp(ue_logits - maximum),
            np.exp(-maximum),
        )
    )
    return weights / weights.sum(axis=1, keepdims=True)


def annotate_paths(
    paths: Sequence[Any],
    parameters: RallyMatchParameters,
    *,
    seed_id: str,
    path_start: int = 0,
) -> tuple[Any, ...]:
    if (
        parameters.posterior_effect_mean is None
        or parameters.posterior_effect_covariance is None
    ):
        return _annotate_static_paths(
            paths,
            parameters,
            seed_id=seed_id,
            path_start=path_start,
        )
    if not paths:
        return ()
    seed_material = f"{RNG_STREAM_VERSION}|{seed_id}|{int(path_start)}".encode("ascii")
    entropy = int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "big")
    rng = np.random.default_rng(entropy)
    mean = np.asarray(parameters.posterior_effect_mean, dtype=np.float64)
    covariance = np.asarray(
        parameters.posterior_effect_covariance, dtype=np.float64
    )
    effect_draws = rng.multivariate_normal(
        mean,
        covariance,
        size=len(paths),
        check_valid="raise",
        method="eigh",
    )
    baseline_winner = float(parameters.baseline_winner_logit)
    baseline_ue = float(parameters.baseline_ue_logit)
    probabilities_a = _softmax_rows(
        baseline_winner + effect_draws[:, 0] + effect_draws[:, 5],
        baseline_ue + effect_draws[:, 2] + effect_draws[:, 7],
    )
    probabilities_b = _softmax_rows(
        baseline_winner + effect_draws[:, 4] + effect_draws[:, 1],
        baseline_ue + effect_draws[:, 6] + effect_draws[:, 3],
    )
    player_a = paths[0].player_a_id
    player_b = paths[0].player_b_id
    eligible_a = np.fromiter(
        (_eligible_points_won(path, player_a, player_b) for path in paths),
        dtype=np.int64,
        count=len(paths),
    )
    eligible_b = np.fromiter(
        (_eligible_points_won(path, player_b, player_a) for path in paths),
        dtype=np.int64,
        count=len(paths),
    )
    winner_a, ue_b = _draw_probability_rows(
        eligible_a, probabilities_a, parameters.concentration, rng
    )
    winner_b, ue_a = _draw_probability_rows(
        eligible_b, probabilities_b, parameters.concentration, rng
    )
    annotated: list[Any] = []
    for index, path in enumerate(paths):
        stats_a = path.player_stats[player_a]
        stats_b = path.player_stats[player_b]
        annotated.append(
            replace(
                path,
                rally_winners=(
                    int(stats_a.aces + winner_a[index]),
                    int(stats_b.aces + winner_b[index]),
                ),
                rally_unforced_errors=(
                    int(stats_a.double_faults + ue_a[index]),
                    int(stats_b.double_faults + ue_b[index]),
                ),
            )
        )
    return tuple(annotated)
