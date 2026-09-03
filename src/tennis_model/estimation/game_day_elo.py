"""Temporary fitness adjustments for the v1.2 game-day Elo candidate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import log
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


FEATURE_NAMES = ("recent_workload", "short_recovery", "return_from_layoff")
ELO_TO_LOGIT = log(10.0) / 400.0


@dataclass(frozen=True, slots=True)
class FitnessFeatures:
    recent_workload: float
    short_recovery: float
    return_from_layoff: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (self.recent_workload, self.short_recovery, self.return_from_layoff),
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class GameDayEloFit:
    feature_names: tuple[str, ...]
    weights_elo: tuple[float, ...]
    l2_penalty: float | None
    converged: bool
    iterations: int
    training_rows: int

    def weights_array(self) -> np.ndarray:
        return np.asarray(self.weights_elo, dtype=float)


def game_day_adjustment(
    features: FitnessFeatures | Sequence[float],
    weights_elo: Sequence[float],
) -> float:
    values = features.as_array() if isinstance(features, FitnessFeatures) else np.asarray(features, dtype=float)
    weights = np.asarray(weights_elo, dtype=float)
    if values.shape != (len(FEATURE_NAMES),) or weights.shape != values.shape:
        raise ValueError("game-day Elo features and weights have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("game-day Elo features and weights must be finite")
    if (values < 0.0).any():
        raise ValueError("game-day Elo fitness features must be nonnegative")
    if (weights > 1e-12).any():
        raise ValueError("game-day Elo fitness weights must be nonpositive")
    return float(values @ weights)


def apply_game_day_elo(
    base_elo: float,
    features: FitnessFeatures | Sequence[float],
    weights_elo: Sequence[float],
) -> float:
    return float(base_elo + game_day_adjustment(features, weights_elo))


def probabilities(base_logits: np.ndarray, feature_differences: np.ndarray, weights_elo: Sequence[float]) -> np.ndarray:
    offsets = np.asarray(base_logits, dtype=float)
    features = np.asarray(feature_differences, dtype=float)
    weights = np.asarray(weights_elo, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature_differences must have one column per fitness feature")
    if offsets.shape != (features.shape[0],) or weights.shape != (features.shape[1],):
        raise ValueError("game-day Elo prediction arrays have incompatible shapes")
    return expit(offsets + ELO_TO_LOGIT * (features @ weights))


def fit_game_day_elo(
    base_logits: np.ndarray,
    feature_differences: np.ndarray,
    outcomes: np.ndarray,
    *,
    l2_penalty: float,
    maximum_absolute_weight_elo: float,
) -> GameDayEloFit:
    offsets = np.asarray(base_logits, dtype=float)
    features = np.asarray(feature_differences, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature_differences must have one column per fitness feature")
    if offsets.shape != observed.shape or offsets.shape != (features.shape[0],):
        raise ValueError("fitness fit arrays have incompatible shapes")
    if len(observed) == 0:
        raise ValueError("fitness fit requires at least one observation")
    if not np.isfinite(offsets).all() or not np.isfinite(features).all() or not np.isfinite(observed).all():
        raise ValueError("fitness fit arrays must be finite")
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("fitness outcomes must be binary")
    if l2_penalty < 0.0 or maximum_absolute_weight_elo <= 0.0:
        raise ValueError("fitness fit penalties and bounds are invalid")

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        linear = offsets + ELO_TO_LOGIT * (features @ weights)
        loss = float(np.mean(np.logaddexp(0.0, linear) - observed * linear))
        penalty = 0.5 * l2_penalty * float(np.sum(np.square(weights / 100.0)))
        residual = expit(linear) - observed
        gradient = ELO_TO_LOGIT * np.mean(residual[:, None] * features, axis=0)
        gradient += l2_penalty * weights / 10000.0
        return loss + penalty, gradient

    result = minimize(
        lambda value: objective(value)[0],
        x0=np.full(len(FEATURE_NAMES), -1.0),
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
        bounds=[(-maximum_absolute_weight_elo, 0.0)] * len(FEATURE_NAMES),
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 2000},
    )
    weights = np.minimum(np.asarray(result.x, dtype=float), 0.0)
    return GameDayEloFit(
        feature_names=FEATURE_NAMES,
        weights_elo=tuple(float(value) for value in weights),
        l2_penalty=float(l2_penalty),
        converged=bool(result.success),
        iterations=int(result.nit),
        training_rows=len(observed),
    )


def zero_adjustment_fit(training_rows: int) -> GameDayEloFit:
    return GameDayEloFit(
        feature_names=FEATURE_NAMES,
        weights_elo=(0.0,) * len(FEATURE_NAMES),
        l2_penalty=None,
        converged=True,
        iterations=0,
        training_rows=training_rows,
    )


def load_game_day_elo_fit(path: str | Path, *, tour: str) -> GameDayEloFit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "game-day-elo-fit/v1":
        raise ValueError("unsupported game-day Elo artifact schema")
    if payload.get("framework_version") != "v1.1":
        raise ValueError("game-day Elo artifact is not promoted for production v1.1")
    try:
        fit = payload["tour_fits"][tour.upper()]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"game-day Elo artifact has no {tour.upper()} fit") from exc
    feature_names = tuple(str(value) for value in fit["feature_names"])
    weights = tuple(float(value) for value in fit["weights_elo"])
    if feature_names != FEATURE_NAMES or len(weights) != len(FEATURE_NAMES):
        raise ValueError("game-day Elo artifact has incompatible fitness features")
    if any(not np.isfinite(value) or value > 1e-12 for value in weights):
        raise ValueError("game-day Elo artifact contains an invalid fitness weight")
    return GameDayEloFit(
        feature_names=feature_names,
        weights_elo=weights,
        l2_penalty=float(fit["l2_penalty"]),
        converged=True,
        iterations=0,
        training_rows=int(fit["training_rows"]),
    )


__all__ = [
    "ELO_TO_LOGIT",
    "FEATURE_NAMES",
    "FitnessFeatures",
    "GameDayEloFit",
    "apply_game_day_elo",
    "fit_game_day_elo",
    "game_day_adjustment",
    "load_game_day_elo_fit",
    "probabilities",
    "zero_adjustment_fit",
]
