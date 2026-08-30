"""Numerical foundations for the frozen time-weighted beta-binomial models.

This module contains no tennis-component routing.  It provides the common
beta-binomial likelihood, frozen recency rule, and deterministic Laplace
curvature calculation used by all five primitive serve components.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaln, digamma, expit, gammaln  # type: ignore[import-untyped]

FROZEN_HALF_LIFE_DAYS = 365.0
FROZEN_WINDOW_DAYS = 1095.0

FloatArray = NDArray[np.float64]


class LikelihoodDomainError(ValueError):
    """A beta-binomial argument lies outside its mathematical support."""


class CurvatureError(RuntimeError):
    """Posterior curvature could not be converted to a finite covariance."""


def _broadcast_inputs(
    successes: ArrayLike,
    trials: ArrayLike,
    mean: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y, n, mu = np.broadcast_arrays(
        np.asarray(successes, dtype=np.float64),
        np.asarray(trials, dtype=np.float64),
        np.asarray(mean, dtype=np.float64),
    )
    return y, n, mu


def _validate_beta_binomial_inputs(
    successes: FloatArray,
    trials: FloatArray,
    mean: FloatArray,
    concentration: float,
) -> None:
    if not np.all(np.isfinite(successes)) or not np.all(np.isfinite(trials)):
        raise LikelihoodDomainError("successes and trials must be finite")
    if not np.all(successes == np.floor(successes)) or not np.all(trials == np.floor(trials)):
        raise LikelihoodDomainError("successes and trials must be exact integers")
    if np.any(trials < 0) or np.any(successes < 0) or np.any(successes > trials):
        raise LikelihoodDomainError("counts must satisfy 0 <= successes <= trials")
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0.0) or np.any(mean >= 1.0):
        raise LikelihoodDomainError(
            "mean probabilities must be finite and strictly between 0 and 1"
        )
    if not isinstance(concentration, (int, float)) or isinstance(concentration, bool):
        raise LikelihoodDomainError("concentration must be a finite positive number")
    if not isfinite(float(concentration)) or concentration <= 0.0:
        raise LikelihoodDomainError("concentration must be a finite positive number")


def beta_binomial_logpmf(
    successes: ArrayLike,
    trials: ArrayLike,
    mean: ArrayLike,
    concentration: float,
) -> FloatArray:
    """Return the beta-binomial log PMF under ``alpha=kappa*mu``.

    The result follows NumPy broadcasting.  Inputs are validated explicitly;
    invalid count or probability data never produce a quiet NaN objective.
    """

    y, n, mu = _broadcast_inputs(successes, trials, mean)
    kappa = float(concentration)
    _validate_beta_binomial_inputs(y, n, mu, kappa)
    alpha = kappa * mu
    beta = kappa * (1.0 - mu)
    result = (
        gammaln(n + 1.0)
        - gammaln(y + 1.0)
        - gammaln(n - y + 1.0)
        + betaln(y + alpha, n - y + beta)
        - betaln(alpha, beta)
    )
    return np.asarray(result, dtype=np.float64)


def beta_binomial_scores(
    successes: FloatArray,
    trials: FloatArray,
    linear_predictor: FloatArray,
    concentration: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return log PMFs and derivatives with respect to eta and log(kappa)."""

    mu = np.asarray(expit(linear_predictor), dtype=np.float64)
    y, n, mu = _broadcast_inputs(successes, trials, mu)
    kappa = float(concentration)
    _validate_beta_binomial_inputs(y, n, mu, kappa)
    alpha = kappa * mu
    beta = kappa * (1.0 - mu)
    logpmf = beta_binomial_logpmf(y, n, mu, kappa)

    alpha_increment = digamma(y + alpha) - digamma(alpha)
    beta_increment = digamma(n - y + beta) - digamma(beta)
    d_logpmf_d_mu = kappa * (alpha_increment - beta_increment)
    d_logpmf_d_eta = d_logpmf_d_mu * mu * (1.0 - mu)

    d_logpmf_d_kappa = (
        mu * alpha_increment + (1.0 - mu) * beta_increment - digamma(n + kappa) + digamma(kappa)
    )
    d_logpmf_d_log_kappa = kappa * d_logpmf_d_kappa
    return logpmf, d_logpmf_d_eta, np.asarray(d_logpmf_d_log_kappa, dtype=np.float64)


def time_weight(age_days: float) -> float:
    """Apply the frozen one-year half-life inside the inclusive 1,095-day window."""

    if isinstance(age_days, bool) or not isinstance(age_days, (int, float)):
        raise TypeError("age_days must be a finite nonnegative number")
    age = float(age_days)
    if not isfinite(age) or age < 0.0:
        raise ValueError("age_days must be a finite nonnegative number")
    if age > FROZEN_WINDOW_DAYS:
        return 0.0
    return float(2.0 ** (-age / FROZEN_HALF_LIFE_DAYS))


@dataclass(frozen=True, slots=True)
class CurvatureResult:
    """A full or diagonal regularized Laplace curvature representation."""

    kind: str
    hessian: FloatArray | None
    covariance: FloatArray | None
    variance_diagonal: FloatArray
    raw_min_eigenvalue: float
    regularization_added: float
    condition_number: float


def _finite_difference_column(
    gradient: Callable[[FloatArray], FloatArray],
    point: FloatArray,
    index: int,
    relative_step: float,
) -> FloatArray:
    step = relative_step * max(1.0, abs(float(point[index])))
    plus = point.copy()
    minus = point.copy()
    plus[index] += step
    minus[index] -= step
    return (gradient(plus) - gradient(minus)) / (2.0 * step)


def laplace_curvature(
    gradient: Callable[[FloatArray], FloatArray],
    map_parameters: FloatArray,
    *,
    max_full_parameters: int,
    relative_step: float,
    eigenvalue_floor: float,
) -> CurvatureResult:
    """Approximate observed posterior curvature deterministically.

    Small and medium fits receive a symmetrized full finite-difference Hessian.
    Larger fits receive the diagonal observed-curvature approximation so a
    production fit cannot accidentally allocate a quadratic dense matrix.
    """

    point = np.asarray(map_parameters, dtype=np.float64)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise CurvatureError("MAP parameters must be a finite one-dimensional vector")
    if max_full_parameters < 1:
        raise ValueError("max_full_parameters must be positive")
    if relative_step <= 0.0 or not isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    if eigenvalue_floor <= 0.0 or not isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be finite and positive")

    size = len(point)
    if size <= max_full_parameters:
        columns = [
            _finite_difference_column(gradient, point, index, relative_step)
            for index in range(size)
        ]
        raw = np.column_stack(columns)
        raw = (raw + raw.T) / 2.0
        if not np.all(np.isfinite(raw)):
            raise CurvatureError("full Hessian contains nonfinite values")
        eigenvalues = np.linalg.eigvalsh(raw)
        raw_min = float(eigenvalues[0])
        ridge = max(0.0, eigenvalue_floor - raw_min)
        regularized = raw + ridge * np.eye(size, dtype=np.float64)
        try:
            covariance = np.linalg.inv(regularized)
        except np.linalg.LinAlgError as exc:
            raise CurvatureError("regularized Hessian is singular") from exc
        covariance = (covariance + covariance.T) / 2.0
        if not np.all(np.isfinite(covariance)):
            raise CurvatureError("Laplace covariance contains nonfinite values")
        condition = float(np.linalg.cond(regularized))
        return CurvatureResult(
            kind="full",
            hessian=regularized,
            covariance=covariance,
            variance_diagonal=np.diag(covariance).copy(),
            raw_min_eigenvalue=raw_min,
            regularization_added=ridge,
            condition_number=condition,
        )

    diagonal = np.empty(size, dtype=np.float64)
    for index in range(size):
        column = _finite_difference_column(gradient, point, index, relative_step)
        diagonal[index] = column[index]
    if not np.all(np.isfinite(diagonal)):
        raise CurvatureError("diagonal Hessian contains nonfinite values")
    raw_min = float(np.min(diagonal))
    ridge = max(0.0, eigenvalue_floor - raw_min)
    regularized_diagonal = diagonal + ridge
    variance = 1.0 / regularized_diagonal
    condition = float(np.max(regularized_diagonal) / np.min(regularized_diagonal))
    return CurvatureResult(
        kind="diagonal",
        hessian=None,
        covariance=None,
        variance_diagonal=variance,
        raw_min_eigenvalue=raw_min,
        regularization_added=ridge,
        condition_number=condition,
    )


__all__ = [
    "FROZEN_HALF_LIFE_DAYS",
    "FROZEN_WINDOW_DAYS",
    "CurvatureError",
    "CurvatureResult",
    "LikelihoodDomainError",
    "beta_binomial_logpmf",
    "beta_binomial_scores",
    "laplace_curvature",
    "time_weight",
]
