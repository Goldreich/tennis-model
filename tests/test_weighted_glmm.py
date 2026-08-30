from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import expit

from tennis_model.estimation.weighted_glmm import (
    CurvatureError,
    LikelihoodDomainError,
    beta_binomial_logpmf,
    beta_binomial_scores,
    laplace_curvature,
    time_weight,
)


def _reference_logpmf(y: int, n: int, mean: float, kappa: float) -> float:
    alpha = kappa * mean
    beta = kappa * (1.0 - mean)
    return (
        math.lgamma(n + 1)
        - math.lgamma(y + 1)
        - math.lgamma(n - y + 1)
        + math.lgamma(y + alpha)
        + math.lgamma(n - y + beta)
        - math.lgamma(n + kappa)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(kappa)
    )


@pytest.mark.parametrize(
    ("y", "n", "mean", "kappa"),
    [(0, 5, 0.2, 3.0), (3, 10, 0.4, 25.0), (10, 10, 0.8, 100.0)],
)
def test_beta_binomial_logpmf_matches_independent_lgamma_reference(
    y: int, n: int, mean: float, kappa: float
) -> None:
    observed = float(beta_binomial_logpmf(y, n, mean, kappa))
    assert observed == pytest.approx(_reference_logpmf(y, n, mean, kappa), abs=1e-12)


def test_large_kappa_approaches_binomial_logpmf() -> None:
    y, n, probability = 37, 60, 0.61
    beta_binomial = float(beta_binomial_logpmf(y, n, probability, 1.0e8))
    binomial = (
        math.lgamma(n + 1)
        - math.lgamma(y + 1)
        - math.lgamma(n - y + 1)
        + y * math.log(probability)
        + (n - y) * math.log1p(-probability)
    )
    assert beta_binomial == pytest.approx(binomial, abs=2e-6)


@pytest.mark.parametrize(
    ("successes", "trials", "mean", "kappa"),
    [
        (-1, 5, 0.5, 10.0),
        (6, 5, 0.5, 10.0),
        (1.5, 5, 0.5, 10.0),
        (1, 5, 0.0, 10.0),
        (1, 5, 1.0, 10.0),
        (1, 5, float("nan"), 10.0),
        (1, 5, 0.5, 0.0),
        (1, 5, 0.5, float("inf")),
    ],
)
def test_invalid_beta_binomial_inputs_fail_explicitly(
    successes: float, trials: int, mean: float, kappa: float
) -> None:
    with pytest.raises(LikelihoodDomainError):
        beta_binomial_logpmf(successes, trials, mean, kappa)


def test_beta_binomial_analytic_scores_match_finite_differences() -> None:
    y = np.asarray([2.0, 17.0, 39.0])
    n = np.asarray([10.0, 30.0, 50.0])
    eta = np.asarray([-0.8, 0.2, 1.1])
    log_kappa = math.log(22.0)
    _logpmf, eta_score, log_kappa_score = beta_binomial_scores(y, n, eta, math.exp(log_kappa))
    step = 1e-6
    for index in range(len(eta)):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (
            float(beta_binomial_logpmf(y[index], n[index], expit(plus[index]), 22.0))
            - float(beta_binomial_logpmf(y[index], n[index], expit(minus[index]), 22.0))
        ) / (2.0 * step)
        assert eta_score[index] == pytest.approx(numeric, rel=2e-6, abs=2e-7)

    plus_kappa = math.exp(log_kappa + step)
    minus_kappa = math.exp(log_kappa - step)
    numeric_kappa = (
        beta_binomial_logpmf(y, n, expit(eta), plus_kappa)
        - beta_binomial_logpmf(y, n, expit(eta), minus_kappa)
    ) / (2.0 * step)
    assert log_kappa_score == pytest.approx(numeric_kappa, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize(
    ("age", "expected"),
    [(0.0, 1.0), (182.5, 2**-0.5), (365.0, 0.5), (1095.0, 0.125), (1095.01, 0.0)],
)
def test_frozen_time_weight(age: float, expected: float) -> None:
    assert time_weight(age) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("age", [-1.0, float("nan"), float("inf")])
def test_time_weight_rejects_invalid_age(age: float) -> None:
    with pytest.raises(ValueError):
        time_weight(age)


def test_laplace_curvature_surfaces_and_regularizes_indefinite_hessian() -> None:
    def gradient(point: np.ndarray) -> np.ndarray:
        return np.asarray([-point[0], 2.0 * point[1]])

    result = laplace_curvature(
        gradient,
        np.asarray([0.2, -0.1]),
        max_full_parameters=5,
        relative_step=1e-5,
        eigenvalue_floor=1e-4,
    )
    assert result.kind == "full"
    assert result.raw_min_eigenvalue == pytest.approx(-1.0, abs=1e-9)
    assert result.regularization_added > 1.0
    assert result.covariance is not None
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)


def test_large_parameter_curvature_uses_auditable_diagonal_approximation() -> None:
    diagonal = np.asarray([1.0, 2.0, 4.0])

    def gradient(point: np.ndarray) -> np.ndarray:
        return diagonal * point

    result = laplace_curvature(
        gradient,
        np.asarray([0.1, 0.2, 0.3]),
        max_full_parameters=2,
        relative_step=1e-5,
        eigenvalue_floor=1e-6,
    )
    assert result.kind == "diagonal"
    assert result.hessian is None
    assert result.covariance is None
    assert result.variance_diagonal == pytest.approx(1.0 / diagonal)


def test_nonfinite_curvature_is_never_silently_serialized() -> None:
    def gradient(_point: np.ndarray) -> np.ndarray:
        return np.asarray([float("nan")])

    with pytest.raises(CurvatureError, match="nonfinite"):
        laplace_curvature(
            gradient,
            np.asarray([0.0]),
            max_full_parameters=2,
            relative_step=1e-5,
            eigenvalue_floor=1e-6,
        )
