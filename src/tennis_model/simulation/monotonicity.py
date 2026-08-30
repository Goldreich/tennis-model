"""Auditable pathwise and explicitly distributional monotonicity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from tennis_model.simulation.match import SimulationBatch, simulate_matches
from tennis_model.simulation.parameters import MatchParameterDistribution
from tennis_model.simulation.point import (
    PointUniforms,
    ServePerformanceDraw,
    generate_point_from_uniforms,
)


@dataclass(frozen=True, slots=True)
class MonotonicityDiagnostic:
    target: Literal[
        "ACE_PROPENSITY",
        "DOUBLE_FAULT_PROPENSITY",
        "SERVICE_POINT_WIN",
        "HOLD_PROBABILITY",
        "MATCH_WIN_PROBABILITY",
    ]
    evidence: Literal["PATHWISE_PREALLOCATED_UNIFORMS", "ANALYTIC", "DISTRIBUTIONAL"]
    direction: Literal["NONDECREASING", "NONINCREASING"]
    low_value: float
    high_value: float
    violations: int
    tolerance: float
    passed: bool
    seed_ids: tuple[str, ...]


def analytic_hold_probability(point_probability: float) -> float:
    if not 0.0 <= point_probability <= 1.0:
        raise ValueError("service-point probability must lie in [0, 1]")
    p = point_probability
    q = 1.0 - p
    denominator = p**2 + q**2
    deuce = 0.0 if denominator == 0 else 20.0 * p**3 * q**3 * p**2 / denominator
    return p**4 * (1.0 + 4.0 * q + 10.0 * q**2) + deuce


def coupled_primitive_monotonicity(
    performance: ServePerformanceDraw,
    *,
    target: Literal["ace", "double_fault"],
    low_probability: float,
    high_probability: float,
    n_draws: int,
    seed: int,
) -> tuple[MonotonicityDiagnostic, MonotonicityDiagnostic, MonotonicityDiagnostic]:
    """Use the same five preallocated uniforms for both primitive settings."""

    if not 0 <= low_probability <= high_probability <= 1 or n_draws <= 0:
        raise ValueError("coupled monotonicity inputs are invalid")
    if target == "ace":
        low = replace(performance, ace_given_first_in=low_probability)
        high = replace(performance, ace_given_first_in=high_probability)
        direction: Literal["NONDECREASING", "NONINCREASING"] = "NONDECREASING"
        target_name: Literal["ACE_PROPENSITY", "DOUBLE_FAULT_PROPENSITY"] = "ACE_PROPENSITY"
    else:
        low = replace(performance, double_fault_given_second_opp=low_probability)
        high = replace(performance, double_fault_given_second_opp=high_probability)
        direction = "NONINCREASING"
        target_name = "DOUBLE_FAULT_PROPENSITY"
    uniforms = np.random.default_rng(seed).random((n_draws, 5))
    low_wins = np.empty(n_draws, dtype=np.int8)
    high_wins = np.empty(n_draws, dtype=np.int8)
    for index, values in enumerate(uniforms):
        latent = PointUniforms(*map(float, values))
        low_wins[index] = generate_point_from_uniforms(low, latent).server_won
        high_wins[index] = generate_point_from_uniforms(high, latent).server_won
    violations = int(
        np.count_nonzero(high_wins < low_wins)
        if direction == "NONDECREASING"
        else np.count_nonzero(high_wins > low_wins)
    )
    low_service = low.service_point_win
    high_service = high.service_point_win
    service_passed = (
        high_service >= low_service if direction == "NONDECREASING" else high_service <= low_service
    )
    low_hold = analytic_hold_probability(low_service)
    high_hold = analytic_hold_probability(high_service)
    hold_passed = high_hold >= low_hold if direction == "NONDECREASING" else high_hold <= low_hold
    seed_ids = (f"preallocated-point-uniforms:{seed}",)
    return (
        MonotonicityDiagnostic(
            target=target_name,
            evidence="PATHWISE_PREALLOCATED_UNIFORMS",
            direction=direction,
            low_value=float(np.mean(low_wins)),
            high_value=float(np.mean(high_wins)),
            violations=violations,
            tolerance=0.0,
            passed=violations == 0,
            seed_ids=seed_ids,
        ),
        MonotonicityDiagnostic(
            target="SERVICE_POINT_WIN",
            evidence="ANALYTIC",
            direction=direction,
            low_value=low_service,
            high_value=high_service,
            violations=int(not service_passed),
            tolerance=0.0,
            passed=service_passed,
            seed_ids=(),
        ),
        MonotonicityDiagnostic(
            target="HOLD_PROBABILITY",
            evidence="ANALYTIC",
            direction=direction,
            low_value=low_hold,
            high_value=high_hold,
            violations=int(not hold_passed),
            tolerance=0.0,
            passed=hold_passed,
            seed_ids=(),
        ),
    )


def _match_win(batch: SimulationBatch, player_id: str) -> tuple[float, int]:
    paths = tuple(path for path in batch.paths if path.started and not path.walkover)
    if not paths:
        raise ValueError("match-win diagnostic requires started paths")
    return sum(path.advancing_player_id == player_id for path in paths) / len(paths), len(paths)


def distributional_match_win_monotonicity(
    low: MatchParameterDistribution,
    high: MatchParameterDistribution,
    *,
    player_id: str,
    direction: Literal["NONDECREASING", "NONINCREASING"],
    n_paths: int,
    low_seed: int,
    high_seed: int,
    absolute_tolerance: float = 1e-3,
) -> MonotonicityDiagnostic:
    """Use independent simulations and label the evidence as distributional."""

    if low_seed == high_seed:
        raise ValueError("distributional comparison requires independent seeds")
    low_batch = simulate_matches(low, n_paths=n_paths, seed=low_seed)
    high_batch = simulate_matches(high, n_paths=n_paths, seed=high_seed)
    low_probability, low_n = _match_win(low_batch, player_id)
    high_probability, high_n = _match_win(high_batch, player_id)
    standard_error = np.sqrt(
        low_probability * (1.0 - low_probability) / low_n
        + high_probability * (1.0 - high_probability) / high_n
    )
    tolerance = float(5.0 * standard_error + absolute_tolerance)
    passed = (
        high_probability + tolerance >= low_probability
        if direction == "NONDECREASING"
        else high_probability <= low_probability + tolerance
    )
    return MonotonicityDiagnostic(
        target="MATCH_WIN_PROBABILITY",
        evidence="DISTRIBUTIONAL",
        direction=direction,
        low_value=low_probability,
        high_value=high_probability,
        violations=int(not passed),
        tolerance=tolerance,
        passed=passed,
        seed_ids=(f"independent:{low_seed}", f"independent:{high_seed}"),
    )


__all__ = [
    "MonotonicityDiagnostic",
    "analytic_hold_probability",
    "coupled_primitive_monotonicity",
    "distributional_match_win_monotonicity",
]
