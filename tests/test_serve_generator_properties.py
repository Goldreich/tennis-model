from __future__ import annotations

from dataclasses import replace
from math import nextafter, sqrt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tennis_model.simulation.point import (
    PointUniforms,
    ServePerformanceDraw,
    ServicePointBranch,
    aggregate_service_points,
    generate_point_from_uniforms,
    generate_service_point,
)

PROBABILITY = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
UNIFORM = st.floats(
    min_value=0.0,
    max_value=nextafter(1.0, 0.0),
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _performance_draws(draw: st.DrawFn) -> ServePerformanceDraw:
    return ServePerformanceDraw(
        first_serve_in=draw(PROBABILITY),
        ace_given_first_in=draw(PROBABILITY),
        returnable_first_win=draw(PROBABILITY),
        double_fault_given_second_opp=draw(PROBABILITY),
        playable_second_win=draw(PROBABILITY),
    )


@st.composite
def _point_uniforms(draw: st.DrawFn) -> PointUniforms:
    return PointUniforms(
        first_serve_in=draw(UNIFORM),
        ace_given_first_in=draw(UNIFORM),
        returnable_first_win=draw(UNIFORM),
        double_fault_given_second_opp=draw(UNIFORM),
        playable_second_win=draw(UNIFORM),
    )


@given(
    performance=_performance_draws(),
    uniforms=_point_uniforms(),
    ace_a=PROBABILITY,
    ace_b=PROBABILITY,
)
@settings(max_examples=300)
def test_increasing_ace_propensity_cannot_turn_a_coupled_server_win_into_loss(
    performance: ServePerformanceDraw,
    uniforms: PointUniforms,
    ace_a: float,
    ace_b: float,
) -> None:
    low, high = sorted((ace_a, ace_b))
    low_result = generate_point_from_uniforms(
        replace(performance, ace_given_first_in=low),
        uniforms,
    )
    high_result = generate_point_from_uniforms(
        replace(performance, ace_given_first_in=high),
        uniforms,
    )

    assert int(low_result.server_won) <= int(high_result.server_won)
    if low_result.first_serve_in and low <= uniforms.ace_given_first_in < high:
        assert high_result.branch is ServicePointBranch.ACE
        assert high_result.server_won


@given(
    performance=_performance_draws(),
    uniforms=_point_uniforms(),
    double_fault_a=PROBABILITY,
    double_fault_b=PROBABILITY,
)
@settings(max_examples=300)
def test_increasing_double_fault_propensity_cannot_create_a_coupled_server_win(
    performance: ServePerformanceDraw,
    uniforms: PointUniforms,
    double_fault_a: float,
    double_fault_b: float,
) -> None:
    low, high = sorted((double_fault_a, double_fault_b))
    low_result = generate_point_from_uniforms(
        replace(performance, double_fault_given_second_opp=low),
        uniforms,
    )
    high_result = generate_point_from_uniforms(
        replace(performance, double_fault_given_second_opp=high),
        uniforms,
    )

    assert int(low_result.server_won) >= int(high_result.server_won)
    if not low_result.first_serve_in and low <= uniforms.double_fault_given_second_opp < high:
        assert high_result.branch is ServicePointBranch.DOUBLE_FAULT
        assert not high_result.server_won


@given(
    cases=st.lists(
        st.tuples(_performance_draws(), _point_uniforms()),
        min_size=0,
        max_size=100,
    )
)
@settings(max_examples=150)
def test_every_coupled_point_and_aggregate_preserves_all_count_identities(
    cases: list[tuple[ServePerformanceDraw, PointUniforms]],
) -> None:
    points = tuple(
        generate_point_from_uniforms(performance, uniforms) for performance, uniforms in cases
    )

    for point in points:
        assert not (point.ace and point.double_fault)
        assert point.ace <= point.first_serve_in
        assert point.double_fault <= (not point.first_serve_in)
        assert point.ace <= point.server_won
        assert point.double_fault <= (not point.server_won)
        assert point.rally_eligible is (point.q1_used or point.q2_used)
        assert not (point.q1_used and point.q2_used)

    counts = aggregate_service_points(points)
    assert counts.service_points == len(points)
    assert counts.service_points == counts.first_serves_in + counts.second_serve_opportunities
    assert counts.first_serves_in == counts.aces + counts.q1_trials
    assert counts.first_serve_points_won == counts.aces + counts.q1_wins
    assert counts.second_serve_opportunities == counts.double_faults + counts.q2_trials
    assert counts.second_serve_points_won == counts.q2_wins


def _assert_binomial_mc_identity(
    *,
    successes: int,
    trials: int,
    expected_probability: float,
    label: str,
) -> None:
    assert trials > 0, f"{label} received no realized trials"
    observed = successes / trials
    standard_error = sqrt(expected_probability * (1.0 - expected_probability) / trials)
    tolerance = max(6.0 * standard_error, 2.0 / trials)
    assert observed == pytest.approx(expected_probability, abs=tolerance), (
        f"{label}: observed={observed:.6f}, expected={expected_probability:.6f}, "
        f"SE={standard_error:.6f}, n={trials}"
    )


@pytest.mark.parametrize(
    ("performance", "seed"),
    [
        (ServePerformanceDraw(0.62, 0.11, 0.67, 0.09, 0.56), 101),
        (ServePerformanceDraw(0.48, 0.26, 0.42, 0.17, 0.72), 202),
        (ServePerformanceDraw(0.75, 0.04, 0.79, 0.24, 0.38), 303),
    ],
)
def test_fixed_interior_production_simulation_reproduces_all_probability_identities(
    performance: ServePerformanceDraw,
    seed: int,
) -> None:
    n_points = 30_000
    rng = np.random.default_rng(seed)
    points = tuple(generate_service_point(performance, rng) for _ in range(n_points))
    counts = aggregate_service_points(points)

    service_wins = sum(point.server_won for point in points)
    _assert_binomial_mc_identity(
        successes=counts.first_serves_in,
        trials=counts.service_points,
        expected_probability=performance.first_serve_in,
        label="F",
    )
    _assert_binomial_mc_identity(
        successes=counts.aces,
        trials=counts.first_serves_in,
        expected_probability=performance.ace_given_first_in,
        label="A | first serve in",
    )
    _assert_binomial_mc_identity(
        successes=counts.q1_wins,
        trials=counts.q1_trials,
        expected_probability=performance.returnable_first_win,
        label="Q1 | returnable first serve",
    )
    _assert_binomial_mc_identity(
        successes=counts.double_faults,
        trials=counts.second_serve_opportunities,
        expected_probability=performance.double_fault_given_second_opp,
        label="D | second-serve opportunity",
    )
    _assert_binomial_mc_identity(
        successes=counts.q2_wins,
        trials=counts.q2_trials,
        expected_probability=performance.playable_second_win,
        label="Q2 | playable second serve",
    )
    _assert_binomial_mc_identity(
        successes=counts.first_serve_points_won,
        trials=counts.first_serves_in,
        expected_probability=performance.first_serve_win,
        label="w1",
    )
    _assert_binomial_mc_identity(
        successes=counts.second_serve_points_won,
        trials=counts.second_serve_opportunities,
        expected_probability=performance.second_serve_win,
        label="w2",
    )
    _assert_binomial_mc_identity(
        successes=service_wins,
        trials=counts.service_points,
        expected_probability=performance.service_point_win,
        label="service-point win",
    )
    _assert_binomial_mc_identity(
        successes=counts.aces,
        trials=counts.service_points,
        expected_probability=performance.ace_rate,
        label="ace rate per service point",
    )
    _assert_binomial_mc_identity(
        successes=counts.double_faults,
        trials=counts.service_points,
        expected_probability=performance.double_fault_rate,
        label="double-fault rate per service point",
    )
