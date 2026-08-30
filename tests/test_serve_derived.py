from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tennis_model.estimation.derived import (
    PrimitiveServeMeans,
    ace_rate_per_service_point,
    double_fault_rate_per_service_point,
    first_serve_win_probability,
    second_serve_win_probability,
    service_point_win_probability,
)

PROBABILITIES = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def test_derived_identities_at_boundaries_and_interior() -> None:
    assert first_serve_win_probability(0.0, 0.0) == 0.0
    assert first_serve_win_probability(1.0, 0.0) == 1.0
    assert first_serve_win_probability(0.0, 1.0) == 1.0
    assert second_serve_win_probability(1.0, 1.0) == 0.0
    assert second_serve_win_probability(0.0, 1.0) == 1.0

    values = PrimitiveServeMeans(0.62, 0.11, 0.67, 0.09, 0.56)
    assert values.first_serve_win == pytest.approx(0.11 + 0.89 * 0.67)
    assert values.second_serve_win == pytest.approx(0.91 * 0.56)
    assert values.service_point_win == pytest.approx(
        0.62 * values.first_serve_win + 0.38 * values.second_serve_win
    )
    assert values.ace_rate == pytest.approx(0.62 * 0.11)
    assert values.double_fault_rate == pytest.approx(0.38 * 0.09)


@given(f=PROBABILITIES, a=PROBABILITIES, q1=PROBABILITIES, d=PROBABILITIES, q2=PROBABILITIES)
def test_derived_identities_remain_in_probability_support(
    f: float, a: float, q1: float, d: float, q2: float
) -> None:
    values = (
        first_serve_win_probability(a, q1),
        second_serve_win_probability(d, q2),
        service_point_win_probability(f, a, q1, d, q2),
        ace_rate_per_service_point(f, a),
        double_fault_rate_per_service_point(f, d),
    )
    assert all(0.0 <= value <= 1.0 for value in values)


@given(
    f=PROBABILITIES,
    low=PROBABILITIES,
    high=PROBABILITIES,
    q1=PROBABILITIES,
    d=PROBABILITIES,
    q2=PROBABILITIES,
)
def test_service_identities_have_the_frozen_monotonic_directions(
    f: float,
    low: float,
    high: float,
    q1: float,
    d: float,
    q2: float,
) -> None:
    lower, upper = sorted((low, high))
    assert first_serve_win_probability(lower, q1) <= first_serve_win_probability(upper, q1)
    assert first_serve_win_probability(low, lower) <= first_serve_win_probability(low, upper)
    assert second_serve_win_probability(lower, q2) >= second_serve_win_probability(upper, q2)
    assert second_serve_win_probability(d, lower) <= second_serve_win_probability(d, upper)
    assert service_point_win_probability(f, lower, q1, d, q2) <= service_point_win_probability(
        f, upper, q1, d, q2
    )
    assert service_point_win_probability(f, low, q1, lower, q2) >= service_point_win_probability(
        f, low, q1, upper, q2
    )


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_derived_identities_reject_invalid_probabilities(bad: float) -> None:
    with pytest.raises(ValueError):
        first_serve_win_probability(bad, 0.5)
