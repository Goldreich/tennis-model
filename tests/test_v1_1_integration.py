from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tennis_model.estimation.strength import StrengthPrediction
from tennis_model.estimation.strength_integration import (
    CrossFittedStrengthRecord,
    StrengthIntegrationConfig,
    fit_strength_integration,
    integrate_serve_performance,
    prepare_strength_match_parameters,
    solve_q_tilt,
)
from tennis_model.exact_probability import exact_match_win_probability
from tennis_model.schemas import Tour
from tennis_model.serve import PrimitiveServeMeans


def _config() -> StrengthIntegrationConfig:
    return StrengthIntegrationConfig(
        schema_version="strength-integration-config/v1",
        l2_penalty=1.0,
        reliability_prior_logit=-1.3862943611198906,
        maximum_absolute_tilt=3.0,
        root_tolerance=1e-8,
        probability_bound=1e-6,
        coefficient_draws_for_summary=64,
        q1_weight=1.0,
        q2_weight=1.0,
    )


def _fit():
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    rows = []
    for index in range(20):
        forecast = cutoff - timedelta(days=40 - index)
        rows.append(
            CrossFittedStrengthRecord(
                match_id=f"cf-{index}",
                tour=Tour.ATP,
                player_a_id="a",
                player_b_id="b",
                forecast_cutoff_utc=forecast,
                scheduled_start_utc=forecast + timedelta(hours=1),
                outcome_available_at_utc=forecast + timedelta(hours=4),
                component_logit=(-0.4 + index / 25.0),
                anchor_logit=(-0.2 + index / 18.0),
                component_variance=0.20,
                anchor_variance=0.15,
                component_instability=0.25,
                component_sparsity=0.30,
                player_a_won=index >= 8,
            )
        )
    return fit_strength_integration(
        tuple(rows),
        tour=Tour.ATP,
        training_cutoff_utc=cutoff,
        fitted_at_utc=cutoff + timedelta(minutes=1),
        config=_config(),
        code_commit="test",
    )


def test_exact_probability_and_q_tilt_are_symmetric_and_monotone() -> None:
    p = exact_match_win_probability(0.65, 0.60, best_of=3)
    reverse = exact_match_win_probability(0.60, 0.65, best_of=3)
    assert p + reverse == pytest.approx(1.0, abs=1e-12)
    a = PrimitiveServeMeans(0.62, 0.12, 0.58, 0.08, 0.52)
    b = PrimitiveServeMeans(0.61, 0.09, 0.54, 0.07, 0.49)
    baseline = exact_match_win_probability(a.service_point_win, b.service_point_win, best_of=3)
    target_logit = np.log(baseline / (1.0 - baseline)) + 0.5
    tilt, attained, saturated = solve_q_tilt(
        a, b, target_logit=float(target_logit), best_of=3, config=_config()
    )
    assert tilt > 0.0
    assert not saturated
    assert attained == pytest.approx(target_logit, abs=2e-8)


@pytest.mark.parametrize(
    ("p_a", "p_b", "best_of", "expected"),
    (
        (0.65, 0.60, 3, 0.7376734700452428),
        (0.65, 0.60, 5, 0.7861870823200439),
        (0.58, 0.72, 3, 0.04159807007857941),
        (0.58, 0.72, 5, 0.015533106653154483),
        (0.50, 0.50, 3, 0.5),
    ),
)
def test_exact_probability_optimization_preserves_reference_values(
    p_a: float, p_b: float, best_of: int, expected: float
) -> None:
    assert exact_match_win_probability(p_a, p_b, best_of=best_of) == pytest.approx(
        expected, abs=1e-15
    )


@pytest.mark.parametrize(
    ("target", "expected_tilt"),
    (
        (-2.0, -0.5561139760538936),
        (-0.5, -0.28983937203884125),
        (0.0, -0.19460939057171345),
        (0.75, -0.05263620615005493),
        (2.0, 0.1657627997919917),
    ),
)
def test_bracketed_tilt_optimization_preserves_reference_solution(
    target: float, expected_tilt: float
) -> None:
    a = PrimitiveServeMeans(0.62, 0.12, 0.70, 0.08, 0.55)
    b = PrimitiveServeMeans(0.64, 0.08, 0.66, 0.10, 0.52)
    tilt, attained, saturated = solve_q_tilt(
        a, b, target_logit=target, best_of=5, config=_config()
    )
    assert not saturated
    assert tilt == pytest.approx(expected_tilt, abs=2e-8)
    assert attained == pytest.approx(target, abs=_config().root_tolerance)


def test_integration_preserves_f_a_d_and_is_reproducible() -> None:
    fit = _fit()
    a_means = PrimitiveServeMeans(0.62, 0.12, 0.58, 0.08, 0.52)
    b_means = PrimitiveServeMeans(0.61, 0.09, 0.54, 0.07, 0.49)
    anchor = StrengthPrediction(
        player_a_id="a",
        player_b_id="b",
        surface="hard",
        best_of=3,
        scheduled_start_utc=datetime(2026, 9, 1, tzinfo=UTC),
        mean_logit=0.8,
        variance_logit=0.2,
        probability=0.68,
        player_a_graph_component=0,
        player_b_graph_component=0,
        weakly_connected=False,
        player_a_known=True,
        player_b_known=True,
    )
    params = prepare_strength_match_parameters(
        anchor_artifact_id="a" * 64,
        integration_artifact_id="b" * 64,
        anchor=anchor,
        integration=fit,
        player_a=a_means,
        player_b=b_means,
        best_of=3,
        component_variance=0.2,
        component_instability=0.2,
        component_sparsity=0.3,
    )
    a = a_means
    b = b_means
    left = integrate_serve_performance(
        a, b, parameters=params, integration=fit, best_of=3, rng=np.random.default_rng(7)
    )
    right = integrate_serve_performance(
        a, b, parameters=params, integration=fit, best_of=3, rng=np.random.default_rng(7)
    )
    assert left == right
    for original, adjusted in ((a, left[0]), (b, left[1])):
        assert adjusted.first_serve_in == original.first_serve_in
        assert adjusted.ace_given_first_in == original.ace_given_first_in
        assert adjusted.double_fault_given_second_opp == original.double_fault_given_second_opp
        assert adjusted.ace_rate == original.ace_rate
        assert adjusted.double_fault_rate == original.double_fault_rate
