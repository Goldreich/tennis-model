from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.linalg import helmert
from scipy.stats import t as student_t

from tennis_model.estimation.duration_model import (
    FLOOR_DURATION_DISPLAY_POLICY,
    NEAREST_DURATION_DISPLAY_POLICY,
    UNRESOLVED_DURATION_DISPLAY_POLICY,
    DurationArtifactIntegrityError,
    DurationCoefficient,
    DurationConditions,
    DurationFitArtifact,
    DurationModelConfig,
    DurationModelError,
    DurationPaceEffect,
    DurationParameterDraw,
    DurationPathExposure,
    _context_activation,
    _design,
    _prior_sds,
    _student_objective,
    build_duration_training_batch,
    draw_duration,
    duration_predictor,
    duration_scale,
    fit_duration_model,
    load_duration_fit_artifact,
    load_duration_model_config,
    map_duration_parameters,
    prepare_duration_parameter_sampler,
    sample_duration_parameters,
    sample_duration_parameters_for_players,
    sample_prepared_duration_parameters,
    write_duration_fit_artifact,
)
from tennis_model.schemas import Tour

_CUTOFF = datetime(2026, 8, 30, 14, 0, tzinfo=UTC)
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


@pytest.fixture(scope="module")
def config() -> DurationModelConfig:
    return load_duration_model_config(Path("config/duration_v1.yaml"))


def _rows(
    match_id: str,
    *,
    score: str = "6-4 6-4",
    points_a: int = 55,
    points_b: int = 53,
    duration: float = 100.0,
    player_a: str = "a",
    player_b: str = "b",
    match_date: date | None = date(2026, 8, 20),
    available_at: datetime = _CUTOFF - timedelta(days=1),
    completed: bool = True,
    retirement: bool = False,
    tour: Tour = Tour.ATP,
    event: str | None = "Fixture Open",
    event_year: int | None = 2026,
    best_of: int = 3,
    indoor: bool | None = None,
    temperature_c: float | None = None,
    night_session: bool | None = None,
) -> list[dict[str, Any]]:
    common: dict[str, Any] = {
        "match_id": match_id,
        "tour": tour.value,
        "match_date": match_date,
        "available_at_utc": available_at,
        "duration_minutes": duration,
        "score": score,
        "best_of": best_of,
        "completed": completed,
        "retirement": retirement,
        "walkover": False,
        "source_id": "fixture-source",
        "snapshot_sha256": _HASH_A,
        "match_date_source_sha256": _HASH_B,
        "exact_date_crosswalk_sha256": _HASH_C,
        "event": event,
        "event_year": event_year,
        "indoor": indoor,
        "temperature_c": temperature_c,
        "night_session": night_session,
    }
    return [
        common
        | {
            "player_id": player_a,
            "opponent_id": player_b,
            "service_points": points_a,
        },
        common
        | {
            "player_id": player_b,
            "opponent_id": player_a,
            "service_points": points_b,
        },
    ]


def test_config_is_strict_immutable_and_hash_stable(
    config: DurationModelConfig, tmp_path: Path
) -> None:
    assert config.schema_version == "duration-model-config/v1"
    assert config.framework_version == "v1.0"
    assert config.sha256 == load_duration_model_config("config/duration_v1.yaml").sha256
    with pytest.raises(ValidationError):
        config.window_days = 1  # type: ignore[misc]

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: duration-model-config/v1\nschema_version: duration-model-config/v1\n",
        encoding="utf-8",
    )
    with pytest.raises(DurationModelError, match="duplicate key"):
        load_duration_model_config(duplicate)


def test_training_builder_uses_reciprocal_exposure_and_retains_match_tiebreak(
    config: DurationModelConfig,
) -> None:
    batch = build_duration_training_batch(
        _rows("match-tb", score="6-4 4-6 [10-7]", points_a=60, points_b=58),
        tour=Tour.ATP,
        information_cutoff_utc=_CUTOFF,
        config=config,
    )
    assert batch.audit.included_match_count == 1
    assert batch.audit.match_tiebreak_format_included_count == 1
    item = batch.observations[0]
    assert item.total_points == 118
    assert item.official_games == 20
    assert item.sets == 3
    assert item.tiebreaks == 1
    assert item.match_tiebreak_format
    assert item.player_a_id == "a"
    assert item.player_b_id == "b"
    assert item.source_sha256s == (_HASH_A,)
    assert item.crosswalk_sha256s == (_HASH_B, _HASH_C)


def test_training_builder_fail_closed_audits_dates_cutoff_retirement_and_anomalies(
    config: DurationModelConfig,
) -> None:
    rows = [
        *_rows("valid"),
        *_rows("undated", match_date=None),
        *_rows("future-availability", available_at=_CUTOFF),
        *_rows("retired", retirement=True),
        *_rows("bad-score", score="6-6 6-4"),
        *_rows("zero-duration", duration=0.0),
    ]
    batch = build_duration_training_batch(
        rows,
        tour=Tour.ATP,
        information_cutoff_utc=_CUTOFF,
        config=config,
    )
    assert batch.audit.included_match_count == 1
    assert batch.audit.excluded_match_count == 5
    assert batch.audit.retirement_match_count == 1
    reasons = {item.reason for item in batch.exclusions}
    assert reasons == {
        "MISSING_OR_INCONSISTENT_EXACT_DATE",
        "NOT_AVAILABLE_BEFORE_CUTOFF",
        "RETIREMENT_EXCLUDED_FROM_ORDINARY_FIT",
        "ILLEGAL_OR_INCOMPLETE_SCORE",
        "MISSING_OR_NONPOSITIVE_EXPOSURE",
    }


def test_training_builder_quarantines_reciprocal_context_and_source_anomalies(
    config: DurationModelConfig,
) -> None:
    event_rows = _rows("event-disagreement")
    event_rows[1]["event"] = "Different Open"
    event_year_rows = _rows("event-year-disagreement")
    event_year_rows[1]["event_year"] = 2025
    indoor_rows = _rows("indoor-disagreement", indoor=True)
    indoor_rows[1]["indoor"] = False
    temperature_rows = _rows("temperature-disagreement", temperature_c=25.0)
    temperature_rows[1]["temperature_c"] = 26.0
    session_rows = _rows("session-disagreement", night_session=True)
    session_rows[1]["night_session"] = False
    missing_source_rows = _rows("missing-source")
    missing_source_rows[1]["source_id"] = None

    batch = build_duration_training_batch(
        [
            *event_rows,
            *event_year_rows,
            *indoor_rows,
            *temperature_rows,
            *session_rows,
            *missing_source_rows,
        ],
        tour=Tour.ATP,
        information_cutoff_utc=_CUTOFF,
        config=config,
    )

    assert batch.audit.included_match_count == 0
    assert batch.audit.excluded_match_count == 6
    reason_counts = {item.reason: item.match_count for item in batch.audit.reason_counts}
    assert reason_counts == {
        "INCONSISTENT_OR_INVALID_CONTEXT": 5,
        "INVALID_OR_MISSING_PROVENANCE": 1,
    }
    assert batch.audit.anomaly_match_count == 6
    assert "truthful source_id" in next(
        item.details for item in batch.exclusions if item.match_id == "missing-source"
    )


def test_contexts_activate_from_known_variation_and_missing_rows_use_zero_baseline(
    config: DurationModelConfig,
) -> None:
    rows: list[dict[str, Any]] = []
    for index in range(42):
        if index < 20:
            indoor: bool | None = False
            temperature: float | None = 20.0
            night: bool | None = False
        elif index < 40:
            indoor = True
            temperature = 30.0
            night = True
        else:
            indoor = None
            temperature = None
            night = None
        if index == 0:
            event: str | None = "US Open"
            event_year: int | None = 2026
        elif index == 41:
            event = None
            event_year = None
        else:
            event = "Fixture Open"
            event_year = 2026
        rows.extend(
            _rows(
                f"context-{index:03d}",
                duration=90.0 + index,
                match_date=date(2026, 8, 20) - timedelta(days=index),
                indoor=indoor,
                temperature_c=temperature,
                night_session=night,
                event=event,
                event_year=event_year,
            )
        )

    batch = build_duration_training_batch(
        rows,
        tour=Tour.ATP,
        information_cutoff_utc=_CUTOFF,
        config=config,
    )
    active, statuses = _context_activation(batch.observations, config)
    assert active == (
        "indoor",
        "temperature_c",
        "night_session",
        "current_usopen_2026",
    )
    assert all(item.active for item in statuses)
    # The current event has only one row, far below minimum_level_rows, but its
    # fixed four-minute prior supplies the required strong small-sample shrinkage.
    assert (
        sum(
            item.conditions.event == "US Open" and item.conditions.event_year == 2026
            for item in batch.observations
        )
        == 1
    )

    design, _response, _weights, names = _design(batch.observations, active, config)
    missing_index = next(
        index for index, item in enumerate(batch.observations) if item.match_id == "context-041"
    )
    missing = batch.observations[missing_index]
    assert missing.conditions == DurationConditions()
    context_columns = [names.index(name) for name in active]
    assert design[missing_index, context_columns] == pytest.approx(np.zeros(len(active)))


def _parameter_draw() -> DurationParameterDraw:
    return DurationParameterDraw(
        artifact_id=_HASH_A,
        tour=Tour.WTA,
        coefficients=(
            DurationCoefficient(name="theta0", value=2.0, standard_error=0.1),
            DurationCoefficient(name="points", value=0.5, standard_error=0.1),
            DurationCoefficient(name="games", value=1.0, standard_error=0.1),
            DurationCoefficient(name="sets", value=3.0, standard_error=0.1),
            DurationCoefficient(name="tiebreaks", value=4.0, standard_error=0.1),
            DurationCoefficient(name="temperature_c", value=2.0, standard_error=0.1),
            DurationCoefficient(name="current_usopen_2026", value=5.0, standard_error=0.1),
        ),
        pace_effects=(
            DurationPaceEffect(
                player_id="a", value=-1.0, standard_error=0.2, weighted_matches=10.0
            ),
            DurationPaceEffect(player_id="b", value=1.0, standard_error=0.2, weighted_matches=10.0),
        ),
        temperature_reference_c=20.0,
        current_event="US Open",
        current_event_year=2026,
        sigma0=2.0,
        sigma1=0.5,
        nu=7.0,
    )


def _exposure(*, tour: Tour = Tour.WTA) -> DurationPathExposure:
    return DurationPathExposure(
        tour=tour,
        player_a_id="a",
        player_b_id="b",
        total_points=100,
        official_games=20,
        sets=2,
        tiebreaks=1,
        conditions=DurationConditions(
            temperature_c=22.0,
            event="US Open",
            event_year=2026,
        ),
    )


def test_predictor_scale_context_and_tour_separation_are_exact() -> None:
    parameters = _parameter_draw()
    exposure = _exposure()
    expected = 2 + 0.5 * 100 + 20 + 3 * 2 + 4 + (-1 + 1) + 2 * (22 - 20) + 5
    baseline = duration_predictor(parameters, exposure)
    assert baseline == expected
    assert duration_scale(parameters, 100) == 7.0
    increments = {
        "total_points": 0.5,
        "official_games": 1.0,
        "sets": 3.0,
        "tiebreaks": 4.0,
    }
    for field, expected_increment in increments.items():
        changed = exposure.model_copy(update={field: getattr(exposure, field) + 1})
        assert duration_predictor(parameters, changed) - baseline == expected_increment
    swapped = exposure.model_copy(
        update={"player_a_id": exposure.player_b_id, "player_b_id": exposure.player_a_id}
    )
    assert duration_predictor(parameters, swapped) == baseline
    with pytest.raises(DurationModelError, match="across tours"):
        duration_predictor(parameters, _exposure(tour=Tour.ATP))


class _FixedResidual:
    def __init__(self, value: float) -> None:
        self.value = value

    def standard_t(self, _nu: float) -> float:
        return self.value


def test_duration_draw_has_exact_one_minute_floor_and_explicit_display_policy() -> None:
    parameters = _parameter_draw()
    exposure = _exposure()
    truncated = draw_duration(
        parameters,
        exposure,
        _FixedResidual(-1e9),  # type: ignore[arg-type]
        display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
        partial=True,
    )
    assert truncated.latent_minutes == 1.0
    assert truncated.official_minutes is None
    assert truncated.candidate_official_minutes == (1,)
    assert truncated.partial

    fractional = draw_duration(
        parameters,
        exposure,
        _FixedResidual((100.2 - duration_predictor(parameters, exposure)) / 7.0),  # type: ignore[arg-type]
        display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
    )
    assert fractional.latent_minutes == pytest.approx(100.2)
    assert fractional.candidate_official_minutes == (100, 101)

    floor_draw = draw_duration(
        parameters,
        exposure,
        _FixedResidual(0.1),  # type: ignore[arg-type]
        display_policy=FLOOR_DURATION_DISPLAY_POLICY,
    )
    nearest_draw = draw_duration(
        parameters,
        exposure,
        _FixedResidual(0.1),  # type: ignore[arg-type]
        display_policy=NEAREST_DURATION_DISPLAY_POLICY,
    )
    assert floor_draw.official_minutes == int(np.floor(floor_draw.latent_minutes))
    assert nearest_draw.official_minutes == int(np.floor(nearest_draw.latent_minutes + 0.5))
    seeded_first = draw_duration(parameters, exposure, np.random.default_rng(994))
    seeded_second = draw_duration(parameters, exposure, np.random.default_rng(994))
    assert seeded_first == seeded_second


def _synthetic_batch(config: DurationModelConfig) -> Any:
    rng = np.random.default_rng(741)
    players = tuple(f"p{index}" for index in range(10))
    pace = {player: (index - 4.5) * 0.35 for index, player in enumerate(players)}
    score_specs = (
        ("6-0 6-0", 12, 2, 0),
        ("7-6(4) 6-4", 23, 2, 1),
        ("6-4 4-6 6-3", 29, 3, 0),
        ("7-6(4) 6-7(5) 7-6(6)", 39, 3, 3),
        ("6-4 4-6 [10-7]", 20, 3, 1),
    )
    rows: list[dict[str, Any]] = []
    for index in range(260):
        pair = rng.choice(len(players), size=2, replace=False)
        player_a = players[int(pair[0])]
        player_b = players[int(pair[1])]
        score, games, sets, tiebreaks = score_specs[index % len(score_specs)]
        total_points = games * 4 + 18 + (index * 7) % 31
        center = (
            8.0
            + 0.48 * total_points
            + 0.72 * games
            + 2.4 * sets
            + 1.3 * tiebreaks
            + pace[player_a]
            + pace[player_b]
        )
        scale = 2.5 + 0.08 * np.sqrt(total_points)
        duration = center + scale * rng.standard_t(8.0)
        points_a = total_points // 2
        points_b = total_points - points_a
        rows.extend(
            _rows(
                f"synthetic-{index:03d}",
                score=score,
                points_a=points_a,
                points_b=points_b,
                duration=duration,
                player_a=player_a,
                player_b=player_b,
                match_date=date(2026, 8, 1) - timedelta(days=index % 300),
                event="Synthetic Open",
            )
        )
    # A single extreme match for a new player exercises the fixed strong pace
    # prior: the fitted q must remain far below the +40-minute raw deviation.
    sparse_total_points = 80
    sparse_center = 8.0 + 0.48 * sparse_total_points + 0.72 * 12 + 2.4 * 2 + pace["p0"]
    rows.extend(
        _rows(
            "synthetic-sparse",
            score="6-0 6-0",
            points_a=40,
            points_b=40,
            duration=sparse_center + 40.0,
            player_a="p_sparse",
            player_b="p0",
            match_date=date(2026, 8, 1),
            event="Synthetic Open",
        )
    )
    return build_duration_training_batch(
        rows,
        tour=Tour.ATP,
        information_cutoff_utc=_CUTOFF,
        config=config,
    )


@pytest.fixture(scope="module")
def fitted(config: DurationModelConfig) -> Any:
    batch = _synthetic_batch(config)
    return fit_duration_model(
        batch,
        config=config,
        source_manifest_id="synthetic-manifest-v1",
        source_manifest_sha256=_HASH_A,
        fitted_at_utc=_CUTOFF + timedelta(minutes=1),
        software_version="test",
        code_sha256=_HASH_C,
        deterministic_test_result_sha256=_HASH_D,
    )


def test_synthetic_fit_recovers_duration_surface_and_strongly_shrinks_pace(
    fitted: Any,
) -> None:
    assert fitted.diagnostics.observation_count == 261
    assert fitted.diagnostics.converged
    assert fitted.diagnostics.gradient_infinity_norm < 0.5
    assert fitted.sigma0 > 0
    assert fitted.sigma1 >= 0
    assert fitted.nu > 2
    assert abs(sum(item.value for item in fitted.pace_effects)) < 1e-8
    assert max(abs(item.value) for item in fitted.pace_effects) < 8.0

    coefficients = {item.name: item.value for item in fitted.coefficients}
    assert coefficients["theta0"] == pytest.approx(8.0, abs=2.5)
    assert coefficients["points"] == pytest.approx(0.48, abs=0.06)
    assert coefficients["games"] == pytest.approx(0.72, abs=0.22)
    assert coefficients["sets"] == pytest.approx(2.4, abs=1.2)
    assert coefficients["tiebreaks"] == pytest.approx(1.3, abs=0.5)
    effects = {item.player_id: item.value for item in fitted.pace_effects}
    assert effects["p0"] == pytest.approx(-1.575, abs=0.85)
    assert effects["p9"] == pytest.approx(1.575, abs=0.85)
    assert 0 < effects["p_sparse"] < 8.0
    assert effects["p_sparse"] < 0.2 * 40.0

    parameters = map_duration_parameters(fitted)
    exposure = DurationPathExposure(
        tour=Tour.ATP,
        player_a_id="unseen-a",
        player_b_id="unseen-b",
        total_points=125,
        official_games=24,
        sets=3,
        tiebreaks=1,
    )
    truth = 8 + 0.48 * 125 + 0.72 * 24 + 2.4 * 3 + 1.3
    assert duration_predictor(parameters, exposure) == pytest.approx(truth, abs=8.0)
    assert duration_scale(parameters, 200) > duration_scale(parameters, 80)
    assert fitted.training_start_date <= fitted.training_end_date
    assert fitted.ridge.player_pace_sd == 4.0
    assert fitted.sigma0 == pytest.approx(2.5, abs=1.0)
    assert fitted.sigma1 == pytest.approx(0.08, abs=0.06)


def test_stored_laplace_covariance_matches_independent_map_hessian_and_predictor_variance(
    config: DurationModelConfig,
    fitted: Any,
) -> None:
    batch = _synthetic_batch(config)
    observations = batch.observations
    active_contexts, _statuses = _context_activation(observations, config)
    design, response, weights, coefficient_names = _design(observations, active_contexts, config)
    players = tuple(
        sorted(
            {item.player_a_id for item in observations}
            | {item.player_b_id for item in observations}
        )
    )
    player_index = {player: index for index, player in enumerate(players)}
    incidence = np.zeros((len(observations), len(players)), dtype=float)
    for row_index, item in enumerate(observations):
        incidence[row_index, player_index[item.player_a_id]] = 1.0
        incidence[row_index, player_index[item.player_b_id]] = 1.0
    contrast = helmert(len(players), full=False).T
    beta_sds = _prior_sds(coefficient_names, config)

    pace_by_player = {item.player_id: item.value for item in fitted.pace_effects}
    pace = np.asarray([pace_by_player[player] for player in players], dtype=float)
    map_unconstrained = np.concatenate(
        (
            np.asarray([item.value for item in fitted.coefficients], dtype=float),
            contrast.T @ pace,
            np.log(np.asarray([fitted.sigma0, fitted.sigma1, fitted.nu - 2.0])),
        )
    )

    def gradient(value: np.ndarray) -> np.ndarray:
        return _student_objective(
            value,
            x=design,
            y=response,
            weights=weights,
            incidence=incidence,
            contrast=contrast,
            beta_sds=beta_sds,
            config=config,
        )[1]

    # Use an independently implemented central difference and a different step
    # from the production curvature helper so this test does not merely replay it.
    columns: list[np.ndarray] = []
    relative_step = config.optimizer.covariance_relative_step / 2.0
    for index, value in enumerate(map_unconstrained):
        step = relative_step * max(1.0, abs(float(value)))
        plus = map_unconstrained.copy()
        minus = map_unconstrained.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((gradient(plus) - gradient(minus)) / (2.0 * step))
    raw_hessian = np.column_stack(columns)
    raw_hessian = (raw_hessian + raw_hessian.T) / 2.0
    raw_minimum = float(np.min(np.linalg.eigvalsh(raw_hessian)))
    regularization = max(0.0, config.optimizer.covariance_eigenvalue_floor - raw_minimum)
    regularized_hessian = raw_hessian + regularization * np.eye(len(map_unconstrained))
    raw_covariance = np.linalg.inv(regularized_hessian)
    raw_covariance = (raw_covariance + raw_covariance.T) / 2.0

    coefficient_count = len(coefficient_names)
    transform = np.zeros(
        (coefficient_count + len(players) + 3, len(map_unconstrained)), dtype=float
    )
    transform[:coefficient_count, :coefficient_count] = np.eye(coefficient_count)
    transform[
        coefficient_count : coefficient_count + len(players),
        coefficient_count : coefficient_count + len(players) - 1,
    ] = contrast
    transform[coefficient_count + len(players) :, coefficient_count + len(players) - 1 :] = np.eye(
        3
    )
    expected_covariance = transform @ raw_covariance @ transform.T
    expected_covariance = (expected_covariance + expected_covariance.T) / 2.0

    posterior = fitted.posterior
    assert config.optimizer.full_covariance_max_parameters >= 700
    assert posterior.method == "finite-difference-map-gradient-laplace/v1"
    assert posterior.curvature_parameterization == "beta-helmert-pace-log-residual/v1"
    assert posterior.curvature_parameter_count == len(map_unconstrained)
    assert posterior.covariance_mode == "full"
    observed_covariance = np.asarray(posterior.covariance, dtype=float)
    np.testing.assert_allclose(
        observed_covariance,
        expected_covariance,
        rtol=2.0e-4,
        atol=2.0e-7,
    )

    index = {name: offset for offset, name in enumerate(posterior.parameter_names)}
    predictor_gradient = np.zeros(len(posterior.parameter_names), dtype=float)
    predictor_gradient[index["coefficient:theta0"]] = 1.0
    predictor_gradient[index["coefficient:points"]] = 125.0
    predictor_gradient[index["coefficient:games"]] = 24.0
    predictor_gradient[index["coefficient:sets"]] = 3.0
    predictor_gradient[index["coefficient:tiebreaks"]] = 1.0
    predictor_gradient[index["pace:p0"]] = 1.0
    predictor_gradient[index["pace:p1"]] = 1.0
    observed_variance = float(predictor_gradient @ observed_covariance @ predictor_gradient)
    expected_variance = float(predictor_gradient @ expected_covariance @ predictor_gradient)
    assert observed_variance > 0.0
    assert observed_variance == pytest.approx(expected_variance, rel=2.0e-4)


def test_parameter_and_residual_draws_are_explicit_reproducible_and_student_t(
    fitted: Any,
) -> None:
    first = sample_duration_parameters(fitted, np.random.default_rng(123))
    second = sample_duration_parameters(fitted, np.random.default_rng(123))
    assert first == second
    prepared = prepare_duration_parameter_sampler(fitted, ("p0", "p1"))
    prepared_draw = sample_prepared_duration_parameters(prepared, np.random.default_rng(456))
    direct_draw = sample_duration_parameters_for_players(
        fitted, ("p0", "p1"), np.random.default_rng(456)
    )
    assert prepared_draw == direct_draw
    assert not prepared.mean.flags.writeable
    assert not prepared.covariance_or_variance.flags.writeable

    parameters = map_duration_parameters(fitted)
    exposure = DurationPathExposure(
        tour=Tour.ATP,
        player_a_id="p0",
        player_b_id="p1",
        total_points=120,
        official_games=22,
        sets=2,
        tiebreaks=0,
    )
    rng = np.random.default_rng(987)
    draws = [draw_duration(parameters, exposure, rng) for _ in range(30_000)]
    residuals = np.asarray([item.standardized_residual for item in draws])
    latent = np.asarray([item.latent_minutes for item in draws])
    assert np.quantile(residuals, 0.1) == pytest.approx(student_t.ppf(0.1, parameters.nu), abs=0.06)
    assert np.quantile(residuals, 0.5) == pytest.approx(0.0, abs=0.04)
    assert np.quantile(residuals, 0.9) == pytest.approx(student_t.ppf(0.9, parameters.nu), abs=0.06)
    center = duration_predictor(parameters, exposure)
    scale = duration_scale(parameters, exposure.total_points)
    for probability in (0.1, 0.5, 0.9):
        expected = center + scale * student_t.ppf(probability, parameters.nu)
        assert np.quantile(latent, probability) == pytest.approx(expected, abs=0.8)

    longer = exposure.model_copy(
        update={"total_points": 180, "official_games": 34, "sets": 3, "tiebreaks": 1}
    )
    long_rng = np.random.default_rng(321)
    long_latent = np.asarray(
        [draw_duration(parameters, longer, long_rng).latent_minutes for _ in range(10_000)]
    )
    assert np.median(long_latent) > np.median(latent) + 20.0


class _CaptureGaussian:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    def multivariate_normal(
        self, mean: np.ndarray, covariance: np.ndarray, *, check_valid: str
    ) -> np.ndarray:
        assert check_valid == "raise"
        self.mean = mean.copy()
        self.covariance = covariance.copy()
        return mean.copy()


def test_target_local_sampler_uses_exact_stored_gaussian_marginal(fitted: Any) -> None:
    assert fitted.posterior.covariance_mode == "full"
    capture = _CaptureGaussian()
    draw = sample_duration_parameters_for_players(
        fitted,
        ("p0", "p9", "unknown-player"),
        capture,  # type: ignore[arg-type]
    )
    assert tuple(item.player_id for item in draw.pace_effects) == ("p0", "p9")
    assert capture.mean is not None
    assert capture.covariance is not None

    selected_names = (
        *(f"coefficient:{item.name}" for item in fitted.coefficients),
        "pace:p0",
        "pace:p9",
        "log_sigma0",
        "log_sigma1",
        "log_nu_minus_two",
    )
    index = {name: offset for offset, name in enumerate(fitted.posterior.parameter_names)}
    selected = np.asarray([index[name] for name in selected_names])
    full_mean = np.asarray(fitted.posterior.map_values)
    full_covariance = np.asarray(fitted.posterior.covariance)
    assert capture.mean == pytest.approx(full_mean[selected], abs=1e-15)
    assert capture.covariance == pytest.approx(
        full_covariance[np.ix_(selected, selected)], abs=1e-15
    )


def test_duration_artifact_round_trip_is_canonical_and_tamper_evident(
    fitted: Any, tmp_path: Path
) -> None:
    persisted = write_duration_fit_artifact(fitted, tmp_path)
    assert persisted.artifact == fitted
    assert write_duration_fit_artifact(fitted, tmp_path) == persisted
    raw = json.loads(persisted.artifact_path.read_text(encoding="utf-8"))
    raw["sigma0"] += 1
    persisted.artifact_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DurationArtifactIntegrityError):
        load_duration_fit_artifact(persisted.directory)


def test_duration_artifact_rejects_internally_inconsistent_rehashed_content(
    fitted: Any,
) -> None:
    payload = fitted.model_dump(mode="json", exclude={"artifact_id"})
    payload["diagnostics"]["observation_count"] += 1
    artifact_id = hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValidationError, match="counts do not reconcile"):
        DurationFitArtifact.model_validate({"artifact_id": artifact_id, **payload})
