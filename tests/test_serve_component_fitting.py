from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from serve_model_test_helpers import (
    TEST_CUTOFF,
    make_model_config,
    make_provenance,
    synthetic_all_component_counts,
    synthetic_component_counts,
)

import tennis_model.estimation.serve_components as serve_components_module
from tennis_model.data.component_counts import build_serve_component_counts
from tennis_model.estimation.serve_components import (
    FitConvergenceError,
    FutureMatchContext,
    ModelDataError,
    ServeComponent,
    fit_all_serve_components,
    fit_serve_component,
    predict_component,
    predict_serve_performance,
    prepare_component_rows,
)
from tennis_model.schemas import Tour


def _context(
    server: str,
    returner: str,
    *,
    tour: Tour = Tour.ATP,
    surface: str = "Hard",
) -> FutureMatchContext:
    return FutureMatchContext(
        serving_player_id=server,
        returning_player_id=returner,
        tour=tour,
        surface=surface,
        indoor=False,
        event="Future Open",
        event_year=2026,
        match_date_utc=TEST_CUTOFF + timedelta(days=2),
        information_cutoff_utc=TEST_CUTOFF,
    )


def test_smoke_fit_returns_converged_map_and_laplace_artifact() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(ServeComponent.A, repetitions=3),
        component=ServeComponent.A,
        tour="ATP",
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance(),
    )
    assert fit.diagnostics.converged
    assert fit.diagnostics.usable_rows == 36
    assert fit.diagnostics.kappa > 0.0
    size = len(fit.posterior.parameter_names)
    assert len(fit.posterior.map_parameters) == size
    assert len(fit.posterior.variance_diagonal) == size
    assert fit.posterior.curvature_kind == "full"
    assert fit.posterior.covariance is not None


def test_full_weighted_map_objective_gradient_matches_finite_differences() -> None:
    config = make_model_config()
    prepared = prepare_component_rows(
        synthetic_component_counts(ServeComponent.A, repetitions=2),
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
    )
    design = serve_components_module._build_design(prepared, ServeComponent.A, config)
    theta = serve_components_module._initial_parameters(prepared, design, config)
    theta = theta + np.linspace(-0.05, 0.05, len(theta))
    value, gradient = serve_components_module._objective_and_gradient(
        theta, prepared, design, config
    )
    assert math.isfinite(value)
    step = 1e-6
    numeric = np.empty_like(theta)
    for index in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = serve_components_module._objective_and_gradient(
            plus, prepared, design, config
        )[0]
        minus_value = serve_components_module._objective_and_gradient(
            minus, prepared, design, config
        )[0]
        numeric[index] = (plus_value - minus_value) / (2.0 * step)
    assert gradient == pytest.approx(numeric, rel=2e-4, abs=2e-5)


def test_primitive_components_consume_only_milestone_one_routed_counts() -> None:
    service_row = pd.DataFrame(
        [
            {
                "snapshot_id": "snapshot-1",
                "match_id": "match-1",
                "tour": "ATP",
                "player_id": "server",
                "opponent_id": "returner",
                "source_date": datetime(2025, 8, 1).date(),
                "match_date": datetime(2025, 8, 1).date(),
                "available_at_utc": datetime(2025, 8, 2, tzinfo=UTC),
                "event": "Routing Open",
                "event_year": 2025,
                "surface": "Hard",
                "indoor": "Outdoor",
                "best_of": 3,
                "service_points": 100,
                "first_serves_in": 60,
                "first_serve_points_won": 42,
                "second_serve_points_won": 20,
                "aces": 10,
                "double_faults": 5,
                "invalid_stat_fields": (),
            }
        ]
    )
    table = build_serve_component_counts(service_row)
    expected = {
        ServeComponent.F: (60, 100),
        ServeComponent.A: (10, 60),
        ServeComponent.Q1: (32, 50),
        ServeComponent.D: (5, 40),
        ServeComponent.Q2: (20, 35),
    }
    for component, (successes, trials) in expected.items():
        prepared = prepare_component_rows(
            table,
            component=component,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(),
        )
        assert prepared.successes.tolist() == [successes]
        assert prepared.trials.tolist() == [trials]


def test_preparation_enforces_status_window_and_strict_cutoff() -> None:
    base = synthetic_component_counts(ServeComponent.F, repetitions=1).iloc[:1].copy()
    records: list[dict[str, object]] = []
    for offset, reason in enumerate(
        ("usable", "at_cutoff", "too_old", "missing", "zero_denominator", "quarantined")
    ):
        row = base.iloc[0].to_dict()
        row["match_id"] = f"match-{reason}"
        row["source_row_number"] = offset + 2
        if reason == "at_cutoff":
            row["available_at_utc"] = TEST_CUTOFF
        elif reason == "too_old":
            old_date = (TEST_CUTOFF - timedelta(days=1096)).date()
            row["match_date"] = old_date
            row["source_date"] = old_date
            row["available_at_utc"] = datetime.combine(
                old_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            )
        elif reason == "missing":
            row["successes"] = pd.NA
            row["trials"] = pd.NA
            row["status"] = "missing_input"
            row["eligible_for_likelihood"] = False
        elif reason == "zero_denominator":
            row["successes"] = 0
            row["trials"] = 0
            row["status"] = "zero_denominator"
            row["eligible_for_likelihood"] = False
        elif reason == "quarantined":
            row["successes"] = 101
            row["trials"] = 100
            row["status"] = "quarantined"
            row["eligible_for_likelihood"] = False
        records.append(row)
    prepared = prepare_component_rows(
        pd.DataFrame.from_records(records),
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
    )
    assert len(prepared.frame) == 1
    assert prepared.excluded == {
        "at_or_after_cutoff": 1,
        "outside_1095_day_window": 1,
        "missing_input": 1,
        "zero_denominator": 1,
        "quarantined": 1,
    }


def test_eligible_invalid_counts_cannot_silently_reenter_fitting() -> None:
    frame = synthetic_component_counts(ServeComponent.F, repetitions=1)
    frame.loc[0, "successes"] = 1000
    with pytest.raises(ModelDataError, match="0 <= successes <= trials"):
        prepare_component_rows(
            frame,
            component=ServeComponent.F,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(),
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("successes", 3.7, "exact integer"),
        ("trials", True, "exact integer"),
        ("eligible_for_likelihood", "False", "exact boolean"),
    ],
)
def test_direct_count_input_rejects_lossy_scalar_coercions(
    column: str, value: object, message: str
) -> None:
    frame = synthetic_component_counts(ServeComponent.F, repetitions=1)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ModelDataError, match=message):
        prepare_component_rows(
            frame,
            component=ServeComponent.F,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(),
        )


@pytest.mark.parametrize("component", [ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2])
def test_opponent_adjusted_components_recover_server_and_returner_directions(
    component: ServeComponent,
) -> None:
    frame = synthetic_component_counts(
        component,
        repetitions=9,
        trials=120,
        intercept=-0.3,
        server_effects={"p0": 0.9, "p1": -0.9},
        returner_effects={"p0": 0.8, "p1": -0.8},
        kappa=100.0,
        seed=100 + list(ServeComponent).index(component),
    )
    fit = fit_serve_component(
        frame,
        component=component,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance(component.value),
    )
    strong_server = predict_component(fit, _context("p0", "p2")).map_mean
    weak_server = predict_component(fit, _context("p1", "p2")).map_mean
    strong_returner = predict_component(fit, _context("p2", "p0")).map_mean
    weak_returner = predict_component(fit, _context("p2", "p1")).map_mean
    assert strong_server > weak_server
    assert strong_returner < weak_returner
    roles = {block.role for block in fit.effect_blocks}
    assert "returner_global" in roles
    assert "returner_surface" in roles


@pytest.mark.parametrize("component", [ServeComponent.F, ServeComponent.D])
def test_f_and_d_have_no_opponent_effect(component: ServeComponent) -> None:
    fit = fit_serve_component(
        synthetic_component_counts(
            component,
            repetitions=5,
            server_effects={"p0": 0.7, "p1": -0.7},
        ),
        component=component,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance(component.value),
    )
    first = predict_component(fit, _context("p0", "p1"))
    second = predict_component(fit, _context("p0", "p3"))
    assert first.map_mean == second.map_mean
    assert first.linear_predictor_sd == second.linear_predictor_sd
    assert first.returning_player_seen is None
    assert all(not block.role.startswith("returner") for block in fit.effect_blocks)


def test_effect_blocks_are_explicitly_centered_in_reported_coefficients() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(ServeComponent.A, repetitions=5),
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(event_components=(ServeComponent.A,)),
        provenance=make_provenance("centered"),
    )
    for block in fit.effect_blocks:
        values = [
            coefficient.value
            for coefficient in fit.coefficient_summary
            if coefficient.name.startswith(f"{block.name}[")
        ]
        assert len(values) == len(block.levels)
        assert sum(values) == pytest.approx(0.0, abs=2e-12)
    assert math.isfinite(fit.posterior.condition_number)


def test_centered_fit_is_invariant_to_player_identifier_relabeling() -> None:
    rows = synthetic_component_counts(
        ServeComponent.A,
        repetitions=6,
        server_effects={"p0": 0.7, "p1": -0.4},
        returner_effects={"p2": 0.6, "p3": -0.5},
        seed=909,
    )
    relabel = {"p0": "zz", "p1": "aa", "p2": "yy", "p3": "bb"}
    renamed = rows.copy()
    renamed["player_id"] = renamed["player_id"].map(relabel)
    renamed["opponent_id"] = renamed["opponent_id"].map(relabel)
    kwargs = {
        "component": ServeComponent.A,
        "tour": Tour.ATP,
        "cutoff": TEST_CUTOFF,
        "config": make_model_config(),
        "provenance": make_provenance("relabel"),
    }
    original_fit = fit_serve_component(rows, **kwargs)  # type: ignore[arg-type]
    renamed_fit = fit_serve_component(renamed, **kwargs)  # type: ignore[arg-type]
    for server, returner in (("p0", "p2"), ("p1", "p3"), ("p2", "p0")):
        original = predict_component(original_fit, _context(server, returner)).map_mean
        mapped = predict_component(
            renamed_fit,
            _context(relabel[server], relabel[returner]),
        ).map_mean
        assert mapped == pytest.approx(original, abs=2e-4)


def test_indoor_hard_context_is_supported_only_when_explicitly_enabled() -> None:
    frame = synthetic_component_counts(ServeComponent.F, repetitions=8, seed=314)
    indoor_mask = frame.index % 2 == 0
    frame.loc[indoor_mask, "indoor"] = "Indoor"
    frame.loc[~indoor_mask, "indoor"] = "Outdoor"
    frame.loc[indoor_mask, "surface"] = "Hard"
    frame.loc[~indoor_mask, "surface"] = "Hard"
    frame.loc[indoor_mask, "successes"] = np.minimum(
        frame.loc[indoor_mask, "trials"].astype(int),
        frame.loc[indoor_mask, "successes"].astype(int) + 18,
    )
    config = make_model_config(include_indoor_hard=True)
    fit = fit_serve_component(
        frame,
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
        provenance=make_provenance("indoor"),
    )
    outdoor_context = _context("p0", "p1")
    indoor_context = outdoor_context.model_copy(update={"indoor": True})
    assert "indoor_hard" in fit.fixed_parameter_indices
    assert (
        predict_component(fit, indoor_context).map_mean
        > predict_component(fit, outdoor_context).map_mean
    )

    missing = frame.copy()
    missing.loc[0, "indoor"] = None
    prepared = prepare_component_rows(
        missing,
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
    )
    assert prepared.excluded["missing_indoor_hard_context"] == 1


def test_shrunk_event_year_effect_is_an_explicit_optional_block() -> None:
    frame = synthetic_component_counts(ServeComponent.A, repetitions=8, seed=2718)
    first_event = frame.index % 2 == 0
    frame.loc[first_event, "event"] = "Fast Open"
    frame.loc[~first_event, "event"] = "Slow Open"
    frame.loc[first_event, "successes"] = np.minimum(
        frame.loc[first_event, "trials"].astype(int),
        frame.loc[first_event, "successes"].astype(int) + 12,
    )
    fit = fit_serve_component(
        frame,
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(event_components=(ServeComponent.A,)),
        provenance=make_provenance("event"),
    )
    assert any(block.role == "event_year" for block in fit.effect_blocks)
    fast = _context("p0", "p1").model_copy(update={"event": "Fast Open", "event_year": 2025})
    slow = fast.model_copy(update={"event": "Slow Open"})
    assert predict_component(fit, fast).map_mean > predict_component(fit, slow).map_mean


def test_missing_surface_is_excluded_and_missing_event_is_not_a_pseudo_level() -> None:
    frame = synthetic_component_counts(ServeComponent.A, repetitions=4, seed=616)
    frame.loc[frame.index[:3], "surface"] = None
    frame.loc[frame.index[:3], ["event", "event_year"]] = None
    fit = fit_serve_component(
        frame,
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(event_components=(ServeComponent.A,)),
        provenance=make_provenance("missing-context"),
    )
    assert fit.diagnostics.usable_rows == len(frame) - 3
    exclusions = {item.reason: item.rows for item in fit.diagnostics.exclusion_counts}
    assert exclusions["missing_surface_context"] == 3
    assert all("__missing" not in level for block in fit.effect_blocks for level in block.levels)
    missing_event = _context("p0", "p1").model_copy(update={"event": None, "event_year": None})
    assert predict_component(fit, missing_event).event_year_seen is None


def test_sparse_player_is_shrunk_more_than_high_exposure_player() -> None:
    rows: list[dict[str, object]] = []
    template = synthetic_component_counts(ServeComponent.F, repetitions=1).iloc[0].to_dict()
    match_index = 0

    def add(player: str, successes: int, repetitions: int) -> None:
        nonlocal match_index
        for repetition in range(repetitions):
            row = dict(template)
            row["match_id"] = f"shrink-{match_index}"
            row["source_row_number"] = match_index + 2
            row["player_id"] = player
            row["opponent_id"] = f"opponent-{match_index % 4}"
            row["surface"] = "Hard" if match_index % 2 == 0 else "Clay"
            row["successes"] = (
                successes if repetitions == 1 else successes + (-8, -4, 0, 4, 8)[repetition % 5]
            )
            row["trials"] = 100
            rows.append(row)
            match_index += 1

    add("baseline-a", 50, 18)
    add("baseline-b", 50, 18)
    add("dense-strong", 80, 18)
    add("sparse-strong", 80, 1)
    fit = fit_serve_component(
        pd.DataFrame.from_records(rows),
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("shrinkage"),
    )
    baseline = predict_component(fit, _context("baseline-a", "x")).map_mean
    dense = predict_component(fit, _context("dense-strong", "x")).map_mean
    sparse = predict_component(fit, _context("sparse-strong", "x")).map_mean
    assert dense > sparse > baseline
    information = {item.player_id: item for item in fit.diagnostics.player_information}
    assert information["sparse-strong"].sparse_warning
    assert not information["dense-strong"].sparse_warning


def test_sparse_hard_deviation_partially_pools_toward_global_player_effect() -> None:
    rows: list[dict[str, object]] = []
    template = synthetic_component_counts(ServeComponent.F, repetitions=1).iloc[0].to_dict()
    match_index = 0
    for player in ("target", "control-a", "control-b"):
        for surface, repetitions, successes in (
            ("Clay", 20, 50),
            ("Hard", 1 if player == "target" else 20, 90 if player == "target" else 50),
        ):
            for _ in range(repetitions):
                row = dict(template)
                row["match_id"] = f"surface-{match_index}"
                row["source_row_number"] = match_index + 2
                row["player_id"] = player
                row["opponent_id"] = f"opponent-{match_index % 5}"
                row["surface"] = surface
                row["successes"] = successes
                row["trials"] = 100
                rows.append(row)
                match_index += 1
    fit = fit_serve_component(
        pd.DataFrame.from_records(rows),
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("surface-pooling"),
    )
    clay = predict_component(fit, _context("target", "x", surface="Clay")).map_mean
    hard = predict_component(fit, _context("target", "x", surface="Hard")).map_mean
    assert clay < hard < 0.9
    assert hard - clay < 0.4
    assert fit.diagnostics.usable_rows == len(rows)


def test_map_recovers_controlled_baseline_within_sampling_tolerance() -> None:
    target = 0.64
    fit = fit_serve_component(
        synthetic_component_counts(
            ServeComponent.F,
            repetitions=18,
            trials=120,
            intercept=math.log(target / (1.0 - target)),
            kappa=120.0,
            seed=8787,
        ),
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("recovery"),
    )
    predictions = [
        predict_component(fit, _context(player, "x")).map_mean
        for player in ("p0", "p1", "p2", "p3")
    ]
    assert float(np.mean(predictions)) == pytest.approx(target, abs=0.06)


def test_map_recovers_server_returner_surface_directions_and_kappa_scale() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(
            ServeComponent.A,
            repetitions=24,
            trials=100,
            intercept=-0.6,
            server_effects={"p0": 0.7, "p1": -0.7},
            returner_effects={"p2": 0.6, "p3": -0.6},
            hard_deviations={"p0": 0.5},
            kappa=35.0,
            seed=4242,
        ),
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("controlled-recovery"),
    )
    strong_server_hard = predict_component(fit, _context("p0", "p3", surface="Hard"))
    strong_server_clay = predict_component(fit, _context("p0", "p3", surface="Clay"))
    weak_server_hard = predict_component(fit, _context("p1", "p3", surface="Hard"))
    strong_returner = predict_component(fit, _context("p0", "p2", surface="Hard"))
    assert strong_server_hard.map_mean > strong_server_clay.map_mean
    assert strong_server_hard.map_mean > weak_server_hard.map_mean
    assert strong_returner.map_mean < strong_server_hard.map_mean
    assert fit.kappa == pytest.approx(35.0, rel=0.5)


def test_effective_sample_diagnostics_match_the_frozen_formulas() -> None:
    rows = synthetic_component_counts(ServeComponent.D, repetitions=4, seed=601)
    config = make_model_config()
    fit = fit_serve_component(
        rows,
        component=ServeComponent.D,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
        provenance=make_provenance("information"),
    )
    prepared = prepare_component_rows(
        rows,
        component=ServeComponent.D,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=config,
    )
    player = "p0"
    positions = prepared.frame.index[prepared.frame["player_id"] == player].to_numpy()
    weighted_opportunities = prepared.weights[positions] * prepared.trials[positions]
    expected_weighted = float(np.sum(weighted_opportunities))
    expected_matches = expected_weighted**2 / float(np.sum(weighted_opportunities**2))
    rho = 1.0 / (fit.kappa + 1.0)
    expected_information = float(
        np.sum(
            prepared.weights[positions]
            * prepared.trials[positions]
            / (1.0 + (prepared.trials[positions] - 1.0) * rho)
        )
    )
    diagnostic = next(
        item for item in fit.diagnostics.player_information if item.player_id == player
    )
    assert diagnostic.weighted_trials == pytest.approx(expected_weighted)
    assert diagnostic.effective_matches == pytest.approx(expected_matches)
    assert diagnostic.information_equivalent_trials == pytest.approx(expected_information)


def test_repeated_fit_is_deterministic_with_symmetric_covariance() -> None:
    rows = synthetic_component_counts(ServeComponent.Q1, repetitions=4, seed=777)
    kwargs = {
        "component": ServeComponent.Q1,
        "tour": Tour.ATP,
        "cutoff": TEST_CUTOFF,
        "config": make_model_config(),
        "provenance": make_provenance("reproducible"),
    }
    first = fit_serve_component(rows, **kwargs)  # type: ignore[arg-type]
    second = fit_serve_component(rows, **kwargs)  # type: ignore[arg-type]
    assert first == second
    covariance = first.posterior.covariance_array()
    assert covariance.shape == (
        len(first.posterior.parameter_names),
        len(first.posterior.parameter_names),
    )
    assert covariance == pytest.approx(covariance.T, abs=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_fitted_component_exercises_recorded_diagonal_curvature_fallback() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(ServeComponent.Q2, repetitions=3, seed=271),
        component=ServeComponent.Q2,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(laplace_max_full_parameters=1),
        provenance=make_provenance("diagonal-fit"),
    )
    assert fit.posterior.curvature_kind == "diagonal"
    assert fit.posterior.hessian is None
    assert fit.posterior.covariance is None
    assert all(value > 0.0 for value in fit.posterior.variance_diagonal)
    assert math.isfinite(predict_component(fit, _context("p0", "p1")).linear_predictor_sd)


def test_unseen_levels_carry_explicit_random_effect_variance() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(
            ServeComponent.A,
            repetitions=4,
            surfaces=("Clay",),
            seed=808,
        ),
        component=ServeComponent.A,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(event_components=(ServeComponent.A,)),
        provenance=make_provenance("unseen"),
    )
    known_context = _context("p0", "p1", surface="Clay").model_copy(
        update={"event": "Synthetic Open", "event_year": 2025}
    )
    known = predict_component(fit, known_context)
    unseen = predict_component(fit, _context("new-server", "new-returner", surface="Hard"))
    assert known.unseen_effect_variance == 0.0
    assert unseen.unseen_effect_variance > 0.0
    assert unseen.linear_predictor_sd**2 >= unseen.unseen_effect_variance
    assert not unseen.surface_seen


def test_fitted_mapping_views_are_immutable() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(ServeComponent.F, repetitions=2),
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("immutable"),
    )
    with pytest.raises(TypeError):
        fit.fixed_parameter_indices["changed"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        fit.diagnostics.shrinkage_scales["changed"] = 1.0  # type: ignore[index]


def test_failed_optimizer_never_emits_a_superficially_valid_fit() -> None:
    with pytest.raises(FitConvergenceError):
        fit_serve_component(
            synthetic_component_counts(ServeComponent.A, repetitions=5),
            component=ServeComponent.A,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(max_iterations=1),
            provenance=make_provenance("failed"),
        )


def test_iteration_cap_gets_one_identical_warm_start_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_minimize(_objective: object, initial: np.ndarray, **_kwargs: object) -> OptimizeResult:
        calls.append(initial.copy())
        if len(calls) == 1:
            return OptimizeResult(
                success=False,
                status=1,
                message="iteration cap",
                nit=5,
                nfev=7,
                x=np.asarray([0.25, -0.5]),
            )
        return OptimizeResult(
            success=True,
            status=0,
            message="converged",
            nit=2,
            nfev=3,
            x=np.asarray([0.2, -0.4]),
        )

    monkeypatch.setattr(serve_components_module, "minimize", fake_minimize)
    config = make_model_config(max_iterations=5)
    result = serve_components_module._minimize_map(
        object(),
        np.zeros(2),
        [(None, None), (None, None)],
        config,
    )
    assert len(calls) == 2
    assert calls[1] == pytest.approx([0.25, -0.5])
    assert result.success
    assert result.nit == 7
    assert result.nfev == 10
    assert str(result.message).startswith("DETERMINISTIC_WARM_START_CONTINUATION:")


def test_all_five_components_fit_one_tour_and_form_typed_future_distribution() -> None:
    rows = synthetic_all_component_counts(repetitions=3)
    fits = fit_all_serve_components(
        rows,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("all-five"),
    )
    assert set(fits) == set(ServeComponent)
    assert all(fit.tour is Tour.ATP for fit in fits.values())
    prediction = predict_serve_performance(fits, _context("p0", "p1"))
    assert (
        prediction.fit_identity.data_snapshot_sha256
        == make_provenance("all-five").data_snapshot_sha256
    )
    assert prediction.context.information_cutoff_utc == TEST_CUTOFF
    assert prediction.first_serve_in.component is ServeComponent.F
    assert prediction.ace_given_first_in.component is ServeComponent.A
    assert prediction.returnable_first_win.component is ServeComponent.Q1
    assert prediction.double_fault_given_second_opp.component is ServeComponent.D
    assert prediction.playable_second_win.component is ServeComponent.Q2
    assert all(
        0.0 < item.map_mean < 1.0
        for item in (
            prediction.first_serve_in,
            prediction.ace_given_first_in,
            prediction.returnable_first_win,
            prediction.double_fault_given_second_opp,
            prediction.playable_second_win,
        )
    )


def test_five_component_prediction_rejects_mixed_fit_provenance() -> None:
    fits = fit_all_serve_components(
        synthetic_all_component_counts(repetitions=2),
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("coherent"),
    )
    fits[ServeComponent.A] = fits[ServeComponent.A].model_copy(
        update={"data_snapshot_sha256": "f" * 64}
    )
    with pytest.raises(ModelDataError, match="must share"):
        predict_serve_performance(fits, _context("p0", "p1"))


def test_strict_cutoff_prediction_is_unchanged_when_future_result_changes() -> None:
    before = synthetic_component_counts(ServeComponent.F, repetitions=4, seed=9123)
    future = before.iloc[:1].copy()
    future.loc[:, "match_id"] = "heldout-historical-match"
    future.loc[:, "source_row_number"] = 9999
    future.loc[:, "match_date"] = (TEST_CUTOFF + timedelta(days=2)).date()
    future.loc[:, "source_date"] = (TEST_CUTOFF + timedelta(days=2)).date()
    future.loc[:, "available_at_utc"] = TEST_CUTOFF + timedelta(days=3)
    low_future = pd.concat([before, future.assign(successes=0)], ignore_index=True)
    high_future = pd.concat(
        [before, future.assign(successes=future["trials"].iloc[0])], ignore_index=True
    )
    kwargs = {
        "component": ServeComponent.F,
        "tour": Tour.ATP,
        "cutoff": TEST_CUTOFF,
        "config": make_model_config(),
        "provenance": make_provenance("cutoff-demo"),
    }
    low_fit = fit_serve_component(low_future, **kwargs)  # type: ignore[arg-type]
    high_fit = fit_serve_component(high_future, **kwargs)  # type: ignore[arg-type]
    assert low_fit == high_fit
    assert low_fit.training_data_sha256 == high_fit.training_data_sha256
    assert (
        dict((item.reason, item.rows) for item in low_fit.diagnostics.exclusion_counts)[
            "at_or_after_cutoff"
        ]
        == 1
    )
    assert predict_component(low_fit, _context("p0", "p1")) == predict_component(
        high_fit, _context("p0", "p1")
    )


def test_all_five_historical_cutoff_predictions_ignore_heldout_match_results() -> None:
    before = synthetic_all_component_counts(repetitions=3, seed=7001)
    future = before.groupby("component", sort=False).head(1).copy()
    future.loc[:, "match_id"] = [f"heldout-{component}" for component in future["component"]]
    future.loc[:, "source_row_number"] = range(9001, 9001 + len(future))
    future.loc[:, "match_date"] = (TEST_CUTOFF + timedelta(days=2)).date()
    future.loc[:, "source_date"] = (TEST_CUTOFF + timedelta(days=2)).date()
    future.loc[:, "available_at_utc"] = TEST_CUTOFF + timedelta(days=3)
    low_results = pd.concat([before, future.assign(successes=0)], ignore_index=True)
    high_results = pd.concat(
        [before, future.assign(successes=future["trials"].to_numpy())],
        ignore_index=True,
    )
    kwargs = {
        "tour": Tour.ATP,
        "cutoff": TEST_CUTOFF,
        "config": make_model_config(),
        "provenance": make_provenance("all-five-cutoff-demo"),
    }
    low_fits = fit_all_serve_components(low_results, **kwargs)  # type: ignore[arg-type]
    high_fits = fit_all_serve_components(high_results, **kwargs)  # type: ignore[arg-type]
    assert low_fits == high_fits
    assert predict_serve_performance(low_fits, _context("p0", "p1")) == predict_serve_performance(
        high_fits, _context("p0", "p1")
    )


def test_atp_and_wta_rows_are_never_combined_in_one_fit() -> None:
    atp = synthetic_component_counts(ServeComponent.F, tour=Tour.ATP, repetitions=3)
    wta = synthetic_component_counts(
        ServeComponent.F,
        tour=Tour.WTA,
        repetitions=3,
        intercept=1.2,
        seed=42,
    )
    combined = pd.concat([atp, wta], ignore_index=True)
    atp_fit = fit_serve_component(
        combined,
        component=ServeComponent.F,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("atp"),
    )
    wta_fit = fit_serve_component(
        combined,
        component=ServeComponent.F,
        tour=Tour.WTA,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("wta", tour=Tour.WTA),
    )
    assert atp_fit.tour is Tour.ATP
    assert wta_fit.tour is Tour.WTA
    assert atp_fit.training_data_sha256 != wta_fit.training_data_sha256
    atp_mean = predict_component(atp_fit, _context("p0", "p1", tour=Tour.ATP)).map_mean
    wta_mean = predict_component(wta_fit, _context("p0", "p1", tour=Tour.WTA)).map_mean
    assert wta_mean > atp_mean


def test_fit_provenance_must_match_row_snapshot_and_optional_count_artifact() -> None:
    frame = synthetic_component_counts(ServeComponent.F, repetitions=2)
    mismatched_snapshot = make_provenance("mismatch").model_copy(
        update={"data_snapshot_sha256": "a" * 64}
    )
    with pytest.raises(ModelDataError, match="snapshot hash"):
        fit_serve_component(
            frame,
            component=ServeComponent.F,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(),
            provenance=mismatched_snapshot,
        )

    frame["component_count_artifact_sha256"] = "b" * 64
    with pytest.raises(ModelDataError, match="component-count artifact"):
        fit_serve_component(
            frame,
            component=ServeComponent.F,
            tour=Tour.ATP,
            cutoff=TEST_CUTOFF,
            config=make_model_config(),
            provenance=make_provenance("mismatch"),
        )


def test_fit_hashes_are_exact_sha256_values() -> None:
    fit = fit_serve_component(
        synthetic_component_counts(ServeComponent.D, repetitions=2),
        component=ServeComponent.D,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("hashes"),
    )
    assert len(fit.training_data_sha256) == 64
    int(fit.training_data_sha256, 16)
    assert (
        fit.model_config_sha256
        == hashlib.sha256(
            (
                json.dumps(
                    fit.config.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
    )
