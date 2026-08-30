from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import nextafter

import numpy as np
import pytest

from tennis_model.estimation.derived import PrimitiveServeMeans
from tennis_model.simulation.point import (
    PointUniforms,
    ServeNumber,
    ServePerformanceDraw,
    ServicePointBranch,
    ServicePointResult,
    aggregate_service_points,
    generate_point_from_uniforms,
    generate_service_point,
)
from tennis_model.simulation.scoring import award_point, new_match


def _performance(**overrides: float) -> ServePerformanceDraw:
    values = {
        "first_serve_in": 0.62,
        "ace_given_first_in": 0.11,
        "returnable_first_win": 0.67,
        "double_fault_given_second_opp": 0.09,
        "playable_second_win": 0.56,
    }
    values.update(overrides)
    return ServePerformanceDraw(**values)


def _uniforms(**overrides: float) -> PointUniforms:
    values = {
        "first_serve_in": 0.1,
        "ace_given_first_in": 0.1,
        "returnable_first_win": 0.1,
        "double_fault_given_second_opp": 0.1,
        "playable_second_win": 0.1,
    }
    values.update(overrides)
    return PointUniforms(**values)


def _point(
    performance: ServePerformanceDraw,
    uniforms: PointUniforms | None = None,
) -> ServicePointResult:
    return generate_point_from_uniforms(
        performance,
        _uniforms() if uniforms is None else uniforms,
        server_id="server",
        receiver_id="receiver",
    )


@pytest.mark.parametrize(
    ("performance", "uniforms", "branch", "server_won"),
    [
        (
            _performance(first_serve_in=1.0, ace_given_first_in=1.0),
            _uniforms(),
            ServicePointBranch.ACE,
            True,
        ),
        (
            _performance(
                first_serve_in=1.0,
                ace_given_first_in=0.0,
                returnable_first_win=1.0,
            ),
            _uniforms(),
            ServicePointBranch.RETURNABLE_FIRST_SERVE,
            True,
        ),
        (
            _performance(
                first_serve_in=1.0,
                ace_given_first_in=0.0,
                returnable_first_win=0.0,
            ),
            _uniforms(),
            ServicePointBranch.RETURNABLE_FIRST_SERVE,
            False,
        ),
        (
            _performance(first_serve_in=0.0, double_fault_given_second_opp=1.0),
            _uniforms(),
            ServicePointBranch.DOUBLE_FAULT,
            False,
        ),
        (
            _performance(
                first_serve_in=0.0,
                double_fault_given_second_opp=0.0,
                playable_second_win=1.0,
            ),
            _uniforms(),
            ServicePointBranch.PLAYABLE_SECOND_SERVE,
            True,
        ),
        (
            _performance(
                first_serve_in=0.0,
                double_fault_given_second_opp=0.0,
                playable_second_win=0.0,
            ),
            _uniforms(),
            ServicePointBranch.PLAYABLE_SECOND_SERVE,
            False,
        ),
    ],
)
def test_all_causal_branches_have_exact_support_and_flags(
    performance: ServePerformanceDraw,
    uniforms: PointUniforms,
    branch: ServicePointBranch,
    server_won: bool,
) -> None:
    result = _point(performance, uniforms)

    assert result.branch is branch
    assert result.server_won is server_won
    assert result.winner_id == ("server" if server_won else "receiver")
    assert result.ace is (branch is ServicePointBranch.ACE)
    assert result.double_fault is (branch is ServicePointBranch.DOUBLE_FAULT)
    assert result.returnable_first_serve is (branch is ServicePointBranch.RETURNABLE_FIRST_SERVE)
    assert result.playable_second_serve is (branch is ServicePointBranch.PLAYABLE_SECOND_SERVE)
    assert result.q1_used is (branch is ServicePointBranch.RETURNABLE_FIRST_SERVE)
    assert result.q2_used is (branch is ServicePointBranch.PLAYABLE_SECOND_SERVE)
    assert result.rally_eligible is (
        branch
        in (
            ServicePointBranch.RETURNABLE_FIRST_SERVE,
            ServicePointBranch.PLAYABLE_SECOND_SERVE,
        )
    )

    if result.first_serve_in:
        assert result.serve_number is ServeNumber.FIRST
        assert not result.double_fault
        assert not result.playable_second_serve
        assert not result.q2_used
    else:
        assert result.serve_number is ServeNumber.SECOND
        assert not result.ace
        assert not result.returnable_first_serve
        assert not result.q1_used

    if result.ace:
        assert result.first_serve_in
        assert result.server_won
        assert not result.rally_eligible
    if result.double_fault:
        assert not result.first_serve_in
        assert not result.server_won
        assert not result.rally_eligible


def test_probability_boundaries_and_strict_uniform_thresholds() -> None:
    near_one = nextafter(1.0, 0.0)

    # At p=0 the event never happens, even for u=0; at p=1 every legal u succeeds.
    zero_first = _point(_performance(first_serve_in=0.0), _uniforms(first_serve_in=0.0))
    one_first = _point(
        _performance(first_serve_in=1.0, ace_given_first_in=1.0),
        _uniforms(first_serve_in=near_one, ace_given_first_in=near_one),
    )
    assert not zero_first.first_serve_in
    assert one_first.first_serve_in and one_first.ace

    # The implementation contract is u < p, not u <= p, at every primitive draw.
    equal_f = _point(
        _performance(first_serve_in=0.5, double_fault_given_second_opp=0.0),
        _uniforms(first_serve_in=0.5),
    )
    equal_a = _point(
        _performance(first_serve_in=1.0, ace_given_first_in=0.5),
        _uniforms(ace_given_first_in=0.5),
    )
    equal_q1 = _point(
        _performance(
            first_serve_in=1.0,
            ace_given_first_in=0.0,
            returnable_first_win=0.5,
        ),
        _uniforms(returnable_first_win=0.5),
    )
    equal_d = _point(
        _performance(first_serve_in=0.0, double_fault_given_second_opp=0.5),
        _uniforms(double_fault_given_second_opp=0.5),
    )
    equal_q2 = _point(
        _performance(
            first_serve_in=0.0,
            double_fault_given_second_opp=0.0,
            playable_second_win=0.5,
        ),
        _uniforms(playable_second_win=0.5),
    )
    assert not equal_f.first_serve_in
    assert not equal_a.ace
    assert not equal_q1.server_won
    assert not equal_d.double_fault
    assert not equal_q2.server_won


def test_production_generator_traverses_every_deterministic_boundary_branch() -> None:
    def generate(performance: ServePerformanceDraw) -> tuple[ServicePointResult, ...]:
        rng = np.random.default_rng(20260829)
        return tuple(generate_service_point(performance, rng) for _ in range(100))

    all_aces = generate(_performance(first_serve_in=1.0, ace_given_first_in=1.0))
    assert all(point.branch is ServicePointBranch.ACE and point.server_won for point in all_aces)

    no_aces = generate(_performance(ace_given_first_in=0.0))
    assert all(not point.ace for point in no_aces)

    all_double_faults = generate(
        _performance(first_serve_in=0.0, double_fault_given_second_opp=1.0)
    )
    assert all(
        point.branch is ServicePointBranch.DOUBLE_FAULT and not point.server_won
        for point in all_double_faults
    )

    no_double_faults = generate(_performance(double_fault_given_second_opp=0.0))
    assert all(not point.double_fault for point in no_double_faults)
    assert all(
        point.first_serve_in or point.branch is ServicePointBranch.PLAYABLE_SECOND_SERVE
        for point in no_double_faults
    )

    deterministic_q1 = generate(
        _performance(
            first_serve_in=1.0,
            ace_given_first_in=0.0,
            returnable_first_win=1.0,
        )
    )
    assert all(
        point.branch is ServicePointBranch.RETURNABLE_FIRST_SERVE and point.server_won
        for point in deterministic_q1
    )

    deterministic_q2 = generate(
        _performance(
            first_serve_in=0.0,
            double_fault_given_second_opp=0.0,
            playable_second_win=0.0,
        )
    )
    assert all(
        point.branch is ServicePointBranch.PLAYABLE_SECOND_SERVE and not point.server_won
        for point in deterministic_q2
    )


def test_uniforms_on_unreached_branches_cannot_change_a_terminal_point() -> None:
    ace_performance = _performance(first_serve_in=1.0, ace_given_first_in=1.0)
    ace_left = generate_point_from_uniforms(
        ace_performance,
        _uniforms(
            returnable_first_win=0.0,
            double_fault_given_second_opp=0.0,
            playable_second_win=0.0,
        ),
    )
    ace_right = generate_point_from_uniforms(
        ace_performance,
        _uniforms(
            returnable_first_win=nextafter(1.0, 0.0),
            double_fault_given_second_opp=nextafter(1.0, 0.0),
            playable_second_win=nextafter(1.0, 0.0),
        ),
    )
    assert ace_left == ace_right

    double_fault_performance = _performance(
        first_serve_in=0.0,
        double_fault_given_second_opp=1.0,
    )
    double_fault_left = generate_point_from_uniforms(
        double_fault_performance,
        _uniforms(
            ace_given_first_in=0.0,
            returnable_first_win=0.0,
            playable_second_win=0.0,
        ),
    )
    double_fault_right = generate_point_from_uniforms(
        double_fault_performance,
        _uniforms(
            ace_given_first_in=nextafter(1.0, 0.0),
            returnable_first_win=nextafter(1.0, 0.0),
            playable_second_win=nextafter(1.0, 0.0),
        ),
    )
    assert double_fault_left == double_fault_right


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        (field, bad)
        for field in (
            "first_serve_in",
            "ace_given_first_in",
            "returnable_first_win",
            "double_fault_given_second_opp",
            "playable_second_win",
        )
        for bad in (-0.01, 1.01, float("nan"), float("inf"), float("-inf"))
    ],
)
def test_performance_draw_rejects_out_of_support_values(field: str, bad: float) -> None:
    with pytest.raises(ValueError):
        _performance(**{field: bad})


@pytest.mark.parametrize("bad", [True, False, "0.5", None])
def test_performance_draw_rejects_non_float_probabilities(bad: object) -> None:
    values: dict[str, object] = {
        "first_serve_in": 0.5,
        "ace_given_first_in": 0.1,
        "returnable_first_win": 0.6,
        "double_fault_given_second_opp": 0.1,
        "playable_second_win": bad,
    }
    with pytest.raises(TypeError):
        ServePerformanceDraw(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.01, float("nan"), float("inf")])
def test_coupled_uniforms_reject_values_outside_half_open_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError):
        _uniforms(playable_second_win=bad)


@pytest.mark.parametrize("bad", [True, False, "0.5", None])
def test_coupled_uniforms_reject_non_float_values(bad: object) -> None:
    values: dict[str, object] = {
        "first_serve_in": 0.5,
        "ace_given_first_in": 0.1,
        "returnable_first_win": 0.6,
        "double_fault_given_second_opp": 0.1,
        "playable_second_win": bad,
    }
    with pytest.raises(TypeError):
        PointUniforms(**values)  # type: ignore[arg-type]


def test_typed_performance_draw_cannot_be_replaced_by_derived_or_aggregate_input() -> None:
    aggregate = PrimitiveServeMeans(0.62, 0.11, 0.67, 0.09, 0.56)
    with pytest.raises(TypeError):
        generate_service_point(aggregate, np.random.default_rng(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        generate_point_from_uniforms(aggregate, _uniforms())  # type: ignore[arg-type]


def test_performance_draw_exposes_only_the_frozen_derived_reporting_identities() -> None:
    performance = _performance()
    assert performance.first_serve_win == pytest.approx(0.11 + 0.89 * 0.67)
    assert performance.second_serve_win == pytest.approx(0.91 * 0.56)
    assert performance.service_point_win == pytest.approx(
        0.62 * performance.first_serve_win + 0.38 * performance.second_serve_win
    )
    assert performance.ace_rate == pytest.approx(0.62 * 0.11)
    assert performance.double_fault_rate == pytest.approx(0.38 * 0.09)


@pytest.mark.parametrize("rng", [None, 17, np.random])
def test_generator_requires_an_explicit_numpy_generator(rng: object) -> None:
    with pytest.raises(TypeError):
        generate_service_point(_performance(), rng)  # type: ignore[arg-type]


def test_seeded_generators_reproduce_the_exact_point_path() -> None:
    performance = _performance()
    left = np.random.default_rng(20260828)
    right = np.random.default_rng(20260828)

    left_path = [generate_service_point(performance, left) for _ in range(250)]
    right_path = [generate_service_point(performance, right) for _ in range(250)]

    assert left_path == right_path


@pytest.mark.parametrize(
    "ids",
    [
        {"server_id": "server"},
        {"receiver_id": "receiver"},
        {"server_id": "same", "receiver_id": "same"},
        {"server_id": "", "receiver_id": "receiver"},
    ],
)
def test_invalid_ids_fail_before_the_generator_consumes_rng(
    ids: dict[str, str],
) -> None:
    actual = np.random.default_rng(8675309)
    with pytest.raises(ValueError):
        generate_service_point(_performance(), actual, **ids)  # type: ignore[arg-type]

    baseline = np.random.default_rng(8675309)
    assert actual.random() == baseline.random()


@pytest.mark.parametrize(
    ("performance", "draws_consumed"),
    [
        (_performance(first_serve_in=1.0, ace_given_first_in=1.0), 2),
        (_performance(first_serve_in=1.0, ace_given_first_in=0.0), 3),
        (_performance(first_serve_in=0.0, double_fault_given_second_opp=1.0), 2),
        (_performance(first_serve_in=0.0, double_fault_given_second_opp=0.0), 3),
    ],
)
def test_generator_consumes_rng_lazily_after_terminal_causal_events(
    performance: ServePerformanceDraw,
    draws_consumed: int,
) -> None:
    actual = np.random.default_rng(314159)
    generate_service_point(performance, actual)
    actual_next = actual.random()

    expected = np.random.default_rng(314159)
    expected.random(draws_consumed)
    expected_next = expected.random()

    assert actual_next == expected_next


def test_explicit_generator_does_not_touch_numpy_global_random_state() -> None:
    state = np.random.get_state()
    try:
        np.random.seed(9876)
        expected = np.random.random()
        np.random.seed(9876)
        generate_service_point(_performance(), np.random.default_rng(55))
        observed = np.random.random()
    finally:
        np.random.set_state(state)
    assert observed == expected


def test_performance_uniforms_and_results_are_immutable() -> None:
    performance = _performance()
    uniforms = _uniforms()
    result = _point(performance, uniforms)

    with pytest.raises(FrozenInstanceError):
        performance.first_serve_in = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        uniforms.first_serve_in = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.server_won = not result.server_won  # type: ignore[misc]


def test_result_constructor_rejects_impossible_causal_states_and_id_pairs() -> None:
    valid_ace = _point(
        _performance(first_serve_in=1.0, ace_given_first_in=1.0),
    )
    for changes in (
        {"first_serve_in": False},
        {"server_won": False},
        {"double_fault": True},
        {"rally_eligible": True},
        {"serve_number": ServeNumber.SECOND},
        {"branch": ServicePointBranch.DOUBLE_FAULT},
    ):
        with pytest.raises(ValueError):
            replace(valid_ace, **changes)  # type: ignore[arg-type]


def test_result_constructor_rejects_impossible_non_ace_branch_states() -> None:
    returnable_first = _point(
        _performance(
            first_serve_in=1.0,
            ace_given_first_in=0.0,
            returnable_first_win=1.0,
        )
    )
    double_fault = _point(_performance(first_serve_in=0.0, double_fault_given_second_opp=1.0))
    playable_second = _point(
        _performance(
            first_serve_in=0.0,
            double_fault_given_second_opp=0.0,
            playable_second_win=1.0,
        )
    )

    impossible_changes = (
        (returnable_first, {"first_serve_in": False}),
        (returnable_first, {"playable_second_serve": True}),
        (returnable_first, {"rally_eligible": False}),
        (double_fault, {"first_serve_in": True}),
        (double_fault, {"server_won": True}),
        (double_fault, {"rally_eligible": True}),
        (playable_second, {"first_serve_in": True}),
        (playable_second, {"returnable_first_serve": True}),
        (playable_second, {"rally_eligible": False}),
    )
    for point, changes in impossible_changes:
        with pytest.raises(ValueError):
            replace(point, **changes)  # type: ignore[arg-type]


def test_result_requires_server_and_receiver_ids_jointly_and_distinct() -> None:
    performance = _performance(first_serve_in=1.0, ace_given_first_in=1.0)
    uniforms = _uniforms()
    with pytest.raises(ValueError):
        generate_point_from_uniforms(performance, uniforms, server_id="server")
    with pytest.raises(ValueError):
        generate_point_from_uniforms(performance, uniforms, receiver_id="receiver")
    with pytest.raises(ValueError):
        generate_point_from_uniforms(
            performance,
            uniforms,
            server_id="same",
            receiver_id="same",
        )


def test_result_without_optional_ids_has_no_invented_winner_identity() -> None:
    result = generate_point_from_uniforms(
        _performance(first_serve_in=1.0, ace_given_first_in=1.0),
        _uniforms(),
    )
    assert result.server_id is None
    assert result.receiver_id is None
    assert result.winner_id is None


def test_aggregation_reconstructs_all_frozen_serve_counts_and_identities() -> None:
    points = (
        _point(_performance(first_serve_in=1.0, ace_given_first_in=1.0)),
        _point(
            _performance(
                first_serve_in=1.0,
                ace_given_first_in=0.0,
                returnable_first_win=1.0,
            )
        ),
        _point(
            _performance(
                first_serve_in=1.0,
                ace_given_first_in=0.0,
                returnable_first_win=0.0,
            )
        ),
        _point(_performance(first_serve_in=0.0, double_fault_given_second_opp=1.0)),
        _point(
            _performance(
                first_serve_in=0.0,
                double_fault_given_second_opp=0.0,
                playable_second_win=1.0,
            )
        ),
        _point(
            _performance(
                first_serve_in=0.0,
                double_fault_given_second_opp=0.0,
                playable_second_win=0.0,
            )
        ),
    )

    counts = aggregate_service_points(points)

    assert counts.service_points == 6
    assert counts.first_serves_in == 3
    assert counts.aces == 1
    assert counts.q1_trials == 2
    assert counts.q1_wins == 1
    assert counts.second_serve_opportunities == 3
    assert counts.double_faults == 1
    assert counts.q2_trials == 2
    assert counts.q2_wins == 1
    assert counts.first_serve_points_won == 2
    assert counts.second_serve_points_won == 1
    assert counts.server_points_won == 3

    assert counts.service_points == counts.first_serves_in + counts.second_serve_opportunities
    assert counts.first_serves_in == counts.aces + counts.q1_trials
    assert counts.first_serve_points_won == counts.aces + counts.q1_wins
    assert counts.second_serve_opportunities == counts.double_faults + counts.q2_trials
    assert counts.second_serve_points_won == counts.q2_wins

    with pytest.raises(ValueError):
        replace(counts, q1_trials=counts.q1_trials + 1)
    with pytest.raises(TypeError):
        aggregate_service_points((*points, object()))  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        counts.aces = 99  # type: ignore[misc]


def test_empty_aggregation_is_zero_without_converting_missing_observations() -> None:
    counts = aggregate_service_points(())
    assert counts.service_points == 0
    assert counts.first_serves_in == 0
    assert counts.aces == 0
    assert counts.q1_trials == 0
    assert counts.q1_wins == 0
    assert counts.second_serve_opportunities == 0
    assert counts.double_faults == 0
    assert counts.q2_trials == 0
    assert counts.q2_wins == 0
    assert counts.first_serve_points_won == 0
    assert counts.second_serve_points_won == 0


def test_point_generation_has_clean_scoring_and_rally_module_boundaries() -> None:
    ace = _point(_performance(first_serve_in=1.0, ace_given_first_in=1.0))
    rally_point = _point(
        _performance(
            first_serve_in=1.0,
            ace_given_first_in=0.0,
            returnable_first_win=0.0,
        )
    )

    # The point generator identifies rally eligibility but does not invent a later rally category.
    assert not ace.rally_eligible
    assert rally_point.rally_eligible
    assert not hasattr(ace, "rally_category")

    # Scoring remains a separate deterministic consumer of the already-realized winner.
    score = new_match("server", "receiver", best_of=3, first_server_index=0)
    transition = award_point(score, 0 if ace.server_won else 1)
    assert transition.before is score
    assert score.total_points_played == 0
    assert transition.after.total_points_played == 1
    assert not hasattr(ace, "game_score")
    assert not hasattr(ace, "tiebreak")
