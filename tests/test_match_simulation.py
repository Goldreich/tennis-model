from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tennis_model.estimation.retirement import (
    CompetingRetirementOutcome,
    CompetingRiskProbabilities,
    RetirementBoundaryDraw,
)
from tennis_model.props import (
    CANONICAL_SETTLEMENT_POLICY,
    ComparisonOperator,
    PolicyBlockedError,
)
from tennis_model.simulation import (
    AND,
    DURATION_MIN,
    MATCH_WIN,
    OR,
    PLAYER_WINS_SET,
    BreakEvent,
    CorePropEstimate,
    MatchPath,
    PlayerMatchStats,
    PropSpec,
    SimulationBatch,
    evaluate_prop,
    evaluate_settlement,
)
from tennis_model.simulation.match import _simulate_one_path
from tennis_model.simulation.point import ServePerformanceDraw
from tennis_model.simulation.scoring import SetResult


def _stats(
    player_id: str,
    *,
    games_won: int,
    service_games: int,
    return_games: int,
    holds: int,
    breaks: int,
    break_points: int,
    aces: int = 2,
    double_faults: int = 1,
) -> PlayerMatchStats:
    service_points = 40
    first_serves_in = 30
    returnable_trials = first_serves_in - aces
    returnable_wins = min(18, returnable_trials)
    second_opportunities = service_points - first_serves_in
    playable_trials = second_opportunities - double_faults
    playable_wins = min(5, playable_trials)
    return PlayerMatchStats(
        player_id=player_id,
        games_won=games_won,
        service_games_played=service_games,
        return_games_played=return_games,
        service_games_held=holds,
        breaks_conceded=service_games - holds,
        breaks_achieved=breaks,
        service_points=service_points,
        first_serve_opportunities=service_points,
        first_serves_in=first_serves_in,
        first_serve_points_won=aces + returnable_wins,
        returnable_first_serve_trials=returnable_trials,
        returnable_first_serve_wins=returnable_wins,
        second_serve_opportunities=second_opportunities,
        double_faults=double_faults,
        playable_second_serve_trials=playable_trials,
        playable_second_serve_wins=playable_wins,
        second_serve_points_won=playable_wins,
        aces=aces,
        break_point_opportunities=break_points,
    )


def _break_event(*, set_number: int, game_number: int, match_game_number: int) -> BreakEvent:
    return BreakEvent(
        set_number=set_number,
        game_number=game_number,
        match_game_number=match_game_number,
        server_id="B",
        receiver_id="A",
        break_player_id="A",
    )


def _completed_path() -> MatchPath:
    return MatchPath(
        winner_id="A",
        player_a_id="A",
        player_b_id="B",
        best_of=3,
        first_server_id="A",
        sets=(
            SetResult(set_number=1, first_server_index=0, games=(6, 4)),
            SetResult(set_number=2, first_server_index=0, games=(6, 4)),
        ),
        player_stats={
            "A": _stats(
                "A",
                games_won=12,
                service_games=10,
                return_games=10,
                holds=10,
                breaks=2,
                break_points=2,
                aces=8,
                double_faults=1,
            ),
            "B": _stats(
                "B",
                games_won=8,
                service_games=10,
                return_games=10,
                holds=8,
                breaks=0,
                break_points=0,
                aces=3,
                double_faults=4,
            ),
        },
        break_events=(
            _break_event(set_number=1, game_number=2, match_game_number=2),
            _break_event(set_number=2, game_number=2, match_game_number=12),
        ),
        total_games=20,
    )


def _batch(*paths: MatchPath) -> SimulationBatch:
    context = type(
        "Context",
        (),
        {"best_of": 3, "player_a_id": "A", "player_b_id": "B"},
    )()
    return SimulationBatch(
        context=context,
        n_paths=len(paths),
        seed_id="fixture",
        paths=paths,
    )


def _retired_path() -> MatchPath:
    return MatchPath(
        winner_id="A",
        player_a_id="A",
        player_b_id="B",
        best_of=3,
        first_server_id="A",
        sets=(SetResult(set_number=1, first_server_index=0, games=(6, 2)),),
        player_stats={
            "A": _stats(
                "A",
                games_won=6,
                service_games=4,
                return_games=4,
                holds=4,
                breaks=2,
                break_points=2,
                aces=8,
            ),
            "B": _stats(
                "B",
                games_won=2,
                service_games=4,
                return_games=4,
                holds=2,
                breaks=0,
                break_points=0,
                aces=1,
            ),
        },
        break_events=(
            _break_event(set_number=1, game_number=2, match_game_number=2),
            _break_event(set_number=1, game_number=6, match_game_number=6),
        ),
        total_games=8,
        completed=False,
        retired_player_id="B",
        sets_started=1,
    )


def test_fixed_match_path_records_exact_game_and_serve_statistics() -> None:
    always_loses = ServePerformanceDraw(1.0, 0.0, 0.0, 0.0, 0.0)
    always_wins = ServePerformanceDraw(1.0, 1.0, 0.0, 0.0, 0.0)

    path = _simulate_one_path(
        "A",
        "B",
        best_of=3,
        first_server_id="A",
        player_a_performance=always_loses,
        player_b_performance=always_wins,
        rng=np.random.default_rng(7),
        trace_points=True,
    )

    assert path.winner_id == "B"
    assert tuple(result.games for result in path.sets) == ((0, 6), (0, 6))
    assert [
        (event.set_number, event.game_number, event.match_game_number)
        for event in path.break_events
    ] == [
        (1, 1, 1),
        (1, 3, 3),
        (1, 5, 5),
        (2, 1, 7),
        (2, 3, 9),
        (2, 5, 11),
    ]
    assert path.player_stats["A"].service_games_held == 0
    assert path.player_stats["A"].breaks_conceded == 6
    assert path.player_stats["B"].service_games_held == 6
    assert path.player_stats["B"].breaks_achieved == 6
    assert path.player_stats["A"].first_serve_opportunities == 24
    assert path.player_stats["B"].first_serve_opportunities == 24
    assert path.point_trace is not None and len(path.point_trace) == 48


def test_fixed_match_path_obeys_best_of_five_termination() -> None:
    always_loses = ServePerformanceDraw(1.0, 0.0, 0.0, 0.0, 0.0)
    always_wins = ServePerformanceDraw(1.0, 1.0, 0.0, 0.0, 0.0)

    path = _simulate_one_path(
        "A",
        "B",
        best_of=5,
        first_server_id="A",
        player_a_performance=always_loses,
        player_b_performance=always_wins,
        rng=np.random.default_rng(8),
    )

    assert path.winner_id == "B"
    assert tuple(result.games for result in path.sets) == ((0, 6), (0, 6), (0, 6))
    assert path.player_stats["B"].breaks_achieved == 9
    assert path.total_games == 18


def test_zero_retirement_hazard_is_path_identical_and_does_not_advance_rng() -> None:
    performance_a = ServePerformanceDraw(0.63, 0.08, 0.68, 0.04, 0.57)
    performance_b = ServePerformanceDraw(0.61, 0.07, 0.65, 0.05, 0.55)
    disabled = _simulate_one_path(
        "A",
        "B",
        best_of=3,
        first_server_id="A",
        player_a_performance=performance_a,
        player_b_performance=performance_b,
        rng=np.random.default_rng(8201),
    )
    retirement_rng = np.random.default_rng(8202)
    state_before = retirement_rng.bit_generator.state
    zero = _simulate_one_path(
        "A",
        "B",
        best_of=3,
        first_server_id="A",
        player_a_performance=performance_a,
        player_b_performance=performance_b,
        rng=np.random.default_rng(8201),
        retirement_intensities=(0.0, 0.0),
        retirement_rng=retirement_rng,
        retirement_scenario_ids=("zero-a", "zero-b"),
    )
    assert retirement_rng.bit_generator.state == state_before
    assert (
        replace(
            zero,
            retirement_intensities=None,
            retirement_scenario_ids=None,
        )
        == disabled
    )


def test_retirement_check_bypasses_match_winning_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def no_retirement(
        intensity_a: float,
        intensity_b: float,
        rng: np.random.Generator,
    ) -> RetirementBoundaryDraw:
        nonlocal calls
        calls += 1
        assert intensity_a == intensity_b == 0.01
        assert isinstance(rng, np.random.Generator)
        return RetirementBoundaryDraw(
            outcome=CompetingRetirementOutcome.NO_RETIREMENT,
            probabilities=CompetingRiskProbabilities(
                p_no_retirement=1.0,
                p_player_a_retires=0.0,
                p_player_b_retires=0.0,
            ),
        )

    monkeypatch.setattr("tennis_model.simulation.match.draw_competing_retirement", no_retirement)
    path = _simulate_one_path(
        "A",
        "B",
        best_of=3,
        first_server_id="A",
        player_a_performance=ServePerformanceDraw(1.0, 1.0, 0.0, 0.0, 0.0),
        player_b_performance=ServePerformanceDraw(1.0, 0.0, 0.0, 0.0, 0.0),
        rng=np.random.default_rng(1),
        retirement_intensities=(0.01, 0.01),
        retirement_rng=np.random.default_rng(2),
        retirement_scenario_ids=("central", "central"),
    )
    assert path.completed
    assert calls == path.total_games - 1


def test_retirement_after_first_game_preserves_partial_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def player_a_retires(
        intensity_a: float,
        intensity_b: float,
        rng: np.random.Generator,
    ) -> RetirementBoundaryDraw:
        del intensity_a, intensity_b, rng
        return RetirementBoundaryDraw(
            outcome=CompetingRetirementOutcome.PLAYER_A_RETIRES,
            probabilities=CompetingRiskProbabilities(
                p_no_retirement=0.0,
                p_player_a_retires=1.0,
                p_player_b_retires=0.0,
            ),
        )

    monkeypatch.setattr("tennis_model.simulation.match.draw_competing_retirement", player_a_retires)
    path = _simulate_one_path(
        "A",
        "B",
        best_of=3,
        first_server_id="A",
        player_a_performance=ServePerformanceDraw(1.0, 1.0, 0.0, 0.0, 0.0),
        player_b_performance=ServePerformanceDraw(1.0, 0.0, 0.0, 0.0, 0.0),
        rng=np.random.default_rng(3),
        retirement_intensities=(1.0, 1.0),
        retirement_rng=np.random.default_rng(4),
        retirement_scenario_ids=("central", "central"),
    )
    assert not path.completed
    assert path.retired_player_id == "A"
    assert path.advancing_player_id == "B"
    assert path.retirement_game_number == path.total_games == 1
    assert path.sets == ()
    assert path.sets_started == 1
    assert sum(item.games_won for item in path.player_stats.values()) == 1


def test_compound_props_use_the_same_joint_paths() -> None:
    batch = _batch(_completed_path())
    at_least_ten_games = PropSpec(
        kind="PLAYER_GAMES",
        subject_ids=("A",),
        operator=ComparisonOperator.AT_LEAST,
        threshold=10,
    )
    at_least_twenty_one_games = replace(at_least_ten_games, threshold=21)

    yes = evaluate_prop(AND(MATCH_WIN("A"), at_least_ten_games), batch)
    no = evaluate_prop(OR(MATCH_WIN("B"), at_least_twenty_one_games), batch)

    assert isinstance(yes, CorePropEstimate)
    assert yes.probability_raw == 1.0
    assert no.probability_raw == 0.0


def test_retirement_walkover_and_monotone_threshold_settlement() -> None:
    retired_batch = _batch(_retired_path())

    assert (
        evaluate_settlement(MATCH_WIN("A"), retired_batch, CANONICAL_SETTLEMENT_POLICY).yes_paths
        == 1
    )
    assert (
        evaluate_settlement(MATCH_WIN("B"), retired_batch, CANONICAL_SETTLEMENT_POLICY).no_paths
        == 1
    )
    assert (
        evaluate_settlement(
            PLAYER_WINS_SET("A"), retired_batch, CANONICAL_SETTLEMENT_POLICY
        ).yes_paths
        == 1
    )
    assert (
        evaluate_settlement(
            PLAYER_WINS_SET("B"), retired_batch, CANONICAL_SETTLEMENT_POLICY
        ).void_paths
        == 1
    )

    ace_threshold = PropSpec(
        kind="PLAYER_ACES",
        subject_ids=("A",),
        operator=ComparisonOperator.AT_LEAST,
        threshold=8,
    )
    unreached_games = PropSpec(
        kind="TOTAL_GAMES",
        operator=ComparisonOperator.AT_LEAST,
        threshold=9,
    )
    assert (
        evaluate_settlement(ace_threshold, retired_batch, CANONICAL_SETTLEMENT_POLICY).yes_paths
        == 1
    )
    assert (
        evaluate_settlement(unreached_games, retired_batch, CANONICAL_SETTLEMENT_POLICY).void_paths
        == 1
    )

    walkover = MatchPath(
        winner_id=None,
        player_a_id="A",
        player_b_id="B",
        best_of=3,
        first_server_id="A",
        sets=(),
        player_stats={},
        completed=False,
        started=False,
        walkover=True,
    )
    result = evaluate_settlement(MATCH_WIN("A"), _batch(walkover), CANONICAL_SETTLEMENT_POLICY)
    assert result.void_paths == 1
    assert result.settled_paths == 0
    assert result.probability_settled == 0.0


def test_settled_probability_fields_use_the_specified_denominators() -> None:
    estimate = evaluate_settlement(
        PLAYER_WINS_SET("B"),
        _batch(_completed_path(), _retired_path()),
        CANONICAL_SETTLEMENT_POLICY,
    )

    assert estimate.yes_paths == 0
    assert estimate.no_paths == 1
    assert estimate.void_paths == 1
    assert estimate.probability_raw == 0.0
    assert estimate.probability_settled == 0.5


def test_first_serve_percentage_uses_exact_ratio_without_rounding() -> None:
    path = _completed_path()
    stats = path.player_stats["A"]
    ambiguous = replace(
        stats,
        service_points=4,
        first_serve_opportunities=4,
        first_serves_in=3,
        first_serve_points_won=2,
        returnable_first_serve_trials=3,
        returnable_first_serve_wins=2,
        second_serve_opportunities=1,
        double_faults=0,
        playable_second_serve_trials=1,
        playable_second_serve_wins=1,
        second_serve_points_won=1,
        aces=0,
    )
    path = replace(path, player_stats={**path.player_stats, "A": ambiguous})
    prop = PropSpec(
        kind="FIRST_SERVE_WIN_PCT",
        subject_ids=("A",),
        operator=ComparisonOperator.AT_LEAST,
        threshold=67,
    )

    estimate = evaluate_settlement(
        prop,
        _batch(path),
        CANONICAL_SETTLEMENT_POLICY,
    )
    assert estimate.unresolved_paths == 0
    assert estimate.settled_paths == 1
    assert estimate.no_paths == 1
    evaluated = evaluate_prop(prop, _batch(path))
    assert evaluated.probability_raw == 0.0


def test_prop_policy_version_mismatch_is_blocked() -> None:
    prop = replace(MATCH_WIN("A"), settlement_policy_version="different-policy")
    with pytest.raises(PolicyBlockedError, match="versions differ"):
        evaluate_settlement(prop, _batch(_completed_path()), CANONICAL_SETTLEMENT_POLICY)


def test_prop_spec_rejects_unsupported_or_incomplete_semantics() -> None:
    with pytest.raises(ValueError, match=r"unsupported frozen-v1.0 prop"):
        PropSpec(kind="UNAPPROVED_FEATURE")
    with pytest.raises(ValueError, match="requires an operator and threshold"):
        PropSpec(kind="TOTAL_GAMES")
    with pytest.raises(ValueError, match="must declare match or set mode"):
        PropSpec(
            kind="FIRST_BREAK_TIMING",
            operator=ComparisonOperator.MORE_THAN,
            threshold=4,
        )


def test_duration_prop_uses_attached_official_minutes_and_reports_display_sensitivity() -> None:
    version = "duration-display-fixture/v1"
    resolved = replace(
        _completed_path(),
        duration_latent=120.6,
        duration_official=121,
        duration_display_policy_version=version,
        duration_display_candidates=(121,),
    )
    prop = DURATION_MIN(
        ComparisonOperator.MORE_THAN,
        120,
        display_conversion_version=version,
    )
    estimate = evaluate_settlement(prop, _batch(resolved), CANONICAL_SETTLEMENT_POLICY)
    assert estimate.yes_paths == 1
    assert estimate.sensitivity_low == estimate.sensitivity_high == 1.0
    assert estimate.display_policy_version == version

    sensitive = replace(
        resolved,
        duration_official=None,
        duration_display_candidates=(120, 121),
    )
    estimate = evaluate_settlement(prop, _batch(sensitive), CANONICAL_SETTLEMENT_POLICY)
    assert estimate.unresolved_paths == 1
    assert estimate.settled_paths == 0
    assert estimate.sensitivity_low == 0.0
    assert estimate.sensitivity_high == 1.0


def test_duration_thresholds_90_120_150_are_evaluated_from_one_joint_path() -> None:
    version = "duration-display-fixture/v1"
    path = replace(
        _completed_path(),
        duration_latent=130.25,
        duration_official=130,
        duration_display_policy_version=version,
        duration_display_candidates=(130,),
    )
    estimates = tuple(
        evaluate_settlement(
            DURATION_MIN(
                ComparisonOperator.MORE_THAN,
                threshold,
                display_conversion_version=version,
            ),
            _batch(path),
            CANONICAL_SETTLEMENT_POLICY,
        )
        for threshold in (90, 120, 150)
    )
    assert tuple(item.probability_raw for item in estimates) == (1.0, 1.0, 0.0)
    assert all(item.total_paths == 1 for item in estimates)


def test_retired_duration_threshold_preserves_frozen_monotone_settlement() -> None:
    version = "duration-display-fixture/v1"
    retired = replace(
        _retired_path(),
        duration_latent=95.0,
        duration_official=95,
        duration_partial=True,
        duration_display_policy_version=version,
        duration_display_candidates=(95,),
    )
    reached = DURATION_MIN(
        ComparisonOperator.MORE_THAN,
        90,
        display_conversion_version=version,
    )
    unreached = DURATION_MIN(
        ComparisonOperator.MORE_THAN,
        120,
        display_conversion_version=version,
    )
    assert (
        evaluate_settlement(reached, _batch(retired), CANONICAL_SETTLEMENT_POLICY).yes_paths
        == 1
    )
    assert (
        evaluate_settlement(unreached, _batch(retired), CANONICAL_SETTLEMENT_POLICY).void_paths
        == 1
    )
