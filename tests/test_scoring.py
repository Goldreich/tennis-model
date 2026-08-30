from __future__ import annotations

from collections.abc import Iterable
from dataclasses import FrozenInstanceError

import pytest

from tennis_model.simulation.scoring import (
    MatchAlreadyCompleteError,
    MatchState,
    PointTransition,
    ScoringInvariantError,
    SetResult,
    SetState,
    TiebreakResult,
    TiebreakState,
    award_point,
    format_game_score,
    is_break_point,
    is_legal_completed_set_score,
    new_match,
    tiebreak_server,
)


def _new(*, best_of: int = 3, first_server: int = 0) -> MatchState:
    return new_match("player-a", "player-b", best_of=best_of, first_server_index=first_server)


def _play(state: MatchState, winners: Iterable[int]) -> tuple[MatchState, list[PointTransition]]:
    transitions: list[PointTransition] = []
    for winner in winners:
        transition = award_point(state, winner)
        transitions.append(transition)
        state = transition.after
    return state, transitions


def _win_game(state: MatchState, winner: int) -> tuple[MatchState, list[PointTransition]]:
    assert state.active_set is not None
    assert not state.active_set.in_tiebreak
    assert state.active_set.game_points == (0, 0)
    before_games = state.total_games
    state, transitions = _play(state, [winner] * 4)
    assert state.total_games == before_games + 1
    assert transitions[-1].regular_game_completed
    assert transitions[-1].game_winner_index == winner
    return state, transitions


def _play_games(
    state: MatchState, winners: Iterable[int]
) -> tuple[MatchState, list[PointTransition]]:
    transitions: list[PointTransition] = []
    for winner in winners:
        state, game_transitions = _win_game(state, winner)
        transitions.extend(game_transitions)
    return state, transitions


def _reach_tiebreak(state: MatchState) -> tuple[MatchState, list[PointTransition]]:
    state, transitions = _play_games(state, [0, 1] * 6)
    assert state.active_set is not None
    assert state.active_set.games == (6, 6)
    assert state.active_set.tiebreak is not None
    return state, transitions


def _complete_set(
    state: MatchState, winner: int, losing_games: int = 0
) -> tuple[MatchState, list[PointTransition]]:
    """Complete one set with a constructive legal score.

    ``losing_games`` 0..4 yields 6-k, 5 yields 7-5, and 6 yields 7-6.
    """

    assert 0 <= losing_games <= 6
    loser = 1 - winner
    transitions: list[PointTransition] = []

    if losing_games <= 4:
        game_winners = [loser] * losing_games + [winner] * 6
    elif losing_games == 5:
        game_winners = [winner, loser] * 5 + [winner, winner]
    else:
        game_winners = [winner, loser] * 6

    state, game_transitions = _play_games(state, game_winners)
    transitions.extend(game_transitions)

    if losing_games == 6:
        assert state.active_set is not None
        assert state.active_set.tiebreak is not None
        target = state.active_set.tiebreak.target_points
        state, tiebreak_transitions = _play(state, [winner] * target)
        transitions.extend(tiebreak_transitions)

    assert transitions[-1].completed_set is not None
    return state, transitions


def _standard_tiebreak_endpoint(winner: int, losing_points: int) -> list[int]:
    loser = 1 - winner
    if 0 <= losing_points <= 5:
        return [winner] * 6 + [loser] * losing_points + [winner]
    if losing_points == 6:
        return [winner, loser] * 6 + [winner, winner]
    raise AssertionError("test helper supports terminal scores 7-0..7-5 and 8-6")


def test_advantage_game_and_saved_break_points() -> None:
    state = _new(first_server=0)

    state, transitions = _play(state, [0, 0, 1, 1, 1, 0, 1, 0, 0, 0])

    assert [transition.break_point_opportunity for transition in transitions] == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert transitions[5].before.active_set is not None
    assert transitions[5].before.active_set.displayed_game_score == ("30", "40")
    assert transitions[6].after.active_set is not None
    assert transitions[6].after.active_set.displayed_game_score == ("40", "AD")
    assert transitions[7].after.active_set is not None
    assert transitions[7].after.active_set.displayed_game_score == ("40", "40")
    assert transitions[8].after.active_set is not None
    assert transitions[8].after.active_set.displayed_game_score == ("AD", "40")
    assert transitions[-1].regular_game_completed
    assert not transitions[-1].break_of_serve
    assert state.break_point_opportunities == (0, 2)
    assert state.breaks_of_serve == (0, 0)
    assert state.active_set is not None
    assert state.active_set.games == (1, 0)
    assert state.active_set.game_points == (0, 0)


def test_multiple_break_points_can_produce_only_one_break() -> None:
    state, transitions = _play(_new(first_server=0), [1, 1, 1, 0, 0, 1])

    assert [
        i for i, transition in enumerate(transitions) if transition.break_point_opportunity
    ] == [
        3,
        4,
        5,
    ]
    assert state.break_point_opportunities == (0, 3)
    assert state.breaks_of_serve == (0, 1)
    assert transitions[-1].regular_game_completed
    assert transitions[-1].break_of_serve
    assert transitions[-1].game_winner_index == 1


@pytest.mark.parametrize(
    ("points", "server", "expected"),
    [
        ((0, 3), 0, True),
        ((1, 3), 0, True),
        ((2, 3), 0, True),
        ((3, 4), 0, True),
        ((5, 6), 0, True),
        ((3, 3), 0, False),
        ((4, 3), 0, False),
        ((3, 2), 0, False),
        ((3, 0), 1, True),
    ],
)
def test_break_point_states(points: tuple[int, int], server: int, expected: bool) -> None:
    assert is_break_point(points, server) is expected


@pytest.mark.parametrize("points", [(4, 0), (0, 4), (5, 3), (3, 5)])
def test_break_point_helper_rejects_completed_games(points: tuple[int, int]) -> None:
    with pytest.raises(ScoringInvariantError, match="completed game"):
        is_break_point(points, server_index=0)


@pytest.mark.parametrize(
    ("points", "display"),
    [
        ((0, 0), ("0", "0")),
        ((1, 0), ("15", "0")),
        ((2, 1), ("30", "15")),
        ((3, 2), ("40", "30")),
        ((3, 3), ("40", "40")),
        ((4, 3), ("AD", "40")),
        ((3, 4), ("40", "AD")),
        ((8, 8), ("40", "40")),
        ((9, 8), ("AD", "40")),
    ],
)
def test_exact_advantage_game_display(points: tuple[int, int], display: tuple[str, str]) -> None:
    assert format_game_score(points) == display


@pytest.mark.parametrize("points", [(4, 0), (4, 2), (5, 3), (8, 6)])
def test_completed_game_has_no_ongoing_display(points: tuple[int, int]) -> None:
    with pytest.raises(ScoringInvariantError, match="completed game"):
        format_game_score(points)


def test_regular_service_alternates_after_holds_and_breaks() -> None:
    state = _new(first_server=0)
    state, first_game = _win_game(state, 0)
    assert {transition.server_index for transition in first_game} == {0}
    assert state.server_index == 1

    state, second_game = _win_game(state, 0)
    assert {transition.server_index for transition in second_game} == {1}
    assert second_game[-1].break_of_serve
    assert state.server_index == 0
    assert state.breaks_of_serve == (1, 0)


def test_service_order_continues_across_even_and_odd_length_sets() -> None:
    even_state, _ = _complete_set(_new(first_server=0), winner=0, losing_games=0)
    assert even_state.completed_sets[0].games == (6, 0)
    assert even_state.completed_sets[0].next_set_server_index == 0
    assert even_state.active_set is not None
    assert even_state.active_set.first_server_index == 0
    assert even_state.server_index == 0

    odd_state, _ = _complete_set(_new(first_server=0), winner=0, losing_games=1)
    assert odd_state.completed_sets[0].games == (6, 1)
    assert odd_state.completed_sets[0].next_set_server_index == 1
    assert odd_state.active_set is not None
    assert odd_state.active_set.first_server_index == 1
    assert odd_state.server_index == 1


def test_standard_tiebreak_is_seven_points_win_by_two() -> None:
    state, _ = _reach_tiebreak(_new())
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.target_points == 7

    state, _ = _play(state, [0, 1] * 6 + [0])
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.points == (7, 6)

    transition = award_point(state, 0)
    assert transition.tiebreak_completed
    assert transition.completed_set is not None
    assert transition.completed_set.games == (7, 6)
    assert transition.completed_set.tiebreak is not None
    assert transition.completed_set.tiebreak.points == (8, 6)


def test_deciding_tiebreak_is_ten_points_win_by_two() -> None:
    state, _ = _complete_set(_new(best_of=3), winner=0)
    state, _ = _complete_set(state, winner=1)
    assert state.deciding_set_began
    state, _ = _reach_tiebreak(state)
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.target_points == 10

    state, _ = _play(state, [0] * 6 + [1] * 5 + [0])
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.points == (7, 5)

    state, _ = _play(state, [1] * 4 + [0] * 3)
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.points == (10, 9)

    transition = award_point(state, 0)
    assert transition.tiebreak_completed
    assert transition.after.is_complete
    assert transition.completed_set is not None
    assert transition.completed_set.tiebreak is not None
    assert transition.completed_set.tiebreak.points == (11, 9)


def test_best_of_five_fourth_set_is_standard_and_fifth_is_deciding_tiebreak() -> None:
    state = _new(best_of=5)
    for winner in (0, 1, 0):
        state, _ = _complete_set(state, winner=winner)
    assert state.active_set is not None
    assert state.active_set.set_number == 4

    state, _ = _reach_tiebreak(state)
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.target_points == 7
    state, _ = _play(state, [1] * 7)

    assert state.active_set is not None
    assert state.active_set.set_number == 5
    assert state.deciding_set_began
    state, _ = _reach_tiebreak(state)
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.target_points == 10


@pytest.mark.parametrize("first_server", [0, 1])
def test_extended_tiebreak_uses_exact_one_two_two_service_sequence(first_server: int) -> None:
    state, _ = _reach_tiebreak(_new(first_server=first_server))
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.first_server_index == first_server

    for points_played, winner in enumerate([0, 1] * 8):
        transition = award_point(state, winner)
        assert transition.was_tiebreak
        assert transition.server_index == tiebreak_server(first_server, points_played)
        assert not transition.tiebreak_completed
        state = transition.after

    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    assert state.active_set.tiebreak.points == (8, 8)


@pytest.mark.parametrize("first_server", [0, 1])
@pytest.mark.parametrize("winner", [0, 1])
@pytest.mark.parametrize("losing_points", [0, 5, 6])
def test_tiebreak_first_receiver_serves_next_set(
    first_server: int, winner: int, losing_points: int
) -> None:
    state, _ = _reach_tiebreak(_new(first_server=first_server))
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    first_tiebreak_server = state.active_set.tiebreak.first_server_index

    state, transitions = _play(state, _standard_tiebreak_endpoint(winner, losing_points))

    result = transitions[-1].completed_set
    assert result is not None
    assert result.tiebreak is not None
    assert result.winner_index == winner
    assert state.active_set is not None
    assert state.active_set.first_server_index == 1 - first_tiebreak_server
    assert state.server_index == 1 - first_tiebreak_server


def test_tiebreak_return_points_are_not_breaks_or_break_points() -> None:
    state, _ = _reach_tiebreak(_new(first_server=0))
    assert state.total_games == 12
    assert state.regular_games == 12
    baseline_breaks = state.breaks_of_serve
    baseline_opportunities = state.break_point_opportunities
    assert state.active_set is not None
    assert state.active_set.tiebreak is not None
    receiver_of_first_point = 1 - state.active_set.tiebreak.first_server_index

    state, transitions = _play(state, [receiver_of_first_point] * 7)

    assert all(transition.was_tiebreak for transition in transitions)
    assert not any(transition.break_point_opportunity for transition in transitions)
    assert not any(transition.break_of_serve for transition in transitions)
    assert not any(transition.regular_game_completed for transition in transitions)
    assert all(transition.after.total_games == 12 for transition in transitions[:-1])
    assert state.breaks_of_serve == baseline_breaks
    assert state.break_point_opportunities == baseline_opportunities
    assert state.total_games == 13
    assert state.regular_games == 12
    result = state.completed_sets[0]
    assert result.games in ((7, 6), (6, 7))
    assert result.total_games == 13
    assert result.regular_games == 12
    assert state.tiebreaks_started == 1


@pytest.mark.parametrize(
    ("games", "tiebreak_played", "expected"),
    [
        ((6, 0), False, True),
        ((0, 6), False, True),
        ((6, 4), False, True),
        ((7, 5), False, True),
        ((5, 7), False, True),
        ((7, 6), True, True),
        ((6, 7), True, True),
        ((6, 5), False, False),
        ((6, 6), False, False),
        ((7, 4), False, False),
        ((7, 6), False, False),
        ((6, 4), True, False),
        ((8, 6), False, False),
    ],
)
def test_legal_completed_set_scores(
    games: tuple[int, int], tiebreak_played: bool, expected: bool
) -> None:
    assert is_legal_completed_set_score(games, tiebreak_played=tiebreak_played) is expected


def test_illegal_game_set_and_tiebreak_terminal_states_are_rejected() -> None:
    with pytest.raises(ScoringInvariantError, match="already terminal"):
        SetState(set_number=1, first_server_index=0, game_points=(4, 2))
    with pytest.raises(ScoringInvariantError, match="6-6"):
        SetState(set_number=1, first_server_index=0, games=(6, 6))
    with pytest.raises(ScoringInvariantError, match="only at 6-6"):
        SetState(
            set_number=1,
            first_server_index=0,
            games=(5, 5),
            tiebreak=TiebreakState(first_server_index=0, target_points=7),
        )
    with pytest.raises(ScoringInvariantError, match="already terminal"):
        TiebreakState(first_server_index=0, target_points=7, points=(8, 6))
    with pytest.raises(ScoringInvariantError, match="minimal terminal"):
        TiebreakResult(first_server_index=0, target_points=7, points=(7, 6))
    with pytest.raises(ScoringInvariantError, match="minimal terminal"):
        TiebreakResult(first_server_index=0, target_points=7, points=(9, 6))
    with pytest.raises(ScoringInvariantError, match="impossible game score"):
        SetResult(set_number=1, first_server_index=0, games=(8, 6))
    with pytest.raises(ScoringInvariantError, match="impossible game score"):
        SetResult(set_number=1, first_server_index=0, games=(7, 6))


def test_scoring_states_reject_wrong_runtime_child_types() -> None:
    with pytest.raises(ScoringInvariantError, match="target must be 7 or 10"):
        TiebreakState(
            first_server_index=0,
            target_points=7.0,  # type: ignore[arg-type]
        )
    with pytest.raises(ScoringInvariantError, match="target must be 7 or 10"):
        TiebreakResult(
            first_server_index=0,
            target_points=10.0,  # type: ignore[arg-type]
            points=(10, 0),
        )
    with pytest.raises(ScoringInvariantError, match="TiebreakState"):
        SetState(
            set_number=1,
            first_server_index=0,
            games=(6, 6),
            tiebreak=TiebreakResult(
                first_server_index=0,
                target_points=7,
                points=(7, 0),
            ),  # type: ignore[arg-type]
        )
    with pytest.raises(ScoringInvariantError, match="TiebreakResult"):
        SetResult(
            set_number=1,
            first_server_index=0,
            games=(7, 6),
            tiebreak=TiebreakState(
                first_server_index=0,
                target_points=7,
            ),  # type: ignore[arg-type]
        )

    active = SetState(set_number=1, first_server_index=0)
    with pytest.raises(ScoringInvariantError, match="immutable tuple"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=[],  # type: ignore[arg-type]
            active_set=active,
        )
    with pytest.raises(ScoringInvariantError, match="only SetResult"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(active,),  # type: ignore[arg-type]
            active_set=active,
        )
    with pytest.raises(ScoringInvariantError, match="SetState or None"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(),
            active_set="not a set",  # type: ignore[arg-type]
        )


def test_match_state_rejects_impossible_audit_aggregates() -> None:
    common = {
        "players": ("player-a", "player-b"),
        "best_of": 3,
        "initial_server_index": 0,
        "completed_sets": (),
    }
    with pytest.raises(ScoringInvariantError, match="total points are inconsistent"):
        MatchState(
            **common,  # type: ignore[arg-type]
            active_set=SetState(set_number=1, first_server_index=0, games=(1, 0)),
            total_points_played=3,
        )
    with pytest.raises(ScoringInvariantError, match="regular points played"):
        MatchState(
            **common,  # type: ignore[arg-type]
            active_set=SetState(set_number=1, first_server_index=0),
            break_point_opportunities=(1, 0),
            total_points_played=0,
        )
    with pytest.raises(ScoringInvariantError, match="regular games won"):
        MatchState(
            **common,  # type: ignore[arg-type]
            active_set=SetState(set_number=1, first_server_index=0),
            break_point_opportunities=(1, 0),
            breaks_of_serve=(1, 0),
            total_points_played=1,
        )


def test_tiebreak_set_requires_consistent_winner_server_and_target() -> None:
    tiebreak = TiebreakResult(first_server_index=0, target_points=7, points=(7, 5))
    with pytest.raises(ScoringInvariantError, match="first server"):
        SetResult(set_number=1, first_server_index=1, games=(7, 6), tiebreak=tiebreak)

    losing_tiebreak = TiebreakResult(first_server_index=0, target_points=7, points=(5, 7))
    with pytest.raises(ScoringInvariantError, match="winners must agree"):
        SetResult(set_number=1, first_server_index=0, games=(7, 6), tiebreak=losing_tiebreak)

    deciding_tiebreak = TiebreakResult(first_server_index=0, target_points=10, points=(10, 8))
    wrong_target_set = SetResult(
        set_number=1,
        first_server_index=0,
        games=(7, 6),
        tiebreak=deciding_tiebreak,
    )
    with pytest.raises(ScoringInvariantError, match="wrong tiebreak target"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(wrong_target_set,),
            active_set=SetState(set_number=2, first_server_index=1),
        )


def test_match_state_rejects_wrong_active_and_deciding_tiebreak_targets() -> None:
    with pytest.raises(ScoringInvariantError, match="wrong tiebreak target"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(),
            active_set=SetState(
                set_number=1,
                first_server_index=0,
                games=(6, 6),
                tiebreak=TiebreakState(first_server_index=0, target_points=10),
            ),
            total_points_played=48,
        )

    first = SetResult(set_number=1, first_server_index=0, games=(6, 0))
    second = SetResult(set_number=2, first_server_index=0, games=(0, 6))
    with pytest.raises(ScoringInvariantError, match="wrong tiebreak target"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(first, second),
            active_set=SetState(
                set_number=3,
                first_server_index=0,
                games=(6, 6),
                tiebreak=TiebreakState(first_server_index=0, target_points=7),
            ),
            total_points_played=96,
        )


def test_best_of_three_stops_at_two_sets_and_has_consistent_winner() -> None:
    state = _new(best_of=3)
    state, _ = _complete_set(state, winner=0, losing_games=4)
    assert not state.is_complete
    assert state.sets_won == (1, 0)

    state, _ = _complete_set(state, winner=0, losing_games=5)
    assert state.is_complete
    assert state.sets_won == (2, 0)
    assert state.set_winners == (0, 0)
    assert state.winner_index == 0
    assert state.winner_id == "player-a"
    assert state.exact_set_scores == ((6, 4), (7, 5))
    assert state.active_set is None
    assert not state.deciding_set_began
    with pytest.raises(MatchAlreadyCompleteError):
        award_point(state, 1)


def test_best_of_three_deciding_set_stops_at_two_one() -> None:
    state = _new(best_of=3)
    for winner in (0, 1):
        state, _ = _complete_set(state, winner=winner)
    assert not state.is_complete
    assert state.deciding_set_began
    assert state.active_set is not None
    assert state.active_set.set_number == 3

    state, _ = _complete_set(state, winner=1, losing_games=1)
    assert state.is_complete
    assert state.sets_won == (1, 2)
    assert state.set_winners == (0, 1, 1)
    assert state.winner_index == 1


def test_best_of_five_stops_at_three_sets_not_two() -> None:
    state = _new(best_of=5)
    for _ in range(2):
        state, _ = _complete_set(state, winner=0)
    assert not state.is_complete
    assert state.sets_won == (2, 0)

    state, _ = _complete_set(state, winner=0)
    assert state.is_complete
    assert state.sets_won == (3, 0)
    assert state.winner_index == 0
    assert len(state.completed_sets) == 3
    assert not state.deciding_set_began


def test_best_of_five_can_end_three_two_only_after_fifth_set_begins() -> None:
    state = _new(best_of=5)
    for winner in (0, 1, 0, 1):
        state, _ = _complete_set(state, winner=winner)
    assert not state.is_complete
    assert state.sets_won == (2, 2)
    assert state.deciding_set_began
    assert state.active_set is not None
    assert state.active_set.set_number == 5

    state, _ = _complete_set(state, winner=0)
    assert state.is_complete
    assert state.sets_won == (3, 2)
    assert state.winner_index == 0
    assert len(state.completed_sets) == 5


def test_match_state_rejects_incomplete_and_post_clinch_set_sequences() -> None:
    first = SetResult(set_number=1, first_server_index=0, games=(6, 0))
    second = SetResult(set_number=2, first_server_index=0, games=(6, 0))

    with pytest.raises(ScoringInvariantError, match="incomplete match"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(first,),
            active_set=None,
        )
    with pytest.raises(ScoringInvariantError, match="completed match"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(first, second),
            active_set=SetState(set_number=3, first_server_index=0),
        )

    third = SetResult(set_number=3, first_server_index=0, games=(0, 6))
    with pytest.raises(ScoringInvariantError, match="clinching set"):
        MatchState(
            players=("player-a", "player-b"),
            best_of=3,
            initial_server_index=0,
            completed_sets=(first, second, third),
            active_set=None,
        )


def test_award_point_is_pure_deterministic_and_immutable() -> None:
    state = _new()
    first = award_point(state, 0)
    repeated = award_point(state, 0)

    assert first == repeated
    assert first.before is state
    assert state.total_points_played == 0
    assert state.active_set is not None
    assert state.active_set.game_points == (0, 0)
    assert first.after.total_points_played == 1
    assert first.after.active_set is not None
    assert first.after.active_set.game_points == (1, 0)
    with pytest.raises(FrozenInstanceError):
        state.total_points_played = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"player_a_id": "same", "player_b_id": "same", "best_of": 3}, "distinct"),
        ({"player_a_id": "a", "player_b_id": "b", "best_of": 1}, "3 or 5"),
        ({"player_a_id": "a", "player_b_id": "b", "best_of": 4}, "3 or 5"),
        ({"player_a_id": "a", "player_b_id": "b", "best_of": 3.0}, "3 or 5"),
    ],
)
def test_new_match_rejects_invalid_identity_and_format(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ScoringInvariantError, match=message):
        new_match(**kwargs, first_server_index=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("winner", [-1, 2, True])
def test_award_point_rejects_invalid_winner(winner: int) -> None:
    with pytest.raises(ScoringInvariantError, match="winner_index"):
        award_point(_new(), winner)  # type: ignore[arg-type]
