from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

from hypothesis import given, settings
from hypothesis import strategies as st

from tennis_model.simulation.scoring import (
    MatchState,
    PointTransition,
    ScoringInvariantError,
    TiebreakResult,
    TiebreakState,
    award_point,
    format_game_score,
    is_break_point,
    is_legal_completed_set_score,
    new_match,
    tiebreak_server,
)


def _legal_match_winner_patterns(best_of: int) -> tuple[tuple[int, ...], ...]:
    sets_to_win = best_of // 2 + 1
    patterns: list[tuple[int, ...]] = []
    for length in range(sets_to_win, best_of + 1):
        for winners in product((0, 1), repeat=length):
            final_winner = winners[-1]
            if winners.count(final_winner) != sets_to_win:
                continue
            if any(
                prefix.count(0) == sets_to_win or prefix.count(1) == sets_to_win
                for end in range(1, length)
                if (prefix := winners[:end])
            ):
                continue
            patterns.append(winners)
    return tuple(patterns)


MATCH_PATTERNS = tuple(
    (best_of, pattern) for best_of in (3, 5) for pattern in _legal_match_winner_patterns(best_of)
)


@st.composite
def _complete_match_specs(draw):
    best_of, set_winners = draw(st.sampled_from(MATCH_PATTERNS))
    losing_games = draw(
        st.lists(
            st.integers(min_value=0, max_value=6),
            min_size=len(set_winners),
            max_size=len(set_winners),
        )
    )
    first_server = draw(st.integers(min_value=0, max_value=1))
    return best_of, set_winners, tuple(losing_games), first_server


@st.composite
def _legal_tiebreak_results(draw):
    target = draw(st.sampled_from((7, 10)))
    winner = draw(st.integers(min_value=0, max_value=1))
    loser_points = draw(st.integers(min_value=0, max_value=40))
    winner_points = target if loser_points <= target - 2 else loser_points + 2
    points = (winner_points, loser_points) if winner == 0 else (loser_points, winner_points)
    return target, winner, points


def _play(state: MatchState, winners: Iterable[int]) -> tuple[MatchState, list[PointTransition]]:
    transitions: list[PointTransition] = []
    for winner in winners:
        transition = award_point(state, winner)
        transitions.append(transition)
        state = transition.after
    return state, transitions


def _win_game(state: MatchState, winner: int) -> tuple[MatchState, list[PointTransition]]:
    assert state.active_set is not None
    assert state.active_set.tiebreak is None
    assert state.active_set.game_points == (0, 0)
    return _play(state, [winner] * 4)


def _complete_set(
    state: MatchState, winner: int, losing_games: int
) -> tuple[MatchState, list[PointTransition]]:
    loser = 1 - winner
    if losing_games <= 4:
        game_winners = [loser] * losing_games + [winner] * 6
    elif losing_games == 5:
        game_winners = [winner, loser] * 5 + [winner, winner]
    else:
        game_winners = [winner, loser] * 6

    transitions: list[PointTransition] = []
    for game_winner in game_winners:
        state, game_transitions = _win_game(state, game_winner)
        transitions.extend(game_transitions)

    if losing_games == 6:
        assert state.active_set is not None
        assert state.active_set.tiebreak is not None
        state, tiebreak_transitions = _play(
            state, [winner] * state.active_set.tiebreak.target_points
        )
        transitions.extend(tiebreak_transitions)

    return state, transitions


def _expected_set_score(winner: int, losing_games: int) -> tuple[int, int]:
    winner_games = 6 if losing_games <= 4 else 7
    return (winner_games, losing_games) if winner == 0 else (losing_games, winner_games)


def _assert_state_invariants(state: MatchState) -> None:
    sets_to_win = state.best_of // 2 + 1
    expected_server = state.initial_server_index
    reconstructed_wins = [0, 0]

    for set_number, result in enumerate(state.completed_sets, start=1):
        assert result.set_number == set_number
        assert result.first_server_index == expected_server
        assert is_legal_completed_set_score(
            result.games, tiebreak_played=result.tiebreak is not None
        )
        assert result.winner_index == (0 if result.games[0] > result.games[1] else 1)
        assert result.total_games == sum(result.games)
        assert result.regular_games == result.total_games - int(result.tiebreak is not None)
        if result.tiebreak is not None:
            expected_target = 10 if set_number == state.best_of else 7
            assert result.games in ((7, 6), (6, 7))
            assert result.tiebreak.target_points == expected_target
            assert result.tiebreak.winner_index == result.winner_index
        reconstructed_wins[result.winner_index] += 1
        expected_server = result.next_set_server_index

    assert state.sets_won == tuple(reconstructed_wins)
    assert all(0 <= value <= sets_to_win for value in state.sets_won)
    assert all(
        0 <= state.breaks_of_serve[player] <= state.break_point_opportunities[player]
        for player in (0, 1)
    )
    assert state.total_points_played >= 0

    completed_games = sum(result.total_games for result in state.completed_sets)
    completed_regular_games = sum(result.regular_games for result in state.completed_sets)
    active_games = 0 if state.active_set is None else sum(state.active_set.games)
    assert state.total_games == completed_games + active_games
    assert state.regular_games == completed_regular_games + active_games
    expected_tiebreaks = sum(result.tiebreak is not None for result in state.completed_sets)

    if state.is_complete:
        assert state.active_set is None
        assert state.winner_index is not None
        assert state.sets_won[state.winner_index] == sets_to_win
        assert sum(value == sets_to_win for value in state.sets_won) == 1
        assert state.tiebreaks_started == expected_tiebreaks
        return

    assert all(value < sets_to_win for value in state.sets_won)
    assert state.winner_index is None
    assert state.active_set is not None
    active = state.active_set
    assert active.set_number == len(state.completed_sets) + 1
    assert active.set_number <= state.best_of
    assert active.first_server_index == expected_server
    if active.tiebreak is not None:
        assert active.games == (6, 6)
        assert active.game_points == (0, 0)
        assert active.tiebreak.first_server_index == active.first_server_index
        assert active.tiebreak.target_points == (10 if active.set_number == state.best_of else 7)
        assert state.tiebreaks_started == expected_tiebreaks + 1
    else:
        assert active.games != (6, 6)
        assert max(active.games) <= 6
        assert not is_legal_completed_set_score(active.games, tiebreak_played=False)
        format_game_score(active.game_points)
        assert state.tiebreaks_started == expected_tiebreaks


def _assert_transition_invariants(transition: PointTransition) -> None:
    before = transition.before
    after = transition.after
    assert before.active_set is not None
    assert transition.server_index == before.server_index
    assert transition.receiver_index == 1 - transition.server_index
    assert transition.was_tiebreak is before.active_set.in_tiebreak
    assert after.total_points_played == before.total_points_played + 1
    assert after.total_games - before.total_games == int(
        transition.regular_game_completed or transition.tiebreak_completed
    )
    assert after.regular_games - before.regular_games == int(transition.regular_game_completed)

    expected_opportunities = list(before.break_point_opportunities)
    expected_breaks = list(before.breaks_of_serve)
    expected_break_point = not transition.was_tiebreak and is_break_point(
        before.active_set.game_points, transition.server_index
    )
    assert transition.break_point_opportunity is expected_break_point
    if expected_break_point:
        expected_opportunities[transition.receiver_index] += 1

    expected_break = (
        transition.regular_game_completed
        and transition.game_winner_index == transition.receiver_index
    )
    assert transition.break_of_serve is expected_break
    if expected_break:
        expected_breaks[transition.receiver_index] += 1
    assert after.break_point_opportunities == tuple(expected_opportunities)
    assert after.breaks_of_serve == tuple(expected_breaks)

    if transition.was_tiebreak:
        tiebreak = before.active_set.tiebreak
        assert tiebreak is not None
        assert transition.server_index == tiebreak_server(
            tiebreak.first_server_index, tiebreak.points_played
        )
        assert not transition.break_point_opportunity
        assert not transition.break_of_serve
        assert not transition.regular_game_completed
        if not transition.tiebreak_completed:
            assert after.active_set is not None
            assert after.total_games == before.total_games
        elif after.active_set is not None:
            assert after.active_set.first_server_index == 1 - tiebreak.first_server_index
    else:
        parity_server = (
            before.initial_server_index
            if before.total_games % 2 == 0
            else 1 - before.initial_server_index
        )
        assert transition.server_index == parity_server
        assert not transition.tiebreak_completed
        if transition.regular_game_completed and after.active_set is not None:
            assert after.server_index == 1 - transition.server_index
        elif not transition.regular_game_completed:
            assert after.server_index == transition.server_index


@given(first_server=st.integers(min_value=0, max_value=1), points_played=st.integers(0, 500))
def test_tiebreak_service_formula_for_arbitrarily_long_tiebreaks(
    first_server: int, points_played: int
) -> None:
    if points_played == 0:
        expected = first_server
    else:
        expected = 1 - first_server if ((points_played - 1) // 2) % 2 == 0 else first_server
    assert tiebreak_server(first_server, points_played) == expected


@given(_legal_tiebreak_results())
def test_tiebreak_terminal_scores_are_minimal_and_win_by_two(
    spec: tuple[int, int, tuple[int, int]],
) -> None:
    target, winner, points = spec
    result = TiebreakResult(first_server_index=0, target_points=target, points=points)
    assert result.winner_index == winner
    assert points[winner] >= target
    assert points[winner] - points[1 - winner] == 2 or (
        points[winner] == target and points[1 - winner] <= target - 2
    )

    before_points = list(points)
    before_points[winner] -= 1
    active = TiebreakState(
        first_server_index=0,
        target_points=target,
        points=tuple(before_points),
    )
    assert active.points[winner] == points[winner] - 1

    overshoot = list(points)
    overshoot[winner] += 1
    try:
        TiebreakResult(
            first_server_index=0,
            target_points=target,
            points=tuple(overshoot),
        )
    except ScoringInvariantError:
        pass
    else:
        raise AssertionError("a tiebreak must stop at its first legal terminal score")


@given(
    games_a=st.integers(min_value=-2, max_value=10),
    games_b=st.integers(min_value=-2, max_value=10),
    tiebreak_played=st.booleans(),
)
@settings(max_examples=300)
def test_completed_set_score_validator_matches_frozen_rule(
    games_a: int, games_b: int, tiebreak_played: bool
) -> None:
    high = max(games_a, games_b)
    low = min(games_a, games_b)
    expected = games_a >= 0 and games_b >= 0 and games_a != games_b
    if tiebreak_played:
        expected = expected and high == 7 and low == 6
    else:
        expected = expected and ((high == 6 and low <= 4) or (high == 7 and low == 5))
    assert (
        is_legal_completed_set_score((games_a, games_b), tiebreak_played=tiebreak_played)
        is expected
    )


@given(_complete_match_specs())
@settings(max_examples=70, deadline=None)
def test_constructed_legal_matches_preserve_exact_scoring_and_termination(
    spec: tuple[int, Sequence[int], Sequence[int], int],
) -> None:
    best_of, set_winners, losing_games, first_server = spec
    state = new_match(
        "player-a",
        "player-b",
        best_of=best_of,
        first_server_index=first_server,
    )
    all_transitions: list[PointTransition] = []
    expected_scores: list[tuple[int, int]] = []

    for index, (winner, set_losing_games) in enumerate(zip(set_winners, losing_games, strict=True)):
        assert not state.is_complete
        state, transitions = _complete_set(state, winner, set_losing_games)
        all_transitions.extend(transitions)
        expected_scores.append(_expected_set_score(winner, set_losing_games))
        if index < len(set_winners) - 1:
            assert not state.is_complete

    assert state.is_complete
    assert state.active_set is None
    assert state.set_winners == tuple(set_winners)
    assert state.exact_set_scores == tuple(expected_scores)
    assert state.winner_index == set_winners[-1]
    assert state.sets_won[state.winner_index] == best_of // 2 + 1
    assert state.total_games == sum(sum(score) for score in expected_scores)
    expected_tiebreaks = sum(score in ((7, 6), (6, 7)) for score in expected_scores)
    assert state.tiebreaks_started == expected_tiebreaks
    assert state.regular_games == state.total_games - expected_tiebreaks
    assert state.deciding_set_began is (len(set_winners) == best_of)
    assert state.total_points_played == len(all_transitions)
    for transition in all_transitions:
        _assert_transition_invariants(transition)
        _assert_state_invariants(transition.after)


@given(
    best_of=st.sampled_from((3, 5)),
    first_server=st.integers(min_value=0, max_value=1),
    winners=st.lists(st.integers(min_value=0, max_value=1), max_size=300),
)
@settings(max_examples=100, deadline=None)
def test_arbitrary_point_streams_never_create_illegal_scoring_states(
    best_of: int, first_server: int, winners: list[int]
) -> None:
    state = new_match(
        "player-a",
        "player-b",
        best_of=best_of,
        first_server_index=first_server,
    )
    _assert_state_invariants(state)

    for winner in winners:
        if state.is_complete:
            break
        transition = award_point(state, winner)
        assert transition.before is state
        _assert_transition_invariants(transition)
        state = transition.after
        _assert_state_invariants(state)
