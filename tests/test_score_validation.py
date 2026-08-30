from dataclasses import FrozenInstanceError

import pytest

from tennis_model.data.validate import (
    SCORE_ILLEGAL_SET,
    SCORE_INCOMPLETE_MATCH,
    SCORE_INVALID_BEST_OF,
    SCORE_INVALID_STATUS,
    SCORE_MALFORMED,
    SCORE_MATCH_TIEBREAK_CONTEXT,
    SCORE_MISSING,
    SCORE_SET_AFTER_MATCH_END,
    SCORE_STATUS_AFTER_COMPLETION,
    SCORE_TOO_MANY_SETS,
    SCORE_WALKOVER_WITH_SCORE,
    SCORE_WINNER_ORIENTATION_MISMATCH,
    ScoreTermination,
    SetScoreKind,
    validate_score,
)


@pytest.mark.parametrize(
    ("score", "best_of", "set_wins", "regular_service_games"),
    [
        ("6-4 3-6 7-6(5)", 3, (2, 1), 31),
        ("6-4 6-3", 3, (2, 0), 19),
        ("6-4 4-6 7-5 3-6 8-6", 5, (3, 2), 55),
        ("70-68 6-4", 3, (2, 0), 148),
        ("6-4 3-6 [10-8]", 3, (2, 1), 19),
        ("6-4 3-6 [7-5]", 3, (2, 1), 19),
    ],
)
def test_accepts_completed_winner_oriented_scores(
    score: str,
    best_of: int,
    set_wins: tuple[int, int],
    regular_service_games: int,
) -> None:
    result = validate_score(score, best_of=best_of)

    assert result.valid
    assert result.completed
    assert not result.retirement
    assert not result.walkover
    assert result.termination is ScoreTermination.COMPLETED
    assert result.parsed_set_wins == set_wins
    assert result.regular_service_games == regular_service_games
    assert result.anomaly_codes == ()


def test_preserves_raw_text_and_excludes_tiebreak_games() -> None:
    raw = "  7-6(10)   6-7(0)  8-6  "

    result = validate_score(raw, best_of=3)

    assert result.raw_score == raw
    assert result.valid
    assert result.parsed_set_wins == (2, 1)
    assert result.regular_service_games == 12 + 12 + 14
    assert result.official_games == 13 + 13 + 14
    assert result.sets[0].tiebreak_loser_points == 10
    assert all(item.kind is SetScoreKind.REGULAR for item in result.sets)


@pytest.mark.parametrize("marker", ["RET", "RET.", "ret"])
def test_accepts_retirement_with_partial_last_set(marker: str) -> None:
    result = validate_score(f"6-4 3-2 {marker}", best_of=3)

    assert result.valid
    assert not result.completed
    assert result.retirement
    assert not result.walkover
    assert result.termination is ScoreTermination.RETIREMENT
    assert result.parsed_set_wins == (1, 0)
    assert result.regular_service_games == 15
    assert not result.sets[-1].completed


def test_retirement_allows_partial_historical_advantage_set() -> None:
    result = validate_score("6-4 7-7 RET", best_of=3)

    assert result.valid
    assert result.parsed_set_wins == (1, 0)
    assert result.regular_service_games == 24


def test_retirement_can_occur_before_a_completed_game_is_reported() -> None:
    result = validate_score("RET", best_of=3)

    assert result.valid
    assert result.retirement
    assert result.parsed_set_wins == (0, 0)
    assert result.regular_service_games is None


def test_accepts_partial_deciding_match_tiebreak_on_retirement() -> None:
    result = validate_score("6-4 3-6 [8-7] RET", best_of=3)

    assert result.valid
    assert result.retirement
    assert result.parsed_set_wins == (1, 1)
    assert result.regular_service_games == 19
    assert result.sets[-1].kind is SetScoreKind.MATCH_TIEBREAK
    assert not result.sets[-1].completed


@pytest.mark.parametrize("marker", ["W/O", "WO", "wo"])
def test_classifies_walkovers(marker: str) -> None:
    result = validate_score(marker, best_of=3)

    assert result.valid
    assert result.walkover
    assert not result.retirement
    assert not result.completed
    assert result.termination is ScoreTermination.WALKOVER
    assert result.parsed_set_wins == (0, 0)
    assert result.regular_service_games == 0


@pytest.mark.parametrize(
    ("marker", "termination"),
    [
        ("DEF", ScoreTermination.DEFAULT),
        ("ABN", ScoreTermination.ABANDONED),
        ("ABD", ScoreTermination.ABANDONED),
    ],
)
def test_recognizes_other_noncompletion_markers(marker: str, termination: ScoreTermination) -> None:
    marker_only = validate_score(marker, best_of=3)
    partial = validate_score(f"6-4 2-3 {marker}", best_of=3)

    assert marker_only.valid
    assert marker_only.termination is termination
    assert marker_only.regular_service_games is None
    assert partial.valid
    assert partial.termination is termination
    assert partial.regular_service_games == 15
    assert not partial.completed
    assert not partial.retirement
    assert not partial.walkover


@pytest.mark.parametrize(
    ("score", "expected_code"),
    [
        ("6-5 6-4", SCORE_ILLEGAL_SET),
        ("7-7 6-4", SCORE_ILLEGAL_SET),
        ("9-3 6-4", SCORE_ILLEGAL_SET),
        ("7-5(3) 6-4", SCORE_ILLEGAL_SET),
        ("6:x 6-4", SCORE_MALFORMED),
        ("6-4 3-6", SCORE_INCOMPLETE_MATCH),
        ("4-6 3-6", SCORE_WINNER_ORIENTATION_MISMATCH),
    ],
)
def test_flags_unambiguously_invalid_completed_scores(score: str, expected_code: str) -> None:
    result = validate_score(score, best_of=3)

    assert not result.valid
    assert expected_code in result.anomaly_codes


def test_flags_sets_after_match_end_and_too_many_sets() -> None:
    result = validate_score("6-4 3-6 6-4 6-4", best_of=3)

    assert not result.valid
    assert SCORE_TOO_MANY_SETS in result.anomaly_codes
    assert SCORE_SET_AFTER_MATCH_END in result.anomaly_codes


def test_status_after_an_already_completed_match_is_invalid() -> None:
    result = validate_score("6-4 6-3 RET", best_of=3)

    assert not result.valid
    assert result.retirement
    assert SCORE_STATUS_AFTER_COMPLETION in result.anomaly_codes


def test_status_marker_must_be_unique_and_last() -> None:
    result = validate_score("RET 6-4", best_of=3)
    multiple = validate_score("6-4 RET ABD", best_of=3)

    assert not result.valid
    assert result.anomaly_codes == (SCORE_INVALID_STATUS,)
    assert not multiple.valid
    assert multiple.anomaly_codes == (SCORE_INVALID_STATUS,)


def test_walkover_cannot_contain_played_sets() -> None:
    result = validate_score("6-4 W/O", best_of=3)

    assert not result.valid
    assert result.walkover
    assert result.anomaly_codes == (SCORE_WALKOVER_WITH_SCORE,)
    assert result.regular_service_games is None


@pytest.mark.parametrize("score", ["[10-8]", "6-4 [10-8]"])
def test_match_tiebreak_requires_a_deciding_set_context(score: str) -> None:
    result = validate_score(score, best_of=3)

    assert not result.valid
    assert SCORE_MATCH_TIEBREAK_CONTEXT in result.anomaly_codes


def test_match_tiebreak_must_be_last_score_token() -> None:
    result = validate_score("6-4 [10-8] 3-6", best_of=3)

    assert not result.valid
    assert SCORE_MATCH_TIEBREAK_CONTEXT in result.anomaly_codes


@pytest.mark.parametrize("score", [None, "", "   "])
def test_missing_scores_are_not_completed(score: str | None) -> None:
    result = validate_score(score, best_of=3)

    assert not result.valid
    assert not result.completed
    assert result.anomaly_codes == (SCORE_MISSING,)
    assert result.raw_score == score


@pytest.mark.parametrize("best_of", [0, 1, 2, 4, 7])
def test_rejects_unsupported_match_formats(best_of: int) -> None:
    result = validate_score("6-4 6-3", best_of=best_of)

    assert not result.valid
    assert result.anomaly_codes == (SCORE_INVALID_BEST_OF,)


def test_result_and_parsed_sets_are_frozen() -> None:
    result = validate_score("6-4 6-3", best_of=3)

    with pytest.raises(FrozenInstanceError):
        result.valid = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.sets[0].player_score = 0  # type: ignore[misc]
