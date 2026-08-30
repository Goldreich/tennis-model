"""Conservative validation for winner-oriented historical score strings.

This module intentionally does not implement tennis scoring.  It recognizes the
small score grammar needed at ingestion time and rejects only states that are
unambiguously impossible.  In particular, long advantage sets remain valid so
that historical rows are not judged under modern tiebreak rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

SCORE_MISSING: Final = "SCORE_MISSING"
SCORE_INVALID_BEST_OF: Final = "SCORE_INVALID_BEST_OF"
SCORE_MALFORMED: Final = "SCORE_MALFORMED"
SCORE_INVALID_STATUS: Final = "SCORE_INVALID_STATUS"
SCORE_WALKOVER_WITH_SCORE: Final = "SCORE_WALKOVER_WITH_SCORE"
SCORE_ILLEGAL_SET: Final = "SCORE_ILLEGAL_SET"
SCORE_TOO_MANY_SETS: Final = "SCORE_TOO_MANY_SETS"
SCORE_SET_AFTER_MATCH_END: Final = "SCORE_SET_AFTER_MATCH_END"
SCORE_INCOMPLETE_MATCH: Final = "SCORE_INCOMPLETE_MATCH"
SCORE_WINNER_ORIENTATION_MISMATCH: Final = "SCORE_WINNER_ORIENTATION_MISMATCH"
SCORE_STATUS_AFTER_COMPLETION: Final = "SCORE_STATUS_AFTER_COMPLETION"
SCORE_MATCH_TIEBREAK_CONTEXT: Final = "SCORE_MATCH_TIEBREAK_CONTEXT"


class ScoreTermination(StrEnum):
    """How the score string says play ended."""

    COMPLETED = "completed"
    RETIREMENT = "retirement"
    WALKOVER = "walkover"
    DEFAULT = "default"
    ABANDONED = "abandoned"


class SetScoreKind(StrEnum):
    """Whether a token records games or a bracketed match tiebreak."""

    REGULAR = "regular"
    MATCH_TIEBREAK = "match_tiebreak"


@dataclass(frozen=True, slots=True)
class ParsedSetScore:
    """One parsed score token, always from the listed winner's perspective."""

    player_score: int
    opponent_score: int
    kind: SetScoreKind
    completed: bool
    winner_index: int | None
    tiebreak_loser_points: int | None = None


@dataclass(frozen=True, slots=True)
class ScoreValidationResult:
    """Typed, immutable result suitable for historical-ingestion decisions.

    ``parsed_set_wins`` is ``(listed_winner, listed_loser)``.  A 7-6 set
    contributes twelve, not thirteen, to ``regular_service_games`` because its
    tiebreak is not a regular service game, while it contributes thirteen to
    ``official_games``. Bracketed match tiebreaks contribute zero to both.
    Consumers must still require ``valid`` before using any derived value.
    """

    raw_score: str | None
    valid: bool
    completed: bool
    retirement: bool
    walkover: bool
    anomaly_codes: tuple[str, ...]
    parsed_set_wins: tuple[int, int]
    regular_service_games: int | None
    official_games: int | None
    sets: tuple[ParsedSetScore, ...]
    termination: ScoreTermination


_REGULAR_SET_RE: Final = re.compile(
    r"^(?P<player>\d+)-(?P<opponent>\d+)(?:\((?P<tiebreak>\d+)\))?$"
)
_MATCH_TIEBREAK_RE: Final = re.compile(r"^\[(?P<player>\d+)-(?P<opponent>\d+)\]$")
_STATUS_ALIASES: Final = {
    "RET": ScoreTermination.RETIREMENT,
    "RET.": ScoreTermination.RETIREMENT,
    "W/O": ScoreTermination.WALKOVER,
    "WO": ScoreTermination.WALKOVER,
    "DEF": ScoreTermination.DEFAULT,
    "ABN": ScoreTermination.ABANDONED,
    "ABD": ScoreTermination.ABANDONED,
}
_MATCH_TIEBREAK_TARGETS: Final = (7, 10)


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _regular_set_winner(player: int, opponent: int) -> int | None:
    high = max(player, opponent)
    low = min(player, opponent)
    if high == 6 and low <= 4:
        return 0 if player > opponent else 1
    if high == 7 and low in (5, 6):
        return 0 if player > opponent else 1
    if high >= 8 and low >= 6 and high - low == 2:
        return 0 if player > opponent else 1
    return None


def _plausible_partial_regular_set(player: int, opponent: int) -> bool:
    if _regular_set_winner(player, opponent) is not None:
        return False
    high = max(player, opponent)
    low = min(player, opponent)
    if high <= 5:
        return True
    if high == 6:
        return low in (5, 6)
    return low >= 6 and high - low <= 1


def _tiebreak_winner_for_target(player: int, opponent: int, target: int) -> int | None:
    high = max(player, opponent)
    low = min(player, opponent)
    if high == target and low <= target - 2:
        return 0 if player > opponent else 1
    if high > target and high - low == 2:
        return 0 if player > opponent else 1
    return None


def _match_tiebreak_winner(player: int, opponent: int) -> int | None:
    for target in _MATCH_TIEBREAK_TARGETS:
        winner = _tiebreak_winner_for_target(player, opponent, target)
        if winner is not None:
            return winner
    return None


def _plausible_partial_match_tiebreak(player: int, opponent: int) -> bool:
    if _match_tiebreak_winner(player, opponent) is not None:
        return False
    high = max(player, opponent)
    low = min(player, opponent)
    return any(high < target or high - low <= 1 for target in _MATCH_TIEBREAK_TARGETS)


def _regular_service_games(parsed_set: ParsedSetScore) -> int:
    if parsed_set.kind is SetScoreKind.MATCH_TIEBREAK:
        return 0
    games = parsed_set.player_score + parsed_set.opponent_score
    if {parsed_set.player_score, parsed_set.opponent_score} == {6, 7}:
        games -= 1
    return games


def _official_games(parsed_set: ParsedSetScore) -> int:
    if parsed_set.kind is SetScoreKind.MATCH_TIEBREAK:
        return 0
    return parsed_set.player_score + parsed_set.opponent_score


def _result(
    *,
    raw_score: str | None,
    termination: ScoreTermination,
    anomalies: list[str],
    parsed_sets: list[ParsedSetScore],
    set_wins: tuple[int, int],
    completed: bool,
    games_are_known: bool,
) -> ScoreValidationResult:
    return ScoreValidationResult(
        raw_score=raw_score,
        valid=not anomalies,
        completed=completed,
        retirement=termination is ScoreTermination.RETIREMENT,
        walkover=termination is ScoreTermination.WALKOVER,
        anomaly_codes=_deduplicate(anomalies),
        parsed_set_wins=set_wins,
        regular_service_games=(
            sum(_regular_service_games(item) for item in parsed_sets) if games_are_known else None
        ),
        official_games=(
            sum(_official_games(item) for item in parsed_sets) if games_are_known else None
        ),
        sets=tuple(parsed_sets),
        termination=termination,
    )


def validate_score(raw_score: str | None, *, best_of: int) -> ScoreValidationResult:
    """Validate a Sackmann-style winner-oriented singles score.

    The absence of a terminal marker means the score claims normal completion.
    ``RET``/``RET.`` may follow one partial last set.  Defaults and abandoned
    matches use the same conservative structural parsing but are kept distinct
    from retirements.  Walkovers must not contain played-set tokens.
    """

    if best_of not in (3, 5):
        return _result(
            raw_score=raw_score,
            termination=ScoreTermination.COMPLETED,
            anomalies=[SCORE_INVALID_BEST_OF],
            parsed_sets=[],
            set_wins=(0, 0),
            completed=False,
            games_are_known=False,
        )

    if raw_score is None or not raw_score.strip():
        return _result(
            raw_score=raw_score,
            termination=ScoreTermination.COMPLETED,
            anomalies=[SCORE_MISSING],
            parsed_sets=[],
            set_wins=(0, 0),
            completed=False,
            games_are_known=False,
        )

    tokens = raw_score.strip().split()
    status_positions = [
        index for index, token in enumerate(tokens) if token.upper() in _STATUS_ALIASES
    ]
    termination = ScoreTermination.COMPLETED
    anomalies: list[str] = []
    if status_positions:
        termination = _STATUS_ALIASES[tokens[status_positions[-1]].upper()]
        if len(status_positions) != 1 or status_positions[0] != len(tokens) - 1:
            anomalies.append(SCORE_INVALID_STATUS)
            return _result(
                raw_score=raw_score,
                termination=termination,
                anomalies=anomalies,
                parsed_sets=[],
                set_wins=(0, 0),
                completed=False,
                games_are_known=False,
            )
        tokens = tokens[:-1]

    if termination is ScoreTermination.WALKOVER:
        if tokens:
            anomalies.append(SCORE_WALKOVER_WITH_SCORE)
        return _result(
            raw_score=raw_score,
            termination=termination,
            anomalies=anomalies,
            parsed_sets=[],
            set_wins=(0, 0),
            completed=False,
            games_are_known=not tokens,
        )

    if not tokens:
        if termination is ScoreTermination.COMPLETED:
            anomalies.append(SCORE_MISSING)
        return _result(
            raw_score=raw_score,
            termination=termination,
            anomalies=anomalies,
            parsed_sets=[],
            set_wins=(0, 0),
            completed=False,
            games_are_known=False,
        )

    permits_partial_last_set = termination is not ScoreTermination.COMPLETED
    parsed_sets: list[ParsedSetScore] = []
    games_are_known = True

    for index, token in enumerate(tokens):
        is_last = index == len(tokens) - 1
        match_tiebreak_match = _MATCH_TIEBREAK_RE.fullmatch(token)
        regular_match = _REGULAR_SET_RE.fullmatch(token)

        if match_tiebreak_match is not None:
            player = int(match_tiebreak_match.group("player"))
            opponent = int(match_tiebreak_match.group("opponent"))
            winner = _match_tiebreak_winner(player, opponent)
            completed_set = winner is not None
            if not is_last:
                anomalies.append(SCORE_MATCH_TIEBREAK_CONTEXT)
            if not completed_set and not (
                permits_partial_last_set
                and is_last
                and _plausible_partial_match_tiebreak(player, opponent)
            ):
                anomalies.append(SCORE_ILLEGAL_SET)
            parsed_sets.append(
                ParsedSetScore(
                    player_score=player,
                    opponent_score=opponent,
                    kind=SetScoreKind.MATCH_TIEBREAK,
                    completed=completed_set,
                    winner_index=winner,
                )
            )
            continue

        if regular_match is None:
            anomalies.append(SCORE_MALFORMED)
            games_are_known = False
            break

        player = int(regular_match.group("player"))
        opponent = int(regular_match.group("opponent"))
        tiebreak_text = regular_match.group("tiebreak")
        winner = _regular_set_winner(player, opponent)
        completed_set = winner is not None
        tiebreak_is_legal = tiebreak_text is None or {player, opponent} == {6, 7}
        partial_is_legal = (
            permits_partial_last_set
            and is_last
            and tiebreak_text is None
            and _plausible_partial_regular_set(player, opponent)
        )
        if not tiebreak_is_legal or (not completed_set and not partial_is_legal):
            anomalies.append(SCORE_ILLEGAL_SET)
        parsed_sets.append(
            ParsedSetScore(
                player_score=player,
                opponent_score=opponent,
                kind=SetScoreKind.REGULAR,
                completed=completed_set,
                winner_index=winner,
                tiebreak_loser_points=(int(tiebreak_text) if tiebreak_text is not None else None),
            )
        )

    if len(parsed_sets) != len(tokens):
        return _result(
            raw_score=raw_score,
            termination=termination,
            anomalies=anomalies,
            parsed_sets=parsed_sets,
            set_wins=(0, 0),
            completed=False,
            games_are_known=games_are_known,
        )

    if len(parsed_sets) > best_of:
        anomalies.append(SCORE_TOO_MANY_SETS)

    target_sets = best_of // 2 + 1
    player_sets = 0
    opponent_sets = 0
    partial_seen = False
    for index, parsed_set in enumerate(parsed_sets):
        if player_sets == target_sets or opponent_sets == target_sets:
            anomalies.append(SCORE_SET_AFTER_MATCH_END)
        if parsed_set.kind is SetScoreKind.MATCH_TIEBREAK and (
            player_sets != target_sets - 1 or opponent_sets != target_sets - 1
        ):
            anomalies.append(SCORE_MATCH_TIEBREAK_CONTEXT)
        if parsed_set.completed:
            if parsed_set.winner_index == 0:
                player_sets += 1
            else:
                opponent_sets += 1
        else:
            partial_seen = True
            if index != len(parsed_sets) - 1:
                anomalies.append(SCORE_ILLEGAL_SET)

    structurally_completed = (
        not partial_seen
        and (player_sets == target_sets or opponent_sets == target_sets)
        and SCORE_SET_AFTER_MATCH_END not in anomalies
        and SCORE_TOO_MANY_SETS not in anomalies
    )

    if termination is ScoreTermination.COMPLETED:
        if opponent_sets == target_sets and player_sets < target_sets:
            anomalies.append(SCORE_WINNER_ORIENTATION_MISMATCH)
        elif player_sets != target_sets:
            anomalies.append(SCORE_INCOMPLETE_MATCH)
    elif player_sets == target_sets or opponent_sets == target_sets:
        anomalies.append(SCORE_STATUS_AFTER_COMPLETION)

    return _result(
        raw_score=raw_score,
        termination=termination,
        anomalies=anomalies,
        parsed_sets=parsed_sets,
        set_wins=(player_sets, opponent_sets),
        completed=(termination is ScoreTermination.COMPLETED and structurally_completed),
        games_are_known=games_are_known,
    )


validate_sackmann_score = validate_score


__all__ = [
    "SCORE_ILLEGAL_SET",
    "SCORE_INCOMPLETE_MATCH",
    "SCORE_INVALID_BEST_OF",
    "SCORE_INVALID_STATUS",
    "SCORE_MALFORMED",
    "SCORE_MATCH_TIEBREAK_CONTEXT",
    "SCORE_MISSING",
    "SCORE_SET_AFTER_MATCH_END",
    "SCORE_STATUS_AFTER_COMPLETION",
    "SCORE_TOO_MANY_SETS",
    "SCORE_WALKOVER_WITH_SCORE",
    "SCORE_WINNER_ORIENTATION_MISMATCH",
    "ParsedSetScore",
    "ScoreTermination",
    "ScoreValidationResult",
    "SetScoreKind",
    "validate_sackmann_score",
    "validate_score",
]
