"""Pure deterministic tennis scoring for the frozen US Open rules.

The state machine consumes only an ordered stream of point-winner indices.  It
does not generate point outcomes, read data, evaluate markets, or apply
settlement policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

type PlayerIndex = Literal[0, 1]
type Score = tuple[int, int]
type GameScore = tuple[str, str]


class ScoringError(ValueError):
    """Base class for invalid scoring input or state."""


class ScoringInvariantError(ScoringError):
    """A constructed scoring value violates the frozen tennis rules."""


class MatchAlreadyCompleteError(ScoringError):
    """A point was supplied after the match reached a legal terminal state."""


def _player_index(value: int, *, field: str) -> PlayerIndex:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ScoringInvariantError(f"{field} must be player index 0 or 1")
    return cast(PlayerIndex, value)


def _score(value: Score, *, field: str) -> Score:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ScoringInvariantError(f"{field} must contain exactly two integers")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ScoringInvariantError(f"{field} values must be nonnegative integers")
    return value


def _increment(score: Score, player: PlayerIndex) -> Score:
    values = [score[0], score[1]]
    values[player] += 1
    return values[0], values[1]


def other_player(player: int) -> PlayerIndex:
    """Return the opposing player index."""

    canonical = _player_index(player, field="player")
    return cast(PlayerIndex, 1 - canonical)


def _terminal_winner(score: Score, target: int) -> PlayerIndex | None:
    for player in (0, 1):
        opponent = 1 - player
        if score[player] >= target and score[player] - score[opponent] >= 2:
            return player
    return None


def _is_minimal_terminal_score(score: Score, target: int) -> bool:
    winner = _terminal_winner(score, target)
    if winner is None:
        return False
    loser = other_player(winner)
    if score[loser] <= target - 2:
        return score[winner] == target
    return score[winner] == score[loser] + 2


def tiebreak_server(first_server_index: int, points_played: int) -> PlayerIndex:
    """Return the server for the next tiebreak point under the 1-2-2 sequence."""

    first_server = _player_index(first_server_index, field="first_server_index")
    if isinstance(points_played, bool) or not isinstance(points_played, int) or points_played < 0:
        raise ScoringInvariantError("points_played must be a nonnegative integer")
    if points_played == 0:
        return first_server
    block = (points_played - 1) // 2
    return other_player(first_server) if block % 2 == 0 else first_server


def is_break_point(game_points: Score, server_index: int) -> bool:
    """Return whether the next regular-game point is a break-point opportunity."""

    points = _score(game_points, field="game_points")
    if _terminal_winner(points, 4) is not None:
        raise ScoringInvariantError("a completed game has no next break-point state")
    server = _player_index(server_index, field="server_index")
    receiver = other_player(server)
    return points[receiver] >= 3 and points[receiver] > points[server]


def format_game_score(game_points: Score) -> GameScore:
    """Render a legal ongoing advantage-game score in player order."""

    points = _score(game_points, field="game_points")
    if _terminal_winner(points, 4) is not None:
        raise ScoringInvariantError("a completed game has no ongoing point score")
    if points[0] >= 3 and points[1] >= 3:
        if points[0] == points[1]:
            return "40", "40"
        if points[0] > points[1]:
            return "AD", "40"
        return "40", "AD"
    labels = ("0", "15", "30", "40")
    return labels[min(points[0], 3)], labels[min(points[1], 3)]


def is_legal_completed_set_score(games: Score, *, tiebreak_played: bool) -> bool:
    """Return whether ``games`` is terminal under the frozen 6-6 tiebreak rules."""

    try:
        canonical = _score(games, field="games")
    except ScoringInvariantError:
        return False
    high = max(canonical)
    low = min(canonical)
    if high == low:
        return False
    if tiebreak_played:
        return high == 7 and low == 6
    return (high == 6 and low <= 4) or (high == 7 and low == 5)


@dataclass(frozen=True, slots=True)
class TiebreakState:
    """An ongoing standard or deciding-set tiebreak."""

    first_server_index: PlayerIndex
    target_points: Literal[7, 10]
    points: Score = (0, 0)

    def __post_init__(self) -> None:
        _player_index(self.first_server_index, field="first_server_index")
        if (
            isinstance(self.target_points, bool)
            or not isinstance(self.target_points, int)
            or self.target_points not in (7, 10)
        ):
            raise ScoringInvariantError("tiebreak target must be 7 or 10")
        _score(self.points, field="tiebreak points")
        if _terminal_winner(self.points, self.target_points) is not None:
            raise ScoringInvariantError("active tiebreak points are already terminal")

    @property
    def points_played(self) -> int:
        return sum(self.points)

    @property
    def server_index(self) -> PlayerIndex:
        return tiebreak_server(self.first_server_index, self.points_played)


@dataclass(frozen=True, slots=True)
class TiebreakResult:
    """A minimally terminal tiebreak score and its first server."""

    first_server_index: PlayerIndex
    target_points: Literal[7, 10]
    points: Score

    def __post_init__(self) -> None:
        _player_index(self.first_server_index, field="first_server_index")
        if (
            isinstance(self.target_points, bool)
            or not isinstance(self.target_points, int)
            or self.target_points not in (7, 10)
        ):
            raise ScoringInvariantError("tiebreak target must be 7 or 10")
        _score(self.points, field="tiebreak points")
        if not _is_minimal_terminal_score(self.points, self.target_points):
            raise ScoringInvariantError("tiebreak result is not a legal minimal terminal score")

    @property
    def winner_index(self) -> PlayerIndex:
        winner = _terminal_winner(self.points, self.target_points)
        if winner is None:  # Guaranteed by validation.
            raise AssertionError("validated tiebreak result lost its winner")
        return winner

    @property
    def points_played(self) -> int:
        return sum(self.points)


@dataclass(frozen=True, slots=True)
class SetState:
    """An ongoing set containing either a regular game or a tiebreak."""

    set_number: int
    first_server_index: PlayerIndex
    games: Score = (0, 0)
    game_points: Score = (0, 0)
    tiebreak: TiebreakState | None = None

    def __post_init__(self) -> None:
        if isinstance(self.set_number, bool) or not isinstance(self.set_number, int):
            raise ScoringInvariantError("set_number must be a positive integer")
        if self.set_number < 1:
            raise ScoringInvariantError("set_number must be a positive integer")
        _player_index(self.first_server_index, field="first_server_index")
        games = _score(self.games, field="games")
        points = _score(self.game_points, field="game_points")

        if self.tiebreak is None:
            if max(games) > 6:
                raise ScoringInvariantError("active regular set cannot exceed six games")
            if games == (6, 6):
                raise ScoringInvariantError("a set at 6-6 must be in a tiebreak")
            if _terminal_winner(games, 6) is not None:
                raise ScoringInvariantError("active set games are already terminal")
            if _terminal_winner(points, 4) is not None:
                raise ScoringInvariantError("active regular game points are already terminal")
            return

        if not isinstance(self.tiebreak, TiebreakState):
            raise ScoringInvariantError("active tiebreak must be a TiebreakState")
        if games != (6, 6):
            raise ScoringInvariantError("a tiebreak may begin only at 6-6")
        if points != (0, 0):
            raise ScoringInvariantError("regular-game points must be zero during a tiebreak")
        if self.tiebreak.first_server_index != self.first_server_index:
            raise ScoringInvariantError(
                "tiebreak must start with the player due after twelve games"
            )

    @property
    def in_tiebreak(self) -> bool:
        return self.tiebreak is not None

    @property
    def server_index(self) -> PlayerIndex:
        if self.tiebreak is not None:
            return self.tiebreak.server_index
        games_played = sum(self.games)
        if games_played % 2 == 0:
            return self.first_server_index
        return other_player(self.first_server_index)

    @property
    def displayed_game_score(self) -> GameScore | None:
        return None if self.tiebreak is not None else format_game_score(self.game_points)


@dataclass(frozen=True, slots=True)
class SetResult:
    """A legal completed set under the frozen current-US-Open rules."""

    set_number: int
    first_server_index: PlayerIndex
    games: Score
    tiebreak: TiebreakResult | None = None

    def __post_init__(self) -> None:
        if isinstance(self.set_number, bool) or not isinstance(self.set_number, int):
            raise ScoringInvariantError("set_number must be a positive integer")
        if self.set_number < 1:
            raise ScoringInvariantError("set_number must be a positive integer")
        _player_index(self.first_server_index, field="first_server_index")
        _score(self.games, field="games")
        if self.tiebreak is not None and not isinstance(self.tiebreak, TiebreakResult):
            raise ScoringInvariantError("completed tiebreak must be a TiebreakResult")
        if not is_legal_completed_set_score(self.games, tiebreak_played=self.tiebreak is not None):
            raise ScoringInvariantError("completed set has an impossible game score")
        if self.tiebreak is None:
            return
        if self.tiebreak.first_server_index != self.first_server_index:
            raise ScoringInvariantError(
                "tiebreak first server is inconsistent with set service order"
            )
        if self.tiebreak.winner_index != self.winner_index:
            raise ScoringInvariantError("tiebreak and set winners must agree")

    @property
    def winner_index(self) -> PlayerIndex:
        return 0 if self.games[0] > self.games[1] else 1

    @property
    def total_games(self) -> int:
        """Official games; a completed tiebreak contributes exactly one."""

        return sum(self.games)

    @property
    def regular_games(self) -> int:
        return self.total_games - int(self.tiebreak is not None)

    @property
    def regular_games_won(self) -> Score:
        games = [self.games[0], self.games[1]]
        if self.tiebreak is not None:
            games[self.tiebreak.winner_index] -= 1
        return games[0], games[1]

    @property
    def next_set_server_index(self) -> PlayerIndex:
        if self.tiebreak is not None:
            return other_player(self.tiebreak.first_server_index)
        if self.total_games % 2 == 0:
            return self.first_server_index
        return other_player(self.first_server_index)


@dataclass(frozen=True, slots=True)
class MatchState:
    """Complete immutable state for one deterministic best-of-three/five match."""

    players: tuple[str, str]
    best_of: Literal[3, 5]
    initial_server_index: PlayerIndex
    completed_sets: tuple[SetResult, ...]
    active_set: SetState | None
    break_point_opportunities: Score = (0, 0)
    breaks_of_serve: Score = (0, 0)
    total_points_played: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.players, tuple) or len(self.players) != 2:
            raise ScoringInvariantError("players must contain exactly two IDs")
        if any(not isinstance(player, str) or not player.strip() for player in self.players):
            raise ScoringInvariantError("player IDs must be nonempty strings")
        if self.players[0] == self.players[1]:
            raise ScoringInvariantError("player IDs must be distinct")
        if (
            isinstance(self.best_of, bool)
            or not isinstance(self.best_of, int)
            or self.best_of not in (3, 5)
        ):
            raise ScoringInvariantError("best_of must be 3 or 5")
        _player_index(self.initial_server_index, field="initial_server_index")
        if not isinstance(self.completed_sets, tuple):
            raise ScoringInvariantError("completed_sets must be an immutable tuple")
        if any(not isinstance(result, SetResult) for result in self.completed_sets):
            raise ScoringInvariantError("completed_sets must contain only SetResult values")
        if self.active_set is not None and not isinstance(self.active_set, SetState):
            raise ScoringInvariantError("active_set must be a SetState or None")
        opportunities = _score(self.break_point_opportunities, field="break_point_opportunities")
        breaks = _score(self.breaks_of_serve, field="breaks_of_serve")
        if any(breaks[player] > opportunities[player] for player in (0, 1)):
            raise ScoringInvariantError("breaks cannot exceed break-point opportunities")
        if (
            isinstance(self.total_points_played, bool)
            or not isinstance(self.total_points_played, int)
            or self.total_points_played < 0
        ):
            raise ScoringInvariantError("total_points_played must be a nonnegative integer")

        regular_game_wins = [0, 0]
        tiebreak_points_played = 0
        minimum_points_played = 0
        for result in self.completed_sets:
            for player in (0, 1):
                regular_game_wins[player] += result.regular_games_won[player]
            minimum_points_played += 4 * result.regular_games
            if result.tiebreak is not None:
                tiebreak_points_played += result.tiebreak.points_played
                minimum_points_played += result.tiebreak.points_played
        if self.active_set is not None:
            for player in (0, 1):
                regular_game_wins[player] += self.active_set.games[player]
            minimum_points_played += 4 * sum(self.active_set.games)
            if self.active_set.tiebreak is None:
                minimum_points_played += sum(self.active_set.game_points)
            else:
                tiebreak_points_played += self.active_set.tiebreak.points_played
                minimum_points_played += self.active_set.tiebreak.points_played

        wins = [0, 0]
        expected_server = self.initial_server_index
        for position, result in enumerate(self.completed_sets, start=1):
            if result.set_number != position:
                raise ScoringInvariantError("completed set numbers must be consecutive")
            if result.first_server_index != expected_server:
                raise ScoringInvariantError("set first server violates continuous service order")
            expected_target = 10 if position == self.best_of else 7
            if result.tiebreak is not None and result.tiebreak.target_points != expected_target:
                raise ScoringInvariantError("completed set used the wrong tiebreak target")
            wins[result.winner_index] += 1
            expected_server = result.next_set_server_index
            if wins[result.winner_index] == self.sets_to_win and position != len(
                self.completed_sets
            ):
                raise ScoringInvariantError("sets appear after the match-clinching set")

        winner = next(
            (player for player in (0, 1) if wins[player] == self.sets_to_win),
            None,
        )
        if winner is not None:
            if self.active_set is not None:
                raise ScoringInvariantError("a completed match cannot contain an active set")
        else:
            if self.active_set is None:
                raise ScoringInvariantError("an incomplete match must contain an active set")
            if self.active_set.set_number != len(self.completed_sets) + 1:
                raise ScoringInvariantError("active set number must follow completed sets")
            if self.active_set.set_number > self.best_of:
                raise ScoringInvariantError("active set exceeds the match format")
            if self.active_set.first_server_index != expected_server:
                raise ScoringInvariantError(
                    "active-set first server violates continuous service order"
                )
            expected_target = 10 if self.active_set.set_number == self.best_of else 7
            if (
                self.active_set.tiebreak is not None
                and self.active_set.tiebreak.target_points != expected_target
            ):
                raise ScoringInvariantError("active set used the wrong tiebreak target")

        if self.total_points_played < minimum_points_played:
            raise ScoringInvariantError("total points are inconsistent with the recorded score")
        regular_points_played = self.total_points_played - tiebreak_points_played
        if sum(opportunities) > regular_points_played:
            raise ScoringInvariantError(
                "break-point opportunities cannot exceed regular points played"
            )
        if any(breaks[player] > regular_game_wins[player] for player in (0, 1)):
            raise ScoringInvariantError("breaks cannot exceed regular games won")

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    @property
    def sets_won(self) -> Score:
        wins = [0, 0]
        for result in self.completed_sets:
            wins[result.winner_index] += 1
        return wins[0], wins[1]

    @property
    def winner_index(self) -> PlayerIndex | None:
        wins = self.sets_won
        for player in (0, 1):
            if wins[player] == self.sets_to_win:
                return player
        return None

    @property
    def winner_id(self) -> str | None:
        winner = self.winner_index
        return None if winner is None else self.players[winner]

    @property
    def is_complete(self) -> bool:
        return self.winner_index is not None

    @property
    def server_index(self) -> PlayerIndex | None:
        return None if self.active_set is None else self.active_set.server_index

    @property
    def server_id(self) -> str | None:
        server = self.server_index
        return None if server is None else self.players[server]

    @property
    def exact_set_scores(self) -> tuple[Score, ...]:
        return tuple(result.games for result in self.completed_sets)

    @property
    def set_winners(self) -> tuple[PlayerIndex, ...]:
        return tuple(result.winner_index for result in self.completed_sets)

    @property
    def total_games(self) -> int:
        completed = sum(result.total_games for result in self.completed_sets)
        active = 0 if self.active_set is None else sum(self.active_set.games)
        return completed + active

    @property
    def regular_games(self) -> int:
        completed = sum(result.regular_games for result in self.completed_sets)
        active = 0 if self.active_set is None else sum(self.active_set.games)
        return completed + active

    @property
    def tiebreaks_started(self) -> int:
        completed = sum(result.tiebreak is not None for result in self.completed_sets)
        active = int(self.active_set is not None and self.active_set.tiebreak is not None)
        return completed + active

    @property
    def deciding_set_began(self) -> bool:
        if any(result.set_number == self.best_of for result in self.completed_sets):
            return True
        return self.active_set is not None and self.active_set.set_number == self.best_of


@dataclass(frozen=True, slots=True)
class PointTransition:
    """Auditable facts produced by one pure point transition."""

    before: MatchState
    after: MatchState
    winner_index: PlayerIndex
    server_index: PlayerIndex
    receiver_index: PlayerIndex
    was_tiebreak: bool
    break_point_opportunity: bool
    regular_game_completed: bool
    break_of_serve: bool
    game_winner_index: PlayerIndex | None
    tiebreak_completed: bool
    completed_set: SetResult | None

    @property
    def match_completed(self) -> bool:
        return self.after.is_complete


def new_match(
    player_a_id: str,
    player_b_id: str,
    *,
    best_of: Literal[3, 5],
    first_server_index: PlayerIndex,
) -> MatchState:
    """Create a validated scoreless match."""

    first_server = _player_index(first_server_index, field="first_server_index")
    active = SetState(set_number=1, first_server_index=first_server)
    return MatchState(
        players=(player_a_id, player_b_id),
        best_of=best_of,
        initial_server_index=first_server,
        completed_sets=(),
        active_set=active,
    )


def _finish_set(
    state: MatchState,
    result: SetResult,
    *,
    break_point_opportunities: Score,
    breaks_of_serve: Score,
) -> MatchState:
    completed_sets = (*state.completed_sets, result)
    wins = [0, 0]
    for completed in completed_sets:
        wins[completed.winner_index] += 1
    active_set: SetState | None
    if wins[result.winner_index] == state.sets_to_win:
        active_set = None
    else:
        active_set = SetState(
            set_number=result.set_number + 1,
            first_server_index=result.next_set_server_index,
        )
    return MatchState(
        players=state.players,
        best_of=state.best_of,
        initial_server_index=state.initial_server_index,
        completed_sets=completed_sets,
        active_set=active_set,
        break_point_opportunities=break_point_opportunities,
        breaks_of_serve=breaks_of_serve,
        total_points_played=state.total_points_played + 1,
    )


def award_point(state: MatchState, winner_index: PlayerIndex) -> PointTransition:
    """Award one point and return the resulting immutable state and audit facts."""

    winner = _player_index(winner_index, field="winner_index")
    if state.is_complete or state.active_set is None:
        raise MatchAlreadyCompleteError("cannot award a point after match completion")

    active = state.active_set
    server = active.server_index
    receiver = other_player(server)
    in_tiebreak = active.tiebreak is not None
    break_point = not in_tiebreak and is_break_point(active.game_points, server)
    opportunities = (
        _increment(state.break_point_opportunities, receiver)
        if break_point
        else state.break_point_opportunities
    )
    breaks = state.breaks_of_serve
    completed_set: SetResult | None = None
    regular_game_completed = False
    break_of_serve = False
    game_winner: PlayerIndex | None = None
    tiebreak_completed = False

    if active.tiebreak is not None:
        tiebreak_points = _increment(active.tiebreak.points, winner)
        tiebreak_winner = _terminal_winner(tiebreak_points, active.tiebreak.target_points)
        if tiebreak_winner is None:
            next_tiebreak = TiebreakState(
                first_server_index=active.tiebreak.first_server_index,
                target_points=active.tiebreak.target_points,
                points=tiebreak_points,
            )
            next_active = SetState(
                set_number=active.set_number,
                first_server_index=active.first_server_index,
                games=active.games,
                tiebreak=next_tiebreak,
            )
            after = MatchState(
                players=state.players,
                best_of=state.best_of,
                initial_server_index=state.initial_server_index,
                completed_sets=state.completed_sets,
                active_set=next_active,
                break_point_opportunities=opportunities,
                breaks_of_serve=breaks,
                total_points_played=state.total_points_played + 1,
            )
        else:
            tiebreak_completed = True
            tiebreak_result = TiebreakResult(
                first_server_index=active.tiebreak.first_server_index,
                target_points=active.tiebreak.target_points,
                points=tiebreak_points,
            )
            completed_set = SetResult(
                set_number=active.set_number,
                first_server_index=active.first_server_index,
                games=_increment(active.games, tiebreak_winner),
                tiebreak=tiebreak_result,
            )
            after = _finish_set(
                state,
                completed_set,
                break_point_opportunities=opportunities,
                breaks_of_serve=breaks,
            )
    else:
        game_points = _increment(active.game_points, winner)
        game_winner = _terminal_winner(game_points, 4)
        if game_winner is None:
            next_active = SetState(
                set_number=active.set_number,
                first_server_index=active.first_server_index,
                games=active.games,
                game_points=game_points,
            )
            after = MatchState(
                players=state.players,
                best_of=state.best_of,
                initial_server_index=state.initial_server_index,
                completed_sets=state.completed_sets,
                active_set=next_active,
                break_point_opportunities=opportunities,
                breaks_of_serve=breaks,
                total_points_played=state.total_points_played + 1,
            )
        else:
            regular_game_completed = True
            break_of_serve = game_winner != server
            if break_of_serve:
                breaks = _increment(breaks, game_winner)
            games = _increment(active.games, game_winner)
            set_winner = _terminal_winner(games, 6)
            if set_winner is not None:
                completed_set = SetResult(
                    set_number=active.set_number,
                    first_server_index=active.first_server_index,
                    games=games,
                )
                after = _finish_set(
                    state,
                    completed_set,
                    break_point_opportunities=opportunities,
                    breaks_of_serve=breaks,
                )
            else:
                starting_tiebreak: TiebreakState | None = None
                if games == (6, 6):
                    target: Literal[7, 10] = 10 if active.set_number == state.best_of else 7
                    starting_tiebreak = TiebreakState(
                        first_server_index=other_player(server),
                        target_points=target,
                    )
                next_active = SetState(
                    set_number=active.set_number,
                    first_server_index=active.first_server_index,
                    games=games,
                    tiebreak=starting_tiebreak,
                )
                after = MatchState(
                    players=state.players,
                    best_of=state.best_of,
                    initial_server_index=state.initial_server_index,
                    completed_sets=state.completed_sets,
                    active_set=next_active,
                    break_point_opportunities=opportunities,
                    breaks_of_serve=breaks,
                    total_points_played=state.total_points_played + 1,
                )

    return PointTransition(
        before=state,
        after=after,
        winner_index=winner,
        server_index=server,
        receiver_index=receiver,
        was_tiebreak=in_tiebreak,
        break_point_opportunity=break_point,
        regular_game_completed=regular_game_completed,
        break_of_serve=break_of_serve,
        game_winner_index=game_winner,
        tiebreak_completed=tiebreak_completed,
        completed_set=completed_set,
    )


def play_points(state: MatchState, winners: Iterable[PlayerIndex]) -> MatchState:
    """Fold a deterministic point-winner stream through ``state``."""

    current = state
    for winner in winners:
        current = award_point(current, winner).after
    return current


__all__ = [
    "GameScore",
    "MatchAlreadyCompleteError",
    "MatchState",
    "PlayerIndex",
    "PointTransition",
    "Score",
    "ScoringError",
    "ScoringInvariantError",
    "SetResult",
    "SetState",
    "TiebreakResult",
    "TiebreakState",
    "award_point",
    "format_game_score",
    "is_break_point",
    "is_legal_completed_set_score",
    "new_match",
    "other_player",
    "play_points",
    "tiebreak_server",
]
