from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import floor, isfinite, log1p, sqrt
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np

from tennis_model.estimation.duration_model import (
    UNRESOLVED_DURATION_DISPLAY_POLICY,
    DurationConditions,
    DurationDisplayPolicy,
    DurationPathExposure,
    draw_duration,
    prepare_duration_parameter_sampler,
    sample_prepared_duration_parameters,
)
from tennis_model.estimation.retirement import (
    CompetingRetirementOutcome,
    draw_competing_retirement,
)
from tennis_model.props.settlement import (
    CANONICAL_SETTLEMENT_POLICY_VERSION,
    ComparisonOperator,
    EventTruth,
    PolicyBlockedError,
    SettlementPolicy,
    SettlementState,
    truth_and,
    truth_or,
)
from tennis_model.simulation.parameters import (
    MatchParameterDistribution,
    SeedReference,
    generator_from_seed_reference,
    sample_match_performance,
)
from tennis_model.simulation.point import (
    ServePerformanceDraw,
    ServicePointResult,
    generate_service_point,
)
from tennis_model.simulation.scoring import (
    MatchState,
    PointTransition,
    SetResult,
    award_point,
    new_match,
)


@dataclass(frozen=True, slots=True)
class BreakEvent:
    set_number: int
    game_number: int
    match_game_number: int
    server_id: str
    receiver_id: str
    break_player_id: str

    def __post_init__(self) -> None:
        locations = (self.set_number, self.game_number, self.match_game_number)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in locations):
            raise TypeError("break locations must be integers")
        if self.set_number < 1 or self.game_number < 1 or self.match_game_number < 1:
            raise ValueError("break locations must be positive game numbers")
        if self.server_id == self.receiver_id:
            raise ValueError("break server and receiver must be distinct")
        if self.break_player_id != self.receiver_id:
            raise ValueError("a break must be credited to the returner")


@dataclass(frozen=True, slots=True)
class PlayerMatchStats:
    player_id: str
    games_won: int
    service_games_played: int
    return_games_played: int
    service_games_held: int
    breaks_conceded: int
    breaks_achieved: int
    service_points: int
    first_serve_opportunities: int
    first_serves_in: int
    first_serve_points_won: int
    returnable_first_serve_trials: int
    returnable_first_serve_wins: int
    second_serve_opportunities: int
    double_faults: int
    playable_second_serve_trials: int
    playable_second_serve_wins: int
    second_serve_points_won: int
    aces: int
    break_point_opportunities: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.player_id, str) or not self.player_id.strip():
            raise ValueError("player_id must be a nonempty string")
        for name in (
            "games_won",
            "service_games_played",
            "return_games_played",
            "service_games_held",
            "breaks_conceded",
            "breaks_achieved",
            "service_points",
            "first_serve_opportunities",
            "first_serves_in",
            "first_serve_points_won",
            "returnable_first_serve_trials",
            "returnable_first_serve_wins",
            "second_serve_opportunities",
            "double_faults",
            "playable_second_serve_trials",
            "playable_second_serve_wins",
            "second_serve_points_won",
            "aces",
            "break_point_opportunities",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.service_points != self.first_serves_in + self.second_serve_opportunities:
            raise ValueError(
                "service_points must equal first_serves_in + second_serve_opportunities"
            )
        if self.first_serve_opportunities != self.service_points:
            raise ValueError("every service point must be a first-serve opportunity")
        if self.first_serves_in != self.aces + self.returnable_first_serve_trials:
            raise ValueError("first_serves_in must equal aces + returnable_first_serve_trials")
        if self.first_serve_points_won != self.aces + self.returnable_first_serve_wins:
            raise ValueError("first_serve_points_won must equal aces + returnable_first_serve_wins")
        if (
            self.second_serve_opportunities
            != self.double_faults + self.playable_second_serve_trials
        ):
            raise ValueError(
                "second_serve_opportunities must equal double_faults + playable_second_serve_trials"
            )
        if self.second_serve_points_won != self.playable_second_serve_wins:
            raise ValueError("second_serve_points_won must equal playable_second_serve_wins")
        if self.first_serve_points_won > self.first_serves_in:
            raise ValueError("first-serve points won cannot exceed first serves in")
        if self.second_serve_points_won > self.second_serve_opportunities:
            raise ValueError("second-serve points won cannot exceed second-serve opportunities")
        if self.service_games_held + self.breaks_conceded != self.service_games_played:
            raise ValueError("service games must partition into holds and breaks conceded")
        if self.breaks_achieved > self.return_games_played:
            raise ValueError("breaks achieved cannot exceed return games played")
        if self.breaks_achieved > self.break_point_opportunities:
            raise ValueError("breaks achieved cannot exceed break-point opportunities")


@dataclass(frozen=True, slots=True)
class MatchPath:
    winner_id: str | None
    player_a_id: str
    player_b_id: str
    best_of: int
    first_server_id: str
    sets: tuple[SetResult, ...]
    player_stats: dict[str, PlayerMatchStats]
    break_events: tuple[BreakEvent, ...] = ()
    tiebreak_count: int = 0
    total_games: int = 0
    completed: bool = True
    started: bool = True
    retired_player_id: str | None = None
    walkover: bool = False
    sets_started: int | None = None
    tiebreak_set_numbers: tuple[int, ...] = ()
    point_trace: tuple[ServicePointResult, ...] | None = None
    retirement_game_number: int | None = None
    retirement_intensities: tuple[float, float] | None = None
    retirement_scenario_ids: tuple[str, str] | None = None
    duration_latent: float | None = None
    duration_official: int | None = None
    duration_partial: bool = False
    duration_display_policy_version: str | None = None
    duration_display_candidates: tuple[int, ...] = ()
    rally_winners: tuple[int, int] | None = None
    rally_unforced_errors: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.best_of, bool) or not isinstance(self.best_of, int):
            raise TypeError("best_of must be an integer")
        if self.best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        if self.player_a_id == self.player_b_id:
            raise ValueError("players must be distinct")
        players = {self.player_a_id, self.player_b_id}
        if self.first_server_id not in players:
            raise ValueError("first_server_id must match one of the players")
        if self.winner_id is not None and self.winner_id not in players:
            raise ValueError("winner_id must match one of the players")
        if self.retired_player_id is not None and self.retired_player_id not in {
            self.player_a_id,
            self.player_b_id,
        }:
            raise ValueError("retired_player_id must match one of the players")
        if self.walkover and self.started:
            raise ValueError("walkover matches must not start")
        if self.walkover and (
            self.completed
            or self.retired_player_id is not None
            or self.sets
            or self.total_games
            or self.break_events
        ):
            raise ValueError("walkover paths must have no played tennis state")
        if self.walkover and self.winner_id is not None:
            raise ValueError("walkover paths must not declare a winner")
        if self.completed and (not self.started or self.retired_player_id is not None):
            raise ValueError("completed paths must be started and cannot contain a retirement")
        if self.completed and self.winner_id is None:
            raise ValueError("completed paths must declare the match winner")
        if not self.completed and self.started and self.retired_player_id is None:
            raise ValueError("started incomplete paths must declare the retired player")
        if self.retired_player_id is not None:
            advancing = (
                self.player_b_id if self.retired_player_id == self.player_a_id else self.player_a_id
            )
            if self.winner_id is not None and self.winner_id != advancing:
                raise ValueError("retirement winner must be the advancing player")
            if self.retirement_game_number is not None and self.retirement_game_number < 1:
                raise ValueError("retirement game number must be a positive boundary")
            if (
                self.retirement_game_number is not None
                and self.retirement_game_number != self.total_games
            ):
                raise ValueError("retirement boundary must equal the stored completed games")
            if (self.retirement_intensities is None) != (self.retirement_scenario_ids is None):
                raise ValueError("B6 intensity and scenario diagnostics must be paired")
        elif self.retirement_game_number is not None:
            raise ValueError("only retirement paths may store a retirement game number")
        if self.retirement_intensities is not None and (
            len(self.retirement_intensities) != 2
            or any(not isfinite(item) or item < 0 for item in self.retirement_intensities)
        ):
            raise ValueError("retirement intensities must be two finite nonnegative values")
        if self.retirement_scenario_ids is not None and (
            len(self.retirement_scenario_ids) != 2
            or any(not item.strip() for item in self.retirement_scenario_ids)
        ):
            raise ValueError("retirement scenario IDs must identify both players")
        duration_attached = self.duration_latent is not None
        if not duration_attached:
            if (
                self.duration_official is not None
                or self.duration_partial
                or self.duration_display_policy_version is not None
                or self.duration_display_candidates
            ):
                raise ValueError("duration metadata requires a latent duration draw")
        else:
            assert self.duration_latent is not None
            if not isfinite(self.duration_latent) or self.duration_latent < 1.0:
                raise ValueError("duration_latent must be finite and at least one minute")
            if self.walkover or not self.started:
                raise ValueError("unstarted paths cannot contain a duration draw")
            if self.duration_partial != (self.retired_player_id is not None):
                raise ValueError("duration_partial must identify exactly retirement paths")
            if (
                self.duration_display_policy_version is None
                or not self.duration_display_policy_version.strip()
            ):
                raise ValueError("duration draws require a display-policy version")
            if not self.duration_display_candidates:
                raise ValueError("duration draws require at least one display candidate")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in self.duration_display_candidates
            ):
                raise ValueError("duration display candidates must be positive whole minutes")
            if len(set(self.duration_display_candidates)) != len(
                self.duration_display_candidates
            ):
                raise ValueError("duration display candidates must be unique")
            if tuple(sorted(self.duration_display_candidates)) != self.duration_display_candidates:
                raise ValueError("duration display candidates must be sorted")
            if self.duration_official is not None:
                if (
                    isinstance(self.duration_official, bool)
                    or not isinstance(self.duration_official, int)
                    or self.duration_official < 1
                ):
                    raise ValueError("duration_official must be a positive whole minute")
                if self.duration_display_candidates != (self.duration_official,):
                    raise ValueError(
                        "a resolved official duration must be the sole display candidate"
                    )
        if self.total_games < 0:
            raise ValueError("total_games must be nonnegative")
        if self.tiebreak_count < 0:
            raise ValueError("tiebreak_count must be nonnegative")
        if self.sets_started is None:
            object.__setattr__(self, "sets_started", len(self.sets))
        if self.sets_started is not None and (
            isinstance(self.sets_started, bool) or not isinstance(self.sets_started, int)
        ):
            raise TypeError("sets_started must be an integer")
        if self.sets_started is None or not len(self.sets) <= self.sets_started <= self.best_of:
            raise ValueError("sets_started must include all completed sets and fit the format")
        completed_tiebreak_sets = tuple(
            result.set_number for result in self.sets if result.tiebreak is not None
        )
        if not self.tiebreak_set_numbers:
            object.__setattr__(self, "tiebreak_set_numbers", completed_tiebreak_sets)
        if len(set(self.tiebreak_set_numbers)) != len(self.tiebreak_set_numbers):
            raise ValueError("tiebreak set numbers must be unique")
        if any(not 1 <= set_no <= self.sets_started for set_no in self.tiebreak_set_numbers):
            raise ValueError("tiebreak set numbers must identify started sets")
        if not set(completed_tiebreak_sets).issubset(self.tiebreak_set_numbers):
            raise ValueError("completed tiebreaks must appear in tiebreak_set_numbers")
        if self.tiebreak_count != len(self.tiebreak_set_numbers):
            raise ValueError("tiebreak_count must match tiebreak_set_numbers")
        if self.completed:
            if self.sets_started != len(self.sets):
                raise ValueError("completed paths cannot contain an incomplete set")
            wins = [0, 0]
            expected_server = 0 if self.first_server_id == self.player_a_id else 1
            for position, result in enumerate(self.sets, start=1):
                if result.set_number != position:
                    raise ValueError("completed set numbers must be consecutive")
                if result.first_server_index != expected_server:
                    raise ValueError("set first servers must preserve service order")
                wins[result.winner_index] += 1
                expected_server = result.next_set_server_index
            sets_to_win = self.best_of // 2 + 1
            winner_index = 0 if self.winner_id == self.player_a_id else 1
            if wins[winner_index] != sets_to_win:
                raise ValueError("winner_id conflicts with the completed set score")
            if wins[1 - winner_index] >= sets_to_win:
                raise ValueError("both players cannot have a match-winning set count")
            if self.total_games != sum(result.total_games for result in self.sets):
                raise ValueError("total_games must equal the completed set scores")
        if self.walkover:
            if self.player_stats or self.point_trace:
                raise ValueError("walkover paths cannot contain statistics or point traces")
            return
        if set(self.player_stats) != players:
            raise ValueError("player_stats must contain exactly both match players")
        left = self.player_stats[self.player_a_id]
        right = self.player_stats[self.player_b_id]
        if left.breaks_achieved != right.breaks_conceded:
            raise ValueError("player A breaks must equal player B breaks conceded")
        if right.breaks_achieved != left.breaks_conceded:
            raise ValueError("player B breaks must equal player A breaks conceded")
        if left.return_games_played != right.service_games_played:
            raise ValueError("player A return games must equal player B service games")
        if right.return_games_played != left.service_games_played:
            raise ValueError("player B return games must equal player A service games")
        if self.total_games != left.games_won + right.games_won:
            raise ValueError("total_games must equal the players' games won")
        if len(self.break_events) != left.breaks_achieved + right.breaks_achieved:
            raise ValueError("break events must match break totals")
        if any({event.server_id, event.receiver_id} != players for event in self.break_events):
            raise ValueError("break events must reference the match players")
        break_locations = tuple(event.match_game_number for event in self.break_events)
        if break_locations != tuple(sorted(break_locations)) or len(set(break_locations)) != len(
            break_locations
        ):
            raise ValueError("break events must have unique chronological locations")
        tiebreak_wins = [0, 0]
        for result in self.sets:
            if result.tiebreak is not None:
                tiebreak_wins[result.winner_index] += 1
        if left.games_won != left.service_games_held + left.breaks_achieved + tiebreak_wins[0]:
            raise ValueError("player A game total conflicts with holds, breaks, and tiebreaks")
        if right.games_won != right.service_games_held + right.breaks_achieved + tiebreak_wins[1]:
            raise ValueError("player B game total conflicts with holds, breaks, and tiebreaks")
        if (
            self.point_trace is not None
            and len(self.point_trace) != left.service_points + right.service_points
        ):
            raise ValueError("point trace length must match service-point totals")

    @property
    def winner_index(self) -> int:
        if self.winner_id is None:
            raise ValueError("winner_id is unavailable for a non-completed or walkover path")
        return 0 if self.winner_id == self.player_a_id else 1

    @property
    def advancing_player_id(self) -> str | None:
        if self.walkover:
            return None
        if self.retired_player_id is not None:
            return (
                self.player_b_id if self.retired_player_id == self.player_a_id else self.player_a_id
            )
        return self.winner_id


@dataclass(frozen=True, slots=True)
class SimulationBatch:
    context: Any
    n_paths: int
    seed_id: str
    paths: tuple[MatchPath, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_paths < 0:
            raise ValueError("n_paths must be nonnegative")
        if len(self.paths) != self.n_paths:
            raise ValueError("path count must equal the number of stored paths")
        if not isinstance(self.seed_id, str):
            raise TypeError("seed_id must be a string")


def _numeric(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _compare(value: float, operator: ComparisonOperator, threshold: float) -> bool:
    if operator is ComparisonOperator.AT_LEAST:
        return value >= threshold
    if operator is ComparisonOperator.MORE_THAN:
        return value > threshold
    if operator is ComparisonOperator.FEWER_THAN:
        return value < threshold
    raise AssertionError("unhandled comparison")


_PROP_SUBJECT_COUNTS = {
    "MATCH_WIN": 1,
    "EXACT_SCORE": 1,
    "STRAIGHT_SETS": 1,
    "PLAYER_WINS_SET": 1,
    "DECIDING_SET": 0,
    "FIRST_SET_WIN": 1,
    "SET_TIEBREAK": 0,
    "ANY_TIEBREAK": 0,
    "TIEBREAK_COUNT": 0,
    "ANY_LOPSIDED_SET": 0,
    "SET_SCORE": 0,
    "FIRST_SET_GAMES": 0,
    "TOTAL_GAMES": 0,
    "PLAYER_GAMES": 1,
    "GAME_HANDICAP": 1,
    "BREAK_COUNT": 1,
    "TOTAL_BREAKS": 0,
    "BOTH_BREAK": 0,
    "FIRST_BREAK_TIMING": 0,
    "PLAYER_ACES": 1,
    "TOTAL_ACES": 0,
    "ACE_COMPARE": 2,
    "PLAYER_DF": 1,
    "TOTAL_DF": 0,
    "DF_COMPARE": 2,
    "FIRST_SERVE_WIN_PCT": 1,
    "WINNERS": 1,
    "WINNER_COMPARE": 2,
    "UNFORCED_ERRORS": 1,
    "TOTAL_UNFORCED_ERRORS": 0,
    "UE_COMPARE": 2,
    "DURATION_MIN": 0,
}
_THRESHOLD_PROP_KINDS = {
    "TIEBREAK_COUNT",
    "FIRST_SET_GAMES",
    "TOTAL_GAMES",
    "PLAYER_GAMES",
    "BREAK_COUNT",
    "TOTAL_BREAKS",
    "FIRST_BREAK_TIMING",
    "PLAYER_ACES",
    "TOTAL_ACES",
    "PLAYER_DF",
    "TOTAL_DF",
    "FIRST_SERVE_WIN_PCT",
    "WINNERS",
    "UNFORCED_ERRORS",
    "TOTAL_UNFORCED_ERRORS",
    "DURATION_MIN",
}


@dataclass(frozen=True, slots=True)
class PropSpec:
    kind: str
    subject_ids: tuple[str, ...] = ()
    operator: ComparisonOperator | None = None
    threshold: float | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    settlement_policy_version: str = CANONICAL_SETTLEMENT_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a nonempty string")
        if self.kind not in _PROP_SUBJECT_COUNTS:
            raise ValueError(f"unsupported frozen-v1.0 prop kind: {self.kind!r}")
        if not isinstance(self.subject_ids, tuple):
            raise TypeError("subject_ids must be a tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.subject_ids):
            raise ValueError("subject_ids must contain nonempty strings")
        if isinstance(self.operator, str):
            try:
                object.__setattr__(self, "operator", ComparisonOperator(self.operator))
            except ValueError as exc:
                raise ValueError(f"unsupported comparison operator: {self.operator!r}") from exc
        elif self.operator is not None and not isinstance(self.operator, ComparisonOperator):
            raise TypeError("operator must be a ComparisonOperator or None")
        if self.threshold is not None:
            _numeric(self.threshold, field="threshold")
        if not isinstance(self.scope, dict):
            raise TypeError("scope must be a dictionary")
        object.__setattr__(self, "scope", dict(self.scope))
        if not isinstance(self.original_text, str):
            raise TypeError("original_text must be a string")
        if not self.settlement_policy_version.strip():
            raise ValueError("settlement_policy_version must not be empty")
        expected_subjects = _PROP_SUBJECT_COUNTS[self.kind]
        if len(self.subject_ids) != expected_subjects:
            raise ValueError(f"{self.kind} requires exactly {expected_subjects} subject IDs")
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("comparison prop subjects must be distinct")
        if self.kind in _THRESHOLD_PROP_KINDS:
            if self.operator is None or self.threshold is None:
                raise ValueError(f"{self.kind} requires an operator and threshold")
        elif self.kind == "GAME_HANDICAP":
            if self.operator is not None or self.threshold is None or "h" not in self.scope:
                raise ValueError("GAME_HANDICAP requires h and no comparison operator")
        elif self.operator is not None or self.threshold is not None:
            raise ValueError(f"{self.kind} does not accept an operator or threshold")
        if self.kind in {"SET_TIEBREAK", "SET_SCORE"}:
            set_number = self.scope.get("set_no")
            if isinstance(set_number, bool) or not isinstance(set_number, int):
                raise ValueError(f"{self.kind} requires an integer set_no")
            if set_number < 1 or set_number > 5:
                raise ValueError("set_no must lie between 1 and 5")
        if self.kind == "SET_SCORE" and not {"x", "y"}.issubset(self.scope):
            raise ValueError("SET_SCORE requires x and y")
        if self.kind == "EXACT_SCORE" and not {"a", "b"}.issubset(self.scope):
            raise ValueError("EXACT_SCORE requires a and b")
        if self.kind == "FIRST_BREAK_TIMING":
            mode = self.scope.get("mode")
            if mode not in {"match", "set"}:
                raise ValueError("FIRST_BREAK_TIMING must declare match or set mode")
            if mode == "set":
                set_number = self.scope.get("set_no")
                if isinstance(set_number, bool) or not isinstance(set_number, int):
                    raise ValueError("set-scoped FIRST_BREAK_TIMING requires integer set_no")


@dataclass(frozen=True, slots=True)
class CorePropEstimate:
    prop: PropSpec | BooleanCompositeSpec
    probability_raw: float
    yes_paths: int
    total_paths: int
    mc_standard_error: float

    def __post_init__(self) -> None:
        _numeric(self.probability_raw, field="probability_raw")
        _numeric(self.mc_standard_error, field="mc_standard_error")
        if isinstance(self.total_paths, bool) or not isinstance(self.total_paths, int):
            raise TypeError("total_paths must be an integer")
        if isinstance(self.yes_paths, bool) or not isinstance(self.yes_paths, int):
            raise TypeError("yes_paths must be an integer")
        if self.total_paths < 0:
            raise ValueError("total_paths must be nonnegative")
        if self.yes_paths < 0 or self.yes_paths > self.total_paths:
            raise ValueError("yes_paths must be in [0, total_paths]")
        if self.total_paths == 0 and (self.probability_raw != 0.0 or self.yes_paths != 0):
            raise ValueError("empty batch must report zero probability")
        if self.total_paths == 0:
            return
        if not 0.0 <= self.probability_raw <= 1.0:
            raise ValueError("probability_raw must lie in [0, 1]")
        if self.mc_standard_error < 0.0:
            raise ValueError("mc_standard_error must be nonnegative")


@dataclass(frozen=True, slots=True)
class PropEstimate:
    prop: PropSpec | BooleanCompositeSpec
    probability_raw: float
    probability_settled: float
    yes_paths: int
    no_paths: int
    void_paths: int
    unresolved_paths: int
    settled_paths: int
    total_paths: int
    mc_standard_error: float
    settlement_policy_version: str = CANONICAL_SETTLEMENT_POLICY_VERSION
    sensitivity_low: float | None = None
    sensitivity_high: float | None = None
    display_policy_version: str | None = None

    def __post_init__(self) -> None:
        _numeric(self.probability_raw, field="probability_raw")
        _numeric(self.probability_settled, field="probability_settled")
        _numeric(self.mc_standard_error, field="mc_standard_error")
        counts = (
            self.yes_paths,
            self.no_paths,
            self.void_paths,
            self.unresolved_paths,
            self.settled_paths,
            self.total_paths,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("path counts must be integers")
        if self.total_paths < 0:
            raise ValueError("total_paths must be nonnegative")
        if self.settled_paths < 0 or self.settled_paths > self.total_paths:
            raise ValueError("settled_paths must be in [0, total_paths]")
        if (
            self.yes_paths < 0
            or self.no_paths < 0
            or self.void_paths < 0
            or self.unresolved_paths < 0
        ):
            raise ValueError("path counts must be nonnegative")
        if (
            self.yes_paths + self.no_paths + self.void_paths + self.unresolved_paths
            != self.total_paths
        ):
            raise ValueError("path count totals must partition the full sample")
        if self.total_paths == 0 and (
            self.probability_raw != 0.0
            or self.yes_paths != 0
            or self.settled_paths != 0
            or self.probability_settled != 0.0
        ):
            raise ValueError(
                "empty batch must report zero settled probability and zero path counts"
            )
        if not 0.0 <= self.probability_raw <= 1.0:
            raise ValueError("probability_raw must lie in [0, 1]")
        if not 0.0 <= self.probability_settled <= 1.0:
            raise ValueError("probability_settled must lie in [0, 1]")
        if self.mc_standard_error < 0.0:
            raise ValueError("mc_standard_error must be nonnegative")
        if (self.sensitivity_low is None) != (self.sensitivity_high is None):
            raise ValueError("probability sensitivities must be supplied as a pair")
        if self.sensitivity_low is not None and self.sensitivity_high is not None:
            _numeric(self.sensitivity_low, field="sensitivity_low")
            _numeric(self.sensitivity_high, field="sensitivity_high")
            if not 0.0 <= self.sensitivity_low <= self.sensitivity_high <= 1.0:
                raise ValueError("probability sensitivity bounds must lie in [0, 1]")
        if self.display_policy_version is not None and not self.display_policy_version.strip():
            raise ValueError("display_policy_version must not be blank")


@dataclass(frozen=True, slots=True)
class BooleanCompositeSpec:
    kind: str
    exprs: tuple[PropSpec | BooleanCompositeSpec, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"AND", "OR"}:
            raise ValueError("compound kind must be AND or OR")
        if not self.exprs:
            raise ValueError("compound props must contain at least one expression")


def AND(*props: PropSpec | BooleanCompositeSpec) -> BooleanCompositeSpec:
    return BooleanCompositeSpec(kind="AND", exprs=tuple(props))


def OR(*props: PropSpec | BooleanCompositeSpec) -> BooleanCompositeSpec:
    return BooleanCompositeSpec(kind="OR", exprs=tuple(props))


def MATCH_WIN(player: str) -> PropSpec:
    return PropSpec(kind="MATCH_WIN", subject_ids=(player,), original_text=f"MATCH_WIN({player})")


def EXACT_SCORE(player: str, a: int, b: int) -> PropSpec:
    return PropSpec(
        kind="EXACT_SCORE",
        subject_ids=(player,),
        scope={"a": int(a), "b": int(b)},
        original_text=f"EXACT_SCORE({player},{a},{b})",
    )


def STRAIGHT_SETS(player: str) -> PropSpec:
    return PropSpec(
        kind="STRAIGHT_SETS", subject_ids=(player,), original_text=f"STRAIGHT_SETS({player})"
    )


def PLAYER_WINS_SET(player: str) -> PropSpec:
    return PropSpec(
        kind="PLAYER_WINS_SET", subject_ids=(player,), original_text=f"PLAYER_WINS_SET({player})"
    )


def DECIDING_SET() -> PropSpec:
    return PropSpec(kind="DECIDING_SET", original_text="DECIDING_SET")


def FIRST_SET_WIN(player: str) -> PropSpec:
    return PropSpec(
        kind="FIRST_SET_WIN", subject_ids=(player,), original_text=f"FIRST_SET_WIN({player})"
    )


def SET_TIEBREAK(set_no: int) -> PropSpec:
    return PropSpec(
        kind="SET_TIEBREAK", scope={"set_no": int(set_no)}, original_text=f"SET_TIEBREAK({set_no})"
    )


def ANY_TIEBREAK() -> PropSpec:
    return PropSpec(kind="ANY_TIEBREAK", original_text="ANY_TIEBREAK")


def TIEBREAK_COUNT(operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="TIEBREAK_COUNT",
        operator=operator,
        threshold=float(k),
        original_text=f"TIEBREAK_COUNT({operator},{k})",
    )


def ANY_LOPSIDED_SET() -> PropSpec:
    return PropSpec(kind="ANY_LOPSIDED_SET", original_text="ANY_LOPSIDED_SET")


def SET_SCORE(set_no: int, x: int, y: int) -> PropSpec:
    return PropSpec(
        kind="SET_SCORE",
        scope={"set_no": int(set_no), "x": int(x), "y": int(y)},
        original_text=f"SET_SCORE({set_no},{x},{y})",
    )


def FIRST_SET_GAMES(operator: ComparisonOperator, k: float) -> PropSpec:
    return PropSpec(
        kind="FIRST_SET_GAMES",
        operator=operator,
        threshold=float(k),
        original_text=f"FIRST_SET_GAMES({operator},{k})",
    )


def TOTAL_GAMES(operator: ComparisonOperator, k: float) -> PropSpec:
    return PropSpec(
        kind="TOTAL_GAMES",
        operator=operator,
        threshold=float(k),
        original_text=f"TOTAL_GAMES({operator},{k})",
    )


def PLAYER_GAMES(player: str, operator: ComparisonOperator, k: float) -> PropSpec:
    return PropSpec(
        kind="PLAYER_GAMES",
        subject_ids=(player,),
        operator=operator,
        threshold=float(k),
        original_text=f"PLAYER_GAMES({player},{operator},{k})",
    )


def GAME_HANDICAP(player: str, h: float) -> PropSpec:
    return PropSpec(
        kind="GAME_HANDICAP",
        subject_ids=(player,),
        threshold=float(h),
        scope={"h": float(h)},
        original_text=f"GAME_HANDICAP({player},{h})",
    )


def BREAK_COUNT(player: str, operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="BREAK_COUNT",
        subject_ids=(player,),
        operator=operator,
        threshold=float(k),
        original_text=f"BREAK_COUNT({player},{operator},{k})",
    )


def TOTAL_BREAKS(operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="TOTAL_BREAKS",
        operator=operator,
        threshold=float(k),
        original_text=f"TOTAL_BREAKS({operator},{k})",
    )


def BOTH_BREAK() -> PropSpec:
    return PropSpec(kind="BOTH_BREAK", original_text="BOTH_BREAK")


def FIRST_BREAK_TIMING(
    scope: dict[str, Any],
    operator: ComparisonOperator,
    k: int,
) -> PropSpec:
    return PropSpec(
        kind="FIRST_BREAK_TIMING",
        operator=operator,
        threshold=float(k),
        scope=dict(scope),
        original_text="FIRST_BREAK_TIMING",
    )


def PLAYER_ACES(player: str, operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="PLAYER_ACES",
        subject_ids=(player,),
        operator=operator,
        threshold=float(k),
        original_text=f"PLAYER_ACES({player},{operator},{k})",
    )


def TOTAL_ACES(operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="TOTAL_ACES",
        operator=operator,
        threshold=float(k),
        original_text=f"TOTAL_ACES({operator},{k})",
    )


def ACE_COMPARE(player_a: str, player_b: str) -> PropSpec:
    return PropSpec(
        kind="ACE_COMPARE",
        subject_ids=(player_a, player_b),
        original_text=f"ACE_COMPARE({player_a},{player_b})",
    )


def PLAYER_DF(player: str, operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="PLAYER_DF",
        subject_ids=(player,),
        operator=operator,
        threshold=float(k),
        original_text=f"PLAYER_DF({player},{operator},{k})",
    )


def TOTAL_DF(operator: ComparisonOperator, k: int) -> PropSpec:
    return PropSpec(
        kind="TOTAL_DF",
        operator=operator,
        threshold=float(k),
        original_text=f"TOTAL_DF({operator},{k})",
    )


def DF_COMPARE(player_a: str, player_b: str) -> PropSpec:
    return PropSpec(
        kind="DF_COMPARE",
        subject_ids=(player_a, player_b),
        original_text=f"DF_COMPARE({player_a},{player_b})",
    )


def FIRST_SERVE_WIN_PCT(
    player: str,
    operator: ComparisonOperator,
    k: float,
) -> PropSpec:
    return PropSpec(
        kind="FIRST_SERVE_WIN_PCT",
        subject_ids=(player,),
        operator=operator,
        threshold=float(k),
        original_text=f"FIRST_SERVE_WIN_PCT({player},{operator},{k})",
    )


def DURATION_MIN(
    operator: ComparisonOperator,
    k: float,
    *,
    display_conversion_version: str = UNRESOLVED_DURATION_DISPLAY_POLICY.policy_version,
) -> PropSpec:
    if not isinstance(display_conversion_version, str) or not display_conversion_version.strip():
        raise ValueError("display_conversion_version must be a nonempty string")
    return PropSpec(
        kind="DURATION_MIN",
        operator=operator,
        threshold=float(k),
        scope={"display_conversion_version": display_conversion_version.strip()},
        original_text=f"DURATION_MIN({operator},{k})",
    )


def _player_index(path: MatchPath, player_id: str) -> int:
    if player_id == path.player_a_id:
        return 0
    if player_id == path.player_b_id:
        return 1
    raise KeyError(f"unknown player {player_id!r}")


def _player_stats(path: MatchPath, player_id: str) -> PlayerMatchStats:
    stats = path.player_stats.get(player_id)
    if stats is None:
        raise KeyError(f"missing stats for {player_id!r}")
    return stats


def _set_score_for_player(path: MatchPath, player_id: str) -> tuple[int, int]:
    player_index = _player_index(path, player_id)
    wins = sum(result.winner_index == player_index for result in path.sets)
    return wins, len(path.sets) - wins


def _comparison_parts(prop: PropSpec) -> tuple[ComparisonOperator, float]:
    if not isinstance(prop.operator, ComparisonOperator) or prop.threshold is None:
        raise ValueError(f"{prop.kind} requires a typed operator and threshold")
    return prop.operator, float(prop.threshold)


def _completed_set(path: MatchPath, set_number: int) -> SetResult | None:
    return next((result for result in path.sets if result.set_number == set_number), None)


def _numeric_measure(path: MatchPath, prop: PropSpec) -> float | None:
    if prop.kind == "TIEBREAK_COUNT":
        return float(path.tiebreak_count)
    if prop.kind == "FIRST_SET_GAMES":
        first_set = _completed_set(path, 1)
        if first_set is not None:
            return float(first_set.total_games)
        return float(path.total_games) if path.sets_started == 1 else None
    if prop.kind == "TOTAL_GAMES":
        return float(path.total_games)
    if prop.kind == "PLAYER_GAMES":
        return float(_player_stats(path, prop.subject_ids[0]).games_won)
    if prop.kind == "BREAK_COUNT":
        return float(_player_stats(path, prop.subject_ids[0]).breaks_achieved)
    if prop.kind == "TOTAL_BREAKS":
        return float(sum(stats.breaks_achieved for stats in path.player_stats.values()))
    if prop.kind == "FIRST_BREAK_TIMING":
        mode = prop.scope.get("mode")
        if mode == "match":
            if not path.break_events:
                return None
            return float(min(event.match_game_number for event in path.break_events))
        if mode == "set":
            set_number = int(prop.scope["set_no"])
            events = [event for event in path.break_events if event.set_number == set_number]
            if not events:
                return None
            return float(min(event.game_number for event in events))
        raise ValueError("FIRST_BREAK_TIMING scope must use mode='match' or mode='set'")
    if prop.kind == "PLAYER_ACES":
        return float(_player_stats(path, prop.subject_ids[0]).aces)
    if prop.kind == "TOTAL_ACES":
        return float(sum(stats.aces for stats in path.player_stats.values()))
    if prop.kind == "PLAYER_DF":
        return float(_player_stats(path, prop.subject_ids[0]).double_faults)
    if prop.kind == "TOTAL_DF":
        return float(sum(stats.double_faults for stats in path.player_stats.values()))
    if prop.kind == "FIRST_SERVE_WIN_PCT":
        stats = _player_stats(path, prop.subject_ids[0])
        if stats.first_serves_in == 0:
            return None
        return 100.0 * stats.first_serve_points_won / stats.first_serves_in
    if prop.kind == "DURATION_MIN":
        return None if path.duration_official is None else float(path.duration_official)
    raise ValueError(f"{prop.kind} is not a numeric prop")


def _published_percentage_truth(path: MatchPath, prop: PropSpec) -> EventTruth:
    value = _numeric_measure(path, prop)
    if value is None:
        return EventTruth.UNRESOLVED
    operator, threshold = _comparison_parts(prop)
    truncated = float(floor(value))
    rounded_half_up = float(floor(value + 0.5))
    outcomes = {
        _compare(truncated, operator, threshold),
        _compare(rounded_half_up, operator, threshold),
    }
    if len(outcomes) != 1:
        return EventTruth.UNRESOLVED
    return EventTruth.from_bool(outcomes.pop())


def _duration_truth(path: MatchPath, prop: PropSpec) -> EventTruth:
    version = prop.scope.get("display_conversion_version")
    if (
        not isinstance(version, str)
        or path.duration_latent is None
        or path.duration_display_policy_version != version
        or not path.duration_display_candidates
    ):
        return EventTruth.UNRESOLVED
    operator, threshold = _comparison_parts(prop)
    outcomes = {
        _compare(float(candidate), operator, threshold)
        for candidate in path.duration_display_candidates
    }
    if len(outcomes) != 1:
        return EventTruth.UNRESOLVED
    return EventTruth.from_bool(outcomes.pop())


def _atomic_truth(path: MatchPath, prop: PropSpec) -> EventTruth:
    if prop.kind == "MATCH_WIN":
        return EventTruth.from_bool(path.advancing_player_id == prop.subject_ids[0])
    if prop.kind == "EXACT_SCORE":
        player = prop.subject_ids[0]
        wins, losses = _set_score_for_player(path, player)
        value = path.winner_id == player and (wins, losses) == (
            int(prop.scope["a"]),
            int(prop.scope["b"]),
        )
        return EventTruth.from_bool(value)
    if prop.kind == "STRAIGHT_SETS":
        player = prop.subject_ids[0]
        wins, losses = _set_score_for_player(path, player)
        return EventTruth.from_bool(path.winner_id == player and (wins, losses) in {(2, 0), (3, 0)})
    if prop.kind == "PLAYER_WINS_SET":
        player_index = _player_index(path, prop.subject_ids[0])
        return EventTruth.from_bool(
            any(result.winner_index == player_index for result in path.sets)
        )
    if prop.kind == "DECIDING_SET":
        return EventTruth.from_bool(cast(int, path.sets_started) >= path.best_of)
    if prop.kind == "FIRST_SET_WIN":
        first_set = _completed_set(path, 1)
        return EventTruth.from_bool(
            first_set is not None
            and first_set.winner_index == _player_index(path, prop.subject_ids[0])
        )
    if prop.kind == "SET_TIEBREAK":
        return EventTruth.from_bool(int(prop.scope["set_no"]) in path.tiebreak_set_numbers)
    if prop.kind == "ANY_TIEBREAK":
        return EventTruth.from_bool(path.tiebreak_count >= 1)
    if prop.kind == "ANY_LOPSIDED_SET":
        return EventTruth.from_bool(
            any(
                result.tiebreak is None and max(result.games) == 6 and min(result.games) <= 2
                for result in path.sets
            )
        )
    if prop.kind == "SET_SCORE":
        result = _completed_set(path, int(prop.scope["set_no"]))
        return EventTruth.from_bool(
            result is not None and result.games == (int(prop.scope["x"]), int(prop.scope["y"]))
        )
    if prop.kind == "GAME_HANDICAP":
        player = prop.subject_ids[0]
        player_stats = _player_stats(path, player)
        opponent = path.player_b_id if player == path.player_a_id else path.player_a_id
        handicap_games = player_stats.games_won + float(prop.scope["h"])
        return EventTruth.from_bool(handicap_games > _player_stats(path, opponent).games_won)
    if prop.kind == "BOTH_BREAK":
        return EventTruth.from_bool(
            all(stats.breaks_achieved >= 1 for stats in path.player_stats.values())
        )
    if prop.kind == "ACE_COMPARE":
        left = _player_stats(path, prop.subject_ids[0]).aces
        right = _player_stats(path, prop.subject_ids[1]).aces
        return EventTruth.from_bool(left > right)
    if prop.kind == "DF_COMPARE":
        left = _player_stats(path, prop.subject_ids[0]).double_faults
        right = _player_stats(path, prop.subject_ids[1]).double_faults
        return EventTruth.from_bool(left > right)
    if prop.kind == "FIRST_SERVE_WIN_PCT":
        return _published_percentage_truth(path, prop)
    if prop.kind == "DURATION_MIN":
        return _duration_truth(path, prop)
    operator, threshold = _comparison_parts(prop)
    measure = _numeric_measure(path, prop)
    if measure is None:
        return EventTruth.FALSE
    return EventTruth.from_bool(_compare(measure, operator, threshold))


def _expression_truth(
    expression: PropSpec | BooleanCompositeSpec,
    path: MatchPath,
) -> EventTruth:
    if isinstance(expression, PropSpec):
        return _atomic_truth(path, expression)
    values = (_expression_truth(item, path) for item in expression.exprs)
    return truth_and(values) if expression.kind == "AND" else truth_or(values)


def evaluate_prop(
    prop: PropSpec | BooleanCompositeSpec,
    batch: SimulationBatch,
) -> CorePropEstimate:
    """Evaluate event truth on completed paths from one joint simulation batch."""

    if any(not path.completed or path.walkover for path in batch.paths):
        raise ValueError("incomplete and walkover paths require settlement-aware evaluation")
    truths = tuple(_expression_truth(prop, path) for path in batch.paths)
    if EventTruth.UNRESOLVED in truths:
        raise PolicyBlockedError("prop truth depends on an unresolved official convention")
    yes_paths = sum(truth is EventTruth.TRUE for truth in truths)
    total = len(batch.paths)
    probability_raw = yes_paths / total if total else 0.0
    mc_standard_error = sqrt(probability_raw * (1.0 - probability_raw) / total) if total else 0.0
    return CorePropEstimate(
        prop=prop,
        probability_raw=probability_raw,
        yes_paths=yes_paths,
        total_paths=total,
        mc_standard_error=mc_standard_error,
    )


_MONOTONE_BOOLEAN_PROPS = {
    "PLAYER_WINS_SET",
    "DECIDING_SET",
    "SET_TIEBREAK",
    "ANY_TIEBREAK",
    "ANY_LOPSIDED_SET",
    "BOTH_BREAK",
}
_MONOTONE_COUNT_PROPS = {
    "TIEBREAK_COUNT",
    "FIRST_SET_GAMES",
    "TOTAL_GAMES",
    "PLAYER_GAMES",
    "BREAK_COUNT",
    "TOTAL_BREAKS",
    "PLAYER_ACES",
    "TOTAL_ACES",
    "PLAYER_DF",
    "TOTAL_DF",
    "DURATION_MIN",
}


def _settlement_state_for_atomic(path: MatchPath, prop: PropSpec) -> SettlementState:
    if path.walkover or not path.started:
        return SettlementState.VOID
    if prop.kind == "MATCH_WIN":
        return SettlementState.from_truth(_atomic_truth(path, prop))
    if path.completed:
        return SettlementState.from_truth(_atomic_truth(path, prop))

    if prop.kind == "FIRST_SET_WIN" and _completed_set(path, 1) is not None:
        return SettlementState.from_truth(_atomic_truth(path, prop))
    if prop.kind == "FIRST_SET_GAMES" and _completed_set(path, 1) is not None:
        return SettlementState.from_truth(_atomic_truth(path, prop))
    if prop.kind in {"SET_SCORE", "SET_TIEBREAK"}:
        set_number = int(prop.scope["set_no"])
        if _completed_set(path, set_number) is not None:
            return SettlementState.from_truth(_atomic_truth(path, prop))
    if prop.kind == "FIRST_BREAK_TIMING" and _numeric_measure(path, prop) is not None:
        return SettlementState.from_truth(_atomic_truth(path, prop))

    truth = _atomic_truth(path, prop)
    if prop.kind == "DURATION_MIN" and truth is EventTruth.UNRESOLVED:
        return SettlementState.UNRESOLVED
    if prop.kind in _MONOTONE_BOOLEAN_PROPS and truth is EventTruth.TRUE:
        return SettlementState.YES
    if prop.kind in _MONOTONE_COUNT_PROPS:
        operator, _threshold = _comparison_parts(prop)
        if (
            operator in {ComparisonOperator.AT_LEAST, ComparisonOperator.MORE_THAN}
            and truth is EventTruth.TRUE
        ):
            return SettlementState.YES
    return SettlementState.VOID


def _settlement_state_for_path(
    path: MatchPath,
    prop: PropSpec | BooleanCompositeSpec,
) -> SettlementState:
    if isinstance(prop, PropSpec):
        return _settlement_state_for_atomic(path, prop)
    states = tuple(_settlement_state_for_path(path, item) for item in prop.exprs)
    truths = tuple(
        EventTruth.TRUE
        if state is SettlementState.YES
        else EventTruth.FALSE
        if state is SettlementState.NO
        else EventTruth.UNRESOLVED
        for state in states
    )
    truth = truth_and(truths) if prop.kind == "AND" else truth_or(truths)
    if truth is EventTruth.TRUE:
        return SettlementState.YES
    if truth is EventTruth.FALSE:
        return SettlementState.NO
    if SettlementState.UNRESOLVED in states:
        return SettlementState.UNRESOLVED
    return SettlementState.VOID


def _policy_versions(prop: PropSpec | BooleanCompositeSpec) -> set[str]:
    if isinstance(prop, PropSpec):
        return {prop.settlement_policy_version}
    versions: set[str] = set()
    for item in prop.exprs:
        versions.update(_policy_versions(item))
    return versions


def _display_policy_versions(prop: PropSpec | BooleanCompositeSpec) -> set[str]:
    if isinstance(prop, PropSpec):
        version = prop.scope.get("display_conversion_version")
        return {version} if prop.kind == "DURATION_MIN" and isinstance(version, str) else set()
    versions: set[str] = set()
    for item in prop.exprs:
        versions.update(_display_policy_versions(item))
    return versions


def evaluate_settlement(
    prop: PropSpec | BooleanCompositeSpec,
    batch: SimulationBatch,
    policy: SettlementPolicy,
) -> PropEstimate:
    """Apply versioned retirement/void semantics after pure path generation."""

    if not isinstance(policy, SettlementPolicy):
        raise TypeError("policy must be a SettlementPolicy")
    if _policy_versions(prop) != {policy.version}:
        raise PolicyBlockedError("prop and evaluator settlement-policy versions differ")
    if not policy.walkover_voids_all or not policy.comparison_tie_is_no:
        raise PolicyBlockedError("policy contradicts frozen Tennis Model v1.0 semantics")

    states = tuple(_settlement_state_for_path(path, prop) for path in batch.paths)
    yes_paths = states.count(SettlementState.YES)
    no_paths = states.count(SettlementState.NO)
    void_paths = states.count(SettlementState.VOID)
    unresolved_paths = states.count(SettlementState.UNRESOLVED)
    settled_paths = yes_paths + no_paths
    total = len(states)
    probability_raw = yes_paths / settled_paths if settled_paths else 0.0
    probability_settled = settled_paths / total if total else 0.0
    mc_standard_error = (
        sqrt(probability_raw * (1.0 - probability_raw) / settled_paths) if settled_paths else 0.0
    )
    sensitivity_denominator = settled_paths + unresolved_paths
    sensitivity_low = (
        yes_paths / sensitivity_denominator if sensitivity_denominator else None
    )
    sensitivity_high = (
        (yes_paths + unresolved_paths) / sensitivity_denominator
        if sensitivity_denominator
        else None
    )
    display_versions = _display_policy_versions(prop)
    return PropEstimate(
        prop=prop,
        probability_raw=probability_raw,
        probability_settled=probability_settled,
        yes_paths=yes_paths,
        no_paths=no_paths,
        void_paths=void_paths,
        unresolved_paths=unresolved_paths,
        settled_paths=settled_paths,
        total_paths=total,
        mc_standard_error=mc_standard_error,
        settlement_policy_version=policy.version,
        sensitivity_low=sensitivity_low,
        sensitivity_high=sensitivity_high,
        display_policy_version=(
            next(iter(display_versions)) if len(display_versions) == 1 else None
        ),
    )


def _empty_stats() -> dict[str, int]:
    return {
        "service_games_played": 0,
        "return_games_played": 0,
        "service_games_held": 0,
        "breaks_conceded": 0,
        "breaks_achieved": 0,
        "break_point_opportunities": 0,
        "service_points": 0,
        "first_serve_opportunities": 0,
        "first_serves_in": 0,
        "first_serve_points_won": 0,
        "returnable_first_serve_trials": 0,
        "returnable_first_serve_wins": 0,
        "second_serve_opportunities": 0,
        "double_faults": 0,
        "playable_second_serve_trials": 0,
        "playable_second_serve_wins": 0,
        "second_serve_points_won": 0,
        "aces": 0,
    }


def _build_player_stats(
    player_id: str,
    player_index: int,
    sets: tuple[SetResult, ...],
    totals_by_player: dict[str, dict[str, int]],
) -> PlayerMatchStats:
    totals = totals_by_player[player_id]
    tiebreak_wins = sum(
        result.tiebreak is not None and result.winner_index == player_index for result in sets
    )
    games_won = totals["service_games_held"] + totals["breaks_achieved"] + tiebreak_wins
    return PlayerMatchStats(
        player_id=player_id,
        games_won=games_won,
        service_games_played=totals["service_games_played"],
        return_games_played=totals["return_games_played"],
        service_games_held=totals["service_games_held"],
        breaks_conceded=totals["breaks_conceded"],
        breaks_achieved=totals["breaks_achieved"],
        service_points=totals["service_points"],
        first_serve_opportunities=totals["first_serve_opportunities"],
        first_serves_in=totals["first_serves_in"],
        first_serve_points_won=totals["first_serve_points_won"],
        returnable_first_serve_trials=totals["returnable_first_serve_trials"],
        returnable_first_serve_wins=totals["returnable_first_serve_wins"],
        second_serve_opportunities=totals["second_serve_opportunities"],
        double_faults=totals["double_faults"],
        playable_second_serve_trials=totals["playable_second_serve_trials"],
        playable_second_serve_wins=totals["playable_second_serve_wins"],
        second_serve_points_won=totals["second_serve_points_won"],
        aces=totals["aces"],
        break_point_opportunities=totals["break_point_opportunities"],
    )


_TIEBREAK_CYCLE_ACCELERATION_THRESHOLD = 1.0e-4
_BINOMIAL_CHUNK_SIZE = 1 << 62


def _service_point_win_probability(performance: ServePerformanceDraw) -> float:
    first = performance.first_serve_in
    ace = performance.ace_given_first_in
    returnable_win = performance.returnable_first_win
    double_fault = performance.double_fault_given_second_opp
    second_win = performance.playable_second_win
    return (
        first * (ace + (1.0 - ace) * returnable_win)
        + (1.0 - first) * (1.0 - double_fault) * second_win
    )


def _draw_large_binomial(
    rng: np.random.Generator,
    trials: int,
    probability: float,
) -> int:
    remaining = trials
    successes = 0
    while remaining:
        chunk = min(remaining, _BINOMIAL_CHUNK_SIZE)
        successes += int(rng.binomial(chunk, probability))
        remaining -= chunk
    return successes


def _draw_three_category_counts(
    rng: np.random.Generator,
    count: int,
    weights: tuple[float, float, float],
) -> tuple[int, int, int]:
    if count == 0:
        return 0, 0, 0
    total = sum(weights)
    if total <= 0.0:
        raise RuntimeError("positive conditional point count has zero probability")
    first_probability = min(1.0, max(0.0, weights[0] / total))
    first = _draw_large_binomial(rng, count, first_probability)
    remaining = count - first
    residual_weight = weights[1] + weights[2]
    if remaining == 0 or residual_weight <= 0.0:
        return first, remaining, 0
    second_probability = min(1.0, max(0.0, weights[1] / residual_weight))
    second = _draw_large_binomial(rng, remaining, second_probability)
    return first, second, remaining - second


def _add_aggregate_service_points(
    totals: dict[str, int],
    performance: ServePerformanceDraw,
    rng: np.random.Generator,
    *,
    server_wins: int,
    server_losses: int,
) -> None:
    first = performance.first_serve_in
    ace = performance.ace_given_first_in
    returnable_win = performance.returnable_first_win
    double_fault = performance.double_fault_given_second_opp
    second_win = performance.playable_second_win

    ace_count, returnable_first_wins, playable_second_wins = (
        _draw_three_category_counts(
            rng,
            server_wins,
            (
                first * ace,
                first * (1.0 - ace) * returnable_win,
                (1.0 - first) * (1.0 - double_fault) * second_win,
            ),
        )
    )
    returnable_first_losses, double_faults, playable_second_losses = (
        _draw_three_category_counts(
            rng,
            server_losses,
            (
                first * (1.0 - ace) * (1.0 - returnable_win),
                (1.0 - first) * double_fault,
                (1.0 - first) * (1.0 - double_fault) * (1.0 - second_win),
            ),
        )
    )

    service_points = server_wins + server_losses
    first_serves_in = ace_count + returnable_first_wins + returnable_first_losses
    second_serve_opportunities = service_points - first_serves_in
    totals["service_points"] += service_points
    totals["first_serve_opportunities"] += service_points
    totals["first_serves_in"] += first_serves_in
    totals["first_serve_points_won"] += ace_count + returnable_first_wins
    totals["returnable_first_serve_trials"] += (
        returnable_first_wins + returnable_first_losses
    )
    totals["returnable_first_serve_wins"] += returnable_first_wins
    totals["second_serve_opportunities"] += second_serve_opportunities
    totals["double_faults"] += double_faults
    totals["playable_second_serve_trials"] += (
        playable_second_wins + playable_second_losses
    )
    totals["playable_second_serve_wins"] += playable_second_wins
    totals["second_serve_points_won"] += playable_second_wins
    totals["aces"] += ace_count


def _accelerate_extended_tiebreak(
    state: MatchState,
    players: tuple[str, str],
    performances: tuple[ServePerformanceDraw, ServePerformanceDraw],
    totals_by_player: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> tuple[MatchState, PointTransition] | None:
    active = state.active_set
    if active is None or active.tiebreak is None:
        return None
    tiebreak = active.tiebreak
    if (
        tiebreak.points[0] != tiebreak.points[1]
        or tiebreak.points[0] < tiebreak.target_points - 1
    ):
        return None

    hold_a = _service_point_win_probability(performances[0])
    hold_b = _service_point_win_probability(performances[1])
    player_a_sweep = hold_a * (1.0 - hold_b)
    player_b_sweep = (1.0 - hold_a) * hold_b
    absorption_probability = player_a_sweep + player_b_sweep
    if absorption_probability >= _TIEBREAK_CYCLE_ACCELERATION_THRESHOLD:
        return None
    if absorption_probability <= 0.0:
        raise RuntimeError("tiebreak cannot terminate under the sampled serve performances")

    uniform = float(rng.random())
    tied_cycles = int(floor(log1p(-uniform) / log1p(-absorption_probability)))
    winner_index = (
        0 if float(rng.random()) < player_a_sweep / absorption_probability else 1
    )
    tied_cycle_probability = 1.0 - absorption_probability
    both_servers_win_probability = hold_a * hold_b / tied_cycle_probability
    both_servers_win = _draw_large_binomial(
        rng,
        tied_cycles,
        min(1.0, max(0.0, both_servers_win_probability)),
    )
    both_servers_lose = tied_cycles - both_servers_win

    _add_aggregate_service_points(
        totals_by_player[players[0]],
        performances[0],
        rng,
        server_wins=both_servers_win + int(winner_index == 0),
        server_losses=both_servers_lose + int(winner_index == 1),
    )
    _add_aggregate_service_points(
        totals_by_player[players[1]],
        performances[1],
        rng,
        server_wins=both_servers_win + int(winner_index == 1),
        server_losses=both_servers_lose + int(winner_index == 0),
    )

    skipped_points = (
        tiebreak.points[0] + tied_cycles,
        tiebreak.points[1] + tied_cycles,
    )
    skipped_state = replace(
        state,
        active_set=replace(active, tiebreak=replace(tiebreak, points=skipped_points)),
        total_points_played=state.total_points_played + 2 * tied_cycles,
    )
    penultimate = award_point(skipped_state, winner_index)
    final = award_point(penultimate.after, winner_index)
    if not final.tiebreak_completed:
        raise AssertionError("accelerated tiebreak did not terminate")
    return final.after, final


def _simulate_one_path(
    player_a_id: str,
    player_b_id: str,
    *,
    best_of: Literal[3, 5],
    first_server_id: str | None,
    player_a_performance: ServePerformanceDraw,
    player_b_performance: ServePerformanceDraw,
    rng: np.random.Generator,
    retirement_intensities: tuple[float, float] = (0.0, 0.0),
    retirement_rng: np.random.Generator | None = None,
    retirement_scenario_ids: tuple[str, str] | None = None,
    trace_points: bool = False,
) -> MatchPath:
    players = (player_a_id, player_b_id)
    if first_server_id is not None and first_server_id not in players:
        raise ValueError("first_server_id must match one of the players")
    initial_server = (
        first_server_id if first_server_id is not None else players[0 if rng.random() < 0.5 else 1]
    )
    state = new_match(
        player_a_id,
        player_b_id,
        best_of=best_of,
        first_server_index=0 if initial_server == player_a_id else 1,
    )
    totals_by_player = {player_a_id: _empty_stats(), player_b_id: _empty_stats()}
    break_events: list[BreakEvent] = []
    point_trace: list[ServicePointResult] | None = [] if trace_points else None
    if any(not isfinite(item) or item < 0 for item in retirement_intensities):
        raise ValueError("retirement intensities must be finite and nonnegative")
    if any(retirement_intensities) and retirement_rng is None:
        raise ValueError("positive retirement intensity requires an explicit retirement RNG")
    retired_player_id: str | None = None
    retirement_game_number: int | None = None

    while not state.is_complete and retired_player_id is None:
        active = state.active_set
        if active is None:
            raise RuntimeError("incomplete match without an active set")
        accelerated = None
        if point_trace is None:
            accelerated = _accelerate_extended_tiebreak(
                state,
                players,
                (player_a_performance, player_b_performance),
                totals_by_player,
                rng,
            )
        if accelerated is not None:
            state, transition = accelerated
        else:
            server_index = active.server_index
            server_id = players[server_index]
            receiver_id = players[1 - server_index]
            performance = (
                player_a_performance if server_id == player_a_id else player_b_performance
            )
            point = generate_service_point(
                performance,
                rng,
                server_id=server_id,
                receiver_id=receiver_id,
            )
            if point_trace is not None:
                point_trace.append(point)
            transition = award_point(state, 0 if point.winner_id == player_a_id else 1)
            state = transition.after

            totals = totals_by_player[server_id]
            totals["service_points"] += 1
            totals["first_serve_opportunities"] += 1
            if point.first_serve_in:
                totals["first_serves_in"] += 1
                totals["first_serve_points_won"] += int(point.server_won)
                if point.q1_used:
                    totals["returnable_first_serve_trials"] += 1
                    totals["returnable_first_serve_wins"] += int(point.server_won)
            else:
                totals["second_serve_opportunities"] += 1
                totals["second_serve_points_won"] += int(point.server_won)
                if point.q2_used:
                    totals["playable_second_serve_trials"] += 1
                    totals["playable_second_serve_wins"] += int(point.server_won)
            if point.ace:
                totals["aces"] += 1
            if point.double_fault:
                totals["double_faults"] += 1
            if transition.break_point_opportunity:
                totals_by_player[receiver_id]["break_point_opportunities"] += 1
            if transition.regular_game_completed:
                server_totals = totals_by_player[server_id]
                receiver_totals = totals_by_player[receiver_id]
                server_totals["service_games_played"] += 1
                receiver_totals["return_games_played"] += 1
                if transition.break_of_serve:
                    server_totals["breaks_conceded"] += 1
                    receiver_totals["breaks_achieved"] += 1
                    game_number = sum(active.games) + 1
                    match_game_number = (
                        sum(result.total_games for result in transition.before.completed_sets)
                        + game_number
                    )
                    break_events.append(
                        BreakEvent(
                            set_number=active.set_number,
                            game_number=game_number,
                            match_game_number=match_game_number,
                            server_id=server_id,
                            receiver_id=receiver_id,
                            break_player_id=receiver_id,
                        )
                    )
                else:
                    server_totals["service_games_held"] += 1

        completed_boundary = transition.regular_game_completed or transition.tiebreak_completed
        if completed_boundary and not state.is_complete:
            assert retirement_rng is not None or not any(retirement_intensities)
            boundary = draw_competing_retirement(
                retirement_intensities[0],
                retirement_intensities[1],
                retirement_rng if retirement_rng is not None else rng,
            )
            if boundary.outcome is not CompetingRetirementOutcome.NO_RETIREMENT:
                retired_player_id = (
                    player_a_id
                    if boundary.outcome is CompetingRetirementOutcome.PLAYER_A_RETIRES
                    else player_b_id
                )
                completed_games = sum(item.total_games for item in state.completed_sets)
                if state.active_set is not None:
                    completed_games += sum(state.active_set.games)
                retirement_game_number = completed_games

    path_sets = tuple(state.completed_sets)
    player_stats = {
        player_a_id: _build_player_stats(player_a_id, 0, path_sets, totals_by_player),
        player_b_id: _build_player_stats(player_b_id, 1, path_sets, totals_by_player),
    }
    if retired_player_id is None and state.winner_id is None:
        raise RuntimeError("completed scoring state did not identify a winner")
    winner_id = (
        state.winner_id
        if retired_player_id is None
        else player_b_id
        if retired_player_id == player_a_id
        else player_a_id
    )
    tiebreak_sets = tuple(result.set_number for result in path_sets if result.tiebreak is not None)
    total_games = sum(item.games_won for item in player_stats.values())
    active_set_started = bool(
        state.active_set is not None
        and (
            sum(state.active_set.games) > 0
            or sum(state.active_set.game_points) > 0
            or state.active_set.tiebreak is not None
        )
    )
    return MatchPath(
        winner_id=winner_id,
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        best_of=best_of,
        first_server_id=initial_server,
        sets=path_sets,
        player_stats=player_stats,
        break_events=tuple(break_events),
        tiebreak_count=len(tiebreak_sets),
        total_games=total_games,
        completed=retired_player_id is None,
        retired_player_id=retired_player_id,
        sets_started=len(path_sets) + int(active_set_started),
        tiebreak_set_numbers=tiebreak_sets,
        point_trace=tuple(point_trace) if point_trace is not None else None,
        retirement_game_number=retirement_game_number,
        retirement_intensities=(
            retirement_intensities if retirement_scenario_ids is not None else None
        ),
        retirement_scenario_ids=retirement_scenario_ids,
    )


def simulate_matches(
    distribution: MatchParameterDistribution,
    *,
    n_paths: int,
    seed: int | np.random.SeedSequence,
    trace_level: Literal["summary", "points"] = "summary",
    first_server_id: str | None = None,
    duration_display_policy: DurationDisplayPolicy = UNRESOLVED_DURATION_DISPLAY_POLICY,
    path_start: int = 0,
) -> SimulationBatch:
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if isinstance(path_start, bool) or not isinstance(path_start, int) or path_start < 0:
        raise ValueError("path_start must be a nonnegative integer")
    if not isinstance(distribution, MatchParameterDistribution):
        raise TypeError("distribution must be a MatchParameterDistribution")
    if trace_level not in {"summary", "points"}:
        raise ValueError("trace_level must be 'summary' or 'points'")
    context = distribution.context
    players = (context.player_a_id, context.player_b_id)
    if first_server_id is not None and first_server_id not in players:
        raise ValueError("first_server_id must match one of the players")
    if isinstance(seed, np.random.SeedSequence):
        root_reference = SeedReference.from_seed_sequence(seed)
    elif isinstance(seed, Integral) and not isinstance(seed, bool):
        root_reference = SeedReference.from_seed_sequence(np.random.SeedSequence(int(seed)))
    else:
        raise TypeError("seed must be an int or numpy.random.SeedSequence")

    root_seed = root_reference.to_seed_sequence()
    paths: list[MatchPath] = []
    duration_parameter_sampler = (
        None
        if distribution.duration is None
        else prepare_duration_parameter_sampler(
            distribution.duration,
            (context.player_a_id, context.player_b_id),
        )
    )
    first_child_index = root_seed.n_children_spawned + path_start
    children = (
        np.random.SeedSequence(
            root_seed.entropy,
            spawn_key=(*root_seed.spawn_key, child_index),
            pool_size=root_seed.pool_size,
        )
        for child_index in range(first_child_index, first_child_index + n_paths)
    )
    for child in children:
        performance = sample_match_performance(distribution, child)
        point_rng = generator_from_seed_reference(performance.seed_plan.point_path)
        retirement_intensities = (
            (0.0, 0.0)
            if not performance.retirement_draws
            else (
                performance.retirement_draws[0].adjusted_intensity_lambda,
                performance.retirement_draws[1].adjusted_intensity_lambda,
            )
        )
        retirement_scenario_ids = (
            None
            if not performance.retirement_draws
            else (
                performance.retirement_draws[0].scenario_id,
                performance.retirement_draws[1].scenario_id,
            )
        )
        retirement_rng = (
            None
            if not performance.retirement_draws
            else generator_from_seed_reference(performance.seed_plan.retirement_boundaries)
        )
        path = _simulate_one_path(
            context.player_a_id,
            context.player_b_id,
            best_of=context.best_of,
            first_server_id=first_server_id,
            player_a_performance=performance.player_a_serving,
            player_b_performance=performance.player_b_serving,
            rng=point_rng,
            retirement_intensities=retirement_intensities,
            retirement_rng=retirement_rng,
            retirement_scenario_ids=retirement_scenario_ids,
            trace_points=trace_level == "points",
        )
        if distribution.duration is not None:
            if path.sets_started is None or path.sets_started < 1:
                raise RuntimeError("a started tennis path must expose at least one started set")
            condition_values = {item.name: item.value for item in context.conditions}
            temperature_value = condition_values.get("temperature_c")
            temperature_c = (
                float(temperature_value)
                if isinstance(temperature_value, (int, float))
                and not isinstance(temperature_value, bool)
                else None
            )
            night_value = condition_values.get("night_session")
            night_session = night_value if isinstance(night_value, bool) else None
            if night_session is None:
                session_value = condition_values.get("session")
                if isinstance(session_value, str):
                    normalized_session = session_value.strip().casefold()
                    if normalized_session in {"day", "night"}:
                        night_session = normalized_session == "night"
            exposure = DurationPathExposure(
                tour=context.tour,
                player_a_id=context.player_a_id,
                player_b_id=context.player_b_id,
                total_points=sum(item.service_points for item in path.player_stats.values()),
                official_games=path.total_games,
                sets=path.sets_started,
                tiebreaks=path.tiebreak_count,
                conditions=DurationConditions(
                    indoor=context.indoor,
                    temperature_c=temperature_c,
                    night_session=night_session,
                    event=context.event,
                    event_year=context.scheduled_start_utc.year,
                ),
            )
            assert duration_parameter_sampler is not None
            duration_parameters = sample_prepared_duration_parameters(
                duration_parameter_sampler,
                generator_from_seed_reference(performance.seed_plan.duration_parameters),
            )
            duration_draw = draw_duration(
                duration_parameters,
                exposure,
                generator_from_seed_reference(performance.seed_plan.duration_residual),
                display_policy=duration_display_policy,
                partial=path.retired_player_id is not None,
            )
            path = replace(
                path,
                duration_latent=duration_draw.latent_minutes,
                duration_official=duration_draw.official_minutes,
                duration_partial=duration_draw.partial,
                duration_display_policy_version=duration_draw.display_policy.policy_version,
                duration_display_candidates=duration_draw.candidate_official_minutes,
            )
        paths.append(path)
    return SimulationBatch(
        context=context,
        n_paths=n_paths,
        seed_id=root_reference.model_dump_json(),
        paths=tuple(paths),
        provenance={
            "bit_generator": "PCG64",
            "root_seed": root_reference.model_dump(mode="json"),
            "path_start": path_start,
            "trace_level": trace_level,
            "snapshot_id": distribution.snapshot.snapshot_id,
            "match_parameter_provenance": distribution.provenance.model_dump(mode="json"),
            "retirement_enabled": distribution.retirement is not None,
            "retirement_artifact_id": (
                None if distribution.retirement is None else distribution.retirement.artifact_id
            ),
            "duration_enabled": distribution.duration is not None,
            "duration_artifact_id": (
                None if distribution.duration is None else distribution.duration.artifact_id
            ),
            "duration_display_policy_version": (
                None
                if distribution.duration is None
                else duration_display_policy.policy_version
            ),
            "duration_rng_stream_version": (
                None
                if distribution.duration is None
                else "seedsequence-duration-parameters-residual/v1"
            ),
            "strength_enabled": distribution.strength is not None,
            "strength_anchor_artifact_id": (
                None
                if distribution.strength is None
                else distribution.strength.anchor_artifact_id
            ),
            "strength_integration_artifact_id": (
                None
                if distribution.strength is None
                else distribution.strength.integration_artifact_id
            ),
            "strength_rng_stream_version": (
                None
                if distribution.strength is None
                else distribution.strength.rng_stream_version
            ),
            "ordinary_termination_before_retirement_version": (
                "ordinary-terminal-bypass-before-b6/v1"
            ),
        },
    )


__all__ = [
    "ACE_COMPARE",
    "AND",
    "ANY_LOPSIDED_SET",
    "ANY_TIEBREAK",
    "BOTH_BREAK",
    "BREAK_COUNT",
    "DECIDING_SET",
    "DF_COMPARE",
    "DURATION_MIN",
    "EXACT_SCORE",
    "FIRST_BREAK_TIMING",
    "FIRST_SERVE_WIN_PCT",
    "FIRST_SET_GAMES",
    "FIRST_SET_WIN",
    "GAME_HANDICAP",
    "MATCH_WIN",
    "OR",
    "PLAYER_ACES",
    "PLAYER_DF",
    "PLAYER_GAMES",
    "PLAYER_WINS_SET",
    "SET_SCORE",
    "SET_TIEBREAK",
    "STRAIGHT_SETS",
    "TIEBREAK_COUNT",
    "TOTAL_ACES",
    "TOTAL_BREAKS",
    "TOTAL_DF",
    "TOTAL_GAMES",
    "BooleanCompositeSpec",
    "BreakEvent",
    "ComparisonOperator",
    "CorePropEstimate",
    "MatchPath",
    "PlayerMatchStats",
    "PropEstimate",
    "PropSpec",
    "SimulationBatch",
    "evaluate_prop",
    "evaluate_settlement",
    "simulate_matches",
]

# --- v1.2 rally-termination extension ---------------------------------------
# This layer is intentionally appended to the frozen scoring implementation.
# It consumes a separate RNG stream and only annotates already-completed paths.

_RALLY_ACCOUNTING_CONVENTION = (
    "usopen-winners-include-aces-ue-include-double-faults/v1"
)
_RALLY_PROP_KINDS = frozenset(
    {
        "WINNERS",
        "WINNER_COMPARE",
        "UNFORCED_ERRORS",
        "TOTAL_UNFORCED_ERRORS",
        "UE_COMPARE",
    }
)


def _rally_prop(
    kind: str,
    subject_ids: tuple[str, ...],
    *,
    operator: ComparisonOperator | str | None = None,
    threshold: float | None = None,
    original_text: str,
) -> PropSpec:
    normalized_operator = (
        None
        if operator is None
        else (
            operator
            if isinstance(operator, ComparisonOperator)
            else ComparisonOperator(operator)
        )
    )
    return PropSpec(
        kind=kind,
        subject_ids=subject_ids,
        operator=normalized_operator,
        threshold=threshold,
        scope={"accounting_convention": _RALLY_ACCOUNTING_CONVENTION},
        original_text=original_text,
    )


def WINNERS(
    player: str, operator: ComparisonOperator | str, k: int
) -> PropSpec:
    return _rally_prop(
        "WINNERS",
        (player,),
        operator=operator,
        threshold=float(k),
        original_text=f"WINNERS({player},{operator},{k})",
    )


def WINNER_COMPARE(player_a: str, player_b: str) -> PropSpec:
    return _rally_prop(
        "WINNER_COMPARE",
        (player_a, player_b),
        original_text=f"WINNER_COMPARE({player_a},{player_b})",
    )


def UNFORCED_ERRORS(
    player: str, operator: ComparisonOperator | str, k: int
) -> PropSpec:
    return _rally_prop(
        "UNFORCED_ERRORS",
        (player,),
        operator=operator,
        threshold=float(k),
        original_text=f"UNFORCED_ERRORS({player},{operator},{k})",
    )


def TOTAL_UNFORCED_ERRORS(
    operator: ComparisonOperator | str, k: int
) -> PropSpec:
    return _rally_prop(
        "TOTAL_UNFORCED_ERRORS",
        (),
        operator=operator,
        threshold=float(k),
        original_text=f"TOTAL_UNFORCED_ERRORS({operator},{k})",
    )


def UE_COMPARE(player_a: str, player_b: str) -> PropSpec:
    return _rally_prop(
        "UE_COMPARE",
        (player_a, player_b),
        original_text=f"UE_COMPARE({player_a},{player_b})",
    )


_first_serve_win_pct_display_dependent = FIRST_SERVE_WIN_PCT


def FIRST_SERVE_WIN_PCT(
    player: str, operator: ComparisonOperator | str, k: float
) -> PropSpec:
    from dataclasses import replace as _replace

    prop = _first_serve_win_pct_display_dependent(player, operator, k)
    return _replace(
        prop,
        scope={
            **prop.scope,
            "comparison_value_version": "exact-ratio/v1",
            "rounding_invariant": True,
        },
    )


_numeric_measure_without_rally = _numeric_measure


def _rally_path_count(
    path: MatchPath, player_id: str, attribute: str
) -> float | None:
    values = getattr(path, attribute, None)
    if values is None:
        return None
    if player_id == path.player_a_id:
        return float(values[0])
    if player_id == path.player_b_id:
        return float(values[1])
    raise ValueError(f"{player_id!r} is not a participant in this path")


def _numeric_measure(path: MatchPath, prop: PropSpec) -> float | None:
    if prop.kind == "WINNERS":
        return _rally_path_count(path, prop.subject_ids[0], "rally_winners")
    if prop.kind == "UNFORCED_ERRORS":
        return _rally_path_count(path, prop.subject_ids[0], "rally_unforced_errors")
    if prop.kind == "TOTAL_UNFORCED_ERRORS":
        values = path.rally_unforced_errors
        return None if values is None else float(sum(values))
    return _numeric_measure_without_rally(path, prop)


_atomic_truth_without_rally = _atomic_truth


def _atomic_truth(path: MatchPath, prop: PropSpec) -> EventTruth:
    if prop.kind in {"WINNER_COMPARE", "UE_COMPARE"}:
        attribute = (
            "rally_winners"
            if prop.kind == "WINNER_COMPARE"
            else "rally_unforced_errors"
        )
        left = _rally_path_count(path, prop.subject_ids[0], attribute)
        right = _rally_path_count(path, prop.subject_ids[1], attribute)
        if left is None or right is None:
            return EventTruth.UNRESOLVED
        return EventTruth.from_bool(left > right)
    return _atomic_truth_without_rally(path, prop)


def _published_percentage_truth(path: MatchPath, prop: PropSpec) -> EventTruth:
    value = _numeric_measure(path, prop)
    if value is None:
        return EventTruth.UNRESOLVED
    operator, threshold = _comparison_parts(prop)
    return EventTruth.from_bool(_compare(value, operator, threshold))


_MONOTONE_COUNT_PROPS = frozenset(
    set(_MONOTONE_COUNT_PROPS)
    | {"WINNERS", "UNFORCED_ERRORS", "TOTAL_UNFORCED_ERRORS"}
)


_simulate_matches_without_rally = simulate_matches


def simulate_matches(distribution: MatchParameterDistribution, **kwargs: Any) -> SimulationBatch:
    from dataclasses import replace as _replace

    batch = _simulate_matches_without_rally(distribution, **kwargs)
    parameters = getattr(distribution, "rally_termination", None)
    if parameters is None:
        return batch
    from tennis_model.estimation.rally_termination import annotate_paths

    paths = annotate_paths(
        batch.paths,
        parameters,
        seed_id=batch.seed_id,
        path_start=int(kwargs.get("path_start", 0)),
    )
    expected: dict[str, dict[str, float]] = {}
    if paths:
        for player_id, index in (
            (paths[0].player_a_id, 0),
            (paths[0].player_b_id, 1),
        ):
            expected[player_id] = {
                "winners": float(
                    sum(path.rally_winners[index] for path in paths) / len(paths)
                ),
                "unforced_errors": float(
                    sum(path.rally_unforced_errors[index] for path in paths) / len(paths)
                ),
            }
    provenance = {
        **batch.provenance,
        "rally_termination": {
            "artifact_id": parameters.artifact_id,
            "schema_version": parameters.schema_version,
            "accounting_convention": parameters.accounting_convention,
            "data_cutoff_utc": parameters.data_cutoff_utc.isoformat(),
            "rng_stream_version": "sha256-seed-id-path-start-pcg64/v1",
            "expected_by_player": expected,
        },
    }
    return _replace(batch, paths=paths, provenance=provenance)
