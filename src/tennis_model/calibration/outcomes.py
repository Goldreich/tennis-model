"""Official historical outcomes and the post-lock revelation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self, cast

from pydantic import Field, field_validator, model_validator

from tennis_model.locking.models import (
    PredictionSnapshot,
    PropSupportStatus,
    SerializedProp,
    deserialize_prop,
)
from tennis_model.props.settlement import SettlementState
from tennis_model.schemas import FrozenModel
from tennis_model.simulation.match import (
    BooleanCompositeSpec,
    MatchPath,
    PropSpec,
    SimulationBatch,
    evaluate_settlement,
)
from tennis_model.simulation.scoring import is_legal_completed_set_score


class HistoricalOutcomeError(ValueError):
    pass


class HistoricalAvailabilityPhase(StrEnum):
    MATCH_EXCLUDED_PRE_REVEAL = "MATCH_EXCLUDED_PRE_REVEAL"
    PROP_AVAILABLE_POST_REVEAL = "PROP_AVAILABLE_POST_REVEAL"
    PROP_UNAVAILABLE_POST_REVEAL = "PROP_UNAVAILABLE_POST_REVEAL"
    TARGET_FAILED_POST_REVEAL = "TARGET_FAILED_POST_REVEAL"


class HistoricalSetResult(FrozenModel):
    set_number: int = Field(ge=1, le=5)
    games: tuple[int, int]
    tiebreak_played: bool = False

    @model_validator(mode="after")
    def score_is_legal(self) -> Self:
        if not is_legal_completed_set_score(self.games, tiebreak_played=self.tiebreak_played):
            raise ValueError("official set score is impossible under frozen US Open rules")
        return self

    @property
    def winner_index(self) -> int:
        return 0 if self.games[0] > self.games[1] else 1

    @property
    def total_games(self) -> int:
        return sum(self.games)

    @property
    def tiebreak(self) -> object | None:
        return _TIEBREAK_MARKER if self.tiebreak_played else None


_TIEBREAK_MARKER = object()


class HistoricalPlayerStats(FrozenModel):
    player_id: str
    games_won: int = Field(ge=0)
    aces: int | None = Field(default=None, ge=0)
    double_faults: int | None = Field(default=None, ge=0)
    breaks_achieved: int | None = Field(default=None, ge=0)
    first_serves_in: int | None = Field(default=None, ge=0)
    first_serve_points_won: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def first_serve_counts_are_coherent(self) -> Self:
        if (self.first_serves_in is None) != (self.first_serve_points_won is None):
            raise ValueError("first-serve denominator and numerator must be jointly available")
        if (
            self.first_serves_in is not None
            and self.first_serve_points_won is not None
            and self.first_serve_points_won > self.first_serves_in
        ):
            raise ValueError("first-serve points won cannot exceed first serves in")
        return self


class HistoricalBreakEvent(FrozenModel):
    set_number: int = Field(ge=1, le=5)
    game_number: int = Field(ge=1)
    match_game_number: int = Field(ge=1)


class OfficialHistoricalOutcome(FrozenModel):
    """Fields revealed only after an immutable forecast lock exists."""

    match_id: str
    player_a_id: str
    player_b_id: str
    best_of: int
    started: bool
    completed: bool
    winner_id: str | None
    retired_player_id: str | None = None
    retirement_completed_games: int | None = Field(default=None, ge=0)
    retirement_timing_missing_reason: str | None = None
    walkover: bool = False
    sets: tuple[HistoricalSetResult, ...] = ()
    sets_started: int = Field(ge=0, le=5)
    player_stats: tuple[HistoricalPlayerStats, HistoricalPlayerStats] | None = None
    break_events: tuple[HistoricalBreakEvent, ...] | None = None
    official_source_id: str
    official_source_sha256: str
    official_source_locator: str
    available_at_utc: datetime
    retrieved_at_utc: datetime

    @field_validator("available_at_utc", "retrieved_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("official_source_sha256")
    @classmethod
    def source_digest_is_valid(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("official_source_sha256 must contain 64 hexadecimal characters")
        return normalized

    @field_validator("retirement_timing_missing_reason")
    @classmethod
    def optional_timing_reason_is_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("retirement_timing_missing_reason must not be empty")
        return normalized

    @model_validator(mode="after")
    def outcome_is_coherent(self) -> Self:
        if self.best_of not in {3, 5}:
            raise ValueError("best_of must be 3 or 5")
        players = {self.player_a_id, self.player_b_id}
        if len(players) != 2:
            raise ValueError("historical players must be distinct")
        if self.winner_id is not None and self.winner_id not in players:
            raise ValueError("winner must be one of the players")
        if self.retired_player_id is not None and self.retired_player_id not in players:
            raise ValueError("retired player must be one of the players")
        if self.walkover:
            if self.started or self.completed or self.sets or self.player_stats is not None:
                raise ValueError("walkover outcome cannot contain played match state")
        elif not self.started:
            raise ValueError("a non-walkover official outcome must have started")
        if self.completed and (self.winner_id is None or self.retired_player_id is not None):
            raise ValueError("completed outcome needs a winner and cannot contain retirement")
        if not self.completed and self.started and self.retired_player_id is None:
            raise ValueError("started incomplete outcome must identify the retired player")
        if self.retired_player_id is not None:
            advancing = (
                self.player_b_id if self.retired_player_id == self.player_a_id else self.player_a_id
            )
            if self.winner_id != advancing:
                raise ValueError("retirement winner must be the advancing player")
            if (
                self.retirement_completed_games is not None
                and self.retirement_timing_missing_reason is not None
            ):
                raise ValueError(
                    "retirement timing and a timing-missing reason are mutually exclusive"
                )
        elif (
            self.retirement_completed_games is not None
            or self.retirement_timing_missing_reason is not None
        ):
            raise ValueError("only a retirement may contain retirement-timing metadata")
        if tuple(item.set_number for item in self.sets) != tuple(range(1, len(self.sets) + 1)):
            raise ValueError("historical completed sets must be consecutive")
        if self.sets_started < len(self.sets):
            raise ValueError("sets_started cannot be smaller than completed sets")
        wins = [0, 0]
        for result in self.sets:
            wins[result.winner_index] += 1
        if self.completed:
            required = self.best_of // 2 + 1
            winner_index = 0 if self.winner_id == self.player_a_id else 1
            if wins[winner_index] != required or wins[1 - winner_index] >= required:
                raise ValueError("winner conflicts with official completed set score")
        if self.player_stats is not None:
            if tuple(item.player_id for item in self.player_stats) != (
                self.player_a_id,
                self.player_b_id,
            ):
                raise ValueError("historical stats must follow player A/player B order")
            expected_games = (
                sum(result.games[0] for result in self.sets),
                sum(result.games[1] for result in self.sets),
            )
            if tuple(item.games_won for item in self.player_stats) != expected_games:
                raise ValueError("historical player game totals conflict with set scores")
        return self


@dataclass(frozen=True, slots=True)
class _HistoricalPath:
    outcome: OfficialHistoricalOutcome

    @property
    def player_a_id(self) -> str:
        return self.outcome.player_a_id

    @property
    def player_b_id(self) -> str:
        return self.outcome.player_b_id

    @property
    def best_of(self) -> int:
        return self.outcome.best_of

    @property
    def started(self) -> bool:
        return self.outcome.started

    @property
    def completed(self) -> bool:
        return self.outcome.completed

    @property
    def winner_id(self) -> str | None:
        return self.outcome.winner_id

    @property
    def retired_player_id(self) -> str | None:
        return self.outcome.retired_player_id

    @property
    def walkover(self) -> bool:
        return self.outcome.walkover

    @property
    def sets(self) -> tuple[HistoricalSetResult, ...]:
        return self.outcome.sets

    @property
    def sets_started(self) -> int:
        return self.outcome.sets_started

    @property
    def tiebreak_set_numbers(self) -> tuple[int, ...]:
        return tuple(item.set_number for item in self.sets if item.tiebreak_played)

    @property
    def tiebreak_count(self) -> int:
        return len(self.tiebreak_set_numbers)

    @property
    def total_games(self) -> int:
        return sum(item.total_games for item in self.sets)

    @property
    def player_stats(self) -> dict[str, HistoricalPlayerStats]:
        return (
            {}
            if self.outcome.player_stats is None
            else {item.player_id: item for item in self.outcome.player_stats}
        )

    @property
    def break_events(self) -> tuple[HistoricalBreakEvent, ...]:
        return self.outcome.break_events or ()

    @property
    def advancing_player_id(self) -> str | None:
        return None if self.walkover else self.winner_id


_SCORE_PROPS = {
    "MATCH_WIN",
    "EXACT_SCORE",
    "STRAIGHT_SETS",
    "PLAYER_WINS_SET",
    "DECIDING_SET",
    "FIRST_SET_WIN",
    "SET_TIEBREAK",
    "ANY_TIEBREAK",
    "TIEBREAK_COUNT",
    "ANY_LOPSIDED_SET",
    "SET_SCORE",
    "FIRST_SET_GAMES",
    "TOTAL_GAMES",
}
_GAME_STAT_PROPS = {"PLAYER_GAMES", "GAME_HANDICAP"}
_BREAK_PROPS = {"BREAK_COUNT", "TOTAL_BREAKS", "BOTH_BREAK"}
_ACE_PROPS = {"PLAYER_ACES", "TOTAL_ACES", "ACE_COMPARE"}
_DF_PROPS = {"PLAYER_DF", "TOTAL_DF", "DF_COMPARE"}


def historical_prop_available(
    prop: PropSpec | BooleanCompositeSpec,
    outcome: OfficialHistoricalOutcome,
) -> bool:
    if isinstance(prop, BooleanCompositeSpec):
        return all(historical_prop_available(child, outcome) for child in prop.exprs)
    if prop.kind in _SCORE_PROPS:
        return True
    if prop.kind in _GAME_STAT_PROPS:
        return outcome.player_stats is not None
    if prop.kind in _BREAK_PROPS:
        return outcome.player_stats is not None and all(
            item.breaks_achieved is not None for item in outcome.player_stats
        )
    if prop.kind == "FIRST_BREAK_TIMING":
        return outcome.break_events is not None
    if prop.kind in _ACE_PROPS:
        return outcome.player_stats is not None and all(
            item.aces is not None for item in outcome.player_stats
        )
    if prop.kind in _DF_PROPS:
        return outcome.player_stats is not None and all(
            item.double_faults is not None for item in outcome.player_stats
        )
    if prop.kind == "FIRST_SERVE_WIN_PCT":
        if outcome.player_stats is None:
            return False
        by_player = {item.player_id: item for item in outcome.player_stats}
        item = by_player[prop.subject_ids[0]]
        return item.first_serves_in is not None and item.first_serve_points_won is not None
    return False


class HistoricalPropResolution(FrozenModel):
    prop_id: str
    prop: SerializedProp
    availability: bool
    state: str
    outcome_binary: int | None
    unavailable_reason: str | None = None
    support_status: PropSupportStatus = PropSupportStatus.SUPPORTED
    availability_phase: HistoricalAvailabilityPhase = (
        HistoricalAvailabilityPhase.PROP_AVAILABLE_POST_REVEAL
    )


class HistoricalLockSettlement(FrozenModel):
    lock_id: str
    outcome_match_id: str
    outcome_available_at_utc: datetime
    resolutions: tuple[HistoricalPropResolution, ...]


def settle_historical_lock(
    lock: PredictionSnapshot,
    outcome: OfficialHistoricalOutcome,
) -> HistoricalLockSettlement:
    """Reveal and settle only after lock creation, using the live settlement evaluator."""

    if outcome.available_at_utc <= lock.context.information_cutoff_utc:
        raise HistoricalOutcomeError("historical outcome was available before the forecast cutoff")
    if (outcome.player_a_id, outcome.player_b_id, outcome.best_of) != (
        lock.context.player_a_id,
        lock.context.player_b_id,
        lock.context.best_of,
    ):
        raise HistoricalOutcomeError("historical outcome does not match the lock")
    from tennis_model.props.settlement import SettlementPolicy

    policy = SettlementPolicy(
        version=lock.settlement_policy.version,
        comparison_tie_is_no=lock.settlement_policy.comparison_tie_is_no,
        walkover_voids_all=lock.settlement_policy.walkover_voids_all,
        allow_policy_blocked=lock.settlement_policy.allow_policy_blocked,
        description=lock.settlement_policy.description,
    )
    path = cast(MatchPath, _HistoricalPath(outcome))
    batch = SimulationBatch(
        context=lock.context, n_paths=1, seed_id="official-outcome", paths=(path,)
    )
    resolutions = []
    for forecast in lock.prop_estimates:
        prop = deserialize_prop(forecast.prop)
        if forecast.support_status is not PropSupportStatus.SUPPORTED:
            resolutions.append(
                HistoricalPropResolution(
                    prop_id=forecast.prop_id,
                    prop=forecast.prop,
                    availability=False,
                    state="unavailable",
                    outcome_binary=None,
                    unavailable_reason=forecast.support_reason_code,
                    support_status=forecast.support_status,
                    availability_phase=HistoricalAvailabilityPhase.PROP_UNAVAILABLE_POST_REVEAL,
                )
            )
            continue
        if not historical_prop_available(prop, outcome):
            resolutions.append(
                HistoricalPropResolution(
                    prop_id=forecast.prop_id,
                    prop=forecast.prop,
                    availability=False,
                    state="unavailable",
                    outcome_binary=None,
                    unavailable_reason="MISSING_SETTLEMENT_FIELDS",
                    support_status=PropSupportStatus.DATA_UNAVAILABLE,
                    availability_phase=(HistoricalAvailabilityPhase.PROP_UNAVAILABLE_POST_REVEAL),
                )
            )
            continue
        estimate = evaluate_settlement(prop, batch, policy)
        if estimate.yes_paths:
            state, binary = SettlementState.YES.value, 1
        elif estimate.no_paths:
            state, binary = SettlementState.NO.value, 0
        elif estimate.void_paths:
            state, binary = SettlementState.VOID.value, None
        else:
            state, binary = SettlementState.UNRESOLVED.value, None
        support_status = (
            PropSupportStatus.SETTLEMENT_BLOCKED
            if state == SettlementState.UNRESOLVED.value
            else PropSupportStatus.SUPPORTED
        )
        resolutions.append(
            HistoricalPropResolution(
                prop_id=forecast.prop_id,
                prop=forecast.prop,
                availability=True,
                state=state,
                outcome_binary=binary,
                unavailable_reason=(
                    "RETIREMENT_SETTLEMENT_EDGE_UNRESOLVED"
                    if support_status is PropSupportStatus.SETTLEMENT_BLOCKED
                    else None
                ),
                support_status=support_status,
            )
        )
    resolutions.extend(
        HistoricalPropResolution(
            prop_id=gate.prop_id,
            prop=gate.prop,
            availability=False,
            state="unavailable",
            outcome_binary=None,
            unavailable_reason=gate.reason_code,
            support_status=gate.support_status,
            availability_phase=HistoricalAvailabilityPhase.PROP_UNAVAILABLE_POST_REVEAL,
        )
        for gate in lock.prop_gates
    )
    return HistoricalLockSettlement(
        lock_id=lock.lock_id,
        outcome_match_id=outcome.match_id,
        outcome_available_at_utc=outcome.available_at_utc,
        resolutions=tuple(resolutions),
    )
