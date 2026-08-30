"""Primitive service-point generation under the frozen v1.0 causal ordering.

The generator consumes one realized match-performance draw containing exactly
the five primitive probabilities ``F``, ``A``, ``Q1``, ``D``, and ``Q2``.  It
never draws a point winner from a derived first-serve, second-serve, or overall
service-point probability and then labels the outcome afterward.  Instead it
draws first-serve status first, then ace or double-fault status on the eligible
branch, and only then a returnable ``Q1`` or playable ``Q2`` outcome.

Ace and double-fault points terminate at this boundary.  The two playable
branches are marked ``rally_eligible`` for a later winner/unforced-error layer;
that later classification is intentionally outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np

from tennis_model.serve import PrimitiveServeMeans


class PointInvariantError(ValueError):
    """A service-point record violates the frozen causal state space."""


class ServeNumber(StrEnum):
    """Serve number on which the point ended or entered play."""

    FIRST = "first"
    SECOND = "second"


class ServicePointBranch(StrEnum):
    """The mutually exclusive terminal branch of the primitive generator."""

    ACE = "ace"
    RETURNABLE_FIRST_SERVE = "returnable_first_serve"
    DOUBLE_FAULT = "double_fault"
    PLAYABLE_SECOND_SERVE = "playable_second_serve"


@dataclass(frozen=True, slots=True)
class ServePerformanceDraw(PrimitiveServeMeans):
    """One realized match-performance vector for ``F/A/Q1/D/Q2``.

    The inherited fields are exactly ``first_serve_in``,
    ``ace_given_first_in``, ``returnable_first_win``,
    ``double_fault_given_second_opp``, and ``playable_second_win``.  Each is
    validated in ``[0, 1]`` by :class:`PrimitiveServeMeans`.  Its inherited
    derived properties exist only for reporting and analytic checks; point
    generation below always traverses the primitive events.
    """


def _uniform(value: float, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite uniform variate")
    if not isfinite(float(value)) or float(value) < 0.0 or float(value) >= 1.0:
        raise ValueError(f"{field} must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class PointUniforms:
    """Five separately supplied uniforms for deterministic coupled generation.

    Field names mirror the primitive event they govern.  All five values are
    supplied up front so tests can reuse the same latent randomness under
    altered probabilities.  The transformation still observes the causal
    branch: for example, ``returnable_first_win`` has no effect after an ace.
    """

    first_serve_in: float
    ace_given_first_in: float
    returnable_first_win: float
    double_fault_given_second_opp: float
    playable_second_win: float

    def __post_init__(self) -> None:
        for field in (
            "first_serve_in",
            "ace_given_first_in",
            "returnable_first_win",
            "double_fault_given_second_opp",
            "playable_second_win",
        ):
            _uniform(getattr(self, field), field=field)


@dataclass(frozen=True, slots=True)
class ServicePointResult:
    """Immutable, invariant-bearing outcome of one primitive service point.

    ``server_id`` and ``receiver_id`` are either both present or both absent.
    ``winner_id`` is derived from those IDs and ``server_won`` so a caller
    cannot construct a point whose winner conflicts with its causal outcome.
    The redundant branch flags are retained deliberately: later scoring and
    official-stat aggregation can consume explicit facts, while validation
    below guarantees that only one of the four legal branch patterns exists.
    """

    server_id: str | None
    receiver_id: str | None
    first_serve_in: bool
    ace: bool
    double_fault: bool
    serve_number: ServeNumber
    returnable_first_serve: bool
    playable_second_serve: bool
    server_won: bool
    rally_eligible: bool
    branch: ServicePointBranch

    def __post_init__(self) -> None:
        if (self.server_id is None) != (self.receiver_id is None):
            raise PointInvariantError("server_id and receiver_id must be supplied together")
        if self.server_id is not None:
            if not isinstance(self.server_id, str) or not self.server_id.strip():
                raise PointInvariantError("server_id must be a nonempty string")
            if not isinstance(self.receiver_id, str) or not self.receiver_id.strip():
                raise PointInvariantError("receiver_id must be a nonempty string")
            if self.server_id == self.receiver_id:
                raise PointInvariantError("server_id and receiver_id must be distinct")

        for field in (
            "first_serve_in",
            "ace",
            "double_fault",
            "returnable_first_serve",
            "playable_second_serve",
            "server_won",
            "rally_eligible",
        ):
            if not isinstance(getattr(self, field), bool):
                raise PointInvariantError(f"{field} must be a boolean")
        if not isinstance(self.serve_number, ServeNumber):
            raise PointInvariantError("serve_number must be a ServeNumber")
        if not isinstance(self.branch, ServicePointBranch):
            raise PointInvariantError("branch must be a ServicePointBranch")

        expected: dict[ServicePointBranch, dict[str, bool | ServeNumber]] = {
            ServicePointBranch.ACE: {
                "first_serve_in": True,
                "ace": True,
                "double_fault": False,
                "serve_number": ServeNumber.FIRST,
                "returnable_first_serve": False,
                "playable_second_serve": False,
                "server_won": True,
                "rally_eligible": False,
            },
            ServicePointBranch.RETURNABLE_FIRST_SERVE: {
                "first_serve_in": True,
                "ace": False,
                "double_fault": False,
                "serve_number": ServeNumber.FIRST,
                "returnable_first_serve": True,
                "playable_second_serve": False,
                "rally_eligible": True,
            },
            ServicePointBranch.DOUBLE_FAULT: {
                "first_serve_in": False,
                "ace": False,
                "double_fault": True,
                "serve_number": ServeNumber.SECOND,
                "returnable_first_serve": False,
                "playable_second_serve": False,
                "server_won": False,
                "rally_eligible": False,
            },
            ServicePointBranch.PLAYABLE_SECOND_SERVE: {
                "first_serve_in": False,
                "ace": False,
                "double_fault": False,
                "serve_number": ServeNumber.SECOND,
                "returnable_first_serve": False,
                "playable_second_serve": True,
                "rally_eligible": True,
            },
        }
        for field, required in expected[self.branch].items():
            if getattr(self, field) != required:
                raise PointInvariantError(
                    f"{field} is inconsistent with branch {self.branch.value}"
                )

    @property
    def winner_id(self) -> str | None:
        """Return the point winner when player IDs were supplied."""

        if self.server_id is None:
            return None
        if self.server_won:
            return self.server_id
        if self.receiver_id is None:  # Joint ID validation makes this unreachable.
            raise AssertionError("validated point lost its receiver ID")
        return self.receiver_id

    @property
    def q1_used(self) -> bool:
        """Whether ``Q1`` causally generated the point outcome."""

        return self.branch is ServicePointBranch.RETURNABLE_FIRST_SERVE

    @property
    def q2_used(self) -> bool:
        """Whether ``Q2`` causally generated the point outcome."""

        return self.branch is ServicePointBranch.PLAYABLE_SECOND_SERVE


def _ace(*, server_id: str | None, receiver_id: str | None) -> ServicePointResult:
    return ServicePointResult(
        server_id=server_id,
        receiver_id=receiver_id,
        first_serve_in=True,
        ace=True,
        double_fault=False,
        serve_number=ServeNumber.FIRST,
        returnable_first_serve=False,
        playable_second_serve=False,
        server_won=True,
        rally_eligible=False,
        branch=ServicePointBranch.ACE,
    )


def _returnable_first(
    *,
    server_won: bool,
    server_id: str | None,
    receiver_id: str | None,
) -> ServicePointResult:
    return ServicePointResult(
        server_id=server_id,
        receiver_id=receiver_id,
        first_serve_in=True,
        ace=False,
        double_fault=False,
        serve_number=ServeNumber.FIRST,
        returnable_first_serve=True,
        playable_second_serve=False,
        server_won=server_won,
        rally_eligible=True,
        branch=ServicePointBranch.RETURNABLE_FIRST_SERVE,
    )


def _double_fault(*, server_id: str | None, receiver_id: str | None) -> ServicePointResult:
    return ServicePointResult(
        server_id=server_id,
        receiver_id=receiver_id,
        first_serve_in=False,
        ace=False,
        double_fault=True,
        serve_number=ServeNumber.SECOND,
        returnable_first_serve=False,
        playable_second_serve=False,
        server_won=False,
        rally_eligible=False,
        branch=ServicePointBranch.DOUBLE_FAULT,
    )


def _playable_second(
    *,
    server_won: bool,
    server_id: str | None,
    receiver_id: str | None,
) -> ServicePointResult:
    return ServicePointResult(
        server_id=server_id,
        receiver_id=receiver_id,
        first_serve_in=False,
        ace=False,
        double_fault=False,
        serve_number=ServeNumber.SECOND,
        returnable_first_serve=False,
        playable_second_serve=True,
        server_won=server_won,
        rally_eligible=True,
        branch=ServicePointBranch.PLAYABLE_SECOND_SERVE,
    )


def _performance(value: ServePerformanceDraw) -> ServePerformanceDraw:
    if not isinstance(value, ServePerformanceDraw):
        raise TypeError("performance must be a ServePerformanceDraw")
    return value


def _player_ids(server_id: str | None, receiver_id: str | None) -> None:
    """Validate optional point metadata before a stochastic call consumes RNG."""

    if (server_id is None) != (receiver_id is None):
        raise PointInvariantError("server_id and receiver_id must be supplied together")
    if server_id is None:
        return
    if not isinstance(server_id, str) or not server_id.strip():
        raise PointInvariantError("server_id must be a nonempty string")
    if not isinstance(receiver_id, str) or not receiver_id.strip():
        raise PointInvariantError("receiver_id must be a nonempty string")
    if server_id == receiver_id:
        raise PointInvariantError("server_id and receiver_id must be distinct")


def generate_point_from_uniforms(
    performance: ServePerformanceDraw,
    uniforms: PointUniforms,
    *,
    server_id: str | None = None,
    receiver_id: str | None = None,
) -> ServicePointResult:
    """Deterministically transform five uniforms through the B3 causal tree.

    This companion to :func:`generate_service_point` is the reference coupling
    interface for pathwise monotonicity tests.  All uniforms are explicit, but
    only uniforms on the realized causal branch influence the returned point.
    """

    draw = _performance(performance)
    if not isinstance(uniforms, PointUniforms):
        raise TypeError("uniforms must be PointUniforms")
    _player_ids(server_id, receiver_id)

    if uniforms.first_serve_in < draw.first_serve_in:
        if uniforms.ace_given_first_in < draw.ace_given_first_in:
            return _ace(server_id=server_id, receiver_id=receiver_id)
        return _returnable_first(
            server_won=uniforms.returnable_first_win < draw.returnable_first_win,
            server_id=server_id,
            receiver_id=receiver_id,
        )

    if uniforms.double_fault_given_second_opp < draw.double_fault_given_second_opp:
        return _double_fault(server_id=server_id, receiver_id=receiver_id)
    return _playable_second(
        server_won=uniforms.playable_second_win < draw.playable_second_win,
        server_id=server_id,
        receiver_id=receiver_id,
    )


def generate_service_point(
    performance: ServePerformanceDraw,
    rng: np.random.Generator,
    *,
    server_id: str | None = None,
    receiver_id: str | None = None,
) -> ServicePointResult:
    """Generate one point using an explicit NumPy RNG and lazy branch draws.

    The function consumes two RNG uniforms for an ace or double fault and three
    for a playable branch.  It does not consume the unused ``Q1``/``Q2`` draw
    after an immediate terminal event.  Consequently, exact replay requires
    the same primitive inputs, RNG algorithm and state, and call sequence.
    No module-global randomness or hidden reseeding is used.
    """

    draw = _performance(performance)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    _player_ids(server_id, receiver_id)

    if float(rng.random()) < draw.first_serve_in:
        if float(rng.random()) < draw.ace_given_first_in:
            return _ace(server_id=server_id, receiver_id=receiver_id)
        return _returnable_first(
            server_won=float(rng.random()) < draw.returnable_first_win,
            server_id=server_id,
            receiver_id=receiver_id,
        )

    if float(rng.random()) < draw.double_fault_given_second_opp:
        return _double_fault(server_id=server_id, receiver_id=receiver_id)
    return _playable_second(
        server_won=float(rng.random()) < draw.playable_second_win,
        server_id=server_id,
        receiver_id=receiver_id,
    )


@dataclass(frozen=True, slots=True)
class ServicePointCounts:
    """Exact primitive and official-style counts reconstructed from points."""

    service_points: int
    first_serves_in: int
    aces: int
    q1_trials: int
    q1_wins: int
    second_serve_opportunities: int
    double_faults: int
    q2_trials: int
    q2_wins: int
    first_serve_points_won: int
    second_serve_points_won: int

    def __post_init__(self) -> None:
        for field in (
            "service_points",
            "first_serves_in",
            "aces",
            "q1_trials",
            "q1_wins",
            "second_serve_opportunities",
            "double_faults",
            "q2_trials",
            "q2_wins",
            "first_serve_points_won",
            "second_serve_points_won",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PointInvariantError(f"{field} must be a nonnegative integer")

        if self.service_points != self.first_serves_in + self.second_serve_opportunities:
            raise PointInvariantError(
                "service_points must equal first serves in plus second-serve opportunities"
            )
        if self.first_serves_in != self.aces + self.q1_trials:
            raise PointInvariantError("first_serves_in must equal aces plus q1_trials")
        if self.first_serve_points_won != self.aces + self.q1_wins:
            raise PointInvariantError("first_serve_points_won must equal aces plus q1_wins")
        if self.q1_wins > self.q1_trials:
            raise PointInvariantError("q1_wins cannot exceed q1_trials")
        if self.second_serve_opportunities != self.double_faults + self.q2_trials:
            raise PointInvariantError(
                "second_serve_opportunities must equal double faults plus q2_trials"
            )
        if self.second_serve_points_won != self.q2_wins:
            raise PointInvariantError("second_serve_points_won must equal q2_wins")
        if self.q2_wins > self.q2_trials:
            raise PointInvariantError("q2_wins cannot exceed q2_trials")

    @property
    def server_points_won(self) -> int:
        """Total service points won across first- and second-serve branches."""

        return self.first_serve_points_won + self.second_serve_points_won


def aggregate_service_points(points: Iterable[ServicePointResult]) -> ServicePointCounts:
    """Reconstruct exact B1 denominators and successes from point records."""

    service_points = 0
    first_serves_in = 0
    aces = 0
    q1_trials = 0
    q1_wins = 0
    second_serve_opportunities = 0
    double_faults = 0
    q2_trials = 0
    q2_wins = 0
    first_serve_points_won = 0
    second_serve_points_won = 0

    for point in points:
        if not isinstance(point, ServicePointResult):
            raise TypeError("points must contain only ServicePointResult values")
        service_points += 1
        if point.first_serve_in:
            first_serves_in += 1
            first_serve_points_won += int(point.server_won)
        else:
            second_serve_opportunities += 1
            second_serve_points_won += int(point.server_won)

        aces += int(point.ace)
        double_faults += int(point.double_fault)
        if point.q1_used:
            q1_trials += 1
            q1_wins += int(point.server_won)
        if point.q2_used:
            q2_trials += 1
            q2_wins += int(point.server_won)

    return ServicePointCounts(
        service_points=service_points,
        first_serves_in=first_serves_in,
        aces=aces,
        q1_trials=q1_trials,
        q1_wins=q1_wins,
        second_serve_opportunities=second_serve_opportunities,
        double_faults=double_faults,
        q2_trials=q2_trials,
        q2_wins=q2_wins,
        first_serve_points_won=first_serve_points_won,
        second_serve_points_won=second_serve_points_won,
    )


__all__ = [
    "PointInvariantError",
    "PointUniforms",
    "ServeNumber",
    "ServePerformanceDraw",
    "ServicePointBranch",
    "ServicePointCounts",
    "ServicePointResult",
    "aggregate_service_points",
    "generate_point_from_uniforms",
    "generate_service_point",
]
