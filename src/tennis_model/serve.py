"""Pure reporting identities for the five primitive Tennis Model v1.0 values.

This lightweight shared module contains no fitting or simulation logic.  The
estimation compatibility namespace and the causal point generator both consume
these exact implementations so derived probabilities cannot drift between
milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _probability(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite probability")
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def first_serve_win_probability(ace_given_first_in: float, returnable_first_win: float) -> float:
    """Return ``w1 = a + (1-a)q1`` without changing the primitive estimands."""

    ace = _probability(ace_given_first_in, field="ace_given_first_in")
    q1 = _probability(returnable_first_win, field="returnable_first_win")
    return ace + (1.0 - ace) * q1


def second_serve_win_probability(
    double_fault_given_second_opp: float,
    playable_second_win: float,
) -> float:
    """Return ``w2 = (1-d)q2`` without fitting a direct w2 model."""

    double_fault = _probability(
        double_fault_given_second_opp,
        field="double_fault_given_second_opp",
    )
    q2 = _probability(playable_second_win, field="playable_second_win")
    return (1.0 - double_fault) * q2


def service_point_win_probability(
    first_serve_in: float,
    ace_given_first_in: float,
    returnable_first_win: float,
    double_fault_given_second_opp: float,
    playable_second_win: float,
) -> float:
    """Return the unconditional service-point win identity from F/A/Q1/D/Q2."""

    first_in = _probability(first_serve_in, field="first_serve_in")
    w1 = first_serve_win_probability(ace_given_first_in, returnable_first_win)
    w2 = second_serve_win_probability(double_fault_given_second_opp, playable_second_win)
    return first_in * w1 + (1.0 - first_in) * w2


def ace_rate_per_service_point(first_serve_in: float, ace_given_first_in: float) -> float:
    """Return ``rA = f*a``."""

    first_in = _probability(first_serve_in, field="first_serve_in")
    ace = _probability(ace_given_first_in, field="ace_given_first_in")
    return first_in * ace


def double_fault_rate_per_service_point(
    first_serve_in: float,
    double_fault_given_second_opp: float,
) -> float:
    """Return ``rD = (1-f)d``."""

    first_in = _probability(first_serve_in, field="first_serve_in")
    double_fault = _probability(
        double_fault_given_second_opp,
        field="double_fault_given_second_opp",
    )
    return (1.0 - first_in) * double_fault


@dataclass(frozen=True, slots=True)
class PrimitiveServeMeans:
    """Five primitive means with derived reporting properties only."""

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
            _probability(getattr(self, field), field=field)

    @property
    def first_serve_win(self) -> float:
        return first_serve_win_probability(
            self.ace_given_first_in,
            self.returnable_first_win,
        )

    @property
    def second_serve_win(self) -> float:
        return second_serve_win_probability(
            self.double_fault_given_second_opp,
            self.playable_second_win,
        )

    @property
    def service_point_win(self) -> float:
        return service_point_win_probability(
            self.first_serve_in,
            self.ace_given_first_in,
            self.returnable_first_win,
            self.double_fault_given_second_opp,
            self.playable_second_win,
        )

    @property
    def ace_rate(self) -> float:
        return ace_rate_per_service_point(self.first_serve_in, self.ace_given_first_in)

    @property
    def double_fault_rate(self) -> float:
        return double_fault_rate_per_service_point(
            self.first_serve_in,
            self.double_fault_given_second_opp,
        )


__all__ = [
    "PrimitiveServeMeans",
    "ace_rate_per_service_point",
    "double_fault_rate_per_service_point",
    "first_serve_win_probability",
    "second_serve_win_probability",
    "service_point_win_probability",
]
