"""Dependency-free truth and settlement primitives.

This module deliberately separates a path event's truth from its settlement
disposition.  In particular, an unresolved or void event is never represented
as a settled ``False`` outcome, and a missing policy ruling is blocked rather
than guessed.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class EventTruth(StrEnum):
    """Three-valued truth for an event evaluated on one match path."""

    TRUE = "true"
    FALSE = "false"
    UNRESOLVED = "unresolved"

    def __bool__(self) -> bool:
        """Reject implicit conversion, especially unresolved-to-false bugs."""
        raise TypeError("EventTruth has no implicit Boolean interpretation")

    @classmethod
    def from_bool(cls, value: bool) -> "EventTruth":
        """Construct a resolved truth value from an actual ``bool``."""
        if type(value) is not bool:
            raise TypeError("value must be a bool")
        return cls.TRUE if value else cls.FALSE


class SettlementState(StrEnum):
    """Canonical final settlement state used by retired or incomplete paths."""

    YES = "yes"
    NO = "no"
    VOID = "void"
    UNRESOLVED = "unresolved"

    @classmethod
    def from_truth(cls, value: "EventTruth | bool | None") -> "SettlementState":
        if value is None:
            return cls.VOID
        if isinstance(value, bool):
            return cls.YES if value else cls.NO
        if not isinstance(value, EventTruth):
            raise TypeError("value must be an EventTruth, bool, or None")
        if value is EventTruth.TRUE:
            return cls.YES
        if value is EventTruth.FALSE:
            return cls.NO
        return cls.UNRESOLVED

    @classmethod
    def from_resolved(cls, value: bool, *, settlement_policy_version: str) -> "SettlementState":
        _ = settlement_policy_version
        return cls.YES if value else cls.NO


class PolicyBlockedError(RuntimeError):
    """Raised when a settlement rule requires a policy ruling not available here."""


def _require_nonempty_text(value: str, *, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


@dataclass(frozen=True, slots=True)
class SettlementPolicy:
    """Versioned settlement semantics for a frozen model policy."""

    version: str
    comparison_tie_is_no: bool = True
    walkover_voids_all: bool = True
    allow_policy_blocked: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_text(self.version, field="version")
        if type(self.comparison_tie_is_no) is not bool:
            raise TypeError("comparison_tie_is_no must be a bool")
        if type(self.walkover_voids_all) is not bool:
            raise TypeError("walkover_voids_all must be a bool")
        if type(self.allow_policy_blocked) is not bool:
            raise TypeError("allow_policy_blocked must be a bool")
        if self.description and not isinstance(self.description, str):
            raise TypeError("description must be a string")


CANONICAL_SETTLEMENT_POLICY_VERSION = "tennis-model-v1.0/settlement-v1"
CANONICAL_SETTLEMENT_POLICY = SettlementPolicy(
    version=CANONICAL_SETTLEMENT_POLICY_VERSION,
    description="Frozen Tennis Model v1.0 canonical US Open settlement policy",
)


def truth_and(values: Iterable[EventTruth]) -> EventTruth:
    """Return the strong three-valued conjunction of *values*.

    A false component determines the conjunction even when another component
    is unresolved.  Otherwise the result is unresolved unless every component
    is true.  The empty conjunction has its standard identity value, true.
    """
    false = False
    unresolved = False
    for value in values:
        _require_truth(value)
        if value is EventTruth.FALSE:
            false = True
        elif value is EventTruth.UNRESOLVED:
            unresolved = True
    if false:
        return EventTruth.FALSE
    return EventTruth.UNRESOLVED if unresolved else EventTruth.TRUE


def truth_or(values: Iterable[EventTruth]) -> EventTruth:
    """Return the strong three-valued disjunction of *values*.

    A true component determines the disjunction even when another component is
    unresolved.  Otherwise the result is unresolved unless every component is
    false.  The empty disjunction has its standard identity value, false.
    """
    true = False
    unresolved = False
    for value in values:
        _require_truth(value)
        if value is EventTruth.TRUE:
            true = True
        elif value is EventTruth.UNRESOLVED:
            unresolved = True
    if true:
        return EventTruth.TRUE
    return EventTruth.UNRESOLVED if unresolved else EventTruth.FALSE


class ComparisonOperator(StrEnum):
    """Canonical numeric comparison operators used by threshold props."""

    AT_LEAST = ">="
    MORE_THAN = ">"
    FEWER_THAN = "<"


type ComparableNumber = int | float


def compare(
    value: ComparableNumber,
    operator: ComparisonOperator,
    threshold: ComparableNumber,
) -> EventTruth:
    """Evaluate a canonical threshold comparison without fuzzy tie handling."""
    _require_finite_number(value, field="value")
    _require_finite_number(threshold, field="threshold")
    if not isinstance(operator, ComparisonOperator):
        raise TypeError("operator must be a ComparisonOperator")

    if operator is ComparisonOperator.AT_LEAST:
        return EventTruth.from_bool(value >= threshold)
    if operator is ComparisonOperator.MORE_THAN:
        return EventTruth.from_bool(value > threshold)
    if operator is ComparisonOperator.FEWER_THAN:
        return EventTruth.from_bool(value < threshold)
    raise AssertionError(f"unhandled comparison operator: {operator!r}")


@dataclass(frozen=True, slots=True)
class Settled:
    """A market path with a genuine Yes/No settlement outcome."""

    outcome: bool
    settlement_policy_version: str

    def __post_init__(self) -> None:
        if type(self.outcome) is not bool:
            raise TypeError("outcome must be a bool")
        _require_nonempty_text(
            self.settlement_policy_version,
            field="settlement_policy_version",
        )


@dataclass(frozen=True, slots=True)
class Voided:
    """A market path that a known policy excludes from Yes/No settlement."""

    settlement_policy_version: str
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self.settlement_policy_version,
            field="settlement_policy_version",
        )
        _require_nonempty_text(self.reason, field="reason")


@dataclass(frozen=True, slots=True)
class Blocked:
    """A market path that cannot be evaluated under the named policy version."""

    settlement_policy_version: str
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self.settlement_policy_version,
            field="settlement_policy_version",
        )
        _require_nonempty_text(self.reason, field="reason")


type SettlementResult = Settled | Voided | Blocked


def _require_truth(value: EventTruth) -> None:
    if not isinstance(value, EventTruth):
        raise TypeError("truth operands must be EventTruth values")


def _require_finite_number(value: ComparableNumber, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be an int or float")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{field} must be finite")
