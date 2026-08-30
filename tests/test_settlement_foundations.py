from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from tennis_model.props.settlement import (
    CANONICAL_SETTLEMENT_POLICY,
    Blocked,
    ComparisonOperator,
    EventTruth,
    Settled,
    SettlementState,
    Voided,
    compare,
    truth_and,
    truth_or,
)


@pytest.mark.parametrize("truth", list(EventTruth))
def test_event_truth_prohibits_implicit_boolean_coercion(truth: EventTruth) -> None:
    with pytest.raises(TypeError, match="no implicit Boolean"):
        bool(truth)


def test_event_truth_from_bool_requires_an_actual_bool() -> None:
    assert EventTruth.from_bool(True) is EventTruth.TRUE
    assert EventTruth.from_bool(False) is EventTruth.FALSE
    with pytest.raises(TypeError, match="must be a bool"):
        EventTruth.from_bool(1)  # type: ignore[arg-type]


def test_unresolved_truth_remains_distinct_from_void_and_no() -> None:
    assert SettlementState.from_truth(EventTruth.UNRESOLVED) is SettlementState.UNRESOLVED
    assert SettlementState.from_truth(None) is SettlementState.VOID
    assert SettlementState.from_truth(False) is SettlementState.NO
    assert CANONICAL_SETTLEMENT_POLICY.comparison_tie_is_no
    assert CANONICAL_SETTLEMENT_POLICY.walkover_voids_all


@pytest.mark.parametrize("left,right", list(product(EventTruth, repeat=2)))
def test_three_valued_and(left: EventTruth, right: EventTruth) -> None:
    if EventTruth.FALSE in (left, right):
        expected = EventTruth.FALSE
    elif EventTruth.UNRESOLVED in (left, right):
        expected = EventTruth.UNRESOLVED
    else:
        expected = EventTruth.TRUE
    assert truth_and((left, right)) is expected


@pytest.mark.parametrize("left,right", list(product(EventTruth, repeat=2)))
def test_three_valued_or(left: EventTruth, right: EventTruth) -> None:
    if EventTruth.TRUE in (left, right):
        expected = EventTruth.TRUE
    elif EventTruth.UNRESOLVED in (left, right):
        expected = EventTruth.UNRESOLVED
    else:
        expected = EventTruth.FALSE
    assert truth_or((left, right)) is expected


def test_empty_compound_expressions_use_boolean_identity_values() -> None:
    assert truth_and(()) is EventTruth.TRUE
    assert truth_or(()) is EventTruth.FALSE


@pytest.mark.parametrize("function", [truth_and, truth_or])
def test_compound_truth_rejects_untyped_values(function: object) -> None:
    with pytest.raises(TypeError, match="must be EventTruth"):
        function((EventTruth.TRUE, True))  # type: ignore[operator, arg-type]


@pytest.mark.parametrize(
    ("value", "operator", "threshold", "expected"),
    [
        (13, ComparisonOperator.AT_LEAST, 13, EventTruth.TRUE),
        (13, ComparisonOperator.MORE_THAN, 13, EventTruth.FALSE),
        (13, ComparisonOperator.FEWER_THAN, 13, EventTruth.FALSE),
        (12, ComparisonOperator.FEWER_THAN, 12.5, EventTruth.TRUE),
        (13, ComparisonOperator.MORE_THAN, 12.5, EventTruth.TRUE),
        (-1, ComparisonOperator.AT_LEAST, 0, EventTruth.FALSE),
    ],
)
def test_comparison_semantics(
    value: int | float,
    operator: ComparisonOperator,
    threshold: int | float,
    expected: EventTruth,
) -> None:
    assert compare(value, operator, threshold) is expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_comparison_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        compare(value, ComparisonOperator.MORE_THAN, 0)


def test_comparison_requires_typed_inputs() -> None:
    with pytest.raises(TypeError, match="operator must be"):
        compare(1, ">", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be"):
        compare(True, ComparisonOperator.MORE_THAN, 0)
    with pytest.raises(TypeError, match="threshold must be"):
        compare(1, ComparisonOperator.MORE_THAN, "0")  # type: ignore[arg-type]


def test_settlement_dispositions_are_distinct_and_do_not_code_void_as_no() -> None:
    no = Settled(outcome=False, settlement_policy_version="us-open-v1")
    void = Voided(
        settlement_policy_version="us-open-v1",
        reason="scope incomplete after retirement",
    )
    blocked = Blocked(
        settlement_policy_version="us-open-v1",
        reason="semantic ruling required",
    )

    assert no.outcome is False
    assert not hasattr(void, "outcome")
    assert not hasattr(blocked, "outcome")
    assert type(no) is Settled
    assert type(void) is Voided
    assert type(blocked) is Blocked


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Settled(outcome=True, settlement_policy_version=""),
        lambda: Voided(settlement_policy_version="  ", reason="known rule"),
        lambda: Voided(settlement_policy_version="v1", reason="\t"),
        lambda: Blocked(settlement_policy_version="", reason="unknown rule"),
        lambda: Blocked(settlement_policy_version="v1", reason="\n"),
    ],
)
def test_settlement_metadata_must_be_nonempty(factory: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        factory()  # type: ignore[operator]


def test_settled_outcome_requires_an_actual_bool() -> None:
    with pytest.raises(TypeError, match="outcome must be a bool"):
        Settled(outcome=1, settlement_policy_version="v1")  # type: ignore[arg-type]


def test_settlement_results_are_frozen() -> None:
    result = Settled(outcome=True, settlement_policy_version="v1")
    with pytest.raises(FrozenInstanceError):
        result.outcome = False  # type: ignore[misc]
