"""Explicit Section K prop gates and model-integer preview policy."""

from __future__ import annotations

from tennis_model.locking.models import (
    PropSupportDecision,
    PropSupportStatus,
)
from tennis_model.props.rounding import (
    MODEL_ROUNDING_POLICY_VERSION,
    model_probability_integer,
)
from tennis_model.simulation.match import BooleanCompositeSpec, PropSpec

SUBMISSION_ROUNDING_POLICY_VERSION = MODEL_ROUNDING_POLICY_VERSION

_WINNER_PROPS = {"WINNERS", "WINNER_COMPARE"}
_UE_PROPS = {"UNFORCED_ERRORS", "TOTAL_UNFORCED_ERRORS", "UE_COMPARE"}
_DURATION_PROPS = {"DURATION_MIN"}
_AUXILIARY_UNAVAILABLE = _WINNER_PROPS | _UE_PROPS


def integer_submission_preview(probability_raw: float) -> int:
    """Return the endpoint-preserving model integer under the centralized policy."""

    return model_probability_integer(probability_raw)


def _disabled(status: PropSupportStatus, code: str, detail: str) -> PropSupportDecision:
    return PropSupportDecision(status=status, reason_code=code, detail=detail)


def assess_prop_support(
    prop: PropSpec | BooleanCompositeSpec,
    *,
    duration_available: bool = False,
) -> PropSupportDecision:
    """Return a typed support decision without guessing unresolved conventions."""

    if isinstance(prop, BooleanCompositeSpec):
        children = tuple(
            assess_prop_support(child, duration_available=duration_available)
            for child in prop.exprs
        )
        disabled = next(
            (item for item in children if item.status is not PropSupportStatus.SUPPORTED),
            None,
        )
        if disabled is None:
            return PropSupportDecision(status=PropSupportStatus.SUPPORTED)
        return _disabled(
            disabled.status,
            f"COMPOUND_CHILD_{disabled.reason_code}",
            "compound contains a child whose accounting or settlement policy is unresolved",
        )
    if prop.kind in _WINNER_PROPS and "official_accounting_version" not in prop.scope:
        return _disabled(
            PropSupportStatus.POLICY_DISABLED,
            "OFFICIAL_WINNERS_ACE_ACCOUNTING_UNRESOLVED",
            "official inclusion of aces in winner totals is unresolved",
        )
    if prop.kind in _UE_PROPS and "official_accounting_version" not in prop.scope:
        return _disabled(
            PropSupportStatus.POLICY_DISABLED,
            "OFFICIAL_UE_DOUBLE_FAULT_ACCOUNTING_UNRESOLVED",
            "official inclusion of double faults in unforced errors is unresolved",
        )
    if prop.kind in _DURATION_PROPS and "display_conversion_version" not in prop.scope:
        return _disabled(
            PropSupportStatus.POLICY_DISABLED,
            "OFFICIAL_DURATION_MINUTE_CONVERSION_UNRESOLVED",
            "official displayed-minute conversion is unresolved",
        )
    if prop.kind in _DURATION_PROPS and not duration_available:
        return _disabled(
            PropSupportStatus.NOT_IMPLEMENTED,
            "DURATION_ARTIFACT_NOT_ATTACHED",
            "the match parameter distribution has no fitted duration artifact",
        )
    if prop.kind == "FIRST_SERVE_WIN_PCT" and not (
        prop.scope.get("rounding_invariant") is True
        or isinstance(prop.scope.get("display_conversion_version"), str)
    ):
        return _disabled(
            PropSupportStatus.POLICY_DISABLED,
            "FIRST_SERVE_WIN_PERCENT_DISPLAY_CONVERSION_UNRESOLVED",
            "official percentage display conversion can change this threshold outcome",
        )
    if prop.kind in _AUXILIARY_UNAVAILABLE:
        return _disabled(
            PropSupportStatus.NOT_IMPLEMENTED,
            "AUXILIARY_PROP_GENERATOR_NOT_IMPLEMENTED",
            "the required winner/UE auxiliary generator is not implemented",
        )
    return PropSupportDecision(status=PropSupportStatus.SUPPORTED)


def prop_generation_available(
    prop: PropSpec | BooleanCompositeSpec,
    *,
    duration_available: bool = False,
) -> bool:
    if isinstance(prop, BooleanCompositeSpec):
        return all(
            prop_generation_available(child, duration_available=duration_available)
            for child in prop.exprs
        )
    if prop.kind in _DURATION_PROPS:
        return duration_available
    return prop.kind not in _AUXILIARY_UNAVAILABLE


__all__ = [
    "SUBMISSION_ROUNDING_POLICY_VERSION",
    "assess_prop_support",
    "integer_submission_preview",
    "prop_generation_available",
]
