"""Centralized model-probability rounding and external-platform transforms."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ROUNDING_POLICY_VERSION = "nearest-percent-half-up-endpoints/v1"


def model_probability_integer(probability_raw: float) -> int:
    """Round a model probability to 0--100 using deterministic half-up ties.

    The model layer deliberately retains the endpoint integers.  Any external
    platform restriction is a separate transform applied by
    :class:`PlatformSubmissionPolicy`.
    """

    if not 0.0 <= probability_raw <= 1.0:
        raise ValueError("raw probability must lie in [0, 1]")
    return int(probability_raw * 100.0 + 0.5)


def integer_bucket(integer: int) -> tuple[float, float, bool, bool]:
    """Return the exact half-up model bucket as ``(low, high, low_closed, high_closed)``."""

    if isinstance(integer, bool) or not isinstance(integer, int):
        raise TypeError("integer bucket must be identified by an integer")
    if not 0 <= integer <= 100:
        raise ValueError("model integer must lie in [0, 100]")
    if integer == 0:
        return (0.0, 0.005, True, False)
    if integer == 100:
        return (0.995, 1.0, True, True)
    return ((integer - 0.5) / 100.0, (integer + 0.5) / 100.0, True, False)


def confidence_interval_model_integer(lower: float, upper: float) -> int | None:
    """Return one integer iff every point in the closed interval maps to it."""

    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("confidence interval must lie in [0, 1]")
    low_integer = model_probability_integer(lower)
    high_integer = model_probability_integer(upper)
    return low_integer if low_integer == high_integer else None


@dataclass(frozen=True, slots=True)
class PlatformSubmissionPolicy:
    """Explicitly transform model integers for an external platform only."""

    version: str
    minimum_integer: int = 1
    maximum_integer: int = 99

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("platform submission policy version must not be blank")
        if not 0 <= self.minimum_integer <= self.maximum_integer <= 100:
            raise ValueError("platform integer bounds must lie in [0, 100]")

    def transform(self, model_integer: int) -> int:
        if isinstance(model_integer, bool) or not isinstance(model_integer, int):
            raise TypeError("model integer must be an integer")
        if not 0 <= model_integer <= 100:
            raise ValueError("model integer must lie in [0, 100]")
        return min(self.maximum_integer, max(self.minimum_integer, model_integer))


SPORTSPREDICT_SUBMISSION_POLICY = PlatformSubmissionPolicy(
    version="sportspredict-clamp-1-99/v1",
)


__all__ = [
    "MODEL_ROUNDING_POLICY_VERSION",
    "SPORTSPREDICT_SUBMISSION_POLICY",
    "PlatformSubmissionPolicy",
    "confidence_interval_model_integer",
    "integer_bucket",
    "model_probability_integer",
]
