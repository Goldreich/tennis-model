"""Versioned fixed and anytime-valid adaptive Monte Carlo policies.

``adaptive_mc_cs_v1`` uses the beta-binomial mixture likelihood-ratio
martingale with a Jeffreys ``Beta(1/2, 1/2)`` mixing distribution.  For a
candidate Bernoulli probability ``p``, successes ``S_n`` and failures
``F_n=n-S_n``, the martingale is

``M_n(p) = B(S_n+a, F_n+b) / (B(a,b) p**S_n (1-p)**F_n)``.

The 99% confidence sequence is ``{p: M_n(p) <= 1/alpha}``, with
``alpha=0.01``.  For every fixed ``p``, ``M_n(p)`` is a nonnegative,
mean-one mixture likelihood-ratio martingale under ``P_p``.  Ville's
inequality therefore gives ``P_p(exists n: M_n(p) >= 1/alpha) <= alpha``;
inverting at every time gives simultaneous coverage under optional stopping.

This is the beta-binomial mixture confidence-sequence construction described
by Howard et al. (2021), *Time-uniform, nonparametric, nonasymptotic confidence
sequences*, Annals of Statistics 49(2), Section 3 / beta-binomial mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import floor, isclose, lgamma, log, log1p

from tennis_model.props.rounding import (
    MODEL_ROUNDING_POLICY_VERSION,
    confidence_interval_model_integer,
    model_probability_integer,
)
from tennis_model.simulation.match import PropEstimate

ADAPTIVE_MC_POLICY_VERSION = "adaptive_mc_cs_v1"
BERNOULLI_CS_METHOD_VERSION = "beta-binomial-jeffreys-mixture/v1"


@dataclass(frozen=True, slots=True)
class PathCountPolicy:
    standard_paths: int = 100_000
    escalated_paths: int = 400_000
    minimum_settled_paths: int = 50_000
    extreme_probability: float = 0.03
    integer_boundary_window: float = 0.005
    integer_boundary_standard_errors: float = 2.0

    def __post_init__(self) -> None:
        if self.standard_paths <= 0 or self.escalated_paths < self.standard_paths:
            raise ValueError("path counts must be positive and nondecreasing")
        if not 0 <= self.minimum_settled_paths <= self.standard_paths:
            raise ValueError("minimum settled paths must fit the standard run")
        if not 0 < self.extreme_probability < 0.5:
            raise ValueError("extreme-probability threshold must lie in (0, 0.5)")
        if self.integer_boundary_window < 0 or self.integer_boundary_standard_errors < 0:
            raise ValueError("integer-boundary thresholds must be nonnegative")


FROZEN_PATH_COUNT_POLICY = PathCountPolicy()
FIXED_50K_V1_POLICY = PathCountPolicy(
    standard_paths=50_000,
    escalated_paths=50_000,
    minimum_settled_paths=50_000,
)
FIXED_100K_V1_POLICY = PathCountPolicy(
    standard_paths=100_000,
    escalated_paths=100_000,
    minimum_settled_paths=100_000,
)


class MCStoppingStatus(StrEnum):
    """Per-prop state at an inspected adaptive checkpoint."""

    INTEGER_STABLE = "INTEGER_STABLE"
    INTEGER_BOUNDARY_SENSITIVE = "INTEGER_BOUNDARY_SENSITIVE"
    CONTINUE = "CONTINUE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class AdaptiveMCPolicy:
    """Auditable checkpoint and confidence-sequence policy for live forecasts."""

    checkpoints: tuple[int, ...] = (5_000, 10_000, 20_000, 40_000, 70_000)
    confidence_level: float = 0.99
    beta_prior_a: float = 0.5
    beta_prior_b: float = 0.5
    version: str = ADAPTIVE_MC_POLICY_VERSION
    confidence_sequence_method: str = BERNOULLI_CS_METHOD_VERSION
    model_rounding_policy_version: str = MODEL_ROUNDING_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != ADAPTIVE_MC_POLICY_VERSION:
            raise ValueError("unknown adaptive Monte Carlo policy version")
        if self.confidence_sequence_method != BERNOULLI_CS_METHOD_VERSION:
            raise ValueError("unknown Bernoulli confidence-sequence method")
        if self.model_rounding_policy_version != MODEL_ROUNDING_POLICY_VERSION:
            raise ValueError("adaptive policy must use the centralized model rounding policy")
        if not self.checkpoints or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.checkpoints
        ):
            raise ValueError("adaptive checkpoints must be positive integers")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("adaptive checkpoints must be strictly increasing")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0, 1)")
        if not isclose(self.confidence_level, 0.99, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("adaptive_mc_cs_v1 freezes the confidence level at 99%")
        if self.beta_prior_a <= 0.0 or self.beta_prior_b <= 0.0:
            raise ValueError("beta mixture parameters must be positive")
        if not (
            isclose(self.beta_prior_a, 0.5, rel_tol=0.0, abs_tol=1e-15)
            and isclose(self.beta_prior_b, 0.5, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError("adaptive_mc_cs_v1 freezes the Jeffreys Beta(1/2, 1/2) mixture")

    @property
    def maximum_paths(self) -> int:
        return self.checkpoints[-1]


ADAPTIVE_MC_CS_V1_POLICY = AdaptiveMCPolicy()


@dataclass(frozen=True, slots=True)
class BernoulliConfidenceSequence:
    successes: int
    trials: int
    lower: float
    upper: float
    confidence_level: float
    method_version: str = BERNOULLI_CS_METHOD_VERSION

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("a confidence sequence requires at least one settled trial")
        if not 0 <= self.successes <= self.trials:
            raise ValueError("successes must lie in [0, trials]")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("confidence-sequence bounds must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdaptivePropDiagnostics:
    confidence_sequence: BernoulliConfidenceSequence | None
    model_probability_raw: float | None
    model_probability_integer: int | None
    stopping_status: MCStoppingStatus
    final_cumulative_path_count: int
    confidence_level: float


def _log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def beta_binomial_mixture_log_martingale(
    successes: int,
    trials: int,
    probability: float,
    *,
    beta_prior_a: float = 0.5,
    beta_prior_b: float = 0.5,
) -> float:
    """Evaluate ``log M_n(p)`` with exact impossible-boundary handling."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if beta_prior_a <= 0.0 or beta_prior_b <= 0.0:
        raise ValueError("beta mixture parameters must be positive")
    failures = trials - successes
    log_mixture = _log_beta(successes + beta_prior_a, failures + beta_prior_b) - _log_beta(
        beta_prior_a, beta_prior_b
    )
    if probability == 0.0:
        return log_mixture if successes == 0 else float("inf")
    if probability == 1.0:
        return log_mixture if failures == 0 else float("inf")
    return log_mixture - successes * log(probability) - failures * log1p(-probability)


def _bisect_martingale_boundary(
    successes: int,
    trials: int,
    left: float,
    right: float,
    target: float,
    *,
    beta_prior_a: float,
    beta_prior_b: float,
    increasing: bool,
) -> float:
    """Deterministically invert one convex-mixture-martingale branch."""

    for _ in range(80):
        midpoint = (left + right) / 2.0
        value = beta_binomial_mixture_log_martingale(
            successes,
            trials,
            midpoint,
            beta_prior_a=beta_prior_a,
            beta_prior_b=beta_prior_b,
        )
        if increasing:
            if value < target:
                left = midpoint
            else:
                right = midpoint
        else:
            if value < target:
                right = midpoint
            else:
                left = midpoint
    return (left + right) / 2.0


def bernoulli_confidence_sequence(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.99,
    beta_prior_a: float = 0.5,
    beta_prior_b: float = 0.5,
) -> BernoulliConfidenceSequence:
    """Invert the beta-binomial mixture martingale into an anytime-valid CS.

    Zero successes return an exact lower endpoint of zero and a positive finite-
    sample upper endpoint.  All successes behave symmetrically with an exact
    upper endpoint of one.  No pseudo-count enters ``successes / trials``; the
    beta parameters occur only in the confidence-sequence martingale.
    """

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    target = log(1.0 / (1.0 - confidence_level))
    estimate = successes / trials
    lower = (
        0.0
        if successes == 0
        else _bisect_martingale_boundary(
            successes,
            trials,
            0.0,
            estimate,
            target,
            beta_prior_a=beta_prior_a,
            beta_prior_b=beta_prior_b,
            increasing=False,
        )
    )
    upper = (
        1.0
        if successes == trials
        else _bisect_martingale_boundary(
            successes,
            trials,
            estimate,
            1.0,
            target,
            beta_prior_a=beta_prior_a,
            beta_prior_b=beta_prior_b,
            increasing=True,
        )
    )
    return BernoulliConfidenceSequence(
        successes=successes,
        trials=trials,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def adaptive_prop_diagnostics(
    estimate: PropEstimate,
    policy: AdaptiveMCPolicy = ADAPTIVE_MC_CS_V1_POLICY,
    *,
    terminal: bool,
) -> AdaptivePropDiagnostics:
    """Compute settlement-conditioned CS and integer-stability state."""

    if estimate.settled_paths == 0:
        return AdaptivePropDiagnostics(
            confidence_sequence=None,
            model_probability_raw=None,
            model_probability_integer=None,
            stopping_status=(
                MCStoppingStatus.UNAVAILABLE if terminal else MCStoppingStatus.CONTINUE
            ),
            final_cumulative_path_count=estimate.total_paths,
            confidence_level=policy.confidence_level,
        )
    raw = estimate.yes_paths / estimate.settled_paths
    if raw != estimate.probability_raw:
        raise ValueError("raw probability must equal Yes / settled paths exactly")
    sequence = bernoulli_confidence_sequence(
        estimate.yes_paths,
        estimate.settled_paths,
        confidence_level=policy.confidence_level,
        beta_prior_a=policy.beta_prior_a,
        beta_prior_b=policy.beta_prior_b,
    )
    integer = model_probability_integer(raw)
    stable_integer = confidence_interval_model_integer(sequence.lower, sequence.upper)
    stable = stable_integer == integer
    return AdaptivePropDiagnostics(
        confidence_sequence=sequence,
        model_probability_raw=raw,
        model_probability_integer=integer,
        stopping_status=(
            MCStoppingStatus.INTEGER_STABLE
            if stable
            else MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE
            if terminal
            else MCStoppingStatus.CONTINUE
        ),
        final_cumulative_path_count=estimate.total_paths,
        confidence_level=policy.confidence_level,
    )


def _distance_to_rounding_boundary(probability: float) -> float:
    percentage = probability * 100.0
    lower = floor(percentage) - 0.5
    boundaries = (lower, lower + 1.0, lower + 2.0)
    return min(abs(percentage - boundary) for boundary in boundaries) / 100.0


def escalation_reasons(
    estimates: tuple[PropEstimate, ...],
    policy: PathCountPolicy = FROZEN_PATH_COUNT_POLICY,
) -> tuple[str, ...]:
    """Return stable trigger labels; callers rerun once and never combine samples."""

    reasons: set[str] = set()
    for estimate in estimates:
        probability = estimate.probability_raw
        if probability < policy.extreme_probability:
            reasons.add("PROBABILITY_BELOW_3_PERCENT")
        if probability > 1.0 - policy.extreme_probability:
            reasons.add("PROBABILITY_ABOVE_97_PERCENT")
        if estimate.settled_paths < policy.minimum_settled_paths:
            reasons.add(f"FEWER_THAN_{policy.minimum_settled_paths}_SETTLED_PATHS")
        distance = _distance_to_rounding_boundary(probability)
        if (
            distance <= policy.integer_boundary_window
            and distance <= policy.integer_boundary_standard_errors * estimate.mc_standard_error
        ):
            reasons.add("INTEGER_SUBMISSION_MC_SENSITIVITY")
    return tuple(sorted(reasons))
