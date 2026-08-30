from __future__ import annotations

from math import log
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import betaln

from tennis_model.locking.models import PropEstimateRecord
from tennis_model.locking.path_counts import (
    AdaptiveMCPolicy,
    MCStoppingStatus,
    adaptive_prop_diagnostics,
    bernoulli_confidence_sequence,
    beta_binomial_mixture_log_martingale,
)
from tennis_model.props.rounding import (
    SPORTSPREDICT_SUBMISSION_POLICY,
    confidence_interval_model_integer,
    integer_bucket,
    model_probability_integer,
)
from tennis_model.simulation.match import MATCH_WIN, PropEstimate


def _estimate(yes: int, settled: int, *, void: int = 0) -> PropEstimate:
    raw = yes / settled if settled else 0.0
    total = settled + void
    return PropEstimate(
        prop=MATCH_WIN("A"),
        probability_raw=raw,
        probability_settled=settled / total if total else 0.0,
        yes_paths=yes,
        no_paths=settled - yes,
        void_paths=void,
        unresolved_paths=0,
        settled_paths=settled,
        total_paths=total,
        mc_standard_error=(raw * (1.0 - raw) / settled) ** 0.5 if settled else 0.0,
    )


def test_pinned_policy_config_matches_code_contract() -> None:
    payload = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "adaptive_mc_cs_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy = AdaptiveMCPolicy()
    assert payload["version"] == policy.version
    assert tuple(payload["checkpoints"]) == policy.checkpoints
    assert payload["confidence_level"] == policy.confidence_level
    assert payload["confidence_sequence"]["method"] == policy.confidence_sequence_method
    assert payload["confidence_sequence"]["beta_prior_a"] == policy.beta_prior_a
    assert payload["confidence_sequence"]["beta_prior_b"] == policy.beta_prior_b
    assert payload["model_rounding_policy"] == policy.model_rounding_policy_version


def test_zero_successes_keep_exact_raw_zero_and_positive_cs_upper() -> None:
    estimate = _estimate(0, 5_000)
    sequence = bernoulli_confidence_sequence(0, 5_000)
    diagnostics = adaptive_prop_diagnostics(estimate, terminal=False)

    assert estimate.probability_raw == 0.0
    assert sequence.lower == 0.0
    assert 0.0 < sequence.upper < 0.005
    assert diagnostics.model_probability_raw == 0.0
    assert diagnostics.model_probability_integer == 0
    assert diagnostics.stopping_status is MCStoppingStatus.INTEGER_STABLE
    assert estimate.yes_paths == 0  # no pseudo-count modifies the observed numerator


def test_all_successes_keep_exact_raw_one_and_finite_cs_lower() -> None:
    estimate = _estimate(5_000, 5_000)
    sequence = bernoulli_confidence_sequence(5_000, 5_000)
    diagnostics = adaptive_prop_diagnostics(estimate, terminal=False)

    assert estimate.probability_raw == 1.0
    assert sequence.upper == 1.0
    assert 0.995 < sequence.lower < 1.0
    assert diagnostics.model_probability_integer == 100
    assert diagnostics.stopping_status is MCStoppingStatus.INTEGER_STABLE


def test_observed_endpoints_need_not_yet_be_integer_stable() -> None:
    zero = adaptive_prop_diagnostics(_estimate(0, 100), terminal=True)
    one = adaptive_prop_diagnostics(_estimate(100, 100), terminal=True)

    assert zero.model_probability_integer == 0
    assert zero.confidence_sequence is not None
    assert zero.confidence_sequence.upper > 0.005
    assert zero.stopping_status is MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE
    assert one.model_probability_integer == 100
    assert one.confidence_sequence is not None
    assert one.confidence_sequence.lower < 0.995
    assert one.stopping_status is MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE


def test_extreme_nonzero_result_stabilizes_only_at_later_checkpoint() -> None:
    early = adaptive_prop_diagnostics(_estimate(10, 5_000), terminal=False)
    later = adaptive_prop_diagnostics(_estimate(20, 10_000), terminal=False)

    assert early.model_probability_raw == 0.002
    assert early.model_probability_integer == 0
    assert early.stopping_status is MCStoppingStatus.CONTINUE
    assert later.model_probability_raw == 0.002
    assert later.model_probability_integer == 0
    assert later.stopping_status is MCStoppingStatus.INTEGER_STABLE


def test_zero_settled_paths_have_no_probability_or_sequence() -> None:
    diagnostics = adaptive_prop_diagnostics(_estimate(0, 0, void=70_000), terminal=True)
    record = PropEstimateRecord.from_estimate(
        _estimate(0, 0, void=70_000),
        adaptive_diagnostics=diagnostics,
    )

    assert diagnostics.model_probability_raw is None
    assert diagnostics.model_probability_integer is None
    assert diagnostics.confidence_sequence is None
    assert diagnostics.stopping_status is MCStoppingStatus.UNAVAILABLE
    assert record.model_probability_raw is None
    assert record.probability_raw is None
    assert record.mc_confidence_sequence_lower is None


def test_rounding_endpoints_ties_and_platform_transform_are_separate() -> None:
    assert model_probability_integer(0.0) == 0
    assert model_probability_integer(0.004999) == 0
    assert model_probability_integer(0.005) == 1
    assert model_probability_integer(0.995) == 100
    assert model_probability_integer(1.0) == 100
    assert integer_bucket(0) == (0.0, 0.005, True, False)
    assert integer_bucket(100) == (0.995, 1.0, True, True)
    assert confidence_interval_model_integer(0.0, 0.004999) == 0
    assert confidence_interval_model_integer(0.0, 0.005) is None

    assert SPORTSPREDICT_SUBMISSION_POLICY.transform(0) == 1
    assert SPORTSPREDICT_SUBMISSION_POLICY.transform(100) == 99
    assert model_probability_integer(0.0) == 0
    assert model_probability_integer(1.0) == 100

    zero_estimate = _estimate(0, 5_000)
    zero_record = PropEstimateRecord.from_estimate(
        zero_estimate,
        adaptive_diagnostics=adaptive_prop_diagnostics(zero_estimate, terminal=False),
        platform_submission_policy=SPORTSPREDICT_SUBMISSION_POLICY,
    )
    one_estimate = _estimate(5_000, 5_000)
    one_record = PropEstimateRecord.from_estimate(
        one_estimate,
        adaptive_diagnostics=adaptive_prop_diagnostics(one_estimate, terminal=False),
        platform_submission_policy=SPORTSPREDICT_SUBMISSION_POLICY,
    )
    assert (zero_record.model_probability_integer, zero_record.platform_submission_integer) == (
        0,
        1,
    )
    assert (one_record.model_probability_integer, one_record.platform_submission_integer) == (
        100,
        99,
    )
    assert zero_record.submitted_integer is None
    assert one_record.submitted_integer is None
    assert zero_record.submission_rounding_policy_version is None
    assert one_record.submission_rounding_policy_version is None


@pytest.mark.parametrize("probability", [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
def test_empirical_simultaneous_coverage_over_all_checkpoints(probability: float) -> None:
    """Deterministically validate at least 99% simultaneous checkpoint coverage."""

    rng = np.random.default_rng(20260830 + round(probability * 1_000_000))
    replications = 20_000
    checkpoints = AdaptiveMCPolicy().checkpoints
    successes = np.zeros(replications, dtype=np.int64)
    crossed = np.zeros(replications, dtype=np.bool_)
    prior = 0
    for checkpoint in checkpoints:
        successes += rng.binomial(
            checkpoint - prior,
            probability,
            size=replications,
        )
        failures = checkpoint - successes
        log_martingale = (
            betaln(successes + 0.5, failures + 0.5)
            - betaln(0.5, 0.5)
            - successes * np.log(probability)
            - failures * np.log1p(-probability)
        )
        crossed |= log_martingale >= log(100.0)
        prior = checkpoint

    assert float(np.mean(crossed)) <= 0.01


def test_scalar_martingale_matches_independent_vectorized_formula() -> None:
    successes = 37
    trials = 1_000
    probability = 0.04
    independent = float(
        betaln(successes + 0.5, trials - successes + 0.5)
        - betaln(0.5, 0.5)
        - successes * np.log(probability)
        - (trials - successes) * np.log1p(-probability)
    )
    assert beta_binomial_mixture_log_martingale(
        successes,
        trials,
        probability,
    ) == pytest.approx(independent, abs=3e-12)
