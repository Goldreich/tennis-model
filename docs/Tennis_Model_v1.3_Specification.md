# Tennis Model v1.3 Specification

## Status and inheritance

Tennis Model v1.3 is the active production framework. It inherits Tennis
Model v1.2 unchanged except for the standalone match-winner forecast defined
below. The framework identifier is `v1.3`.

A v1.3 model snapshot must immutably identify the v1.2 snapshot it inherited
as `base_snapshot_id` and record the hash of `config/model_v1_3.yaml`.

## Probability-affecting change from v1.2

For a standalone `MATCH_WIN(player)` prop, the effective v1.3 probability is
the no-vig probability derived from the latest available, cutoff-safe,
two-sided Pinnacle match-winner quote:

```text
raw_a = 1 / decimal_odds_a
raw_b = 1 / decimal_odds_b
p_a = raw_a / (raw_a + raw_b)
p_b = raw_b / (raw_a + raw_b)
```

The policy identifier is `pinnacle-two-way-multiplicative-no-vig/v1`.
American prices must be converted to decimal before entering the immutable
snapshot; the retained raw-source payload hash preserves the original evidence.

This is a targeted forecast override. It does not alter serve components,
surface Elo, game-day fitness, match-performance draws, score paths, duration,
aces, double faults, first-serve statistics, winners, unforced errors,
tiebreaks, deciding sets, lopsided sets, or retirement generation.

The intrinsic simulated match-winner probability remains stored for audit and
for interpreting the simulated joint distribution. It is not the effective
answer to the standalone v1.3 match-winner prop. Exact-score probabilities and
compound propositions remain path-derived and are not rescaled to the market
winner marginal. Consequently, users must not infer that their sums reproduce
the standalone market-derived winner probability.

## Quote selection and failure policy

Each production lock requires an immutable
`pinnacle-moneyline-snapshot/v1` artifact containing explicit player IDs,
decimal prices, observation time, source identity, source URL, and raw-payload
SHA-256.

The selected quote must:

- identify Pinnacle and the two scheduled participants;
- be a two-sided match-winner price;
- have been observed no later than the lock information cutoff;
- have been observed strictly before the scheduled start; and
- be the latest eligible observation in the supplied immutable snapshot.

The snapshot itself must have been captured no later than the information
cutoff. Conflicting prices at the same latest timestamp are invalid. Missing,
one-sided, malformed, participant-mismatched, post-cutoff, or post-start quotes
block lock creation. There is no fallback to the simulated winner probability
and no silent substitution of another bookmaker.

The selection policy identifier is
`latest-cutoff-safe-pinnacle-moneyline/v1`.

## Lock and reporting contract

Production v1.3 uses `prediction-lock/v5`. The lock retains:

- the complete v1.2 simulation and all Monte Carlo path counts;
- the immutable Pinnacle snapshot as a retained artifact;
- the selected quote and both raw implied probabilities;
- the overround and both normalized no-vig probabilities;
- source and capture timestamps and hashes; and
- both the effective market forecast and intrinsic simulation forecast in
  exported reports.

Market-derived winner probabilities have zero Monte Carlo error. Monte Carlo
diagnostics attached to the intrinsic `MATCH_WIN` path estimate remain
simulation diagnostics and must not be presented as uncertainty around the
Pinnacle probability.

Newer eligible odds create new forecast information and therefore require a
new immutable lock revision or a new lock under the operational workflow.
Existing locks are never overwritten.

## Submission rounding

The effective no-vig probability is the archived point estimate. When the
external platform requires an integer percentage, v1.3 applies nearest-percent
rounding and clamps only the submitted integer to the platform range 1 through
99. The point estimate itself is never clipped or rounded.

The submission policy identifier is
`pinnacle-no-vig-nearest-percent-clamp-1-99/v1`.

## Frozen inheritance

Every v1.2 rule not explicitly changed above remains frozen. In particular,
Pinnacle is not an input to the joint simulation and no other market, crowd,
ranking, head-to-head, momentum, or clutch feature is activated. Any expansion
of market use to exact scores, compound props, simulation paths, or any other
forecast requires a later framework version.
