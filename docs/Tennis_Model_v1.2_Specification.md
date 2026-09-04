# Tennis Model v1.2 Specification

## Status and inheritance

Tennis Model v1.2 is the active production framework. It inherits the complete
frozen Tennis Model v1.1 framework, including surface Elo, game-day Elo fitness,
serve-component estimation, score simulation, duration generation, settlement,
and reproducibility requirements. Unless this document explicitly changes a
behavior, the v1.1 specification remains authoritative.

The framework identifier is `v1.2`. A v1.2 model snapshot must retain the
immutable v1.1 snapshot it inherited from as `base_snapshot_id` and must record
the hash of `config/model_v1_2.yaml`.

## Probability-affecting change from v1.1

v1.2 adds the rally-termination auxiliary layer defined in
`docs/RALLY_TERMINATION_V1.md`. The layer predicts official winners and
unforced errors from each already-simulated match path. It does not modify the
primitive serve probabilities, Elo or game-day-fitness strength tilt, score
generation, duration generation, aces, double faults, match winner, or
tiebreak probabilities.

The active accounting convention is
`usopen-winners-include-aces-ue-include-double-faults/v1`. Official winners
include aces and official unforced errors include double faults. The fitted
residual model therefore subtracts those deterministic categories during
estimation and adds them back exactly once during simulation.

## Active posterior

The active artifact schema is `rally-termination-posterior/v2`. Its initial
state comes from `rally-termination-fit/v1`. Following each newly completed
eligible match, the complete sparse Gaussian player-effect posterior is updated
with both directional multinomial observations using
`sequential-laplace-multinomial-logit/v1`.

Updates may change only the four player-specific latent effects: aggression,
allows-winners, pressure, and error tendency. Tour intercepts, player-effect
prior scale, residual concentration, and every inherited v1.1 parameter remain
fixed. Prediction integrates the joint eight-effect posterior for the two
scheduled players. Unknown players use the tour baseline posterior.

Each posterior update is immutable, parent-linked, source-hashed, and strictly
ordered by information cutoff. Active pointer files select an immutable
posterior artifact; they do not mutate it.

## Supported derived props

The v1.2 layer supports player winners, winner comparisons, player unforced
errors, total unforced errors, and unforced-error comparisons. These outcomes
must be generated from the same joint match paths used for all other props.

First-serve-win percentage props compare the exact simulated ratio with the
platform threshold. No whole-percent rounding is applied.

## Frozen invariants

- All v1.0 and v1.1 causal serve, data-cutoff, settlement, RNG, and provenance
  invariants remain in force.
- Existing prop probabilities must be invariant to enabling the v1.2 layer.
- Rally generation must use its own explicit RNG stream and must not consume or
  reorder any inherited RNG stream.
- Market probabilities remain prohibited model inputs.
- Any later probability-affecting change requires a new framework version.
