# Milestone 6 implementation contract

The frozen specification remains authoritative. This note records the joint
match-simulation boundary and does not define new tennis or probability rules.

## Implemented

- `simulate_matches` accepts a verified Milestone 5 `MatchParameterDistribution`
  and an explicit integer or NumPy `SeedSequence`. Each path samples posterior
  uncertainty and one fixed `F/A/Q1/D/Q2` performance vector per serving
  direction, then uses the reserved point-path RNG for the entire match.
- Known first servers are accepted explicitly; otherwise each path uses the
  frozen fair pre-toss draw. Standard best-of-three/five scoring, service order,
  tiebreaks, and termination delegate to the pure Milestone 2 state machine.
- Each path retains exact set scores, official game totals, tiebreak sets,
  chronological breaks, break-point opportunities, service games and holds, and
  all primitive serve sufficient statistics. Summary paths omit points;
  `trace_level="points"` retains the immutable point trace.
- Path invariants cross-check winners, service order, total games, holds, breaks,
  tiebreak games, serve identities, player identities, and optional trace length.
  Batch provenance records the root seed, PCG64, trace level, snapshot ID, and
  match-parameter provenance.

## Verification and boundary

Deterministic extreme-probability tests verify exact breaks and game locations,
hold/break accounting, first-serve denominators, and point traces. An integration
test proves the public simulator consumes the real Milestone 5 distribution,
replays exactly, and does not mutate the caller's `SeedSequence`.

The simulator currently generates completed core serve/scoring paths. No
retirement hazard, rally winner/error, or duration distribution is activated;
those require their own fitted or explicitly configured artifacts and must not be
invented inside the core serve simulator.

