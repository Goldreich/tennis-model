# Repository Agent Rules

These instructions apply to the entire repository. The authoritative source for
all probability-affecting behavior is
`docs/Tennis_Model_v1.0_Specification.md`; read it before model work. This file is
a guardrail, not a substitute for that specification.

## Authority and scope

- Implement the frozen Tennis Model v1.0 as specified. Do not redesign,
  reinterpret, or silently "improve" its architecture.
- Any probability-affecting methodological change requires an explicit framework
  version change. Never introduce one incidentally during implementation,
  refactoring, optimization, or bug fixing.
- Surface and document probability-affecting ambiguities rather than resolving
  them arbitrarily. Small organizational and non-probability-affecting engineering
  decisions may be made autonomously when they improve maintainability.
- Preserve completed milestone behavior unless a later milestone explicitly
  requires an interface change. Do not enter a later milestone unless the current
  task explicitly requests it.

## Frozen modeling invariants

- The primitive serve components are exactly `F`, `A`, `Q1`, `D`, and `Q2`, with
  the estimands and denominators defined in the specification.
- Preserve the causal serve ordering: first serve; ace conditional on first serve
  in and causing a server win; otherwise returnable first-serve outcome; after a
  missed first serve, double fault causing a server loss or a playable
  second-serve outcome.
- Derive related and compound props from the same joint simulated match paths.
- Frozen v1.0 uses conditionally independent match-performance draws for `F`,
  `A`, `Q1`, `D`, and `Q2`. Do not activate a dependence model without an explicit
  later-version decision.
- Do not use bookmaker odds, SportsPredict crowd forecasts, prediction-market
  probabilities, or similar market information as model inputs.
- Do not add Elo, ranking, head-to-head, momentum, clutch, or other unapproved
  features to fitted component models. Do not silently add hyperparameters or
  tune the model to one tournament.

## Data, state, and reproducibility

- Enforce strict information cutoffs. Future-data leakage is a correctness
  failure.
- Raw source snapshots are immutable. Preserve invalid or inconsistent inputs and
  quarantine them with explicit anomaly information; never silently clip, repair,
  replace, or impute them. Derived missing values remain missing, never zero.
- Every stochastic interface must use explicit RNG/seed handling; hidden global
  randomness is prohibited.
- Every numerical forecast must be reproducible from its data snapshot,
  model/config artifacts, code version, information cutoff, scenario, seed, and
  settlement-policy version.
- Match locks are immutable. New information creates a new lock revision; never
  overwrite an existing lock.
- Keep settlement semantics separate from event generation. Unsupported or
  unresolved settlement cases remain blocked rather than guessed.

## Engineering and milestone workflow

- Prefer simple, auditable implementations over unnecessary abstraction or
  optimization.
- Before declaring a milestone complete, run its required tests and the existing
  regression suite.
- Work milestone-by-milestone against the frozen specification. End each milestone
  with:

  1. implementation summary;
  2. tests and results;
  3. deviations from the specification;
  4. genuine ambiguities or blockers;
  5. principal files changed;
  6. readiness for the next milestone.
