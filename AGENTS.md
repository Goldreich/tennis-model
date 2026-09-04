# Repository Agent Rules

These instructions apply to the entire repository. The authoritative production
source for probability-affecting behavior is
`docs/Tennis_Model_v1.3_Specification.md`; read it before model work. Tennis Model
v1.3 inherits the frozen v1.2 framework documented by
`docs/Tennis_Model_v1.2_Specification.md`, which inherits the frozen v1.1 framework documented by
`docs/Tennis_Model_v1.1_Production_Specification.md`. The frozen v1.0
specification remains authoritative only for archived v1.0 replay and rollback.
This file is a guardrail, not a substitute for the versioned specifications.

## Authority and scope

- Implement Tennis Model v1.3 as the active production framework governed by
  `docs/Tennis_Model_v1.3_Specification.md`. It inherits frozen Tennis Model v1.2,
  governed by `docs/Tennis_Model_v1.2_Specification.md`, and Tennis Model v1.1,
  governed by `docs/Tennis_Model_v1.1_Production_Specification.md`. Frozen Tennis
  Model v1.0 remains unchanged as the documented archive and rollback
  implementation.
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
  v1.3's standalone `MATCH_WIN(player)` probability is the sole explicit
  exception; exact-score and compound props remain path-derived.
- Frozen v1.0 uses conditionally independent match-performance draws for `F`,
  `A`, `Q1`, `D`, and `Q2`. Do not activate a dependence model without an explicit
  later-version decision.
- Do not use bookmaker odds, SportsPredict crowd forecasts, prediction-market
  probabilities, or similar market information as model inputs except for the
  exact v1.3 standalone match-winner Pinnacle policy. No market input may alter
  joint simulation paths or any other prop.
- Frozen v1.0 component models remain unchanged. Frozen v1.1 adds only the
  approved surface-Elo strength anchor, fixed Q1/Q2 integration, and temporary
  minute-based game-day fitness adjustment. Frozen v1.2 adds only its
  specified winners/unforced-errors rally-termination layer and sequential
  Bayesian player-effect updates. Production v1.3 adds only the specified
  cutoff-safe Pinnacle no-vig standalone match-winner forecast. Ranking,
  head-to-head, momentum, clutch, and
  unapproved features remain disabled.

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
## Active production framework

- Tennis Model v1.1 is the frozen Elo and game-day-fitness framework.
- Tennis Model v1.2 is the frozen winners/unforced-errors framework.
- Tennis Model v1.3 is the active production successor governed by
  `docs/Tennis_Model_v1.3_Specification.md`. It inherits v1.2 unchanged and adds
  only the cutoff-safe Pinnacle no-vig probability for standalone match-winner
  props.
