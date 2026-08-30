# Milestone 7 implementation contract

The frozen specification remains authoritative. This note records the core prop
and settlement boundary without replacing the ontology in specification section
E.

## Implemented

- Typed constructors cover the supported core match, set, game, break, ace,
  double-fault, comparison, and first-serve-win-percentage props. Related and
  compound events are evaluated pathwise on the same `SimulationBatch`.
- Settlement is a pure post-generation operation with an explicit versioned
  policy. Started-match retirement results use the advancing player; completed
  scopes settle; irreversible Yes conditions settle; incomplete mutable scopes
  void; and walkovers void every prop.
- Compound settlement uses strong three-valued logic. Yes, No, Void, and
  unresolved-policy cases remain distinct. Unsupported prop kinds and policy
  mismatches fail explicitly.
- `probability_raw` is `Yes / settled paths`; `probability_settled` is `settled
  paths / all paths`; Monte Carlo standard error uses the settled denominator.
- First-serve-win-percentage outcomes compare the plausible whole-number
  truncation and half-up rounding conventions. Convention-sensitive paths remain
  unresolved rather than being guessed, as required by specification section K.

## Verification and boundary

Tests cover pathwise compounds, retirement and walkover behavior, monotone
thresholds, settlement denominators, policy-version mismatch, and unresolved
official rounding. Market-text parsing, question-specific SportsPredict
overrides, unsupported official-stat models, and lock persistence are not
implemented here and remain later explicit work.

