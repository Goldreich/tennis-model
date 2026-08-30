# Milestone 4 implementation contract

The authoritative probability model remains
`Tennis_Model_v1.0_Specification.md`. This document records the implemented
point-generation boundary and reproducibility choices without redefining the
frozen serve components.

## Primitive point-generation boundary

- `ServePerformanceDraw` is one realized match-performance vector containing
  exactly `F`, `A`, `Q1`, `D`, and `Q2`. It is distinct from Milestone 3's
  posterior/predictive distribution and accepts no derived serve-win probability
  in place of those primitives.
- The unchanged Milestone 3 derived identities have one lightweight canonical
  implementation in `tennis_model.serve`; `tennis_model.estimation.derived`
  remains a compatibility re-export of the same objects.
- A service point follows the frozen causal order: first serve; then ace before a
  returnable first-serve outcome, or double fault before a playable second-serve
  outcome. Ace and double-fault branches terminate immediately.
- The immutable result records the causal branch, serve number, primitive support
  flags, server-won outcome, optional player identities, and eligibility for the
  future rally-stat layer. Construction rejects combinations that contradict the
  selected branch.
- Only non-ace returnable first serves and playable second serves are rally
  eligible. This milestone does not classify winners or unforced errors.
- Pure aggregation reconstructs all primitive denominators and official first-
  and second-serve point-win counts. The B1 accounting identities are enforced
  exactly, including on empty batches.

## RNG and coupled-uniform semantics

`generate_service_point` requires a caller-owned NumPy `Generator`. It consumes
uniforms lazily in causal order: terminal ace and double-fault branches consume
two draws, while returnable first-serve and playable second-serve branches consume
three. It never reads global random state or reseeds the supplied generator.

`generate_point_from_uniforms` is the deterministic coupled-testing interface.
Its immutable input supplies five independent uniforms in `[0, 1)`, one for each
primitive event. All five are predefined so the same stochastic inputs can be
reused under changed parameters; only uniforms on the realized causal branch are
treated as outcomes. Bernoulli events use the documented `u < p` rule.

Because changing a branch probability can change how many values a lazy generator
consumes, equal production seeds are not a coupled multi-point experiment. The
five-uniform interface is the point-level coupling contract; later match-level
monotonicity tests will need preallocated or counter-keyed per-point randomness.

## Integration boundaries

The point result exposes `server_won` and, when both player IDs are supplied, the
derived winner ID. The deterministic scoring engine can therefore consume the
winner without knowing component probabilities. Conversely, point generation
does not import or inspect game, set, tiebreak, break-point, or match state.

Milestone 4 does not draw posterior parameters or beta match performance, build
directional match parameters, simulate games or matches, classify rally
terminations, evaluate props, or create locks. Those remain later-milestone work.

## Ambiguities and blockers

There is no unresolved ambiguity in the requested causal point generator. Requiring
server and receiver IDs together (or omitting both), using half-open uniforms, and
defining lazy production consumption are non-probability-affecting interface and
reproducibility choices.

Production forecasts remain blocked on the data and inactivity-rule issues
recorded for Milestone 3; they do not block verification of this fixed-parameter
point generator.

## Next milestone boundary

The scalar point generator is ready to accept a realized directional performance
draw from Milestone 5. Milestone 5 must construct that draw with explicit
posterior and match-performance uncertainty while preserving the five primitive
semantics; none of that sampling is implemented here.
