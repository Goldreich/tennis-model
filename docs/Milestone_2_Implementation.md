# Milestone 2 implementation contract

The authoritative rules remain in `Tennis_Model_v1.0_Specification.md`. This
document records only the implemented boundary and engineering decisions.

## Deterministic scoring boundary

- The scoring engine is a pure fold over an explicit stream of point-winner
  indices. It contains no probabilities, RNG, fitted parameters, data access,
  market access, or live research.
- Frozen immutable states represent an ongoing regular game or tiebreak, completed
  legal sets, and best-of-three/five match progress. Redundant winners and service
  order are derived from the score history rather than independently stored.
- Regular games use advantage scoring. Servers alternate after each regular game,
  independent of the game winner, and the order carries continuously across sets.
- At 6-6, non-deciding sets use a 7-point win-by-two tiebreak and the numeric
  deciding set uses a 10-point win-by-two tiebreak. Service follows the 1-2-2
  sequence. The receiver of the first tiebreak point serves first in the next set.
- Break-point opportunities are identified from the pre-point regular-game state.
  A break is recorded only when the receiver wins a regular service game.
  Tiebreak return points are never breaks.
- A completed tiebreak adds one official game, producing 7-6; its internal points
  add no games. No points can be applied after a legal match terminal state.
- Each point transition reports auditable server, receiver, phase, break-point,
  break, game, tiebreak, set, and match-completion facts without mutating its input.

## Settlement foundation boundary

- Settlement is a separate dependency-free module. `Settled(False)`, `Voided`,
  and `Blocked` are distinct types; void or unsupported paths cannot silently
  become No outcomes.
- Three-valued event truth supports deterministic AND/OR composition, and typed
  threshold comparisons preserve strict ties and half-point boundaries.
- This milestone does not implement market parsing, stochastic prop evaluation,
  retirement hazards, walkover handling, Monte Carlo aggregation, or the complete
  settlement ontology. Those later layers will consume scoring facts.

## Ambiguities and blockers

There is no unresolved ambiguity in the requested core scoring rules. The frozen
specification explicitly leaves some retirement settlement cases, official
rounding, and winner/error accounting unresolved. The typed `Blocked` result is
the foundation for refusing those cases; this milestone does not guess them.

## Milestone status

Milestone 2 remains complete and regression-covered. Milestone 3 was subsequently
implemented without changing this deterministic scoring or settlement boundary.
The next authorized work is Milestone 4, which may consume the primitive
component distributions but must preserve this scoring engine unchanged.
