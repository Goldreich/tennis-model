# Tennis Model v1.2 Fitness Candidate

**Status:** promoted, unchanged, as the fitness layer of production v1.1
**Authorized:** 2 September 2026
**Historical framework identifier:** `Tennis Model v1.2-fitness-candidate`

The production identity and locked deployment contract are now specified in
`docs/Tennis_Model_v1.1_Production_Specification.md`. This document remains the
historical fit design and is not the active version authority.

## Purpose

This candidate adds a temporary, match-specific fitness adjustment to the
surface Elo strength anchor. It does not change a player's persistent Elo and
does not modify frozen v1.0 serve primitives, scoring, duration, settlement, or
retirement behavior.

For player `i` in match `m`:

```text
game_day_elo[i,m] = base_surface_elo[i,m] + fitness_adjustment[i,m]
```

Every fitness feature is nonnegative and every fitted coefficient is constrained
to be nonpositive. A normally rested player therefore receives no penalty, and a
fitness signal cannot increase the player's individual Elo above the underlying
base rating.

## Initial feature set

The first candidate deliberately has only three fitted weights:

1. `recent_workload`: recorded main-tour match minutes over the preceding 21
   days, with a fixed mixture of 3-day and 14-day exponential decay, expressed
   in units of 300 weighted minutes.
2. `short_recovery`: a fixed penalty basis for less than two proxy days since
   the previous recorded main-tour match.
3. `return_from_layoff`: the excess over a 42-day main-tour absence, transformed
   by `log1p(excess_days / 30)` and decayed by matches since return with a
   three-match exponential scale.

Missing recorded duration inside the workload window makes workload unavailable;
it is never replaced by zero. Absence of any eligible main-tour match is a
structural zero only within the explicitly agreed main-tour-only data scope.

## Date limitation and fixed proxy

The source records event start dates rather than exact match timestamps. The fit
therefore uses a predeclared round-to-day proxy. Grand Slam main-draw rounds use
offsets `0, 2, 4, 6, 8, 11, 13` from R128 through the final. Other main-draw
rounds use consecutive offsets. Qualifying rounds receive fixed negative
offsets. Round-robin rows are retained for Elo and exposure history but are not
eligible coefficient-estimation targets.

No exact rest-hour claim may be made from these data.

## Estimation

ATP and WTA weights are fitted separately. For completed match `m`, player order
is deterministic and independent of the result:

```text
logit(P(A wins)) = base_elo_logit + log(10) / 400 * (X_A - X_B) weights
```

Weights are estimated by constrained ridge logistic regression. The ridge
strength is selected on 2024-2025 matches using only earlier matches for the
coefficient fit. A zero-adjustment candidate is retained and wins unless a
nonzero candidate improves validation Brier by the frozen minimum margin.
Selected settings are refitted on all eligible pre-US-Open matches from 2019
through 29 August 2026.

Retirements, walkovers, defaults, abandonments, and invalid results are excluded.
The retirement model remains separate.

## Held-out test

All available completed 2026 US Open matches are excluded from fitting and model
selection. They are evaluated in round order. Earlier completed rounds may update
base Elo and workload state before later-round forecasts; no same-round or future
result enters a forecast.

The primary diagnostic compares unadjusted surface Elo with game-day Elo on the
same matches. Where an existing v1.0 probability is available, a secondary test
compares the existing 75% Elo-logit integration before and after the fitness
adjustment.

This is a one-event test and cannot by itself promote the candidate. Promotion
requires broader rolling-origin validation and explicit authorization.

## Invariants

- Persistent Elo updates are unchanged.
- Game-day adjustments are never written back to persistent Elo.
- Individual fitness adjustments cannot be positive.
- Current-US-Open outcomes never enter coefficient fitting or selection.
- Ace and double-fault primitive rates are untouched.
- Retirement risk remains a separate model.
- Market or crowd probabilities are never model inputs.
