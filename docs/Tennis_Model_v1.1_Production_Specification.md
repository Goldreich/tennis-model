# Tennis Model v1.1 Production Specification

**Status:** production
**Promotion date:** 2 September 2026
**Archived fallback:** frozen Tennis Model v1.0

## 1. Production decision

Tennis Model v1.1 is the default framework for every new prediction. It retains
the complete frozen v1.0 joint simulator and adds two approved strength layers:

1. a persistent surface-aware Elo anchor;
2. a temporary minute-based game-day fitness adjustment to that Elo anchor.

Frozen v1.0 remains available under its original specification and configuration
for documentation, replay, and rollback. Existing v1.0 locks are immutable and
must never be relabeled as v1.1.

## 2. Preserved v1.0 model

The following remain unchanged:

- primitive serve components `F`, `A`, `Q1`, `D`, and `Q2`;
- serve-event causal ordering;
- component fitting and uncertainty;
- exact point, game, set, and match simulation;
- ace and double-fault generation;
- duration and retirement models;
- settlement semantics;
- joint-path prop derivation;
- fixed 100,000-path production simulations and 12-worker default.

## 3. Persistent surface Elo

ATP and WTA are rated independently from completed main-tour singles matches.
Walkovers, retirements, defaults, abandonments, byes, and unsupported levels are
excluded. Each player has a global rating and a surface rating, initialized at
1500. The production settings are:

```text
K = 16
effective Elo = 0.50 global Elo + 0.50 current-surface Elo
P(A wins) = 1 / (1 + 10 ^ ((Elo_B - Elo_A) / 400))
```

The Elo anchor receives 75% of the integrated strength logit and the unchanged
v1.0 component model receives 25%. The target is implemented through symmetric
Q1/Q2 tilting inside the joint simulator. Elo never replaces a simulated match
winner after the fact.

## 4. Game-day fitness Elo

Persistent Elo estimates normal underlying ability. For player `i` in match `m`:

```text
game_day_elo[i,m] = base_surface_elo[i,m] + fitness_adjustment[i,m]
```

The adjustment uses three nonnegative features. Fitted weights are constrained
to be nonpositive, so an individual fitness assessment can never raise Elo above
the player's persistent rating.

### 4.1 Features

`recent_workload` is recorded main-tour match minutes over the preceding 21 days.
Minutes receive a fixed mixture of 70% three-day and 30% fourteen-day exponential
decay and are expressed in units of 300 weighted minutes.

`short_recovery` is positive only when fewer than two date-proxy days separate
the forecast and the player's previous recorded main-tour match.

`return_from_layoff` begins after a 42-day main-tour absence:

```text
log1p((absence_days - 42) / 30)
```

After return, this quantity decays exponentially with a three-match scale.

Historical sources contain event dates rather than exact match timestamps. The
fixed round-date mapping from the promoted fit remains authoritative. It must not
be represented as exact rest hours.

### 4.2 Locked weights

| Tour | Recent workload | Short recovery | Return from layoff |
|---|---:|---:|---:|
| ATP | -12.347905 | -13.228820 | -63.333417 |
| WTA | 0.000000 | 0.000000 | -65.190815 |

Weights are Elo points per feature unit. The immutable fit artifact and its
source hashes must be retained with every production lock.

If any recorded match inside the 21-day workload window lacks duration, the
minute-based assessment is `unavailable`. The operational fallback is exactly
zero temporary adjustment with an explicit unavailable status. This is not an
assertion of zero workload and must not be reported as one.

## 5. Strength integration

The game-day Elo difference is converted to an anchor logit and integrated with
the component strength target using the locked 75/25 weights. The resulting
target is attained by symmetric Q1/Q2 point-strength tilting. `F`, `A`, and `D`
are not directly modified. Ace and double-fault match counts may change only
through simulated exposure.

## 6. Separate retirement and health mechanisms

Game-day Elo estimates conditional performance while the player remains in the
match. The existing retirement-risk model remains independent and unchanged.
Credible acute health information not represented by recorded workload remains a
timestamped scenario; it cannot be converted into an undocumented Elo penalty.

## 7. Reproducibility and locks

Every v1.1 lock records or retains:

- the persistent Elo artifact;
- the fixed strength-integration artifact;
- the game-day fitness fit artifact;
- both players' feature values, availability status, and Elo adjustment;
- source-file hashes used for the fitness assessment;
- model configuration, code, cutoff, seed, and simulation path count.

Game-day adjustments never update persistent Elo. New information creates a new
lock revision rather than mutating an existing lock.

## 8. Prohibited inputs

Bookmaker odds, SportsPredict forecasts, crowd probabilities, prediction-market
prices, ranking, head-to-head, momentum, and clutch effects are not production
v1.1 inputs. The superseded dynamic-anchor, ranking-prior, trajectory, and H2H
candidates remain disabled.

## 9. Promotion evidence and limitation

The surface-Elo integration passed the accepted historical evaluation. The
fitness layer improved the held-out existing-v1.0 integration cohort while the
standalone Elo comparison was inconclusive and slightly worse. The user has
explicitly authorized production promotion with this limitation understood.
