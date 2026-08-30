# Milestone 5 implementation contract

The authoritative probability model remains
`Tennis_Model_v1.0_Specification.md`. This document records the implemented
match-parameter boundary, uncertainty separation, and reproducibility choices
without redefining the frozen serve components.

## Snapshot and cutoff boundary

- `ModelSnapshot` identifies exactly one coherent persisted `F`, `A`, `Q1`, `D`,
  and `Q2` fit bundle by framework/implementation version, tour, fit timestamp,
  data cutoff, component artifact IDs, source-data and count-artifact hashes,
  configuration hash, and fit-code commit. Creation and loading verify the
  referenced content-addressed artifacts; neither path fits a model or reads raw
  or current data.
- A snapshot data cutoff and fit timestamp may precede a match information
  cutoff, but neither may follow it. `estimate_match` rejects a wrong-tour,
  future-data, or not-yet-fitted snapshot and records the relevant timestamps.
- `MatchContext` separates fitted inputs from lock/reporting metadata. Frozen-v1.0
  hard surface, configured indoor/event-year terms, and only already-approved
  active fit terms enter component predictors. Round, arbitrary named conditions,
  and the scenario identifier are retained as metadata and cannot silently shift
  primitive means.
- Stable JSON records retain the snapshot, exact artifact references, both
  serving-direction MAP summaries, dependence mode, context, scenario, hashes,
  the fit commit, and the match-parameter implementation version. Posterior
  matrices remain in referenced fit artifacts, and live RNG objects are never
  serialized. Reconstruction reloads and verifies the artifacts and then
  requires an exact record match.

Timestamps are normalized to UTC before hashing. Snapshot identity excludes
artifact directory locations, so relocating identical verified artifacts does
not create a different statistical snapshot.

## Two-stage uncertainty architecture

For each simulated match path, `sample_matchup_parameters` first draws the full,
ordered Laplace parameter vector for each fitted primitive component. A full
stored covariance is sampled by Cholesky factorization; the diagonal path is used
only when the Milestone 3 artifact explicitly records its large-fit diagonal
curvature approximation. Stable parameter names remain attached to every draw.
The sampled `log_kappa` is transformed with that same draw rather than replaced
by its MAP value.

One component parameter draw is evaluated for both serving directions. This
preserves their common fitted-component parameter uncertainty while applying the
players in their distinct server and receiver roles. Zero-centered effects for
an unseen player, surface level, or supplied event-year level are realized as
named latent Gaussian draws and shared whenever both directions reference the
same fitted effect.

`sample_serve_performance` then applies the separate beta predictive layer to one
fixed directional mean vector:

\[
\phi_k \sim \operatorname{Beta}\{\kappa_k\mu_k,
\kappa_k(1-\mu_k)\}.
\]

The full `sample_match_performance` API performs both stages for both directions.
The five component-model posterior draws are independent across `F`, `A`, `Q1`,
`D`, and `Q2`; their beta performance draws are conditionally independent as
well. The two directions receive separate beta draws. Frozen v1.0 enforces
`PerformanceDependenceSpec(mode="independent", loadings=(),
validation_artifact_id=None)` and rejects activation of a dependence model.

This gives three explicit layers:

```text
stored fitted posterior
    -> parameter draw and directional matchup means       (Milestone 5)
    -> one fixed ServePerformanceDraw per direction/path  (Milestone 5)
    -> repeated point outcomes from that fixed draw        (Milestone 4; path use in Milestone 6)
```

The inverse link and beta sampler require finite, strictly interior means,
positive finite concentration, and positive finite shapes. Invalid inputs and
shapes raise. If NumPy rounds a mathematically interior beta variate to exactly
zero or one for an extreme valid shape, the sampler moves only that represented
endpoint to the adjacent interior float with `nextafter`; ordinary draws are not
clipped.

## Directional matchup construction

`estimate_match(snapshot, context)` constructs `A serving against B` and `B
serving against A` separately from the same verified five-artifact snapshot.
`F` and `D` remain server-led. `A`, `Q1`, and `Q2` apply the fitted server effect
and subtractive returner effect. Surface deviations and configured active context
terms enter once. No new predictor, narrative adjustment, ranking, Elo,
head-to-head, or refit path is introduced.

The sampled output is the exact immutable Milestone 4 `ServePerformanceDraw`
containing only `F`, `A`, `Q1`, `D`, and `Q2`. Derived `w1`, `w2`, overall service
win probability, ace rate, and double-fault rate remain reporting identities, not
additional stochastic primitives. The caller samples this object once per
serving direction and match path and reuses it for every corresponding service
point.

Milestone 3 gained one public explicit-parameter projection function and a public
five-fit identity validator. Its existing MAP prediction path now delegates to
the same linear-term evaluator; regression tests verify unchanged MAP behavior.

## Reproducibility and numerical validation

A caller-supplied NumPy `SeedSequence` is copied, not advanced, and split into
independent child streams for the component-parameter stage, Player A
performance, Player B performance, and the later point path. The stable seed
record includes entropy, spawn key, pool size, child counter, and the explicitly
fixed `PCG64` bit-generator choice. Repeating snapshot + context + seed reproduces the exact
two primitive vectors; consuming one direction's stream cannot advance the
other, and no global NumPy random state is used.

Controlled tests recover a synthetic Laplace covariance with diagonal entries
`0.04`, `0.09` and covariance `0.03` to absolute tolerance `0.004`, verify exact
Cholesky replay and stable parameter indexing, and exercise the explicit
Milestone 3 diagonal fallback.

For `mu = 0.65`, `kappa = 80`, and 25,000 draws using seed `5101`, the empirical
mean was `0.649325782180` and empirical variance was `0.002834628084`, versus the
target

\[
0.65(1-0.65)/(80+1)=0.002808641975.
\]

At the same mean, seeded 10,000-draw variances were `0.037804793069` for
`kappa = 5` and `0.000461417362` for `kappa = 500`, a width ratio of about
`81.9`. Tests separately exercise parameter-only, performance-only, and full
two-stage variation. A fixed performance draw also feeds 400 varying Milestone 4
points without mutation.

## Synthetic historical-cutoff demonstration

No audited production ATP or WTA snapshot is checked into the repository, so a
real-player historical forecast would be a false provenance claim. The
reproducible demonstration therefore uses the synthetic ATP chronology from the
tests: pre-cutoff observations only, data cutoff `2026-01-01T00:00:00Z`, and a
fit timestamp and match-information cutoff of `2026-01-02T00:00:00Z`, and a
`p0` versus `p1` match scheduled for `2026-01-11T00:00:00Z` at Synthetic Open.

The conditional MAP means and a 4,096-parameter-draw posterior-integrated estimate
(seed `2026082905`; beta expectation conditional on each parameter draw) are:

| Direction and summary | F | A | Q1 | D | Q2 |
|---|---:|---:|---:|---:|---:|
| p0 serving p1 — conditional MAP | 0.805894 | 0.441566 | 0.935366 | 0.231844 | 0.882172 |
| p0 serving p1 — posterior-integrated MC | 0.818119 | 0.487183 | 0.925518 | 0.246399 | 0.869754 |
| p1 serving p0 — conditional MAP | 0.325788 | 0.026587 | 0.263664 | 0.049045 | 0.128879 |
| p1 serving p0 — posterior-integrated MC | 0.316113 | 0.035020 | 0.258393 | 0.052116 | 0.146112 |

Three full two-stage seeded realizations give:

| Seed | Direction | F | A | Q1 | D | Q2 | w1 | w2 | p_srv | ace rate | DF rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260829 | p0 -> p1 | 0.738639 | 0.538344 | 0.922614 | 0.241484 | 0.897050 | 0.964274 | 0.680427 | 0.890088 | 0.397642 | 0.063114 |
| 20260829 | p1 -> p0 | 0.270676 | 0.029864 | 0.370898 | 0.043889 | 0.076992 | 0.389685 | 0.073613 | 0.159166 | 0.008083 | 0.032009 |
| 20260830 | p0 -> p1 | 0.205182 | 0.650041 | 0.875479 | 0.101385 | 0.814466 | 0.956423 | 0.731891 | 0.777961 | 0.133376 | 0.080583 |
| 20260830 | p1 -> p0 | 0.361692 | 0.064812 | 0.290592 | 0.168052 | 0.097395 | 0.336570 | 0.081027 | 0.173455 | 0.023442 | 0.107269 |
| 20260831 | p0 -> p1 | 0.914998 | 0.209166 | 0.946545 | 0.286660 | 0.897198 | 0.957726 | 0.640008 | 0.930719 | 0.191386 | 0.024367 |
| 20260831 | p1 -> p0 | 0.346606 | 0.044793 | 0.302843 | 0.028157 | 0.106084 | 0.334071 | 0.103097 | 0.183154 | 0.015525 | 0.018397 |

The post-cutoff result is not an input to snapshot creation, artifact loading,
matchup construction, or sampling. The existing Milestone 3 leakage regression
also changes held-out post-cutoff component results from zero successes to all
successes and obtains identical fits and predictions. This demonstrates leakage
exclusion and reproducibility only; it is not an accuracy claim.

## Ambiguities and blockers

Milestone 5 introduces no new unresolved probability-affecting choice. Two prior
production blockers remain:

1. no independently audited immutable ATP/WTA production snapshot is available,
   so only synthetic fixtures are claimed here;
2. the specification requires uncertainty inflation and hard-deviation mean
   reversion after more than 90 days of inactivity but does not define their
   functional form or scale. No adjustment was invented.

These do not block the typed two-stage sampler or its synthetic verification, but
they continue to block a production forecast claim.

## Next milestone boundary

The repository is structurally ready for **Milestone 6 — Joint Match Simulation
and Core Prop Engine** once the Milestone 5 tests and full regression suite pass.
Milestone 6 must sample one directional performance vector per path, reuse it
throughout that path, and reserve the established point-path child seed for point
outcomes. This milestone does not implement full games/sets/matches,
first-server orchestration, props, rally statistics, retirement, settlement
aggregation, or locks, and none should begin without an explicit request.
