# Tennis Model v1.1 Candidate Backtest Plan

**Plan status:** FROZEN BEFORE OUTER-FOLD MODEL COMPARISONS
**Freeze date:** 2026-09-01
**Plan version:** `tennis-v1.1-rolling-backtest/v1.0.1`
**Production consequence:** none. Tennis Model v1.0 remains the production default.

Any change to this document after comparative outer-fold results exist requires a new
plan version and a clean rerun. The runner stores this file's SHA-256 in every output
manifest.

## 1. Purpose and hypotheses

The primary comparison is frozen v1.0 against the implemented v1.1 candidate. The
intended improvement is lower match-winner Brier score and log loss, especially fewer
high-confidence strength inversions, without direct changes to `F`, `A`, or `D` and
without a material regression in ace- or double-fault-comparison Brier score.

The backtest is not tuned to the 2026 US Open, the supplied 53-market scorecard, or its
three motivating tail failures. That scorecard is a separate diagnostic only.

## 2. Source mode and its limitation

The immutable inputs are the 18 annual ATP/WTA CSV files for 2017-2025 pinned in
`data/backtest-v1.1-source/source_manifest.json`. Their bytes must match the SHA-256
values already recorded by the repository's historical source audit.

The source is `RETROSPECTIVE_FINALIZED_EVENT_EMBARGO`, not a fully reconstructed
point-in-time vintage. Sackmann's `tourney_date` is an event start date and the files do
not contain original result-publication or correction timestamps. To prevent result
leakage computationally:

- every forecast cutoff is 00:00 UTC on the first day of a calendar month;
- a historical event is available for fitting only when `event_start + 21 days` is
  strictly before that cutoff;
- all matches in a target month are forecast from the same pre-month snapshot;
- no result from the target event or target month can enter its forecast;
- finalized corrections cannot be proven to have existed at a historical cutoff.

Because the source has an event date but no scheduled-start timestamp, an event dated
exactly on the monthly cutoff is represented in the typed OOF chronology as
`cutoff + 1 microsecond`. This is an ordering sentinel only; the component and anchor
forecast remains the frozen month-start forecast and no additional information enters.

The missing vintage publication/correction timestamps are a predeclared promotion
limitation. Even favorable scores can produce at most `INCONCLUSIVE` unless a later
point-in-time-vintage audit independently closes this gap. They can still produce
`FAIL` if a statistical or coherence gate fails.

## 3. Eligibility

Tours are ATP and WTA. Surfaces are Hard, Clay, and Grass. Match types are completed
singles with best-of-three or best-of-five scoring. Eligible competition levels are:

- ATP: `A`, `F`, `G`, `M`;
- WTA: `35+H`, `50+H`, `F`, `G`, `I`, `P`, `PM`, `W`.

Davis/Billie Jean King Cup (`D`), Olympics (`O`), unsupported surfaces, missing or
invalid identities, invalid best-of values, walkovers, retirements, defaults,
abandonments, cancellations, and unparseable/non-completed scores are excluded.
Cancellation-only records cannot be observed in result-only files and are disclosed as
a source-cohort limitation.

Winner evaluation additionally requires a valid winner and loser. Protected ace/DF
evaluation requires both players' service-point, first-serve-in, first-serve-won,
second-serve-won, ace, and double-fault totals. Component accounting failures are
quarantined by component and an evaluation match with any required protected component
invalid is excluded from the common protected cohort. Missing ranking does not exclude
a match; ranking-prior candidates fall back to the no-ranking initialization for that
player.

## 4. Rolling folds and cutoffs

Outer folds are calendar years 2022, 2023, 2024, and 2025. Within each outer year,
forecasts are generated at monthly rolling cutoffs. The initial history begins in 2017,
and frozen v1.0 retains its three-year, one-year-half-life training rule.

For outer year `Y`, the inner validation years are `Y-2` and `Y-1`:

- candidate integration coefficients are first fitted on out-of-fold predictions from
  `Y-2` and hyperparameters are evaluated on `Y-1`;
- after selection, integration coefficients are refitted on eligible out-of-fold
  predictions from both inner years whose outcomes clear the outer-year cutoff;
- selected settings and coefficients are frozen for the full outer year;
- outer outcomes never affect settings for that outer fold.

No random train/test split is permitted. Models may expand through time only at a later
monthly cutoff and only with rows clearing the 21-day embargo.

## 5. Variants

Identical outer matches and settlement rules are used for:

1. frozen v1.0;
2. v1.0 with winner temperature only;
3. v1.0 blended on the logit scale with conventional surface Elo;
4. v1.0 blended with the dynamic anchor at a constant weight;
5. v1.0 with the fitted reliability-gated dynamic anchor;
6. complete v1.1 with dynamic level change, sparse ranking prior, and shrunk H2H;
7. complete v1.1 without the ranking prior;
8. complete v1.1 without dynamic level change;
9. complete v1.1 without H2H.

All non-v1.0 winner targets are implemented through the symmetric `Q1`/`Q2` point tilt
and exact-scoring root, not by replacing winner probabilities after simulation.

## 6. Frozen candidate grids

- Winner logit temperature multipliers: `0.80, 0.90, 1.00, 1.10, 1.20`.
- Surface Elo K: `16, 24, 32`; surface/global blend: `0.50, 0.75`.
- Elo/component and constant-anchor/component logit blends: `0.25, 0.50, 0.75`.
- Dynamic core decay days: `365, 730`.
- Dynamic core process SD: `0.03, 0.07`.
- Dynamic initial SD: `1.25`; surface SD: `0.35`; variance floor: `1e-6`.
- Complete ranking log-slope: `0.35`, prior SD `0.75`, only through the implemented
  sparse-player initialization.
- Complete H2H prior SD: `0.15`.
- No-level-change ablation: decay `1,000,000` days and process SD `1e-6`.
- Integration L2 penalties: `0.1, 1.0, 10.0`.
- Reliability prior logit: `-0.5`; maximum absolute tilt: `2.5`.
- Q weights: `Q1=1`, `Q2=1`; probability bound `1e-6`; root tolerance `1e-9`.

Component reliability diagnostics are fixed as follows: component variance is the mean
squared Q1/Q2 linear-predictor SD over both service directions; raw instability is their
maximum SD and the integration feature is its monotone bounded transform `x/(1+x)`;
sparsity is `exp(-min(prior matches of either player)/25)`.

**Pre-evaluation implementation erratum (2026-09-02):** plan v1 initially named the
raw maximum SD as the integration instability feature, but the implemented typed
interface requires that feature in `[0,1]`. The first inner-fold fit failed validation
before producing or viewing any outer result. Version 1.0.1 therefore predeclares the
monotone `x/(1+x)` normalization required by that interface. No candidate grid, outcome,
outer metric, or promotion gate was inspected or changed.

**Pre-evaluation chronology clarification (2026-09-02):** the first final-inner refit
encountered an event-date proxy equal to its month-start forecast cutoff. Before any
outer result was produced, version 1.0.1 fixed the strictly ordered typed representation
to `cutoff + 1 microsecond` for equality only. This is not an estimated match time and
does not change a model input, decay interval, outcome, or probability.

**Pre-evaluation serialization correction (2026-09-02):** the first completed outer
folds failed during atomic parquet publication because generic outer-metadata access
via `getattr(row, key)` resolved the `"round"` field to the `Series.round` method. No
outer predictions or metrics were published or viewed. Every copied source field now
uses explicit item access as `row[key]`, and all outer folds are rerun under the
corrected code. This is a probability-neutral serialization correction; no grid,
outcome, metric, or promotion gate changed.

## 7. Metrics and uncertainty

Primary winner metrics are mean Brier score and log loss. Secondary metrics are
calibration intercept/slope, ten-bin reliability, ROC AUC, high-confidence error rate
at 80%, worst-decile Brier, favorite/underdog performance, and slices by tour,
competition level, surface, sparse/well-observed status, anchor/component agreement,
and graph connectivity.

Paired uncertainty uses a 2,000-replicate percentile bootstrap over event blocks,
stratified by outer year and tour, with seed `11092026`. Multiple props from one match
are never treated as independent blocks.

Practical winner promotion thresholds for complete v1.1 versus v1.0 are:

- mean Brier improvement at least `0.002`, with the paired 95% interval upper bound
  below zero;
- mean log-loss improvement at least `0.005`, with the paired interval upper bound
  below zero;
- absolute calibration-intercept deterioration no more than `0.05` and absolute
  slope-distance-from-one deterioration no more than `0.10`;
- no increase above `0.005` in high-confidence error rate or worst-decile Brier;
- improvement must appear in both tours and at least two outer years, rather than only
  a handful of outliers.

## 8. Protected ace and double-fault families

Protected comparisons settle ties as No. The primary non-inferiority margin is an
increase of `0.001` in mean Brier for each family. Complete-v1.1-minus-v1.0 uses paired
event-block intervals; each interval's upper bound must be no greater than `0.001`.

The protected audit also reports calibration, expected-count residuals, dispersion,
zero mass, tie rates, primitive exposure-adjusted rates, expected match-length strata,
and strength-disagreement strata. Direct `F/A/D` draw equality is checked separately
from count changes caused by service-point exposure.

The protected cohort is a deterministic SHA-256 sample of up to 192 eligible matches
per tour/outer-year, selected without outcomes. The common-random-number simulation
starts at 256 paths and escalates through `1024` and `4096` until the aggregate paired
Monte Carlo SE for each protected-family Brier difference is at most `0.00035`, or the
maximum is reached. Reaching the maximum above the target is reported as inconclusive.

## 9. Coherence and numerical checks

The same protected paths report exact score, sets, games, hold/break rates, tiebreaks,
service-point exposure, and a duration-exposure proxy. Duration calibration and
retirement-aware historical settlement are not promotion-eligible from this finalized
source; synthetic repository tests cover their semantics and the limitation is explicit.

Required numerical checks are player-order antisymmetry, monotonic response to positive
tilt, root residual at or below `1e-9` unless saturated, rare/reported saturation,
bitwise-identical coupled `F/A/D` draws, exposure reconciliation, integrated winner
probability equal to exact-score mass, deterministic replay, and no market inputs.

## 10. Recommendation rule

`PASS` requires every winner, calibration, tail, protected-prop, coherence,
reproducibility, and provenance gate. `FAIL` means at least one substantive statistical
or coherence gate fails. `INCONCLUSIVE` means no substantive gate fails but uncertainty,
Monte Carlo precision, protected coverage, or point-in-time provenance is insufficient.

Because the present source lacks historical publication/correction vintages, this run
is predeclared as unable to return `PASS`. Frozen v1.0 remains production after either
`FAIL` or `INCONCLUSIVE`, and no production default may be changed without explicit user
authorization.
