# Retrospective-finalized historical validation

Assessment date: 2026-08-30  
Policy: `historical-validation-data-policy/v1`  
Mode: `RETROSPECTIVE_FINALIZED`

## Result

The mode and exact-date augmentation path are implemented, but the genuine rolling-origin
backtest was **not run**. The finalized 2017–2025 crosswalk resolves 41,041 of 47,854
Sackmann-style match rows (85.763%), leaving 6,813 residual rows. That is insufficient to
assert complete B6 retirement history or the C6 last-started-match date. Treating the residuals
as nonmatches, nonretirements, or zero-valued observations would violate the frozen
specification.

The retained target-source assessment also has no general 2017–2025 corpus of genuinely
pre-result schedule/order-of-play records. The backtest interface now requires that provenance
for every retrospective target, but the exact-date history gate fails before target revelation
in any event.

No model was fitted, no target outcome was revealed, and no Tennis Model v1.0 methodology was
changed.

## Mode boundary

`POINT_IN_TIME_VINTAGE` remains the default. It still requires the original row/source
availability to be strictly before the forecast cutoff.

`RETROSPECTIVE_FINALIZED` is explicit and noninterchangeable. It may use a later-retrieved
finalized or corrected record only when a uniquely crosswalked exact calendar date proves the
match was complete before the historical cutoff. Because the augmentation is date-only, a match
on the cutoff date is excluded for the entire date. Missing, inconsistent, ambiguous, reused,
or structurally conflicting dates fail closed. Original source publication/retrieval timestamps
remain in provenance; they are not rewritten to historical dates.

Future and same-day matches are excluded before their probability-affecting statistics are
normalized. Historical targets must come from a pinned schedule/order-of-play record available
strictly before their forecast cutoff. Outcomes remain behind the existing lock-then-reveal
interface and cannot be supplied to cohort construction.

## Exact-date crosswalk

The deterministic `sackmann-tennis-data-exact-date/v1` procedure partitions by tour/year,
preserves winner/loser direction, normalizes names without fuzzy matching, constrains dates to
the event window, checks surface/best-of/round/set-score structure, and requires a global
one-to-one match. It imports no odds or market fields.

| Tour | Year | Source rows | Matched | Residual | Coverage |
|---|---:|---:|---:|---:|---:|
| ATP | 2017 | 2,911 | 2,484 | 427 | 85.332% |
| ATP | 2018 | 2,897 | 2,561 | 336 | 88.402% |
| ATP | 2019 | 2,806 | 2,543 | 263 | 90.627% |
| ATP | 2020 | 1,462 | 1,234 | 228 | 84.405% |
| ATP | 2021 | 2,733 | 2,364 | 369 | 86.498% |
| ATP | 2022 | 2,917 | 2,554 | 363 | 87.556% |
| ATP | 2023 | 2,986 | 2,623 | 363 | 87.843% |
| ATP | 2024 | 3,076 | 2,628 | 448 | 85.436% |
| ATP | 2025 | 2,944 | 2,524 | 420 | 85.734% |
| **ATP** |  | **24,732** | **21,515** | **3,217** | **86.993%** |
| WTA | 2017 | 2,862 | 2,247 | 615 | 78.512% |
| WTA | 2018 | 2,756 | 2,229 | 527 | 80.878% |
| WTA | 2019 | 2,743 | 2,275 | 468 | 82.938% |
| WTA | 2020 | 1,276 | 954 | 322 | 74.765% |
| WTA | 2021 | 2,597 | 2,291 | 306 | 88.217% |
| WTA | 2022 | 2,594 | 2,264 | 330 | 87.278% |
| WTA | 2023 | 2,810 | 2,391 | 419 | 85.089% |
| WTA | 2024 | 2,689 | 2,429 | 260 | 90.331% |
| WTA | 2025 | 2,795 | 2,446 | 349 | 87.513% |
| **WTA** |  | **23,122** | **19,526** | **3,596** | **84.448%** |
| **Total** |  | **47,854** | **41,041** | **6,813** | **85.763%** |

Residual status counts are 6,585 unmatched, 220 structural conflicts, 4 ambiguous, and 4
duplicate-source matches. Residuals are not confined to irrelevant edge cases: they include
3,541 Davis/Billie-Jean-King-Cup-level rows and 1,823 ordinary `A`/`I`-level rows, among other
levels. This prevents a truthful completeness assertion for rolling B6/C6 inputs.

## Pins and generated artifacts

The source-by-source Sackmann SHA-256, Tennis-Data SHA-256, and crosswalk IDs are recorded in
`config/historical_validation_retrospective_finalized_v1.yaml`. The Sackmann objects are the
finalized files at the repository commits already pinned by `config/sources.yaml`; the
Tennis-Data objects are the 18 retained workbooks retrieved on 2026-08-30. Their late retrieval
and mutable publication history are stated provenance, not disguised point-in-time availability.

- Crosswalk-set SHA-256: `4f25376b139d5e4871cf7d9f304644431703179c7ec144e8cee9638a18548495`
- Aggregate manifest SHA-256: `35db7a63950447ebe65b69864b65a82f59d9715eee980ed57e31521526ffa331`
- Summary SHA-256: `d9ba4b4973c9976ae9a532c209703664ec72b182e17c30755500c59a06bf7b89`
- Residual-detail SHA-256: `df1733beddc6796a82186eb2fa381d3b09027546489afb1f85f56479b9d81eb9`

The complete per-year crosswalks, manifests, summary, and residual detail are generated under
`data/processed/retrospective-finalized-crosswalk-v1/`, which is intentionally ignored as a
local immutable artifact tree. They are reproducible with
`scripts/assess_retrospective_crosswalk.py`; the script verifies every input hash before use.

## Backtest readiness

Blocked by data completeness, not by model code. A genuine run may begin only after the
crosswalk set (or an independently pinned exact-date source) resolves the residual history
needed for B6/C6 and a pinned pre-result target schedule/order-of-play corpus is available.
The new gate prevents an incomplete retrospective cohort from reaching fitting, simulation, or
outcome reveal.
