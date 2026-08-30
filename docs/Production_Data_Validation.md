# Production Data and Historical Validation Attempt

Date: 2026-08-30  
Framework: frozen Tennis Model v1.0 plus normative B6/C6 amendment

## Pinned sources

`config/sources.yaml` is the authoritative strict registry. It pins nine yearly
objects per tour for 2017–2025:

- ATP: `Kadantte/tennis_atp` commit
  `712be0c5ade693cdab9e69c23a71a0edf5a23c44`;
- WTA: `VictorSquidWei/tennis_wta` commit
  `85ef6efaa04ca860ed592a935498dcda9460ea91`;
- upstream attribution: Jeff Sackmann / Tennis Abstract;
- license: CC BY-NC-SA 4.0;
- manifest SHA-256:
  `5b11845fd11bfc80ee9f21ac604d415255c06e19f4cf1f4e8e5d98be1a194ee5`;
- retrieval time: `2026-08-30T08:07:34Z`;
- schema: `sackmann-match-csv/v1`;
- normalization availability rule: tournament-start date plus a conservative
  21-day lag.

Every yearly Git blob ID, payload SHA-256, exact immutable locator, observed
coverage bound, and source availability timestamp is retained in the registry.
Local raw snapshots and processed Parquet bundles are content-addressed,
checksum-verified, and ignored by Git.

## Intake audit

| Measure | ATP | WTA |
|---|---:|---:|
| Raw matches | 24,732 | 23,122 |
| Normalized matches | 24,376 | 22,939 |
| Normalized player-service rows | 48,752 | 45,878 |
| Unique canonical players | 1,103 | 1,139 |
| Matches with all five component rows usable | 23,535 (96.55%) | 22,375 (97.54%) |
| Quarantined source rows with error anomalies | 246 | 506 |
| Cross-source duplicate component rows | 0 | 0 |
| Unresolved identity rows | 0 | 0 |
| Exact normalized match dates | 0 | 0 |
| Usable tournament-start range | 2017-01-02–2025-11-23 | 2017-01-02–2025-11-01 |

ATP has 1,674 missing-input player rows for each primitive. Seven Q2 rows are
quarantined. WTA has 1,106 missing-input rows for F/A/Q1, 1,110 for D/Q2,
three quarantined Q1 rows, and eleven quarantined Q2 rows. Missing values remain
missing; zero-denominator rows remain explicit and no invalid count was clipped.

Terminal strings cover 650 ATP and 683 WTA retirement-marked matches, with
started evidence for 650 and 682 respectively. Exact retirement timing is absent.
There are 155 ATP and 158 WTA walkovers; ATP also has five default and one
abandonment records. Duration is present for 23,227 ATP and 21,927 WTA rows.
Winners and unforced errors are absent.

## Correctness repairs

Three non-methodological defects were repaired:

1. the operational source audit now validates the actual strict
   `SourceManifest` instead of an incompatible placeholder `status` wrapper;
2. optional `SvGms` discrepancies no longer discard otherwise valid primitive
   counts, and both observed tiebreak service-game accounting conventions are
   recognized; genuine mismatches remain explicit and quarantine only
   `service_games`;
3. verified nonoverlapping yearly processed bundles can now form one fit input
   with a schema-tagged composite provenance hash while retaining every per-row
   raw snapshot identity.

No statistical formula, constant, feature, dependence assumption, simulator, or
settlement rule changed.

## Hard validation blockers

The pinned files use `tourney_date`, a tournament-start date, for every match.
The normalized contract correctly leaves `match_date` missing. The frozen B6
artifact requires an exact official match date for its calendar-day window and
weights. Frozen C6 requires exact scheduled-start and last-started-match dates;
missing coverage must not be treated as a cold start. Historical cohort targets
also require genuine scheduled starts.

In addition, both retained repository objects became available in 2026. They
cannot represent the exact source versions and corrections available before
2022–2025 historical target cutoffs. Date-filtering current files is not a
substitute for cutoff-contemporaneous source versions.

Accordingly:

- B6 artifacts fitted: 0;
- cutoff-safe ModelSnapshots published: 0;
- pre-reveal historical targets frozen: 0;
- production locks and ledger rows: 0;
- rolling validation, comparators, dependence gate, and replay demonstration:
  not run.

A real multi-year in-memory fit smoke was started after the multi-bundle repair.
It remained CPU-bound after more than four minutes and was stopped without
publishing an artifact because the upstream B6/C6 and historical-version gates
already prohibited a valid catalog. This is a performance observation, not a
statistical result.

## Required next inputs

Before genuine rolling validation can proceed, obtain and pin:

1. exact match/scheduled-start dates for the selected ATP/WTA cohort and its
   preceding history, from sources permitted by the frozen hierarchy;
2. source objects or correction histories demonstrably available before each
   historical target cutoff;
3. a clean committed code/config state containing the repairs and registry.

Probability-affecting changes: **None**.  
Deviations from Tennis Model v1.0 or the B6/C6 amendment: **None**.

```text
GENUINE ROLLING-ORIGIN VALIDATION NOT COMPLETE
CORE v1.0 NOT READY FOR FINAL REVIEW
```
