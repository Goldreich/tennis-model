# Historical source and exact-date augmentation assessment

**Assessment date:** 2026-08-30  
**Scope:** retained Sackmann-style ATP/WTA Git history and Tennis-Data exact-match-date
augmentation for 2017–2025. No model was fitted and no probability-affecting method was
changed.

> This document assesses strict `POINT_IN_TIME_VINTAGE` reconstruction. The separately
> authorized `RETROSPECTIVE_FINALIZED` mode and its finalized-file crosswalk are reported in
> `docs/Retrospective_Finalized_Historical_Validation.md`; the two availability policies are
> not interchangeable.

## Conclusion

The retained repositories preserve real Git history, but not a sufficiently fresh and
continuous history for arbitrary rolling cutoffs throughout 2017–2025. At the representative
cutoff of **15 August 00:00 UTC in each year**, the ATP yearly file exists for 2017–2024 but
not 2025; the WTA yearly file exists for 2017–2018 and 2020–2024 but not 2019 or 2025.
Several existing snapshots are also months stale.

Tennis-Data is useful as a retrospective exact-date crosswalk. Against the available
pre-cutoff Git blobs, the conservative deterministic join uniquely matched 17,213 of 20,066
rows (85.782%). Coverage rises to about 93.1% after excluding team/Olympic/other event levels
that Tennis-Data does not cover. That exclusion is descriptive only: those matches cannot be
silently removed from the frozen history universe.

The combination is **not sufficient** to reconstruct fully cutoff-safe B6/C6 inputs or a
pre-reveal rolling target cohort:

- the same-year Tennis-Data workbooks currently served are mutable, finalized files whose
  `Last-Modified` timestamps are after the representative cutoffs;
- the Internet Archive has a strictly pre-cutoff workbook capture for only ATP 2023 among the
  18 tour-years checked;
- unmatched and conflicting joins prevent a complete last-started-match assertion;
- the Git histories have missing/stale same-year files at material cutoffs; and
- neither results source is a pre-result draw/schedule manifest from which target eligibility
  can be frozen before outcomes are revealed.

These are source/provenance findings, not a reason to alter B6, C6, or any other frozen model
rule.

## Method and cutoff discipline

The representative cutoff for each tour-year is `YYYY-08-15T00:00:00Z`, chosen as a consistent
pre-US-Open historical checkpoint. For each repository the selected commit was produced by:

```text
git rev-list -1 --before=YYYY-08-15T00:00:00Z HEAD
```

The yearly CSV was read only as `COMMIT:atp_matches_YYYY.csv` or
`COMMIT:wta_matches_YYYY.csv`. Current `HEAD` file contents were not substituted. Each
downloaded CSV's computed Git blob SHA-1 was checked against `git ls-tree`, then a SHA-256 was
computed over the exact bytes.

The retained repositories are:

- [Kadantte/tennis_atp](https://github.com/Kadantte/tennis_atp): 512 commits, root
  `5a152232f6d3ac1803834d4acddd82bc659972e8`, current assessed head
  `712be0c5ade693cdab9e69c23a71a0edf5a23c44`;
- [VictorSquidWei/tennis_wta](https://github.com/VictorSquidWei/tennis_wta): 515 commits, root
  `23f1299c9400a4629513f36d17ba6222d9a0f935`, current assessed head
  `85ef6efaa04ca860ed592a935498dcda9460ea91`.

Their README files attribute the data to Jeff Sackmann/Tennis Abstract under CC BY-NC-SA 4.0.
The forks retain the original commit topology, but a Git commit timestamp is not an external
notarization of when a commit was first pushed. Production use still requires retaining the
objects and hashes locally.

## Representative Git cutoff audit

`Max date` is the maximum Sackmann `tourney_date` (normally event start, not exact match date).
`Content lag` is the number of days from that date to the cutoff. All commit timestamps are
strictly before the cutoff.

| Tour | Year | Latest repository commit strictly pre-cutoff | Commit time | File rows | Max date | Content lag |
|---|---:|---|---|---:|---|---:|
| ATP | 2017 | `2251028fd29e6035b4b7fefd5fb8610a94945329` | 2017-03-10T08:30:37+01:00 | 388 | 2017-02-03 | 193 |
| ATP | 2018 | `a4ec6b5c920ec7bc2ba4b6ab698db93a13dd6d82` | 2018-05-28T19:46:10+02:00 | 1,416 | 2018-05-21 | 86 |
| ATP | 2019 | `6e2981f29fc7cdcc4bbd50225a7077350c2e6845` | 2019-05-16T00:05:17+02:00 | 651 | 2019-02-25 | 171 |
| ATP | 2020 | `90a1480c7a0cb7d642d27857898372b84b323361` | 2020-07-28T14:36:53+02:00 | 747 | 2020-03-06 | 162 |
| ATP | 2021 | `bf9cf7fa70f2997170f8c5f58d4f00797637a6f0` | 2021-08-10T12:02:03+02:00 | 1,895 | 2021-08-02 | 13 |
| ATP | 2022 | `3a017cb9c9ef14d68358557d84a3d69327d41890` | 2022-08-12T19:39:41+02:00 | 2,000 | 2022-08-01 | 14 |
| ATP | 2023 | `780d5ffa65c732aa19aba68c16f8e9c08ad1e5ec` | 2023-08-14T11:07:22+02:00 | 2,140 | 2023-08-07 | 8 |
| ATP | 2024 | `a36a13fe21f9d0e8ea45a78b3a425ac9bf7a6991` | 2024-05-27T08:59:53+02:00 | 1,417 | 2024-05-20 | 87 |
| ATP | 2025 | `5b6263dc487b4fa4fa35326764bc9b0215042a42` | 2024-12-30T11:23:38+01:00 | absent | — | — |
| WTA | 2017 | `81524e2fd19d154a6b9f931653908f695e216d89` | 2017-02-07T15:25:46+01:00 | 337 | 2017-01-30 | 197 |
| WTA | 2018 | `0860dddabb8391e5625ec8bf598eac3b18bc5643` | 2018-05-28T19:46:33+02:00 | 1,330 | 2018-05-21 | 86 |
| WTA | 2019 | `8a2d7568f6581eba9c82e6fdf8f27d1d93ba17fe` | 2019-01-01T22:46:17+01:00 | absent | — | — |
| WTA | 2020 | `ca6476a75d180758723ac892c56bf334343053ab` | 2020-07-28T15:07:03+02:00 | 797 | 2020-03-07 | 161 |
| WTA | 2021 | `177fb0f266440eff541cd5484912ec51006c0899` | 2021-08-14T14:48:30+02:00 | 1,838 | 2021-08-02 | 13 |
| WTA | 2022 | `7fc728fcaf6e2c3f27eec095ae1a15c225f3a623` | 2022-08-11T13:12:12+02:00 | 1,871 | 2022-08-01 | 14 |
| WTA | 2023 | `33bac212d2a780e29d4f6905bc791e40e13051f2` | 2023-08-14T11:09:00+02:00 | 2,064 | 2023-08-07 | 8 |
| WTA | 2024 | `b12002990a9aeeeaa42e53bdce180a27f5ddbe34` | 2024-05-27T09:18:25+02:00 | 1,175 | 2024-05-20 | 87 |
| WTA | 2025 | `225f6afd12d906cbe9bfab507551cdc5f346a540` | 2024-12-30T11:39:16+01:00 | absent | — | — |

WTA 2019 first appears only in commit
`5d3d6f78713de31948e85673c14878cf9b2012c2` on 2019-12-02. The 2025 ATP and WTA files first
appear only in May 2026 (`2c40e40c0f3ac51cc11eabe6c7c25932234e83e0` and
`85ef6efaa04ca860ed592a935498dcda9460ea91`, respectively). Those later contents were not used
for the historical-cutoff checks.

For all 15 present pre-cutoff files:

- the downloaded Git blob SHA-1 equals the tree blob ID;
- zero `tourney_date` values are on or after the cutoff;
- among uniquely joined rows, zero exact dates are on or after the cutoff, after the repository
  commit date, or on the repository commit date.

This verifies the dated subset. Rows that do not join to an exact date cannot be fully verified
from event-start dates alone, so the assessment does not claim a complete row-level availability
proof.

## Pinned pre-cutoff Git blobs

| Tour | Year | Git blob SHA-1 | SHA-256 of exact CSV bytes |
|---|---:|---|---|
| ATP | 2017 | `165d81f375d02fcfdf21ce295636860306edc30c` | `f1cd2008a43574d675079791499b5fa9c971ff74ee5a79bbb048b859a3a47a55` |
| ATP | 2018 | `e5bbf3a30dc5db14cac5f06aacc83d6e7f537ce6` | `33a6b987b02f550d66599f3a6b1df5a6d075c7406b312493bfc10e50021fded2` |
| ATP | 2019 | `e48646dd557db55432212ae1d255024f24db5316` | `3075b80d01c4b7dfdb13044953bc37cc3bf2d015027ac289bbaa9187adc9b099` |
| ATP | 2020 | `40c1a5a5617a789cbc81f46109d71fc38be497b7` | `f26848845e6ad59a56a5c9ba6328b2c0ab7b5892a2ecaa1ca66bfaa509f09b0a` |
| ATP | 2021 | `5bfe5cf6b9460285734d1046d95d7c8f49901e56` | `91d2260e246a97cd914a9af9f0fe9ba1d0d0c0cbba4a4588d89625fe8baf418e` |
| ATP | 2022 | `fd86b0b17cc84551642dedcb918b7400c1e4c457` | `680ac9ec27a1bd7428a21c6c444274cc80ae734cf59bab1513304a757ae6df4f` |
| ATP | 2023 | `abedd0f22a531e7bd3b9f6b6a22518548a1ddb22` | `ae9eef8d6623cfc1640021ee7431ac9837d5eee2fc45d475547833d304c6ae41` |
| ATP | 2024 | `36c000339cb7a3305311964f67994eb6d6022d1d` | `ca89ea5db5c526c7730d732535c5af44b4e3c4c8a6912a3e743ee05600222420` |
| WTA | 2017 | `d3eaa4320df6fbe1534f794707775e9a70467cda` | `ef00004c2ee7d1a6084b2f0bf5fb311e3059ceb427b51c13da161596b4497712` |
| WTA | 2018 | `f9f9eccd460a7be02a67f0217a0feafaaa1af910` | `fbf6f4c35f132447dafd2599e84c6937e4b77a6d60329b4498fd1b2128196aad` |
| WTA | 2020 | `7013dbedf76f9f2ae54b8e48aff292a808411dc3` | `d47eb5cda0a5542bae1c64f08e11dbd9474b14d742ba455cfa7d57d4706c4ee2` |
| WTA | 2021 | `36a6243c311df4184c7081ada388bee446a59583` | `357ed0951ec2846d064474d4c8e0b5db288cf8513232ec6023b3210b545ece37` |
| WTA | 2022 | `9ee6a3eaf0639bfcd465f19134692444a1dfe3f6` | `60da87d85e5bf2fe981e3cc1354ef4720607a9c0e98e4e77148ae4e2ac01035e` |
| WTA | 2023 | `1314f4fb744c142fb5dd1b22611dfb272892fc58` | `fcb225da4f01007847125b4d57e7f7bb688df18b8c62747f46f8a6fc180705c8` |
| WTA | 2024 | `b11feca30e7db1e01ba7cd3fcb2fddc68afab7c2` | `05d5fc31aae9724cdea6e96538d85db8d43a5aeae908641fce9d58ea951a2b0f` |

## Tennis-Data candidates and pins

The 18 current annual workbooks were retrieved from the
[Tennis-Data all-files page](http://www.tennis-data.co.uk/alldata.php) on 2026-08-30 between
09:49 and 09:55 UTC. ATP URLs follow
`http://www.tennis-data.co.uk/YYYY/YYYY.xlsx`; WTA URLs follow
`http://www.tennis-data.co.uk/YYYYw/YYYY.xlsx`. The site describes `Date` as the date of the
match for this period. The workbooks also contain bookmaker odds; the assessment used only
`Date`, event, directed player identities, surface, round, best-of, and set scores. No odds field
was used or carried into a model artifact.

| Tour | Year | Bytes | Server `Last-Modified` | SHA-256 |
|---|---:|---:|---|---|
| ATP | 2017 | 526,053 | 2017-11-20T14:32:05Z | `d8d2e2af5bda1f7891de39b18739ba229719e00672c22d2892b3e347dc6e1537` |
| ATP | 2018 | 474,033 | 2019-09-05T14:40:29Z | `3e52c70872c87736110bfe22f682bd224bb5883bf5367a7f8997aab991c7162a` |
| ATP | 2019 | 411,531 | 2019-11-18T00:05:48Z | `a267a7a779406ccd89762b98cbe4bb6370839a658f2a95ac098310ea3394e827` |
| ATP | 2020 | 206,869 | 2020-11-22T21:07:46Z | `310467bb81360e1981e9f3775946c63a5dc707a1d941e5ad4d9713da97daa752` |
| ATP | 2021 | 395,807 | 2021-11-21T17:33:48Z | `dcbd273a4fa101f4783384dbfe3410bfc7ee7bed0aed4ab9a09afbffb294c4ca` |
| ATP | 2022 | 417,917 | 2022-11-20T23:33:44Z | `9feaa1567783cb063e23b6f2d653d4c97210a48b001373eb367eb2d8b6a60a86` |
| ATP | 2023 | 447,088 | 2023-11-19T20:46:27Z | `5789a33720cbd5da9c7909713cfca131927eedaf305ab2b111ec6f8dda842b29` |
| ATP | 2024 | 445,729 | 2025-10-12T21:29:51Z | `d92a4d4167cfece60b624e81ba8d6724d90a4704637341bb0bc87539a36c746a` |
| ATP | 2025 | 426,707 | 2026-04-13T13:24:43Z | `941aaa1abc49131f51e1f7f6eee93dac829dd72578ca10b341dfc4c9d41ba013` |
| WTA | 2017 | 463,132 | 2021-05-05T21:48:33Z | `5476fa98ea38a885a5f7141e6aa6c4979dfb2d7f93715f7d13b46a17c1f140e0` |
| WTA | 2018 | 488,666 | 2018-11-04T16:10:44Z | `e96cc8317a634f0238c51f375896bfc47f53de96554adb0291003cc9b26c7278` |
| WTA | 2019 | 383,919 | 2021-04-20T21:02:56Z | `3302aa2b04e4873cd1079ef470d8b63778b35b9b821d2ca52621b933f88a567d` |
| WTA | 2020 | 168,741 | 2020-10-25T22:17:03Z | `f7dd7c632d8fc3a02c259983ab2dfd83a2e4516a3011d9b70caef6cd49c3eae1` |
| WTA | 2021 | 382,230 | 2021-11-18T12:25:33Z | `a364ca800b49b6ffc770b6766b8b15cec50ecfa9baf850c7973076dc936b1cc3` |
| WTA | 2022 | 370,069 | 2022-12-27T16:53:14Z | `d173f08e2607e2a3259448261f50cbe5bed6f298dd204427ea07b7795b4cc155` |
| WTA | 2023 | 407,251 | 2023-11-06T23:03:45Z | `ea4c4556c841ad696cd417885015fe0c63b7a2414c94d856f9515a238f2fef2a` |
| WTA | 2024 | 406,783 | 2025-07-28T09:20:52Z | `0ce3a4e87c269253dd61e1ab4a3bbeb2d9acdd512f316189e5ad872c86c67b53` |
| WTA | 2025 | 399,398 | 2026-01-05T12:27:52Z | `aac890b7465d0c74812578b2a962b4e5cb4c76455dccb6cece2efa5067448c13` |

These hashes pin the bytes used for the retrospective coverage assessment; they do not make the
mutable URLs retrospectively cutoff-safe. The site exposes no release ID or version history and
its disclaimer does not state a dataset reuse license. That licensing point needs resolution
before production redistribution. HTTPS negotiation also failed in the assessment environment,
so the direct files were retrieved over HTTP and verified by retained hashes.

The Internet Archive CDX index was checked for each exact workbook URL. Only ATP 2023 had a
distinct capture strictly before its representative cutoff:

- capture: [20230514150432](https://web.archive.org/web/20230514150432id_/http://www.Tennis-Data.co.uk/2023/2023.xlsx);
- WARC source: `MEGA002-20230514150203-crawl901/MEGA002-20230514150346-02570.warc.gz`;
- archive SHA-1 base32 digest: `TOVSWTOMPD45ZIUURYSRXB5FSKBJSKM4` (verified);
- bytes: 197,760;
- SHA-256: `830d731ffa2ea0de3fc44408aef9db40175f0102b6c16f1c5558c1f218b41ba0`;
- 1,101 workbook rows, dated through 2023-05-07.

It uniquely joins 1,056 of the 2,140 ATP rows present in the August Git snapshot (49.346%). All
joined dates are no later than 2023-05-07. The other 17 workbook URLs had no distinct CDX
capture strictly before their annual representative cutoff.

Supporting source-page snapshots used only for schema/provenance review were also hashed:

- `alldata.php`: `ef8fbc556ee4831043df152a1434c7aad2273fbfebedb46610be8ed0349a683d`;
- `notes.txt`: `13e57641a7bd67304ad9b98141aa6f413dff748676a703f5beff0078e8a27f29`;
- `disclaimer.php`: `0722c05050964e0230c287509f304b0d0b980efef955461ebcfc15c5ff0e92a3`.

## Deterministic join procedure

The measured procedure is intentionally conservative and contains no fuzzy matching:

1. Partition by tour and source year. Preserve winner/loser direction.
2. Unicode-NFKD normalize player names, remove combining marks, case-fold, and retain only
   alphanumeric tokens.
3. Convert Sackmann full names to candidate `(surname, first-initial)` forms using one through
   four trailing surname tokens. Convert Tennis-Data's `Surname F.` or `Surname F.M.` form to
   one canonical `(surname, first-initial)` key.
4. Require both directed player keys and a Tennis-Data date from three days before through 21
   days after Sackmann `tourney_date`. The negative allowance covers matches played on the
   Sunday before the source's conventional Monday event date.
5. Require compatible surface and best-of. Derive the expected round from Sackmann `draw_size`
   and round code; normalize the Tennis-Data round label. Compare ordered set-score signatures
   where both sources provide them.
6. Accept only one structurally consistent candidate and enforce a global one-to-one target.
   Multiple candidates, a reused target, a duplicate/conflicting source match, or a structural
   disagreement is blocked and retained with its reason.

The score is used only to validate/disambiguate this retrospective result-to-result crosswalk.
It cannot be used to construct a pre-result target cohort.

## Join measurements

The table below uses the pinned current Tennis-Data workbooks to measure potential retrospective
coverage against the exact pre-cutoff Git blobs. It is not a claim that the current workbooks
were available at those historical cutoffs.

| Tour | Year | Git rows | Unique joins | Unmatched | Ambiguous | Duplicate source match | Structural conflict | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ATP | 2017 | 388 | 254 | 133 | 0 | 0 | 1 | 65.464% |
| ATP | 2018 | 1,416 | 1,178 | 234 | 0 | 0 | 4 | 83.192% |
| ATP | 2019 | 651 | 595 | 56 | 0 | 0 | 0 | 91.398% |
| ATP | 2020 | 747 | 531 | 210 | 0 | 0 | 6 | 71.084% |
| ATP | 2021 | 1,895 | 1,671 | 211 | 1 | 0 | 12 | 88.179% |
| ATP | 2022 | 2,000 | 1,825 | 166 | 0 | 0 | 9 | 91.250% |
| ATP | 2023 | 2,140 | 1,930 | 198 | 0 | 0 | 12 | 90.187% |
| ATP | 2024 | 1,417 | 1,252 | 161 | 0 | 0 | 4 | 88.356% |
| WTA | 2017 | 337 | 324 | 10 | 0 | 2 | 1 | 96.142% |
| WTA | 2018 | 1,330 | 1,060 | 242 | 0 | 20 | 8 | 79.699% |
| WTA | 2020 | 797 | 478 | 319 | 0 | 0 | 0 | 59.975% |
| WTA | 2021 | 1,838 | 1,672 | 155 | 1 | 0 | 10 | 90.968% |
| WTA | 2022 | 1,871 | 1,632 | 235 | 1 | 0 | 3 | 87.226% |
| WTA | 2023 | 2,064 | 1,709 | 349 | 0 | 0 | 6 | 82.800% |
| WTA | 2024 | 1,175 | 1,102 | 70 | 0 | 0 | 3 | 93.787% |
| **Total** |  | **20,066** | **17,213** | **2,749** | **3** | **22** | **79** | **85.782%** |

ATP coverage is 86.690%; WTA coverage is 84.754%. Candidate ambiguity is low (3 rows,
0.015%), but low ambiguity does not offset missing coverage. Team events account for 729
unmatched ATP rows and team/125K/Olympic levels account for 851 WTA rows; after descriptively
excluding those levels, coverage is 93.058% ATP and 93.178% WTA.

The 22 duplicate WTA source rows are not harmless duplicate bytes. They include conflicting
records for the same directed players/event/round, such as an incomplete row alongside a
started-retirement row, and therefore remain quarantined.

As a direct C6-oriented diagnostic, only 1,555 of 2,486 ATP player-years (62.550%) and 1,379 of
2,006 WTA player-years (68.744%) have every row in their latest source event uniquely joined.
This is an upper-bound structural diagnostic based on the current workbooks, not a cutoff-safe
last-match reconstruction.

## B6/C6 and rolling-target sufficiency

### B6 retirement inputs

Not sufficient. B6 requires an exact normalized match date, cutoff-valid result availability,
complete source coverage, and reliable terminal classification. The pre-cutoff Git blob can
establish coarse batch availability and provides score/status evidence, but `tourney_date` is
normally the event start. Current Tennis-Data can attach exact dates to most main-tour rows, but
its same-year bytes were not versioned at the historical cutoffs and it misses material event
classes. Missing joins cannot be treated as non-retirements or silently excluded as if coverage
were complete.

### C6 last-started-match history

Not sufficient. C6 requires the true latest eligible started singles match, not merely the latest
successfully joined row. Any later unmatched team, Olympic, qualifying, or ordinary-tour row can
change `D`, the inactivity band, and therefore probabilities. The frozen cold-start branch is not
available when coverage or identity resolution is uncertain.

### Rolling historical targets

Not sufficient. Both evaluated sources are result datasets. A target manifest built from their
eventual result rows would reveal which matches occurred and survived data publication. Freezing
target eligibility before outcome reveal requires a separately pinned pre-result draw/schedule
source, including scheduled-start date and revision history. Post-result missing statistics may
then disable props without changing the frozen cohort.

## Required source work before genuine rolling validation

1. Obtain an immutable, redistribution-compatible exact-match-date source with historical
   versions or archives available before each target cutoff.
2. Cover or explicitly resolve team, Olympic, qualifying, and other eligible events in the
   frozen tour manifest; do not infer absence from a failed join.
3. Obtain a pinned pre-result schedule/draw archive for target-cohort freezing.
4. Retain source availability/correction timestamps, not only match dates and Git author dates.
5. Re-run this deterministic crosswalk and quarantine every ambiguous, duplicate, conflicting,
   or unmatched row before populating `config/sources.yaml`.

No production source registry was changed, no historical target cohort was frozen, and no model
fit or validation run was started as part of this assessment.
