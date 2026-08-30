# Tennis Model v1.0 B5 Match-Duration Implementation

This note records the implemented B5 auxiliary duration layer. The authoritative
probability specification remains `Tennis_Model_v1.0_Specification.md`; this is
an implementation and artifact report, not a replacement specification.

## Data and fit scope

The offline builder verifies the retained 2017–2025 ATP/WTA Sackmann-style raw
objects, their normalized bundles, the pinned exact-date augmentation, and the
official 2026 US Open completed-match capture. Historical duration provenance is
grade B; the current event feed is official/authoritative. The source-manifest
SHA-256 is
`b5e68e78712891ac750597a84742525b6529e65050a568ddee247f6021d1f945`
and the exact-date crosswalk-set SHA-256 is
`4f25376b139d5e4871cf7d9f304644431703179c7ec144e8cee9638a18548495`.

The full retained audit is:

| Tour | Candidates | Positive duration | Positive points | Both | Players | US Open | Other majors | Retirement rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATP | 24,732 | 23,168 | 23,723 | 23,162 | 1,118 | 1,143 | 3,302 | 650 |
| WTA | 23,122 | 21,825 | 22,409 | 21,730 | 1,139 | 1,143 | 3,302 | 685 |

Raw ATP has 1,564 missing/nonpositive duration rows and 1,009
missing/nonpositive point-total rows; WTA has 1,297 and 713 respectively. These
overlap. No value is inferred from score, clipped, or converted to zero.

The frozen 1,095-day, one-year-half-life fit uses verified exact match dates from
2023-08-31 through 2026-08-28:

| Tour | Combined candidates | Included | Historical | 2026 US Open | Players | Excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ATP | 12,708 | 5,334 | 5,227 | 107 | 409 | 7,374 |
| WTA | 11,809 | 4,999 | 4,888 | 111 | 460 | 6,810 |

ATP exclusions are 6,969 outside the fit window, 237 missing/nonpositive
exposure, 166 retirements, and 2 non-normal completions. WTA exclusions are
6,356 outside the window, 308 missing/nonpositive exposure, and 146 retirements.
The current official input has 111 ATP matches (four retirements) and 112 WTA
matches (one retirement); the retirement rows are audited but excluded from the
ordinary completed-match response. At the retained cutoff these completed
official rows are qualifying matches; no completed main-draw match was yet in
the pinned capture.

## Model and artifacts

The implementation fits ATP and WTA separately using the frozen predictor
`theta0 + thetaP*P + thetaG*G + thetaS*S + thetaB*B + qi + qj + cm`, centered
strongly shrunk player pace effects, and scale
`sigma0 + sigma1*sqrt(P)` with centered Student-t noise. Unknown optional
conditions remain missing in source data and contribute the declared central
zero baseline. At this cutoff, indoor, temperature, and session terms are
inactive because no fit rows carry those fields; the 2026 US Open effect is
active and shrunk with its fixed four-minute prior standard deviation.

| Tour/term | Estimate | SE | Tour/term | Estimate | SE |
| --- | ---: | ---: | --- | ---: | ---: |
| ATP intercept | -3.1952 | 0.8043 | WTA intercept | -7.1771 | 0.8798 |
| ATP points | 0.7283 | 0.0146 | WTA points | 0.6603 | 0.0127 |
| ATP games | 0.3697 | 0.1218 | WTA games | 0.6454 | 0.1079 |
| ATP sets | -2.3367 | 0.8354 | WTA sets | 1.4865 | 0.7373 |
| ATP tiebreaks | -0.8080 | 0.4536 | WTA tiebreaks | 0.6580 | 0.4711 |
| ATP US Open 2026 | -0.8259 | 0.9034 | WTA US Open 2026 | -2.6132 | 0.8335 |

The correlated exposure terms are deliberately retained; individual signs are
not monotonicity constraints. ATP has `sigma0=0.1544`, `sigma1=0.4851`, and
`nu=5.9435`. WTA has `sigma0=0.1771`, `sigma1=0.4029`, and `nu=4.3519`.
Both MAP optimizations converged. The full posterior dimensions are 418 (ATP)
and 469 (WTA). Their evaluated finite-difference Hessians were positive definite
before any repair (minimum eigenvalues 0.0623 and 0.0580; added regularization
zero), and the full covariance preserves the compensating exposure
correlations.

The valid content-addressed build is
`4c9d944a03931055df58b5ec8405eb22e7e69160958c9d28d4c4396e6a6c078b`:

| Tour | Duration artifact | Upgraded v3 snapshot |
| --- | --- | --- |
| ATP | `e09190b75097d9e9427e4c7751366d45c8844b153817d1490164331a9add21c4` | `23eb9a9a00af42226316e691f041325a93b6d4db1ddc1fa8dac30dda17385e7c` |
| WTA | `cd801ed4db698e42df9223a07eac015bfec2cc5a97a50c78338d4eef64bdb65d` | `30e0ca75ece12ad329f5d8211003219c76e2fb9ba5cf687599fb5db3db18b4fc` |

All ten output-file hashes, both artifact identities, both upgraded snapshots,
and the rejection-aware latest-build selector were independently verified.

## Validation

Controlled synthetic tests recover the configured fixed effects, adequately
exposed pace effects, heteroskedastic scale, and Student-t quantiles; demonstrate
sparse-player shrinkage; and compare the stored covariance and predictor
variance with an independently finite-differenced Hessian.

Historical diagnostics are descriptive in-sample conditional-MAP checks, not a
new rolling backtest:

| Tour | Mean residual | MAE | RMSE | 50% coverage | 80% | 90% | 95% | PIT mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATP | 0.646 | 6.167 | 8.532 | 48.46% | 78.50% | 88.47% | 93.81% | 0.5149 |
| WTA | 0.442 | 5.089 | 7.446 | 47.09% | 78.74% | 90.26% | 95.88% | 0.5060 |

All Pearson residual correlations with points, games, sets, and tiebreaks have
absolute magnitude below 0.036. For official integer outcomes, the analytic
floor-to-ceiling probability ranges at 90/120/150 minutes are respectively
69.88–71.06%, 40.15–40.98%, and 18.30–18.89% for ATP, and 59.23–60.53%,
30.75–31.51%, and 10.94–11.46% for WTA. The corresponding observed frequencies
are 70.12%, 40.81%, 19.67% and 59.33%, 31.37%, 11.68%.

Sparse pace histories shrink as intended. Mean absolute pace effect rises from
0.92 minutes below one weighted ATP match to 3.29 minutes at ten or more; WTA
rises from 0.98 to 3.34. Mean SE falls from 3.68 to 1.38 minutes for ATP and from
3.59 to 1.24 for WTA. Centering sums are below `1e-13` minutes.

The current-event check contains 107 ATP and 111 WTA matches. ATP actual and
predicted-center means are 110.42 and 109.83 minutes (MAE 3.91); WTA values are
105.23 and 104.86 (MAE 6.27). Event effects are -0.83 minutes (approximate 95%
Wald interval -2.60 to 0.94) and -2.61 minutes (-4.25 to -0.98). These are
descriptive in-sample stability checks, not held-out claims.

## Joint-path integration and settlement

Duration parameter and Student-t residual RNG streams are appended to the seed
tree without changing pre-existing streams. A target matchup's exact posterior
marginal is prepared once, then one parameter draw and one residual draw are
made per already-completed tennis path. The duration exposure is the same path's
realized points, official games, sets started, and tiebreaks. Retirement paths
use only partial realized exposure and set `duration_partial=True`. Regression
tests show that stripping duration fields leaves every score and tennis statistic
bit-for-bit unchanged.

`DURATION_MIN(>,90)`, `DURATION_MIN(>,120)`, and `DURATION_MIN(>,150)` are tested
pathwise on the same batch. Settlement preserves floor/nearest/ceiling candidate
sensitivity. Production cannot select a guessed display conversion: affected
paths remain unresolved and locks/cards expose probability bounds and a warning.
Test-mode resolved policies verify ordinary evaluation, immutable lock v4,
retained-artifact verification, replay, and duration-only Match Card rendering.

## Reproducibility and performance

The build cutoff is `2026-08-30T12:13:53.667208Z`, seed `20260830`, duration
config identity
`ddbaedbb5312dfbbe638463ab49fd96aca618b1e2a5c90fcfd4c102d57892939`,
code identity
`850850a32085c86ff5441917723d169b4516f834e1f273a1c9299b0f3655b9f4`,
and deterministic-test receipt SHA-256
`0b6d56f4a851fe11369a2f16146fe1684cafd381f6aa469aa38cdf8d6037d1e8`.
Repeated seeds reproduce duration paths and lock replay exactly.

Incremental duration-only timings on this machine, conditional on one retained
realized exposure, are:

| Tour | One-time preparation | 1,000 paths | 5,000 paths | 100,000 paths |
| --- | ---: | ---: | ---: | ---: |
| ATP | 0.0049 s | 0.1171 s | 0.5012 s | 10.8896 s |
| WTA | 0.0070 s | 0.1133 s | 0.5409 s | 10.5425 s |

These timings include one-time preparation in each displayed total and exclude
the upstream tennis simulation.

## Deviations and limitations

There is no deviation from the frozen B5 formula or architecture. Two trial
builds exposed an invalid optimizer-history covariance approximation and are
preserved but explicitly rejected by immutable receipts; selectors cannot use
them. The released build uses the repository's evaluated MAP-Hessian covariance
without changing the MAP fit or probability formula.

Remaining limitations are the unresolved official whole-minute display
conversion, strongly baseline-shrunk sparse pace histories, in-sample-only
current-event diagnostics based only on completed qualifying matches at this
cutoff, inactive unavailable roof/weather/session terms, and grade-B historical
duration provenance. No winner/unforced-error or other auxiliary model was
implemented.
