# Milestone 3 implementation contract

The authoritative probability model remains
`Tennis_Model_v1.0_Specification.md`. This document records the implemented
boundary, explicit numerical choices, and unresolved items without redefining the
frozen estimands.

## Implemented estimation boundary

- The only fitted primitives are `F`, `A`, `Q1`, `D`, and `Q2`, consumed directly
  from Milestone 1 successes/trials. `F` and `D` are server-only; `A`, `Q1`, and
  `Q2` jointly fit server and subtractive returner effects. ATP and WTA never share
  a fit.
- Each component uses a recency-weighted beta-binomial likelihood, an inclusive
  1,095-day rolling window, the fixed 365-day half-life, and strict
  `available_at_utc < cutoff` filtering. Invalid, missing, zero-denominator, and
  quarantined rows cannot re-enter through the estimation API.
- Global player effects and per-surface deviations are estimated jointly. Every
  usable row appears once. Effects use exact centered contrasts with zero-centered
  Gaussian shrinkage; relabeling player identifiers does not select a privileged
  reference player.
- Missing surface rows are excluded with an explicit diagnostic because the
  specified surface decomposition cannot be identified for them. Missing
  event/year rows remain in the base likelihood but receive no event effect.
  Neither case creates fitted pseudo-surface or pseudo-event levels. The default
  approved event-year block is active for `A`, `Q1`, and `Q2`.
- Indoor-hard support is implemented but disabled in the checked-in production
  config because the Milestone 1 source does not reliably identify roof state.
  Handedness and best-of-five fields are typed but inactive; no ranking, Elo,
  head-to-head, form, or other unapproved feature is fitted.

## MAP and posterior representation

L-BFGS-B minimizes the explicit negative weighted log posterior with an analytic
gradient. A failed optimizer raises and cannot produce a fitted artifact.
Diagnostics record convergence, objective and gradient status, fitted
concentration and shrinkage scales, exclusions, raw/weighted exposure,
calibration summaries, and the frozen effective-information formulas.

Laplace curvature is a deterministic central finite difference of the analytic
gradient. Fits with at most 200 parameters store a symmetrized dense Hessian and
covariance. Larger fits store a diagonal observed-curvature approximation to avoid
quadratic memory. Both paths record the unregularized minimum curvature,
regularization, condition number, stable parameter ordering, and positive
variance diagonal. The diagonal path is an explicit approximation, not a switch
to another inference method.

## Prediction and artifacts

The typed future-match result contains the five conditional MAP means,
linear-predictor uncertainty, beta shape parameters at the MAP, support flags,
context, and a common fit identity. Unseen player, surface-deviation, or supplied
event levels use the centered zero mean and add the corresponding fitted MAP
shrinkage-scale variance. These are summaries only: Milestone 3 draws no posterior
parameters, match-performance probabilities, points, or match paths.

Fit JSON and human diagnostics are append-only and content-addressed. Writes and
loads revalidate finite posterior state, hashes, parameter dimensions, covariance
symmetry, configuration identity, concentration consistency, and convergence.
The production entry point accepts a verified Milestone 1 processed bundle and
binds the raw snapshot and component-count Parquet checksums rather than trusting
caller labels. Component-count transformation `serve-component-counts-v1.1`
changes only lineage/context schema; the primitive count arithmetic and status
behavior are unchanged.

## Explicit choices and unresolved specification details

The prose specification does not numerically select Gaussian prior scales, the
log shrinkage-scale hyperprior, concentration bounds, optimizer tolerances, or the
dense-curvature threshold. `config/model_v1.yaml` makes these choices explicit,
strictly validated, embedded in every fit, and hash-visible. Concentration is
likelihood-estimated within its configured bounds and has no additional prior.
These values require framework-version review before any probability-affecting
change.

The specification requires posterior-uncertainty inflation and mean reversion of
hard-specific deviations after more than 90 days of inactivity, but gives no
functional form or numerical scale. The fit currently records player exposure but
does not apply an inactivity adjustment. This remains a genuine
probability-affecting blocker rather than an invented rule.

When exact `match_date` is unavailable, the estimator may use the preserved
`source_date` for recency and records `date_fallback_rows`. For tournament-start
dated sources this is an approximation; precise within-event updates still
require a separately pinned timing source.

No audited production ATP/WTA snapshot is checked into the repository. The
cutoff demonstration is therefore synthetic and proves leakage invariance only;
it is not presented as a historical forecast or accuracy claim.

## Next milestone boundary

The implemented component fits and posterior artifacts are ready to be consumed
by Milestone 4's revised causal point generator once the user explicitly requests
that milestone. Production forecasts remain blocked on audited source data and a
versioned inactivity rule. Milestone 4 must retain joint posterior parameter-draw
semantics and must not infer independent two-direction uncertainty from detached
univariate summaries.
