> **Superseded preproduction design.** This dynamic-anchor candidate is retained
> for historical documentation only. The authoritative production v1.1 model is
> specified in `docs/Tennis_Model_v1.1_Production_Specification.md`.

# Tennis Model v1.1 Candidate Specification

**Strength-anchored point model for coherent match and prop prediction**
**Specification date:** 1 September 2026
**Status:** design-complete candidate; not approved for production until the
rolling-origin validation gates in Section J pass

This document specifies a probability-affecting successor to frozen Tennis
Model v1.0. It does not amend, reinterpret, or replace the v1.0 specification,
its fitted artifacts, or any lock created under it. The proposed framework
identifier is `Tennis Model v1.1`. Until validation and an explicit promotion
decision, production forecasts remain on v1.0 and experimental forecasts must
be labeled `Tennis Model v1.1-candidate`.

The sole architectural objective is to correct occasional catastrophic
player-strength inversions and excessive match-winner certainty while
preserving the point-level mechanism that has performed well for ace and
double-fault props. The change adds an independently estimated match-outcome
strength anchor, learns how much incremental information it supplies under
strict rolling cutoffs, and expresses the resulting correction through
returnable first-serve and playable second-serve point outcomes. It never
replaces a simulated match-winner probability after the path has been drawn.

---

## A. Evidence, scope, and design decision

### A1. Triggering evidence

The available 53-market live scorecard is diagnostic evidence, not a training
or validation sample:

- 33 winner forecasts beat the crowd benchmark and 20 did not;
- the median Relative Brier Points gap was `+1.9`;
- the mean gap was `-0.55`;
- 31 forecasts were within `+/-5` points;
- excluding the three worst forecasts, the mean was `+1.43`;
- Barrios Vera, Djokovic, and Darderi accounted for the dominant losses.

This pattern is consistent with a generally useful model whose aggregate
winner performance is damaged by a small number of tail failures. Barrios Vera
and Darderi are candidate strength-ordering failures. Djokovic is principally
an excessive-certainty failure because both the model and benchmark favored
the same player. These examples motivate the change but must not determine its
hyperparameters or promotion decision.

### A2. Authorization correction

Ranking, Elo, independent overall strength, head-to-head information, and
recent trajectory are not categorically prohibited in v1.1. Their omission
from v1.0 was a framework choice, not a user requirement. Each receives an
explicit role or validation gate below.

Market odds, crowd probabilities, prediction-market forecasts, and the live
Relative Brier scorecard remain prohibited as model inputs. They may be used
only as quarantined, timestamped external evaluation benchmarks.

### A3. Selected architecture

v1.1 selects the validated tour-specific, surface-aware Elo model as the
independent overall-strength anchor. ATP and WTA are updated separately; every
player begins at 1500; K is 16; and the effective pre-match rating is 50 percent
global Elo plus 50 percent current-surface Elo. The dynamic Bradley-Terry model
is retained as an experimental ablation, not the selected production candidate.

The anchor is not directly substituted for the point model. A constrained,
cross-fitted integration model combines the component-implied match logit and
the anchor logit. For every posterior/performance draw, the combined target is
mapped back into a symmetric adjustment of `Q1` and `Q2` before points are
simulated. Therefore winner, score, duration, tiebreak, ace, and double-fault
props remain functions of the same joint match paths.

### A4. Version boundary

The following are probability-affecting v1.1 changes:

1. the versioned surface-Elo outcome-strength artifact;
2. the cross-fitted strength-integration artifact;
3. a draw-level symmetric `Q1`/`Q2` tilt;
4. joint uncertainty induced between `Q1`, `Q2`, and the two serving
   directions by that tilt;
5. expanded strength, disagreement, and tail-calibration diagnostics.

All unlisted v1.0 behavior remains unchanged, including the primitive
estimands, causal serve ordering, scoring state machine, rally attachment,
duration model, retirement generator, prop ontology, settlement policy,
information scenarios, immutable locks, and explicit RNG handling.

---

## B. Preserved primitive model

### B1. Primitive estimands

For server `i` against returner `j`, v1.1 retains exactly:

\[
\begin{aligned}
f_i &= P(\text{first serve in}),\\
a_{ij} &= P(\text{ace}\mid\text{first serve in}),\\
q_{1,ij} &= P(\text{server wins}\mid\text{first serve in, non-ace}),\\
d_i &= P(\text{double fault}\mid\text{second-serve opportunity}),\\
q_{2,ij} &= P(\text{server wins}\mid\text{playable second serve}).
\end{aligned}
\]

The observable successes, denominators, anomaly rules, zero-denominator
policy, source transformations, and hierarchical beta-binomial likelihoods
are unchanged from v1.0.

The identities remain

\[
w_{1,ij}=a_{ij}+(1-a_{ij})q_{1,ij},
\qquad
w_{2,ij}=(1-d_i)q_{2,ij},
\]

\[
p^{\mathrm{srv}}_{ij}
=f_i\{a_{ij}+(1-a_{ij})q_{1,ij}\}
+(1-f_i)(1-d_i)q_{2,ij},
\]

\[
r^A_{ij}=f_i a_{ij},
\qquad
r^D_{ij}=(1-f_i)d_i.
\]

### B2. Causal point ordering

The v1.0 point generator is unchanged:

1. draw first serve in or out from `F`;
2. after a first serve lands in, draw an ace from `A`;
3. if it is not an ace, draw the point result from integrated `Q1`;
4. after a missed first serve, draw a double fault from `D`;
5. if it is not a double fault, draw the point result from integrated `Q2`;
6. attach rally termination only to eligible non-ace/non-double-fault points.

An ace is still an immediate server win and a double fault is still an
immediate server loss. The strength anchor cannot create, suppress, or relabel
either event.

### B3. Directly protected quantities

The strength layer does not alter the fitted linear predictors, posterior
draws, or match-performance beta marginals for `F`, `A`, or `D`. Conditional on
a service opportunity and the corresponding primitive draw, their event
probabilities are identical to v1.0.

Ace and double-fault match counts may change indirectly because integrated
point strength can change games, sets, service points, and which player serves
more often. Such exposure changes are required for path coherence. They are
not changes to ace or double-fault propensity.

---

## C. Outcome-strength anchors

### C1. Eligible outcome rows

Fit ATP and WTA anchors separately. The unit is a started, completed,
non-retired singles match with known players, date, surface, format, and
official winner. Exclude walkovers. The default skill likelihood excludes
retirements because retirement is already generated by a separate health
process and should not be mislabeled as ordinary playing strength. A
predeclared sensitivity fit may include retirements, but it cannot replace the
default without passing the same rolling validation gates.

Every row must have an observation timestamp and source snapshot available by
the historical forecast cutoff. Corrected outcomes enter only forecasts after
the correction timestamp. No rank, rating, or outcome observed after a cutoff
may be reconstructed into that cutoff.

### C2. Dynamic latent-state ablation

For player `i`, date `t`, and surface `s`, define

\[
z_{i,s}(t)=g_i(t)+u_{i,s},
\]

where `g_i(t)` is dynamic all-surface strength and `u_{i,s}` is a shrunk
surface deviation. Within a tour, impose weighted centering at each fitted
cutoff,

\[
\sum_i w_i g_i(t)=0,
\qquad
\sum_i w_i u_{i,s}=0,
\]

using active-player information weights `w_i`. Centering is an identification
operation and does not discard posterior uncertainty.

Between appearances, all-surface strength follows a mean-reverting Gaussian
state transition:

\[
g_i(t_2)\mid g_i(t_1)
\sim N\left(\rho(\Delta t)g_i(t_1),
\sigma_g^2\{1-\rho(\Delta t)^2\}\right),
\]

\[
\rho(\Delta t)=\exp(-\Delta t/\tau_g).
\]

Surface deviations receive

\[
u_{i,s}\sim N(0,\tau_s^2).
\]

The finite candidate set for `tau_g`, surface pooling, and innovation scale is
declared in configuration before an outer backtest. Selection occurs only in
inner rolling folds. If the data cannot identify a dynamic parameter, the
fallback is the one-year decay and prior scales fixed from earlier events, not
a tournament-specific estimate.

### C3. Outcome likelihood

For match `m` in which `i` plays `j`, let `Y_m=1` if `i` wins. The anchor uses

\[
Y_m\sim\operatorname{Bernoulli}(\pi_m^S),
\]

\[
\operatorname{logit}\pi_m^S
=c_{b_m}\{z_{i,s_m}(t_m)-z_{j,s_m}(t_m)\},
\]

where `c_b > 0` is a tour-specific format scale for best-of-three or
best-of-five. There is no player-order intercept. Reversing the player order
negates the anchor logit exactly.

Event level is recorded for diagnostics. It is not an arbitrary additive
advantage. Challenger, qualifying, and main-tour observations share a player
graph so crossover players identify relative competition strength. Slices
that are disconnected or weakly connected receive inflated uncertainty and a
lower data grade rather than an undocumented level correction.

### C4. Estimation and posterior

Fit the dynamic Bradley-Terry model by deterministic MAP state-space
optimization with a sparse Laplace approximation. A sequential Gaussian
filter/update is permitted only if it reproduces the batch rolling-cutoff fit
within declared numerical tolerances. Store the state mean, sparse precision
or reproducible factorization, hyperparameters, player graph diagnostics, and
optimizer diagnostics.

For a future matchup, draw a joint state vector from the cutoff posterior and
advance each player through the state transition to the scheduled date. The
resulting sampled anchor logit is

\[
L^S_b=c_b\{z_{A,s}^{(b)}(t)-z_{B,s}^{(b)}(t)\}.
\]

This draw represents both fitted-state uncertainty and uncertainty accumulated
during inactivity. It is not a new match-performance draw for aces or double
faults.

### C5. Selected surface-Elo anchor

The selected anchor is deterministic surface Elo with fixed, versioned rules:

- ATP and WTA have independent rating pools;
- every previously unseen player starts at 1500;
- K is 16;
- global and current-surface ratings are each updated after every eligible match;
- the rating used for prediction is 50 percent global and 50 percent current surface;
- the win probability is the conventional base-10 Elo transform with scale 400;
- walkovers, retirements, defaults, abandonments, and byes do not update Elo.

Rows are processed in deterministic event and match order and must be available
strictly before the forecast cutoff. The artifact stores all player ratings,
the cutoff, source hashes, configuration, code provenance, and identity aliases.
Unknown players retain the 1500 prior. Elo contains no injury or fitness shift.

---

## D. Cross-fitted integration with Q1 and Q2

### D1. Why integration is learned out of sample

The strength and component models use overlapping historical matches. v1.1
does not pretend that their errors are independent and does not precision-blend
them as if they were based on unrelated evidence. Instead, it generates strict
rolling-origin, out-of-fold predictions from both models and learns only the
incremental outcome information carried by the anchor conditional on the
component forecast.

For each integration-training match `m`, construct using its historical
cutoff:

- `L_m^C`, the logit match-win probability from the unintegrated component
  model and exact scorer;
- `L_m^S`, the dynamic strength-anchor logit;
- `V_m^C` and `V_m^S`, their predictive variances;
- predeclared component exposure and fit-stability diagnostics;
- the official completed outcome.

The match itself and all later observations are absent from both fitted
artifacts. Integration coefficients are fitted only to these out-of-fold
records.

### D2. Reliability gate

Define a compact diagnostic vector `x_m` containing only:

\[
x_m=(1,
\log\{(V_m^C+\epsilon)/(V_m^S+\epsilon)\},
U_m^C,
E_m^C),
\]

where `U_m^C` is a versioned component-instability score and `E_m^C` is a
versioned component-sparsity score. Instability may use optimizer convergence,
posterior condition number, hyperparameter boundary contact, and predictive
sensitivity. Sparsity uses `N^info` and `M^eff` from v1.0. Definitions and
normalizations are fitted-artifact metadata and cannot be changed per match.

The anchor weight is

\[
G_m=\operatorname{logit}^{-1}(\eta^\top x_m),
\qquad 0<G_m<1.
\]

Coefficients multiplying relative component uncertainty, instability, and
sparsity are constrained so that worse component evidence cannot reduce anchor
weight, all else equal. Greater anchor uncertainty reduces its weight through
the variance ratio. `eta` receives a shrinkage prior centered on a constant,
modest anchor weight.

### D3. Integrated target logit

The cross-fitted target is

\[
L_m^T=\beta_C L_m^C+G_m(L_m^S-L_m^C),
\qquad \beta_C>0.
\]

For the selected surface-Elo configuration, completed validation fixes
`beta_C=1` and `G_m=0.75`, so the target logit is 25 percent component-model
logit and 75 percent surface-Elo logit. These weights are versioned constants,
not reselected for an individual tournament or matchup. The reliability-gated
integration remains available only for the dynamic-anchor ablations.

`beta_C` is a global, tour-specific calibration temperature. It controls
component-model overconfidence or underconfidence. There is no free intercept,
because player-order symmetry requires

\[
L^T(B,A)=-L^T(A,B).
\]

Estimate `beta_C` and `eta` by penalized Bernoulli log likelihood on the
rolling-origin integration records. Hyperparameters are selected in nested
rolling folds. Store a Laplace or event-block bootstrap approximation for
coefficient uncertainty.

The nested validation must compare at least:

1. v1.0 unchanged;
2. temperature calibration only;
3. constant anchor weight;
4. reliability-gated anchor weight;
5. the complete integrated candidate.

### D4. Mapping the target back to point probabilities

For posterior/performance draw `b`, first draw the v1.0 primitive probabilities
for both serving directions and calculate the baseline exact-scoring match
logit `L_b^C`. Draw `L_b^S` from the anchor posterior and integration
coefficients from their uncertainty distribution. Calculate `L_b^T` using the
same equation as D3.

For a scalar point-strength tilt `delta_b`, define

\[
\begin{aligned}
\operatorname{logit}q^{*,b}_{k,AB}
&=\operatorname{logit}q^b_{k,AB}+\delta_b/2,\\
\operatorname{logit}q^{*,b}_{k,BA}
&=\operatorname{logit}q^b_{k,BA}-\delta_b/2,
\qquad k\in\{Q1,Q2\}.
\end{aligned}
\]

Positive `delta_b` favors player A by improving A's playable serve outcomes
and A's playable return outcomes symmetrically. The v1.1 default applies equal
logit shifts to `Q1` and `Q2`. Unequal positive weights may be evaluated only
as a predeclared ablation with the identification constraint that their mean
is one.

Let `M_b(delta)` be player A's match-win probability from the deterministic
exact-scoring operator under the tilted primitives, format, and first-server
mixture, excluding retirement scenarios. Solve

\[
\operatorname{logit}M_b(\delta_b)=L_b^T.
\]

`M_b(delta)` is monotone, so use deterministic bracketed root finding. Cache or
interpolate the monotone mapping only under a tested numerical error bound.
The full stochastic match is then simulated using `q1*` and `q2*`; the target
probability itself is never returned as a prop estimate.

If the target is outside the attainable interval under configured numerical
probability bounds, use the nearest attainable tilt, record `tilt_saturated`,
and expose the target-versus-attained difference. This is a diagnostic event,
not silent clipping. Frequent saturation fails validation.

### D5. Consequences for joint paths

The strength tilt changes only playable point outcomes. It can consequently
change:

- match, set, game, hold, break, and tiebreak probabilities;
- the number and allocation of service opportunities;
- duration through changed exposure;
- expected ace and double-fault counts through changed exposure;
- comparative ace and double-fault probabilities when exposure changes
  asymmetrically.

It does not directly change:

- first-serve-in probability;
- ace probability conditional on first serve in;
- double-fault probability conditional on a second-serve opportunity;
- official settlement semantics.

This small, coherent exposure effect is preferred to a winner-only post-hoc
blend that would make the reported winner probability inconsistent with the
simulated score and stat paths.

---

## E. Ranking, trajectory, head-to-head, and information features

### E1. Ranking

Ranking is permitted but is not a routine additive predictor in the complete
candidate. Its principled candidate role is initialization of an unseen or
extremely sparse player's anchor state through a tour- and date-specific noisy
measurement model. Any such prior must use a timestamped pre-cutoff ranking
snapshot, account for unranked and protected-ranking cases, and have uncertainty
large enough for observed matches to dominate.

Ranking enters v1.1 production only if an inner/outer rolling ablation improves
sparse-player winner calibration beyond the dynamic anchor and does not harm
other groups. Otherwise the fitted coefficient or prior link is zero. This is
an empirical gate, not a categorical exclusion.

### E2. Recent form and trajectory

Recent information enters in two distinct ways:

1. the component likelihood retains v1.0 exponential weighting;
2. the dynamic anchor permits latent strength innovations and mean reversion.

The second mechanism can represent rapid improvement or decline that a fixed
weighted average misses. Do not also add an arbitrary last-five-match win
percentage or tournament-form multiplier. A separate level-change process may
be tested if posterior predictive residuals show persistent changes, but its
change points must be inferred from pre-cutoff data and validated across
events.

### E3. Head-to-head

Head-to-head is permitted as a strongly shrunk antisymmetric interaction:

\[
h_{ij}=-h_{ji},
\qquad
h_{ij}\sim N(0,\tau_H^2).
\]

A candidate anchor logit may add `h_ij` only when the observation set, recency,
and posterior shrinkage are explicit. The default complete candidate fixes
`tau_H=0` because the current evidence has not established incremental value.
The feature is promoted only if nested rolling validation improves outcomes
for previously unseen future meetings, not by fitting and evaluating the same
pair history. Report posterior mass near zero and sensitivity to one-match
samples.

### E4. Injury, inactivity, and workload

Verified injury and availability information remains a timestamped scenario,
not a hidden strength shift. Inactivity naturally increases anchor uncertainty
through the state transition and retains v1.0 component mean reversion. A
verified limitation may affect retirement, pace, or component scenarios only
through a named information bundle with explicit provenance.

### E5. Market-information quarantine

Odds, crowd forecasts, prediction markets, SportsPredict submissions, and
post-match Relative Brier results must be stored outside every feature-building
interface. Backtest code may join them only after model probabilities and
cutoffs are frozen. A model artifact containing a market-derived input is
invalid for v1.1.

---

## F. Fitting, uncertainty, and reproducibility

### F1. Fitting sequence

For each tour and cutoff:

1. construct immutable eligible historical rows;
2. fit unchanged v1.0 primitive component models;
3. fit or update the dynamic strength anchor;
4. load a pre-cutoff integration artifact trained on earlier out-of-fold
   forecasts;
5. estimate matchup primitives and diagnostics;
6. draw component, anchor, and integration uncertainty;
7. map strength targets to integrated `Q1`/`Q2` draws;
8. run the unchanged joint match simulator and prop evaluators.

An integration artifact may never train on outcomes from the event interval it
is used to evaluate unless those outcomes occurred before the individual
forecast cutoff and the evaluation design explicitly permits expanding-window
updates.

### F2. Uncertainty layers

v1.1 propagates:

| Layer | Treatment |
|---|---|
| Primitive parameter uncertainty | unchanged component posterior draw |
| Primitive match-performance variation | unchanged beta draw for `F,A,Q1,D,Q2` before integration |
| Anchor state uncertainty | joint dynamic-state posterior draw advanced to match date |
| Integration uncertainty | draw calibrated coefficients from stored approximation |
| Match-path uncertainty | simulate points, scoring, stats, duration, and retirement |
| Information uncertainty | explicit scenario mixture or separate sensitivity |

The integrated tilt induces intentional dependence between `Q1`, `Q2`, and
serving directions. It does not activate the optional v1.0 shared serving-day
factor. If that factor is later validated, its combination with strength
integration requires a separately tested framework revision.

### F3. RNG streams

Derive named child streams from the lock seed sequence for:

- primitive posterior draws;
- primitive match-performance draws;
- anchor state draws;
- integration-coefficient draws;
- first-server choice;
- point paths;
- rally categories;
- duration;
- retirement;
- information scenarios.

Store RNG algorithm, root seed identifier, stream derivation version, path
count, settled count, and Monte Carlo error. Hidden global randomness is
prohibited.

### F4. Numerical controls

Root finding must use fixed tolerances declared in configuration. Store maximum
residual, saturation count, and cache/interpolation error. Failure to converge
for any draw cannot be silently converted to zero tilt; quarantine the draw or
fail the lock according to a versioned numerical policy.

Simulation path counts and escalation rules remain those in the active
production configuration. Development backtests may use fewer paths only when
Monte Carlo uncertainty is measured and negligible relative to predeclared
model-comparison margins.

---

## G. Simulation and prop semantics

### G1. Match simulation

After the v1.1 parameter draw is integrated, use the v1.0 simulation sequence
without modification: scenario, parameter draw, performance draw, first server,
exact games and sets, tiebreak service order, retirement, duration, official
accounting, and settlement-aware prop evaluation.

All related and compound props are evaluated from the same integrated paths.
It is invalid to use the anchor probability for `MATCH_WIN` while using
unintegrated paths for other props.

### G2. Match-winner calibration

The `beta_C` temperature in D3 is the v1.1 winner-calibration mechanism. It is
learned on historical out-of-fold forecasts and implemented through the point
tilt. No additional post-simulation Platt scaling, isotonic regression, manual
favorite adjustment, or probability cap is permitted in the default candidate.

Submitted integer clipping to `1-99` remains a championship interface rule,
not a fitted calibration method.

### G3. Settlement

The v1.0 settlement-policy version remains authoritative because v1.1 changes
event generation, not question meaning. Walkovers, retirements, comparison
ties, official rounding, and void filtering are unchanged. Any later settlement
clarification creates a settlement-policy version, not a v1.1 model amendment.

---

## H. Safeguards and diagnostics

### H1. Component-anchor disagreement

Every estimate records:

\[
D_L=|L^S-L^C|,
\]

the posterior probability that component and anchor logits have opposite
signs, the reliability weight `G`, the applied tilt, and target-versus-attained
logit residual.

Flag a disagreement when a threshold fixed from historical training
quantiles is crossed. The flag triggers review and a sensitivity report; it
does not automatically choose the anchor or force a forecast toward 50%.

### H2. Sparse and unstable players

The anchor receives more integration weight only through the fitted monotone
reliability gate. Low `N^info`, low `M^eff`, disconnected competition graphs,
Q1/Q2 optimizer boundary contact, ill-conditioned posterior curvature, and
large predictive sensitivity lower the data grade and widen uncertainty.

No single diagnostic may produce a hand-coded player adjustment. The mapping
from diagnostics to anchor weight is fixed by the integration artifact.

### H3. Extreme probabilities

Monitor forecasts below 0.05 and above 0.95 separately. Report calibration,
log loss, high-confidence upset frequency, and the worst-decile Brier score.
The calibration temperature and strength uncertainty should control unjustified
extremes. Arbitrary probability ceilings are prohibited.

### H4. Lower-level and new players

For weakly connected or new players, report:

- component information diagnostics;
- anchor graph connectivity and posterior variance;
- whether a ranking prior was active;
- sensitivity with and without the anchor;
- integration weight and tilt saturation.

If neither components nor anchor provide a defensible estimate, assign the
lowest data grade and widen uncertainty. Do not manufacture certainty from a
ranking number alone.

### H5. Protected prop families

For every backtest and live monitoring period, report ace and double-fault
comparison Brier scores, expected-count residuals, calibration, dispersion,
tie rates, and exposure-adjusted primitive rates. Separate direct rate
diagnostics from count changes caused by match exposure.

---

## I. Data, configuration, artifacts, and locks

### I1. Data additions

The core strength anchor uses the same immutable match identities, dates,
surfaces, formats, completion states, and winners already present in normalized
historical data. Add explicit fields for:

- result observation timestamp;
- result correction timestamp, if any;
- outcome eligibility reason;
- competition-graph component identifier;
- optional timestamped ranking snapshot identifier.

Derived missing values remain missing. Invalid or temporally ambiguous rows are
quarantined with reason codes rather than repaired.

### I2. Proposed configuration

Create `config/model_v1_1.yaml` during implementation with at least:

```yaml
framework_version: "Tennis Model v1.1-candidate"

strength_anchor:
  kind: dynamic_bradley_terry
  tours: [ATP, WTA]
  exclude_walkovers: true
  exclude_retirements: true
  surface_pooling: hierarchical
  dynamic_decay_candidates_days: []
  innovation_scale_candidates: []
  laplace_draws: null

strength_integration:
  kind: cross_fitted_q_tilt
  q1_weight: 1.0
  q2_weight: 1.0
  enforce_player_order_symmetry: true
  reliability_features:
    - relative_predictive_variance
    - component_instability
    - component_sparsity
  root_tolerance: null
  probability_bound: null

optional_features:
  ranking_sparse_prior: false
  dynamic_level_change: false
  head_to_head_interaction: false

market_inputs:
  permitted: false
```

Empty candidate sets and null numerical tolerances must be filled by a
pre-registered implementation/backtest plan before fitting. They are explicit
unresolved empirical settings, not permission to select values after viewing
outer-fold results.

### I3. Fitted artifacts

A v1.1 snapshot references:

- unchanged primitive component artifact IDs;
- strength-anchor artifact ID;
- strength-integration artifact ID;
- optional feature-gate artifact IDs;
- data, configuration, and code hashes;
- training and validation cutoff ranges;
- strength-state and integration diagnostics;
- settlement-policy version.

Artifacts are immutable and content addressed. Refitting at a new cutoff creates
new artifact IDs without changing the framework version.

### I4. Typed interface additions

Implementation should add immutable equivalents of:

```python
@dataclass(frozen=True)
class StrengthPosterior:
    artifact_id: str
    tour: str
    cutoff_utc: datetime
    player_state_reference: str
    covariance_reference: str
    hyperparameters: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StrengthIntegrationDraw:
    component_logit: float
    anchor_logit: float
    target_logit: float
    reliability_weight: float
    q_tilt: float
    attained_logit: float
    saturated: bool


@dataclass(frozen=True)
class StrengthIntegrationSpec:
    artifact_id: str
    calibration_temperature: float
    reliability_parameters: dict[str, float]
    q1_weight: float
    q2_weight: float
    numerical_policy_version: str
```

`ModelSnapshot` gains strength and integration artifact IDs.
`MatchParameterDistribution` stores baseline and integrated `Q1`/`Q2`
posteriors plus provenance. The simulator consumes only integrated draws for a
v1.1 path.

### I5. Lock additions

A v1.1 lock records:

- baseline component winner probability;
- anchor winner probability;
- integrated path winner probability;
- component and anchor uncertainty;
- reliability weight distribution;
- `Q1`/`Q2` tilt distribution;
- disagreement flags and sign-disagreement probability;
- tilt saturation and numerical residuals;
- expected ace/DF counts and primitive rates before and after integration;
- every new artifact and RNG stream identifier.

Match locks remain immutable. A v1.0 lock can be recomputed under v1.1 only as
a separate counterfactual lock; it is never relabeled or overwritten.

---

## J. Validation and promotion gates

### J1. Required design

Use nested rolling-origin validation with strict historical information
cutoffs. Inner folds select anchor and integration settings. Untouched outer
folds estimate performance. Include multiple events and years for both tours;
do not tune to the 2026 US Open or the 53-market diagnostic sample.

Compare identical eligible matches, settlement rules, and where practical
common random numbers across:

1. frozen v1.0;
2. temperature calibration only;
3. surface Elo integration;
4. dynamic anchor integration with constant weight;
5. dynamic anchor integration with reliability gating;
6. optional ranking, level-change, and head-to-head ablations.

### J2. Primary winner gates

Report paired differences and event/time-block bootstrap intervals for:

- Brier score;
- log loss;
- calibration intercept and slope;
- reliability by probability band;
- discrimination;
- high-confidence error rate;
- worst-decile Brier score;
- ATP/WTA and competition-level slices;
- sparse versus well-observed players;
- anchor-component sign disagreements.

Promotion requires a practically meaningful winner improvement whose paired
uncertainty does not indicate a material regression, improved or non-worsened
calibration, and evidence that tail failures decline across more than one event.
A gain produced entirely by the three motivating examples is insufficient.

### J3. Ace and double-fault non-inferiority

For ace and double-fault comparison props, expected counts, and primitive
rates, report paired Brier, calibration, count residual, dispersion, zero mass,
tie rate, and exposure-stratified diagnostics.

The default non-inferiority margin is an increase of `0.001` in mean Brier for
each protected core prop family, matching the v1.0 standard for adopting a
probability-affecting dependence correction. The upper confidence bound for
the v1.1-minus-v1.0 change must not exceed that margin. A pre-registered
backtest plan may justify a different margin before outcomes are examined.

Failure of either protected family blocks promotion of the complete candidate,
even if winner prediction improves.

### J4. Coherence and numerical gates

Required checks include:

- player-order reversal negates every component, anchor, and target logit;
- positive point tilt weakly increases the favored player's exact-scoring win
  probability under coupled randomness;
- the root solver attains its target within configured tolerance;
- saturation is rare, reported, and not concentrated in a player group;
- `F`, `A`, and `D` draws are bitwise unchanged under coupled v1.0/v1.1 RNG;
- differences in ace/DF counts reconcile to changed exposure;
- winner probability equals exact-score probability mass from integrated paths;
- all v1.0 serve identities and scoring properties still hold;
- no market or future information reaches a fitted artifact;
- repeated runs with the same artifacts and seed reproduce outputs.

### J5. Promotion outcomes

The validation report must conclude one of:

- **pass:** all winner, protected-prop, coherence, and reproducibility gates
  pass, permitting an explicit production-promotion decision;
- **fail:** at least one gate fails, leaving v1.0 as production default;
- **inconclusive:** uncertainty is too wide or data coverage is insufficient,
  leaving v1.0 as production default while identifying the needed evidence.

No automatic promotion is permitted.

---

## K. Implementation plan

### K1. Milestone 1: data and anchor

1. Add timestamped outcome eligibility and optional ranking snapshot schemas.
2. Implement competition-graph diagnostics.
3. Implement the tour-specific dynamic Bradley-Terry fit and posterior draws.
4. Implement a conventional surface Elo diagnostic.
5. Add cutoff, player-order symmetry, state-transition, and reproducibility
   tests.

### K2. Milestone 2: historical cross-fitting

1. Generate strict rolling-origin v1.0 component probabilities.
2. Generate matching anchor predictions without future data.
3. Persist a reusable out-of-fold integration table.
4. Implement instability and sparsity diagnostics without outcome peeking.
5. Fit constrained temperature and reliability models.

### K3. Milestone 3: path integration

1. Implement the deterministic exact-scoring probability operator used by the
   tilt solver.
2. Implement symmetric `Q1`/`Q2` tilting and bracketed root finding.
3. Add anchor and integration uncertainty streams.
4. Extend match parameters, snapshots, locks, and reports.
5. Run causal, identity, monotonicity, numerical, and coupled-RNG tests.

### K4. Milestone 4: rolling backtest

1. Freeze the backtest plan and non-inferiority margins.
2. Run nested rolling-origin model and feature ablations.
3. Measure winner improvement, tail behavior, and protected-prop
   non-inferiority.
4. Produce fold-level forecasts, aggregate metrics, uncertainty intervals, and
   failure analysis.
5. Issue pass, fail, or inconclusive recommendation without changing criteria.

### K5. Milestone 5: optional promotion

Only after a pass and explicit approval:

1. change framework status from candidate to production;
2. update repository guardrails to recognize v1.1 as an authorized framework;
3. freeze production configuration and artifacts;
4. create new v1.1 locks for future matches;
5. retain v1.0 code, artifacts, and historical locks unchanged.

Each milestone ends with implementation summary, tests and results, deviations,
ambiguities, principal files, and readiness for the next milestone.

---

## L. Migration and changelog

### L1. Migration from v1.0

No data migration mutates raw snapshots or historical locks. New derived
outcome-state tables and strength artifacts are append-only. Existing v1.0
component artifacts may be referenced by a v1.1 experimental snapshot when
their cutoff, schema, and code hashes are compatible.

The v1.0 settlement policy, prop parser, scoring engine, duration model, rally
model, and retirement model require no probability-affecting migration.

### L2. Probability-affecting changelog

| Area | v1.0 | v1.1 candidate |
|---|---|---|
| Overall strength | external diagnostic only | dynamic outcome anchor |
| Winner calibration | raw joint simulation | cross-fitted temperature and anchor integration |
| Q1/Q2 | component posterior and beta performance draw | same draw plus symmetric strength tilt |
| F/A/D | primitive posterior and beta performance draw | unchanged |
| Recent trajectory | exponential component weighting | weighting plus dynamic anchor state |
| Ranking | diagnostic only | permitted sparse-player prior after validation |
| Head-to-head | diagnostic only | permitted strongly shrunk interaction after validation |
| Joint dependence | conditional primitive independence | strength-induced Q1/Q2 and direction dependence |
| Locks | component artifacts | component, anchor, integration, and disagreement artifacts |

### L3. Expected prop effects

| Prop family | Expected direct effect |
|---|---|
| Match winner | primary intended improvement |
| Exact score / sets / games | changes coherently with integrated point strength |
| Tiebreaks | may change through hold and matchup balance |
| Duration | may change through games, sets, and points |
| Ace propensity | none |
| Double-fault propensity | none |
| Ace/DF match counts | possible exposure-only change |
| Rally winner/error props | possible playable-point and exposure change |
| Settlement | none |

### L4. Explicit unresolved empirical decisions

The following must be fixed in the pre-registered backtest plan, not chosen
after outer results:

1. dynamic-state decay and innovation candidate sets;
2. exact instability-score normalization;
3. integration shrinkage strength;
4. numerical root tolerance and probability bounds;
5. whether ranking improves sparse-player initialization;
6. whether a level-change process adds stable value;
7. whether head-to-head shrinkage remains exactly zero;
8. whether equal Q1/Q2 tilt weights should remain fixed;
9. final practical non-inferiority margins if different from Section J3.

These are validation choices, not permission to alter the architecture or tune
to one tournament.

---

## M. Candidate freeze decision

This document is sufficiently explicit to implement and test the proposed
v1.1 framework without inventing its central statistical mechanism. It does
not establish that v1.1 is superior to v1.0.

The candidate is ready for implementation and rolling-origin backtesting when:

- the unresolved candidate grids and numerical tolerances are pre-registered;
- historical cutoff-safe outcome coverage is audited;
- the backtest's protected-prop non-inferiority rules are frozen.

Until the validation report passes and the user explicitly authorizes
promotion, the operative decision is:

**TENNIS MODEL v1.1 IS SPECIFIED AS A BACKTEST CANDIDATE, NOT FROZEN FOR
PRODUCTION.**
