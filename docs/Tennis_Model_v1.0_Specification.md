# Tennis Model v1.0

**US Open Prediction Championship — statistical and implementation specification**  
**Specification date:** 28 August 2026  
**Status:** design-complete; approved for freeze after the v1.0-rc1 targeted review; no individual match has yet been locked under this model

This document is the clean, consolidated Tennis Model v1.0 design incorporating the accepted v1.0-rc1 review. It distinguishes the **framework version** (`Tennis Model v1.0`) from a **match lock**, which is a timestamped snapshot of player information, parameters, simulation output, and exact championship questions for one match.

Required-output map: **A. Revision summary** is in Section A; **B. Revised specification** is the consolidated Sections B–H; **C. Validation additions** are explicit in Section I3; **D. Updated Codex handoff** is Section J; and **E. Freeze decision** is Section L.

The championship facts used here are current as of the specification date. SportsPredict states that the event contains 1,280 binary probability questions over 190 matches, uses cumulative Relative Brier Points with round multipliers, accepts integer forecasts from 1 through 99, permits revisions until the scheduled match start, and settles from the official US Open match tracker and statistics. Its public pages also identify the central question families: tiebreaks, duration, ace and double-fault comparisons, unforced-error thresholds, first-serve point-win percentage, winner comparisons, first-set game counts, lopsided set scores, and deciding sets. See the [challenge overview](https://sportspredict.com/probability/grand-slam), [scoring rules](https://sportspredict.com/probability/grand-slam/scoring), [settlement FAQ](https://sportspredict.com/probability/grand-slam/faq), and [API documentation](https://sportspredict.com/probability/api).

---

## A. Revision summary and executive design decision

The v1.0-rc1 review makes four substantive amendments:

1. **Serve-event ordering:** aces are now generated conditional on a first serve landing in and determine an immediate server win; double faults are generated conditional on a second-serve opportunity and determine an immediate server loss. Returnable first-serve and playable second-serve outcomes are modeled separately. This lets exceptional ace and double-fault performance change overall service-point performance rather than merely relabel an outcome already drawn.
2. **Component estimands and identities:** the five core serve components are now first-serve-in rate, ace propensity, returnable first-serve effectiveness, double-fault propensity, and playable second-serve effectiveness. Their observable denominators, reconstruction identities, zero-denominator policy, anomaly checks, interfaces, and generator tests are explicit.
3. **Cross-component dependence gate:** rolling historical validation must estimate out-of-sample residual dependence among the five fitted components before any shared “serving-day” effect is introduced. Conditional independence remains the v1.0 default unless material, stable, predictively useful dependence is demonstrated.
4. **Historical ingestion:** versioned Jeff Sackmann ATP/WTA structured match files are the preferred core historical input where a provenance and coverage audit passes. Official US Open pages remain authoritative for current-tournament updates and settlement; Tennis Abstract and official ATP/WTA aggregates become cross-checks; official-major and cautiously used Match Charting Project data remain the auxiliary-stat sources. The implementation order and typed serve interfaces are updated accordingly.

No change is made to the point → game → set → match architecture, opponent-adjustment philosophy, opportunity-conditioned winner/unforced-error layer, exposure-conditioned duration model, retirement generator, prop ontology, settlement separation, immutable locks, append-only versioning, or ban on market probabilities as inputs.

Retain the accepted **point → game → set → match** architecture, with the targeted serve-event correction described below.

First, a service point is not a single Bernoulli trial. It is generated through a small, constrained serve mechanism:

1. first serve in or out;
2. after a first serve lands in, ace or non-ace; an ace wins immediately, while a non-ace proceeds to a returnable first-serve outcome;
3. after a first serve misses, double fault or playable second serve; a double fault loses immediately, while a playable second serve proceeds to its point outcome;
4. for every returnable/non-double-fault point, an official-stat termination category: winner, unforced error, or other.

This nesting guarantees logical identities—an ace is a won service point and a double fault is a lost service point—while making the match-level ace and double-fault draws directly affect service-point performance and all downstream score and exposure distributions.

Second, match duration cannot be obtained credibly from scoring alone. v1.0 therefore attaches a conditional duration model to the realized points, games, sets, tiebreaks, players, and conditions. It does not simulate every rally second-by-second.

The matchup parameters are estimated with separate ATP and WTA hierarchical, time-weighted models. For returnable first-serve and playable second-serve outcomes, player serving effects and opponent returning effects are fitted jointly on the logit scale. This is the default opponent adjustment; raw percentages are never inserted directly into a matchup or averaged naively. First-serve-in rate is principally a server attribute. Ace propensity includes server and returner effects; double-fault propensity is server-led. Hard-court player effects are partially pooled toward all-surface effects. Surface Elo is an external diagnostic and match-winner benchmark, not an additional predictor layered on top of the same match history.

Use a three-year rolling history with a one-year exponential half-life as the v1.0 default. Completed matches at the current US Open enter with their actual point exposure and ordinary recency weight; they receive no arbitrary “tournament form” multiplier. A tournament-wide court/conditions effect is updated from all completed US Open matches. This lets the event teach us about court speed without letting one player’s first-round match overwhelm an established record.

Routine simulation includes exact scoring, serve order, standard and deciding-set tiebreak rules, parameter uncertainty, between-match performance variation, and retirement/void semantics. It does **not** routinely include momentum, player-specific clutch, head-to-head effects, set-score psychology, or player-specific tiebreak skill. Global fatigue or format effects enter only if rolling backtests improve out-of-sample calibration; credible current injury or endurance concerns are handled as explicit information scenarios.

The rally layer is deliberately modest. On each eligible point won by player $i$ over player $j$, it chooses among a rally winner by $i$, an unforced error by $j$, and “other” using a heavily shrunk Dirichlet-multinomial model fitted to official-major data. It is not a shot-by-shot physics model. The public [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) is useful for priors and diagnostics, but its volunteer-selected coverage and scorer variation make it unsuitable as the sole calibration source for official US Open winner/error props.

All props are deterministic functions of simulated paths plus one versioned settlement policy. Comparison ties settle **No**; walkovers void every question; after a retirement, the match result settles but other questions settle only if their scope was completed or the event had already become irreversible. Those are confirmed championship rules, not sportsbook conventions.

Every ordinary forecast is the nearest permitted integer to the simulated probability, clipped to 1–99. The model never shades a probability to “beat the crowd”: Brier scoring remains proper under relative scoring. A match lock may be superseded before first serve when new information arrives, but it is never silently mutated, and prior submissions remain attached to their original lock and model version.

---

## B. Tennis Model v1.0 mathematical specification

### B1. Observational units and notation

Fit ATP and WTA models separately. For historical match $m$, let $i$ be the server/player whose service statistics form the row and $j$ the opponent. Define:

| Component $k$ | Successes $y_{mik}$ | Trials $n_{mik}$ | Match quantity modeled |
|---|---:|---:|---|
| $F$ | first serves in | service points | $f_i=P(\text{first serve in})$ |
| $A$ | aces | first serves in | $a_{ij}=P(\text{ace}\mid\text{first serve in})$ |
| $Q1$ | first-serve points won minus aces | first serves in minus aces | $q_{1,ij}=P(\text{server wins}\mid\text{first serve in, non-ace})$ |
| $D$ | double faults | service points minus first serves in | $d_i=P(\text{double fault}\mid\text{second-serve opportunity})$ |
| $Q2$ | second-serve points won | service points minus first serves in minus double faults | $q_{2,ij}=P(\text{server wins}\mid\text{second serve enters play})$ |

Here a **second-serve opportunity** is a service point on which the first serve missed. A **playable second serve** is such an opportunity that did not end in a double fault. The double-fault component is server-led; an opponent term is permitted only if it improves rolling held-out prediction and remains stable. Rows with a zero denominator or a missing official component are omitted only from that component’s likelihood, never recorded as a zero rate.

Ordinary official match totals produce the historical component estimates

\[
\widehat a=
\frac{\text{aces}}{\text{first serves in}},
\qquad
\widehat q_1=
\frac{\text{first-serve points won}-\text{aces}}
{\text{first serves in}-\text{aces}},
\]

\[
\widehat d=
\frac{\text{double faults}}
{\text{service points}-\text{first serves in}},
\qquad
\widehat q_2=
\frac{\text{second-serve points won}}
{\text{service points}-\text{first serves in}-\text{double faults}}.
\]

For an individual source row, reject only the affected component if its denominator is negative, its successes are outside $[0,n]$, or an accounting identity fails. In particular, flag `aces > first_serves_in`, `first_serve_points_won < aces`, `double_faults > service_points - first_serves_in`, or `second_serve_points_won > service_points - first_serves_in - double_faults`. Preserve the raw row and anomaly code; do not clip, silently repair, or impute it. A zero trial count supplies no likelihood contribution for that component but may remain usable for the other components.

Standard aggregate scorecards do not identify the serve number of each ace. The equations therefore use the ordinary-stat convention that reported aces are assigned to first serves; rare second-serve aces are a measurement limitation to quantify against point-level samples, not a reason to change the v1.0 generator. A source that reliably identifies serve number may apply a versioned correction before constructing these counts.

The derived first- and second-serve point-win probabilities are

\[
w_{1,ij}=a_{ij}+(1-a_{ij})q_{1,ij},
\qquad
w_{2,ij}=(1-d_i)q_{2,ij}.
\]

The expected unconditional service-point win probability is therefore

\[
p^{\mathrm{srv}}_{ij}
=f_i\{a_{ij}+(1-a_{ij})q_{1,ij}\}
+(1-f_i)(1-d_i)q_{2,ij}.
\]

Expected ace and double-fault rates per service point are

\[
r^{A}_{ij}=f_i a_{ij},\qquad
r^{D}_{ij}=(1-f_i)d_i.
\]

### B2. Hierarchical opponent adjustment

For each component, use a time-weighted beta-binomial generalized linear mixed model. The conditional match mean is linked by

\[
\begin{aligned}
\operatorname{logit}\mu^F_{mi} &= \alpha_F+u^F_i+x_m^\top\beta_F,\\
\operatorname{logit}\mu^A_{mij} &= \alpha_A+s^A_i-r^A_j+x_m^\top\beta_A,\\
\operatorname{logit}\mu^{Q1}_{mij} &= \alpha_{Q1}+s^{Q1}_i-r^{Q1}_j+x_m^\top\beta_{Q1},\\
\operatorname{logit}\mu^D_{mi} &= \alpha_D+u^D_i+x_m^\top\beta_D,\\
\operatorname{logit}\mu^{Q2}_{mij} &= \alpha_{Q2}+s^{Q2}_i-r^{Q2}_j+x_m^\top\beta_{Q2}.
\end{aligned}
\]

Positive $s$ denotes better serving; positive $r$ denotes better returning or ace suppression and therefore reduces the server’s probability. $x_m$ contains only predeclared low-dimensional context: indoor/outdoor hard court, a shrunk event-year speed effect, and—only if validated—global handedness or best-of-five indicators. It does not contain ranking, Elo, “form,” and raw component statistics as additional overlapping predictors.

Player effects receive zero-centered Gaussian priors, with identifiability supplied by those priors and explicit centering after fitting. Estimate surface pooling jointly so that every historical row enters the likelihood exactly once. Write the hard-court effect as a global player effect plus a shrunk hard deviation:

\[
b^{\mathrm{hard}}_{ik}=b^{\mathrm{global}}_{ik}+\delta^{\mathrm{hard}}_{ik},
\qquad \delta^{\mathrm{hard}}_{ik}\sim N\!\left(0,\tau^2_{k,\mathrm{surface}}\right).
\]

Non-hard observations identify the shared global effect and their own surface deviations; they are not reused as an empirical prior after the hard observations have already entered the likelihood.

For component $k$, the beta-binomial likelihood is

\[
y_{mik}\mid n_{mik},\mu_{mik},\kappa_k
\sim \operatorname{BetaBinomial}
\left(n_{mik},\;\kappa_k\mu_{mik},\;\kappa_k(1-\mu_{mik})\right).
\]

Its concentration $\kappa_k$ represents genuine match-to-match variation beyond binomial point randomness. MAP fitting uses the weighted log posterior

\[
\mathcal L_k=\sum_m \omega_{mk}\log p(y_{mik}\mid n_{mik},\mu_{mik},\kappa_k)+\log p(\alpha_k,\beta_k,b_k),
\]

where the default time weight is

\[
\omega_{mk}=2^{-\Delta t_m/365},\qquad 0\leq \Delta t_m\leq1095\ \text{days}.
\]

One half-life is the v1.0 default to limit tuning. A component-specific half-life may replace it only after rolling-origin validation.

For a future matchup, draw a posterior parameter vector from the Laplace approximation around the MAP, compute the mean $\mu_k$, then draw each match-specific probability

\[
\phi_k\sim\operatorname{Beta}\{\kappa_k\mu_k,\kappa_k(1-\mu_k)\}.
\]

The posterior draw is **parameter uncertainty**; the beta draw is **new-match performance variation**. Both are drawn once per simulated match, not once per point. The baseline draws $(\phi_F,\phi_A,\phi_{Q1},\phi_D,\phi_{Q2})$ are conditionally independent given the fitted player, opponent, time, surface, and context terms. Section I3 tests this assumption out of sample. Do not introduce a multivariate performance model unless that test finds material, stable residual dependence and the simplest shared-factor correction improves held-out joint and prop prediction.

This is a simplified, implementation-oriented member of the point-based hierarchical family. A published reference model likewise estimates time-, surface-, tournament-, serve-, and return-specific latent skills and shows that such point models can be competitive for match prediction; see [Ingram (2019)](https://martiningram.github.io/papers/bayes_point_based.pdf). v1.0 uses exponential weighting and predictive overdispersion in place of a full Gaussian random walk because they are easier to update and audit during a two-week event.

### B3. Point generator

On a service point for $i$ against $j$, with match-specific draws $(\phi_F,\phi_A,\phi_{Q1},\phi_D,\phi_{Q2})$:

1. Draw $F\sim\mathrm{Bernoulli}(\phi_F)$.
2. If $F=1$, draw $A\sim\mathrm{Bernoulli}(\phi_A)$.
   - If $A=1$, record an ace and a server point win; the point ends immediately.
   - If $A=0$, draw $W\sim\mathrm{Bernoulli}(\phi_{Q1})$ for the returnable first-serve point, then send that realized winner/loser to the rally-stat layer.
3. If $F=0$, draw $D\sim\mathrm{Bernoulli}(\phi_D)$.
   - If $D=1$, record a double fault and a server point loss; the point ends immediately.
   - If $D=0$, draw $W\sim\mathrm{Bernoulli}(\phi_{Q2})$ for the playable second-serve point, then send that realized winner/loser to the rally-stat layer.

Thus an unusually high ace draw raises first-serve and overall service performance, while an unusually high double-fault draw lowers second-serve and overall service performance. Ace and double-fault draws are causes of their point outcomes, not labels attached afterward.

Every point record stores server, receiver, first-serve status, whether the point reached a returnable first serve or playable second serve, winner, ace, double fault, game/set score before and after, and whether it occurred in a tiebreak. This is sufficient to reconstruct official serve denominators and every scoring prop.

For a stationary service-point win probability $p$, the analytic hold probability used as a sanity check is

\[
h(p)=p^4\{1+4(1-p)+10(1-p)^2\}
+20p^3(1-p)^3\frac{p^2}{p^2+(1-p)^2}.
\]

Simulation, rather than this formula, is authoritative whenever the first/second-serve mechanism, tiebreaks, or state scenarios are active.

### B4. Rally winners and unforced errors

Let a non-ace/non-double-fault point be won by $i$ over $j$. Its official-stat termination category is

\[
C_{i\to j}\in\{W_i,\,UE_j,\,O\},
\]

where $O$ includes forced errors and all other endings. Historical directional counts use

\[
N_{i\to j}=\text{points won by }i-\text{aces}_i-\text{double faults}_j.
\]

After reconciling the official inclusion rules, subtract aces from published winners if winners include aces, and subtract double faults from published unforced errors if errors include double faults. The residual triplet is fitted as

\[
(W_i^{r},UE_j^{r},O)\sim\operatorname{DirichletMultinomial}
(N_{i\to j},\kappa_R\boldsymbol\pi_{i\to j}),
\]

with

\[
\begin{aligned}
\log(\pi_W/\pi_O)&=\gamma_W+a^W_i+d^W_j+z_m^\top\delta_W,\\
\log(\pi_{UE}/\pi_O)&=\gamma_U+e^U_j+p^U_i+z_m^\top\delta_U.
\end{aligned}
\]

The effects represent the winner’s aggression, the loser’s tendency to allow winners, the loser’s error tendency, and the winner’s pressure tendency. They are heavily pooled because official-major rally data are much thinner than serve data. For each simulated matchup direction, draw one $\boldsymbol\pi$ from its Dirichlet distribution and categorize every eligible point. This prevents a point from being counted simultaneously as both a winner and an unforced error.

Published totals are reconstructed by adding back aces and/or double faults only when the official US Open convention requires it. The convention is a versioned configuration value, never an implicit assumption.

### B5. Match duration

After a completed path produces total points $P$, regular/tiebreak games $G$, sets $S$, and tiebreaks $B$, simulate latent elapsed minutes from the robust conditional model

\[
T^*=\max\left\{1,
\theta_0+\theta_PP+\theta_GG+\theta_S S+\theta_BB
+q_i+q_j+c_m+\sigma(P)t_\nu
\right\},
\]

where $(q_i,q_j)$ are shrunk player-pace effects, $c_m$ contains roof/temperature/session and current-tournament effects when known, $t_\nu$ is a centered Student-$t$ residual, and $\sigma(P)=\sigma_0+\sigma_1\sqrt{P}$. Ridge priors prevent unstable allocation between the correlated exposure terms. Fit this only on matches with an official duration and point total.

The prop evaluator converts $T^*$ to the official published whole-minute value through a versioned rounding function. Until the tournament’s exact convention is reconciled, duration props within one minute of a threshold receive a sensitivity flag.

### B6. Retirement and walkover process

Estimate a player retirement probability from recent match-level retirement outcomes with strong tour-level beta-binomial shrinkage, then map it to a small per-completed-game competing hazard. Current credible injury, illness, heat, or workload evidence may multiply that hazard only through a named information scenario. v1.0 checks retirement at game boundaries; the resulting approximation is acceptable at normal base rates and must be stress-tested for visibly injured players.

A walkover occurs before any point and voids every market. It is handled outside the match simulator once the schedule changes. If a simulated retirement occurs, the retiring player loses the match, but the prop settlement layer separately determines whether each other market is settled or void.

For prop $q$, let $S_q=1$ mean “this market would be settled” and $Y_q\in\{0,1\}$ its outcome when settled. The forecast sent to the championship is

\[
\widehat p_q=
\frac{\sum_{b=1}^{N}\mathbf 1\{S_{qb}=1,Y_{qb}=1\}}
{\sum_{b=1}^{N}\mathbf 1\{S_{qb}=1\}},
\]

not the unconditional probability of “Yes” with void paths silently coded as “No.” Store $\widehat P(S_q=1)$ as a diagnostic.

### B7. Three uncertainty layers

| Layer | Routine treatment | Rule |
|---|---|---|
| Parameter uncertainty | Propagate | Draw the fitted matchup parameters once per simulated path from a Laplace approximation or calibrated bootstrap. |
| Match-path uncertainty | Propagate | Simulate points, scoring, statistics, duration, and retirement conditional on the parameter draw. |
| Information uncertainty | Scenario mixture or sensitivity | Blend only when scenario weights are defensible; otherwise publish the central estimate plus alternative-scenario probabilities. |

Examples of information scenarios are “roof open/closed,” “fully fit/limited,” or “starts normally/withdraws.” Vague narrative uncertainty is not converted into a parameter shift.

### B8. ATP and WTA separation

ATP and WTA have distinct intercepts, player effects, shrinkage variances, beta/Dirichlet concentrations, duration coefficients, and retirement baselines. The scoring engine changes best-of-three to best-of-five for men’s singles but shares audited tennis rules. A men’s best-of-five fatigue adjustment is not automatically imposed; it must beat the stationary model in historical major backtests. This is more than changing the number of sets, without requiring two unrelated architectures.

### B9. State-dependent effect policy

| Class | v1.0 treatment | Effects |
|---|---|---|
| **A — routine** | Always encode | exact serve order and score state in the rules engine; correct tiebreak service; tour/format; draw-once match performance variation; retirement hazard and settlement |
| **B — evidence-triggered** | Named information scenario or a prevalidated global coefficient | credible injury/medical limitation; exceptional heat/humidity; accumulated workload or short rest; documented late-match serve degradation; uncertain roof/conditions |
| **C — omitted** | No v1.0 adjustment | “momentum”; generic pressure/clutch; being ahead or behind by a set; player-specific tiebreak magic; narrative mental strength; score-state effects without stable held-out evidence |

The scoring engine necessarily knows whether a point is a break point, but v1.0 uses the same component probability there. Break-point conversion/saving and tiebreak records are validation targets, not extra “clutch” predictors that would reuse the same outcomes.

### B10. Central stochastic-component contract

| Component | Inputs | Fitted parameters | Distribution/transition | Outputs | Update rule |
|---|---|---|---|---|---|
| First serve in | server, date, hard context | intercept, server effect, context, κ | beta-binomial; beta match draw | first/second-serve opportunities | append completed match; refit/update at daily cutoff |
| Ace propensity | first serve in, server, returner, context | ace/anti-ace effects, context, κ | beta match draw; Bernoulli before point outcome | ace flag and immediate server win | same |
| Returnable first serve | first serve in and non-ace, server, returner, context | serve and return effects, context, κ | hierarchical beta-binomial; beta match draw | non-ace first-serve point outcome | same |
| Double-fault propensity | second-serve opportunity, server, context | server effect, context, κ | beta match draw; Bernoulli before playable outcome | DF flag and immediate server loss | same |
| Playable second serve | non-DF second-serve opportunity, server, returner, context | serve and return effects, context, κ | hierarchical beta-binomial; beta match draw | playable second-serve point outcome | same |
| Rally termination | eligible winner/loser, context | two multinomial logits, player effects, $\kappa_R$ | Dirichlet-multinomial/categorical | winners, UEs, other | update when official stats become available |
| Tennis scoring | ordered point winners, server order, format | none | deterministic state machine | games, breaks, sets, tiebreaks, winner | rule-version change only |
| Duration | simulated exposure, players, conditions | robust regression, pace effects, residual scale | truncated Student-t regression | latent and published minutes | update from official completed matches |
| Retirement | player, format, workload, health scenario | shrunk baseline and hazard multipliers | competing hazard at game boundaries | retirement time/player | update from official status and retirement history |
| Information scenario | verified condition/injury alternatives | explicit scenario weights/shifts | finite mixture or sensitivity runs | scenario-specific prop probabilities | change only on cited new information |

---

## C. Data and estimation specification

### C1. Source hierarchy

Use the following role-specific hierarchy, with every extraction cached, timestamped, checksummed, and named in the source manifest:

1. **Core historical serve/return modeling — structured Jeff Sackmann ATP/WTA match data:** use a pinned, provenance-verified snapshot of the public Sackmann match-file schema as the primary reproducible input wherever coverage passes validation. The structured winner/loser columns include, where available, score and metadata plus `svpt`, `1stIn`, `1stWon`, `2ndWon`, `ace`, `df`, `SvGms`, `bpSaved`, `bpFaced`, and `minutes`. They directly identify the five v1.0 serve denominators after normalization to two player-service rows. The inspected [ATP preserved repository](https://github.com/Kadantte/tennis_atp) documents integer match totals, missing-stat caveats, and the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license; its [2026 match file](https://github.com/Kadantte/tennis_atp/blob/master/atp_matches_2026.csv) exposes the expected schema.
2. **Current tournament update and settlement — official US Open:** the [official live-scores page](https://www.usopen.org/en_US/scores/index.html) and individual IBM SlamTracker/stat pages are authoritative for current-tournament updates, corrections, final settlement, scores, serve statistics, aces, double faults, winners, unforced errors, total points, and published duration. SportsPredict explicitly makes this the settlement source.
3. **Cross-checks and diagnostics — Tennis Abstract and official tours:** use [Tennis Abstract](https://www.tennisabstract.com/), the [ATP Stats hub](https://www.atptour.com/en/stats), and [WTA Stats Hub](https://www.wtatennis.com/stats) for player identity, aggregate-rate, and spot-row reconciliation. Do not make fragile live webpage scraping a required historical dependency when a validated structured row already contains the field. A webpage may fill a documented date/component gap only with explicit provenance and a source-specific data grade.
4. **Winners, unforced errors, and specialized rally statistics:** use sufficiently consistent official-major historical pages for target-label training. The public [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) may supplement priors, structural checks, and diagnostics, but not serve as the representative calibration sample; its repository describes volunteer, match-selected coverage, and the [Tennis Abstract winner/error documentation](https://tennisabstract.com/reports/winners_errors_leaders_men_career.html) describes incomplete mixed-source coverage.
5. **News and conditions:** official player/tournament statements first; credible reporting second. Store the source, publication time, observation time, and exact modeled implication. Never turn rumors into silent parameter changes.

The structured-data recommendation has an operational qualification. As of this review, the former canonical `JeffSackmann/tennis_atp` and `JeffSackmann/tennis_wta` URLs return 404 and the [author’s GitHub profile](https://github.com/JeffSackmann) lists the Match Charting Project but not those two repositories. Therefore the ingestion job must not depend on an unpinned fork’s moving default branch. The manifest must preserve the upstream attribution, exact archive/fork URL, commit or immutable object identifier, retrieval time, file checksum, schema version, license text/version, and the verified coverage end date for ATP and WTA separately. Confirm that the intended use complies with the recorded license before using or redistributing the data.

The inspected schema supports the core count model but does **not** supply winners or unforced errors, rally/shot detail, serve-number-specific aces, roof/weather/session, exact retirement timing, or the US Open’s displayed rounding rules. `minutes` and match statistics are missing for some rows; score strings may mark a retirement without its point/game time. Current WTA snapshot availability and recency must be audited independently rather than inferred from ATP. For a date or component that fails this intake audit, retain the previous Tennis Abstract/official-history route only for that documented gap; do not silently substitute incompatible fields. These are data/operations issues, not changes to the model architecture.

Bookmaker odds and prediction-market probabilities are excluded from fitting and locking. If separately requested as a post-lock sanity check, record them in a quarantined field that the model code cannot consume.

### C2. Canonical match row

Each player-match row should contain:

`match_id, date, tour, event, event_year, level, round, surface, indoor, best_of, player_id, opponent_id, player_hand, opponent_hand, score, completed, retirement, walkover, service_points, first_serves_in, first_serve_points_won, second_serve_points_won, aces, double_faults, service_games, break_points_faced, break_points_saved, total_points_won, winners, unforced_errors, duration_minutes, source_url, retrieved_at`.

Derived denominators and official accounting flags live in a separate, versioned transformation layer. Never overwrite raw source columns.

For a normalized player-service row, construct the core component counts exactly as follows:

| Component | Successes | Trials |
|---|---:|---:|
| $F$ | `first_serves_in` | `service_points` |
| $A$ | `aces` | `first_serves_in` |
| $Q1$ | `first_serve_points_won - aces` | `first_serves_in - aces` |
| $D$ | `double_faults` | `service_points - first_serves_in` |
| $Q2$ | `second_serve_points_won` | `service_points - first_serves_in - double_faults` |

Sackmann-format match rows are winner/loser oriented; normalization emits one service row for each player with a common immutable `match_id` and the other player as `opponent_id`. Preserve the original columns, source row number, and winner/loser orientation for audit. Missing raw inputs yield missing derived counts, not zeros.

Required validation includes:

- unique player identities and symmetric opponent rows;
- score legality and agreement with games/sets;
- $0\leq y\leq n$ for every component, with zero-trial rows excluded only from that component fit;
- first- plus second-serve opportunities reconciling to service points;
- `first_serve_points_won = aces + returnable_first_serve_points_won` and `second_serve_opportunities = double_faults + playable_second_serve_opportunities` after transformation;
- aces not exceeding first serves in or first-serve points won; double faults not exceeding second-serve opportunities; second-serve points won not exceeding playable second-serve opportunities;
- rally triplets nonnegative after official-accounting reconciliation;
- duplicate and corrected-stat detection;
- no post-cutoff information in a historical lock or backtest.

### C3. Time, surface, and current-tournament weighting

The base window is the preceding 1,095 days. All observations receive the one-year half-life weight given in B2. Outdoor hard-court data identify the target surface deviation directly. Indoor hard is retained with an indicator. Non-hard data identify the shared player component and their own surface deviations in the same joint fit; they never enter as if they were equivalent US Open observations, and no row is counted twice.

Current-US-Open player matches receive their normal near-1 recency weight and their actual denominators. A first-round match with roughly 50 service points therefore updates a player much less than an established record containing thousands of weighted points. Within the same joint likelihood, all completed tournament matches also identify a shrunk event-year speed vector for the relevant primitive components—principally $A,Q1,Q2$, with $F$ or $D$ included only if historically supported; the duration fit has its corresponding event effect. Each row still enters each component only once. Freeze both data cutoff and event effect inside every lock.

Do not add a second “recent form” feature computed from the same matches. Recency already enters through the weighted likelihood. Match wins, ranking, and Elo are retained only as external diagnostics.

### C4. Shrinkage and effective sample size

Gaussian player priors and beta-binomial concentration provide the primary shrinkage. For component $k$, report weighted trials and two diagnostics:

\[
N^{w}_{ik}=\sum_m\omega_{mk}n_{mik},\qquad
M^{\mathrm{eff}}_{ik}=\frac{(\sum_m\omega_{mk}n_{mik})^2}
{\sum_m(\omega_{mk}n_{mik})^2}
\]

and the approximate information-equivalent trials

\[
N^{\mathrm{info}}_{ik}\approx
\sum_m\frac{\omega_{mk}n_{mik}}{1+(n_{mik}-1)\rho_k},
\qquad \rho_k=\frac{1}{\kappa_k+1}.
\]

$M^{\mathrm{eff}}$ diagnoses domination by a few matches; $N^{\mathrm{info}}$ discounts raw opportunities for match-level overdispersion. Neither is presented as an exact posterior sample size.

When hard-court exposure is thin, the hard effect shrinks toward the all-surface player effect and then the tour baseline. Surface Elo and ranking remain diagnostics even then; v1.0 does not reuse the same historical outcomes through an Elo-derived prior. A player too sparse for a defensible component estimate receives a lower data grade rather than an undocumented blend.

### C5. Opponent-adjustment alternatives

| Method | Merits | Failure mode | v1.0 role |
|---|---|---|---|
| Raw player percentage | trivial | confounds schedule strength and double-counts quality when mixed with opponent stats | reject |
| Average server and return percentages | easy | wrong probability geometry and no principled shrinkage | reject |
| Logit combination | coherent and easy | depends on already adjusted component estimates | fallback |
| Joint hierarchical serve/return logit | schedule-adjusted, shrunk, auditable | requires match-level counts and fitting | **default** |
| Elo-informed residual | helps match-winner prediction for sparse players | overlaps historical outcomes and cannot generate all props alone | diagnostic/benchmark only in v1.0 |

If a pre-fitted joint model is unavailable, the allowed fallback for component $k$ is

\[
\operatorname{logit}p_{A\text{ serves},B\text{ returns}}
=\operatorname{logit}p^{\mathrm{srv}}_{A,k}
+\operatorname{logit}p^{\mathrm{allow}}_{B,k}
-\operatorname{logit}p^{\mathrm{tour}}_k,
\]

followed by uncertainty-aware shrinkage. Here $p^{\mathrm{allow}}_{B,k}$ is the server success B allows after schedule adjustment. This adds each deviation from the tour baseline once.

### C6. Injury, inactivity, head-to-head, and conditions

- **Inactivity:** after 90 days without a match, inflate posterior uncertainty and mean-revert hard-specific deviations toward the all-surface prior as a function of the gap. Do not automatically downgrade the mean solely for absence.
- **Age and trajectory:** exponential time weighting captures gradual trajectory for observed players. Age may inform only a sparse-player prior or a backtest-validated global drift; it is not stacked on top of a well-observed recent record.
- **Return from injury:** create central/limited scenarios from verified information. The first one or two return matches update the ordinary likelihood but do not erase the pre-injury base.
- **Structural technical change:** downweight pre-change observations only after a documented change and enough post-change opportunity to estimate it with shrinkage. A single striking match is insufficient.
- **Head-to-head:** never a routine additive feature. Use it to diagnose a specific unmodeled interaction; any adjustment must state the mechanism, sample, recency, and prior shrinkage.
- **Workload:** store recent minutes, days of rest, and current-tournament minutes. It affects an information/fatigue scenario only when validated or match-specifically material.
- **Court/weather:** apply only measured global effects whose direction and scale were fitted historically. Otherwise report sensitivity rather than a narrative adjustment.
- **Break-point statistics:** retain for diagnostics and realized-prop accounting, but do not add saving/conversion percentages as independent strength features; they mostly reuse the service/return points already modeled.

### C7. Estimation and update schedule

Run a full rolling fit before the tournament, then a daily update after official corrections settle. For each match lock, use the most recent completed data cutoff and produce a content hash of data, configuration, code commit, and fitted artifact. A new match result updates future locks; it never alters past locks.

---

## D. Simulation specification

One simulated match is generated as follows.

1. **Choose information scenario.** Draw from explicit scenario weights if they are defensible; otherwise run scenarios separately.
2. **Draw parameter uncertainty.** Draw the relevant player/context linear predictors from the fitted posterior approximation.
3. **Draw match performance.** Draw $(F,A,Q1,D,Q2)$ probabilities from their beta predictive distributions and the two directional rally-category vectors from their Dirichlet predictive distributions. Draw them conditionally independently in v1.0; any later shared-factor model must first pass the I3 residual-dependence gate.
4. **Choose first server.** Use the known server if available. Before the toss, use 50/50 unless a validated player choice model exists; v1.0 has none.
5. **Initialize score.** Use best-of-three for women’s singles and best-of-five for men’s singles. Carry service order continuously across sets.
6. **Simulate each service point.** Apply B3, record all sufficient statistics, and—if eligible—draw its rally termination category.
7. **Advance a regular game.** Standard advantage scoring applies. Record every break-point opportunity at 0-40, 15-40, 30-40, or advantage receiver; a returner winning a service game is one break. A tiebreak is never a break of serve.
8. **Advance the set.** A player wins a set at six games with a two-game margin, except at 6-6. Use a seven-point, win-by-two tiebreak in non-deciding sets and the official ten-point, win-by-two deciding-set tiebreak. The US Open confirms the 10-point deciding-set rule at 6-6 in its [rules explainer](https://www.usopen.org/en_US/us-open-at-home/how_tennis_works_us_open_101.html).
9. **Record tiebreak service correctly.** The player due to serve starts with one point; service then alternates in blocks of two. All tiebreak service points count toward aces, double faults, serve percentages, winners, errors, and total points. The tiebreak contributes one game to the set score; its internal points do not count as games.
   After the tiebreak, the player who received its first point serves first in the next set.
10. **Check retirement.** At each completed game, apply the two competing retirement hazards. If one triggers, stop the path and mark the advancing player and partial statistics.
11. **End the match.** Stop at two set wins in best-of-three or three in best-of-five. Store exact score, set scores, first-set games, total games, tiebreaks, breaks, all point/stat totals, and whether a deciding set began.
12. **Generate duration.** For completed paths, draw duration from B5. For retired paths, apply the same exposure model to the partial path and mark it partial.
13. **Apply official accounting.** Convert latent rally counts to official winners/errors, round official percentages and duration according to the settlement configuration, and retain both raw and published representations.
14. **Evaluate every prop.** The evaluator returns `(settled, yes/no)` for that path. Aggregate only settled paths using B6.

Use 100,000 paths for a standard lock. At (p=0.5), this gives Monte Carlo standard error about 0.16 percentage points before void filtering. Increase to at least 400,000 when the estimate is below 3%, above 97%, has fewer than 50,000 settled paths, or lies within 0.5 percentage points of an integer-rounding boundary that could change the submitted forecast. Store the RNG algorithm, seed sequence, path count, settled count, and Monte Carlo error.

Sanity checks before accepting a lock:

- simulated mean hold agrees with the analytic formula under fixed parameters;
- fixed-parameter point simulations reproduce $w_1=a+(1-a)q_1$, $w_2=(1-d)q_2$, $p^{\mathrm{srv}}=fw_1+(1-f)w_2$, $r^A=fa$, and $r^D=(1-f)d$ within Monte Carlo tolerance;
- all match-score probabilities sum to one conditional on completion;
- winner probability equals the sum of that player’s exact-score cells;
- total aces increase with service-point exposure in conditional checks;
- holding other uniforms fixed, increasing ace propensity cannot turn a server win into a loss, and increasing double-fault propensity cannot turn a server loss into a win;
- every ace is a first serve in and a won service point; every double fault follows a missed first serve and is a lost service point; neither receives a rally category;
- simulated `first_serve_points_won` equals aces plus returnable first-serve wins, and simulated second-serve opportunities equal double faults plus playable second serves;
- set/game/tiebreak counts obey tennis identities;
- comparison events treat ties as No;
- no prop uses future data or a different parameter snapshot.

---
## E. Prop ontology and settlement dictionary

### E1. Canonical settlement policy

The following policy is authoritative for v1.0 unless SportsPredict publishes a question-specific override:

| Concept | Canonical interpretation |
|---|---|
| Source of truth | The final official US Open match tracker and match-stat page. Third-party discrepancies are ignored. |
| Published integers | Settle on the whole-number percentage or duration displayed by the official source, even if a reconstructed unrounded value crosses the threshold. |
| Comparison tie | For “A records more/fewer than B,” an exact tie is **No**. |
| Walkover | Match never begins; every question is void. |
| Retirement—match result | The player who officially advances is the winner; match-result questions settle normally. |
| Retirement—completed scope | A question about a completed set or other completed scope settles Yes or No from that scope. |
| Retirement—monotone threshold | If the stated threshold/condition was already reached, settle Yes. If it was not reached and the relevant match scope remained incomplete, void rather than settle No. |
| Tiebreak | Any official set tiebreak, including a deciding-set 10-point tiebreak at 6-6. A match tiebreak would count only if used in the relevant competition format. |
| Games in a tiebreak set | A 7-6 set contains 13 games. Tiebreak points do not add games. |
| Break of serve | Returner wins a regular service game. Winning points on the opponent’s serve during a tiebreak is not a break. |
| Straight sets | Winner loses no set: 2-0 in best-of-three or 3-0 in best-of-five. |
| Wins a set | At least one officially completed set credited to that player. |
| Deciding set/goes the distance | The deciding set begins: third set in best-of-three or fifth in best-of-five. |
| “At least X” / “X+” | Statistic (\ge X). |
| “More than X” / “Over X” | Statistic (>X); preserve half-points exactly. |
| “Fewer than X” / “Under X” | Statistic (<X). |
| Official correction | Amend the realized value and score with a new audit timestamp; never replace the submitted forecast or original lock. |

The tie, walkover, retirement, official-source, and published-rounding rules are confirmed on the [SportsPredict settlement FAQ](https://sportspredict.com/probability/grand-slam/faq). The exact duration rounding algorithm and official inclusion of aces/double faults within winners/errors require empirical reconciliation before the first affected market is locked.

Every market is stored twice: original text exactly as published and a typed `PropSpec`. Parsing is strict. If wording cannot be mapped without judgment, the system refuses automatic evaluation and requests a one-time semantic ruling that is then versioned.

### E2. Match and set props

| Canonical ID | Simulation event | Required path data | Retirement behavior |
|---|---|---|---|
| `MATCH_WIN(player)` | official advancing player is `player` | completion/retiring player | always settles after a started match |
| `EXACT_SCORE(player,a,b)` | set wins equal `(a,b)` | completed match score | void if retired before exact score determined |
| `STRAIGHT_SETS(player)` | player wins 2-0 or 3-0 | completed set score | a completed straight-sets win settles Yes; otherwise incomplete paths void |
| `PLAYER_WINS_SET(player)` | completed set wins ≥1 | set results | Yes once achieved; otherwise void on retirement |
| `DECIDING_SET` | set 3/5 begins | set sequence | Yes once begun; otherwise void on early retirement |
| `FIRST_SET_WIN(player)` | player wins completed first set | first-set result | completed first set settles Yes/No; earlier retirement voids |
| `SET_TIEBREAK(set_no)` | specified set reaches 6-6 and starts a tiebreak | set path | completed scope or reached condition settles; otherwise void |
| `ANY_TIEBREAK` | number of tiebreaks ≥1 | set paths | Yes once reached; otherwise void if match retires |
| `TIEBREAK_COUNT(op,k)` | compare number of tiebreaks with $k$ | set paths | completed match settles; reached lower-bound Yes may settle early |
| `ANY_LOPSIDED_SET` | any completed set is 6-0, 6-1, or 6-2 | exact set scores | Yes once achieved; otherwise void on retirement |
| `SET_SCORE(set_no,x,y)` | specified set ends at exact game score | set result | settle when set completes; void if it does not |

### E3. Game and break props

| Canonical ID | Simulation event | Notes |
|---|---|---|
| `FIRST_SET_GAMES(op,k)` | compare first-set games with $k$ | 7-6 contributes 13; settle when first set completes or monotone threshold is reached |
| `TOTAL_GAMES(op,k)` | compare total official games with $k$ | tiebreak set contributes 13; no phantom games after retirement |
| `GAME_HANDICAP(player,h)` | `games_player + h > games_opponent` | exact push/tie semantics must follow wording; normally full completion required |
| `PLAYER_GAMES(player,op,k)` | compare player’s won games with $k$ | opportunity generated by match path |
| `BREAK_COUNT(player,op,k)` | regular return games won by player | excludes tiebreak return points |
| `TOTAL_BREAKS(op,k)` | sum of both players’ breaks | same |
| `BOTH_BREAK` | each player records ≥1 break | joint path event |
| `FIRST_BREAK_TIMING(scope,op,k)` | first regular break occurs before/after specified game or set state | parser must encode whether boundary game is included |

### E4. Serve and official-stat props

| Canonical ID | Simulation event | Exposure/submodel |
|---|---|---|
| `PLAYER_ACES(player,op,k)` | compare simulated official ace count with $k$ | first serves in and ace draws on the same path |
| `TOTAL_ACES(op,k)` | compare both ace totals with $k$ | joint exposure |
| `ACE_COMPARE(A,B)` | `aces_A > aces_B`; tie is No | joint match path |
| `PLAYER_DF(player,op,k)` | compare double-fault count with $k$ | second-serve opportunities and DF draws on the same path |
| `TOTAL_DF(op,k)` | compare both DF totals with $k$ | joint exposure |
| `DF_COMPARE(A,B)` | `df_A > df_B`; tie is No | joint match path |
| `FIRST_SERVE_WIN_PCT(player,op,k)` | compare official published $100\times W_1/N_1$ with $k$ | first-serve numerator and denominator; apply official rounding |
| `WINNERS(player,op,k)` | compare official winner count with $k$ | rally termination plus ace accounting |
| `WINNER_COMPARE(A,B)` | `winners_A > winners_B`; tie is No | two directional rally processes |
| `UNFORCED_ERRORS(player,op,k)` | compare official UE count with $k$ | rally termination plus DF accounting |
| `TOTAL_UNFORCED_ERRORS(op,k)` | compare sum with $k$ | joint rally exposure |
| `UE_COMPARE(A,B)` | `UE_A > UE_B`; tie is No | joint path |

Full-match percentages and comparisons normally void after retirement because their final values remain mutable. Count thresholds already crossed settle Yes under the confirmed rule. A comparison is not treated as irreversible merely because one player leads when play stops.

### E5. Duration, compound, and tournament props

| Canonical ID | Evaluator |
|---|---|
| `DURATION_MIN(op,k)` | compare the official published whole-minute duration with $k$; use conditional duration simulation |
| `AND(q1,...,qr)` | Yes only when all component events are true on the same path; never multiply marginals |
| `OR(q1,...,qr)` | Yes when at least one component event is true on the same path |
| `REACH_ROUND(player,r)` | player officially advances far enough to appear in round $r$ |
| `WIN_TOURNAMENT(player)` | player wins final in a full bracket simulation |
| `WIN_QUARTER(player)` | player emerges from the specified draw quarter |

Compound settlement is evaluated as a logical expression over path-level component states `{true, false, unresolved}`. It settles only when the expression itself is determined under the championship retirement rule. Any compound wording not covered by the public FAQ receives a manual settlement-policy test before prediction.

Tournament props use one simulated draw per replicate. Every match calls the same matchup estimator and match simulator; realized winners populate later rounds. Known withdrawals and walkovers update the draw. Future court/weather and workload are either integrated over declared scenarios or omitted—not filled in with false certainty.

### E6. Unsupported or lower-confidence categories

The following need an additional model and are not inferred from generic point outcomes:

- rally length, net approaches, shot-type/direction, fastest serve, and distance covered require shot-tracking or dedicated official-stat models;
- medical timeouts require a health/event model;
- exact elapsed time of a particular set requires a set-level duration model rather than the match-duration layer;
- break-point conversion percentage can be generated from the point path, but player-specific “clutch” adjustments are not part of v1.0;
- any newly published official statistic must be added to the ontology with a denominator, accounting identity, settlement test, and backtest before automatic submission.

---

## F. Locked Match Card template

### F1. Initialization protocol

For every new lock, execute the same sequence:

1. confirm the official players, draw, round, scheduled start, and match status;
2. confirm best-of-three/best-of-five and current US Open scoring rules;
3. fetch the exact open SportsPredict market wording and closing time for that match;
4. freeze the information cutoff and validate the latest player/match data;
5. research verified health, inactivity, current-tournament workload, and recent workload;
6. record court, session, roof, and weather information that is known or materially uncertain;
7. update player component fits and the tournament-wide environment effect without future data;
8. estimate the two serving directions, compare against diagnostics, and resolve any identity/data anomaly;
9. simulate the joint match distribution, evaluate every parsed market, and run all consistency checks;
10. persist the immutable lock artifacts and render the card. Do not submit unless separately authorized.

A subsequent prop answer must reference this lock. New information first produces a `LockDiff`; only a material change creates the next revision. The diff lists the evidence, parameters changed, probability changes for every affected open market, and prior submissions that may need an authorized update.

### F2. Card template

```markdown
# LOCKED MATCH CARD

Lock ID: USO26-[ROUND]-[MATCH]-L[REV]
Framework: Tennis Model v1.0
Created (UTC): [timestamp]
Information cutoff: [timestamp]
Match: [Player A] v [Player B]
Draw / round: [ATP|WTA] singles / [round]
Scheduled start: [official time + uncertainty]
Format: [best of 3|best of 5]; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50) | known: [player]
Conditions scenario: [central]; alternatives: [if material]
Health/workload status: [verified concise summary]
Data/model hash: [hash]
Simulation: [N] paths; seed [id]; settlement policy [version]

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | --.-% | --.-% |
| Ace given first serve in | --.-% | --.-% |
| Returnable first-serve points won | --.-% | --.-% |
| Derived first-serve points won | --.-% | --.-% |
| Double fault given second-serve opportunity | --.-% | --.-% |
| Playable second-serve points won | --.-% | --.-% |
| Derived second-serve points won | --.-% | --.-% |
| Overall service points won | --.-% | --.-% |
| Implied hold probability | --.-% | --.-% |
| Ace rate / service point | --.-% | --.-% |
| Double-fault rate / service point | --.-% | --.-% |
| Weighted hard-court service ESS | --- | --- |

## Core simulated outputs

| Output | Player A / Yes | Player B / No |
|---|---:|---:|
| Match win | --% | --% |
| Exact scores | [cells] | [cells] |
| Expected service games | --.- | --.- |
| Expected breaks conceded | --.- | --.- |
| Expected aces | --.- | --.- |
| Expected double faults | --.- | --.- |
| Expected winners* | --.- | --.- |
| Expected unforced errors* | --.- | --.- |

Expected total games: --.-  
Total-games 10/50/90% quantiles: -- / -- / --  
At least one tiebreak: --%  
Deciding set: --%  
Expected match duration*: --- minutes  
Expected total breaks: --.-

*Lower-confidence attached submodel; include data grade A/B/C.

## Championship markets

| Market ID | Canonical prop | Probability | P(settled) | Submitted integer | Data grade |
|---|---|---:|---:|---:|---|
| ... | ... | --.-% | --.-% | -- | A/B/C |

## Audit and sensitivities

- Sanity checks: [pass/fail list]
- Material alternative scenario(s): [probability changes]
- Sparse or conflicting evidence: [only material items]
- Market semantics requiring review: [none/list]

LOCK STATUS: LOCKED
```

The starred models are shown only when the current questions require them. The card remains compact by placing full parameter intervals, source logs, and simulation diagnostics in the machine-readable lock artifact.

---

## G. Match-initialization prompt

> **Initialize and lock Tennis Model v1.0 for [PLAYER A] v [PLAYER B] at the 2026 US Open. Use only information available by [CUTOFF, or “now”]. Confirm the official matchup, round, scheduled start, format, current SportsPredict questions, health/workload, and relevant conditions. Update the player and tournament state, estimate the matchup parameters, run the joint simulation and required checks, then save and return the LOCKED MATCH CARD. Do not use market odds and do not submit predictions unless I separately ask.**

After an existing lock, the revision command is:

> **Review lock [LOCK ID] for genuinely new information. If nothing material changed, keep it. If a revision is warranted, show the new evidence, parameter and prop changes, affected prior predictions, and create the next lock revision without overwriting the old one.**

---

## H. Prop-response template

Routine response:

```text
Prop: [canonicalized Yes event]
Probability: 63%
Model: Tennis Model v1.0 · Lock [LOCK ID]
Derivation: joint locked-match simulation
```

Add only when material:

```text
Settlement probability: 98% (retirement/void exposure)
Key sensitivity: 59% if [specific credible alternative]; central case remains 63%.
Data grade: C — official-error history is sparse.
```

For direct championship submission, convert the raw simulated probability to the nearest integer and clamp to 1–99. Preserve both raw and submitted values in the ledger. Never report a range in place of the required forecast.

---

## I. Calibration ledger schema and validation

### I1. Append-only prediction ledger

Maintain one primary row per championship question. Use child tables for submission revisions and official-stat corrections so the primary identity never changes.

**Question and event identity**

| Field | Type / meaning |
|---|---|
| `prediction_id` | internal immutable UUID |
| `championship_event_id`, `lobby_id` | SportsPredict identifiers |
| `market_id`, `match_id` | SportsPredict identifiers |
| `market_text_original` | exact published wording |
| `prop_type`, `prop_spec_json` | canonical typed representation |
| `tour`, `round`, `stage_multiplier` | ATP/WTA and scoring stage |
| `player_a_id`, `player_b_id` | canonical player identities |
| `scheduled_start_utc`, `market_close_utc` | lock deadline metadata |
| `created_at_utc` | first ledger insertion |

**Model provenance**

| Field | Type / meaning |
|---|---|
| `framework_version` | `Tennis Model v1.0` or later |
| `lock_id`, `lock_revision` | immutable match snapshot |
| `information_cutoff_utc` | latest information allowed in lock |
| `data_hash`, `fit_hash`, `config_hash`, `code_commit` | reproducibility chain |
| `settlement_policy_version` | ontology/rule dictionary version |
| `scenario_id`, `scenario_weights_json` | central or mixture specification |
| `simulation_paths`, `settled_paths`, `rng_seed_id` | Monte Carlo provenance |
| `mc_standard_error` | simulation error for raw probability |
| `parameter_summary_json` | compact matchup parameter reference |
| `data_grade` | A core; B conditional auxiliary; C sparse/fragile |
| `uncertainty_class` | routine / parameter-sensitive / information-sensitive |

**Forecast and submission**

| Field | Type / meaning |
|---|---|
| `probability_raw` | unrounded $P(Y=1\mid S=1)$ |
| `probability_settled` | (P(S=1)) |
| `probability_submitted` | integer 1–99 |
| `side` | normally `YES`; retain if UI reverses orientation |
| `forecast_status` | draft / locked / submitted / superseded / skipped |
| `skip_reason` | only for semantic failure, absent data, or no defensible model |
| `sensitivity_low`, `sensitivity_high` | named credible scenarios, not generic interval |
| `notes` | concise audit note |

**Resolution and scoring**

| Field | Type / meaning |
|---|---|
| `resolution_status` | pending / yes / no / void / corrected |
| `outcome_binary` | 1/0/null |
| `official_value_json` | exact official score/stat used |
| `official_source_url` | US Open source page |
| `resolved_at_utc`, `retrieved_at_utc` | timing |
| `brier_score` | $(p-y)^2$, using submitted $p$ unless explicitly analyzing raw model output |
| `brier_raw_model` | optional score using `probability_raw` |
| `log_loss` | diagnostic; clipped only at submitted 0.01/0.99 |
| `field_average_brier` | when made available |
| `rbp_unweighted` | $100(\overline{B}_{\mathrm{field}}-B_{\mathrm{user}})$ |
| `rbp_weighted` | stage multiplier times unweighted RBP |
| `correction_revision` | append-only official correction pointer |

The submission-history child table records `prediction_id, submission_revision, prior_probability, new_probability, submitted_at, API_prediction_id, reason, lock_id, response_status`. This preserves every pre-close revision. Only the final accepted submission before close is scored.

Use the confirmed round multipliers: 1× in Rounds 1–2, 2× in Round 3 and the Round of 16, and 3× in the quarterfinals, semifinals, and finals. The multiplier changes leaderboard impact, not the truthful probability or raw Brier calculation.

### I2. Dashboard metrics

Report:

- total questions forecast, settled, void, skipped, and revised;
- mean Brier score and Brier skill against predeclared simple baselines;
- cumulative unweighted and stage-weighted RBP when field data are available;
- optional mean log loss;
- reliability table/plot by broad probability buckets, with counts and bootstrap uncertainty;
- calibration intercept and slope from a held-out logistic calibration regression;
- sharpness/distribution of issued probabilities;
- predicted versus realized means and full count distributions for aces, double faults, winners, unforced errors, games, tiebreaks, and duration;
- results by prop family, ATP/WTA, round, favorite/close-match class, core versus auxiliary statistic, and data grade;
- raw-versus-submitted quantization loss and Monte Carlo error.

Tournament-only subgroup results are descriptive because cells will be small. Do not declare a model defect from one bad high-confidence outcome or a five-question subgroup. Use historical rolling backtests as the evidentiary base; use live standardized residuals and posterior-predictive checks as alerts for review, not automatic retuning.

### I3. Pre-tournament validation gates

Backtest with strict rolling cutoffs on prior US Opens and comparable outdoor hard-court events. At minimum use the 2022–2025 US Opens for official-stat questions, plus recent Australian Opens and the North American hard-court swing where compatible fields exist. Reconstruct championship-like binary questions at multiple thresholds without choosing thresholds after observing outcomes.

Compare v1.0 against predeclared baselines:

1. tour/event base rate;
2. time-weighted raw player statistics with simple logit combination;
3. surface Elo for match winner only;
4. stationary point model without parameter/performance uncertainty;
5. full v1.0.

Required gates:

| Target | Validation |
|---|---|
| Match winner | Brier/log loss, calibration intercept/slope, exact-score coherence, Elo benchmark |
| Hold/break | predicted versus observed hold and break rates by decile and tour |
| Set scores | probability mass over legal scores; observed frequencies of 7-6 and lopsided sets |
| Total/first-set games | PIT or randomized quantile checks; means and tails |
| Tiebreaks | any-TB and TB-count calibration by hold environment |
| Serve components | calibration and dispersion for $F,A,Q1,D,Q2$ on their observable denominators; derived $w_1,w_2,p^{\mathrm{srv}}$ identities |
| Aces/DFs | player and total means, dispersion, zero mass, comparison/tie rates, upper tails, and exposure-adjusted rates |
| First-serve win % | reconstructed and simulated $a+(1-a)q_1$, official rounding reconciliation, and threshold Brier scores |
| Winners/UEs | means, dispersion, tail coverage, player comparisons, scorer/event stability |
| Duration | residuals versus points/games/sets, threshold calibration, rounded boundary behavior |
| Retirement | overall/tour/player shrinkage, timing distribution, settlement-frequency calibration |
| Tournament | simulated advancement frequencies and bracket probability conservation |

#### Revised serve-generator validation

The revised ordering adds the following mandatory tests:

1. **Transformation identities:** on every valid historical row, reconstruct `first_serve_points_won = aces + Q1_successes`, `first_serves_in = aces + Q1_trials`, `second_serve_opportunities = double_faults + Q2_trials`, and `second_serve_points_won = Q2_successes`. Test missing and zero denominators component by component. An invalid component is quarantined with an anomaly code while unaffected components remain usable.
2. **Generator support:** on every simulated point, an ace implies first serve in and server win; a double fault implies first serve missed and server loss; ace and double-fault points never enter the rally termination process. A point cannot be both an ace and a double fault.
3. **Probability identities:** at fixed $(f,a,q_1,d,q_2)$, large point simulations must reproduce $w_1=a+(1-a)q_1$, $w_2=(1-d)q_2$, $p^{\mathrm{srv}}=fw_1+(1-f)w_2$, $r^A=fa$, and $r^D=(1-f)d$ within a tolerance defined from Monte Carlo standard error.
4. **Causal monotonicity under coupled randomness:** with the same uniform variates and all other inputs held fixed, increasing $a$ cannot reduce server point wins; increasing $d$ cannot increase them. The resulting game-hold and match-win probabilities must move in the same weak directions in high-precision simulation.
5. **Distributional validation:** rolling holdouts compare observed and posterior-predictive means, dispersion, zero mass, and tails for each primitive component and for the derived first-serve, second-serve, overall service, ace-count, and double-fault-count distributions. This verifies that the new event ordering improves more than algebraic coherence.
6. **Aggregate-stat limitation:** quantify the frequency and sensitivity of second-serve aces or other source accounting anomalies using point-level/official samples. Do not silently change the aggregate formulas; any correction is a versioned ingestion transform.

#### Cross-component match-day residual-dependence test

Before adding any dependence among the five match-performance draws, run this focused historical test separately for ATP and WTA:

1. Generate strict rolling-origin, out-of-fold posterior predictive distributions for each player-service-match row and each of $F,A,Q1,D,Q2$, using only data available before that match.
2. Conditional on each realized denominator, convert the observed beta-binomial count to a randomized quantile residual. Repeat the randomization with fixed recorded seeds as a sensitivity check. This removes the fitted player, opponent, surface, time, event, and other modeled context while respecting unequal and nested denominators; do not correlate raw percentages.
3. Estimate the residual Pearson and rank-correlation matrices overall and by chronological fold. Use event-block bootstrap intervals, with a player-cluster sensitivity analysis, and report stability by tour and surface/context slice where sample size permits.
4. Treat a pattern as a **candidate material dependence** only when at least one pooled pair has $|\rho|\geq0.10$, its 95% block-bootstrap interval excludes zero, and the same sign with $|\rho|\geq0.05$ appears in at least three of four chronological holdout folds. A small, sign-unstable, or in-sample-only correlation fails the gate.
5. If the residual gate passes, compare the independent model with one shared server-match factor implemented as a one-factor copula over the already calibrated beta marginals. For component $k$, one admissible construction is

   \[
   Z_m,\epsilon_{mk}\stackrel{\mathrm{iid}}{\sim}N(0,1),\qquad
   U_{mk}=\Phi\!\left(\lambda_k Z_m+\sqrt{1-\lambda_k^2}\,\epsilon_{mk}\right),
   \qquad
   \phi_{mk}=F^{-1}_{\mathrm{Beta}(\kappa_k\mu_{mk},\kappa_k(1-\mu_{mk}))}(U_{mk}),
   \]

   with $|\lambda_k|\leq1$, one loading sign fixed for identification, and irrelevant loadings shrunk toward zero. This preserves each beta marginal and adds only one latent serving-day dimension.
6. Adopt that correction only if it improves held-out joint log predictive density with a paired block-bootstrap interval above zero, improves at least one affected core-prop calibration/Brier target, and worsens no core prop family by more than 0.001 mean Brier. Otherwise retain conditionally independent component draws. Do not fit an unrestricted multivariate random-effects model unless a later version demonstrates additional stable predictive gain over the one-factor candidate.

The v1.0 default is conditional independence. A positive result obtained after freeze is evidence for a versioned v1.1 change, not a reason to keep this design open.

Also perform deterministic property tests for every scoring and settlement identity. A methodological option enters v1.x only if it improves a relevant held-out metric across more than one event or fixes a demonstrated calibration defect without materially harming other prop families. Do not tune dozens of half-lives, state effects, or player-specific adjustments on four US Opens.

### I4. Live model-failure review

For each family, monitor cumulative forecast residuals (\sum(y-p)), cumulative Brier difference from the simple baseline, and distributional residuals for the underlying counts. A review should ask in order:

1. Is the apparent error compatible with forecast variance and sample size?
2. Is it a settlement/parser or official-stat accounting error?
3. Did event conditions shift for the whole field?
4. Is one player or one extreme match driving the signal?
5. Does the same pattern appear in prior-event holdout data?

Only after these checks should the framework be revised. A change that affects open predictions creates a new model version or lock revision and an explicit affected-market list.

---

## J. Implementation specification for Codex

### J1. Repository architecture

```text
tennis-model/
├── pyproject.toml
├── README.md
├── config/
│   ├── model_v1.yaml
│   ├── settlement_v1.yaml
│   └── sources.yaml
├── data/
│   ├── raw/                 # immutable, timestamped source snapshots
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── fits/                # versioned fitted models
│   ├── locks/               # immutable lock JSON + rendered card
│   └── backtests/
├── src/tennis_model/
│   ├── schemas.py
│   ├── identity.py
│   ├── data/
│   │   ├── ingest_sackmann.py
│   │   ├── ingest_usopen.py
│   │   ├── ingest_official_tours.py
│   │   ├── crosscheck_tennis_abstract.py
│   │   ├── source_manifest.py
│   │   ├── normalize.py
│   │   └── validate.py
│   ├── estimation/
│   │   ├── weighted_glmm.py
│   │   ├── serve_components.py
│   │   ├── dependence_diagnostics.py
│   │   ├── rally_model.py
│   │   ├── duration_model.py
│   │   ├── retirement_model.py
│   │   ├── event_update.py
│   │   └── posterior.py
│   ├── simulation/
│   │   ├── point.py
│   │   ├── scoring.py
│   │   ├── match.py
│   │   └── tournament.py
│   ├── props/
│   │   ├── ontology.py
│   │   ├── parser.py
│   │   ├── evaluators.py
│   │   └── settlement.py
│   ├── locking/
│   │   ├── initialize.py
│   │   ├── card.py
│   │   ├── revise.py
│   │   └── provenance.py
│   ├── championship/
│   │   ├── client.py
│   │   ├── market_sync.py
│   │   ├── submission.py
│   │   └── results_sync.py
│   ├── calibration/
│   │   ├── ledger.py
│   │   ├── backtest.py
│   │   ├── metrics.py
│   │   └── dashboard.py
│   └── cli.py
└── tests/
    ├── test_scoring_properties.py
    ├── test_component_counts.py
    ├── test_serve_generator_properties.py
    ├── test_component_dependence.py
    ├── test_prop_semantics.py
    ├── test_retirement_settlement.py
    ├── test_reproducibility.py
    └── fixtures/
```

Use Python with `pandas`/`polars` for tabular work, `numpy` and `scipy` for fitting and simulation, `pydantic` or frozen dataclasses for schemas, `pyarrow`/Parquet for data, and SQLite or DuckDB for the ledger. Keep the scoring state machine dependency-free and pure. Vectorize independent paths or JIT only after correctness tests pass.

### J2. Core typed interfaces

```python
@dataclass(frozen=True)
class MatchContext:
    player_a_id: str
    player_b_id: str
    tour: Literal["ATP", "WTA"]
    event: str
    round: str
    scheduled_start_utc: datetime
    best_of: Literal[3, 5]
    surface: Literal["hard"]
    indoor: bool | None
    conditions: dict[str, Any]
    information_cutoff_utc: datetime


@dataclass(frozen=True)
class ModelSnapshot:
    framework_version: str
    fitted_at_utc: datetime
    data_cutoff_utc: datetime
    component_artifact_ids: dict[str, str]
    data_hash: str
    config_hash: str
    code_commit: str


@dataclass(frozen=True)
class ServePerformanceDraw:
    first_serve_in: float  # f
    ace_given_first_in: float  # a
    returnable_first_win: float  # q1
    double_fault_given_second_opp: float  # d
    playable_second_win: float  # q2

    @property
    def first_serve_win(self) -> float:
        return self.ace_given_first_in + (1.0 - self.ace_given_first_in) * self.returnable_first_win

    @property
    def second_serve_win(self) -> float:
        return (1.0 - self.double_fault_given_second_opp) * self.playable_second_win

    @property
    def service_point_win(self) -> float:
        return (
            self.first_serve_in * self.first_serve_win
            + (1.0 - self.first_serve_in) * self.second_serve_win
        )


@dataclass(frozen=True)
class PerformanceDependenceSpec:
    mode: Literal["independent", "one_factor_beta_copula"]
    loadings: dict[str, float]  # empty for frozen v1.0 default
    validation_artifact_id: str | None


@dataclass(frozen=True)
class MatchParameterDistribution:
    context: MatchContext
    component_posteriors: dict[str, Any]  # keys: F, A, Q1, D, Q2
    predictive_concentrations: dict[str, float]
    performance_dependence: PerformanceDependenceSpec
    rally_posteriors: dict[str, Any]
    duration_posterior: Any
    retirement_scenarios: tuple[Any, ...]
    provenance: dict[str, str]


@dataclass(frozen=True)
class MatchPath:
    completed: bool
    retired_player_id: str | None
    winner_id: str
    sets: tuple[Any, ...]
    points: Any  # compact sufficient-stat or optional full trace
    stats: dict[str, Any]
    duration_latent: float
    duration_official: int


@dataclass(frozen=True)
class PropSpec:
    kind: str
    subject_ids: tuple[str, ...]
    operator: str | None
    threshold: float | None
    scope: dict[str, Any]
    original_text: str
    settlement_policy_version: str


@dataclass(frozen=True)
class PropEstimate:
    prop: PropSpec
    probability_raw: float
    probability_settled: float
    submitted_integer: int
    mc_standard_error: float
    settled_paths: int
    data_grade: str
    sensitivities: dict[str, float]
```

Required service interfaces:

```python
ingest_sackmann_snapshot(
    source: PinnedSource,
    tour: Literal["ATP", "WTA"],
) -> RawMatchSnapshot

normalize_player_service_rows(raw: RawMatchSnapshot) -> NormalizedServiceRows
build_serve_component_counts(rows: NormalizedServiceRows) -> ComponentCountTable

fit_snapshot(cutoff: datetime, tour: str, config: ModelConfig) -> ModelSnapshot

estimate_match(
    snapshot: ModelSnapshot,
    context: MatchContext,
    information: InformationBundle,
) -> MatchParameterDistribution

simulate_matches(
    params: MatchParameterDistribution,
    n_paths: int,
    seed: SeedSequence,
    trace_level: Literal["summary", "points"],
) -> SimulationBatch

parse_market(text: str, players: tuple[str, str], policy: SettlementPolicy) -> PropSpec

evaluate_prop(
    prop: PropSpec,
    simulations: SimulationBatch,
    policy: SettlementPolicy,
) -> PropEstimate

initialize_lock(request: LockRequest) -> LockedMatch
revise_lock(lock_id: str, new_information: InformationBundle) -> LockDiff | NoChange

simulate_tournament(
    draw: Draw,
    snapshot: ModelSnapshot,
    scenarios: TournamentScenarios,
    n_paths: int,
    seed: SeedSequence,
) -> TournamentSimulation
```

Every stochastic interface accepts an explicit RNG/seed and returns enough metadata to reproduce the result. Every evaluator is pure: it cannot read current data, refit a model, or mutate a lock.

`point.py` consumes one `ServePerformanceDraw` per serving direction per simulated path and applies the B3 ordering exactly. It must not accept derived $w_1$ or $w_2$ as substitutes for the primitive five-component draw. Derived probabilities may be cached for reporting and analytic checks only. `PerformanceDependenceSpec.mode` is `independent` in frozen v1.0; activating the one-factor option requires the I3 validation artifact and a new probability-affecting framework version.

### J3. Model-fitting contract

Implement each central component behind the same protocol:

```python
class StochasticComponent(Protocol):
    def required_columns(self) -> set[str]: ...
    def fit(self, rows: DataFrame, cutoff: datetime, config: Any) -> FittedComponent: ...
    def predict_linear(self, fitted: FittedComponent, context: Any) -> PosteriorLinearPredictor: ...
    def sample_predictive(self, predictor: PosteriorLinearPredictor, rng: Generator) -> Any: ...
    def update(
        self, fitted: FittedComponent, new_rows: DataFrame, cutoff: datetime
    ) -> FittedComponent: ...
    def diagnostics(self, fitted: FittedComponent) -> dict[str, Any]: ...
```

An “update” may perform a warm-started full MAP refit; it need not be an approximate online algorithm. Correct cutoff handling and reproducibility matter more than milliseconds.

### J4. SportsPredict integration boundary

The public API accepts 1–99 integer probabilities, supports single and batches of up to 50 predictions, permits pre-close updates, and documents a 60-request-per-minute limit. Implement the workflow as:

```text
discover event → discover matches → fetch markets per match
→ parse and verify every market → initialize/reuse lock
→ produce dry-run submission manifest → explicit user-authorized submission
→ verify every per-market response → append submission history
→ poll settled results → reconcile against official source
```

`submission.py` defaults to dry-run. It cannot estimate probabilities or reinterpret text. Credentials come only from a runtime secret such as `SPORTSPREDICT_API_KEY` and are never written to logs, locks, manifests, notebooks, or git. Partial batch failures are recorded and retried only for failed markets. A fetched market closing time is authoritative for the client, while the human-facing lock card also notes schedule uncertainty.

### J5. Persistence and versioning

- **Framework version:** statistical method. `v1.0` is this specification; `v1.1` is a small probability-affecting methodological correction such as a validated shared serving-day factor; `v2.0` requires a material generative redesign. A documentation-only correction may use a patch suffix.
- **Fitted-artifact version:** same framework, new data cutoff or refit. Daily updates create new artifacts without renaming the framework.
- **Lock revision:** same scheduled match, new information or fitted artifact. `L1`, `L2`, and so on coexist.
- **Settlement-policy version:** changes only when a semantic/official-stat rule changes or is clarified.
- **Prediction revision:** every accepted API update before close is append-only and points to the lock that generated it.

Never retroactively score an old prediction under a newer model. For comparative research, recompute a separate counterfactual forecast record rather than replacing history.

### J6. Initial implementation sequence

1. **Schemas and reproducible historical-data ingestion:** identity resolution; pinned Sackmann ATP/WTA snapshots and manifests; official-US-Open ingestion; immutable raw data; normalization; component counts; anomaly quarantine; provenance and license metadata.
2. **Deterministic tennis scoring and settlement property tests:** exact games, sets, serve order, both tiebreak formats, retirement/void separation, and canonical prop semantics.
3. **Revised component models:** first serve in, ace given first serve in, returnable first-serve win, double fault given second-serve opportunity, and playable second-serve win, with B1/B3 identity tests.
4. **Opponent adjustment and parameter uncertainty:** joint serve/return effects, surface pooling, recency weighting, beta-binomial dispersion, posterior approximation, and draw-once match variation.
5. **Joint match simulation:** common paths for scoring, exposure, serve statistics, and settlement; preserve exact seeds and sufficient statistics.
6. **Core SportsPredict prop evaluators:** match outcomes where needed; tiebreaks; deciding sets; first-set games; total games; aces; double faults; and first-serve point-win percentage.
7. **Immutable match locks and provenance:** versioned artifacts, lock diffs, append-only prediction revisions, and reproducibility hashes.
8. **Historical rolling backtests:** primitive and derived serve calibration, generator identities, cross-component residual dependence, scoring props, settlement behavior, and predeclared baselines.
9. **Auxiliary winner/UE/duration models:** official-major ingestion, opportunity-conditioned rally attachment, exposure-conditioned duration, and affected-family data grades.
10. **SportsPredict API dry-run integration:** read-only discovery, typed parsing, manifests, response verification, and no submission without separate authorization.
11. **Tournament simulation:** implement only after match-level validation and calibration are acceptable.

Correctness, reproducibility, calibration, and auditability take priority over performance optimization. Vectorization, JIT compilation, and API automation follow passing deterministic and rolling validation tests.

---

## K. Open questions

1. **Winners and aces:** determine whether the official US Open “winners” total includes aces, consistently across courts and both draws. Until reconciled, disable automatic winner props whose settlement depends on adding or subtracting aces; do not disable the ace or core score models.
2. **Unforced errors and double faults:** determine whether the official “unforced errors” total includes double faults. Until reconciled, disable affected automatic UE props; do not disable the double-fault or core score models.
3. **Displayed first-serve-win percentage:** determine whether the official match tracker rounds, truncates, or applies another conversion before displaying an integer. Until reconciled, disable automatic submission only for first-serve-win-percentage questions whose answer can change under the plausible display conventions; retain raw-model validation and unambiguous thresholds.
4. **Displayed duration:** determine how official duration is converted to whole displayed minutes, including any treatment of partial minutes. Until reconciled, disable automatic duration questions at convention-sensitive boundaries; retain the latent duration model and unambiguous thresholds for validation.
5. **Historical official winner/UE availability:** establish whether enough consistently defined official US Open/major winner and unforced-error rows can support player effects. If not, use stronger pooling, grade the affected families C, or skip them rather than importing incompatible labels. This does not affect the core serve/scoring fit.
6. **SportsPredict retirement settlement semantics:** obtain a ruling for retirement cases not explicitly covered by the public examples, including completed comparison props, completed full-set No outcomes, and compounds. Encode it in a new settlement-policy version and property tests before those affected markets are submitted. It does not alter event generation.
7. **Backtest-gated dynamics:** test whether a global best-of-five fatigue term, component-specific decay rates, handedness effects, or the I3 shared serving-day factor improve rolling out-of-sample calibration enough for a later version. The v1.0 default remains omission.

These unresolved items are empirical, data-reconciliation, or operational questions. They disable only the affected automatic prop families or boundary cases and do not block implementation, fitting, simulation, or validation of the core serve/scoring model.

---

## L. Freeze decision

The amended framework is design-complete:

- the primitive serve estimands have observable historical denominators and a coherent causal ordering;
- the identities $w_1=a+(1-a)q_1$, $w_2=(1-d)q_2$, and $p^{\mathrm{srv}}=fw_1+(1-f)w_2$ connect the component models to the unchanged tennis scoring engine;
- the hierarchical opponent adjustment, surface pooling, recency weighting, parameter uncertainty, and match-performance variation remain implementable;
- every core and compound prop is evaluated from the same joint simulated paths;
- settlement/void rules remain separate, typed, and versioned;
- data sources, transforms, anomalies, provenance, interfaces, validation gates, locks, and implementation priorities are explicit enough for Codex to implement without inventing a major statistical choice;
- the remaining questions concern empirical calibration, source availability/accounting, displayed-value conversion, or external settlement clarification—not architecture.

Future held-out evidence may justify a v1.1 correction under the existing append-only versioning policy. It is not a reason to keep v1.0-rc1 open.

**READY TO FREEZE AS TENNIS MODEL v1.0**
