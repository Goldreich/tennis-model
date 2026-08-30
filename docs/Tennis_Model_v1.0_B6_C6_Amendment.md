# Tennis Model v1.0 — B6/C6 Probability-Definition Amendment

**Status:** normative, ready-to-merge probability specification; implementation and historical-validation verification remain outstanding

**Scope:** This amendment resolves only B6 retirement generation and C6 post-90-day inactivity handling. It does not alter the five serve components, opponent adjustment, surface pooling, Laplace posterior approximation, beta match-performance draws, conditional independence, scoring, settlement, auxiliary models, or immutable versioning. Sections A and B below are the exact replacement text for B6 and the inactivity part of C6. Section C supplies the consequential B10, simulation, validation, and provenance insertions.

## A. Replacement B6 — Retirement and walkover process

### B6.1 Historical response and eligibility

The retirement model is fitted separately for ATP and WTA. Its observational unit is one player in one started singles match. A match is **started** when the normalized official record establishes that at least one point was played, using at least one of: a positive point-stat count, a legal score containing a completed game or tiebreak, or an explicit official started/in-play record. A score or status that does not establish play is not enough.

For player \(i\) in match \(m\), define \(R_{im}\) only under the following exhaustive coding rules:

| Official terminal record | Retirement-model treatment |
|---|---|
| Normal completion | Include both players with \(R_{im}=0\). |
| Started retirement, retiring player reliably identified | Include the retiring player with \(R_{im}=1\) and the opponent with \(R_{jm}=0\). The retirement may be included for incidence even when its exact game is unavailable. |
| Walkover or pre-start withdrawal | Exclude both players. A walkover is handled outside the simulator and is never a zero-retirement observation. |
| Default, disqualification, penalty termination, or misconduct termination | Exclude both players. These are not physical retirements and the opponent’s exposure was administratively censored. |
| Abandonment, cancellation, no contest, suspended match without a resolved terminal record, conflicting score/status, retirement with no reliable retiring-player identity, or any other ambiguous termination | Exclude both players and store a specific anomaly code. |

An official correction available before the fit cutoff replaces the earlier label in the input snapshot through the ordinary append-only correction process. A correction learned after the cutoff cannot enter that fit. A purported retirement in which the identified retiree is also recorded as the advancing winner, or in which the record simultaneously says normal completion, is conflicting and therefore excluded until reconciled.

Matches with a reliably identified retiree but no exact retirement game remain usable for the incidence estimator and unusable for timing diagnostics. Retirements before the first completed game are included in the match-level incidence response if play was confirmed; the simulator’s game-boundary approximation can first realize them after game 1.

### B6.2 Window and recency weights

For a fit with information cutoff \(C\), use only eligible outcomes known before \(C\) whose normalized match date is no more than 1,826 calendar days before the cutoff date. For match \(m\), let

\[
\Delta_m=(\operatorname{date}(C)-\operatorname{match\_date}_m)\text{ in whole calendar days},
\]

and include it only when \(0\leq \Delta_m\leq 1826\). Its weight is

\[
\omega_m=2^{-\Delta_m/730}.
\]

Thus the retirement fit has a fixed five-year window and a fixed 730-day half-life. These constants are not selected by backtesting. The result-availability timestamp, not merely the match date, must precede \(C\).

The history universe and identity rules are those of the pinned ATP or WTA normalized singles snapshot. Observations are never borrowed across tours.

### B6.3 Separate ATP and WTA tour baselines

For tour \(T\in\{\mathrm{ATP},\mathrm{WTA}\}\), form weighted player-start totals

\[
Y_T=\sum_{(i,m)\in T}\omega_m R_{im},
\qquad
N_T=\sum_{(i,m)\in T}\omega_m.
\]

The tour baseline is the Jeffreys-smoothed weighted rate

\[
\bar\rho_T=\frac{Y_T+\tfrac12}{N_T+1}.
\]

ATP and WTA have distinct \((Y_T,N_T,\bar\rho_T)\); there is no cross-tour pooling. The corresponding diagnostic tour posterior is

\[
\rho_T\mid\mathcal D_T\sim
\operatorname{Beta}\!\left(Y_T+\tfrac12,\;N_T-Y_T+\tfrac12\right),
\]

but v1.0 uses its mean \(\bar\rho_T\) as the fixed center of the player prior. Player-level beta draws below propagate the retirement parameter uncertainty used by the simulator; no additional common tour draw is introduced.

A production retirement artifact requires \(N_T\geq500\) weighted player-starts. If this coverage gate fails, the tour retirement component is unavailable and a production lock is prohibited; no other tour’s baseline or fixed emergency rate may be substituted. A zero-retirement tour sample with \(N_T\geq500\) remains estimable because of the Jeffreys terms.

### B6.4 Strongly shrunk player estimator

For player \(i\) on tour \(T\), define

\[
Y_i=\sum_{m:i\in m}\omega_mR_{im},
\qquad
N_i=\sum_{m:i\in m}\omega_m.
\]

Use a fixed prior effective sample size of

\[
\nu_R=100\text{ weighted player-starts}.
\]

The player posterior parameters and reported mean are

\[
\alpha_i=\nu_R\bar\rho_T+Y_i,
\qquad
\beta_i=\nu_R(1-\bar\rho_T)+N_i-Y_i,
\]

\[
\rho_i\mid\mathcal D\sim\operatorname{Beta}(\alpha_i,\beta_i),
\qquad
\bar\rho_i=\frac{\alpha_i}{\alpha_i+\beta_i}
=\frac{100\bar\rho_T+Y_i}{100+N_i}.
\]

Fractional \(Y\) and \(N\) are the recency-weighted beta-likelihood pseudo-counts. There is no player-age, ranking, form, score-state, format, or automatic injury covariate in this estimator.

The formula has the following exact behavior and no branching estimator:

- **No usable player history:** \(N_i=Y_i=0\), so \(\bar\rho_i=\bar\rho_T\) and the posterior is \(\operatorname{Beta}(100\bar\rho_T,100(1-\bar\rho_T))\).
- **Zero observed retirements:** if \(N_i>0\) and \(Y_i=0\), then \(\bar\rho_i=100\bar\rho_T/(100+N_i)>0\).
- **Sparse history:** \(0<N_i<25\). The tour prior supplies more than 80% of the posterior-mean weight.
- **Intermediate history:** \(25\leq N_i<100\). The same continuous formula applies.
- **Substantial history:** \(N_i\geq100\). The player data supply at least 50% of the posterior-mean weight, and as \(N_i/100\to\infty\), \(\bar\rho_i\to Y_i/N_i\).

These labels are diagnostics only; they do not change the formula.

### B6.5 Per-completed-game intensity

For each simulated path, independently for each player, draw

\[
\rho_i^{(b)}\sim\operatorname{Beta}(\alpha_i,\beta_i).
\]

The reference exposure is exactly

\[
G_0=22
\]

completed official match games for both ATP and WTA. A tiebreak counts as the one official game represented in the set score; its internal points do not create retirement checks. The same \(G_0\) is used for both tours because it defines a standard best-of-three reference match. ATP and WTA differ through their fitted baselines. Best-of-five matches acquire greater retirement probability only through their greater realized number of completed-game opportunities; no separate format multiplier is applied.

Convert the path draw to a constant per-completed-game intensity and its single-player discrete hazard:

\[
\lambda_i^{(b)}
=-\frac{\log(1-\rho_i^{(b)})}{22},
\qquad
h_i^{(b)}=1-\exp\{-\lambda_i^{(b)}\}
=1-(1-\rho_i^{(b)})^{1/22}.
\]

The identity

\[
1-(1-h_i^{(b)})^{22}=\rho_i^{(b)}
\]

must hold up to floating-point tolerance. Implement the transform with stable \(\log1p\) and \(\expm1\) equivalents. The intensity is constant over a path: score, set, game number, serving player, workload, and elapsed simulated duration do not change it in routine v1.0.

### B6.6 Named health or injury scenarios

A named scenario \(s\) may modify only the affected player’s retirement intensity through a supplied log-hazard ratio

\[
\eta_{i,s}\in\mathbb R,
\qquad
M_{i,s}=\exp(\eta_{i,s}),
\qquad
\lambda_{i,s}^{(b)}=M_{i,s}\lambda_i^{(b)}.
\]

The central scenario is exactly \(\eta_{i,0}=0\), hence \(M_{i,0}=1\). This specification assigns no nonzero injury, illness, heat, or workload multiplier. A nonzero \(\eta_{i,s}\) is legal only when it comes from a separately approved, versioned information-scenario record that contains the named state, affected player, numerical log-hazard ratio, source, observation time, publication time, authoring method, and scenario weight if a mixture is used. In the absence of such a record, the multiplier is 1.

Scenario weights must be fixed before simulation and sum to 1. Draw the scenario once per path before the retirement posterior draws. If weights are not defensible, run the named scenarios separately as sensitivities and do not blend them. A known pre-start withdrawal is a walkover/status update, not an infinite hazard scenario.

Inactivity under C6 never automatically changes \(\lambda_i\). Any return-from-injury retirement effect must enter through this explicit scenario interface.

### B6.7 Exact two-player competing risks

After any completed game at which the match remains ordinarily alive, let

\[
\Lambda^{(b)}=\lambda_{A,s}^{(b)}+\lambda_{B,s}^{(b)}.
\]

If \(\Lambda^{(b)}=0\), set

\[
p_0=1,\qquad p_A=p_B=0.
\]

Otherwise use the exact one-boundary categorical probabilities

\[
p_0=\exp(-\Lambda^{(b)}),
\]

\[
p_A=\{1-\exp(-\Lambda^{(b)})\}
\frac{\lambda_{A,s}^{(b)}}{\Lambda^{(b)}},
\qquad
p_B=\{1-\exp(-\Lambda^{(b)})\}
\frac{\lambda_{B,s}^{(b)}}{\Lambda^{(b)}}.
\]

Use one categorical draw from \((p_0,p_A,p_B)\). There is no simultaneous-retirement state. The retiring player loses and the other player advances; settlement remains a separate operation.

Compute \(1-\exp(-\Lambda)\) as \(-\operatorname{expm1}(-\Lambda)\). All adjusted intensities must be finite and nonnegative before the categorical probabilities are formed.

### B6.8 Ordering and edge cases

At a completed-game boundary, perform operations in this order:

1. record the game and update the set and match score;
2. if the ordinary tennis rules now end the match, record normal completion and perform no retirement draw;
3. otherwise draw the three-way competing-risk category;
4. if a player retires, stop the path immediately and preserve all already realized statistics.

There is no retirement draw before the first completed game. A completed regular game and a completed tiebreak each create one boundary. A match-winning game never receives a post-completion retirement draw.

Additional edge rules are:

- \(\rho_i=0\) maps to \(\lambda_i=h_i=0\). A mathematical \(\rho_i=1\) maps to infinite intensity; beta draws with positive parameters are in the open interval, but an exact floating-point 1 must be replaced by the greatest representable value below 1 before applying \(\log1p\).
- Scenario log-hazard ratios and their exponentiated multipliers must be finite. Missing, NaN, infinite, underflowed-zero, or overflowed values invalidate the scenario. A nonpositive multiplier cannot be supplied because \(M=\exp(\eta)>0\).
- If one intensity is zero, the nonzero player is the only possible retiree at that boundary.
- If both are zero, the RNG must not be advanced for a retirement draw.
- Exact timing missing from a historical retirement does not prevent incidence fitting.
- Walkovers, defaults, and ambiguous terminations never enter \(Y_i\) or \(N_i\).
- All weighted sufficient statistics must be finite and satisfy \(0\leq Y_T\leq N_T\) and \(0\leq Y_i\leq N_i\). A violation invalidates the artifact; counts are never clipped.
- Missing or incomplete tour-source coverage cannot be treated as no player history. It fails provenance validation. The no-history branch is permitted only when the pinned, coverage-valid source contains no eligible observation for that canonical player.
- The generator does not create defaults, disqualifications, walkovers, or simultaneous retirements.

### B6.9 Mandatory validation

The following tests are part of v1.0:

**Deterministic tests**

1. A fixture for every terminal-status class must produce the exact include/exclude and \(R\) coding above. The two player rows of a started retirement must be \(1\) and \(0\).
2. Window-boundary fixtures at ages 0, 1, 730, 1,826, and 1,827 days must yield weights \(1\), \(2^{-1/730}\), \(1/2\), \(2^{-1826/730}\), and exclusion, respectively.
3. Separate ATP and WTA fixtures must never share counts or baselines.
4. Posterior fixtures must reproduce \(\alpha_i,\beta_i,\bar\rho_i\) to relative tolerance \(10^{-12}\), including no-history and zero-retirement cases.
5. For \(\rho\in\{10^{-6},0.001,0.01,0.10,0.50\}\), the 22-game mapping identity must hold to absolute tolerance \(10^{-12}\).
6. For a grid of finite nonnegative intensities, \(p_0,p_A,p_B\) must each lie in \([0,1]\) and sum to 1 within \(10^{-15}\); swapping player intensities must swap \(p_A,p_B\).
7. Zero-hazard paths must be bit-for-bit identical to paths with retirement disabled, apart from recorded diagnostic fields.
8. A match-winning game must bypass retirement. A non-match-winning completed game must execute exactly one competing-risk check.
9. Fixed seed, snapshot, scenario, and path count must reproduce the same retirement player/game and all downstream settlement states.

**Statistical tests**

10. With fixed \(\rho\), simulate 1,000,000 independent 22-boundary single-player exposures and require the empirical retirement rate to differ from \(\rho\) by no more than

\[
5\sqrt{\rho(1-\rho)/10^6}+10^{-6}.
\]

11. With fixed \((\lambda_A,\lambda_B)\), simulate 1,000,000 one-boundary draws and require each empirical category frequency to differ from its analytic probability by no more than five multinomial standard errors plus \(10^{-6}\).
12. From 1,000,000 player posterior draws, require the empirical beta mean and variance to lie within five Monte Carlo standard errors of their analytic values.
13. In strict rolling historical evaluation, compute each player-start’s simulated retirement probability before revealing the result. For each tour, report

\[
Z_T=
\frac{\sum_o(R_o-\widehat p_o)}
{\sqrt{\sum_o\widehat p_o(1-\widehat p_o)}}.
\]

Flag retirement calibration if \(|Z_T|>3\) and a 99% event-block bootstrap interval for the observed-minus-predicted rate excludes zero. Report the same diagnostic by best-of-three/best-of-five and by the four history bands above; subgroup results with fewer than 20 observed retirements are explicitly underpowered rather than pass/fail.
14. Where at least 50 retirements have reliable game timing for a tour, compare the observed discrete retirement-game distribution with posterior-predictive timing by randomized PIT and a fixed-seed parametric-bootstrap Kolmogorov–Smirnov test. Flag at \(p<0.01\). If timing coverage is lower, mark this diagnostic unavailable; do not invent retirement times.
15. Compare predicted and observed settlement/void frequencies by prop family under the unchanged settlement policy. A settlement discrepancy is not repaired by altering the retirement generator without a versioned model change.

For all stated Monte Carlo tests with \(n\) draws, use \(\operatorname{SE}(\widehat p)=\sqrt{p(1-p)/n}\), \(\operatorname{SE}(\bar X)=\sqrt{v/n}\), and, for the unbiased sample variance \(S^2\),

\[
\operatorname{SE}(S^2)=
\sqrt{\frac{\mu_4-\frac{n-3}{n-1}v^2}{n}},
\]

where \(v\) and \(\mu_4\) are the analytic variance and fourth central moment of the tested distribution.

The statistical checks validate the frozen constants; they are not a license to tune them against the Milestone 8 synthetic fixture or the first production backtest.

## B. Replacement C6 inactivity text

Replace only the current **Inactivity** bullet in C6 with the following subsection. All other C6 bullets remain unchanged.

### C6.1 Post-90-day inactivity

#### Match-date definition

For target player \(i\), let \(S\) be the target’s scheduled-start calendar date in the event’s local timezone. Let \(L_i\) be the latest normalized official calendar date before the information cutoff on which \(i\) played at least one point in an eligible competitive singles match. Eligibility is the same pinned, identity-resolved, non-exhibition singles universe admitted to the tour’s historical model. Qualifying and official team or Olympic singles records count when that source is part of the frozen tour manifest. Doubles, exhibitions, walkovers, pre-start withdrawals, defaults with no play, and abandoned records that do not establish a played point do not count. A started retirement does count.

Define

\[
D_i=(S-L_i)\text{ in whole calendar days}.
\]

Only a last match whose outcome or started status was available before the information cutoff may be used. If \(D_i<0\), the chronology is invalid and production estimation stops; it is not clipped to zero. A match completed earlier on the same calendar date gives \(D_i=0\).

If no official scheduled-start calendar date is available, a production lock is unavailable; no estimated date is substituted. A schedule-date change creates a new lock revision and recomputes \(D_i\).

If the pinned history has verified coverage but contains no prior started match for the canonical player, use the cold-start rule below. If source coverage or identity resolution is insufficient to decide whether a prior match exists, production estimation stops; missing coverage is not treated as infinite inactivity.

#### Threshold and mean reversion

Set

\[
g_i=\max(0,D_i-90).
\]

The hard-surface mean-reversion multiplier is

\[
m_i=
\begin{cases}
1, & D_i\leq90,\\[2mm]
2^{-g_i/180}, & D_i>90.
\end{cases}
\]

Thus inactivity begins only on day 91 and the post-threshold half-life is exactly 180 days. There is no finite lower cap: \(m_i\to0\) as \(D_i\to\infty\).

For every affected player/component/role coordinate, write the hard-court effect as in B2:

\[
b^{\mathrm{hard}}_{ik}
=b^{\mathrm{global}}_{ik}+\delta^{\mathrm{hard}}_{ik}.
\]

Replace only the posterior central hard deviation by

\[
\mathbb E[\delta^{\mathrm{hard}}_{ik}\mid\mathcal D,\ D_i]
=m_i\,\widehat\delta^{\mathrm{hard}}_{ik}.
\]

Equivalently, the adjusted central hard effect is

\[
\widehat b^{\mathrm{hard,adj}}_{ik}
=\widehat b^{\mathrm{global}}_{ik}
+m_i\widehat\delta^{\mathrm{hard}}_{ik}.
\]

The all-surface central effect \(\widehat b^{\mathrm{global}}_{ik}\) is unchanged. Inactivity therefore removes confidence in a player’s hard-specific departure from all-surface ability; it does not mechanically make the player worse.

#### Affected player blocks

Apply the rule to the inactive player wherever that player appears in the five primitive component models:

| Component | Server block affected | Returner block affected |
|---|---|---|
| \(F\) | \(u_i^F\), including its global and hard-deviation coordinates | none |
| \(A\) | \(s_i^A\), including global and hard-deviation coordinates | \(r_i^A\), including global and hard-deviation coordinates |
| \(Q1\) | \(s_i^{Q1}\), including global and hard-deviation coordinates | \(r_i^{Q1}\), including global and hard-deviation coordinates |
| \(D\) | \(u_i^D\), including its global and hard-deviation coordinates | none |
| \(Q2\) | \(s_i^{Q2}\), including global and hard-deviation coordinates | \(r_i^{Q2}\), including global and hard-deviation coordinates |

Mean reversion applies only to the hard-deviation coordinate in each affected block. Uncertainty inflation applies to both the player-specific global coordinate and the player-specific hard-deviation coordinate. Tour intercepts, event-year effects, indoor/outdoor and other context coefficients, other players’ coordinates, beta-binomial concentration parameters \(\kappa_k\), rally and duration models, and the B6 retirement process are unchanged. If both target players are inactive, apply each player’s own multiplier to that player’s coordinates.

#### Exact posterior uncertainty inflation

For known \(D_i\), define the variance-inflation factor

\[
c_i=1+\{1-m_i^2\}=2-m_i^2.
\]

Therefore \(1\leq c_i<2\), \(c_i=1\) for \(D_i\leq90\), and \(c_i\to2\) for a very long known absence. The standard-deviation cap is \(\sqrt2\) times the fitted value.

Numerically, constrain the analytically computed \(m_i\) to \([0,1]\) only to remove floating-point overshoot, then compute \(c_i=\min\{2,\max(1,2-m_i^2)\}\). Underflow of \(m_i\) to zero for an extremely long known absence validly gives \(c_i=2\); there is no data-dependent cap.

Within each component’s Laplace posterior, let \(\widehat\theta_k\) and \(\Sigma_k\) be the fitted posterior mean and covariance. Construct a diagonal matrix \(A_{ik}\) whose entry is \(\sqrt{c_i}\) for every affected player-specific global or hard-deviation coordinate listed above and 1 for every other coordinate. When both players are inactive, use the corresponding player-specific factor on each player’s coordinates. Define the adjusted mean \(\widehat\theta_k^{\,\mathrm{adj}}\) by leaving all entries unchanged except

\[
\widehat\delta_{ik}^{\,\mathrm{hard,adj}}
=m_i\widehat\delta_{ik}^{\,\mathrm{hard}},
\]

and define

\[
\Sigma_k^{\mathrm{adj}}
=A_k\Sigma_kA_k^\top.
\]

This congruence transform is positive semidefinite whenever \(\Sigma_k\) is. Marginal variances of the affected coordinates are multiplied by \(c_i\); their correlations with the rest of the same component posterior are preserved. Marginal means and variances of unaffected coordinates are unchanged. Because component fits remain separate and no shared random variable is added, the frozen cross-component conditional-independence rule is preserved.

The beta match-performance layer is not directly inflated. After the adjusted posterior parameter draw produces \(\mu_k\), draw

\[
\phi_k\sim\operatorname{Beta}\{\kappa_k\mu_k,\kappa_k(1-\mu_k)\}
\]

with the original fitted \(\kappa_k\). Inactivity changes predictive uncertainty only through the adjusted player-parameter posterior.

#### Required ordering

For each matchup:

1. determine \(D_i,g_i,m_i,c_i\) from pre-cutoff data for both players;
2. construct \(\widehat\theta_k^{\,\mathrm{adj}}\) and \(\Sigma_k^{\mathrm{adj}}\) for each of \(F,A,Q1,D,Q2\);
3. draw the posterior parameter vector from

\[
\theta_k^{(b)}\sim
N(\widehat\theta_k^{\,\mathrm{adj}},\Sigma_k^{\mathrm{adj}});
\]

4. construct the two directional matchup linear predictors and means from those draws;
5. draw the five beta match-performance probabilities using the unchanged \(\kappa_k\);
6. run the unchanged point, game, set, match, and settlement layers.

No surface mean reversion or variance multiplier may be applied a second time after step 3.

#### Boundary and missing-history behavior

- **\(D_i\leq90\):** \(g_i=0,m_i=1,c_i=1\). The adjusted mean and covariance equal the fitted values exactly. With the same snapshot, seed, and inputs, active-player forecasts must be bit-for-bit compatible with the pre-amendment serve model.
- **\(D_i=91\):** \(g_i=1\),

\[
m_i=2^{-1/180}=0.9961565872205752,
\qquad
c_i=2-m_i^2=1.0076720537370566.
\]

- **Known long absence:** the hard-specific central deviation tends continuously to zero, while affected player-specific posterior variances tend to, but never exceed, twice their fitted values. The all-surface central effects remain unchanged.
- **No known prior match with verified coverage:** instantiate the existing tour-specific cold-start player prior. Set every hard-deviation central value to zero and apply no additional inactivity inflation, \(c_i=1\), because the cold-start prior already represents the model’s no-data uncertainty. Record \(D_i\) and \(g_i\) as null, \(m_i=0\), and **cold_start=true**. This branch does not reuse another player’s history.
- **Coverage or identity uncertainty:** do not use the cold-start branch. Mark the matchup unavailable for a production lock until the source or identity issue is resolved.

ATP and WTA use the same 90-day threshold, 180-day post-threshold half-life, and 2-times variance cap, but all fitted player effects, covariance matrices, shrinkage variances, intercepts, and component concentrations remain tour-specific.

#### Mandatory validation

**Deterministic tests**

1. Date fixtures must establish that \(D=90\) is unchanged, \(D=91\) uses the exact constants above, and a negative \(D\) fails.
2. For \(D\in\{0,90,91,270,450,810\}\), reproduce the following values to absolute tolerance \(10^{-12}\):

| \(D\) | \(g\) | \(m\) | \(c=2-m^2\) |
|---:|---:|---:|---:|
| 0 or 90 | 0 | 1 | 1 |
| 91 | 1 | 0.9961565872205752 | 1.0076720537370566 |
| 270 | 180 | 0.5 | 1.75 |
| 450 | 360 | 0.25 | 1.9375 |
| 810 | 720 | 0.0625 | 1.99609375 |

3. For every affected role, verify that the adjusted hard central effect is \(\widehat b^{global}+m\widehat\delta^{hard}\), that the global central effect is unchanged, and that the sign convention for returner effects is unchanged.
4. At \(D\leq90\), adjusted posterior means, covariances, sampled parameters, beta draws, and match outputs must be bit-for-bit identical under the same seed.
5. For a positive-semidefinite fixture covariance, \(\Sigma^{adj}\) must remain positive semidefinite to numerical eigenvalue tolerance \(-10^{-12}\), affected marginal variances must equal \(c\Sigma_{jj}\), unaffected marginal variances must be unchanged, and all correlations that are defined must be preserved.
6. Applying the transform twice must be rejected by an adjustment-version/state guard.
7. The cold-start branch must produce zero hard-deviation central means, the existing tour-specific prior covariance, and no 2-times inflation.
8. No C6 operation may alter \(\kappa_k\), the B6 posterior or intensity, context coefficients, or an active opponent’s marginal posterior.
9. Component-level RNG streams must remain separate; inactivity must not create a shared match-day factor.

**Statistical tests**

10. For each adjusted Gaussian fixture, draw 1,000,000 parameter vectors and require every affected sample mean and variance to fall within five Monte Carlo standard errors of the analytic adjusted mean and variance.
11. With fixed adjusted \(\mu_k,\kappa_k\), beta-draw samples must reproduce the ordinary beta mean and variance within five Monte Carlo standard errors, confirming that C6 did not change the match-performance law.
12. In strict rolling-origin historical validation, compute inactivity using only the target’s pre-match source snapshot and report primitive and derived serve calibration in fixed bands \(D\leq90\), 91–180, 181–365, and \(D>365\), separately for ATP and WTA. Report count calibration, randomized quantile residuals, log predictive density, and interval coverage. The bands and formulas are fixed before outcomes are revealed and are not used to retune v1.0.
13. Compare active-player forecasts produced by the amended implementation with the pre-amendment reference fixtures. Any numerical difference for \(D\leq90\) is an implementation failure.
14. Verify monotonic mechanics on held parameters: as \(D\) increases, \(|m\widehat\delta^{hard}|\) cannot increase and \(c\) cannot decrease; \(\widehat b^{global}\) cannot change. This is a mechanism test, not a requirement that match-win probability be monotone, because different component signs and opponent roles can offset.

## C. Consequential ready-to-merge patch text

### C1. Replace the B10 Retirement row and add the Inactivity row

Replace the current B10 Retirement row with:

| Retirement | tour, player, format through realized scoring exposure, started-retirement history, named health/workload scenario | tour baseline \((Y_T,N_T,\bar\rho_T)\), player \((Y_i,N_i,\alpha_i,\beta_i)\), \(\nu_R=100\), \(G_0=22\), scenario log-hazard ratio | beta draw; intensity transform; three-way competing risk after a nonterminal completed game | retirement player and completed-game boundary | refit from eligible official terminal outcomes in 1,826-day window with 730-day half-life; scenario changes only on cited versioned information |

Add immediately after it:

| Inactivity adjustment | scheduled start, cutoff-valid last started match, player-specific posterior blocks | threshold 90 days, hard-deviation half-life 180 days, variance factor \(c=2-m^2\) capped below 2 | deterministic posterior mean/covariance transform before parameter draw | adjusted \(F,A,Q1,D,Q2\) posterior distributions | recompute for every lock; no fitted-parameter refit and no change at \(D\leq90\) |

### C2. Replace D steps 10 and 11

Replace simulation steps 10 and 11 with:

10. **Resolve ordinary termination, then retirement.** After each completed regular game or tiebreak, update the score. If that game ordinarily completes the match, stop with normal completion and do not draw retirement. Otherwise apply the B6 three-way competing-risk draw once. If one player retires, stop the path, mark that player, and preserve partial statistics.
11. **Store the terminal state.** Store normal completion or retirement, exact score and completed set scores, first-set games, total completed games, tiebreaks, breaks, all point/stat totals, and whether a deciding set began. A retiring player loses the match; prop settlement is evaluated later and separately.

Insert before current simulation step 2:

**Apply inactivity transform.** Compute and freeze each player’s C6 inactivity record, then transform the five component posterior means and covariance matrices exactly once. Parameter draws in the next step must come from those adjusted distributions.

### C3. Replace the I3 Retirement validation row and add Inactivity

Use:

| Retirement | B6 terminal-status coding; ATP/WTA tour baselines; 100-start shrinkage identities; 22-game hazard identity; competing-risk mass; rolling incidence, timing where observed, and settlement-frequency calibration |
| Inactivity | date/90-day boundary; hard-deviation mean transform; covariance congruence and 2-times cap; active-player identity compatibility; rolling calibration and coverage by fixed inactivity band and tour |

Add the following sentence after the I3 required-gates table:

> The complete deterministic and statistical gates in B6.9 and C6.1 are mandatory. Their constants were predeclared before genuine historical evaluation. A failed historical diagnostic may motivate a versioned v1.1 proposal, but it must not silently retune v1.0 or alter an existing lock.

### C4. Provenance and artifact requirements

Add the following normative text to C7/F1/J5:

**Retirement fit artifact.** Each ATP/WTA artifact must store: artifact/schema version; response-coding version; canonical source-manifest ID and hash; source coverage result; fit and information cutoffs; 1,826-day window and 730-day half-life; included and excluded weighted/unweighted counts by terminal-status reason; \(Y_T,N_T,\bar\rho_T\) and diagnostic tour beta parameters; \(\nu_R=100\); every player’s \(Y_i,N_i,\alpha_i,\beta_i\); \(G_0=22\); intensity-transform version; software/config/data hashes; and deterministic-test result hash. Corrections create a new artifact.

**Inactivity configuration artifact.** Store: schema/version; threshold 90; post-threshold half-life 180; variance formula \(c=2-m^2\); limiting variance factor 2; cold-start rule; eligible-match/date rule; complete source-coverage assertion; affected coordinate names for every component/role; transform implementation/version; and config/code hashes.

**Prediction lock.** For each player, the machine-readable lock must contain:

- retirement artifact ID, tour baseline, player \(Y,N,\alpha,\beta\), posterior mean, \(G_0\), central scenario intensity summary, scenario ID/log-hazard ratio/weight/source hash, and competing-risk version;
- last eligible match ID/date/source pin and availability timestamp, or the verified cold-start record;
- \(D,g,m,c\), **cold_start**, affected coordinate IDs, and hashes of the unadjusted and adjusted posterior mean/covariance records;
- B6/C6 schema versions, model/config/data/code hashes, root seed, RNG, path count, ordinary-termination-before-retirement policy version, warnings, and test/validation artifact references.

The rendered card may show only retirement probability, inactivity days/multiplier, material scenarios, and warnings; the machine-readable lock remains authoritative.

**Historical backtests.** Every historical target must reconstruct B6 retirement fits and C6 last-match records from source versions and corrections available before that target cutoff. Target construction must not read the target outcome. The outcome revealer supplies retirement/normal/other terminal status and timing only after the forecast lock is verified. Missing termination or timing data remain missing and are reported by reason; they are never imputed. Backtest rows must retain the retirement/inactivity artifact IDs, source pins, constants, player sufficient statistics, \(D,g,m,c\), scenario record, and run ID.

**Compatibility.** A snapshot or production lock that lacks the normative B6/C6 fields is schema-incompatible and must fail explicitly. Existing non-production development locks remain immutable historical artifacts and are not rewritten. For every player with \(D\leq90\), the C6 transform is the identity and existing active-player probability fixtures must remain bit-for-bit unchanged. The retirement and inactivity additions do not change prop settlement-policy versions.

## D. Codex implementation handoff

No model formula remains for Codex to choose. Implement the contracts above with the following bounded changes.

### Required new or changed types

- **HistoricalTerminationClass:** normal completion, started retirement, walkover, default/disqualification, ambiguous; includes started evidence, retiree identity, timing availability, source pin, availability time, and anomaly code.
- **RetirementObservation:** tour, player, match, date, \(R\), recency weight, cutoff, and source/coding version.
- **RetirementFitArtifact:** all tour/player sufficient statistics, posterior parameters, fixed constants, exclusions, coverage gate, hashes, and schema version specified in C4.
- **PlayerRetirementPosterior:** player/tour IDs, \(\alpha,\beta,\bar\rho\), history band, artifact ID.
- **RetirementScenario:** stable scenario ID, player, finite log-hazard ratio, weight if blended, evidence/source timestamps and hash.
- **RetirementPathDraw:** player posterior draw, base and scenario-adjusted intensity, and scenario ID. It is path-local and is not persisted one row per path.
- **InactivityRecord:** player, scheduled-start date, last eligible match/source or verified cold start, \(D,g,m,c\), coverage status, and adjustment version.
- **InactivityAdjustment:** affected component/coordinate IDs plus adjusted-mean and adjusted-covariance references/hashes.
- **MatchParameterDistribution:** add explicit retirement posteriors/scenarios and inactivity records; component posteriors must identify whether the C6 transform has already been applied.
- **ModelSnapshot and PredictionSnapshot:** add retirement artifact/config references, inactivity-config reference, and B6/C6 schema versions.

### Affected interfaces

- Normalize official terminal outcomes into the exhaustive B6 classes without changing the existing score/stat normalization.
- Fit and retrieve the separate ATP/WTA retirement artifact at a strict cutoff.
- Compute a player retirement posterior from artifact sufficient statistics.
- Compute the C6 inactivity record from scheduled start, information cutoff, source coverage, and last eligible match.
- Apply the C6 mean/covariance transform exactly once before the existing posterior sampler.
- Convert beta retirement draws to intensities and sample the three-way competing risk only after a nonterminal completed game.
- Extend lock creation, reproduction, comparison, rendering, and rolling backtests with the required B6/C6 records and hashes.
- Keep settlement evaluators pure and unchanged; they consume only the terminal path state.

### Required tests

- Implement every deterministic and statistical test in B6.9 and C6.1.
- Extend scoring tests to prove ordinary match completion precedes retirement.
- Extend lock round-trip and reproduction tests to cover B6/C6 fields, hashes, scenario records, and adjustment-once guards.
- Extend cutoff/leakage tests so retirement labels, official corrections, last-match records, and scenario evidence cannot cross the historical target cutoff.
- Extend compatibility fixtures to require bit-for-bit active-player outputs at \(D\leq90\).
- Extend backtest coverage reports with terminal-status exclusions, retirement timing availability, cold-start counts, and fixed inactivity bands.

### Production-lock and backtest contents

Production locks must include all C4 fields and must reject: failed tour retirement coverage; unresolved player identity; incomplete last-match coverage; an unversioned nonzero scenario multiplier; a missing B6/C6 artifact; a duplicate C6 transform; or a simulator that checks retirement after an ordinarily completed match. Historical backtests must select only cutoff-valid artifacts, derive seeds under the existing policy, persist and verify the forecast lock before outcome reveal, and report unavailable data without substitution.

### Compatibility boundaries

The five serve-component likelihoods, fitted \(\kappa_k\), posterior approximation method, beta performance draws, cross-component independence, scoring, auxiliary models, settlement, prop definitions, path-count policy, and lock append-only behavior are unchanged. C6 is exactly the identity for known \(D\leq90\). B6 adds the already intended ancillary probability process; it does not authorize a detailed injury model or automatic workload/inactivity multiplier.

## E. Versioning and readiness

B6 and C6 were already part of the intended Tennis Model v1.0 architecture, and no production forecast has been issued. Fixing their previously omitted numerical definitions completes, rather than redesigns, v1.0. The framework may therefore remain **Tennis Model v1.0** after implementation and test verification. During implementation it may be identified operationally as a v1.0 release candidate or as a document revision, but the probability method does not require a v1.1 label.

This conclusion does not operationally freeze the model. The accepted Milestone 8 audit remains controlling: Milestone 8’s architecture is only provisionally implemented and requires the separate Milestone 8.1 operational-correctness remediation. Genuine cutoff-safe historical validation is also still required. Those are implementation and validation gates, not unresolved B6/C6 probability definitions.

B6/C6 PROBABILITY SPECIFICATION: COMPLETE
OPERATIONAL TENNIS MODEL v1.0: NOT YET READY — MILESTONE 8.1 AND GENUINE HISTORICAL VALIDATION STILL REQUIRED
