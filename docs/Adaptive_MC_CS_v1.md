# Adaptive Monte Carlo confidence-sequence policy v1

`adaptive_mc_cs_v1` is an operational simulation, stopping, and reporting
policy. It changes no Tennis Model v1.0 probability formula or fitted model
parameter. It supersedes the fixed 100,000/400,000 live-path escalation and
the model-layer 1--99 clamp. Historical locks retain their recorded policies.

## Confidence sequence

For one binary prop, let `Y` be Yes outcomes and `N` be settled outcomes.
Void and unresolved paths do not enter either count. When `N > 0`, the raw
estimate is exactly

\[
\widehat p_N=Y/N.
\]

No pseudo-count, smoothing, or clipping is applied to this estimate.

For candidate Bernoulli probability `p`, define the Jeffreys beta-binomial
mixture likelihood-ratio martingale

\[
M_N(p)=
\frac{B(Y+1/2,N-Y+1/2)}
{B(1/2,1/2)p^Y(1-p)^{N-Y}}.
\]

The 99% confidence sequence is the interval

\[
C_N=\{p\in[0,1]:M_N(p)\le100\}=[L_N,U_N].
\]

For every fixed `p`, `M_N(p)` is a nonnegative mean-one mixture
likelihood-ratio martingale under `P_p`. Ville's inequality gives

\[
P_p\left(\sup_N M_N(p)\ge100\right)\le0.01,
\]

so `P_p(p in C_N for every inspected N) >= 0.99`, including when stopping
depends on the observed confidence sequence. This is the standard
beta-binomial mixture construction; see Howard, Ramdas, McAuliffe, and
Sekhon (2021), "Time-uniform, nonparametric, nonasymptotic confidence
sequences," *Annals of Statistics* 49(2), especially its mixture-martingale
construction.

The implementation evaluates `log(M_N(p))` with `lgamma`, `log`, and
`log1p`, then performs 80 deterministic bisection iterations on each side of
`Y/N`. Impossible boundary likelihoods are treated as infinity. If `Y=0`,
the lower endpoint is exactly zero and the finite-sample upper endpoint is
positive. If `Y=N`, the upper endpoint is exactly one and the lower endpoint
is below one. A prop with `N=0` has no probability or confidence sequence and
is `UNAVAILABLE`.

## Stopping and integer reporting

The cumulative path checkpoints are 5,000, 10,000, 20,000, 40,000, and
70,000. Simulation stops when every generated requested prop is
`INTEGER_STABLE`, or unconditionally at 70,000. A prop is integer-stable only
when every value in its full confidence sequence maps to the same model
integer as the raw estimate.

Model probabilities use centralized nearest-percent, half-up rounding and
retain 0 and 100. The endpoint buckets are exactly `[0, 0.005)` and
`[0.995, 1]`. At 70,000, a confidence sequence crossing any rounding boundary
is `INTEGER_BOUNDARY_SENSITIVE`. Zero settled paths remain `UNAVAILABLE`.

An external 1--99 constraint is not part of model estimation or reporting.
It is represented by a separate `PlatformSubmissionPolicy`; locks retain the
raw model probability, model integer, and any platform integer as distinct
fields.

An estimate of 0% means that no Yes outcome was observed among settled
paths. An estimate of 100% means every settled path was Yes. Neither statement
proves the underlying model probability is analytically zero or one; the
confidence sequence is the authoritative Monte Carlo uncertainty statement.

## Deterministic endpoint examples

- `0 / 5,000`: raw `0%`, CS approximately `[0%, 0.1885%]`, stable at `0%`.
- `5,000 / 5,000`: raw `100%`, CS approximately `[99.8115%, 100%]`, stable
  at `100%`.
- `0 / 100`: raw `0%`, but the positive upper bound crosses the 0% bucket,
  so it is not yet stable.
- `100 / 100`: raw `100%`, but the lower bound crosses the 100% bucket, so it
  is not yet stable.
- `10 / 5,000`: raw `0.2%` and still boundary-sensitive; `20 / 10,000` has
  the same raw proportion but a sufficiently narrower sequence to stabilize
  at model integer `0%`.

The deterministic test suite also simulates 20,000 Bernoulli sequences at
each of seven probabilities from 0.001 through 0.999 and verifies at least
99% simultaneous coverage across all five inspected checkpoints. With seed
rule `20260830 + round(p * 1,000,000)`, the observed crossing rates were
`0.005%`, `0.005%`, `0%`, `0.010%`, `0%`, `0.010%`, and `0.020%` for
`p = 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999`, respectively. The worst
observed simultaneous coverage was therefore `99.980%`; this is empirical
validation, while the Ville inequality above supplies the formal guarantee.
