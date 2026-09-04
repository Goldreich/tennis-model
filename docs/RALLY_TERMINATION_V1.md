# Tennis Model v1.2 Rally-Termination Layer

## Scope

This versioned auxiliary layer supports official winners and unforced-error
props. It does not alter the frozen serve components, Elo/game-day Elo, point
winners, score paths, aces, double faults, tiebreaks, or duration.

For each completed simulated path and each direction `i -> j`, the eligible
count is:

```text
points won by i - aces by i - double faults by j
```

Those points are classified mutually exclusively as a rally winner by `i`, an
unforced error by `j`, or other/forced. Classification is performed in an
aggregate vectorized draw; it is distributionally equivalent to independent
point labels conditional on the path and directional latent probabilities.

## Accounting convention

The active US Open convention is
`usopen-winners-include-aces-ue-include-double-faults/v1`:

```text
official winners = aces + rally winners
official unforced errors = double faults + rally unforced errors
```

Aces and double faults are therefore subtracted before fitting the residual
classifier and added back exactly once after simulation.

## Fit

ATP and WTA are fit separately. Each uses two multinomial logits against an
`other/forced` baseline:

```text
winner logit = tour intercept + winner aggression + loser allows-winners
UE logit     = tour intercept + winner pressure   + loser error-tendency
```

Player effects receive a zero-centered Gaussian ridge prior with standard
deviation `0.35`. This deliberately pools a small official-stat sample heavily.
An unseen player receives zero player effects and therefore the relevant tour
baseline. Conditional match-to-match dispersion is represented by a fitted
Dirichlet concentration.

## Settlement

The supported atomic kinds are `WINNERS`, `WINNER_COMPARE`,
`UNFORCED_ERRORS`, `TOTAL_UNFORCED_ERRORS`, and `UE_COMPARE`. Props must carry
the accounting convention above. Comparison ties settle No for "more than."

First-serve-win percentage uses the exact simulated ratio. Thus `66.1 > 66`
settles Yes; no whole-percent display rounding is applied.

## Reproducibility

The fitted artifact records its immutable source snapshot, manifest hash, data
cutoff, fit settings, player exposures, and artifact content hash. Rally draws
use the separate `sha256-seed-id-path-start-pcg64/v1` stream, so enabling this
layer cannot consume or shift the established simulation RNG streams.

## Sequential Bayesian updates

The static fit is the initialization prior for
`rally-termination-posterior/v2`. Tour intercepts, the player-effect prior
scale, and the fitted Dirichlet concentration remain fixed. The four player
effects are updated after each newly available completed match.

The update likelihood uses both directional residual triplets:

```text
(rally winners by A, rally UEs by B, other points won by A)
(rally winners by B, rally UEs by A, other points won by B)
```

The implementation applies a sequential Laplace update to the complete sparse
Gaussian posterior. Each row couples the point winner's aggression and pressure
with the point loser's allows-winners and error-tendency effects. A new player
enters with mean zero and covariance equal to the original player-effect prior.

Every update writes an immutable child under
`artifacts/production/tennis-model-v1.2/rally-posterior/{tour}/{artifact_id}`.
The mutable ATP/WTA current pointers select the latest child for new forecasts.
Historical forecasts traverse the parent chain to the newest artifact strictly
preceding their information cutoff.

Initialize a chain with `scripts/initialize_rally_posterior.py`. Apply every
unseen completed singles match in a later immutable capture with
`scripts/update_rally_posterior.py`. Previously seen match IDs are skipped only
when the source hash is identical; changed payloads are blocked pending an
explicit correction policy.

For prediction, the simulator extracts the joint posterior mean and covariance
of the two players' eight effects. Each simulated match draws that vector once
and uses it in both directional classifications, integrating parameter
uncertainty without changing the score path.
