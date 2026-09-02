# LOCKED MATCH CARD

Lock ID: TMV1-WTA-e31287c370fb554f2aac-L1
Canonical match identity: match_e31287c370fb554f2aace6dbf2ba365406b5466e222d3e1b35e2edfe4fb921aa
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T21:45:15.891231+00:00
Information cutoff: 2026-09-01T20:50:16.183925+00:00
Event: US Open
Match: player_b97a5f95-4205-58f3-96ca-c63411436250 v player_ad84ea61-9ba3-53b4-86bd-395303127dc6
Draw / round: WTA singles / R64
Scheduled start: 2026-09-02T15:00:00+00:00
Format: best of 3; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50 by path)
Conditions scenario: central
Model snapshot: 488a8f77251e3eb929affcf4c81082201b3597cc899c388e0ac80536634423cd
Data snapshot: 9754208a618c6bcf5b78432fb1372eeae2133e3155985ecda4dca5f50702c103
Source manifest: b5e68e78712891ac750597a84742525b6529e65050a568ddee247f6021d1f945
Model configuration: 4beecd53003c934c6ee8bfa2c7a2fb9e64b98fa7623a01473e271515699e483a
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty 97477c9da8c60468730686c46f5d2f6a8170e62f44c84963b1d63990f9b52632)
Simulation: 100000 paths; seed {"entropy":202608302216,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 56.4% | 62.0% |
| Ace given first serve in | 4.8% | 12.5% |
| Returnable first-serve points won | 64.3% | 68.7% |
| Double fault given second-serve opportunity | 18.0% | 7.5% |
| Playable second-serve points won | 56.6% | 59.5% |
| Derived first-serve points won | 66.1% | 72.6% |
| Derived second-serve points won | 46.4% | 55.1% |
| Overall service points won | 57.5% | 65.9% |
| Implied hold probability | 68.1% | 84.5% |
| Ace rate / service point | 2.7% | 7.8% |
| Double-fault rate / service point | 7.8% | 2.9% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.0889 | 0.0834 | 320.0 | 320.0 | 1700.6 | 1715.2 |
| A | 0.3118 | 0.2809 | 338.4 | 338.4 | 971.4 | 1061.3 |
| Q1 | 0.1512 | 0.1505 | 368289.7 | 368289.7 | 894.7 | 963.7 |
| D | 0.1673 | 0.2313 | 290.3 | 290.3 | 729.2 | 653.9 |
| Q2 | 0.1873 | 0.1882 | 1000000.0 | 1000000.0 | 588.8 | 601.5 |

### B6 retirement and C6 inactivity

- player_b97a5f95-4205-58f3-96ca-c63411436250: inactivity 3 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_ad84ea61-9ba3-53b4-86bd-395303127dc6: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 101.6 minutes
- Duration 10/50/90% quantiles: 56.4 / 95.6 / 156.4 minutes
- Duration data grade: B
- Duration artifact: cd801ed4db698e42df9223a07eac015bfec2cc5a97a50c78338d4eef64bdb65d
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -2.61 minutes

## Core simulated outputs

- player_b97a5f95-4205-58f3-96ca-c63411436250: win 24.8%; expected aces 1.97; expected DFs 5.49; expected breaks 2.03
- player_ad84ea61-9ba3-53b4-86bd-395303127dc6: win 75.2%; expected aces 5.25; expected DFs 1.94; expected breaks 3.41
- Expected total games: 21.74
- Total-games 10/50/90% quantiles: 13.0 / 20.0 / 32.0
- At least one tiebreak: 33.1%
- Deciding set: 28.9%
- Expected total breaks: 5.44
- Retirement probability: 3.2%
- Exact score probabilities: player_ad84ea61-9ba3-53b4-86bd-395303127dc6 2-0 55.7%; player_ad84ea61-9ba3-53b4-86bd-395303127dc6 2-1 20.4%; player_b97a5f95-4205-58f3-96ca-c63411436250 2-0 14.9%; player_b97a5f95-4205-58f3-96ca-c63411436250 2-1 9.0%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_b97a5f95-4205-58f3-96ca-c63411436250) | 24815 | 75185 | 0 | 0 | 100000 | 24.815% | 25% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_b97a5f95-4205-58f3-96ca-c63411436250,player_ad84ea61-9ba3-53b4-86bd-395303127dc6) | 9547 | 87214 | 3239 | 0 | 96761 | 9.867% | 10% | -- | legacy fixed-sample | 100000 | 96.76% | SUPPORTED | -- | A |
| DF_COMPARE(player_b97a5f95-4205-58f3-96ca-c63411436250,player_ad84ea61-9ba3-53b4-86bd-395303127dc6) | 81670 | 15091 | 3239 | 0 | 96761 | 84.404% | 84% | -- | legacy fixed-sample | 100000 | 96.76% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,1) | 33062 | 64225 | 2713 | 0 | 97287 | 33.984% | 34% | -- | legacy fixed-sample | 100000 | 97.29% | SUPPORTED | -- | A |
| DURATION_MIN(>,85.0) | 61997 | 35427 | 2576 | 0 | 97424 | 63.636% | 64% | -- | legacy fixed-sample | 100000 | 97.42% | SUPPORTED | -- | B |

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:97477c9da8c60468730686c46f5d2f6a8170e62f44c84963b1d63990f9b52632
- Warning: HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE
- Warning: UNKNOWN_INDOOR_ROOF_STATE
- Warning: MISSING_CURRENT_CONDITION:roof state
- Check: MATCH_WIN_PROBABILITIES_SUM_TO_ONE
- Check: EXACT_SCORE_PROBABILITIES_SUM_TO_ONE_CONDITIONAL_ON_COMPLETION
- Check: FROZEN_V1_PERFORMANCE_DRAWS_INDEPENDENT
- Check: ALL_PROP_ESTIMATES_SHARE_ONE_SIMULATION_BATCH
- Check: DURATION_CONDITIONAL_ON_REALIZED_JOINT_PATH
- Check: DURATION_DRAW_CANNOT_ALTER_SCORE_OR_STATISTICS

LOCK STATUS: LOCKED
