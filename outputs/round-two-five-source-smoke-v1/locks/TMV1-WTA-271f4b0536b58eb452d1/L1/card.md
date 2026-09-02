# LOCKED MATCH CARD

Lock ID: TMV1-WTA-271f4b0536b58eb452d1-L1
Canonical match identity: match_271f4b0536b58eb452d14eabce5e38320dd3e46799fa1484811d4f402a338d05
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T20:50:38.124067+00:00
Information cutoff: 2026-09-01T20:50:16.232366+00:00
Event: US Open
Match: player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636 v player_16892af7-47cb-5adb-ae40-5d5617f870ec
Draw / round: WTA singles / R64
Scheduled start: 2026-09-02T15:00:00+00:00
Format: best of 3; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50 by path)
Conditions scenario: central
Model snapshot: 488a8f77251e3eb929affcf4c81082201b3597cc899c388e0ac80536634423cd
Data snapshot: 9754208a618c6bcf5b78432fb1372eeae2133e3155985ecda4dca5f50702c103
Source manifest: b5e68e78712891ac750597a84742525b6529e65050a568ddee247f6021d1f945
Model configuration: 4beecd53003c934c6ee8bfa2c7a2fb9e64b98fa7623a01473e271515699e483a
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty 47d286fe7abefd3e204335581dc2f9b39400545a20fd39e7cd9390a317b971a3)
Simulation: 1 paths; seed {"entropy":202608302207,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 58.9% | 53.6% |
| Ace given first serve in | 12.8% | 7.4% |
| Returnable first-serve points won | 66.4% | 65.0% |
| Double fault given second-serve opportunity | 7.4% | 16.9% |
| Playable second-serve points won | 52.8% | 54.9% |
| Derived first-serve points won | 70.7% | 67.6% |
| Derived second-serve points won | 49.0% | 45.6% |
| Overall service points won | 61.7% | 57.4% |
| Implied hold probability | 77.1% | 67.9% |
| Ace rate / service point | 7.6% | 4.0% |
| Double-fault rate / service point | 3.0% | 7.9% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.0848 | 0.0884 | 320.0 | 320.0 | 1820.7 | 1781.1 |
| A | 0.2801 | 0.3239 | 338.4 | 338.4 | 1072.8 | 985.0 |
| Q1 | 0.1559 | 0.1546 | 368289.7 | 368289.7 | 971.8 | 883.6 |
| D | 0.2356 | 0.1681 | 290.3 | 290.3 | 747.9 | 796.1 |
| Q2 | 0.1867 | 0.1854 | 1000000.0 | 1000000.0 | 693.9 | 665.9 |

### B6 retirement and C6 inactivity

- player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_16892af7-47cb-5adb-ae40-5d5617f870ec: inactivity 3 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 126.7 minutes
- Duration 10/50/90% quantiles: 126.7 / 126.7 / 126.7 minutes
- Duration data grade: B
- Duration artifact: cd801ed4db698e42df9223a07eac015bfec2cc5a97a50c78338d4eef64bdb65d
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -2.61 minutes

## Core simulated outputs

- player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636: win 0.0%; expected aces 10.00; expected DFs 3.00; expected breaks 5.00
- player_16892af7-47cb-5adb-ae40-5d5617f870ec: win 100.0%; expected aces 5.00; expected DFs 4.00; expected breaks 8.00
- Expected total games: 26.00
- Total-games 10/50/90% quantiles: 26.0 / 26.0 / 26.0
- At least one tiebreak: 0.0%
- Deciding set: 100.0%
- Expected total breaks: 13.00
- Retirement probability: 0.0%
- Exact score probabilities: player_16892af7-47cb-5adb-ae40-5d5617f870ec 2-1 100.0%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636) | 0 | 1 | 0 | 0 | 1 | 0.000% | 1% | -- | legacy fixed-sample | 1 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636,player_16892af7-47cb-5adb-ae40-5d5617f870ec) | 1 | 0 | 0 | 0 | 1 | 100.000% | 99% | -- | legacy fixed-sample | 1 | 100.00% | SUPPORTED | -- | A |
| DF_COMPARE(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636,player_16892af7-47cb-5adb-ae40-5d5617f870ec) | 0 | 1 | 0 | 0 | 1 | 0.000% | 1% | -- | legacy fixed-sample | 1 | 100.00% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,1) | 0 | 1 | 0 | 0 | 1 | 0.000% | 1% | -- | legacy fixed-sample | 1 | 100.00% | SUPPORTED | -- | A |
| DURATION_MIN(>,100.0) | 1 | 0 | 0 | 0 | 1 | 100.000% | 99% | -- | legacy fixed-sample | 1 | 100.00% | SUPPORTED | -- | B |

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:47d286fe7abefd3e204335581dc2f9b39400545a20fd39e7cd9390a317b971a3
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
