# LOCKED MATCH CARD

Lock ID: TMV1-WTA-271f4b0536b58eb452d1-L1
Canonical match identity: match_271f4b0536b58eb452d14eabce5e38320dd3e46799fa1484811d4f402a338d05
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T21:45:26.978248+00:00
Information cutoff: 2026-09-01T20:50:16.183925+00:00
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
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty bf3fbfff52ed21b5645d62434061f739b7b74e0584cf30ada6be841c0f7cad71)
Simulation: 100000 paths; seed {"entropy":202608302207,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

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

- Expected duration: 96.6 minutes
- Duration 10/50/90% quantiles: 48.3 / 91.7 / 150.8 minutes
- Duration data grade: B
- Duration artifact: cd801ed4db698e42df9223a07eac015bfec2cc5a97a50c78338d4eef64bdb65d
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -2.61 minutes

## Core simulated outputs

- player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636: win 64.5%; expected aces 5.43; expected DFs 2.19; expected breaks 3.59
- player_16892af7-47cb-5adb-ae40-5d5617f870ec: win 35.5%; expected aces 2.98; expected DFs 5.70; expected breaks 2.82
- Expected total games: 22.34
- Total-games 10/50/90% quantiles: 14.0 / 21.0 / 32.0
- At least one tiebreak: 33.1%
- Deciding set: 34.1%
- Expected total breaks: 6.41
- Retirement probability: 2.0%
- Exact score probabilities: player_16892af7-47cb-5adb-ae40-5d5617f870ec 2-0 21.7%; player_16892af7-47cb-5adb-ae40-5d5617f870ec 2-1 13.4%; player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636 2-0 43.7%; player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636 2-1 21.1%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636) | 64518 | 35482 | 0 | 0 | 100000 | 64.518% | 65% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636,player_16892af7-47cb-5adb-ae40-5d5617f870ec) | 69816 | 28136 | 2048 | 0 | 97952 | 71.276% | 71% | -- | legacy fixed-sample | 100000 | 97.95% | SUPPORTED | -- | A |
| DF_COMPARE(player_eb7315d9-e8c2-51c0-9ea7-6fc8fced8636,player_16892af7-47cb-5adb-ae40-5d5617f870ec) | 9283 | 88669 | 2048 | 0 | 97952 | 9.477% | 9% | -- | legacy fixed-sample | 100000 | 97.95% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,1) | 33104 | 65156 | 1740 | 0 | 98260 | 33.690% | 34% | -- | legacy fixed-sample | 100000 | 98.26% | SUPPORTED | -- | A |
| DURATION_MIN(>,100.0) | 42206 | 56000 | 1794 | 0 | 98206 | 42.977% | 43% | -- | legacy fixed-sample | 100000 | 98.21% | SUPPORTED | -- | B |

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:bf3fbfff52ed21b5645d62434061f739b7b74e0584cf30ada6be841c0f7cad71
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
