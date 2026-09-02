# LOCKED MATCH CARD

Lock ID: TMV1-WTA-210fc1f9ca2b002f6416-L1
Canonical match identity: match_210fc1f9ca2b002f6416bf92b4900f26c683adb74a3786a377fd484bb798aeaa
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T20:25:50.582749+00:00
Information cutoff: 2026-09-01T19:38:27.928754+00:00
Event: US Open
Match: player_a9454392-7fa6-5f67-a6b4-86f1d55a8438 v player_0f4d8c09-0cb2-53f1-847f-522541cb9131
Draw / round: WTA singles / R128
Scheduled start: 2026-09-01T19:38:27.928755+00:00
Format: best of 3; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50 by path)
Conditions scenario: central
Model snapshot: 488a8f77251e3eb929affcf4c81082201b3597cc899c388e0ac80536634423cd
Data snapshot: 9754208a618c6bcf5b78432fb1372eeae2133e3155985ecda4dca5f50702c103
Source manifest: b5e68e78712891ac750597a84742525b6529e65050a568ddee247f6021d1f945
Model configuration: 4beecd53003c934c6ee8bfa2c7a2fb9e64b98fa7623a01473e271515699e483a
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty bb23e0a185ea66d15766f3a0e27bb8adba49d9154dbe62b9f5fba68fb702670d)
Simulation: 100000 paths; seed {"entropy":202608302137,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 53.5% | 65.3% |
| Ace given first serve in | 16.2% | 2.7% |
| Returnable first-serve points won | 66.7% | 46.4% |
| Double fault given second-serve opportunity | 13.6% | 4.7% |
| Playable second-serve points won | 56.3% | 44.7% |
| Derived first-serve points won | 72.1% | 47.8% |
| Derived second-serve points won | 48.6% | 42.6% |
| Overall service points won | 61.2% | 46.0% |
| Implied hold probability | 76.0% | 40.1% |
| Ace rate / service point | 8.7% | 1.8% |
| Double-fault rate / service point | 6.3% | 1.6% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.0818 | 0.1964 | 320.0 | 320.0 | 1931.4 | 264.9 |
| A | 0.5226 | 0.8096 | 338.4 | 338.4 | 1048.3 | 172.7 |
| Q1 | 0.2388 | 0.2399 | 368289.7 | 368289.7 | 986.3 | 168.8 |
| D | 0.1677 | 0.7133 | 290.3 | 290.3 | 883.1 | 92.3 |
| Q2 | 0.3406 | 0.3197 | 1000000.0 | 1000000.0 | 762.5 | 88.2 |

### B6 retirement and C6 inactivity

- player_a9454392-7fa6-5f67-a6b4-86f1d55a8438: inactivity 15 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_0f4d8c09-0cb2-53f1-847f-522541cb9131: inactivity 4 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 81.5 minutes
- Duration 10/50/90% quantiles: 44.3 / 74.6 / 133.1 minutes
- Duration data grade: B
- Duration artifact: cd801ed4db698e42df9223a07eac015bfec2cc5a97a50c78338d4eef64bdb65d
- Duration display policy: duration-display-unresolved/v1
- Current-event duration effect: -2.61 minutes
- Warning: official whole-minute conversion affects at least one requested duration threshold.

## Core simulated outputs

- player_a9454392-7fa6-5f67-a6b4-86f1d55a8438: win 82.7%; expected aces 5.45; expected DFs 3.80; expected breaks 5.17
- player_0f4d8c09-0cb2-53f1-847f-522541cb9131: win 17.3%; expected aces 1.47; expected DFs 1.21; expected breaks 2.70
- Expected total games: 18.92
- Total-games 10/50/90% quantiles: 13.0 / 17.0 / 28.0
- At least one tiebreak: 18.1%
- Deciding set: 17.8%
- Expected total breaks: 7.88
- Retirement probability: 2.5%
- Exact score probabilities: player_0f4d8c09-0cb2-53f1-847f-522541cb9131 2-0 11.1%; player_0f4d8c09-0cb2-53f1-847f-522541cb9131 2-1 5.2%; player_a9454392-7fa6-5f67-a6b4-86f1d55a8438 2-0 70.8%; player_a9454392-7fa6-5f67-a6b4-86f1d55a8438 2-1 12.9%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_a9454392-7fa6-5f67-a6b4-86f1d55a8438) | 82687 | 17313 | 0 | 0 | 100000 | 82.687% | 83% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_a9454392-7fa6-5f67-a6b4-86f1d55a8438,player_0f4d8c09-0cb2-53f1-847f-522541cb9131) | 82382 | 15093 | 2525 | 0 | 97475 | 84.516% | 85% | -- | legacy fixed-sample | 100000 | 97.47% | SUPPORTED | -- | A |
| DF_COMPARE(player_a9454392-7fa6-5f67-a6b4-86f1d55a8438,player_0f4d8c09-0cb2-53f1-847f-522541cb9131) | 76030 | 21445 | 2525 | 0 | 97475 | 77.999% | 78% | -- | legacy fixed-sample | 100000 | 97.47% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,1) | 18083 | 79613 | 2304 | 0 | 97696 | 18.509% | 19% | -- | legacy fixed-sample | 100000 | 97.70% | SUPPORTED | -- | A |
| DURATION_MIN(>,85.0) | 35802 | 60908 | 2238 | 1052 | 96710 | 37.020% | -- | -- | legacy fixed-sample | 100000 | 96.71% | SETTLEMENT_BLOCKED | -- | B |
  - Policy issue for `e9320382994c4edff675e871908f8bc1d5076e4bae547cdb658efb779b847d9e`: settlement semantics unresolved on one or more simulated paths
  - Support gate for `e9320382994c4edff675e871908f8bc1d5076e4bae547cdb658efb779b847d9e`: DURATION_DISPLAY_POLICY_BOUNDARY_SENSITIVE — one or more paths differ under the preserved official-minute display-policy candidates
  - Display-policy sensitivity for `e9320382994c4edff675e871908f8bc1d5076e4bae547cdb658efb779b847d9e`: 36.62% to 37.70%

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:bb23e0a185ea66d15766f3a0e27bb8adba49d9154dbe62b9f5fba68fb702670d
- Warning: HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE
- Warning: UNKNOWN_INDOOR_ROOF_STATE
- Warning: DURATION_OFFICIAL_MINUTE_CONVERSION_BOUNDARY_SENSITIVE
- Warning: MISSING_CURRENT_CONDITION:roof state
- Warning: SPARSE_PLAYER_COMPONENT_HISTORY
- Warning: POLICY_BLOCKED_PROP:PropSpec
- Warning: PROP_SETTLEMENT_BLOCKED:DURATION_DISPLAY_POLICY_BOUNDARY_SENSITIVE
- Check: MATCH_WIN_PROBABILITIES_SUM_TO_ONE
- Check: EXACT_SCORE_PROBABILITIES_SUM_TO_ONE_CONDITIONAL_ON_COMPLETION
- Check: FROZEN_V1_PERFORMANCE_DRAWS_INDEPENDENT
- Check: ALL_PROP_ESTIMATES_SHARE_ONE_SIMULATION_BATCH
- Check: DURATION_CONDITIONAL_ON_REALIZED_JOINT_PATH
- Check: DURATION_DRAW_CANNOT_ALTER_SCORE_OR_STATISTICS

LOCK STATUS: LOCKED
