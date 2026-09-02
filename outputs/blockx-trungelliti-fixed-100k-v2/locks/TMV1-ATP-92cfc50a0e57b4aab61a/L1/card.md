# LOCKED MATCH CARD

Lock ID: TMV1-ATP-92cfc50a0e57b4aab61a-L1
Canonical match identity: match_92cfc50a0e57b4aab61a1b3105058a9f47d5875beaa205d9387cbff91c4cfe1a
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T21:43:39.821200+00:00
Information cutoff: 2026-09-01T20:50:16.183925+00:00
Event: US Open
Match: player_9403311e-fbaf-51f1-8395-03ad26efd937 v player_4ced8d9b-d3ba-56d7-8098-17166dea551d
Draw / round: ATP singles / R64
Scheduled start: 2026-09-02T15:00:00+00:00
Format: best of 5; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50 by path)
Conditions scenario: central
Model snapshot: fcd43e372ca11d3b32359da58659984780bebe284b291f51cb267ea010f0a712
Data snapshot: 0e3564a97faeb96f05edf36decc158993f1507ee7a4e3c193aec5e57b3756b71
Source manifest: b5e68e78712891ac750597a84742525b6529e65050a568ddee247f6021d1f945
Model configuration: 4beecd53003c934c6ee8bfa2c7a2fb9e64b98fa7623a01473e271515699e483a
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty bf3fbfff52ed21b5645d62434061f739b7b74e0584cf30ada6be841c0f7cad71)
Simulation: 100000 paths; seed {"entropy":202608301215,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 65.2% | 55.8% |
| Ace given first serve in | 24.5% | 3.4% |
| Returnable first-serve points won | 89.7% | 54.0% |
| Double fault given second-serve opportunity | 8.3% | 14.5% |
| Playable second-serve points won | 92.2% | 34.5% |
| Derived first-serve points won | 92.2% | 55.5% |
| Derived second-serve points won | 84.6% | 29.5% |
| Overall service points won | 89.6% | 44.0% |
| Implied hold probability | 99.8% | 35.4% |
| Ace rate / service point | 16.0% | 1.9% |
| Double-fault rate / service point | 2.9% | 6.4% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.1939 | 0.6192 | 258.7 | 258.7 | 319.4 | 143.5 |
| A | 1.1313 | 1.4324 | 217.9 | 217.9 | 208.6 | 86.0 |
| Q1 | 1.3066 | 0.8155 | 572197.7 | 572197.7 | 180.7 | 81.8 |
| D | 0.5243 | 1.2225 | 453.4 | 453.4 | 110.8 | 57.5 |
| Q2 | 1.5421 | 0.9832 | 1000000.0 | 1000000.0 | 102.4 | 51.8 |

### B6 retirement and C6 inactivity

- player_9403311e-fbaf-51f1-8395-03ad26efd937: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_4ced8d9b-d3ba-56d7-8098-17166dea551d: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 97.0 minutes
- Duration 10/50/90% quantiles: 54.5 / 86.8 / 152.1 minutes
- Duration data grade: B
- Duration artifact: e09190b75097d9e9427e4c7751366d45c8844b153817d1490164331a9add21c4
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -0.83 minutes

## Core simulated outputs

- player_9403311e-fbaf-51f1-8395-03ad26efd937: win 94.2%; expected aces 11.47; expected DFs 2.03; expected breaks 6.32
- player_4ced8d9b-d3ba-56d7-8098-17166dea551d: win 5.8%; expected aces 3.21; expected DFs 6.36; expected breaks 0.67
- Expected total games: 25.09
- Total-games 10/50/90% quantiles: 18.0 / 22.0 / 37.0
- At least one tiebreak: 20.3%
- Deciding set: 2.5%
- Expected total breaks: 6.99
- Retirement probability: 2.8%
- Exact score probabilities: player_4ced8d9b-d3ba-56d7-8098-17166dea551d 3-0 2.5%; player_4ced8d9b-d3ba-56d7-8098-17166dea551d 3-1 1.2%; player_4ced8d9b-d3ba-56d7-8098-17166dea551d 3-2 0.9%; player_9403311e-fbaf-51f1-8395-03ad26efd937 3-0 89.4%; player_9403311e-fbaf-51f1-8395-03ad26efd937 3-1 4.4%; player_9403311e-fbaf-51f1-8395-03ad26efd937 3-2 1.6%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_9403311e-fbaf-51f1-8395-03ad26efd937) | 94160 | 5840 | 0 | 0 | 100000 | 94.160% | 94% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_9403311e-fbaf-51f1-8395-03ad26efd937,player_4ced8d9b-d3ba-56d7-8098-17166dea551d) | 80628 | 16540 | 2832 | 0 | 97168 | 82.978% | 83% | -- | legacy fixed-sample | 100000 | 97.17% | SUPPORTED | -- | A |
| DF_COMPARE(player_9403311e-fbaf-51f1-8395-03ad26efd937,player_4ced8d9b-d3ba-56d7-8098-17166dea551d) | 20005 | 77163 | 2832 | 0 | 97168 | 20.588% | 21% | -- | legacy fixed-sample | 100000 | 97.17% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,1) | 20271 | 77372 | 2357 | 0 | 97643 | 20.760% | 21% | -- | legacy fixed-sample | 100000 | 97.64% | SUPPORTED | -- | A |
| DURATION_MIN(>,125.0) | 19355 | 77999 | 2646 | 0 | 97354 | 19.881% | 20% | -- | legacy fixed-sample | 100000 | 97.35% | SUPPORTED | -- | B |

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:bf3fbfff52ed21b5645d62434061f739b7b74e0584cf30ada6be841c0f7cad71
- Warning: HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE
- Warning: UNKNOWN_INDOOR_ROOF_STATE
- Warning: MISSING_CURRENT_CONDITION:roof state
- Warning: SPARSE_PLAYER_COMPONENT_HISTORY
- Check: MATCH_WIN_PROBABILITIES_SUM_TO_ONE
- Check: EXACT_SCORE_PROBABILITIES_SUM_TO_ONE_CONDITIONAL_ON_COMPLETION
- Check: FROZEN_V1_PERFORMANCE_DRAWS_INDEPENDENT
- Check: ALL_PROP_ESTIMATES_SHARE_ONE_SIMULATION_BATCH
- Check: DURATION_CONDITIONAL_ON_REALIZED_JOINT_PATH
- Check: DURATION_DRAW_CANNOT_ALTER_SCORE_OR_STATISTICS

LOCK STATUS: LOCKED
