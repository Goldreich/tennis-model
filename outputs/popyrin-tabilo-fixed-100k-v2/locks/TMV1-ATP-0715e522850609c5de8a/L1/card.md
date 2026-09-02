# LOCKED MATCH CARD

Lock ID: TMV1-ATP-0715e522850609c5de8a-L1
Canonical match identity: match_0715e522850609c5de8a84b148498c04c024dfdfd394677f6f4a398aeedc2cf9
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T21:48:04.902775+00:00
Information cutoff: 2026-09-01T20:50:16.183925+00:00
Event: US Open
Match: player_4f0fc189-678e-5dda-9fc1-8ee86830a805 v player_7964db64-8d38-5bff-936f-43e3dfa9f65c
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
Simulation: 100000 paths; seed {"entropy":202608301202,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 61.7% | 66.3% |
| Ace given first serve in | 30.0% | 15.4% |
| Returnable first-serve points won | 70.1% | 67.8% |
| Double fault given second-serve opportunity | 10.7% | 8.0% |
| Playable second-serve points won | 55.0% | 56.8% |
| Derived first-serve points won | 79.0% | 72.7% |
| Derived second-serve points won | 49.1% | 52.3% |
| Overall service points won | 67.6% | 65.8% |
| Implied hold probability | 86.9% | 84.3% |
| Ace rate / service point | 18.5% | 10.2% |
| Double-fault rate / service point | 4.1% | 2.7% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.0812 | 0.0913 | 258.7 | 258.7 | 2376.2 | 1968.4 |
| A | 0.1943 | 0.2125 | 217.9 | 217.9 | 1494.1 | 1270.7 |
| Q1 | 0.1570 | 0.1496 | 572197.7 | 572197.7 | 1225.4 | 1110.8 |
| D | 0.1886 | 0.2442 | 453.4 | 453.4 | 882.1 | 697.7 |
| Q2 | 0.1809 | 0.1794 | 1000000.0 | 1000000.0 | 791.2 | 642.5 |

### B6 retirement and C6 inactivity

- player_4f0fc189-678e-5dda-9fc1-8ee86830a805: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_7964db64-8d38-5bff-936f-43e3dfa9f65c: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 168.0 minutes
- Duration 10/50/90% quantiles: 86.4 / 167.9 / 248.7 minutes
- Duration data grade: B
- Duration artifact: e09190b75097d9e9427e4c7751366d45c8844b153817d1490164331a9add21c4
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -0.83 minutes

## Core simulated outputs

- player_4f0fc189-678e-5dda-9fc1-8ee86830a805: win 55.8%; expected aces 21.62; expected DFs 4.85; expected breaks 3.36
- player_7964db64-8d38-5bff-936f-43e3dfa9f65c: win 44.2%; expected aces 12.12; expected DFs 3.23; expected breaks 2.97
- Expected total games: 38.45
- Total-games 10/50/90% quantiles: 22.0 / 39.0 / 55.0
- At least one tiebreak: 60.0%
- Deciding set: 24.3%
- Expected total breaks: 6.33
- Retirement probability: 3.3%
- Exact score probabilities: player_4f0fc189-678e-5dda-9fc1-8ee86830a805 3-0 25.5%; player_4f0fc189-678e-5dda-9fc1-8ee86830a805 3-1 17.0%; player_4f0fc189-678e-5dda-9fc1-8ee86830a805 3-2 13.3%; player_7964db64-8d38-5bff-936f-43e3dfa9f65c 3-0 19.4%; player_7964db64-8d38-5bff-936f-43e3dfa9f65c 3-1 13.2%; player_7964db64-8d38-5bff-936f-43e3dfa9f65c 3-2 11.6%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_4f0fc189-678e-5dda-9fc1-8ee86830a805) | 55822 | 44178 | 0 | 0 | 100000 | 55.822% | 56% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_4f0fc189-678e-5dda-9fc1-8ee86830a805,player_7964db64-8d38-5bff-936f-43e3dfa9f65c) | 85762 | 10927 | 3311 | 0 | 96689 | 88.699% | 89% | -- | legacy fixed-sample | 100000 | 96.69% | SUPPORTED | -- | A |
| DF_COMPARE(player_4f0fc189-678e-5dda-9fc1-8ee86830a805,player_7964db64-8d38-5bff-936f-43e3dfa9f65c) | 60645 | 36044 | 3311 | 0 | 96689 | 62.722% | 63% | -- | legacy fixed-sample | 100000 | 96.69% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,2) | 31544 | 65533 | 2923 | 0 | 97077 | 32.494% | 32% | -- | legacy fixed-sample | 100000 | 97.08% | SUPPORTED | -- | A |
| DURATION_MIN(>,165.0) | 51261 | 45878 | 2861 | 0 | 97139 | 52.771% | 53% | -- | legacy fixed-sample | 100000 | 97.14% | SUPPORTED | -- | B |

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
