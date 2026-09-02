# LOCKED MATCH CARD

Lock ID: TMV1-ATP-eda526d2f1d3894edcae-L1
Canonical match identity: match_eda526d2f1d3894edcae3499a2ed742d9eaed402f1e92e32ed1e3d8af2fe6267
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T21:48:07.271407+00:00
Information cutoff: 2026-09-01T20:50:16.183925+00:00
Event: US Open
Match: player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2 v player_f3daabf9-7d11-53e4-b281-4f8b1823136f
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
Simulation: 100000 paths; seed {"entropy":202608301221,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 65.6% | 65.8% |
| Ace given first serve in | 16.8% | 10.7% |
| Returnable first-serve points won | 68.7% | 67.1% |
| Double fault given second-serve opportunity | 3.9% | 10.3% |
| Playable second-serve points won | 51.2% | 56.9% |
| Derived first-serve points won | 74.0% | 70.6% |
| Derived second-serve points won | 49.2% | 51.0% |
| Overall service points won | 65.5% | 63.9% |
| Implied hold probability | 83.7% | 81.1% |
| Ace rate / service point | 11.1% | 7.0% |
| Double-fault rate / service point | 1.3% | 3.5% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.0742 | 0.0781 | 258.7 | 258.7 | 2854.7 | 2532.7 |
| A | 0.1909 | 0.2029 | 217.9 | 217.9 | 1886.6 | 1686.9 |
| Q1 | 0.1322 | 0.1285 | 572197.7 | 572197.7 | 1577.1 | 1538.8 |
| D | 0.2794 | 0.1897 | 453.4 | 453.4 | 968.1 | 845.8 |
| Q2 | 0.1554 | 0.1600 | 1000000.0 | 1000000.0 | 930.1 | 760.2 |

### B6 retirement and C6 inactivity

- player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_f3daabf9-7d11-53e4-b281-4f8b1823136f: inactivity 2 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 167.7 minutes
- Duration 10/50/90% quantiles: 83.0 / 168.2 / 249.3 minutes
- Duration data grade: B
- Duration artifact: e09190b75097d9e9427e4c7751366d45c8844b153817d1490164331a9add21c4
- Duration display policy: duration-display-nearest-half-up/v1
- Current-event duration effect: -0.83 minutes

## Core simulated outputs

- player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2: win 56.6%; expected aces 12.96; expected DFs 1.60; expected breaks 3.89
- player_f3daabf9-7d11-53e4-b281-4f8b1823136f: win 43.4%; expected aces 8.34; expected DFs 4.18; expected breaks 3.53
- Expected total games: 37.85
- Total-games 10/50/90% quantiles: 21.0 / 38.0 / 54.0
- At least one tiebreak: 55.7%
- Deciding set: 24.9%
- Expected total breaks: 7.43
- Retirement probability: 3.4%
- Exact score probabilities: player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2 3-0 25.2%; player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2 3-1 17.4%; player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2 3-2 13.8%; player_f3daabf9-7d11-53e4-b281-4f8b1823136f 3-0 19.4%; player_f3daabf9-7d11-53e4-b281-4f8b1823136f 3-1 12.5%; player_f3daabf9-7d11-53e4-b281-4f8b1823136f 3-2 11.7%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_f3daabf9-7d11-53e4-b281-4f8b1823136f) | 43397 | 56603 | 0 | 0 | 100000 | 43.397% | 43% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_f3daabf9-7d11-53e4-b281-4f8b1823136f,player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2) | 17981 | 78643 | 3376 | 0 | 96624 | 18.609% | 19% | -- | legacy fixed-sample | 100000 | 96.62% | SUPPORTED | -- | A |
| DF_COMPARE(player_f3daabf9-7d11-53e4-b281-4f8b1823136f,player_50d26cca-b544-5c93-b8eb-61f5e4b67ee2) | 74670 | 21954 | 3376 | 0 | 96624 | 77.279% | 77% | -- | legacy fixed-sample | 100000 | 96.62% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,2) | 26997 | 69959 | 3044 | 0 | 96956 | 27.845% | 28% | -- | legacy fixed-sample | 100000 | 96.96% | SUPPORTED | -- | A |
| DURATION_MIN(>,185.0) | 40774 | 56141 | 3085 | 0 | 96915 | 42.072% | 42% | -- | legacy fixed-sample | 100000 | 96.91% | SUPPORTED | -- | B |

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
