# LOCKED MATCH CARD

Lock ID: TMV1-ATP-c7835a68aec717b6f078-L1
Canonical match identity: match_c7835a68aec717b6f0785efb32cde555a9ad158101ec0d677891abc3fda27269
Framework: Tennis Model v1.0
Created (UTC): 2026-09-01T20:32:58.196536+00:00
Information cutoff: 2026-09-01T19:38:27.928642+00:00
Event: US Open
Match: player_72f48298-3469-51b1-999e-6ace673695ba v player_0b714ccd-16ea-5950-9a2f-5fed169b9763
Draw / round: ATP singles / R128
Scheduled start: 2026-09-01T19:38:27.928643+00:00
Format: best of 5; standard TB to 7; deciding TB to 10 at 6-6
First server: unknown (50/50 by path)
Conditions scenario: central
Model snapshot: 016e90567b633d5e7ba9036dd5b125ccbcb94a8eead2feb542b8024b8cf4b383
Data snapshot: 0e3564a97faeb96f05edf36decc158993f1507ee7a4e3c193aec5e57b3756b71
Source manifest: 6bb5862dddcab4884d3fff1ed1a8479179ea70dde488a5fc990cfd3fddcc5890
Model configuration: 4beecd53003c934c6ee8bfa2c7a2fb9e64b98fa7623a01473e271515699e483a
Code: 042979b94a30556f26b0a871384b17e24b6afb0f (dirty ab10d927c4f9bb3ff3f99ccd67314371648e87edbc52379baa51861a08cea9ab)
Simulation: 100000 paths; seed {"entropy":202608301109,"spawn_key":[],"pool_size":4,"n_children_spawned":0}; settlement policy tennis-model-v1.0/settlement-v1

## Matchup parameters

| Parameter | Player A serving | Player B serving |
|---|---:|---:|
| First serve in | 60.6% | 58.4% |
| Ace given first serve in | 16.4% | 17.5% |
| Returnable first-serve points won | 69.2% | 74.2% |
| Double fault given second-serve opportunity | 9.4% | 11.4% |
| Playable second-serve points won | 49.9% | 59.0% |
| Derived first-serve points won | 74.3% | 78.7% |
| Derived second-serve points won | 45.2% | 52.2% |
| Overall service points won | 62.8% | 67.7% |
| Implied hold probability | 79.1% | 87.1% |
| Ace rate / service point | 10.0% | 10.2% |
| Double-fault rate / service point | 3.7% | 4.7% |

### Primitive uncertainty diagnostics

| Component | A logit SD | B logit SD | A concentration | B concentration | A weighted trials | B weighted trials |
|---|---:|---:|---:|---:|---:|---:|
| F | 3.0228 | 0.2082 | 258.7 | 258.7 | -- | 230.3 |
| A | 4.2615 | 4.2559 | 217.9 | 217.9 | -- | 134.0 |
| Q1 | 3.0340 | 3.0388 | 572197.7 | 572197.7 | -- | 107.2 |
| D | 4.2427 | 0.4669 | 453.4 | 453.4 | -- | 96.3 |
| Q2 | 3.0386 | 3.0387 | 1000000.0 | 1000000.0 | -- | 85.4 |

### B6 retirement and C6 inactivity

- player_72f48298-3469-51b1-999e-6ace673695ba: inactivity 13 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)
- player_0b714ccd-16ea-5950-9a2f-5fed169b9763: inactivity 4 days; hard multiplier 1.000000; variance factor 1.000000; retirement scenario central (eta=0, w=1)

### Match duration

- Expected duration: 145.2 minutes
- Duration 10/50/90% quantiles: 60.6 / 110.7 / 208.1 minutes
- Duration data grade: B
- Duration artifact: 283330a498b04ede7d71e5d95aa8c81a10f57915ee7551871ca39e0ea8061be2
- Duration display policy: duration-display-unresolved/v1
- Current-event duration effect: -0.81 minutes
- Warning: official whole-minute conversion affects at least one requested duration threshold.

## Core simulated outputs

- player_72f48298-3469-51b1-999e-6ace673695ba: win 45.4%; expected aces 17.12; expected DFs 9.83; expected breaks 3.97
- player_0b714ccd-16ea-5950-9a2f-5fed169b9763: win 54.6%; expected aces 18.47; expected DFs 5.18; expected breaks 5.20
- Expected total games: 29.27
- Total-games 10/50/90% quantiles: 18.0 / 26.0 / 48.0
- At least one tiebreak: 36.3%
- Deciding set: 6.0%
- Expected total breaks: 9.17
- Retirement probability: 3.5%
- Exact score probabilities: player_0b714ccd-16ea-5950-9a2f-5fed169b9763 3-0 46.3%; player_0b714ccd-16ea-5950-9a2f-5fed169b9763 3-1 5.4%; player_0b714ccd-16ea-5950-9a2f-5fed169b9763 3-2 3.0%; player_72f48298-3469-51b1-999e-6ace673695ba 3-0 36.3%; player_72f48298-3469-51b1-999e-6ace673695ba 3-1 5.9%; player_72f48298-3469-51b1-999e-6ace673695ba 3-2 3.2%

## Championship markets

| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | Model integer | 99% anytime-valid CS | MC status | Final paths | P(settled) | Support | Platform integer | Grade |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|
| MATCH_WIN(player_0b714ccd-16ea-5950-9a2f-5fed169b9763) | 54563 | 45437 | 0 | 0 | 100000 | 54.563% | 55% | -- | legacy fixed-sample | 100000 | 100.00% | SUPPORTED | -- | A |
| ACE_COMPARE(player_0b714ccd-16ea-5950-9a2f-5fed169b9763,player_72f48298-3469-51b1-999e-6ace673695ba) | 48264 | 48234 | 3502 | 0 | 96498 | 50.016% | 50% | -- | legacy fixed-sample | 100000 | 96.50% | SUPPORTED | -- | A |
| DF_COMPARE(player_0b714ccd-16ea-5950-9a2f-5fed169b9763,player_72f48298-3469-51b1-999e-6ace673695ba) | 54887 | 41611 | 3502 | 0 | 96498 | 56.879% | 57% | -- | legacy fixed-sample | 100000 | 96.50% | SUPPORTED | -- | A |
| TIEBREAK_COUNT(>=,2) | 24144 | 72853 | 3003 | 0 | 96997 | 24.891% | 25% | -- | legacy fixed-sample | 100000 | 97.00% | SUPPORTED | -- | A |
| DURATION_MIN(>,155.0) | 23386 | 72954 | 3210 | 450 | 96340 | 24.274% | -- | -- | legacy fixed-sample | 100000 | 96.34% | SETTLEMENT_BLOCKED | -- | B |
  - Policy issue for `baa9d34f5b624926d8e261082b73c7f968c016b77f17bd35cffc4cbb3cdff38d`: settlement semantics unresolved on one or more simulated paths
  - Support gate for `baa9d34f5b624926d8e261082b73c7f968c016b77f17bd35cffc4cbb3cdff38d`: DURATION_DISPLAY_POLICY_BOUNDARY_SENSITIVE — one or more paths differ under the preserved official-minute display-policy candidates
  - Display-policy sensitivity for `baa9d34f5b624926d8e261082b73c7f968c016b77f17bd35cffc4cbb3cdff38d`: 24.16% to 24.63%

## Audit and sensitivities

- Warning: DIRTY_CODE_TREE_RECORDED:ab10d927c4f9bb3ff3f99ccd67314371648e87edbc52379baa51861a08cea9ab
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
