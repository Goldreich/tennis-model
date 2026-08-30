# Milestone 8 implementation contract

The frozen specification remains authoritative. This note documents the immutable-lock,
historical-evaluation, and diagnostic implementation; it does not redefine model or settlement
semantics.

## Immutable locks and provenance

- `PredictionSnapshot` is a frozen, schema-versioned payload. Its stable base identity is derived
  from framework, tour, event, round, scheduled start, and the unordered player pair. Revisions are
  separate `L1`, `L2`, ... artifacts with the prior revision number/hash and a structured reason.
- `LockStore` publishes `lock.json`, a purely rendered `card.md`, and a file-hash manifest into a
  new revision directory. Existing revisions are never overwritten. Reload verifies file hashes,
  the payload content hash, schema compatibility, and card reproducibility.
- Locks embed the exact pinned source manifest, model snapshot and five component references,
  parameter record, data/model/config hashes, match/information cutoffs, scenario and information
  source hashes, first-server status, settlement policy, root seed, RNG, path policy/count, code
  commit, dirty status/diff hash, prop definitions/counts, warnings, and summary outputs.
- Dirty trees are rejected unless the caller explicitly permits a lock carrying the recorded dirty
  diff hash. The original Milestone 8 implementation used the frozen
  100,000/400,000 path policy; current live locks use the later versioned
  `adaptive_mc_cs_v1` confidence-sequence policy while old locks retain this
  historical policy identity.
  Development/test counts require an explicit non-production label and store their complete policy.
- Under the historical fixed policy, standard-path results were discarded and rerun from the same
  root seed when a frozen escalation trigger applied. Current adaptive live runs extend one
  deterministic seed prefix and inspect only the versioned checkpoints documented in
  `docs/Adaptive_MC_CS_v1.md`.
- `reproduce_prediction_lock` reloads the referenced fit artifacts, rebuilds the matchup, reruns the
  recorded path count from the root seed, and compares every prop count/probability and core summary.

## Locked Match Card and diffs

The machine-readable lock is authoritative. `render_locked_match_card` reads only stored fields and
shows identity, cutoff, model/code/source identity, seed/count/policy, primitive and derived serving
summaries, match distribution summaries, prop probabilities, settlement probabilities, integer
previews, warnings, and checks. It performs no estimation or simulation.

`compare_locks` separately reports source/config/code/cutoff/scenario/seed/prop-set changes and
numerical probability differences for overlapping props. It makes no causal attribution.

## Historical result boundary and ledger

- `HistoricalForecastTarget` contains pre-result inputs only. `forecast_historical_match` has no
  outcome parameter. `run_rolling_backtest` persists and verifies the lock before calling the
  `HistoricalOutcomeRevealer` protocol.
- `OfficialHistoricalOutcome` normalizes official winner, legal set scores, games, tiebreaks,
  optional aces/DFs/breaks/first-serve counts, retirement/walkover state, and an exact source pin.
  Missing historical statistics remain missing.
- Historical reconciliation adapts the normalized outcome to the existing `PropSpec` plus
  `SettlementPolicy` evaluator. A prop without the required official fields is marked
  `unavailable`; it is not guessed and does not force exclusion of other supported props.
- `CalibrationLedger` uses SQLite from the standard library. Schema metadata is explicit; database
  triggers reject updates and deletes; entries are unique and hash-chained. Official corrections
  append a new row pointing to the prior row. Unknown schema versions fail explicitly.
- Ledger rows retain event identity, lock warnings/policy flags, and a backtest run ID. Reports are
  scoped to the current run even when its ledger already contains prior immutable rows.
- Brier/log-loss calculations include Yes/No rows only. Fixed 0.1 reliability bins, distinct core
  prop-family, tour, confidence-band, retirement-rate, and Monte Carlo error summaries are
  implemented. Diagnostic confidence bands are predeclared by distance from 0.5: Low below 0.25,
  Medium from 0.25 through below 0.40, and High from 0.40; they never adjust a forecast.

## Rolling evaluation and diagnostics

- `SnapshotCatalog` selects the latest tour-matched snapshot whose data and fit timestamps are both
  no later than the target cutoff. Historical seeds are SHA-256-derived from framework version,
  match ID, lock revision, and backtest run ID. The market grid is fixed before outcome reveal.
- Date/tour/event filters, structured match exclusions with per-reason counts, prop-level
  unavailability, and pinned-source coverage reports are explicit. Tour-source retrieval and
  verification times must precede the lock cutoff. A real historical run is not silently
  substituted when data or source coverage is absent.
- The fixed pre-outcome grid covers match winner, exact score, straight sets, deciding set,
  first-set winner/games, tiebreak, total/player games, total/player breaks, ace and double-fault
  thresholds/totals/comparisons, and a representative joint-path compound.
- The I3 diagnostic constructs fixed-seed randomized beta-binomial quantile residuals conditional on
  observed denominators, then reports Pearson/rank correlations, four chronological folds,
  event-block intervals, and player-cluster sensitivity. Its exact frozen gate is implemented.
- The beta-copula candidate is labeled `CANDIDATE_V1_1_ONLY`, cannot be constructed unless the
  residual gate passes, and is absent from the live simulator. Candidate adoption evidence applies
  the joint-density/core-prop/no-family-worse-than-0.001 gate but never changes v1.0.
- Match-level shared-seed synthetic checks confirm that raising ace propensity improves match-win
  and hold rates and raising double-fault propensity worsens them. Existing point, game, scoring,
  generator, and settlement checks remain in the full suite.

## Commands

The installed `tennis` command (or `python -m tennis_model.cli`) exposes:

```text
tennis lock-match ...
tennis verify-lock ... [--reproduce]
tennis render-lock ...
tennis backtest ... [--tour ATP|WTA] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
tennis summarize-ledger ...
```

Backtests use separate target and outcome locations. Outcome files are named only by match ID and
are opened lazily after lock verification. SportsPredict reads/submission and notebooks are absent.

No dependency was added: NumPy, SciPy, and Pydantic already support the numerical/schema work, and
the ledger uses Python's standard-library SQLite implementation.

## Demonstration and current blockers

The automated ATP synthetic pipeline demonstration covers 11--12 January 2026 at the Synthetic
Audit Open. It produced 2 locks, no exclusions, and 62 settled prop rows (zero void, unavailable,
or unresolved) across match-winner, exact-score, set, tiebreak, games, breaks, ace, double-fault,
and compound families. Raw mean Brier was 0.13026 and mean log loss 0.43141. The family mean Brier
values were: aces 0.11390 (N=12), breaks 0.15213 (N=6), compounds 0.05252 (N=2), double faults
0.10818 (N=8), exact score 0.10200 (N=8), games 0.17325 (N=12), match winner 0.12587 (N=4), sets
0.16710 (N=6), and tiebreak 0.10634 (N=4). Runtime was 2.40 seconds after fitting. This is a
deterministic pipeline fixture using 24/48 explicitly labeled test paths, not a historical
model-quality result; median MC error was 0.0680 and its deliberately large simulation error must
not be interpreted statistically.

A genuine historical mini-backtest cannot yet be run because the repository deliberately has no
`config/sources.yaml`, pinned ATP/WTA source snapshot, or historical fitted snapshot. More
importantly, production locks remain correctly blocked by two pre-existing probability-affecting
gaps:

1. specification B6's fitted retirement hazard/generator is not implemented; and
2. specification C6's post-90-day uncertainty inflation and surface mean-reversion rule is still an
   explicitly unresolved Milestone 3 item.

The frozen specification does not provide enough numerical methodology to invent either behavior
incidentally in Milestone 8. Development/test locks record both warnings and leave retirement
probability unavailable rather than treating it as zero.

Consequently the current status is:

```text
CORE TENNIS MODEL v1.0 IMPLEMENTATION NOT READY
```
