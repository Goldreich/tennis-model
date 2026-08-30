# Milestone 1 implementation contract

## Concrete implementation plan

The frozen design is implemented in the specification's order:

1. **Milestone 1 — data foundation.** Freeze source provenance and
   exact bytes; resolve source-backed identities; normalize each winner/loser row
   to two player-service rows; construct F/A/Q1/D/Q2 integer counts; quarantine
   anomalies; enforce point-in-time access; persist processed tables as Parquet.
2. **Milestone 2 — deterministic scoring.** Add a dependency-light pure state
   machine for games, sets, best-of-three/five, both tiebreak formats, service
   order, breaks, and retirement/settlement separation. Prove tennis identities
   with example and property tests before simulation is attached.
3. **Milestone 3 — component estimation.** Use NumPy/SciPy to implement the five
   separate ATP/WTA time-weighted beta-binomial GLMMs with explicit MAP objectives,
   Laplace artifacts, and typed future-match component summaries.
4. **Milestone 4 — revised point generator.** Feed only primitive F/A/Q1/D/Q2
   probabilities to the causal serve generator and prove its accounting and
   monotonicity identities.
5. **Later matchup uncertainty.** Draw coherent posterior parameters and the five
   conditionally independent match-performance probabilities once per simulated
   match. The shared-factor interface remains disabled in v1.0.
6. **Milestone 6 — joint simulation and core props.** Drive every score and core
   prop from common match paths. All random APIs accept explicit NumPy generators
   or seed sequences.
7. **Milestone 7 — retirement and settlement.** Add the separate typed retirement
   process and pathwise settlement/void rules.
8. **Milestone 8 — locks and historical validation.** Add immutable lock revisions,
   a DuckDB append-only ledger, Parquet/JSON backtest artifacts, and strict
   rolling-origin evaluation before external dry runs.
9. **Auxiliary and external integrations.** Winner/error, duration, tournament,
   and read-only SportsPredict work begins only after the core gates pass. Live
   submission remains disabled absent separate authorization.

## Milestone 1 behavior

- `PinnedSource` records a stable identity namespace separately from a particular
  file ID, upstream attribution, an exact locator, optional archive/object
  identifiers where available, raw SHA-256, schema, license, UTC retrieval time,
  tour-specific verified coverage, date semantics, and an explicit availability
  lag.
- Raw snapshot publication is local-only, staged, atomic, content-addressed, and
  idempotent for identical bytes. Any checksum/provenance mismatch fails loudly.
- Player IDs use only source namespace, tour, and stable external ID. Names remain
  aliases. Match IDs use source namespace, tour, tournament ID, and match number.
- Exact duplicate records are represented once with audit lineage. Conflicting
  records sharing a match identity are all quarantined; there is no first/last
  correction rule.
- Clearly illegal or unknown score forms quarantine the whole match. A missing
  score is retained as an explicit warning because it does not alter otherwise
  observable primitive serve counts. Walkovers emit no service rows.
- Count anomalies are component-local. Invalid signed derived values remain in
  the quarantine table and are never clipped.
- Historical ingestion requires a timezone-aware cutoff and applies strict `<`.
  Unknown availability fails closed; future raw statistics are filtered before
  normalization.

## True implementation ambiguities

These do not reopen the frozen serve architecture:

1. No exact production ATP/WTA objects, checksums, or license approval were
   supplied. The pipeline is complete, but production data are intentionally not
   fabricated or downloaded.
2. Sackmann `tourney_date` commonly identifies tournament start, not precise
   completion/publication. The pin therefore declares a conservative lag; exact
   same-event updates need a separately sourced match timestamp.
3. A current historical snapshot cannot prove which later stat corrections were
   known at an old cutoff. Fully vintage-safe backtests need archived source
   objects or bitemporal correction metadata.
4. The frozen specification provides no fuzzy cross-source identity precedence.
   Missing or ambiguous stable IDs are quarantined rather than resolved by name.
5. It mandates duplicate/correction detection but no precedence rule. Exact
   copies are deduplicated; conflicts are excluded pending a versioned ruling.
6. Historical score dialects vary by date and event. Milestone 1 recognizes only
   conservative, well-defined forms and does not attempt Milestone 2 scoring.

## Milestone status

Milestone 1 remains complete and regression-covered. Milestones 2 and 3 were
subsequently implemented without changing its primitive count construction,
quarantine policy, or strict cutoff behavior. A production source audit remains
separate from fixture validation.
