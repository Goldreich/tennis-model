# US Open live simulation workflow

All new production locks use Tennis Model v1.1: frozen v1.0 serve components,
surface Elo with the fixed 75% strength integration, and the retained
minute-based game-day fitness assessment. Frozen v1.0 remains the archive and
rollback model only.

The live workflow is append-only. Each refresh creates a new source capture,
model snapshot, operational snapshot bundle, and lock directory. Never reuse a
post-start capture for a match lock.

## 1. Capture official tournament data

```powershell
.venv\Scripts\python.exe scripts/build_current_usopen_snapshot.py `
  --repo . `
  --acquire `
  --acquire-only
```

Retain the printed content-addressed capture path. Network-restricted agents may
need approval for this command.

## 2. Refresh and prepare model snapshots

```powershell
.venv\Scripts\python.exe scripts/refresh_current_from_verified_bundles.py `
  --repo . `
  --capture <capture-path> `
  --historical-root .runtime-worktree-81aa8db/data/processed/current-usopen-2026-exact-date `
  --output-root artifacts/current-usopen-2026 `
  --deterministic-test-result-sha256 <test-receipt-sha256>
```

Use the printed run ID to prepare production B6/C6 metadata without creating a
hardcoded demonstration lock:

```powershell
.venv\Scripts\python.exe scripts/create_current_usopen_lock.py `
  --repo . `
  --run-id <run-id> `
  --operational-name snapshot-<run-prefix>-v1 `
  --prepare-only
```

## 3. Simulate configured matches

`simulate_usopen_match_batch.py` accepts one or more `--match-id` arguments.
Use `--policy smoke --smoke-paths 5000` for a development smoke lock. Use
`--policy adaptive` for the frozen production Bernoulli confidence-sequence
policy. Production mode is rejected unless the official capture is pre-start
and all immutable provenance requirements pass.

```powershell
.venv\Scripts\python.exe scripts/simulate_usopen_match_batch.py `
  --fixture-file <fixture.json> `
  --match-id <official-match-id> `
  --output artifacts/live-usopen-2026/<batch-name> `
  --base-run-id <run-id> `
  --operational-name snapshot-<run-prefix>-v1 `
  --policy adaptive `
  --round R128 `
  --schedule-date 2026-08-31 `
  --schedule-source-id official-usopen-2026-schedule-day-9
```

Omit `--source-capture` to acquire and retain the schedule plus each configured
player's C6 evidence. A fixture file contains:

```json
{
  "schedule_url": "https://www.usopen.org/en_US/scores/feeds/2026/schedule/schedule9.json",
  "players": {
    "player-key": {
      "name": "Player Name",
      "id": "canonical-player-id",
      "tour": "ATP",
      "latest_date": "2026-08-20",
      "latest_match_id": "official-latest-match-id",
      "competition": "MAIN_DRAW",
      "source_url": "https://official.example/player-evidence"
    }
  },
  "matches": [
    {
      "official_id": "1000",
      "a": "player-key",
      "b": "opponent-key",
      "tour": "ATP"
    }
  ]
}
```

The player IDs and latest-match facts must be cutoff-safe and supported by the
captured official source. Market odds, rankings, and crowd forecasts remain
prohibited model inputs.
