# Tennis Model v1.0

This repository implements the frozen statistical design in
[`docs/Tennis_Model_v1.0_Specification.md`](docs/Tennis_Model_v1.0_Specification.md).
The current repository state contains the implemented engineering layers for
**Milestones 1 through 8 plus the frozen B5 duration layer**: reproducible
historical-data ingestion and provenance, a pure deterministic tennis scoring
state machine with typed settlement foundations, the five frozen ATP/WTA
serve-component estimators, the causal service-point generator, and cutoff-safe
two-direction match-parameter construction with two-stage uncertainty draws,
joint match simulation, the core pathwise prop/settlement evaluator, immutable
forecast locks, leakage-safe historical evaluation, and separate ATP/WTA
exposure-conditioned match-duration models attached to those same paths.

No market-text parser, auxiliary winner/error model, or tournament simulator is
included. The B6 retirement, C6 inactivity, and B5 duration contracts are
implemented. Current/live locks and retrospective historical validation use
separate eligibility policies. Current/live fitting may use only verified
exact-dated rows and must record every exclusion; retrospective rolling-origin
validation remains blocked by incomplete exact-date coverage and the absence of
a sufficiently complete pinned pre-result target schedule.

## Environment

The project targets Python 3.12 and uses `uv` for reproducible local environments.

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src/tennis_model
```

Milestone 1 uses:

- Pydantic v2 for frozen provenance schemas;
- pandas nullable dtypes for auditable tabular transformations;
- PyArrow for processed Parquet artifacts;
- PyYAML for source-manifest loading;
- pytest and Hypothesis for deterministic and property tests.

Milestone 3 adds NumPy and SciPy for explicit weighted beta-binomial MAP fitting
and deterministic finite-difference Laplace curvature. The exact resolved package
versions are recorded in `uv.lock`.

Milestone 5 and the B5 duration implementation reuse Pydantic, NumPy, SciPy, and
PyYAML and add no runtime dependency. Milestones 6 through 8 add no runtime
dependency.

Milestone 2's scoring and settlement foundations use only the Python standard
library. The scoring engine consumes an explicit stream of point winners and has
no access to model probabilities, historical data, live sources, or APIs.

DuckDB and any additional simulation-acceleration dependencies remain deferred
until their first milestone-authorized use.

## Milestone 1 workflow

The data API deliberately separates source acquisition from ingestion. Network
retrieval is out of scope: an operator first obtains a local source file and
audits its attribution, license, and coverage, then constructs a `PinnedSource`
containing an exact locator, any available archive/object identifiers, and the
SHA-256.

```python
from datetime import UTC, datetime
from tennis_model.data.cutoff import InformationCutoff
from tennis_model.data.ingest_sackmann import ingest_sackmann_snapshot
from tennis_model.data.snapshot import materialize_snapshot

snapshot = materialize_snapshot(source_pin, downloaded_file, "data/raw")
result = ingest_sackmann_snapshot(
    snapshot,
    cutoff=InformationCutoff(datetime(2026, 8, 28, tzinfo=UTC)),
)

service_rows = result.service_rows
component_counts = result.counts
anomalies = result.anomalies
```

Raw bytes are stored under a content-addressed directory and verified on every
read. Existing content is never overwritten or repaired. Missing count operands
stay missing; zero denominators are explicit non-likelihood rows; invalid counts
are preserved and quarantined without suppressing unaffected components.

Every public historical ingestion call requires an aware UTC cutoff. Inclusion
uses a strict `available_at_utc < cutoff` rule. Because a standard Sackmann
`tourney_date` is commonly a tournament-start date rather than an exact match
completion timestamp, each source pin must declare its date semantics and a
conservative availability lag. Such rows are unsuitable for within-event rolling
updates unless they are enriched from a separately pinned timing source.

## Data, attribution, and publication boundary

External dataset payloads and generated model/lock artifacts are deliberately
not versioned. They remain under the ignored `data/` and `artifacts/` trees.
The repository instead retains manifests, immutable locators, content hashes,
provenance, configuration, and reconstruction scripts:

- `config/sources.yaml` pins audited 2017–2025 ATP and WTA Sackmann-style yearly
  objects by commit, Git blob, SHA-256, coverage, retrieval time, license, and
  date semantics. The data are attributed to Jeff Sackmann / Tennis Abstract,
  through the retained
  [ATP](https://github.com/Kadantte/tennis_atp) and
  [WTA](https://github.com/VictorSquidWei/tennis_wta) repositories, under the
  stated
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  terms.
- `config/historical_validation_retrospective_finalized_v1.yaml` pins the
  Tennis-Data workbook hashes used only for exact-date augmentation. The
  workbooks are not redistributed because the assessed site did not state a
  dataset reuse license. No bookmaker or market fields enter the model.
- Current-tournament acquisition uses official US Open public scoring feeds and
  stores captures locally with truthful retrieval provenance; those captures
  are not redistributed here and remain subject to the source site's terms.

The Sackmann-style intake supplies the five core count components, but its
native dates are tournament-start dates. The separately pinned exact-date
crosswalk covers 85.763% of the retained 2017–2025 rows and leaves 6,813 rows
unresolved, so it must not be represented as a complete historical B6/C6 or
rolling-validation corpus. See
[`docs/Production_Data_Validation.md`](docs/Production_Data_Validation.md),
[`docs/Historical_Source_Date_Augmentation_Assessment.md`](docs/Historical_Source_Date_Augmentation_Assessment.md),
and
[`docs/Retrospective_Finalized_Historical_Validation.md`](docs/Retrospective_Finalized_Historical_Validation.md).

The immutable source locators and hashes are sufficient to verify independently
obtained bytes. `scripts/assess_retrospective_crosswalk.py` reconstructs the
retrospective crosswalk from a prepared pinned-source assessment root;
`scripts/build_current_usopen_snapshot.py --acquire` captures the current
official feed and builds a content-addressed live snapshot; and
`scripts/build_duration_models.py` performs its documented offline upgrade from
retained inputs. None of these scripts bypasses source licensing or cutoff
requirements.

See [`docs/Milestone_1_Implementation.md`](docs/Milestone_1_Implementation.md)
for the ingestion contract and ambiguity policy.

## Milestone 2 scoring

```python
from tennis_model.simulation.scoring import award_point, new_match

state = new_match(
    "player-a",
    "player-b",
    best_of=3,
    first_server_index=0,
)
transition = award_point(state, winner_index=0)
state = transition.after
```

States and transitions are immutable. The engine implements advantage games,
continuous service order, standard and deciding-set tiebreaks, break-point and
break accounting, exact scores, and strict legal termination. Settlement truth,
void, and blocked states are separate from score generation.

See [`docs/Milestone_2_Implementation.md`](docs/Milestone_2_Implementation.md)
for the Milestone 2 contract and the boundary with later stochastic work.

## Milestone 8 audit and historical evaluation

The repository now has schema-versioned immutable prediction locks, explicit lock revisions,
content/file hashing, complete data/model/code/config/seed provenance, a pure Locked Match Card
renderer, anytime-valid adaptive stopping, full lock replay, SQLite append-only calibration
ledger, lock-before-result historical orchestration, Brier/reliability summaries, coverage and
exclusion reports, and the frozen diagnostic-only cross-component dependence gate.

Useful entrypoints are `tennis lock-match`, `tennis verify-lock`, `tennis render-lock`, `tennis
backtest`, and `tennis summarize-ledger` after installing the package. None reads or submits
SportsPredict markets. Backtests accept optional `--tour`, `--start`, `--end`, and repeatable
`--event` filters.

Production locks enforce the implemented B6/C6 schema, the versioned
`adaptive_mc_cs_v1` 5k/10k/20k/40k/70k anytime-valid path policy, canonical
identity, retained-artifact, and clean-tree gates. Reduced-count development/test locks are
explicitly labeled and cannot masquerade as production artifacts. See
[`docs/Milestone_8_Implementation.md`](docs/Milestone_8_Implementation.md) for the architecture,
validation boundary, and exact readiness status, and
[`docs/Adaptive_MC_CS_v1.md`](docs/Adaptive_MC_CS_v1.md) for the confidence-sequence formula and
stopping contract.

## Milestone 3 estimation

The estimation package consumes the exact long-form Milestone 1 `F`, `A`, `Q1`,
`D`, and `Q2` counts. It fits ATP and WTA separately with the frozen 1,095-day
window and 365-day half-life, joint global/surface shrinkage, the specified
server/returner structures, MAP optimization, and a recorded Laplace posterior
approximation. A verified processed-bundle entry point binds fit provenance to
the source snapshot and component-count Parquet checksum.

Future-match output contains conditional MAP means, linear-predictor uncertainty,
beta shapes at the MAP, context, and coherent fit provenance. It contains no RNG,
posterior draw, match-performance beta draw, point generation, or match path.

See [`docs/Milestone_3_Implementation.md`](docs/Milestone_3_Implementation.md)
for the statistical implementation, explicit numerical choices, and unresolved
inactivity rule.

## Milestone 4 point generation

```python
import numpy as np

from tennis_model.simulation.point import ServePerformanceDraw, generate_service_point

performance = ServePerformanceDraw(
    first_serve_in=0.62,
    ace_given_first_in=0.11,
    returnable_first_win=0.67,
    double_fault_given_second_opp=0.09,
    playable_second_win=0.56,
)
point = generate_service_point(
    performance,
    np.random.default_rng(20260829),
    server_id="player-a",
    receiver_id="player-b",
)
```

The generator consumes only the five realized primitive probabilities and an
explicit RNG. It draws first-serve status before ace/double-fault status and draws
`Q1` or `Q2` only on the eligible causal branch. Immutable results expose a clean
winner boundary for scoring and a rally-eligibility boundary for later auxiliary
models. A deterministic five-uniform interface supports coupled monotonicity
tests, and pure aggregation reconstructs the exact serve-count identities.

See [`docs/Milestone_4_Implementation.md`](docs/Milestone_4_Implementation.md)
for RNG semantics, enforced invariants, and the boundary with Milestone 5.

## Milestone 5 match parameters

Milestone 5 loads an explicit immutable five-component snapshot, enforces that
neither its data cutoff nor fit timestamp follows the requested information
cutoff, and constructs separate `A -> B` and `B -> A` serving distributions. Each
match path draws one full Laplace parameter vector per component and then one
conditionally independent beta performance value per component and serving
direction. The resulting two immutable `ServePerformanceDraw` objects feed
Milestone 4 directly and must be reused throughout that path.

Sampling uses a caller-owned `SeedSequence`, without advancing it, and records
stable `PCG64` parameter, directional-performance, and reserved point-path child
streams. Stable matchup records retain artifact IDs, hashes, both cutoffs,
context/scenario, fit and match-parameter implementation versions, and the
frozen-independent dependence policy without copying posterior matrices or RNG
objects.

See [`docs/Milestone_5_Implementation.md`](docs/Milestone_5_Implementation.md)
for snapshot safety, uncertainty separation, numerical checks, and the synthetic
historical-cutoff demonstration.

## Milestone 6 joint match simulation

The joint simulator consumes a verified Milestone 5 distribution, samples one
fixed primitive performance vector per player and path, and reuses it for every
corresponding service point. It delegates tennis rules to the deterministic
scoring engine and records exact scores, games, tiebreaks, breaks, break points,
holds, and serve sufficient statistics. Explicit hierarchical seeds and optional
point traces make every batch replayable.

See [`docs/Milestone_6_Implementation.md`](docs/Milestone_6_Implementation.md).

## Milestone 7 prop settlement

Core props and compounds are evaluated on the same simulated match paths.
Settlement is a separate pure layer with explicit policy versioning and distinct
Yes, No, Void, and unresolved states. Retirement/walkover tests enforce completed
scope and irreversible-threshold rules; official rounding-sensitive percentage
cases remain blocked.

See [`docs/Milestone_7_Implementation.md`](docs/Milestone_7_Implementation.md).

## B5 match duration

The duration layer fits separate ATP and WTA robust regressions to legal,
normally completed, exact-dated matches with official duration and total-point
data. It retains the frozen point/game/set/tiebreak exposure terms, strongly
shrunk player pace effects, a shrunk current-event effect, heteroskedastic
Student-t residuals, and an audited parameter-covariance approximation.

For simulation, duration is drawn only after a joint tennis path has produced
its realized exposure. Dedicated seed streams make the added draw replayable and
leave every pre-duration score and statistic unchanged. `DURATION_MIN` uses the
same paths and reports official-minute conversion sensitivity while the exact
display convention remains unresolved. Generated fits, upgraded snapshots, data
audit, diagnostics, and 1k/5k/100k incremental timings are produced offline by:

```powershell
uv run python scripts/build_duration_models.py --fitted-at-utc <UTC_TIMESTAMP>
```

See
[`docs/Match_Duration_Model_Implementation.md`](docs/Match_Duration_Model_Implementation.md)
for the retained source coverage, fitted artifacts, diagnostics, and remaining
limitations.
