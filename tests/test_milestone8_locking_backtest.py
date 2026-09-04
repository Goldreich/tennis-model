from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from serve_model_test_helpers import (
    TEST_CUTOFF,
    make_model_config,
    make_provenance,
    synthetic_all_component_counts,
)

from tennis_model.calibration import (
    CalibrationLedger,
    HistoricalForecastTarget,
    HistoricalOutcomeError,
    HistoricalPlayerStats,
    HistoricalSetResult,
    LedgerError,
    OfficialHistoricalOutcome,
    SnapshotCatalog,
    brier_score,
    fixed_core_market_grid,
    ledger_rows_from_settlement,
    load_backtest_run_manifest,
    run_rolling_backtest,
    settle_historical_lock,
)
from tennis_model.estimation.artifacts import write_fit_artifact
from tennis_model.estimation.serve_components import ServeComponent, fit_all_serve_components
from tennis_model.estimation.snapshot import ModelSnapshot, create_model_snapshot
from tennis_model.locking import (
    AdaptiveMCPolicy,
    CodeProvenance,
    InformationBundle,
    InformationItem,
    LockCreationError,
    LockRevisionReason,
    LockStore,
    MCStoppingStatus,
    PathCountPolicy,
    compare_locks,
    create_prediction_lock,
    render_locked_match_card,
    reproduce_prediction_lock,
)
from tennis_model.locking.provenance import CodeProvenanceError
from tennis_model.locking.store import (
    LockAlreadyExistsError,
    LockIntegrityError,
    LockPublicationInterrupted,
    LockReservationConflict,
)
from tennis_model.props import CANONICAL_SETTLEMENT_POLICY, ComparisonOperator
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RowDateSemantics,
    SourceManifest,
    Tour,
    TourCoverage,
)
from tennis_model.simulation import (
    ANY_TIEBREAK,
    FIRST_SERVE_WIN_PCT,
    MATCH_WIN,
    PLAYER_ACES,
    PLAYER_DF,
    TOTAL_BREAKS,
    TOTAL_GAMES,
)
from tennis_model.simulation.parameters import MatchCondition, MatchContext


@pytest.fixture(scope="module")
def m8_snapshot(tmp_path_factory: pytest.TempPathFactory) -> ModelSnapshot:
    counts = synthetic_all_component_counts(repetitions=3, seed=8800)
    fits = fit_all_serve_components(
        counts,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(max_iterations=1200),
        provenance=make_provenance("milestone-8"),
    )
    root = tmp_path_factory.mktemp("m8-fit-artifacts")
    artifacts = {
        component: write_fit_artifact(fits[component], root) for component in ServeComponent
    }
    return create_model_snapshot(artifacts)


def _source_manifest(snapshot: ModelSnapshot) -> SourceManifest:
    coverage = CoverageRange(
        first_match_date=date(2023, 1, 1),
        last_match_date=date(2025, 12, 31),
        verified_at_utc=TEST_CUTOFF,
    )
    return SourceManifest(
        manifest_version="m8-test/v1",
        sources=(
            PinnedSource(
                source_id="m8-history",
                identity_namespace="m8-history",
                tour=Tour.ATP,
                upstream_attribution="synthetic Milestone 8 fixture",
                locator="fixture://m8-history",
                object_identifier="fixture-object-v1",
                sha256="1" * 64,
                schema_version="fixture/v1",
                stated_license="test-only",
                retrieved_at_utc=TEST_CUTOFF,
                verified_coverage=coverage,
                row_date_semantics=RowDateSemantics.MATCH_DATE,
                availability_lag_days=1,
            ),
        ),
        coverage_by_tour=TourCoverage(atp=coverage, wta=None),
    )


def _context(*, day: int = 10, cutoff_day: int = 2) -> MatchContext:
    return MatchContext(
        player_a_id="p0",
        player_b_id="p1",
        tour=Tour.ATP,
        event="Synthetic Audit Open",
        round="R32",
        scheduled_start_utc=TEST_CUTOFF + timedelta(days=day),
        best_of=3,
        indoor=False,
        conditions=(MatchCondition(name="roof", value="open"),),
        information_cutoff_utc=TEST_CUTOFF + timedelta(days=cutoff_day),
        information_scenario_id="historical-central",
    )


def _information(context: MatchContext, *, label: str = "initial") -> InformationBundle:
    return InformationBundle(
        bundle_id=f"info-{label}",
        scenario_id=context.information_scenario_id,
        information_cutoff_utc=context.information_cutoff_utc,
        items=(
            InformationItem(
                category="schedule",
                summary=f"Pre-match schedule state {label}",
                source_id="official-schedule",
                source_sha256="a" * 64,
                observed_at_utc=context.information_cutoff_utc - timedelta(hours=2),
                available_at_utc=context.information_cutoff_utc - timedelta(hours=1),
            ),
        ),
        missing_current_conditions=("temperature",),
    )


_TEST_POLICY = PathCountPolicy(
    standard_paths=24,
    escalated_paths=48,
    minimum_settled_paths=0,
    extreme_probability=0.001,
    integer_boundary_window=0.0,
    integer_boundary_standard_errors=0.0,
)
_CLEAN_CODE = CodeProvenance(commit="test-commit", dirty=False, diff_sha256=None)


def _create_lock(
    snapshot: ModelSnapshot,
    store: LockStore | None,
    *,
    context: MatchContext | None = None,
    props: tuple[object, ...] | None = None,
    seed: int = 81,
    parent: object | None = None,
    reason: LockRevisionReason | None = None,
) -> object:
    selected_context = _context() if context is None else context
    selected_props = (
        (MATCH_WIN("p0"), ANY_TIEBREAK(), TOTAL_BREAKS(ComparisonOperator.MORE_THAN, 3))
        if props is None
        else props
    )
    return create_prediction_lock(
        snapshot,
        selected_context,
        _information(selected_context, label=f"r{1 if parent is None else 2}"),
        selected_props,  # type: ignore[arg-type]
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=_source_manifest(snapshot),
        code=_CLEAN_CODE,
        seed=seed,
        store=store,
        n_paths=_TEST_POLICY.standard_paths,
        execution_mode="test",
        path_count_policy=_TEST_POLICY,
        created_at_utc=selected_context.information_cutoff_utc,
        parent=parent,  # type: ignore[arg-type]
        revision_reason=reason,
    )


def _outcome(match_id: str, *, start_day: int = 10) -> OfficialHistoricalOutcome:
    played = TEST_CUTOFF + timedelta(days=start_day, hours=3)
    return OfficialHistoricalOutcome(
        match_id=match_id,
        player_a_id="p0",
        player_b_id="p1",
        best_of=3,
        started=True,
        completed=True,
        winner_id="p0",
        sets=(
            HistoricalSetResult(set_number=1, games=(6, 4)),
            HistoricalSetResult(set_number=2, games=(6, 4)),
        ),
        sets_started=2,
        player_stats=(
            HistoricalPlayerStats(
                player_id="p0",
                games_won=12,
                aces=8,
                double_faults=2,
                breaks_achieved=2,
                first_serves_in=42,
                first_serve_points_won=31,
            ),
            HistoricalPlayerStats(
                player_id="p1",
                games_won=8,
                aces=3,
                double_faults=4,
                breaks_achieved=0,
                first_serves_in=38,
                first_serve_points_won=25,
            ),
        ),
        break_events=None,
        official_source_id="official-result",
        official_source_sha256="b" * 64,
        official_source_locator="fixture://official-result",
        available_at_utc=played,
        retrieved_at_utc=played + timedelta(minutes=5),
    )


def test_lock_round_trip_immutability_revision_and_full_reproduction(
    m8_snapshot: ModelSnapshot,
    tmp_path: Path,
) -> None:
    store = LockStore(tmp_path / "locks")
    lock = _create_lock(m8_snapshot, store)
    loaded = store.load(lock.base_lock_id, 1).lock  # type: ignore[attr-defined]
    assert loaded == lock
    assert store.verify(lock.base_lock_id, 1) == lock.content_sha256  # type: ignore[attr-defined]
    assert reproduce_prediction_lock(loaded).reproduced
    prior_json = (store.revision_directory(lock.base_lock_id, 1) / "lock.json").read_bytes()  # type: ignore[attr-defined]
    card = (store.revision_directory(lock.base_lock_id, 1) / "card.md").read_text(  # type: ignore[attr-defined]
        encoding="utf-8"
    )
    assert "Exact score probabilities:" in card
    assert "Source manifest:" in card
    assert "99% anytime-valid CS | MC status | Final paths" in card

    with pytest.raises(LockAlreadyExistsError):
        store.write(lock)  # type: ignore[arg-type]

    revised_context = _context(cutoff_day=3)
    revision = _create_lock(
        m8_snapshot,
        store,
        context=revised_context,
        props=(MATCH_WIN("p0"), TOTAL_GAMES(ComparisonOperator.MORE_THAN, 22.5)),
        seed=82,
        parent=lock,
        reason=LockRevisionReason(
            category="conditions",
            summary="Official roof status changed",
            evidence_source_ids=("official-schedule",),
        ),
    )
    assert revision.revision == 2  # type: ignore[attr-defined]
    assert revision.parent_content_sha256 == lock.content_sha256  # type: ignore[attr-defined]
    assert revision.content_sha256 != lock.content_sha256  # type: ignore[attr-defined]
    assert (store.revision_directory(lock.base_lock_id, 1) / "lock.json").read_bytes() == prior_json  # type: ignore[attr-defined]
    diff = compare_locks(lock, revision)  # type: ignore[arg-type]
    assert {item.field for item in diff.metadata_changes} >= {
        "information_cutoff",
        "seed_path_count",
        "prop_set",
    }


def test_lock_hash_covers_probability_and_provenance_content(
    m8_snapshot: ModelSnapshot,
) -> None:
    lock = _create_lock(m8_snapshot, None)
    variants = (
        lock.model_copy(
            update={
                "context": lock.context.model_copy(
                    update={
                        "information_cutoff_utc": lock.context.information_cutoff_utc
                        + timedelta(seconds=1)
                    }
                )
            }
        ),
        lock.model_copy(
            update={
                "match_parameters": lock.match_parameters.model_copy(
                    update={"snapshot_id": "f" * 64}
                )
            }
        ),
        lock.model_copy(
            update={"simulation": lock.simulation.model_copy(update={"seed_id": "changed-seed"})}
        ),
        lock.model_copy(
            update={
                "simulation": lock.simulation.model_copy(
                    update={"actual_paths": lock.simulation.actual_paths + 1}
                )
            }
        ),
        lock.model_copy(
            update={
                "prop_estimates": (
                    lock.prop_estimates[0].model_copy(update={"prop_id": "e" * 64}),
                    *lock.prop_estimates[1:],
                )
            }
        ),
        lock.model_copy(
            update={
                "settlement_policy": lock.settlement_policy.model_copy(
                    update={"version": "changed-policy"}
                )
            }
        ),
        lock.model_copy(
            update={
                "match_summary": lock.match_summary.model_copy(
                    update={"expected_total_games": lock.match_summary.expected_total_games + 1.0}
                )
            }
        ),
    )
    assert all(item.content_sha256 != lock.content_sha256 for item in variants)


def test_lock_integrity_dirty_policy_cutoff_and_production_block(
    m8_snapshot: ModelSnapshot,
    tmp_path: Path,
) -> None:
    store = LockStore(tmp_path / "locks")
    lock = _create_lock(m8_snapshot, store)
    card = store.revision_directory(lock.base_lock_id, 1) / "card.md"  # type: ignore[attr-defined]
    card.write_text("tampered", encoding="utf-8")
    with pytest.raises(LockIntegrityError):
        store.load(lock.base_lock_id, 1)  # type: ignore[attr-defined]

    schema_store = LockStore(tmp_path / "schema-locks")
    schema_lock = _create_lock(m8_snapshot, schema_store, seed=83)
    manifest_path = (
        schema_store.revision_directory(schema_lock.base_lock_id, 1) / "manifest.json"  # type: ignore[attr-defined]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "prediction-lock-files/v99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LockIntegrityError, match="unsupported lock-file manifest schema"):
        schema_store.load(schema_lock.base_lock_id, 1)  # type: ignore[attr-defined]

    context = _context()
    with pytest.raises(CodeProvenanceError, match="dirty"):
        create_prediction_lock(
            m8_snapshot,
            context,
            _information(context),
            (MATCH_WIN("p0"),),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=_source_manifest(m8_snapshot),
            code=CodeProvenance(commit="test", dirty=True, diff_sha256="c" * 64),
            seed=1,
            n_paths=_TEST_POLICY.standard_paths,
            execution_mode="test",
            path_count_policy=_TEST_POLICY,
        )
    with pytest.raises(LockCreationError, match="retirement generator"):
        create_prediction_lock(
            m8_snapshot,
            context,
            _information(context),
            (MATCH_WIN("p0"),),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=_source_manifest(m8_snapshot),
            code=_CLEAN_CODE,
            seed=1,
        )

    future_source = (
        _source_manifest(m8_snapshot)
        .sources[0]
        .model_copy(
            update={
                "retrieved_at_utc": context.information_cutoff_utc + timedelta(seconds=2),
                "source_effective_at_utc": context.information_cutoff_utc - timedelta(seconds=1),
                "source_available_at_utc": context.information_cutoff_utc + timedelta(seconds=1),
            }
        )
    )
    future_manifest = _source_manifest(m8_snapshot).model_copy(update={"sources": (future_source,)})
    with pytest.raises(LockCreationError, match="probability-relevant information unavailable"):
        create_prediction_lock(
            m8_snapshot,
            context,
            _information(context),
            (MATCH_WIN("p0"),),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=future_manifest,
            code=_CLEAN_CODE,
            seed=1,
            n_paths=_TEST_POLICY.standard_paths,
            execution_mode="test",
            path_count_policy=_TEST_POLICY,
        )


def test_result_reveal_settlement_and_append_only_ledger(
    m8_snapshot: ModelSnapshot,
    tmp_path: Path,
) -> None:
    lock = _create_lock(m8_snapshot, None)
    outcome = _outcome("historical-001")
    leaked = outcome.model_copy(update={"available_at_utc": lock.context.information_cutoff_utc})
    with pytest.raises(HistoricalOutcomeError, match="available before the forecast cutoff"):
        settle_historical_lock(lock, leaked)  # type: ignore[arg-type]
    settlement = settle_historical_lock(lock, outcome)  # type: ignore[arg-type]
    assert {item.state for item in settlement.resolutions} <= {"yes", "no", "void"}
    rows = ledger_rows_from_settlement(lock, outcome, settlement)  # type: ignore[arg-type]
    ledger = CalibrationLedger(tmp_path / "ledger.sqlite3")
    for row in rows:
        ledger.append(row)
    assert ledger.read() == rows
    original_chain = ledger.verify_chain()
    assert original_chain.terminal_sha256 is not None
    assert all(row.brier_raw_model is not None for row in rows)
    with pytest.raises(LedgerError, match="already exists"):
        ledger.append(rows[0])
    with (
        sqlite3.connect(ledger.path) as connection,
        pytest.raises(
            sqlite3.IntegrityError,
            match="append-only",
        ),
    ):
        connection.execute(
            "UPDATE calibration_entries SET row_json = row_json WHERE row_id = ?",
            (rows[0].row_id,),
        )
    correction = ledger.append_correction(
        rows[0].row_id,
        rows[0],
        reason="official source correction fixture",
    )
    assert correction.correction_of_row_id == rows[0].row_id
    assert len(ledger.read()) == len(rows) + 1
    corrected_chain = ledger.verify_chain()
    assert corrected_chain.terminal_sha256 != original_chain.terminal_sha256
    assert corrected_chain.correction_rows == 1
    effective = ledger.effective_rows(correction_policy="latest-appended/v1")
    assert rows[0] not in effective
    assert correction in effective
    second_correction = ledger.append_correction(
        rows[0].row_id,
        rows[0],
        reason="later official correction supersedes the earlier correction",
    )
    latest = ledger.effective_rows(correction_policy="latest-appended/v1")
    assert correction not in latest
    assert second_correction in latest
    assert ledger.verify_chain().correction_rows == 2
    with pytest.raises(LedgerError, match="terminal ledger digest differs"):
        ledger.verify_chain(expected_terminal_sha256="0" * 64)
    assert brier_score(0.25, 1) == pytest.approx(0.5625)
    assert brier_score(0.25, 0) == pytest.approx(0.0625)

    retired_time = TEST_CUTOFF + timedelta(days=10, hours=1)
    retired = OfficialHistoricalOutcome(
        match_id="historical-retired",
        player_a_id="p0",
        player_b_id="p1",
        best_of=3,
        started=True,
        completed=False,
        winner_id="p0",
        retired_player_id="p1",
        sets=(),
        sets_started=1,
        player_stats=None,
        break_events=None,
        official_source_id="official-result",
        official_source_sha256="e" * 64,
        official_source_locator="fixture://official-retirement",
        available_at_utc=retired_time,
        retrieved_at_utc=retired_time + timedelta(minutes=5),
    )
    retired_settlement = settle_historical_lock(lock, retired)  # type: ignore[arg-type]
    retired_rows = ledger_rows_from_settlement(
        lock,
        retired,
        retired_settlement,  # type: ignore[arg-type]
    )
    assert {row.resolution_status for row in retired_rows} == {
        "yes",
        "void",
        "unavailable",
    }
    assert all(
        row.brier_raw_model is None
        for row in retired_rows
        if row.resolution_status in {"void", "unavailable"}
    )
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DROP TRIGGER calibration_entries_no_update")
        connection.execute(
            "UPDATE calibration_entries SET entry_sha256 = ? WHERE sequence = 1",
            ("0" * 64,),
        )
    with pytest.raises(LedgerError, match="modified at sequence 1"):
        ledger.verify_chain()


def test_unknown_ledger_schema_fails_closed(tmp_path: Path) -> None:
    ledger = CalibrationLedger(tmp_path / "unknown-schema.sqlite3")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "CREATE TABLE ledger_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO ledger_metadata(key, value) VALUES('schema_version', 'future/v99')"
        )
    with pytest.raises(LedgerError, match="unsupported calibration ledger schema"):
        ledger.read()


def test_missing_stats_and_unresolved_display_policy_are_prop_level_only(
    m8_snapshot: ModelSnapshot,
) -> None:
    lock = _create_lock(
        m8_snapshot,
        None,
        props=(
            PLAYER_ACES("p0", ComparisonOperator.MORE_THAN, 4),
            PLAYER_DF("p0", ComparisonOperator.MORE_THAN, 3),
            FIRST_SERVE_WIN_PCT("p0", ComparisonOperator.MORE_THAN, 65),
        ),
    )
    complete = _outcome("missing-stat-outcome")
    assert complete.player_stats is not None
    missing = complete.model_copy(
        update={
            "player_stats": (
                HistoricalPlayerStats(
                    player_id="p0",
                    games_won=12,
                    aces=None,
                    double_faults=None,
                    breaks_achieved=2,
                    first_serves_in=None,
                    first_serve_points_won=None,
                ),
                complete.player_stats[1],
            )
        }
    )
    settlement = settle_historical_lock(lock, missing)  # type: ignore[arg-type]
    by_kind = {item.prop.kind: item for item in settlement.resolutions}
    assert by_kind["PLAYER_ACES"].support_status.value == "DATA_UNAVAILABLE"
    assert by_kind["PLAYER_DF"].support_status.value == "DATA_UNAVAILABLE"
    assert by_kind["FIRST_SERVE_WIN_PCT"].support_status.value == "DATA_UNAVAILABLE"
    assert all(
        item.availability_phase.value == "PROP_UNAVAILABLE_POST_REVEAL"
        for item in settlement.resolutions
    )


class _Revealer:
    def __init__(self, outcomes: dict[str, OfficialHistoricalOutcome]) -> None:
        self.outcomes = outcomes
        self.revealed: list[tuple[str, str]] = []

    def reveal(self, match_id: str, *, locked_content_sha256: str) -> OfficialHistoricalOutcome:
        assert len(locked_content_sha256) == 64
        self.revealed.append((match_id, locked_content_sha256))
        return self.outcomes[match_id]


def test_rolling_origin_selects_only_safe_snapshot_and_reveals_after_lock(
    m8_snapshot: ModelSnapshot,
    tmp_path: Path,
) -> None:
    targets = []
    outcomes = {}
    for index, day in enumerate((10, 11), start=1):
        context = _context(day=day)
        match_id = f"historical-{index:03d}"
        targets.append(
            HistoricalForecastTarget(
                match_id=match_id,
                context=context,
                information=_information(context, label=match_id),
                props=fixed_core_market_grid(context),
            )
        )
        outcomes[match_id] = _outcome(match_id, start_day=day)
    revealer = _Revealer(outcomes)
    lock_store = LockStore(tmp_path / "locks")
    ledger = CalibrationLedger(tmp_path / "ledger.sqlite3")
    prior_lock = _create_lock(m8_snapshot, None, seed=999)
    prior_outcome = _outcome("historical-prior", start_day=9)
    prior_settlement = settle_historical_lock(prior_lock, prior_outcome)  # type: ignore[arg-type]
    prior_row = ledger_rows_from_settlement(
        prior_lock,
        prior_outcome,
        prior_settlement,  # type: ignore[arg-type]
        backtest_run_id="earlier-run",
    )[0]
    ledger.append(prior_row)
    report = run_rolling_backtest(
        tuple(targets),
        snapshots=SnapshotCatalog((m8_snapshot,)),
        outcomes=revealer,
        source_manifest=_source_manifest(m8_snapshot),
        code=_CLEAN_CODE,
        lock_store=lock_store,
        ledger=ledger,
        backtest_run_id="m8-reproducible-mini",
        path_count_policy=_TEST_POLICY,
        execution_mode="test",
    )
    all_rows = ledger.read()
    assert report.targets == 2
    assert report.locks_created == 2
    assert report.ledger_rows == len(all_rows) - 1
    assert (
        report.calibration.settled_rows + report.calibration.unavailable_rows == report.ledger_rows
    )
    assert {row.backtest_run_id for row in all_rows} == {
        "earlier-run",
        "m8-reproducible-mini",
    }
    assert len(revealer.revealed) == 2
    assert report.calibration.settled_rows > 0
    # The fixture includes first-serve counts for both players in both matches,
    # so every row in the fixed core grid is settleable.
    assert report.calibration.unavailable_rows == 0
    assert report.exclusion_counts == ()
    persisted_run = load_backtest_run_manifest(lock_store.root / "_backtest_runs" / report.run_id)
    assert persisted_run == report.run_manifest
    assert persisted_run.terminal_ledger_sha256 == ledger.verify_chain().terminal_sha256
    kinds = {item.kind for item in fixed_core_market_grid(targets[0].context)}
    assert {
        "MATCH_WIN",
        "EXACT_SCORE",
        "STRAIGHT_SETS",
        "DECIDING_SET",
        "ANY_TIEBREAK",
        "FIRST_SET_WIN",
        "FIRST_SET_GAMES",
        "TOTAL_GAMES",
        "PLAYER_GAMES",
        "BREAK_COUNT",
        "TOTAL_BREAKS",
        "PLAYER_ACES",
        "TOTAL_ACES",
        "ACE_COMPARE",
        "PLAYER_DF",
        "TOTAL_DF",
        "DF_COMPARE",
        "FIRST_SERVE_WIN_PCT",
        "AND",
    } <= kinds

    replay_revealer = _Revealer(outcomes)
    replay_ledger = CalibrationLedger(tmp_path / "replay-ledger.sqlite3")
    replay_report = run_rolling_backtest(
        tuple(targets),
        snapshots=SnapshotCatalog((m8_snapshot,)),
        outcomes=replay_revealer,
        source_manifest=_source_manifest(m8_snapshot),
        code=_CLEAN_CODE,
        lock_store=LockStore(tmp_path / "replay-locks"),
        ledger=replay_ledger,
        backtest_run_id="m8-reproducible-mini",
        path_count_policy=_TEST_POLICY,
        execution_mode="test",
    )
    current_rows = tuple(row for row in all_rows if row.backtest_run_id == "m8-reproducible-mini")
    replay_rows = replay_ledger.read()
    assert [
        (
            row.match_id,
            row.prop_id,
            row.probability_raw,
            row.probability_settled,
            row.outcome_binary,
            row.brier_raw_model,
        )
        for row in replay_rows
    ] == [
        (
            row.match_id,
            row.prop_id,
            row.probability_raw,
            row.probability_settled,
            row.outcome_binary,
            row.brier_raw_model,
        )
        for row in current_rows
    ]
    assert [item[0] for item in replay_revealer.revealed] == [item[0] for item in revealer.revealed]
    assert replay_report.calibration == report.calibration
    assert replay_report.exclusion_counts == report.exclusion_counts

    unsafe = m8_snapshot.model_copy(
        update={"data_cutoff_utc": targets[0].context.information_cutoff_utc + timedelta(seconds=1)}
    )
    assert SnapshotCatalog((unsafe,)).select(targets[0].context) is None
    run_manifest_path = lock_store.root / "_backtest_runs" / report.run_id / "manifest.json"
    tampered_run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    tampered_run["terminal_ledger_sha256"] = "0" * 64
    run_manifest_path.write_text(json.dumps(tampered_run), encoding="utf-8")
    with pytest.raises(LedgerError, match="cannot load backtest run manifest"):
        load_backtest_run_manifest(run_manifest_path)


def test_information_bundle_rejects_future_items() -> None:
    context = _context()
    with pytest.raises(ValueError, match="before its cutoff"):
        InformationBundle(
            bundle_id="future-leak",
            scenario_id=context.information_scenario_id,
            information_cutoff_utc=context.information_cutoff_utc,
            items=(
                InformationItem(
                    category="conditions",
                    summary="future weather observation",
                    source_id="future",
                    source_sha256="d" * 64,
                    observed_at_utc=context.information_cutoff_utc,
                    available_at_utc=context.information_cutoff_utc + timedelta(seconds=1),
                ),
            ),
        )
    with pytest.raises(ValueError, match="before its cutoff"):
        InformationBundle(
            bundle_id="cutoff-equality",
            scenario_id=context.information_scenario_id,
            information_cutoff_utc=context.information_cutoff_utc,
            items=(
                InformationItem(
                    category="conditions",
                    summary="published exactly at cutoff",
                    source_id="equal-cutoff",
                    source_sha256="d" * 64,
                    observed_at_utc=context.information_cutoff_utc - timedelta(seconds=1),
                    available_at_utc=context.information_cutoff_utc,
                ),
            ),
        )
    with pytest.raises(ValueError, match="before it was observed"):
        InformationItem(
            category="conditions",
            summary="impossible source timing",
            source_id="reversed-time",
            source_sha256="d" * 64,
            observed_at_utc=context.information_cutoff_utc,
            available_at_utc=context.information_cutoff_utc - timedelta(seconds=1),
        )


def test_retrospective_source_retrieval_is_truthful_and_not_probability_leakage(
    m8_snapshot: ModelSnapshot,
) -> None:
    context = _context()
    source = (
        _source_manifest(m8_snapshot)
        .sources[0]
        .model_copy(
            update={
                "source_effective_at_utc": context.information_cutoff_utc - timedelta(days=1),
                "source_available_at_utc": context.information_cutoff_utc - timedelta(seconds=1),
                "retrieved_at_utc": context.information_cutoff_utc + timedelta(days=100),
            }
        )
    )
    manifest = _source_manifest(m8_snapshot).model_copy(update={"sources": (source,)})
    lock = create_prediction_lock(
        m8_snapshot,
        context,
        _information(context),
        (MATCH_WIN("p0"),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=_CLEAN_CODE,
        seed=991,
        n_paths=_TEST_POLICY.standard_paths,
        execution_mode="test",
        path_count_policy=_TEST_POLICY,
    )
    assert manifest.sources[0].retrieved_at_utc > context.information_cutoff_utc
    assert lock.prop_estimates


def test_atomic_publication_concurrency_interruptions_recovery_and_retry(
    m8_snapshot: ModelSnapshot,
    tmp_path: Path,
) -> None:
    lock = _create_lock(m8_snapshot, None, seed=772)
    concurrent_store = LockStore(tmp_path / "concurrent-locks")
    barrier = Barrier(2)

    def publish() -> str:
        barrier.wait()
        try:
            concurrent_store.write(lock)
        except (LockReservationConflict, LockAlreadyExistsError) as exc:
            return type(exc).__name__
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: publish(), range(2)))
    assert results.count("published") == 1
    assert set(results) <= {
        "published",
        "LockReservationConflict",
        "LockAlreadyExistsError",
    }
    assert concurrent_store.verify(lock.base_lock_id, 1) == lock.content_sha256

    for index, stage in enumerate(
        ("before_manifest", "after_payload", "after_verification"), start=1
    ):
        store = LockStore(tmp_path / f"interrupted-{index}")
        with pytest.raises(LockPublicationInterrupted):
            store.write(lock, interrupt_at=stage)  # type: ignore[arg-type]
        assert not store.revision_directory(lock.base_lock_id, 1).exists()
        incomplete = store.incomplete_publications()
        assert len(incomplete) == 1
        assert incomplete[0].temporary_paths
        if stage == "after_payload":
            corrupt = incomplete[0].temporary_paths[0] / "lock.json"
            corrupt.write_bytes(corrupt.read_bytes() + b"corruption")
            with pytest.raises(LockIntegrityError):
                store._verify_directory(incomplete[0].temporary_paths[0])
        quarantined = store.quarantine_stale_publication(
            lock.base_lock_id,
            1,
            minimum_age=timedelta(seconds=1),
            now_utc=datetime.now(UTC) + timedelta(minutes=1),
        )
        assert quarantined
        assert not store.incomplete_publications()
        store.write(lock)
        assert store.verify(lock.base_lock_id, 1) == lock.content_sha256


def test_end_to_end_escalation_discards_the_standard_path_run(
    m8_snapshot: ModelSnapshot,
) -> None:
    context = _context()
    policy = PathCountPolicy(
        standard_paths=12,
        escalated_paths=37,
        minimum_settled_paths=0,
        extreme_probability=0.03,
        integer_boundary_window=0.0,
        integer_boundary_standard_errors=0.0,
    )
    lock = create_prediction_lock(
        m8_snapshot,
        context,
        _information(context),
        (TOTAL_GAMES(ComparisonOperator.MORE_THAN, 1_000),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=_source_manifest(m8_snapshot),
        code=_CLEAN_CODE,
        seed=5005,
        n_paths=policy.standard_paths,
        execution_mode="test",
        path_count_policy=policy,
    )
    assert lock.simulation.requested_paths == 12
    assert lock.simulation.actual_paths == 37
    assert lock.simulation.escalated
    assert "PROBABILITY_BELOW_3_PERCENT" in lock.simulation.escalation_reasons
    assert lock.prop_estimates[0].total_paths == 37


def test_adaptive_lock_records_confidence_sequence_and_boundary_status(
    m8_snapshot: ModelSnapshot,
) -> None:
    context = _context()
    policy = AdaptiveMCPolicy(checkpoints=(12, 37))
    lock = create_prediction_lock(
        m8_snapshot,
        context,
        _information(context),
        (TOTAL_GAMES(ComparisonOperator.MORE_THAN, 1_000),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=_source_manifest(m8_snapshot),
        code=_CLEAN_CODE,
        seed=5006,
        execution_mode="test",
        path_count_policy=policy,
    )

    estimate = lock.prop_estimates[0]
    assert lock.simulation.path_count_policy.version == "adaptive_mc_cs_v1"
    assert lock.simulation.inspected_path_counts == (12, 37)
    assert lock.simulation.requested_paths == 37
    assert lock.simulation.actual_paths == 37
    assert estimate.yes_paths == 0
    assert estimate.no_paths == 37
    assert estimate.model_probability_raw == 0.0
    assert estimate.model_probability_integer == 0
    assert estimate.mc_confidence_sequence_lower == 0.0
    assert estimate.mc_confidence_sequence_upper is not None
    assert estimate.mc_confidence_sequence_upper > 0.005
    assert estimate.mc_stopping_status is MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE
    assert estimate.submitted_integer is None
    assert reproduce_prediction_lock(lock).reproduced
    card = render_locked_match_card(lock)
    assert "0.000%" in card
    assert "INTEGER_BOUNDARY_SENSITIVE" in card
    assert "does not prove the model probability is zero" in card
