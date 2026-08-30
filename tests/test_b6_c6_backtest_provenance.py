from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from tennis_model.calibration import (
    CalibrationLedger,
    OfficialHistoricalOutcome,
    SnapshotCatalog,
    historical_coverage_report,
    run_rolling_backtest,
)
from tennis_model.estimation.inactivity import (
    InactivityBand,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityRecord,
    InactivityTerminalStatus,
    LastEligibleMatch,
    PlayedPointEvidence,
    inactivity_factors,
)
from tennis_model.estimation.retirement import (
    HistoricalTerminationInput,
    OfficialTerminalStatus,
    RetirementObservationBatch,
    StartedEvidence,
    build_retirement_observations,
    normalize_historical_termination,
)
from tennis_model.locking import CodeProvenance, LockStore
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RowDateSemantics,
    SourceManifest,
    Tour,
    TourCoverage,
)

_CUTOFF = datetime(2026, 8, 20, 12, tzinfo=UTC)
_SCHEDULED_LOCAL_DATE = date(2026, 8, 30)


def _manifest() -> SourceManifest:
    coverage = CoverageRange(
        first_match_date=date(2021, 8, 20),
        last_match_date=date(2026, 8, 19),
        verified_at_utc=_CUTOFF - timedelta(days=1),
    )
    return SourceManifest(
        manifest_version="b6-c6-coverage-fixture/v1",
        sources=(
            PinnedSource(
                source_id="b6-c6-history",
                identity_namespace="b6-c6-history",
                tour=Tour.ATP,
                upstream_attribution="synthetic cutoff-safe audit fixture",
                locator="fixture://b6-c6-history",
                object_identifier="fixture-object-v1",
                sha256="1" * 64,
                schema_version="fixture/v1",
                stated_license="test-only",
                retrieved_at_utc=_CUTOFF - timedelta(days=2),
                verified_coverage=coverage,
                row_date_semantics=RowDateSemantics.MATCH_DATE,
                availability_lag_days=1,
            ),
        ),
        coverage_by_tour=TourCoverage(atp=coverage, wta=None),
    )


def _retirement_batch() -> RetirementObservationBatch:
    records = (
        HistoricalTerminationInput(
            match_id="timed-retirement",
            tour=Tour.ATP,
            player_a_id="p0",
            player_b_id="p1",
            match_date=_CUTOFF.date() - timedelta(days=10),
            official_status=OfficialTerminalStatus.RETIREMENT,
            started_evidence=(StartedEvidence.POSITIVE_POINT_STAT_COUNT,),
            retiring_player_id="p0",
            advancing_winner_id="p1",
            retirement_completed_games=12,
            source_id="official-history",
            source_sha256="2" * 64,
            available_at_utc=_CUTOFF - timedelta(days=9),
        ),
        HistoricalTerminationInput(
            match_id="untimed-retirement",
            tour=Tour.ATP,
            player_a_id="p2",
            player_b_id="p3",
            match_date=_CUTOFF.date() - timedelta(days=20),
            official_status=OfficialTerminalStatus.RETIREMENT,
            started_evidence=(StartedEvidence.EXPLICIT_OFFICIAL_STARTED_OR_IN_PLAY,),
            retiring_player_id="p3",
            advancing_winner_id="p2",
            source_id="official-history",
            source_sha256="3" * 64,
            available_at_utc=_CUTOFF - timedelta(days=19),
        ),
        HistoricalTerminationInput(
            match_id="walkover",
            tour=Tour.ATP,
            player_a_id="p4",
            player_b_id="p5",
            match_date=_CUTOFF.date() - timedelta(days=30),
            official_status=OfficialTerminalStatus.WALKOVER_OR_PRESTART_WITHDRAWAL,
            source_id="official-history",
            source_sha256="4" * 64,
            available_at_utc=_CUTOFF - timedelta(days=29),
        ),
    )
    normalized = tuple(normalize_historical_termination(item) for item in records)
    return build_retirement_observations(normalized, _CUTOFF)


def _inactivity_record(player_id: str, days: int) -> InactivityRecord:
    gap, multiplier, inflation = inactivity_factors(days)
    return InactivityRecord(
        player_id=player_id,
        tour=Tour.ATP,
        scheduled_start_local_date=_SCHEDULED_LOCAL_DATE,
        information_cutoff_utc=_CUTOFF,
        coverage=InactivityCoverageAssertion(
            state=InactivityCoverageState.VERIFIED_COMPLETE,
            source_manifest_id="b6-c6-coverage-fixture/v1",
            source_manifest_sha256="5" * 64,
            canonical_player_id=player_id,
            asserted_at_utc=_CUTOFF - timedelta(days=1),
        ),
        last_eligible_match=LastEligibleMatch(
            match_id=f"last-{player_id}",
            match_date_local=_SCHEDULED_LOCAL_DATE - timedelta(days=days),
            terminal_status=InactivityTerminalStatus.NORMAL_COMPLETION,
            started_evidence=(PlayedPointEvidence.POSITIVE_POINT_STAT_COUNT,),
            source_pin=f"fixture-row-{player_id}",
            source_sha256="6" * 64,
            available_at_utc=_CUTOFF - timedelta(days=1),
        ),
        inactivity_days=days,
        post_threshold_days=gap,
        hard_deviation_multiplier=multiplier,
        variance_inflation_factor=inflation,
        cold_start=False,
    )


class _NeverReveal:
    def reveal(self, match_id: str, *, locked_content_sha256: str) -> OfficialHistoricalOutcome:
        raise AssertionError((match_id, locked_content_sha256))


def test_rolling_report_aggregates_supplied_b6_c6_coverage(tmp_path: Path) -> None:
    batch = _retirement_batch()
    inactivity_records = (
        _inactivity_record("p0", 90),
        _inactivity_record("p1", 270),
    )

    report = run_rolling_backtest(
        (),
        snapshots=SnapshotCatalog(()),
        outcomes=_NeverReveal(),
        source_manifest=_manifest(),
        code=CodeProvenance(commit="test", dirty=False, diff_sha256=None),
        lock_store=LockStore(tmp_path / "locks"),
        ledger=CalibrationLedger(tmp_path / "ledger.sqlite3"),
        backtest_run_id="b6-c6-coverage",
        retirement_batches=(batch,),
    )

    coverage = report.coverage
    assert coverage.retirement_timing_available == 1
    assert coverage.retirement_timing_missing == 1
    assert "2 eligible started matches" in coverage.retirement_coverage
    assert "2 retirements" in coverage.retirement_coverage
    assert "1 excluded terminal records" in coverage.retirement_coverage
    assert "exact retirement timing (1 of 2 retirements missing)" in coverage.unavailable_fields
    exclusion_counts = dict(coverage.terminal_status_exclusions)
    assert sum(exclusion_counts.values()) == 1
    assert any("WALKOVER" in reason for reason in exclusion_counts)
    detailed_coverage = historical_coverage_report(
        _manifest(),
        retirement_batches=(batch,),
        inactivity_records=inactivity_records,
    )
    assert detailed_coverage.inactivity_band_counts == (
        (Tour.ATP, InactivityBand.ACTIVE_DAYS_0_90, 1),
        (Tour.ATP, InactivityBand.DAYS_181_365, 1),
    )

    no_b6 = historical_coverage_report(_manifest())
    assert no_b6.retirement_timing_available == 0
    assert no_b6.retirement_timing_missing == 0
    assert no_b6.retirement_coverage.startswith("not supplied")
    assert "exact retirement timing (not audited)" in no_b6.unavailable_fields


def _retirement_outcome(**updates: object) -> OfficialHistoricalOutcome:
    payload: dict[str, object] = {
        "match_id": "historical-retirement",
        "player_a_id": "p0",
        "player_b_id": "p1",
        "best_of": 3,
        "started": True,
        "completed": False,
        "winner_id": "p1",
        "retired_player_id": "p0",
        "sets_started": 1,
        "official_source_id": "official-result",
        "official_source_sha256": "7" * 64,
        "official_source_locator": "fixture://official-result",
        "available_at_utc": _CUTOFF + timedelta(hours=1),
        "retrieved_at_utc": _CUTOFF + timedelta(hours=2),
    }
    payload.update(updates)
    return OfficialHistoricalOutcome.model_validate(payload)


def test_historical_retirement_timing_reason_is_explicit_but_v1_compatible() -> None:
    legacy = _retirement_outcome()
    assert legacy.retirement_completed_games is None
    assert legacy.retirement_timing_missing_reason is None

    explained = _retirement_outcome(
        retirement_timing_missing_reason="  official source has no exact boundary  "
    )
    assert explained.retirement_timing_missing_reason == ("official source has no exact boundary")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _retirement_outcome(
            retirement_completed_games=8,
            retirement_timing_missing_reason="official source has no exact boundary",
        )
    with pytest.raises(ValueError, match="retirement-timing metadata"):
        _retirement_outcome(
            completed=True,
            retired_player_id=None,
            retirement_timing_missing_reason="not applicable",
        )
