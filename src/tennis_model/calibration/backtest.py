"""Strict rolling-origin forecast -> lock -> reveal -> settle orchestration."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from tennis_model.calibration.ledger import (
    CalibrationLedger,
    LedgerError,
    ledger_rows_from_settlement,
)
from tennis_model.calibration.metrics import CalibrationReport, summarize_calibration
from tennis_model.calibration.outcomes import (
    HistoricalAvailabilityPhase,
    HistoricalOutcomeError,
    OfficialHistoricalOutcome,
    settle_historical_lock,
)
from tennis_model.data.historical_validation import (
    POINT_IN_TIME_VINTAGE_POLICY,
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
)
from tennis_model.estimation.inactivity import InactivityBand, InactivityRecord
from tennis_model.estimation.retirement import (
    RetirementObservationBatch,
    RetirementScenarioMixture,
)
from tennis_model.estimation.snapshot import ModelSnapshot
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking._json import canonical_json_bytes, sha256_json
from tennis_model.locking.models import (
    CodeProvenance,
    InformationBundle,
    PredictionSnapshot,
    RetainedArtifactRecord,
    SerializedProp,
    deserialize_prop,
)
from tennis_model.locking.path_counts import FROZEN_PATH_COUNT_POLICY, PathCountPolicy
from tennis_model.locking.provenance import capture_runtime_fingerprint
from tennis_model.locking.service import LockCreationError, create_prediction_lock
from tennis_model.locking.store import LockStore
from tennis_model.props.settlement import CANONICAL_SETTLEMENT_POLICY, SettlementPolicy
from tennis_model.schemas import FrozenModel, SourceManifest, Tour
from tennis_model.simulation.match import (
    ACE_COMPARE,
    AND,
    ANY_TIEBREAK,
    BREAK_COUNT,
    DECIDING_SET,
    DF_COMPARE,
    EXACT_SCORE,
    FIRST_SERVE_WIN_PCT,
    FIRST_SET_GAMES,
    FIRST_SET_WIN,
    MATCH_WIN,
    PLAYER_ACES,
    PLAYER_DF,
    PLAYER_GAMES,
    STRAIGHT_SETS,
    TOTAL_ACES,
    TOTAL_BREAKS,
    TOTAL_DF,
    TOTAL_GAMES,
    BooleanCompositeSpec,
    ComparisonOperator,
    PropSpec,
)
from tennis_model.simulation.parameters import MatchContext

MARKET_GRID_VERSION = "historical-core-grid/v2"
BACKTEST_SEED_POLICY_VERSION = "paired-canonical-match-cutoff-purpose/v2"
PRODUCTION_SEED_POLICY_VERSION = "production-lock-root-seed/v1"


class BacktestExclusionReason(StrEnum):
    MISSING_COMPONENT_COUNTS = "MISSING_COMPONENT_COUNTS"
    INVALID_SCORE = "INVALID_SCORE"
    QUARANTINED_SOURCE_ROW = "QUARANTINED_SOURCE_ROW"
    INSUFFICIENT_PREMATCH_HISTORY = "INSUFFICIENT_PREMATCH_HISTORY"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    MISSING_SETTLEMENT_FIELDS = "MISSING_SETTLEMENT_FIELDS"
    NO_CUTOFF_SAFE_SNAPSHOT = "NO_CUTOFF_SAFE_SNAPSHOT"
    LOCK_CREATION_FAILED = "LOCK_CREATION_FAILED"
    OUTCOME_RECONCILIATION_FAILED = "OUTCOME_RECONCILIATION_FAILED"


class HistoricalTargetSourceProvenance(FrozenModel):
    """Pinned, genuinely pre-result schedule/order-of-play target evidence."""

    schema_version: Literal["historical-target-source/v1"] = "historical-target-source/v1"
    source_id: str
    source_sha256: str
    record_available_at_utc: datetime
    retrieved_at_utc: datetime

    @field_validator("record_available_at_utc", "retrieved_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("historical target source timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def provenance_is_valid(self) -> Self:
        if not self.source_id.strip():
            raise ValueError("historical target source ID must not be empty")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("historical target source requires a lowercase SHA-256")
        return self


class HistoricalForecastTarget(FrozenModel):
    """Pre-result input only; no result or result-bearing filename is permitted."""

    match_id: str
    context: MatchContext
    information: InformationBundle
    props: tuple[SerializedProp, ...]
    first_server_id: str | None = None
    inactivity_records: tuple[InactivityRecord, ...] = ()
    retirement_scenario_mixtures: tuple[RetirementScenarioMixture, ...] = ()
    canonical_match_identity: CanonicalMatchIdentity | None = None
    retained_artifacts: tuple[RetainedArtifactRecord, ...] = ()
    target_source_provenance: HistoricalTargetSourceProvenance | None = None

    @model_validator(mode="after")
    def target_is_coherent(self) -> Self:
        if not self.match_id.strip():
            raise ValueError("historical match ID must not be empty")
        if not self.props:
            raise ValueError("historical target requires a predeclared prop grid")
        if self.information.information_cutoff_utc != self.context.information_cutoff_utc:
            raise ValueError("target information and context cutoffs differ")
        if self.first_server_id is not None and self.first_server_id not in {
            self.context.player_a_id,
            self.context.player_b_id,
        }:
            raise ValueError("first server must be one of the target players")
        if self.inactivity_records and {item.player_id for item in self.inactivity_records} != {
            self.context.player_a_id,
            self.context.player_b_id,
        }:
            raise ValueError("historical C6 records must identify exactly the target players")
        if any(
            item.information_cutoff_utc != self.context.information_cutoff_utc
            for item in self.inactivity_records
        ):
            raise ValueError("historical C6 records must use the target information cutoff")
        if self.canonical_match_identity is not None and (
            self.canonical_match_identity.tour is not self.context.tour
            or set(self.canonical_match_identity.participant_ids)
            != {self.context.player_a_id, self.context.player_b_id}
        ):
            raise ValueError("historical canonical identity differs from the target context")
        if (
            self.target_source_provenance is not None
            and self.target_source_provenance.record_available_at_utc
            >= self.context.information_cutoff_utc
        ):
            raise ValueError("historical target schedule was not available before its cutoff")
        return self


class HistoricalOutcomeRevealer(Protocol):
    def reveal(
        self,
        match_id: str,
        *,
        locked_content_sha256: str,
    ) -> OfficialHistoricalOutcome: ...


@dataclass(frozen=True, slots=True)
class SnapshotCatalog:
    snapshots: tuple[ModelSnapshot, ...]

    def __post_init__(self) -> None:
        identities = tuple(item.snapshot_id for item in self.snapshots)
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot catalog identities must be unique")

    def select(self, context: MatchContext) -> ModelSnapshot | None:
        eligible = [
            snapshot
            for snapshot in self.snapshots
            if snapshot.tour is context.tour
            and snapshot.data_cutoff_utc <= context.information_cutoff_utc
            and (
                snapshot.retirement_artifact is None
                or snapshot.retirement_artifact.information_cutoff_utc
                <= context.information_cutoff_utc
            )
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (item.data_cutoff_utc, item.snapshot_id),
        )


class HistoricalFilter(FrozenModel):
    tour: Tour | None = None
    start_date: date | None = None
    end_date: date | None = None
    events: tuple[str, ...] = ()

    @model_validator(mode="after")
    def dates_are_ordered(self) -> Self:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("backtest start date cannot follow end date")
        return self

    def includes(self, target: HistoricalForecastTarget) -> bool:
        match_date = target.context.scheduled_start_utc.date()
        return (
            (self.tour is None or target.context.tour is self.tour)
            and (self.start_date is None or match_date >= self.start_date)
            and (self.end_date is None or match_date <= self.end_date)
            and (not self.events or target.context.event in self.events)
        )


class BacktestExclusion(FrozenModel):
    match_id: str
    reason: BacktestExclusionReason
    detail: str
    phase: HistoricalAvailabilityPhase = HistoricalAvailabilityPhase.MATCH_EXCLUDED_PRE_REVEAL


class TargetCohortManifest(FrozenModel):
    """Content-addressed target cohort frozen before any outcome access."""

    schema_version: Literal[
        "historical-target-cohort/v1", "historical-target-cohort/v2"
    ] = "historical-target-cohort/v2"
    cohort_id: str
    constructed_at_utc: datetime
    historical_source_index_sha256: str
    filter_sha256: str
    target_ids: tuple[str, ...]
    target_sha256: tuple[str, ...]
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        if self.constructed_at_utc.tzinfo is None or self.constructed_at_utc.utcoffset() is None:
            raise ValueError("cohort construction time must be timezone-aware")
        if len(self.target_ids) != len(self.target_sha256):
            raise ValueError("cohort target identities and hashes must align")
        excluded = {"cohort_id", "constructed_at_utc"}
        if self.schema_version == "historical-target-cohort/v1":
            if self.historical_validation_policy != POINT_IN_TIME_VINTAGE_POLICY:
                raise ValueError("v1 target cohorts can only represent strict vintage data")
            excluded.add("historical_validation_policy")
        payload = self.model_dump(mode="json", exclude=excluded)
        if self.cohort_id != sha256_json(payload):
            raise ValueError("cohort ID does not match its pre-outcome content")
        return self


def freeze_target_cohort(
    targets: tuple[HistoricalForecastTarget, ...],
    *,
    historical_filter: HistoricalFilter,
    historical_source_index_sha256: str,
    constructed_at_utc: datetime | None = None,
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
) -> tuple[TargetCohortManifest, tuple[HistoricalForecastTarget, ...]]:
    """Select and hash targets through an interface that cannot accept outcomes."""

    if len(historical_source_index_sha256) != 64 or any(
        item not in "0123456789abcdef" for item in historical_source_index_sha256
    ):
        raise ValueError("historical source index requires a SHA-256 digest")
    ordered = tuple(
        sorted(
            (item for item in targets if historical_filter.includes(item)),
            key=lambda item: (item.context.scheduled_start_utc, item.match_id),
        )
    )
    target_ids = tuple(item.match_id for item in ordered)
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("historical cohort cannot contain duplicate target IDs")
    if historical_validation_policy.mode is HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED:
        if not historical_validation_policy.exact_date_history_complete:
            raise ValueError("retrospective cohort requires complete exact-date history")
        if any(item.target_source_provenance is None for item in ordered):
            raise ValueError(
                "retrospective cohort requires pre-result schedule provenance for every target"
            )
    target_hashes = tuple(sha256_json(item.model_dump(mode="json")) for item in ordered)
    filter_sha256 = sha256_json(historical_filter.model_dump(mode="json"))
    payload = {
        "schema_version": "historical-target-cohort/v2",
        "historical_source_index_sha256": historical_source_index_sha256,
        "filter_sha256": filter_sha256,
        "target_ids": target_ids,
        "target_sha256": target_hashes,
        "historical_validation_policy": historical_validation_policy.model_dump(mode="json"),
    }
    manifest = TargetCohortManifest(
        cohort_id=sha256_json(payload),
        constructed_at_utc=(
            datetime.now(UTC) if constructed_at_utc is None else constructed_at_utc
        ),
        historical_source_index_sha256=historical_source_index_sha256,
        filter_sha256=filter_sha256,
        target_ids=target_ids,
        target_sha256=target_hashes,
        historical_validation_policy=historical_validation_policy,
    )
    return manifest, ordered


class SourceCoverage(FrozenModel):
    source_id: str
    tour: Tour
    earliest_date: date
    latest_date: date
    source_sha256: str


class HistoricalCoverageReport(FrozenModel):
    manifest_sha256: str
    sources: tuple[SourceCoverage, ...]
    service_stat_completeness: str
    retirement_coverage: str
    ace_df_coverage: str
    unavailable_fields: tuple[str, ...]
    quarantined_anomaly_count: int | None
    terminal_status_exclusions: tuple[tuple[str, int], ...] = ()
    retirement_timing_available: int = Field(default=0, ge=0)
    retirement_timing_missing: int = Field(default=0, ge=0)
    cold_start_count: int = Field(default=0, ge=0)
    inactivity_band_counts: tuple[tuple[Tour, InactivityBand, int], ...] = ()


def historical_coverage_report(
    manifest: SourceManifest,
    *,
    quarantined_anomaly_count: int | None = None,
    retirement_batches: tuple[RetirementObservationBatch, ...] = (),
    inactivity_records: tuple[InactivityRecord, ...] = (),
) -> HistoricalCoverageReport:
    from tennis_model.locking.models import SourceManifestProvenance

    pin = SourceManifestProvenance.from_manifest(manifest)
    exclusion_counts: Counter[str] = Counter()
    retirement_timing_available = 0
    retirement_timing_missing = 0
    eligible_started_matches = 0
    excluded_terminal_records = 0
    for batch in retirement_batches:
        exclusion_counts.update(item.exclusion_reason for item in batch.exclusions)
        eligible_started_matches += len(batch.observations) // 2
        excluded_terminal_records += len(batch.exclusions)
        retirements = tuple(item for item in batch.observations if item.response == 1)
        retirement_timing_available += sum(
            item.retirement_completed_games is not None for item in retirements
        )
        retirement_timing_missing += sum(
            item.retirement_completed_games is None for item in retirements
        )
    band_counts = Counter((item.tour, item.band) for item in inactivity_records)
    retirement_count = retirement_timing_available + retirement_timing_missing
    if retirement_batches:
        retirement_coverage = (
            f"{len(retirement_batches)} cutoff-safe B6 batch(es): "
            f"{eligible_started_matches} eligible started matches, "
            f"{retirement_count} retirements, and "
            f"{excluded_terminal_records} excluded terminal records"
        )
    else:
        retirement_coverage = (
            "not supplied; terminal-status and retirement-timing coverage not measured"
        )
    band_order = {band: index for index, band in enumerate(InactivityBand)}
    unavailable_fields = ["winners", "unforced_errors", "duration convention"]
    if not retirement_batches:
        unavailable_fields.append("exact retirement timing (not audited)")
    elif retirement_timing_missing:
        unavailable_fields.append(
            "exact retirement timing "
            f"({retirement_timing_missing} of {retirement_count} retirements missing)"
        )
    return HistoricalCoverageReport(
        manifest_sha256=pin.manifest_sha256,
        sources=tuple(
            SourceCoverage(
                source_id=source.source_id,
                tour=source.tour,
                earliest_date=source.verified_coverage.first_match_date,
                latest_date=source.verified_coverage.last_match_date,
                source_sha256=source.sha256,
            )
            for source in manifest.sources
        ),
        service_stat_completeness=(
            "requires processed-table field audit; not inferred from manifest"
        ),
        retirement_coverage=retirement_coverage,
        ace_df_coverage="source-dependent and must be measured from normalized rows",
        unavailable_fields=tuple(unavailable_fields),
        quarantined_anomaly_count=quarantined_anomaly_count,
        terminal_status_exclusions=tuple(sorted(exclusion_counts.items())),
        retirement_timing_available=retirement_timing_available,
        retirement_timing_missing=retirement_timing_missing,
        cold_start_count=sum(item.cold_start for item in inactivity_records),
        inactivity_band_counts=tuple(
            (tour, band, count)
            for (tour, band), count in sorted(
                band_counts.items(),
                key=lambda item: (item[0][0].value, band_order[item[0][1]]),
            )
        ),
    )


def fixed_core_market_grid(context: MatchContext) -> tuple[SerializedProp, ...]:
    """Predeclared championship-like thresholds, chosen without historical outcomes."""

    from tennis_model.locking.models import serialize_prop

    total_games = (19.5, 22.5, 25.5) if context.best_of == 3 else (35.5, 39.5, 43.5)
    target_sets = context.best_of // 2 + 1
    props: list[PropSpec | BooleanCompositeSpec] = [
        MATCH_WIN(context.player_a_id),
        MATCH_WIN(context.player_b_id),
        *(
            EXACT_SCORE(player, target_sets, loser_sets)
            for player in (context.player_a_id, context.player_b_id)
            for loser_sets in range(target_sets)
        ),
        STRAIGHT_SETS(context.player_a_id),
        STRAIGHT_SETS(context.player_b_id),
        ANY_TIEBREAK(),
        DECIDING_SET(),
        FIRST_SET_WIN(context.player_a_id),
        FIRST_SET_GAMES(ComparisonOperator.MORE_THAN, 9.5),
        TOTAL_BREAKS(ComparisonOperator.MORE_THAN, 6),
        BREAK_COUNT(context.player_a_id, ComparisonOperator.MORE_THAN, 2),
        BREAK_COUNT(context.player_b_id, ComparisonOperator.MORE_THAN, 2),
        PLAYER_GAMES(
            context.player_a_id,
            ComparisonOperator.MORE_THAN,
            10.5 if context.best_of == 3 else 18.5,
        ),
        PLAYER_GAMES(
            context.player_b_id,
            ComparisonOperator.MORE_THAN,
            10.5 if context.best_of == 3 else 18.5,
        ),
    ]
    props.extend(TOTAL_GAMES(ComparisonOperator.MORE_THAN, threshold) for threshold in total_games)
    for player in (context.player_a_id, context.player_b_id):
        props.extend(
            (
                PLAYER_ACES(player, ComparisonOperator.MORE_THAN, 4),
                PLAYER_ACES(player, ComparisonOperator.MORE_THAN, 8),
                PLAYER_DF(player, ComparisonOperator.MORE_THAN, 3),
                FIRST_SERVE_WIN_PCT(player, ComparisonOperator.MORE_THAN, 65),
            )
        )
    props.extend(
        (
            TOTAL_ACES(ComparisonOperator.MORE_THAN, 15),
            ACE_COMPARE(context.player_a_id, context.player_b_id),
            TOTAL_DF(ComparisonOperator.MORE_THAN, 6),
            DF_COMPARE(context.player_a_id, context.player_b_id),
            AND(MATCH_WIN(context.player_a_id), ANY_TIEBREAK()),
        )
    )
    return tuple(serialize_prop(prop) for prop in props)


def derive_backtest_seed(
    *,
    canonical_match_id: str,
    forecast_cutoff_utc: datetime,
    simulation_purpose: str = "paired-historical-evaluation",
    path_index: int = 0,
) -> int:
    """Paired-comparison seed independent of framework and backtest run identity."""

    if forecast_cutoff_utc.tzinfo is None or forecast_cutoff_utc.utcoffset() is None:
        raise ValueError("paired evaluation cutoff must be timezone-aware")
    if not canonical_match_id.strip() or not simulation_purpose.strip() or path_index < 0:
        raise ValueError("paired seed inputs must be nonempty and path index nonnegative")
    payload = "\0".join(
        (
            BACKTEST_SEED_POLICY_VERSION,
            canonical_match_id,
            forecast_cutoff_utc.astimezone(UTC).isoformat(),
            simulation_purpose,
            str(path_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big", signed=False)


def derive_production_seed(
    *,
    canonical_match_id: str,
    lock_revision: int,
    purpose: str,
    entropy: str,
) -> int:
    """Versioned production seed policy, distinct from paired evaluation."""

    seed_parts = (canonical_match_id, purpose, entropy)
    if lock_revision < 1 or not all(item.strip() for item in seed_parts):
        raise ValueError("production seed inputs are invalid")
    payload = "\0".join(
        (
            PRODUCTION_SEED_POLICY_VERSION,
            canonical_match_id,
            str(lock_revision),
            purpose,
            entropy,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big", signed=False)


class BacktestRunManifest(FrozenModel):
    """Immutable binding of cohort, locks, settlements, ledger, and runtime."""

    schema_version: Literal[
        "backtest-run-manifest/v1", "backtest-run-manifest/v2"
    ] = "backtest-run-manifest/v2"
    manifest_id: str
    run_id: str
    created_at_utc: datetime
    cohort_id: str
    lock_content_sha256: tuple[str, ...]
    settlement_row_ids: tuple[str, ...]
    terminal_ledger_sha256: str | None
    source_manifest_sha256: str
    code_commit: str
    code_dirty_sha256: str | None
    runtime_fingerprint_sha256: str
    seed_policy_version: str
    market_grid_version: str
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY

    @model_validator(mode="after")
    def manifest_identity_matches(self) -> Self:
        if self.created_at_utc.tzinfo is None or self.created_at_utc.utcoffset() is None:
            raise ValueError("backtest manifest creation time must be timezone-aware")
        excluded = {"manifest_id"}
        if self.schema_version == "backtest-run-manifest/v1":
            if self.historical_validation_policy != POINT_IN_TIME_VINTAGE_POLICY:
                raise ValueError("v1 backtest manifests can only represent strict vintage data")
            excluded.add("historical_validation_policy")
        expected = sha256_json(self.model_dump(mode="json", exclude=excluded))
        if self.manifest_id != expected:
            raise ValueError("backtest run manifest ID does not match its content")
        return self


def _backtest_manifest_bytes(manifest: BacktestRunManifest) -> bytes:
    value = manifest.model_dump(mode="json")
    if manifest.schema_version == "backtest-run-manifest/v1":
        value.pop("historical_validation_policy")
    return canonical_json_bytes(value)


def persist_backtest_run_manifest(
    manifest: BacktestRunManifest,
    root: str | Path,
) -> Path:
    """Publish a completed run manifest immutably after ledger anchoring."""

    parent = Path(root).resolve()
    target = parent / manifest.run_id
    payload = _backtest_manifest_bytes(manifest)
    if target.exists():
        existing = target / "manifest.json"
        if existing.is_file() and existing.read_bytes() == payload:
            return target
        raise LedgerError(f"backtest run manifest already exists: {manifest.run_id}")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".partial-backtest-run-", dir=parent))
    try:
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        verified = BacktestRunManifest.model_validate_json(manifest_path.read_bytes())
        if verified != manifest:
            raise LedgerError("backtest run manifest changed during prepublication verification")
        temporary.rename(target)
    except OSError as exc:
        raise LedgerError(f"cannot publish backtest run manifest: {exc}") from exc
    return target


def load_backtest_run_manifest(path: str | Path) -> BacktestRunManifest:
    """Load and content-verify one immutably published run manifest."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = BacktestRunManifest.model_validate_json(payload)
    except (OSError, ValueError) as exc:
        raise LedgerError(f"cannot load backtest run manifest: {exc}") from exc
    if _backtest_manifest_bytes(manifest) != payload:
        raise LedgerError("backtest run manifest is not canonical JSON")
    return manifest


@dataclass(frozen=True, slots=True)
class BacktestReport:
    run_id: str
    targets: int
    locks_created: int
    ledger_rows: int
    exclusions: tuple[BacktestExclusion, ...]
    exclusion_counts: tuple[tuple[BacktestExclusionReason, int], ...]
    calibration: CalibrationReport
    coverage: HistoricalCoverageReport
    cohort_manifest: TargetCohortManifest
    run_manifest: BacktestRunManifest
    runtime_seconds: float
    seed_policy_version: str = BACKTEST_SEED_POLICY_VERSION
    market_grid_version: str = MARKET_GRID_VERSION


def forecast_historical_match(
    target: HistoricalForecastTarget,
    snapshot: ModelSnapshot,
    *,
    source_manifest: SourceManifest,
    code: CodeProvenance,
    policy: SettlementPolicy,
    seed: int,
    lock_store: LockStore,
    path_count_policy: PathCountPolicy,
    execution_mode: Literal["production", "development", "test"],
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
) -> PredictionSnapshot:
    """Forecast-only API: deliberately has no result parameter."""

    return create_prediction_lock(
        snapshot,
        target.context,
        target.information,
        tuple(deserialize_prop(item) for item in target.props),
        policy,
        source_manifest=source_manifest,
        code=code,
        seed=seed,
        store=lock_store,
        n_paths=path_count_policy.standard_paths,
        first_server_id=target.first_server_id,
        execution_mode=execution_mode,
        path_count_policy=path_count_policy,
        allow_dirty=True,
        inactivity_records=target.inactivity_records,
        retirement_scenario_mixtures=target.retirement_scenario_mixtures,
        canonical_match_identity=target.canonical_match_identity,
        retained_artifacts=target.retained_artifacts,
        historical_validation_policy=historical_validation_policy,
    )


def run_rolling_backtest(
    targets: tuple[HistoricalForecastTarget, ...],
    *,
    snapshots: SnapshotCatalog,
    outcomes: HistoricalOutcomeRevealer,
    source_manifest: SourceManifest,
    code: CodeProvenance,
    lock_store: LockStore,
    ledger: CalibrationLedger,
    backtest_run_id: str,
    historical_filter: HistoricalFilter | None = None,
    policy: SettlementPolicy = CANONICAL_SETTLEMENT_POLICY,
    path_count_policy: PathCountPolicy = FROZEN_PATH_COUNT_POLICY,
    execution_mode: Literal["production", "development", "test"] = "development",
    retirement_batches: tuple[RetirementObservationBatch, ...] = (),
    historical_source_index_sha256: str | None = None,
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
) -> BacktestReport:
    """Create each lock before the outcome provider is allowed to reveal its result."""

    if not backtest_run_id.strip():
        raise ValueError("backtest_run_id must not be empty")
    selected_filter = HistoricalFilter() if historical_filter is None else historical_filter
    source_manifest_sha256 = sha256_json(source_manifest.model_dump(mode="json"))
    cohort, ordered = freeze_target_cohort(
        targets,
        historical_filter=selected_filter,
        historical_source_index_sha256=(
            source_manifest_sha256
            if historical_source_index_sha256 is None
            else historical_source_index_sha256
        ),
        historical_validation_policy=historical_validation_policy,
    )
    started = time.perf_counter()
    exclusions: list[BacktestExclusion] = []
    current_run_rows = []
    locks = 0
    rows_written = 0
    lock_hashes: list[str] = []
    for target in ordered:
        snapshot = snapshots.select(target.context)
        if snapshot is None:
            exclusions.append(
                BacktestExclusion(
                    match_id=target.match_id,
                    reason=BacktestExclusionReason.NO_CUTOFF_SAFE_SNAPSHOT,
                    detail=(
                        "no fitted snapshot has both data and fit timestamps before the "
                        "forecast cutoff"
                    ),
                )
            )
            continue
        seed = derive_backtest_seed(
            canonical_match_id=(
                target.match_id
                if target.canonical_match_identity is None
                else target.canonical_match_identity.canonical_match_id
            ),
            forecast_cutoff_utc=target.context.information_cutoff_utc,
        )
        try:
            lock = forecast_historical_match(
                target,
                snapshot,
                source_manifest=source_manifest,
                code=code,
                policy=policy,
                seed=seed,
                lock_store=lock_store,
                path_count_policy=path_count_policy,
                execution_mode=execution_mode,
                historical_validation_policy=historical_validation_policy,
            )
            # The result boundary is crossed only after the lock store accepted the artifact.
            lock_store.verify(lock.base_lock_id, lock.revision)
            lock_hashes.append(lock.content_sha256)
            outcome = outcomes.reveal(
                target.match_id,
                locked_content_sha256=lock.content_sha256,
            )
            settlement = settle_historical_lock(lock, outcome)
            rows = ledger_rows_from_settlement(
                lock,
                outcome,
                settlement,
                created_at_utc=outcome.retrieved_at_utc,
                backtest_run_id=backtest_run_id,
            )
            for row in rows:
                ledger.append(row)
                current_run_rows.append(row)
                rows_written += 1
            locks += 1
        except LockCreationError as exc:
            exclusions.append(
                BacktestExclusion(
                    match_id=target.match_id,
                    reason=BacktestExclusionReason.LOCK_CREATION_FAILED,
                    detail=str(exc),
                )
            )
        except (HistoricalOutcomeError, LedgerError) as exc:
            exclusions.append(
                BacktestExclusion(
                    match_id=target.match_id,
                    reason=BacktestExclusionReason.OUTCOME_RECONCILIATION_FAILED,
                    detail=str(exc),
                    phase=HistoricalAvailabilityPhase.TARGET_FAILED_POST_REVEAL,
                )
            )
    chain = ledger.verify_chain()
    runtime = capture_runtime_fingerprint(
        simulator_algorithm_version="rolling-origin-backtest/v1",
        chunk_size=path_count_policy.standard_paths,
        thread_count=1,
        process_count=1,
    )
    manifest_created = datetime.now(UTC)
    manifest_payload = {
        "schema_version": "backtest-run-manifest/v2",
        "run_id": backtest_run_id,
        "created_at_utc": manifest_created,
        "cohort_id": cohort.cohort_id,
        "lock_content_sha256": tuple(lock_hashes),
        "settlement_row_ids": tuple(row.row_id for row in current_run_rows),
        "terminal_ledger_sha256": chain.terminal_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "code_commit": code.commit,
        "code_dirty_sha256": code.diff_sha256,
        "runtime_fingerprint_sha256": sha256_json(runtime.model_dump(mode="json")),
        "seed_policy_version": BACKTEST_SEED_POLICY_VERSION,
        "market_grid_version": MARKET_GRID_VERSION,
        "historical_validation_policy": historical_validation_policy.model_dump(mode="json"),
    }
    provisional_manifest = BacktestRunManifest.model_construct(
        manifest_id="0" * 64,
        schema_version="backtest-run-manifest/v2",
        run_id=backtest_run_id,
        created_at_utc=manifest_created,
        cohort_id=cohort.cohort_id,
        lock_content_sha256=tuple(lock_hashes),
        settlement_row_ids=tuple(row.row_id for row in current_run_rows),
        terminal_ledger_sha256=chain.terminal_sha256,
        source_manifest_sha256=source_manifest_sha256,
        code_commit=code.commit,
        code_dirty_sha256=code.diff_sha256,
        runtime_fingerprint_sha256=sha256_json(runtime.model_dump(mode="json")),
        seed_policy_version=BACKTEST_SEED_POLICY_VERSION,
        market_grid_version=MARKET_GRID_VERSION,
        historical_validation_policy=historical_validation_policy,
    )
    run_manifest = BacktestRunManifest.model_validate(
        {
            "manifest_id": sha256_json(
                provisional_manifest.model_dump(mode="json", exclude={"manifest_id"})
            ),
            **manifest_payload,
        }
    )
    persist_backtest_run_manifest(run_manifest, lock_store.root / "_backtest_runs")
    return BacktestReport(
        run_id=backtest_run_id,
        targets=len(ordered),
        locks_created=locks,
        ledger_rows=rows_written,
        exclusions=tuple(exclusions),
        exclusion_counts=tuple(
            sorted(
                Counter(item.reason for item in exclusions).items(),
                key=lambda item: item[0].value,
            )
        ),
        calibration=summarize_calibration(tuple(current_run_rows)),
        coverage=historical_coverage_report(
            source_manifest,
            retirement_batches=retirement_batches,
            inactivity_records=tuple(
                record for target in ordered for record in target.inactivity_records
            ),
        ),
        cohort_manifest=cohort,
        run_manifest=run_manifest,
        runtime_seconds=time.perf_counter() - started,
    )
