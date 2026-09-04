"""Append-only, hash-chained calibration ledger."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from tennis_model.calibration.outcomes import (
    HistoricalLockSettlement,
    OfficialHistoricalOutcome,
)
from tennis_model.locking._json import canonical_json_bytes, require_sha256, sha256_json
from tennis_model.locking.models import (
    LEDGER_B6_C6_SCHEMA_VERSION,
    LEDGER_OPERATIONAL_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    PredictionSnapshot,
    PropSupportStatus,
    SerializedProp,
)
from tennis_model.schemas import FrozenModel, Tour
from tennis_model.simulation.parameters import (
    InactivityMatchParameters,
    RetirementMatchParameters,
)


class LedgerError(RuntimeError):
    pass


class B6C6LedgerProvenance(FrozenModel):
    """Probability-contract inputs retained on each amended historical row."""

    retirement: RetirementMatchParameters
    inactivity: InactivityMatchParameters
    retirement_schema_version: str
    inactivity_schema_version: str
    match_retirement_probability: float = Field(ge=0, le=1)
    player_retirement_probabilities: tuple[tuple[str, float], tuple[str, float]]
    retired_player_id: str | None = None
    retirement_completed_games: int | None = Field(default=None, ge=0)
    retirement_timing_missing_reason: str | None = None

    @field_validator("retired_player_id", "retirement_timing_missing_reason")
    @classmethod
    def optional_text_is_present(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @model_validator(mode="after")
    def identities_are_coherent(self) -> Self:
        posterior_players = tuple(item.player_id for item in self.retirement.player_posteriors)
        if tuple(item[0] for item in self.player_retirement_probabilities) != posterior_players:
            raise ValueError("ledger B6 player probabilities differ from posterior ordering")
        if tuple(item.player_id for item in self.inactivity.records) != posterior_players:
            raise ValueError("ledger C6 player records differ from B6 player ordering")
        if any(not 0.0 <= item[1] <= 1.0 for item in self.player_retirement_probabilities):
            raise ValueError("ledger B6 player probabilities must lie in [0, 1]")
        if not isclose(
            sum(item[1] for item in self.player_retirement_probabilities),
            self.match_retirement_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("ledger B6 player probabilities do not sum to match retirement")
        if self.retirement.artifact_schema_version != self.retirement_schema_version:
            raise ValueError("ledger B6 schema differs from retirement parameters")
        if any(
            item.schema_version != self.inactivity_schema_version
            for item in self.inactivity.records
        ):
            raise ValueError("ledger C6 schema differs from inactivity records")
        if self.retired_player_id is None:
            if (
                self.retirement_completed_games is not None
                or self.retirement_timing_missing_reason is not None
            ):
                raise ValueError("non-retirement provenance cannot contain timing metadata")
        else:
            if self.retired_player_id not in posterior_players:
                raise ValueError("ledger retiree is not one of the modeled players")
            if (self.retirement_completed_games is None) == (
                self.retirement_timing_missing_reason is None
            ):
                raise ValueError(
                    "retirement provenance requires timing or one explicit missing reason"
                )
        return self


class CalibrationLedgerRow(FrozenModel):
    schema_version: Literal[
        "calibration-ledger/v1", "calibration-ledger/v2", "calibration-ledger/v3"
    ] = LEDGER_SCHEMA_VERSION
    row_id: str
    correction_of_row_id: str | None = None
    correction_reason: str | None = None
    created_at_utc: datetime
    backtest_run_id: str | None = None
    match_id: str
    event: str
    lock_id: str
    lock_revision: int = Field(ge=1)
    lock_content_sha256: str
    prop_id: str
    prop: SerializedProp
    prop_family: str
    tour: Tour
    round: str
    player_a_id: str
    player_b_id: str
    scheduled_start_utc: datetime
    information_cutoff_utc: datetime
    framework_version: Literal["v1.0"]
    settlement_policy_version: str
    source_manifest_sha256: str
    data_hash: str
    fit_hash: str
    config_hash: str
    code_commit: str
    code_dirty: bool
    code_diff_sha256: str | None
    scenario_id: str
    simulation_paths: int = Field(gt=0)
    settled_paths: int = Field(ge=0)
    rng_seed_id: str
    mc_standard_error: float = Field(ge=0)
    probability_raw: float | None = Field(default=None, ge=0, le=1)
    probability_settled: float | None = Field(default=None, ge=0, le=1)
    probability_submitted: int | None = Field(default=None, ge=1, le=99)
    data_grade: Literal["A", "B", "C"]
    resolution_status: Literal["yes", "no", "void", "unavailable", "unresolved"]
    outcome_binary: int | None = Field(default=None, ge=0, le=1)
    official_value: dict[str, Any]
    official_source_id: str
    official_source_sha256: str
    official_source_locator: str
    resolved_at_utc: datetime
    retrieved_at_utc: datetime
    brier_raw_model: float | None = Field(default=None, ge=0, le=1)
    brier_submitted: float | None = Field(default=None, ge=0, le=1)
    quantization_loss: float | None = None
    submission_rounding_policy_version: str | None = None
    support_status: PropSupportStatus = PropSupportStatus.SUPPORTED
    support_reason_code: str | None = None
    match_retired: bool
    policy_flags: tuple[str, ...] = ()
    b6_c6_provenance: B6C6LedgerProvenance | None = None

    @field_validator(
        "row_id",
        "lock_content_sha256",
        "source_manifest_sha256",
        "data_hash",
        "fit_hash",
        "config_hash",
        "official_source_sha256",
    )
    @classmethod
    def hashes_are_valid(cls, value: str, info: Any) -> str:
        return require_sha256(value, field=info.field_name)

    @field_validator("code_diff_sha256")
    @classmethod
    def optional_hash_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field="code_diff_sha256")

    @field_validator("backtest_run_id")
    @classmethod
    def optional_run_id_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("backtest_run_id must not be empty")
        return value

    @field_validator(
        "created_at_utc",
        "scheduled_start_utc",
        "information_cutoff_utc",
        "resolved_at_utc",
        "retrieved_at_utc",
    )
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def row_is_coherent(self) -> Self:
        settled = self.resolution_status in {"yes", "no"}
        if settled and self.probability_raw is None:
            raise ValueError("settled ledger rows require a raw probability")
        if self.support_status is PropSupportStatus.SUPPORTED and self.probability_raw is None:
            raise ValueError("supported ledger rows require a raw probability")
        if settled != (self.outcome_binary is not None):
            raise ValueError("only yes/no ledger rows may have a binary outcome")
        if settled != (self.brier_raw_model is not None):
            raise ValueError("raw Brier scores exist exactly for settled yes/no rows")
        if (settled and self.probability_submitted is not None) != (
            self.brier_submitted is not None
        ):
            raise ValueError("submitted Brier requires a settled row and integer preview")
        if (self.brier_submitted is not None) != (self.quantization_loss is not None):
            raise ValueError("quantization loss exists exactly when submitted Brier exists")
        if self.quantization_loss is not None and not isclose(
            self.quantization_loss,
            cast(float, self.brier_submitted) - cast(float, self.brier_raw_model),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("quantization loss differs from submitted minus raw Brier")
        supported = self.support_status is PropSupportStatus.SUPPORTED
        if supported != (self.support_reason_code is None):
            raise ValueError("ledger support status and reason are inconsistent")
        if not supported and self.probability_submitted is not None:
            raise ValueError("unsupported ledger rows cannot contain a submitted preview")
        if (self.probability_submitted is not None) != (
            self.submission_rounding_policy_version is not None
        ):
            raise ValueError("submitted preview requires its rounding policy version")
        if (self.correction_of_row_id is None) != (self.correction_reason is None):
            raise ValueError("correction pointer and reason must be supplied together")
        if self.schema_version == LEDGER_SCHEMA_VERSION:
            if self.b6_c6_provenance is not None:
                raise ValueError("v1 ledger rows cannot contain B6/C6 provenance")
            excluded = {"row_id", "b6_c6_provenance"}
        else:
            if self.b6_c6_provenance is None:
                raise ValueError("v2 ledger rows require complete B6/C6 provenance")
            if self.match_retired != (self.b6_c6_provenance.retired_player_id is not None):
                raise ValueError("v2 ledger retirement flag differs from B6 provenance")
            excluded = {"row_id"}
        expected = sha256_json(self.model_dump(mode="json", exclude=excluded))
        if self.row_id != expected:
            raise ValueError("ledger row ID does not match its content")
        return self


def _prop_family(kind: str) -> str:
    if kind == "MATCH_WIN":
        return "MATCH_WINNER"
    if kind == "EXACT_SCORE":
        return "EXACT_SCORE"
    if kind in {
        "STRAIGHT_SETS",
        "PLAYER_WINS_SET",
        "FIRST_SET_WIN",
        "ANY_LOPSIDED_SET",
        "SET_SCORE",
    }:
        return "SETS"
    if "TIEBREAK" in kind or kind == "DECIDING_SET":
        return "TIEBREAK"
    if "GAME" in kind:
        return "GAMES"
    if "BREAK" in kind or kind == "BOTH_BREAK":
        return "BREAKS"
    if "ACE" in kind:
        return "ACES"
    if kind in {"PLAYER_DF", "TOTAL_DF", "DF_COMPARE"}:
        return "DOUBLE_FAULTS"
    if kind == "FIRST_SERVE_WIN_PCT":
        return "FIRST_SERVE_WIN_PERCENTAGE"
    return "COMPOUND"


def ledger_rows_from_settlement(
    lock: PredictionSnapshot,
    outcome: OfficialHistoricalOutcome,
    settlement: HistoricalLockSettlement,
    *,
    created_at_utc: datetime | None = None,
    backtest_run_id: str | None = None,
) -> tuple[CalibrationLedgerRow, ...]:
    if settlement.lock_id != lock.lock_id or settlement.outcome_match_id != outcome.match_id:
        raise LedgerError("settlement identity differs from lock or outcome")
    forecasts = {item.prop_id: item for item in lock.prop_estimates}
    gates = {item.prop_id: item for item in lock.prop_gates}
    created = datetime.now(UTC) if created_at_utc is None else created_at_utc
    rows = []
    b6_c6_provenance: B6C6LedgerProvenance | None = None
    row_schema_version: Literal[
        "calibration-ledger/v1", "calibration-ledger/v2", "calibration-ledger/v3"
    ] = LEDGER_SCHEMA_VERSION
    if lock.schema_version in {"prediction-lock/v2", "prediction-lock/v3"}:
        retirement = lock.match_parameters.retirement
        inactivity = lock.match_parameters.inactivity
        if retirement is None or inactivity is None:
            raise LedgerError("modern lock lost its required B6/C6 match parameters")
        player_retirement_probabilities = tuple(
            (item.player_id, item.retirement_probability) for item in lock.match_summary.players
        )
        if any(item[1] is None for item in player_retirement_probabilities):
            raise LedgerError("modern lock lacks player-level B6 probabilities")
        match_retirement_probability = lock.match_summary.retirement_probability
        if match_retirement_probability is None:
            raise LedgerError("modern lock lacks match-level B6 probability")
        if (
            outcome.retired_player_id is not None
            and outcome.retirement_completed_games is None
            and outcome.retirement_timing_missing_reason is None
        ):
            raise LedgerError(
                "modern retirement outcome needs timing or an explicit timing-missing reason"
            )
        b6_c6_provenance = B6C6LedgerProvenance(
            retirement=retirement,
            inactivity=inactivity,
            retirement_schema_version=retirement.artifact_schema_version,
            inactivity_schema_version=inactivity.records[0].schema_version,
            match_retirement_probability=match_retirement_probability,
            player_retirement_probabilities=cast(
                tuple[tuple[str, float], tuple[str, float]],
                player_retirement_probabilities,
            ),
            retired_player_id=outcome.retired_player_id,
            retirement_completed_games=outcome.retirement_completed_games,
            retirement_timing_missing_reason=outcome.retirement_timing_missing_reason,
        )
        row_schema_version = (
            LEDGER_OPERATIONAL_SCHEMA_VERSION
            if lock.schema_version == "prediction-lock/v3"
            else LEDGER_B6_C6_SCHEMA_VERSION
        )
    official_value = outcome.model_dump(mode="json")
    if row_schema_version == LEDGER_SCHEMA_VERSION:
        for optional_timing_field in (
            "retirement_completed_games",
            "retirement_timing_missing_reason",
        ):
            if official_value[optional_timing_field] is None:
                del official_value[optional_timing_field]
    for resolution in settlement.resolutions:
        forecast = forecasts.get(resolution.prop_id)
        gate = gates.get(resolution.prop_id)
        if (forecast is None) == (gate is None):
            raise LedgerError("settlement prop is absent or duplicated across lock estimates/gates")
        binary = resolution.outcome_binary
        market_override = (
            forecast is not None
            and lock.market_match_winner is not None
            and forecast.prop.kind == "MATCH_WIN"
        )
        probability_raw = (
            None
            if forecast is None
            else lock.effective_prop_probability(forecast.prop_id)
        )
        brier_raw = (
            None if binary is None or probability_raw is None else (probability_raw - binary) ** 2
        )
        submitted_integer: int | None = None
        rounding_policy: str | None = None
        if (
            forecast is not None
            and resolution.support_status is PropSupportStatus.SUPPORTED
        ):
            if market_override:
                submitted_integer = lock.effective_prop_submission_integer(forecast.prop_id)
                rounding_policy = (
                    lock.market_match_winner.submission_rounding_policy_version
                    if lock.market_match_winner is not None
                    else None
                )
            elif forecast.platform_submission_integer is not None:
                submitted_integer = forecast.platform_submission_integer
                rounding_policy = forecast.platform_submission_policy_version
            elif forecast.submitted_integer is not None:
                submitted_integer = forecast.submitted_integer
                rounding_policy = forecast.submission_rounding_policy_version
        submitted_probability = None if submitted_integer is None else submitted_integer / 100.0
        brier_submitted = (
            None
            if binary is None or submitted_probability is None
            else (submitted_probability - binary) ** 2
        )
        quantization_loss = (
            None if brier_submitted is None or brier_raw is None else brier_submitted - brier_raw
        )
        payload: dict[str, Any] = {
            "schema_version": row_schema_version,
            "created_at_utc": created,
            "backtest_run_id": backtest_run_id,
            "match_id": outcome.match_id,
            "event": lock.context.event,
            "lock_id": lock.lock_id,
            "lock_revision": lock.revision,
            "lock_content_sha256": lock.content_sha256,
            "prop_id": resolution.prop_id,
            "prop": resolution.prop,
            "prop_family": _prop_family(resolution.prop.kind),
            "tour": lock.context.tour,
            "round": lock.context.round,
            "player_a_id": lock.context.player_a_id,
            "player_b_id": lock.context.player_b_id,
            "scheduled_start_utc": lock.context.scheduled_start_utc,
            "information_cutoff_utc": lock.context.information_cutoff_utc,
            "framework_version": lock.framework_version,
            "settlement_policy_version": lock.settlement_policy.version,
            "source_manifest_sha256": lock.source_manifest.manifest_sha256,
            "data_hash": lock.match_parameters.snapshot.data_hash,
            "fit_hash": lock.match_parameters.snapshot_id,
            "config_hash": lock.match_parameters.snapshot.config_hash,
            "code_commit": lock.code.commit,
            "code_dirty": lock.code.dirty,
            "code_diff_sha256": lock.code.diff_sha256,
            "scenario_id": lock.information.scenario_id,
            "simulation_paths": lock.simulation.actual_paths,
            "settled_paths": 0 if forecast is None else forecast.settled_paths,
            "rng_seed_id": lock.simulation.seed_id,
            "mc_standard_error": (
                0.0 if forecast is None or market_override else forecast.mc_standard_error
            ),
            "probability_raw": probability_raw,
            "probability_settled": probability_raw,
            "probability_submitted": submitted_integer,
            "data_grade": "C" if forecast is None else forecast.data_grade,
            "resolution_status": resolution.state,
            "outcome_binary": binary,
            "official_value": official_value,
            "official_source_id": outcome.official_source_id,
            "official_source_sha256": outcome.official_source_sha256,
            "official_source_locator": outcome.official_source_locator,
            "resolved_at_utc": outcome.available_at_utc,
            "retrieved_at_utc": outcome.retrieved_at_utc,
            "brier_raw_model": brier_raw,
            "brier_submitted": brier_submitted,
            "quantization_loss": quantization_loss,
            "submission_rounding_policy_version": rounding_policy,
            "support_status": resolution.support_status,
            "support_reason_code": resolution.unavailable_reason,
            "match_retired": outcome.retired_player_id is not None,
            "policy_flags": lock.warnings
            + (("MATCH_WIN_PINNACLE_NO_VIG_V1",) if market_override else ())
            + (
                ()
                if forecast is None or forecast.policy_issue is None
                else (forecast.policy_issue,)
            ),
            "b6_c6_provenance": b6_c6_provenance,
        }
        provisional = CalibrationLedgerRow.model_construct(row_id="0" * 64, **payload)
        excluded = (
            {"row_id", "b6_c6_provenance"}
            if row_schema_version == LEDGER_SCHEMA_VERSION
            else {"row_id"}
        )
        row_id = sha256_json(provisional.model_dump(mode="json", exclude=excluded))
        rows.append(CalibrationLedgerRow(row_id=row_id, **payload))
    return tuple(rows)


class LedgerChainVerification(FrozenModel):
    schema_version: Literal["ledger-chain-verification/v1"] = "ledger-chain-verification/v1"
    rows: int = Field(ge=0)
    genesis_sha256: str | None
    terminal_sha256: str | None
    correction_rows: int = Field(ge=0)
    run_ids: tuple[str, ...]


class CalibrationLedger:
    """SQLite-backed ledger whose entries cannot be updated or deleted through SQL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calibration_entries (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                row_id TEXT NOT NULL UNIQUE,
                previous_entry_sha256 TEXT,
                entry_sha256 TEXT NOT NULL UNIQUE,
                row_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS calibration_entries_no_update
            BEFORE UPDATE ON calibration_entries
            BEGIN
                SELECT RAISE(ABORT, 'calibration ledger entries are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS calibration_entries_no_delete
            BEFORE DELETE ON calibration_entries
            BEGIN
                SELECT RAISE(ABORT, 'calibration ledger entries are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS ledger_metadata_no_update
            BEFORE UPDATE ON ledger_metadata
            BEGIN
                SELECT RAISE(ABORT, 'calibration ledger metadata is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS ledger_metadata_no_delete
            BEFORE DELETE ON ledger_metadata
            BEGIN
                SELECT RAISE(ABORT, 'calibration ledger metadata is immutable');
            END;
            """
        )
        observed = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if observed is None:
            connection.execute(
                "INSERT INTO ledger_metadata(key, value) VALUES('schema_version', ?)",
                (LEDGER_SCHEMA_VERSION,),
            )
        elif observed[0] != LEDGER_SCHEMA_VERSION:
            raise LedgerError(f"unsupported calibration ledger schema: {observed[0]}")
        connection.commit()

    def read(self) -> tuple[CalibrationLedgerRow, ...]:
        if not self.path.exists():
            return ()
        rows: list[CalibrationLedgerRow] = []
        previous: str | None = None
        try:
            with self._connect() as connection:
                self._ensure_schema(connection)
                records = connection.execute(
                    """
                    SELECT sequence, row_id, previous_entry_sha256, entry_sha256, row_json
                    FROM calibration_entries ORDER BY sequence
                    """
                ).fetchall()
                seen_rows: dict[str, CalibrationLedgerRow] = {}
                for expected_sequence, (
                    sequence,
                    row_id,
                    prior_hash,
                    entry_hash,
                    row_json,
                ) in enumerate(records, start=1):
                    if sequence != expected_sequence:
                        raise LedgerError(
                            f"ledger sequence is discontinuous at {sequence}; "
                            f"expected {expected_sequence}"
                        )
                    if prior_hash != previous:
                        raise LedgerError(f"ledger chain is broken at sequence {sequence}")
                    row = CalibrationLedgerRow.model_validate_json(row_json)
                    if row.row_id != row_id:
                        raise LedgerError(f"row identity differs at sequence {sequence}")
                    expected = sha256_json(
                        {
                            "previous_entry_sha256": previous,
                            "row": row.model_dump(mode="json"),
                        }
                    )
                    if expected != entry_hash:
                        raise LedgerError(f"ledger entry was modified at sequence {sequence}")
                    if row.correction_of_row_id is not None:
                        original = seen_rows.get(row.correction_of_row_id)
                        if original is None:
                            raise LedgerError(
                                f"correction target is absent or not prior at sequence {sequence}"
                            )
                        if (row.match_id, row.prop_id) != (
                            original.match_id,
                            original.prop_id,
                        ):
                            raise LedgerError(
                                "correction identity differs from its target at "
                                f"sequence {sequence}"
                            )
                    previous = entry_hash
                    rows.append(row)
                    seen_rows[row.row_id] = row
        except LedgerError:
            raise
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise LedgerError(f"cannot read calibration ledger: {exc}") from exc
        return tuple(rows)

    def verify_chain(
        self,
        *,
        expected_terminal_sha256: str | None = None,
        expected_run_id: str | None = None,
    ) -> LedgerChainVerification:
        """Verify the entire chain and return the digest that reports must anchor."""

        rows = self.read()
        if not self.path.exists():
            terminal = None
        else:
            try:
                with self._connect() as connection:
                    self._ensure_schema(connection)
                    record = connection.execute(
                        "SELECT entry_sha256 FROM calibration_entries "
                        "ORDER BY sequence DESC LIMIT 1"
                    ).fetchone()
                    terminal = None if record is None else str(record[0])
            except sqlite3.Error as exc:
                raise LedgerError(f"cannot read terminal ledger digest: {exc}") from exc
        if expected_terminal_sha256 is not None and terminal != expected_terminal_sha256:
            raise LedgerError(
                "terminal ledger digest differs from the backtest run manifest: "
                f"expected {expected_terminal_sha256}, observed {terminal}"
            )
        run_ids = tuple(sorted({row.backtest_run_id for row in rows if row.backtest_run_id}))
        if expected_run_id is not None and expected_run_id not in run_ids:
            raise LedgerError(f"ledger has no rows linked to expected run {expected_run_id}")
        return LedgerChainVerification(
            rows=len(rows),
            genesis_sha256=(
                None
                if not rows
                else sha256_json(
                    {"schema_version": LEDGER_SCHEMA_VERSION, "previous_entry_sha256": None}
                )
            ),
            terminal_sha256=terminal,
            correction_rows=sum(row.correction_of_row_id is not None for row in rows),
            run_ids=run_ids,
        )

    def effective_rows(
        self,
        *,
        correction_policy: Literal["latest-appended/v1"] = "latest-appended/v1",
    ) -> tuple[CalibrationLedgerRow, ...]:
        """Return verified effective rows under an explicit append-only correction policy."""

        if correction_policy != "latest-appended/v1":
            raise LedgerError("unsupported ledger correction policy")
        rows = self.read()
        root_by_row: dict[str, str] = {}
        active_by_root: dict[str, CalibrationLedgerRow] = {}
        for row in rows:
            if row.correction_of_row_id is None:
                root = row.row_id
            else:
                try:
                    root = root_by_row[row.correction_of_row_id]
                except KeyError as exc:
                    raise LedgerError(f"correction lineage is absent for row {row.row_id}") from exc
            root_by_row[row.row_id] = root
            active_by_root[root] = row
        active_ids = {row.row_id for row in active_by_root.values()}
        return tuple(row for row in rows if row.row_id in active_ids)

    def append(self, row: CalibrationLedgerRow) -> str:
        try:
            with self._connect() as connection:
                self._ensure_schema(connection)
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM calibration_entries WHERE row_id = ?", (row.row_id,)
                ).fetchone():
                    raise LedgerError("ledger row already exists; rows cannot be overwritten")
                last = connection.execute(
                    "SELECT entry_sha256 FROM calibration_entries ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                previous = None if last is None else str(last[0])
                content = {
                    "previous_entry_sha256": previous,
                    "row": row.model_dump(mode="json"),
                }
                entry_hash = sha256_json(content)
                row_json = canonical_json_bytes(row.model_dump(mode="json")).decode("utf-8")
                connection.execute(
                    """
                    INSERT INTO calibration_entries(
                        row_id, previous_entry_sha256, entry_sha256, row_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (row.row_id, previous, entry_hash, row_json),
                )
                connection.commit()
                return entry_hash
        except LedgerError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"cannot append calibration ledger row: {exc}") from exc

    def append_correction(
        self,
        original_row_id: str,
        corrected_row: CalibrationLedgerRow,
        *,
        reason: str,
    ) -> CalibrationLedgerRow:
        rows = self.read()
        original = next((row for row in rows if row.row_id == original_row_id), None)
        if original is None:
            raise LedgerError("correction target is absent from the ledger")
        if not reason.strip():
            raise LedgerError("correction reason must not be empty")
        if (corrected_row.match_id, corrected_row.prop_id) != (
            original.match_id,
            original.prop_id,
        ):
            raise LedgerError("correction must preserve match and prop identity")
        provisional = corrected_row.model_copy(
            update={
                "row_id": "0" * 64,
                "correction_of_row_id": original_row_id,
                "correction_reason": reason.strip(),
            }
        )
        excluded = (
            {"row_id", "b6_c6_provenance"}
            if provisional.schema_version == LEDGER_SCHEMA_VERSION
            else {"row_id"}
        )
        row_id = sha256_json(provisional.model_dump(mode="json", exclude=excluded))
        correction = CalibrationLedgerRow.model_validate(
            provisional.model_dump(mode="python") | {"row_id": row_id}
        )
        self.append(correction)
        return correction
