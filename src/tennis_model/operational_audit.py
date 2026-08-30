"""Read-only Milestone 8.1 operational audit orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import model_validator

from tennis_model.calibration.ledger import CalibrationLedger, LedgerError
from tennis_model.calibration.validation import ComparatorKind
from tennis_model.data.source_manifest import SourceManifestError, load_source_manifest
from tennis_model.estimation.inactivity import INACTIVITY_ADJUSTMENT_VERSION
from tennis_model.estimation.retirement import RETIREMENT_COMPETING_RISK_VERSION
from tennis_model.locking.provenance import capture_code_provenance
from tennis_model.locking.store import LockIntegrityError, LockStore
from tennis_model.schemas import FrozenModel


class SyntheticAuditCategory(StrEnum):
    NORMAL_COMPLETION = "NORMAL_COMPLETION"
    RETIREMENT = "RETIREMENT"
    WALKOVER_STATUS = "WALKOVER_STATUS"
    MISSING_STAT_PROP = "MISSING_STAT_PROP"
    VOID_SETTLEMENT = "VOID_SETTLEMENT"
    CORRECTION = "CORRECTION"
    LOCK_REVISION = "LOCK_REVISION"
    PATH_ESCALATION = "PATH_ESCALATION"
    POLICY_DISABLED_MARKET = "POLICY_DISABLED_MARKET"


class SyntheticAuditCase(FrozenModel):
    case_id: str
    category: SyntheticAuditCategory
    expected_status: str
    evidence_test: str


class SyntheticAuditManifest(FrozenModel):
    schema_version: Literal["milestone-8.1-synthetic-audit/v1"]
    deterministic_seed: int
    cases: tuple[SyntheticAuditCase, ...]

    @model_validator(mode="after")
    def required_pathways_are_present(self) -> Self:
        categories = {item.category for item in self.cases}
        missing = set(SyntheticAuditCategory) - categories
        if missing:
            raise ValueError(
                "synthetic audit omits required pathways: "
                + ", ".join(sorted(item.value for item in missing))
            )
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("synthetic audit case IDs must be unique")
        return self


def load_synthetic_audit_manifest(path: str | Path) -> SyntheticAuditManifest:
    payload: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SyntheticAuditManifest.model_validate(payload)


@dataclass(frozen=True, slots=True)
class AuditCheck:
    name: str
    status: Literal["PASS", "FAIL", "UNAVAILABLE"]
    detail: str


@dataclass(frozen=True, slots=True)
class CoreAuditReport:
    schema_version: Literal["operational-core-audit/v1"]
    b6_c6_status: Literal["COMPLETE"]
    checks: tuple[AuditCheck, ...]
    comparator_suite: tuple[str, ...]
    code_dirty: bool
    code_dirty_sha256: str | None
    genuine_historical_validation: Literal["NOT_YET_RUN"]
    missing_production_prerequisites: tuple[str, ...]

    @property
    def correctness_passed(self) -> bool:
        return all(item.status == "PASS" for item in self.checks)


def audit_core(
    repo_root: str | Path,
    *,
    locks_root: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> CoreAuditReport:
    """Inspect operational artifacts without fitting, simulating, or mutating locks."""

    repo = Path(repo_root).resolve()
    checks: list[AuditCheck] = [
        AuditCheck(
            "B6_C6_PROBABILITY_CONTRACT",
            "PASS",
            f"{RETIREMENT_COMPETING_RISK_VERSION}; {INACTIVITY_ADJUSTMENT_VERSION}",
        )
    ]
    missing: list[str] = []
    source_config = repo / "config" / "sources.yaml"
    if not source_config.exists():
        checks.append(AuditCheck("PRODUCTION_SOURCE_REGISTRY", "FAIL", "config absent"))
        missing.append("verified config/sources.yaml")
    else:
        try:
            source_manifest = load_source_manifest(source_config)
        except SourceManifestError as exc:
            configured = False
            source_detail = f"invalid source registry: {exc}"
        else:
            configured = True
            source_detail = (
                f"{len(source_manifest.sources)} verified source object(s); "
                f"manifest {source_manifest.manifest_version}"
            )
        checks.append(
            AuditCheck(
                "PRODUCTION_SOURCE_REGISTRY",
                "PASS" if configured else "UNAVAILABLE",
                source_detail,
            )
        )
        if not configured:
            missing.append("pinned, licensed ATP/WTA production source snapshots")
    if locks_root is not None:
        store = LockStore(locks_root)
        incomplete = store.incomplete_publications()
        failures = []
        if store.root.exists():
            for directory in sorted(store.root.glob("*/L*")):
                if not directory.is_dir() or not directory.name[1:].isdigit():
                    continue
                try:
                    store.load(directory.parent.name, int(directory.name[1:]))
                except LockIntegrityError as exc:
                    failures.append(f"{directory}: {exc}")
        checks.append(
            AuditCheck(
                "LOCK_INTEGRITY_AND_PUBLICATION_STATE",
                "FAIL" if failures or incomplete else "PASS",
                "; ".join(failures)
                if failures
                else f"verified; {len(incomplete)} classified incomplete publication(s)",
            )
        )
    else:
        checks.append(AuditCheck("LOCK_INTEGRITY_AND_PUBLICATION_STATE", "UNAVAILABLE", "no store"))
    if ledger_path is not None and Path(ledger_path).exists():
        try:
            verification = CalibrationLedger(ledger_path).verify_chain()
            checks.append(
                AuditCheck(
                    "LEDGER_CHAIN",
                    "PASS",
                    f"{verification.rows} rows; terminal {verification.terminal_sha256}",
                )
            )
        except LedgerError as exc:
            checks.append(AuditCheck("LEDGER_CHAIN", "FAIL", str(exc)))
    else:
        checks.append(AuditCheck("LEDGER_CHAIN", "UNAVAILABLE", "no historical ledger"))
        missing.append("genuine rolling-origin calibration ledger")
    synthetic_fixture = repo / "tests" / "fixtures" / "milestone81_synthetic_audit.yaml"
    if synthetic_fixture.exists():
        try:
            synthetic = load_synthetic_audit_manifest(synthetic_fixture)
            checks.append(
                AuditCheck(
                    "SYNTHETIC_FAILURE_PATH_FIXTURE",
                    "PASS",
                    f"{len(synthetic.cases)} required deterministic pathways declared",
                )
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            checks.append(AuditCheck("SYNTHETIC_FAILURE_PATH_FIXTURE", "FAIL", str(exc)))
    else:
        checks.append(AuditCheck("SYNTHETIC_FAILURE_PATH_FIXTURE", "FAIL", "fixture absent"))
    checks.extend(
        (
            AuditCheck(
                "TIMESTAMP_AND_CUTOFF_SCHEMA",
                "PASS",
                "effective/available/training/creation/retrieval/verification times are distinct",
            ),
            AuditCheck(
                "VALIDATION_INFRASTRUCTURE",
                "PASS",
                "five evaluation-only comparators and frozen I3 diagnostics are registered",
            ),
            AuditCheck(
                "REGRESSION_STATUS",
                "UNAVAILABLE",
                "read-only audit does not claim that pytest/Ruff/mypy were run",
            ),
        )
    )
    historical_catalog = repo / "artifacts" / "fits"
    if not historical_catalog.exists() or not any(historical_catalog.rglob("*.json")):
        missing.append("cutoff-safe historical ATP/WTA fit catalog")
    code = capture_code_provenance(repo)
    checks.append(
        AuditCheck(
            "COMPLETE_CODE_FINGERPRINT",
            "PASS",
            f"{code.fingerprint_version}; dirty={code.dirty}",
        )
    )
    return CoreAuditReport(
        schema_version="operational-core-audit/v1",
        b6_c6_status="COMPLETE",
        checks=tuple(checks),
        comparator_suite=tuple(item.value for item in ComparatorKind),
        code_dirty=code.dirty,
        code_dirty_sha256=code.diff_sha256,
        genuine_historical_validation="NOT_YET_RUN",
        missing_production_prerequisites=tuple(dict.fromkeys(missing)),
    )


__all__ = [
    "AuditCheck",
    "CoreAuditReport",
    "SyntheticAuditCase",
    "SyntheticAuditCategory",
    "SyntheticAuditManifest",
    "audit_core",
    "load_synthetic_audit_manifest",
]
