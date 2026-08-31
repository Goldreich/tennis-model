"""Reissue current exact-date eligibility metadata and create US Open lock 2117.

This command never refits a probability model.  It content-verifies the
already-fitted current snapshots, amends only B6 eligibility/provenance
metadata, and runs the versioned anytime-valid adaptive production policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from tennis_model.data.source_manifest import load_source_manifest
from tennis_model.estimation.artifacts import load_fit_artifact
from tennis_model.estimation.retirement import (
    RetirementSourceCoverage,
    load_retirement_fit_artifact,
    reissue_retirement_fit_eligibility_metadata,
    write_retirement_fit_artifact,
)
from tennis_model.estimation.snapshot import ModelSnapshot, create_model_snapshot
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking import (
    ADAPTIVE_MC_CS_V1_POLICY,
    HistoricalTrainingEligibilityProvenance,
    InformationBundle,
    LockStore,
    ReplayLevel,
    RetainedArtifactRecord,
    SourceManifestProvenance,
    TrainingInputCoverageRecord,
    capture_code_provenance,
    create_prediction_lock,
    reproduce_prediction_lock,
)
from tennis_model.props import CANONICAL_SETTLEMENT_POLICY
from tennis_model.schemas import Tour
from tennis_model.simulation import MATCH_WIN
from tennis_model.simulation.parameters import MatchContext

_RUN_ID = "2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664"
_PEGULA_ID = "player_d208168d-af45-58e6-b9d6-5fee97fffa3d"
_COMPONENT_ORDER = ("F", "A", "Q1", "D", "Q2")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    path.write_bytes(payload)


def _code_archive(repo: Path, output: Path) -> Path:
    result = subprocess.run(
        ("git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"),
        check=True,
        capture_output=True,
        text=True,
    )
    admitted = (
        "src/",
        "scripts/",
        "tests/",
        "config/",
        "docs/",
    )
    files = tuple(
        relative
        for relative in sorted(result.stdout.splitlines())
        if relative.startswith(admitted)
        or relative in {"AGENTS.md", "README.md", "pyproject.toml", "uv.lock"}
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=output, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (repo / relative).read_bytes())
        digest = _sha256_file(temporary_path)
        target = output / f"working-tree-{digest}.zip"
        if target.exists():
            if _sha256_file(target) != digest:
                raise RuntimeError("existing code archive failed its content hash")
        else:
            temporary_path.replace(target)
        return target
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _evidence_hashes(
    repo: Path, manifest: Any, tour: Tour
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_hashes = {
        source.sha256
        for source in manifest.sources
        if source.tour is tour
        and (
            source.source_id.endswith(tuple(str(year) for year in range(2021, 2026)))
            or source.source_id.endswith("current")
        )
    }
    crosswalk_hashes: set[str] = set()
    crosswalk_root = repo / "data/processed/retrospective-finalized-crosswalk-v1"
    for year in range(2021, 2026):
        stem = f"{tour.value.lower()}_{year}"
        detail = crosswalk_root / f"crosswalk_{stem}.csv"
        receipt = crosswalk_root / f"manifest_{stem}.json"
        crosswalk_hashes.update((_sha256_file(detail), _sha256_file(receipt)))
        manifest_payload = json.loads(receipt.read_bytes())
        source_hashes.add(manifest_payload["augmentation_source"]["sha256"])
    return tuple(sorted(source_hashes)), tuple(sorted(crosswalk_hashes))


def _statistical_payload_sha256(artifact: Any) -> str:
    payload = artifact.model_dump(
        mode="json",
        exclude={"artifact_id", "schema_version", "source_coverage", "production_eligible"},
    )
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _training_provenance(
    repo: Path,
    base: Path,
    report: dict[str, Any],
    manifest: Any,
    snapshot: ModelSnapshot,
    retirement: Any,
    run_id: str,
) -> HistoricalTrainingEligibilityProvenance:
    tour = snapshot.tour
    source_hashes, crosswalk_hashes = _evidence_hashes(repo, manifest, tour)
    undated_b6_matches = sum(
        count
        for key, count in report["exact_date_exclusions"].items()
        if key.startswith(f"{tour.value}:")
    )
    undated_serve_matches = sum(
        count
        for key, count in report["exact_date_exclusions"].items()
        if key.startswith(f"{tour.value}:") and int(key.split(":")[1]) >= 2023
    )
    serve_undated_rows = 2 * undated_serve_matches
    fit_by_component = {
        reference.component.value: load_fit_artifact(reference.directory).fit
        for reference in snapshot.component_artifacts
    }
    records = tuple(
        TrainingInputCoverageRecord(
            component=component,  # type: ignore[arg-type]
            row_unit="component_rows",
            included_exact_dated_rows=fit_by_component[component].diagnostics.usable_rows,
            excluded_undated_candidate_rows=serve_undated_rows,
            exclusion_rate=(
                serve_undated_rows
                / (fit_by_component[component].diagnostics.usable_rows + serve_undated_rows)
            ),
            date_fallback_rows=fit_by_component[component].diagnostics.date_fallback_rows,
        )
        for component in _COMPONENT_ORDER
    )
    included_matches = sum(item.match_count for item in retirement.included_counts)
    included_player_starts = sum(item.player_start_count for item in retirement.included_counts)
    b6 = TrainingInputCoverageRecord(
        component="B6",
        row_unit="matches",
        included_exact_dated_rows=included_matches,
        excluded_undated_candidate_rows=undated_b6_matches,
        exclusion_rate=undated_b6_matches / (included_matches + undated_b6_matches),
        included_unweighted_player_starts=included_player_starts,
        included_weighted_player_starts=retirement.tour_starts_n,
        excluded_undated_player_starts=2 * undated_b6_matches,
    )
    pin = SourceManifestProvenance.from_manifest(manifest)
    return HistoricalTrainingEligibilityProvenance(
        tour=tour,
        assertion_id=f"current-exact-dated-fit-inputs-{run_id[:16]}-{tour.value.lower()}",
        verified_at_utc=snapshot.data_cutoff_utc - timedelta(microseconds=1),
        all_included_rows_have_verified_exact_dates=True,
        historical_exact_date_coverage_complete=False,
        source_manifest_sha256=pin.manifest_sha256,
        source_sha256s=source_hashes,
        crosswalk_sha256s=crosswalk_hashes,
        records=(*records, b6),
        warning="HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE",
    )


def _retained_record(kind: str, path: Path) -> RetainedArtifactRecord:
    digest = _hash_path(path)
    return RetainedArtifactRecord(
        kind=kind,  # type: ignore[arg-type]
        artifact_id=f"{kind}:{digest}",
        path=str(path.resolve()),
        sha256=digest,
    )


def run(
    repo: Path,
    *,
    run_id: str = _RUN_ID,
    operational_name: str = "official-2117-v1",
    prepare_only: bool = False,
) -> Path:
    base = repo / "artifacts/current-usopen-2026" / run_id
    operational = repo / "artifacts/live-usopen-2026" / operational_name
    retained = operational / "retained"
    report = json.loads((base / "build_report.json").read_bytes())
    manifest = load_source_manifest(base / "source_manifest.yaml")
    code_before = capture_code_provenance(repo)

    snapshots: dict[Tour, ModelSnapshot] = {}
    eligibility: dict[Tour, HistoricalTrainingEligibilityProvenance] = {}
    amendment_report: dict[str, Any] = {}
    for tour in Tour:
        original_snapshot = ModelSnapshot.from_json(
            (base / f"model_snapshot_{tour.value.lower()}.json").read_bytes()
        )
        if original_snapshot.retirement_artifact is None:
            raise RuntimeError("current snapshot has no B6 artifact")
        original = load_retirement_fit_artifact(
            original_snapshot.retirement_artifact.directory
        ).artifact
        provenance = _training_provenance(
            repo, base, report, manifest, original_snapshot, original, run_id
        )
        b6_record = provenance.records[-1]
        coverage = RetirementSourceCoverage(
            tour=tour,
            complete=False,
            assertion_id=provenance.assertion_id,
            verified_at_utc=provenance.verified_at_utc,
            details=(
                "Every B6 row admitted to the current time-weighted fit has a verified exact "
                "match date. Undated rows remain excluded; global historical exact-date "
                "coverage is incomplete and cannot prove a no-history player state."
            ),
            fit_input_date_eligibility_verified=True,
            historical_exact_date_coverage_complete=False,
            included_exact_dated_matches=b6_record.included_exact_dated_rows,
            excluded_undated_matches=b6_record.excluded_undated_candidate_rows,
            included_exact_dated_player_starts=b6_record.included_unweighted_player_starts,
            excluded_undated_player_starts=b6_record.excluded_undated_player_starts or 0,
            source_sha256s=provenance.source_sha256s,
            crosswalk_sha256s=provenance.crosswalk_sha256s,
            eligibility_rule_version="exact-dated-fit-inputs/v1",
        )
        amended = reissue_retirement_fit_eligibility_metadata(original, coverage)
        if _statistical_payload_sha256(original) != _statistical_payload_sha256(amended):
            raise RuntimeError("B6 metadata amendment changed fitted probability fields")
        persisted = write_retirement_fit_artifact(amended, operational / "retirement_fits")
        fits = {
            reference.component: load_fit_artifact(reference.directory)
            for reference in original_snapshot.component_artifacts
        }
        if original_snapshot.inactivity_configuration is None:
            raise RuntimeError("current snapshot has no C6 configuration")
        snapshot = create_model_snapshot(
            fits,
            retirement_artifact=persisted,
            inactivity_configuration=original_snapshot.inactivity_configuration,
        )
        _write_immutable(
            operational / f"model_snapshot_{tour.value.lower()}.json",
            snapshot.canonical_json().encode(),
        )
        _write_immutable(
            operational / f"training_eligibility_{tour.value.lower()}.json",
            _canonical_bytes(provenance.model_dump(mode="json")),
        )
        snapshots[tour] = snapshot
        eligibility[tour] = provenance
        amendment_report[tour.value] = {
            "original_snapshot_id": original_snapshot.snapshot_id,
            "amended_snapshot_id": snapshot.snapshot_id,
            "original_retirement_artifact_id": original.artifact_id,
            "amended_retirement_artifact_id": amended.artifact_id,
            "probability_fields_unchanged": True,
            "b6_production_eligible": amended.production_eligible,
            "weighted_player_starts": amended.tour_starts_n,
            "training_eligibility": provenance.model_dump(mode="json"),
        }

    if prepare_only:
        result_path = operational / "snapshot_preparation_report.json"
        _write_immutable(
            result_path,
            _canonical_bytes(
                {
                    "schema_version": "current-operational-snapshot-preparation/v1",
                    "base_run_id": run_id,
                    "official_capture_id": report["official_capture_id"],
                    "methodology_changed": False,
                    "refit_performed": False,
                    "artifact_amendments": amendment_report,
                }
            ),
        )
        return result_path

    snapshot = snapshots[Tour.WTA]
    provenance = eligibility[Tour.WTA]
    information = InformationBundle.model_validate_json(
        (base / "lock/information_bundle.json").read_bytes()
    )
    canonical_match = CanonicalMatchIdentity.model_validate_json(
        (base / "lock/canonical_match_identity.json").read_bytes()
    )
    attempt = json.loads((base / "lock/lock_attempt.json").read_bytes())
    context = MatchContext.model_validate(attempt["context"])

    source_receipt = retained / "source_snapshot_receipt.json"
    _write_immutable(
        source_receipt,
        _canonical_bytes(
            {
                "source_manifest_sha256": provenance.source_manifest_sha256,
                "source_sha256s": provenance.source_sha256s,
                "crosswalk_sha256s": provenance.crosswalk_sha256s,
                "official_capture_id": report["official_capture_id"],
            }
        ),
    )
    counts_receipt = retained / "component_counts_receipt.json"
    _write_immutable(
        counts_receipt,
        _canonical_bytes(
            {
                "component_count_artifact_hash": snapshot.component_count_artifact_hash,
                "current_component_counts_sha256": _sha256_file(
                    base / "data/current_component_counts.parquet"
                ),
                "records": [item.model_dump(mode="json") for item in provenance.records],
            }
        ),
    )
    inactivity_path = retained / "inactivity_config.json"
    assert snapshot.inactivity_configuration is not None
    _write_immutable(
        inactivity_path,
        _canonical_bytes(snapshot.inactivity_configuration.model_dump(mode="json")),
    )
    settlement_path = retained / "settlement_policy.json"
    _write_immutable(settlement_path, _canonical_bytes(asdict(CANONICAL_SETTLEMENT_POLICY)))
    code_archive = _code_archive(repo, retained / "code")
    code = capture_code_provenance(repo)
    if code != code_before:
        raise RuntimeError("code provenance changed while preparing ignored lock artifacts")

    wta_fit_root = base / "fits/wta"
    b6_path = snapshot.retirement_artifact.directory if snapshot.retirement_artifact else None
    if b6_path is None:
        raise RuntimeError("amended snapshot lost B6")
    retained_artifacts = (
        _retained_record("source_snapshot", source_receipt),
        _retained_record("normalized_snapshot", operational / "training_eligibility_wta.json"),
        _retained_record("component_counts", counts_receipt),
        _retained_record("component_fit", wta_fit_root),
        _retained_record("retirement_fit", b6_path),
        _retained_record("inactivity_config", inactivity_path),
        _retained_record("model_config", repo / "config/model_v1.yaml"),
        _retained_record("settlement_policy", settlement_path),
        _retained_record("code_archive", code_archive),
    )
    store = LockStore(operational / "locks")
    lock = create_prediction_lock(
        snapshot,
        context,
        information,
        (MATCH_WIN(_PEGULA_ID),),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=manifest,
        code=code,
        seed=20260830,
        store=store,
        execution_mode="production",
        path_count_policy=ADAPTIVE_MC_CS_V1_POLICY,
        allow_dirty=True,
        canonical_match_identity=canonical_match,
        retained_artifacts=retained_artifacts,
        training_eligibility=provenance,
    )
    verified_sha256 = store.verify(lock.base_lock_id, lock.revision)
    replay = reproduce_prediction_lock(lock, replay_level=ReplayLevel.SAME_RUNTIME_EXACT)
    result = {
        "schema_version": "current-live-lock-remediation/v1",
        "official_match_id": "2117",
        "match": "Jessica Pegula vs Elena-Gabriela Ruse",
        "historical_validation_mode": lock.historical_validation_policy.mode.value,
        "methodology_changed": False,
        "refit_performed": False,
        "artifact_amendments": amendment_report,
        "lock_id": lock.lock_id,
        "revision": lock.revision,
        "content_sha256": lock.content_sha256,
        "verified_sha256": verified_sha256,
        "actual_paths": lock.simulation.actual_paths,
        "escalated": lock.simulation.escalated,
        "escalation_reasons": lock.simulation.escalation_reasons,
        "same_runtime_exact_replay": asdict(replay),
        "c6": [item.model_dump(mode="json") for item in lock.match_parameters.inactivity.records],
        "warnings": lock.warnings,
        "card_path": str(store.revision_directory(lock.base_lock_id, lock.revision) / "card.md"),
    }
    result_path = operational / "operational_report.json"
    _write_immutable(result_path, _canonical_bytes(result))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=_RUN_ID)
    parser.add_argument("--operational-name", default="official-2117-v1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if len(args.run_id) != 64 or any(char not in "0123456789abcdef" for char in args.run_id):
        parser.error("--run-id must be 64 lowercase hex characters")
    if (
        not args.operational_name.strip()
        or Path(args.operational_name).name != args.operational_name
    ):
        parser.error("--operational-name must be one nonempty path component")
    print(
        run(
            args.repo.resolve(),
            run_id=args.run_id,
            operational_name=args.operational_name,
            prepare_only=args.prepare_only,
        )
    )


if __name__ == "__main__":
    main()
