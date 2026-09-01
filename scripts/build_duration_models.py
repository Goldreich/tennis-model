"""Build the frozen B5 duration upgrade for a retained current US Open build.

This command is deliberately offline.  It verifies and reuses the retained raw,
processed, crosswalk, current-US-Open, serve, B6, and C6 artifacts; it does not
download, refit, or overwrite any of them.  New duration artifacts, v3 snapshots,
and reports are published under a separate content-addressed build root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from tennis_model.data.artifacts import (
    ProcessedArtifactBundle,
    ProcessedArtifactManifest,
    load_processed_bundle,
    read_processed_table,
)
from tennis_model.data.snapshot import verify_snapshot
from tennis_model.data.source_manifest import load_source_manifest, manifest_sha256
from tennis_model.estimation.artifacts import load_fit_artifact
from tennis_model.estimation.duration_model import (
    UNRESOLVED_DURATION_DISPLAY_POLICY,
    DurationFitArtifact,
    DurationModelConfig,
    DurationPathExposure,
    DurationTrainingBatch,
    build_duration_training_batch,
    draw_duration,
    duration_predictor,
    duration_scale,
    fit_duration_model,
    load_duration_model_config,
    map_duration_parameters,
    prepare_duration_parameter_sampler,
    sample_prepared_duration_parameters,
    write_duration_fit_artifact,
)
from tennis_model.estimation.retirement import load_retirement_fit_artifact
from tennis_model.estimation.snapshot import ModelSnapshot, create_model_snapshot
from tennis_model.locking.provenance import capture_code_provenance
from tennis_model.schemas import RawSourceSnapshot, Tour

_SCHEMA_VERSION = "current-usopen-duration-build/v1"
_DATA_AUDIT_SCHEMA_VERSION = "duration-data-audit/v1"
_VALIDATION_SCHEMA_VERSION = "duration-validation-report/v2"
_PERFORMANCE_SCHEMA_VERSION = "duration-performance-report/v1"
_REJECTION_SCHEMA_VERSION = "duration-build-rejection/v1"
_CROSSWALK_SET_SHA256 = (
    "4f25376b139d5e4871cf7d9f304644431703179c7ec144e8cee9638a18548495"
)
_THRESHOLDS = (90, 120, 150)
_INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _validate_run_id(value: object, *, field_name: str = "run_id") -> str:
    run_id = str(value)
    if len(run_id) != 64 or any(character not in "0123456789abcdef" for character in run_id):
        raise RuntimeError(f"{field_name} must be a lowercase SHA-256 digest")
    return run_id


def _load_duration_build_rejections(output_root: Path) -> dict[str, dict[str, Any]]:
    directory = output_root.resolve() / "rejected-runs"
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"duration rejection registry is not a regular directory: {directory}")
    receipts: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"duration rejection receipt is not a regular file: {path}")
        raw = path.read_bytes()
        try:
            receipt = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid duration rejection receipt: {path}") from exc
        if not isinstance(receipt, dict) or raw != _canonical_bytes(receipt):
            raise RuntimeError(f"duration rejection receipt is not canonical: {path}")
        run_id = _validate_run_id(receipt.get("run_id"))
        if (
            path.name != f"{run_id}.json"
            or receipt.get("schema_version") != _REJECTION_SCHEMA_VERSION
            or receipt.get("framework_version") != "v1.0"
            or receipt.get("status") != "REJECTED"
            or receipt.get("replacement_run_id") is not None
        ):
            raise RuntimeError(f"duration rejection receipt is inconsistent: {path}")
        receipts[run_id] = receipt
    return receipts


def assert_duration_build_not_rejected(build_directory: Path, output_root: Path) -> str:
    """Return a duration run identity only when no immutable rejection exists."""

    report_path = build_directory.resolve() / "build_report.json"
    try:
        report = json.loads(report_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"duration build report is unreadable: {report_path}") from exc
    run_id = _validate_run_id(report.get("run_id"))
    if build_directory.name != run_id[:32]:
        raise RuntimeError("duration build directory and run identity differ")
    if run_id in _load_duration_build_rejections(output_root):
        raise RuntimeError(f"duration build is explicitly rejected: {run_id}")
    return run_id


def select_latest_ready_duration_build(output_root: Path, base_build_prefix: str) -> Path:
    """Select the newest non-rejected READY duration build for one base build."""

    family = output_root.resolve() / base_build_prefix
    if family.is_symlink() or not family.is_dir():
        raise RuntimeError(f"duration build family is unavailable: {family}")
    rejected = _load_duration_build_rejections(output_root)
    candidates: list[tuple[datetime, str, Path]] = []
    for directory in family.iterdir():
        if directory.is_symlink() or not directory.is_dir() or directory.name.startswith("."):
            continue
        report_path = directory / "build_report.json"
        if not report_path.is_file() or report_path.is_symlink():
            continue
        try:
            report = json.loads(report_path.read_bytes())
            run_id = _validate_run_id(report.get("run_id"))
            fitted_at = _parse_utc(str(report.get("fitted_at_utc")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, argparse.ArgumentTypeError):
            continue
        if (
            report.get("schema_version") != _SCHEMA_VERSION
            or report.get("status") != "READY"
            or directory.name != run_id[:32]
            or run_id in rejected
        ):
            continue
        candidates.append((fitted_at, run_id, directory))
    if not candidates:
        raise RuntimeError(f"no non-rejected READY duration build exists under {family}")
    return max(candidates)[2]


def write_duration_build_rejection(
    output_root: Path,
    build_directory: Path,
    *,
    detected_at_utc: datetime,
) -> Path:
    """Publish an exclusive canonical rejection receipt without altering a run."""

    if detected_at_utc.tzinfo is None or detected_at_utc.utcoffset() is None:
        raise RuntimeError("rejection detected_at_utc must be timezone-aware")
    detected_at = detected_at_utc.astimezone(UTC)
    build_directory = build_directory.resolve()
    output_root = output_root.resolve()
    try:
        relative_build = build_directory.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise RuntimeError("rejected duration build must be inside its output root") from exc
    build_report_path = build_directory / "build_report.json"
    performance_path = build_directory / "performance_report.json"
    validation_path = build_directory / "validation_report.json"
    build_report = json.loads(build_report_path.read_bytes())
    performance = json.loads(performance_path.read_bytes())
    validation = json.loads(validation_path.read_bytes())
    run_id = _validate_run_id(build_report.get("run_id"))
    if build_directory.name != run_id[:32]:
        raise RuntimeError("rejected duration build directory and run identity differ")
    evidence_by_tour: dict[str, Any] = {}
    for tour in Tour:
        checkpoints = performance["by_tour"][tour.value]["checkpoints"]
        final = checkpoints[-1]
        fit = validation["by_tour"][tour.value]["fit"]
        evidence_by_tour[tour.value] = {
            "benchmark_paths": int(final["paths"]),
            "mean_latent_minutes": float(final["mean_minutes"]),
            "p90_latent_minutes": float(final["p90_minutes"]),
            "posterior_covariance_mode": str(fit["posterior_covariance_mode"]),
            "posterior_dimension": int(fit["posterior_dimension"]),
        }
    receipt = {
        "schema_version": _REJECTION_SCHEMA_VERSION,
        "framework_version": "v1.0",
        "status": "REJECTED",
        "run_id": run_id,
        "relative_build_path": relative_build,
        "detected_at_utc": detected_at.isoformat(),
        "reason_code": "PATHOLOGICAL_DURATION_PARAMETER_UNCERTAINTY",
        "reason": (
            "The stored diagonal L-BFGS covariance loses material parameter "
            "correlations and produces implausibly diffuse duration draws; this "
            "run is not production-usable despite its immutable internal READY label."
        ),
        "evidence": {
            "build_report_sha256": _sha256_file(build_report_path),
            "performance_report_sha256": _sha256_file(performance_path),
            "validation_report_sha256": _sha256_file(validation_path),
            "by_tour": evidence_by_tour,
        },
        "replacement_run_id": None,
        "methodology_change": False,
    }
    target = output_root / "rejected-runs" / f"{run_id}.json"
    _write_canonical(target, receipt)
    loaded = _load_duration_build_rejections(output_root)
    if loaded.get(run_id) != receipt:
        raise RuntimeError("published duration rejection receipt failed verification")
    return target


def _raw_snapshot(repo: Path, source: Any) -> RawSourceSnapshot:
    directory = repo / "data/raw" / source.tour.value.lower() / source.source_id / source.sha256
    payload = directory / "payload"
    snapshot = RawSourceSnapshot(
        source=source,
        payload_path=payload,
        provenance_path=directory / "source.json",
        size_bytes=payload.stat().st_size,
        sha256=source.sha256,
    )
    verify_snapshot(snapshot)
    return snapshot


def _load_full_processed_bundle(repo: Path, source: Any) -> ProcessedArtifactBundle:
    parent = (
        repo
        / "data/processed"
        / source.tour.value.lower()
        / source.source_id
        / source.sha256[:16]
    )
    candidates = sorted(parent.glob("*/manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one retained processed bundle for {source.source_id}; "
            f"found {len(candidates)}"
        )
    return load_processed_bundle(candidates[0].parent)


def _historical_raw_audit(repo: Path, source_manifest: Any) -> dict[str, Any]:
    """Audit all 18 pinned 2017--2025 raw CSVs before normalization."""

    frames: dict[Tour, list[pd.DataFrame]] = {tour: [] for tour in Tour}
    raw_pins: dict[Tour, list[dict[str, Any]]] = {tour: [] for tour in Tour}
    processed_totals: dict[Tour, Counter[str]] = {tour: Counter() for tour in Tour}
    processed_pins: dict[Tour, list[dict[str, Any]]] = {tour: [] for tour in Tour}
    for source in sorted(source_manifest.sources, key=lambda item: item.source_id):
        year_segment = source.source_id.rsplit("-", 1)[-1]
        if not year_segment.isdigit():
            continue
        year = int(year_segment)
        if not 2017 <= year <= 2025:
            continue
        snapshot = _raw_snapshot(repo, source)
        frame = pd.read_csv(snapshot.payload_path, low_memory=False)
        frame["_source_id"] = source.source_id
        frames[source.tour].append(frame)
        raw_pins[source.tour].append(
            {
                "source_id": source.source_id,
                "year": year,
                "payload_path": _relative(snapshot.payload_path, repo),
                "payload_sha256": source.sha256,
                "payload_size_bytes": snapshot.size_bytes,
                "source_record_path": _relative(snapshot.provenance_path, repo),
                "source_record_sha256": _sha256_file(snapshot.provenance_path),
            }
        )
        bundle = _load_full_processed_bundle(repo, source)
        manifest = bundle.manifest
        processed_totals[source.tour]["accepted_matches"] += manifest.accepted_match_count
        processed_totals[source.tour]["anomaly_rows"] += manifest.receipt_for(
            "anomalies"
        ).row_count
        processed_totals[source.tour]["cutoff_exclusion_rows"] += manifest.receipt_for(
            "cutoff_exclusions"
        ).row_count
        processed_pins[source.tour].append(
            {
                "source_id": source.source_id,
                "bundle_id": bundle.bundle_id,
                "bundle_path": _relative(bundle.directory, repo),
                "service_rows_sha256": manifest.receipt_for("service_rows").sha256,
                "anomalies_sha256": manifest.receipt_for("anomalies").sha256,
            }
        )

    result: dict[str, Any] = {}
    for tour in Tour:
        if len(frames[tour]) != 9:
            raise RuntimeError(f"full {tour.value} audit requires nine retained yearly sources")
        rows = pd.concat(frames[tour], ignore_index=True)
        duration = pd.to_numeric(rows["minutes"], errors="coerce")
        winner_points = pd.to_numeric(rows["w_svpt"], errors="coerce")
        loser_points = pd.to_numeric(rows["l_svpt"], errors="coerce")
        duration_valid = duration.gt(0)
        points_valid = (winner_points + loser_points).gt(0)
        dates = pd.to_datetime(
            rows["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
        )
        player_ids = pd.concat(
            (rows["winner_id"], rows["loser_id"]), ignore_index=True
        ).dropna()
        player_ids = player_ids.astype("string").str.strip()
        player_ids = player_ids.loc[player_ids.ne("")]
        level = rows["tourney_level"].astype("string").str.upper()
        event_key = (
            rows["tourney_name"]
            .astype("string")
            .str.casefold()
            .str.replace(r"[^a-z]", "", regex=True)
        )
        major = level.eq("G")
        us_open = major & event_key.eq("usopen")
        retirement = rows["score"].astype("string").str.contains(
            r"\bRET\b", case=False, na=False
        )
        processed = processed_totals[tour]
        candidate_count = len(rows)
        result[tour.value] = {
            "source_kind": "retained_sackmann_style_finalized_records",
            "duration_data_grade": "B",
            "candidate_matches": candidate_count,
            "matches_with_positive_official_duration": int(duration_valid.sum()),
            "matches_with_positive_total_points": int(points_valid.sum()),
            "matches_with_duration_and_total_points": int(
                (duration_valid & points_valid).sum()
            ),
            "date_coverage": {
                "first_tournament_start_date": None
                if dates.notna().sum() == 0
                else dates.min().date().isoformat(),
                "last_tournament_start_date": None
                if dates.notna().sum() == 0
                else dates.max().date().isoformat(),
                "missing_tournament_start_dates": int(dates.isna().sum()),
                "note": (
                    "Raw Sackmann dates are tournament-start dates, not fit-eligible "
                    "exact match dates."
                ),
            },
            "unique_players_with_source_id": int(player_ids.nunique()),
            "us_open_matches": int(us_open.sum()),
            "other_major_matches": int((major & ~us_open).sum()),
            "retirement_rows": int(retirement.sum()),
            "missing_or_invalid": {
                "duration_missing_or_nonpositive": int((~duration_valid).sum()),
                "total_points_missing_or_nonpositive": int((~points_valid).sum()),
                "duration_or_total_points_missing_or_nonpositive": int(
                    (~(duration_valid & points_valid)).sum()
                ),
                "score_missing": int(rows["score"].isna().sum()),
                "winner_or_loser_id_missing": int(
                    (rows["winner_id"].isna() | rows["loser_id"].isna()).sum()
                ),
            },
            "post_normalization": {
                "accepted_matches": processed["accepted_matches"],
                "excluded_raw_candidates": candidate_count - processed["accepted_matches"],
                "anomaly_rows": processed["anomaly_rows"],
                "cutoff_exclusion_rows": processed["cutoff_exclusion_rows"],
            },
            "raw_source_pins": raw_pins[tour],
            "processed_bundle_pins": processed_pins[tour],
        }
    return result


def _select_exact_date_bundles(
    repo: Path, cutoff: datetime
) -> dict[Tour, tuple[ProcessedArtifactBundle, ...]]:
    root = repo / "data/processed/current-usopen-2026-exact-date"
    selected: dict[Tour, list[ProcessedArtifactBundle]] = {tour: [] for tour in Tour}
    for manifest_path in sorted(root.glob("**/manifest.json")):
        manifest = ProcessedArtifactManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.information_cutoff_utc != cutoff:
            continue
        if (
            manifest.historical_validation_policy.exact_date_crosswalk_sha256
            != _CROSSWALK_SET_SHA256
        ):
            continue
        bundle = load_processed_bundle(manifest_path.parent)
        if bundle.manifest.exact_date_crosswalk_manifest is None:
            raise RuntimeError(f"selected bundle lacks exact-date provenance: {bundle.directory}")
        selected[bundle.manifest.source.tour].append(bundle)
    output: dict[Tour, tuple[ProcessedArtifactBundle, ...]] = {}
    for tour in Tour:
        bundles = tuple(sorted(selected[tour], key=lambda item: item.manifest.source.source_id))
        source_ids = tuple(item.manifest.source.source_id for item in bundles)
        if len(bundles) != 5 or len(set(source_ids)) != 5:
            raise RuntimeError(
                f"retained build cutoff requires five unique 2021--2025 {tour.value} bundles"
            )
        years = tuple(int(source_id.rsplit("-", 1)[1]) for source_id in source_ids)
        if years != (2021, 2022, 2023, 2024, 2025):
            raise RuntimeError(f"unexpected exact-date source years for {tour.value}: {years}")
        output[tour] = bundles
    return output


def _bundle_pin(bundle: ProcessedArtifactBundle, repo: Path) -> dict[str, Any]:
    manifest = bundle.manifest
    crosswalk = manifest.exact_date_crosswalk_manifest
    return {
        "source_id": manifest.source.source_id,
        "source_sha256": manifest.source.sha256,
        "bundle_id": bundle.bundle_id,
        "bundle_path": _relative(bundle.directory, repo),
        "information_cutoff_utc": manifest.information_cutoff_utc.isoformat(),
        "service_rows_sha256": manifest.receipt_for("service_rows").sha256,
        "service_row_count": manifest.receipt_for("service_rows").row_count,
        "crosswalk_id": None if crosswalk is None else crosswalk.crosswalk_id,
        "crosswalk_detail_sha256": None if crosswalk is None else crosswalk.detail_sha256,
        "augmentation_source_sha256": None
        if crosswalk is None
        else crosswalk.augmentation_source.sha256,
        "crosswalk_set_sha256": _CROSSWALK_SET_SHA256,
    }


def _current_match_audit(rows: pd.DataFrame) -> dict[str, Any]:
    candidate = 0
    duration_count = 0
    points_count = 0
    both_count = 0
    dates: list[Any] = []
    players: set[str] = set()
    retirements = 0
    for _match_id, group in rows.groupby("match_id", sort=True):
        candidate += 1
        duration = pd.to_numeric(group["duration_minutes"], errors="coerce")
        service_points = pd.to_numeric(group["service_points"], errors="coerce")
        duration_ok = len(duration) == 2 and duration.notna().all() and duration.gt(0).all()
        points_ok = len(service_points) == 2 and service_points.sum() > 0
        duration_count += int(duration_ok)
        points_count += int(points_ok)
        both_count += int(duration_ok and points_ok)
        dates.extend(item for item in group["match_date"] if not pd.isna(item))
        players.update(group["player_id"].dropna().astype(str))
        retirements += int(bool(group["retirement"].fillna(False).any()))
    return {
        "source_kind": "official_2026_us_open_completed_match_feeds",
        "authority": "official_authoritative",
        "candidate_matches": candidate,
        "matches_with_positive_official_duration": duration_count,
        "matches_with_positive_total_points": points_count,
        "matches_with_duration_and_total_points": both_count,
        "date_coverage": {
            "first_match_date": None if not dates else min(dates).isoformat(),
            "last_match_date": None if not dates else max(dates).isoformat(),
        },
        "unique_players": len(players),
        "retirements": retirements,
    }


def _prepare_fit_inputs(
    repo: Path,
    base_build: Path,
    cutoff: datetime,
    config: DurationModelConfig,
) -> tuple[
    dict[Tour, DurationTrainingBatch],
    dict[Tour, pd.DataFrame],
    dict[Tour, tuple[ProcessedArtifactBundle, ...]],
    dict[str, Any],
]:
    bundles = _select_exact_date_bundles(repo, cutoff)
    current_path = base_build / "data/current_service_rows.parquet"
    current_rows = pd.read_parquet(current_path)
    if current_rows.empty:
        raise RuntimeError("retained current service-row artifact is empty")
    if pd.to_datetime(current_rows["retrieved_at_utc"], utc=True).ge(cutoff).any():
        raise RuntimeError("current official row was not available strictly before build cutoff")

    batches: dict[Tour, DurationTrainingBatch] = {}
    combined_rows: dict[Tour, pd.DataFrame] = {}
    fit_audit: dict[str, Any] = {}
    for tour in Tour:
        historical_frames: list[pd.DataFrame] = []
        for bundle in bundles[tour]:
            frame = read_processed_table(bundle, "service_rows")
            existing = set(
                frame["exact_date_crosswalk_sha256"].dropna().astype(str).unique()
            )
            if existing and existing != {_CROSSWALK_SET_SHA256}:
                raise RuntimeError(
                    f"{bundle.manifest.source.source_id} has an unexpected crosswalk-set hash"
                )
            # Stamp the pinned aggregate crosswalk-set identity into the in-memory
            # fit rows.  The immutable processed bundles remain unchanged.
            frame["exact_date_crosswalk_sha256"] = _CROSSWALK_SET_SHA256
            historical_frames.append(frame)
        current = current_rows.loc[current_rows["tour"].eq(tour.value)].copy()
        rows = pd.concat((*historical_frames, current), ignore_index=True, sort=False)
        duplicated = rows.groupby("match_id").size().ne(2)
        if duplicated.any():
            raise RuntimeError(
                f"{tour.value} duration input has non-reciprocal or duplicated match IDs"
            )
        batch = build_duration_training_batch(
            rows,
            tour=tour,
            information_cutoff_utc=cutoff,
            config=config,
        )
        if not batch.observations:
            raise RuntimeError(f"{tour.value} duration batch contains no eligible observations")
        batches[tour] = batch
        combined_rows[tour] = rows
        current_ids = set(current["match_id"].astype(str))
        current_included = sum(item.match_id in current_ids for item in batch.observations)
        fit_audit[tour.value] = {
            "selected_exact_date_bundles": [
                _bundle_pin(bundle, repo) for bundle in bundles[tour]
            ],
            "combined_input_rows": len(rows),
            "combined_candidate_matches": int(rows["match_id"].nunique()),
            "current_official_input": _current_match_audit(current),
            "fit_window_audit": batch.audit.model_dump(mode="json"),
            "fit_eligible_date_coverage": {
                "first_match_date": min(item.match_date for item in batch.observations).isoformat(),
                "last_match_date": max(item.match_date for item in batch.observations).isoformat(),
            },
            "fit_eligible_unique_players": len(
                {
                    player
                    for item in batch.observations
                    for player in (item.player_a_id, item.player_b_id)
                }
            ),
            "fit_eligible_current_us_open_2026_matches": current_included,
            "fit_eligible_historical_matches": len(batch.observations) - current_included,
            "training_batch_sha256": batch.data_sha256,
            "included_source_sha256s": sorted(
                {value for item in batch.observations for value in item.source_sha256s}
            ),
            "included_crosswalk_sha256s": sorted(
                {value for item in batch.observations for value in item.crosswalk_sha256s}
            ),
        }
    return batches, combined_rows, bundles, {
        "current_service_rows": {
            "path": _relative(current_path, repo),
            "sha256": _sha256_file(current_path),
            "row_count": len(current_rows),
        },
        "by_tour": fit_audit,
    }


def _exposure(item: Any) -> DurationPathExposure:
    return DurationPathExposure(
        tour=item.tour,
        player_a_id=item.player_a_id,
        player_b_id=item.player_b_id,
        total_points=item.total_points,
        official_games=item.official_games,
        sets=item.sets,
        tiebreaks=item.tiebreaks,
        conditions=item.conditions,
    )


def _correlations(residuals: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    if len(values) < 2 or np.ptp(values) == 0 or np.ptp(residuals) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(residuals, values)[0, 1]),
        "spearman": float(stats.spearmanr(residuals, values).statistic),
    }


def _interval_coverage(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    nu: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in _INTERVAL_LEVELS:
        critical = float(stats.t.ppf((1.0 + level) / 2.0, df=nu))
        covered = (actual >= center - critical * scale) & (
            actual <= center + critical * scale
        )
        result[f"{int(level * 100)}pct"] = {
            "nominal": level,
            "empirical": float(covered.mean()),
            "covered": int(covered.sum()),
            "matches": len(actual),
        }
    return result


def _threshold_diagnostics(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    nu: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    official_integer = np.isclose(actual, np.rint(actual), rtol=0.0, atol=1e-9)
    for threshold in _THRESHOLDS:
        # For a continuous latent duration T*, the display event D > k has a
        # different analytic boundary under each plausible whole-minute rule:
        # floor(T*) > k iff T* >= k+1; nearest-half-up(T*) > k iff
        # T* >= k+0.5; and ceil(T*) > k iff T* > k.  Equality has probability
        # zero under the fitted Student-t distribution.
        latent_boundaries = {
            "floor": float(threshold + 1.0),
            "nearest_half_up": float(threshold + 0.5),
            "ceiling": float(threshold),
        }
        observed = actual > threshold
        policy_results: dict[str, Any] = {}
        mean_probabilities: list[float] = []
        for policy, boundary in latent_boundaries.items():
            predicted = 1.0 - stats.t.cdf((boundary - center) / scale, df=nu)
            mean_probability = float(np.mean(predicted))
            mean_probabilities.append(mean_probability)
            policy_results[policy] = {
                "latent_boundary_minutes": boundary,
                "mean_predicted_probability": mean_probability,
                "calibration_gap_vs_official_integer": (
                    mean_probability - float(np.mean(observed))
                ),
                "brier_score_vs_official_integer": float(
                    np.mean(np.square(predicted - observed))
                ),
            }
        # Retain the pre-v2 latent comparison explicitly for report consumers.
        # It is numerically the ceiling-policy probability for continuous T*.
        latent_predicted = 1.0 - stats.t.cdf(
            (threshold - center) / scale, df=nu
        )
        result[str(threshold)] = {
            "operator": ">",
            "minute_threshold": threshold,
            "actual_official_integer_frequency": float(np.mean(observed)),
            "official_integer_rows": int(official_integer.sum()),
            "noninteger_observed_rows": int((~official_integer).sum()),
            "latent_continuous_probability": {
                "mean_predicted_probability": float(np.mean(latent_predicted)),
                "note": (
                    "For a continuous latent duration this equals the analytic "
                    "ceiling-policy probability; it is not an asserted official "
                    "display convention."
                ),
            },
            "unresolved_display_policy_sensitivity": {
                "status": "UNRESOLVED",
                "analytic_probabilities": policy_results,
                "mean_probability_minimum": min(mean_probabilities),
                "mean_probability_maximum": max(mean_probabilities),
                "mean_probability_range": max(mean_probabilities)
                - min(mean_probabilities),
            },
            "matches": len(actual),
        }
    return result


def _current_event_stability(
    *,
    event_effect: Any,
    row_count: int,
    prior_standard_deviation: float,
) -> dict[str, Any]:
    effect = None if event_effect is None else float(event_effect.value)
    standard_error = (
        None if event_effect is None else float(event_effect.standard_error)
    )
    wald_interval = (
        None
        if effect is None or standard_error is None
        else {
            "lower_minutes": effect - 1.96 * standard_error,
            "upper_minutes": effect + 1.96 * standard_error,
        }
    )
    return {
        "row_count": row_count,
        "event_effect_minutes": effect,
        "laplace_standard_error_minutes": standard_error,
        "laplace_z_ratio": None
        if effect is None or standard_error in {None, 0.0}
        else effect / standard_error,
        "prior_standard_deviation_minutes": prior_standard_deviation,
        "effect_to_prior_standard_deviation": None
        if effect is None
        else effect / prior_standard_deviation,
        "approximate_95pct_wald_interval_minutes": wald_interval,
        "scope": "descriptive in-sample current-event stability diagnostic",
        "approximation": (
            "Laplace/Hessian standard error and Wald interval at the conditional "
            "MAP; not a held-out or rolling validation interval."
        ),
    }


def _descriptive_errors(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    nu: float,
) -> dict[str, Any]:
    residual = actual - center
    return {
        "matches": len(actual),
        "actual_mean_minutes": float(actual.mean()),
        "predicted_center_mean_minutes": float(center.mean()),
        "mean_residual_minutes": float(residual.mean()),
        "mae_minutes": float(np.mean(np.abs(residual))),
        "rmse_minutes": float(np.sqrt(np.mean(np.square(residual)))),
        "interval_coverage": _interval_coverage(actual, center, scale, nu),
        "thresholds": _threshold_diagnostics(actual, center, scale, nu),
    }


def _pace_shrinkage(artifact: DurationFitArtifact) -> dict[str, Any]:
    effects = artifact.pace_effects
    bins = (
        ("lt_1_weighted_match", 0.0, 1.0),
        ("1_to_lt_3", 1.0, 3.0),
        ("3_to_lt_10", 3.0, 10.0),
        ("ge_10", 10.0, math.inf),
    )
    summaries: dict[str, Any] = {}
    for name, lower, upper in bins:
        selected = [item for item in effects if lower <= item.weighted_matches < upper]
        summaries[name] = {
            "players": len(selected),
            "mean_absolute_effect_minutes": None
            if not selected
            else float(np.mean([abs(item.value) for item in selected])),
            "median_absolute_effect_minutes": None
            if not selected
            else float(np.median([abs(item.value) for item in selected])),
            "mean_standard_error_minutes": None
            if not selected
            else float(np.mean([item.standard_error for item in selected])),
        }
    return {
        "player_count": len(effects),
        "identifiability_sum_minutes": float(math.fsum(item.value for item in effects)),
        "prior_standard_deviation_minutes": artifact.ridge.player_pace_sd,
        "weighted_history_bins": summaries,
    }


def _validation_report(
    artifacts: dict[Tour, DurationFitArtifact],
    batches: dict[Tour, DurationTrainingBatch],
    base_build_id: str,
) -> dict[str, Any]:
    by_tour: dict[str, Any] = {}
    for tour in Tour:
        artifact = artifacts[tour]
        observations = batches[tour].observations
        mapped = map_duration_parameters(artifact)
        actual = np.asarray([item.duration_minutes for item in observations], dtype=float)
        center = np.asarray(
            [duration_predictor(mapped, _exposure(item)) for item in observations],
            dtype=float,
        )
        scale = np.asarray(
            [duration_scale(mapped, item.total_points) for item in observations],
            dtype=float,
        )
        residual = actual - center
        standardized = residual / scale
        pit = stats.t.cdf(standardized, df=artifact.nu)
        weights = np.asarray([item.recency_weight for item in observations], dtype=float)
        exposure_values = {
            "total_points": np.asarray([item.total_points for item in observations], dtype=float),
            "official_games": np.asarray(
                [item.official_games for item in observations], dtype=float
            ),
            "sets": np.asarray([item.sets for item in observations], dtype=float),
            "tiebreaks": np.asarray([item.tiebreaks for item in observations], dtype=float),
        }
        current_mask = np.asarray(
            [
                item.conditions.event == artifact.current_event
                and item.conditions.event_year == artifact.current_event_year
                for item in observations
            ],
            dtype=bool,
        )
        event_effect = next(
            (item for item in artifact.coefficients if item.name == "current_usopen_2026"),
            None,
        )
        by_tour[tour.value] = {
            "artifact_id": artifact.artifact_id,
            "validation_scope": (
                "in_sample_conditional_MAP_diagnostics; not a rolling or out-of-sample backtest"
            ),
            "matches": len(observations),
            "map_residuals": {
                "mean_minutes": float(residual.mean()),
                "recency_weighted_mean_minutes": float(np.average(residual, weights=weights)),
                "median_minutes": float(np.median(residual)),
                "standard_deviation_minutes": float(residual.std(ddof=0)),
                "mae_minutes": float(np.mean(np.abs(residual))),
                "rmse_minutes": float(np.sqrt(np.mean(np.square(residual)))),
                "correlations": {
                    name: _correlations(residual, values)
                    for name, values in exposure_values.items()
                },
            },
            "conditional_student_t_interval_coverage": _interval_coverage(
                actual, center, scale, artifact.nu
            ),
            "threshold_calibration": _threshold_diagnostics(
                actual, center, scale, artifact.nu
            ),
            "pit": {
                "mean": float(pit.mean()),
                "variance": float(pit.var()),
                "quantiles": {
                    str(value): float(np.quantile(pit, value))
                    for value in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
                },
                "decile_counts": np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))[0]
                .astype(int)
                .tolist(),
                "uniform_ks_statistic": float(stats.kstest(pit, "uniform").statistic),
                "uniform_ks_p_value_descriptive": float(stats.kstest(pit, "uniform").pvalue),
            },
            "pace_shrinkage": _pace_shrinkage(artifact),
            "current_event_stability": _current_event_stability(
                event_effect=event_effect,
                row_count=int(current_mask.sum()),
                prior_standard_deviation=artifact.ridge.current_event_sd,
            ),
            "current_us_open_2026_descriptive": None
            if not current_mask.any()
            else {
                **_descriptive_errors(
                    actual[current_mask],
                    center[current_mask],
                    scale[current_mask],
                    artifact.nu,
                ),
                "event_effect_minutes": None if event_effect is None else event_effect.value,
                "event_effect_standard_error": None
                if event_effect is None
                else event_effect.standard_error,
                "event_effect_z_ratio": None
                if event_effect is None or event_effect.standard_error == 0
                else event_effect.value / event_effect.standard_error,
                "interpretation": "Descriptive in-sample check on cutoff-safe official matches.",
            },
            "fit": {
                "coefficients": [item.model_dump(mode="json") for item in artifact.coefficients],
                "sigma0": artifact.sigma0,
                "sigma1": artifact.sigma1,
                "nu": artifact.nu,
                "context_status": [
                    item.model_dump(mode="json") for item in artifact.context_status
                ],
                "ridge": artifact.ridge.model_dump(mode="json"),
                "posterior_covariance_mode": artifact.posterior.covariance_mode,
                "posterior_dimension": len(artifact.posterior.parameter_names),
                "diagnostics": artifact.diagnostics.model_dump(mode="json"),
            },
        }
    return {
        "schema_version": _VALIDATION_SCHEMA_VERSION,
        "framework_version": "v1.0",
        "base_current_build_id": base_build_id,
        "thresholds_predeclared_minutes": list(_THRESHOLDS),
        "official_display_policy": UNRESOLVED_DURATION_DISPLAY_POLICY.model_dump(mode="json"),
        "notes": [
            (
                "Threshold diagnostics use direct conditional Student-t probabilities at "
                "MAP and expose analytic floor, nearest-half-up, and ceiling sensitivity."
            ),
            (
                "Whole-minute conversion remains unresolved; policy-sensitive markets "
                "stay disabled."
            ),
            (
                "Current-event effect stability uses a descriptive in-sample "
                "Laplace/Hessian approximation, not held-out validation."
            ),
            "Historical Sackmann-style durations are grade B; 2026 US Open feeds are official.",
        ],
        "by_tour": by_tour,
    }


def _representative_observation(batch: DurationTrainingBatch, artifact: DurationFitArtifact) -> Any:
    current = [
        item
        for item in batch.observations
        if item.conditions.event == artifact.current_event
        and item.conditions.event_year == artifact.current_event_year
    ]
    return (current or list(batch.observations))[0]


def _benchmark_tour(
    artifact: DurationFitArtifact,
    batch: DurationTrainingBatch,
    counts: tuple[int, int, int],
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    item = _representative_observation(batch, artifact)
    exposure = _exposure(item)
    parameter_seed, residual_seed = seed.spawn(2)
    parameter_rng = np.random.Generator(np.random.PCG64(parameter_seed))
    residual_rng = np.random.Generator(np.random.PCG64(residual_seed))
    preparation_start_ns = time.perf_counter_ns()
    prepared = prepare_duration_parameter_sampler(
        artifact, (exposure.player_a_id, exposure.player_b_id)
    )
    preparation_ns = time.perf_counter_ns() - preparation_start_ns
    preparation_seconds = preparation_ns / 1_000_000_000
    values = np.empty(counts[-1], dtype=float)
    digest = hashlib.sha256()
    checkpoints: list[dict[str, Any]] = []
    start_ns = time.perf_counter_ns()
    previous_ns = start_ns
    previous_count = 0
    for index in range(counts[-1]):
        parameters = sample_prepared_duration_parameters(prepared, parameter_rng)
        draw = draw_duration(
            parameters,
            exposure,
            residual_rng,
            display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
        )
        values[index] = draw.latent_minutes
        digest.update(struct.pack("!d", draw.latent_minutes))
        completed = index + 1
        if completed in counts:
            now_ns = time.perf_counter_ns()
            elapsed = (now_ns - start_ns) / 1_000_000_000
            segment = (now_ns - previous_ns) / 1_000_000_000
            observed = values[:completed]
            checkpoints.append(
                {
                    "paths": completed,
                    "cumulative_seconds": elapsed,
                    "cumulative_seconds_including_one_time_preparation": (
                        preparation_seconds + elapsed
                    ),
                    "cumulative_microseconds_per_path": elapsed * 1_000_000 / completed,
                    "cumulative_microseconds_per_path_including_one_time_preparation": (
                        (preparation_seconds + elapsed) * 1_000_000 / completed
                    ),
                    "incremental_paths": completed - previous_count,
                    "incremental_seconds": segment,
                    "incremental_microseconds_per_path": (
                        segment * 1_000_000 / (completed - previous_count)
                    ),
                    "latent_duration_sha256": digest.copy().hexdigest(),
                    "mean_minutes": float(observed.mean()),
                    "p10_minutes": float(np.quantile(observed, 0.10)),
                    "p50_minutes": float(np.quantile(observed, 0.50)),
                    "p90_minutes": float(np.quantile(observed, 0.90)),
                    "threshold_probabilities": {
                        str(threshold): float(np.mean(observed > threshold))
                        for threshold in _THRESHOLDS
                    },
                }
            )
            previous_ns = now_ns
            previous_count = completed
    return {
        "artifact_id": artifact.artifact_id,
        "benchmark_scope": (
            "duration layer only, conditional on one retained realized exposure; "
            "one-time target-local sampler preparation is timed separately and path "
            "timings contain Gaussian parameter draws plus Student-t residual draws"
        ),
        "one_time_target_sampler_preparation": {
            "seconds": preparation_seconds,
            "microseconds": preparation_ns / 1_000,
            "included_in_path_timings": False,
            "player_ids": [exposure.player_a_id, exposure.player_b_id],
        },
        "representative_match_id": item.match_id,
        "exposure": exposure.model_dump(mode="json"),
        "parameter_seed": {
            "entropy": parameter_seed.entropy,
            "spawn_key": list(parameter_seed.spawn_key),
        },
        "residual_seed": {
            "entropy": residual_seed.entropy,
            "spawn_key": list(residual_seed.spawn_key),
        },
        "checkpoints": checkpoints,
    }


def _performance_report(
    artifacts: dict[Tour, DurationFitArtifact],
    batches: dict[Tour, DurationTrainingBatch],
    counts: tuple[int, int, int],
    seed: int,
) -> dict[str, Any]:
    root = np.random.SeedSequence(seed)
    children = root.spawn(len(tuple(Tour)))
    return {
        "schema_version": _PERFORMANCE_SCHEMA_VERSION,
        "framework_version": "v1.0",
        "timer": "time.perf_counter_ns",
        "seed": seed,
        "requested_path_checkpoints": list(counts),
        "timings_are_machine_specific": True,
        "numerical_checksums_are_seed_reproducible": True,
        "by_tour": {
            tour.value: _benchmark_tour(artifacts[tour], batches[tour], counts, child)
            for tour, child in zip(Tour, children, strict=True)
        },
    }


def _validate_artifact(
    artifact: DurationFitArtifact,
    batch: DurationTrainingBatch,
    source_manifest: Any,
    source_manifest_hash: str,
) -> None:
    expected_sources = tuple(
        sorted({value for item in batch.observations for value in item.source_sha256s})
    )
    expected_crosswalks = tuple(
        sorted({value for item in batch.observations for value in item.crosswalk_sha256s})
    )
    if not artifact.diagnostics.converged:
        raise RuntimeError(
            f"{artifact.tour.value} duration optimizer did not converge: "
            f"{artifact.diagnostics.optimizer_message}"
        )
    if (
        artifact.tour is not batch.tour
        or artifact.information_cutoff_utc != batch.information_cutoff_utc
        or artifact.fit_cutoff_utc != batch.information_cutoff_utc
        or artifact.data_sha256 != batch.data_sha256
        or artifact.source_manifest_id != source_manifest.manifest_version
        or artifact.source_manifest_sha256 != source_manifest_hash
        or artifact.source_sha256s != expected_sources
        or artifact.crosswalk_sha256s != expected_crosswalks
    ):
        raise RuntimeError(f"{batch.tour.value} duration artifact provenance mismatch")


def _upgrade_snapshot(
    base_build: Path,
    tour: Tour,
    persisted_duration: Any,
    staging: Path,
    target: Path,
) -> tuple[ModelSnapshot, ModelSnapshot]:
    source = ModelSnapshot.from_json(
        (base_build / f"model_snapshot_{tour.value.lower()}.json").read_bytes()
    )
    if source.schema_version != "serve-model-snapshot/v2" or not source.b6_c6_complete:
        raise RuntimeError(f"retained {tour.value} snapshot is not a complete v2 snapshot")
    fits = {
        reference.component: load_fit_artifact(reference.directory)
        for reference in source.component_artifacts
    }
    if source.retirement_artifact is None or source.inactivity_configuration is None:
        raise RuntimeError(f"retained {tour.value} snapshot lost B6/C6 references")
    retirement = load_retirement_fit_artifact(source.retirement_artifact.directory)
    upgraded = create_model_snapshot(
        fits,
        retirement_artifact=retirement,
        inactivity_configuration=source.inactivity_configuration,
        duration_artifact=persisted_duration,
    )
    if upgraded.duration_artifact is None:
        raise RuntimeError("v3 snapshot creation lost its duration reference")
    relative_duration = persisted_duration.directory.relative_to(staging)
    future_reference = upgraded.duration_artifact.model_copy(
        update={"directory": (target / relative_duration).resolve()}
    )
    upgraded = ModelSnapshot.model_validate(
        upgraded.model_dump(mode="python") | {"duration_artifact": future_reference}
    )
    if upgraded.schema_version != "serve-model-snapshot/v3":
        raise RuntimeError("duration upgrade did not produce snapshot v3")
    return source, upgraded


def build(
    repo: Path,
    data_repo: Path,
    base_build: Path,
    output_root: Path,
    config_path: Path,
    fitted_at_utc: datetime,
    deterministic_test_receipt: Path,
    seed: int,
    benchmark_counts: tuple[int, int, int],
) -> Path:
    repo = repo.resolve()
    data_repo = data_repo.resolve()
    base_build = base_build.resolve()
    base_report_path = base_build / "build_report.json"
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_build_id = str(base_report["run_id"])
    if base_build.name != base_build_id:
        raise RuntimeError("base build directory and retained build identity differ")
    cutoff = _parse_utc(str(base_report["information_cutoff_utc"]))
    if fitted_at_utc < cutoff:
        raise RuntimeError("duration fitted_at_utc cannot precede retained build cutoff")
    if not deterministic_test_receipt.is_file():
        raise RuntimeError("deterministic test receipt is missing")
    deterministic_test_hash = _sha256_file(deterministic_test_receipt)
    config = load_duration_model_config(config_path)
    source_manifest = load_source_manifest(base_build / "source_manifest.yaml")
    source_manifest_hash = manifest_sha256(source_manifest)
    if source_manifest_hash != base_report["source_manifest_sha256"]:
        raise RuntimeError("retained source manifest differs from base build report")

    historical_audit = _historical_raw_audit(data_repo, source_manifest)
    batches, _combined, _bundles, fit_input_audit = _prepare_fit_inputs(
        data_repo, base_build, cutoff, config
    )
    code = capture_code_provenance(repo)
    code_hash = code.diff_sha256 or hashlib.sha256(code.commit.encode("ascii")).hexdigest()
    run_identity = {
        "schema_version": _SCHEMA_VERSION,
        "base_build_id": base_build_id,
        "base_build_report_sha256": _sha256_file(base_report_path),
        "information_cutoff_utc": cutoff.isoformat(),
        "fitted_at_utc": fitted_at_utc.isoformat(),
        "duration_config_sha256": config.sha256,
        "duration_config_file_sha256": _sha256_file(config_path),
        "source_manifest_sha256": source_manifest_hash,
        "current_service_rows_sha256": fit_input_audit["current_service_rows"]["sha256"],
        "training_batch_sha256s": {
            tour.value: batches[tour].data_sha256 for tour in Tour
        },
        "code_provenance": code.model_dump(mode="json"),
        "deterministic_test_result_sha256": deterministic_test_hash,
        "seed": seed,
        "benchmark_counts": list(benchmark_counts),
    }
    run_id = hashlib.sha256(_canonical_bytes(run_identity)).hexdigest()
    if run_id in _load_duration_build_rejections(output_root):
        raise RuntimeError(f"refusing to republish explicitly rejected duration run: {run_id}")
    parent = output_root.resolve() / base_build_id[:16]
    target = parent / run_id[:32]
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"duration build already exists and will not be overwritten: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id[:16]}.partial-", dir=parent))
    try:
        receipt_copy = staging / "deterministic_test_receipt.json"
        with receipt_copy.open("xb") as handle:
            handle.write(deterministic_test_receipt.read_bytes())
        if _sha256_file(receipt_copy) != deterministic_test_hash:
            raise RuntimeError("copied deterministic test receipt changed")
        for tour in Tour:
            _write_canonical(
                staging / f"duration_training_batch_{tour.value.lower()}.json",
                batches[tour].model_dump(mode="json"),
            )

        artifacts: dict[Tour, DurationFitArtifact] = {}
        persisted: dict[Tour, Any] = {}
        for tour in Tour:
            artifact = fit_duration_model(
                batches[tour],
                config=config,
                source_manifest_id=source_manifest.manifest_version,
                source_manifest_sha256=source_manifest_hash,
                fitted_at_utc=fitted_at_utc,
                software_version="tennis-model/0.1.0",
                code_sha256=code_hash,
                deterministic_test_result_sha256=deterministic_test_hash,
            )
            _validate_artifact(artifact, batches[tour], source_manifest, source_manifest_hash)
            artifacts[tour] = artifact
            persisted[tour] = write_duration_fit_artifact(
                artifact, staging / "duration_fits"
            )

        snapshots: dict[Tour, ModelSnapshot] = {}
        source_snapshots: dict[Tour, ModelSnapshot] = {}
        for tour in Tour:
            source_snapshot, upgraded = _upgrade_snapshot(
                base_build,
                tour,
                persisted[tour],
                staging,
                target,
            )
            source_snapshots[tour] = source_snapshot
            snapshots[tour] = upgraded
            _write_canonical(
                staging / f"model_snapshot_{tour.value.lower()}_v3.json",
                upgraded.model_dump(mode="json"),
            )

        data_audit = {
            "schema_version": _DATA_AUDIT_SCHEMA_VERSION,
            "framework_version": "v1.0",
            "base_current_build_id": base_build_id,
            "information_cutoff_utc": cutoff.isoformat(),
            "source_policy": {
                "historical_sackmann_style_duration_grade": "B",
                "current_2026_us_open_source": "official_authoritative",
                "fit_requires_verified_exact_match_date": True,
                "missing_duration_or_total_points_imputed": False,
                "retirements_in_ordinary_completed_fit": False,
            },
            "full_retained_raw_2017_2025": historical_audit,
            "exact_date_and_window_fit_inputs": fit_input_audit,
        }
        data_audit_hash = _write_canonical(staging / "data_audit.json", data_audit)
        validation = _validation_report(artifacts, batches, base_build_id)
        validation_hash = _write_canonical(
            staging / "validation_report.json", validation
        )
        performance = _performance_report(
            artifacts, batches, benchmark_counts, seed
        )
        performance_hash = _write_canonical(
            staging / "performance_report.json", performance
        )

        output_files = {
            path.relative_to(staging).as_posix(): _sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        build_report = {
            "schema_version": _SCHEMA_VERSION,
            "framework_version": "v1.0",
            "status": "READY",
            "run_id": run_id,
            "base_current_build": {
                "run_id": base_build_id,
                "path": _relative(base_build, repo),
                "build_report_sha256": _sha256_file(base_report_path),
                "snapshot_schema_version": "serve-model-snapshot/v2",
                "retained_files_overwritten": False,
            },
            "information_cutoff_utc": cutoff.isoformat(),
            "fitted_at_utc": fitted_at_utc.isoformat(),
            "source_manifest_id": source_manifest.manifest_version,
            "source_manifest_sha256": source_manifest_hash,
            "crosswalk_set_sha256": _CROSSWALK_SET_SHA256,
            "duration_config": {
                "path": _relative(config_path, repo),
                "config_sha256": config.sha256,
                "file_sha256": _sha256_file(config_path),
            },
            "code_provenance": code.model_dump(mode="json"),
            "code_sha256": code_hash,
            "deterministic_test_receipt": {
                "source_path": _relative(deterministic_test_receipt, repo),
                "retained_copy_path": "deterministic_test_receipt.json",
                "sha256": deterministic_test_hash,
            },
            "seed": seed,
            "artifacts": {
                tour.value: {
                    "artifact_id": artifacts[tour].artifact_id,
                    "relative_directory": persisted[tour]
                    .directory.relative_to(staging)
                    .as_posix(),
                    "training_batch_sha256": batches[tour].data_sha256,
                    "converged": artifacts[tour].diagnostics.converged,
                    "included_matches": len(batches[tour].observations),
                    "date_range": [
                        artifacts[tour].training_start_date.isoformat(),
                        artifacts[tour].training_end_date.isoformat(),
                    ],
                }
                for tour in Tour
            },
            "snapshots": {
                tour.value: {
                    "source_v2_snapshot_id": source_snapshots[tour].snapshot_id,
                    "upgraded_v3_snapshot_id": snapshots[tour].snapshot_id,
                    "relative_path": f"model_snapshot_{tour.value.lower()}_v3.json",
                }
                for tour in Tour
            },
            "reports": {
                "data_audit": {"path": "data_audit.json", "sha256": data_audit_hash},
                "validation": {
                    "path": "validation_report.json",
                    "sha256": validation_hash,
                },
                "performance": {
                    "path": "performance_report.json",
                    "sha256": performance_hash,
                },
            },
            "output_file_sha256s_excluding_build_report": output_files,
            "methodology_change": False,
            "deviations_from_B5": [],
            "remaining_limitations": [
                "Official whole-minute display conversion is unresolved.",
                "Sparse player pace histories shrink toward the tour baseline.",
                "Current-event validation is descriptive and in-sample at this cutoff.",
                "Historical Sackmann-style duration records have grade B provenance.",
            ],
        }
        _write_canonical(staging / "build_report.json", build_report)
        staging.rename(target)
    finally:
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != parent.resolve() or not resolved.name.startswith(
                f".{run_id[:16]}.partial-"
            ):
                raise RuntimeError(f"refusing to clean unexpected staging path: {resolved}")
            shutil.rmtree(resolved)

    summary = {
        "status": "READY",
        "run_id": run_id,
        "output": str(target),
        "artifacts": {
            tour.value: artifacts[tour].artifact_id for tour in Tour
        },
        "snapshots": {tour.value: snapshots[tour].snapshot_id for tour in Tour},
    }
    print(_canonical_bytes(summary).decode("utf-8"), end="")
    return target


def _parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    default_base = (
        repo
        / "artifacts/current-usopen-2026"
        / "2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--data-repo",
        type=Path,
        default=repo,
        help="Optional root containing staged data/raw and data/processed artifacts.",
    )
    parser.add_argument("--base-build", type=Path, default=default_base)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "artifacts/duration-usopen-2026",
    )
    parser.add_argument(
        "--config", type=Path, default=repo / "config/duration_v1.yaml"
    )
    parser.add_argument(
        "--fitted-at-utc",
        type=_parse_utc,
        required=True,
        help="Truthful, explicit artifact creation time (ISO-8601 with UTC offset).",
    )
    parser.add_argument(
        "--deterministic-test-receipt",
        type=Path,
        help="Pinned deterministic test receipt; defaults to the retained build receipt.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--benchmark-counts",
        type=int,
        nargs=3,
        metavar=("FIRST", "SECOND", "FINAL"),
        default=(1_000, 5_000, 100_000),
    )
    parser.add_argument(
        "--crosswalk-set-sha256",
        default=_CROSSWALK_SET_SHA256,
        help="Pinned aggregate exact-date crosswalk-set SHA-256.",
    )
    return parser


def main() -> None:
    global _CROSSWALK_SET_SHA256
    args = _parser().parse_args()
    if len(args.crosswalk_set_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.crosswalk_set_sha256
    ):
        raise SystemExit("--crosswalk-set-sha256 must be 64 lowercase hex characters")
    _CROSSWALK_SET_SHA256 = args.crosswalk_set_sha256
    counts = tuple(args.benchmark_counts)
    if len(counts) != 3 or any(value <= 0 for value in counts) or tuple(sorted(counts)) != counts:
        raise SystemExit("--benchmark-counts must be three strictly increasing positive integers")
    if len(set(counts)) != 3:
        raise SystemExit("--benchmark-counts must be strictly increasing")
    receipt = args.deterministic_test_receipt or (
        args.base_build / "deterministic_test_receipt.json"
    )
    build(
        args.repo,
        args.data_repo,
        args.base_build,
        args.output_root,
        args.config,
        args.fitted_at_utc,
        receipt,
        args.seed,
        counts,
    )


if __name__ == "__main__":
    main()
