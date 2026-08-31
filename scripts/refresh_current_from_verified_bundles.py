"""Refresh current US Open snapshots from pinned official and verified historical data."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import build_current_usopen_snapshot as builder
import pandas as pd

from tennis_model.data.artifacts import ProcessedArtifactManifest, load_processed_bundle
from tennis_model.data.source_manifest import load_source_manifest
from tennis_model.schemas import Tour

# These projections contain every field used by fitting, identity resolution,
# B6 exposure construction, or retained fit provenance. The complete immutable
# tables remain in their content-addressed bundles.
FIT_COUNT_COLUMNS = (
    "snapshot_id",
    "match_id",
    "tour",
    "component",
    "player_id",
    "opponent_id",
    "successes",
    "trials",
    "status",
    "eligible_for_likelihood",
    "available_at_utc",
    "snapshot_sha256",
    "transformation_version",
    "surface",
    "indoor",
    "event",
    "event_year",
    "match_date",
    "source_date",
)
SERVICE_COLUMNS = (
    "match_id",
    "tour",
    "player_id",
    "player_name",
    "match_date",
    "retirement",
    "walkover",
    "orientation",
    "completed",
    "service_points",
    "source_id",
    "snapshot_sha256",
    "retrieved_at_utc",
)
_HISTORICAL_ROOT: Path | None = None


def _latest_verified_bundle(parent: Path, cutoff: datetime):
    eligible: list[tuple[datetime, Path]] = []
    for manifest_path in parent.glob("*/manifest.json"):
        manifest = ProcessedArtifactManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.information_cutoff_utc <= cutoff:
            eligible.append((manifest.information_cutoff_utc, manifest_path.parent))
    if not eligible:
        raise RuntimeError(f"no cutoff-valid retained bundle under {parent}")
    return load_processed_bundle(max(eligible, key=lambda item: item[0])[1])


def _reuse_historical(repo: Path, cutoff: datetime):
    manifest = load_source_manifest(repo / "config/sources.yaml")
    root = (
        repo / "data/processed/current-usopen-2026-exact-date"
        if _HISTORICAL_ROOT is None
        else _HISTORICAL_ROOT
    )
    bundles = {Tour.ATP: [], Tour.WTA: []}
    service_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    exclusions: Counter[str] = Counter()
    for source in manifest.sources:
        year = int(source.source_id.rsplit("-", 1)[1])
        if year < 2021:
            continue
        parent = root / source.tour.value.lower() / source.source_id / source.sha256[:16]
        bundle = _latest_verified_bundle(parent, cutoff)
        if bundle.manifest.source != source:
            raise RuntimeError(f"retained bundle pin differs for {source.source_id}")
        bundles[source.tour].append(bundle)
        service_frames.append(
            pd.read_parquet(
                bundle.table_path("service_rows"),
                columns=list(SERVICE_COLUMNS),
                engine="pyarrow",
            )
        )
        if year >= 2023:
            counts = pd.read_parquet(
                bundle.table_path("component_counts"),
                columns=list(FIT_COUNT_COLUMNS),
                engine="pyarrow",
            )
            counts["component_count_artifact_sha256"] = bundle.manifest.receipt_for(
                "component_counts"
            ).sha256
            count_frames.append(counts)
        cutoff_exclusions = pd.read_parquet(
            bundle.table_path("cutoff_exclusions"),
            columns=["cutoff_exclusion_code"],
            engine="pyarrow",
        )
        for reason, rows in cutoff_exclusions.groupby("cutoff_exclusion_code"):
            exclusions[f"{source.tour.value}:{year}:{reason}"] += len(rows)
    return (
        manifest,
        bundles,
        pd.concat(service_frames, ignore_index=True),
        pd.concat(count_frames, ignore_index=True),
        dict(sorted(exclusions.items())),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/current-usopen-2026"),
    )
    parser.add_argument(
        "--historical-root",
        type=Path,
        help="optional readable mirror of current-usopen-2026-exact-date",
    )
    parser.add_argument("--deterministic-test-result-sha256", required=True)
    args = parser.parse_args()
    if len(args.deterministic_test_result_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.deterministic_test_result_sha256
    ):
        parser.error("--deterministic-test-result-sha256 must be 64 lowercase hex characters")

    repo = args.repo.resolve()
    capture = args.capture.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root.is_absolute()
        else (repo / args.output_root).resolve()
    )
    global _HISTORICAL_ROOT
    _HISTORICAL_ROOT = None if args.historical_root is None else args.historical_root.resolve()
    builder._prepare_historical = _reuse_historical
    output = builder.build(
        repo,
        capture,
        output_root,
        args.deterministic_test_result_sha256,
    )
    print(output)


if __name__ == "__main__":
    main()
