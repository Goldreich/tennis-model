"""Compose prepared B6/C6 snapshots with matching duration-enabled snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tennis_model.estimation.duration_model import (
    DurationFitArtifact,
    load_duration_fit_artifact,
    write_duration_fit_artifact,
)
from tennis_model.estimation.snapshot import ModelSnapshot


def _rebased_duration_artifact(
    source: Path, cutoff, source_manifest_sha256: str, output: Path
):
    persisted = load_duration_fit_artifact(source)
    updates = {
        "artifact_id": "0" * 64,
        "information_cutoff_utc": cutoff,
        "fit_cutoff_utc": cutoff,
        "source_manifest_sha256": source_manifest_sha256,
    }
    provisional = persisted.artifact.model_copy(update=updates)
    canonical = json.dumps(
        provisional.model_dump(mode="json", exclude={"artifact_id"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    canonical = (canonical + "\n").encode("utf-8")
    artifact_id = hashlib.sha256(canonical).hexdigest()
    artifact = DurationFitArtifact.model_validate(
        provisional.model_copy(update={"artifact_id": artifact_id}).model_dump(mode="python")
    )
    return write_duration_fit_artifact(artifact, output / "duration_fits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--duration-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebase-duration-cutoff", action="store_true")
    parser.add_argument("--source-manifest-sha256")
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    duration_build = args.duration_build.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("composed output already exists")
    shutil.copytree(prepared, output)
    try:
        for tour in ("atp", "wta"):
            target_path = output / f"model_snapshot_{tour}.json"
            production = ModelSnapshot.from_json(target_path.read_bytes())
            duration_source = ModelSnapshot.from_json(
                (duration_build / f"model_snapshot_{tour}_v3.json").read_bytes()
            )
            for field in (
                "tour",
                "data_hash",
                "component_count_artifact_hash",
                "config_hash",
            ):
                if getattr(production, field) != getattr(duration_source, field):
                    raise RuntimeError(f"snapshot mismatch for {tour}: {field}")
            if production.retirement_artifact is None or duration_source.duration_artifact is None:
                raise RuntimeError(f"required artifact missing for {tour}")
            if production.data_cutoff_utc != duration_source.data_cutoff_utc:
                if not args.rebase_duration_cutoff:
                    raise RuntimeError(f"snapshot mismatch for {tour}: data_cutoff_utc")
                if args.source_manifest_sha256 is None:
                    raise RuntimeError(
                        "--source-manifest-sha256 is required when rebasing duration cutoff"
                    )
                duration = _rebased_duration_artifact(
                    Path(duration_source.duration_artifact.directory),
                    production.data_cutoff_utc,
                    args.source_manifest_sha256,
                    output,
                )
                duration_reference = duration_source.duration_artifact.model_copy(
                    update={
                        "artifact_id": duration.artifact_id,
                        "directory": duration.directory,
                        "information_cutoff_utc": duration.artifact.information_cutoff_utc,
                    }
                )
            else:
                duration_reference = duration_source.duration_artifact
            retirement = production.retirement_artifact
            retirement_directory = output / Path(retirement.directory).relative_to(prepared)
            composed = production.model_copy(
                update={
                    "schema_version": "serve-model-snapshot/v3",
                    "retirement_artifact": retirement.model_copy(
                        update={"directory": retirement_directory}
                    ),
                    "duration_artifact": duration_reference,
                    "duration_schema_version": duration_source.duration_schema_version,
                }
            )
            target_path.write_text(composed.canonical_json(), encoding="utf-8")
    except Exception:
        shutil.rmtree(output)
        raise
    print(output)


if __name__ == "__main__":
    main()
