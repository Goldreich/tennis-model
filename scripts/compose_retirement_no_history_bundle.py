from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tennis_model.estimation.retirement import (
    RetirementFitArtifact,
    RetirementPlayerSufficientStatistics,
    _canonical_json_bytes,
    _retirement_artifact_payload,
    load_retirement_fit_artifact,
    write_retirement_fit_artifact,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "player_id",
        "tour",
        "identity_namespace",
        "external_id",
        "observed_rows",
        "snapshots",
        "verified_at_utc",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ValueError(f"no-history evidence is missing fields: {missing}")
    if evidence["observed_rows"] != 0:
        raise ValueError("no-history evidence must report exactly zero observed rows")
    if not evidence["snapshots"]:
        raise ValueError("no-history evidence must identify the checked snapshots")
    for snapshot in evidence["snapshots"]:
        if set(snapshot) != {"source_id", "sha256"}:
            raise ValueError("each checked snapshot must contain source_id and sha256")
        if len(snapshot["sha256"]) != 64:
            raise ValueError("snapshot sha256 must contain 64 hexadecimal characters")
        int(snapshot["sha256"], 16)
    return evidence


def _with_asserted_no_history(
    artifact: RetirementFitArtifact,
    evidence: dict[str, Any],
    evidence_sha256: str,
) -> RetirementFitArtifact:
    player_id = str(evidence["player_id"])
    if artifact.tour.value != evidence["tour"]:
        return artifact
    if any(item.player_id == player_id for item in artifact.player_statistics):
        raise ValueError(f"player {player_id} already has retirement statistics")

    prior_starts = artifact.prior_effective_starts
    baseline = artifact.tour_baseline_rho
    no_history = RetirementPlayerSufficientStatistics(
        player_id=player_id,
        retirements_y=0.0,
        starts_n=0.0,
        alpha=prior_starts * baseline,
        beta=prior_starts * (1.0 - baseline),
    )
    statistics = tuple(
        sorted((*artifact.player_statistics, no_history), key=lambda item: item.player_id)
    )
    data_sha256 = _sha256(
        _canonical_json_bytes(
            {
                "base_data_sha256": artifact.data_sha256,
                "retirement_no_history_evidence_sha256": evidence_sha256,
            }
        )
    )
    candidate = artifact.model_copy(
        update={
            "artifact_id": "0" * 64,
            "data_sha256": data_sha256,
            "player_statistics": statistics,
        }
    )
    artifact_id = _sha256(
        _canonical_json_bytes(_retirement_artifact_payload(candidate))
    )
    return RetirementFitArtifact.model_validate(
        candidate.model_copy(update={"artifact_id": artifact_id}).model_dump(mode="python")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose an immutable bundle with a source-proven no-history retirement row."
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_bundle.resolve()
    output = args.output_bundle.resolve()
    evidence = _load_evidence(args.evidence.resolve())
    evidence_bytes = _canonical_json_bytes(evidence)
    evidence_sha256 = _sha256(evidence_bytes)

    if not source.is_dir():
        raise FileNotFoundError(f"source bundle does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output bundle already exists: {output}")

    retirement_files = sorted(source.glob("retirement_fits/*/*/*/retirement-fit.json"))
    if len(retirement_files) != 2:
        raise ValueError(f"expected two tour retirement artifacts, found {len(retirement_files)}")

    shutil.copytree(source, output, ignore=shutil.ignore_patterns("retirement_fits"))
    written: dict[str, str] = {}
    retirement_refs: dict[str, dict[str, Any]] = {}
    for artifact_file in retirement_files:
        artifact = load_retirement_fit_artifact(artifact_file.parent).artifact
        revised = _with_asserted_no_history(artifact, evidence, evidence_sha256)
        persisted = write_retirement_fit_artifact(revised, output / "retirement_fits")
        written[artifact.tour.value] = str(persisted.directory)
        retirement_refs[artifact.tour.value] = {
            "artifact_id": revised.artifact_id,
            "directory": str(persisted.directory),
            "fitted_at_utc": revised.fitted_at_utc.isoformat().replace("+00:00", "Z"),
            "information_cutoff_utc": revised.information_cutoff_utc.isoformat().replace(
                "+00:00", "Z"
            ),
            "tour": revised.tour.value,
        }

    for tour, retirement_ref in retirement_refs.items():
        snapshot_path = output / f"model_snapshot_{tour.casefold()}.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["retirement_artifact"] = retirement_ref
        snapshot_path.write_bytes(_canonical_json_bytes(snapshot))

    assertion = {
        "schema_version": "retirement-no-history-assertion/v1",
        "evidence": evidence,
        "evidence_sha256": evidence_sha256,
        "artifacts": written,
    }
    (output / "retirement_no_history_assertions.json").write_bytes(
        _canonical_json_bytes(assertion)
    )
    print(json.dumps(assertion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
