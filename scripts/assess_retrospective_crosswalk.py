"""Build the pinned 2017-2025 retrospective exact-date crosswalk assessment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_model.data.exact_date_crosswalk import (
    ExactDateSourcePin,
    build_exact_date_crosswalk,
)
from tennis_model.data.historical_validation import crosswalk_set_sha256
from tennis_model.data.source_manifest import load_source_manifest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob(repository: Path, revision: str, filename: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{filename}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _successful_retrieval(root: Path, tour: str, year: int) -> dict[str, Any]:
    path = root / f"tennis_data_retrieval_{tour}_{year}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    successful = [item for item in records if "local_file" in item and "sha256" in item]
    if not successful:
        raise RuntimeError(f"no successful Tennis-Data retrieval receipt: {path}")
    return successful[-1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=Path("config/sources.yaml"))
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/retrospective-finalized-crosswalk-v1"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    assessment_root = args.assessment_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = load_source_manifest(args.source_manifest)
    rows: list[dict[str, Any]] = []
    residuals: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []

    for source in source_manifest.sources:
        tour = source.tour.value.lower()
        year_match = source.source_id.rsplit("-", 1)[-1]
        year = int(year_match)
        repository = assessment_root / f"tennis_{tour}"
        filename = f"{tour}_matches_{year}.csv"
        sackmann_payload = _git_blob(repository, source.archive_identifier or "HEAD", filename)
        if _sha256(sackmann_payload) != source.sha256:
            raise RuntimeError(f"Sackmann pin mismatch for {source.source_id}")
        sackmann = pd.read_csv(io.BytesIO(sackmann_payload))

        receipt = _successful_retrieval(assessment_root, tour, year)
        workbook_path = assessment_root / "tennis_data" / receipt["local_file"]
        workbook_payload = workbook_path.read_bytes()
        if _sha256(workbook_payload) != receipt["sha256"]:
            raise RuntimeError(f"Tennis-Data pin mismatch for {tour} {year}")
        tennis_data = pd.read_excel(workbook_path)
        retrieved = datetime.fromisoformat(receipt["retrieved_at"]).astimezone(UTC)
        last_modified = (
            None
            if not receipt.get("last_modified")
            else parsedate_to_datetime(receipt["last_modified"]).astimezone(UTC)
        )
        augmentation = ExactDateSourcePin(
            source_id=f"tennis-data-{tour}-{year}",
            tour=source.tour,
            year=year,
            locator=receipt["url"],
            sha256=receipt["sha256"],
            size_bytes=receipt["size_bytes"],
            retrieved_at_utc=retrieved,
            source_last_modified_at_utc=last_modified,
        )
        result = build_exact_date_crosswalk(
            sackmann,
            tennis_data,
            sackmann_source_id=source.source_id,
            sackmann_source_sha256=source.sha256,
            augmentation_source=augmentation,
        )
        detail = result.detail.copy()
        detail.insert(0, "tour", source.tour.value)
        detail.insert(1, "year", year)
        detail.to_csv(output / f"crosswalk_{tour}_{year}.csv", index=False)
        residual = detail.loc[~detail["status"].eq("MATCHED")].copy()
        residuals.append(residual)
        manifest_payload = result.manifest.model_dump(mode="json")
        (output / f"manifest_{tour}_{year}.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest_payload)
        counts = dict(result.manifest.status_counts)
        rows.append(
            {
                "tour": source.tour.value,
                "year": year,
                "sackmann_source_id": source.source_id,
                "sackmann_sha256": source.sha256,
                "tennis_data_source_id": augmentation.source_id,
                "tennis_data_sha256": augmentation.sha256,
                "crosswalk_id": result.manifest.crosswalk_id,
                "source_rows": result.manifest.source_rows,
                "matched_rows": result.manifest.matched_rows,
                "residual_rows": result.manifest.residual_rows,
                "coverage": result.coverage,
                **{f"status_{status.value.lower()}": counts[status] for status in counts},
            }
        )

    summary = pd.DataFrame(rows).sort_values(["tour", "year"])
    summary.to_csv(output / "summary.csv", index=False)
    residual_detail = pd.concat(residuals, ignore_index=True, sort=False)
    residual_detail.to_csv(output / "residual_detail.csv", index=False)
    member_ids = tuple(sorted(item["crosswalk_id"] for item in manifests))
    aggregate = {
        "schema_version": "retrospective-finalized-crosswalk-assessment/v1",
        "source_manifest_version": source_manifest.manifest_version,
        "member_crosswalk_ids": member_ids,
        "crosswalk_set_sha256": crosswalk_set_sha256(member_ids),
        "source_rows": int(summary["source_rows"].sum()),
        "matched_rows": int(summary["matched_rows"].sum()),
        "residual_rows": int(summary["residual_rows"].sum()),
        "coverage": float(summary["matched_rows"].sum() / summary["source_rows"].sum()),
        "complete_for_b6_c6_history": bool(summary["residual_rows"].sum() == 0),
        "members": manifests,
    }
    (output / "aggregate_manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {key: value for key, value in aggregate.items() if key != "members"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
