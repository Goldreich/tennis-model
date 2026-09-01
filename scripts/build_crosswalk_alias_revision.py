"""Build an immutable v2 exact-date crosswalk revision from pinned local inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from tennis_model.data.exact_date_crosswalk import (
    ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION,
    ExactDateCrosswalkManifest,
    ExactDateJoinStatus,
    build_exact_date_crosswalk,
)
from tennis_model.data.historical_validation import crosswalk_set_sha256


OSORIO_ALIASES = {"Camila Osorio": ("osorio|m",)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--workbook-root", type=Path, required=True)
    parser.add_argument("--policy-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    old_root = args.old_root.resolve()
    output = args.output.resolve()
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise RuntimeError("crosswalk revision output already exists")
    partial.mkdir(parents=True)
    manifests: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    residuals: list[pd.DataFrame] = []

    for old_manifest_path in sorted(old_root.glob("manifest_*.json")):
        stem = old_manifest_path.stem.removeprefix("manifest_")
        tour, year_text = stem.split("_")
        year = int(year_text)
        old_manifest = ExactDateCrosswalkManifest.model_validate_json(
            old_manifest_path.read_bytes()
        )
        revised = tour == "wta" and 2021 <= year <= 2025
        if revised:
            raw_path = (
                args.raw_root.resolve()
                / tour
                / f"sackmann-{tour}-{year}"
                / old_manifest.sackmann_source_sha256
                / "payload"
            )
            workbook_path = (
                args.workbook_root.resolve() / f"tennis-data-{tour}-{year}.xlsx"
            )
            if _sha256(raw_path) != old_manifest.sackmann_source_sha256:
                raise RuntimeError(f"raw snapshot hash mismatch for {stem}")
            if _sha256(workbook_path) != old_manifest.augmentation_source.sha256:
                raise RuntimeError(f"augmentation workbook hash mismatch for {stem}")
            result = build_exact_date_crosswalk(
                pd.read_csv(raw_path),
                pd.read_excel(workbook_path),
                sackmann_source_id=old_manifest.sackmann_source_id,
                sackmann_source_sha256=old_manifest.sackmann_source_sha256,
                augmentation_source=old_manifest.augmentation_source,
                sackmann_name_key_aliases=OSORIO_ALIASES,
                algorithm_version=ALIASED_EXACT_DATE_MATCHING_ALGORITHM_VERSION,
            )
            detail = result.detail.copy()
            detail.insert(0, "tour", tour.upper())
            detail.insert(1, "year", year)
            detail.to_csv(partial / f"crosswalk_{stem}.csv", index=False)
            manifest_payload = result.manifest.model_dump(mode="json")
            (partial / f"manifest_{stem}.json").write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = result.manifest
        else:
            shutil.copy2(old_root / f"crosswalk_{stem}.csv", partial)
            shutil.copy2(old_manifest_path, partial)
            detail = pd.read_csv(partial / f"crosswalk_{stem}.csv", keep_default_na=False)
            manifest_payload = old_manifest.model_dump(
                mode="json", exclude={"sackmann_name_key_aliases"}
            )
            manifest = old_manifest
        manifests.append(manifest_payload)
        residuals.append(detail.loc[~detail["status"].eq("MATCHED")].copy())
        counts = dict(manifest.status_counts)
        summaries.append(
            {
                "tour": tour.upper(),
                "year": year,
                "sackmann_source_id": manifest.sackmann_source_id,
                "sackmann_sha256": manifest.sackmann_source_sha256,
                "tennis_data_source_id": manifest.augmentation_source.source_id,
                "tennis_data_sha256": manifest.augmentation_source.sha256,
                "crosswalk_id": manifest.crosswalk_id,
                "source_rows": manifest.source_rows,
                "matched_rows": manifest.matched_rows,
                "residual_rows": manifest.residual_rows,
                "coverage": manifest.matched_rows / manifest.source_rows,
                **{
                    f"status_{status.value.lower()}": counts[status]
                    for status in ExactDateJoinStatus
                },
            }
        )

    summary = pd.DataFrame(summaries).sort_values(["tour", "year"])
    summary_path = partial / "summary.csv"
    summary.to_csv(summary_path, index=False)
    residual_path = partial / "residual_detail.csv"
    pd.concat(residuals, ignore_index=True, sort=False).to_csv(residual_path, index=False)
    member_ids = tuple(sorted(str(item["crosswalk_id"]) for item in manifests))
    set_hash = crosswalk_set_sha256(member_ids)
    aggregate = {
        "schema_version": "retrospective-finalized-crosswalk-assessment/v2",
        "source_manifest_version": "production-sackmann-2017-2025/v1",
        "member_crosswalk_ids": member_ids,
        "crosswalk_set_sha256": set_hash,
        "source_rows": int(summary["source_rows"].sum()),
        "matched_rows": int(summary["matched_rows"].sum()),
        "residual_rows": int(summary["residual_rows"].sum()),
        "coverage": float(summary["matched_rows"].sum() / summary["source_rows"].sum()),
        "complete_for_b6_c6_history": bool(summary["residual_rows"].sum() == 0),
        "members": manifests,
    }
    aggregate_path = partial / "aggregate_manifest.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    policy = yaml.safe_load(args.policy_template.read_text(encoding="utf-8"))
    policy["exact_date_algorithm_version"] = (
        "sackmann-tennis-data-exact-date/v1+osorio-alias/v2"
    )
    policy["crosswalk_set_sha256"] = set_hash
    policy["exact_date_history_complete"] = aggregate["complete_for_b6_c6_history"]
    policy["coverage_gate"] = {
        "source_rows": aggregate["source_rows"],
        "matched_rows": aggregate["matched_rows"],
        "residual_rows": aggregate["residual_rows"],
        "coverage": aggregate["coverage"],
        "sufficient_for_b6_c6_and_rolling_backtest": aggregate[
            "complete_for_b6_c6_history"
        ],
        "reason": (
            "Residual rows remain explicitly excluded; the v2 revision adds the pinned "
            "Camila Osorio to Osorio M. identity alias without asserting global completeness."
        ),
    }
    policy["generated_artifacts"] = {
        "aggregate_manifest": {
            "path": str(aggregate_path),
            "sha256": _sha256(aggregate_path),
        },
        "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "residual_detail": {
            "path": str(residual_path),
            "sha256": _sha256(residual_path),
        },
    }
    policy["members"] = [
        {
            "tour": str(item["augmentation_source"]["tour"]),
            "year": int(item["augmentation_source"]["year"]),
            "sackmann_source_id": item["sackmann_source_id"],
            "sackmann_sha256": item["sackmann_source_sha256"],
            "tennis_data_sha256": item["augmentation_source"]["sha256"],
            "crosswalk_id": item["crosswalk_id"],
        }
        for item in sorted(
            manifests,
            key=lambda value: (
                str(value["augmentation_source"]["tour"]),
                int(value["augmentation_source"]["year"]),
            ),
        )
    ]
    (partial / "historical_validation_policy.yaml").write_text(
        yaml.safe_dump(policy, sort_keys=False), encoding="utf-8"
    )
    partial.replace(output)
    print(json.dumps({"output": str(output), "crosswalk_set_sha256": set_hash}))


if __name__ == "__main__":
    main()
