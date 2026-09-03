from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser(
    description="Aggregate independent fixed-path match reports without replaying paths."
)
parser.add_argument(
    "--pair",
    action="append",
    nargs=2,
    metavar=("BASELINE", "INCREMENT"),
    required=True,
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _report_matches(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if "matches" in payload:
        seed_offset = int(payload.get("seed_offset", 0))
        matches = []
        for match in payload["matches"]:
            item = deepcopy(match)
            item["_input_path"] = str(path.resolve())
            item["_input_kind"] = "batch_report"
            item["_seed_offset"] = seed_offset
            item["_source_capture_id"] = payload.get("official_source_capture_id")
            matches.append(item)
        return matches

    if "lock" in payload:
        lock = payload["lock"]
        identity = lock["canonical_match_identity"]
        total_paths = max(
            int(estimate["total_paths"]) for estimate in lock["prop_estimates"]
        )
        return [
            {
                "official_match_id": str(identity["official_match_id"]),
                "match": None,
                "tour": identity.get("tour"),
                "snapshot_id": None,
                "paths": total_paths,
                "prop_estimates": deepcopy(lock["prop_estimates"]),
                "_input_path": str(path.resolve()),
                "_input_kind": "immutable_lock",
                "_seed_offset": 0,
                "_source_capture_id": None,
                "_content_sha256": payload.get("content_sha256"),
            }
        ]

    raise ValueError(f"unsupported simulation artifact: {path}")


def _prop_key(estimate: dict[str, Any]) -> str:
    return json.dumps(estimate["prop"], sort_keys=True, separators=(",", ":"))


def _submission_integer(probability: float) -> int:
    rounded = int(math.floor(probability * 100.0 + 0.5))
    return max(1, min(99, rounded))


def _aggregate_estimates(
    baseline: list[dict[str, Any]], increment: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    increment_by_prop = {_prop_key(item): item for item in increment}
    if {_prop_key(item) for item in baseline} != set(increment_by_prop):
        raise ValueError("baseline and increment prop definitions differ")

    combined = []
    count_fields = (
        "yes_paths",
        "no_paths",
        "void_paths",
        "unresolved_paths",
        "settled_paths",
        "total_paths",
    )
    for original in baseline:
        added = increment_by_prop[_prop_key(original)]
        estimate = deepcopy(original)
        for field in count_fields:
            estimate[field] = int(original[field]) + int(added[field])
        settled = int(estimate["settled_paths"])
        total = int(estimate["total_paths"])
        probability = float(estimate["yes_paths"]) / settled
        estimate["probability_raw"] = probability
        estimate["probability_settled"] = float(settled) / total
        estimate["submitted_integer"] = _submission_integer(probability)
        estimate["mc_standard_error"] = math.sqrt(
            probability * (1.0 - probability) / settled
        )
        if estimate.get("sensitivity_low") is not None:
            estimate["sensitivity_low"] = probability
        if estimate.get("sensitivity_high") is not None:
            estimate["sensitivity_high"] = probability
        combined.append(estimate)
    return combined


def _select_match(
    candidates: list[dict[str, Any]], official_match_id: str, path: Path
) -> dict[str, Any]:
    selected = [
        item
        for item in candidates
        if str(item["official_match_id"]) == official_match_id
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one match {official_match_id} in {path}, found {len(selected)}"
        )
    return selected[0]


aggregated_matches = []
input_pairs = []
for baseline_name, increment_name in args.pair:
    baseline_path = Path(baseline_name)
    increment_path = Path(increment_name)
    increment_matches = _report_matches(_read(increment_path), increment_path)
    if len(increment_matches) != 1:
        raise ValueError(f"increment must contain exactly one match: {increment_path}")
    increment = increment_matches[0]
    match_id = str(increment["official_match_id"])
    baseline = _select_match(
        _report_matches(_read(baseline_path), baseline_path), match_id, baseline_path
    )

    if int(baseline["_seed_offset"]) == int(increment["_seed_offset"]):
        raise ValueError(f"seed streams are not distinct for match {match_id}")
    if baseline.get("snapshot_id") and increment.get("snapshot_id"):
        if baseline["snapshot_id"] != increment["snapshot_id"]:
            raise ValueError(f"model snapshots differ for match {match_id}")
    if int(baseline["paths"]) != 10_000 or int(increment["paths"]) != 10_000:
        raise ValueError(f"expected two 10K inputs for match {match_id}")

    prop_estimates = _aggregate_estimates(
        baseline["prop_estimates"], increment["prop_estimates"]
    )
    combined = {
        "official_match_id": match_id,
        "match": increment.get("match") or baseline.get("match"),
        "tour": increment.get("tour") or baseline.get("tour"),
        "snapshot_id": increment.get("snapshot_id") or baseline.get("snapshot_id"),
        "paths": 20_000,
        "prop_estimates": prop_estimates,
        "mc_error": max(item["mc_standard_error"] for item in prop_estimates),
        "input_artifacts": [
            {
                "path": baseline["_input_path"],
                "kind": baseline["_input_kind"],
                "paths": baseline["paths"],
                "seed_offset": baseline["_seed_offset"],
                "source_capture_id": baseline.get("_source_capture_id"),
                "content_sha256": baseline.get("_content_sha256"),
            },
            {
                "path": increment["_input_path"],
                "kind": increment["_input_kind"],
                "paths": increment["paths"],
                "seed_offset": increment["_seed_offset"],
                "source_capture_id": increment.get("_source_capture_id"),
                "content_sha256": increment.get("_content_sha256"),
            },
        ],
    }
    aggregated_matches.append(combined)
    input_pairs.append(
        {
            "official_match_id": match_id,
            "baseline": str(baseline_path.resolve()),
            "increment": str(increment_path.resolve()),
        }
    )

result = {
    "schema_version": "aggregated-configured-usopen-match-batch/v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "aggregation_method": "independent-fixed-batch-count-summation/v1",
    "status": "AGGREGATED_20K_DEVELOPMENT_LOCK_SUMMARIES",
    "paths_per_match": 20_000,
    "exact_scope": "requested prop counts and probabilities",
    "limitations": [
        "path-level traces are not reconstructed",
        "quantiles outside the requested prop bundle are not aggregated",
    ],
    "input_pairs": input_pairs,
    "matches": aggregated_matches,
}

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(args.output.resolve())
