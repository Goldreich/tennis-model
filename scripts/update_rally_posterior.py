"""Apply every unseen completed singles match in an immutable source capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fit_rally_termination import _direction, _name, _tour
from tennis_model.estimation.rally_posterior import (
    activate_posterior,
    load_active_posterior,
    update_posterior,
)
from tennis_model.estimation.rally_termination import canonical_player_key


def _crosswalk_aliases(path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in pd.read_parquet(path).to_dict(orient="records"):
        name = canonical_player_key(str(row["normalized_name"]))
        for value in (
            row.get("canonical_player_id"),
            row.get("official_player_id"),
            row.get("official_player_name"),
            name,
        ):
            if value:
                aliases[canonical_player_key(str(value))] = name
    return aliases


def _candidate(
    capture: Path,
    key: str,
    receipt: dict[str, Any],
) -> tuple[int, datetime, str, str, list[Any], dict[str, str]] | None:
    payload_path = capture / receipt["relative_path"]
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise RuntimeError(f"source object hash mismatch: {key}")
    parsed = json.loads(payload)
    matches = parsed.get("matches") or []
    if len(matches) != 1:
        return None
    match = matches[0]
    tour = _tour(str(match.get("eventName") or ""))
    if tour is None or match.get("statusCode") != "D":
        return None
    first_name = canonical_player_key(_name(match.get("team1") or {}))
    second_name = canonical_player_key(_name(match.get("team2") or {}))
    base = (match.get("base_stats") or {}).get("match") or {}
    first = _direction(
        first_name,
        second_name,
        base.get("team_1") or {},
        base.get("team_2") or {},
    )
    second = _direction(
        second_name,
        first_name,
        base.get("team_2") or {},
        base.get("team_1") or {},
    )
    if first is None or second is None:
        raise ValueError(f"new completed match has inconsistent winner/UE counts: {key}")
    aliases: dict[str, str] = {}
    for term, name in (
        (match.get("team1") or {}, first_name),
        (match.get("team2") or {}, second_name),
    ):
        for value in (term.get("idA"), term.get("displayNameA"), _name(term), name):
            if value:
                aliases[canonical_player_key(str(value))] = name
    available = datetime.fromisoformat(receipt["retrieved_at_utc"]).astimezone(UTC)
    try:
        round_order = int(str(match.get("roundCode") or ""))
    except ValueError:
        round_order = 10_000
    return (
        round_order,
        available,
        str(match["match_id"]),
        tour,
        [first, second],
        aliases,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--equivalent-prior-capture", type=Path)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=Path("artifacts/production/tennis-model-v1.2"),
    )
    args = parser.parse_args()
    capture = args.capture.resolve()
    production = args.production_root.resolve()
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for key, receipt in manifest["objects"].items():
        if key.startswith("complete_match_"):
            candidate = _candidate(capture, key, receipt)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: (item[3], item[0], item[2]))
    prior_equivalents: dict[tuple[str, str], tuple[str, list[Any]]] = {}
    if args.equivalent_prior_capture is not None:
        prior_capture = args.equivalent_prior_capture.resolve()
        prior_manifest = json.loads(
            (prior_capture / "manifest.json").read_text(encoding="utf-8")
        )
        for key, receipt in prior_manifest["objects"].items():
            if not key.startswith("complete_match_"):
                continue
            candidate = _candidate(prior_capture, key, receipt)
            if candidate is None:
                continue
            _, _, prior_match_id, prior_tour, prior_rows, _ = candidate
            prior_equivalents[(prior_tour, prior_match_id)] = (
                str(receipt["sha256"]),
                prior_rows,
            )
    crosswalk_aliases = _crosswalk_aliases(args.crosswalk.resolve())
    active = {
        tour: load_active_posterior(production, tour) for tour in ("ATP", "WTA")
    }
    updates: list[dict[str, Any]] = []
    equivalent_source_revisions: list[dict[str, str]] = []
    for _, available, match_id, tour, rows, source_aliases in candidates:
        current = active[tour]
        source_sha256 = str(
            manifest["objects"][f"complete_match_{match_id}"]["sha256"]
        )
        if match_id in current.seen_matches:
            if current.seen_matches[match_id] != source_sha256:
                prior = prior_equivalents.get((tour, match_id))
                if (
                    prior is None
                    or prior[0] != current.seen_matches[match_id]
                    or prior[1] != rows
                ):
                    raise ValueError(
                        f"immutable correction requires an explicit policy: {match_id}"
                    )
                equivalent_source_revisions.append(
                    {
                        "tour": tour,
                        "match_id": match_id,
                        "prior_source_sha256": prior[0],
                        "current_source_sha256": source_sha256,
                        "decision": "accepted_modeled_observation_equivalence",
                    }
                )
            continue
        cutoff = max(
            available + timedelta(microseconds=1),
            current.data_cutoff_utc + timedelta(microseconds=1),
        )
        updated = update_posterior(
            current,
            rows,
            match_id=match_id,
            source_sha256=source_sha256,
            available_at_utc=cutoff,
            aliases={**crosswalk_aliases, **source_aliases},
            artifact_root=production / "rally-posterior",
            updated_at_utc=datetime.now(UTC),
        )
        activate_posterior(updated, production)
        active[tour] = updated
        updates.append(
            {
                "tour": tour,
                "match_id": match_id,
                "parent_artifact_id": current.artifact_id,
                "artifact_id": updated.artifact_id,
                "data_cutoff_utc": updated.data_cutoff_utc.isoformat(),
            }
        )
    report = {
        "schema_version": "rally-posterior-update-report/v1",
        "source_snapshot_id": capture.name,
        "ordering": "tour, numeric roundCode, match_id",
        "updates": updates,
        "equivalent_source_revisions": equivalent_source_revisions,
        "active": {tour: artifact.artifact_id for tour, artifact in active.items()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
