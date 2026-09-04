"""Fit v1.2 winners/unforced-errors artifacts from an immutable US Open capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tennis_model.estimation.rally_termination import (
    ACCOUNTING_CONVENTION,
    RallyFitRow,
    canonical_player_key,
    fit_directional_model,
)


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _name(team: dict[str, Any]) -> str:
    parts = (team.get("firstNameA"), team.get("lastNameA"))
    full = " ".join(str(part).strip() for part in parts if part)
    return full or str(team.get("displayNameA") or team.get("idA") or "").strip()


def _tour(event_name: str) -> str | None:
    folded = event_name.casefold()
    if "women's" in folded and "singles" in folded:
        return "WTA"
    if "men's" in folded and "singles" in folded:
        return "ATP"
    return None


def _direction(
    point_winner: str,
    point_loser: str,
    winner_stats: dict[str, Any],
    loser_stats: dict[str, Any],
) -> RallyFitRow | None:
    first_won = _integer(winner_stats.get("t_f_srv_w"))
    second_won = _integer(winner_stats.get("t_s_srv_w"))
    return_won = _integer(winner_stats.get("t_p_w_opp_srv"))
    aces = _integer(winner_stats.get("t_ace"))
    loser_df = _integer(loser_stats.get("df"))
    winners = _integer(winner_stats.get("t_w"))
    loser_ues = _integer(loser_stats.get("t_ue"))
    values = (first_won, second_won, return_won, aces, loser_df, winners, loser_ues)
    if any(value is None for value in values):
        return None
    eligible = first_won + second_won + return_won - aces - loser_df
    rally_winners = winners - aces
    rally_ues = loser_ues - loser_df
    other = eligible - rally_winners - rally_ues
    if min(eligible, rally_winners, rally_ues, other) < 0:
        return None
    return RallyFitRow(point_winner, point_loser, rally_winners, rally_ues, other)


def collect(
    capture: Path,
    crosswalk_path: Path | None = None,
) -> tuple[dict[str, list[RallyFitRow]], dict[str, str], dict[str, Any]]:
    manifest_path = capture / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows: dict[str, list[RallyFitRow]] = {"ATP": [], "WTA": []}
    aliases: dict[str, str] = {}
    quarantined: list[str] = []
    accepted_matches = {"ATP": 0, "WTA": 0}
    retrievals: list[datetime] = []
    for key, receipt in sorted(manifest["objects"].items()):
        if not key.startswith("complete_match_"):
            continue
        retrievals.append(datetime.fromisoformat(receipt["retrieved_at_utc"]).astimezone(UTC))
        payload_path = capture / receipt["relative_path"]
        payload = payload_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            raise RuntimeError(f"source object hash mismatch: {key}")
        parsed = json.loads(payload)
        matches = parsed.get("matches") or []
        if len(matches) != 1:
            quarantined.append(f"{key}:match_cardinality")
            continue
        match = matches[0]
        tour = _tour(str(match.get("eventName") or ""))
        if tour is None or match.get("statusCode") != "D":
            continue
        base = (match.get("base_stats") or {}).get("match") or {}
        team_1 = base.get("team_1") or {}
        team_2 = base.get("team_2") or {}
        name_1 = canonical_player_key(_name(match.get("team1") or {}))
        name_2 = canonical_player_key(_name(match.get("team2") or {}))
        if not name_1 or not name_2:
            quarantined.append(f"{key}:player_name")
            continue
        first = _direction(name_1, name_2, team_1, team_2)
        second = _direction(name_2, name_1, team_2, team_1)
        if first is None or second is None:
            quarantined.append(f"{key}:inconsistent_counts")
            continue
        rows[tour].extend((first, second))
        accepted_matches[tour] += 1
        for team, canonical in ((match.get("team1") or {}, name_1), (match.get("team2") or {}, name_2)):
            for alias in (team.get("idA"), team.get("displayNameA"), _name(team)):
                if alias:
                    aliases[canonical_player_key(str(alias))] = canonical
    if not retrievals:
        raise RuntimeError("capture contains no completed-match retrieval receipts")
    crosswalk_sha256: str | None = None
    if crosswalk_path is not None:
        import pandas as pd

        crosswalk_bytes = crosswalk_path.read_bytes()
        crosswalk_sha256 = hashlib.sha256(crosswalk_bytes).hexdigest()
        crosswalk = pd.read_parquet(crosswalk_path)
        for record in crosswalk.to_dict(orient="records"):
            canonical_name = canonical_player_key(str(record["normalized_name"]))
            if canonical_name not in {
                row.point_winner for tour_rows in rows.values() for row in tour_rows
            } and canonical_name not in {
                row.point_loser for tour_rows in rows.values() for row in tour_rows
            }:
                continue
            for alias in (
                record.get("canonical_player_id"),
                record.get("official_player_id"),
                record.get("official_player_name"),
            ):
                if alias:
                    aliases[canonical_player_key(str(alias))] = canonical_name
    audit = {
        "source_snapshot_id": capture.name,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "data_cutoff_utc": (max(retrievals) + timedelta(microseconds=1)).isoformat(),
        "accepted_matches": accepted_matches,
        "quarantined_count": len(quarantined),
        "quarantined": quarantined,
        "accounting_convention": ACCOUNTING_CONVENTION,
        "identity_crosswalk_path": (
            None if crosswalk_path is None else str(crosswalk_path.resolve())
        ),
        "identity_crosswalk_sha256": crosswalk_sha256,
    }
    return rows, aliases, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/production/tennis-model-v1.2"),
    )
    parser.add_argument("--prior-sd", type=float, default=0.35)
    parser.add_argument("--crosswalk", type=Path)
    args = parser.parse_args()
    rows, aliases, audit = collect(
        args.capture.resolve(),
        None if args.crosswalk is None else args.crosswalk.resolve(),
    )
    fitted_at = datetime.now(UTC)
    cutoff = datetime.fromisoformat(audit["data_cutoff_utc"]).astimezone(UTC)
    reports: dict[str, Any] = {}
    for tour in ("ATP", "WTA"):
        artifact = fit_directional_model(
            rows[tour],
            tour=tour,
            fitted_at_utc=fitted_at,
            data_cutoff_utc=cutoff,
            source_snapshot_id=audit["source_snapshot_id"],
            source_manifest_sha256=audit["source_manifest_sha256"],
            aliases=aliases,
            prior_sd=args.prior_sd,
        )
        target = args.output_dir / f"rally_termination_{tour.casefold()}.json"
        artifact.write(target)
        reports[tour] = {
            "artifact_id": artifact.payload()["artifact_id"],
            "path": str(target.resolve()),
            **artifact.fit_summary,
            "concentration": artifact.concentration,
        }
    report = {
        "schema_version": "rally-termination-fit-report/v1",
        "audit": audit,
        "fits": reports,
    }
    report_path = args.output_dir / "rally_termination_fit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
