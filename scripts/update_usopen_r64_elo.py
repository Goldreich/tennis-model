"""Advance the frozen pre-tournament surface Elo state through US Open R64."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tennis_model.estimation.elo import (
    SurfaceEloConfig,
    import_surface_elo_csv,
    write_surface_elo_artifact,
)
from tennis_model.schemas import Tour


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "artifacts" / "elo" / "current-main-tour-2026-08-30"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "elo" / "v1_2-live-through-r64"
DEFAULT_OPERATIONAL = (
    ROOT
    / "artifacts"
    / "live-usopen-2026"
    / "official-2117-v1.2"
    / "model_snapshot_atp.json"
)
K_FACTOR = 16.0
SURFACE_WEIGHT = 0.5
RATING_SCALE = 400.0
ELIGIBLE_ROUNDS = {1, 2}
EVENT_TOURS = {"Men's Singles": Tour.ATP, "Women's Singles": Tour.WTA}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.strip())
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.casefold())


def _team_name(team: dict[str, Any]) -> str:
    full = " ".join(
        str(value).strip()
        for value in (team.get("firstNameA"), team.get("lastNameA"))
        if value
    )
    return full or str(team.get("displayNameA") or team.get("idA") or "").strip()


def _effective(row: dict[str, str]) -> float:
    return (1.0 - SURFACE_WEIGHT) * float(row["global_elo"]) + SURFACE_WEIGHT * float(
        row["hard_elo"]
    )


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / RATING_SCALE))


def _integer(value: str | None) -> int:
    try:
        return int(float(str(value or 0)))
    except ValueError:
        return 0


def _resolve_row(
    rows: list[dict[str, str]],
    by_name: dict[tuple[str, str], list[int]],
    *,
    tour: Tour,
    team: dict[str, Any],
) -> tuple[int, bool]:
    name = _team_name(team)
    key = (tour.value, _normalized_name(name))
    candidates = by_name.get(key, [])
    if candidates:
        return max(
            candidates,
            key=lambda index: (
                _integer(rows[index].get("matches")),
                _integer(rows[index].get("last_event_start")),
            ),
        ), False

    player_id = str(team.get("idA") or "").strip() or f"draw:{key[1]}"
    prefix = tour.value.casefold()
    row = {
        "tour": tour.value,
        "player_key": f"{prefix}:official:{player_id}",
        "player_id": player_id,
        "player_name": name,
        "global_elo": "1500.000000",
        "hard_elo": "1500.000000",
        "effective_hard_elo": "1500.000000",
        "clay_elo": "1500.000000",
        "effective_clay_elo": "1500.000000",
        "grass_elo": "1500.000000",
        "effective_grass_elo": "1500.000000",
        "carpet_elo": "1500.000000",
        "effective_carpet_elo": "1500.000000",
        "matches": "0",
        "wins": "0",
        "losses": "0",
        "hard_matches": "0",
        "clay_matches": "0",
        "grass_matches": "0",
        "carpet_matches": "0",
        "last_event_start": "0",
        "last_event": "",
    }
    rows.append(row)
    index = len(rows) - 1
    by_name.setdefault(key, []).append(index)
    return index, True


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--operational-snapshot", type=Path, default=DEFAULT_OPERATIONAL)
    args = parser.parse_args()

    capture = args.capture.resolve()
    baseline = args.baseline.resolve()
    artifact_root = args.artifact_root.resolve()
    capture_manifest_path = capture / "manifest.json"
    capture_manifest_bytes = capture_manifest_path.read_bytes()
    capture_manifest = json.loads(capture_manifest_bytes)
    baseline_manifest_bytes = (baseline / "manifest.json").read_bytes()
    baseline_ratings_bytes = (baseline / "all_player_elos.csv").read_bytes()
    operational = json.loads(args.operational_snapshot.resolve().read_text(encoding="utf-8"))

    rows = list(
        csv.DictReader(baseline_ratings_bytes.decode("utf-8-sig").splitlines())
    )
    if not rows:
        raise RuntimeError("pre-tournament Elo table is empty")
    by_name: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["tour"]).upper(), _normalized_name(str(row["player_name"])))
        by_name.setdefault(key, []).append(index)

    matches: list[tuple[Tour, int, int, str, dict[str, Any], str]] = []
    retrievals: list[datetime] = []
    exclusions: list[dict[str, str]] = []
    for key, receipt in capture_manifest["objects"].items():
        retrieved = datetime.fromisoformat(receipt["retrieved_at_utc"]).astimezone(UTC)
        retrievals.append(retrieved)
        if not key.startswith("complete_match_"):
            continue
        payload_path = capture / receipt["relative_path"]
        payload = payload_path.read_bytes()
        if _sha256(payload) != receipt["sha256"]:
            raise RuntimeError(f"source object hash mismatch: {key}")
        parsed = json.loads(payload)
        cards = parsed.get("matches") or []
        if len(cards) != 1:
            exclusions.append({"source_key": key, "reason": "match_cardinality"})
            continue
        match = cards[0]
        tour = EVENT_TOURS.get(str(match.get("eventName") or ""))
        if tour is None:
            continue
        try:
            round_code = int(str(match.get("roundCode") or ""))
        except ValueError:
            exclusions.append({"source_key": key, "reason": "round_code"})
            continue
        if round_code not in ELIGIBLE_ROUNDS:
            exclusions.append({"source_key": key, "reason": "after_r64"})
            continue
        if str(match.get("statusCode")) != "D":
            exclusions.append({"source_key": key, "reason": "noncompletion"})
            continue
        match_id = str(match["match_id"])
        try:
            numeric_match_id = int(match_id)
        except ValueError:
            numeric_match_id = 10**9
        matches.append(
            (tour, round_code, numeric_match_id, match_id, match, receipt["sha256"])
        )

    matches.sort(key=lambda item: (item[0].value, item[1], item[2], item[3]))
    seen: set[tuple[str, str]] = set()
    updates: list[dict[str, Any]] = []
    initialized: list[dict[str, str]] = []
    for tour, round_code, _, match_id, match, source_sha256 in matches:
        dedupe = (tour.value, match_id)
        if dedupe in seen:
            raise RuntimeError(f"duplicate official match: {dedupe}")
        seen.add(dedupe)
        team_1 = match.get("team1") or {}
        team_2 = match.get("team2") or {}
        winner_code = str(match.get("winner") or "")
        if winner_code == "1":
            winner_team, loser_team = team_1, team_2
        elif winner_code == "2":
            winner_team, loser_team = team_2, team_1
        else:
            raise RuntimeError(f"completed match lacks winner: {tour.value}:{match_id}")
        winner_index, winner_new = _resolve_row(
            rows, by_name, tour=tour, team=winner_team
        )
        loser_index, loser_new = _resolve_row(rows, by_name, tour=tour, team=loser_team)
        winner = rows[winner_index]
        loser = rows[loser_index]
        if winner_new:
            initialized.append(
                {"tour": tour.value, "player_name": winner["player_name"]}
            )
        if loser_new:
            initialized.append({"tour": tour.value, "player_name": loser["player_name"]})

        winner_before = _effective(winner)
        loser_before = _effective(loser)
        probability = _expected(winner_before, loser_before)
        delta = K_FACTOR * (1.0 - probability)
        for row, sign in ((winner, 1.0), (loser, -1.0)):
            row["global_elo"] = f"{float(row['global_elo']) + sign * delta:.6f}"
            row["hard_elo"] = f"{float(row['hard_elo']) + sign * delta:.6f}"
            row["effective_hard_elo"] = f"{_effective(row):.6f}"
            row["matches"] = str(_integer(row.get("matches")) + 1)
            row["hard_matches"] = str(_integer(row.get("hard_matches")) + 1)
            row["last_event_start"] = "20260831"
            row["last_event"] = "US Open"
        winner["wins"] = str(_integer(winner.get("wins")) + 1)
        loser["losses"] = str(_integer(loser.get("losses")) + 1)
        updates.append(
            {
                "tour": tour.value,
                "round_code": round_code,
                "match_id": match_id,
                "source_sha256": source_sha256,
                "winner": winner["player_name"],
                "loser": loser["player_name"],
                "winner_effective_hard_before": winner_before,
                "loser_effective_hard_before": loser_before,
                "winner_expected_probability": probability,
                "elo_delta": delta,
                "winner_effective_hard_after": _effective(winner),
                "loser_effective_hard_after": _effective(loser),
            }
        )

    if not retrievals:
        raise RuntimeError("capture contains no retrieval timestamps")
    cutoff = max(retrievals) + timedelta(microseconds=1)
    policy = {
        "schema_version": "usopen-through-r64-elo-update/v1",
        "base_ratings_sha256": _sha256(baseline_ratings_bytes),
        "base_manifest_sha256": _sha256(baseline_manifest_bytes),
        "capture_id": capture.name,
        "capture_manifest_sha256": _sha256(capture_manifest_bytes),
        "information_cutoff_utc": cutoff.isoformat(),
        "eligible_event_names": sorted(EVENT_TOURS),
        "eligible_round_codes": sorted(ELIGIBLE_ROUNDS),
        "terminal_status_policy": "statusCode D only; retirements and other noncompletions excluded",
        "ordering": "tour, numeric roundCode, numeric match_id",
        "initial_rating": 1500.0,
        "k_factor": K_FACTOR,
        "effective_rating": {
            "global_weight": 1.0 - SURFACE_WEIGHT,
            "hard_weight": SURFACE_WEIGHT,
        },
        "update_count": len(updates),
        "updates_by_tour_round": {
            f"{tour.value}:R{round_code}": sum(
                1
                for item in updates
                if item["tour"] == tour.value and item["round_code"] == round_code
            )
            for tour in Tour
            for round_code in sorted(ELIGIBLE_ROUNDS)
        },
        "initialized_players": initialized,
        "exclusions": exclusions,
        "update_script_sha256": _sha256(Path(__file__).read_bytes()),
    }
    run_id = _sha256(_canonical_bytes(policy))
    output = artifact_root / "source" / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows.sort(
        key=lambda row: (
            row["tour"],
            -float(row["effective_hard_elo"]),
            row["player_name"],
        )
    )
    ratings_path = output / "all_player_elos.csv"
    manifest_path = output / "manifest.json"
    updates_path = output / "updates.json"
    _write_csv(ratings_path, rows)
    manifest_path.write_bytes(_canonical_bytes(policy))
    updates_path.write_bytes(_canonical_bytes(updates))

    config = SurfaceEloConfig(
        schema_version="surface-elo-config/v1",
        initial_rating=1500.0,
        k_factor=K_FACTOR,
        surface_weight=SURFACE_WEIGHT,
        rating_scale=RATING_SCALE,
        deterministic_logit_variance_floor=1e-12,
    )
    fitted_at = datetime.now(UTC)
    artifacts: dict[str, dict[str, str]] = {}
    for tour in Tour:
        fit = import_surface_elo_csv(
            ratings_path,
            manifest_path,
            tour=tour,
            cutoff_utc=cutoff,
            fitted_at_utc=fitted_at,
            config=config,
            code_commit=str(operational.get("code_commit") or "unknown"),
        )
        artifact = write_surface_elo_artifact(fit, artifact_root / "artifacts")
        artifacts[tour.value] = {
            "artifact_id": artifact.artifact_id,
            "directory": str(artifact.directory),
        }

    report = {
        "schema_version": "usopen-through-r64-elo-report/v1",
        "run_id": run_id,
        "information_cutoff_utc": cutoff.isoformat(),
        "updates": len(updates),
        "updates_by_tour_round": policy["updates_by_tour_round"],
        "initialized_players": initialized,
        "source_directory": str(output),
        "artifacts": artifacts,
    }
    (output / "artifact_report.json").write_bytes(_canonical_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
