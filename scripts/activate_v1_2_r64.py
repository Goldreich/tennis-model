"""Create the v1.2 through-R64 operational bundle and readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_model.estimation.elo import load_surface_elo_artifact  # noqa: E402
from tennis_model.estimation.rally_posterior import (  # noqa: E402
    EFFECT_NAMES,
    load_active_posterior,
)
from tennis_model.estimation.rally_termination import (  # noqa: E402
    canonical_player_key,
)
from tennis_model.estimation.snapshot import (  # noqa: E402
    ModelSnapshot,
    revise_v1_2_strength_anchor,
)
from tennis_model.schemas import Tour  # noqa: E402


DEFAULT_BASE = (
    ROOT / "artifacts" / "live-usopen-2026" / "official-2117-v1.2"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "live-usopen-2026" / "through-r64-v1.2"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _team_name(team: dict[str, Any]) -> str:
    full = " ".join(
        str(value).strip()
        for value in (team.get("firstNameA"), team.get("lastNameA"))
        if value
    )
    return full or str(team.get("displayNameA") or team.get("idA") or "").strip()


def _r32_players(capture: Path) -> dict[str, list[dict[str, str]]]:
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    players: dict[str, list[dict[str, str]]] = {"ATP": [], "WTA": []}
    event_tour = {"Men's Singles": "ATP", "Women's Singles": "WTA"}
    seen: set[tuple[str, str]] = set()
    for key, receipt in manifest["objects"].items():
        if not key.startswith("complete_match_"):
            continue
        payload = (capture / receipt["relative_path"]).read_bytes()
        if _sha256(payload) != receipt["sha256"]:
            raise RuntimeError(f"capture object hash mismatch: {key}")
        cards = json.loads(payload).get("matches") or []
        if len(cards) != 1:
            continue
        match = cards[0]
        tour = event_tour.get(str(match.get("eventName") or ""))
        if tour is None or str(match.get("roundCode")) != "2":
            continue
        winner = str(match.get("winner") or "")
        team = match.get("team1" if winner == "1" else "team2") or {}
        if winner not in {"1", "2"}:
            raise RuntimeError(f"R64 match has no winner: {match.get('match_id')}")
        name = _team_name(team)
        identity = (tour, canonical_player_key(name))
        if identity in seen:
            raise RuntimeError(f"duplicate R32 qualifier: {identity}")
        seen.add(identity)
        players[tour].append(
            {
                "player_name": name,
                "official_player_id": str(team.get("idA") or ""),
                "r64_match_id": str(match["match_id"]),
                "r64_status_code": str(match.get("statusCode") or ""),
            }
        )
    for tour in players:
        players[tour].sort(key=lambda item: canonical_player_key(item["player_name"]))
        if len(players[tour]) != 32:
            raise RuntimeError(f"expected 32 {tour} R32 players, found {len(players[tour])}")
    return players


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-bundle", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--atp-elo", type=Path, required=True)
    parser.add_argument("--wta-elo", type=Path, required=True)
    parser.add_argument("--rally-root", type=Path, required=True)
    args = parser.parse_args()

    base = args.base_bundle.resolve()
    output = args.output_bundle.resolve()
    capture = args.capture.resolve()
    rally_root = args.rally_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite operational bundle: {output}")
    output.mkdir(parents=True)

    elo_artifacts = {
        Tour.ATP: load_surface_elo_artifact(args.atp_elo.resolve()),
        Tour.WTA: load_surface_elo_artifact(args.wta_elo.resolve()),
    }
    posterior_artifacts = {
        Tour.ATP: load_active_posterior(rally_root, "ATP"),
        Tour.WTA: load_active_posterior(rally_root, "WTA"),
    }
    r32 = _r32_players(capture)
    readiness: dict[str, list[dict[str, Any]]] = {"ATP": [], "WTA": []}
    snapshots: dict[str, dict[str, str]] = {}

    for tour in Tour:
        source_path = base / f"model_snapshot_{tour.value.casefold()}.json"
        source = ModelSnapshot.model_validate_json(source_path.read_bytes())
        revised = revise_v1_2_strength_anchor(
            source,
            strength_anchor=elo_artifacts[tour],
        )
        target = output / source_path.name
        target.write_text(revised.canonical_json(), encoding="utf-8")
        snapshots[tour.value] = {
            "prior_snapshot_id": source.snapshot_id,
            "snapshot_id": revised.snapshot_id,
            "strength_anchor_artifact_id": elo_artifacts[tour].artifact_id,
        }

        posterior = posterior_artifacts[tour]
        player_index = {player: index for index, player in enumerate(posterior.players)}
        missing: list[str] = []
        for player in r32[tour.value]:
            name_key = canonical_player_key(player["player_name"])
            official_key = canonical_player_key(player["official_player_id"])
            resolved = posterior.aliases.get(
                official_key,
                posterior.aliases.get(name_key, name_key),
            )
            index = player_index.get(resolved)
            if index is None:
                missing.append(player["player_name"])
                effects = None
            else:
                start = 4 * index
                effects = {
                    effect: float(posterior.posterior_mean[start + offset])
                    for offset, effect in enumerate(EFFECT_NAMES)
                }
            readiness[tour.value].append(
                {
                    **player,
                    "posterior_player_key": resolved,
                    "initialized": index is not None,
                    "posterior_effect_means": effects,
                }
            )
        if missing:
            raise RuntimeError(
                f"{tour.value} R32 players lack rally posterior state: {missing}"
            )

    for name in ("training_eligibility_atp.json", "training_eligibility_wta.json"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, output / name)

    readiness_path = output / "r32_rally_effect_readiness.json"
    readiness_path.write_bytes(_canonical_bytes(readiness))
    report = {
        "schema_version": "tennis-model-v1.2-r64-activation/v1",
        "capture_id": capture.name,
        "capture_manifest_sha256": _sha256((capture / "manifest.json").read_bytes()),
        "snapshots": snapshots,
        "rally": {
            tour.value: {
                "artifact_id": posterior_artifacts[tour].artifact_id,
                "data_cutoff_utc": posterior_artifacts[tour].data_cutoff_utc.isoformat(),
                "update_sequence": posterior_artifacts[tour].update_sequence,
                "players": len(posterior_artifacts[tour].players),
                "r32_initialized": sum(
                    1 for item in readiness[tour.value] if item["initialized"]
                ),
                "r32_missing": [],
            }
            for tour in Tour
        },
        "runtime_rally_root": str(rally_root),
        "readiness_path": str(readiness_path),
    }
    (output / "activation_report.json").write_bytes(_canonical_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
