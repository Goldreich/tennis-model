"""Create adaptive production locks for the requested remaining US Open matches.

This operational command does not refit or alter any fitted probability field.
It pins the live schedule and target-local C6 evidence, combines the already
amended production B6 snapshots with the retained duration artifacts, and runs
the frozen adaptive Monte Carlo workflow.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tennis_model.data.source_manifest import load_source_manifest
from tennis_model.estimation.inactivity import (
    CompetitionClass,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityMatchCandidate,
    InactivityTerminalStatus,
    PlayedPointEvidence,
)
from tennis_model.estimation.snapshot import ModelSnapshot
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking import (
    ADAPTIVE_MC_CS_V1_POLICY,
    HistoricalTrainingEligibilityProvenance,
    InformationBundle,
    InformationItem,
    LockStore,
    PlayerInactivityInformation,
    RetainedArtifactRecord,
    SourceManifestProvenance,
    capture_code_provenance,
    create_prediction_lock,
)
from tennis_model.props import CANONICAL_SETTLEMENT_POLICY
from tennis_model.schemas import Tour
from tennis_model.simulation import MATCH_WIN
from tennis_model.simulation.parameters import MatchCondition, MatchContext


BASE_RUN_ID = "2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664"
DURATION_RUN_ID = "4c9d944a03931055df58b5ec8405eb22e7e69160958c9d28d4c4396e6a6c078b"
SCHEDULE_URL = "https://www.usopen.org/en_US/scores/feeds/2026/schedule/schedule8.json"
USER_AGENT = "TennisModel-v1.0 live remaining-four provenance capture"
NY = ZoneInfo("America/New_York")


PLAYERS = {
    "navarro": {
        "name": "Emma Navarro",
        "id": "player_ce23ce28-23c8-574a-8f50-f0e7e31effb3",
        "tour": Tour.WTA,
        "latest_date": date(2026, 8, 17),
        "latest_match_id": "wta-cincinnati-2026-navarro-pegula-r32",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.wtatennis.com/news/4561658/pegula-remains-unbeaten-against-navarro-with-straight-sets-cincinnati-victory",
    },
    "boisson": {
        "name": "Lois Boisson",
        "id": "player_f38d37f7-0989-5cf2-9754-1ae439f5c9cf",
        "tour": Tour.WTA,
        "latest_date": date(2026, 6, 30),
        "latest_match_id": "wta-wimbledon-2026-boisson-rybakina-r128",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.wtatennis.com/news/4528838/rybakina-survives-boisson-scare-to-advance-at-wimbledon",
    },
    "navone": {
        "name": "Mariano Navone",
        "id": "player_bde060e7-6f01-5c5e-badf-2af11a0f786d",
        "tour": Tour.ATP,
        "latest_date": date(2026, 8, 6),
        "latest_match_id": "atp-montreal-2026-navone-fils-r32",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.atptour.com/en/scores/archive/montreal/421/2026/results",
    },
    "djokovic": {
        "name": "Novak Djokovic",
        "id": "player_992cc5b1-63b5-5e03-b18d-2d264d3fc45b",
        "tour": Tour.ATP,
        "latest_date": date(2026, 7, 10),
        "latest_match_id": "atp-wimbledon-2026-djokovic-sinner-sf",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.atptour.com/en/news/djokovic-wimbledon-2026-sf-reaction-friday",
    },
    "svitolina": {
        "name": "Elina Svitolina",
        "id": "player_17a16c2f-ed1e-55e3-aeee-7cad19de1572",
        "tour": Tour.WTA,
        "latest_date": date(2026, 8, 16),
        "latest_match_id": "wta-cincinnati-2026-svitolina-valentova-r64",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.wtatennis.com/news/4561997/svitolina-withdraws-from-cincinnati-open-wang-xiyu-moves-last-16",
    },
    "sierra": {
        "name": "Solana Sierra",
        "id": "player_c81dae89-76a0-5ba6-a30e-dd07c68f5f62",
        "tour": Tour.WTA,
        "latest_date": date(2026, 7, 1),
        "latest_match_id": "wta-wimbledon-2026-sierra-gauff-r64",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.wtatennis.com/news/4529459/gauff-escapes-sierra-in-third-set-tiebreak-thriller-at-wimbledon",
    },
    "bublik": {
        "name": "Alexander Bublik",
        "id": "player_ff651893-77ce-5825-84e7-8afde4281fb2",
        "tour": Tour.ATP,
        "latest_date": date(2026, 7, 25),
        "latest_match_id": "atp-kitzbuhel-2026-bublik-halys-final",
        "competition": CompetitionClass.MAIN_DRAW,
        "source_url": "https://www.atptour.com/en/news/bublik-halys-kitzbuhel-2026-final",
    },
    "wolf": {
        "name": "J.J. Wolf",
        "id": "player_48ab0de7-0021-574f-8a7d-edd8a3587f3e",
        "tour": Tour.ATP,
        "latest_date": date(2026, 8, 12),
        "latest_match_id": "atp-cincinnati-2026-wolf-shimabukuro-q2",
        "competition": CompetitionClass.QUALIFYING,
        "source_url": "https://www.atptour.com/en/scores/archive/cincinnati/422/2026/results",
    },
}

MATCHES = (
    {"official_id": "2129", "a": "navarro", "b": "boisson", "tour": Tour.WTA},
    {"official_id": "1148", "a": "navone", "b": "djokovic", "tour": Tour.ATP},
    {"official_id": "2125", "a": "svitolina", "b": "sierra", "tour": Tour.WTA},
    {"official_id": "1157", "a": "bublik", "b": "wolf", "tour": Tour.ATP},
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    path.write_bytes(payload)


def fetch(locator: str) -> tuple[bytes, datetime]:
    request = urllib.request.Request(locator, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(f"source retrieval failed for {locator}: {exc}") from exc
    if not payload:
        raise RuntimeError(f"official source returned an empty payload: {locator}")
    return payload, datetime.now(UTC)


def acquire_sources(root: Path) -> tuple[Path, dict[str, dict[str, Any]], datetime]:
    locators = {"schedule": SCHEDULE_URL}
    for key, player in PLAYERS.items():
        locators[f"c6_{key}"] = str(player["source_url"])
    captured: dict[str, dict[str, Any]] = {}
    for key, locator in locators.items():
        payload, retrieved = fetch(locator)
        captured[key] = {
            "locator": locator,
            "retrieved_at_utc": retrieved,
            "sha256": sha256_bytes(payload),
            "payload": payload,
        }
    manifest = {
        "schema_version": "remaining-four-source-capture/v1",
        "objects": {
            key: {
                "locator": item["locator"],
                "retrieved_at_utc": item["retrieved_at_utc"].isoformat(),
                "sha256": item["sha256"],
                "size_bytes": len(item["payload"]),
                "relative_path": f"objects/{key}.{'json' if key == 'schedule' else 'html'}",
            }
            for key, item in sorted(captured.items())
        },
    }
    manifest_payload = canonical_bytes(manifest)
    capture_id = sha256_bytes(manifest_payload)
    target = root / "source-captures" / capture_id
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".partial-capture-", dir=target.parent))
        try:
            (staging / "objects").mkdir()
            for key, receipt in manifest["objects"].items():
                (staging / receipt["relative_path"]).write_bytes(captured[key]["payload"])
            (staging / "manifest.json").write_bytes(manifest_payload)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif (target / "manifest.json").read_bytes() != manifest_payload:
        raise RuntimeError("existing source capture conflicts with its content identity")
    cutoff = datetime.now(UTC)
    return target, captured, cutoff


def load_captured_sources(target: Path) -> tuple[Path, dict[str, dict[str, Any]], datetime]:
    manifest_payload = (target / "manifest.json").read_bytes()
    if sha256_bytes(manifest_payload) != target.name:
        raise RuntimeError("readable source capture directory differs from its manifest hash")
    manifest = json.loads(manifest_payload)
    captured: dict[str, dict[str, Any]] = {}
    availability: list[datetime] = []
    for key, receipt in manifest["objects"].items():
        payload = (target / receipt["relative_path"]).read_bytes()
        if sha256_bytes(payload) != receipt["sha256"]:
            raise RuntimeError(f"readable source capture object failed verification: {key}")
        retrieved = datetime.fromisoformat(receipt["retrieved_at_utc"]).astimezone(UTC)
        availability.append(retrieved)
        captured[key] = {
            "locator": receipt["locator"],
            "retrieved_at_utc": retrieved,
            "sha256": receipt["sha256"],
            "payload": payload,
        }
    return target, captured, max(availability) + timedelta(microseconds=1)


def code_archive(repo: Path, output: Path) -> Path:
    result = subprocess.run(
        ("git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"),
        check=True,
        capture_output=True,
        text=True,
    )
    admitted = ("src/", "scripts/", "tests/", "config/", "docs/")
    files = tuple(
        relative
        for relative in sorted(result.stdout.splitlines())
        if relative.startswith(admitted)
        or relative in {"AGENTS.md", "README.md", "pyproject.toml", "uv.lock"}
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=output, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (repo / relative).read_bytes())
        digest = sha256_file(temporary_path)
        target = output / f"working-tree-{digest}.zip"
        if target.exists():
            if sha256_file(target) != digest:
                raise RuntimeError("existing code archive failed verification")
        else:
            temporary_path.replace(target)
        return target
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def retained_record(kind: str, path: Path) -> RetainedArtifactRecord:
    digest = hash_path(path)
    return RetainedArtifactRecord(
        kind=kind,  # type: ignore[arg-type]
        artifact_id=f"{kind}:{digest}",
        path=str(path.resolve()),
        sha256=digest,
    )


def upgraded_snapshots(repo: Path, artifact_repo: Path) -> dict[Tour, ModelSnapshot]:
    operational = artifact_repo / "artifacts/live-usopen-2026/official-2117-v1"
    snapshots: dict[Tour, ModelSnapshot] = {}
    for tour in Tour:
        amended = ModelSnapshot.from_json(
            (operational / f"model_snapshot_{tour.value.lower()}.json").read_bytes()
        )
        if amended.retirement_artifact is None or amended.inactivity_configuration is None:
            raise RuntimeError(f"{tour.value} amended snapshot is missing B6/C6")
        snapshots[tour] = amended
    return snapshots


def schedule_matches(payload: bytes) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    schedule = json.loads(payload)
    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    requested = {str(item["official_id"]) for item in MATCHES}
    for court in schedule["courts"]:
        for match in court.get("matches", []):
            match_id = str(match.get("match_id"))
            if match_id in requested:
                selected[match_id] = (court, match)
    if set(selected) != requested:
        raise RuntimeError("official schedule does not contain every requested match")
    return selected


def scheduled_start(court: dict[str, Any], match: dict[str, Any]) -> datetime:
    not_before = match.get("notBefore")
    if not_before:
        local = datetime.strptime(f"2026-08-30 {not_before}", "%Y-%m-%d %I:%M %p").replace(
            tzinfo=NY
        )
        return local.astimezone(UTC)
    return datetime.fromtimestamp(int(court["startEpoch"]), tz=UTC)


def inactivity_information(
    player_key: str,
    *,
    manifest_id: str,
    manifest_sha256: str,
    cutoff: datetime,
    captured: dict[str, dict[str, Any]],
) -> PlayerInactivityInformation:
    player = PLAYERS[player_key]
    source = captured[f"c6_{player_key}"]
    player_id = str(player["id"])
    candidates = [
        InactivityMatchCandidate(
            player_id=player_id,
            identity_resolved=True,
            tour=player["tour"],
            match_id=str(player["latest_match_id"]),
            match_date_local=player["latest_date"],
            discipline="singles",
            competition_class=player["competition"],
            terminal_status=InactivityTerminalStatus.NORMAL_COMPLETION,
            started_evidence=(
                PlayedPointEvidence.LEGAL_SCORE_WITH_COMPLETED_GAME_OR_TIEBREAK,
            ),
            source_manifest_id=manifest_id,
            source_pin=f"official-current-activity-{player_key}",
            source_sha256=source["sha256"],
            available_at_utc=source["retrieved_at_utc"],
        )
    ]
    if player_key == "svitolina":
        candidates.append(
            InactivityMatchCandidate(
                player_id=player_id,
                identity_resolved=True,
                tour=Tour.WTA,
                match_id="wta-cincinnati-2026-svitolina-wang-r32",
                match_date_local=date(2026, 8, 18),
                discipline="singles",
                competition_class=CompetitionClass.MAIN_DRAW,
                terminal_status=InactivityTerminalStatus.PRE_START_WITHDRAWAL,
                started_evidence=(),
                source_manifest_id=manifest_id,
                source_pin="official-current-activity-svitolina",
                source_sha256=source["sha256"],
                available_at_utc=source["retrieved_at_utc"],
            )
        )
    return PlayerInactivityInformation(
        player_id=player_id,
        coverage=InactivityCoverageAssertion(
            state=InactivityCoverageState.VERIFIED_COMPLETE,
            source_manifest_id=manifest_id,
            source_manifest_sha256=manifest_sha256,
            canonical_player_id=player_id,
            asserted_at_utc=cutoff - timedelta(microseconds=1),
        ),
        candidates=tuple(candidates),
    )


def retained_artifacts(
    repo: Path,
    artifact_repo: Path,
    output: Path,
    tour: Tour,
    snapshot: ModelSnapshot,
    capture: Path,
    code_zip: Path,
    eligibility_path: Path,
) -> tuple[RetainedArtifactRecord, ...]:
    base = artifact_repo / "artifacts/current-usopen-2026" / BASE_RUN_ID
    source_operational = artifact_repo / "artifacts/live-usopen-2026/official-2117-v1"
    if snapshot.retirement_artifact is None:
        raise RuntimeError("production snapshot lost B6")
    counts_receipt = output / "retained" / f"component_counts_{tour.value.lower()}.json"
    write_immutable(
        counts_receipt,
        canonical_bytes(
            {
                "component_count_artifact_hash": snapshot.component_count_artifact_hash,
                "current_component_counts_sha256": sha256_file(
                    base / "data/current_component_counts.parquet"
                ),
                "training_eligibility_sha256": sha256_file(eligibility_path),
            }
        ),
    )
    return (
        retained_record("source_snapshot", capture),
        retained_record("normalized_snapshot", eligibility_path),
        retained_record("component_counts", counts_receipt),
        retained_record("component_fit", base / "fits" / tour.value.lower()),
        retained_record("retirement_fit", snapshot.retirement_artifact.directory),
        retained_record("inactivity_config", source_operational / "retained/inactivity_config.json"),
        retained_record("model_config", repo / "config/model_v1.yaml"),
        retained_record("settlement_policy", source_operational / "retained/settlement_policy.json"),
        retained_record("code_archive", code_zip),
    )


def run(repo: Path) -> Path:
    # The full source is isolated in a worktree while retained artifacts remain
    # at the shared workspace root; avoid relying on Windows reparse traversal.
    artifact_repo = repo.parent
    output = artifact_repo / "artifacts/live-usopen-2026/navarro-boisson-preliminary-5k-v1"
    selected_matches = MATCHES[:1]
    preliminary_policy = replace(ADAPTIVE_MC_CS_V1_POLICY, checkpoints=(5_000,))
    output.mkdir(parents=True, exist_ok=True)
    readable = (
        artifact_repo
        / "artifacts/live-usopen-2026/remaining-four-adaptive-v1/source-captures-readable"
    )
    retained_captures = tuple(sorted(readable.glob("*/manifest.json")))
    if retained_captures:
        capture, captured, cutoff = load_captured_sources(retained_captures[-1].parent)
    else:
        capture, captured, cutoff = acquire_sources(output)
    selected = schedule_matches(captured["schedule"]["payload"])
    for item in selected_matches:
        court, scheduled = selected[str(item["official_id"])]
        if str(scheduled.get("statusCode")) != "B":
            raise RuntimeError(
                f"official match {item['official_id']} is no longer unstarted: "
                f"status={scheduled.get('statusCode')}"
            )
        start = scheduled_start(court, scheduled)
        if cutoff >= start:
            raise RuntimeError(
                f"official match {item['official_id']} cannot be locked after its schedule anchor"
            )

    base = artifact_repo / "artifacts/current-usopen-2026" / BASE_RUN_ID
    source_manifest = load_source_manifest(base / "source_manifest.yaml")
    manifest_pin = SourceManifestProvenance.from_manifest(source_manifest)
    snapshots = upgraded_snapshots(repo, artifact_repo)
    source_operational = artifact_repo / "artifacts/live-usopen-2026/official-2117-v1"
    eligibility: dict[Tour, HistoricalTrainingEligibilityProvenance] = {
        tour: HistoricalTrainingEligibilityProvenance.model_validate_json(
            (source_operational / f"training_eligibility_{tour.value.lower()}.json").read_bytes()
        )
        for tour in Tour
    }
    for tour, snapshot in snapshots.items():
        write_immutable(
            output / f"model_snapshot_{tour.value.lower()}_v2.json",
            snapshot.canonical_json().encode("utf-8"),
        )

    code_zip = code_archive(repo, output / "retained/code")
    code = capture_code_provenance(repo)
    store = LockStore(output / "locks")
    schedule_source = captured["schedule"]
    schedule_payload = json.loads(schedule_source["payload"])
    schedule_observed = min(
        datetime.fromtimestamp(int(schedule_payload["epoch"]), tz=UTC),
        schedule_source["retrieved_at_utc"],
    )
    results: list[dict[str, Any]] = []
    for item in selected_matches:
        official_id = str(item["official_id"])
        tour: Tour = item["tour"]
        left = PLAYERS[str(item["a"])]
        right = PLAYERS[str(item["b"])]
        court, scheduled = selected[official_id]
        start = scheduled_start(court, scheduled)
        context = MatchContext(
            player_a_id=str(left["id"]),
            player_b_id=str(right["id"]),
            tour=tour,
            event="US Open",
            round="R128",
            scheduled_start_utc=start,
            scheduled_start_local_date=date(2026, 8, 30),
            best_of=5 if tour is Tour.ATP else 3,
            indoor=None,
            conditions=(
                MatchCondition(name="court", value=str(court["courtName"])),
                MatchCondition(name="official_match_id", value=official_id),
                MatchCondition(name="court_order", value=int(scheduled["order"])),
                MatchCondition(
                    name="schedule_anchor_semantics",
                    value=(
                        "not-before"
                        if scheduled.get("notBefore")
                        else "session-start; later ordered matches follow"
                    ),
                ),
            ),
            information_cutoff_utc=cutoff,
            information_scenario_id="central",
        )
        c6 = tuple(
            inactivity_information(
                key,
                manifest_id=source_manifest.manifest_version,
                manifest_sha256=manifest_pin.manifest_sha256,
                cutoff=cutoff,
                captured=captured,
            )
            for key in (str(item["a"]), str(item["b"]))
        )
        information = InformationBundle(
            bundle_id=f"usopen-2026-{official_id}-{capture.name[:16]}",
            scenario_id="central",
            information_cutoff_utc=cutoff,
            items=(
                InformationItem(
                    category="schedule",
                    summary=(
                        f"Official US Open schedule match {official_id}: "
                        f"{left['name']} vs {right['name']}; status B at capture"
                    ),
                    source_id="official-usopen-2026-schedule-day-8",
                    source_sha256=schedule_source["sha256"],
                    observed_at_utc=schedule_observed,
                    available_at_utc=schedule_source["retrieved_at_utc"],
                ),
                *(
                    InformationItem(
                        category="workload",
                        summary=(
                            f"Official activity evidence for {PLAYERS[key]['name']}: latest "
                            f"eligible singles match {PLAYERS[key]['latest_date'].isoformat()}"
                        ),
                        source_id=f"official-current-activity-{key}",
                        source_sha256=captured[f"c6_{key}"]["sha256"],
                        observed_at_utc=captured[f"c6_{key}"]["retrieved_at_utc"],
                        available_at_utc=captured[f"c6_{key}"]["retrieved_at_utc"],
                    )
                    for key in (str(item["a"]), str(item["b"]))
                ),
            ),
            player_inactivity=c6,
            missing_current_conditions=("roof state",),
        )
        canonical = CanonicalMatchIdentity.from_official_id(
            source_namespace="usopen-official",
            tour=tour,
            official_match_id=official_id,
            participant_ids=(str(left["id"]), str(right["id"])),
            source_id="official-usopen-2026-schedule-day-8",
            source_sha256=schedule_source["sha256"],
            source_locator=SCHEDULE_URL,
            resolved_at_utc=schedule_source["retrieved_at_utc"],
        )
        eligibility_path = (
            source_operational / f"training_eligibility_{tour.value.lower()}.json"
        )
        artifacts = retained_artifacts(
            repo,
            artifact_repo,
            output,
            tour,
            snapshots[tour],
            capture,
            code_zip,
            eligibility_path,
        )
        existing = store.revision_directory(canonical.base_lock_id, 1)
        if existing.exists():
            lock = store.load(canonical.base_lock_id, 1).lock
        else:
            lock = create_prediction_lock(
                snapshots[tour],
                context,
                information,
                (MATCH_WIN(str(left["id"])),),
                CANONICAL_SETTLEMENT_POLICY,
                source_manifest=source_manifest,
                code=code,
                seed=202608300000 + int(official_id),
                store=store,
                execution_mode="development",
                path_count_policy=preliminary_policy,
                allow_dirty=True,
                canonical_match_identity=canonical,
                retained_artifacts=artifacts,
                training_eligibility=eligibility[tour],
            )
        verified = store.verify(lock.base_lock_id, lock.revision)
        player_names = {str(left["id"]): str(left["name"]), str(right["id"]): str(right["name"])}
        estimate = lock.prop_estimates[0]
        results.append(
            {
                "official_match_id": official_id,
                "match": f"{left['name']} vs {right['name']}",
                "tour": tour.value,
                "schedule_status_at_cutoff": scheduled["statusCode"],
                "court": court["courtName"],
                "scheduled_start_utc": start.isoformat(),
                "lock_id": lock.lock_id,
                "revision": lock.revision,
                "content_sha256": lock.content_sha256,
                "verified_sha256": verified,
                "snapshot_id": lock.match_parameters.snapshot_id,
                "paths": lock.simulation.actual_paths,
                "inspected_path_counts": lock.simulation.inspected_path_counts,
                "mc_stopping_status": (
                    None
                    if estimate.mc_stopping_status is None
                    else estimate.mc_stopping_status.value
                ),
                "mc_error": estimate.mc_standard_error,
                "match_win_probability": {
                    player_names[row.player_id]: row.match_win_probability
                    for row in lock.match_summary.players
                },
                "exact_score_probability": [
                    {
                        "winner": player_names[row.winner_id],
                        "score": f"{row.winner_sets}-{row.loser_sets}",
                        "probability": row.probability,
                    }
                    for row in lock.match_summary.exact_scores
                ],
                "any_tiebreak_probability": lock.match_summary.any_tiebreak_probability,
                "deciding_set_probability": lock.match_summary.deciding_set_probability,
                "expected_total_games": lock.match_summary.expected_total_games,
                "total_games_quantiles": lock.match_summary.total_games_quantiles,
                "players": [
                    {
                        "name": player_names[row.player_id],
                        "expected_aces": row.expected_aces,
                        "expected_double_faults": row.expected_double_faults,
                        "expected_breaks": row.expected_breaks,
                        "retirement_probability": row.retirement_probability,
                        "first_serve_win_expectation": next(
                            direction.first_serve_win
                            for direction in lock.parameter_summaries
                            if direction.server_id == row.player_id
                        ),
                    }
                    for row in lock.match_summary.players
                ],
                "duration": (
                    None
                    if lock.match_summary.duration is None
                    else lock.match_summary.duration.model_dump(mode="json")
                ),
                "c6": [
                    {
                        "player": player_names[row.player_id],
                        "latest_match_date": (
                            None
                            if row.last_eligible_match is None
                            else row.last_eligible_match.match_date_local.isoformat()
                        ),
                        "inactivity_days": row.inactivity_days,
                        "band": row.band.value,
                    }
                    for row in lock.match_parameters.inactivity.records
                ],
                "warnings": lock.warnings,
                "card_path": str(
                    store.revision_directory(lock.base_lock_id, lock.revision) / "card.md"
                ),
            }
        )

    report = {
        "schema_version": "remaining-four-live-simulation/v1",
        "batch_information_cutoff_utc": cutoff.isoformat(),
        "official_source_capture_id": capture.name,
        "methodology_changed": False,
        "refit_performed": False,
        "snapshot_staleness_deviation": (
            "The required same-day snapshot refresh could not initialize under host memory "
            "pressure. Locks use the latest previously valid 2026 US Open snapshots at "
            "2026-08-30T12:13:53.667208Z and therefore omit later completed main-draw matches."
        ),
        "adaptive_policy": asdict(preliminary_policy),
        "status": "PRELIMINARY_5K_NOT_PRODUCTION_LOCK",
        "matches": results,
    }
    report_path = output / f"batch-report-{capture.name[:16]}.json"
    write_immutable(report_path, canonical_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report_path


if __name__ == "__main__":
    run(Path.cwd().resolve())
