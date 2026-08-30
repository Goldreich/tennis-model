"""Acquire, fit, and persist the current 2026 US Open serve/B6 snapshot.

This is an operational build command, not a backtest.  It admits historical
rows only after the retained exact-date crosswalk resolves the match date and
adds official completed US Open singles with their exact completion timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_model.data.artifacts import read_processed_table, write_processed_bundle
from tennis_model.data.current_usopen import (
    OfficialJsonObject,
    build_official_player_crosswalk,
    normalize_completed_singles,
)
from tennis_model.data.cutoff import InformationCutoff
from tennis_model.data.exact_date_crosswalk import (
    ExactDateCrosswalkManifest,
    ExactDateCrosswalkResult,
)
from tennis_model.data.historical_validation import load_historical_validation_policy
from tennis_model.data.ingest_sackmann import ingest_sackmann_snapshot
from tennis_model.data.snapshot import verify_snapshot
from tennis_model.data.source_manifest import (
    dump_source_manifest,
    load_source_manifest,
    manifest_sha256,
)
from tennis_model.estimation.artifacts import write_fit_artifact
from tennis_model.estimation.config import load_serve_model_config
from tennis_model.estimation.inactivity import (
    CompetitionClass,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityMatchCandidate,
    InactivityTerminalStatus,
    PlayedPointEvidence,
    create_inactivity_configuration_artifact,
)
from tennis_model.estimation.retirement import (
    HistoricalTerminationInput,
    OfficialTerminalStatus,
    RetirementSourceCoverage,
    StartedEvidence,
    build_retirement_observations,
    fit_retirement_artifact,
    normalize_historical_termination,
    write_retirement_fit_artifact,
)
from tennis_model.estimation.serve_components import (
    FitProvenance,
    ServeComponent,
    fit_all_serve_components,
    fit_input_set_sha256,
)
from tennis_model.estimation.snapshot import create_model_snapshot
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.models import (
    InformationBundle,
    InformationItem,
    PlayerInactivityInformation,
    SourceManifestProvenance,
)
from tennis_model.locking.path_counts import PathCountPolicy
from tennis_model.locking.provenance import capture_code_provenance
from tennis_model.locking.service import LockCreationError, create_prediction_lock
from tennis_model.props import CANONICAL_SETTLEMENT_POLICY
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RawSourceSnapshot,
    RowDateSemantics,
    SourceManifest,
    Tour,
    TourCoverage,
)
from tennis_model.simulation import MATCH_WIN
from tennis_model.simulation.parameters import MatchContext

_USOPEN = "https://www.usopen.org"
_CONFIG_URL = f"{_USOPEN}/en_US/json/gen/config_web.json"
_EVENT_DAYS_URL = f"{_USOPEN}/en_US/scores/feeds/2026/completed_matches/eventDays.json"
_SCHEDULE_URL = f"{_USOPEN}/en_US/scores/feeds/2026/schedule/schedule8.json"
_WTA_PEGULA_RECORD = "https://www.wtatennis.com/players/316956/jessica-pegula/record"
_WTA_RUSE_RECORD = "https://www.wtatennis.com/players/320408/elena-gabriela-ruse/record"
_WTA_PEGULA_LATEST = "https://www.wtatennis.com/tournaments/cincinnati-open/scores/LS001"
_WTA_RUSE_LATEST = "https://www.wtatennis.com/tournaments/1017/cincinnati/2026/scores/LS050"
_TARGET_MATCH_ID = "2117"
_USER_AGENT = "TennisModel-v1.0 current snapshot provenance capture"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(locator: str) -> tuple[bytes, datetime]:
    request = urllib.request.Request(locator, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    return payload, datetime.now(UTC)


def _capture_object(locator: str) -> dict[str, Any]:
    payload, retrieved = _fetch(locator)
    return {
        "locator": locator,
        "retrieved_at_utc": retrieved.isoformat(),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "payload": payload,
    }


def acquire_official_capture(raw_root: Path) -> Path:
    """Retain one immutable official feed capture and return its directory."""

    objects: dict[str, dict[str, Any]] = {
        "config": _capture_object(_CONFIG_URL),
        "event_days": _capture_object(_EVENT_DAYS_URL),
        "schedule_day_8": _capture_object(_SCHEDULE_URL),
        "wta_pegula_record": _capture_object(_WTA_PEGULA_RECORD),
        "wta_ruse_record": _capture_object(_WTA_RUSE_RECORD),
        "wta_pegula_latest_match": _capture_object(_WTA_PEGULA_LATEST),
        "wta_ruse_latest_match": _capture_object(_WTA_RUSE_LATEST),
    }
    event_days = json.loads(objects["event_days"]["payload"])["eventDays"]
    current_tournament_day = max(1, (datetime.now(UTC).date() - date(2026, 8, 22)).days)
    day_entries = [
        item
        for item in event_days
        if int(item["tournDay"]) <= current_tournament_day and int(item["tournDay"]) <= 22
    ]
    with ThreadPoolExecutor(max_workers=6) as executor:
        day_results = list(executor.map(lambda item: _capture_object(item["url"]), day_entries))
    for item, captured in zip(day_entries, day_results, strict=True):
        objects[f"completed_day_{int(item['tournDay'])}"] = captured

    match_locators: dict[str, str] = {}
    for key, captured in objects.items():
        if not key.startswith("completed_day_"):
            continue
        for match in json.loads(captured["payload"])["matches"]:
            if match.get("eventCode") not in {"MQ", "WQ", "MS", "WS"}:
                continue
            if str(match.get("statusCode")) not in {"D", "E"}:
                continue
            match_id = str(match["match_id"])
            match_locators[match_id] = (
                f"{_USOPEN}/en_US/scores/feeds/2026/matches/complete/{match_id}.json"
            )
    with ThreadPoolExecutor(max_workers=8) as executor:
        match_results = list(executor.map(_capture_object, match_locators.values()))
    for match_id, captured in zip(match_locators, match_results, strict=True):
        objects[f"complete_match_{match_id}"] = captured

    manifest_objects: dict[str, dict[str, Any]] = {}
    for key in sorted(objects):
        captured = objects[key]
        suffix = ".json" if key not in {
            "wta_pegula_record",
            "wta_ruse_record",
            "wta_pegula_latest_match",
            "wta_ruse_latest_match",
        } else ".html"
        manifest_objects[key] = {
            name: captured[name]
            for name in ("locator", "retrieved_at_utc", "sha256", "size_bytes")
        }
        manifest_objects[key]["relative_path"] = f"objects/{key}{suffix}"
    manifest_payload = {
        "schema_version": "official-usopen-current-capture/v1",
        "event_year": 2026,
        "objects": manifest_objects,
    }
    manifest_bytes = _canonical_bytes(manifest_payload)
    capture_id = _sha256_bytes(manifest_bytes)
    parent = raw_root.resolve() / "usopen-2026-current"
    target = parent / capture_id
    if target.exists():
        if (target / "manifest.json").read_bytes() != manifest_bytes:
            raise RuntimeError("existing current capture has conflicting manifest bytes")
        return target
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
    try:
        (staging / "objects").mkdir()
        for key, receipt in manifest_objects.items():
            path = staging / receipt["relative_path"]
            path.write_bytes(objects[key]["payload"])
            if _sha256_file(path) != receipt["sha256"]:
                raise RuntimeError(f"staged official object changed: {key}")
        (staging / "manifest.json").write_bytes(manifest_bytes)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def _load_capture(capture: Path) -> tuple[dict[str, Any], dict[str, OfficialJsonObject]]:
    manifest = json.loads((capture / "manifest.json").read_bytes())
    if _sha256_bytes((capture / "manifest.json").read_bytes()) != capture.name:
        raise RuntimeError("official capture directory does not match its manifest hash")
    objects: dict[str, OfficialJsonObject] = {}
    for key, receipt in manifest["objects"].items():
        payload = (capture / receipt["relative_path"]).read_bytes()
        objects[key] = OfficialJsonObject(
            source_id=f"official-usopen-2026-{key}",
            locator=receipt["locator"],
            retrieved_at_utc=datetime.fromisoformat(receipt["retrieved_at_utc"]),
            payload=payload,
            sha256=receipt["sha256"],
        )
    return manifest, objects


def _raw_snapshot(source: PinnedSource, raw_root: Path) -> RawSourceSnapshot:
    directory = raw_root / source.tour.value.lower() / source.source_id / source.sha256
    payload = directory / "payload"
    snapshot = RawSourceSnapshot(
        source=source,
        payload_path=payload,
        provenance_path=directory / "source.json",
        size_bytes=payload.stat().st_size,
        sha256=source.sha256,
    )
    verify_snapshot(snapshot)
    return snapshot


def _load_crosswalk(root: Path, tour: Tour, year: int) -> ExactDateCrosswalkResult:
    stem = f"{tour.value.lower()}_{year}"
    detail = pd.read_csv(root / f"crosswalk_{stem}.csv", keep_default_na=False)
    manifest = ExactDateCrosswalkManifest.model_validate_json(
        (root / f"manifest_{stem}.json").read_bytes()
    )
    return ExactDateCrosswalkResult(detail=detail, manifest=manifest)


def _prepare_historical(
    repo: Path,
    cutoff: datetime,
) -> tuple[
    SourceManifest,
    dict[Tour, list[Any]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
]:
    manifest = load_source_manifest(repo / "config/sources.yaml")
    policy = load_historical_validation_policy(
        repo / "config/historical_validation_retrospective_finalized_v1.yaml"
    )
    crosswalk_root = repo / "data/processed/retrospective-finalized-crosswalk-v1"
    bundle_root = repo / "data/processed/current-usopen-2026-exact-date"
    cutoff_record = InformationCutoff(at_utc=cutoff)
    bundles: dict[Tour, list[Any]] = {Tour.ATP: [], Tour.WTA: []}
    service_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    exclusions: Counter[str] = Counter()
    for source in manifest.sources:
        year = int(source.source_id.rsplit("-", 1)[1])
        if year < 2021:
            continue
        result = ingest_sackmann_snapshot(
            _raw_snapshot(source, repo / "data/raw"),
            cutoff=cutoff_record,
            historical_validation_policy=policy,
            exact_date_crosswalk=_load_crosswalk(crosswalk_root, source.tour, year),
        )
        bundle = write_processed_bundle(result, bundle_root)
        bundles[source.tour].append(bundle)
        service = read_processed_table(bundle, "service_rows")
        service_frames.append(service)
        if year >= 2023:
            counts = read_processed_table(bundle, "component_counts")
            counts["component_count_artifact_sha256"] = bundle.manifest.receipt_for(
                "component_counts"
            ).sha256
            count_frames.append(counts)
        for reason, rows in result.cutoff_exclusions.groupby("cutoff_exclusion_code"):
            exclusions[f"{source.tour.value}:{year}:{reason}"] += len(rows)
    return (
        manifest,
        bundles,
        pd.concat(service_frames, ignore_index=True),
        pd.concat(count_frames, ignore_index=True),
        dict(sorted(exclusions.items())),
    )


def _combined_manifest(
    historical: SourceManifest,
    capture: Path,
    current_rows: pd.DataFrame,
    cutoff: datetime,
    capture_retrieved_at_utc: datetime,
) -> SourceManifest:
    capture_hash = capture.name
    current_sources: list[PinnedSource] = []
    for tour in Tour:
        rows = current_rows.loc[current_rows["tour"].eq(tour.value)]
        first = min(rows["match_date"])
        last = max(rows["match_date"])
        current_sources.append(
            PinnedSource(
                source_id=f"official-usopen-2026-{tour.value.lower()}-current",
                identity_namespace="usopen-official",
                tour=tour,
                upstream_attribution="United States Tennis Association / official US Open",
                locator=str((capture / "manifest.json").resolve()),
                object_identifier=capture_hash,
                sha256=capture_hash,
                schema_version="official-usopen-current-capture/v1",
                stated_license="Official public scoring data; use subject to US Open site terms",
                retrieved_at_utc=capture_retrieved_at_utc,
                verified_coverage=CoverageRange(
                    first_match_date=first,
                    last_match_date=last,
                    verified_at_utc=cutoff - timedelta(microseconds=1),
                ),
                row_date_semantics=RowDateSemantics.MATCH_DATE,
                availability_lag_days=1,
                source_effective_at_utc=capture_retrieved_at_utc,
                source_available_at_utc=capture_retrieved_at_utc,
            )
        )
    sources = (*historical.sources, *current_sources)

    def coverage(tour: Tour) -> CoverageRange:
        ranges = [item.verified_coverage for item in sources if item.tour is tour]
        return CoverageRange(
            first_match_date=min(item.first_match_date for item in ranges),
            last_match_date=max(item.last_match_date for item in ranges),
            verified_at_utc=cutoff - timedelta(microseconds=1),
        )

    return SourceManifest(
        manifest_version=f"current-usopen-2026-{capture_hash[:16]}/v1",
        sources=sources,
        coverage_by_tour=TourCoverage(atp=coverage(Tour.ATP), wta=coverage(Tour.WTA)),
    )


def _termination_inputs(rows: pd.DataFrame) -> tuple[HistoricalTerminationInput, ...]:
    records: list[HistoricalTerminationInput] = []
    for match_id, match_rows in rows.groupby("match_id", sort=True):
        if len(match_rows) != 2:
            continue
        first = match_rows.iloc[0]
        players = tuple(match_rows["player_id"].astype(str))
        retirement = bool(first["retirement"])
        walkover = bool(first["walkover"])
        orientations = dict(
            zip(match_rows["orientation"].astype(str), players, strict=True)
        )
        winner_id: str | None = orientations.get("winner")
        retiree_id: str | None = orientations.get("loser") if retirement else None
        if winner_id is None:
            raw = json.loads(str(first["raw_record_json"]))
            winner_team = int(raw.get("winner", 0))
            winner_id = orientations.get(f"team{winner_team}")
            retiree_id = orientations.get(f"team{3 - winner_team}") if retirement else None
        evidence: list[StartedEvidence] = []
        points = pd.to_numeric(match_rows["service_points"], errors="coerce")
        if bool(points.fillna(0).gt(0).any()):
            evidence.append(StartedEvidence.POSITIVE_POINT_STAT_COUNT)
        if not walkover and bool(first["completed"]):
            evidence.append(StartedEvidence.LEGAL_SCORE_COMPLETED_GAME)
        if walkover:
            status = OfficialTerminalStatus.WALKOVER_OR_PRESTART_WITHDRAWAL
        elif retirement:
            status = OfficialTerminalStatus.RETIREMENT
        else:
            status = OfficialTerminalStatus.NORMAL_COMPLETION
        records.append(
            HistoricalTerminationInput(
                match_id=str(match_id),
                tour=Tour(str(first["tour"])),
                player_a_id=players[0],
                player_b_id=players[1],
                match_date=first["match_date"],
                official_status=status,
                started_evidence=tuple(sorted(set(evidence), key=lambda item: item.value)),
                retiring_player_id=retiree_id,
                advancing_winner_id=winner_id,
                source_id=str(first["source_id"]),
                source_sha256=str(first["snapshot_sha256"]),
                available_at_utc=pd.Timestamp(first["retrieved_at_utc"]).to_pydatetime(),
            )
        )
    return tuple(records)


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return _sha256_file(path)


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build(repo: Path, capture: Path, output_root: Path, deterministic_test_hash: str) -> Path:
    capture_manifest, official = _load_capture(capture)
    latest_retrieval = max(item.retrieved_at_utc for item in official.values())
    cutoff = max(datetime.now(UTC), latest_retrieval + timedelta(microseconds=1))
    historical_manifest, _bundles, historical_rows, historical_counts, date_exclusions = (
        _prepare_historical(repo, cutoff)
    )
    day_objects = tuple(
        value for key, value in official.items() if key.startswith("completed_day_")
    )
    match_objects = {
        key.removeprefix("complete_match_"): value
        for key, value in official.items()
        if key.startswith("complete_match_")
    }
    current = normalize_completed_singles(
        day_objects,
        match_objects,
        historical_identity_rows=historical_rows,
        information_cutoff_utc=cutoff,
    )
    if current.included_match_count == 0:
        raise RuntimeError("official capture contains no usable completed singles matches")
    source_manifest = _combined_manifest(
        historical_manifest,
        capture,
        current.rows,
        cutoff,
        latest_retrieval,
    )
    source_manifest_hash = manifest_sha256(source_manifest)
    config = load_serve_model_config(repo / "config/model_v1.yaml")
    code = capture_code_provenance(repo)
    code_hash = code.diff_sha256 or hashlib.sha256(code.commit.encode()).hexdigest()
    fitted_at = datetime.now(UTC)
    run_id = hashlib.sha256(
        _canonical_bytes(
            {
                "capture": capture.name,
                "cutoff": cutoff.isoformat(),
                "config": config.sha256,
                "code": code.model_dump(mode="json"),
            }
        )
    ).hexdigest()
    output = output_root.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_manifest.yaml").write_text(
        dump_source_manifest(source_manifest), encoding="utf-8"
    )
    (output / "official_capture_manifest.json").write_bytes(
        (capture / "manifest.json").read_bytes()
    )
    _write_frame(current.rows, output / "data/current_service_rows.parquet")
    current_count_hash = _write_frame(
        current.counts.counts, output / "data/current_component_counts.parquet"
    )
    _write_frame(current.counts.anomalies, output / "data/current_component_anomalies.parquet")
    _write_frame(current.identity_crosswalk, output / "data/current_identity_crosswalk.parquet")
    _write_frame(current.exclusions, output / "data/current_exclusions.parquet")
    current_counts = current.counts.counts.copy()
    current_counts["component_count_artifact_sha256"] = current_count_hash
    all_counts = pd.concat((historical_counts, current_counts), ignore_index=True)
    all_service = pd.concat((historical_rows, current.rows), ignore_index=True)

    fits_by_tour: dict[Tour, dict[ServeComponent, Any]] = {}
    snapshots: dict[Tour, Any] = {}
    retirement_artifacts: dict[Tour, Any] = {}
    input_snapshots: dict[Tour, str] = {}
    input_count_artifacts: dict[Tour, str] = {}
    for tour in Tour:
        selected_counts = all_counts.loc[all_counts["tour"].eq(tour.value)].copy()
        input_snapshots[tour] = fit_input_set_sha256(
            "source_snapshots", selected_counts["snapshot_sha256"].astype(str)
        )
        input_count_artifacts[tour] = fit_input_set_sha256(
            "component_count_artifacts",
            selected_counts["component_count_artifact_sha256"].astype(str),
        )
        fits = fit_all_serve_components(
            selected_counts,
            tour=tour,
            cutoff=cutoff,
            config=config,
            provenance=FitProvenance(
                data_snapshot_sha256=input_snapshots[tour],
                component_count_artifact_sha256=input_count_artifacts[tour],
                code_commit=_git_commit(repo),
                fitted_at_utc=fitted_at,
            ),
        )
        fits_by_tour[tour] = {
            component: write_fit_artifact(fit, output / "fits")
            for component, fit in fits.items()
        }

    classified = tuple(
        normalize_historical_termination(item) for item in _termination_inputs(all_service)
    )
    retirement_batch = build_retirement_observations(classified, cutoff)
    retirement_data_hash = hashlib.sha256(
        _canonical_bytes(retirement_batch.model_dump(mode="json"))
    ).hexdigest()
    unresolved_total = sum(date_exclusions.values())
    for tour in Tour:
        retirement = fit_retirement_artifact(
            retirement_batch,
            tour=tour,
            source_manifest_id=source_manifest.manifest_version,
            source_manifest_sha256=source_manifest_hash,
            source_coverage=RetirementSourceCoverage(
                tour=tour,
                complete=False,
                assertion_id=f"exact-date-residuals-{capture.name[:16]}-{tour.value.lower()}",
                verified_at_utc=cutoff - timedelta(microseconds=1),
                details=(
                    "All uniquely exact-dated rows were processed; unresolved exact-date rows "
                    f"remain excluded and reported. Aggregate exclusions={unresolved_total}."
                ),
            ),
            fitted_at_utc=fitted_at,
            software_version="tennis-model/0.1.0",
            config_sha256=config.sha256,
            data_sha256=retirement_data_hash,
            code_sha256=code_hash,
            deterministic_test_result_sha256=deterministic_test_hash,
        )
        persisted_retirement = write_retirement_fit_artifact(
            retirement, output / "retirement_fits"
        )
        retirement_artifacts[tour] = persisted_retirement
        inactivity_config = create_inactivity_configuration_artifact(
            config_sha256=config.sha256, code_sha256=code_hash
        )
        snapshot = create_model_snapshot(
            fits_by_tour[tour],
            retirement_artifact=persisted_retirement,
            inactivity_configuration=inactivity_config,
        )
        snapshots[tour] = snapshot
        (output / f"model_snapshot_{tour.value.lower()}.json").write_text(
            snapshot.canonical_json(), encoding="utf-8"
        )

    schedule = official["schedule_day_8"].json()
    target: dict[str, Any] | None = None
    target_start: int | None = None
    for court in schedule["courts"]:
        for match in court["matches"]:
            if str(match.get("match_id")) == _TARGET_MATCH_ID:
                target = match
                target_start = int(court["startEpoch"])
                break
    if target is None or target_start is None:
        raise RuntimeError("official schedule capture lacks target match 2117")
    target_players = tuple(
        (
            Tour.WTA,
            str(target[key][0]["idA"]),
            f"{target[key][0]['firstNameA']} {target[key][0]['lastNameA']}",
        )
        for key in ("team1", "team2")
    )
    target_map, target_crosswalk = build_official_player_crosswalk(
        historical_rows, target_players
    )
    _write_frame(target_crosswalk, output / "lock/target_identity_crosswalk.parquet")
    pegula_id = target_map[(Tour.WTA, "wta316956")]
    ruse_id = target_map[(Tour.WTA, "wta320408")]
    schedule_source = official["schedule_day_8"]
    canonical_match = CanonicalMatchIdentity.from_official_id(
        source_namespace="usopen-official",
        tour=Tour.WTA,
        official_match_id=_TARGET_MATCH_ID,
        participant_ids=(pegula_id, ruse_id),
        source_id=schedule_source.source_id,
        source_sha256=schedule_source.sha256,
        source_locator=schedule_source.locator,
        resolved_at_utc=schedule_source.retrieved_at_utc,
    )
    match_source_by_player = {
        pegula_id: (
            official["wta_pegula_latest_match"],
            date(2026, 8, 23),
            "LS001",
            InactivityTerminalStatus.NORMAL_COMPLETION,
        ),
        ruse_id: (
            official["wta_ruse_latest_match"],
            date(2026, 8, 15),
            "LS050",
            InactivityTerminalStatus.STARTED_RETIREMENT,
        ),
    }
    manifest_pin = SourceManifestProvenance.from_manifest(source_manifest)
    player_inactivity: list[PlayerInactivityInformation] = []
    for player_id, (
        latest_source,
        latest_date,
        latest_id,
        terminal_status,
    ) in match_source_by_player.items():
        player_inactivity.append(
            PlayerInactivityInformation(
                player_id=player_id,
                coverage=InactivityCoverageAssertion(
                    state=InactivityCoverageState.VERIFIED_COMPLETE,
                    source_manifest_id=source_manifest.manifest_version,
                    source_manifest_sha256=manifest_pin.manifest_sha256,
                    canonical_player_id=player_id,
                    asserted_at_utc=cutoff - timedelta(microseconds=1),
                ),
                candidates=(
                    InactivityMatchCandidate(
                        player_id=player_id,
                        identity_resolved=True,
                        tour=Tour.WTA,
                        match_id=f"wta-cincinnati-2026-{latest_id}",
                        match_date_local=latest_date,
                        discipline="singles",
                        competition_class=CompetitionClass.MAIN_DRAW,
                        terminal_status=terminal_status,
                        started_evidence=(PlayedPointEvidence.POSITIVE_POINT_STAT_COUNT,),
                        source_manifest_id=source_manifest.manifest_version,
                        source_pin=latest_source.source_id,
                        source_sha256=latest_source.sha256,
                        available_at_utc=latest_source.retrieved_at_utc,
                    ),
                ),
            )
        )
    information = InformationBundle(
        bundle_id=f"usopen-2026-2117-{capture.name[:16]}",
        scenario_id="central",
        information_cutoff_utc=cutoff,
        items=(
            InformationItem(
                category="schedule",
                summary="Official US Open Day 1 schedule: Pegula vs Ruse, match 2117",
                source_id=schedule_source.source_id,
                source_sha256=schedule_source.sha256,
                observed_at_utc=datetime.fromtimestamp(int(schedule["epoch"]), tz=UTC),
                available_at_utc=schedule_source.retrieved_at_utc,
            ),
        ),
        player_inactivity=tuple(player_inactivity),
        missing_current_conditions=("roof state",),
    )
    (output / "lock/information_bundle.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "lock/information_bundle.json").write_bytes(
        _canonical_bytes(information.model_dump(mode="json"))
    )
    (output / "lock/canonical_match_identity.json").write_bytes(
        _canonical_bytes(canonical_match.model_dump(mode="json"))
    )
    context = MatchContext(
        player_a_id=pegula_id,
        player_b_id=ruse_id,
        tour=Tour.WTA,
        event="US Open",
        round="R128",
        scheduled_start_utc=datetime.fromtimestamp(target_start, tz=UTC),
        scheduled_start_local_date=date(2026, 8, 30),
        best_of=3,
        indoor=None,
        information_cutoff_utc=cutoff,
        information_scenario_id="central",
    )
    lock_attempt: dict[str, Any]
    retrospective_policy = load_historical_validation_policy(
        repo / "config/historical_validation_retrospective_finalized_v1.yaml"
    )
    try:
        lock = create_prediction_lock(
            snapshots[Tour.WTA],
            context,
            information,
            (MATCH_WIN(pegula_id),),
            CANONICAL_SETTLEMENT_POLICY,
            source_manifest=source_manifest,
            code=code,
            seed=20260830,
            n_paths=20,
            execution_mode="test",
            path_count_policy=PathCountPolicy(
                standard_paths=20,
                escalated_paths=40,
                minimum_settled_paths=10,
            ),
            allow_dirty=True,
            created_at_utc=datetime.now(UTC),
            canonical_match_identity=canonical_match,
            historical_validation_policy=retrospective_policy,
        )
    except LockCreationError as exc:
        expected = "retrospective-finalized lock blocked: exact-date history is incomplete"
        if str(exc) != expected:
            raise RuntimeError(f"unexpected current lock failure: {exc}") from exc
        lock_attempt = {
            "status": "BLOCKED_AS_REQUIRED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "seed": 20260830,
            "requested_paths": 20,
            "snapshot_id": snapshots[Tour.WTA].snapshot_id,
            "context": context.model_dump(mode="json"),
            "information_bundle_id": information.bundle_id,
        }
    else:
        lock_attempt = {
            "status": "CREATED",
            "lock_id": lock.lock_id,
            "snapshot_id": lock.model_snapshot_id,
        }
    (output / "lock/lock_attempt.json").write_bytes(_canonical_bytes(lock_attempt))
    report = {
        "schema_version": "current-usopen-snapshot-build/v1",
        "run_id": run_id,
        "information_cutoff_utc": cutoff.isoformat(),
        "fitted_at_utc": fitted_at.isoformat(),
        "official_capture_id": capture.name,
        "source_manifest_id": source_manifest.manifest_version,
        "source_manifest_sha256": source_manifest_hash,
        "completed_official_singles_seen": current.completed_match_count,
        "completed_official_singles_included": current.included_match_count,
        "completed_main_draw_matches_included": int(
            current.rows.loc[current.rows["level"].eq("G"), "match_id"].nunique()
        ),
        "completed_qualifying_matches_included": int(
            current.rows.loc[current.rows["level"].eq("Q"), "match_id"].nunique()
        ),
        "exact_date_exclusions": date_exclusions,
        "current_match_exclusions": current.exclusions.to_dict(orient="records"),
        "snapshots": {tour.value: snapshots[tour].snapshot_id for tour in Tour},
        "retirement": {
            tour.value: {
                "artifact_id": retirement_artifacts[tour].artifact_id,
                "production_eligible": retirement_artifacts[tour].artifact.production_eligible,
                "weighted_starts": retirement_artifacts[tour].artifact.tour_starts_n,
                "weighted_retirements": retirement_artifacts[tour].artifact.tour_retirements_y,
            }
            for tour in Tour
        },
        "target_lock": {
            "official_match_id": _TARGET_MATCH_ID,
            "canonical_match_id": canonical_match.canonical_match_id,
            "scheduled_start_utc": datetime.fromtimestamp(target_start, tz=UTC).isoformat(),
            "player_ids": [pegula_id, ruse_id],
            "c6_latest_dates": ["2026-08-23", "2026-08-15"],
            "status": lock_attempt["status"],
            "reason": lock_attempt.get("error"),
            "attempt_receipt": "lock/lock_attempt.json",
        },
        "code_provenance": code.model_dump(mode="json"),
        "official_capture_objects": len(capture_manifest["objects"]),
    }
    (output / "build_report.json").write_bytes(_canonical_bytes(report))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/current-usopen-2026"))
    parser.add_argument("--deterministic-test-result-sha256", required=True)
    args = parser.parse_args()
    if len(args.deterministic_test_result_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.deterministic_test_result_sha256
    ):
        parser.error("--deterministic-test-result-sha256 must be 64 lowercase hex characters")
    repo = args.repo.resolve()
    if bool(args.capture) == bool(args.acquire):
        parser.error("choose exactly one of --capture or --acquire")
    capture = (
        acquire_official_capture(repo / "data/raw")
        if args.acquire
        else args.capture.resolve()
    )
    output = build(repo, capture, repo / args.output_root, args.deterministic_test_result_sha256)
    print(output)


if __name__ == "__main__":
    main()
