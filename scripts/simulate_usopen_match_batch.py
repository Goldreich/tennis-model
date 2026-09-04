"""Run configured US Open match simulations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

repo = Path.cwd().resolve()
sys.path.insert(0, str(repo / "src"))
artifact_repo = repo

spec = importlib.util.spec_from_file_location(
    "simulate_remaining_four",
    repo / "scripts" / "simulate_remaining_four.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from tennis_model.estimation.duration_model import (  # noqa: E402
    NEAREST_DURATION_DISPLAY_POLICY,
    UNRESOLVED_DURATION_DISPLAY_POLICY,
)
from tennis_model.estimation.fitness_history import (  # noqa: E402
    assess_game_day_fitness,
    load_main_tour_fitness_history,
)
from tennis_model.estimation.game_day_elo import (  # noqa: E402
    game_day_adjustment,
    load_game_day_elo_fit,
)
from tennis_model.locking import FIXED_100K_V1_POLICY, PathCountPolicy  # noqa: E402
from tennis_model.props.settlement import ComparisonOperator  # noqa: E402
from tennis_model.simulation import (  # noqa: E402
    ACE_COMPARE,
    DF_COMPARE,
    DURATION_MIN,
    TIEBREAK_COUNT,
)
from tennis_model.simulation.r32_props import R32_PROP_BUNDLE  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-file", type=Path)
parser.add_argument("--match-id", action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-capture", type=Path)
parser.add_argument("--base-run-id", required=True)
parser.add_argument("--operational-name", required=True)
parser.add_argument("--policy", choices=("smoke", "fixed"), default="fixed")
parser.add_argument("--smoke-paths", type=int, default=5_000)
parser.add_argument("--seed-offset", type=int, default=0)
parser.add_argument("--workers", type=int, default=12)
parser.add_argument("--checkpoint-paths", type=int, default=5_000)
parser.add_argument(
    "--checkpoint-root",
    type=Path,
    help="Reuse checkpointed paths from this root instead of storing them under --output.",
)
parser.add_argument("--round", default="R128")
parser.add_argument("--schedule-date", type=date.fromisoformat, required=True)
parser.add_argument("--schedule-source-id", required=True)
parser.add_argument(
    "--match-win-side",
    choices=("left", "right"),
    default="left",
)
parser.add_argument(
    "--comparison-side",
    choices=("left", "right"),
    default="right",
)
parser.add_argument("--duration-threshold", type=float, default=155.0)
parser.add_argument("--tiebreak-threshold", type=int, default=2)
parser.add_argument(
    "--artifact-root",
    type=Path,
    help="Optional local root containing staged artifacts; defaults to the repository root.",
)
args = parser.parse_args()
if args.seed_offset < 0:
    parser.error("--seed-offset must be nonnegative")
artifact_repo = args.artifact_root.resolve() if args.artifact_root is not None else repo

if len(args.base_run_id) != 64 or any(char not in "0123456789abcdef" for char in args.base_run_id):
    parser.error("--base-run-id must be 64 lowercase hex characters")
if args.smoke_paths <= 0:
    parser.error("--smoke-paths must be positive")
if args.workers <= 0:
    parser.error("--workers must be positive")
if args.checkpoint_paths <= 0:
    parser.error("--checkpoint-paths must be positive")
if Path(args.operational_name).name != args.operational_name:
    parser.error("--operational-name must be one path component")

fixture = None
if args.fixture_file is not None:
    fixture = json.loads(args.fixture_file.read_bytes())
    mod.SCHEDULE_URL = str(fixture["schedule_url"])
    mod.PLAYERS = {
        key: {
            **player,
            "tour": mod.Tour(player["tour"]),
            "latest_date": date.fromisoformat(player["latest_date"]),
            "competition": mod.CompetitionClass(player["competition"]),
        }
        for key, player in fixture["players"].items()
    }
    mod.MATCHES = tuple({**match, "tour": mod.Tour(match["tour"])} for match in fixture["matches"])

requested_ids = set(args.match_id)
configured_ids = {str(item["official_id"]) for item in mod.MATCHES}
unknown_ids = requested_ids - configured_ids
if unknown_ids:
    parser.error(f"match IDs absent from fixture: {sorted(unknown_ids)}")
selected_matches = tuple(item for item in mod.MATCHES if str(item["official_id"]) in requested_ids)
if args.policy == "fixed":
    path_policy = FIXED_100K_V1_POLICY
    n_paths = 100_000
    execution_mode = "production"
else:
    path_policy = PathCountPolicy(
        standard_paths=args.smoke_paths,
        escalated_paths=args.smoke_paths,
        minimum_settled_paths=args.smoke_paths,
    )
    n_paths = args.smoke_paths
    execution_mode = "development"

output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
checkpoint_root = (
    args.checkpoint_root.resolve()
    if args.checkpoint_root is not None
    else output / "simulation-checkpoints"
)


if args.source_capture is not None:
    capture, captured, cutoff = mod.load_captured_sources(args.source_capture.resolve())
else:
    capture, captured, cutoff = mod.acquire_sources(output)

schedule_overrides = {} if fixture is None else fixture.get("schedule_overrides", {})
try:
    selected = mod.schedule_matches(captured["schedule"]["payload"])
except RuntimeError:
    if not requested_ids.issubset({str(key) for key in schedule_overrides}):
        raise
    selected = {}
if fixture is not None:
    for official_id, override in schedule_overrides.items():
        selected[str(official_id)] = (
            {
                "courtName": str(override.get("courtName", "TBD")),
                "startEpoch": int(override["startEpoch"]),
            },
            {
                "notBefore": override.get("notBefore"),
                "statusCode": str(override.get("statusCode", "B")),
                "order": int(override.get("order", 1)),
            },
        )
base = artifact_repo / "artifacts/current-usopen-2026" / args.base_run_id
source_manifest = mod.load_source_manifest(base / "source_manifest.yaml")
manifest_pin = mod.SourceManifestProvenance.from_manifest(source_manifest)
def current_inactivity_information(player_key):
    player = mod.PLAYERS[player_key]
    source = captured[f"c6_{player_key}"]
    player_id = str(player["id"])
    return mod.PlayerInactivityInformation(
        player_id=player_id,
        coverage=mod.InactivityCoverageAssertion(
            state=mod.InactivityCoverageState.VERIFIED_COMPLETE,
            source_manifest_id=source_manifest.manifest_version,
            source_manifest_sha256=manifest_pin.manifest_sha256,
            canonical_player_id=player_id,
            asserted_at_utc=cutoff - timedelta(microseconds=1),
        ),
        candidates=(
            mod.InactivityMatchCandidate(
                player_id=player_id,
                identity_resolved=True,
                tour=player["tour"],
                match_id=str(player["latest_match_id"]),
                match_date_local=player["latest_date"],
                discipline="singles",
                competition_class=player["competition"],
                terminal_status=mod.InactivityTerminalStatus.NORMAL_COMPLETION,
                started_evidence=(
                    mod.PlayedPointEvidence.LEGAL_SCORE_WITH_COMPLETED_GAME_OR_TIEBREAK,
                ),
                source_manifest_id=source_manifest.manifest_version,
                source_pin=f"current-activity-{player_key}",
                source_sha256=source["sha256"],
                available_at_utc=source["retrieved_at_utc"],
            ),
        ),
    )


source_operational = artifact_repo / "artifacts/live-usopen-2026" / args.operational_name
snapshots = {
    tour: mod.ModelSnapshot.from_json(
        (source_operational / f"model_snapshot_{tour.value.lower()}.json").read_bytes()
    )
    for tour in mod.Tour
}
framework_versions = {snapshot.framework_version for snapshot in snapshots.values()}
if len(framework_versions) != 1:
    raise RuntimeError(f"mixed operational framework versions: {framework_versions}")
framework_version = next(iter(framework_versions))
uses_v1_1 = framework_version in {"v1.1-candidate", "v1.1", "v1.2"}

fitness_config = None
fitness_fit_path = None
fitness_fit_sha256 = None
fitness_fits = None
fitness_history = None
if uses_v1_1:
    v1_1_config_path = repo / (
        "config/model_v1_2.yaml"
        if framework_version == "v1.2"
        else "config/model_v1_1.yaml"
    )
    v1_1_config_payload = v1_1_config_path.read_bytes()
    v1_1_config = yaml.safe_load(v1_1_config_payload)
    v1_1_config_hash = __import__("hashlib").sha256(v1_1_config_payload).hexdigest()
    fitness_config = v1_1_config["game_day_fitness"]
    fitness_fit_path = repo / str(fitness_config["fit_artifact"])
    fitness_fit_sha256 = mod.sha256_file(fitness_fit_path)
    fitness_fits = {
        tour: load_game_day_elo_fit(fitness_fit_path, tour=tour.value)
        for tour in mod.Tour
    }
    fitness_history = load_main_tour_fitness_history(
        repo / str(fitness_config["history_source_directory"]),
        source_years=tuple(int(value) for value in fitness_config["source_years"]),
        tours=tuple(tour.value for tour in mod.Tour),
        levels={
            "ATP": ("A", "F", "G", "M"),
            "WTA": ("35+H", "50+H", "F", "G", "I", "P", "PM", "W"),
        },
    )
    if any(not snapshot.strength_complete for snapshot in snapshots.values()):
        raise RuntimeError("production v1.1 requires strength-complete v4 snapshots")
elif framework_version != "v1.0":
    raise RuntimeError(f"unsupported operational framework version: {framework_version}")
missing_duration_tours = sorted(
    {
        item["tour"].value
        for item in selected_matches
        if not snapshots[item["tour"]].duration_complete
    }
)
if missing_duration_tours:
    raise RuntimeError(
        "required day-one prop bundle blocked before simulation: "
        f"cutoff-matched duration artifact missing for {missing_duration_tours}"
    )
eligibility = {
    tour: mod.HistoricalTrainingEligibilityProvenance.model_validate_json(
        (source_operational / f"training_eligibility_{tour.value.lower()}.json").read_bytes()
    )
    for tour in mod.Tour
}

for tour, snapshot in snapshots.items():
    mod.write_immutable(
        output / f"model_snapshot_{tour.value.lower()}_v2.json",
        snapshot.canonical_json().encode("utf-8"),
    )

code_zip = mod.code_archive(repo, output / "retained/code")
code = mod.capture_code_provenance(repo)
store = mod.LockStore(output / "locks")
schedule_source = captured["schedule"]
schedule_payload = json.loads(schedule_source["payload"])
schedule_observed = min(
    datetime.fromtimestamp(int(schedule_payload["epoch"]), tz=UTC),
    schedule_source["retrieved_at_utc"],
)
results = []


def retained_artifacts(tour, snapshot, eligibility_path, fitness_input_path=None):
    if (
        snapshot.retirement_artifact is None
        or snapshot.inactivity_configuration is None
        or snapshot.duration_artifact is None
    ):
        raise RuntimeError(f"{tour.value} operational snapshot lacks B5/B6/C6")
    counts_receipt = output / "retained" / f"component_counts_{tour.value.lower()}.json"
    mod.write_immutable(
        counts_receipt,
        mod.canonical_bytes(
            {
                "component_count_artifact_hash": snapshot.component_count_artifact_hash,
                "current_component_counts_sha256": mod.sha256_file(
                    base / "data/current_component_counts.parquet"
                ),
                "training_eligibility_sha256": mod.sha256_file(eligibility_path),
            }
        ),
    )
    inactivity_path = output / "retained" / f"inactivity_config_{tour.value.lower()}.json"
    mod.write_immutable(
        inactivity_path,
        mod.canonical_bytes(snapshot.inactivity_configuration.model_dump(mode="json")),
    )
    settlement_path = output / "retained" / "settlement_policy.json"
    mod.write_immutable(
        settlement_path,
        mod.canonical_bytes(mod.asdict(mod.CANONICAL_SETTLEMENT_POLICY)),
    )
    records = [
        mod.retained_record("source_snapshot", capture),
        mod.retained_record("normalized_snapshot", eligibility_path),
        mod.retained_record("component_counts", counts_receipt),
        mod.retained_record("component_fit", base / "fits" / tour.value.lower()),
        mod.RetainedArtifactRecord(
            kind="duration_fit",
            artifact_id=snapshot.duration_artifact.artifact_id,
            path=str(snapshot.duration_artifact.directory.resolve()),
            sha256=mod.hash_path(snapshot.duration_artifact.directory),
        ),
        mod.retained_record("retirement_fit", snapshot.retirement_artifact.directory),
        mod.retained_record("inactivity_config", inactivity_path),
        mod.retained_record(
            "model_config",
            repo
            / (
                "config/model_v1_2.yaml"
                if snapshot.framework_version == "v1.2"
                else "config/model_v1_1.yaml"
                if snapshot.framework_version in {"v1.1-candidate", "v1.1"}
                else "config/model_v1.yaml"
            ),
        ),
        mod.retained_record("settlement_policy", settlement_path),
        mod.retained_record("code_archive", code_zip),
    ]
    if fitness_fit_path is not None and fitness_input_path is not None:
        records.extend(
            (
                mod.retained_record("fitness_fit", fitness_fit_path),
                mod.retained_record("fitness_input", fitness_input_path),
            )
        )
    if snapshot.strength_complete:
        if (
            snapshot.strength_anchor_artifact is None
            or snapshot.strength_integration_artifact is None
        ):
            raise RuntimeError(f"{tour.value} v1.1 snapshot lost its strength artifacts")
        records.extend(
            (
                mod.RetainedArtifactRecord(
                    kind="strength_fit",
                    artifact_id=snapshot.strength_anchor_artifact.artifact_id,
                    path=str(snapshot.strength_anchor_artifact.directory.resolve()),
                    sha256=mod.hash_path(snapshot.strength_anchor_artifact.directory),
                ),
                mod.RetainedArtifactRecord(
                    kind="strength_integration",
                    artifact_id=snapshot.strength_integration_artifact.artifact_id,
                    path=str(snapshot.strength_integration_artifact.directory.resolve()),
                    sha256=mod.hash_path(snapshot.strength_integration_artifact.directory),
                ),
            )
        )
    return tuple(records)


for item in selected_matches:
    official_id = str(item["official_id"])
    tour = item["tour"]
    left = mod.PLAYERS[str(item["a"])]
    right = mod.PLAYERS[str(item["b"])]
    court, scheduled = selected[official_id]
    if scheduled.get("notBefore"):
        local = datetime.strptime(
            f"{args.schedule_date.isoformat()} {scheduled['notBefore']}",
            "%Y-%m-%d %I:%M %p",
        ).replace(tzinfo=mod.NY)
        start = local.astimezone(UTC)
    else:
        start = datetime.fromtimestamp(int(court["startEpoch"]), tz=UTC)
    official_schedule_anchor = start
    if str(scheduled.get("statusCode")) != "B":
        raise RuntimeError(
            f"official match {official_id} is not unstarted at cutoff: "
            f"status={scheduled.get('statusCode')}"
        )
    # A court/session anchor is not the actual start of a later ordered match,
    # and a delayed not-before match can also remain unstarted after its anchor.
    # Official status B is the pre-start observation; retain the published anchor
    # below while satisfying the lock schema with the tightest valid lower bound.
    start_adjusted_from_status_b = cutoff >= start
    if start_adjusted_from_status_b:
        start = cutoff + timedelta(microseconds=1)

    fitness_input_path = None
    fitness_conditions = ()
    fitness_information = ()
    if uses_v1_1:
        assessments = tuple(
            assess_game_day_fitness(
                fitness_history,
                tour=tour.value,
                player_name=str(player["name"]),
                scheduled_local_date=args.schedule_date,
                information_cutoff_utc=cutoff,
                config=fitness_config,
            )
            for player in (left, right)
        )
        adjustments = tuple(
            0.0
            if assessment.features is None
            else game_day_adjustment(
                assessment.features,
                fitness_fits[tour].weights_elo,
            )
            for assessment in assessments
        )
        fitness_input_path = output / "retained" / f"game_day_fitness_{official_id}.json"
        mod.write_immutable(
            fitness_input_path,
            mod.canonical_bytes(
                {
                    "schema_version": "game-day-fitness-input/v1",
                    "framework_version": "v1.1",
                    "fit_artifact_sha256": fitness_fit_sha256,
                    "tour": tour.value,
                    "information_cutoff_utc": cutoff.isoformat(),
                    "scheduled_local_date": args.schedule_date.isoformat(),
                    "players": [
                        {
                            "player_id": str(player["id"]),
                            "player_name": str(player["name"]),
                            "available": assessment.available,
                            "reason": assessment.reason,
                            "last_match_date": (
                                None
                                if assessment.last_match_date is None
                                else assessment.last_match_date.isoformat()
                            ),
                            "recent_match_count": assessment.recent_match_count,
                            "features": (
                                None
                                if assessment.features is None
                                else {
                                    "recent_workload": assessment.features.recent_workload,
                                    "short_recovery": assessment.features.short_recovery,
                                    "return_from_layoff": assessment.features.return_from_layoff,
                                }
                            ),
                            "adjustment_elo": adjustment,
                        }
                        for player, assessment, adjustment in zip(
                            (left, right), assessments, adjustments, strict=True
                        )
                    ],
                    "source_manifest": fitness_history.source_manifest,
                }
            ),
        )
        fitness_input_sha256 = mod.sha256_file(fitness_input_path)
        fitness_conditions = (
            mod.MatchCondition(
                name="game_day_fitness_artifact_sha256",
                value=fitness_fit_sha256,
            ),
            mod.MatchCondition(
                name="game_day_fitness_status_player_a",
                value="available" if assessments[0].available else "unavailable",
            ),
            mod.MatchCondition(
                name="game_day_fitness_status_player_b",
                value="available" if assessments[1].available else "unavailable",
            ),
            mod.MatchCondition(name="game_day_elo_adjustment_player_a", value=adjustments[0]),
            mod.MatchCondition(name="game_day_elo_adjustment_player_b", value=adjustments[1]),
        )
        fitness_information = tuple(
            mod.InformationItem(
                category="workload",
                summary=(
                    f"Production v1.1 game-day fitness for {player['name']}: "
                    f"status={'available' if assessment.available else 'unavailable'}; "
                    f"temporary Elo adjustment={adjustment:.6f}"
                ),
                source_id=f"game-day-fitness-v1.1-{official_id}-{player['id']}",
                source_sha256=fitness_input_sha256,
                observed_at_utc=cutoff - timedelta(microseconds=1),
                available_at_utc=cutoff - timedelta(microseconds=1),
            )
            for player, assessment, adjustment in zip(
                (left, right), assessments, adjustments, strict=True
            )
        )

    context = mod.MatchContext(
        player_a_id=str(left["id"]),
        player_b_id=str(right["id"]),
        tour=tour,
        event="US Open",
        round=args.round,
        scheduled_start_utc=start,
        scheduled_start_local_date=args.schedule_date,
        best_of=5 if tour is mod.Tour.ATP else 3,
        indoor=None,
        conditions=(
            mod.MatchCondition(name="court", value=str(court["courtName"])),
            mod.MatchCondition(name="official_match_id", value=official_id),
            mod.MatchCondition(name="court_order", value=int(scheduled["order"])),
            mod.MatchCondition(
                name="schedule_anchor_semantics",
                value="not-before"
                if scheduled.get("notBefore")
                else "session-start; later ordered matches follow",
            ),
            mod.MatchCondition(
                name="official_schedule_anchor_utc",
                value=official_schedule_anchor.isoformat(),
            ),
            mod.MatchCondition(
                name="scheduled_start_utc_semantics",
                value=(
                    "status-B pre-start lower bound"
                    if start_adjusted_from_status_b
                    else "official schedule anchor"
                ),
            ),
            *fitness_conditions,
        ),
        information_cutoff_utc=cutoff,
        information_scenario_id="central",
    )
    c6 = tuple(
        current_inactivity_information(key)
        for key in (str(item["a"]), str(item["b"]))
    )
    information = mod.InformationBundle(
        bundle_id=f"usopen-2026-{official_id}-{capture.name[:16]}",
        scenario_id="central",
        information_cutoff_utc=cutoff,
        items=(
            mod.InformationItem(
                category="schedule",
                summary=(
                    f"Official US Open schedule match {official_id}: "
                    f"{left['name']} vs {right['name']}; status B at capture"
                ),
                source_id=args.schedule_source_id,
                source_sha256=schedule_source["sha256"],
                observed_at_utc=schedule_observed,
                available_at_utc=schedule_source["retrieved_at_utc"],
            ),
            *(
                mod.InformationItem(
                    category="workload",
                    summary=(
                        f"Current activity evidence for {mod.PLAYERS[key]['name']}: "
                        "latest eligible singles match "
                        f"{mod.PLAYERS[key]['latest_date'].isoformat()}"
                    ),
                    source_id=f"current-activity-{key}",
                    source_sha256=captured[f"c6_{key}"]["sha256"],
                    observed_at_utc=captured[f"c6_{key}"]["retrieved_at_utc"],
                    available_at_utc=captured[f"c6_{key}"]["retrieved_at_utc"],
                )
                for key in (str(item["a"]), str(item["b"]))
            ),
            *fitness_information,
        ),
        player_inactivity=c6,
        missing_current_conditions=("roof state",),
    )
    canonical = mod.CanonicalMatchIdentity.from_official_id(
        source_namespace="usopen-official",
        tour=tour,
        official_match_id=official_id,
        participant_ids=(str(left["id"]), str(right["id"])),
        source_id=args.schedule_source_id,
        source_sha256=schedule_source["sha256"],
        source_locator=mod.SCHEDULE_URL,
        resolved_at_utc=schedule_source["retrieved_at_utc"],
    )
    eligibility_path = source_operational / f"training_eligibility_{tour.value.lower()}.json"
    artifacts = retained_artifacts(
        tour, snapshots[tour], eligibility_path, fitness_input_path
    )
    existing = store.revision_directory(canonical.base_lock_id, 1)
    if existing.exists():
        lock = store.load(canonical.base_lock_id, 1).lock
    else:
        prop_config = item.get("prop_config", {})
        match_win_side = str(prop_config.get("match_win_side", args.match_win_side))
        comparison_side = str(prop_config.get("comparison_side", args.comparison_side))
        duration_threshold = float(
            prop_config.get("duration_threshold", args.duration_threshold)
        )
        tiebreak_threshold = int(
            prop_config.get("tiebreak_threshold", args.tiebreak_threshold)
        )
        if match_win_side not in {"left", "right"} or comparison_side not in {
            "left",
            "right",
        }:
            raise RuntimeError("fixture prop sides must be left or right")
        match_win_player = right if match_win_side == "right" else left
        comparison_player = right if comparison_side == "right" else left
        comparison_opponent = left if comparison_side == "right" else right
        duration_display_policy = (
            UNRESOLVED_DURATION_DISPLAY_POLICY
            if execution_mode == "production"
            else NEAREST_DURATION_DISPLAY_POLICY
        )
        if str(prop_config.get("bundle", "standard")) == "r32":
            def player_for_side(side_key: str, default: str):
                side = str(prop_config.get(side_key, default))
                if side not in {"left", "right"}:
                    raise RuntimeError(f"fixture {side_key} must be left or right")
                return right if side == "right" else left

            unforced_error_player = player_for_side(
                "unforced_error_side", comparison_side
            )
            first_serve_player = player_for_side(
                "first_serve_side", "left" if comparison_side == "right" else "right"
            )
            winner_player = player_for_side("winner_side", comparison_side)
            winner_opponent = left if winner_player is right else right
            requested_props = R32_PROP_BUNDLE(
                duration_minutes=duration_threshold,
                ace_subject=str(comparison_player["id"]),
                ace_opponent=str(comparison_opponent["id"]),
                double_fault_subject=str(comparison_player["id"]),
                double_fault_opponent=str(comparison_opponent["id"]),
                unforced_error_subject=str(unforced_error_player["id"]),
                unforced_error_threshold=float(
                    prop_config["unforced_error_threshold"]
                ),
                first_serve_subject=str(first_serve_player["id"]),
                first_serve_threshold_pct=float(
                    prop_config["first_serve_threshold_pct"]
                ),
                winner_subject=str(winner_player["id"]),
                winner_opponent=str(winner_opponent["id"]),
                match_win_subject=str(match_win_player["id"]),
                tiebreak_threshold=tiebreak_threshold,
            )
        else:
            requested_props = (
                mod.MATCH_WIN(str(match_win_player["id"])),
                ACE_COMPARE(str(comparison_player["id"]), str(comparison_opponent["id"])),
                DF_COMPARE(str(comparison_player["id"]), str(comparison_opponent["id"])),
                TIEBREAK_COUNT(ComparisonOperator.AT_LEAST, tiebreak_threshold),
                DURATION_MIN(
                    ComparisonOperator.MORE_THAN,
                    duration_threshold,
                    display_conversion_version=duration_display_policy.policy_version,
                ),
            )
        lock = mod.create_prediction_lock(
            snapshots[tour],
            context,
            information,
            requested_props,
            mod.CANONICAL_SETTLEMENT_POLICY,
            source_manifest=source_manifest,
            code=code,
            seed=202608300000 + int(official_id) + args.seed_offset,
            store=store,
            execution_mode=execution_mode,
            n_paths=n_paths,
            path_count_policy=path_policy,
            allow_dirty=True,
            canonical_match_identity=canonical,
            retained_artifacts=artifacts,
            training_eligibility=eligibility[tour],
            duration_display_policy=duration_display_policy,
            simulation_workers=args.workers,
            simulation_checkpoint_dir=checkpoint_root / official_id,
            simulation_checkpoint_paths=args.checkpoint_paths,
            simulation_progress=True,
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
            "official_schedule_anchor_utc": official_schedule_anchor.isoformat(),
            "scheduled_start_adjusted_from_status_b": start_adjusted_from_status_b,
            "lock_id": lock.lock_id,
            "revision": lock.revision,
            "content_sha256": lock.content_sha256,
            "verified_sha256": verified,
            "snapshot_id": lock.match_parameters.snapshot_id,
            "paths": lock.simulation.actual_paths,
            "inspected_path_counts": lock.simulation.inspected_path_counts,
            "mc_stopping_status": None
            if estimate.mc_stopping_status is None
            else estimate.mc_stopping_status.value,
            "mc_error": estimate.mc_standard_error,
            "prop_estimates": [item.model_dump(mode="json") for item in lock.prop_estimates],
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
            "duration": None
            if lock.match_summary.duration is None
            else lock.match_summary.duration.model_dump(mode="json"),
            "c6": [
                {
                    "player": player_names[row.player_id],
                    "latest_match_date": None
                    if row.last_eligible_match is None
                    else row.last_eligible_match.match_date_local.isoformat(),
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
    "schema_version": "configured-usopen-match-batch/v1",
    "batch_information_cutoff_utc": cutoff.isoformat(),
    "official_source_capture_id": capture.name,
    "base_run_id": args.base_run_id,
    "operational_name": args.operational_name,
    "methodology_changed": True,
    "execution_policy_version": "fixed-100k/v1" if args.policy == "fixed" else "fixed-smoke/v1",
    "refit_performed": False,
    "execution_mode": execution_mode,
    "seed_offset": args.seed_offset,
    "simulation_workers": args.workers,
    "simulation_checkpoint_paths": args.checkpoint_paths,
    "simulation_checkpoint_root": str(checkpoint_root),
    "path_count_policy": mod.asdict(path_policy),
    "status": (
        "FIXED_100K_PRODUCTION_LOCKS"
        if args.policy == "fixed"
        else "FIXED_CHECKPOINT_DEVELOPMENT_SMOKE_LOCKS"
    ),
    "matches": results,
}
report_path = output / f"batch-report-{capture.name[:16]}.json"
mod.write_immutable(report_path, mod.canonical_bytes(report))
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
print(f"\nREPORT_PATH={report_path}")
