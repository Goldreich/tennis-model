"""Run configured US Open match simulations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

repo = Path.cwd().resolve()
sys.path.insert(0, str(repo / "src"))
artifact_repo = repo

spec = importlib.util.spec_from_file_location(
    "simulate_remaining_four",
    repo / ".runtime-worktree-81aa8db" / "scripts" / "simulate_remaining_four.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from tennis_model.estimation.duration_model import UNRESOLVED_DURATION_DISPLAY_POLICY
from tennis_model.locking import FIXED_100K_V1_POLICY, PathCountPolicy
from tennis_model.props.settlement import ComparisonOperator
from tennis_model.simulation import ACE_COMPARE, DF_COMPARE, DURATION_MIN, TIEBREAK_COUNT

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-file", type=Path)
parser.add_argument("--match-id", action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-capture", type=Path)
parser.add_argument("--base-run-id", required=True)
parser.add_argument("--operational-name", required=True)
parser.add_argument("--policy", choices=("smoke", "fixed"), default="fixed")
parser.add_argument("--smoke-paths", type=int, default=5_000)
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
artifact_repo = args.artifact_root.resolve() if args.artifact_root is not None else repo

if len(args.base_run_id) != 64 or any(char not in "0123456789abcdef" for char in args.base_run_id):
    parser.error("--base-run-id must be 64 lowercase hex characters")
if args.smoke_paths <= 0:
    parser.error("--smoke-paths must be positive")
if Path(args.operational_name).name != args.operational_name:
    parser.error("--operational-name must be one path component")

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

if args.source_capture is not None:
    capture, captured, cutoff = mod.load_captured_sources(args.source_capture.resolve())
else:
    capture, captured, cutoff = mod.acquire_sources(output)

selected = mod.schedule_matches(captured["schedule"]["payload"])
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


def retained_artifacts(tour, snapshot, eligibility_path):
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
    return (
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
        mod.retained_record("model_config", repo / "config/model_v1.yaml"),
        mod.retained_record("settlement_policy", settlement_path),
        mod.retained_record("code_archive", code_zip),
    )


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
    artifacts = retained_artifacts(tour, snapshots[tour], eligibility_path)
    existing = store.revision_directory(canonical.base_lock_id, 1)
    if existing.exists():
        lock = store.load(canonical.base_lock_id, 1).lock
    else:
        match_win_player = right if args.match_win_side == "right" else left
        comparison_player = right if args.comparison_side == "right" else left
        comparison_opponent = left if args.comparison_side == "right" else right
        requested_props = (
            mod.MATCH_WIN(str(match_win_player["id"])),
            ACE_COMPARE(str(comparison_player["id"]), str(comparison_opponent["id"])),
            DF_COMPARE(str(comparison_player["id"]), str(comparison_opponent["id"])),
            TIEBREAK_COUNT(ComparisonOperator.AT_LEAST, args.tiebreak_threshold),
            DURATION_MIN(
                ComparisonOperator.MORE_THAN,
                args.duration_threshold,
                display_conversion_version=UNRESOLVED_DURATION_DISPLAY_POLICY.policy_version,
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
            seed=202608300000 + int(official_id),
            store=store,
            execution_mode=execution_mode,
            n_paths=n_paths,
            path_count_policy=path_policy,
            allow_dirty=True,
            canonical_match_identity=canonical,
            retained_artifacts=artifacts,
            training_eligibility=eligibility[tour],
            duration_display_policy=UNRESOLVED_DURATION_DISPLAY_POLICY,
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
