from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

repo = Path.cwd().resolve()
sys.path.insert(0, str(repo / 'src'))
artifact_repo = repo

spec = importlib.util.spec_from_file_location(
    'simulate_remaining_four',
    repo / '.runtime-worktree-81aa8db' / 'scripts' / 'simulate_remaining_four.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

selected_matches = tuple(item for item in mod.MATCHES if str(item['official_id']) in {'1148', '2125', '1157'})
preliminary_policy = replace(mod.ADAPTIVE_MC_CS_V1_POLICY, checkpoints=(5_000,))
output = artifact_repo / 'artifacts/live-usopen-2026/targeted-three-v1'
output.mkdir(parents=True, exist_ok=True)

readable = artifact_repo / 'artifacts/live-usopen-2026/remaining-four-adaptive-v1/source-captures-readable'
retained_captures = tuple(sorted(readable.glob('*/manifest.json')))
if retained_captures:
    capture, captured, cutoff = mod.load_captured_sources(retained_captures[-1].parent)
else:
    capture, captured, cutoff = mod.acquire_sources(output)

selected = mod.schedule_matches(captured['schedule']['payload'])
base = artifact_repo / 'artifacts/current-usopen-2026' / mod.BASE_RUN_ID
source_manifest = mod.load_source_manifest(base / 'source_manifest.yaml')
manifest_pin = mod.SourceManifestProvenance.from_manifest(source_manifest)
snapshots = mod.upgraded_snapshots(repo, artifact_repo)
source_operational = artifact_repo / 'artifacts/live-usopen-2026/official-2117-v1'
eligibility = {
    tour: mod.HistoricalTrainingEligibilityProvenance.model_validate_json(
        (source_operational / f'training_eligibility_{tour.value.lower()}.json').read_bytes()
    )
    for tour in mod.Tour
}

for tour, snapshot in snapshots.items():
    mod.write_immutable(output / f'model_snapshot_{tour.value.lower()}_v2.json', snapshot.canonical_json().encode('utf-8'))

code_zip = mod.code_archive(repo, output / 'retained/code')
code = mod.capture_code_provenance(repo)
store = mod.LockStore(output / 'locks')
schedule_source = captured['schedule']
schedule_payload = json.loads(schedule_source['payload'])
schedule_observed = min(datetime.fromtimestamp(int(schedule_payload['epoch']), tz=UTC), schedule_source['retrieved_at_utc'])
results = []

for item in selected_matches:
    official_id = str(item['official_id'])
    tour = item['tour']
    left = mod.PLAYERS[str(item['a'])]
    right = mod.PLAYERS[str(item['b'])]
    court, scheduled = selected[official_id]
    start = mod.scheduled_start(court, scheduled)

    context = mod.MatchContext(
        player_a_id=str(left['id']),
        player_b_id=str(right['id']),
        tour=tour,
        event='US Open',
        round='R128',
        scheduled_start_utc=start,
        scheduled_start_local_date=date(2026, 8, 30),
        best_of=5 if tour is mod.Tour.ATP else 3,
        indoor=None,
        conditions=(
            mod.MatchCondition(name='court', value=str(court['courtName'])),
            mod.MatchCondition(name='official_match_id', value=official_id),
            mod.MatchCondition(name='court_order', value=int(scheduled['order'])),
            mod.MatchCondition(
                name='schedule_anchor_semantics',
                value='not-before' if scheduled.get('notBefore') else 'session-start; later ordered matches follow',
            ),
        ),
        information_cutoff_utc=cutoff,
        information_scenario_id='central',
    )
    c6 = tuple(
        mod.inactivity_information(
            key,
            manifest_id=source_manifest.manifest_version,
            manifest_sha256=manifest_pin.manifest_sha256,
            cutoff=cutoff,
            captured=captured,
        )
        for key in (str(item['a']), str(item['b']))
    )
    information = mod.InformationBundle(
        bundle_id=f'usopen-2026-{official_id}-{capture.name[:16]}',
        scenario_id='central',
        information_cutoff_utc=cutoff,
        items=(
            mod.InformationItem(
                category='schedule',
                summary=f'Official US Open schedule match {official_id}: {left["name"]} vs {right["name"]}; status B at capture',
                source_id='official-usopen-2026-schedule-day-8',
                source_sha256=schedule_source['sha256'],
                observed_at_utc=schedule_observed,
                available_at_utc=schedule_source['retrieved_at_utc'],
            ),
            *(
                mod.InformationItem(
                    category='workload',
                    summary=f'Official activity evidence for {mod.PLAYERS[key]["name"]}: latest eligible singles match {mod.PLAYERS[key]["latest_date"].isoformat()}',
                    source_id=f'official-current-activity-{key}',
                    source_sha256=captured[f'c6_{key}']['sha256'],
                    observed_at_utc=captured[f'c6_{key}']['retrieved_at_utc'],
                    available_at_utc=captured[f'c6_{key}']['retrieved_at_utc'],
                )
                for key in (str(item['a']), str(item['b']))
            ),
        ),
        player_inactivity=c6,
        missing_current_conditions=('roof state',),
    )
    canonical = mod.CanonicalMatchIdentity.from_official_id(
        source_namespace='usopen-official',
        tour=tour,
        official_match_id=official_id,
        participant_ids=(str(left['id']), str(right['id'])),
        source_id='official-usopen-2026-schedule-day-8',
        source_sha256=schedule_source['sha256'],
        source_locator=mod.SCHEDULE_URL,
        resolved_at_utc=schedule_source['retrieved_at_utc'],
    )
    eligibility_path = source_operational / f'training_eligibility_{tour.value.lower()}.json'
    artifacts = mod.retained_artifacts(repo, artifact_repo, output, tour, snapshots[tour], capture, code_zip, eligibility_path)
    lock = mod.create_prediction_lock(
        snapshots[tour],
        context,
        information,
        (mod.MATCH_WIN(str(left['id'])),),
        mod.CANONICAL_SETTLEMENT_POLICY,
        source_manifest=source_manifest,
        code=code,
        seed=202608300000 + int(official_id),
        store=store,
        execution_mode='development',
        path_count_policy=preliminary_policy,
        allow_dirty=True,
        canonical_match_identity=canonical,
        retained_artifacts=artifacts,
        training_eligibility=eligibility[tour],
    )
    verified = store.verify(lock.base_lock_id, lock.revision)
    player_names = {str(left['id']): str(left['name']), str(right['id']): str(right['name'])}
    estimate = lock.prop_estimates[0]
    results.append({
        'official_match_id': official_id,
        'match': f"{left['name']} vs {right['name']}",
        'tour': tour.value,
        'schedule_status_at_cutoff': scheduled['statusCode'],
        'court': court['courtName'],
        'scheduled_start_utc': start.isoformat(),
        'lock_id': lock.lock_id,
        'revision': lock.revision,
        'content_sha256': lock.content_sha256,
        'verified_sha256': verified,
        'snapshot_id': lock.match_parameters.snapshot_id,
        'paths': lock.simulation.actual_paths,
        'inspected_path_counts': lock.simulation.inspected_path_counts,
        'mc_stopping_status': None if estimate.mc_stopping_status is None else estimate.mc_stopping_status.value,
        'mc_error': estimate.mc_standard_error,
        'match_win_probability': {player_names[row.player_id]: row.match_win_probability for row in lock.match_summary.players},
        'exact_score_probability': [{'winner': player_names[row.winner_id], 'score': f'{row.winner_sets}-{row.loser_sets}', 'probability': row.probability} for row in lock.match_summary.exact_scores],
        'any_tiebreak_probability': lock.match_summary.any_tiebreak_probability,
        'deciding_set_probability': lock.match_summary.deciding_set_probability,
        'expected_total_games': lock.match_summary.expected_total_games,
        'total_games_quantiles': lock.match_summary.total_games_quantiles,
        'players': [{
            'name': player_names[row.player_id],
            'expected_aces': row.expected_aces,
            'expected_double_faults': row.expected_double_faults,
            'expected_breaks': row.expected_breaks,
            'retirement_probability': row.retirement_probability,
            'first_serve_win_expectation': next(direction.first_serve_win for direction in lock.parameter_summaries if direction.server_id == row.player_id),
        } for row in lock.match_summary.players],
        'duration': None if lock.match_summary.duration is None else lock.match_summary.duration.model_dump(mode='json'),
        'c6': [{
            'player': player_names[row.player_id],
            'latest_match_date': None if row.last_eligible_match is None else row.last_eligible_match.match_date_local.isoformat(),
            'inactivity_days': row.inactivity_days,
            'band': row.band.value,
        } for row in lock.match_parameters.inactivity.records],
        'warnings': lock.warnings,
        'card_path': str(store.revision_directory(lock.base_lock_id, lock.revision) / 'card.md'),
    })

report = {
    'schema_version': 'targeted-three-live-simulation/v1',
    'batch_information_cutoff_utc': cutoff.isoformat(),
    'official_source_capture_id': capture.name,
    'methodology_changed': False,
    'refit_performed': False,
    'status': 'TARGETED_THREE_DEVELOPMENT_LOCK',
    'matches': results,
}
report_path = output / f'batch-report-{capture.name[:16]}.json'
mod.write_immutable(report_path, mod.canonical_bytes(report))
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
print(f'\nREPORT_PATH={report_path}')
