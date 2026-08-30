from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tennis_model.calibration.backtest import (
    derive_backtest_seed,
    derive_production_seed,
)
from tennis_model.calibration.validation import (
    ComparatorForecast,
    ComparatorKind,
    ComparatorResolution,
    CorrectnessGate,
    DiscretePredictiveObservation,
    GateStatus,
    OpportunityObservation,
    PrimitivePredictiveObservation,
    brier_skill,
    build_validation_report,
    calibration_regression,
    exact_score_coherence,
    randomized_discrete_pit,
    reliability_with_event_bootstrap,
)
from tennis_model.cli import main as cli_main
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.models import PropSupportStatus
from tennis_model.locking.provenance import (
    capture_code_provenance,
    capture_runtime_fingerprint,
)
from tennis_model.operational_audit import (
    SyntheticAuditCategory,
    audit_core,
    load_synthetic_audit_manifest,
)
from tennis_model.props.policy import assess_prop_support, integer_submission_preview
from tennis_model.props.settlement import ComparisonOperator
from tennis_model.schemas import Tour
from tennis_model.simulation.match import (
    FIRST_SERVE_WIN_PCT,
    PropSpec,
    SimulationBatch,
    _simulate_one_path,
)
from tennis_model.simulation.point import ServePerformanceDraw


def test_checked_in_production_source_registry_passes_operational_audit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = audit_core(repo_root)
    source_check = next(
        item for item in report.checks if item.name == "PRODUCTION_SOURCE_REGISTRY"
    )
    assert source_check.status == "PASS"
    assert "18 verified source object(s)" in source_check.detail


def test_canonical_match_identity_is_independent_of_forecast_state() -> None:
    first = CanonicalMatchIdentity.from_official_id(
        source_namespace="Official Draw",
        tour=Tour.ATP,
        official_match_id="match-101",
        participant_ids=("A", "B"),
        source_id="draw-v1",
        source_sha256="a" * 64,
        source_locator="fixture://draw/old",
        resolved_at_utc=datetime(2026, 8, 1, tzinfo=UTC),
    )
    later = CanonicalMatchIdentity.from_official_id(
        source_namespace="official draw",
        tour=Tour.ATP,
        official_match_id="match-101",
        participant_ids=("B", "A"),
        source_id="draw-v2",
        source_sha256="b" * 64,
        source_locator="fixture://draw/rescheduled",
        resolved_at_utc=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert first.canonical_match_id == later.canonical_match_id
    assert first.base_lock_id == later.base_lock_id

    slot = CanonicalMatchIdentity.from_draw_slot(
        source_namespace="official draw",
        tour=Tour.WTA,
        event_edition_id="USO-2026",
        draw_slot="QF-2",
        participant_ids=("C", "D"),
        source_id="draw-slot",
        source_sha256="c" * 64,
        source_locator="fixture://draw/QF-2",
        resolved_at_utc=datetime(2026, 8, 3, tzinfo=UTC),
    )
    assert slot.base_lock_id.startswith("TMV1-WTA-")
    assert slot.official_match_id is None


def test_milestone81_prop_policy_gates_and_integer_preview() -> None:
    percent = FIRST_SERVE_WIN_PCT("A", ComparisonOperator.MORE_THAN, 65)
    decision = assess_prop_support(percent)
    assert decision.status is PropSupportStatus.POLICY_DISABLED
    assert decision.reason_code == "FIRST_SERVE_WIN_PERCENT_DISPLAY_CONVERSION_UNRESOLVED"

    resolved_percent = PropSpec(
        kind="FIRST_SERVE_WIN_PCT",
        subject_ids=("A",),
        operator=ComparisonOperator.MORE_THAN,
        threshold=65,
        scope={"rounding_invariant": True},
    )
    assert assess_prop_support(resolved_percent).status is PropSupportStatus.SUPPORTED
    winners = PropSpec(
        kind="WINNERS",
        subject_ids=("A",),
        operator=ComparisonOperator.MORE_THAN,
        threshold=20,
    )
    assert assess_prop_support(winners).status is PropSupportStatus.POLICY_DISABLED
    implemented_policy_but_missing_generator = PropSpec(
        kind="WINNERS",
        subject_ids=("A",),
        operator=ComparisonOperator.MORE_THAN,
        threshold=20,
        scope={"official_accounting_version": "fixture/v1"},
    )
    assert (
        assess_prop_support(implemented_policy_but_missing_generator).status
        is PropSupportStatus.NOT_IMPLEMENTED
    )
    assert integer_submission_preview(0.0) == 0
    assert integer_submission_preview(0.5049) == 50
    assert integer_submission_preview(0.505) == 51
    assert integer_submission_preview(1.0) == 100


def _comparator_rows() -> tuple[ComparatorResolution, ...]:
    rows = []
    start = datetime(2026, 8, 20, tzinfo=UTC)
    for index in range(20):
        outcome = index % 2
        tour = Tour.ATP if index < 10 else Tour.WTA
        for comparator in ComparatorKind:
            probability = {
                ComparatorKind.TOUR_EVENT_BASELINE: 0.5,
                ComparatorKind.RAW_LOGIT_BASELINE: 0.55 if outcome else 0.45,
                ComparatorKind.SURFACE_ELO_BENCHMARK: 0.60 if outcome else 0.40,
                ComparatorKind.STATIONARY_POINT_MODEL: 0.65 if outcome else 0.35,
                ComparatorKind.TENNIS_MODEL_V1: 0.75 if outcome else 0.25,
            }[comparator]
            forecast = ComparatorForecast(
                comparator=comparator,
                target_id=f"target-{index}",
                event_id=f"event-{index // 2}",
                tour=tour,
                forecast_cutoff_utc=start + timedelta(days=index),
                target_start_utc=start + timedelta(days=index, hours=2),
                artifact_created_at_utc=start + timedelta(days=100 + index),
                probability=probability,
            )
            assert forecast.evaluation_only
            rows.append(
                ComparatorResolution(
                    forecast=forecast,
                    outcome=outcome,
                    outcome_available_at_utc=forecast.target_start_utc + timedelta(hours=3),
                )
            )
    return tuple(rows)


def test_complete_i3_validation_infrastructure_is_seeded_and_structured() -> None:
    comparator_rows = _comparator_rows()
    probabilities = np.linspace(0.1, 0.9, 40)
    outcomes = (probabilities >= 0.5).astype(np.float64)
    regression = calibration_regression(probabilities, outcomes)
    assert regression.observations == 40
    assert regression.status is GateStatus.PASS
    assert brier_skill(0.10, 0.20) == pytest.approx(0.5)
    assert brier_skill(0.0, 0.0) is None

    reliability = reliability_with_event_bootstrap(
        comparator_rows,
        replicates=40,
        seed=711,
    )
    assert len(reliability) == 10
    assert any(item.mean_brier_interval.replicates == 40 for item in reliability)

    opportunities = tuple(
        OpportunityObservation(
            kind="HOLD" if index % 2 else "BREAK",
            event_id=f"event-{index // 4}",
            tour=Tour.ATP if index < 20 else Tour.WTA,
            predicted_probability=(index + 1) / 41,
            observed=index % 3 == 0,
        )
        for index in range(40)
    )
    games = tuple(
        DiscretePredictiveObservation(
            observation_id=f"games-{index}",
            event_id=f"event-{index // 2}",
            observed_value=20 + index % 3,
            mass=((20, 0.2), (21, 0.5), (22, 0.3)),
        )
        for index in range(12)
    )
    first_pit = randomized_discrete_pit(games, seed=991)
    second_pit = randomized_discrete_pit(games, seed=991)
    assert first_pit == second_pit
    assert first_pit.ks_statistic is not None

    primitives = tuple(
        PrimitivePredictiveObservation(
            observation_id=f"primitive-{component.value}",
            event_id="primitive-event",
            tour=Tour.ATP,
            component=component,
            observed=1,
            predictive_mass=(0.2, 0.6, 0.2),
            interval_lower=0,
            interval_upper=2,
        )
        for component in ServeComponent
    )
    report = build_validation_report(
        hard_correctness_gates=(CorrectnessGate("HASH_INTEGRITY", GateStatus.PASS, "fixture"),),
        comparator_resolutions=comparator_rows,
        ledger_rows=(),
        opportunity_observations=opportunities,
        game_observations=games,
        primitive_observations=primitives,
        randomization_seed=20260830,
        bootstrap_replicates=30,
    )
    assert len(report.comparator_rows) == 5
    assert {item.comparator for item in report.comparator_rows} == set(ComparatorKind)
    full = next(
        item for item in report.comparator_rows if item.comparator is ComparatorKind.TENNIS_MODEL_V1
    )
    assert full.brier_skill is not None and full.brier_skill > 0
    assert full.brier_skill_observations == 20
    assert full.brier_interval.replicates == 30
    assert {item.tour for item in report.comparator_rows_by_tour} == {Tour.ATP, Tour.WTA}
    assert report.comparator_rows_by_event
    assert report.hold_break_deciles
    assert report.game_distribution_pit is not None
    assert {item.component for item in report.primitive_diagnostics} == set(ServeComponent)
    assert not report.genuine_historical_validation


def test_exact_score_coherence_uses_the_same_joint_paths() -> None:
    performance_a = ServePerformanceDraw(0.65, 0.10, 0.60, 0.06, 0.54)
    performance_b = ServePerformanceDraw(0.62, 0.08, 0.56, 0.08, 0.50)
    paths = tuple(
        _simulate_one_path(
            "A",
            "B",
            best_of=3,
            first_server_id="A",
            player_a_performance=performance_a,
            player_b_performance=performance_b,
            rng=np.random.default_rng(seed),
            trace_points=False,
        )
        for seed in range(40)
    )
    batch = SimulationBatch(
        context=SimpleNamespace(best_of=3, player_a_id="A", player_b_id="B"),
        n_paths=len(paths),
        seed_id="explicit-path-fixture",
        paths=paths,
    )
    coherence = exact_score_coherence(batch)
    assert coherence.coherent
    assert coherence.legal_support
    assert sum(item[3] for item in coherence.exact_score_distribution) == pytest.approx(1)
    assert coherence.deciding_set_probability == pytest.approx(
        coherence.deciding_set_probability_from_scores
    )


def test_runtime_and_complete_git_fingerprint_cover_all_dirty_classes(tmp_path: Path) -> None:
    repo = tmp_path / "git-fixture"
    repo.mkdir()
    subprocess.run(("git", "init", str(repo)), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "fixture@example.test"),
        check=True,
    )
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Fixture"), check=True)
    tracked = repo / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repo), "add", "tracked.py"), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "commit", "-m", "fixture"),
        check=True,
        capture_output=True,
    )
    empty = hashlib.sha256(b"").hexdigest()
    clean = capture_code_provenance(repo)
    assert not clean.dirty
    assert (clean.staged_sha256, clean.unstaged_sha256, clean.untracked_sha256) == (
        empty,
        empty,
        empty,
    )

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repo), "add", "tracked.py"), check=True)
    staged = capture_code_provenance(repo)
    assert staged.dirty and staged.staged_sha256 != empty
    tracked.write_text("VALUE = 3\n", encoding="utf-8")
    unstaged = capture_code_provenance(repo)
    assert unstaged.unstaged_sha256 != empty
    (repo / "probability_config.yaml").write_text("version: fixture\n", encoding="utf-8")
    untracked = capture_code_provenance(repo)
    assert untracked.untracked_sha256 != empty
    assert untracked.relevant_untracked_files == ("probability_config.yaml",)
    assert len({staged.diff_sha256, unstaged.diff_sha256, untracked.diff_sha256}) == 3

    runtime = capture_runtime_fingerprint(
        simulator_algorithm_version="fixture-simulator/v1",
        chunk_size=128,
        thread_count=2,
        process_count=1,
    )
    assert runtime.numpy_version
    assert runtime.scipy_version
    assert runtime.pydantic_version
    assert runtime.rng_bit_generator == "PCG64"
    assert runtime.chunk_size == 128
    assert runtime.thread_count == 2


def test_paired_seed_policy_and_synthetic_audit_manifest() -> None:
    cutoff = datetime(2026, 8, 20, tzinfo=UTC)
    paired = derive_backtest_seed(
        canonical_match_id="stable-match",
        forecast_cutoff_utc=cutoff,
    )
    assert paired == derive_backtest_seed(
        canonical_match_id="stable-match",
        forecast_cutoff_utc=cutoff,
    )
    assert paired != derive_backtest_seed(
        canonical_match_id="stable-match",
        forecast_cutoff_utc=cutoff,
        path_index=1,
    )
    assert paired != derive_production_seed(
        canonical_match_id="stable-match",
        lock_revision=1,
        purpose="production",
        entropy="fixture",
    )

    fixture = Path(__file__).parent / "fixtures" / "milestone81_synthetic_audit.yaml"
    manifest = load_synthetic_audit_manifest(fixture)
    assert manifest.deterministic_seed == 20260830
    assert {item.category for item in manifest.cases} == set(SyntheticAuditCategory)


def test_read_only_audit_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    repo = Path(__file__).parents[1]
    assert cli_main(["audit-core", "--repo", str(repo)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["b6_c6_status"] == "COMPLETE"
    assert payload["genuine_historical_validation"] == "NOT_YET_RUN"
    assert len(payload["comparator_suite"]) == 5
    checks = {item["name"]: item["status"] for item in payload["checks"]}
    assert checks["SYNTHETIC_FAILURE_PATH_FIXTURE"] == "PASS"
    assert checks["VALIDATION_INFRASTRUCTURE"] == "PASS"
