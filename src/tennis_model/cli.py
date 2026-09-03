"""Reproducible command-line entrypoints for Milestone 8 artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
import hashlib
from pathlib import Path
from typing import Any

from tennis_model.calibration.backtest import (
    HistoricalFilter,
    HistoricalForecastTarget,
    SnapshotCatalog,
    run_rolling_backtest,
)
from tennis_model.calibration.ledger import CalibrationLedger
from tennis_model.calibration.metrics import summarize_calibration
from tennis_model.calibration.outcomes import OfficialHistoricalOutcome
from tennis_model.data.source_manifest import load_source_manifest
from tennis_model.estimation.inactivity import InactivityRecord
from tennis_model.estimation.retirement import RetirementScenarioMixture
from tennis_model.estimation.snapshot import ModelSnapshot
from tennis_model.estimation.snapshot import (
    create_v1_1_snapshot,
    load_v1_1_strength_anchor_artifact,
)
from tennis_model.estimation.elo import (
    SurfaceEloConfig,
    import_surface_elo_csv,
    write_surface_elo_artifact,
)
from tennis_model.estimation.strength import (
    DynamicStrengthConfig,
    StrengthOutcomeRecord,
    fit_dynamic_strength,
    load_strength_artifact,
    write_strength_artifact,
)
from tennis_model.estimation.strength_integration import (
    CrossFittedStrengthRecord,
    StrengthIntegrationConfig,
    create_fixed_strength_integration,
    fit_strength_integration,
    load_strength_integration_artifact,
    write_strength_integration_artifact,
)
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.card import render_locked_match_card
from tennis_model.locking.models import (
    InformationBundle,
    ReplayLevel,
    RetainedArtifactRecord,
    SerializedProp,
)
from tennis_model.locking.path_counts import (
    ADAPTIVE_MC_CS_V1_POLICY,
    FROZEN_PATH_COUNT_POLICY,
    AdaptiveMCPolicy,
    PathCountPolicy,
)
from tennis_model.locking.provenance import capture_code_provenance
from tennis_model.locking.service import create_prediction_lock, reproduce_prediction_lock
from tennis_model.locking.store import LockStore
from tennis_model.operational_audit import audit_core
from tennis_model.props.settlement import CANONICAL_SETTLEMENT_POLICY
from tennis_model.schemas import Tour
from tennis_model.simulation.parameters import MatchContext
import yaml


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, default=_json_default, sort_keys=True, indent=2))


def _path_policy(args: argparse.Namespace) -> PathCountPolicy:
    if args.mode == "production":
        return FROZEN_PATH_COUNT_POLICY
    return PathCountPolicy(
        standard_paths=args.standard_paths,
        escalated_paths=args.escalated_paths,
        minimum_settled_paths=args.minimum_settled_paths,
    )


def _live_path_policy(args: argparse.Namespace) -> PathCountPolicy | AdaptiveMCPolicy:
    return ADAPTIVE_MC_CS_V1_POLICY if args.mode == "production" else _path_policy(args)


def _lock_match(args: argparse.Namespace) -> int:
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    snapshot = ModelSnapshot.model_validate_json(Path(request["snapshot_path"]).read_bytes())
    context = MatchContext.model_validate(request["context"])
    information = InformationBundle.model_validate(request["information"])
    props = tuple(SerializedProp.model_validate(item) for item in request["props"])
    inactivity_records = tuple(
        InactivityRecord.model_validate(item) for item in request.get("inactivity_records", ())
    )
    retirement_scenario_mixtures = tuple(
        RetirementScenarioMixture.model_validate(item)
        for item in request.get("retirement_scenario_mixtures", ())
    )
    canonical_identity = (
        None
        if request.get("canonical_match_identity") is None
        else CanonicalMatchIdentity.model_validate(request["canonical_match_identity"])
    )
    retained_artifacts = tuple(
        RetainedArtifactRecord.model_validate(item)
        for item in request.get("retained_artifacts", ())
    )
    from tennis_model.locking.models import deserialize_prop

    code = capture_code_provenance(args.repo)
    policy = _live_path_policy(args)
    lock = create_prediction_lock(
        snapshot,
        context,
        information,
        tuple(deserialize_prop(item) for item in props),
        CANONICAL_SETTLEMENT_POLICY,
        source_manifest=load_source_manifest(request["source_manifest_path"]),
        code=code,
        seed=int(request["seed"]),
        store=LockStore(args.locks),
        n_paths=(policy.standard_paths if isinstance(policy, PathCountPolicy) else None),
        first_server_id=request.get("first_server_id"),
        execution_mode=args.mode,
        path_count_policy=policy,
        allow_dirty=args.allow_dirty,
        inactivity_records=inactivity_records,
        retirement_scenario_mixtures=retirement_scenario_mixtures,
        canonical_match_identity=canonical_identity,
        retained_artifacts=retained_artifacts,
    )
    _print_json(
        {
            "lock_id": lock.lock_id,
            "content_sha256": lock.content_sha256,
            "artifact_directory": str(
                LockStore(args.locks).revision_directory(lock.base_lock_id, lock.revision)
            ),
        }
    )
    return 0


def _verify_lock(args: argparse.Namespace) -> int:
    envelope = LockStore(args.locks).load(args.base_lock_id, args.revision)
    result: dict[str, Any] = {
        "lock_id": envelope.lock.lock_id,
        "content_sha256": envelope.content_sha256,
        "hashes_verified": True,
    }
    if args.reproduce:
        result["reproduction"] = asdict(
            reproduce_prediction_lock(
                envelope.lock,
                replay_level=ReplayLevel(args.replay_level),
                semantic_tolerance=args.semantic_tolerance,
            )
        )
    _print_json(result)
    return 0


def _render_lock(args: argparse.Namespace) -> int:
    lock = LockStore(args.locks).load(args.base_lock_id, args.revision).lock
    print(render_locked_match_card(lock), end="")
    return 0


def _summarize_ledger(args: argparse.Namespace) -> int:
    ledger = CalibrationLedger(args.ledger)
    verification = ledger.verify_chain()
    report = summarize_calibration(ledger.effective_rows())
    _print_json({"chain": verification.model_dump(mode="json"), "calibration": asdict(report)})
    return 0


def _audit_core(args: argparse.Namespace) -> int:
    _print_json(
        asdict(
            audit_core(
                args.repo,
                locks_root=args.locks,
                ledger_path=args.ledger,
            )
        )
    )
    return 0


def _load_v1_1_config(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = Path(path).read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict) or raw.get("framework_version") != "v1.1":
        raise ValueError("v1.1 config must identify framework_version v1.1")
    if raw.get("market_inputs") != {"permitted": False}:
        raise ValueError("v1.1 config must explicitly prohibit market inputs")
    return raw, hashlib.sha256(payload).hexdigest()


def _fit_strength_v1_1(args: argparse.Namespace) -> int:
    raw_config, _config_hash = _load_v1_1_config(args.config)
    records = tuple(
        StrengthOutcomeRecord.model_validate(item)
        for item in json.loads(Path(args.records).read_text(encoding="utf-8"))
    )
    cutoff = datetime.fromisoformat(args.cutoff).astimezone(UTC)
    fit = fit_dynamic_strength(
        records,
        tour=Tour(args.tour),
        cutoff_utc=cutoff,
        fitted_at_utc=datetime.now(UTC),
        config=DynamicStrengthConfig.model_validate(raw_config["strength_anchor"]),
        code_commit=capture_code_provenance(args.repo).commit,
    )
    artifact = write_strength_artifact(fit, args.artifact_root)
    _print_json({"artifact_id": artifact.artifact_id, "directory": str(artifact.directory)})
    return 0


def _import_elo_v1_1(args: argparse.Namespace) -> int:
    raw_config, _config_hash = _load_v1_1_config(args.config)
    cutoff = datetime.fromisoformat(args.cutoff).astimezone(UTC)
    fitted_at = datetime.now(UTC)
    fit = import_surface_elo_csv(
        args.ratings,
        args.source_manifest,
        tour=Tour(args.tour),
        cutoff_utc=cutoff,
        fitted_at_utc=fitted_at,
        config=SurfaceEloConfig.model_validate(raw_config["surface_elo_anchor"]),
        code_commit=capture_code_provenance(args.repo).commit,
    )
    artifact = write_surface_elo_artifact(fit, args.artifact_root)
    _print_json({"artifact_id": artifact.artifact_id, "directory": str(artifact.directory)})
    return 0


def _create_fixed_elo_integration_v1_1(args: argparse.Namespace) -> int:
    raw_config, _config_hash = _load_v1_1_config(args.config)
    cutoff = datetime.fromisoformat(args.cutoff).astimezone(UTC)
    fitted_at = datetime.now(UTC)
    fit = create_fixed_strength_integration(
        tour=Tour(args.tour),
        training_cutoff_utc=cutoff,
        fitted_at_utc=fitted_at,
        config=StrengthIntegrationConfig.model_validate(raw_config["strength_integration"]),
        anchor_weight=float(raw_config["surface_elo_integration"]["anchor_weight"]),
        selection_reference=str(
            raw_config["surface_elo_integration"]["selection_reference"]
        ),
        code_commit=capture_code_provenance(args.repo).commit,
    )
    artifact = write_strength_integration_artifact(fit, args.artifact_root)
    _print_json({"artifact_id": artifact.artifact_id, "directory": str(artifact.directory)})
    return 0


def _fit_integration_v1_1(args: argparse.Namespace) -> int:
    raw_config, _config_hash = _load_v1_1_config(args.config)
    records = tuple(
        CrossFittedStrengthRecord.model_validate(item)
        for item in json.loads(Path(args.records).read_text(encoding="utf-8"))
    )
    cutoff = datetime.fromisoformat(args.cutoff).astimezone(UTC)
    fit = fit_strength_integration(
        records,
        tour=Tour(args.tour),
        training_cutoff_utc=cutoff,
        fitted_at_utc=datetime.now(UTC),
        config=StrengthIntegrationConfig.model_validate(
            raw_config["strength_integration"]
        ),
        code_commit=capture_code_provenance(args.repo).commit,
    )
    artifact = write_strength_integration_artifact(fit, args.artifact_root)
    _print_json({"artifact_id": artifact.artifact_id, "directory": str(artifact.directory)})
    return 0


def _create_snapshot_v1_1(args: argparse.Namespace) -> int:
    base = ModelSnapshot.model_validate_json(Path(args.base_snapshot).read_bytes())
    anchor = load_v1_1_strength_anchor_artifact(args.strength_artifact)
    integration = load_strength_integration_artifact(args.integration_artifact)
    _raw, config_hash = _load_v1_1_config(args.config)
    snapshot = create_v1_1_snapshot(
        base,
        strength_anchor=anchor,
        strength_integration=integration,
        framework_config_hash=config_hash,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(snapshot.canonical_json(), encoding="utf-8")
    _print_json({"snapshot_id": snapshot.snapshot_id, "path": str(output.resolve())})
    return 0


class _DirectoryOutcomeRevealer:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def reveal(
        self,
        match_id: str,
        *,
        locked_content_sha256: str,
    ) -> OfficialHistoricalOutcome:
        if not locked_content_sha256:
            raise ValueError("outcome reveal requires an already-persisted lock hash")
        path = (self.directory / f"{match_id}.json").resolve()
        path.relative_to(self.directory)
        return OfficialHistoricalOutcome.model_validate_json(path.read_bytes())


def _backtest(args: argparse.Namespace) -> int:
    targets_payload = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    targets = tuple(HistoricalForecastTarget.model_validate(item) for item in targets_payload)
    snapshots = tuple(
        ModelSnapshot.model_validate_json(path.read_bytes())
        for path in sorted(Path(args.snapshots).glob("*.json"))
    )
    policy = _path_policy(args)
    report = run_rolling_backtest(
        targets,
        snapshots=SnapshotCatalog(snapshots),
        outcomes=_DirectoryOutcomeRevealer(args.outcomes),
        source_manifest=load_source_manifest(args.source_manifest),
        code=capture_code_provenance(args.repo),
        lock_store=LockStore(args.locks),
        ledger=CalibrationLedger(args.ledger),
        backtest_run_id=args.run_id,
        historical_filter=HistoricalFilter(
            tour=None if args.tour is None else Tour(args.tour),
            start_date=None if args.start is None else date.fromisoformat(args.start),
            end_date=None if args.end is None else date.fromisoformat(args.end),
            events=tuple(args.event),
        ),
        path_count_policy=policy,
        execution_mode=args.mode,
    )
    _print_json(asdict(report))
    return 0


def _add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", choices=("production", "development", "test"), default="production"
    )
    parser.add_argument("--standard-paths", type=int, default=1_000)
    parser.add_argument("--escalated-paths", type=int, default=4_000)
    parser.add_argument("--minimum-settled-paths", type=int, default=500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tennis", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lock = commands.add_parser("lock-match", help="create a lock from a prepared request")
    lock.add_argument("--request", required=True)
    lock.add_argument("--locks", required=True)
    lock.add_argument("--repo", default=".")
    lock.add_argument("--allow-dirty", action="store_true")
    _add_path_options(lock)
    lock.set_defaults(handler=_lock_match)

    verify = commands.add_parser("verify-lock", help="verify one immutable lock")
    verify.add_argument("--locks", required=True)
    verify.add_argument("base_lock_id")
    verify.add_argument("revision", type=int)
    verify.add_argument("--reproduce", action="store_true")
    verify.add_argument(
        "--replay-level",
        choices=tuple(item.value for item in ReplayLevel),
        default=ReplayLevel.SAME_RUNTIME_EXACT.value,
    )
    verify.add_argument("--semantic-tolerance", type=float, default=1e-12)
    verify.set_defaults(handler=_verify_lock)

    render = commands.add_parser("render-lock", help="render a stored locked match card")
    render.add_argument("--locks", required=True)
    render.add_argument("base_lock_id")
    render.add_argument("revision", type=int)
    render.set_defaults(handler=_render_lock)

    ledger = commands.add_parser("summarize-ledger", help="calculate calibration diagnostics")
    ledger.add_argument("ledger")
    ledger.set_defaults(handler=_summarize_ledger)

    backtest = commands.add_parser("backtest", help="run strict lock-before-reveal backtests")
    backtest.add_argument("--targets", required=True)
    backtest.add_argument("--snapshots", required=True)
    backtest.add_argument("--outcomes", required=True)
    backtest.add_argument("--source-manifest", required=True)
    backtest.add_argument("--locks", required=True)
    backtest.add_argument("--ledger", required=True)
    backtest.add_argument("--run-id", required=True)
    backtest.add_argument("--tour", choices=("ATP", "WTA"))
    backtest.add_argument("--start", help="inclusive target date, YYYY-MM-DD")
    backtest.add_argument("--end", help="inclusive target date, YYYY-MM-DD")
    backtest.add_argument("--event", action="append", default=[], help="exact event filter")
    backtest.add_argument("--repo", default=".")
    _add_path_options(backtest)
    backtest.set_defaults(handler=_backtest)

    audit = commands.add_parser("audit-core", help="run read-only operational audit checks")
    audit.add_argument("--repo", default=".")
    audit.add_argument("--locks")
    audit.add_argument("--ledger")
    audit.set_defaults(handler=_audit_core)

    strength = commands.add_parser(
        "fit-strength-v1.1", help="fit a cutoff-safe experimental strength anchor"
    )
    strength.add_argument("--records", required=True)
    strength.add_argument("--config", default="config/model_v1_1.yaml")
    strength.add_argument("--tour", choices=("ATP", "WTA"), required=True)
    strength.add_argument("--cutoff", required=True)
    strength.add_argument("--artifact-root", required=True)
    strength.add_argument("--repo", default=".")
    strength.set_defaults(handler=_fit_strength_v1_1)

    elo = commands.add_parser(
        "import-elo-v1.1", help="package a cutoff-safe surface Elo table"
    )
    elo.add_argument("--ratings", required=True)
    elo.add_argument("--source-manifest", required=True)
    elo.add_argument("--config", default="config/model_v1_1.yaml")
    elo.add_argument("--tour", choices=("ATP", "WTA"), required=True)
    elo.add_argument("--cutoff", required=True)
    elo.add_argument("--artifact-root", required=True)
    elo.add_argument("--repo", default=".")
    elo.set_defaults(handler=_import_elo_v1_1)

    fixed_integration = commands.add_parser(
        "create-fixed-elo-integration-v1.1",
        help="package the validated fixed Elo/component blend",
    )
    fixed_integration.add_argument("--config", default="config/model_v1_1.yaml")
    fixed_integration.add_argument("--tour", choices=("ATP", "WTA"), required=True)
    fixed_integration.add_argument("--cutoff", required=True)
    fixed_integration.add_argument("--artifact-root", required=True)
    fixed_integration.add_argument("--repo", default=".")
    fixed_integration.set_defaults(handler=_create_fixed_elo_integration_v1_1)

    integration = commands.add_parser(
        "fit-integration-v1.1", help="fit the cross-fitted strength integration layer"
    )
    integration.add_argument("--records", required=True)
    integration.add_argument("--config", default="config/model_v1_1.yaml")
    integration.add_argument("--tour", choices=("ATP", "WTA"), required=True)
    integration.add_argument("--cutoff", required=True)
    integration.add_argument("--artifact-root", required=True)
    integration.add_argument("--repo", default=".")
    integration.set_defaults(handler=_fit_integration_v1_1)

    snapshot = commands.add_parser(
        "create-snapshot-v1.1", help="bind v1.1 artifacts to a complete v1.0 snapshot"
    )
    snapshot.add_argument("--base-snapshot", required=True)
    snapshot.add_argument("--strength-artifact", required=True)
    snapshot.add_argument("--integration-artifact", required=True)
    snapshot.add_argument("--config", default="config/model_v1_1.yaml")
    snapshot.add_argument("--output", required=True)
    snapshot.set_defaults(handler=_create_snapshot_v1_1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
