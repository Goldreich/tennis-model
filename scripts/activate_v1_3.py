"""Create an operational v1.3 bundle from immutable frozen v1.2 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_model.estimation.snapshot import (  # noqa: E402
    ModelSnapshot,
    create_v1_3_snapshot,
)
from tennis_model.estimation.strength_integration import (  # noqa: E402
    load_strength_integration_artifact,
)

DEFAULT_SOURCE = ROOT / "artifacts" / "live-usopen-2026" / "through-r64-v1.2"
DEFAULT_OUTPUT = ROOT / "artifacts" / "live-usopen-2026" / "through-r64-v1.3"
DEFAULT_CONFIG = ROOT / "config" / "model_v1_3.yaml"
PORTABLE_INTEGRATION_ROOT = ROOT / "model_artifacts" / "v1_1" / "strength_integration"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bind_portable_strength_integration(
    snapshot: ModelSnapshot,
    *,
    output: Path,
    tour: str,
) -> tuple[ModelSnapshot, Path, str]:
    reference = snapshot.strength_integration_artifact
    if reference is None:
        raise ValueError(f"{tour.upper()} snapshot has no strength-integration artifact")
    portable = PORTABLE_INTEGRATION_ROOT / tour / reference.artifact_id[:32]
    persisted = load_strength_integration_artifact(portable)
    if persisted.artifact_id != reference.artifact_id:
        raise ValueError(
            f"portable {tour.upper()} strength-integration artifact does not match "
            f"snapshot reference {reference.artifact_id}"
        )
    target = (
        output
        / "runtime-artifacts"
        / "strength-integration"
        / tour
        / reference.artifact_id[:32]
    )
    shutil.copytree(portable, target)
    rebound = reference.model_copy(update={"directory": target.resolve()})
    return (
        snapshot.model_copy(update={"strength_integration_artifact": rebound}),
        target,
        persisted.artifact_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    config = args.config.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {output}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not config.is_file():
        raise FileNotFoundError(config)

    shutil.copytree(source, output)
    config_hash = _sha256(config)
    activated: dict[str, dict[str, str]] = {}
    for tour in ("atp", "wta"):
        snapshot_path = output / f"model_snapshot_{tour}.json"
        base = ModelSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
        base, portable_integration, integration_artifact_id = (
            _bind_portable_strength_integration(
            base,
            output=output,
            tour=tour,
        )
        )
        snapshot = create_v1_3_snapshot(base, framework_config_hash=config_hash)
        _write_json(snapshot_path, snapshot.model_dump(mode="json"))
        activated[tour] = {
            "base_snapshot_id": base.snapshot_id,
            "snapshot_id": snapshot.snapshot_id,
            "strength_integration_artifact_id": integration_artifact_id,
            "strength_integration_directory": str(portable_integration),
        }

    report = {
        "schema_version": "tennis-model-v1.3-activation/v1",
        "framework_version": "v1.3",
        "created_at": datetime.now(UTC).isoformat(),
        "source_bundle": str(source.relative_to(ROOT)),
        "output_bundle": str(output.relative_to(ROOT)),
        "framework_config": str(config.relative_to(ROOT)),
        "framework_config_hash": config_hash,
        "snapshots": activated,
        "market_input": {
            "scope": "standalone_match_winner_only",
            "bookmaker": "Pinnacle",
            "required_at_lock_creation": True,
            "snapshot_schema": "pinnacle-moneyline-snapshot/v1",
        },
    }
    _write_json(output / "v1_3_activation_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
