"""Create an operational v1.2 bundle from immutable frozen v1.1 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_model.estimation.snapshot import (  # noqa: E402
    ModelSnapshot,
    create_v1_2_snapshot,
)


DEFAULT_SOURCE = (
    ROOT / "artifacts" / "live-usopen-2026" / "official-2117-v1.1-christening"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "live-usopen-2026" / "official-2117-v1.2"
DEFAULT_CONFIG = ROOT / "config" / "model_v1_2.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        base = ModelSnapshot.model_validate_json(
            snapshot_path.read_text(encoding="utf-8")
        )
        snapshot = create_v1_2_snapshot(
            base,
            framework_config_hash=config_hash,
        )
        _write_json(snapshot_path, snapshot.model_dump(mode="json"))
        activated[tour] = {
            "base_snapshot_id": base.snapshot_id,
            "snapshot_id": snapshot.snapshot_id,
        }

    rally_root = ROOT / "artifacts" / "production" / "tennis-model-v1.2"
    report = {
        "schema_version": "tennis-model-v1.2-activation/v1",
        "framework_version": "v1.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(source.relative_to(ROOT)),
        "output_bundle": str(output.relative_to(ROOT)),
        "framework_config": str(config.relative_to(ROOT)),
        "framework_config_hash": config_hash,
        "snapshots": activated,
        "rally_artifacts": {
            "atp_active_pointer": str(
                (rally_root / "rally_termination_atp_current.json").relative_to(ROOT)
            ),
            "wta_active_pointer": str(
                (rally_root / "rally_termination_wta_current.json").relative_to(ROOT)
            ),
        },
    }
    _write_json(output / "v1_2_activation_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
