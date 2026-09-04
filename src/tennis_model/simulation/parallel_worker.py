"""Worker entry point for checkpointed deterministic simulation ranges."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from tennis_model.simulation.match import simulate_matches
from tennis_model.simulation.parallel import (
    _THREAD_LIMIT_VARIABLES,
    PreparedParallelSimulation,
    _atomic_pickle,
    _checkpoint_path,
    _load_pickle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ranges", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    args = parser.parse_args()

    prepared = _load_pickle(args.prepared.resolve())
    if not isinstance(prepared, PreparedParallelSimulation):
        raise RuntimeError("parallel worker received an invalid prepared payload")
    ranges = tuple(
        tuple(int(value) for value in item.split(":"))
        for item in args.ranges.split(",")
        if item
    )
    active_ranges = ranges[:10]
    remaining_ranges = ranges[10:]
    for start, count in active_ranges:
        target = _checkpoint_path(args.checkpoint_dir.resolve(), start, count)
        if target.exists():
            continue
        batch = simulate_matches(
            prepared.distribution,
            n_paths=count,
            seed=prepared.seed.to_seed_sequence(),
            trace_level=prepared.trace_level,
            first_server_id=prepared.first_server_id,
            duration_display_policy=prepared.duration_display_policy,
            path_start=start,
        )
        _atomic_pickle(
            target,
            {
                "schema_version": "simulation-path-checkpoint/v1",
                "setup_id": prepared.setup_id,
                "path_start": start,
                "path_count": count,
                "worker_pid": os.getpid(),
                "thread_limits": {
                    name: os.environ.get(name) for name in _THREAD_LIMIT_VARIABLES
                },
                "batch": batch,
            },
        )
        print(
            f"worker {args.worker_index}: checkpointed paths {start}-{start + count - 1}",
            flush=True,
        )
    if remaining_ranges:
        subprocess.run(
            (
                sys.executable,
                "-m",
                "tennis_model.simulation.parallel_worker",
                "--prepared",
                str(args.prepared.resolve()),
                "--checkpoint-dir",
                str(args.checkpoint_dir.resolve()),
                "--ranges",
                ",".join(f"{start}:{count}" for start, count in remaining_ranges),
                "--worker-index",
                str(args.worker_index),
            ),
            check=True,
        )


if __name__ == "__main__":
    main()
