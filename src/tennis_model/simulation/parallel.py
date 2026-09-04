"""Checkpointed subprocess execution for deterministic simulation path ranges."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from tennis_model.estimation.duration_model import DurationDisplayPolicy
from tennis_model.simulation.match import SimulationBatch
from tennis_model.simulation.parameters import (
    MatchParameterDistribution,
    SeedReference,
)

_THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class PreparedParallelSimulation:
    setup_id: str
    distribution: MatchParameterDistribution
    seed: SeedReference
    trace_level: Literal["summary", "points"]
    first_server_id: str | None
    duration_display_policy: DurationDisplayPolicy


def _canonical_setup_id(
    distribution: MatchParameterDistribution,
    seed: SeedReference,
    trace_level: str,
    first_server_id: str | None,
    duration_display_policy: DurationDisplayPolicy,
) -> str:
    payload = {
        "match_parameters": distribution.to_record().model_dump(mode="json"),
        "seed": seed.model_dump(mode="json"),
        "trace_level": trace_level,
        "first_server_id": first_server_id,
        "duration_display_policy": duration_display_policy.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _checkpoint_path(directory: Path, start: int, count: int) -> Path:
    return directory / f"chunk-{start:06d}-{count:05d}.pickle"


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ("taskkill", "/PID", str(process.pid), "/T", "/F"),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _validated_checkpoint(
    path: Path,
    *,
    setup_id: str,
    start: int,
    count: int,
) -> SimulationBatch:
    payload = _load_pickle(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"parallel checkpoint has invalid payload: {path}")
    if (
        payload.get("schema_version") != "simulation-path-checkpoint/v1"
        or payload.get("setup_id") != setup_id
        or payload.get("path_start") != start
        or payload.get("path_count") != count
        or not isinstance(payload.get("batch"), SimulationBatch)
        or payload.get("thread_limits")
        != {name: "1" for name in _THREAD_LIMIT_VARIABLES}
    ):
        raise RuntimeError(f"parallel checkpoint does not match this simulation: {path}")
    batch = payload["batch"]
    if batch.n_paths != count or batch.provenance.get("path_start") != start:
        raise RuntimeError(f"parallel checkpoint path range is inconsistent: {path}")
    return batch


def _materialize_from_covering_checkpoint(
    directory: Path,
    *,
    setup_id: str,
    start: int,
    count: int,
    cache: dict[Path, SimulationBatch],
    sources: tuple[tuple[int, int, Path], ...],
) -> bool:
    target = _checkpoint_path(directory, start, count)
    for source_start, source_count, source in sources:
        if (
            source_count <= count
            or source_start > start
            or start + count > source_start + source_count
        ):
            continue
        source_batch = cache.get(source)
        if source_batch is None:
            source_batch = _validated_checkpoint(
                source,
                setup_id=setup_id,
                start=source_start,
                count=source_count,
            )
            cache[source] = source_batch
        offset = start - source_start
        batch = SimulationBatch(
            context=source_batch.context,
            n_paths=count,
            seed_id=source_batch.seed_id,
            paths=source_batch.paths[offset : offset + count],
            provenance={**source_batch.provenance, "path_start": start},
        )
        _atomic_pickle(
            target,
            {
                "schema_version": "simulation-path-checkpoint/v1",
                "setup_id": setup_id,
                "path_start": start,
                "path_count": count,
                "worker_pid": os.getpid(),
                "thread_limits": {name: "1" for name in _THREAD_LIMIT_VARIABLES},
                "batch": batch,
            },
        )
        return True
    return False


def simulate_matches_parallel(
    distribution: MatchParameterDistribution,
    *,
    n_paths: int,
    seed: int | np.random.SeedSequence,
    workers: int,
    checkpoint_dir: str | Path,
    checkpoint_paths: int = 5_000,
    trace_level: Literal["summary", "points"] = "summary",
    first_server_id: str | None = None,
    duration_display_policy: DurationDisplayPolicy,
    show_progress: bool = False,
) -> SimulationBatch:
    """Simulate deterministic path ranges in restartable single-threaded workers."""

    if workers <= 0:
        raise ValueError("checkpointed simulation requires at least one worker")
    if checkpoint_paths <= 0:
        raise ValueError("checkpoint_paths must be positive")
    if isinstance(seed, np.random.SeedSequence):
        seed_reference = SeedReference.from_seed_sequence(seed)
    else:
        seed_reference = SeedReference.from_seed_sequence(np.random.SeedSequence(int(seed)))

    directory = Path(checkpoint_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    setup_id = _canonical_setup_id(
        distribution,
        seed_reference,
        trace_level,
        first_server_id,
        duration_display_policy,
    )
    prepared = PreparedParallelSimulation(
        setup_id=setup_id,
        distribution=distribution,
        seed=seed_reference,
        trace_level=trace_level,
        first_server_id=first_server_id,
        duration_display_policy=duration_display_policy,
    )
    prepared_path = directory / f"prepared-{setup_id}.pickle"
    if not prepared_path.exists():
        _atomic_pickle(prepared_path, prepared)

    chunks = tuple(
        (start, min(checkpoint_paths, n_paths - start))
        for start in range(0, n_paths, checkpoint_paths)
    )
    covering_sources: list[tuple[int, int, Path]] = []
    for source in directory.glob("chunk-*.pickle"):
        parts = source.stem.split("-")
        if len(parts) != 3:
            continue
        try:
            covering_sources.append((int(parts[1]), int(parts[2]), source))
        except ValueError:
            continue
    covering_source_index = tuple(covering_sources)
    batches: dict[int, SimulationBatch] = {}
    pending: list[tuple[int, int]] = []
    covering_cache: dict[Path, SimulationBatch] = {}
    for start, count in chunks:
        path = _checkpoint_path(directory, start, count)
        if not path.exists():
            _materialize_from_covering_checkpoint(
                directory,
                setup_id=setup_id,
                start=start,
                count=count,
                cache=covering_cache,
                sources=covering_source_index,
            )
        if path.exists():
            batches[start] = _validated_checkpoint(
                path, setup_id=setup_id, start=start, count=count
            )
        else:
            pending.append((start, count))

    if show_progress and batches:
        reused = sum(batch.n_paths for batch in batches.values())
        print(f"simulation resume: {reused}/{n_paths} paths already checkpointed", flush=True)

    processes: list[subprocess.Popen[bytes]] = []
    if pending:
        worker_count = min(workers, len(pending))
        reverse_pending = list(reversed(pending))
        assignments = [reverse_pending[index::worker_count] for index in range(worker_count)]
        environment = os.environ.copy()
        for name in _THREAD_LIMIT_VARIABLES:
            environment[name] = "1"
        source_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else source_root + os.pathsep + existing_pythonpath
        )
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            for worker_index, assignment in enumerate(assignments, start=1):
                ranges = ",".join(f"{start}:{count}" for start, count in assignment)
                command = (
                    sys.executable,
                    "-m",
                    "tennis_model.simulation.parallel_worker",
                    "--prepared",
                    str(prepared_path),
                    "--checkpoint-dir",
                    str(directory),
                    "--ranges",
                    ranges,
                    "--worker-index",
                    str(worker_index),
                )
                processes.append(
                    subprocess.Popen(
                        command,
                        env=environment,
                        creationflags=creationflags,
                    )
                )

            last_completed = len(batches)
            while processes:
                failed = [process for process in processes if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        "parallel simulation worker failed with exit code "
                        + ", ".join(str(process.returncode) for process in failed)
                    )
                processes = [process for process in processes if process.poll() is None]
                completed = sum(
                    _checkpoint_path(directory, start, count).exists()
                    for start, count in chunks
                )
                if show_progress and completed != last_completed:
                    completed_paths = sum(
                        count
                        for start, count in chunks
                        if _checkpoint_path(directory, start, count).exists()
                    )
                    print(
                        f"simulation progress: {completed_paths}/{n_paths} paths checkpointed",
                        flush=True,
                    )
                    last_completed = completed
                if processes:
                    time.sleep(2.0)
        except BaseException:
            for process in processes:
                _terminate_process_tree(process)
            raise

    for start, count in chunks:
        if start not in batches:
            batches[start] = _validated_checkpoint(
                _checkpoint_path(directory, start, count),
                setup_id=setup_id,
                start=start,
                count=count,
            )
    ordered = tuple(batches[start] for start, _count in chunks)
    first = ordered[0]
    paths = tuple(path for batch in ordered for path in batch.paths)
    provenance = {
        **first.provenance,
        "path_start": 0,
        "parallel_execution": "checkpointed-subprocess-ranges/v1",
        "parallel_workers": min(workers, len(chunks)),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_setup_id": setup_id,
    }
    return SimulationBatch(
        context=first.context,
        n_paths=len(paths),
        seed_id=first.seed_id,
        paths=paths,
        provenance=provenance,
    )


__all__ = ["PreparedParallelSimulation", "simulate_matches_parallel"]
