"""Code-state capture for reproducible prediction locks."""

from __future__ import annotations

import hashlib
import io
import platform
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pydantic
import scipy  # type: ignore[import-untyped]

from tennis_model.locking.models import CodeProvenance, RuntimeFingerprint


class CodeProvenanceError(RuntimeError):
    """The repository state could not be captured unambiguously."""


def _git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CodeProvenanceError(f"cannot inspect git state: {exc}") from exc
    return result.stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_code_provenance(
    repo_root: str | Path,
    *,
    excluded_untracked_prefixes: tuple[str, ...] = (
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
    ),
) -> CodeProvenance:
    """Record staged, unstaged, and relevant untracked content separately."""

    repo = Path(repo_root).resolve()
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    staged = _git(repo, "diff", "--cached", "--binary", "--")
    unstaged = _git(repo, "diff", "--binary", "--")
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    untracked_digest = hashlib.sha256()
    untracked_files: list[str] = []
    for raw_entry in status.split(b"\0"):
        if not raw_entry.startswith(b"?? "):
            continue
        relative = raw_entry[3:].decode("utf-8", errors="surrogateescape")
        normalized_relative = relative.replace("\\", "/")
        if any(normalized_relative.startswith(prefix) for prefix in excluded_untracked_prefixes):
            continue
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise CodeProvenanceError(
                "git reported an untracked path outside the repository"
            ) from exc
        untracked_files.append(normalized_relative)
        untracked_digest.update(normalized_relative.encode("utf-8", errors="surrogateescape"))
        untracked_digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    untracked_digest.update(chunk)
        untracked_digest.update(b"\0")
    staged_sha256 = _sha256(staged)
    unstaged_sha256 = _sha256(unstaged)
    untracked_sha256 = untracked_digest.hexdigest()
    dirty = bool(staged or unstaged or untracked_files)
    combined = None
    if dirty:
        digest = hashlib.sha256()
        for label, value in (
            (b"staged", staged_sha256),
            (b"unstaged", unstaged_sha256),
            (b"untracked", untracked_sha256),
        ):
            digest.update(label + b"\0" + value.encode("ascii") + b"\0")
        combined = digest.hexdigest()
    return CodeProvenance(
        commit=commit,
        dirty=dirty,
        diff_sha256=combined,
        fingerprint_version="complete-git-state/v2",
        staged_sha256=staged_sha256,
        unstaged_sha256=unstaged_sha256,
        untracked_sha256=untracked_sha256,
        relevant_untracked_files=tuple(sorted(untracked_files)),
    )


def capture_runtime_fingerprint(
    *,
    simulator_algorithm_version: str,
    rng_bit_generator: str = "PCG64",
    chunk_size: int = 1,
    thread_count: int = 1,
    process_count: int = 1,
) -> RuntimeFingerprint:
    """Capture the execution context needed to qualify replay claims."""

    output = io.StringIO()
    with redirect_stdout(output):
        np.show_config()
    blas_lines = tuple(
        line.strip()
        for line in output.getvalue().splitlines()
        if line.strip() and any(token in line.casefold() for token in ("blas", "lapack"))
    )
    return RuntimeFingerprint(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        operating_system=platform.system(),
        architecture=platform.architecture()[0],
        machine=platform.machine(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        pydantic_version=pydantic.__version__,
        rng_bit_generator=rng_bit_generator,
        numpy_random_context="numpy.random.SeedSequence + Generator",
        simulator_algorithm_version=simulator_algorithm_version,
        chunk_size=chunk_size,
        thread_count=thread_count,
        process_count=process_count,
        blas_backend=blas_lines,
    )


def enforce_dirty_tree_policy(code: CodeProvenance, *, allow_dirty: bool) -> None:
    """Reject dirty production state unless the caller explicitly permits its recorded hash."""

    if code.dirty and not allow_dirty:
        raise CodeProvenanceError(
            "prediction lock refused: repository is dirty; commit changes or explicitly allow "
            "a lock that records the dirty diff hash"
        )
