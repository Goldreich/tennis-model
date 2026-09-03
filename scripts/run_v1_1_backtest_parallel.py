from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import tennis_model.calibration.v1_1_backtest as backtest


_SERIAL_PROTECTED_BACKTEST = backtest.protected_prop_backtest


def _run_protected_chunk(
    payload: tuple[dict[str, Any], pd.DataFrame, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config, variants, checkpoint = payload
    shard_config = deepcopy(config)
    shard_config["protected_props"] = dict(shard_config["protected_props"])
    shard_config["protected_props"]["checkpoints"] = [checkpoint]
    estimates, _diagnostics, paths = _SERIAL_PROTECTED_BACKTEST(
        shard_config, variants
    )
    return estimates, paths


def protected_prop_backtest_parallel(
    config: Mapping[str, Any],
    variants: pd.DataFrame,
    *,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = variants.loc[variants["variant"].eq("complete_v1_1")]
    sample = backtest._protected_sample(config, complete)
    match_ids = sample["match_id"].to_numpy(object)
    worker_count = max(1, min(int(workers), len(match_ids)))
    if worker_count == 1:
        return _SERIAL_PROTECTED_BACKTEST(config, variants)

    id_chunks = [chunk.tolist() for chunk in np.array_split(match_ids, worker_count)]
    variant_chunks = [
        variants.loc[variants["match_id"].isin(chunk)].copy()
        for chunk in id_chunks
        if chunk
    ]

    checkpoints = [int(value) for value in config["protected_props"]["checkpoints"]]
    target = float(config["protected_props"]["aggregate_mcse_target"])
    diagnostics: list[dict[str, float | int]] = []
    final_estimates: pd.DataFrame | None = None
    final_paths: pd.DataFrame | None = None

    for checkpoint in checkpoints:
        jobs = [
            (dict(config), chunk, checkpoint)
            for chunk in variant_chunks
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(_run_protected_chunk, jobs))

        estimates = pd.concat(
            [result[0] for result in results], ignore_index=True, sort=False
        ).sort_values(["match_id", "variant"], kind="stable", ignore_index=True)
        paths = pd.concat(
            [result[1] for result in results], ignore_index=True, sort=False
        ).sort_values(
            ["match_id", "path_index", "variant"],
            kind="stable",
            ignore_index=True,
        )

        wide = estimates.pivot(
            index="match_id", columns="variant", values=["ace_brier", "df_brier"]
        )
        ace_delta = (
            wide[("ace_brier", "complete_v1_1")]
            - wide[("ace_brier", "v1_0")]
        )
        df_delta = (
            wide[("df_brier", "complete_v1_1")]
            - wide[("df_brier", "v1_0")]
        )
        ace_mcse = float(
            np.sqrt(np.var(ace_delta, ddof=1) / len(ace_delta) / checkpoint)
        )
        df_mcse = float(
            np.sqrt(np.var(df_delta, ddof=1) / len(df_delta) / checkpoint)
        )
        diagnostics.append(
            {
                "paths": checkpoint,
                "ace_brier_delta_mcse": ace_mcse,
                "df_brier_delta_mcse": df_mcse,
            }
        )
        final_estimates = estimates
        final_paths = paths
        if max(ace_mcse, df_mcse) <= target:
            break

    if final_estimates is None or final_paths is None:
        raise RuntimeError("parallel protected simulation produced no estimates")
    return (
        final_estimates,
        pd.DataFrame.from_records(diagnostics),
        final_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Tennis Model v1.1 backtest with parallel protected paths"
    )
    parser.add_argument("--config", default="config/model_v1_1_backtest.yaml")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")

    backtest.protected_prop_backtest = lambda config, variants: (
        protected_prop_backtest_parallel(config, variants, workers=args.workers)
    )
    output = backtest.run_backtest(args.config, repo_root=Path(args.repo))
    print(output)


if __name__ == "__main__":
    main()
