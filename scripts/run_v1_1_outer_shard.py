from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tennis_model.calibration.v1_1_backtest import (
    _json_dump,
    _run_paths,
    _sha256_file,
    construct_outer_variants,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one disjoint v1.1 nested outer fold")
    parser.add_argument("--config", default="config/model_v1_1_backtest.yaml")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--tour", required=True, choices=("ATP", "WTA"))
    parser.add_argument("--year", required=True, type=int)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config = dict(load_config(repo / args.config))
    config["tours"] = [args.tour]
    config["outer_years"] = [args.year]
    paths = _run_paths(config, repo)
    forecasts = pd.read_parquet(paths.root / "cross_fitted_base_forecasts.parquet")
    module_path = repo / "src/tennis_model/calibration/v1_1_backtest.py"
    outer, selections = construct_outer_variants(
        config, forecasts, _sha256_file(module_path)
    )
    destination = paths.root / "outer_parts" / f"{args.tour.lower()}-{args.year}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    recovery = destination.with_suffix(".recovery.pkl")
    recovery_selections = destination.with_suffix(".recovery.json")
    outer.to_pickle(recovery)
    _json_dump(recovery_selections, selections)
    outer.to_parquet(temporary, index=False)
    temporary.replace(destination)
    _json_dump(destination.with_suffix(".json"), selections)
    recovery.unlink(missing_ok=True)
    recovery_selections.unlink(missing_ok=True)
    print(f"{args.tour} {args.year}: {outer['match_id'].nunique()} matches")


if __name__ == "__main__":
    main()
