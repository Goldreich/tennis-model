from __future__ import annotations

import argparse
from pathlib import Path

from tennis_model.calibration.v1_1_backtest import (
    _run_paths,
    generate_cross_fitted_forecasts,
    load_config,
    prepare_historical_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run disjoint v1.1 OOF forecast shards")
    parser.add_argument("--config", default="config/model_v1_1_backtest.yaml")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--tour", required=True, choices=("ATP", "WTA"))
    parser.add_argument("--year", required=True, action="append", type=int)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config = dict(load_config(repo / args.config))
    config["tours"] = [args.tour]
    config["_runtime_forecast_years"] = sorted(set(args.year))
    paths = _run_paths(config, repo)
    matches, counts, _excluded, manifest = prepare_historical_data(config, repo)
    forecasts = generate_cross_fitted_forecasts(
        config, repo, paths, matches, counts, manifest
    )
    print(f"{args.tour} {config['_runtime_forecast_years']}: {len(forecasts)} forecast rows")


if __name__ == "__main__":
    main()
