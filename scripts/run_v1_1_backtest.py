from __future__ import annotations

import argparse

from tennis_model.calibration.v1_1_backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Tennis Model v1.1 backtest")
    parser.add_argument("--config", default="config/model_v1_1_backtest.yaml")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    output = run_backtest(args.config, repo_root=args.repo)
    print(output)


if __name__ == "__main__":
    main()
