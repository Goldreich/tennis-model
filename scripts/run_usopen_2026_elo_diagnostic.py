from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_model.calibration.v1_1_backtest import (
    _elo_predictions,
    load_config,
    prepare_historical_data,
)


# Frozen before this tournament diagnostic is evaluated. These are the settings
# selected in both 2025 outer folds of the historical v1.1 backtest.
ELO_K = 16.0
SURFACE_BLEND = 0.50
ELO_LOGIT_BLEND = 0.75
FORECAST_CUTOFF = datetime(2026, 8, 30, tzinfo=UTC)


# Each tuple is tour, target player, opponent, submitted v1.0 percentage,
# crowd percentage, and settled Relative Brier Points gap. Crowd forecasts are
# used only to recover the already-revealed binary outcome from the settlement
# display; they never enter an Elo or blended forecast.
SCORECARD = [
    ("WTA", "A. Potapova", "D. Semenistaja", 83, 73, 6.8),
    ("ATP", "D. Blanch", "T. Fritz", 11, 12, 2.4),
    ("WTA", "E. Mertens", "K. Quevedo", 59, 76, -8.7),
    ("WTA", "T. Frodin", "E. Rybakina", 37, 15, -7.6),
    ("WTA", "R. Montgomery", "M. Sakkari", 57, 39, -15.6),
    ("ATP", "A. Fery", "L. Musetti", 21, 27, 4.8),
    ("ATP", "F. Cerundolo", "F. Misolic", 72, 78, 0.3),
    ("ATP", "F. Cobolli", "F. Comesana", 55, 74, -11.4),
    ("WTA", "B. Bencic", "Y. Putintseva", 68, 73, 8.8),
    ("ATP", "M. Damm", "F. Tiafoe", 41, 25, -8.8),
    ("WTA", "N. Osaka", "A. Zakharova", 82, 80, 3.4),
    ("WTA", "I. Swiatek", "X. Wang", 86, 85, 3.0),
    ("ATP", "T. Griekspoor", "B. Shelton", 34, 23, -4.6),
    ("ATP", "F. Auger-Aliassime", "R. Hijikata", 84, 80, 3.5),
    ("ATP", "N. Borges", "L. Tien", 41, 37, -1.3),
    ("WTA", "L. Noskova", "K. Volynets", 69, 68, 2.4),
    ("WTA", "S. Cirstea", "J. Grabher", 86, 79, 5.5),
    ("ATP", "M. Arnaldi", "J. Duckworth", 58, 59, 2.6),
    ("WTA", "D. Shnaider", "D. Snigur", 61, 66, -2.6),
    ("WTA", "A. Anisimova", "A. Krueger", 75, 75, 2.1),
    ("ATP", "C. Alcaraz", "R. Safiullin", 80, 79, 2.6),
    ("ATP", "L. Darderi", "H. Wendelken", 40, 69, -24.1),
    ("ATP", "J. Mensik", "S. Mochizuki", 66, 81, -5.6),
    ("ATP", "S. Baez", "B. Nakashima", 25, 26, 2.8),
    ("ATP", "T. Barrios Vera", "A. Blockx", 79, 28, -52.2),
    ("ATP", "A. Rublev", "O. Virtanen", 74, 82, -3.2),
    ("WTA", "M. Chwalinska", "T. Townsend", 20, 34, 9.5),
    ("WTA", "S. Bejlek", "C. Bucsa", 27, 67, 40.8),
    ("WTA", "A. Blinkova", "A. Kalinskaya", 32, 30, -0.2),
    ("WTA", "M. Bouzkova", "E. Jacquemot", 69, 75, -0.3),
    ("WTA", "A. Li", "A. Ruzic", 66, 66, 2.0),
    ("WTA", "C. Osorio", "A. Sabalenka", 14, 12, 0.8),
    ("WTA", "M. Keys", "A. Korneeva", 74, 70, 3.8),
    ("ATP", "Y. Hanfmann", "A. Tabilo", 55, 43, -10.9),
    ("WTA", "M. Andreeva", "J. Tjen", 63, 80, -7.4),
    ("WTA", "S. Kraus", "K. Muchova", 18, 19, 3.0),
    ("ATP", "A. Kovacevic", "V. Vacherot", 18, 44, 17.3),
    ("ATP", "A. Fils", "S. Tsitsipas", 50, 69, 25.6),
    ("ATP", "A. Bublik", "J. Wolf", 68, 68, 1.4),
    ("ATP", "N. Djokovic", "M. Navone", 97, 82, -24.7),
    ("WTA", "S. Sierra", "E. Svitolina", 20, 16, 0.7),
    ("WTA", "L. Boisson", "E. Navarro", 25, 27, 2.3),
    ("ATP", "H. Gaston", "D. Medvedev", 15, 17, 2.7),
    ("WTA", "J. Pegula", "E. Ruse", 83, 82, 3.1),
    ("WTA", "S. Hunter", "M. Kostyuk", 35, 18, -7.3),
    ("ATP", "T. Etcheverry", "V. Kopriva", 76, 63, 8.9),
    ("WTA", "E. Alexandrova", "M. Kessler", 63, 61, 2.9),
    ("ATP", "C. Norrie", "L. Van Assche", 63, 63, 1.9),
    ("ATP", "T. Paul", "C. Wong", 76, 78, 1.5),
    ("WTA", "V. Erjavec", "J. Paolini", 18, 19, 1.8),
    ("WTA", "B. Krejcikova", "K. Rakhimova", 72, 68, -3.3),
    ("ATP", "A. Rinderknech", "S. Shimabukuro", 48, 70, -16.3),
    ("WTA", "L. Fernandez", "S. Zhang", 70, 63, 5.7),
]


# Rounded submission and crowd probabilities coincide in these four rows, so
# the RBP sign alone cannot recover the result. These completed results were
# resolved separately from the official US Open record.
EQUAL_ROUNDED_OUTCOMES = {
    ("WTA", "A. Anisimova", "A. Krueger"): 1,
    ("WTA", "A. Li", "A. Ruzic"): 1,
    ("ATP", "A. Bublik", "J. Wolf"): 1,
    ("ATP", "C. Norrie", "L. Van Assche"): 0,
}


def _ascii_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z]+", ascii_value)


def _name_signature(value: str) -> tuple[str, str]:
    tokens = _ascii_tokens(value)
    if not tokens:
        raise ValueError(f"empty player name: {value!r}")
    return tokens[0][0], tokens[-1]


def _player_directory(matches: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for side in ("winner", "loser"):
        name_column = f"{side}_name"
        id_column = f"{side}_player_id"
        for row in matches[[name_column, id_column, "tour", "event_start_date"]].itertuples(index=False, name=None):
            name, player_id, tour, date = row
            if pd.isna(name) or pd.isna(player_id):
                continue
            initial, surname = _name_signature(str(name))
            records.append(
                {
                    "tour": str(tour),
                    "initial": initial,
                    "surname": surname,
                    "player_name": str(name),
                    "player_id": str(player_id),
                    "last_seen": date,
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values("last_seen")
        .drop_duplicates(["tour", "initial", "surname"], keep="last")
    )


def _resolve_player(directory: pd.DataFrame, tour: str, name: str) -> tuple[str, bool]:
    initial, surname = _name_signature(name)
    found = directory.loc[
        directory["tour"].eq(tour)
        & directory["initial"].eq(initial)
        & directory["surname"].eq(surname)
    ]
    if len(found) == 1:
        return str(found.iloc[0]["player_id"]), True
    slug = "-".join(_ascii_tokens(name))
    return f"{tour}:UNSEEN:{slug}", False


def _outcome(record: tuple[str, str, str, int, int, float]) -> tuple[int, str]:
    tour, target, opponent, submitted, crowd, gap = record
    if submitted == crowd:
        return EQUAL_ROUNDED_OUTCOMES[(tour, target, opponent)], "official-result-tie-resolution"
    return int((submitted - crowd) * gap > 0), "settled-rbp-sign"


def _expit(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(probability: float) -> float:
    bounded = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in [("overall", frame), *[(f"tour={tour}", part) for tour, part in frame.groupby("tour")]]:
        outcome = group["target_won"].to_numpy(float)
        for variant, column in (
            ("v1_0_submission", "v1_0_probability"),
            ("surface_elo", "elo_probability"),
            ("v1_0_elo_blend_75", "blended_probability"),
        ):
            probability = group[column].to_numpy(float)
            rows.append(
                {
                    "variant": variant,
                    "stratum": stratum,
                    "n": len(group),
                    "brier": float(np.mean(np.square(probability - outcome))),
                    "log_loss": float(
                        np.mean(
                            -outcome * np.log(np.clip(probability, 1e-6, 1.0))
                            - (1.0 - outcome) * np.log(np.clip(1.0 - probability, 1e-6, 1.0))
                        )
                    ),
                    "accuracy": float(np.mean((probability >= 0.5) == outcome)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 2026 US Open Elo diagnostic")
    parser.add_argument("--config", default="config/model_v1_1_backtest.yaml")
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--output",
        default="artifacts/backtests/usopen-2026-elo-diagnostic-v1",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config_path = (repo / args.config).resolve()
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    matches, _counts, _excluded, _manifest = prepare_historical_data(config, repo)
    directory = _player_directory(matches)

    records: list[dict[str, Any]] = []
    for scorecard_row in SCORECARD:
        tour, target, opponent, submitted, crowd, gap = scorecard_row
        target_id, target_seen = _resolve_player(directory, tour, target)
        opponent_id, opponent_seen = _resolve_player(directory, tour, opponent)
        target_won, outcome_source = _outcome(scorecard_row)
        records.append(
            {
                "match_id": f"usopen-2026-{tour.lower()}-{len(records) + 1:03d}",
                "tour": tour,
                "target_player": target,
                "opponent": opponent,
                "player_a_id": target_id,
                "player_b_id": opponent_id,
                "target_seen_in_history": target_seen,
                "opponent_seen_in_history": opponent_seen,
                "surface": "Hard",
                "v1_0_probability": submitted / 100.0,
                "target_won": target_won,
                "outcome_source": outcome_source,
                "settled_rbp_gap": gap,
            }
        )
    diagnostic = pd.DataFrame.from_records(records)

    elo_logits = _elo_predictions(
        matches,
        diagnostic,
        cutoff=FORECAST_CUTOFF,
        k=ELO_K,
        surface_blend=SURFACE_BLEND,
    )
    diagnostic["elo_logit"] = elo_logits
    diagnostic["elo_probability"] = diagnostic["elo_logit"].map(_expit)
    diagnostic["blended_logit"] = (
        (1.0 - ELO_LOGIT_BLEND) * diagnostic["v1_0_probability"].map(_logit)
        + ELO_LOGIT_BLEND * diagnostic["elo_logit"]
    )
    diagnostic["blended_probability"] = diagnostic["blended_logit"].map(_expit)
    metrics = _metric_rows(diagnostic)

    diagnostic.to_csv(output / "match_forecasts.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)

    overall = metrics.loc[metrics["stratum"].eq("overall")].set_index("variant")
    baseline = overall.loc["v1_0_submission"]
    report_lines = [
        "# 2026 US Open Surface-Elo Diagnostic",
        "",
        "**Status:** contaminated tournament diagnostic; not an independent promotion backtest.",
        "",
        "The Elo settings and blend were frozen from the historical 2025 outer-fold selections before this diagnostic was scored.",
        "Ranking, head-to-head, crowd probabilities, and tournament outcomes do not enter any forecast.",
        "",
        "## Overall metrics",
        "",
        "| Variant | N | Brier | Difference vs v1.0 | Log loss | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("v1_0_submission", "surface_elo", "v1_0_elo_blend_75"):
        row = overall.loc[variant]
        report_lines.append(
            f"| {variant} | {int(row['n'])} | {row['brier']:.6f} | "
            f"{row['brier'] - baseline['brier']:+.6f} | {row['log_loss']:.6f} | "
            f"{row['accuracy']:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "## Cohort notes",
            "",
            f"- Settled winner markets: {len(diagnostic)}.",
            f"- Players absent from the 2017-2025 source receive the preregistered 1500 Elo default: {int((~diagnostic['target_seen_in_history']).sum() + (~diagnostic['opponent_seen_in_history']).sum())} player-slots.",
            "- Four outcomes with equal rounded submission/crowd percentages were resolved from the official result; all others are algebraically recovered from the settled RBP sign.",
            "- No Elo K, surface blend, integration weight, or model feature was selected using these outcomes.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "usopen-2026-elo-diagnostic/v1",
        "status": "CONTAMINATED_DIAGNOSTIC_ONLY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "matches": len(diagnostic),
        "forecast_cutoff_utc": FORECAST_CUTOFF.isoformat(),
        "elo_k": ELO_K,
        "surface_blend": SURFACE_BLEND,
        "elo_logit_blend": ELO_LOGIT_BLEND,
        "history": "pinned 2017-2025 backtest source; single chronological Elo pass",
        "v1_0_input": "submitted integer probability from the settled 53-market scorecard",
        "outcome_policy": "settled RBP sign; official result when rounded probabilities tie",
        "config_sha256": _sha256(config_path),
        "script_sha256": _sha256(Path(__file__)),
        "production_default_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
