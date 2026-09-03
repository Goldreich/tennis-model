"""Fit game-day Elo fitness weights and test them on the 2026 US Open."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit

from tennis_model.estimation.game_day_elo import (
    FEATURE_NAMES,
    FitnessFeatures,
    GameDayEloFit,
    fit_game_day_elo,
    game_day_adjustment,
    probabilities,
    zero_adjustment_fit,
)


INVALID_SCORE = re.compile(r"(?:W[./]?O|WALKOVER|\bRET\b|\bDEF\b|\bABD\b|\bBYE\b)", re.I)
GRAND_SLAM_OFFSETS = {"Q1": -6, "Q2": -4, "Q3": -2, "R128": 0, "R64": 2, "R32": 4, "R16": 6, "QF": 8, "SF": 11, "F": 13}
STANDARD_OFFSETS = {"Q1": -3, "Q2": -2, "Q3": -1, "R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}


@dataclass(slots=True)
class Exposure:
    played_at: pd.Timestamp
    minutes: float | None


@dataclass(slots=True)
class PlayerState:
    global_elo: float
    surface_elo: dict[str, float]
    exposures: list[Exposure] = field(default_factory=list)
    last_played_at: pd.Timestamp | None = None
    return_severity: float = 0.0
    matches_since_return: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value).strip())
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _name_signature(value: str) -> tuple[str, str]:
    folded = unicodedata.normalize("NFKD", str(value).strip())
    ascii_text = folded.encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z]+", ascii_text)
    if not tokens:
        raise ValueError(f"empty player name: {value!r}")
    return tokens[0][0], tokens[-1]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def _round_offset(round_name: str, level: str) -> tuple[int, bool]:
    key = str(round_name).strip().upper()
    if key == "RR":
        return 0, False
    offsets = GRAND_SLAM_OFFSETS if str(level).strip().upper() == "G" else STANDARD_OFFSETS
    return offsets.get(key, 0), key in offsets


def _load_rows(config: Mapping[str, Any], repo: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source = repo / str(config["source_directory"])
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for tour in config["tours"]:
        prefix = str(tour).lower()
        names = [f"{prefix}_{year}.csv" for year in config["source_years"]]
        names.append(f"{prefix}_ongoing.csv")
        for priority, name in enumerate(names):
            path = source / name
            if not path.is_file():
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["tour"] = str(tour)
            frame["source_file"] = name
            frame["source_priority"] = priority
            frame["source_row"] = np.arange(2, len(frame) + 2)
            frames.append(frame)
            manifest.append(
                {
                    "path": str(path.relative_to(repo)).replace("\\", "/"),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                }
            )
    if not frames:
        raise RuntimeError("no main-tour source rows were found")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    required = (
        "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date",
        "match_num", "winner_name", "loser_name", "score", "best_of", "round", "minutes",
    )
    missing = [column for column in required if column not in raw]
    if missing:
        raise RuntimeError(f"source rows are missing required columns: {missing}")
    for column in ("tourney_date", "match_num", "best_of", "minutes"):
        raw[f"num_{column}"] = _numeric(raw[column])
    raw["event_start_date"] = pd.to_datetime(
        raw["tourney_date"], format="%Y%m%d", errors="coerce", utc=True
    )
    raw["winner_key"] = raw["tour"] + ":" + raw["winner_name"].map(_normalized_name)
    raw["loser_key"] = raw["tour"] + ":" + raw["loser_name"].map(_normalized_name)
    raw["event_name_key"] = raw["tourney_name"].map(_normalized_name)
    raw["dedupe_key"] = (
        raw["tour"] + "|" + raw["tourney_id"] + "|" + raw["tourney_date"] + "|"
        + raw["match_num"] + "|" + raw["winner_key"] + "|" + raw["loser_key"]
    )
    raw = (
        raw.sort_values(["source_priority", "source_row"])
        .drop_duplicates("dedupe_key", keep="last")
        .reset_index(drop=True)
    )
    levels = {tour: set(values) for tour, values in config["levels"].items()}
    eligible_level = pd.Series(
        [level in levels[tour] for level, tour in zip(raw["tourney_level"], raw["tour"])],
        index=raw.index,
    )
    eligible = (
        raw["event_start_date"].notna()
        & raw["surface"].isin(config["surfaces"])
        & eligible_level
        & raw["num_best_of"].isin((3, 5))
        & raw["winner_name"].ne("")
        & raw["loser_name"].ne("")
        & raw["winner_key"].ne(raw["loser_key"])
        & raw["score"].ne("")
        & ~raw["score"].str.contains(INVALID_SCORE, na=True)
    )
    raw = raw.loc[eligible].copy()
    offsets = [
        _round_offset(round_name, level)
        for round_name, level in zip(raw["round"], raw["tourney_level"])
    ]
    raw["round_offset_days"] = [item[0] for item in offsets]
    raw["round_proxy_reliable"] = [item[1] for item in offsets]
    raw["proxy_played_at"] = raw["event_start_date"] + pd.to_timedelta(
        raw["round_offset_days"], unit="D"
    )
    raw["event_id"] = raw["tour"] + "|" + raw["tourney_id"] + "|" + raw["tourney_date"]
    raw["match_id"] = raw["dedupe_key"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )
    return raw, manifest


def _new_state(config: Mapping[str, Any]) -> PlayerState:
    initial = float(config["base_elo"]["initial_rating"])
    return PlayerState(initial, defaultdict(lambda: initial))


def _fitness_features(
    state: PlayerState,
    played_at: pd.Timestamp,
    config: Mapping[str, Any],
) -> tuple[FitnessFeatures | None, float | None, bool]:
    feature = config["fitness_features"]
    lookback = int(feature["workload_lookback_days"])
    recent = [
        exposure
        for exposure in state.exposures
        if 0.0 <= (played_at - exposure.played_at).total_seconds() / 86400.0 <= lookback
    ]
    missing_minutes = any(exposure.minutes is None for exposure in recent)
    workload = 0.0
    if not missing_minutes:
        short_half_life = float(feature["workload_short_half_life_days"])
        long_half_life = float(feature["workload_long_half_life_days"])
        short_weight = float(feature["workload_short_weight"])
        unit = float(feature["workload_unit_minutes"])
        for exposure in recent:
            days = (played_at - exposure.played_at).total_seconds() / 86400.0
            decay = short_weight * 2.0 ** (-days / short_half_life)
            decay += (1.0 - short_weight) * 2.0 ** (-days / long_half_life)
            workload += float(exposure.minutes) * decay / unit

    gap_days: float | None = None
    short_recovery = 0.0
    return_from_layoff = 0.0
    if state.last_played_at is not None:
        gap_days = max(0.0, (played_at - state.last_played_at).total_seconds() / 86400.0)
        normal = float(feature["normal_recovery_days"])
        short_recovery = max(0.0, normal - gap_days) / normal
        threshold = float(feature["layoff_threshold_days"])
        if gap_days > threshold:
            return_from_layoff = math.log1p(
                (gap_days - threshold) / float(feature["layoff_unit_days"])
            )
        elif state.return_severity > 0.0:
            return_from_layoff = state.return_severity * math.exp(
                -state.matches_since_return / float(feature["return_decay_matches"])
            )
    if missing_minutes:
        return None, gap_days, True
    return FitnessFeatures(workload, short_recovery, return_from_layoff), gap_days, False


def _update_fitness_state(
    state: PlayerState,
    played_at: pd.Timestamp,
    minutes: float | None,
    gap_days: float | None,
    config: Mapping[str, Any],
) -> None:
    feature = config["fitness_features"]
    threshold = float(feature["layoff_threshold_days"])
    if gap_days is not None and gap_days > threshold:
        state.return_severity = math.log1p(
            (gap_days - threshold) / float(feature["layoff_unit_days"])
        )
        state.matches_since_return = 1
    elif state.return_severity > 0.0:
        state.matches_since_return += 1
        if state.matches_since_return >= 12:
            state.return_severity = 0.0
            state.matches_since_return = 0
    state.last_played_at = played_at
    state.exposures.append(Exposure(played_at, minutes))
    keep_days = int(feature["workload_lookback_days"]) + 1
    state.exposures = [
        exposure
        for exposure in state.exposures
        if (played_at - exposure.played_at).total_seconds() / 86400.0 <= keep_days
    ]


def _build_forecast_rows(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    states: defaultdict[str, PlayerState] = defaultdict(lambda: _new_state(config))
    base = config["base_elo"]
    surface_blend = float(base["surface_blend"])
    rating_scale = float(base["rating_scale"])
    k_factor = float(base["k_factor"])
    records: list[dict[str, Any]] = []
    ordered = raw.sort_values(
        ["proxy_played_at", "event_start_date", "event_id", "round_offset_days", "num_match_num", "match_id"]
    )
    for row in ordered.itertuples(index=False):
        a_key, b_key = sorted((row.winner_key, row.loser_key))
        a_is_winner = a_key == row.winner_key
        a_name = row.winner_name if a_is_winner else row.loser_name
        b_name = row.loser_name if a_is_winner else row.winner_name
        state_a, state_b = states[a_key], states[b_key]
        rating_a = (1.0 - surface_blend) * state_a.global_elo + surface_blend * state_a.surface_elo[row.surface]
        rating_b = (1.0 - surface_blend) * state_b.global_elo + surface_blend * state_b.surface_elo[row.surface]
        base_logit = math.log(10.0) * (rating_a - rating_b) / rating_scale
        features_a, gap_a, missing_a = _fitness_features(state_a, row.proxy_played_at, config)
        features_b, gap_b, missing_b = _fitness_features(state_b, row.proxy_played_at, config)
        record: dict[str, Any] = {
            "match_id": row.match_id,
            "event_id": row.event_id,
            "tour": row.tour,
            "tournament": row.tourney_name,
            "event_start_date": row.event_start_date,
            "proxy_played_at": row.proxy_played_at,
            "round": row.round,
            "round_proxy_reliable": bool(row.round_proxy_reliable),
            "surface": row.surface,
            "player_a_id": a_key,
            "player_b_id": b_key,
            "player_a_name": a_name,
            "player_b_name": b_name,
            "player_a_won": int(a_is_winner),
            "base_elo_a": rating_a,
            "base_elo_b": rating_b,
            "base_elo_logit": base_logit,
            "base_elo_probability": float(expit(base_logit)),
            "player_a_gap_days": gap_a,
            "player_b_gap_days": gap_b,
            "player_a_missing_recent_minutes": missing_a,
            "player_b_missing_recent_minutes": missing_b,
            "feature_complete": features_a is not None and features_b is not None,
            "source_file": row.source_file,
            "source_row": int(row.source_row),
        }
        for index, name in enumerate(FEATURE_NAMES):
            value_a = np.nan if features_a is None else features_a.as_array()[index]
            value_b = np.nan if features_b is None else features_b.as_array()[index]
            record[f"player_a_{name}"] = value_a
            record[f"player_b_{name}"] = value_b
            record[f"difference_{name}"] = value_a - value_b
        records.append(record)

        outcome = float(a_is_winner)
        expected = float(expit(base_logit))
        delta = k_factor * (outcome - expected)
        state_a.global_elo += delta
        state_b.global_elo -= delta
        state_a.surface_elo[row.surface] += delta
        state_b.surface_elo[row.surface] -= delta
        minutes = None if pd.isna(row.num_minutes) or row.num_minutes <= 0 else float(row.num_minutes)
        _update_fitness_state(state_a, row.proxy_played_at, minutes, gap_a, config)
        _update_fitness_state(state_b, row.proxy_played_at, minutes, gap_b, config)
    return pd.DataFrame.from_records(records)


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["base_elo_logit"].to_numpy(float),
        frame[[f"difference_{name}" for name in FEATURE_NAMES]].to_numpy(float),
        frame["player_a_won"].to_numpy(float),
    )


def _brier(outcome: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(probability - outcome)))


def _log_loss(outcome: np.ndarray, probability: np.ndarray) -> float:
    bounded = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(np.mean(-outcome * np.log(bounded) - (1.0 - outcome) * np.log(1.0 - bounded)))


def _fit_tour(
    frame: pd.DataFrame,
    tour: str,
    config: Mapping[str, Any],
) -> tuple[GameDayEloFit, list[dict[str, Any]]]:
    fit_config = config["fit"]
    start = pd.Timestamp(fit_config["fit_start_utc"])
    validation_start = pd.Timestamp(fit_config["validation_start_utc"])
    validation_end = pd.Timestamp(fit_config["validation_end_utc"])
    cutoff = pd.Timestamp(fit_config["final_cutoff_utc"])
    eligible = frame.loc[
        frame["tour"].eq(tour)
        & frame["feature_complete"]
        & frame["round_proxy_reliable"]
        & frame["proxy_played_at"].ge(start)
        & frame["proxy_played_at"].lt(cutoff)
    ]
    inner_train = eligible.loc[eligible["proxy_played_at"].lt(validation_start)]
    validation = eligible.loc[
        eligible["proxy_played_at"].ge(validation_start)
        & eligible["proxy_played_at"].lt(validation_end)
    ]
    if inner_train.empty or validation.empty:
        raise RuntimeError(f"{tour} fitness selection cohort is empty")
    train_arrays = _arrays(inner_train)
    validation_arrays = _arrays(validation)
    rows: list[dict[str, Any]] = []
    null_fit = zero_adjustment_fit(len(inner_train))
    null_probability = probabilities(validation_arrays[0], validation_arrays[1], null_fit.weights_elo)
    null_brier = _brier(validation_arrays[2], null_probability)
    rows.append(
        {
            "tour": tour,
            "candidate": "zero_adjustment",
            "l2_penalty": np.nan,
            "training_rows": len(inner_train),
            "validation_rows": len(validation),
            "validation_brier": null_brier,
            "validation_log_loss": _log_loss(validation_arrays[2], null_probability),
            **{f"weight_{name}_elo": 0.0 for name in FEATURE_NAMES},
        }
    )
    candidates: list[tuple[float, GameDayEloFit, float]] = []
    for l2_penalty in fit_config["l2_candidates"]:
        fitted = fit_game_day_elo(
            *train_arrays,
            l2_penalty=float(l2_penalty),
            maximum_absolute_weight_elo=float(fit_config["maximum_absolute_weight_elo"]),
        )
        predicted = probabilities(validation_arrays[0], validation_arrays[1], fitted.weights_elo)
        candidate_brier = _brier(validation_arrays[2], predicted)
        candidates.append((float(l2_penalty), fitted, candidate_brier))
        rows.append(
            {
                "tour": tour,
                "candidate": "ridge",
                "l2_penalty": float(l2_penalty),
                "training_rows": len(inner_train),
                "validation_rows": len(validation),
                "validation_brier": candidate_brier,
                "validation_log_loss": _log_loss(validation_arrays[2], predicted),
                **{
                    f"weight_{name}_elo": fitted.weights_elo[index]
                    for index, name in enumerate(FEATURE_NAMES)
                },
            }
        )
    best_l2, _best_inner, best_brier = min(candidates, key=lambda item: (item[2], -item[0]))
    minimum_gain = float(fit_config["minimum_validation_brier_gain"])
    final_arrays = _arrays(eligible)
    if null_brier - best_brier < minimum_gain:
        selected = zero_adjustment_fit(len(eligible))
    else:
        selected = fit_game_day_elo(
            *final_arrays,
            l2_penalty=best_l2,
            maximum_absolute_weight_elo=float(fit_config["maximum_absolute_weight_elo"]),
        )
    for row in rows:
        row["selected"] = (
            selected.l2_penalty is None and row["candidate"] == "zero_adjustment"
        ) or (
            selected.l2_penalty is not None
            and row["candidate"] == "ridge"
            and row["l2_penalty"] == selected.l2_penalty
        )
    return selected, rows


def _metric_records(frame: pd.DataFrame, variants: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups = [("overall", frame), *[(f"tour={tour}", group) for tour, group in frame.groupby("tour")]]
    for stratum, group in groups:
        outcome = group["player_a_won"].to_numpy(float)
        for variant, column in variants.items():
            probability = group[column].to_numpy(float)
            records.append(
                {
                    "stratum": stratum,
                    "variant": variant,
                    "n": len(group),
                    "brier": _brier(outcome, probability),
                    "log_loss": _log_loss(outcome, probability),
                    "accuracy": float(np.mean((probability >= 0.5) == outcome)),
                }
            )
    return records


def _paired_interval(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    config: Mapping[str, Any],
) -> dict[str, float]:
    outcome = frame["player_a_won"].to_numpy(float)
    baseline = frame[baseline_column].to_numpy(float)
    candidate = frame[candidate_column].to_numpy(float)
    brier_delta = np.square(candidate - outcome) - np.square(baseline - outcome)
    base_log = -(outcome * np.log(np.clip(baseline, 1e-9, 1.0)) + (1.0 - outcome) * np.log(np.clip(1.0 - baseline, 1e-9, 1.0)))
    candidate_log = -(outcome * np.log(np.clip(candidate, 1e-9, 1.0)) + (1.0 - outcome) * np.log(np.clip(1.0 - candidate, 1e-9, 1.0)))
    log_delta = candidate_log - base_log
    test_config = config["held_out_test"]
    rng = np.random.default_rng(int(test_config["bootstrap_seed"]))
    replicates = int(test_config["bootstrap_replicates"])
    indices = rng.integers(0, len(frame), size=(replicates, len(frame)))
    brier_bootstrap = brier_delta[indices].mean(axis=1)
    log_bootstrap = log_delta[indices].mean(axis=1)
    return {
        "brier_difference": float(np.mean(brier_delta)),
        "brier_ci_low": float(np.quantile(brier_bootstrap, 0.025)),
        "brier_ci_high": float(np.quantile(brier_bootstrap, 0.975)),
        "log_loss_difference": float(np.mean(log_delta)),
        "log_loss_ci_low": float(np.quantile(log_bootstrap, 0.025)),
        "log_loss_ci_high": float(np.quantile(log_bootstrap, 0.975)),
    }


def _scorecard_test(
    test: pd.DataFrame,
    weights: Mapping[str, GameDayEloFit],
    config: Mapping[str, Any],
    repo: Path,
) -> pd.DataFrame:
    path = repo / str(config["scorecard_forecasts_path"])
    if not path.is_file():
        return pd.DataFrame()
    scorecard = pd.read_csv(path)
    lookup: dict[tuple[str, frozenset[tuple[str, str]]], Any] = {}
    for row in test.itertuples(index=False):
        key = (row.tour, frozenset((_name_signature(row.player_a_name), _name_signature(row.player_b_name))))
        lookup[key] = row
    records: list[dict[str, Any]] = []
    blend = float(config["held_out_test"]["elo_logit_blend"])
    for row in scorecard.itertuples(index=False):
        key = (row.tour, frozenset((_name_signature(row.target_player), _name_signature(row.opponent))))
        match = lookup.get(key)
        if match is None or not match.feature_complete:
            continue
        target_is_a = _name_signature(row.target_player) == _name_signature(match.player_a_name)
        v1_a = float(row.v1_0_probability) if target_is_a else 1.0 - float(row.v1_0_probability)
        differences = np.asarray([getattr(match, f"difference_{name}") for name in FEATURE_NAMES])
        fitness_logit = float(match.base_elo_logit + math.log(10.0) / 400.0 * (differences @ weights[row.tour].weights_array()))
        records.append(
            {
                "match_id": match.match_id,
                "tour": row.tour,
                "player_a_name": match.player_a_name,
                "player_b_name": match.player_b_name,
                "player_a_won": match.player_a_won,
                "v1_0_probability_a": v1_a,
                "base_elo_probability": float(expit(match.base_elo_logit)),
                "game_day_elo_probability": float(expit(fitness_logit)),
                "base_integrated_probability": float(expit((1.0 - blend) * logit(np.clip(v1_a, 1e-6, 1.0 - 1e-6)) + blend * match.base_elo_logit)),
                "fitness_integrated_probability": float(expit((1.0 - blend) * logit(np.clip(v1_a, 1e-6, 1.0 - 1e-6)) + blend * fitness_logit)),
            }
        )
    return pd.DataFrame.from_records(records)


def _render_report(
    weights: Mapping[str, GameDayEloFit],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: Mapping[str, float],
    scorecard: pd.DataFrame,
    scorecard_metrics: pd.DataFrame,
    scorecard_paired: Mapping[str, float] | None,
) -> str:
    lines = [
        "# Tennis Model v1.2 Fitness Candidate Fit",
        "",
        "**Status:** experimental; production models unchanged.",
        "",
        "## Fitted game-day Elo weights",
        "",
        "| Tour | Recent workload | Short recovery | Return from layoff | Selected L2 | Training rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tour, fitted in weights.items():
        l2 = "zero adjustment" if fitted.l2_penalty is None else f"{fitted.l2_penalty:g}"
        lines.append(
            f"| {tour} | {fitted.weights_elo[0]:.3f} | {fitted.weights_elo[1]:.3f} | "
            f"{fitted.weights_elo[2]:.3f} | {l2} | {fitted.training_rows} |"
        )
    lines.extend(
        [
            "",
            "Weights are Elo points per feature unit and are constrained to be nonpositive.",
            "One recent-workload unit is 300 exponentially weighted main-tour minutes.",
            "",
            "## Held-out 2026 US Open: Elo anchor",
            "",
            "| Variant | N | Brier | Log loss | Accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    overall = metrics.loc[metrics["stratum"].eq("overall")]
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.variant} | {row.n} | {row.brier:.6f} | {row.log_loss:.6f} | {row.accuracy:.1%} |")
    lines.extend(
        [
            "",
            f"Paired Brier difference (fitness minus base): {paired['brier_difference']:+.6f} "
            f"(match-bootstrap 95% interval {paired['brier_ci_low']:+.6f} to {paired['brier_ci_high']:+.6f}).",
            f"Paired log-loss difference: {paired['log_loss_difference']:+.6f} "
            f"(match-bootstrap 95% interval {paired['log_loss_ci_low']:+.6f} to {paired['log_loss_ci_high']:+.6f}).",
        ]
    )
    if not scorecard.empty and not scorecard_metrics.empty and scorecard_paired is not None:
        lines.extend(
            [
                "",
                "## Existing v1.0 plus 75% Elo integration cohort",
                "",
                "| Variant | N | Brier | Log loss | Accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        score_overall = scorecard_metrics.loc[scorecard_metrics["stratum"].eq("overall")]
        for row in score_overall.itertuples(index=False):
            lines.append(f"| {row.variant} | {row.n} | {row.brier:.6f} | {row.log_loss:.6f} | {row.accuracy:.1%} |")
        lines.extend(
            [
                "",
                f"Integrated paired Brier difference: {scorecard_paired['brier_difference']:+.6f} "
                f"(95% interval {scorecard_paired['brier_ci_low']:+.6f} to {scorecard_paired['brier_ci_high']:+.6f}).",
            ]
        )
    lines.extend(
        [
            "",
            "## Coverage and limitations",
            "",
            f"- Completed US Open source matches: {len(test)}.",
            f"- Held-out matches with complete fitness features: {int(test['feature_complete'].sum())}.",
            f"- Existing v1.0 integration matches joined: {len(scorecard)}.",
            "- The US Open was excluded from coefficient fitting and ridge selection.",
            "- Exact match timestamps are unavailable; fixed round-day proxies are used.",
            "- Workload covers main-tour matches only and excludes training and lower-tour play.",
            "- Bootstrap intervals resample matches inside one tournament and do not establish cross-event generalization.",
            "- Retirement risk, serve primitives, and persistent Elo updates are unchanged.",
            "",
            "The candidate requires broader rolling-origin validation before production use.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_v1_2_fitness.yaml")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    config_path = (repo / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = repo / str(config["output_directory"])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fitness-partial-", dir=output.parent))
    try:
        raw, source_manifest = _load_rows(config, repo)
        forecasts = _build_forecast_rows(raw, config)
        weights: dict[str, GameDayEloFit] = {}
        validation_rows: list[dict[str, Any]] = []
        for tour in config["tours"]:
            weights[tour], rows = _fit_tour(forecasts, tour, config)
            validation_rows.extend(rows)
        validation = pd.DataFrame.from_records(validation_rows)

        test_config = config["held_out_test"]
        test_event = _normalized_name(str(test_config["tournament_name"]))
        test_date = pd.Timestamp(test_config["event_start_date"], tz="UTC")
        test = forecasts.loc[
            forecasts["tournament"].map(_normalized_name).eq(test_event)
            & forecasts["event_start_date"].eq(test_date)
            & forecasts["round_proxy_reliable"]
        ].copy()
        test = test.loc[test["feature_complete"]].copy()
        if test.empty:
            raise RuntimeError("held-out US Open fitness cohort is empty")
        test["game_day_elo_logit"] = np.nan
        test["game_day_elo_probability"] = np.nan
        test["player_a_fitness_adjustment_elo"] = np.nan
        test["player_b_fitness_adjustment_elo"] = np.nan
        for tour, group in test.groupby("tour"):
            indices = group.index
            feature_differences = group[[f"difference_{name}" for name in FEATURE_NAMES]].to_numpy(float)
            fitted = weights[tour]
            adjusted_probability = probabilities(group["base_elo_logit"].to_numpy(float), feature_differences, fitted.weights_elo)
            test.loc[indices, "game_day_elo_probability"] = adjusted_probability
            test.loc[indices, "game_day_elo_logit"] = logit(np.clip(adjusted_probability, 1e-9, 1.0 - 1e-9))
            a_features = group[[f"player_a_{name}" for name in FEATURE_NAMES]].to_numpy(float)
            b_features = group[[f"player_b_{name}" for name in FEATURE_NAMES]].to_numpy(float)
            test.loc[indices, "player_a_fitness_adjustment_elo"] = a_features @ fitted.weights_array()
            test.loc[indices, "player_b_fitness_adjustment_elo"] = b_features @ fitted.weights_array()
        if (test[["player_a_fitness_adjustment_elo", "player_b_fitness_adjustment_elo"]] > 1e-10).any().any():
            raise RuntimeError("fitness invariant failed: a player received a positive adjustment")

        metrics = pd.DataFrame.from_records(
            _metric_records(
                test,
                {"base_surface_elo": "base_elo_probability", "game_day_elo": "game_day_elo_probability"},
            )
        )
        paired = _paired_interval(test, "base_elo_probability", "game_day_elo_probability", config)
        scorecard = _scorecard_test(test, weights, config, repo)
        if scorecard.empty:
            scorecard_metrics = pd.DataFrame()
            scorecard_paired = None
        else:
            scorecard_metrics = pd.DataFrame.from_records(
                _metric_records(
                    scorecard,
                    {
                        "base_v1_0_elo_integration": "base_integrated_probability",
                        "fitness_v1_0_elo_integration": "fitness_integrated_probability",
                    },
                )
            )
            scorecard_paired = _paired_interval(
                scorecard,
                "base_integrated_probability",
                "fitness_integrated_probability",
                config,
            )

        fit_payload = {
            "schema_version": config["schema_version"],
            "framework_version": config["framework_version"],
            "fitted_at_utc": datetime.now(UTC).isoformat(),
            "training_cutoff_utc": config["fit"]["final_cutoff_utc"].isoformat(),
            "feature_names": FEATURE_NAMES,
            "tour_fits": {tour: asdict(fitted) for tour, fitted in weights.items()},
            "production_default_modified": False,
        }
        artifact_bytes = json.dumps(fit_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fit_payload["artifact_id"] = hashlib.sha256(artifact_bytes).hexdigest()
        (staging / "game_day_elo_fit.json").write_text(
            json.dumps(fit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation.to_csv(staging / "ridge_selection.csv", index=False)
        test.to_csv(staging / "usopen_test_forecasts.csv", index=False)
        metrics.to_csv(staging / "usopen_test_metrics.csv", index=False)
        if not scorecard.empty:
            scorecard.to_csv(staging / "scorecard_integration_test.csv", index=False)
            scorecard_metrics.to_csv(staging / "scorecard_integration_metrics.csv", index=False)
        report = _render_report(
            weights, validation, test, metrics, paired, scorecard, scorecard_metrics, scorecard_paired
        )
        (staging / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": "tennis-game-day-elo-run/v1",
            "framework_version": config["framework_version"],
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_files": source_manifest,
            "config_sha256": _sha256(config_path),
            "specification_sha256": _sha256(repo / str(config["specification_path"])),
            "script_sha256": _sha256(Path(__file__)),
            "module_sha256": _sha256(repo / "src/tennis_model/estimation/game_day_elo.py"),
            "test_matches": len(test),
            "scorecard_matches": len(scorecard),
            "paired_anchor_differences": paired,
            "paired_integration_differences": scorecard_paired,
            "production_default_modified": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output)


if __name__ == "__main__":
    main()
