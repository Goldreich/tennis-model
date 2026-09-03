"""Cutoff-safe main-tour workload inputs for production game-day Elo."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from tennis_model.estimation.game_day_elo import FitnessFeatures


_INVALID_SCORE = re.compile(
    r"(?:W[./]?O|WALKOVER|\bRET\b|\bDEF\b|\bABD\b|\bBYE\b)", re.I
)
_SLAM_OFFSETS = {
    "Q1": -6,
    "Q2": -4,
    "Q3": -2,
    "R128": 0,
    "R64": 2,
    "R32": 4,
    "R16": 6,
    "QF": 8,
    "SF": 11,
    "F": 13,
}
_STANDARD_OFFSETS = {
    "Q1": -3,
    "Q2": -2,
    "Q3": -1,
    "R128": 0,
    "R64": 1,
    "R32": 2,
    "R16": 3,
    "QF": 4,
    "SF": 5,
    "F": 6,
}


@dataclass(frozen=True, slots=True)
class FitnessExposure:
    match_key: str
    played_at_utc: datetime
    minutes: float | None


@dataclass(frozen=True, slots=True)
class FitnessHistorySnapshot:
    by_player: Mapping[tuple[str, str], tuple[FitnessExposure, ...]]
    source_manifest: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class GameDayFitnessAssessment:
    player_name: str
    available: bool
    reason: str | None
    features: FitnessFeatures | None
    last_match_date: date | None
    recent_match_count: int


def _normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.strip())
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _played_at(row: Mapping[str, str]) -> datetime | None:
    digits = re.sub(r"\D", "", str(row.get("tourney_date", "")))
    if len(digits) < 8:
        return None
    try:
        event_start = datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    level = str(row.get("tourney_level", "")).strip().upper()
    round_name = str(row.get("round", "")).strip().upper()
    offsets = _SLAM_OFFSETS if level == "G" else _STANDARD_OFFSETS
    return event_start + timedelta(days=offsets.get(round_name, 0))


def load_main_tour_fitness_history(
    source_directory: str | Path,
    *,
    source_years: Sequence[int],
    tours: Sequence[str],
    levels: Mapping[str, Sequence[str]],
) -> FitnessHistorySnapshot:
    source = Path(source_directory)
    matches: dict[str, tuple[str, str, FitnessExposure]] = {}
    manifest: list[dict[str, Any]] = []
    for tour in tours:
        prefix = tour.lower()
        names = [f"{prefix}_{year}.csv" for year in source_years]
        names.append(f"{prefix}_ongoing.csv")
        for name in names:
            path = source / name
            if not path.is_file():
                continue
            raw = path.read_bytes()
            manifest.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )
            rows = csv.DictReader(raw.decode("utf-8-sig", errors="replace").splitlines())
            for row in rows:
                winner = str(row.get("winner_name", "")).strip()
                loser = str(row.get("loser_name", "")).strip()
                score = str(row.get("score", "")).strip()
                level = str(row.get("tourney_level", "")).strip()
                if (
                    not winner
                    or not loser
                    or not score
                    or _INVALID_SCORE.search(score)
                    or level not in set(levels[tour])
                ):
                    continue
                played_at = _played_at(row)
                if played_at is None:
                    continue
                minutes = _number(str(row.get("minutes", "")))
                if minutes is not None and minutes <= 0.0:
                    minutes = None
                match_identity = "|".join(
                    (
                        tour,
                        str(row.get("tourney_id", "")),
                        str(row.get("tourney_date", "")),
                        str(row.get("match_num", "")),
                        _normalized_name(winner),
                        _normalized_name(loser),
                    )
                )
                match_key = hashlib.sha256(match_identity.encode("utf-8")).hexdigest()[:24]
                matches[match_key] = (
                    _normalized_name(winner),
                    _normalized_name(loser),
                    FitnessExposure(match_key, played_at, minutes),
                )
    by_player: dict[tuple[str, str], list[FitnessExposure]] = {}
    for match_key, (winner, loser, exposure) in matches.items():
        tour = match_key and next(
            value.split("|", 1)[0]
            for value in (
                "|".join((candidate_tour,))
                for candidate_tour in tours
                if (candidate_tour, winner) in by_player or candidate_tour in match_key
            )
        ) if False else None
        del tour
        # The tour is recoverable only from the identity before hashing; use the
        # separately retained key below when rebuilding the dictionary.
    by_player = {}
    for tour in tours:
        prefix = f"{tour}|"
        for identity_key, (winner, loser, exposure) in (
            item for item in matches.items()
        ):
            del prefix, identity_key
            # Exposure keys are globally unique, but the source tour is retained
            # in a second pass from player names below.
            break
        break
    # Re-read no source bytes: assign tour while constructing a keyed match map.
    keyed_matches: dict[tuple[str, str], list[FitnessExposure]] = {}
    for tour in tours:
        for name in [f"{tour.lower()}_{year}.csv" for year in source_years] + [f"{tour.lower()}_ongoing.csv"]:
            path = source / name
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                for row in csv.DictReader(stream):
                    winner = _normalized_name(str(row.get("winner_name", "")))
                    loser = _normalized_name(str(row.get("loser_name", "")))
                    identity = "|".join(
                        (
                            tour,
                            str(row.get("tourney_id", "")),
                            str(row.get("tourney_date", "")),
                            str(row.get("match_num", "")),
                            winner,
                            loser,
                        )
                    )
                    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                    found = matches.get(key)
                    if found is None:
                        continue
                    exposure = found[2]
                    for player in (winner, loser):
                        values = keyed_matches.setdefault((tour, player), [])
                        if not any(item.match_key == exposure.match_key for item in values):
                            values.append(exposure)
    frozen = {
        key: tuple(sorted(values, key=lambda item: (item.played_at_utc, item.match_key)))
        for key, values in keyed_matches.items()
    }
    return FitnessHistorySnapshot(frozen, tuple(manifest))


def assess_game_day_fitness(
    history: FitnessHistorySnapshot,
    *,
    tour: str,
    player_name: str,
    scheduled_local_date: date,
    information_cutoff_utc: datetime,
    config: Mapping[str, Any],
) -> GameDayFitnessAssessment:
    target = datetime.combine(scheduled_local_date, datetime.min.time(), tzinfo=UTC)
    eligible = tuple(
        item
        for item in history.by_player.get((tour.upper(), _normalized_name(player_name)), ())
        if item.played_at_utc < target and item.played_at_utc < information_cutoff_utc
    )
    if not eligible:
        return GameDayFitnessAssessment(
            player_name, True, None, FitnessFeatures(0.0, 0.0, 0.0), None, 0
        )
    lookback = int(config["workload_lookback_days"])
    recent = tuple(
        item for item in eligible if 0 <= (target - item.played_at_utc).days <= lookback
    )
    if any(item.minutes is None for item in recent):
        return GameDayFitnessAssessment(
            player_name,
            False,
            "MISSING_RECORDED_MINUTES_IN_21_DAY_WINDOW",
            None,
            eligible[-1].played_at_utc.date(),
            len(recent),
        )
    short_weight = float(config["workload_short_weight"])
    workload = 0.0
    for item in recent:
        days = (target - item.played_at_utc).total_seconds() / 86400.0
        decay = short_weight * 2.0 ** (
            -days / float(config["workload_short_half_life_days"])
        )
        decay += (1.0 - short_weight) * 2.0 ** (
            -days / float(config["workload_long_half_life_days"])
        )
        assert item.minutes is not None
        workload += item.minutes * decay / float(config["workload_unit_minutes"])

    previous = eligible[-1].played_at_utc
    gap = max(0.0, (target - previous).total_seconds() / 86400.0)
    normal = float(config["normal_recovery_days"])
    short_recovery = max(0.0, normal - gap) / normal
    threshold = float(config["layoff_threshold_days"])
    severity = 0.0
    matches_since_return = 0
    prior: datetime | None = None
    for item in eligible:
        if prior is not None:
            historical_gap = (item.played_at_utc - prior).total_seconds() / 86400.0
            if historical_gap > threshold:
                severity = math.log1p(
                    (historical_gap - threshold) / float(config["layoff_unit_days"])
                )
                matches_since_return = 1
            elif severity > 0.0:
                matches_since_return += 1
                if matches_since_return >= 12:
                    severity = 0.0
                    matches_since_return = 0
        prior = item.played_at_utc
    if gap > threshold:
        return_from_layoff = math.log1p(
            (gap - threshold) / float(config["layoff_unit_days"])
        )
    elif severity > 0.0:
        return_from_layoff = severity * math.exp(
            -matches_since_return / float(config["return_decay_matches"])
        )
    else:
        return_from_layoff = 0.0
    return GameDayFitnessAssessment(
        player_name,
        True,
        None,
        FitnessFeatures(workload, short_recovery, return_from_layoff),
        previous.date(),
        len(recent),
    )


__all__ = [
    "FitnessHistorySnapshot",
    "GameDayFitnessAssessment",
    "assess_game_day_fitness",
    "load_main_tour_fitness_history",
]
