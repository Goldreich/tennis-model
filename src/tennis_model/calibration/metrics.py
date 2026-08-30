"""Transparent Brier, reliability, subgroup, and Monte Carlo diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log
from statistics import median

import numpy as np

from tennis_model.calibration.ledger import CalibrationLedgerRow


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    mean_probability: float | None
    empirical_frequency: float | None
    mean_brier: float | None


@dataclass(frozen=True, slots=True)
class GroupCalibration:
    group: str
    rows: int
    mean_probability: float | None
    empirical_frequency: float | None
    mean_brier: float | None


@dataclass(frozen=True, slots=True)
class MonteCarloAudit:
    count: int
    median_standard_error: float | None
    p95_standard_error: float | None
    fraction_above_tolerance: float | None
    tolerance: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    total_rows: int
    settled_rows: int
    void_rows: int
    unavailable_rows: int
    unresolved_rows: int
    mean_brier: float | None
    mean_log_loss: float | None
    reliability: tuple[ReliabilityBin, ...]
    by_prop_family: tuple[GroupCalibration, ...]
    by_tour: tuple[GroupCalibration, ...]
    by_confidence: tuple[GroupCalibration, ...]
    retirement_rate_by_tour: tuple[GroupCalibration, ...]
    mc_audit: MonteCarloAudit


def brier_score(probability: float, outcome: int) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if outcome not in {0, 1}:
        raise ValueError("outcome must be zero or one")
    return (probability - outcome) ** 2


def _settled(rows: tuple[CalibrationLedgerRow, ...]) -> tuple[CalibrationLedgerRow, ...]:
    return tuple(row for row in rows if row.resolution_status in {"yes", "no"})


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def reliability_table(rows: tuple[CalibrationLedgerRow, ...]) -> tuple[ReliabilityBin, ...]:
    settled = _settled(rows)
    bins: list[list[CalibrationLedgerRow]] = [[] for _ in range(10)]
    for row in settled:
        if row.probability_raw is None:
            raise ValueError("settled calibration row lacks raw probability")
        bins[min(int(row.probability_raw * 10.0), 9)].append(row)
    result = []
    for index, members in enumerate(bins):
        probabilities = [row.probability_raw for row in members if row.probability_raw is not None]
        outcomes = [float(row.outcome_binary) for row in members if row.outcome_binary is not None]
        briers = [row.brier_raw_model for row in members if row.brier_raw_model is not None]
        result.append(
            ReliabilityBin(
                lower=index / 10.0,
                upper=(index + 1) / 10.0,
                upper_inclusive=index == 9,
                count=len(members),
                mean_probability=_mean(probabilities),
                empirical_frequency=_mean(outcomes),
                mean_brier=_mean(briers),
            )
        )
    return tuple(result)


def confidence_band(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance >= 0.4:
        return "High"
    if distance >= 0.25:
        return "Medium"
    return "Low"


def _groups(
    rows: tuple[CalibrationLedgerRow, ...],
    key: Callable[[CalibrationLedgerRow], object],
) -> tuple[GroupCalibration, ...]:
    grouped: dict[str, list[CalibrationLedgerRow]] = {}
    for row in _settled(rows):
        grouped.setdefault(str(key(row)), []).append(row)
    return tuple(
        GroupCalibration(
            group=group,
            rows=len(members),
            mean_probability=_mean(
                [row.probability_raw for row in members if row.probability_raw is not None]
            ),
            empirical_frequency=_mean(
                [float(row.outcome_binary) for row in members if row.outcome_binary is not None]
            ),
            mean_brier=_mean(
                [row.brier_raw_model for row in members if row.brier_raw_model is not None]
            ),
        )
        for group, members in sorted(grouped.items())
    )


def summarize_calibration(
    rows: tuple[CalibrationLedgerRow, ...],
    *,
    mc_tolerance: float = 0.0025,
) -> CalibrationReport:
    settled = _settled(rows)
    briers = [row.brier_raw_model for row in settled if row.brier_raw_model is not None]
    losses = []
    for row in settled:
        if row.outcome_binary is None:
            continue
        if row.probability_raw is None:
            raise ValueError("settled calibration row lacks raw probability")
        p = min(1.0 - 1e-15, max(1e-15, row.probability_raw))
        losses.append(-(row.outcome_binary * log(p) + (1 - row.outcome_binary) * log(1 - p)))
    errors = [row.mc_standard_error for row in rows]
    if errors:
        ordered = sorted(errors)
        p95 = float(np.quantile(np.asarray(ordered), 0.95, method="linear"))
        audit = MonteCarloAudit(
            count=len(errors),
            median_standard_error=median(errors),
            p95_standard_error=p95,
            fraction_above_tolerance=sum(value > mc_tolerance for value in errors) / len(errors),
            tolerance=mc_tolerance,
        )
    else:
        audit = MonteCarloAudit(0, None, None, None, mc_tolerance)

    match_rows: dict[tuple[str, str], CalibrationLedgerRow] = {}
    for row in rows:
        match_rows.setdefault((row.lock_id, row.tour.value), row)
    retirement_grouped: dict[str, list[float]] = {}
    for (_lock_id, tour), row in match_rows.items():
        retirement_grouped.setdefault(tour, []).append(float(row.match_retired))
    retirement = tuple(
        GroupCalibration(
            group=tour,
            rows=len(values),
            mean_probability=None,
            empirical_frequency=_mean(values),
            mean_brier=None,
        )
        for tour, values in sorted(retirement_grouped.items())
    )
    return CalibrationReport(
        total_rows=len(rows),
        settled_rows=len(settled),
        void_rows=sum(row.resolution_status == "void" for row in rows),
        unavailable_rows=sum(row.resolution_status == "unavailable" for row in rows),
        unresolved_rows=sum(row.resolution_status == "unresolved" for row in rows),
        mean_brier=_mean(briers),
        mean_log_loss=_mean(losses),
        reliability=reliability_table(rows),
        by_prop_family=_groups(rows, lambda row: row.prop_family),
        by_tour=_groups(rows, lambda row: row.tour.value),
        by_confidence=_groups(
            rows,
            lambda row: confidence_band(
                row.probability_raw if row.probability_raw is not None else 0.5
            ),
        ),
        retirement_rate_by_tour=retirement,
        mc_audit=audit,
    )
