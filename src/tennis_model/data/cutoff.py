"""Fail-closed, strict historical information-cutoff access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd


class CutoffViolationError(ValueError):
    """Raised when a supposedly cutoff-safe table contains future data."""


@dataclass(frozen=True, slots=True)
class InformationCutoff:
    """An explicit UTC boundary; rows at the boundary are not yet available."""

    at_utc: datetime

    def __post_init__(self) -> None:
        if self.at_utc.tzinfo is None or self.at_utc.utcoffset() is None:
            raise ValueError("information cutoff must be timezone-aware")
        object.__setattr__(self, "at_utc", self.at_utc.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class CutoffSelection:
    """Rows visible before a cutoff and rows excluded by the guard."""

    rows: pd.DataFrame
    excluded: pd.DataFrame
    cutoff: InformationCutoff


def availability_from_source_date(source_date: date, *, availability_lag_days: int) -> datetime:
    """Return the declared conservative availability upper bound.

    A Sackmann pin must declare what its date column means and the lag used for
    cutoff access.  This helper performs no silent inference about that source
    metadata.
    """

    if availability_lag_days < 1:
        raise ValueError("availability_lag_days must be positive for a date-only source")
    return datetime.combine(source_date, time.min, tzinfo=UTC) + timedelta(
        days=availability_lag_days
    )


def _aware_utc(value: Any) -> datetime | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError(f"availability value is not a datetime: {value!r}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("row availability timestamps must be timezone-aware")
    return value.astimezone(UTC)


def select_before_cutoff(rows: pd.DataFrame, cutoff: InformationCutoff) -> CutoffSelection:
    """Select only whole matches known strictly before ``cutoff``.

    Unknown availability is excluded.  If directions for one match disagree
    about their availability, the whole match is excluded so a cutoff can never
    leave an asymmetric player-service pair.
    """

    required = {"match_id", "available_at_utc"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError("rows missing cutoff columns: " + ", ".join(sorted(missing)))

    working = rows.copy(deep=True)
    normalized_times: list[datetime | None] = []
    for value in working["available_at_utc"]:
        normalized_times.append(_aware_utc(value))
    # Object dtype is intentional: pandas otherwise coerces ``None`` alongside
    # aware datetimes back to ``NaT``, obscuring the fail-closed unknown state.
    working["available_at_utc"] = pd.Series(normalized_times, index=working.index, dtype="object")

    include_indices: list[Any] = []
    exclusion_codes: dict[Any, str] = {}
    for _, group in working.groupby("match_id", sort=False, dropna=False):
        group_times = list(group["available_at_utc"])
        if any(value is None for value in group_times):
            code = "UNKNOWN_AVAILABILITY"
        elif len(set(group_times)) != 1:
            code = "INCONSISTENT_MATCH_AVAILABILITY"
        elif group_times[0] >= cutoff.at_utc:
            code = "AT_OR_AFTER_INFORMATION_CUTOFF"
        else:
            include_indices.extend(group.index)
            continue
        for index in group.index:
            exclusion_codes[index] = code

    selected = working.loc[include_indices].copy()
    selected["information_cutoff_utc"] = cutoff.at_utc
    excluded = working.loc[list(exclusion_codes)].copy()
    if excluded.empty:
        excluded["cutoff_exclusion_code"] = pd.Series(dtype="string")
    else:
        excluded["cutoff_exclusion_code"] = pd.Series(exclusion_codes).loc[excluded.index]
    assert_cutoff_safe(selected, cutoff)
    return CutoffSelection(rows=selected, excluded=excluded, cutoff=cutoff)


def assert_cutoff_safe(rows: pd.DataFrame, cutoff: InformationCutoff) -> None:
    """Fail loudly if any row violates an asserted strict cutoff."""

    if "available_at_utc" not in rows:
        raise CutoffViolationError("cutoff-safe rows lack available_at_utc")
    for value in rows["available_at_utc"]:
        available_at = _aware_utc(value)
        if available_at is None:
            raise CutoffViolationError("cutoff-safe rows contain unknown availability")
        if available_at >= cutoff.at_utc:
            raise CutoffViolationError(
                f"row available at {available_at.isoformat()} is not before "
                f"cutoff {cutoff.at_utc.isoformat()}"
            )
