"""Versioned information policies for historical validation datasets.

The point-in-time policy is the production/live-data rule.  The retrospective
policy is narrower than a general "as of now" data view: it may use a finalized
record only after an exact date proves that the represented match finished
before the historical forecast cutoff.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import pandas as pd
import yaml
from pydantic import model_validator

from tennis_model.data.cutoff import (
    CutoffSelection,
    CutoffViolationError,
    InformationCutoff,
    select_before_cutoff,
)
from tennis_model.schemas import FrozenModel

HISTORICAL_VALIDATION_POLICY_VERSION = "historical-validation-data-policy/v1"


class HistoricalValidationDataMode(StrEnum):
    """Mutually exclusive historical source-availability interpretations."""

    POINT_IN_TIME_VINTAGE = "POINT_IN_TIME_VINTAGE"
    RETROSPECTIVE_FINALIZED = "RETROSPECTIVE_FINALIZED"


class HistoricalValidationPolicy(FrozenModel):
    """Content-addressable declaration of one historical validation data mode."""

    schema_version: str = HISTORICAL_VALIDATION_POLICY_VERSION
    mode: HistoricalValidationDataMode = HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE
    exact_date_crosswalk_sha256: str | None = None
    exact_date_member_crosswalk_ids: tuple[str, ...] = ()
    exact_date_algorithm_version: str | None = None
    exact_date_history_complete: bool | None = None

    @model_validator(mode="after")
    def policy_is_complete(self) -> Self:
        if self.schema_version != HISTORICAL_VALIDATION_POLICY_VERSION:
            raise ValueError("unsupported historical validation policy version")
        fields = (
            self.exact_date_crosswalk_sha256,
            self.exact_date_algorithm_version,
            self.exact_date_history_complete,
        )
        if self.mode is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE:
            if any(value is not None for value in fields):
                raise ValueError("strict vintage mode cannot declare a finalized crosswalk")
            if self.exact_date_member_crosswalk_ids:
                raise ValueError("strict vintage mode cannot declare crosswalk members")
            return self
        if any(value is None for value in fields):
            raise ValueError("retrospective-finalized mode requires exact-date provenance")
        assert self.exact_date_crosswalk_sha256 is not None
        if len(self.exact_date_crosswalk_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.exact_date_crosswalk_sha256
        ):
            raise ValueError("exact-date crosswalk requires a lowercase SHA-256 digest")
        members = self.exact_date_member_crosswalk_ids
        if not members or tuple(sorted(set(members))) != members:
            raise ValueError("crosswalk member IDs must be nonempty, unique, and sorted")
        if any(
            len(member) != 64
            or any(character not in "0123456789abcdef" for character in member)
            for member in members
        ):
            raise ValueError("crosswalk member IDs must be lowercase SHA-256 digests")
        if self.exact_date_crosswalk_sha256 != crosswalk_set_sha256(members):
            raise ValueError("crosswalk-set SHA-256 does not match its members")
        return self


POINT_IN_TIME_VINTAGE_POLICY = HistoricalValidationPolicy()


def crosswalk_set_sha256(member_crosswalk_ids: tuple[str, ...]) -> str:
    """Hash an ordered set of independently content-addressed crosswalks."""

    members = tuple(sorted(set(member_crosswalk_ids)))
    payload = json.dumps(
        {
            "schema_version": "exact-date-crosswalk-set/v1",
            "member_crosswalk_ids": members,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_historical_validation_policy(path: str | Path) -> HistoricalValidationPolicy:
    """Load the policy fields from a pinned assessment configuration."""

    policy_path = Path(path)
    try:
        value = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load historical validation policy {policy_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("historical validation policy root must be a mapping")
    raw_members = value.get("members", ())
    if not isinstance(raw_members, list) or any(
        not isinstance(item, Mapping) for item in raw_members
    ):
        raise ValueError("historical validation policy members must be mappings")
    member_ids = tuple(sorted(str(item.get("crosswalk_id", "")) for item in raw_members))
    coverage = value.get("coverage_gate")
    if not isinstance(coverage, Mapping):
        raise ValueError("historical validation policy requires a coverage gate")
    return HistoricalValidationPolicy(
        schema_version=str(value.get("schema_version", "")),
        mode=HistoricalValidationDataMode(str(value.get("mode", ""))),
        exact_date_crosswalk_sha256=str(value.get("crosswalk_set_sha256", "")),
        exact_date_member_crosswalk_ids=member_ids,
        exact_date_algorithm_version=str(value.get("exact_date_algorithm_version", "")),
        exact_date_history_complete=bool(value.get("exact_date_history_complete")),
    )


def conservative_match_completion_utc(match_date: date) -> datetime:
    """Return the first instant after a date-only match could have occurred.

    Exact augmentation supplies a calendar date, not a start/end timestamp.  A
    row dated on the forecast-cutoff day is therefore excluded for the whole
    day.  This is deliberately conservative and avoids inferring intraday order.
    """

    return datetime.combine(match_date + timedelta(days=1), time.min, tzinfo=UTC)


def _exact_date(value: Any) -> date | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid exact match date: {value!r}") from exc


def select_historical_training_rows(
    rows: pd.DataFrame,
    cutoff: InformationCutoff,
    *,
    policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
) -> CutoffSelection:
    """Select training rows under an explicit, non-interchangeable policy.

    Strict vintage delegates to the original availability selector unchanged.
    Retrospective-finalized ignores publication time only after a unique exact
    match date proves the whole match predates the cutoff.  Missing or
    inconsistent dates fail closed.
    """

    if policy.mode is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE:
        return select_before_cutoff(rows, cutoff)
    required = {"match_id", "match_date"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(
            "retrospective rows missing exact-date columns: " + ", ".join(sorted(missing))
        )
    working = rows.copy(deep=True)
    working["match_date"] = pd.Series(
        [_exact_date(value) for value in working["match_date"]],
        index=working.index,
        dtype="object",
    )
    include: list[Any] = []
    exclusions: dict[Any, str] = {}
    for _, group in working.groupby("match_id", sort=False, dropna=False):
        dates = list(group["match_date"])
        if any(value is None for value in dates):
            code = "MISSING_EXACT_MATCH_DATE"
        elif len(set(dates)) != 1:
            code = "INCONSISTENT_EXACT_MATCH_DATE"
        elif conservative_match_completion_utc(dates[0]) > cutoff.at_utc:
            code = "AT_OR_AFTER_FORECAST_CUTOFF"
        else:
            include.extend(group.index)
            continue
        exclusions.update({index: code for index in group.index})
    selected = working.loc[include].copy()
    selected["information_cutoff_utc"] = cutoff.at_utc
    selected["historical_validation_data_mode"] = policy.mode.value
    selected["historical_validation_policy_version"] = policy.schema_version
    selected["exact_date_crosswalk_sha256"] = policy.exact_date_crosswalk_sha256
    excluded = working.loc[list(exclusions)].copy()
    if excluded.empty:
        excluded["cutoff_exclusion_code"] = pd.Series(dtype="string")
    else:
        excluded["cutoff_exclusion_code"] = pd.Series(exclusions).loc[excluded.index]
    assert_historical_training_safe(selected, cutoff, policy=policy)
    return CutoffSelection(rows=selected, excluded=excluded, cutoff=cutoff)


def assert_historical_training_safe(
    rows: pd.DataFrame,
    cutoff: InformationCutoff,
    *,
    policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY,
) -> None:
    """Verify a selected table under the declared policy."""

    if policy.mode is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE:
        from tennis_model.data.cutoff import assert_cutoff_safe

        assert_cutoff_safe(rows, cutoff)
        return
    if "match_date" not in rows:
        raise CutoffViolationError("retrospective training rows lack exact match_date")
    for value in rows["match_date"]:
        parsed = _exact_date(value)
        if parsed is None:
            raise CutoffViolationError("retrospective training rows contain a missing date")
        if conservative_match_completion_utc(parsed) > cutoff.at_utc:
            raise CutoffViolationError(
                f"match dated {parsed.isoformat()} is not proven complete before "
                f"cutoff {cutoff.at_utc.isoformat()}"
            )


__all__ = [
    "HISTORICAL_VALIDATION_POLICY_VERSION",
    "POINT_IN_TIME_VINTAGE_POLICY",
    "HistoricalValidationDataMode",
    "HistoricalValidationPolicy",
    "assert_historical_training_safe",
    "conservative_match_completion_utc",
    "crosswalk_set_sha256",
    "load_historical_validation_policy",
    "select_historical_training_rows",
]
