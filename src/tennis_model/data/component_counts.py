"""Construction and validation of the five frozen serve-component counts.

The transformation in this module is deliberately integer-only.  Rates are a
modeling concern for a later milestone; Milestone 1 records successes, trials,
eligibility, and any component-local anomaly without clipping or imputation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any

import pandas as pd

TRANSFORMATION_VERSION = "serve-component-counts-v1.1"
SERVE_COMPONENTS = ("F", "A", "Q1", "D", "Q2")


class ComponentStatus(StrEnum):
    """Whether a count row may contribute to a component likelihood."""

    VALID = "valid"
    MISSING_INPUT = "missing_input"
    ZERO_DENOMINATOR = "zero_denominator"
    QUARANTINED = "quarantined"


_STAT_FIELDS = (
    "service_points",
    "first_serves_in",
    "first_serve_points_won",
    "second_serve_points_won",
    "aces",
    "double_faults",
)

_FIELD_CODE_NAMES = {
    "service_points": "SERVICE_POINTS",
    "first_serves_in": "FIRST_SERVES_IN",
    "first_serve_points_won": "FIRST_SERVE_POINTS_WON",
    "second_serve_points_won": "SECOND_SERVE_POINTS_WON",
    "aces": "ACES",
    "double_faults": "DOUBLE_FAULTS",
}

_COMPONENT_INPUTS: Mapping[str, tuple[str, ...]] = {
    "F": ("first_serves_in", "service_points"),
    "A": ("aces", "first_serves_in"),
    "Q1": ("first_serve_points_won", "aces", "first_serves_in"),
    "D": ("double_faults", "service_points", "first_serves_in"),
    "Q2": (
        "second_serve_points_won",
        "service_points",
        "first_serves_in",
        "double_faults",
    ),
}

_LINEAGE_COLUMNS = (
    "snapshot_id",
    "snapshot_sha256",
    "source_id",
    "source_url",
    "source_schema_version",
    "retrieved_at_utc",
    "normalization_version",
    "source_row_number",
    "orientation",
    "match_id",
    "source_date",
    "match_date",
    "match_date_source_id",
    "match_date_source_sha256",
    "match_date_crosswalk_id",
    "event_start_date",
    "source_date_semantics",
    "available_at_utc",
    "tour",
    "event",
    "event_year",
    "level",
    "round",
    "surface",
    "indoor",
    "best_of",
    "player_id",
    "opponent_id",
    "player_hand",
    "opponent_hand",
    "raw_record_json",
)


@dataclass(frozen=True, slots=True)
class ComponentCountTable:
    """Long-form counts and a separate append-only anomaly table."""

    counts: pd.DataFrame
    anomalies: pd.DataFrame
    transformation_version: str = TRANSFORMATION_VERSION

    def likelihood_rows(self) -> pd.DataFrame:
        """Return a defensive copy of rows that contribute to likelihoods."""

        return self.counts.loc[self.counts["eligible_for_likelihood"]].copy()


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _as_optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid count")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        converted = int(float(value))
        if float(value) == converted:
            return converted
    raise TypeError(f"count is not an exact integer: {value!r}")


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _deduplicate(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _computed_counts(component: str, values: Mapping[str, int]) -> tuple[int, int]:
    if component == "F":
        return values["first_serves_in"], values["service_points"]
    if component == "A":
        return values["aces"], values["first_serves_in"]
    if component == "Q1":
        return (
            values["first_serve_points_won"] - values["aces"],
            values["first_serves_in"] - values["aces"],
        )
    if component == "D":
        return (
            values["double_faults"],
            values["service_points"] - values["first_serves_in"],
        )
    if component == "Q2":
        return (
            values["second_serve_points_won"],
            values["service_points"] - values["first_serves_in"] - values["double_faults"],
        )
    raise ValueError(f"unknown serve component: {component}")


def _component_anomalies(
    component: str, values: Mapping[str, int], successes: int, trials: int
) -> tuple[str, ...]:
    codes: list[str] = []
    for field in _COMPONENT_INPUTS[component]:
        if values[field] < 0:
            codes.append(f"NEGATIVE_{_FIELD_CODE_NAMES[field]}")

    if component == "F":
        if values["first_serves_in"] > values["service_points"]:
            codes.append("FIRST_SERVES_IN_GT_SERVICE_POINTS")
    elif component == "A":
        if values["aces"] > values["first_serves_in"]:
            codes.append("ACES_GT_FIRST_SERVES_IN")
    elif component == "Q1":
        if values["aces"] > values["first_serves_in"]:
            codes.append("ACES_GT_FIRST_SERVES_IN")
        if values["aces"] > values["first_serve_points_won"]:
            codes.append("ACES_GT_FIRST_SERVE_POINTS_WON")
        if values["first_serve_points_won"] > values["first_serves_in"]:
            codes.append("FIRST_SERVE_POINTS_WON_GT_FIRST_SERVES_IN")
    elif component == "D":
        second_opportunities = values["service_points"] - values["first_serves_in"]
        if values["first_serves_in"] > values["service_points"]:
            codes.append("FIRST_SERVES_IN_GT_SERVICE_POINTS")
        if values["double_faults"] > second_opportunities:
            codes.append("DOUBLE_FAULTS_GT_SECOND_SERVE_OPPORTUNITIES")
    elif component == "Q2":
        second_opportunities = values["service_points"] - values["first_serves_in"]
        playable_second_serves = second_opportunities - values["double_faults"]
        if values["first_serves_in"] > values["service_points"]:
            codes.append("FIRST_SERVES_IN_GT_SERVICE_POINTS")
        if values["double_faults"] > second_opportunities:
            codes.append("DOUBLE_FAULTS_GT_SECOND_SERVE_OPPORTUNITIES")
        if values["second_serve_points_won"] > playable_second_serves:
            codes.append("SECOND_SERVE_POINTS_WON_GT_PLAYABLE_SECOND_SERVES")

    if trials < 0:
        codes.append("NEGATIVE_TRIALS")
    if successes < 0:
        codes.append("NEGATIVE_SUCCESSES")
    if successes > trials:
        codes.append("SUCCESSES_GT_TRIALS")
    return _deduplicate(codes)


def _lineage(row: pd.Series) -> dict[str, Any]:
    return {column: row.get(column, pd.NA) for column in _LINEAGE_COLUMNS}


def build_serve_component_counts(rows: pd.DataFrame) -> ComponentCountTable:
    """Build F/A/Q1/D/Q2 counts from normalized player-service rows.

    Missing operands make only dependent components missing.  Invalid integer
    tokens and impossible count identities quarantine only dependent
    components.  A legitimate ``(successes, trials) == (0, 0)`` is retained as
    ``zero_denominator`` and supplies no likelihood.
    """

    missing_columns = set(_STAT_FIELDS).difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"normalized rows missing required columns: {missing}")

    count_records: list[dict[str, Any]] = []
    anomaly_records: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        invalid_fields = set(_coerce_string_tuple(row.get("invalid_stat_fields")))
        parsed_values: dict[str, int | None] = {}
        parse_failures: set[str] = set(invalid_fields)
        for field in _STAT_FIELDS:
            try:
                parsed_values[field] = _as_optional_int(row[field])
            except TypeError:
                parsed_values[field] = None
                parse_failures.add(field)

        for component in SERVE_COMPONENTS:
            inputs = _COMPONENT_INPUTS[component]
            base = _lineage(row)
            base.update(
                {
                    "transformation_version": TRANSFORMATION_VERSION,
                    "component": component,
                }
            )
            malformed = [field for field in inputs if field in parse_failures]
            if malformed:
                codes = tuple(f"MALFORMED_{_FIELD_CODE_NAMES[field]}" for field in malformed)
                count_records.append(
                    {
                        **base,
                        "successes": pd.NA,
                        "trials": pd.NA,
                        "status": ComponentStatus.QUARANTINED.value,
                        "eligible_for_likelihood": False,
                        "anomaly_codes": codes,
                    }
                )
                for code in codes:
                    anomaly_records.append(_anomaly_record(row, component, code, parsed_values))
                continue

            if any(parsed_values[field] is None for field in inputs):
                count_records.append(
                    {
                        **base,
                        "successes": pd.NA,
                        "trials": pd.NA,
                        "status": ComponentStatus.MISSING_INPUT.value,
                        "eligible_for_likelihood": False,
                        "anomaly_codes": (),
                    }
                )
                continue

            complete_values: dict[str, int] = {}
            for field in inputs:
                parsed_value = parsed_values[field]
                if parsed_value is None:  # Guarded by the missing branch above.
                    raise AssertionError("complete component input became missing")
                complete_values[field] = parsed_value
            successes, trials = _computed_counts(component, complete_values)
            codes = _component_anomalies(component, complete_values, successes, trials)
            if codes:
                status = ComponentStatus.QUARANTINED
            elif trials == 0:
                status = ComponentStatus.ZERO_DENOMINATOR
            else:
                status = ComponentStatus.VALID

            count_records.append(
                {
                    **base,
                    "successes": successes,
                    "trials": trials,
                    "status": status.value,
                    "eligible_for_likelihood": status is ComponentStatus.VALID,
                    "anomaly_codes": codes,
                }
            )
            for code in codes:
                anomaly_records.append(_anomaly_record(row, component, code, complete_values))

    counts = pd.DataFrame.from_records(count_records)
    if counts.empty:
        counts = pd.DataFrame(
            columns=(
                *_LINEAGE_COLUMNS,
                "transformation_version",
                "component",
                "successes",
                "trials",
                "status",
                "eligible_for_likelihood",
                "anomaly_codes",
            )
        )
    counts["successes"] = counts["successes"].astype("Int64")
    counts["trials"] = counts["trials"].astype("Int64")
    counts["eligible_for_likelihood"] = counts["eligible_for_likelihood"].astype("boolean")

    anomalies = pd.DataFrame.from_records(anomaly_records)
    if anomalies.empty:
        anomalies = pd.DataFrame(
            columns=(
                *_LINEAGE_COLUMNS,
                "scope",
                "component",
                "code",
                "raw_values_json",
            )
        )
    return ComponentCountTable(counts=counts, anomalies=anomalies)


def _anomaly_record(
    row: pd.Series,
    component: str,
    code: str,
    values: Mapping[str, int | None],
) -> dict[str, Any]:
    raw_values = {field: values.get(field) for field in _COMPONENT_INPUTS[component]}
    return {
        **_lineage(row),
        "scope": "component",
        "component": component,
        "code": code,
        "raw_values_json": json.dumps(raw_values, sort_keys=True, separators=(",", ":")),
    }
