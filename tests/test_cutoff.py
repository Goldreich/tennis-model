from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest

from tennis_model.data.cutoff import (
    CutoffViolationError,
    InformationCutoff,
    assert_cutoff_safe,
    availability_from_source_date,
    select_before_cutoff,
)


def cutoff_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "past",
                "player_id": "a",
                "available_at_utc": datetime(2026, 8, 27, 23, 59, tzinfo=UTC),
            },
            {
                "match_id": "past",
                "player_id": "b",
                "available_at_utc": datetime(2026, 8, 27, 23, 59, tzinfo=UTC),
            },
            {
                "match_id": "equal",
                "player_id": "c",
                "available_at_utc": datetime(2026, 8, 28, tzinfo=UTC),
            },
            {
                "match_id": "future",
                "player_id": "d",
                "available_at_utc": datetime(2026, 8, 29, tzinfo=UTC),
            },
        ]
    )


def test_cutoff_is_strict_and_preserves_both_directions() -> None:
    cutoff = InformationCutoff(datetime(2026, 8, 28, tzinfo=UTC))
    result = select_before_cutoff(cutoff_rows(), cutoff)
    assert list(result.rows["match_id"]) == ["past", "past"]
    excluded = result.excluded.set_index("match_id")["cutoff_exclusion_code"]
    assert excluded["equal"] == "AT_OR_AFTER_INFORMATION_CUTOFF"
    assert excluded["future"] == "AT_OR_AFTER_INFORMATION_CUTOFF"


def test_unknown_or_inconsistent_availability_excludes_whole_match() -> None:
    rows = pd.DataFrame(
        [
            {"match_id": "unknown", "available_at_utc": pd.NaT},
            {"match_id": "unknown", "available_at_utc": datetime(2020, 1, 1, tzinfo=UTC)},
            {"match_id": "split", "available_at_utc": datetime(2020, 1, 1, tzinfo=UTC)},
            {"match_id": "split", "available_at_utc": datetime(2020, 1, 2, tzinfo=UTC)},
        ]
    )
    result = select_before_cutoff(rows, InformationCutoff(datetime(2021, 1, 1, tzinfo=UTC)))
    assert result.rows.empty
    codes = result.excluded.groupby("match_id")["cutoff_exclusion_code"].first()
    assert codes.to_dict() == {
        "split": "INCONSISTENT_MATCH_AVAILABILITY",
        "unknown": "UNKNOWN_AVAILABILITY",
    }


def test_timezones_are_normalized_to_utc() -> None:
    central = timezone(timedelta(hours=-5))
    cutoff = InformationCutoff(datetime(2026, 8, 27, 19, 0, tzinfo=central))
    assert cutoff.at_utc == datetime(2026, 8, 28, 0, 0, tzinfo=UTC)
    result = select_before_cutoff(cutoff_rows(), cutoff)
    assert set(result.rows["match_id"]) == {"past"}


def test_naive_cutoff_and_naive_row_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        InformationCutoff(datetime(2026, 8, 28))
    rows = pd.DataFrame([{"match_id": "m", "available_at_utc": datetime(2020, 1, 1)}])
    with pytest.raises(ValueError, match="timezone-aware"):
        select_before_cutoff(rows, InformationCutoff(datetime(2021, 1, 1, tzinfo=UTC)))


def test_date_only_availability_requires_an_explicit_lag() -> None:
    assert availability_from_source_date(date(2026, 8, 27), availability_lag_days=1) == datetime(
        2026, 8, 28, tzinfo=UTC
    )
    for invalid_lag in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            availability_from_source_date(date(2026, 8, 27), availability_lag_days=invalid_lag)


def test_cutoff_assertion_catches_future_data() -> None:
    with pytest.raises(CutoffViolationError, match="not before"):
        assert_cutoff_safe(
            cutoff_rows().loc[lambda frame: frame["match_id"] == "equal"],
            InformationCutoff(datetime(2026, 8, 28, tzinfo=UTC)),
        )
