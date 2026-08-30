from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from tennis_model.calibration.backtest import HistoricalFilter, freeze_target_cohort
from tennis_model.data.cutoff import InformationCutoff
from tennis_model.data.exact_date_crosswalk import (
    EXACT_DATE_MATCHING_ALGORITHM_VERSION,
    ExactDateJoinStatus,
    ExactDateSourcePin,
    build_exact_date_crosswalk,
)
from tennis_model.data.historical_validation import (
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
    crosswalk_set_sha256,
    load_historical_validation_policy,
    select_historical_training_rows,
)
from tennis_model.schemas import Tour

MEMBER_ID = "1" * 64


def retrospective_policy(*, complete: bool) -> HistoricalValidationPolicy:
    members = (MEMBER_ID,)
    return HistoricalValidationPolicy(
        mode=HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED,
        exact_date_crosswalk_sha256=crosswalk_set_sha256(members),
        exact_date_member_crosswalk_ids=members,
        exact_date_algorithm_version=EXACT_DATE_MATCHING_ALGORITHM_VERSION,
        exact_date_history_complete=complete,
    )


def test_modes_are_explicit_and_noninterchangeable() -> None:
    strict = HistoricalValidationPolicy()
    assert strict.mode is HistoricalValidationDataMode.POINT_IN_TIME_VINTAGE
    with pytest.raises(ValidationError, match="requires exact-date provenance"):
        HistoricalValidationPolicy(mode=HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED)
    with pytest.raises(ValidationError, match="cannot declare a finalized crosswalk"):
        HistoricalValidationPolicy(exact_date_crosswalk_sha256="0" * 64)

    configured = load_historical_validation_policy(
        "config/historical_validation_retrospective_finalized_v1.yaml"
    )
    assert configured.mode is HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED
    assert configured.exact_date_history_complete is False
    assert len(configured.exact_date_member_crosswalk_ids) == 18


def test_retrospective_selector_uses_occurrence_not_later_retrieval_time() -> None:
    cutoff = InformationCutoff(datetime(2024, 1, 3, tzinfo=UTC))
    later = datetime(2026, 8, 30, tzinfo=UTC)
    rows = pd.DataFrame(
        [
            {"match_id": "past", "match_date": date(2024, 1, 1), "available_at_utc": later},
            {"match_id": "past", "match_date": date(2024, 1, 1), "available_at_utc": later},
            {"match_id": "same-day", "match_date": date(2024, 1, 3), "available_at_utc": later},
            {"match_id": "missing", "match_date": None, "available_at_utc": later},
        ]
    )
    strict = select_historical_training_rows(rows, cutoff)
    assert strict.rows.empty
    finalized = select_historical_training_rows(
        rows, cutoff, policy=retrospective_policy(complete=False)
    )
    assert set(finalized.rows["match_id"]) == {"past"}
    assert set(finalized.excluded["cutoff_exclusion_code"]) == {
        "AT_OR_AFTER_FORECAST_CUTOFF",
        "MISSING_EXACT_MATCH_DATE",
    }
    assert set(finalized.rows["historical_validation_data_mode"]) == {
        "RETROSPECTIVE_FINALIZED"
    }


def test_retrospective_selector_rejects_inconsistent_direction_dates() -> None:
    rows = pd.DataFrame(
        [
            {"match_id": "m", "match_date": date(2024, 1, 1)},
            {"match_id": "m", "match_date": date(2024, 1, 2)},
        ]
    )
    selection = select_historical_training_rows(
        rows,
        InformationCutoff(datetime(2024, 2, 1, tzinfo=UTC)),
        policy=retrospective_policy(complete=False),
    )
    assert selection.rows.empty
    assert set(selection.excluded["cutoff_exclusion_code"]) == {
        "INCONSISTENT_EXACT_MATCH_DATE"
    }


def test_exact_date_crosswalk_is_deterministic_conservative_and_market_free() -> None:
    sackmann = pd.DataFrame(
        [
            {
                "tourney_id": "2024-001",
                "tourney_name": "Audit Open",
                "tourney_level": "A",
                "tourney_date": 20240101,
                "match_num": 1,
                "round": "R32",
                "draw_size": 32,
                "surface": "Hard",
                "best_of": 3,
                "score": "6-4 6-4",
                "winner_name": "John Doe",
                "loser_name": "Alex Roe",
            },
            {
                "tourney_id": "2024-D01",
                "tourney_name": "Uncovered Team Event",
                "tourney_level": "D",
                "tourney_date": 20240101,
                "match_num": 2,
                "round": "RR",
                "draw_size": 4,
                "surface": "Hard",
                "best_of": 3,
                "score": "6-0 6-0",
                "winner_name": "Missing Player",
                "loser_name": "Absent Player",
            },
        ]
    )
    tennis_data = pd.DataFrame(
        [
            {
                "Date": "2024-01-02",
                "Tournament": "Audit Open",
                "Winner": "Doe J.",
                "Loser": "Roe A.",
                "Surface": "Hard",
                "Round": "1st Round",
                "Best of": 3,
                "W1": 6,
                "L1": 4,
                "W2": 6,
                "L2": 4,
                "B365W": 1.25,
                "B365L": 4.0,
            }
        ]
    )
    source = ExactDateSourcePin(
        source_id="tennis-data-atp-2024",
        tour=Tour.ATP,
        year=2024,
        locator="https://example.invalid/2024.xlsx",
        sha256="2" * 64,
        size_bytes=123,
        retrieved_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    first = build_exact_date_crosswalk(
        sackmann,
        tennis_data,
        sackmann_source_id="sackmann-atp-2024",
        sackmann_source_sha256="3" * 64,
        augmentation_source=source,
    )
    second = build_exact_date_crosswalk(
        sackmann,
        tennis_data,
        sackmann_source_id="sackmann-atp-2024",
        sackmann_source_sha256="3" * 64,
        augmentation_source=source,
    )
    assert first.manifest == second.manifest
    assert first.detail["status"].tolist() == [
        ExactDateJoinStatus.MATCHED.value,
        ExactDateJoinStatus.UNMATCHED.value,
    ]
    assert first.detail.loc[0, "exact_match_date"] == "2024-01-02"
    assert first.manifest.complete_for_b6_c6_history is False
    assert not any("B365" in column for column in first.detail.columns)


def test_backtest_gate_rejects_incomplete_retrospective_history_before_reveal() -> None:
    with pytest.raises(ValueError, match="complete exact-date history"):
        freeze_target_cohort(
            (),
            historical_filter=HistoricalFilter(),
            historical_source_index_sha256="4" * 64,
            historical_validation_policy=retrospective_policy(complete=False),
        )
