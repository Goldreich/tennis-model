from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tennis_model.locking.models import PredictionSnapshot, PropEstimateRecord, SerializedProp
from tennis_model.market import (
    PinnacleMoneylineQuote,
    PinnacleMoneylineSnapshot,
    select_latest_pinnacle_moneyline,
)

_PAYLOAD_A = "a" * 64
_PAYLOAD_B = "b" * 64
_SNAPSHOT = "c" * 64


def _quote(
    *,
    observed_hour: int,
    player_a_id: str = "left",
    player_b_id: str = "right",
    player_a_odds: float = 2.4,
    player_b_odds: float = 1.6,
    payload_sha256: str = _PAYLOAD_A,
) -> PinnacleMoneylineQuote:
    return PinnacleMoneylineQuote(
        official_match_id="1234",
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        player_a_decimal_odds=player_a_odds,
        player_b_decimal_odds=player_b_odds,
        observed_at_utc=datetime(2026, 9, 4, observed_hour, tzinfo=UTC),
        source_name="retained-test-source",
        source_url="https://example.test/match/1234",
        source_payload_sha256=payload_sha256,
    )


def test_latest_quote_is_reoriented_and_normalized_without_vig() -> None:
    snapshot = PinnacleMoneylineSnapshot(
        captured_at_utc=datetime(2026, 9, 4, 14, tzinfo=UTC),
        quotes=(
            _quote(observed_hour=12),
            _quote(
                observed_hour=13,
                player_a_id="right",
                player_b_id="left",
                player_a_odds=1.5,
                player_b_odds=2.75,
                payload_sha256=_PAYLOAD_B,
            ),
        ),
    )

    selected = select_latest_pinnacle_moneyline(
        snapshot,
        snapshot_sha256=_SNAPSHOT,
        official_match_id="1234",
        player_a_id="left",
        player_b_id="right",
        information_cutoff_utc=datetime(2026, 9, 4, 14, tzinfo=UTC),
        scheduled_start_utc=datetime(2026, 9, 4, 16, tzinfo=UTC),
    )

    expected_left = (1.0 / 2.75) / ((1.0 / 2.75) + (1.0 / 1.5))
    assert selected.observed_at_utc == datetime(2026, 9, 4, 13, tzinfo=UTC)
    assert selected.player_a_decimal_odds == 2.75
    assert selected.player_b_decimal_odds == 1.5
    assert selected.player_a_no_vig_probability == pytest.approx(expected_left)
    assert selected.player_b_no_vig_probability == pytest.approx(1.0 - expected_left)

    prop = SerializedProp.model_construct(
        prop_id="winner-prop",
        kind="MATCH_WIN",
        subject_ids=("left",),
        original_text="MATCH_WIN(left)",
    )
    estimate = PropEstimateRecord.model_construct(
        prop_id="winner-prop",
        prop=prop,
        probability_raw=0.61,
        probability_settled=0.61,
    )
    lock = PredictionSnapshot.model_construct(
        prop_estimates=(estimate,),
        market_match_winner=selected,
    )
    assert lock.effective_prop_probability("winner-prop") == pytest.approx(expected_left)
    assert lock.effective_prop_submission_integer("winner-prop") == 35
    assert lock.effective_prop_source("winner-prop") == selected.no_vig_policy_version


def test_snapshot_captured_after_cutoff_is_rejected() -> None:
    snapshot = PinnacleMoneylineSnapshot(
        captured_at_utc=datetime(2026, 9, 4, 15, tzinfo=UTC),
        quotes=(_quote(observed_hour=13),),
    )
    with pytest.raises(ValueError, match="captured after the information cutoff"):
        select_latest_pinnacle_moneyline(
            snapshot,
            snapshot_sha256=_SNAPSHOT,
            official_match_id="1234",
            player_a_id="left",
            player_b_id="right",
            information_cutoff_utc=datetime(2026, 9, 4, 14, tzinfo=UTC),
            scheduled_start_utc=datetime(2026, 9, 4, 16, tzinfo=UTC),
        )


def test_conflicting_latest_prices_are_rejected() -> None:
    snapshot = PinnacleMoneylineSnapshot(
        captured_at_utc=datetime(2026, 9, 4, 14, tzinfo=UTC),
        quotes=(
            _quote(observed_hour=13),
            _quote(
                observed_hour=13,
                player_a_odds=2.5,
                player_b_odds=1.55,
                payload_sha256=_PAYLOAD_B,
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicting two-sided prices"):
        select_latest_pinnacle_moneyline(
            snapshot,
            snapshot_sha256=_SNAPSHOT,
            official_match_id="1234",
            player_a_id="left",
            player_b_id="right",
            information_cutoff_utc=datetime(2026, 9, 4, 14, tzinfo=UTC),
            scheduled_start_utc=datetime(2026, 9, 4, 16, tzinfo=UTC),
        )
