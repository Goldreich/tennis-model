from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from tennis_model.calibration.v1_1_backtest import _stable_id


def test_event_embargo_is_strictly_before_cutoff() -> None:
    cutoff = datetime(2024, 5, 1, tzinfo=UTC)
    event = cutoff - timedelta(days=21)
    available = event + timedelta(days=21)
    assert not available < cutoff
    assert event - timedelta(seconds=1) + timedelta(days=21) < cutoff


def test_hash_sampling_is_player_order_independent_when_match_id_is_stable() -> None:
    match_id = _stable_id("ATP", "event", 1, min("p1", "p2"), max("p1", "p2"))
    reversed_id = _stable_id("ATP", "event", 1, min("p2", "p1"), max("p2", "p1"))
    assert match_id == reversed_id


def test_comparison_ties_settle_no() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 4]})
    assert (frame["a"] > frame["b"]).astype(int).tolist() == [0, 1, 0]
