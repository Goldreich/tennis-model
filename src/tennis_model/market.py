"""Cutoff-safe Pinnacle match-winner inputs for Tennis Model v1.3."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from tennis_model.schemas import FrozenModel

PINNACLE_NO_VIG_POLICY_VERSION: Literal[
    "pinnacle-two-way-multiplicative-no-vig/v1"
] = "pinnacle-two-way-multiplicative-no-vig/v1"
PINNACLE_QUOTE_SELECTION_POLICY_VERSION: Literal[
    "latest-cutoff-safe-pinnacle-moneyline/v1"
] = "latest-cutoff-safe-pinnacle-moneyline/v1"
PINNACLE_SUBMISSION_ROUNDING_POLICY_VERSION: Literal[
    "pinnacle-no-vig-nearest-percent-clamp-1-99/v1"
] = (
    "pinnacle-no-vig-nearest-percent-clamp-1-99/v1"
)


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _sha256(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


class _MarketModel(FrozenModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class PinnacleMoneylineQuote(_MarketModel):
    """One observed two-sided Pinnacle match-winner quote."""

    schema_version: Literal["pinnacle-moneyline-quote/v1"] = (
        "pinnacle-moneyline-quote/v1"
    )
    bookmaker: Literal["Pinnacle"] = "Pinnacle"
    market: Literal["match_winner"] = "match_winner"
    official_match_id: str
    player_a_id: str
    player_b_id: str
    player_a_decimal_odds: float = Field(gt=1.0)
    player_b_decimal_odds: float = Field(gt=1.0)
    observed_at_utc: datetime
    source_name: str
    source_url: str
    source_payload_sha256: str

    @field_validator("observed_at_utc")
    @classmethod
    def observed_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="observed_at_utc")

    @field_validator("source_payload_sha256")
    @classmethod
    def payload_digest_is_valid(cls, value: str) -> str:
        return _sha256(value, field="source_payload_sha256")

    @field_validator(
        "official_match_id", "player_a_id", "player_b_id", "source_name", "source_url"
    )
    @classmethod
    def identifiers_are_present(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @model_validator(mode="after")
    def players_are_distinct(self) -> Self:
        if self.player_a_id == self.player_b_id:
            raise ValueError("Pinnacle quote players must be distinct")
        return self


class PinnacleMoneylineSnapshot(_MarketModel):
    """Immutable capture containing one or more timestamped Pinnacle quotes."""

    schema_version: Literal["pinnacle-moneyline-snapshot/v1"] = (
        "pinnacle-moneyline-snapshot/v1"
    )
    captured_at_utc: datetime
    quotes: tuple[PinnacleMoneylineQuote, ...] = Field(min_length=1)

    @field_validator("captured_at_utc")
    @classmethod
    def captured_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="captured_at_utc")

    @model_validator(mode="after")
    def capture_is_coherent(self) -> Self:
        if any(quote.observed_at_utc > self.captured_at_utc for quote in self.quotes):
            raise ValueError("a quote cannot be observed after its snapshot was captured")
        identities = tuple(
            (
                quote.official_match_id,
                tuple(sorted((quote.player_a_id, quote.player_b_id))),
                quote.observed_at_utc,
                quote.source_payload_sha256,
            )
            for quote in self.quotes
        )
        if len(identities) != len(set(identities)):
            raise ValueError("Pinnacle snapshot contains duplicate quote identities")
        return self


class PinnacleMatchWinnerSelection(_MarketModel):
    """The exact cutoff-safe quote selected for one v1.3 prediction lock."""

    schema_version: Literal["pinnacle-match-winner-selection/v1"] = (
        "pinnacle-match-winner-selection/v1"
    )
    bookmaker: Literal["Pinnacle"] = "Pinnacle"
    market: Literal["match_winner"] = "match_winner"
    selection_policy_version: Literal[
        "latest-cutoff-safe-pinnacle-moneyline/v1"
    ] = PINNACLE_QUOTE_SELECTION_POLICY_VERSION
    no_vig_policy_version: Literal[
        "pinnacle-two-way-multiplicative-no-vig/v1"
    ] = PINNACLE_NO_VIG_POLICY_VERSION
    submission_rounding_policy_version: Literal[
        "pinnacle-no-vig-nearest-percent-clamp-1-99/v1"
    ] = PINNACLE_SUBMISSION_ROUNDING_POLICY_VERSION
    snapshot_sha256: str
    snapshot_captured_at_utc: datetime
    official_match_id: str
    player_a_id: str
    player_b_id: str
    player_a_decimal_odds: float = Field(gt=1.0)
    player_b_decimal_odds: float = Field(gt=1.0)
    player_a_raw_implied_probability: float = Field(gt=0.0, lt=1.0)
    player_b_raw_implied_probability: float = Field(gt=0.0, lt=1.0)
    overround: float = Field(gt=0.0)
    player_a_no_vig_probability: float = Field(gt=0.0, lt=1.0)
    player_b_no_vig_probability: float = Field(gt=0.0, lt=1.0)
    observed_at_utc: datetime
    source_name: str
    source_url: str
    source_payload_sha256: str

    @field_validator("snapshot_captured_at_utc", "observed_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @field_validator("snapshot_sha256", "source_payload_sha256")
    @classmethod
    def digests_are_valid(cls, value: str, info: Any) -> str:
        return _sha256(value, field=info.field_name)

    @model_validator(mode="after")
    def probabilities_match_prices(self) -> Self:
        raw_a = 1.0 / self.player_a_decimal_odds
        raw_b = 1.0 / self.player_b_decimal_odds
        overround = raw_a + raw_b
        no_vig_a = raw_a / overround
        no_vig_b = raw_b / overround
        expected = (
            (self.player_a_raw_implied_probability, raw_a),
            (self.player_b_raw_implied_probability, raw_b),
            (self.overround, overround),
            (self.player_a_no_vig_probability, no_vig_a),
            (self.player_b_no_vig_probability, no_vig_b),
        )
        if any(
            not isclose(observed, target, rel_tol=0.0, abs_tol=1e-12)
            for observed, target in expected
        ):
            raise ValueError("stored Pinnacle probabilities do not reproduce from quoted odds")
        if not isclose(
            self.player_a_no_vig_probability + self.player_b_no_vig_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Pinnacle no-vig probabilities must sum to one")
        if self.player_a_id == self.player_b_id:
            raise ValueError("Pinnacle selection players must be distinct")
        if self.observed_at_utc > self.snapshot_captured_at_utc:
            raise ValueError("selected quote was observed after its snapshot capture")
        return self

    def probability_for(self, player_id: str) -> float:
        if player_id == self.player_a_id:
            return self.player_a_no_vig_probability
        if player_id == self.player_b_id:
            return self.player_b_no_vig_probability
        raise ValueError("requested player is absent from the Pinnacle selection")


def load_pinnacle_moneyline_snapshot(
    path: str | Path,
) -> tuple[PinnacleMoneylineSnapshot, str]:
    payload = Path(path).read_bytes()
    return (
        PinnacleMoneylineSnapshot.model_validate_json(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def select_latest_pinnacle_moneyline(
    snapshot: PinnacleMoneylineSnapshot,
    *,
    snapshot_sha256: str,
    official_match_id: str,
    player_a_id: str,
    player_b_id: str,
    information_cutoff_utc: datetime,
    scheduled_start_utc: datetime,
) -> PinnacleMatchWinnerSelection:
    cutoff = _utc(information_cutoff_utc, field="information_cutoff_utc")
    scheduled_start = _utc(scheduled_start_utc, field="scheduled_start_utc")
    if snapshot.captured_at_utc > cutoff:
        raise ValueError("Pinnacle snapshot was captured after the information cutoff")
    participants = {player_a_id, player_b_id}
    candidates = tuple(
        quote
        for quote in snapshot.quotes
        if quote.official_match_id == str(official_match_id)
        and {quote.player_a_id, quote.player_b_id} == participants
        and quote.observed_at_utc <= cutoff
        and quote.observed_at_utc < scheduled_start
    )
    if not candidates:
        raise ValueError("no cutoff-safe two-sided Pinnacle quote exists for the match")
    latest_time = max(quote.observed_at_utc for quote in candidates)
    latest = tuple(quote for quote in candidates if quote.observed_at_utc == latest_time)

    def oriented_prices(quote: PinnacleMoneylineQuote) -> tuple[float, float]:
        if quote.player_a_id == player_a_id:
            return quote.player_a_decimal_odds, quote.player_b_decimal_odds
        return quote.player_b_decimal_odds, quote.player_a_decimal_odds

    price_pairs = {oriented_prices(quote) for quote in latest}
    if len(price_pairs) != 1:
        raise ValueError("latest Pinnacle timestamp contains conflicting two-sided prices")
    selected = min(latest, key=lambda quote: quote.source_payload_sha256)
    decimal_a, decimal_b = oriented_prices(selected)
    raw_a = 1.0 / decimal_a
    raw_b = 1.0 / decimal_b
    overround = raw_a + raw_b
    return PinnacleMatchWinnerSelection(
        snapshot_sha256=_sha256(snapshot_sha256, field="snapshot_sha256"),
        snapshot_captured_at_utc=snapshot.captured_at_utc,
        official_match_id=str(official_match_id),
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        player_a_decimal_odds=decimal_a,
        player_b_decimal_odds=decimal_b,
        player_a_raw_implied_probability=raw_a,
        player_b_raw_implied_probability=raw_b,
        overround=overround,
        player_a_no_vig_probability=raw_a / overround,
        player_b_no_vig_probability=raw_b / overround,
        observed_at_utc=selected.observed_at_utc,
        source_name=selected.source_name,
        source_url=selected.source_url,
        source_payload_sha256=selected.source_payload_sha256,
    )


__all__ = [
    "PINNACLE_NO_VIG_POLICY_VERSION",
    "PINNACLE_QUOTE_SELECTION_POLICY_VERSION",
    "PINNACLE_SUBMISSION_ROUNDING_POLICY_VERSION",
    "PinnacleMatchWinnerSelection",
    "PinnacleMoneylineQuote",
    "PinnacleMoneylineSnapshot",
    "load_pinnacle_moneyline_snapshot",
    "select_latest_pinnacle_moneyline",
]
