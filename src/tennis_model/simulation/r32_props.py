"""Canonical ten-question R32 prop bundle constructors."""

from __future__ import annotations

from tennis_model.simulation.match import (
    ACE_COMPARE,
    ANY_LOPSIDED_SET,
    DECIDING_SET,
    DF_COMPARE,
    DURATION_MIN,
    FIRST_SERVE_WIN_PCT,
    MATCH_WIN,
    TIEBREAK_COUNT,
    UNFORCED_ERRORS,
    WINNER_COMPARE,
    ComparisonOperator,
    PropSpec,
)


def R32_PROP_BUNDLE(
    *,
    duration_minutes: float,
    ace_subject: str,
    ace_opponent: str,
    double_fault_subject: str,
    double_fault_opponent: str,
    unforced_error_subject: str,
    unforced_error_threshold: int,
    first_serve_subject: str,
    first_serve_threshold_pct: float,
    winner_subject: str,
    winner_opponent: str,
    match_win_subject: str,
    tiebreak_threshold: int,
) -> tuple[PropSpec, ...]:
    """Return the platform's ten props in display order."""

    return (
        DURATION_MIN(ComparisonOperator.MORE_THAN, duration_minutes),
        ACE_COMPARE(ace_subject, ace_opponent),
        DF_COMPARE(double_fault_subject, double_fault_opponent),
        UNFORCED_ERRORS(
            unforced_error_subject,
            ComparisonOperator.MORE_THAN,
            unforced_error_threshold,
        ),
        FIRST_SERVE_WIN_PCT(
            first_serve_subject,
            ComparisonOperator.MORE_THAN,
            first_serve_threshold_pct,
        ),
        WINNER_COMPARE(winner_subject, winner_opponent),
        DECIDING_SET(),
        ANY_LOPSIDED_SET(),
        MATCH_WIN(match_win_subject),
        TIEBREAK_COUNT(ComparisonOperator.AT_LEAST, tiebreak_threshold),
    )


__all__ = ["R32_PROP_BUNDLE"]
