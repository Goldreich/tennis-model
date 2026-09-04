"""Deterministic exact-scoring probabilities used by v1.1 Q tilting."""

from __future__ import annotations

from functools import cache
from math import isfinite, nextafter
from typing import Literal


def _probability(value: float, *, field: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be inside [0, 1]")
    if result == 0.0:
        return nextafter(0.0, 1.0)
    if result == 1.0:
        return nextafter(1.0, 0.0)
    return result


def hold_probability(point_win: float) -> float:
    p = _probability(point_win, field="point_win")
    q = 1.0 - p
    return p**4 * (1.0 + 4.0 * q + 10.0 * q**2) + (
        20.0 * p**3 * q**3 * p**2 / (p**2 + q**2)
    )


def _point_a_probability(
    point_index: int,
    first_server: int,
    p_a_serve: float,
    p_b_serve: float,
) -> float:
    if point_index == 0:
        server = first_server
    else:
        block = (point_index - 1) // 2
        server = 1 - first_server if block % 2 == 0 else first_server
    return p_a_serve if server == 0 else 1.0 - p_b_serve


def tiebreak_probability(
    p_a_serve: float,
    p_b_serve: float,
    *,
    first_server: int,
    target: Literal[7, 10] = 7,
) -> float:
    pa = _probability(p_a_serve, field="p_a_serve")
    pb = _probability(p_b_serve, field="p_b_serve")
    if first_server not in (0, 1):
        raise ValueError("first_server must be 0 or 1")

    @cache
    def win(a: int, b: int) -> float:
        if max(a, b) >= target and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0
        if a == b and a >= target - 1:
            index = a + b
            p0 = _point_a_probability(index, first_server, pa, pb)
            p1 = _point_a_probability(index + 1, first_server, pa, pb)
            win_two_0 = p0 * p1
            split_0 = p0 * (1.0 - p1) + (1.0 - p0) * p1
            p2 = _point_a_probability(index + 2, first_server, pa, pb)
            p3 = _point_a_probability(index + 3, first_server, pa, pb)
            win_two_1 = p2 * p3
            split_1 = p2 * (1.0 - p3) + (1.0 - p2) * p3
            denominator = 1.0 - split_0 * split_1
            if denominator == 0.0:
                return 0.5
            return (win_two_0 + split_0 * win_two_1) / denominator
        index = a + b
        p = _point_a_probability(index, first_server, pa, pb)
        return p * win(a + 1, b) + (1.0 - p) * win(a, b + 1)

    return win(0, 0)


def _set_outcomes(
    p_a_serve: float,
    p_b_serve: float,
    *,
    first_server: int,
    deciding_tiebreak_target: Literal[7, 10],
) -> dict[tuple[int, int], float]:
    hold_a = hold_probability(p_a_serve)
    hold_b = hold_probability(p_b_serve)

    @cache
    def recurse(a_games: int, b_games: int, server: int) -> tuple[tuple[int, int, float], ...]:
        if max(a_games, b_games) >= 6 and abs(a_games - b_games) >= 2:
            return ((0 if a_games > b_games else 1, server, 1.0),)
        if a_games == 6 and b_games == 6:
            p_tb = tiebreak_probability(
                p_a_serve,
                p_b_serve,
                first_server=server,
                target=deciding_tiebreak_target,
            )
            next_server = 1 - server
            return ((0, next_server, p_tb), (1, next_server, 1.0 - p_tb))
        p_a_game = hold_a if server == 0 else 1.0 - hold_b
        combined: dict[tuple[int, int], float] = {}
        for winner, next_server, probability in recurse(a_games + 1, b_games, 1 - server):
            combined[(winner, next_server)] = combined.get((winner, next_server), 0.0) + (
                p_a_game * probability
            )
        for winner, next_server, probability in recurse(a_games, b_games + 1, 1 - server):
            combined[(winner, next_server)] = combined.get((winner, next_server), 0.0) + (
                (1.0 - p_a_game) * probability
            )
        return tuple(
            (winner, server_id, probability)
            for (winner, server_id), probability in combined.items()
        )

    return {
        (winner, server): probability
        for winner, server, probability in recurse(0, 0, first_server)
    }


def exact_match_win_probability(
    p_a_serve: float,
    p_b_serve: float,
    *,
    best_of: Literal[3, 5],
    first_server: int | None = None,
) -> float:
    pa = _probability(p_a_serve, field="p_a_serve")
    pb = _probability(p_b_serve, field="p_b_serve")
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")
    needed = best_of // 2 + 1

    @cache
    def set_outcomes(
        server: int, target: Literal[7, 10]
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (winner, next_server, probability)
            for (winner, next_server), probability in _set_outcomes(
                pa,
                pb,
                first_server=server,
                deciding_tiebreak_target=target,
            ).items()
        )

    @cache
    def match(a_sets: int, b_sets: int, server: int) -> float:
        if a_sets == needed:
            return 1.0
        if b_sets == needed:
            return 0.0
        deciding = a_sets == needed - 1 and b_sets == needed - 1
        target: Literal[7, 10] = 10 if deciding else 7
        result = 0.0
        for winner, next_server, probability in set_outcomes(server, target):
            result += probability * match(
                a_sets + int(winner == 0),
                b_sets + int(winner == 1),
                next_server,
            )
        return result

    if first_server is None:
        return 0.5 * (match(0, 0, 0) + match(0, 0, 1))
    if first_server not in (0, 1):
        raise ValueError("first_server must be 0, 1, or None")
    return match(0, 0, first_server)


__all__ = ["exact_match_win_probability", "hold_probability", "tiebreak_probability"]
