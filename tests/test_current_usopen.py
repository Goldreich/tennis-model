from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tennis_model.data.current_usopen import (
    OfficialJsonObject,
    build_official_player_crosswalk,
    normalize_completed_singles,
)
from tennis_model.estimation.inactivity import (
    CompetitionClass,
    InactivityCoverageAssertion,
    InactivityCoverageState,
    InactivityMatchCandidate,
    InactivityTerminalStatus,
    PlayedPointEvidence,
)
from tennis_model.locking.models import InformationBundle, PlayerInactivityInformation
from tennis_model.locking.service import LockCreationError, _resolve_inactivity_records
from tennis_model.schemas import Tour
from tennis_model.simulation.parameters import MatchContext


def _object(source_id: str, value: object, retrieved_at: datetime) -> OfficialJsonObject:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return OfficialJsonObject(
        source_id=source_id,
        locator=f"https://www.usopen.org/fixture/{source_id}.json",
        retrieved_at_utc=retrieved_at,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_official_identity_crosswalk_is_exact_and_fails_ambiguous_names() -> None:
    history = pd.DataFrame(
        [
            {"tour": "WTA", "player_id": "canonical-pegula", "player_name": "Jessica Pegula"},
            {"tour": "WTA", "player_id": "duplicate-a", "player_name": "Same Name"},
            {"tour": "WTA", "player_id": "duplicate-b", "player_name": "Same-Name"},
        ]
    )
    mapping, detail = build_official_player_crosswalk(
        history,
        (
            (Tour.WTA, "wta316956", "Jessica Pegula"),
            (Tour.WTA, "wta-new", "New Player"),
            (Tour.WTA, "wta-ambiguous", "Same Name"),
        ),
    )
    assert mapping[(Tour.WTA, "wta316956")] == "canonical-pegula"
    assert (Tour.WTA, "wta-new") in mapping
    assert (Tour.WTA, "wta-ambiguous") not in mapping
    assert set(detail["status"]) == {
        "MATCHED_UNIQUE_EXACT_NAME",
        "NEW_OFFICIAL_ID",
        "AMBIGUOUS_EXACT_NAME",
    }


def test_completed_official_match_reconstructs_frozen_serve_counts() -> None:
    cutoff = datetime(2026, 8, 30, 12, tzinfo=UTC)
    retrieved = cutoff - timedelta(minutes=10)
    completed_ms = int((cutoff - timedelta(days=3)).timestamp() * 1000)
    day_match = {
        "match_id": "12101",
        "eventCode": "WQ",
        "roundNameShort": "R1",
        "status": "Completed",
        "statusCode": "D",
        "winner": "1",
        "epoch": completed_ms,
        "team1": {
            "idA": "wta-a",
            "firstNameA": "Player",
            "lastNameA": "One",
        },
        "team2": {
            "idA": "wta-b",
            "firstNameA": "Player",
            "lastNameA": "Two",
        },
        "scores": {
            "sets": [
                [{"score": 6, "tiebreak": None}, {"score": 4, "tiebreak": None}],
                [{"score": 6, "tiebreak": None}, {"score": 3, "tiebreak": None}],
            ]
        },
    }
    complete_match = {
        **day_match,
        "duration": "1:20",
        "team1": {**day_match["team1"], "idB": None},
        "team2": {**day_match["team2"], "idB": None},
        "base_stats": {
            "match": {
                "team_1": {
                    "t_f_srv": 60,
                    "t_f_srv_in": 36,
                    "t_f_srv_w": 28,
                    "t_s_srv_w": 12,
                    "df": 3,
                    "t_p_w": 70,
                    "t_w": 20,
                    "t_ue": 15,
                },
                "team_2": {
                    "t_f_srv": 55,
                    "t_f_srv_in": 33,
                    "t_f_srv_w": 20,
                    "t_s_srv_w": 9,
                    "df": 2,
                    "t_p_w": 52,
                    "t_w": 12,
                    "t_ue": 20,
                },
            }
        },
        "serve_stats": {
            "match": {
                "team_1": {"f_srv_ace": 6, "t_gms": 10},
                "team_2": {"f_srv_ace": 2, "t_gms": 9},
            }
        },
    }
    history = pd.DataFrame(
        [
            {"tour": "WTA", "player_id": "p1", "player_name": "Player One"},
            {"tour": "WTA", "player_id": "p2", "player_name": "Player Two"},
        ]
    )
    normalized = normalize_completed_singles(
        (_object("day", {"matches": [day_match]}, retrieved),),
        {"12101": _object("match", {"matches": [complete_match]}, retrieved)},
        historical_identity_rows=history,
        information_cutoff_utc=cutoff,
    )
    assert normalized.completed_match_count == 1
    assert normalized.included_match_count == 1
    assert normalized.exclusions.empty
    player = normalized.counts.counts.loc[
        normalized.counts.counts["player_id"].eq("p1")
    ].set_index("component")
    assert tuple(player.loc["F", ["successes", "trials"]]) == (36, 60)
    assert tuple(player.loc["A", ["successes", "trials"]]) == (6, 36)
    assert tuple(player.loc["Q1", ["successes", "trials"]]) == (22, 30)
    assert tuple(player.loc["D", ["successes", "trials"]]) == (3, 24)
    assert tuple(player.loc["Q2", ["successes", "trials"]]) == (12, 21)


def test_completed_match_at_cutoff_is_excluded() -> None:
    cutoff = datetime(2026, 8, 30, 12, tzinfo=UTC)
    day_match = {
        "match_id": "late",
        "eventCode": "WQ",
        "statusCode": "D",
        "epoch": int(cutoff.timestamp() * 1000),
        "team1": {"idA": "a", "firstNameA": "A", "lastNameA": "One"},
        "team2": {"idA": "b", "firstNameA": "B", "lastNameA": "Two"},
    }
    source = _object("late", {"matches": [day_match]}, cutoff - timedelta(minutes=1))
    normalized = normalize_completed_singles(
        (_object("day", {"matches": [day_match]}, cutoff - timedelta(minutes=1)),),
        {"late": source},
        historical_identity_rows=pd.DataFrame(
            [
                {"tour": "WTA", "player_id": "a", "player_name": "A One"},
                {"tour": "WTA", "player_id": "b", "player_name": "B Two"},
            ]
        ),
        information_cutoff_utc=cutoff,
    )
    assert normalized.rows.empty
    assert normalized.exclusions.iloc[0]["reason"] == "AT_OR_AFTER_CUTOFF"


def test_information_bundle_builds_c6_only_for_the_two_forecast_players() -> None:
    cutoff = datetime(2026, 8, 30, 12, tzinfo=UTC)
    context = MatchContext(
        player_a_id="pegula",
        player_b_id="ruse",
        tour=Tour.WTA,
        event="US Open",
        round="R1",
        scheduled_start_utc=cutoff + timedelta(hours=4),
        scheduled_start_local_date=datetime(2026, 8, 30).date(),
        best_of=3,
        indoor=None,
        information_cutoff_utc=cutoff,
    )

    def player_information(player_id: str, latest_days_ago: int) -> PlayerInactivityInformation:
        return PlayerInactivityInformation(
            player_id=player_id,
            coverage=InactivityCoverageAssertion(
                state=InactivityCoverageState.VERIFIED_COMPLETE,
                source_manifest_id="current-sources/v1",
                source_manifest_sha256="a" * 64,
                canonical_player_id=player_id,
                asserted_at_utc=cutoff - timedelta(minutes=2),
            ),
            candidates=(
                InactivityMatchCandidate(
                    player_id=player_id,
                    identity_resolved=True,
                    tour=Tour.WTA,
                    match_id=f"latest-{player_id}",
                    match_date_local=context.scheduled_start_local_date
                    - timedelta(days=latest_days_ago),
                    discipline="singles",
                    competition_class=CompetitionClass.MAIN_DRAW,
                    terminal_status=InactivityTerminalStatus.NORMAL_COMPLETION,
                    started_evidence=(PlayedPointEvidence.POSITIVE_POINT_STAT_COUNT,),
                    source_manifest_id="current-sources/v1",
                    source_pin=f"official-{player_id}",
                    source_sha256="b" * 64,
                    available_at_utc=cutoff - timedelta(minutes=1),
                ),
            ),
        )

    information = InformationBundle(
        bundle_id="current-match",
        scenario_id="central",
        information_cutoff_utc=cutoff,
        player_inactivity=(player_information("pegula", 7), player_information("ruse", 15)),
    )
    records = _resolve_inactivity_records(information, context, ())
    assert tuple(record.player_id for record in records) == ("pegula", "ruse")
    assert tuple(record.inactivity_days for record in records) == (7, 15)

    incomplete = information.model_copy(
        update={"player_inactivity": (player_information("pegula", 7),)}
    )
    with pytest.raises(LockCreationError, match="exactly the two forecast players"):
        _resolve_inactivity_records(incomplete, context, ())
