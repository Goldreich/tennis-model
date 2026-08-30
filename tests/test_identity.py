from __future__ import annotations

import pytest

from tennis_model.identity import (
    IdentityAnomalyCode,
    IdentityValidationError,
    MatchDuplicateKind,
    MatchRecordIdentity,
    canonical_match_id,
    canonical_player_id,
    detect_match_duplicate_groups,
    make_match_identity,
    make_match_record_identity,
    make_player_identity,
)
from tennis_model.schemas import Tour


def test_player_id_is_deterministic_and_names_are_aliases_only() -> None:
    first = make_player_identity(
        "Sackmann",
        Tour.ATP,
        104925,
        display_name="Novak Djokovic",
    )
    renamed = make_player_identity(
        " sackmann ",
        "ATP",
        "104925",
        display_name="N. Djokovic",
    )

    assert first.player_id == renamed.player_id
    assert first.display_aliases == ("Novak Djokovic",)
    assert renamed.display_aliases == ("N. Djokovic",)
    assert canonical_player_id("sackmann", Tour.ATP, 104925) == first.player_id


def test_player_namespaces_prevent_cross_source_and_cross_tour_collisions() -> None:
    atp = canonical_player_id("sackmann", Tour.ATP, "123")

    assert atp != canonical_player_id("official-usopen", Tour.ATP, "123")
    assert atp != canonical_player_id("sackmann", Tour.WTA, "123")
    assert atp != canonical_player_id("sackmann", Tour.ATP, "124")


@pytest.mark.parametrize(
    ("source", "tour", "external_id", "expected"),
    [
        ("", Tour.ATP, "1", IdentityAnomalyCode.MISSING_SOURCE_NAMESPACE),
        ("source", "", "1", IdentityAnomalyCode.MISSING_TOUR),
        ("source", "ITF", "1", IdentityAnomalyCode.INVALID_TOUR),
        ("source", Tour.ATP, "  ", IdentityAnomalyCode.MISSING_PLAYER_EXTERNAL_ID),
    ],
)
def test_player_identity_rejects_missing_or_invalid_key_fields(
    source: object,
    tour: Tour | str,
    external_id: object,
    expected: IdentityAnomalyCode,
) -> None:
    with pytest.raises(IdentityValidationError) as caught:
        canonical_player_id(source, tour, external_id)

    assert caught.value.code is expected


@pytest.mark.parametrize(
    ("source", "tour", "tourney_id", "match_num", "expected"),
    [
        (None, Tour.ATP, "2026-USO", 1, IdentityAnomalyCode.MISSING_SOURCE_NAMESPACE),
        ("source", None, "2026-USO", 1, IdentityAnomalyCode.MISSING_TOUR),
        ("source", Tour.ATP, "", 1, IdentityAnomalyCode.MISSING_TOURNEY_ID),
        ("source", Tour.ATP, "2026-USO", None, IdentityAnomalyCode.MISSING_MATCH_NUM),
    ],
)
def test_match_id_rejects_missing_key_fields(
    source: object,
    tour: Tour | str | None,
    tourney_id: object,
    match_num: object,
    expected: IdentityAnomalyCode,
) -> None:
    with pytest.raises(IdentityValidationError) as caught:
        canonical_match_id(source, tour, tourney_id, match_num)  # type: ignore[arg-type]

    assert caught.value.code is expected


def test_match_id_uses_only_source_tour_tourney_and_match_number() -> None:
    player_a = canonical_player_id("sackmann", Tour.ATP, "p1")
    player_b = canonical_player_id("sackmann", Tour.ATP, "p2")
    first = make_match_identity(
        "sackmann",
        Tour.ATP,
        "2026-USO",
        17,
        player_a_id=player_a,
        player_b_id=player_b,
    )
    reversed_players = make_match_identity(
        "SACKMANN",
        "ATP",
        "2026-USO",
        "17",
        player_a_id=player_b,
        player_b_id=player_a,
    )

    assert first.match_id == reversed_players.match_id
    assert first.match_id == canonical_match_id("sackmann", Tour.ATP, "2026-USO", 17)
    assert first.participant_ids == reversed_players.participant_ids


def test_match_identity_rejects_missing_participant_and_self_match() -> None:
    player = canonical_player_id("sackmann", Tour.WTA, "p1")

    with pytest.raises(IdentityValidationError) as missing:
        make_match_identity(
            "sackmann",
            Tour.WTA,
            "2026-USO",
            3,
            player_a_id=player,
            player_b_id="",
        )
    assert missing.value.code is IdentityAnomalyCode.MISSING_PLAYER_ID

    with pytest.raises(IdentityValidationError) as self_match:
        make_match_identity(
            "sackmann",
            Tour.WTA,
            "2026-USO",
            3,
            player_a_id=player,
            player_b_id=player,
        )
    assert self_match.value.code is IdentityAnomalyCode.SELF_MATCH


def _record(
    *,
    match_num: int,
    player_a: str,
    player_b: str,
    row: str,
    fingerprint: str,
) -> MatchRecordIdentity:
    match = make_match_identity(
        "sackmann",
        Tour.ATP,
        "2026-USO",
        match_num,
        player_a_id=player_a,
        player_b_id=player_b,
    )
    return make_match_record_identity(
        match,
        source_row_ref=row,
        record_fingerprint=fingerprint,
    )


def test_exact_duplicate_group_is_reported_without_dropping_records() -> None:
    player_a = canonical_player_id("sackmann", Tour.ATP, "p1")
    player_b = canonical_player_id("sackmann", Tour.ATP, "p2")
    records = [
        _record(
            match_num=7,
            player_a=player_a,
            player_b=player_b,
            row="file.csv:2",
            fingerprint="sha256:identical",
        ),
        _record(
            match_num=7,
            player_a=player_a,
            player_b=player_b,
            row="file.csv:9",
            fingerprint="sha256:identical",
        ),
    ]

    groups = detect_match_duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0].kind is MatchDuplicateKind.EXACT_DUPLICATE
    assert groups[0].anomaly_code is IdentityAnomalyCode.DUPLICATE_MATCH
    assert {record.source_row_ref for record in groups[0].records} == {
        "file.csv:2",
        "file.csv:9",
    }


def test_conflicting_correction_group_preserves_every_candidate() -> None:
    player_a = canonical_player_id("sackmann", Tour.ATP, "p1")
    player_b = canonical_player_id("sackmann", Tour.ATP, "p2")
    records = [
        _record(
            match_num=8,
            player_a=player_a,
            player_b=player_b,
            row="old.csv:2",
            fingerprint="sha256:old",
        ),
        _record(
            match_num=8,
            player_a=player_a,
            player_b=player_b,
            row="new.csv:2",
            fingerprint="sha256:corrected",
        ),
    ]

    groups = detect_match_duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0].kind is MatchDuplicateKind.CONFLICTING_RECORDS
    assert groups[0].anomaly_code is IdentityAnomalyCode.CONFLICTING_MATCH_RECORDS
    assert len(groups[0].records) == 2


def test_match_key_collision_is_reported_for_different_participants() -> None:
    player_a = canonical_player_id("sackmann", Tour.ATP, "p1")
    player_b = canonical_player_id("sackmann", Tour.ATP, "p2")
    player_c = canonical_player_id("sackmann", Tour.ATP, "p3")
    records = [
        _record(
            match_num=9,
            player_a=player_a,
            player_b=player_b,
            row="file.csv:2",
            fingerprint="sha256:first",
        ),
        _record(
            match_num=9,
            player_a=player_a,
            player_b=player_c,
            row="file.csv:3",
            fingerprint="sha256:second",
        ),
    ]

    groups = detect_match_duplicate_groups(records)

    assert len(groups) == 1
    assert groups[0].kind is MatchDuplicateKind.MATCH_KEY_COLLISION
    assert groups[0].anomaly_code is IdentityAnomalyCode.MATCH_KEY_COLLISION
    assert len(groups[0].records) == 2


def test_unique_match_records_do_not_form_duplicate_groups() -> None:
    player_a = canonical_player_id("sackmann", Tour.WTA, "p1")
    player_b = canonical_player_id("sackmann", Tour.WTA, "p2")
    records = [
        make_match_record_identity(
            make_match_identity(
                "sackmann",
                Tour.WTA,
                "2026-USO",
                match_num,
                player_a_id=player_a,
                player_b_id=player_b,
            ),
            source_row_ref=f"file.csv:{match_num}",
            record_fingerprint=f"sha256:{match_num}",
        )
        for match_num in (1, 2)
    ]

    assert detect_match_duplicate_groups(records) == ()
