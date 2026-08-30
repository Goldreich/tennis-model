from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from tennis_model.data.artifacts import (
    ProcessedArtifactIntegrityError,
    load_processed_bundle,
    read_processed_table,
    verify_processed_bundle,
    write_processed_bundle,
)
from tennis_model.data.cutoff import InformationCutoff
from tennis_model.data.exact_date_crosswalk import (
    EXACT_DATE_MATCHING_ALGORITHM_VERSION,
    ExactDateSourcePin,
    build_exact_date_crosswalk,
)
from tennis_model.data.historical_validation import (
    HistoricalValidationDataMode,
    HistoricalValidationPolicy,
    crosswalk_set_sha256,
)
from tennis_model.data.ingest_sackmann import (
    SACKMANN_SCHEMA_VERSION,
    CoverageValidationError,
    SackmannIngestionError,
    ingest_sackmann_snapshot,
    read_sackmann_snapshot,
)
from tennis_model.data.normalize import normalize_player_service_rows
from tennis_model.data.snapshot import materialize_snapshot, read_snapshot_bytes
from tennis_model.estimation.config import load_serve_model_config
from tennis_model.estimation.serve_components import (
    FutureMatchContext,
    ServeComponent,
    fit_all_serve_components_from_bundle,
    fit_all_serve_components_from_bundles,
    fit_input_set_sha256,
    predict_serve_performance,
)
from tennis_model.schemas import (
    CoverageRange,
    PinnedSource,
    RowDateSemantics,
    Tour,
)

HEADERS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "winner_hand",
    "loser_id",
    "loser_name",
    "loser_hand",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


def valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "tourney_id": "2026-TEST",
        "tourney_name": "Test Open",
        "surface": "Hard",
        "tourney_level": "A",
        "tourney_date": "20260801",
        "match_num": "1",
        "winner_id": "1001",
        "winner_name": "Winner One",
        "winner_hand": "R",
        "loser_id": "1002",
        "loser_name": "Loser Two",
        "loser_hand": "L",
        "score": "6-4 6-4",
        "best_of": "3",
        "round": "F",
        "minutes": "120",
        "w_ace": "10",
        "w_df": "5",
        "w_svpt": "100",
        "w_1stIn": "60",
        "w_1stWon": "42",
        "w_2ndWon": "20",
        "w_SvGms": "10",
        "w_bpSaved": "3",
        "w_bpFaced": "5",
        "l_ace": "4",
        "l_df": "6",
        "l_svpt": "90",
        "l_1stIn": "50",
        "l_1stWon": "30",
        "l_2ndWon": "15",
        "l_SvGms": "10",
        "l_bpSaved": "2",
        "l_bpFaced": "6",
    }
    row.update(overrides)
    return row


def csv_payload(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def pinned_source(
    payload: bytes,
    *,
    tour: Tour = Tour.ATP,
    date_semantics: RowDateSemantics = RowDateSemantics.MATCH_DATE,
    availability_lag_days: int = 1,
) -> PinnedSource:
    return PinnedSource(
        source_id=f"fixture-{tour.value.lower()}-2026",
        identity_namespace="fixture-sackmann",
        tour=tour,
        upstream_attribution="Synthetic Jeff Sackmann-style test fixture",
        locator=f"https://archive.example/{tour.value.lower()}-2026.csv",
        archive_identifier=f"fixture:{tour.value}:2026",
        object_identifier="fixture-object-v1",
        sha256=hashlib.sha256(payload).hexdigest(),
        schema_version="sackmann-match-csv/v1",
        stated_license="CC BY-NC-SA 4.0 (fixture metadata)",
        retrieved_at_utc=datetime(2026, 8, 28, 12, tzinfo=UTC),
        verified_coverage=CoverageRange(
            first_match_date=date(2026, 1, 1),
            last_match_date=date(2026, 8, 27),
            verified_at_utc=datetime(2026, 8, 28, 13, tzinfo=UTC),
        ),
        row_date_semantics=date_semantics,
        availability_lag_days=availability_lag_days,
    )


def snapshot_for(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    tour: Tour = Tour.ATP,
    date_semantics: RowDateSemantics = RowDateSemantics.MATCH_DATE,
    availability_lag_days: int = 1,
):
    payload = csv_payload(rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_file = tmp_path / f"{tour.value.lower()}-matches.csv"
    source_file.write_bytes(payload)
    snapshot = materialize_snapshot(
        pinned_source(
            payload,
            tour=tour,
            date_semantics=date_semantics,
            availability_lag_days=availability_lag_days,
        ),
        source_file,
        tmp_path / "raw",
    )
    return snapshot, payload


def snapshot_from_payload(tmp_path: Path, payload: bytes, *, source: PinnedSource | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_file = tmp_path / "source.csv"
    source_file.write_bytes(payload)
    snapshot = materialize_snapshot(
        source or pinned_source(payload),
        source_file,
        tmp_path / "raw",
    )
    return snapshot


def test_declared_sackmann_schema_version_is_enforced(tmp_path: Path) -> None:
    payload = csv_payload([valid_row()])
    value = pinned_source(payload).model_dump(mode="python")
    value["schema_version"] = "unknown-schema/v9"
    snapshot = snapshot_from_payload(
        tmp_path,
        payload,
        source=PinnedSource.model_validate(value),
    )

    with pytest.raises(SackmannIngestionError, match="unsupported Sackmann schema version"):
        read_sackmann_snapshot(snapshot)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"tourney_id,winner_id\n2026-TEST,1001\n", "missing required columns"),
        (
            (",".join([*HEADERS, "winner_id"]) + "\n").encode(),
            "duplicate CSV columns",
        ),
    ],
)
def test_structural_csv_schema_failures_are_rejected(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    snapshot = snapshot_from_payload(tmp_path, payload)

    with pytest.raises(SackmannIngestionError, match=message):
        read_sackmann_snapshot(snapshot)


def test_supported_schema_version_constant_matches_fixture_contract() -> None:
    assert SACKMANN_SCHEMA_VERSION == "sackmann-match-csv/v1"


def test_winner_loser_normalization_is_symmetric_and_not_cross_wired(
    tmp_path: Path,
) -> None:
    snapshot, payload = snapshot_for(tmp_path, [valid_row()])
    raw = read_sackmann_snapshot(snapshot)
    normalized = normalize_player_service_rows(raw, snapshot=snapshot)

    assert len(normalized.rows) == 2
    assert normalized.accepted_match_count == 1
    winner = normalized.rows.loc[normalized.rows["orientation"] == "winner"].iloc[0]
    loser = normalized.rows.loc[normalized.rows["orientation"] == "loser"].iloc[0]
    assert winner["match_id"] == loser["match_id"]
    assert (winner["player_id"], winner["opponent_id"]) == (
        loser["opponent_id"],
        loser["player_id"],
    )
    assert (winner["service_points"], winner["aces"], winner["double_faults"]) == (
        100,
        10,
        5,
    )
    assert (loser["service_points"], loser["aces"], loser["double_faults"]) == (
        90,
        4,
        6,
    )
    assert winner["player_source_id"] == "1001"
    assert loser["player_source_id"] == "1002"
    assert json.loads(winner["raw_record_json"])["winner_name"] == "Winner One"
    assert read_snapshot_bytes(snapshot) == payload


def test_tiebreak_service_game_accounting_accepts_official_game_convention(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [
            valid_row(
                score="6-4 3-6 7-6(5)",
                w_SvGms="16",
                l_SvGms="16",
            )
        ],
    )
    normalized = normalize_player_service_rows(
        read_sackmann_snapshot(snapshot), snapshot=snapshot
    )

    assert normalized.accepted_match_count == 1
    assert "SERVICE_GAMES_SCORE_MISMATCH" not in set(normalized.anomalies["code"])


def test_service_game_mismatch_quarantines_only_that_optional_field(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [valid_row(w_SvGms="5", l_SvGms="5")],
    )
    normalized = normalize_player_service_rows(
        read_sackmann_snapshot(snapshot), snapshot=snapshot
    )

    assert normalized.accepted_match_count == 1
    assert len(normalized.rows) == 2
    assert all("service_games" in fields for fields in normalized.rows["invalid_stat_fields"])
    mismatches = normalized.anomalies.loc[
        normalized.anomalies["code"] == "SERVICE_GAMES_SCORE_MISMATCH"
    ]
    assert len(mismatches) == 1


def test_tour_is_explicit_and_names_do_not_define_identity(tmp_path: Path) -> None:
    atp_snapshot, _ = snapshot_for(tmp_path / "atp", [valid_row()])
    wta_snapshot, _ = snapshot_for(
        tmp_path / "wta",
        [valid_row(winner_name="A Different Display Name")],
        tour=Tour.WTA,
    )
    atp = normalize_player_service_rows(
        read_sackmann_snapshot(atp_snapshot), snapshot=atp_snapshot
    ).rows
    wta = normalize_player_service_rows(
        read_sackmann_snapshot(wta_snapshot), snapshot=wta_snapshot
    ).rows
    assert set(atp["tour"]) == {"ATP"}
    assert set(wta["tour"]) == {"WTA"}
    assert atp.iloc[0]["player_id"] != wta.iloc[0]["player_id"]


def test_tournament_start_date_is_never_mislabeled_as_match_date(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [valid_row()],
        date_semantics=RowDateSemantics.TOURNAMENT_START_DATE,
        availability_lag_days=21,
    )
    rows = normalize_player_service_rows(read_sackmann_snapshot(snapshot), snapshot=snapshot).rows
    assert rows["match_date"].isna().all()
    assert rows["event_start_date"].tolist() == [
        date(2026, 8, 1),
        date(2026, 8, 1),
    ]
    assert rows["available_at_utc"].iloc[0] == datetime(2026, 8, 22, tzinfo=UTC)


def test_exact_duplicate_is_preserved_raw_but_cannot_double_count(
    tmp_path: Path,
) -> None:
    row = valid_row()
    snapshot, payload = snapshot_for(tmp_path, [row, row.copy()])
    normalized = normalize_player_service_rows(read_sackmann_snapshot(snapshot), snapshot=snapshot)
    assert normalized.raw_row_count == 2
    assert len(normalized.rows) == 2
    assert normalized.accepted_match_count == 1
    assert "duplicate_match" in set(normalized.anomalies["code"])
    assert read_snapshot_bytes(snapshot) == payload


def test_conflicting_duplicate_corrections_are_quarantined_together(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(), valid_row(w_ace="11")])
    normalized = normalize_player_service_rows(read_sackmann_snapshot(snapshot), snapshot=snapshot)
    assert normalized.rows.empty
    conflicts = normalized.anomalies.loc[
        normalized.anomalies["code"] == "conflicting_match_records"
    ]
    assert set(conflicts["source_row_number"]) == {2, 3}


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"winner_id": ""}, "missing_player_external_id"),
        ({"loser_id": "1001"}, "self_match"),
        ({"score": "6-6 6-4"}, "SCORE_ILLEGAL_SET"),
    ],
)
def test_match_level_failures_quarantine_both_directions(
    tmp_path: Path, overrides: dict[str, str], expected_code: str
) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(**overrides)])
    normalized = normalize_player_service_rows(read_sackmann_snapshot(snapshot), snapshot=snapshot)
    assert normalized.rows.empty
    assert expected_code in set(normalized.anomalies["code"])


def test_walkover_is_audited_and_emits_no_service_rows(tmp_path: Path) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [
            valid_row(
                score="W/O",
                w_ace="",
                w_df="",
                w_svpt="",
                w_1stIn="",
                w_1stWon="",
                w_2ndWon="",
                w_SvGms="",
                w_bpSaved="",
                w_bpFaced="",
                l_ace="",
                l_df="",
                l_svpt="",
                l_1stIn="",
                l_1stWon="",
                l_2ndWon="",
                l_SvGms="",
                l_bpSaved="",
                l_bpFaced="",
            )
        ],
    )
    normalized = normalize_player_service_rows(read_sackmann_snapshot(snapshot), snapshot=snapshot)
    assert normalized.rows.empty
    assert "WALKOVER_EXCLUDED" in set(normalized.anomalies["code"])


def test_missing_primitive_input_is_not_converted_to_zero(tmp_path: Path) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(w_ace="")])
    result = ingest_sackmann_snapshot(
        snapshot,
        cutoff=InformationCutoff(datetime(2026, 8, 10, tzinfo=UTC)),
    )
    winner = result.counts.loc[result.counts["orientation"].eq("winner")].set_index("component")
    assert winner.loc["A", "status"] == "missing_input"
    assert winner.loc["Q1", "status"] == "missing_input"
    assert pd.isna(winner.loc["A", "successes"])
    assert winner.loc["F", "status"] == "valid"
    assert winner.loc["D", "status"] == "valid"
    assert winner.loc["Q2", "status"] == "valid"


def test_retrospective_finalized_ingestion_uses_exact_dates_and_preserves_late_provenance(
    tmp_path: Path,
) -> None:
    rows = [
        valid_row(tourney_date="20260801", match_num="1"),
        valid_row(
            tourney_date="20260801",
            match_num="2",
            winner_id="2001",
            winner_name="Future Three",
            loser_id="2002",
            loser_name="Future Four",
            w_ace="future-secret-malformed-value",
        ),
    ]
    payload = csv_payload(rows)
    late = datetime(2026, 8, 28, 12, tzinfo=UTC)
    source = pinned_source(
        payload, date_semantics=RowDateSemantics.TOURNAMENT_START_DATE
    ).model_copy(
        update={
            "retrieved_at_utc": late,
            "source_effective_at_utc": late,
            "source_available_at_utc": late,
        }
    )
    snapshot = snapshot_from_payload(tmp_path, payload, source=source)
    tennis_data = pd.DataFrame(
        [
            {
                "Date": "2026-08-02",
                "Tournament": "Test Open",
                "Winner": "One W.",
                "Loser": "Two L.",
                "Surface": "Hard",
                "Round": "The Final",
                "Best of": 3,
                "W1": 6,
                "L1": 4,
                "W2": 6,
                "L2": 4,
            },
            {
                "Date": "2026-08-04",
                "Tournament": "Test Open",
                "Winner": "Three F.",
                "Loser": "Four F.",
                "Surface": "Hard",
                "Round": "The Final",
                "Best of": 3,
                "W1": 6,
                "L1": 4,
                "W2": 6,
                "L2": 4,
            },
        ]
    )
    augmentation = ExactDateSourcePin(
        source_id="tennis-data-atp-2026",
        tour=Tour.ATP,
        year=2026,
        locator="https://example.invalid/2026.xlsx",
        sha256="9" * 64,
        size_bytes=100,
        retrieved_at_utc=late,
    )
    crosswalk = build_exact_date_crosswalk(
        pd.DataFrame(rows),
        tennis_data,
        sackmann_source_id=source.source_id,
        sackmann_source_sha256=source.sha256,
        augmentation_source=augmentation,
    )
    assert crosswalk.manifest.complete_for_b6_c6_history
    members = (crosswalk.manifest.crosswalk_id,)
    policy = HistoricalValidationPolicy(
        mode=HistoricalValidationDataMode.RETROSPECTIVE_FINALIZED,
        exact_date_crosswalk_sha256=crosswalk_set_sha256(members),
        exact_date_member_crosswalk_ids=members,
        exact_date_algorithm_version=EXACT_DATE_MATCHING_ALGORITHM_VERSION,
        exact_date_history_complete=True,
    )
    cutoff = InformationCutoff(datetime(2026, 8, 3, tzinfo=UTC))
    strict = ingest_sackmann_snapshot(snapshot, cutoff=cutoff)
    assert strict.service_rows.empty

    finalized = ingest_sackmann_snapshot(
        snapshot,
        cutoff=cutoff,
        historical_validation_policy=policy,
        exact_date_crosswalk=crosswalk,
    )
    assert finalized.selected_raw_row_count == 1
    assert set(finalized.service_rows["match_date"]) == {date(2026, 8, 2)}
    assert (finalized.service_rows["available_at_utc"] == late).all()
    assert "MALFORMED_ACES" not in set(finalized.anomalies["code"])
    assert set(finalized.cutoff_exclusions["source_row_number"]) == {3}
    bundle = write_processed_bundle(finalized, tmp_path / "processed-finalized")
    assert bundle.manifest.historical_validation_policy == policy
    assert bundle.manifest.exact_date_crosswalk_manifest == crosswalk.manifest


def test_malformed_primitive_is_preserved_and_component_locally_quarantined(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(w_ace="10.5")])
    result = ingest_sackmann_snapshot(
        snapshot,
        cutoff=InformationCutoff(datetime(2026, 8, 10, tzinfo=UTC)),
    )

    service_row = result.service_rows.loc[result.service_rows["orientation"].eq("winner")].iloc[0]
    assert pd.isna(service_row["aces"])
    assert service_row["invalid_stat_fields"] == ("aces",)
    assert json.loads(service_row["raw_record_json"])["w_ace"] == "10.5"
    winner = result.counts.loc[result.counts["orientation"].eq("winner")].set_index("component")
    assert winner.loc["A", "status"] == "quarantined"
    assert winner.loc["Q1", "status"] == "quarantined"
    assert winner.loc["F", "status"] == "valid"
    assert winner.loc["D", "status"] == "valid"
    assert winner.loc["Q2", "status"] == "valid"
    assert "MALFORMED_ACES" in set(result.anomalies["code"])


def test_zero_denominators_contribute_no_likelihood(tmp_path: Path) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [
            valid_row(
                w_ace="0",
                w_df="0",
                w_svpt="0",
                w_1stIn="0",
                w_1stWon="0",
                w_2ndWon="0",
            )
        ],
    )
    result = ingest_sackmann_snapshot(
        snapshot,
        cutoff=InformationCutoff(datetime(2026, 8, 10, tzinfo=UTC)),
    )
    winner = result.counts.loc[result.counts["orientation"].eq("winner")]
    assert set(winner["status"]) == {"zero_denominator"}
    assert not winner["eligible_for_likelihood"].any()


def test_cutoff_excludes_equal_and_future_rows_before_stat_parsing(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(
        tmp_path,
        [
            valid_row(tourney_date="20260801", match_num="1"),
            valid_row(
                tourney_date="20260802",
                match_num="2",
                winner_id="1003",
                loser_id="1004",
                w_ace="future-secret-malformed-value",
            ),
        ],
    )
    cutoff = InformationCutoff(datetime(2026, 8, 3, tzinfo=UTC))
    result = ingest_sackmann_snapshot(snapshot, cutoff=cutoff)
    assert result.raw_row_count == 2
    assert result.selected_raw_row_count == 1
    assert result.service_rows["match_date"].tolist() == [
        date(2026, 8, 1),
        date(2026, 8, 1),
    ]
    assert (result.counts["available_at_utc"] < cutoff.at_utc).all()
    assert "MALFORMED_ACES" not in set(result.anomalies["code"])
    assert set(result.cutoff_exclusions["source_row_number"]) == {3}
    assert set(result.cutoff_exclusions["cutoff_exclusion_code"]) == {
        "AT_OR_AFTER_INFORMATION_CUTOFF"
    }


def test_observed_dates_must_fit_the_verified_tour_coverage(tmp_path: Path) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(tourney_date="20261231")])
    with pytest.raises(CoverageValidationError, match="outside declared ATP coverage"):
        ingest_sackmann_snapshot(
            snapshot,
            cutoff=InformationCutoff(datetime(2027, 1, 10, tzinfo=UTC)),
        )


def test_processed_parquet_bundle_round_trips_and_is_idempotent(
    tmp_path: Path,
) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row(w_ace="61")])
    result = ingest_sackmann_snapshot(
        snapshot,
        cutoff=InformationCutoff(datetime(2026, 8, 10, tzinfo=UTC)),
    )
    first = write_processed_bundle(result, tmp_path / "processed")
    second = write_processed_bundle(result, tmp_path / "processed")
    loaded = load_processed_bundle(first.directory)

    assert first.bundle_id == second.bundle_id == loaded.bundle_id
    assert first.directory == second.directory
    counts = read_processed_table(loaded, "component_counts")
    assert str(counts["successes"].dtype) == "Int64"
    q1 = counts.loc[counts["orientation"].eq("winner") & counts["component"].eq("Q1")].iloc[0]
    assert q1["anomaly_codes"]
    assert "ACES_GT_FIRST_SERVES_IN" in q1["anomaly_codes"]
    service_rows = read_processed_table(loaded, "service_rows")
    assert service_rows.loc[0, "invalid_stat_fields"] == ()
    assert loaded.manifest.source.verified_coverage == snapshot.source.verified_coverage
    verify_processed_bundle(loaded)


def test_processed_parquet_tampering_fails_loudly(tmp_path: Path) -> None:
    snapshot, _ = snapshot_for(tmp_path, [valid_row()])
    result = ingest_sackmann_snapshot(
        snapshot,
        cutoff=InformationCutoff(datetime(2026, 8, 10, tzinfo=UTC)),
    )
    bundle = write_processed_bundle(result, tmp_path / "processed")
    with bundle.table_path("component_counts").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ProcessedArtifactIntegrityError, match="checksum/size"):
        verify_processed_bundle(bundle)


def test_verified_processed_bundle_binds_all_five_fit_provenance(tmp_path: Path) -> None:
    rows = [
        valid_row(
            match_num=str(index + 1),
            w_ace=str(7 + index % 4),
            w_df=str(3 + index % 3),
            w_1stIn=str(57 + index % 6),
            w_1stWon=str(38 + index % 5),
            w_2ndWon=str(18 + index % 4),
            l_ace=str(3 + index % 3),
            l_df=str(4 + index % 2),
            l_1stIn=str(48 + index % 5),
            l_1stWon=str(27 + index % 4),
            l_2ndWon=str(13 + index % 4),
        )
        for index in range(10)
    ]
    snapshot, _ = snapshot_for(tmp_path, rows)
    cutoff = datetime(2026, 8, 10, tzinfo=UTC)
    result = ingest_sackmann_snapshot(snapshot, cutoff=InformationCutoff(cutoff))
    bundle = write_processed_bundle(result, tmp_path / "processed")
    fits = fit_all_serve_components_from_bundle(
        bundle,
        tour=Tour.ATP,
        cutoff=cutoff,
        config=load_serve_model_config("config/model_v1.yaml"),
        code_commit="integration-test-commit",
        fitted_at_utc=datetime(2026, 8, 11, tzinfo=UTC),
    )
    receipt = bundle.manifest.receipt_for("component_counts")
    assert set(fits) == set(ServeComponent)
    assert all(fit.data_snapshot_sha256 == snapshot.sha256 for fit in fits.values())
    assert all(fit.component_count_artifact_sha256 == receipt.sha256 for fit in fits.values())

    first_direction = result.counts.loc[result.counts["component"].eq("F")].iloc[0]
    context = FutureMatchContext(
        serving_player_id=str(first_direction["player_id"]),
        returning_player_id=str(first_direction["opponent_id"]),
        tour=Tour.ATP,
        surface="Hard",
        indoor=False,
        event="Test Open",
        event_year=2026,
        match_date_utc=datetime(2026, 8, 12, tzinfo=UTC),
        information_cutoff_utc=cutoff,
    )
    prediction = predict_serve_performance(fits, context)
    assert prediction.fit_identity.component_count_artifact_sha256 == receipt.sha256
    assert prediction.context == context


def test_multiple_verified_yearly_bundles_form_one_auditable_fit_input(
    tmp_path: Path,
) -> None:
    def rows(tourney_id: str, offset: int) -> list[dict[str, str]]:
        return [
            valid_row(
                tourney_id=tourney_id,
                match_num=str(index + 1),
                w_ace=str(7 + (index + offset) % 4),
                w_df=str(3 + (index + offset) % 3),
                w_1stIn=str(57 + (index + offset) % 6),
                w_1stWon=str(38 + (index + offset) % 5),
                w_2ndWon=str(18 + (index + offset) % 4),
                l_ace=str(3 + (index + offset) % 3),
                l_df=str(4 + (index + offset) % 2),
                l_1stIn=str(48 + (index + offset) % 5),
                l_1stWon=str(27 + (index + offset) % 4),
                l_2ndWon=str(13 + (index + offset) % 4),
            )
            for index in range(8)
        ]

    cutoff = datetime(2026, 8, 10, tzinfo=UTC)
    bundles = []
    for name, offset in (("2026-FIRST", 0), ("2026-SECOND", 5)):
        snapshot, _ = snapshot_for(tmp_path / name, rows(name, offset))
        result = ingest_sackmann_snapshot(snapshot, cutoff=InformationCutoff(cutoff))
        bundles.append(write_processed_bundle(result, tmp_path / "processed"))

    fits = fit_all_serve_components_from_bundles(
        bundles,
        tour=Tour.ATP,
        cutoff=cutoff,
        config=load_serve_model_config("config/model_v1.yaml"),
        code_commit="multi-source-integration-test",
        fitted_at_utc=datetime(2026, 8, 11, tzinfo=UTC),
    )
    expected_snapshot_hash = fit_input_set_sha256(
        "source_snapshots", [bundle.manifest.snapshot_sha256 for bundle in bundles]
    )
    expected_count_hash = fit_input_set_sha256(
        "component_count_artifacts",
        [bundle.manifest.receipt_for("component_counts").sha256 for bundle in bundles],
    )
    assert all(fit.data_snapshot_sha256 == expected_snapshot_hash for fit in fits.values())
    assert all(
        fit.component_count_artifact_sha256 == expected_count_hash for fit in fits.values()
    )
