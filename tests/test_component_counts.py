from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tennis_model.data.component_counts import (
    SERVE_COMPONENTS,
    ComponentStatus,
    build_serve_component_counts,
)


def service_row(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "snapshot_id": "snapshot-1",
        "source_row_number": 2,
        "orientation": "winner",
        "match_id": "match-1",
        "tour": "ATP",
        "player_id": "sackmann:atp:1",
        "opponent_id": "sackmann:atp:2",
        "service_points": 100,
        "first_serves_in": 60,
        "first_serve_points_won": 42,
        "second_serve_points_won": 20,
        "aces": 10,
        "double_faults": 5,
        "invalid_stat_fields": (),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def by_component(frame: pd.DataFrame, component: str) -> pd.Series:
    return frame.loc[frame["component"] == component].iloc[0]


@st.composite
def valid_primitive_totals(draw: st.DrawFn) -> dict[str, int]:
    service_points = draw(st.integers(min_value=0, max_value=500))
    first_serves_in = draw(st.integers(min_value=0, max_value=service_points))
    aces = draw(st.integers(min_value=0, max_value=first_serves_in))
    first_serve_points_won = draw(st.integers(min_value=aces, max_value=first_serves_in))
    second_opportunities = service_points - first_serves_in
    double_faults = draw(st.integers(min_value=0, max_value=second_opportunities))
    second_serve_points_won = draw(
        st.integers(
            min_value=0,
            max_value=second_opportunities - double_faults,
        )
    )
    return {
        "service_points": service_points,
        "first_serves_in": first_serves_in,
        "first_serve_points_won": first_serve_points_won,
        "second_serve_points_won": second_serve_points_won,
        "aces": aces,
        "double_faults": double_faults,
    }


def test_constructs_the_frozen_primitive_counts_and_identities() -> None:
    table = build_serve_component_counts(service_row())
    observed = {
        component: (
            int(by_component(table.counts, component)["successes"]),
            int(by_component(table.counts, component)["trials"]),
        )
        for component in SERVE_COMPONENTS
    }
    assert observed == {
        "F": (60, 100),
        "A": (10, 60),
        "Q1": (32, 50),
        "D": (5, 40),
        "Q2": (20, 35),
    }
    assert observed["A"][0] + observed["Q1"][0] == 42
    assert observed["A"][0] + observed["Q1"][1] == 60
    assert observed["D"][0] + observed["Q2"][1] == 40
    assert observed["Q2"][0] == 20
    assert len(table.likelihood_rows()) == 5
    assert table.anomalies.empty


@given(valid_primitive_totals())
def test_all_valid_integer_totals_obey_the_frozen_reconstruction_identities(
    totals: dict[str, int],
) -> None:
    table = build_serve_component_counts(service_row(**totals))
    counts = table.counts.set_index("component")
    assert set(counts["status"]) <= {"valid", "zero_denominator"}
    assert table.anomalies.empty
    assert (
        counts.loc["A", "successes"] + counts.loc["Q1", "successes"]
        == totals["first_serve_points_won"]
    )
    assert counts.loc["A", "successes"] + counts.loc["Q1", "trials"] == totals["first_serves_in"]
    assert counts.loc["D", "successes"] + counts.loc["Q2", "trials"] == (
        totals["service_points"] - totals["first_serves_in"]
    )
    assert counts.loc["Q2", "successes"] == totals["second_serve_points_won"]


@pytest.mark.parametrize(
    ("missing_field", "missing_components"),
    [
        ("service_points", {"F", "D", "Q2"}),
        ("first_serves_in", set(SERVE_COMPONENTS)),
        ("aces", {"A", "Q1"}),
        ("first_serve_points_won", {"Q1"}),
        ("double_faults", {"D", "Q2"}),
        ("second_serve_points_won", {"Q2"}),
    ],
)
def test_missing_inputs_are_component_local(
    missing_field: str, missing_components: set[str]
) -> None:
    table = build_serve_component_counts(service_row(**{missing_field: pd.NA}))
    observed = {
        component
        for component in SERVE_COMPONENTS
        if by_component(table.counts, component)["status"] == ComponentStatus.MISSING_INPUT.value
    }
    assert observed == missing_components
    assert (
        table.counts.loc[table.counts["component"].isin(missing_components), "successes"]
        .isna()
        .all()
    )


def test_zero_denominators_are_not_missing_or_anomalous() -> None:
    table = build_serve_component_counts(
        service_row(
            service_points=0,
            first_serves_in=0,
            first_serve_points_won=0,
            second_serve_points_won=0,
            aces=0,
            double_faults=0,
        )
    )
    assert set(table.counts["status"]) == {ComponentStatus.ZERO_DENOMINATOR.value}
    assert not table.counts["eligible_for_likelihood"].any()
    assert table.anomalies.empty


@pytest.mark.parametrize(
    ("overrides", "code", "invalid_components", "valid_components"),
    [
        (
            {"aces": 61},
            "ACES_GT_FIRST_SERVES_IN",
            {"A", "Q1"},
            {"F", "D", "Q2"},
        ),
        (
            {"aces": 43},
            "ACES_GT_FIRST_SERVE_POINTS_WON",
            {"Q1"},
            {"F", "A", "D", "Q2"},
        ),
        (
            {"double_faults": 41},
            "DOUBLE_FAULTS_GT_SECOND_SERVE_OPPORTUNITIES",
            {"D", "Q2"},
            {"F", "A", "Q1"},
        ),
        (
            {"second_serve_points_won": 36},
            "SECOND_SERVE_POINTS_WON_GT_PLAYABLE_SECOND_SERVES",
            {"Q2"},
            {"F", "A", "Q1", "D"},
        ),
    ],
)
def test_named_anomalies_quarantine_only_affected_components(
    overrides: dict[str, int],
    code: str,
    invalid_components: set[str],
    valid_components: set[str],
) -> None:
    table = build_serve_component_counts(service_row(**overrides))
    statuses = table.counts.set_index("component")["status"].to_dict()
    assert {
        component
        for component, status in statuses.items()
        if status == ComponentStatus.QUARANTINED.value
    } == invalid_components
    assert all(statuses[component] == "valid" for component in valid_components)
    assert code in set(table.anomalies["code"])


def test_invalid_derived_counts_are_preserved_not_clipped() -> None:
    table = build_serve_component_counts(service_row(aces=61))
    q1 = by_component(table.counts, "Q1")
    assert int(q1["successes"]) == -19
    assert int(q1["trials"]) == -1


def test_malformed_integer_is_quarantined_not_treated_as_missing() -> None:
    table = build_serve_component_counts(service_row(aces=pd.NA, invalid_stat_fields=("aces",)))
    assert by_component(table.counts, "A")["status"] == "quarantined"
    assert by_component(table.counts, "Q1")["status"] == "quarantined"
    assert by_component(table.counts, "F")["status"] == "valid"
    assert "MALFORMED_ACES" in set(table.anomalies["code"])


def test_required_normalized_columns_are_enforced() -> None:
    with pytest.raises(ValueError, match="second_serve_points_won"):
        build_serve_component_counts(service_row().drop(columns=["second_serve_points_won"]))


def test_nan_is_missing_not_a_malformed_or_zero_count() -> None:
    table = build_serve_component_counts(service_row(aces=float("nan")))
    statuses = table.counts.set_index("component")["status"]
    assert statuses["A"] == "missing_input"
    assert statuses["Q1"] == "missing_input"
    assert statuses["F"] == "valid"
    assert table.anomalies.empty
