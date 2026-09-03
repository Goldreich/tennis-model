from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tennis_model.estimation.elo import (
    SurfaceEloConfig,
    SurfaceEloDiagnostics,
    SurfaceEloFit,
    SurfaceEloPlayerState,
    SurfaceEloRating,
    load_surface_elo_artifact,
    predict_surface_elo,
    write_surface_elo_artifact,
)
from tennis_model.estimation.strength_integration import (
    FixedStrengthIntegrationFit,
    StrengthIntegrationConfig,
    prepare_strength_match_parameters,
)
from tennis_model.schemas import Tour
from tennis_model.serve import PrimitiveServeMeans


def _elo_fit() -> SurfaceEloFit:
    cutoff = datetime(2026, 8, 30, tzinfo=UTC)
    config = SurfaceEloConfig(schema_version="surface-elo-config/v1")
    players = (
        SurfaceEloPlayerState(
            player_id="a",
            player_name="Player A",
            aliases=("player-a",),
            global_rating=1600.0,
            surfaces=(SurfaceEloRating(surface="hard", rating=1700.0),),
            eligible_matches=20,
        ),
        SurfaceEloPlayerState(
            player_id="b",
            player_name="Player B",
            aliases=("player-b",),
            global_rating=1500.0,
            surfaces=(SurfaceEloRating(surface="hard", rating=1500.0),),
            eligible_matches=20,
        ),
    )
    return SurfaceEloFit(
        tour=Tour.ATP,
        information_cutoff_utc=cutoff,
        fitted_at_utc=cutoff,
        config=config,
        ratings_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        code_commit="test",
        player_states=players,
        diagnostics=SurfaceEloDiagnostics(
            players=2, eligible_matches=20, initialized_players=0
        ),
    )


def test_surface_elo_prediction_is_symmetric_and_alias_aware() -> None:
    fit = _elo_fit()
    start = datetime(2026, 9, 1, tzinfo=UTC)
    forward = predict_surface_elo(
        fit,
        player_a_id="player-a",
        player_b_id="b",
        surface="hard",
        best_of=5,
        scheduled_start_utc=start,
    )
    reverse = predict_surface_elo(
        fit,
        player_a_id="b",
        player_b_id="player-a",
        surface="hard",
        best_of=5,
        scheduled_start_utc=start,
    )
    assert forward.mean_logit == pytest.approx(-reverse.mean_logit, abs=1e-12)
    assert forward.probability + reverse.probability == pytest.approx(1.0, abs=1e-12)


def test_surface_elo_artifact_is_content_addressed(tmp_path) -> None:
    artifact = write_surface_elo_artifact(_elo_fit(), tmp_path)
    assert load_surface_elo_artifact(artifact.directory) == artifact


def test_fixed_elo_blend_targets_seventy_five_percent_anchor_logit() -> None:
    fit = _elo_fit()
    anchor = predict_surface_elo(
        fit,
        player_a_id="a",
        player_b_id="b",
        surface="hard",
        best_of=3,
        scheduled_start_utc=datetime(2026, 9, 1, tzinfo=UTC),
    )
    integration = FixedStrengthIntegrationFit(
        tour=Tour.ATP,
        training_cutoff_utc=datetime(2026, 8, 30, tzinfo=UTC),
        fitted_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
        config=StrengthIntegrationConfig(
            schema_version="strength-integration-config/v1",
            l2_penalty=1.0,
            reliability_prior_logit=-1.3862943611198906,
            maximum_absolute_tilt=3.0,
            root_tolerance=1e-8,
            probability_bound=1e-6,
            coefficient_draws_for_summary=64,
            q1_weight=1.0,
            q2_weight=1.0,
        ),
        anchor_weight=0.75,
        selection_reference="test",
        code_commit="test",
    )
    a = PrimitiveServeMeans(0.62, 0.12, 0.58, 0.08, 0.52)
    b = PrimitiveServeMeans(0.61, 0.09, 0.54, 0.07, 0.49)
    parameters = prepare_strength_match_parameters(
        anchor_artifact_id="a" * 64,
        integration_artifact_id="b" * 64,
        anchor=anchor,
        integration=integration,
        player_a=a,
        player_b=b,
        best_of=3,
        component_variance=0.2,
        component_instability=0.2,
        component_sparsity=0.2,
    )
    expected = 0.25 * parameters.component_logit + 0.75 * anchor.mean_logit
    assert parameters.reliability_weight == pytest.approx(0.75)
    assert parameters.target_logit == pytest.approx(expected)
