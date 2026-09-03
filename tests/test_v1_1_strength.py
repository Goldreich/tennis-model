from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tennis_model.estimation.strength import (
    DynamicStrengthConfig,
    StrengthOutcomeRecord,
    fit_dynamic_strength,
    load_strength_artifact,
    predict_strength,
    write_strength_artifact,
)
from tennis_model.schemas import Tour


def _config() -> DynamicStrengthConfig:
    return DynamicStrengthConfig(
        schema_version="dynamic-strength-config/v1",
        decay_days=365.0,
        process_sd=0.35,
        initial_sd=1.5,
        surface_sd=0.55,
        observation_scale_best_of_3=1.0,
        observation_scale_best_of_5=1.15,
        variance_floor=1e-4,
        ranking_sparse_prior_enabled=False,
        ranking_intercept=0.0,
        ranking_log_slope=0.0,
        ranking_prior_sd=1.5,
        head_to_head_enabled=False,
        head_to_head_prior_sd=0.15,
    )


def _records(cutoff: datetime) -> tuple[StrengthOutcomeRecord, ...]:
    values = []
    for index in range(24):
        start = cutoff - timedelta(days=30 - index)
        values.append(
            StrengthOutcomeRecord(
                match_id=f"m-{index}",
                tour=Tour.ATP,
                player_a_id="strong",
                player_b_id="weak",
                winner_id="strong" if index % 5 else "weak",
                start_utc=start,
                available_at_utc=start + timedelta(hours=3),
                surface="hard",
                best_of=3,
            )
        )
    values.append(
        StrengthOutcomeRecord(
            match_id="future",
            tour=Tour.ATP,
            player_a_id="weak",
            player_b_id="strong",
            winner_id="weak",
            start_utc=cutoff + timedelta(days=1),
            available_at_utc=cutoff + timedelta(days=1, hours=3),
            surface="hard",
            best_of=3,
        )
    )
    return tuple(values)


def test_strength_fit_is_cutoff_safe_reversible_and_content_addressed(tmp_path) -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    fit = fit_dynamic_strength(
        _records(cutoff),
        tour=Tour.ATP,
        cutoff_utc=cutoff,
        fitted_at_utc=cutoff + timedelta(minutes=1),
        config=_config(),
        code_commit="test",
    )
    assert fit.diagnostics.future_rows_excluded == 1
    forward = predict_strength(
        fit,
        player_a_id="strong",
        player_b_id="weak",
        surface="hard",
        best_of=3,
        scheduled_start_utc=cutoff + timedelta(days=1),
    )
    reverse = predict_strength(
        fit,
        player_a_id="weak",
        player_b_id="strong",
        surface="hard",
        best_of=3,
        scheduled_start_utc=cutoff + timedelta(days=1),
    )
    assert forward.mean_logit == pytest.approx(-reverse.mean_logit, abs=1e-12)
    assert forward.probability + reverse.probability == pytest.approx(1.0, abs=1e-12)
    artifact = write_strength_artifact(fit, tmp_path)
    assert load_strength_artifact(artifact.directory) == artifact
