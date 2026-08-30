from __future__ import annotations

from pathlib import Path

import pytest
from serve_model_test_helpers import (
    TEST_CUTOFF,
    make_model_config,
    make_provenance,
    synthetic_component_counts,
)

from tennis_model.estimation.artifacts import (
    FitArtifactError,
    FitArtifactIntegrityError,
    load_fit_artifact,
    write_fit_artifact,
)
from tennis_model.estimation.serve_components import ServeComponent, fit_serve_component
from tennis_model.schemas import Tour


def _fit():
    return fit_serve_component(
        synthetic_component_counts(ServeComponent.D, repetitions=3),
        component=ServeComponent.D,
        tour=Tour.ATP,
        cutoff=TEST_CUTOFF,
        config=make_model_config(),
        provenance=make_provenance("artifact"),
    )


def test_fit_artifact_is_content_addressed_verified_and_idempotent(tmp_path: Path) -> None:
    fit = _fit()
    first = write_fit_artifact(fit, tmp_path / "fits")
    second = write_fit_artifact(fit, tmp_path / "fits")
    loaded = load_fit_artifact(first.directory)

    assert first.directory == second.directory
    assert first.artifact_id == second.artifact_id == loaded.artifact_id
    assert len(first.artifact_id) == 64
    assert loaded.fit == fit
    assert first.fit_path.is_file()
    assert first.diagnostics_path.is_file()
    diagnostics = first.diagnostics_path.read_text(encoding="utf-8")
    assert "tour/component: ATP/D" in diagnostics
    assert "optimizer: converged=True" in diagnostics


def test_different_provenance_creates_a_new_fit_without_overwrite(tmp_path: Path) -> None:
    first_fit = _fit()
    second_fit = first_fit.model_copy(update={"code_commit": "different-test-commit"})
    first = write_fit_artifact(first_fit, tmp_path / "fits")
    second = write_fit_artifact(second_fit, tmp_path / "fits")

    assert first.artifact_id != second.artifact_id
    assert first.directory != second.directory
    assert load_fit_artifact(first.directory).fit.code_commit == "test-fixture-commit"
    assert load_fit_artifact(second.directory).fit.code_commit == "different-test-commit"


def test_fit_artifact_tampering_is_detected(tmp_path: Path) -> None:
    artifact = write_fit_artifact(_fit(), tmp_path / "fits")
    artifact.fit_path.write_bytes(artifact.fit_path.read_bytes() + b" ")
    with pytest.raises(FitArtifactIntegrityError, match=r"invalid|canonical"):
        load_fit_artifact(artifact.directory)


def test_human_diagnostics_are_verified_not_advisory(tmp_path: Path) -> None:
    artifact = write_fit_artifact(_fit(), tmp_path / "fits")
    artifact.diagnostics_path.write_text("incorrect\n", encoding="utf-8")
    with pytest.raises(FitArtifactIntegrityError, match="diagnostics differ"):
        load_fit_artifact(artifact.directory)


@pytest.mark.parametrize(
    "invalid_fit",
    [
        lambda fit: fit.model_copy(update={"training_data_sha256": "not-a-hash"}),
        lambda fit: fit.model_copy(update={"code_commit": ""}),
        lambda fit: fit.model_copy(
            update={
                "posterior": fit.posterior.model_copy(
                    update={
                        "map_parameters": (
                            float("nan"),
                            *fit.posterior.map_parameters[1:],
                        )
                    }
                )
            }
        ),
    ],
)
def test_invalid_copied_fit_is_revalidated_before_persistence(invalid_fit, tmp_path: Path) -> None:
    with pytest.raises(FitArtifactError, match="invalid fitted component"):
        write_fit_artifact(invalid_fit(_fit()), tmp_path / "fits")
