from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_model.estimation.config import load_serve_model_config


def test_repository_model_config_is_complete_stable_and_finite() -> None:
    first = load_serve_model_config("config/model_v1.yaml")
    second = load_serve_model_config("config/model_v1.yaml")
    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    int(first.sha256, 16)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    source = Path("config/model_v1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text(source + "\nframework_version: v1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_serve_model_config(path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("intercept_sd: 2.5", 'intercept_sd: "2.5"'),
        ("include_indoor_hard: false", 'include_indoor_hard: "false"'),
        ("intercept_sd: 2.5", "intercept_sd: .nan"),
    ],
)
def test_probability_config_rejects_coerced_or_nonfinite_scalars(
    old: str, new: str, tmp_path: Path
) -> None:
    source = Path("config/model_v1.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace(old, new), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_serve_model_config(path)
