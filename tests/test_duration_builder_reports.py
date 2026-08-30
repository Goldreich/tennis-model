import json
from datetime import UTC, datetime
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

_BUILD_SCRIPT = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "build_duration_models.py")
)
_current_event_stability = _BUILD_SCRIPT["_current_event_stability"]
_threshold_diagnostics = _BUILD_SCRIPT["_threshold_diagnostics"]
assert_duration_build_not_rejected = _BUILD_SCRIPT[
    "assert_duration_build_not_rejected"
]
select_latest_ready_duration_build = _BUILD_SCRIPT[
    "select_latest_ready_duration_build"
]
write_duration_build_rejection = _BUILD_SCRIPT["write_duration_build_rejection"]


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fake_duration_build(
    output_root: Path,
    *,
    base_prefix: str,
    run_id: str,
    fitted_at_utc: str,
) -> Path:
    directory = output_root / base_prefix / run_id[:32]
    _write_canonical(
        directory / "build_report.json",
        {
            "fitted_at_utc": fitted_at_utc,
            "run_id": run_id,
            "schema_version": "current-usopen-duration-build/v1",
            "status": "READY",
        },
    )
    _write_canonical(
        directory / "performance_report.json",
        {
            "by_tour": {
                tour: {
                    "checkpoints": [
                        {
                            "mean_minutes": 100_000_000.0,
                            "p90_minutes": 200_000_000.0,
                            "paths": 100_000,
                        }
                    ]
                }
                for tour in ("ATP", "WTA")
            }
        },
    )
    _write_canonical(
        directory / "validation_report.json",
        {
            "by_tour": {
                tour: {
                    "fit": {
                        "posterior_covariance_mode": "diagonal",
                        "posterior_dimension": 400,
                    }
                }
                for tour in ("ATP", "WTA")
            }
        },
    )
    return directory


def test_threshold_report_exposes_all_unresolved_display_boundaries() -> None:
    report = _threshold_diagnostics(
        np.asarray([120.0, 121.0]),
        np.asarray([120.0, 120.0]),
        np.asarray([1.0, 1.0]),
        6.0,
    )["120"]

    assert report["actual_official_integer_frequency"] == 0.5
    assert report["official_integer_rows"] == 2
    assert report["noninteger_observed_rows"] == 0
    sensitivity = report["unresolved_display_policy_sensitivity"]
    policies = sensitivity["analytic_probabilities"]
    assert policies["floor"]["latent_boundary_minutes"] == 121.0
    assert policies["nearest_half_up"]["latent_boundary_minutes"] == 120.5
    assert policies["ceiling"]["latent_boundary_minutes"] == 120.0
    assert (
        policies["floor"]["mean_predicted_probability"]
        < policies["nearest_half_up"]["mean_predicted_probability"]
        < policies["ceiling"]["mean_predicted_probability"]
    )
    assert sensitivity["mean_probability_range"] > 0.0


def test_current_event_stability_is_explicitly_laplace_and_descriptive() -> None:
    report = _current_event_stability(
        event_effect=SimpleNamespace(value=-2.0, standard_error=1.5),
        row_count=111,
        prior_standard_deviation=4.0,
    )

    assert report["row_count"] == 111
    assert report["event_effect_minutes"] == -2.0
    assert report["laplace_standard_error_minutes"] == 1.5
    assert report["laplace_z_ratio"] == pytest.approx(-2.0 / 1.5)
    assert report["effect_to_prior_standard_deviation"] == -0.5
    assert report["approximate_95pct_wald_interval_minutes"] == {
        "lower_minutes": pytest.approx(-4.94),
        "upper_minutes": pytest.approx(0.94),
    }
    assert "descriptive in-sample" in report["scope"]
    assert "not a held-out" in report["approximation"]


def test_rejection_receipt_is_exclusive_and_selection_skips_run(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "duration-builds"
    base_prefix = "base-build-prefix"
    older = _fake_duration_build(
        output_root,
        base_prefix=base_prefix,
        run_id="a" * 64,
        fitted_at_utc="2026-08-30T15:00:00+00:00",
    )
    newest = _fake_duration_build(
        output_root,
        base_prefix=base_prefix,
        run_id="b" * 64,
        fitted_at_utc="2026-08-30T16:00:00+00:00",
    )
    assert select_latest_ready_duration_build(output_root, base_prefix) == newest

    receipt = write_duration_build_rejection(
        output_root,
        newest,
        detected_at_utc=datetime(2026, 8, 30, 16, 30, tzinfo=UTC),
    )
    assert receipt.name == f"{'b' * 64}.json"
    assert select_latest_ready_duration_build(output_root, base_prefix) == older
    assert assert_duration_build_not_rejected(older, output_root) == "a" * 64
    with pytest.raises(RuntimeError, match="explicitly rejected"):
        assert_duration_build_not_rejected(newest, output_root)
    with pytest.raises(FileExistsError):
        write_duration_build_rejection(
            output_root,
            newest,
            detected_at_utc=datetime(2026, 8, 30, 16, 31, tzinfo=UTC),
        )
