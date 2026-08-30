from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from math import exp

import numpy as np
import pandas as pd

from tennis_model.estimation.config import load_serve_model_config
from tennis_model.estimation.serve_components import (
    ContextConfig,
    FitProvenance,
    ServeComponent,
    ServeModelConfig,
)
from tennis_model.schemas import Tour

TEST_CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_provenance(label: str = "synthetic", *, tour: Tour = Tour.ATP) -> FitProvenance:
    return FitProvenance(
        data_snapshot_sha256=_digest(f"snapshot:{tour.value}"),
        component_count_artifact_sha256=_digest(f"counts:{label}"),
        code_commit="test-fixture-commit",
        fitted_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )


def make_model_config(
    *,
    event_components: tuple[ServeComponent, ...] = (),
    include_indoor_hard: bool = False,
    max_iterations: int = 1200,
    laplace_max_full_parameters: int = 100,
) -> ServeModelConfig:
    config = load_serve_model_config("config/model_v1.yaml")
    optimizer = config.optimizer.model_copy(
        update={
            "max_iterations": max_iterations,
            "laplace_max_full_parameters": laplace_max_full_parameters,
        }
    )
    context = ContextConfig(
        include_indoor_hard=include_indoor_hard,
        event_year_effect_components=event_components,
    )
    return config.model_copy(update={"optimizer": optimizer, "context": context})


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def synthetic_component_counts(
    component: ServeComponent,
    *,
    tour: Tour = Tour.ATP,
    players: tuple[str, ...] = ("p0", "p1", "p2", "p3"),
    repetitions: int = 5,
    trials: int = 80,
    intercept: float = -0.2,
    server_effects: dict[str, float] | None = None,
    returner_effects: dict[str, float] | None = None,
    hard_deviations: dict[str, float] | None = None,
    surfaces: tuple[str, ...] = ("Hard", "Clay"),
    kappa: float = 40.0,
    seed: int = 20260829,
) -> pd.DataFrame:
    server_values = server_effects or {}
    returner_values = returner_effects or {}
    hard_values = hard_deviations or {}
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    row_number = 0
    for repetition in range(repetitions):
        for server in players:
            for returner in players:
                if server == returner:
                    continue
                surface = surfaces[(row_number + repetition) % len(surfaces)]
                eta = intercept + server_values.get(server, 0.0)
                if component in (ServeComponent.A, ServeComponent.Q1, ServeComponent.Q2):
                    eta -= returner_values.get(returner, 0.0)
                if surface == "Hard":
                    eta += hard_values.get(server, 0.0)
                mean = _logistic(eta)
                match_probability = rng.beta(kappa * mean, kappa * (1.0 - mean))
                successes = int(rng.binomial(trials, match_probability))
                match_date = (TEST_CUTOFF - timedelta(days=30 + 7 * repetition)).date()
                available_at = datetime.combine(
                    match_date + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=UTC,
                )
                records.append(
                    {
                        "snapshot_id": _digest(f"snapshot:{tour.value}"),
                        "snapshot_sha256": _digest(f"snapshot:{tour.value}"),
                        "transformation_version": "serve-component-counts-v1.1",
                        "source_id": f"synthetic-{tour.value.lower()}",
                        "source_row_number": row_number + 2,
                        "orientation": "synthetic",
                        "match_id": f"{tour.value}-{component.value}-{row_number}",
                        "source_date": match_date,
                        "match_date": match_date,
                        "available_at_utc": available_at,
                        "tour": tour.value,
                        "event": "Synthetic Open",
                        "event_year": match_date.year,
                        "surface": surface,
                        "indoor": "Outdoor",
                        "best_of": 3,
                        "player_id": server,
                        "opponent_id": returner,
                        "component": component.value,
                        "successes": successes,
                        "trials": trials,
                        "status": "valid",
                        "eligible_for_likelihood": True,
                        "anomaly_codes": (),
                    }
                )
                row_number += 1
    return pd.DataFrame.from_records(records)


def synthetic_all_component_counts(
    *,
    tour: Tour = Tour.ATP,
    repetitions: int = 4,
    seed: int = 20260829,
) -> pd.DataFrame:
    intercepts = {
        ServeComponent.F: 0.55,
        ServeComponent.A: -2.0,
        ServeComponent.Q1: 0.65,
        ServeComponent.D: -2.3,
        ServeComponent.Q2: 0.2,
    }
    return pd.concat(
        [
            synthetic_component_counts(
                component,
                tour=tour,
                repetitions=repetitions,
                intercept=intercepts[component],
                seed=seed + index,
            )
            for index, component in enumerate(ServeComponent)
        ],
        ignore_index=True,
    )
