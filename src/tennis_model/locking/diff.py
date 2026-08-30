"""Non-causal audit and numerical diffs between immutable lock revisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tennis_model.locking.models import PredictionSnapshot


@dataclass(frozen=True, slots=True)
class MetadataChange:
    field: str
    old: Any
    new: Any


@dataclass(frozen=True, slots=True)
class ProbabilityChange:
    prop_id: str
    prop: str
    old_probability: float | None
    new_probability: float | None
    difference: float | None
    old_probability_settled: float
    new_probability_settled: float


@dataclass(frozen=True, slots=True)
class LockDiff:
    old_lock_id: str
    new_lock_id: str
    metadata_changes: tuple[MetadataChange, ...]
    probability_changes: tuple[ProbabilityChange, ...]


def compare_locks(old: PredictionSnapshot, new: PredictionSnapshot) -> LockDiff:
    fields = {
        "information_cutoff": (
            old.context.information_cutoff_utc,
            new.context.information_cutoff_utc,
        ),
        "data_snapshot": (
            old.match_parameters.snapshot.data_hash,
            new.match_parameters.snapshot.data_hash,
        ),
        "model_snapshot": (old.match_parameters.snapshot_id, new.match_parameters.snapshot_id),
        "duration_artifact": (
            old.duration_model_artifact_id,
            new.duration_model_artifact_id,
        ),
        "duration_display_policy": (
            old.simulation.duration_display_policy_version,
            new.simulation.duration_display_policy_version,
        ),
        "code_commit": (old.code.model_dump(mode="json"), new.code.model_dump(mode="json")),
        "configuration": (
            old.lock_configuration_sha256,
            new.lock_configuration_sha256,
        ),
        "scenario": (old.information.scenario_id, new.information.scenario_id),
        "settlement_policy": (
            old.settlement_policy.model_dump(mode="json"),
            new.settlement_policy.model_dump(mode="json"),
        ),
        "seed_path_count": (
            (old.simulation.seed_id, old.simulation.actual_paths),
            (new.simulation.seed_id, new.simulation.actual_paths),
        ),
        "prop_set": (
            tuple(item.prop_id for item in old.prop_estimates),
            tuple(item.prop_id for item in new.prop_estimates),
        ),
    }
    metadata = tuple(
        MetadataChange(field=name, old=values[0], new=values[1])
        for name, values in fields.items()
        if values[0] != values[1]
    )
    old_props = {item.prop_id: item for item in old.prop_estimates}
    new_props = {item.prop_id: item for item in new.prop_estimates}
    probability = []
    for prop_id in sorted(old_props.keys() & new_props.keys()):
        left = old_props[prop_id]
        right = new_props[prop_id]
        difference = (
            None
            if left.probability_raw is None or right.probability_raw is None
            else right.probability_raw - left.probability_raw
        )
        probability.append(
            ProbabilityChange(
                prop_id=prop_id,
                prop=left.prop.original_text or left.prop.kind,
                old_probability=left.probability_raw,
                new_probability=right.probability_raw,
                difference=difference,
                old_probability_settled=left.probability_settled,
                new_probability_settled=right.probability_settled,
            )
        )
    return LockDiff(
        old_lock_id=old.lock_id,
        new_lock_id=new.lock_id,
        metadata_changes=metadata,
        probability_changes=tuple(probability),
    )
