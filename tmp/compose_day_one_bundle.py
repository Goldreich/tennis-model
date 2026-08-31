from pathlib import Path

from tennis_model.estimation.snapshot import ModelSnapshot


RUNTIME = Path(r"C:\Users\orgol\AppData\Local\tm-usopen")
DESTINATION = RUNTIME / "artifacts/live-usopen-2026/official-2117-duration-v1"
DONOR = Path(
    r"C:\Users\orgol\AppData\Local\tennis-model-runtime\usopen-2026-08-31-five-v3"
) / "artifacts/live-usopen-2026/day-one-prop-bundle-v1"
OLD_RUNTIME = Path(
    r"C:\Users\orgol\AppData\Local\tennis-model-runtime\usopen-2026-08-31-five-v3"
)


for tour in ("atp", "wta"):
    target_path = DESTINATION / f"model_snapshot_{tour}.json"
    donor_path = DONOR / f"model_snapshot_{tour}.json"
    production = ModelSnapshot.from_json(target_path.read_bytes())
    duration_source = ModelSnapshot.from_json(donor_path.read_bytes())
    if production.tour != duration_source.tour:
        raise RuntimeError(f"tour mismatch while composing {tour} snapshot")
    if production.data_cutoff_utc != duration_source.data_cutoff_utc:
        raise RuntimeError(f"cutoff mismatch while composing {tour} snapshot")
    if duration_source.duration_artifact is None:
        raise RuntimeError(f"duration artifact missing from {tour} donor snapshot")

    retirement = production.retirement_artifact
    if retirement is None:
        raise RuntimeError(f"retirement artifact missing from {tour} production snapshot")
    retirement_directory = DESTINATION / Path(retirement.directory).relative_to(
        Path(r"C:\Users\orgol\OneDrive\Documents\Independent-Research\tennis-model")
        / "artifacts/live-usopen-2026/official-2117-v1"
    )
    rebased_retirement = retirement.model_copy(
        update={"directory": retirement_directory}
    )

    duration = duration_source.duration_artifact
    duration_directory = RUNTIME / Path(duration.directory).relative_to(OLD_RUNTIME)
    rebased_duration = duration.model_copy(update={"directory": duration_directory})

    for field in (
        "tour",
        "data_cutoff_utc",
        "data_hash",
        "component_count_artifact_hash",
        "config_hash",
    ):
        if getattr(production, field) != getattr(duration_source, field):
            raise RuntimeError(f"snapshot mismatch for {tour}: {field}")

    composed = duration_source.model_copy(
        update={
            "retirement_artifact": rebased_retirement,
            "retirement_schema_version": production.retirement_schema_version,
            "inactivity_configuration": production.inactivity_configuration,
            "inactivity_schema_version": production.inactivity_schema_version,
            "duration_artifact": rebased_duration,
        }
    )
    target_path.write_text(composed.canonical_json(), encoding="utf-8")

print(DESTINATION)
