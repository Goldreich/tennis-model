"""Typed, immutable schemas for Tennis Model v1.0 prediction locks."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from math import isclose
from typing import Any, Literal, Self, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from tennis_model.data.historical_validation import (
    POINT_IN_TIME_VINTAGE_POLICY,
    HistoricalValidationPolicy,
)
from tennis_model.estimation.inactivity import (
    InactivityCoverageAssertion,
    InactivityMatchCandidate,
)
from tennis_model.estimation.serve_components import ServeComponent
from tennis_model.identity import CanonicalMatchIdentity
from tennis_model.locking.path_counts import (
    ADAPTIVE_MC_POLICY_VERSION,
    BERNOULLI_CS_METHOD_VERSION,
    AdaptivePropDiagnostics,
    MCStoppingStatus,
)
from tennis_model.props.rounding import (
    MODEL_ROUNDING_POLICY_VERSION,
    PlatformSubmissionPolicy,
    confidence_interval_model_integer,
    model_probability_integer,
)
from tennis_model.schemas import FrozenModel, SourceManifest, Tour
from tennis_model.simulation.match import BooleanCompositeSpec, PropEstimate, PropSpec
from tennis_model.simulation.parameters import MatchContext, MatchParameterRecord

from ._json import require_sha256, sha256_json

LOCK_SCHEMA_VERSION: Literal["prediction-lock/v1"] = "prediction-lock/v1"
LOCK_OPERATIONAL_SCHEMA_VERSION: Literal["prediction-lock/v3"] = "prediction-lock/v3"
LOCK_DURATION_SCHEMA_VERSION: Literal["prediction-lock/v4"] = "prediction-lock/v4"
LEDGER_SCHEMA_VERSION: Literal["calibration-ledger/v1"] = "calibration-ledger/v1"
LEDGER_B6_C6_SCHEMA_VERSION: Literal["calibration-ledger/v2"] = "calibration-ledger/v2"
LEDGER_OPERATIONAL_SCHEMA_VERSION: Literal["calibration-ledger/v3"] = "calibration-ledger/v3"


class PropSupportStatus(StrEnum):
    """Operational availability state; never inferred from a probability value."""

    SUPPORTED = "SUPPORTED"
    POLICY_DISABLED = "POLICY_DISABLED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    SETTLEMENT_BLOCKED = "SETTLEMENT_BLOCKED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class ReplayLevel(StrEnum):
    HASH_VERIFICATION = "HASH_VERIFICATION"
    SAME_RUNTIME_EXACT = "SAME_RUNTIME_EXACT"
    CROSS_RUNTIME_SEMANTIC = "CROSS_RUNTIME_SEMANTIC"


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


class LockModel(FrozenModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class PropSupportDecision(LockModel):
    status: PropSupportStatus
    reason_code: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def reason_matches_status(self) -> Self:
        supported = self.status is PropSupportStatus.SUPPORTED
        if supported != (self.reason_code is None and self.detail is None):
            raise ValueError("supported props have no disable reason; other states require one")
        return self


class SourceTimeProvenance(LockModel):
    """Truthful source knowledge and modern reconstruction timestamps."""

    source_id: str
    source_effective_through: date
    source_effective_at_utc: datetime | None = None
    source_available_at_utc: datetime | None = None
    information_availability_rule: str
    retrieved_at_utc: datetime
    verified_at_utc: datetime

    @field_validator(
        "source_effective_at_utc",
        "source_available_at_utc",
        "retrieved_at_utc",
        "verified_at_utc",
    )
    @classmethod
    def audit_times_are_utc(cls, value: datetime | None, info: Any) -> datetime | None:
        if value is None:
            return None
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def information_times_are_coherent(self) -> Self:
        if (self.source_effective_at_utc is None) != (self.source_available_at_utc is None):
            raise ValueError("source effective/available timestamps must be jointly present")
        if (
            self.source_effective_at_utc is not None
            and self.source_available_at_utc is not None
            and self.source_effective_at_utc > self.source_available_at_utc
        ):
            raise ValueError("source cannot be available before it is effective")
        return self


class HistoricalTimeProvenance(LockModel):
    """Separate probability cutoffs from physical artifact construction times."""

    schema_version: Literal[
        "historical-time-provenance/v1", "historical-time-provenance/v2"
    ] = "historical-time-provenance/v2"
    information_cutoff_utc: datetime
    training_data_cutoff_utc: datetime
    artifact_created_at_utc: datetime
    sources: tuple[SourceTimeProvenance, ...]
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY

    @field_validator(
        "information_cutoff_utc", "training_data_cutoff_utc", "artifact_created_at_utc"
    )
    @classmethod
    def times_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def cutoff_is_safe_but_reconstruction_may_be_later(self) -> Self:
        if self.training_data_cutoff_utc > self.information_cutoff_utc:
            raise ValueError("training data cutoff cannot follow the forecast cutoff")
        if not self.sources:
            raise ValueError("historical time provenance requires source records")
        if len({item.source_id for item in self.sources}) != len(self.sources):
            raise ValueError("historical time provenance source IDs must be unique")
        return self


class TrainingInputCoverageRecord(LockModel):
    """Exact-date inclusion audit for one current fitted component."""

    component: Literal["F", "A", "Q1", "D", "Q2", "B6"]
    row_unit: Literal["component_rows", "matches"]
    included_exact_dated_rows: int = Field(ge=0)
    excluded_undated_candidate_rows: int = Field(ge=0)
    exclusion_rate: float = Field(ge=0, le=1)
    date_fallback_rows: int = Field(default=0, ge=0)
    included_unweighted_player_starts: int | None = Field(default=None, ge=0)
    included_weighted_player_starts: float | None = Field(default=None, ge=0)
    excluded_undated_player_starts: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def rate_and_b6_exposure_are_coherent(self) -> Self:
        denominator = self.included_exact_dated_rows + self.excluded_undated_candidate_rows
        expected = 0.0 if denominator == 0 else self.excluded_undated_candidate_rows / denominator
        if not isclose(self.exclusion_rate, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("training exclusion rate does not match retained counts")
        if self.date_fallback_rows != 0:
            raise ValueError("current fit eligibility cannot contain date-fallback rows")
        b6_exposure = (
            self.included_unweighted_player_starts,
            self.included_weighted_player_starts,
            self.excluded_undated_player_starts,
        )
        if self.component == "B6":
            if self.row_unit != "matches" or any(item is None for item in b6_exposure):
                raise ValueError("B6 coverage requires match rows and player-start exposure")
        elif self.row_unit != "component_rows" or any(item is not None for item in b6_exposure):
            raise ValueError("serve-component coverage cannot contain B6 exposure")
        return self


class HistoricalTrainingEligibilityProvenance(LockModel):
    """Current-lock audit of exact-dated training inputs and retained exclusions."""

    schema_version: Literal["current-training-eligibility/v1"] = (
        "current-training-eligibility/v1"
    )
    lock_semantics: Literal["CURRENT_POINT_IN_TIME"] = "CURRENT_POINT_IN_TIME"
    tour: Tour
    assertion_id: str
    verified_at_utc: datetime
    all_included_rows_have_verified_exact_dates: bool
    historical_exact_date_coverage_complete: bool
    source_manifest_sha256: str
    source_sha256s: tuple[str, ...]
    crosswalk_sha256s: tuple[str, ...]
    records: tuple[TrainingInputCoverageRecord, ...]
    warning: Literal["HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE"] | None = None

    @field_validator("verified_at_utc")
    @classmethod
    def verification_time_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="verified_at_utc")

    @field_validator("source_manifest_sha256")
    @classmethod
    def manifest_hash_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="source_manifest_sha256")

    @field_validator("source_sha256s", "crosswalk_sha256s")
    @classmethod
    def hashes_are_valid_and_canonical(
        cls, value: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        normalized = tuple(require_sha256(item, field=info.field_name) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError(f"{info.field_name} must be unique and sorted")
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @model_validator(mode="after")
    def records_and_warning_are_coherent(self) -> Self:
        if not self.assertion_id.strip():
            raise ValueError("training eligibility assertion_id must not be empty")
        components = tuple(item.component for item in self.records)
        if components != ("F", "A", "Q1", "D", "Q2", "B6"):
            raise ValueError("training coverage records must contain F/A/Q1/D/Q2/B6 in order")
        if not self.all_included_rows_have_verified_exact_dates:
            raise ValueError("a current fitted artifact cannot admit unverified dates")
        incomplete = not self.historical_exact_date_coverage_complete
        if incomplete != (self.warning == "HISTORICAL_EXACT_DATE_COVERAGE_INCOMPLETE"):
            raise ValueError("incomplete exact-date coverage requires the explicit warning")
        return self


class RuntimeFingerprint(LockModel):
    schema_version: Literal["runtime-fingerprint/v1"] = "runtime-fingerprint/v1"
    python_version: str
    python_implementation: str
    platform: str
    operating_system: str
    architecture: str
    machine: str
    numpy_version: str
    scipy_version: str
    pydantic_version: str
    rng_bit_generator: str
    numpy_random_context: str
    simulator_algorithm_version: str
    chunk_size: int = Field(gt=0)
    thread_count: int = Field(gt=0)
    process_count: int = Field(gt=0)
    blas_backend: tuple[str, ...] = ()


class RetainedArtifactRecord(LockModel):
    """One immutable local artifact required for full lock reproduction."""

    kind: Literal[
        "source_snapshot",
        "normalized_snapshot",
        "component_counts",
        "component_fit",
        "duration_fit",
        "retirement_fit",
        "inactivity_config",
        "model_config",
        "settlement_policy",
        "code_archive",
    ]
    artifact_id: str
    path: str
    sha256: str

    @field_validator("artifact_id", "path")
    @classmethod
    def retained_identity_is_present(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def retained_digest_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="sha256")


class InformationItem(LockModel):
    """One pre-match fact with an auditable availability time and source object."""

    category: Literal["schedule", "conditions", "health", "workload", "status", "other"]
    summary: str
    source_id: str
    source_sha256: str
    observed_at_utc: datetime
    available_at_utc: datetime

    @field_validator("summary", "source_id")
    @classmethod
    def text_is_present(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()

    @field_validator("source_sha256")
    @classmethod
    def digest_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="source_sha256")

    @field_validator("observed_at_utc", "available_at_utc")
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, field=info.field_name)

    @model_validator(mode="after")
    def timing_is_coherent(self) -> Self:
        if self.observed_at_utc > self.available_at_utc:
            raise ValueError("information cannot be available before it was observed")
        return self


class PlayerInactivityInformation(LockModel):
    """Cutoff-safe C6 source facts for one player in the forecast matchup."""

    player_id: str
    coverage: InactivityCoverageAssertion
    candidates: tuple[InactivityMatchCandidate, ...]

    @field_validator("player_id")
    @classmethod
    def player_identity_is_present(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("player_id must not be empty")
        return normalized

    @model_validator(mode="after")
    def facts_resolve_the_declared_player(self) -> Self:
        if self.coverage.canonical_player_id != self.player_id:
            raise ValueError("C6 coverage resolves a different player")
        if any(candidate.player_id != self.player_id for candidate in self.candidates):
            raise ValueError("C6 candidates must belong to the declared player")
        identities = tuple(
            (candidate.tour, candidate.match_id, candidate.available_at_utc)
            for candidate in self.candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("C6 candidate revisions must be unique")
        return self


class InformationBundle(LockModel):
    bundle_id: str
    scenario_id: str
    information_cutoff_utc: datetime
    items: tuple[InformationItem, ...] = ()
    player_inactivity: tuple[PlayerInactivityInformation, ...] = ()
    missing_current_conditions: tuple[str, ...] = ()

    @field_validator("bundle_id", "scenario_id")
    @classmethod
    def identity_is_present(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()

    @field_validator("information_cutoff_utc")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="information_cutoff_utc")

    @model_validator(mode="after")
    def contains_only_available_information(self) -> Self:
        if any(item.available_at_utc >= self.information_cutoff_utc for item in self.items):
            raise ValueError("information bundle contains an item not available before its cutoff")
        identities = tuple(
            (item.source_id, item.source_sha256, item.summary) for item in self.items
        )
        if len(identities) != len(set(identities)):
            raise ValueError("information items must be unique")
        player_ids = tuple(item.player_id for item in self.player_inactivity)
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("C6 information must contain at most one entry per player")
        if any(
            item.coverage.asserted_at_utc >= self.information_cutoff_utc
            or any(
                candidate.available_at_utc >= self.information_cutoff_utc
                for candidate in item.candidates
            )
            for item in self.player_inactivity
        ):
            raise ValueError("C6 information contains facts unavailable at the bundle cutoff")
        return self


class CodeProvenance(LockModel):
    commit: str
    dirty: bool
    diff_sha256: str | None
    fingerprint_version: Literal["legacy-combined/v1", "complete-git-state/v2"] = (
        "legacy-combined/v1"
    )
    staged_sha256: str | None = None
    unstaged_sha256: str | None = None
    untracked_sha256: str | None = None
    relevant_untracked_files: tuple[str, ...] = ()

    @field_validator("commit")
    @classmethod
    def commit_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("commit must not be empty")
        return value.strip()

    @field_validator("diff_sha256", "staged_sha256", "unstaged_sha256", "untracked_sha256")
    @classmethod
    def diff_digest_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field="diff_sha256")

    @model_validator(mode="after")
    def dirty_state_is_complete(self) -> Self:
        if self.dirty != (self.diff_sha256 is not None):
            raise ValueError("dirty code provenance requires exactly one diff_sha256")
        component_hashes = (self.staged_sha256, self.unstaged_sha256, self.untracked_sha256)
        if self.fingerprint_version == "legacy-combined/v1":
            if any(item is not None for item in component_hashes) or self.relevant_untracked_files:
                raise ValueError("legacy code provenance cannot contain complete-state fields")
        elif any(item is None for item in component_hashes):
            raise ValueError("complete code provenance requires staged/unstaged/untracked digests")
        if not self.dirty and self.relevant_untracked_files:
            raise ValueError("clean code provenance cannot list untracked files")
        return self


class SourceManifestProvenance(LockModel):
    manifest: SourceManifest
    manifest_sha256: str

    @field_validator("manifest_sha256")
    @classmethod
    def digest_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="manifest_sha256")

    @model_validator(mode="after")
    def digest_matches_manifest(self) -> Self:
        observed = sha256_json(self.manifest.model_dump(mode="json"))
        if observed != self.manifest_sha256:
            raise ValueError("source manifest digest does not match its content")
        return self

    @classmethod
    def from_manifest(cls, manifest: SourceManifest) -> SourceManifestProvenance:
        content = manifest.model_dump(mode="json")
        return cls(manifest=manifest, manifest_sha256=sha256_json(content))


class SettlementPolicyRecord(LockModel):
    version: str
    comparison_tie_is_no: bool
    walkover_voids_all: bool
    allow_policy_blocked: bool
    description: str


class SerializedProp(LockModel):
    """Stable recursive representation of an atomic or compound prop."""

    node: Literal["atomic", "and", "or"]
    kind: str
    subject_ids: tuple[str, ...] = ()
    operator: str | None = None
    threshold: float | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    original_text: str = ""
    settlement_policy_version: str | None = None
    children: tuple[SerializedProp, ...] = ()

    @model_validator(mode="after")
    def node_is_coherent(self) -> Self:
        if self.node == "atomic":
            if self.children or self.settlement_policy_version is None:
                raise ValueError("atomic props require a policy version and no children")
        elif not self.children:
            raise ValueError("compound props require children")
        elif any((self.subject_ids, self.operator, self.threshold is not None, self.scope)):
            raise ValueError("compound props cannot contain atomic fields")
        return self

    @property
    def prop_id(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def serialize_prop(prop: PropSpec | BooleanCompositeSpec) -> SerializedProp:
    if isinstance(prop, PropSpec):
        return SerializedProp(
            node="atomic",
            kind=prop.kind,
            subject_ids=prop.subject_ids,
            operator=None if prop.operator is None else prop.operator.value,
            threshold=prop.threshold,
            scope=prop.scope,
            original_text=prop.original_text,
            settlement_policy_version=prop.settlement_policy_version,
        )
    node: Literal["and", "or"] = "and" if prop.kind == "AND" else "or"
    return SerializedProp(
        node=node,
        kind=prop.kind,
        children=tuple(serialize_prop(child) for child in prop.exprs),
    )


def deserialize_prop(record: SerializedProp) -> PropSpec | BooleanCompositeSpec:
    from tennis_model.props.settlement import ComparisonOperator

    if record.node == "atomic":
        operator = None if record.operator is None else ComparisonOperator(record.operator)
        return PropSpec(
            kind=record.kind,
            subject_ids=record.subject_ids,
            operator=operator,
            threshold=record.threshold,
            scope=record.scope,
            original_text=record.original_text,
            settlement_policy_version=record.settlement_policy_version or "",
        )
    return BooleanCompositeSpec(
        kind=record.kind,
        exprs=tuple(deserialize_prop(child) for child in record.children),
    )


class PropEstimateRecord(LockModel):
    prop_id: str
    prop: SerializedProp
    probability_raw: float | None = Field(default=None, ge=0, le=1)
    probability_settled: float = Field(ge=0, le=1)
    model_probability_raw: float | None = Field(default=None, ge=0, le=1)
    model_probability_integer: int | None = Field(default=None, ge=0, le=100)
    platform_submission_integer: int | None = Field(default=None, ge=1, le=99)
    platform_submission_policy_version: str | None = None
    submitted_integer: int | None = Field(default=None, ge=1, le=99)
    yes_paths: int = Field(ge=0)
    no_paths: int = Field(ge=0)
    void_paths: int = Field(ge=0)
    unresolved_paths: int = Field(ge=0)
    settled_paths: int = Field(ge=0)
    total_paths: int = Field(ge=0)
    mc_standard_error: float = Field(ge=0)
    sensitivity_low: float | None = Field(default=None, ge=0, le=1)
    sensitivity_high: float | None = Field(default=None, ge=0, le=1)
    display_policy_version: str | None = None
    data_grade: Literal["A", "B", "C"] = "A"
    policy_status: Literal["evaluated", "partially_unresolved", "blocked"]
    policy_issue: str | None = None
    support_status: PropSupportStatus = PropSupportStatus.SUPPORTED
    support_reason_code: str | None = None
    support_detail: str | None = None
    submission_rounding_policy_version: str | None = (
        "nearest-percent-clamp-1-99/v1"
    )
    model_rounding_policy_version: str | None = None
    mc_policy_version: str | None = None
    mc_confidence_level: float | None = Field(default=None, gt=0, lt=1)
    mc_confidence_sequence_method: str | None = None
    mc_confidence_sequence_lower: float | None = Field(default=None, ge=0, le=1)
    mc_confidence_sequence_upper: float | None = Field(default=None, ge=0, le=1)
    mc_stopping_status: MCStoppingStatus | None = None
    final_cumulative_path_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def values_are_coherent(self) -> Self:
        if self.prop_id != self.prop.prop_id:
            raise ValueError("prop_id does not match serialized prop")
        if self.yes_paths + self.no_paths != self.settled_paths:
            raise ValueError("settled path counts are inconsistent")
        if self.settled_paths + self.void_paths + self.unresolved_paths != self.total_paths:
            raise ValueError("prop path counts do not partition the simulation")
        expected_status = (
            "blocked"
            if self.unresolved_paths and not self.settled_paths
            else "partially_unresolved"
            if self.unresolved_paths
            else "evaluated"
        )
        if self.policy_status != expected_status:
            raise ValueError("policy status does not match unresolved/settled path counts")
        if (self.policy_status == "evaluated") != (self.policy_issue is None):
            raise ValueError("policy issue exists exactly for unresolved policy states")
        supported = self.support_status is PropSupportStatus.SUPPORTED
        if supported != (self.support_reason_code is None and self.support_detail is None):
            raise ValueError("prop support status and disable reason are inconsistent")
        if not supported and self.submitted_integer is not None:
            raise ValueError("disabled or unavailable props cannot have an integer submission")
        if (self.platform_submission_integer is None) != (
            self.platform_submission_policy_version is None
        ):
            raise ValueError("platform integer and platform policy must be supplied together")
        if (self.sensitivity_low is None) != (self.sensitivity_high is None):
            raise ValueError("prop sensitivity bounds must be supplied as a pair")
        if (
            self.sensitivity_low is not None
            and self.sensitivity_high is not None
            and self.sensitivity_low > self.sensitivity_high
        ):
            raise ValueError("prop sensitivity bounds are reversed")
        if self.display_policy_version is not None and not self.display_policy_version.strip():
            raise ValueError("display policy version must not be blank")
        adaptive_fields = (
            self.model_rounding_policy_version,
            self.mc_policy_version,
            self.mc_confidence_level,
            self.mc_confidence_sequence_method,
            self.mc_stopping_status,
            self.final_cumulative_path_count,
        )
        adaptive = self.mc_policy_version is not None
        if adaptive != all(value is not None for value in adaptive_fields):
            raise ValueError("adaptive Monte Carlo metadata must be complete or absent")
        if adaptive:
            if self.mc_policy_version != ADAPTIVE_MC_POLICY_VERSION:
                raise ValueError("unknown adaptive Monte Carlo policy")
            if self.model_rounding_policy_version != MODEL_ROUNDING_POLICY_VERSION:
                raise ValueError("adaptive record uses an unknown model rounding policy")
            if self.mc_confidence_sequence_method != BERNOULLI_CS_METHOD_VERSION:
                raise ValueError("adaptive record uses an unknown confidence sequence")
            if not isclose(
                cast(float, self.mc_confidence_level), 0.99, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("adaptive record must use a 99% confidence sequence")
            if self.final_cumulative_path_count != self.total_paths:
                raise ValueError("final cumulative path count must equal the joint batch size")
            if self.submitted_integer is not None:
                raise ValueError("adaptive records cannot conflate legacy submitted integers")
            if self.submission_rounding_policy_version is not None:
                raise ValueError(
                    "adaptive records cannot retain a legacy submission rounding policy"
                )
            if self.settled_paths == 0:
                if any(
                    value is not None
                    for value in (
                        self.model_probability_raw,
                        self.model_probability_integer,
                        self.probability_raw,
                        self.mc_confidence_sequence_lower,
                        self.mc_confidence_sequence_upper,
                    )
                ):
                    raise ValueError(
                        "zero-settled props have no model estimate or confidence sequence"
                    )
                if self.mc_stopping_status is not MCStoppingStatus.UNAVAILABLE:
                    raise ValueError("zero-settled final props must remain unavailable")
                if self.support_status is not PropSupportStatus.DATA_UNAVAILABLE:
                    raise ValueError("zero-settled final props require a data-unavailable gate")
            else:
                expected_raw = self.yes_paths / self.settled_paths
                if (
                    self.model_probability_raw != expected_raw
                    or self.probability_raw != expected_raw
                ):
                    raise ValueError("model raw probability must equal Yes / settled paths")
                if self.model_probability_integer != model_probability_integer(expected_raw):
                    raise ValueError("model integer does not match the centralized rounding policy")
                if (
                    self.mc_confidence_sequence_lower is None
                    or self.mc_confidence_sequence_upper is None
                    or not self.mc_confidence_sequence_lower
                    <= expected_raw
                    <= self.mc_confidence_sequence_upper
                ):
                    raise ValueError("confidence sequence must contain the raw estimate")
                stable_integer = confidence_interval_model_integer(
                    self.mc_confidence_sequence_lower,
                    self.mc_confidence_sequence_upper,
                )
                if self.mc_stopping_status is MCStoppingStatus.INTEGER_STABLE:
                    if stable_integer != self.model_probability_integer:
                        raise ValueError(
                            "INTEGER_STABLE requires one integer over the full sequence"
                        )
                elif self.mc_stopping_status is MCStoppingStatus.INTEGER_BOUNDARY_SENSITIVE:
                    if stable_integer is not None:
                        raise ValueError("boundary-sensitive sequence already maps to one integer")
                else:
                    raise ValueError("final settled adaptive props require a final stopping status")
        else:
            if self.probability_raw is None:
                raise ValueError("legacy fixed-path records require a raw probability")
            if self.submission_rounding_policy_version != "nearest-percent-clamp-1-99/v1":
                raise ValueError("legacy record must retain its original submission policy")
            if any(
                value is not None
                for value in (
                    self.model_probability_raw,
                    self.model_probability_integer,
                    self.platform_submission_integer,
                    self.platform_submission_policy_version,
                    self.mc_confidence_sequence_lower,
                    self.mc_confidence_sequence_upper,
                )
            ):
                raise ValueError("legacy MC records cannot contain adaptive probability fields")
        return self

    @classmethod
    def from_estimate(
        cls,
        estimate: PropEstimate,
        support: PropSupportDecision | None = None,
        *,
        data_grade: Literal["A", "B", "C"] = "A",
        adaptive_diagnostics: AdaptivePropDiagnostics | None = None,
        platform_submission_policy: PlatformSubmissionPolicy | None = None,
    ) -> PropEstimateRecord:
        prop = serialize_prop(estimate.prop)
        decision = (
            PropSupportDecision(status=PropSupportStatus.SUPPORTED) if support is None else support
        )
        if (
            decision.status is PropSupportStatus.SUPPORTED
            and adaptive_diagnostics is not None
            and adaptive_diagnostics.stopping_status is MCStoppingStatus.UNAVAILABLE
        ):
            decision = PropSupportDecision(
                status=PropSupportStatus.DATA_UNAVAILABLE,
                reason_code="ZERO_SETTLED_PATHS",
                detail="no simulated path settled this prop by the final adaptive checkpoint",
            )
        submitted = None
        if adaptive_diagnostics is None and (
            decision.status is PropSupportStatus.SUPPORTED
            and estimate.settled_paths
            and not estimate.unresolved_paths
        ):
            submitted = min(99, max(1, int(estimate.probability_raw * 100.0 + 0.5)))
        platform_integer = None
        if (
            adaptive_diagnostics is not None
            and platform_submission_policy is not None
            and adaptive_diagnostics.model_probability_integer is not None
            and decision.status is PropSupportStatus.SUPPORTED
            and not estimate.unresolved_paths
        ):
            platform_integer = platform_submission_policy.transform(
                adaptive_diagnostics.model_probability_integer
            )
        platform_policy_version = None
        if platform_integer is not None:
            assert platform_submission_policy is not None
            platform_policy_version = platform_submission_policy.version
        policy_status: Literal["evaluated", "partially_unresolved", "blocked"] = (
            "blocked"
            if estimate.unresolved_paths and not estimate.settled_paths
            else "partially_unresolved"
            if estimate.unresolved_paths
            else "evaluated"
        )
        return cls(
            prop_id=prop.prop_id,
            prop=prop,
            probability_raw=(
                estimate.probability_raw
                if adaptive_diagnostics is None
                or adaptive_diagnostics.model_probability_raw is not None
                else None
            ),
            probability_settled=estimate.probability_settled,
            model_probability_raw=(
                None
                if adaptive_diagnostics is None
                else adaptive_diagnostics.model_probability_raw
            ),
            model_probability_integer=(
                None
                if adaptive_diagnostics is None
                else adaptive_diagnostics.model_probability_integer
            ),
            platform_submission_integer=platform_integer,
            platform_submission_policy_version=platform_policy_version,
            submitted_integer=submitted,
            submission_rounding_policy_version=(
                "nearest-percent-clamp-1-99/v1"
                if adaptive_diagnostics is None
                else None
            ),
            yes_paths=estimate.yes_paths,
            no_paths=estimate.no_paths,
            void_paths=estimate.void_paths,
            unresolved_paths=estimate.unresolved_paths,
            settled_paths=estimate.settled_paths,
            total_paths=estimate.total_paths,
            mc_standard_error=estimate.mc_standard_error,
            sensitivity_low=(
                estimate.sensitivity_low if estimate.display_policy_version is not None else None
            ),
            sensitivity_high=(
                estimate.sensitivity_high if estimate.display_policy_version is not None else None
            ),
            display_policy_version=estimate.display_policy_version,
            data_grade=data_grade,
            policy_status=policy_status,
            policy_issue=(
                None
                if policy_status == "evaluated"
                else "settlement semantics unresolved on one or more simulated paths"
            ),
            support_status=decision.status,
            support_reason_code=decision.reason_code,
            support_detail=decision.detail,
            model_rounding_policy_version=(
                None if adaptive_diagnostics is None else MODEL_ROUNDING_POLICY_VERSION
            ),
            mc_policy_version=(
                None if adaptive_diagnostics is None else ADAPTIVE_MC_POLICY_VERSION
            ),
            mc_confidence_level=(
                None if adaptive_diagnostics is None else adaptive_diagnostics.confidence_level
            ),
            mc_confidence_sequence_method=(
                None if adaptive_diagnostics is None else BERNOULLI_CS_METHOD_VERSION
            ),
            mc_confidence_sequence_lower=(
                None
                if adaptive_diagnostics is None
                or adaptive_diagnostics.confidence_sequence is None
                else adaptive_diagnostics.confidence_sequence.lower
            ),
            mc_confidence_sequence_upper=(
                None
                if adaptive_diagnostics is None
                or adaptive_diagnostics.confidence_sequence is None
                else adaptive_diagnostics.confidence_sequence.upper
            ),
            mc_stopping_status=(
                None if adaptive_diagnostics is None else adaptive_diagnostics.stopping_status
            ),
            final_cumulative_path_count=(
                None
                if adaptive_diagnostics is None
                else adaptive_diagnostics.final_cumulative_path_count
            ),
        )


class PropGateRecord(LockModel):
    prop_id: str
    prop: SerializedProp
    support_status: PropSupportStatus
    reason_code: str
    detail: str

    @model_validator(mode="after")
    def gate_is_disabled_and_identified(self) -> Self:
        if self.prop_id != self.prop.prop_id:
            raise ValueError("gated prop ID does not match serialized prop")
        if self.support_status is PropSupportStatus.SUPPORTED:
            raise ValueError("supported props do not belong in the disabled gate list")
        if not self.reason_code.strip() or not self.detail.strip():
            raise ValueError("gated prop requires a structured reason")
        return self


class PrimitiveSummary(LockModel):
    component: ServeComponent
    map_mean: float = Field(gt=0, lt=1)
    linear_predictor_sd: float = Field(ge=0)
    predictive_concentration: float = Field(gt=0)
    weighted_trials: float | None = Field(default=None, ge=0)
    effective_matches: float | None = Field(default=None, ge=0)
    information_equivalent_trials: float | None = Field(default=None, ge=0)
    sparse_warning: bool


class ServingDirectionSummary(LockModel):
    server_id: str
    receiver_id: str
    primitives: tuple[PrimitiveSummary, ...]
    first_serve_win: float = Field(ge=0, le=1)
    second_serve_win: float = Field(ge=0, le=1)
    service_point_win: float = Field(ge=0, le=1)
    analytic_hold_probability: float = Field(ge=0, le=1)
    ace_rate_per_service_point: float = Field(ge=0, le=1)
    double_fault_rate_per_service_point: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def primitives_are_complete(self) -> Self:
        if tuple(item.component for item in self.primitives) != tuple(ServeComponent):
            raise ValueError("direction summary must contain F/A/Q1/D/Q2")
        return self


class ExactScoreProbability(LockModel):
    winner_id: str
    winner_sets: int = Field(ge=2, le=3)
    loser_sets: int = Field(ge=0, le=2)
    probability: float = Field(ge=0, le=1)


class PlayerSimulationSummary(LockModel):
    player_id: str
    match_win_probability: float = Field(ge=0, le=1)
    expected_service_games: float = Field(ge=0)
    expected_breaks: float = Field(ge=0)
    expected_breaks_conceded: float = Field(ge=0)
    expected_aces: float = Field(ge=0)
    expected_double_faults: float = Field(ge=0)
    retirement_probability: float | None = Field(default=None, ge=0, le=1)


class MatchSimulationSummary(LockModel):
    players: tuple[PlayerSimulationSummary, PlayerSimulationSummary]
    exact_scores: tuple[ExactScoreProbability, ...]
    any_tiebreak_probability: float = Field(ge=0, le=1)
    deciding_set_probability: float = Field(ge=0, le=1)
    expected_total_games: float = Field(ge=0)
    total_games_quantiles: tuple[float, float, float]
    expected_total_breaks: float = Field(ge=0)
    completion_probability: float = Field(ge=0, le=1)
    retirement_probability: float | None = Field(default=None, ge=0, le=1)
    walkover_probability: float = Field(ge=0, le=1)
    duration: DurationSimulationSummary | None = None


class DurationSimulationSummary(LockModel):
    expected_minutes: float = Field(ge=1)
    quantiles: tuple[float, float, float]
    data_grade: Literal["A", "B", "C"]
    artifact_id: str
    current_event_effect_minutes: float | None = None
    display_policy_version: str
    display_boundary_sensitive: bool

    @field_validator("artifact_id")
    @classmethod
    def artifact_digest_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="duration artifact_id")

    @model_validator(mode="after")
    def duration_summary_is_coherent(self) -> Self:
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("duration quantiles must be ordered")
        if any(value < 1 for value in self.quantiles):
            raise ValueError("duration quantiles must be at least one minute")
        if not self.display_policy_version.strip():
            raise ValueError("duration display-policy version must not be blank")
        return self


class AdaptiveMCPolicyRecord(LockModel):
    version: Literal["adaptive_mc_cs_v1"] = "adaptive_mc_cs_v1"
    checkpoints: tuple[int, ...]
    confidence_level: float = Field(gt=0, lt=1)
    confidence_sequence_method: Literal["beta-binomial-jeffreys-mixture/v1"] = (
        "beta-binomial-jeffreys-mixture/v1"
    )
    beta_prior_a: float = Field(gt=0)
    beta_prior_b: float = Field(gt=0)
    model_rounding_policy_version: Literal["nearest-percent-half-up-endpoints/v1"] = (
        "nearest-percent-half-up-endpoints/v1"
    )

    @model_validator(mode="after")
    def policy_is_frozen(self) -> Self:
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("adaptive checkpoints must be positive and strictly increasing")
        if any(value <= 0 for value in self.checkpoints):
            raise ValueError("adaptive checkpoints must be positive")
        if not isclose(self.confidence_level, 0.99, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("adaptive_mc_cs_v1 requires 99% confidence")
        if not (
            isclose(self.beta_prior_a, 0.5, rel_tol=0.0, abs_tol=1e-15)
            and isclose(self.beta_prior_b, 0.5, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError("adaptive_mc_cs_v1 requires a Jeffreys beta mixture")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class PathCountPolicyRecord(LockModel):
    version: Literal["frozen-v1.0", "explicit-development-test"]
    standard_paths: int = Field(gt=0)
    escalated_paths: int = Field(gt=0)
    minimum_settled_paths: int = Field(ge=0)
    extreme_probability: float = Field(gt=0, lt=0.5)
    integer_boundary_window: float = Field(ge=0)
    integer_boundary_standard_errors: float = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_ordered(self) -> Self:
        if self.escalated_paths < self.standard_paths:
            raise ValueError("escalated path count cannot be below the standard count")
        if self.minimum_settled_paths > self.standard_paths:
            raise ValueError("minimum settled paths cannot exceed the standard count")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class SimulationProvenance(LockModel):
    rng_algorithm: Literal["PCG64"] = "PCG64"
    seed_id: str
    requested_paths: int = Field(gt=0)
    actual_paths: int = Field(gt=0)
    trace_level: Literal["summary", "points"]
    first_server_id: str | None
    path_count_mode: Literal["production", "development", "test"]
    escalated: bool
    escalation_reasons: tuple[str, ...] = ()
    performance_dependence_mode: Literal["independent"] = "independent"
    path_count_policy: PathCountPolicyRecord | AdaptiveMCPolicyRecord
    inspected_path_counts: tuple[int, ...] = ()
    platform_submission_policy_version: str | None = None
    retirement_rng_stream_version: (
        Literal["seedsequence-retirement-parameters-boundaries/v1"] | None
    ) = None
    ordinary_termination_before_retirement_version: (
        Literal["ordinary-terminal-bypass-before-b6/v1"] | None
    ) = None
    duration_rng_stream_version: (
        Literal["seedsequence-duration-parameters-residual/v1"] | None
    ) = None
    duration_display_policy_version: str | None = None
    seed_policy_version: str = "production-seed-policy/v1"
    chunk_size: int = Field(default=1, gt=0)
    thread_count: int = Field(default=1, gt=0)
    process_count: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def adaptive_inspection_is_coherent(self) -> Self:
        if isinstance(self.path_count_policy, AdaptiveMCPolicyRecord):
            expected_prefix = self.path_count_policy.checkpoints[: len(self.inspected_path_counts)]
            if not self.inspected_path_counts or self.inspected_path_counts != expected_prefix:
                raise ValueError(
                    "adaptive inspected path counts must be a nonempty checkpoint prefix"
                )
            if self.inspected_path_counts[-1] != self.actual_paths:
                raise ValueError("last adaptive inspection must equal actual paths")
            if self.requested_paths != self.path_count_policy.checkpoints[-1]:
                raise ValueError("adaptive requested paths must record the frozen maximum")
            if self.escalated != (self.actual_paths > self.path_count_policy.checkpoints[0]):
                raise ValueError("adaptive escalation flag does not match inspected checkpoints")
        elif self.inspected_path_counts:
            raise ValueError("legacy fixed-path simulations cannot claim adaptive inspections")
        if (
            not isinstance(self.path_count_policy, AdaptiveMCPolicyRecord)
            and self.platform_submission_policy_version is not None
        ):
            raise ValueError("external platform transforms require adaptive MC records")
        return self
class LockRevisionReason(LockModel):
    category: Literal[
        "initial",
        "schedule",
        "conditions",
        "health",
        "workload",
        "model_refresh",
        "official_correction",
        "scenario",
        "configuration",
        "other",
    ]
    summary: str
    evidence_source_ids: tuple[str, ...] = ()


class PredictionSnapshot(LockModel):
    """Complete immutable numerical forecast payload; storage adds its content hash."""

    schema_version: Literal[
        "prediction-lock/v1",
        "prediction-lock/v2",
        "prediction-lock/v3",
        "prediction-lock/v4",
    ] = LOCK_SCHEMA_VERSION
    identity_schema_version: Literal["legacy-forecast-state/v1", "canonical-match-identity/v2"] = (
        "legacy-forecast-state/v1"
    )
    base_lock_id: str
    canonical_match_identity: CanonicalMatchIdentity | None = None
    revision: int = Field(ge=1)
    created_at_utc: datetime
    parent_revision: int | None = Field(default=None, ge=1)
    parent_content_sha256: str | None = None
    revision_reason: LockRevisionReason
    framework_version: Literal["v1.0"]
    settlement_policy: SettlementPolicyRecord
    context: MatchContext
    information: InformationBundle
    source_manifest: SourceManifestProvenance
    code: CodeProvenance
    historical_validation_policy: HistoricalValidationPolicy = POINT_IN_TIME_VINTAGE_POLICY
    historical_time: HistoricalTimeProvenance | None = None
    training_eligibility: HistoricalTrainingEligibilityProvenance | None = None
    runtime: RuntimeFingerprint | None = None
    retained_artifacts: tuple[RetainedArtifactRecord, ...] = ()
    lock_configuration_sha256: str
    match_parameters: MatchParameterRecord
    parameter_summaries: tuple[ServingDirectionSummary, ServingDirectionSummary]
    simulation: SimulationProvenance
    match_summary: MatchSimulationSummary
    prop_estimates: tuple[PropEstimateRecord, ...]
    prop_gates: tuple[PropGateRecord, ...] = ()
    retirement_model_artifact_id: str | None = None
    inactivity_configuration_artifact_id: str | None = None
    duration_model_artifact_id: str | None = None
    warnings: tuple[str, ...] = ()
    validation_checks: tuple[str, ...] = ()

    @field_validator("created_at_utc")
    @classmethod
    def created_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, field="created_at_utc")

    @field_validator(
        "parent_content_sha256",
        "retirement_model_artifact_id",
        "inactivity_configuration_artifact_id",
        "duration_model_artifact_id",
        "lock_configuration_sha256",
    )
    @classmethod
    def optional_digests_are_valid(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else require_sha256(value, field=info.field_name)

    @model_validator(mode="after")
    def lock_is_coherent(self) -> Self:
        if self.schema_version in {"prediction-lock/v3", "prediction-lock/v4"}:
            if self.identity_schema_version != "canonical-match-identity/v2":
                raise ValueError("v3 prediction locks require canonical identity schema v2")
            if self.canonical_match_identity is None:
                raise ValueError("v3 prediction locks require canonical match identity")
            if self.base_lock_id != self.canonical_match_identity.base_lock_id:
                raise ValueError("base lock ID differs from canonical match identity")
            if self.canonical_match_identity.tour is not self.context.tour:
                raise ValueError("canonical identity tour differs from match context")
            if set(self.canonical_match_identity.participant_ids) != {
                self.context.player_a_id,
                self.context.player_b_id,
            }:
                raise ValueError("canonical identity participants differ from match context")
            if self.historical_time is None or self.runtime is None:
                raise ValueError("v3 locks require historical-time and runtime provenance")
            if (
                self.historical_time.information_cutoff_utc != self.context.information_cutoff_utc
                or self.historical_time.training_data_cutoff_utc
                != self.match_parameters.snapshot.data_cutoff_utc
                or self.historical_time.artifact_created_at_utc != self.created_at_utc
            ):
                raise ValueError("v3 timestamp provenance differs from lock/snapshot timestamps")
            if (
                self.historical_time.historical_validation_policy
                != self.historical_validation_policy
            ):
                raise ValueError("v3 historical data policy provenance differs")
            if self.training_eligibility is not None:
                if self.training_eligibility.tour is not self.context.tour:
                    raise ValueError("training eligibility tour differs from match context")
                if (
                    self.training_eligibility.source_manifest_sha256
                    != self.source_manifest.manifest_sha256
                ):
                    raise ValueError("training eligibility source manifest differs")
                if self.training_eligibility.verified_at_utc >= self.context.information_cutoff_utc:
                    raise ValueError("training eligibility was not verified before cutoff")
                if self.training_eligibility.warning is not None and (
                    self.training_eligibility.warning not in self.warnings
                ):
                    raise ValueError("training coverage warning is absent from lock warnings")
            required_artifact_kinds = {
                "source_snapshot",
                "normalized_snapshot",
                "component_counts",
                "component_fit",
                "retirement_fit",
                "inactivity_config",
                "model_config",
                "settlement_policy",
                "code_archive",
            }
            if self.schema_version == "prediction-lock/v4":
                required_artifact_kinds.add("duration_fit")
            if {item.kind for item in self.retained_artifacts} != required_artifact_kinds:
                raise ValueError("v3 locks require one retained artifact for every required kind")
            if len({item.artifact_id for item in self.retained_artifacts}) != len(
                self.retained_artifacts
            ):
                raise ValueError("retained artifact IDs must be unique")
            if self.code.fingerprint_version != "complete-git-state/v2":
                raise ValueError(
                    "v3 locks require a complete staged/unstaged/untracked fingerprint"
                )
        elif (
            self.canonical_match_identity is not None
            or self.identity_schema_version != "legacy-forecast-state/v1"
        ):
            raise ValueError("legacy locks cannot silently adopt v3 canonical identity")
        if self.revision == 1:
            if self.parent_revision is not None or self.parent_content_sha256 is not None:
                raise ValueError("initial lock cannot have a parent")
            if self.revision_reason.category != "initial":
                raise ValueError("initial lock requires an initial revision reason")
        elif self.parent_revision != self.revision - 1 or self.parent_content_sha256 is None:
            raise ValueError("lock revisions require the immediately preceding parent")
        if self.framework_version != self.match_parameters.snapshot.framework_version:
            raise ValueError("lock framework version differs from model snapshot")
        if self.context != self.match_parameters.context:
            raise ValueError("lock context differs from match parameters")
        if self.information.information_cutoff_utc != self.context.information_cutoff_utc:
            raise ValueError("information and match cutoffs differ")
        if self.information.scenario_id != self.context.information_scenario_id:
            raise ValueError("information and match scenarios differ")
        if self.simulation.actual_paths != next(
            (item.total_paths for item in self.prop_estimates), self.simulation.actual_paths
        ):
            raise ValueError("prop estimates do not match the stored simulation path count")
        if any(item.total_paths != self.simulation.actual_paths for item in self.prop_estimates):
            raise ValueError("all prop estimates must derive from the same joint paths")
        adaptive_mc = isinstance(self.simulation.path_count_policy, AdaptiveMCPolicyRecord)
        if adaptive_mc:
            if any(
                item.mc_policy_version != ADAPTIVE_MC_POLICY_VERSION
                for item in self.prop_estimates
            ):
                raise ValueError("adaptive simulation requires adaptive metadata on every estimate")
        elif any(item.mc_policy_version is not None for item in self.prop_estimates):
            raise ValueError("legacy fixed-path simulations cannot contain adaptive estimates")
        prop_ids = tuple(item.prop_id for item in self.prop_estimates) + tuple(
            item.prop_id for item in self.prop_gates
        )
        if not prop_ids or len(prop_ids) != len(set(prop_ids)):
            raise ValueError("lock props must be nonempty and unique across estimates/gates")
        if self.simulation.performance_dependence_mode != "independent":
            raise ValueError("frozen v1.0 locks must use independent performance draws")
        if self.schema_version == "prediction-lock/v1":
            if (
                self.retirement_model_artifact_id is not None
                or self.inactivity_configuration_artifact_id is not None
                or self.match_parameters.retirement is not None
                or self.match_parameters.inactivity is not None
            ):
                raise ValueError("v1 prediction locks cannot contain B6/C6 parameters")
        else:
            if self.match_parameters.retirement is None or self.match_parameters.inactivity is None:
                raise ValueError("v2 prediction locks require complete B6/C6 parameters")
            if (
                self.retirement_model_artifact_id != self.match_parameters.retirement.artifact_id
                or self.inactivity_configuration_artifact_id
                != self.match_parameters.inactivity.configuration_artifact_id
            ):
                raise ValueError("lock B6/C6 artifact IDs differ from match parameters")
            if (
                self.simulation.retirement_rng_stream_version is None
                or self.simulation.ordinary_termination_before_retirement_version is None
            ):
                raise ValueError("v2 lock lacks B6 simulation ordering/RNG provenance")
            player_retirement_probabilities = tuple(
                item.retirement_probability for item in self.match_summary.players
            )
            if any(item is None for item in player_retirement_probabilities):
                raise ValueError("v2 lock lacks player-level B6 incidence probabilities")
            assert self.match_summary.retirement_probability is not None
            if not isclose(
                sum(item for item in player_retirement_probabilities if item is not None),
                self.match_summary.retirement_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("player-level B6 incidence does not sum to match retirement")
        duration_parameters = getattr(self.match_parameters, "duration", None)
        if self.schema_version == "prediction-lock/v4":
            if duration_parameters is None:
                raise ValueError("v4 prediction locks require duration match parameters")
            if self.duration_model_artifact_id != duration_parameters.artifact_id:
                raise ValueError("lock duration artifact differs from match parameters")
            if self.match_summary.duration is None:
                raise ValueError("v4 prediction locks require a duration simulation summary")
            if self.match_summary.duration.artifact_id != self.duration_model_artifact_id:
                raise ValueError("duration summary references another fit artifact")
            retained_duration = tuple(
                item for item in self.retained_artifacts if item.kind == "duration_fit"
            )
            if (
                len(retained_duration) != 1
                or retained_duration[0].artifact_id != self.duration_model_artifact_id
            ):
                raise ValueError(
                    "v4 retained duration fit must identify the lock duration artifact"
                )
            if (
                self.simulation.duration_rng_stream_version is None
                or self.simulation.duration_display_policy_version is None
            ):
                raise ValueError("v4 prediction lock lacks duration RNG/display provenance")
            if (
                self.match_summary.duration.display_policy_version
                != self.simulation.duration_display_policy_version
            ):
                raise ValueError("duration summary and simulation use different display policies")
        elif (
            duration_parameters is not None
            or self.duration_model_artifact_id is not None
            or self.match_summary.duration is not None
            or self.simulation.duration_rng_stream_version is not None
            or self.simulation.duration_display_policy_version is not None
        ):
            raise ValueError("pre-v4 locks cannot contain duration-model state")
        configuration: dict[str, Any] = {
            "model_config_sha256": self.match_parameters.snapshot.config_hash,
            "settlement_policy": self.settlement_policy.model_dump(mode="json"),
            "information_scenario_id": self.information.scenario_id,
            "trace_level": self.simulation.trace_level,
            "first_server_id": self.simulation.first_server_id,
            "path_count_mode": self.simulation.path_count_mode,
            "path_count_policy": self.simulation.path_count_policy.model_dump(mode="json"),
            "performance_dependence_mode": self.simulation.performance_dependence_mode,
            "historical_validation_policy": self.historical_validation_policy.model_dump(
                mode="json"
            ),
        }
        platform_policy_versions = {
            item.platform_submission_policy_version
            for item in self.prop_estimates
            if item.platform_submission_policy_version is not None
        }
        if len(platform_policy_versions) > 1:
            raise ValueError("one lock cannot mix external platform submission policies")
        if platform_policy_versions and platform_policy_versions != {
            self.simulation.platform_submission_policy_version
        }:
            raise ValueError("prop and simulation platform submission policies differ")
        if self.simulation.platform_submission_policy_version is not None:
            configuration["platform_submission_policy_version"] = (
                self.simulation.platform_submission_policy_version
            )
        if self.training_eligibility is not None:
            configuration["training_eligibility"] = self.training_eligibility.model_dump(
                mode="json"
            )
        if self.schema_version != "prediction-lock/v1":
            assert self.match_parameters.retirement is not None
            assert self.match_parameters.inactivity is not None
            configuration.update(
                {
                    "retirement_artifact_id": self.match_parameters.retirement.artifact_id,
                    "retirement_scenario_mixtures": [
                        item.model_dump(mode="json")
                        for item in self.match_parameters.retirement.scenario_mixtures
                    ],
                    "inactivity_configuration_artifact_id": (
                        self.match_parameters.inactivity.configuration_artifact_id
                    ),
                    "inactivity_record_hashes": [
                        item.sha256 for item in self.match_parameters.inactivity.records
                    ],
                    "ordinary_termination_before_retirement_version": (
                        self.simulation.ordinary_termination_before_retirement_version
                    ),
                }
            )
        if self.schema_version == "prediction-lock/v4":
            configuration.update(
                {
                    "duration_artifact_id": self.duration_model_artifact_id,
                    "duration_display_policy_version": (
                        self.simulation.duration_display_policy_version
                    ),
                    "duration_rng_stream_version": self.simulation.duration_rng_stream_version,
                }
            )
        if self.lock_configuration_sha256 != sha256_json(configuration):
            raise ValueError("lock configuration hash does not match configuration content")
        return self

    @property
    def lock_id(self) -> str:
        return f"{self.base_lock_id}-L{self.revision}"

    @property
    def content_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        simulation = payload.get("simulation")
        adaptive_mc = bool(
            isinstance(simulation, dict)
            and isinstance(simulation.get("path_count_policy"), dict)
            and simulation["path_count_policy"].get("version") == ADAPTIVE_MC_POLICY_VERSION
        )
        if not adaptive_mc:
            if isinstance(simulation, dict):
                simulation.pop("inspected_path_counts", None)
                simulation.pop("platform_submission_policy_version", None)
            estimates = payload.get("prop_estimates")
            if isinstance(estimates, list):
                adaptive_estimate_fields = (
                    "model_probability_raw",
                    "model_probability_integer",
                    "platform_submission_integer",
                    "platform_submission_policy_version",
                    "model_rounding_policy_version",
                    "mc_policy_version",
                    "mc_confidence_level",
                    "mc_confidence_sequence_method",
                    "mc_confidence_sequence_lower",
                    "mc_confidence_sequence_upper",
                    "mc_stopping_status",
                    "final_cumulative_path_count",
                )
                for estimate in estimates:
                    if isinstance(estimate, dict):
                        for field_name in adaptive_estimate_fields:
                            estimate.pop(field_name, None)
        if self.schema_version != "prediction-lock/v4":
            # These fields did not exist in the immutable v1--v3 payload
            # schemas.  Excluding their default values preserves the content
            # identity of already-published locks when they are loaded by the
            # duration-capable code.
            payload.pop("duration_model_artifact_id", None)
            if isinstance(simulation, dict):
                simulation.pop("duration_rng_stream_version", None)
                simulation.pop("duration_display_policy_version", None)
            summary = payload.get("match_summary")
            if isinstance(summary, dict):
                summary.pop("duration", None)
            match_parameters = payload.get("match_parameters")
            if isinstance(match_parameters, dict):
                match_parameters.pop("duration", None)
                parameter_provenance = match_parameters.get("provenance")
                if isinstance(parameter_provenance, dict):
                    parameter_provenance.pop("duration_artifact_id", None)
                model_snapshot = match_parameters.get("snapshot")
                if isinstance(model_snapshot, dict):
                    model_snapshot.pop("duration_artifact", None)
                    model_snapshot.pop("duration_schema_version", None)
            estimates = payload.get("prop_estimates")
            if isinstance(estimates, list):
                for estimate in estimates:
                    if isinstance(estimate, dict):
                        estimate.pop("sensitivity_low", None)
                        estimate.pop("sensitivity_high", None)
                        estimate.pop("display_policy_version", None)
        return sha256_json(payload)


class StoredPredictionLock(LockModel):
    schema_version: Literal["prediction-lock-envelope/v1"] = "prediction-lock-envelope/v1"
    content_sha256: str
    lock: PredictionSnapshot

    @field_validator("content_sha256")
    @classmethod
    def content_digest_is_valid(cls, value: str) -> str:
        return require_sha256(value, field="content_sha256")

    @model_validator(mode="after")
    def digest_matches_payload(self) -> Self:
        if self.content_sha256 != self.lock.content_sha256:
            raise ValueError("lock content hash does not match its payload")
        return self
