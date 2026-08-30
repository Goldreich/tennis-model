"""Normalize winner/loser source rows into symmetric player-service rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from tennis_model.data.cutoff import availability_from_source_date
from tennis_model.data.validate import SCORE_MISSING, ScoreValidationResult, validate_score
from tennis_model.identity import (
    IdentityValidationError,
    MatchDuplicateKind,
    MatchRecordIdentity,
    detect_match_duplicate_groups,
    make_match_identity,
    make_match_record_identity,
    make_player_identity,
)
from tennis_model.schemas import RawSourceSnapshot, RowDateSemantics

NORMALIZATION_VERSION = "sackmann-player-service-v1.0"

_MISSING_TOKENS = frozenset({"", "NA", "N/A", "NAN", "NULL", "NONE"})
_INTEGER_RE = re.compile(r"^-?\d+$")

_CORE_RAW_COLUMNS = {
    "tourney_id",
    "tourney_level",
    "tourney_name",
    "tourney_date",
    "match_num",
    "round",
    "surface",
    "best_of",
    "score",
    "winner_id",
    "winner_name",
    "winner_hand",
    "loser_id",
    "loser_name",
    "loser_hand",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_ace",
    "w_df",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_ace",
    "l_df",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
}

_STAT_SUFFIXES: Mapping[str, str] = {
    "service_points": "svpt",
    "first_serves_in": "1stIn",
    "first_serve_points_won": "1stWon",
    "second_serve_points_won": "2ndWon",
    "aces": "ace",
    "double_faults": "df",
    "service_games": "SvGms",
    "break_points_saved": "bpSaved",
    "break_points_faced": "bpFaced",
}

_CORE_STAT_FIELDS = frozenset(
    {
        "service_points",
        "first_serves_in",
        "first_serve_points_won",
        "second_serve_points_won",
        "aces",
        "double_faults",
    }
)

_NORMALIZED_COLUMNS = (
    "snapshot_id",
    "snapshot_sha256",
    "source_id",
    "source_url",
    "source_schema_version",
    "retrieved_at_utc",
    "normalization_version",
    "source_row_number",
    "orientation",
    "match_id",
    "source_date",
    "match_date",
    "match_date_source_id",
    "match_date_source_sha256",
    "match_date_crosswalk_id",
    "event_start_date",
    "source_date_semantics",
    "available_at_utc",
    "tour",
    "event",
    "event_year",
    "level",
    "round",
    "surface",
    "indoor",
    "best_of",
    "player_id",
    "opponent_id",
    "player_source_id",
    "opponent_source_id",
    "player_name",
    "opponent_name",
    "player_hand",
    "opponent_hand",
    "score",
    "completed",
    "retirement",
    "walkover",
    "service_points",
    "first_serves_in",
    "first_serve_points_won",
    "second_serve_points_won",
    "aces",
    "double_faults",
    "service_games",
    "break_points_faced",
    "break_points_saved",
    "total_points_won",
    "winners",
    "unforced_errors",
    "duration_minutes",
    "invalid_stat_fields",
    "raw_record_json",
)

_ANOMALY_COLUMNS = (
    "snapshot_id",
    "snapshot_sha256",
    "source_id",
    "source_row_number",
    "match_id",
    "orientation",
    "player_id",
    "scope",
    "severity",
    "code",
    "field",
    "raw_value",
    "raw_record_json",
)


class SackmannSchemaError(ValueError):
    """The raw file does not expose the frozen ingestion schema."""


class SymmetryValidationError(ValueError):
    """Normalized rows do not form reciprocal two-player matches."""


@dataclass(frozen=True, slots=True)
class NormalizedServiceRows:
    """Normalized rows plus every exclusion/warning raised during intake."""

    rows: pd.DataFrame
    anomalies: pd.DataFrame
    raw_row_count: int
    accepted_match_count: int
    normalization_version: str = NORMALIZATION_VERSION


@dataclass(slots=True)
class _Candidate:
    raw: dict[str, Any]
    raw_record_json: str
    source_row_number: int
    match_id: str
    winner_id: str
    loser_id: str
    match_record_identity: MatchRecordIdentity
    source_date: date
    exact_match_date: date | None
    available_at_utc: datetime
    best_of: int
    score: ScoreValidationResult
    fatal_codes: list[tuple[str, str | None, Any]]
    warning_codes: list[tuple[str, str | None, Any]]


def validate_sackmann_columns(raw: pd.DataFrame) -> None:
    """Require stable identity metadata and nullable primitive-stat columns."""

    missing = _CORE_RAW_COLUMNS.difference(raw.columns)
    if missing:
        raise SackmannSchemaError(
            "Sackmann snapshot missing required columns: " + ", ".join(sorted(missing))
        )


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip().upper() in _MISSING_TOKENS


def _optional_text(value: Any) -> str | None:
    return None if _is_missing(value) else str(value).strip()


def _parse_required_date(value: Any) -> date:
    text = _optional_text(value)
    if text is None:
        raise ValueError("missing source date")
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"invalid source date: {text!r}")


def _parse_required_best_of(value: Any) -> int:
    text = _optional_text(value)
    if text is None or not _INTEGER_RE.fullmatch(text):
        raise ValueError("best_of must be an integer")
    best_of = int(text)
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")
    return best_of


def _parse_optional_integer(value: Any) -> tuple[int | None, bool]:
    text = _optional_text(value)
    if text is None:
        return None, False
    if not _INTEGER_RE.fullmatch(text):
        return None, True
    return int(text), False


def _raw_json(row: Mapping[str, Any]) -> str:
    raw_values = {
        str(key): (None if _is_missing(value) else str(value))
        for key, value in row.items()
        if not str(key).startswith("_")
    }
    return json.dumps(raw_values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _anomaly(
    snapshot: RawSourceSnapshot,
    *,
    source_row_number: int,
    raw_record_json: str,
    code: str,
    scope: str,
    severity: str = "error",
    match_id: str | None = None,
    orientation: str | None = None,
    player_id: str | None = None,
    field: str | None = None,
    raw_value: Any = None,
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.source.sha256,
        "snapshot_sha256": snapshot.sha256,
        "source_id": snapshot.source.source_id,
        "source_row_number": source_row_number,
        "match_id": match_id,
        "orientation": orientation,
        "player_id": player_id,
        "scope": scope,
        "severity": severity,
        "code": code,
        "field": field,
        "raw_value": None if _is_missing(raw_value) else str(raw_value),
        "raw_record_json": raw_record_json,
    }


def _candidate_from_raw(
    row: Mapping[str, Any], snapshot: RawSourceSnapshot
) -> tuple[_Candidate | None, list[dict[str, Any]]]:
    source_row_number = int(row.get("_source_row_number", 0))
    raw_record_json = str(row.get("_raw_record_json") or _raw_json(row))
    source = snapshot.source
    anomalies: list[dict[str, Any]] = []

    try:
        winner = make_player_identity(
            source.identity_namespace,
            source.tour,
            _optional_text(row.get("winner_id")),
            display_name=_optional_text(row.get("winner_name")),
        )
        loser = make_player_identity(
            source.identity_namespace,
            source.tour,
            _optional_text(row.get("loser_id")),
            display_name=_optional_text(row.get("loser_name")),
        )
        match = make_match_identity(
            source.identity_namespace,
            source.tour,
            _optional_text(row.get("tourney_id")),
            _optional_text(row.get("match_num")),
            player_a_id=winner.player_id,
            player_b_id=loser.player_id,
        )
    except IdentityValidationError as exc:
        anomalies.append(
            _anomaly(
                snapshot,
                source_row_number=source_row_number,
                raw_record_json=raw_record_json,
                scope="identity",
                code=exc.code.value,
                field=exc.field,
                raw_value=exc.value,
            )
        )
        return None, anomalies

    fatal: list[tuple[str, str | None, Any]] = []
    warnings: list[tuple[str, str | None, Any]] = []
    try:
        source_date = _parse_required_date(row.get("tourney_date"))
    except ValueError:
        fatal.append(("MALFORMED_MATCH_DATE", "tourney_date", row.get("tourney_date")))
        source_date = date.min

    exact_match_date: date | None = None
    if not _is_missing(row.get("_exact_match_date")):
        try:
            exact_match_date = _parse_required_date(row.get("_exact_match_date"))
        except ValueError:
            fatal.append(
                ("MALFORMED_EXACT_MATCH_DATE", "_exact_match_date", row.get("_exact_match_date"))
            )

    try:
        best_of = _parse_required_best_of(row.get("best_of"))
    except ValueError:
        fatal.append(("MALFORMED_BEST_OF", "best_of", row.get("best_of")))
        best_of = 3

    score = validate_score(_optional_text(row.get("score")), best_of=best_of)
    if not score.valid:
        for code in score.anomaly_codes:
            target = warnings if code == SCORE_MISSING else fatal
            target.append((code, "score", row.get("score")))

    derived_available_at_utc = availability_from_source_date(
        source_date, availability_lag_days=source.availability_lag_days
    )
    available_at_utc = max(
        derived_available_at_utc,
        source.source_available_at_utc or derived_available_at_utc,
    )
    fingerprint = hashlib.sha256(raw_record_json.encode("utf-8")).hexdigest()
    record_identity = make_match_record_identity(
        match,
        source_row_ref=str(source_row_number),
        record_fingerprint=fingerprint,
    )
    return (
        _Candidate(
            raw=dict(row),
            raw_record_json=raw_record_json,
            source_row_number=source_row_number,
            match_id=match.match_id,
            winner_id=winner.player_id,
            loser_id=loser.player_id,
            match_record_identity=record_identity,
            source_date=source_date,
            exact_match_date=exact_match_date,
            available_at_utc=available_at_utc,
            best_of=best_of,
            score=score,
            fatal_codes=fatal,
            warning_codes=warnings,
        ),
        anomalies,
    )


def _stat_values(
    candidate: _Candidate, orientation: str
) -> tuple[dict[str, int | None], tuple[str, ...], list[tuple[str, str, Any]]]:
    prefix = "w" if orientation == "winner" else "l"
    values: dict[str, int | None] = {}
    invalid: list[str] = []
    warnings: list[tuple[str, str, Any]] = []
    for normalized_field, suffix in _STAT_SUFFIXES.items():
        raw_field = f"{prefix}_{suffix}"
        parsed, malformed = _parse_optional_integer(candidate.raw.get(raw_field))
        values[normalized_field] = parsed
        if malformed:
            warnings.append(
                (
                    f"MALFORMED_{normalized_field.upper()}",
                    raw_field,
                    candidate.raw.get(raw_field),
                )
            )
            if normalized_field in _CORE_STAT_FIELDS:
                invalid.append(normalized_field)
    return values, tuple(invalid), warnings


def _optional_integer_field(
    raw: Mapping[str, Any], *names: str
) -> tuple[int | None, bool, str | None, Any]:
    for name in names:
        if name in raw:
            value, malformed = _parse_optional_integer(raw.get(name))
            return value, malformed, name, raw.get(name)
    return None, False, None, None


def _normalized_row(
    candidate: _Candidate,
    snapshot: RawSourceSnapshot,
    *,
    orientation: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    is_winner = orientation == "winner"
    player_prefix = "w" if is_winner else "l"
    player_source_field = "winner_id" if is_winner else "loser_id"
    opponent_source_field = "loser_id" if is_winner else "winner_id"
    player_name_field = "winner_name" if is_winner else "loser_name"
    opponent_name_field = "loser_name" if is_winner else "winner_name"
    player_hand_field = "winner_hand" if is_winner else "loser_hand"
    opponent_hand_field = "loser_hand" if is_winner else "winner_hand"
    player_id = candidate.winner_id if is_winner else candidate.loser_id
    opponent_id = candidate.loser_id if is_winner else candidate.winner_id
    stats, invalid_fields, stat_warnings = _stat_values(candidate, orientation)
    source = snapshot.source
    anomalies = [
        _anomaly(
            snapshot,
            source_row_number=candidate.source_row_number,
            raw_record_json=candidate.raw_record_json,
            match_id=candidate.match_id,
            orientation=orientation,
            player_id=player_id,
            scope="raw_stat",
            severity="warning",
            code=code,
            field=field,
            raw_value=value,
        )
        for code, field, value in stat_warnings
    ]

    total_points_won, total_points_malformed, total_points_field, total_points_raw = (
        _optional_integer_field(
            candidate.raw,
            f"{player_prefix}_pts",
            f"{player_prefix}_total_points_won",
        )
    )
    if total_points_malformed:
        anomalies.append(
            _anomaly(
                snapshot,
                source_row_number=candidate.source_row_number,
                raw_record_json=candidate.raw_record_json,
                match_id=candidate.match_id,
                orientation=orientation,
                player_id=player_id,
                scope="raw_stat",
                severity="warning",
                code="MALFORMED_TOTAL_POINTS_WON",
                field=total_points_field,
                raw_value=total_points_raw,
            )
        )

    duration, duration_malformed, duration_field, duration_raw = _optional_integer_field(
        candidate.raw, "minutes"
    )
    if duration_malformed:
        anomalies.append(
            _anomaly(
                snapshot,
                source_row_number=candidate.source_row_number,
                raw_record_json=candidate.raw_record_json,
                match_id=candidate.match_id,
                orientation=orientation,
                player_id=player_id,
                scope="raw_stat",
                severity="warning",
                code="MALFORMED_DURATION_MINUTES",
                field=duration_field,
                raw_value=duration_raw,
            )
        )

    score_missing = SCORE_MISSING in candidate.score.anomaly_codes
    source_is_match_date = source.row_date_semantics is RowDateSemantics.MATCH_DATE
    match_date = (
        candidate.exact_match_date
        if candidate.exact_match_date is not None
        else candidate.source_date
        if source_is_match_date
        else None
    )
    row = {
        "snapshot_id": source.sha256,
        "snapshot_sha256": snapshot.sha256,
        "source_id": source.source_id,
        "source_url": source.locator,
        "source_schema_version": source.schema_version,
        "retrieved_at_utc": source.retrieved_at_utc,
        "normalization_version": NORMALIZATION_VERSION,
        "source_row_number": candidate.source_row_number,
        "orientation": orientation,
        "match_id": candidate.match_id,
        "source_date": candidate.source_date,
        "match_date": match_date,
        "match_date_source_id": _optional_text(candidate.raw.get("_match_date_source_id")),
        "match_date_source_sha256": _optional_text(
            candidate.raw.get("_match_date_source_sha256")
        ),
        "match_date_crosswalk_id": _optional_text(
            candidate.raw.get("_match_date_crosswalk_id")
        ),
        "event_start_date": None if source_is_match_date else candidate.source_date,
        "source_date_semantics": source.row_date_semantics.value,
        "available_at_utc": candidate.available_at_utc,
        "tour": source.tour.value,
        "event": _optional_text(candidate.raw.get("tourney_name")),
        "event_year": candidate.source_date.year,
        "level": _optional_text(candidate.raw.get("tourney_level")),
        "round": _optional_text(candidate.raw.get("round")),
        "surface": _optional_text(candidate.raw.get("surface")),
        "indoor": _optional_text(candidate.raw.get("indoor")),
        "best_of": candidate.best_of,
        "player_id": player_id,
        "opponent_id": opponent_id,
        "player_source_id": _optional_text(candidate.raw.get(player_source_field)),
        "opponent_source_id": _optional_text(candidate.raw.get(opponent_source_field)),
        "player_name": _optional_text(candidate.raw.get(player_name_field)),
        "opponent_name": _optional_text(candidate.raw.get(opponent_name_field)),
        "player_hand": _optional_text(candidate.raw.get(player_hand_field)),
        "opponent_hand": _optional_text(candidate.raw.get(opponent_hand_field)),
        "score": _optional_text(candidate.raw.get("score")),
        "completed": None if score_missing else candidate.score.completed,
        "retirement": candidate.score.retirement,
        "walkover": candidate.score.walkover,
        **stats,
        "total_points_won": total_points_won,
        "winners": None,
        "unforced_errors": None,
        "duration_minutes": duration,
        "invalid_stat_fields": invalid_fields,
        "raw_record_json": candidate.raw_record_json,
    }
    return row, anomalies


def _empty_normalized_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_NORMALIZED_COLUMNS)


def _empty_anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_ANOMALY_COLUMNS)


def normalize_player_service_rows(
    raw: pd.DataFrame, *, snapshot: RawSourceSnapshot
) -> NormalizedServiceRows:
    """Emit two symmetric service rows for every accepted source match.

    Exact duplicate records are represented once and audited.  Conflicting
    corrections, identity failures, unambiguous score failures, and walkovers
    never enter the normalized service table; their raw records remain in the
    immutable snapshot and are referenced by the anomaly table.
    """

    validate_sackmann_columns(raw)
    candidates: list[_Candidate] = []
    anomaly_records: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        string_keyed_row = {str(key): value for key, value in row.items()}
        candidate, candidate_anomalies = _candidate_from_raw(string_keyed_row, snapshot)
        anomaly_records.extend(candidate_anomalies)
        if candidate is not None:
            candidates.append(candidate)

    duplicate_groups = detect_match_duplicate_groups(
        candidate.match_record_identity for candidate in candidates
    )
    excluded_rows: set[int] = set()
    for group in duplicate_groups:
        group_candidates = [item for item in candidates if item.match_id == group.match_id]
        if group.kind is MatchDuplicateKind.EXACT_DUPLICATE:
            keep = min(item.source_row_number for item in group_candidates)
            affected = [item for item in group_candidates if item.source_row_number != keep]
            severity = "warning"
        else:
            affected = group_candidates
            severity = "error"
        for candidate in affected:
            excluded_rows.add(candidate.source_row_number)
            anomaly_records.append(
                _anomaly(
                    snapshot,
                    source_row_number=candidate.source_row_number,
                    raw_record_json=candidate.raw_record_json,
                    match_id=candidate.match_id,
                    scope="match",
                    severity=severity,
                    code=group.anomaly_code.value,
                )
            )

    normalized_records: list[dict[str, Any]] = []
    for candidate in candidates:
        for code, field, raw_value in candidate.warning_codes:
            anomaly_records.append(
                _anomaly(
                    snapshot,
                    source_row_number=candidate.source_row_number,
                    raw_record_json=candidate.raw_record_json,
                    match_id=candidate.match_id,
                    scope="match",
                    severity="warning",
                    code=code,
                    field=field,
                    raw_value=raw_value,
                )
            )
        if candidate.fatal_codes:
            excluded_rows.add(candidate.source_row_number)
            for code, field, raw_value in candidate.fatal_codes:
                anomaly_records.append(
                    _anomaly(
                        snapshot,
                        source_row_number=candidate.source_row_number,
                        raw_record_json=candidate.raw_record_json,
                        match_id=candidate.match_id,
                        scope="match",
                        code=code,
                        field=field,
                        raw_value=raw_value,
                    )
                )
        if candidate.score.walkover:
            excluded_rows.add(candidate.source_row_number)
            anomaly_records.append(
                _anomaly(
                    snapshot,
                    source_row_number=candidate.source_row_number,
                    raw_record_json=candidate.raw_record_json,
                    match_id=candidate.match_id,
                    scope="match",
                    severity="info",
                    code="WALKOVER_EXCLUDED",
                    field="score",
                    raw_value=candidate.raw.get("score"),
                )
            )
        if candidate.source_row_number in excluded_rows:
            continue

        winner_stats, _, _ = _stat_values(candidate, "winner")
        loser_stats, _, _ = _stat_values(candidate, "loser")
        service_games_mismatch = (
            candidate.score.completed
            and candidate.score.regular_service_games is not None
            and candidate.score.official_games is not None
            and winner_stats["service_games"] is not None
            and loser_stats["service_games"] is not None
            and winner_stats["service_games"] + loser_stats["service_games"]
            not in {
                candidate.score.regular_service_games,
                candidate.score.official_games,
            }
        )
        if service_games_mismatch:
            anomaly_records.append(
                _anomaly(
                    snapshot,
                    source_row_number=candidate.source_row_number,
                    raw_record_json=candidate.raw_record_json,
                    match_id=candidate.match_id,
                    scope="match",
                    code="SERVICE_GAMES_SCORE_MISMATCH",
                    field="w_SvGms,l_SvGms,score",
                    raw_value=(
                        candidate.raw.get("w_SvGms"),
                        candidate.raw.get("l_SvGms"),
                        candidate.raw.get("score"),
                    ),
                )
            )

        for orientation in ("winner", "loser"):
            normalized, anomalies = _normalized_row(candidate, snapshot, orientation=orientation)
            if service_games_mismatch:
                normalized["invalid_stat_fields"] = tuple(
                    dict.fromkeys((*normalized["invalid_stat_fields"], "service_games"))
                )
            normalized_records.append(normalized)
            anomaly_records.extend(anomalies)

    rows = pd.DataFrame.from_records(normalized_records, columns=_NORMALIZED_COLUMNS)
    if rows.empty:
        rows = _empty_normalized_frame()
    for column in (
        "source_row_number",
        "event_year",
        "best_of",
        *_STAT_SUFFIXES.keys(),
        "total_points_won",
        "winners",
        "unforced_errors",
        "duration_minutes",
    ):
        rows[column] = rows[column].astype("Int64")

    anomaly_frame = pd.DataFrame.from_records(anomaly_records, columns=_ANOMALY_COLUMNS)
    if anomaly_frame.empty:
        anomaly_frame = _empty_anomaly_frame()
    validate_symmetric_service_rows(rows)
    return NormalizedServiceRows(
        rows=rows,
        anomalies=anomaly_frame,
        raw_row_count=len(raw),
        accepted_match_count=int(rows["match_id"].nunique()),
    )


def validate_symmetric_service_rows(rows: pd.DataFrame) -> None:
    """Assert the two reciprocal service directions for every normalized match."""

    if rows.empty:
        return
    required = {"match_id", "player_id", "opponent_id", "orientation"}
    missing = required.difference(rows.columns)
    if missing:
        raise SymmetryValidationError(
            "normalized symmetry columns missing: " + ", ".join(sorted(missing))
        )
    for match_id, group in rows.groupby("match_id", dropna=False):
        if len(group) != 2:
            raise SymmetryValidationError(
                f"match {match_id!r} has {len(group)} service rows, expected 2"
            )
        if set(group["orientation"]) != {"winner", "loser"}:
            raise SymmetryValidationError(f"match {match_id!r} lacks winner/loser symmetry")
        pairs = {(row.player_id, row.opponent_id) for row in group.itertuples()}
        if len(pairs) != 2 or any((opponent, player) not in pairs for player, opponent in pairs):
            raise SymmetryValidationError(
                f"match {match_id!r} does not contain reciprocal opponents"
            )
        if group["player_id"].nunique() != 2:
            raise SymmetryValidationError(f"match {match_id!r} is a self-match")


def combine_anomaly_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    """Concatenate compatible anomaly tables without mutating their inputs."""

    nonempty = [table.copy() for table in tables if not table.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True, sort=False)
    return _empty_anomaly_frame()


__all__ = [
    "NORMALIZATION_VERSION",
    "NormalizedServiceRows",
    "SackmannSchemaError",
    "SymmetryValidationError",
    "combine_anomaly_tables",
    "normalize_player_service_rows",
    "validate_sackmann_columns",
    "validate_symmetric_service_rows",
]
