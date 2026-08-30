"""Normalize pinned official 2026 US Open singles payloads for current fitting.

The module is deliberately network-free.  Callers must first retain and hash
the official day and per-match JSON objects, then pass the verified bytes here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from tennis_model.data.component_counts import ComponentCountTable, build_serve_component_counts
from tennis_model.identity import make_player_identity
from tennis_model.schemas import Tour

CURRENT_USOPEN_NORMALIZATION_VERSION = "official-usopen-player-service/v1"
_NEW_YORK = ZoneInfo("America/New_York")
_EVENT_CODES = {"MQ": Tour.ATP, "WQ": Tour.WTA, "MS": Tour.ATP, "WS": Tour.WTA}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class OfficialJsonObject:
    """One already-retained official JSON object and its truthful receipt."""

    source_id: str
    locator: str
    retrieved_at_utc: datetime
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.locator.strip():
            raise ValueError("official source identity and locator must not be empty")
        if self.retrieved_at_utc.tzinfo is None or self.retrieved_at_utc.utcoffset() is None:
            raise ValueError("official retrieval time must be timezone-aware")
        observed = hashlib.sha256(self.payload).hexdigest()
        if not _SHA256_RE.fullmatch(self.sha256) or observed != self.sha256:
            raise ValueError("official payload SHA-256 does not match its receipt")

    def json(self) -> dict[str, Any]:
        try:
            value = json.loads(self.payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid official JSON object {self.source_id}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"official JSON object {self.source_id} is not a mapping")
        return value


@dataclass(frozen=True, slots=True)
class CurrentUSOpenRows:
    rows: pd.DataFrame
    counts: ComponentCountTable
    identity_crosswalk: pd.DataFrame
    exclusions: pd.DataFrame
    completed_match_count: int
    included_match_count: int


def normalized_name(value: str) -> str:
    """Return the deterministic exact-name key; this is not fuzzy matching."""

    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def build_official_player_crosswalk(
    historical_rows: pd.DataFrame,
    official_players: tuple[tuple[Tour, str, str], ...],
) -> tuple[dict[tuple[Tour, str], str], pd.DataFrame]:
    """Resolve official IDs by unique exact full name, otherwise retain a new official ID.

    An exact name that maps to multiple historical identities is unresolved and
    receives no mapping.  A name absent from history is a valid unseen player,
    anchored to the stable official US Open ID.
    """

    required = {"tour", "player_id", "player_name"}
    missing = required.difference(historical_rows.columns)
    if missing:
        raise ValueError("historical identity rows missing: " + ", ".join(sorted(missing)))
    historical: dict[tuple[Tour, str], set[str]] = {}
    for row in historical_rows.loc[:, sorted(required)].drop_duplicates().itertuples(index=False):
        tour = Tour(str(row.tour).upper())
        key = normalized_name(str(row.player_name))
        if key:
            historical.setdefault((tour, key), set()).add(str(row.player_id))

    mapping: dict[tuple[Tour, str], str] = {}
    records: list[dict[str, Any]] = []
    for tour, official_id, official_name in sorted(
        set(official_players), key=lambda item: (item[0].value, item[1])
    ):
        key = normalized_name(official_name)
        candidates = tuple(sorted(historical.get((tour, key), ())))
        if len(candidates) == 1:
            canonical_id = candidates[0]
            status = "MATCHED_UNIQUE_EXACT_NAME"
        elif not candidates:
            canonical_id = make_player_identity(
                "usopen-official", tour, official_id, display_name=official_name
            ).player_id
            status = "NEW_OFFICIAL_ID"
        else:
            canonical_id = None
            status = "AMBIGUOUS_EXACT_NAME"
        if canonical_id is not None:
            mapping[(tour, official_id)] = canonical_id
        records.append(
            {
                "tour": tour.value,
                "official_player_id": official_id,
                "official_player_name": official_name,
                "normalized_name": key,
                "status": status,
                "historical_candidate_ids": candidates,
                "canonical_player_id": canonical_id,
            }
        )
    return mapping, pd.DataFrame.from_records(records)


def _integer(value: Any) -> tuple[int | None, bool]:
    if value is None or value == "":
        return None, False
    if isinstance(value, bool):
        return None, True
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None, True
    if isinstance(value, float) and value != converted:
        return None, True
    return converted, False


def _score_string(match: dict[str, Any]) -> str | None:
    scores = match.get("scores")
    if not isinstance(scores, dict) or not isinstance(scores.get("sets"), list):
        return None
    result: list[str] = []
    for item in scores["sets"]:
        if not isinstance(item, list) or len(item) != 2:
            return None
        try:
            left, right = item
            token = f"{int(left['score'])}-{int(right['score'])}"
            loser_tiebreak = (
                right.get("tiebreak")
                if int(left["score"]) > int(right["score"])
                else left.get("tiebreak")
            )
            if loser_tiebreak is not None:
                token += f"({int(loser_tiebreak)})"
            result.append(token)
        except (KeyError, TypeError, ValueError):
            return None
    return " ".join(result) or None


def _player_row(
    *,
    match: dict[str, Any],
    day_match: dict[str, Any],
    event_code: str,
    team_number: int,
    tour: Tour,
    player_id: str,
    opponent_id: str,
    source: OfficialJsonObject,
    completed_at_utc: datetime,
) -> dict[str, Any]:
    team_key = f"team{team_number}"
    opponent_key = f"team{3 - team_number}"
    team = match[team_key]
    opponent = match[opponent_key]
    base = match.get("base_stats", {}).get("match", {}).get(f"team_{team_number}", {})
    serve = match.get("serve_stats", {}).get("match", {}).get(f"team_{team_number}", {})
    raw_stats = {
        "service_points": base.get("t_f_srv"),
        "first_serves_in": base.get("t_f_srv_in"),
        "first_serve_points_won": base.get("t_f_srv_w"),
        "second_serve_points_won": base.get("t_s_srv_w"),
        "aces": serve.get("f_srv_ace"),
        "double_faults": base.get("df"),
        "service_games": serve.get("t_gms"),
        "total_points_won": base.get("t_p_w"),
        "winners": base.get("t_w"),
        "unforced_errors": base.get("t_ue"),
    }
    parsed: dict[str, int | None] = {}
    invalid: list[str] = []
    for field, value in raw_stats.items():
        parsed[field], malformed = _integer(value)
        if malformed:
            invalid.append(field)
    duration_text = match.get("duration")
    duration_minutes: int | None = None
    if isinstance(duration_text, str) and re.fullmatch(r"\d+:\d{2}", duration_text):
        hours, minutes = duration_text.split(":")
        duration_minutes = int(hours) * 60 + int(minutes)

    status_code = str(day_match.get("statusCode", match.get("statusCode", "")))
    retirement = status_code == "E" or str(day_match.get("status", "")).casefold() == "retired"
    match_date = completed_at_utc.astimezone(_NEW_YORK).date()
    match_id = f"usopen-official:2026:{day_match['match_id']}"
    return {
        "snapshot_id": source.sha256,
        "snapshot_sha256": source.sha256,
        "source_id": source.source_id,
        "source_url": source.locator,
        "source_schema_version": "official-usopen-complete-match-json/2026",
        "retrieved_at_utc": source.retrieved_at_utc.astimezone(UTC),
        "normalization_version": CURRENT_USOPEN_NORMALIZATION_VERSION,
        "source_row_number": 1,
        "orientation": team_key,
        "match_id": match_id,
        "source_date": match_date,
        "match_date": match_date,
        "match_date_source_id": source.source_id,
        "match_date_source_sha256": source.sha256,
        "match_date_crosswalk_id": None,
        "event_start_date": None,
        "source_date_semantics": "match_date",
        "available_at_utc": source.retrieved_at_utc.astimezone(UTC),
        "tour": tour.value,
        "event": "US Open",
        "event_year": 2026,
        "level": "Q" if event_code in {"MQ", "WQ"} else "G",
        "round": str(day_match.get("roundNameShort") or match.get("roundNameShort") or ""),
        "surface": "Hard",
        "indoor": None,
        "best_of": 3 if tour is Tour.WTA or event_code in {"MQ", "WQ"} else 5,
        "player_id": player_id,
        "opponent_id": opponent_id,
        "player_source_id": str(team["idA"]),
        "opponent_source_id": str(opponent["idA"]),
        "player_name": f"{team['firstNameA']} {team['lastNameA']}",
        "opponent_name": f"{opponent['firstNameA']} {opponent['lastNameA']}",
        "player_hand": None,
        "opponent_hand": None,
        "score": _score_string(day_match),
        "completed": status_code in {"D", "E"},
        "retirement": retirement,
        "walkover": False,
        **parsed,
        "break_points_faced": None,
        "break_points_saved": None,
        "duration_minutes": duration_minutes,
        "invalid_stat_fields": tuple(invalid),
        "raw_record_json": json.dumps(match, sort_keys=True, separators=(",", ":")),
    }


def normalize_completed_singles(
    day_objects: tuple[OfficialJsonObject, ...],
    match_objects: dict[str, OfficialJsonObject],
    *,
    historical_identity_rows: pd.DataFrame,
    information_cutoff_utc: datetime,
) -> CurrentUSOpenRows:
    """Normalize completed MQ/WQ/MS/WS matches available strictly before cutoff."""

    cutoff = information_cutoff_utc.astimezone(UTC)
    selected: list[tuple[dict[str, Any], str, Tour, datetime]] = []
    official_players: list[tuple[Tour, str, str]] = []
    exclusions: list[dict[str, Any]] = []
    completed_count = 0
    for day_object in day_objects:
        day = day_object.json()
        matches = day.get("matches")
        if not isinstance(matches, list):
            raise ValueError(f"official day object {day_object.source_id} lacks matches")
        for item in matches:
            if not isinstance(item, dict) or item.get("eventCode") not in _EVENT_CODES:
                continue
            if str(item.get("statusCode")) not in {"D", "E"}:
                continue
            completed_count += 1
            event_code = str(item["eventCode"])
            tour = _EVENT_CODES[event_code]
            epoch = item.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, (int, str)):
                exclusions.append(
                    {"match_id": item.get("match_id"), "reason": "INVALID_COMPLETION_TIMESTAMP"}
                )
                continue
            try:
                completed_at = datetime.fromtimestamp(int(epoch) / 1000, tz=UTC)
            except (TypeError, ValueError, OSError):
                exclusions.append(
                    {"match_id": item.get("match_id"), "reason": "INVALID_COMPLETION_TIMESTAMP"}
                )
                continue
            match_id = str(item.get("match_id", ""))
            source = match_objects.get(match_id)
            if source is None:
                exclusions.append({"match_id": match_id, "reason": "MISSING_MATCH_PAYLOAD"})
                continue
            if source.retrieved_at_utc.astimezone(UTC) >= cutoff or completed_at >= cutoff:
                exclusions.append({"match_id": match_id, "reason": "AT_OR_AFTER_CUTOFF"})
                continue
            for team_key in ("team1", "team2"):
                team = item.get(team_key)
                if not isinstance(team, dict) or team.get("idA") is None:
                    exclusions.append({"match_id": match_id, "reason": "INVALID_SINGLES_IDENTITY"})
                    break
                official_players.append(
                    (tour, str(team["idA"]), f"{team['firstNameA']} {team['lastNameA']}")
                )
            else:
                selected.append((item, event_code, tour, completed_at))

    identity_map, identity_crosswalk = build_official_player_crosswalk(
        historical_identity_rows, tuple(official_players)
    )
    rows: list[dict[str, Any]] = []
    for item, event_code, tour, completed_at in selected:
        match_id = str(item["match_id"])
        source = match_objects[match_id]
        payload = source.json()
        matches = payload.get("matches")
        if not isinstance(matches, list) or len(matches) != 1 or not isinstance(matches[0], dict):
            exclusions.append({"match_id": match_id, "reason": "INVALID_MATCH_PAYLOAD"})
            continue
        match = matches[0]
        ids = (str(item["team1"]["idA"]), str(item["team2"]["idA"]))
        canonical = (identity_map.get((tour, ids[0])), identity_map.get((tour, ids[1])))
        if canonical[0] is None or canonical[1] is None or canonical[0] == canonical[1]:
            exclusions.append({"match_id": match_id, "reason": "UNRESOLVED_PLAYER_IDENTITY"})
            continue
        rows.extend(
            (
                _player_row(
                    match=match,
                    day_match=item,
                    event_code=event_code,
                    team_number=1,
                    tour=tour,
                    player_id=canonical[0],
                    opponent_id=canonical[1],
                    source=source,
                    completed_at_utc=completed_at,
                ),
                _player_row(
                    match=match,
                    day_match=item,
                    event_code=event_code,
                    team_number=2,
                    tour=tour,
                    player_id=canonical[1],
                    opponent_id=canonical[0],
                    source=source,
                    completed_at_utc=completed_at,
                ),
            )
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=(
                "service_points",
                "first_serves_in",
                "first_serve_points_won",
                "second_serve_points_won",
                "aces",
                "double_faults",
            )
        )
    counts = build_serve_component_counts(frame)
    return CurrentUSOpenRows(
        rows=frame,
        counts=counts,
        identity_crosswalk=identity_crosswalk,
        exclusions=pd.DataFrame.from_records(exclusions),
        completed_match_count=completed_count,
        included_match_count=len(frame) // 2,
    )


__all__ = [
    "CURRENT_USOPEN_NORMALIZATION_VERSION",
    "CurrentUSOpenRows",
    "OfficialJsonObject",
    "build_official_player_crosswalk",
    "normalize_completed_singles",
    "normalized_name",
]
