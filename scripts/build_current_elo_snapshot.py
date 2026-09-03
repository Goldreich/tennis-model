"""Build the current main-tour Elo snapshot used by production v1.1.

This is a data-preparation utility, not a change to frozen Tennis Model v1.0.
It computes ATP and WTA ratings independently from completed main-tour singles
matches and writes both the complete rating table and the current US Open field.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "elo-main-tour-current" / "source" / "tennismylife"
OUTPUT_DIR = ROOT / "artifacts" / "elo" / "current-main-tour-2026-08-30"
DRAW_FILES = {
    "atp": ("us_open_2026_ms_draw.pdf", "us_open_2026_ms_draw.txt", "2026_MS_draw.pdf"),
    "wta": ("us_open_2026_ws_draw.pdf", "us_open_2026_ws_draw.txt", "2026_WS_draw.pdf"),
}

YEARS = range(2017, 2027)
CUTOFF_DATE = 20260830
INITIAL_ELO = 1500.0
K_FACTOR = 16.0
GLOBAL_WEIGHT = 0.5
SURFACE_WEIGHT = 0.5
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

SOURCE_URL = "https://stats.tennismylife.org/data/{name}"
INVALID_SCORE = re.compile(r"(?:W[./]?O|WALKOVER|\bRET\b|\bDEF\b|\bABD\b|\bBYE\b)", re.I)


@dataclass
class PlayerState:
    player_key: str
    player_id: str
    player_name: str
    global_elo: float = INITIAL_ELO
    surface_elo: dict[str, float] = field(
        default_factory=lambda: {surface: INITIAL_ELO for surface in SURFACES}
    )
    matches: int = 0
    wins: int = 0
    losses: int = 0
    surface_matches: dict[str, int] = field(
        default_factory=lambda: {surface: 0 for surface in SURFACES}
    )
    last_date: int = 0
    last_event: str = ""


def clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def normalized_id(value: str) -> str:
    value = clean(value)
    return value[:-2] if re.fullmatch(r"\d+\.0", value) else value


def parse_date(value: str) -> int:
    digits = re.sub(r"\D", "", clean(value))
    return int(digits[:8]) if len(digits) >= 8 else 0


def parse_match_num(value: str) -> int:
    value = clean(value)
    try:
        return int(float(value))
    except ValueError:
        return 0


def canonical_surface(value: str) -> str:
    value = clean(value).title()
    return value if value in SURFACES else ""


def player_identity(tour: str, row: dict[str, str], side: str) -> tuple[str, str, str]:
    player_id = normalized_id(row.get(f"{side}_id", ""))
    player_name = clean(row.get(f"{side}_name", ""))
    if player_id:
        return f"{tour}:id:{player_id}", player_id, player_name
    name_key = normalized_name(player_name)
    return f"{tour}:name:{name_key}", "", player_name


def effective_elo(player: PlayerState, surface: str) -> float:
    return GLOBAL_WEIGHT * player.global_elo + SURFACE_WEIGHT * player.surface_elo[surface]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def source_names(tour: str) -> list[str]:
    names = [f"{tour}_{year}.csv" for year in YEARS]
    names.append(f"{tour}_ongoing.csv")
    return names


def remote_name(local_name: str) -> str:
    if local_name == "atp_ongoing.csv":
        return "ongoing_tourneys.csv"
    if local_name == "wta_ongoing.csv":
        return "wta_ongoing_tourneys.csv"
    match = re.fullmatch(r"(atp|wta)_(\d{4})\.csv", local_name)
    if not match:
        raise ValueError(f"Unexpected source name: {local_name}")
    tour, year = match.groups()
    return f"{year}.csv" if tour == "atp" else f"{year}_wta.csv"


def load_sources() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for tour in ("atp", "wta"):
        for name in source_names(tour):
            path = SOURCE_DIR / name
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            count = 0
            for source_row in reader:
                row = {clean(key): clean(value) for key, value in source_row.items() if key is not None}
                row["_tour"] = tour
                row["_source"] = name
                rows.append(row)
                count += 1
            remote = remote_name(name)
            manifest.append(
                {
                    "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "source_url": SOURCE_URL.format(name=remote),
                    "sha256": digest,
                    "bytes": len(raw),
                    "rows": count,
                }
            )
    return rows, manifest


def draw_player_name(line: str) -> tuple[int, str] | None:
    match = re.match(r"^\s*(\d{1,3})\.\s+(.+?)\s*$", line)
    if not match:
        return None
    position = int(match.group(1))
    if not 1 <= position <= 128:
        return None
    value = re.sub(r"\[[^]]*\]|\([^)]*\)", "", match.group(2)).strip()
    if "," not in value:
        return None
    surname, given = (part.strip() for part in value.split(",", 1))
    given = re.sub(r"\b[A-Z]{3}\b", "", given)
    given = re.sub(r"\s+", " ", given).strip()
    surname = re.sub(r"\s+", " ", surname).strip().title()
    if not surname or not given:
        return None
    return position, f"{given} {surname}"


def load_official_draws() -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    fields: dict[str, dict[str, str]] = {"atp": {}, "wta": {}}
    manifest: list[dict[str, object]] = []
    for tour, (pdf_name, text_name, remote_name) in DRAW_FILES.items():
        pdf_path = SOURCE_DIR / pdf_name
        text_path = SOURCE_DIR / text_name
        pdf_raw = pdf_path.read_bytes()
        text_raw = text_path.read_bytes()
        text = text_raw.decode("utf-8", errors="replace")
        positions: dict[int, str] = {}
        for line in text.splitlines():
            parsed = draw_player_name(line)
            if parsed is not None and parsed[0] not in positions:
                positions[parsed[0]] = parsed[1]
        if set(positions) != set(range(1, 129)):
            missing = sorted(set(range(1, 129)) - set(positions))
            raise RuntimeError(f"Official {tour.upper()} draw did not yield 128 entries; missing {missing}")
        for position in range(1, 129):
            name = positions[position]
            fields[tour][normalized_name(name)] = name
        if len(fields[tour]) != 128:
            raise RuntimeError(f"Official {tour.upper()} draw contains duplicate normalized names")
        manifest.extend(
            [
                {
                    "local_path": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_url": f"https://www.usopen.org/en_US/scores/draws/{remote_name}",
                    "sha256": hashlib.sha256(pdf_raw).hexdigest(),
                    "bytes": len(pdf_raw),
                    "rows": 128,
                },
                {
                    "local_path": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                    "derived_from": pdf_name,
                    "sha256": hashlib.sha256(text_raw).hexdigest(),
                    "bytes": len(text_raw),
                    "rows": len(text.splitlines()),
                },
            ]
        )
    return fields, manifest


def is_current_us_open(row: dict[str, object]) -> bool:
    event = normalized_name(clean(row.get("tourney_name", "")))
    level = clean(row.get("tourney_level", "")).upper()
    return "usopen" in event and level in {"G", "GRAND SLAM", "GRAND_SLAM", ""}


def is_eligible(row: dict[str, object]) -> bool:
    event_date = parse_date(clean(row.get("tourney_date", "")))
    surface = canonical_surface(clean(row.get("surface", "")))
    winner = clean(row.get("winner_name", ""))
    loser = clean(row.get("loser_name", ""))
    score = clean(row.get("score", ""))
    level = clean(row.get("tourney_level", "")).upper()
    if not (event_date and event_date < CUTOFF_DATE and surface and winner and loser and score):
        return False
    if level in {"C", "CH", "CHALLENGER", "S", "SATELLITE", "ITF", "Q"}:
        return False
    if "/" in winner or "/" in loser or INVALID_SCORE.search(score):
        return False
    return True


def ensure_player(
    states: dict[str, PlayerState], tour: str, row: dict[str, object], side: str
) -> PlayerState:
    key, player_id, player_name = player_identity(tour, row, side)
    if key not in states:
        states[key] = PlayerState(key, player_id, player_name)
    elif player_name:
        states[key].player_name = player_name
    return states[key]


def match_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        parse_date(clean(row.get("tourney_date", ""))),
        clean(row.get("tourney_id", "")),
        normalized_name(clean(row.get("tourney_name", ""))),
        parse_match_num(clean(row.get("match_num", ""))),
        clean(row.get("round", "")),
        normalized_name(clean(row.get("winner_name", ""))),
        normalized_name(clean(row.get("loser_name", ""))),
    )


def build_ratings(
    rows: list[dict[str, object]], us_open_field: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, PlayerState]], dict[str, set[str]], dict[str, int], int]:
    states: dict[str, dict[str, PlayerState]] = {"atp": {}, "wta": {}}
    us_open_matches: dict[str, int] = {"atp": 0, "wta": 0}

    for row in rows:
        if not is_current_us_open(row):
            continue
        tour = clean(row["_tour"])
        event_date = parse_date(clean(row.get("tourney_date", "")))
        if event_date < CUTOFF_DATE:
            continue
        us_open_matches[tour] += 1

    eligible = [row for row in rows if is_eligible(row)]
    eligible.sort(key=match_sort_key)
    seen_matches: set[tuple[object, ...]] = set()
    processed = 0

    for row in eligible:
        tour = clean(row["_tour"])
        winner_key, _, _ = player_identity(tour, row, "winner")
        loser_key, _, _ = player_identity(tour, row, "loser")
        dedupe_key = (
            tour,
            parse_date(clean(row.get("tourney_date", ""))),
            clean(row.get("tourney_id", "")) or normalized_name(clean(row.get("tourney_name", ""))),
            parse_match_num(clean(row.get("match_num", ""))),
            winner_key,
            loser_key,
        )
        if dedupe_key in seen_matches:
            continue
        seen_matches.add(dedupe_key)

        winner = ensure_player(states[tour], tour, row, "winner")
        loser = ensure_player(states[tour], tour, row, "loser")
        surface = canonical_surface(clean(row.get("surface", "")))
        event_date = parse_date(clean(row.get("tourney_date", "")))
        event = clean(row.get("tourney_name", ""))
        probability = expected_score(effective_elo(winner, surface), effective_elo(loser, surface))
        delta = K_FACTOR * (1.0 - probability)

        winner.global_elo += delta
        loser.global_elo -= delta
        winner.surface_elo[surface] += delta
        loser.surface_elo[surface] -= delta
        winner.matches += 1
        loser.matches += 1
        winner.wins += 1
        loser.losses += 1
        winner.surface_matches[surface] += 1
        loser.surface_matches[surface] += 1
        winner.last_date = max(winner.last_date, event_date)
        loser.last_date = max(loser.last_date, event_date)
        if event_date >= winner.last_date:
            winner.last_event = event
        if event_date >= loser.last_date:
            loser.last_event = event
        processed += 1

    # The live WTA feed uses a different player-ID namespace from the annual files,
    # so the official draw is joined to history by normalized name. New main-draw
    # players with no eligible historical match retain the agreed 1500 prior.
    known_names = {
        tour: {normalized_name(player.player_name) for player in states[tour].values()}
        for tour in ("atp", "wta")
    }
    for tour in ("atp", "wta"):
        for name_key, name in us_open_field[tour].items():
            if name_key not in known_names[tour]:
                key = f"{tour}:draw:{name_key}"
                states[tour][key] = PlayerState(key, "", name)
                known_names[tour].add(name_key)

    field_keys = {tour: set(us_open_field[tour]) for tour in ("atp", "wta")}
    return states, field_keys, us_open_matches, processed


def rating_row(tour: str, player: PlayerState) -> dict[str, object]:
    row: dict[str, object] = {
        "tour": tour.upper(),
        "player_key": player.player_key,
        "player_id": player.player_id,
        "player_name": player.player_name,
        "global_elo": round(player.global_elo, 6),
        "hard_elo": round(player.surface_elo["Hard"], 6),
        "effective_hard_elo": round(effective_elo(player, "Hard"), 6),
        "clay_elo": round(player.surface_elo["Clay"], 6),
        "effective_clay_elo": round(effective_elo(player, "Clay"), 6),
        "grass_elo": round(player.surface_elo["Grass"], 6),
        "effective_grass_elo": round(effective_elo(player, "Grass"), 6),
        "carpet_elo": round(player.surface_elo["Carpet"], 6),
        "effective_carpet_elo": round(effective_elo(player, "Carpet"), 6),
        "matches": player.matches,
        "wins": player.wins,
        "losses": player.losses,
        "hard_matches": player.surface_matches["Hard"],
        "clay_matches": player.surface_matches["Clay"],
        "grass_matches": player.surface_matches["Grass"],
        "carpet_matches": player.surface_matches["Carpet"],
        "last_event_start": player.last_date,
        "last_event": player.last_event,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_rows, sources = load_sources()
    official_draws, draw_sources = load_official_draws()
    sources.extend(draw_sources)
    states, field_keys, field_matches, processed = build_ratings(source_rows, official_draws)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = [
        rating_row(tour, player)
        for tour in ("atp", "wta")
        for player in states[tour].values()
    ]
    all_rows.sort(key=lambda row: (row["tour"], -float(row["effective_hard_elo"]), row["player_name"]))

    field_rows: list[dict[str, object]] = []
    for tour in ("atp", "wta"):
        for name_key in field_keys[tour]:
            candidates = [
                player
                for player in states[tour].values()
                if normalized_name(player.player_name) == name_key
            ]
            player = max(candidates, key=lambda item: (item.matches, item.last_date))
            field_rows.append(rating_row(tour, player))
    field_rows.sort(key=lambda row: (row["tour"], -float(row["effective_hard_elo"]), row["player_name"]))
    top_atp = [row for row in field_rows if row["tour"] == "ATP"][:10]
    top_wta = [row for row in field_rows if row["tour"] == "WTA"][:10]

    write_csv(OUTPUT_DIR / "all_player_elos.csv", all_rows)
    write_csv(OUTPUT_DIR / "us_open_field_elos.csv", field_rows)
    write_csv(OUTPUT_DIR / "top_10_men.csv", top_atp)
    write_csv(OUTPUT_DIR / "top_10_women.csv", top_wta)

    manifest = {
        "artifact": "current-main-tour-elo-snapshot",
        "cutoff": "2026-08-30T00:00:00Z",
        "initial_elo": INITIAL_ELO,
        "k_factor": K_FACTOR,
        "effective_rating": {"global_weight": GLOBAL_WEIGHT, "current_surface_weight": SURFACE_WEIGHT},
        "tour_scope": "ATP and WTA main-tour files; Challenger and qualifying files excluded",
        "match_policy": "Completed singles before cutoff; walkovers, retirements, defaults, abandonments, and byes excluded",
        "processed_matches": processed,
        "players_rated": {tour: len(states[tour]) for tour in ("atp", "wta")},
        "us_open_matches_observed": field_matches,
        "us_open_field_players": {tour: len(field_keys[tour]) for tour in ("atp", "wta")},
        "ranking_metric": "effective_hard_elo = 0.5 * global_elo + 0.5 * hard_elo",
        "sources": sources,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Current main-tour Elo snapshot",
        "",
        "Cutoff: 2026-08-30 00:00 UTC (before the US Open main draw).",
        "",
        "Method: separate ATP/WTA pools; every player starts at 1500; K=16; effective rating is 50% global and 50% current-surface Elo.",
        "",
        "## Men - US Open top 10 by effective hard-court Elo",
        "",
        "| Rank | Player | Effective hard Elo | Global Elo | Hard Elo | Matches |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top_atp, 1):
        lines.append(
            f"| {rank} | {row['player_name']} | {float(row['effective_hard_elo']):.1f} | "
            f"{float(row['global_elo']):.1f} | {float(row['hard_elo']):.1f} | {row['matches']} |"
        )
    lines.extend(
        [
            "",
            "## Women - US Open top 10 by effective hard-court Elo",
            "",
            "| Rank | Player | Effective hard Elo | Global Elo | Hard Elo | Matches |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_wta, 1):
        lines.append(
            f"| {rank} | {row['player_name']} | {float(row['effective_hard_elo']):.1f} | "
            f"{float(row['global_elo']):.1f} | {float(row['hard_elo']):.1f} | {row['matches']} |"
        )
    lines.extend(
        [
            "",
            f"Processed eligible matches: {processed}.",
            f"Detected US Open field: {len(field_keys['atp'])} men and {len(field_keys['wta'])} women.",
            "",
            "See manifest.json for source URLs, hashes, row counts, and the complete computation policy.",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(OUTPUT_DIR), "top_atp": top_atp, "top_wta": top_wta, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
