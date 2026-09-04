from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "v1.0-counterfactual-2026-09-03-final"
OUTPUT = ROOT / "outputs" / "v1.0-v1.1-actual-comparison-2026-09-03"
SOURCE = OUTPUT / "official-match-payloads"
MANIFEST = INPUT / "replay_manifest.json"
PAIRED = INPUT / "paired_predictions.csv"
FEED = "https://www.usopen.org/en_US/scores/feeds/2026/matches/complete/{match_id}.json"
NETWORK_WORKERS = 4


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if character.isalnum()).casefold()


def _duration_minutes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]) * 60 + int(parts[1])


def _fetch(job: dict[str, Any]) -> dict[str, Any]:
    match_id = str(job["official_match_id"])
    url = FEED.format(match_id=match_id)
    request = urllib.request.Request(url, headers={"User-Agent": "tennis-model-settlement/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    target = SOURCE / f"{match_id}.json"
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"official payload changed for match {match_id}")
    if not target.exists():
        target.write_bytes(payload)
    parsed = json.loads(payload)
    matches = parsed.get("matches", [])
    selected = [item for item in matches if str(item.get("match_id")) == match_id]
    if len(selected) != 1:
        raise RuntimeError(f"expected one official match {match_id}, found {len(selected)}")
    return {
        "match_id": match_id,
        "source_url": url,
        "source_sha256": digest,
        "captured_payload": str(target),
        "match": selected[0],
    }


def _contains_marker(value: Any, markers: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_marker(key, markers) or _contains_marker(item, markers)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_marker(item, markers) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(marker in lowered for marker in markers)
    return False


def _team_name(team: dict[str, Any]) -> str:
    first = str(team.get("firstNameA") or "").strip()
    last = str(team.get("lastNameA") or "").strip()
    return " ".join(item for item in (first, last) if item)


def _tiebreak_count(match: dict[str, Any]) -> int:
    count = 0
    for pair in match.get("scores", {}).get("sets", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        explicit = any(item.get("tiebreak") is not None for item in pair if isinstance(item, dict))
        scores: list[int] = []
        for item in pair:
            try:
                scores.append(int(item.get("score")))
            except (AttributeError, TypeError, ValueError):
                scores = []
                break
        if explicit or (len(scores) == 2 and sorted(scores) == [6, 7]):
            count += 1
    return count


def _actual_record(job: dict[str, Any], captured: dict[str, Any]) -> dict[str, Any]:
    match = captured["match"]
    status = str(match.get("status") or "")
    status_code = str(match.get("statusCode") or "")
    cancelled = _contains_marker(match, ("cancel", "walkover", "w/o", "abandon"))
    retired = _contains_marker(match, ("retired", "retirement"))
    completed = status.casefold() == "completed" and str(match.get("winner")) in {"1", "2"}
    left_expected, right_expected = str(job["match"]).split(" vs ", 1)
    left_actual = _team_name(match.get("team1", {}))
    right_actual = _team_name(match.get("team2", {}))
    if not (
        _normalize_name(left_expected).endswith(_normalize_name(left_actual.split()[-1]))
        and _normalize_name(right_expected).endswith(_normalize_name(right_actual.split()[-1]))
    ):
        raise RuntimeError(
            f"participant order mismatch for {job['official_match_id']}: "
            f"{left_expected} vs {right_expected}; feed has {left_actual} vs {right_actual}"
        )
    stats = match.get("base_stats", {}).get("match", {})
    left_stats = stats.get("team_1") or {}
    right_stats = stats.get("team_2") or {}
    duration = _duration_minutes(match.get("duration"))
    return {
        "official_match_id": str(job["official_match_id"]),
        "match": str(job["match"]),
        "status": status,
        "status_code": status_code,
        "completed": completed,
        "cancelled": cancelled,
        "retired": retired,
        "left_player": left_actual,
        "right_player": right_actual,
        "winner_side": "left" if str(match.get("winner")) == "1" else "right" if str(match.get("winner")) == "2" else None,
        "duration_display": match.get("duration"),
        "duration_minutes": duration,
        "left_aces": left_stats.get("t_ace"),
        "right_aces": right_stats.get("t_ace"),
        "left_double_faults": left_stats.get("df"),
        "right_double_faults": right_stats.get("df"),
        "tiebreaks": _tiebreak_count(match),
        "source_url": captured["source_url"],
        "source_sha256": captured["source_sha256"],
        "captured_payload": captured["captured_payload"],
    }


def _settle(
    row: dict[str, str],
    job: dict[str, Any],
    actual: dict[str, Any],
) -> tuple[int | None, str]:
    if actual["cancelled"]:
        return None, "cancelled"
    if not actual["completed"]:
        return None, "not_completed"
    kind = row["prop_kind"]
    side = str(job["side"])
    other = "right" if side == "left" else "left"
    if kind == "MATCH_WIN":
        return int(actual["winner_side"] == side), "settled"
    if kind == "DURATION_MIN":
        if actual["duration_minutes"] is None:
            return None, "missing_duration"
        threshold = int(job["duration_threshold"])
        outcome = int(int(actual["duration_minutes"]) >= threshold)
        if actual["retired"] and outcome == 0:
            return None, "retirement_not_irreversible"
        return outcome, "settled"
    if kind == "TIEBREAK_COUNT":
        outcome = int(int(actual["tiebreaks"]) >= int(job["tiebreak_threshold"]))
        if actual["retired"] and outcome == 0:
            return None, "retirement_not_irreversible"
        return outcome, "settled"
    if actual["retired"]:
        return None, "retirement_incomplete_scope"
    field = "aces" if kind == "ACE_COMPARE" else "double_faults"
    target = actual[f"{side}_{field}"]
    opponent = actual[f"{other}_{field}"]
    if target is None or opponent is None:
        return None, f"missing_{field}"
    return int(int(target) > int(opponent)), "settled"


def _log_loss(probability: float, outcome: int) -> float:
    clipped = min(1.0 - 1.0e-15, max(1.0e-15, probability))
    return -(outcome * math.log(clipped) + (1 - outcome) * math.log(1.0 - clipped))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row["settlement_status"] == "settled"]
    if not settled:
        return {"settled_props": 0}
    v10 = mean(row["v1_0_brier"] for row in settled)
    v11 = mean(row["v1_1_brier"] for row in settled)
    v10_submitted = mean(row["v1_0_submitted_brier"] for row in settled)
    v11_submitted = mean(row["v1_1_submitted_brier"] for row in settled)
    return {
        "settled_props": len(settled),
        "v1_0_mean_brier": v10,
        "v1_1_mean_brier": v11,
        "v1_1_minus_v1_0_brier": v11 - v10,
        "better_model_raw": "v1.1" if v11 < v10 else "v1.0" if v10 < v11 else "tie",
        "v1_0_mean_log_loss": mean(row["v1_0_log_loss"] for row in settled),
        "v1_1_mean_log_loss": mean(row["v1_1_log_loss"] for row in settled),
        "v1_0_submitted_mean_brier": v10_submitted,
        "v1_1_submitted_mean_brier": v11_submitted,
        "v1_1_minus_v1_0_submitted_brier": v11_submitted - v10_submitted,
        "v1_1_better_props": sum(row["v1_1_brier"] < row["v1_0_brier"] for row in settled),
        "v1_0_better_props": sum(row["v1_0_brier"] < row["v1_1_brier"] for row in settled),
        "tied_props": sum(row["v1_0_brier"] == row["v1_1_brier"] for row in settled),
    }


def _format_markdown(summary: dict[str, Any], actuals: list[dict[str, Any]]) -> str:
    overall = summary["overall"]
    lines = [
        "# v1.0 versus v1.1 actual-outcome comparison",
        "",
        f"- Matches in paired forecast set: {summary['matches_total']}",
        f"- Completed, non-cancelled matches: {summary['matches_scored']}",
        f"- Excluded/cancelled matches: {summary['matches_excluded']}",
        f"- Settled props: {overall['settled_props']}",
        f"- Overall winner: **{overall['better_model_raw']}**",
        "",
        "## Overall",
        "",
        "| Metric | v1.0 | v1.1 | v1.1 - v1.0 |",
        "|---|---:|---:|---:|",
        f"| Raw Brier | {overall['v1_0_mean_brier']:.6f} | {overall['v1_1_mean_brier']:.6f} | {overall['v1_1_minus_v1_0_brier']:+.6f} |",
        f"| Submitted Brier | {overall['v1_0_submitted_mean_brier']:.6f} | {overall['v1_1_submitted_mean_brier']:.6f} | {overall['v1_1_minus_v1_0_submitted_brier']:+.6f} |",
        f"| Log loss | {overall['v1_0_mean_log_loss']:.6f} | {overall['v1_1_mean_log_loss']:.6f} | {overall['v1_1_mean_log_loss'] - overall['v1_0_mean_log_loss']:+.6f} |",
        "",
        "## By prop family",
        "",
        "| Family | N | v1.0 Brier | v1.1 Brier | Difference | Winner |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind in sorted(summary["by_prop_kind"]):
        item = summary["by_prop_kind"][kind]
        lines.append(
            f"| {kind} | {item['settled_props']} | {item['v1_0_mean_brier']:.6f} | "
            f"{item['v1_1_mean_brier']:.6f} | {item['v1_1_minus_v1_0_brier']:+.6f} | "
            f"{item['better_model_raw']} |"
        )
    lines.extend(
        [
            "",
            "## Match-level prop bundles",
            "",
            "| Match | Settled | v1.0 Brier | v1.1 Brier | Difference | Winner |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for match_name in sorted(summary["by_match"]):
        item = summary["by_match"][match_name]
        lines.append(
            f"| {match_name} | {item['settled_props']} | {item['v1_0_mean_brier']:.6f} | "
            f"{item['v1_1_mean_brier']:.6f} | {item['v1_1_minus_v1_0_brier']:+.6f} | "
            f"{item['better_model_raw']} |"
        )
    unresolved = [item for item in actuals if not item["completed"] or item["cancelled"]]
    if unresolved:
        lines.extend(["", "## Excluded matches", ""])
        for item in unresolved:
            lines.append(f"- {item['match']}: {item['status'] or 'unknown status'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    jobs = manifest["jobs"]
    jobs_by_id = {str(item["official_match_id"]): item for item in jobs}
    with ThreadPoolExecutor(max_workers=NETWORK_WORKERS) as executor:
        captures = list(executor.map(_fetch, jobs))
    captures_by_id = {item["match_id"]: item for item in captures}
    actuals = [_actual_record(job, captures_by_id[str(job["official_match_id"])]) for job in jobs]
    actuals_by_id = {item["official_match_id"]: item for item in actuals}

    settled_rows: list[dict[str, Any]] = []
    with PAIRED.open("r", encoding="utf-8", newline="") as handle:
        for source_row in csv.DictReader(handle):
            match_id = source_row["official_match_id"]
            outcome, settlement_status = _settle(
                source_row,
                jobs_by_id[match_id],
                actuals_by_id[match_id],
            )
            row: dict[str, Any] = dict(source_row)
            row["cancelled"] = actuals_by_id[match_id]["cancelled"]
            row["true_outcome"] = outcome
            row["settlement_status"] = settlement_status
            if outcome is not None:
                p10 = float(source_row["v1_0_probability"])
                p11 = float(source_row["v1_1_probability"])
                s10 = float(source_row["v1_0_submit_percent"]) / 100.0
                s11 = float(source_row["v1_1_submit_percent"]) / 100.0
                row.update(
                    {
                        "v1_0_brier": (p10 - outcome) ** 2,
                        "v1_1_brier": (p11 - outcome) ** 2,
                        "v1_1_minus_v1_0_brier": (p11 - outcome) ** 2 - (p10 - outcome) ** 2,
                        "v1_0_log_loss": _log_loss(p10, outcome),
                        "v1_1_log_loss": _log_loss(p11, outcome),
                        "v1_0_submitted_brier": (s10 - outcome) ** 2,
                        "v1_1_submitted_brier": (s11 - outcome) ** 2,
                    }
                )
            else:
                for key in (
                    "v1_0_brier",
                    "v1_1_brier",
                    "v1_1_minus_v1_0_brier",
                    "v1_0_log_loss",
                    "v1_1_log_loss",
                    "v1_0_submitted_brier",
                    "v1_1_submitted_brier",
                ):
                    row[key] = None
            settled_rows.append(row)

    grouped_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in settled_rows:
        grouped_kind[row["prop_kind"]].append(row)
        grouped_match[row["match"]].append(row)
    summary = {
        "schema_version": "v1.0-v1.1-actual-comparison/v1",
        "source_event": "US Open 2026",
        "source_schedule_date": "2026-09-03",
        "duration_settlement_rule": "official displayed whole minute >= threshold settles Yes",
        "comparison_tie_rule": "No",
        "matches_total": len(actuals),
        "matches_scored": sum(item["completed"] and not item["cancelled"] for item in actuals),
        "matches_excluded": sum(item["cancelled"] or not item["completed"] for item in actuals),
        "overall": _summarize(settled_rows),
        "by_prop_kind": {key: _summarize(value) for key, value in grouped_kind.items()},
        "by_match": {key: _summarize(value) for key, value in grouped_match.items()},
    }

    _write_json(OUTPUT / "actual_results.json", actuals)
    _write_json(OUTPUT / "settled_paired_predictions.json", settled_rows)
    _write_json(OUTPUT / "comparison_summary.json", summary)
    fields = list(settled_rows[0])
    with (OUTPUT / "settled_paired_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(settled_rows)
    (OUTPUT / "comparison_report.md").write_text(
        _format_markdown(summary, actuals), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"COMPARISON_OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
