"""Replay all v1.1 US Open forecasts through frozen v1.0.

The replay preserves each v1.1 source capture, fixture, prop orientation,
threshold, path count, seed offset, worker count, and settlement policy.  The
only intended probability-affecting difference is the operational model bundle.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "tm-usopen"
ARTIFACT_ROOT = LOCAL / "v11-followup-stage-v1"
BASE_RUN_ID = "2edefbc0b1c8522b241d2b8305fc10b3d473df13b23fc063c6391876fa3d3664"
V10_OPERATIONAL_NAME = "official-2117-duration-v1"
V10_SPARSE_HISTORY_OPERATIONAL_NAME = (
    "official-2117-v1.0-counterfactual-sparse-history-local-v2"
)
SPARSE_HISTORY_JOBS = {"oliynykova-eala", "bu-zheng"}
LOCAL_OUTPUT = LOCAL / "outputs" / "v1.0-counterfactual-2026-09-03"
FINAL_OUTPUT = REPO / "outputs" / "v1.0-counterfactual-2026-09-03-final"
RUNNER = REPO / "scripts" / "simulate_usopen_match_batch.py"
CONCURRENT_MATCHES = 2
WORKERS_PER_MATCH = 2
CHECKPOINT_PATHS = 5


def job(slug: str, report: str, fixture: str, side: str, duration: int, tiebreaks: int) -> dict:
    return {
        "slug": slug,
        "v1_1_report": REPO / report,
        "fixture": REPO / "artifacts" / "live-usopen-2026" / fixture,
        "side": side,
        "duration_threshold": duration,
        "tiebreak_threshold": tiebreaks,
    }


JOBS = [
    job("blockx-trungelliti", "outputs/blockx-trungelliti-v1.1-repaired-100k-corrected/batch-report-840195f829af9e00.json", "round-two-five-fixture-v1.json", "left", 125, 1),
    job("popyrin-tabilo", "outputs/popyrin-tabilo-v1.1-repaired-100k/batch-report-840195f829af9e00.json", "round-two-five-fixture-v1.json", "left", 165, 2),
    job("darderi-svrcina", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/darderi-svrcina.json", "overnight-round-two-twelve-fixture-v1.json", "left", 190, 1),
    job("timofeeva-mertens", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/timofeeva-mertens.json", "overnight-round-two-twelve-fixture-v1.json", "right", 90, 1),
    job("potapova-jeanjean", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/potapova-jeanjean.json", "overnight-round-two-twelve-fixture-v1.json", "right", 95, 1),
    job("struff-cerundolo", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/struff-cerundolo.json", "overnight-round-two-twelve-fixture-v1.json", "right", 190, 3),
    job("keys-bondar", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/keys-bondar.json", "overnight-round-two-twelve-fixture-v1.json", "right", 85, 1),
    job("swiatek-podoroska", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/swiatek-podoroska.json", "overnight-round-two-twelve-fixture-v1.json", "right", 70, 1),
    job("auger-aliassime-khachanov", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/auger-aliassime-khachanov.json", "overnight-round-two-twelve-fixture-v1.json", "left", 175, 2),
    job("osaka-siniakova", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/osaka-siniakova.json", "overnight-round-two-twelve-fixture-v1.json", "left", 110, 1),
    job("dart-bouzkova", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/dart-bouzkova.json", "overnight-round-two-twelve-fixture-v1.json", "right", 90, 1),
    job("tagger-anisimova", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/tagger-anisimova.json", "overnight-round-two-twelve-fixture-v1.json", "right", 75, 1),
    job("bucsa-sakatsume", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/bucsa-sakatsume.json", "overnight-round-two-twelve-fixture-v1.json", "left", 95, 1),
    job("mensik-rodionov", "outputs/overnight-round-two-twelve-v1.1-100k-final/reports/mensik-rodionov.json", "overnight-round-two-twelve-fixture-v1.json", "left", 135, 2),
    job("oliynykova-eala", "outputs/round-two-followup-five-v1.1-100k-final/reports/oliynykova-eala.json", "round-two-followup-five-fixture-v1.json", "right", 90, 1),
    job("sakkari-starodubtseva", "outputs/round-two-followup-five-v1.1-100k-final/reports/sakkari-starodubtseva.json", "round-two-followup-five-fixture-v1.json", "left", 110, 1),
    job("monfils-tien", "outputs/round-two-followup-five-v1.1-100k-final/reports/monfils-tien.json", "round-two-followup-five-fixture-v1.json", "left", 170, 2),
    job("schoolkate-cobolli", "outputs/round-two-followup-five-v1.1-100k-final/reports/schoolkate-cobolli.json", "round-two-followup-five-fixture-v1.json", "right", 175, 1),
    job("bouzas-maneiro-rybakina", "outputs/round-two-followup-five-v1.1-100k-final/reports/bouzas-maneiro-rybakina.json", "round-two-followup-five-fixture-v1.json", "left", 75, 1),
    job("andreeva-lys", "outputs/round-two-next-four-v1.1-100k-final/reports/andreeva-lys.json", "round-two-next-four-fixture-v1.json", "left", 80, 1),
    job("sweeny-musetti", "outputs/round-two-next-four-v1.1-100k-final/reports/sweeny-musetti.json", "round-two-next-four-fixture-v1.json", "right", 135, 1),
    job("zheng-putintseva", "outputs/round-two-next-four-v1.1-100k-final/reports/zheng-putintseva.json", "round-two-next-four-fixture-v1.json", "right", 105, 1),
    job("fritz-bellucci", "outputs/round-two-next-four-v1.1-100k-final/reports/fritz-bellucci.json", "round-two-next-four-fixture-v1.json", "right", 125, 2),
    job("bartunkova-maria", "outputs/bartunkova-maria-v1.1-100k-final/reports/bartunkova-maria.json", "bartunkova-maria-round-two-fixture-v1.json", "left", 110, 1),
    job("svajda-gea", "outputs/round-two-next-five-v1.1-100k-final/reports/svajda-gea.json", "round-two-next-five-fixture-v1.json", "right", 180, 2),
    job("bergs-de-jong", "outputs/round-two-next-five-v1.1-100k-final/reports/bergs-de-jong.json", "round-two-next-five-fixture-v1.json", "left", 170, 1),
    job("bu-zheng", "outputs/round-two-next-five-v1.1-100k-final/reports/bu-zheng.json", "round-two-next-five-fixture-v1.json", "left", 155, 1),
    job("bonzi-buse", "outputs/round-two-next-five-v1.1-100k-final/reports/bonzi-buse.json", "round-two-next-five-fixture-v1.json", "left", 165, 2),
    job("jovic-jones", "outputs/round-two-next-five-v1.1-100k-final/reports/jovic-jones.json", "round-two-next-five-fixture-v1.json", "right", 105, 1),
    job("zandschulp-de-minaur", "outputs/zandschulp-de-minaur-v1.1-100k-final/reports/zandschulp-de-minaur.json", "zandschulp-de-minaur-round-two-fixture-v1.json", "right", 140, 1),
    job("badosa-gauff", "outputs/badosa-gauff-zverev-halys-v1.1-100k-final/reports/badosa-gauff.json", "badosa-gauff-zverev-halys-round-two-fixture-v1.json", "left", 95, 1),
    job("zverev-halys", "outputs/badosa-gauff-zverev-halys-v1.1-100k-final/reports/zverev-halys.json", "badosa-gauff-zverev-halys-round-two-fixture-v1.json", "right", 150, 2),
]


PROP_ORDER = {
    "DURATION_MIN": 1,
    "ACE_COMPARE": 2,
    "DF_COMPARE": 3,
    "MATCH_WIN": 4,
    "TIEBREAK_COUNT": 5,
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def probability(prop: dict) -> float:
    if prop["prop"]["kind"] == "DURATION_MIN":
        value = prop.get("sensitivity_high")
        if value is not None:
            return float(value)
    return float(prop["probability_raw"])


def submitted_integer(value: float) -> int:
    return min(99, max(1, math.floor(value * 100.0 + 0.5)))


def find_capture(capture_id: str) -> Path:
    candidates = []
    for root in LOCAL.glob("source-captures*"):
        candidate = root / capture_id
        if candidate.is_dir():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"source capture {capture_id} not found under {LOCAL}")
    return sorted(candidates)[0]


def schedule_date(match: dict) -> str:
    value = match["scheduled_start_utc"].replace("Z", "+00:00")
    from datetime import datetime

    return datetime.fromisoformat(value).astimezone(ZoneInfo("America/New_York")).date().isoformat()


def prepare_manifest() -> list[dict]:
    prepared = []
    for spec in JOBS:
        report = load_json(spec["v1_1_report"])
        if len(report.get("matches", [])) != 1:
            raise ValueError(f"expected one match in {spec['v1_1_report']}")
        match = report["matches"][0]
        capture_id = report["official_source_capture_id"]
        paths = int(match["paths"])
        if paths != 100000:
            raise ValueError(f"{spec['slug']} has {paths} v1.1 paths, expected 100000")
        original = {p["prop"]["kind"]: p for p in match["prop_estimates"]}
        if set(PROP_ORDER) - set(original):
            raise ValueError(f"{spec['slug']} is missing a required prop")
        item = dict(spec)
        item.update(
            {
                "official_match_id": str(match["official_match_id"]),
                "match": match["match"],
                "tour": match["tour"],
                "paths": paths,
                "source_capture_id": capture_id,
                "source_capture": find_capture(capture_id),
                "schedule_date": schedule_date(match),
                "operational_name": (
                    V10_SPARSE_HISTORY_OPERATIONAL_NAME
                    if spec["slug"] in SPARSE_HISTORY_JOBS
                    else V10_OPERATIONAL_NAME
                ),
                "v1_1_probabilities": {kind: probability(prop) for kind, prop in original.items()},
            }
        )
        prepared.append(item)
    return prepared


def serializable_manifest(jobs: list[dict]) -> dict:
    fields = (
        "slug",
        "match",
        "tour",
        "official_match_id",
        "paths",
        "side",
        "duration_threshold",
        "tiebreak_threshold",
        "schedule_date",
        "source_capture_id",
        "operational_name",
        "source_capture",
        "fixture",
        "v1_1_report",
        "v1_1_probabilities",
    )
    return {
        "schema_version": "v1.0-counterfactual-replay-manifest/v1",
        "base_run_id": BASE_RUN_ID,
        "v1_0_operational_name": V10_OPERATIONAL_NAME,
        "total_worker_ceiling": CONCURRENT_MATCHES * WORKERS_PER_MATCH,
        "concurrent_matches": CONCURRENT_MATCHES,
        "workers_per_match": WORKERS_PER_MATCH,
        "checkpoint_paths": CHECKPOINT_PATHS,
        "duration_boundary_convention": "exact official whole-minute boundary settles YES",
        "jobs": [{key: str(item[key]) if isinstance(item[key], Path) else item[key] for key in fields} for item in jobs],
    }


def run_job(item: dict, index: int, total: int) -> Path:
    output_slug = item["slug"]
    if item["operational_name"] == V10_SPARSE_HISTORY_OPERATIONAL_NAME:
        output_slug = f"{output_slug}-sparse-history-v2"
    output = LOCAL_OUTPUT / output_slug
    existing = sorted(output.glob("batch-report-*.json"))
    if existing:
        print(f"[{index}/{total}] reuse {item['match']}", flush=True)
        return existing[-1]

    day_number = 4 if item["schedule_date"] == "2026-09-02" else 5
    command = [
        sys.executable,
        str(RUNNER),
        "--fixture-file",
        str(item["fixture"]),
        "--match-id",
        item["official_match_id"],
        "--output",
        str(output),
        "--artifact-root",
        str(ARTIFACT_ROOT),
        "--base-run-id",
        BASE_RUN_ID,
        "--operational-name",
        item["operational_name"],
        "--source-capture",
        str(item["source_capture"]),
        "--policy",
        "fixed",
        "--workers",
        str(WORKERS_PER_MATCH),
        "--checkpoint-paths",
        str(CHECKPOINT_PATHS),
        "--round",
        "R64",
        "--schedule-date",
        item["schedule_date"],
        "--schedule-source-id",
        f"usopen-2026-day{day_number}",
        "--match-win-side",
        item["side"],
        "--comparison-side",
        item["side"],
        "--duration-threshold",
        str(item["duration_threshold"]),
        "--tiebreak-threshold",
        str(item["tiebreak_threshold"]),
    ]
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    print(f"[{index}/{total}] start {item['match']}", flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    reports = sorted(output.glob("batch-report-*.json"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one report for {item['slug']}, found {len(reports)}")
    print(f"[{index}/{total}] complete {item['match']}", flush=True)
    return reports[0]


def export(jobs: list[dict], reports: list[Path]) -> None:
    report_output = FINAL_OUTPUT / "reports"
    report_output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, report_path in zip(jobs, reports, strict=True):
        report = load_json(report_path)
        match = report["matches"][0]
        props = {p["prop"]["kind"]: p for p in match["prop_estimates"]}
        shutil.copy2(report_path, report_output / f"{item['slug']}.json")
        for kind, order in sorted(PROP_ORDER.items(), key=lambda pair: pair[1]):
            prop = props[kind]
            v10 = probability(prop)
            v11 = float(item["v1_1_probabilities"][kind])
            rows.append(
                {
                    "match": item["match"],
                    "official_match_id": item["official_match_id"],
                    "question_order": order,
                    "prop_kind": kind,
                    "prop": prop["prop"]["original_text"],
                    "v1_0_probability": v10,
                    "v1_1_probability": v11,
                    "v1_1_minus_v1_0": v11 - v10,
                    "v1_0_submit_percent": submitted_integer(v10),
                    "v1_1_submit_percent": submitted_integer(v11),
                    "true_outcome": None,
                    "cancelled": None,
                    "source_capture_id": item["source_capture_id"],
                    "paths": item["paths"],
                }
            )

    payload = {
        "schema_version": "paired-v1.0-v1.1-counterfactual-predictions/v1",
        "match_count": len(jobs),
        "prop_count": len(rows),
        "duration_boundary_convention": "exact official whole-minute boundary settles YES",
        "predictions": rows,
    }
    with (FINAL_OUTPUT / "paired_predictions.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with (FINAL_OUTPUT / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = prepare_manifest()
    manifest = serializable_manifest(jobs)
    with (FINAL_OUTPUT / "replay_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"preflight complete: {len(jobs)} matches, {len(jobs) * 5} props", flush=True)
    reports: list[Path | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=CONCURRENT_MATCHES) as executor:
        futures = {
            executor.submit(run_job, item, index, len(jobs)): index - 1
            for index, item in enumerate(jobs, start=1)
        }
        for future in as_completed(futures):
            reports[futures[future]] = future.result()
    if any(report is None for report in reports):
        raise RuntimeError("counterfactual replay completed without every report")
    export(jobs, [report for report in reports if report is not None])
    print(f"COUNTERFACTUAL_OUTPUT={FINAL_OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
