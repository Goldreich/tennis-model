"""Pre-registered rolling-origin comparison of frozen v1.0 and v1.1 candidates.

This module is intentionally separate from production locking. It consumes pinned
historical bytes, writes immutable comparative artifacts, and never changes the active
production framework.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from tennis_model import exact_probability
from tennis_model.data.component_counts import build_serve_component_counts
from tennis_model.data.validate import validate_score
from tennis_model.estimation.config import load_serve_model_config
from tennis_model.estimation.serve_components import (
    FitProvenance,
    FutureMatchContext,
    ServeComponent,
    fit_input_set_sha256,
    predict_serve_performance,
    fit_all_serve_components,
)
from tennis_model.estimation.strength import (
    DynamicStrengthConfig,
    StrengthOutcomeRecord,
    StrengthPrediction,
    fit_dynamic_strength,
    predict_strength,
)
from tennis_model.estimation.strength_integration import (
    CrossFittedStrengthRecord,
    StrengthIntegrationConfig,
    fit_strength_integration,
    prepare_strength_match_parameters,
    solve_q_tilt,
)
from tennis_model.schemas import Tour
from tennis_model.serve import PrimitiveServeMeans
from tennis_model.simulation.parameters import ServePerformanceDraw
from tennis_model.simulation.point import generate_service_point
from tennis_model.simulation.scoring import award_point, new_match, tiebreak_server


VARIANTS = (
    "v1_0",
    "temperature_only",
    "surface_elo",
    "dynamic_constant",
    "dynamic_gated",
    "complete_v1_1",
    "ranking_prior_ablation",
    "level_change_ablation",
    "head_to_head_ablation",
)

PROTECTED_VARIANTS = ("v1_0", "complete_v1_1")
PRIMITIVE_NAMES = (
    "first_serve_in",
    "ace_given_first_in",
    "returnable_first_win",
    "double_fault_given_second_opp",
    "playable_second_win",
)
COMPONENT_FOR_PRIMITIVE = dict(zip(PRIMITIVE_NAMES, ("F", "A", "Q1", "D", "Q2")))
EPS = 1e-12


@dataclass(frozen=True)
class RunPaths:
    root: Path
    cache: Path
    forecasts: Path
    figures: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    payload = "\x1f".join(str(item) for item in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 1e-9, 1.0 - 1e-9))


def _logit(value: float) -> float:
    return float(logit(_clip_probability(value)))


def _bounded_instability(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("component instability must be finite and nonnegative")
    return float(value / (1.0 + value))


def _utc(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("backtest configuration must be frozen before evaluation")
    return payload


def _run_paths(config: Mapping[str, Any], repo_root: Path) -> RunPaths:
    root = repo_root / str(config["output_directory"])
    return RunPaths(
        root=root,
        cache=root / "cache",
        forecasts=root / "forecast_parts",
        figures=root / "figures",
    )


def _source_manifest(source_dir: Path) -> list[dict[str, Any]]:
    path = source_dir / "source_manifest.json"
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(records) != 18:
        raise ValueError(f"expected 18 pinned source members, found {len(records)}")
    for record in records:
        file_path = source_dir / f"{str(record['tour']).lower()}_matches_{record['year']}.csv"
        actual = _sha256_file(file_path)
        if actual != record["sha256"]:
            raise ValueError(f"source hash mismatch for {file_path}: {actual}")
    return records


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _score_status(score: Any, best_of: Any) -> tuple[bool, str | None]:
    text = "" if pd.isna(score) else str(score).strip()
    upper = text.upper()
    for token, reason in (
        ("W/O", "WALKOVER"),
        ("RET", "RETIREMENT"),
        ("DEF", "DEFAULT"),
        ("ABD", "ABANDONED"),
        ("BYE", "BYE"),
    ):
        if token in upper:
            return False, reason
    try:
        result = validate_score(text or None, best_of=int(best_of))
    except (TypeError, ValueError):
        return False, "INVALID_SCORE"
    if not result.valid or not result.completed:
        return False, "INVALID_OR_INCOMPLETE_SCORE"
    return True, None


def prepare_historical_data(
    config: Mapping[str, Any], repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load pinned rows, assign cutoffs, and construct frozen component counts."""

    source_dir = repo_root / str(config["source_directory"])
    manifest = _source_manifest(source_dir)
    hash_by_file = {
        f"{str(item['tour']).lower()}_matches_{item['year']}.csv": item["sha256"]
        for item in manifest
    }
    frames: list[pd.DataFrame] = []
    for tour in ("ATP", "WTA"):
        for year in range(2017, 2026):
            name = f"{tour.lower()}_matches_{year}.csv"
            frame = pd.read_csv(source_dir / name, dtype=str, keep_default_na=False)
            frame["tour"] = tour
            frame["source_year"] = year
            frame["source_file"] = name
            frame["source_sha256"] = hash_by_file[name]
            frame["source_row_number"] = np.arange(2, len(frame) + 2)
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True, sort=False)

    for column in (
        "tourney_date",
        "best_of",
        "winner_id",
        "loser_id",
        "winner_rank",
        "loser_rank",
        "w_svpt",
        "l_svpt",
        "w_1stIn",
        "l_1stIn",
        "w_1stWon",
        "l_1stWon",
        "w_2ndWon",
        "l_2ndWon",
        "w_ace",
        "l_ace",
        "w_df",
        "l_df",
        "w_SvGms",
        "l_SvGms",
        "minutes",
    ):
        raw[f"num_{column}"] = _numeric(raw[column])

    raw["best_of_int"] = raw["num_best_of"].astype("Int64")

    raw["event_start_date"] = pd.to_datetime(
        raw["tourney_date"], format="%Y%m%d", errors="coerce", utc=True
    )
    raw["available_at_utc"] = raw["event_start_date"] + pd.to_timedelta(
        int(config["result_embargo_days"]), unit="D"
    )
    raw["forecast_cutoff_utc"] = raw["event_start_date"].dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    raw["event_id"] = [
        _stable_id(tour, event_date, event, tournament_id)
        for tour, event_date, event, tournament_id in zip(
            raw["tour"], raw["tourney_date"], raw["tourney_name"], raw["tourney_id"]
        )
    ]
    raw["match_id"] = [
        _stable_id(tour, tournament_id, match_num, winner, loser)
        for tour, tournament_id, match_num, winner, loser in zip(
            raw["tour"],
            raw["tourney_id"],
            raw["match_num"],
            raw["winner_id"],
            raw["loser_id"],
        )
    ]
    raw["winner_player_id"] = raw["tour"] + ":" + raw["winner_id"].astype(str)
    raw["loser_player_id"] = raw["tour"] + ":" + raw["loser_id"].astype(str)
    raw["player_a_id"] = raw[["winner_player_id", "loser_player_id"]].min(axis=1)
    raw["player_b_id"] = raw[["winner_player_id", "loser_player_id"]].max(axis=1)
    raw["player_a_won"] = (raw["player_a_id"] == raw["winner_player_id"]).astype(int)
    raw["player_a_rank"] = np.where(
        raw["player_a_won"].eq(1), raw["num_winner_rank"], raw["num_loser_rank"]
    )
    raw["player_b_rank"] = np.where(
        raw["player_a_won"].eq(1), raw["num_loser_rank"], raw["num_winner_rank"]
    )

    allowed_levels = {
        tour: set(values) for tour, values in config["levels"].items()
    }
    reasons: list[str | None] = []
    score_ok: list[bool] = []
    for row in raw.itertuples(index=False):
        reason: str | None = None
        if pd.isna(row.event_start_date):
            reason = "MISSING_EVENT_START_DATE"
        elif not row.winner_id or not row.loser_id or row.winner_id == row.loser_id:
            reason = "INVALID_PLAYER_IDENTITY"
        elif row.surface not in config["surfaces"]:
            reason = "UNSUPPORTED_SURFACE"
        elif row.tourney_level not in allowed_levels[row.tour]:
            reason = "EXCLUDED_COMPETITION_LEVEL"
        elif row.best_of_int not in (3, 5):
            reason = "INVALID_BEST_OF"
        else:
            valid, score_reason = _score_status(row.score, row.best_of_int)
            if not valid:
                reason = score_reason
        reasons.append(reason)
        score_ok.append(reason is None)
    raw["outcome_exclusion_reason"] = reasons
    raw["outcome_eligible"] = score_ok

    stat_columns = (
        "svpt",
        "1stIn",
        "1stWon",
        "2ndWon",
        "ace",
        "df",
    )
    protected_complete = np.ones(len(raw), dtype=bool)
    accounting_ok = np.ones(len(raw), dtype=bool)
    for prefix in ("w", "l"):
        protected_complete &= raw[[f"num_{prefix}_{name}" for name in stat_columns]].notna().all(axis=1)
        sv = raw[f"num_{prefix}_svpt"]
        first = raw[f"num_{prefix}_1stIn"]
        first_won = raw[f"num_{prefix}_1stWon"]
        second_won = raw[f"num_{prefix}_2ndWon"]
        ace = raw[f"num_{prefix}_ace"]
        df = raw[f"num_{prefix}_df"]
        accounting_ok &= (
            (first <= sv)
            & (ace <= first)
            & (first_won >= ace)
            & (first_won <= first)
            & (df <= sv - first)
            & (second_won <= sv - first - df)
            & (sv >= 0)
            & (first >= 0)
            & (ace >= 0)
            & (df >= 0)
        ).fillna(False)
    raw["protected_complete"] = protected_complete
    raw["component_accounting_ok"] = accounting_ok
    raw["evaluation_eligible"] = (
        raw["outcome_eligible"] & protected_complete & accounting_ok
    )

    excluded = raw.loc[~raw["evaluation_eligible"], [
        "match_id", "tour", "source_year", "tourney_name", "event_id", "surface",
        "tourney_level", "score", "outcome_exclusion_reason", "protected_complete",
        "component_accounting_ok", "source_file", "source_row_number",
    ]].copy()
    excluded["exclusion_reason"] = excluded["outcome_exclusion_reason"]
    excluded.loc[
        excluded["exclusion_reason"].isna() & ~excluded["protected_complete"],
        "exclusion_reason",
    ] = "MISSING_PROTECTED_SERVICE_STATS"
    excluded.loc[
        excluded["exclusion_reason"].isna() & ~excluded["component_accounting_ok"],
        "exclusion_reason",
    ] = "PRIMITIVE_ACCOUNTING_ANOMALY"

    service_records: list[dict[str, Any]] = []
    training = raw.loc[raw["outcome_eligible"]]
    for row in training.itertuples(index=False):
        for orientation, prefix, player_id, opponent_id, player_hand, opponent_hand in (
            (
                "winner", "w", row.winner_player_id, row.loser_player_id,
                row.winner_hand or None, row.loser_hand or None,
            ),
            (
                "loser", "l", row.loser_player_id, row.winner_player_id,
                row.loser_hand or None, row.winner_hand or None,
            ),
        ):
            service_records.append(
                {
                    "snapshot_id": row.source_sha256,
                    "snapshot_sha256": row.source_sha256,
                    "source_id": row.source_file,
                    "source_url": f"pinned://{row.source_file}",
                    "source_schema_version": "sackmann-match-csv/v1",
                    "retrieved_at_utc": datetime(2026, 9, 1, tzinfo=UTC),
                    "normalization_version": "v1.1-backtest-event-embargo/v1",
                    "source_row_number": int(row.source_row_number),
                    "orientation": orientation,
                    "match_id": row.match_id,
                    "source_date": row.event_start_date.date(),
                    "match_date": row.event_start_date.date(),
                    "match_date_source_id": None,
                    "match_date_source_sha256": None,
                    "match_date_crosswalk_id": None,
                    "event_start_date": row.event_start_date.date(),
                    "source_date_semantics": "EVENT_START_DATE",
                    "available_at_utc": row.available_at_utc.to_pydatetime(),
                    "tour": row.tour,
                    "event": row.tourney_name,
                    "event_year": int(row.source_year),
                    "level": row.tourney_level,
                    "round": row.round,
                    "surface": row.surface,
                    "indoor": None,
                    "best_of": int(row.best_of_int),
                    "player_id": player_id,
                    "opponent_id": opponent_id,
                    "player_hand": player_hand,
                    "opponent_hand": opponent_hand,
                    "service_points": getattr(row, f"num_{prefix}_svpt"),
                    "first_serves_in": getattr(row, f"num_{prefix}_1stIn"),
                    "first_serve_points_won": getattr(row, f"num_{prefix}_1stWon"),
                    "second_serve_points_won": getattr(row, f"num_{prefix}_2ndWon"),
                    "aces": getattr(row, f"num_{prefix}_ace"),
                    "double_faults": getattr(row, f"num_{prefix}_df"),
                    "invalid_stat_fields": (),
                    "raw_record_json": "{}",
                }
            )
    component_table = build_serve_component_counts(pd.DataFrame.from_records(service_records))
    counts = component_table.counts
    counts["available_at_utc"] = pd.to_datetime(counts["available_at_utc"], utc=True)
    counts["match_date"] = pd.to_datetime(counts["match_date"], utc=True)
    counts["event_start_date"] = pd.to_datetime(counts["event_start_date"], utc=True)

    raw["observed_a_aces"] = np.where(raw["player_a_won"].eq(1), raw["num_w_ace"], raw["num_l_ace"])
    raw["observed_b_aces"] = np.where(raw["player_a_won"].eq(1), raw["num_l_ace"], raw["num_w_ace"])
    raw["observed_a_df"] = np.where(raw["player_a_won"].eq(1), raw["num_w_df"], raw["num_l_df"])
    raw["observed_b_df"] = np.where(raw["player_a_won"].eq(1), raw["num_l_df"], raw["num_w_df"])
    raw["ace_compare_outcome"] = (raw["observed_a_aces"] > raw["observed_b_aces"]).astype(int)
    raw["df_compare_outcome"] = (raw["observed_a_df"] > raw["observed_b_df"]).astype(int)
    return raw, counts, excluded, manifest


def _training_hash(counts: pd.DataFrame) -> str:
    ordered = counts.sort_values(["match_id", "player_id", "component"])
    payload = ordered[["match_id", "player_id", "component", "successes", "trials"]].to_csv(
        index=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _fit_v1_bundle(
    counts: pd.DataFrame,
    *,
    tour: str,
    cutoff: datetime,
    source_hash: str,
    code_hash: str,
    repo_root: Path,
) -> Mapping[ServeComponent, Any]:
    config = load_serve_model_config(repo_root / "config/model_v1.yaml")
    selected = counts.loc[
        counts["tour"].eq(tour)
        & (counts["available_at_utc"] < cutoff)
        & (counts["event_start_date"] >= cutoff - timedelta(days=1095))
    ].copy()
    training_hash = _training_hash(selected)
    snapshot_set_hash = fit_input_set_sha256(
        "source_snapshots", selected["snapshot_sha256"].dropna().astype(str).unique()
    )
    provenance = FitProvenance(
        data_snapshot_sha256=snapshot_set_hash,
        component_count_artifact_sha256=training_hash,
        code_commit=code_hash,
        fitted_at_utc=cutoff,
    )
    return fit_all_serve_components(
        selected,
        tour=Tour(tour),
        cutoff=cutoff,
        config=config,
        provenance=provenance,
    )


def _means_and_uncertainty(performance: Any) -> tuple[PrimitiveServeMeans, dict[str, Any]]:
    predictions = {
        "first_serve_in": performance.first_serve_in,
        "ace_given_first_in": performance.ace_given_first_in,
        "returnable_first_win": performance.returnable_first_win,
        "double_fault_given_second_opp": performance.double_fault_given_second_opp,
        "playable_second_win": performance.playable_second_win,
    }
    means = PrimitiveServeMeans(**{name: float(item.map_mean) for name, item in predictions.items()})
    uncertainty = {
        name: {
            "mean": float(item.map_mean),
            "linear_sd": float(item.linear_predictor_sd),
            "concentration": float(item.predictive_concentration),
            "serving_seen": bool(item.serving_player_seen),
            "returning_seen": bool(item.returning_player_seen),
        }
        for name, item in predictions.items()
    }
    return means, uncertainty


def _service_point_win(means: PrimitiveServeMeans) -> float:
    return float(
        means.first_serve_in
        * (
            means.ace_given_first_in
            + (1.0 - means.ace_given_first_in) * means.returnable_first_win
        )
        + (1.0 - means.first_serve_in)
        * (1.0 - means.double_fault_given_second_opp)
        * means.playable_second_win
    )


def _primitive_json(means: PrimitiveServeMeans) -> str:
    return json.dumps(asdict(means), sort_keys=True, separators=(",", ":"))


def _predict_component_match(bundle: Mapping[ServeComponent, Any], row: Any, cutoff: datetime) -> dict[str, Any]:
    common = dict(
        tour=Tour(row.tour),
        surface=row.surface,
        indoor=False,
        event=row.tourney_name,
        event_year=int(row.source_year),
        match_date_utc=row.event_start_date.to_pydatetime(),
        information_cutoff_utc=cutoff,
        best_of=int(row.best_of_int),
    )
    a_perf = predict_serve_performance(
        bundle,
        FutureMatchContext(
            serving_player_id=row.player_a_id,
            returning_player_id=row.player_b_id,
            serving_hand=None,
            returning_hand=None,
            **common,
        ),
    )
    b_perf = predict_serve_performance(
        bundle,
        FutureMatchContext(
            serving_player_id=row.player_b_id,
            returning_player_id=row.player_a_id,
            serving_hand=None,
            returning_hand=None,
            **common,
        ),
    )
    a, a_unc = _means_and_uncertainty(a_perf)
    b, b_unc = _means_and_uncertainty(b_perf)
    p = exact_probability.exact_match_win_probability(
        _service_point_win(a), _service_point_win(b), best_of=int(row.best_of_int)
    )
    q_sds = [
        a_unc["returnable_first_win"]["linear_sd"],
        a_unc["playable_second_win"]["linear_sd"],
        b_unc["returnable_first_win"]["linear_sd"],
        b_unc["playable_second_win"]["linear_sd"],
    ]
    return {
        "component_probability": float(p),
        "component_logit": _logit(p),
        "component_variance": float(np.mean(np.square(q_sds))),
        "component_instability": float(np.max(q_sds)),
        "primitive_a": _primitive_json(a),
        "primitive_b": _primitive_json(b),
        "primitive_uncertainty_a": json.dumps(a_unc, sort_keys=True),
        "primitive_uncertainty_b": json.dumps(b_unc, sort_keys=True),
    }


def _strength_records(matches: pd.DataFrame) -> tuple[StrengthOutcomeRecord, ...]:
    records: list[StrengthOutcomeRecord] = []
    for row in matches.loc[matches["outcome_eligible"]].itertuples(index=False):
        records.append(
            StrengthOutcomeRecord(
                match_id=row.match_id,
                tour=Tour(row.tour),
                player_a_id=row.player_a_id,
                player_b_id=row.player_b_id,
                winner_id=row.player_a_id if row.player_a_won else row.player_b_id,
                start_utc=row.event_start_date.to_pydatetime(),
                available_at_utc=row.available_at_utc.to_pydatetime(),
                surface=row.surface,
                best_of=int(row.best_of_int),
                player_a_rank=None if pd.isna(row.player_a_rank) else int(row.player_a_rank),
                player_b_rank=None if pd.isna(row.player_b_rank) else int(row.player_b_rank),
            )
        )
    return tuple(records)


def _dynamic_configs(config: Mapping[str, Any]) -> dict[str, DynamicStrengthConfig]:
    block = config["dynamic_anchor"]

    def make(
        decay: float,
        process: float,
        *,
        ranking: bool = False,
        h2h: bool = False,
    ) -> DynamicStrengthConfig:
        return DynamicStrengthConfig(
            schema_version="dynamic-strength-config/v1",
            decay_days=decay,
            process_sd=process,
            initial_sd=block["initial_sd"],
            surface_sd=block["surface_sd"],
            observation_scale_best_of_3=block["observation_scale_best_of_3"],
            observation_scale_best_of_5=block["observation_scale_best_of_5"],
            variance_floor=block["variance_floor"],
            ranking_sparse_prior_enabled=ranking,
            ranking_intercept=0.0,
            ranking_log_slope=block["ranking_log_slope"],
            ranking_prior_sd=block["ranking_prior_sd"],
            head_to_head_enabled=h2h,
            head_to_head_prior_sd=block["head_to_head_prior_sd"],
        )

    result = {
        f"core_d{int(decay)}_p{process:.2f}": make(decay, process)
        for decay in block["decay_days"]
        for process in block["process_sd"]
    }
    complete_decay = block["complete_decay_days"]
    complete_process = block["complete_process_sd"]
    result.update(
        {
            "complete": make(complete_decay, complete_process, ranking=True, h2h=True),
            "no_ranking": make(complete_decay, complete_process, ranking=False, h2h=True),
            "no_h2h": make(complete_decay, complete_process, ranking=True, h2h=False),
            "no_level": make(
                block["no_level_change_decay_days"],
                block["no_level_change_process_sd"],
                ranking=True,
                h2h=True,
            ),
        }
    )
    return result


def _fit_anchor_predictions(
    records: tuple[StrengthOutcomeRecord, ...],
    target: pd.DataFrame,
    *,
    cutoff: datetime,
    tour: str,
    configs: Mapping[str, DynamicStrengthConfig],
    code_hash: str,
) -> dict[str, list[dict[str, Any]]]:
    eligible_records = tuple(
        item for item in records if item.tour.value == tour and item.available_at_utc < cutoff
    )
    outputs: dict[str, list[dict[str, Any]]] = {}
    for config_id, dynamic_config in configs.items():
        fit = fit_dynamic_strength(
            eligible_records,
            tour=Tour(tour),
            cutoff_utc=cutoff,
            fitted_at_utc=cutoff,
            config=dynamic_config,
            code_commit=code_hash,
        )
        predictions: list[dict[str, Any]] = []
        for row in target.itertuples(index=False):
            prediction = predict_strength(
                fit,
                player_a_id=row.player_a_id,
                player_b_id=row.player_b_id,
                surface=row.surface,
                best_of=int(row.best_of_int),
                scheduled_start_utc=row.event_start_date.to_pydatetime(),
                player_a_rank=None if pd.isna(row.player_a_rank) else int(row.player_a_rank),
                player_b_rank=None if pd.isna(row.player_b_rank) else int(row.player_b_rank),
            )
            predictions.append(
                {
                    f"anchor_{config_id}_logit": prediction.mean_logit,
                    f"anchor_{config_id}_probability": prediction.probability,
                    f"anchor_{config_id}_variance": prediction.variance_logit,
                    f"anchor_{config_id}_weak": prediction.weakly_connected,
                    f"anchor_{config_id}_a_known": prediction.player_a_known,
                    f"anchor_{config_id}_b_known": prediction.player_b_known,
                    f"anchor_{config_id}_a_graph": prediction.player_a_graph_component,
                    f"anchor_{config_id}_b_graph": prediction.player_b_graph_component,
                }
            )
        outputs[config_id] = predictions
    return outputs


def _elo_predictions(
    history: pd.DataFrame,
    target: pd.DataFrame,
    *,
    cutoff: datetime,
    k: float,
    surface_blend: float,
) -> list[float]:
    selected = history.loc[
        history["outcome_eligible"] & (history["available_at_utc"] < cutoff)
    ].sort_values(["event_start_date", "event_id", "match_num", "match_id"])
    global_rating: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    surface_rating: defaultdict[tuple[str, str], float] = defaultdict(lambda: 1500.0)
    for row in selected.itertuples(index=False):
        a, b = row.player_a_id, row.player_b_id
        ra = (1.0 - surface_blend) * global_rating[a] + surface_blend * surface_rating[(a, row.surface)]
        rb = (1.0 - surface_blend) * global_rating[b] + surface_blend * surface_rating[(b, row.surface)]
        expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        delta = k * (float(row.player_a_won) - expected)
        global_rating[a] += delta
        global_rating[b] -= delta
        surface_rating[(a, row.surface)] += delta
        surface_rating[(b, row.surface)] -= delta
    result: list[float] = []
    for row in target.itertuples(index=False):
        ra = (1.0 - surface_blend) * global_rating[row.player_a_id] + surface_blend * surface_rating[(row.player_a_id, row.surface)]
        rb = (1.0 - surface_blend) * global_rating[row.player_b_id] + surface_blend * surface_rating[(row.player_b_id, row.surface)]
        result.append(float(math.log(10.0) * (ra - rb) / 400.0))
    return result


def generate_cross_fitted_forecasts(
    config: Mapping[str, Any],
    repo_root: Path,
    paths: RunPaths,
    matches: pd.DataFrame,
    counts: pd.DataFrame,
    manifest: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Generate reusable monthly OOF component, anchor, and Elo predictions."""

    paths.forecasts.mkdir(parents=True, exist_ok=True)
    code_hash = _sha256_file(Path(__file__))
    source_hash = _sha256_bytes(
        json.dumps(list(manifest), sort_keys=True, default=str).encode("utf-8")
    )
    strength_records = _strength_records(matches)
    dynamic_configs = _dynamic_configs(config)
    forecast_years = config.get(
        "_runtime_forecast_years",
        range(min(config["outer_years"]) - 2, max(config["outer_years"]) + 1),
    )
    part_paths: list[Path] = []
    for tour in config["tours"]:
        tour_history = matches.loc[matches["tour"].eq(tour)]
        for year in forecast_years:
            for month in range(1, 13):
                cutoff = datetime(year, month, 1, tzinfo=UTC)
                part_path = paths.forecasts / f"{tour.lower()}-{year:04d}-{month:02d}.parquet"
                part_paths.append(part_path)
                if part_path.exists():
                    continue
                target = matches.loc[
                    matches["evaluation_eligible"]
                    & matches["tour"].eq(tour)
                    & matches["forecast_cutoff_utc"].eq(pd.Timestamp(cutoff))
                ].copy()
                if target.empty:
                    pd.DataFrame().to_parquet(part_path, index=False)
                    continue
                started = time.perf_counter()
                component_cache = paths.cache / f"component-{tour.lower()}-{year:04d}-{month:02d}.parquet"
                if component_cache.exists():
                    component_frame = pd.read_parquet(component_cache)
                    if component_frame["match_id"].tolist() != target["match_id"].tolist():
                        raise ValueError(f"component cache target mismatch: {component_cache}")
                    component_frame = component_frame.drop(columns="match_id")
                    component_frame.index = target.index
                else:
                    bundle = _fit_v1_bundle(
                        counts,
                        tour=tour,
                        cutoff=cutoff,
                        source_hash=source_hash,
                        code_hash=code_hash,
                        repo_root=repo_root,
                    )
                    component_rows = [
                        _predict_component_match(bundle, row, cutoff)
                        for row in target.itertuples(index=False)
                    ]
                    component_frame = pd.DataFrame(component_rows, index=target.index)
                    cache_frame = component_frame.copy()
                    cache_frame.insert(0, "match_id", target["match_id"].to_numpy())
                    cache_frame.to_parquet(component_cache, index=False)
                output = target.copy()
                for column in component_frame:
                    output[column] = component_frame[column]
                prior = tour_history.loc[
                    tour_history["outcome_eligible"]
                    & (tour_history["available_at_utc"] < cutoff)
                ]
                counts_by_player = pd.concat(
                    [prior["winner_player_id"], prior["loser_player_id"]]
                ).value_counts()
                output["prior_matches_a"] = output["player_a_id"].map(counts_by_player).fillna(0).astype(int)
                output["prior_matches_b"] = output["player_b_id"].map(counts_by_player).fillna(0).astype(int)
                output["component_sparsity"] = np.exp(
                    -output[["prior_matches_a", "prior_matches_b"]].min(axis=1) / 25.0
                )

                anchors = _fit_anchor_predictions(
                    strength_records,
                    target,
                    cutoff=cutoff,
                    tour=tour,
                    configs=dynamic_configs,
                    code_hash=code_hash,
                )
                for config_id, records_for_config in anchors.items():
                    anchor_frame = pd.DataFrame(records_for_config, index=output.index)
                    for column in anchor_frame:
                        output[column] = anchor_frame[column]
                for k in config["elo"]["k_values"]:
                    for blend in config["elo"]["surface_blends"]:
                        key = f"elo_k{int(k)}_s{blend:.2f}_logit"
                        output[key] = _elo_predictions(
                            tour_history,
                            target,
                            cutoff=cutoff,
                            k=float(k),
                            surface_blend=float(blend),
                        )
                output["monthly_fit_runtime_seconds"] = time.perf_counter() - started
                output["code_sha256"] = code_hash
                output["source_set_sha256"] = source_hash
                output.to_parquet(part_path, index=False)
    frames = [pd.read_parquet(path) for path in part_paths if path.exists() and path.stat().st_size]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False)


def _integration_config(config: Mapping[str, Any], l2: float) -> StrengthIntegrationConfig:
    block = config["integration"]
    return StrengthIntegrationConfig(
        schema_version="strength-integration-config/v1",
        l2_penalty=l2,
        reliability_prior_logit=block["reliability_prior_logit"],
        maximum_absolute_tilt=block["maximum_absolute_tilt"],
        root_tolerance=block["root_tolerance"],
        probability_bound=block["probability_bound"],
        coefficient_draws_for_summary=block["coefficient_draws_for_summary"],
        q1_weight=block["q1_weight"],
        q2_weight=block["q2_weight"],
    )


def _cross_fitted_records(frame: pd.DataFrame, anchor_id: str) -> tuple[CrossFittedStrengthRecord, ...]:
    result: list[CrossFittedStrengthRecord] = []
    for _, row in frame.iterrows():
        forecast_cutoff = row.forecast_cutoff_utc.to_pydatetime()
        scheduled_start = row.event_start_date.to_pydatetime()
        if scheduled_start <= forecast_cutoff:
            scheduled_start = forecast_cutoff + timedelta(microseconds=1)
        result.append(
            CrossFittedStrengthRecord(
                match_id=row.match_id,
                tour=Tour(row.tour),
                player_a_id=row.player_a_id,
                player_b_id=row.player_b_id,
                forecast_cutoff_utc=forecast_cutoff,
                scheduled_start_utc=scheduled_start,
                outcome_available_at_utc=row.available_at_utc.to_pydatetime(),
                component_logit=float(row.component_logit),
                anchor_logit=float(getattr(row, f"anchor_{anchor_id}_logit")),
                component_variance=float(row.component_variance),
                anchor_variance=float(getattr(row, f"anchor_{anchor_id}_variance")),
                component_instability=_bounded_instability(float(row.component_instability)),
                component_sparsity=float(row.component_sparsity),
                player_a_won=bool(row.player_a_won),
            )
        )
    return tuple(result)


def _fit_integration(
    frame: pd.DataFrame,
    *,
    anchor_id: str,
    l2: float,
    cutoff: datetime,
    tour: str,
    config: Mapping[str, Any],
    code_hash: str,
) -> Any:
    eligible = frame.loc[frame["available_at_utc"] < cutoff]
    return fit_strength_integration(
        _cross_fitted_records(eligible, anchor_id),
        tour=Tour(tour),
        training_cutoff_utc=cutoff,
        fitted_at_utc=cutoff,
        config=_integration_config(config, l2),
        code_commit=code_hash,
    )


def _row_means(row: Any, side: str) -> PrimitiveServeMeans:
    return PrimitiveServeMeans(**json.loads(getattr(row, f"primitive_{side}")))


def _strength_prediction_from_row(row: Any, anchor_id: str) -> StrengthPrediction:
    player_a_graph = getattr(row, f"anchor_{anchor_id}_a_graph")
    player_b_graph = getattr(row, f"anchor_{anchor_id}_b_graph")
    return StrengthPrediction(
        player_a_id=row.player_a_id,
        player_b_id=row.player_b_id,
        surface=row.surface,
        best_of=int(row.best_of_int),
        scheduled_start_utc=row.event_start_date.to_pydatetime(),
        mean_logit=float(getattr(row, f"anchor_{anchor_id}_logit")),
        variance_logit=float(getattr(row, f"anchor_{anchor_id}_variance")),
        probability=float(getattr(row, f"anchor_{anchor_id}_probability")),
        player_a_graph_component=None if pd.isna(player_a_graph) else int(player_a_graph),
        player_b_graph_component=None if pd.isna(player_b_graph) else int(player_b_graph),
        weakly_connected=bool(getattr(row, f"anchor_{anchor_id}_weak")),
        player_a_known=bool(getattr(row, f"anchor_{anchor_id}_a_known")),
        player_b_known=bool(getattr(row, f"anchor_{anchor_id}_b_known")),
    )


def _integrated_target(row: Any, fit: Any, anchor_id: str) -> tuple[float, float, bool, float]:
    parameters = prepare_strength_match_parameters(
        anchor_artifact_id=_stable_id("anchor", anchor_id),
        integration_artifact_id=_stable_id("integration", fit.records_sha256),
        anchor=_strength_prediction_from_row(row, anchor_id),
        integration=fit,
        player_a=_row_means(row, "a"),
        player_b=_row_means(row, "b"),
        best_of=int(row.best_of_int),
        component_variance=float(row.component_variance),
        component_instability=_bounded_instability(float(row.component_instability)),
        component_sparsity=float(row.component_sparsity),
    )
    return (
        parameters.target_attained_logit,
        parameters.q_tilt_mean,
        parameters.tilt_saturated,
        parameters.reliability_weight,
    )


def _apply_target(
    row: Any, target_logit: float, integration_config: StrengthIntegrationConfig
) -> tuple[float, float, bool, float]:
    tilt, attained, saturated = solve_q_tilt(
        _row_means(row, "a"),
        _row_means(row, "b"),
        target_logit=target_logit,
        best_of=int(row.best_of_int),
        config=integration_config,
    )
    return attained, tilt, saturated, float("nan")


def _brier(frame: pd.DataFrame, probability_column: str) -> float:
    return float(np.mean(np.square(frame[probability_column].to_numpy(float) - frame["player_a_won"].to_numpy(float))))


def construct_outer_variants(
    config: Mapping[str, Any], forecasts: pd.DataFrame, code_hash: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    core_ids = [key for key in _dynamic_configs(config) if key.startswith("core_")]
    for outer_year in config["outer_years"]:
        outer_start = datetime(int(outer_year), 1, 1, tzinfo=UTC)
        inner_train_year = int(outer_year) - 2
        inner_validation_year = int(outer_year) - 1
        for tour in config["tours"]:
            cohort = forecasts.loc[forecasts["tour"].eq(tour)]
            train = cohort.loc[cohort["source_year"].eq(inner_train_year)]
            validation = cohort.loc[cohort["source_year"].eq(inner_validation_year)]
            both_inner = cohort.loc[cohort["source_year"].isin([inner_train_year, inner_validation_year])]
            outer = cohort.loc[cohort["source_year"].eq(outer_year)]
            if min(len(train), len(validation), len(outer)) == 0:
                raise ValueError(f"empty nested fold for {tour} {outer_year}")
            selection: dict[str, Any] = {}

            beta = min(
                config["temperature_multipliers"],
                key=lambda value: np.mean(
                    np.square(expit(float(value) * validation["component_logit"].to_numpy(float)) - validation["player_a_won"].to_numpy(float))
                ),
            )
            selection["temperature_multiplier"] = beta

            elo_candidates: list[tuple[float, float, float, float]] = []
            for k in config["elo"]["k_values"]:
                for surface_blend in config["elo"]["surface_blends"]:
                    anchor = validation[f"elo_k{int(k)}_s{surface_blend:.2f}_logit"].to_numpy(float)
                    for blend in config["elo"]["component_blends"]:
                        target = (1.0 - blend) * validation["component_logit"].to_numpy(float) + blend * anchor
                        score = float(np.mean(np.square(expit(target) - validation["player_a_won"].to_numpy(float))))
                        elo_candidates.append((score, float(k), float(surface_blend), float(blend)))
            _, elo_k, elo_surface, elo_blend = min(elo_candidates)
            selection["elo"] = {"k": elo_k, "surface_blend": elo_surface, "component_blend": elo_blend}

            constant_candidates: list[tuple[float, str, float]] = []
            for anchor_id in core_ids:
                anchor = validation[f"anchor_{anchor_id}_logit"].to_numpy(float)
                for blend in config["constant_anchor_blends"]:
                    target = (1.0 - blend) * validation["component_logit"].to_numpy(float) + blend * anchor
                    score = float(np.mean(np.square(expit(target) - validation["player_a_won"].to_numpy(float))))
                    constant_candidates.append((score, anchor_id, float(blend)))
            _, constant_anchor, constant_blend = min(constant_candidates)
            selection["dynamic_constant"] = {"anchor_id": constant_anchor, "blend": constant_blend}

            gated_candidates: list[tuple[float, str, float]] = []
            fitted_for_selection: dict[tuple[str, float], Any] = {}
            validation_cutoff = datetime(inner_validation_year, 1, 1, tzinfo=UTC)
            for anchor_id in core_ids:
                for l2 in config["integration"]["l2_penalties"]:
                    fit = _fit_integration(
                        train,
                        anchor_id=anchor_id,
                        l2=float(l2),
                        cutoff=validation_cutoff,
                        tour=tour,
                        config=config,
                        code_hash=code_hash,
                    )
                    fitted_for_selection[(anchor_id, float(l2))] = fit
                    probabilities = [
                        expit(_integrated_target(row, fit, anchor_id)[0])
                        for _, row in validation.iterrows()
                    ]
                    score = float(np.mean(np.square(np.asarray(probabilities) - validation["player_a_won"].to_numpy(float))))
                    gated_candidates.append((score, anchor_id, float(l2)))
            _, gated_anchor, gated_l2 = min(gated_candidates)
            selection["dynamic_gated"] = {"anchor_id": gated_anchor, "l2": gated_l2}

            integration_fits = {
                "dynamic_gated": _fit_integration(
                    both_inner,
                    anchor_id=gated_anchor,
                    l2=gated_l2,
                    cutoff=outer_start,
                    tour=tour,
                    config=config,
                    code_hash=code_hash,
                )
            }
            for name, anchor_id in (
                ("complete_v1_1", "complete"),
                ("ranking_prior_ablation", "no_ranking"),
                ("level_change_ablation", "no_level"),
                ("head_to_head_ablation", "no_h2h"),
            ):
                integration_fits[name] = _fit_integration(
                    both_inner,
                    anchor_id=anchor_id,
                    l2=gated_l2,
                    cutoff=outer_start,
                    tour=tour,
                    config=config,
                    code_hash=code_hash,
                )
            selection["optional_l2"] = gated_l2
            selections[f"{tour}-{outer_year}"] = selection
            root_config = _integration_config(config, gated_l2)

            for _, row in outer.iterrows():
                base = {
                    key: row[key]
                    for key in (
                        "match_id", "event_id", "tour", "source_year", "tourney_name",
                        "tourney_level", "surface", "round", "player_a_id", "player_b_id",
                        "player_a_won", "event_start_date", "available_at_utc", "forecast_cutoff_utc",
                        "component_probability", "component_logit", "component_variance",
                        "component_instability", "component_sparsity", "prior_matches_a",
                        "prior_matches_b", "primitive_a", "primitive_b", "primitive_uncertainty_a",
                        "primitive_uncertainty_b", "best_of_int", "observed_a_aces", "observed_b_aces",
                        "observed_a_df", "observed_b_df", "ace_compare_outcome", "df_compare_outcome",
                    )
                }
                variant_targets: dict[str, tuple[float, float, bool, float, str]] = {
                    "v1_0": (float(row.component_logit), 0.0, False, 0.0, "component"),
                }
                variant_targets["temperature_only"] = (*_apply_target(row, float(beta) * row.component_logit, root_config), "temperature")
                elo_anchor = getattr(row, f"elo_k{int(elo_k)}_s{elo_surface:.2f}_logit")
                elo_target = (1.0 - elo_blend) * row.component_logit + elo_blend * elo_anchor
                variant_targets["surface_elo"] = (*_apply_target(row, elo_target, root_config), "surface_elo")
                constant_anchor_logit = getattr(row, f"anchor_{constant_anchor}_logit")
                constant_target = (1.0 - constant_blend) * row.component_logit + constant_blend * constant_anchor_logit
                variant_targets["dynamic_constant"] = (*_apply_target(row, constant_target, root_config), constant_anchor)
                variant_targets["dynamic_gated"] = (*_integrated_target(row, integration_fits["dynamic_gated"], gated_anchor), gated_anchor)
                for name, anchor_id in (
                    ("complete_v1_1", "complete"),
                    ("ranking_prior_ablation", "no_ranking"),
                    ("level_change_ablation", "no_level"),
                    ("head_to_head_ablation", "no_h2h"),
                ):
                    variant_targets[name] = (*_integrated_target(row, integration_fits[name], anchor_id), anchor_id)

                for variant, (attained, tilt, saturated, reliability, anchor_id) in variant_targets.items():
                    record = dict(base)
                    record.update(
                        {
                            "outer_fold": int(outer_year),
                            "variant": variant,
                            "probability": float(expit(attained)),
                            "attained_logit": float(attained),
                            "q_tilt": float(tilt),
                            "tilt_saturated": bool(saturated),
                            "reliability_weight": float(reliability),
                            "anchor_id": anchor_id,
                            "anchor_logit": float(
                                row.component_logit if anchor_id in ("component", "temperature")
                                else elo_anchor if anchor_id == "surface_elo"
                                else getattr(row, f"anchor_{anchor_id}_logit")
                            ),
                        }
                    )
                    record["root_residual"] = float(record["attained_logit"] - attained)
                    rows.append(record)
    result = pd.DataFrame.from_records(rows)
    return result, selections


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(p)), np.asarray([_logit(value) for value in p])])

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    fit = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    return float(fit.x[0]), float(fit.x[1])


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def winner_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groupings: list[tuple[str, pd.Series]] = [("overall", pd.Series(True, index=frame.index))]
    for column in ("tour", "outer_fold", "surface", "tourney_level"):
        for value in sorted(frame[column].dropna().unique(), key=str):
            groupings.append((f"{column}={value}", frame[column].eq(value)))
    groupings.extend(
        [
            ("sparse", frame["component_sparsity"] >= 0.5),
            ("well_observed", frame["component_sparsity"] < 0.5),
            ("anchor_component_agree", np.sign(frame["anchor_logit"]) == np.sign(frame["component_logit"])),
            ("anchor_component_disagree", np.sign(frame["anchor_logit"]) != np.sign(frame["component_logit"])),
        ]
    )
    for variant in VARIANTS:
        variant_frame = frame.loc[frame["variant"].eq(variant)]
        for stratum, mask in groupings:
            subset = variant_frame.loc[mask.reindex(variant_frame.index, fill_value=False)]
            if len(subset) < 20:
                continue
            y = subset["player_a_won"].to_numpy(float)
            p = subset["probability"].to_numpy(float)
            errors = np.square(p - y)
            intercept, slope = _calibration(y, p)
            high = ((p >= 0.8) & (y == 0)) | ((p <= 0.2) & (y == 1))
            tail_n = max(1, math.ceil(0.1 * len(errors)))
            records.append(
                {
                    "variant": variant,
                    "stratum": stratum,
                    "n": len(subset),
                    "brier": float(errors.mean()),
                    "log_loss": float(-np.mean(y * np.log(np.clip(p, EPS, 1)) + (1-y) * np.log(np.clip(1-p, EPS, 1)))),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "auc": _auc(y, p),
                    "high_confidence_error_rate": float(high.mean()),
                    "high_confidence_n": int(((p >= 0.8) | (p <= 0.2)).sum()),
                    "worst_decile_brier": float(np.partition(errors, -tail_n)[-tail_n:].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def reliability_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["probability_band"] = pd.cut(
        result["probability"], bins=np.linspace(0, 1, 11), include_lowest=True
    ).astype(str)
    return (
        result.groupby(["variant", "probability_band"], observed=True)
        .agg(n=("match_id", "size"), mean_probability=("probability", "mean"), observed_rate=("player_a_won", "mean"))
        .reset_index()
    )


def paired_bootstrap(config: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index="match_id", columns="variant", values="probability")
    metadata = frame.drop_duplicates("match_id").set_index("match_id")
    wide = wide.join(metadata[["player_a_won", "event_id", "tour", "outer_fold"]]).dropna()
    y = wide["player_a_won"].to_numpy(float)
    base_p = wide["v1_0"].to_numpy(float)
    blocks = wide.groupby(["tour", "outer_fold", "event_id"], sort=True).indices
    strata: defaultdict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for (tour, fold, _event), indices in blocks.items():
        strata[(tour, int(fold))].append(np.asarray(indices, dtype=int))
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    records: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        p = wide[variant].to_numpy(float)
        brier_delta = np.square(p-y) - np.square(base_p-y)
        log_delta = -(y*np.log(np.clip(p,EPS,1))+(1-y)*np.log(np.clip(1-p,EPS,1))) + (y*np.log(np.clip(base_p,EPS,1))+(1-y)*np.log(np.clip(1-base_p,EPS,1)))
        draws_brier = np.empty(replicates)
        draws_log = np.empty(replicates)
        for rep in range(replicates):
            sampled: list[np.ndarray] = []
            for event_blocks in strata.values():
                choices = rng.integers(0, len(event_blocks), len(event_blocks))
                sampled.extend(event_blocks[index] for index in choices)
            indices = np.concatenate(sampled)
            draws_brier[rep] = brier_delta[indices].mean()
            draws_log[rep] = log_delta[indices].mean()
        for metric, values, point in (
            ("brier", draws_brier, brier_delta.mean()),
            ("log_loss", draws_log, log_delta.mean()),
        ):
            records.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "difference_vs_v1_0": float(point),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": replicates,
                    "block_kind": "tour-outer-year-event",
                }
            )
    return pd.DataFrame.from_records(records)


def _integrated_means(means: PrimitiveServeMeans, tilt: float, direction: int) -> PrimitiveServeMeans:
    payload = asdict(means)
    for name in ("returnable_first_win", "playable_second_win"):
        payload[name] = float(expit(_logit(float(payload[name])) + direction * tilt))
    return PrimitiveServeMeans(**payload)


def _sample_performance(uncertainty_json: str, rng: np.random.Generator) -> PrimitiveServeMeans:
    payload = json.loads(uncertainty_json)
    values: dict[str, float] = {}
    for name in PRIMITIVE_NAMES:
        item = payload[name]
        eta = _logit(float(item["mean"])) + float(item["linear_sd"]) * rng.normal()
        mean = _clip_probability(float(expit(eta)))
        concentration = max(float(item["concentration"]), 1e-3)
        values[name] = float(rng.beta(concentration * mean, concentration * (1.0 - mean)))
    return PrimitiveServeMeans(**values)


def _current_server(state: Any) -> int:
    active = state.active_set
    if active is None:
        raise RuntimeError("completed match has no current server")
    if active.tiebreak is not None:
        return int(tiebreak_server(active.tiebreak.first_server_index, sum(active.tiebreak.points)))
    return int((active.first_server_index + sum(active.games)) % 2)


def _simulate_path(
    row: Any,
    *,
    tilt: float,
    seed: int,
    first_server: int,
) -> dict[str, Any]:
    performance_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    point_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    a_base = _sample_performance(row.primitive_uncertainty_a, performance_rng)
    b_base = _sample_performance(row.primitive_uncertainty_b, performance_rng)
    a = _integrated_means(a_base, tilt, 1)
    b = _integrated_means(b_base, tilt, -1)
    draws = (ServePerformanceDraw(**asdict(a)), ServePerformanceDraw(**asdict(b)))
    state = new_match(row.player_a_id, row.player_b_id, best_of=int(row.best_of_int), first_server_index=first_server)
    aces = [0, 0]
    dfs = [0, 0]
    service_points = [0, 0]
    holds = [0, 0]
    service_games = [0, 0]
    tiebreaks = 0
    while state.active_set is not None:
        server = _current_server(state)
        point = generate_service_point(
            draws[server], point_rng, server_id=state.players[server], receiver_id=state.players[1-server]
        )
        service_points[server] += 1
        aces[server] += int(point.ace)
        dfs[server] += int(point.double_fault)
        transition = award_point(state, server if point.server_won else 1-server)
        if transition.regular_game_completed:
            service_games[server] += 1
            holds[server] += int(transition.game_winner_index == server)
        if transition.tiebreak_completed:
            tiebreaks += 1
        state = transition.after
    games = sum(sum(item.games) for item in state.completed_sets)
    winner = int(sum(item.games[0] > item.games[1] for item in state.completed_sets) < sum(item.games[1] > item.games[0] for item in state.completed_sets))
    return {
        "a_won": int(winner == 0),
        "a_aces": aces[0],
        "b_aces": aces[1],
        "a_df": dfs[0],
        "b_df": dfs[1],
        "a_service_points": service_points[0],
        "b_service_points": service_points[1],
        "sets": len(state.completed_sets),
        "games": games,
        "tiebreaks": tiebreaks,
        "a_holds": holds[0],
        "b_holds": holds[1],
        "a_service_games": service_games[0],
        "b_service_games": service_games[1],
        "duration_exposure_proxy": sum(service_points) + 4 * games + 8 * tiebreaks,
        "a_f": a_base.first_serve_in,
        "a_a": a_base.ace_given_first_in,
        "a_d": a_base.double_fault_given_second_opp,
        "b_f": b_base.first_serve_in,
        "b_a": b_base.ace_given_first_in,
        "b_d": b_base.double_fault_given_second_opp,
    }


def _protected_sample(config: Mapping[str, Any], complete: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    n = int(config["protected_props"]["sample_per_tour_year"])
    for (_tour, _year), group in complete.groupby(["tour", "outer_fold"]):
        hashes = group["match_id"].map(lambda value: _stable_id("protected", value))
        selected.append(group.assign(_sample_hash=hashes).sort_values("_sample_hash").head(n))
    return pd.concat(selected, ignore_index=True).drop(columns="_sample_hash")


def protected_prop_backtest(config: Mapping[str, Any], variants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = variants.loc[variants["variant"].eq("complete_v1_1")]
    sample = _protected_sample(config, complete)
    baseline = variants.loc[variants["variant"].eq("v1_0")].set_index("match_id")
    checkpoints = [int(value) for value in config["protected_props"]["checkpoints"]]
    root_seed = int(config["protected_props"]["seed"])
    path_records: list[dict[str, Any]] = []
    summary: pd.DataFrame | None = None
    diagnostics: list[dict[str, Any]] = []
    previous = 0
    for checkpoint in checkpoints:
        for row in sample.itertuples(index=False):
            base = baseline.loc[row.match_id]
            for path_index in range(previous, checkpoint):
                seed = int.from_bytes(hashlib.sha256(f"{root_seed}|{row.match_id}|{path_index}".encode()).digest()[:8], "big")
                first_server = seed & 1
                for variant, tilt in (("v1_0", float(base.q_tilt)), ("complete_v1_1", float(row.q_tilt))):
                    simulated = _simulate_path(row, tilt=tilt, seed=seed, first_server=first_server)
                    path_records.append({"match_id": row.match_id, "event_id": row.event_id, "tour": row.tour, "outer_fold": row.outer_fold, "variant": variant, "path_index": path_index, **simulated})
        paths = pd.DataFrame.from_records(path_records)
        summary = paths.groupby(["match_id", "variant"]).agg(
            ace_probability=("a_aces", lambda x: 0.0),
        ).reset_index().drop(columns="ace_probability")
        estimates = paths.groupby(["match_id", "variant"]).apply(
            lambda group: pd.Series({
                "ace_probability": float((group["a_aces"] > group["b_aces"]).mean()),
                "df_probability": float((group["a_df"] > group["b_df"]).mean()),
                "expected_a_aces": group["a_aces"].mean(),
                "expected_b_aces": group["b_aces"].mean(),
                "expected_a_df": group["a_df"].mean(),
                "expected_b_df": group["b_df"].mean(),
                "ace_tie_rate": (group["a_aces"] == group["b_aces"]).mean(),
                "df_tie_rate": (group["a_df"] == group["b_df"]).mean(),
                "zero_ace_rate": ((group["a_aces"] + group["b_aces"]) == 0).mean(),
                "zero_df_rate": ((group["a_df"] + group["b_df"]) == 0).mean(),
                "mean_games": group["games"].mean(),
                "mean_sets": group["sets"].mean(),
                "mean_tiebreaks": group["tiebreaks"].mean(),
                "mean_service_points": (group["a_service_points"] + group["b_service_points"]).mean(),
                "duration_exposure_proxy": group["duration_exposure_proxy"].mean(),
            }),
            include_groups=False,
        ).reset_index()
        observed = sample.set_index("match_id")
        estimates["ace_outcome"] = estimates["match_id"].map(observed["ace_compare_outcome"])
        estimates["df_outcome"] = estimates["match_id"].map(observed["df_compare_outcome"])
        estimates["ace_brier"] = np.square(estimates["ace_probability"] - estimates["ace_outcome"])
        estimates["df_brier"] = np.square(estimates["df_probability"] - estimates["df_outcome"])
        wide = estimates.pivot(index="match_id", columns="variant", values=["ace_brier", "df_brier"])
        ace_delta = wide[("ace_brier", "complete_v1_1")] - wide[("ace_brier", "v1_0")]
        df_delta = wide[("df_brier", "complete_v1_1")] - wide[("df_brier", "v1_0")]
        # Conservative independent-path upper bound; common random numbers usually reduce it.
        ace_mcse = float(np.sqrt(np.var(ace_delta, ddof=1) / len(ace_delta) / checkpoint))
        df_mcse = float(np.sqrt(np.var(df_delta, ddof=1) / len(df_delta) / checkpoint))
        diagnostics.append({"paths": checkpoint, "ace_brier_delta_mcse": ace_mcse, "df_brier_delta_mcse": df_mcse})
        summary = estimates
        previous = checkpoint
        if max(ace_mcse, df_mcse) <= float(config["protected_props"]["aggregate_mcse_target"]):
            break
    if summary is None:
        raise RuntimeError("protected simulation produced no estimates")
    return summary, pd.DataFrame.from_records(diagnostics), pd.DataFrame.from_records(path_records)


def protected_metrics(config: Mapping[str, Any], estimates: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    metadata = variants.loc[variants["variant"].eq("complete_v1_1")].drop_duplicates("match_id").set_index("match_id")
    enriched = estimates.copy()
    for column in ("event_id", "tour", "outer_fold"):
        enriched[column] = enriched["match_id"].map(metadata[column])
    wide = enriched.pivot(index="match_id", columns="variant", values=["ace_brier", "df_brier"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]) + 1)
    records: list[dict[str, Any]] = []
    meta = metadata.loc[wide.index]
    block_indices = list(meta.groupby(["tour", "outer_fold", "event_id"]).indices.values())
    for family in ("ace", "df"):
        delta = wide[(f"{family}_brier", "complete_v1_1")] - wide[(f"{family}_brier", "v1_0")]
        draws = []
        for _ in range(int(config["bootstrap"]["replicates"])):
            chosen = rng.integers(0, len(block_indices), len(block_indices))
            indices = np.concatenate([np.asarray(block_indices[index]) for index in chosen])
            draws.append(float(delta.to_numpy()[indices].mean()))
        records.append({
            "family": family,
            "n": len(delta),
            "difference_vs_v1_0": float(delta.mean()),
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "noninferiority_margin": float(config["protected_props"]["noninferiority_margin_brier"]),
        })
    return pd.DataFrame.from_records(records)


def _plot_calibration(reliability: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 800, 700, 80
    plot_width, plot_height = width - 2 * margin, height - 2 * margin

    def point(x: float, y: float) -> tuple[float, float]:
        return margin + x * plot_width, height - margin - y * plot_height

    colors = {"v1_0": "#1b4965", "dynamic_gated": "#ca6702", "complete_v1_1": "#2a9d8f"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f5ef"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#777" stroke-dasharray="6 6"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#222"/>',
        '<text x="400" y="35" text-anchor="middle" font-family="Georgia" font-size="24">Rolling-origin winner calibration</text>',
        '<text x="400" y="680" text-anchor="middle" font-family="Georgia" font-size="17">Mean forecast probability</text>',
        '<text x="22" y="350" text-anchor="middle" transform="rotate(-90 22 350)" font-family="Georgia" font-size="17">Observed frequency</text>',
    ]
    legend_y = 65
    for variant in ("v1_0", "dynamic_gated", "complete_v1_1"):
        subset = reliability.loc[reliability["variant"].eq(variant)].dropna(subset=["mean_probability", "observed_rate"])
        coordinates = [point(float(x), float(y)) for x, y in zip(subset["mean_probability"], subset["observed_rate"])]
        if coordinates:
            joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
            elements.append(f'<polyline points="{joined}" fill="none" stroke="{colors[variant]}" stroke-width="3"/>')
            elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[variant]}"/>' for x, y in coordinates)
        elements.append(f'<text x="{width-margin-180}" y="{legend_y}" font-family="Georgia" font-size="15" fill="{colors[variant]}">{variant}</text>')
        legend_y += 22
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _recommendation(
    config: Mapping[str, Any], metrics: pd.DataFrame, intervals: pd.DataFrame, protected: pd.DataFrame
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    complete = intervals.loc[intervals["variant"].eq("complete_v1_1")].set_index("metric")
    gates = config["winner_gates"]
    if complete.loc["brier", "difference_vs_v1_0"] > -float(gates["brier_improvement"]) or complete.loc["brier", "ci_upper"] >= 0:
        reasons.append("complete v1.1 did not pass the preregistered winner Brier gate")
    if complete.loc["log_loss", "difference_vs_v1_0"] > -float(gates["log_loss_improvement"]) or complete.loc["log_loss", "ci_upper"] >= 0:
        reasons.append("complete v1.1 did not pass the preregistered winner log-loss gate")
    for row in protected.itertuples(index=False):
        if row.ci_upper > row.noninferiority_margin:
            reasons.append(f"{row.family} comparison failed protected-prop non-inferiority")
    if reasons:
        return "FAIL", reasons
    if not config["publication_vintage_complete"]:
        reasons.append("historical publication/correction vintages are not complete")
        return "INCONCLUSIVE", reasons
    return "PASS", reasons


def run_backtest(config_path: str | Path, *, repo_root: str | Path = ".") -> Path:
    repo = Path(repo_root).resolve()
    config_file = (repo / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    config = load_config(config_file)
    plan_path = repo / str(config["plan_path"])
    paths = _run_paths(config, repo)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    matches, counts, excluded, source_manifest = prepare_historical_data(config, repo)
    excluded.to_csv(paths.root / "excluded_matches.csv", index=False)
    cutoff_audit = matches.groupby(["tour", "source_year", "forecast_cutoff_utc"]).agg(
        source_rows=("match_id", "size"),
        evaluation_eligible=("evaluation_eligible", "sum"),
        earliest_available=("available_at_utc", "min"),
        latest_available=("available_at_utc", "max"),
    ).reset_index()
    cutoff_audit.to_csv(paths.root / "cutoff_audit.csv", index=False)

    forecasts = generate_cross_fitted_forecasts(config, repo, paths, matches, counts, source_manifest)
    forecasts.to_parquet(paths.root / "cross_fitted_base_forecasts.parquet", index=False)
    code_hash = _sha256_file(Path(__file__))
    outer_parts = paths.root / "outer_parts"
    expected_outer_parts = [
        outer_parts / f"{tour.lower()}-{year}.parquet"
        for tour in config["tours"]
        for year in config["outer_years"]
    ]
    if expected_outer_parts and all(path.exists() for path in expected_outer_parts):
        outer = pd.concat(
            [pd.read_parquet(path) for path in expected_outer_parts],
            ignore_index=True,
            sort=False,
        )
        selections = {}
        for path in expected_outer_parts:
            selection_path = path.with_suffix(".json")
            selections.update(json.loads(selection_path.read_text(encoding="utf-8")))
    else:
        outer, selections = construct_outer_variants(config, forecasts, code_hash)
    outer.to_parquet(paths.root / "fold_level_forecasts.parquet", index=False)
    _json_dump(paths.root / "selected_hyperparameters.json", selections)

    metrics = winner_metrics(outer)
    reliability = reliability_table(outer)
    intervals = paired_bootstrap(config, outer)
    metrics.to_csv(paths.root / "winner_metrics.csv", index=False)
    reliability.to_csv(paths.root / "winner_reliability.csv", index=False)
    intervals.to_csv(paths.root / "paired_winner_intervals.csv", index=False)
    _plot_calibration(reliability, paths.figures / "winner_calibration.svg")

    prop_estimates, mc_diagnostics, path_records = protected_prop_backtest(config, outer)
    prop_summary = protected_metrics(config, prop_estimates, outer)
    prop_estimates.to_parquet(paths.root / "protected_prop_forecasts.parquet", index=False)
    prop_summary.to_csv(paths.root / "protected_prop_noninferiority.csv", index=False)
    mc_diagnostics.to_csv(paths.root / "monte_carlo_diagnostics.csv", index=False)
    path_records.to_parquet(paths.root / "joint_path_audit.parquet", index=False)

    recommendation, reasons = _recommendation(config, metrics, intervals, prop_summary)
    implementation_bugs = [
        {
            "id": "BACKTEST-INSTABILITY-BOUND-001",
            "stage": "first inner-fold integration fit",
            "outer_results_viewed_before_fix": False,
            "symptom": "raw maximum Q-logit SD exceeded the typed [0,1] feature contract",
            "fix": "apply the preregistered monotone x/(1+x) transform at every integration boundary",
            "affected_folds_rerun": "all inner and outer integration folds",
        },
        {
            "id": "BACKTEST-GRAPH-NULL-002",
            "stage": "inner-fold integrated prediction",
            "outer_results_viewed_before_fix": False,
            "symptom": "optional graph-component None values round-tripped through parquet as NaN",
            "fix": "restore missing graph-component identifiers to None at the typed interface",
            "affected_folds_rerun": "all inner and outer integrated predictions",
        },
        {
            "id": "BACKTEST-DATE-PROXY-003",
            "stage": "final inner-fold integration refit",
            "outer_results_viewed_before_fix": False,
            "symptom": "event-date proxy equaled the month-start forecast cutoff",
            "fix": "use cutoff + 1 microsecond as a typed chronology sentinel for equality only",
            "affected_folds_rerun": "all inner and outer integration folds",
        },
        {
            "id": "BACKTEST-ROUND-SERIALIZE-004",
            "stage": "outer-fold atomic publication",
            "outer_results_viewed_before_fix": False,
            "symptom": "generic outer-metadata attribute access resolved the round field to pandas Series.round",
            "fix": "use explicit item access row[key] for every copied outer source field",
            "affected_folds_rerun": "all outer folds",
        },
    ]
    _json_dump(paths.root / "implementation_bugs.json", implementation_bugs)
    overall = metrics.loc[metrics["stratum"].eq("overall")]
    report_lines = [
        "# Tennis Model v1.1 Candidate Backtest Report",
        "",
        f"**Recommendation: {recommendation}**",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Overall winner metrics",
        "",
        "```csv",
        overall.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired winner intervals",
        "",
        "```csv",
        intervals.to_csv(index=False).strip(),
        "```",
        "",
        "## Protected prop non-inferiority",
        "",
        "```csv",
        prop_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Predeclared limitations",
        "",
        "- The source is retrospective-finalized with a conservative event embargo, not a complete point-in-time publication/correction vintage.",
        "- Historical cancellation-only target rows are not observable in result-only files.",
        "- Duration effects are represented by joint-path exposure diagnostics; historical duration and retirement settlement are not promotion-eligible in this source mode.",
        "- The supplied 53-market scorecard was not used for training, tuning, or selection.",
        "",
        "## Implementation bugs corrected before outer evaluation",
        "",
        "- `BACKTEST-INSTABILITY-BOUND-001`: normalized raw maximum Q-logit SD with `x/(1+x)` to satisfy the typed reliability-feature contract; every integration fold was run only after the fix.",
        "- `BACKTEST-GRAPH-NULL-002`: restored parquet `NaN` graph identifiers to semantic `None` before typed strength prediction construction.",
        "- `BACKTEST-DATE-PROXY-003`: represented an event-date proxy equal to its cutoff as `cutoff + 1 microsecond` solely to satisfy strict typed chronology.",
        "- `BACKTEST-ROUND-SERIALIZE-004`: changed all copied outer source fields from generic pandas attribute access to explicit item access after the `round` field resolved to `Series.round` and atomic parquet publication failed; no outer predictions or metrics were viewed before the fix, and every outer fold was rerun.",
    ]
    (paths.root / "BACKTEST_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "schema_version": config["schema_version"],
        "recommendation": recommendation,
        "plan_sha256": _sha256_file(plan_path),
        "config_sha256": _sha256_file(config_file),
        "code_sha256": code_hash,
        "source_manifest_sha256": _sha256_file(repo / str(config["source_directory"]) / "source_manifest.json"),
        "source_members": source_manifest,
        "outer_forecast_rows": len(outer),
        "outer_matches": int(outer["match_id"].nunique()),
        "protected_matches": int(prop_estimates["match_id"].nunique()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "production_default_modified": False,
    }
    _json_dump(paths.root / "manifest.json", manifest)
    return paths.root


__all__ = [
    "VARIANTS",
    "construct_outer_variants",
    "load_config",
    "prepare_historical_data",
    "run_backtest",
]
