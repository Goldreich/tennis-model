"""Pure renderers over stored lock content; no probability is recomputed here."""

from __future__ import annotations

from tennis_model.locking.models import PredictionSnapshot, SerializedProp


def _contains_duration(prop: SerializedProp) -> bool:
    if prop.node == "atomic":
        return prop.kind == "DURATION_MIN"
    return any(_contains_duration(child) for child in prop.children)


def render_locked_match_card(lock: PredictionSnapshot) -> str:
    context = lock.context
    duration_requested = any(
        _contains_duration(item.prop) for item in lock.prop_estimates
    ) or any(_contains_duration(item.prop) for item in lock.prop_gates)
    dirty_suffix = f" (dirty {lock.code.diff_sha256})" if lock.code.dirty else ""
    lines = [
        "# LOCKED MATCH CARD",
        "",
        f"Lock ID: {lock.lock_id}",
        "Canonical match identity: "
        + (
            "legacy identity schema"
            if lock.canonical_match_identity is None
            else lock.canonical_match_identity.canonical_match_id
        ),
        f"Framework: Tennis Model {lock.framework_version}",
        f"Created (UTC): {lock.created_at_utc.isoformat()}",
        f"Information cutoff: {context.information_cutoff_utc.isoformat()}",
        f"Event: {context.event}",
        f"Match: {context.player_a_id} v {context.player_b_id}",
        f"Draw / round: {context.tour.value} singles / {context.round}",
        f"Scheduled start: {context.scheduled_start_utc.isoformat()}",
        f"Format: best of {context.best_of}; standard TB to 7; deciding TB to 10 at 6-6",
        f"First server: {lock.simulation.first_server_id or 'unknown (50/50 by path)'}",
        f"Conditions scenario: {lock.information.scenario_id}",
        f"Model snapshot: {lock.match_parameters.snapshot_id}",
        f"Data snapshot: {lock.match_parameters.snapshot.data_hash}",
        f"Source manifest: {lock.source_manifest.manifest_sha256}",
        f"Model configuration: {lock.match_parameters.snapshot.config_hash}",
        f"Code: {lock.code.commit}{dirty_suffix}",
        (
            f"Simulation: {lock.simulation.actual_paths} paths; seed {lock.simulation.seed_id}; "
            f"settlement policy {lock.settlement_policy.version}"
        ),
        "",
        "## Matchup parameters",
        "",
        "| Parameter | Player A serving | Player B serving |",
        "|---|---:|---:|",
    ]
    directions = lock.parameter_summaries
    primitive_labels = {
        "F": "First serve in",
        "A": "Ace given first serve in",
        "Q1": "Returnable first-serve points won",
        "D": "Double fault given second-serve opportunity",
        "Q2": "Playable second-serve points won",
    }
    for index, component in enumerate(("F", "A", "Q1", "D", "Q2")):
        lines.append(
            f"| {primitive_labels[component]} | "
            f"{directions[0].primitives[index].map_mean:.1%} | "
            f"{directions[1].primitives[index].map_mean:.1%} |"
        )
    derived = (
        ("Derived first-serve points won", "first_serve_win"),
        ("Derived second-serve points won", "second_serve_win"),
        ("Overall service points won", "service_point_win"),
        ("Implied hold probability", "analytic_hold_probability"),
        ("Ace rate / service point", "ace_rate_per_service_point"),
        ("Double-fault rate / service point", "double_fault_rate_per_service_point"),
    )
    for label, field in derived:
        lines.append(
            f"| {label} | {getattr(directions[0], field):.1%} | "
            f"{getattr(directions[1], field):.1%} |"
        )
    lines.extend(
        [
            "",
            "### Primitive uncertainty diagnostics",
            "",
            "| Component | A logit SD | B logit SD | A concentration | B concentration | "
            "A weighted trials | B weighted trials |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for left, right in zip(directions[0].primitives, directions[1].primitives, strict=True):
        left_trials = "--" if left.weighted_trials is None else f"{left.weighted_trials:.1f}"
        right_trials = "--" if right.weighted_trials is None else f"{right.weighted_trials:.1f}"
        lines.append(
            f"| {left.component.value} | {left.linear_predictor_sd:.4f} | "
            f"{right.linear_predictor_sd:.4f} | {left.predictive_concentration:.1f} | "
            f"{right.predictive_concentration:.1f} | {left_trials} | {right_trials} |"
        )

    if (
        lock.match_parameters.inactivity is not None
        and lock.match_parameters.retirement is not None
    ):
        lines.extend(["", "### B6 retirement and C6 inactivity", ""])
        scenario_by_player = {
            item.player_id: item for item in lock.match_parameters.retirement.scenario_mixtures
        }
        for record in lock.match_parameters.inactivity.records:
            days = "cold start" if record.inactivity_days is None else str(record.inactivity_days)
            scenarios = ", ".join(
                f"{item.scenario_id} (eta={item.log_hazard_ratio:.4g}, w={item.weight:.4g})"
                for item in scenario_by_player[record.player_id].scenarios
                if item.weight is not None
            )
            lines.append(
                f"- {record.player_id}: inactivity {days} days; hard multiplier "
                f"{record.hard_deviation_multiplier:.6f}; variance factor "
                f"{record.variance_inflation_factor:.6f}; retirement scenario {scenarios}"
            )

    if duration_requested:
        duration = lock.match_summary.duration
        lines.extend(["", "### Match duration", ""])
        if duration is None:
            lines.append("- Duration model unavailable for this lock.")
        else:
            lines.extend(
                [
                    f"- Expected duration: {duration.expected_minutes:.1f} minutes",
                    "- Duration 10/50/90% quantiles: "
                    f"{duration.quantiles[0]:.1f} / {duration.quantiles[1]:.1f} / "
                    f"{duration.quantiles[2]:.1f} minutes",
                    f"- Duration data grade: {duration.data_grade}",
                    f"- Duration artifact: {duration.artifact_id}",
                    f"- Duration display policy: {duration.display_policy_version}",
                ]
            )
            if duration.current_event_effect_minutes is not None:
                lines.append(
                    "- Current-event duration effect: "
                    f"{duration.current_event_effect_minutes:+.2f} minutes"
                )
            if duration.display_boundary_sensitive:
                lines.append(
                    "- Warning: official whole-minute conversion affects at least one "
                    "requested duration threshold."
                )

    lines.extend(["", "## Core simulated outputs", ""])
    for player in lock.match_summary.players:
        lines.append(
            f"- {player.player_id}: win {player.match_win_probability:.1%}; "
            f"expected aces {player.expected_aces:.2f}; expected DFs "
            f"{player.expected_double_faults:.2f}; expected breaks {player.expected_breaks:.2f}"
        )
    quantiles = lock.match_summary.total_games_quantiles
    lines.extend(
        [
            f"- Expected total games: {lock.match_summary.expected_total_games:.2f}",
            "- Total-games 10/50/90% quantiles: "
            f"{quantiles[0]:.1f} / {quantiles[1]:.1f} / {quantiles[2]:.1f}",
            f"- At least one tiebreak: {lock.match_summary.any_tiebreak_probability:.1%}",
            f"- Deciding set: {lock.match_summary.deciding_set_probability:.1%}",
            f"- Expected total breaks: {lock.match_summary.expected_total_breaks:.2f}",
            "- Retirement probability: "
            + (
                "unavailable (pre-amendment development/test lock)"
                if lock.match_summary.retirement_probability is None
                else f"{lock.match_summary.retirement_probability:.1%}"
            ),
            "- Exact score probabilities: "
            + "; ".join(
                f"{item.winner_id} {item.winner_sets}-{item.loser_sets} {item.probability:.1%}"
                for item in lock.match_summary.exact_scores
            ),
            "",
            "## Championship markets",
            "",
            "| Canonical prop | Yes | No | Void | Unresolved | Settled | Raw model | "
            "Model integer | 99% anytime-valid CS | MC status | Final paths | "
            "P(settled) | Support | Platform integer | Grade |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|---|",
        ]
    )
    for estimate in lock.prop_estimates:
        raw_probability = (
            estimate.probability_raw
            if estimate.mc_policy_version is None
            else estimate.model_probability_raw
        )
        raw = "--" if raw_probability is None else f"{raw_probability:.3%}"
        model_integer = (
            estimate.submitted_integer
            if estimate.mc_policy_version is None
            else estimate.model_probability_integer
        )
        integer = "--" if model_integer is None else f"{model_integer}%"
        confidence_sequence = (
            "--"
            if estimate.mc_confidence_sequence_lower is None
            or estimate.mc_confidence_sequence_upper is None
            else (
                f"[{estimate.mc_confidence_sequence_lower:.3%}, "
                f"{estimate.mc_confidence_sequence_upper:.3%}]"
            )
        )
        stopping_status = (
            "legacy fixed-sample"
            if estimate.mc_stopping_status is None
            else estimate.mc_stopping_status.value
        )
        platform_integer = (
            "--"
            if estimate.platform_submission_integer is None
            else f"{estimate.platform_submission_integer}%"
        )
        lines.append(
            f"| {estimate.prop.original_text or estimate.prop.kind} | "
            f"{estimate.yes_paths} | {estimate.no_paths} | {estimate.void_paths} | "
            f"{estimate.unresolved_paths} | {estimate.settled_paths} | {raw} | {integer} | "
            f"{confidence_sequence} | {stopping_status} | {estimate.total_paths} | "
            f"{estimate.probability_settled:.2%} | {estimate.support_status.value} | "
            f"{platform_integer} | {estimate.data_grade} |"
        )
        if estimate.model_probability_raw == 0.0:
            lines.append(
                "  - Endpoint interpretation: 0% means no Yes outcomes were observed among "
                "settled simulated paths; it does not prove the model probability is zero."
            )
        elif estimate.model_probability_raw == 1.0:
            lines.append(
                "  - Endpoint interpretation: 100% means every settled simulated path was "
                "Yes; it does not prove the model probability is one."
            )
        if estimate.policy_issue is not None:
            lines.append(f"  - Policy issue for `{estimate.prop_id}`: {estimate.policy_issue}")
        if estimate.support_reason_code is not None:
            lines.append(
                f"  - Support gate for `{estimate.prop_id}`: "
                f"{estimate.support_reason_code} — {estimate.support_detail}"
            )
        if (
            duration_requested
            and _contains_duration(estimate.prop)
            and estimate.sensitivity_low is not None
            and estimate.sensitivity_high is not None
            and estimate.sensitivity_low != estimate.sensitivity_high
        ):
            lines.append(
                f"  - Display-policy sensitivity for `{estimate.prop_id}`: "
                f"{estimate.sensitivity_low:.2%} to {estimate.sensitivity_high:.2%}"
            )
    for gate in lock.prop_gates:
        lines.append(
            f"| {gate.prop.original_text or gate.prop.kind} | 0 | 0 | 0 | 0 | 0 | -- | -- | "
            f"-- | UNAVAILABLE | 0 | -- | {gate.support_status.value} | -- | -- |"
        )
        lines.append(f"  - Support gate for `{gate.prop_id}`: {gate.reason_code} — {gate.detail}")
    lines.extend(["", "## Audit and sensitivities", ""])
    lines.extend(f"- Warning: {warning}" for warning in lock.warnings)
    lines.extend(f"- Check: {check}" for check in lock.validation_checks)
    lines.extend(["", "LOCK STATUS: LOCKED", ""])
    return "\n".join(lines)
