from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import METRIC_NAMES, average_metrics
from .schema import (
    DEFAULT_SEED,
    QUERY_LABELS,
    SAME_POOL_VARIANTS,
    Benchmark,
    atomic_write_bytes,
    atomic_write_json,
    candidate_pool_sha256,
    canonical_json,
    deduplicate_ranking,
    normalize_query,
    ranking_ids,
)


def _group_items(artifacts: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(artifacts, Mapping):
        return [(str(group), dict(artifact)) for group, artifact in sorted(artifacts.items())]
    result = []
    for artifact in artifacts:
        value = dict(artifact)
        group = str(value.get("query_group", value.get("group_id", "")))
        if not group:
            raise ValueError("Group artifact has no query_group")
        result.append((group, value))
    return sorted(result)


def _numeric_summary(values: Iterable[Any], *, total_count: int | None = None) -> dict[str, Any]:
    materialized = list(values)
    observed = [float(value) for value in materialized if value is not None and not isinstance(value, bool)]
    total = len(materialized) if total_count is None else total_count
    missing = total - len(observed)
    complete = missing == 0 and total > 0
    return {
        "count": total,
        "observed_count": len(observed),
        "missing_count": missing,
        "coverage": len(observed) / total if total else None,
        "sum": math.fsum(observed) if complete else None,
        "observed_sum": math.fsum(observed) if observed else None,
        "mean": math.fsum(observed) / len(observed) if observed else None,
        "median": float(np.median(observed)) if observed else None,
        "p95": float(np.quantile(observed, 0.95)) if observed else None,
        "min": min(observed) if observed else None,
        "max": max(observed) if observed else None,
    }


def _stage_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_calls = []
    for key in ("llm_calls", "calls"):
        if isinstance(artifact.get(key), list):
            raw_calls = [dict(record) for record in artifact[key]]
            break
    top_usage = [dict(record) for record in artifact.get("token_usage", [])] if isinstance(artifact.get("token_usage"), list) else []
    has_nested_usage = any(isinstance(call.get("token_usage"), list) for call in raw_calls)
    if top_usage or has_nested_usage:
        records = [dict(record) for record in top_usage]
        matched = set()
        extras = []

        def enrich(usage: Mapping[str, Any], call: Mapping[str, Any]) -> dict[str, Any]:
            record = dict(usage)
            record.setdefault("stage", call.get("stage"))
            record.setdefault("model", call.get("model"))
            record["status"] = call.get("status", record.get("status", "success"))
            record["events"] = call.get("events", record.get("events", []))
            record["fallback_used"] = call.get("fallback_used", record.get("fallback_used", False))
            record["attempts"] = call.get("attempts", record.get("attempts", 1))
            if call.get("latency_ms") is not None:
                record["stage_latency_ms"] = call["latency_ms"]
            return record

        for call in raw_calls:
            nested = call.get("token_usage")
            if isinstance(nested, list):
                for raw_usage in nested:
                    usage = dict(raw_usage)
                    match = next(
                        (
                            index
                            for index, record in enumerate(records)
                            if index not in matched and record == usage
                        ),
                        None,
                    )
                    if match is None:
                        extras.append(enrich(usage, call))
                    else:
                        records[match] = enrich(records[match], call)
                        matched.add(match)
            elif any(key in call for key in ("prompt_tokens", "completion_tokens", "total_tokens", "token_usage_status")):
                if not any(record == call for record in records):
                    extras.append(call)
            else:
                attempts = max(1, int(call.get("attempts", 1)))
                for attempt in range(1, attempts + 1):
                    extras.append(
                        enrich(
                            {
                                "stage": call.get("stage"),
                                "model": call.get("model"),
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None,
                                "token_usage_status": "missing",
                                "attempt": attempt,
                                "latency_ms": None,
                            },
                            call,
                        )
                    )
        records.extend(extras)
        return records
    if raw_calls:
        return raw_calls
    records = []
    stages = artifact.get("stages", {})
    if isinstance(stages, Mapping):
        for stage, value in stages.items():
            if isinstance(value, Mapping) and isinstance(value.get("calls"), list):
                for call in value["calls"]:
                    record = dict(call)
                    record.setdefault("stage", stage)
                    records.append(record)
            elif isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("stage", stage)
                records.append(record)
    elif isinstance(stages, list):
        records.extend(dict(record) for record in stages)
    return records


def _raw_stage_calls(artifact: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    for key in ("llm_calls", "calls"):
        if isinstance(artifact.get(key), list):
            return [dict(record) for record in artifact[key]]
    return [dict(record) for record in records]


def _is_jina_record(record: Mapping[str, Any]) -> bool:
    stage = str(record.get("stage", record.get("component", ""))).casefold()
    model = str(record.get("model", "")).casefold()
    return "jina" in stage or "jina" in model or str(record.get("token_usage_status", "")) == "not_applicable" and "selector" in stage and "llm" not in stage


def _llm_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw in records:
        record = dict(raw)
        if record.get("token_usage_status") == "not_applicable" or _is_jina_record(record):
            continue
        stage = str(record.get("stage", "")).casefold()
        if stage in {"bm25", "dense", "rrf", "search", "retrieval", "fusion"}:
            continue
        result.append(record)
    return result


def _usage(record: Mapping[str, Any], field: str) -> float | int | None:
    if record.get("token_usage_status") == "missing":
        return None
    value = record.get(field)
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid token usage field: {field}")
    return value


def _event_name(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, Mapping):
        for key in ("event", "type", "error_class", "kind", "reason"):
            if event.get(key):
                return str(event[key])
    return "unspecified"


def _artifact_events(artifact: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[str]:
    events = []
    for key in ("reliability_events", "events", "errors"):
        raw = artifact.get(key, [])
        if isinstance(raw, list):
            events.extend(_event_name(event) for event in raw)
        elif isinstance(raw, Mapping):
            events.extend(_event_name(raw))
    for record in records:
        for key in ("events", "errors"):
            raw = record.get(key, [])
            if isinstance(raw, list):
                events.extend(_event_name(event) for event in raw)
        if record.get("error_class"):
            events.append(str(record["error_class"]))
    return events


def _latency_values(records: Iterable[Mapping[str, Any]]) -> list[float | int | None]:
    return [record.get("latency_ms") for record in records]


def _group_latency(artifact: Mapping[str, Any]) -> Any:
    for key in ("measured_group_latency_ms", "group_latency_ms", "latency_ms"):
        if key in artifact:
            return artifact[key]
    return None


def _budgets(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ledgers = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if {"requested_raw_hits", "actual_raw_hits"} <= set(value):
                ledgers.append(value)
                return
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(artifact.get("budget", {}))
    pool = artifact.get("candidate_pool", artifact.get("pool", {}))
    if isinstance(pool, Mapping):
        walk(pool.get("budget", {}))
    return ledgers


def aggregate_cost_reliability(
    group_artifacts: Any,
    *,
    variant: str | None = None,
    pricing_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    groups = _group_items(group_artifacts)
    all_records = []
    records_by_group = {}
    for group, artifact in groups:
        records = _stage_records(artifact)
        records_by_group[group] = records
        all_records.extend(records)
    llm_records = _llm_records(all_records)
    token_fields = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        token_fields[field] = _numeric_summary([_usage(record, field) for record in llm_records])
        token_fields[field]["unit"] = "tokens"
        token_fields[field]["missing_policy"] = "null_not_zero"
    group_usage = {}
    for group, records in records_by_group.items():
        llm = _llm_records(records)
        if not llm:
            group_usage[group] = "not_applicable"
            continue
        statuses = []
        for record in llm:
            usages = [_usage(record, field) for field in ("prompt_tokens", "completion_tokens", "total_tokens")]
            if all(value is None for value in usages):
                statuses.append("missing")
            elif any(value is None for value in usages):
                statuses.append("partial")
            else:
                statuses.append("reported")
        if all(status == "missing" for status in statuses):
            group_usage[group] = "missing"
        elif all(status == "reported" for status in statuses):
            group_usage[group] = "reported"
        else:
            group_usage[group] = "partial"
    usage_counts = Counter(group_usage.values())
    applicable_groups = len(groups) - usage_counts["not_applicable"]
    jina_records = [dict(record) for record in all_records if _is_jina_record(record)]
    stage_latencies = defaultdict(list)
    phase_latencies = defaultdict(list)
    attempt_latencies = defaultdict(list)
    for group, artifact in groups:
        for call in _raw_stage_calls(artifact, records_by_group[group]):
            stage = str(call.get("stage", "unspecified"))
            if call.get("latency_ms") is not None:
                stage_latencies[stage].append(call["latency_ms"])
        if isinstance(artifact.get("stage_timings_ms"), Mapping):
            for stage, latency in artifact["stage_timings_ms"].items():
                phase_latencies[str(stage)].append(latency)
    for record in all_records:
        if record.get("latency_ms") is not None:
            attempt_latencies[str(record.get("stage", "unspecified"))].append(record["latency_ms"])
    measured_latencies = [_group_latency(artifact) for _, artifact in groups]
    reconstructed_latencies = [
        artifact.get("reconstructed_full_system_latency_ms", artifact.get("reconstructed_latency_ms"))
        for _, artifact in groups
    ]
    reconstructed_costs = [
        artifact.get("reconstructed_full_system_cost", artifact.get("reconstructed_cost"))
        for _, artifact in groups
    ]
    actual_cost = None
    observed_cost = None
    pricing_status = "not_available"
    if pricing_profile is not None:
        price_date = pricing_profile.get("priced_at", pricing_profile.get("price_date"))
        required_prices = ("prompt_per_1m", "completion_per_1m")
        if not price_date or any(pricing_profile.get(key) is None for key in required_prices):
            raise ValueError("Pricing profile requires a price date and prompt/completion prices")
        pricing_status = "documented"
        individual_costs = []
        for record in llm_records:
            prompt = _usage(record, "prompt_tokens")
            completion = _usage(record, "completion_tokens")
            if prompt is None or completion is None:
                individual_costs.append(None)
            else:
                individual_costs.append(
                    float(prompt) * float(pricing_profile["prompt_per_1m"]) / 1_000_000
                    + float(completion) * float(pricing_profile["completion_per_1m"]) / 1_000_000
                )
        observed = [value for value in individual_costs if value is not None]
        observed_cost = math.fsum(observed) if observed else None
        actual_cost = math.fsum(observed) if len(observed) == len(individual_costs) and individual_costs else None
    fallbacks = 0
    failures = 0
    repairs = 0
    retry_calls = 0
    transient_retries_total = 0
    empty_outputs = 0
    short_outputs = 0
    saturated_groups = 0
    stage_statuses = Counter()
    fallback_stages = Counter()
    all_events = []
    requested = 0
    actual = 0
    target = 0
    unspent = 0
    unique_candidates = []
    for group, artifact in groups:
        records = records_by_group[group]
        status = str(artifact.get("status", artifact.get("group_status", "success")))
        fallback_used = bool(artifact.get("fallback_used")) or status == "fallback"
        fallbacks += fallback_used
        failures += status in {"failed", "error"}
        declared_fallbacks = artifact.get("fallbacks", [])
        if isinstance(declared_fallbacks, list) and declared_fallbacks:
            for fallback in declared_fallbacks:
                if isinstance(fallback, Mapping) and fallback.get("stage"):
                    fallback_stages[str(fallback["stage"])] += 1
        elif artifact.get("fallback_stage"):
            fallback_stages[str(artifact["fallback_stage"])] += 1
        try:
            output = deduplicate_ranking(artifact, max_items=10)
            empty_outputs += not output
            short_outputs += 0 < len(output) < 10
        except (ValueError, KeyError, TypeError):
            pass
        ledgers = _budgets(artifact)
        saturated_groups += any(ledger.get("channel_status", ledger.get("status")) == "saturated" for ledger in ledgers)
        target += sum(int(ledger.get("target_raw_hits", 0)) for ledger in ledgers)
        requested += sum(int(ledger["requested_raw_hits"]) for ledger in ledgers)
        actual += sum(int(ledger["actual_raw_hits"]) for ledger in ledgers)
        unspent += sum(int(ledger.get("unspent_raw_hits", int(ledger["requested_raw_hits"]) - int(ledger["actual_raw_hits"]))) for ledger in ledgers)
        pool = artifact.get("candidate_pool", artifact.get("pool", {}))
        if isinstance(pool, Mapping) and isinstance(pool.get("candidates"), list):
            unique_candidates.append(len({str(candidate["chunk_id"]) for candidate in pool["candidates"]}))
        else:
            unique_candidates.append(None)
        group_repaired = status == "success_after_json_repair"
        for call in _raw_stage_calls(artifact, records):
            record_status = str(call.get("status", "success"))
            stage = str(call.get("stage", "unspecified"))
            stage_statuses[f"{stage}:{record_status}"] += 1
            group_repaired = group_repaired or record_status == "success_after_json_repair"
            attempts = int(call.get("attempts", 1))
            call_transient_retries = 0
            raw_events = call.get("events", [])
            if isinstance(raw_events, list):
                for event in raw_events:
                    event_attempt = int(event.get("attempt", 1)) if isinstance(event, Mapping) else 1
                    if _event_name(event) in {"timeout", "rate_limit", "provider_error"} and event_attempt < attempts:
                        call_transient_retries += 1
            transient_retries_total += call_transient_retries
            retry_calls += call_transient_retries > 0
        repairs += group_repaired
        all_events.extend(_artifact_events(artifact, records))
    event_counts = Counter(all_events)
    denominator = len(groups)
    reliability = {
        "groups": denominator,
        "fallback_groups": fallbacks,
        "fallback_rate": fallbacks / denominator if denominator else None,
        "failed_groups": failures,
        "failure_rate": failures / denominator if denominator else None,
        "repair_groups": repairs,
        "repair_rate": repairs / denominator if denominator else None,
        "retry_calls": retry_calls,
        "transient_retries_total": transient_retries_total,
        "timeout_groups": sum(any(_event_name(event) == "timeout" for event in _artifact_events(artifact, records_by_group[group])) for group, artifact in groups),
        "malformed_json_events": event_counts["malformed_json"],
        "event_counts": dict(sorted(event_counts.items())),
        "stage_status_counts": dict(sorted(stage_statuses.items())),
        "fallback_stage_groups": dict(sorted(fallback_stages.items())),
        "empty_output_groups": empty_outputs,
        "empty_output_rate": empty_outputs / denominator if denominator else None,
        "short_output_groups": short_outputs,
        "short_output_rate": short_outputs / denominator if denominator else None,
        "saturated_groups": saturated_groups,
        "saturation_group_rate": saturated_groups / denominator if denominator else None,
    }
    cost = {
        "variant": variant,
        "groups": denominator,
        "llm_calls_total": len(llm_records),
        "llm_calls_per_group": len(llm_records) / denominator if denominator else None,
        "tokens": token_fields,
        "prompt_tokens_sum": token_fields["prompt_tokens"]["sum"],
        "completion_tokens_sum": token_fields["completion_tokens"]["sum"],
        "total_tokens_sum": token_fields["total_tokens"]["sum"],
        "prompt_tokens_observed_sum": token_fields["prompt_tokens"]["observed_sum"],
        "completion_tokens_observed_sum": token_fields["completion_tokens"]["observed_sum"],
        "total_tokens_observed_sum": token_fields["total_tokens"]["observed_sum"],
        "token_usage_groups": dict(sorted(group_usage.items())),
        "token_usage_group_counts": dict(sorted(usage_counts.items())),
        "token_usage_group_coverage": {
            status: (usage_counts[status] / applicable_groups if applicable_groups else None)
            for status in ("reported", "missing", "partial")
        },
        "token_usage_call_coverage": {
            field: values["coverage"] for field, values in token_fields.items()
        },
        "jina_local_compute": {
            "calls": len(jina_records),
            "latency_ms": _numeric_summary(_latency_values(jina_records)),
            "models": sorted({str(record["model"]) for record in jina_records if record.get("model")}),
            "devices": sorted({str(record["device"]) for record in jina_records if record.get("device")}),
            "llm_api_tokens": None,
            "accounting": "local_compute_separate_from_llm_api_tokens",
        },
        "stage_latency_ms": {stage: _numeric_summary(values) for stage, values in sorted(stage_latencies.items())},
        "phase_latency_ms": {stage: _numeric_summary(values) for stage, values in sorted(phase_latencies.items())},
        "attempt_latency_ms": {stage: _numeric_summary(values) for stage, values in sorted(attempt_latencies.items())},
        "measured_group_latency_ms": _numeric_summary(measured_latencies),
        "target_raw_hits": target,
        "requested_raw_hits": requested,
        "actual_raw_hits": actual,
        "unspent_raw_hits": unspent,
        "unique_candidates": _numeric_summary(unique_candidates),
        "monetary_cost": actual_cost,
        "observed_monetary_cost": observed_cost,
        "pricing_status": pricing_status,
        "pricing_date": pricing_profile.get("priced_at", pricing_profile.get("price_date")) if pricing_profile else None,
        "reconstructed_full_system_cost": _numeric_summary(reconstructed_costs),
        "reconstructed_full_system_latency_ms": _numeric_summary(reconstructed_latencies),
        "missing_usage_policy": "null_aggregates_with_coverage_never_zero_imputation",
    }
    return {"cost": cost, "reliability": reliability}


def aggregate_variant_cost_reliability(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return aggregate_cost_reliability(*args, **kwargs)


def core_metrics_table(scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, score in scores.items():
        summary = score["summary"]
        rows.append(
            {
                "variant": variant,
                "answerable_groups": summary["answerable_unique_queries"],
                "answerable_rows": summary["answerable_rows"],
                "groups_total": summary["groups_total"],
                "rows_without_qrels": summary["rows_without_qrels"],
                **{f"unique_{key}": value for key, value in summary["unique_query_macro"].items()},
                **{f"row_{key}": value for key, value in summary["row_macro"].items()},
                "fallback_groups": sum(bool(row.get("fallback_used")) for row in score.get("all_groups", [])),
                "output_policy": summary["output_policy"],
            }
        )
    return rows


def kendall_tau_common(left: Any, right: Any) -> float | None:
    left_ids = deduplicate_ranking(left, max_items=None, reject_duplicates=True)
    right_ids = deduplicate_ranking(right, max_items=None, reject_duplicates=True)
    common = sorted(set(left_ids) & set(right_ids))
    if len(common) < 2:
        return None
    left_rank = {chunk_id: rank for rank, chunk_id in enumerate(left_ids)}
    right_rank = {chunk_id: rank for rank, chunk_id in enumerate(right_ids)}
    concordant = 0
    discordant = 0
    for index, first in enumerate(common):
        for second in common[index + 1 :]:
            product = (left_rank[first] - left_rank[second]) * (right_rank[first] - right_rank[second])
            concordant += product > 0
            discordant += product < 0
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else None


def top10_overlap(left: Any, right: Any) -> dict[str, Any]:
    left_ids = deduplicate_ranking(left, max_items=10, reject_duplicates=True)
    right_ids = deduplicate_ranking(right, max_items=10, reject_duplicates=True)
    left_set = set(left_ids)
    right_set = set(right_ids)
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "left_count": len(left_ids),
        "right_count": len(right_ids),
        "intersection_count": len(intersection),
        "overlap_fraction_of_10": len(intersection) / 10,
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "ordered_overlap": sum(first == second for first, second in zip(left_ids, right_ids)) / max(len(left_ids), len(right_ids), 1),
    }


def _full_ranking(artifact: Mapping[str, Any]) -> list[str]:
    for key in ("full_ranked_chunk_ids", "selector_permutation", "permutation"):
        if key in artifact:
            return deduplicate_ranking(artifact[key], max_items=None, reject_duplicates=True)
    return deduplicate_ranking(artifact, max_items=10, reject_duplicates=True)


def _pool_from_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    pool = artifact.get("candidate_pool", artifact.get("pool", {}))
    return pool if isinstance(pool, Mapping) else {}


def _pool_hash(artifact: Mapping[str, Any]) -> str | None:
    pool = _pool_from_artifact(artifact)
    value = artifact.get("pool_sha256", pool.get("pool_sha256"))
    if value:
        return str(value)
    if isinstance(pool.get("candidates"), list):
        return candidate_pool_sha256(pool["candidates"])
    return None


def _supporting_ids(artifact: Mapping[str, Any]) -> set[str]:
    result = set()

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            if key == "supporting_chunk_ids":
                result.update(str(item) for item in value)
            else:
                for child in value:
                    walk(child, key)

    walk(artifact.get("reports", {}))
    return result


def rank_movement(
    left_ranking: Any,
    right_ranking: Any,
    relevant: Mapping[str, int],
    *,
    supporting_chunk_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    left = deduplicate_ranking(left_ranking, max_items=None, reject_duplicates=True)
    right = deduplicate_ranking(right_ranking, max_items=None, reject_duplicates=True)
    left_ranks = {chunk_id: rank for rank, chunk_id in enumerate(left, 1)}
    right_ranks = {chunk_id: rank for rank, chunk_id in enumerate(right, 1)}
    supporting = set(supporting_chunk_ids)
    rows = []
    for chunk_id in sorted(set(relevant) & (set(left) | set(right))):
        old_rank = left_ranks.get(chunk_id)
        new_rank = right_ranks.get(chunk_id)
        movement = old_rank - new_rank if old_rank is not None and new_rank is not None else None
        rows.append(
            {
                "chunk_id": chunk_id,
                "relevance": int(relevant[chunk_id]),
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_improvement": movement,
                "entered_top10": (old_rank is None or old_rank > 10) and new_rank is not None and new_rank <= 10,
                "exited_top10": old_rank is not None and old_rank <= 10 and (new_rank is None or new_rank > 10),
                "moved_up": movement is not None and movement > 0,
                "moved_down": movement is not None and movement < 0,
                "report_supported": chunk_id in supporting,
            }
        )
    return rows


def same_pool_group_analysis(
    left_artifact: Mapping[str, Any],
    right_artifact: Mapping[str, Any],
    relevant: Mapping[str, int],
    *,
    contrast: str,
) -> dict[str, Any]:
    left_hash = _pool_hash(left_artifact)
    right_hash = _pool_hash(right_artifact)
    if left_hash is None or right_hash is None or left_hash != right_hash:
        raise ValueError(f"Same-pool hash mismatch for {contrast}")
    left_pool = _pool_from_artifact(left_artifact)
    right_pool = _pool_from_artifact(right_artifact)
    if left_pool.get("candidates") is not None and right_pool.get("candidates") is not None:
        left_visible_hash = candidate_pool_sha256(left_pool["candidates"])
        right_visible_hash = candidate_pool_sha256(right_pool["candidates"])
        if left_visible_hash != right_visible_hash:
            raise ValueError(f"Selector-visible pool mismatch for {contrast}")
        if left_hash != left_visible_hash or right_hash != right_visible_hash:
            raise ValueError(f"Declared pool hash does not match pool content for {contrast}")
    full_ranking_available = all(
        any(key in artifact for key in ("full_ranked_chunk_ids", "selector_permutation", "permutation"))
        for artifact in (left_artifact, right_artifact)
    )
    left_full = _full_ranking(left_artifact)
    right_full = _full_ranking(right_artifact)
    left_top = left_full[:10]
    right_top = right_full[:10]
    supporting = _supporting_ids(right_artifact)
    movement = rank_movement(left_full, right_full, relevant, supporting_chunk_ids=supporting)
    result = {
        "contrast": contrast,
        "query_group": left_artifact.get("query_group", right_artifact.get("query_group")),
        "pool_sha256": left_hash,
        **top10_overlap(left_top, right_top),
        "kendall_tau_common": kendall_tau_common(left_full, right_full),
        "relevant_entered_top10": sum(row["entered_top10"] for row in movement),
        "relevant_exited_top10": sum(row["exited_top10"] for row in movement),
        "rank_movement": movement,
        "report_supported_moved_up": sorted(row["chunk_id"] for row in movement if row["report_supported"] and row["moved_up"]),
        "report_supported_moved_down": sorted(row["chunk_id"] for row in movement if row["report_supported"] and row["moved_down"]),
        "trace_is_not_causal_proof": True,
        "ranking_scope": "full_permutation" if full_ranking_available else "observed_top10_only",
    }
    if left_pool.get("candidates") is not None and relevant:
        left_recall = len({str(candidate["chunk_id"]) for candidate in left_pool["candidates"]} & set(relevant)) / len(relevant)
        right_recall = len({str(candidate["chunk_id"]) for candidate in right_pool["candidates"]} & set(relevant)) / len(relevant)
        if left_recall != right_recall:
            raise ValueError(f"Pool recall changed for {contrast}")
        result["pool_recall"] = left_recall
        result["pool_recall_constant"] = True
    return result


def same_pool_ranking_analysis(
    artifacts: Mapping[str, Any],
    relevant: Benchmark | Mapping[str, Mapping[str, int]],
    *,
    pairs: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    selected_pairs = pairs or {"M3-M2": ("M3", "M2"), "M4-M3": ("M4", "M3")}
    rows = []
    for contrast, (left_variant, right_variant) in selected_pairs.items():
        left_groups = dict(_group_items(artifacts[left_variant]))
        right_groups = dict(_group_items(artifacts[right_variant]))
        if set(left_groups) != set(right_groups):
            raise ValueError(f"Same-pool group mismatch for {contrast}")
        for group in sorted(left_groups):
            if isinstance(relevant, Benchmark):
                qrels = relevant.group_by_id[group].relevant
            else:
                qrels = relevant[group]
            if not qrels:
                continue
            rows.append(
                same_pool_group_analysis(
                    left_groups[group],
                    right_groups[group],
                    qrels,
                    contrast=contrast,
                )
            )
    summary = []
    for contrast in selected_pairs:
        subset = [row for row in rows if row["contrast"] == contrast]
        summary.append(
            {
                "contrast": contrast,
                "answerable_groups": len(subset),
                "top10_jaccard_mean": _numeric_summary(row["jaccard"] for row in subset)["mean"],
                "kendall_tau_common_mean": _numeric_summary(row["kendall_tau_common"] for row in subset)["mean"],
                "relevant_entered_top10": sum(row["relevant_entered_top10"] for row in subset),
                "relevant_exited_top10": sum(row["relevant_exited_top10"] for row in subset),
            }
        )
    return {"rows": rows, "summary": summary}


def select_stability_groups(
    query_groups: Iterable[str],
    *,
    count: int = 50,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    unique = sorted(set(query_groups))
    selected = sorted(
        unique,
        key=lambda group: hashlib.sha256(f"{group}{seed}".encode("utf-8")).hexdigest(),
    )[:count]
    return sorted(selected)


def _generated_queries(artifact: Mapping[str, Any]) -> set[str]:
    result = set()

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            if key in {"queries", "generated_queries", "lexical_queries", "semantic_queries"}:
                for child in value:
                    if isinstance(child, str) and child.strip():
                        result.add(normalize_query(child))
                    else:
                        walk(child, key)
            else:
                for child in value:
                    walk(child, key)

    walk(artifact.get("generated_queries", {}))
    walk(_pool_from_artifact(artifact).get("generated_queries", {}))
    walk(artifact.get("worker_queries", {}))
    return result


def _repair_used(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("status") == "success_after_json_repair":
        return True
    return any(record.get("status") == "success_after_json_repair" for record in _stage_records(artifact))


def _set_comparison(left: set[str], right: set[str]) -> dict[str, Any]:
    union = left | right
    return {
        "exact_set_match": left == right,
        "jaccard": len(left & right) / len(union) if union else 1.0,
    }


def compare_stability_artifacts(base: Mapping[str, Any], repeat: Mapping[str, Any]) -> dict[str, Any]:
    base_queries = _generated_queries(base)
    repeat_queries = _generated_queries(repeat)
    base_pool = _pool_from_artifact(base)
    repeat_pool = _pool_from_artifact(repeat)
    base_candidates = [str(candidate["chunk_id"]) for candidate in base_pool.get("candidates", [])]
    repeat_candidates = [str(candidate["chunk_id"]) for candidate in repeat_pool.get("candidates", [])]
    query_comparison = _set_comparison(base_queries, repeat_queries)
    pool_comparison = _set_comparison(set(base_candidates), set(repeat_candidates))
    base_top = deduplicate_ranking(base, max_items=10)
    repeat_top = deduplicate_ranking(repeat, max_items=10)
    top_comparison = _set_comparison(set(base_top), set(repeat_top))
    return {
        "query_group": base.get("query_group", repeat.get("query_group")),
        "exact_query_set_match": query_comparison["exact_set_match"] if base_queries or repeat_queries else None,
        "generated_query_jaccard": query_comparison["jaccard"] if base_queries or repeat_queries else None,
        "pool_id_jaccard": pool_comparison["jaccard"],
        "pool_ordered_overlap": sum(first == second for first, second in zip(base_candidates, repeat_candidates)) / max(len(base_candidates), len(repeat_candidates), 1),
        "pool_sha256_exact_match": _pool_hash(base) is not None and _pool_hash(base) == _pool_hash(repeat),
        "top10_jaccard": top_comparison["jaccard"],
        "kendall_tau_common": kendall_tau_common(base_top, repeat_top),
        "fallback_changed": bool(base.get("fallback_used")) != bool(repeat.get("fallback_used")),
        "repair_changed": _repair_used(base) != _repair_used(repeat),
    }


def _run_metric(run: Mapping[str, Mapping[str, Any]], metric: str) -> dict[str, float | None]:
    return {
        group: value.get("metrics", {}).get(metric, value.get(metric))
        for group, value in run.items()
    }


def summarize_stability(
    runs: Mapping[str, Sequence[Mapping[str, Mapping[str, Any]]]],
    *,
    metric: str = "recall@10",
) -> dict[str, Any]:
    comparison_rows = []
    variant_rows = []
    for variant, repeat_runs in runs.items():
        if len(repeat_runs) < 2:
            raise ValueError(f"Stability variant {variant} requires a base and at least one repeat")
        normalized_runs = [dict(_group_items(run)) for run in repeat_runs]
        base = normalized_runs[0]
        variant_comparisons = []
        for repeat_index, repeat in enumerate(normalized_runs[1:], 1):
            if set(base) != set(repeat):
                raise ValueError(f"Stability groups differ for {variant}")
            for group in sorted(base):
                row = compare_stability_artifacts(base[group], repeat[group])
                row.update({"variant": variant, "repeat": repeat_index})
                comparison_rows.append(row)
                variant_comparisons.append(row)
        metric_values = [_run_metric(run, metric) for run in normalized_runs]
        ranges = []
        for group in sorted(base):
            values = [values[group] for values in metric_values if values[group] is not None]
            ranges.append(max(values) - min(values) if values else None)
        per_run_cost = [aggregate_cost_reliability(run, variant=variant)["cost"] for run in normalized_runs]
        token_totals = [cost["total_tokens_sum"] for cost in per_run_cost]
        latency_means = [cost["measured_group_latency_ms"]["mean"] for cost in per_run_cost]
        variant_rows.append(
            {
                "variant": variant,
                "groups": len(base),
                "additional_repeats": len(normalized_runs) - 1,
                "comparisons": len(variant_comparisons),
                "exact_query_set_match_rate": _numeric_summary(float(row["exact_query_set_match"]) if row["exact_query_set_match"] is not None else None for row in variant_comparisons)["mean"],
                "generated_query_jaccard_mean": _numeric_summary(row["generated_query_jaccard"] for row in variant_comparisons)["mean"],
                "pool_id_jaccard_mean": _numeric_summary(row["pool_id_jaccard"] for row in variant_comparisons)["mean"],
                "pool_ordered_overlap_mean": _numeric_summary(row["pool_ordered_overlap"] for row in variant_comparisons)["mean"],
                "pool_sha256_exact_match_rate": _numeric_summary(float(row["pool_sha256_exact_match"]) for row in variant_comparisons)["mean"],
                "top10_jaccard_mean": _numeric_summary(row["top10_jaccard"] for row in variant_comparisons)["mean"],
                "kendall_tau_common_mean": _numeric_summary(row["kendall_tau_common"] for row in variant_comparisons)["mean"],
                "metric_range_mean": _numeric_summary(ranges)["mean"],
                "metric_range_max": _numeric_summary(ranges)["max"],
                "fallback_change_rate": _numeric_summary(float(row["fallback_changed"]) for row in variant_comparisons)["mean"],
                "repair_change_rate": _numeric_summary(float(row["repair_changed"]) for row in variant_comparisons)["mean"],
                "llm_calls_range": max(cost["llm_calls_total"] for cost in per_run_cost) - min(cost["llm_calls_total"] for cost in per_run_cost),
                "total_tokens_observed_range": _numeric_summary(token_totals)["max"] - _numeric_summary(token_totals)["min"] if all(value is not None for value in token_totals) else None,
                "total_tokens_complete_coverage": all(value is not None for value in token_totals),
                "measured_group_latency_range_ms": max(latency_means) - min(latency_means) if all(value is not None for value in latency_means) else None,
                "independent_primary_n_added": 0,
            }
        )
    return {"rows": comparison_rows, "variants": variant_rows}


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    if columns is None:
        columns = list(rows[0]) if rows else []
    if not columns:
        return "Нет данных."

    def format_value(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, (dict, list, tuple)):
            return canonical_json(value).replace("|", "\\|")
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


DEFAULT_LIMITATIONS = (
    "Benchmark является development benchmark, а не независимым test set.",
    "Известные неточности qrels и исходных вопросов сохранены без неявной коррекции.",
    "Независимое обобщение на другом наборе данных не проверялось.",
    "Возможная contamination LLM не исключена.",
    "Часть литературы имеет статус preprint.",
    "Подгруппы с шестью multi-label аннотациями пересекаются.",
    "Отсутствующий token usage не заменяется нулём и ограничивает стоимостные выводы.",
)


def render_report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    trace_examples: Sequence[str] = (),
    limitations: Sequence[str] = DEFAULT_LIMITATIONS,
) -> str:
    titles = {
        "T1": "T1. Качество десяти систем, unique-query macro",
        "T2": "T2. Candidate pool quality и вклад BM25/Dense",
        "T3": "T3. Подтверждающие контрасты C1–C4",
        "T4": "T4. Same-pool ranking и перемещения relevant IDs",
        "T5": "T5. Шесть НПА-labels и interactions",
        "T6": "T6. Calls, tokens, latency, search cost и missing usage",
        "T7": "T7. Retries, repairs, fallback, saturation и failures",
        "T8": "T8. Stability",
    }
    sections = ["# Отчёт npa_mas_component_v1", "", "Отчёт содержит наблюдения и не принимает автоматического публикационного решения."]
    for table_id in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"):
        sections.extend(["", f"## {titles[table_id]}", "", markdown_table(tables.get(table_id, []))])
    sections.extend(["", "## Trace examples", ""])
    sections.extend(f"- {example}" for example in trace_examples[:5])
    if not trace_examples:
        sections.append("Примеры не переданы.")
    sections.extend(["", "## Limitations", ""])
    sections.extend(f"- {limitation}" for limitation in limitations)
    return "\n".join(sections) + "\n"


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str] | None = None) -> Path:
    if columns is None:
        columns = list(rows[0]) if rows else []
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
        )
    return atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def write_analysis_report(
    output_dir: str | Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    trace_examples: Sequence[str] = (),
    limitations: Sequence[str] = DEFAULT_LIMITATIONS,
) -> Path:
    destination = Path(output_dir)
    report = render_report(tables, trace_examples=trace_examples, limitations=limitations)
    return atomic_write_bytes(destination / "report.md", report.encode("utf-8"))
