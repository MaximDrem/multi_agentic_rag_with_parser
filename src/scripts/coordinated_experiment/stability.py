import csv
import io
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.coordinated_experiment import analysis
from scripts.coordinated_experiment.decision import STABLE
from scripts.coordinated_experiment.legacy_reuse import load_snapshots
from scripts.coordinated_experiment.metrics import rank_overlap_and_correlation, score_group_records
from scripts.coordinated_experiment.pools import assert_same_pool
from scripts.coordinated_experiment.schema import atomic_write_json, atomic_write_text, load_config
from scripts.score_retrieval import load_qrels, load_questions

EXECUTION_BATCHES = ("main", "repeat_1", "repeat_2")
COMMUNICATION_CONTRASTS = ("MECH1", "MECH2", "MECH3", "MECH4", "MECH5", "SEL4", "SEL5")
ITERATIVE_CONTRASTS = ("MECH1", "SEL5")
CANDIDATE_CONTRASTS = ("CG2", "CG4", "CG5", "CG6", "CG7")
FULL_SELECTOR_CONTRASTS = ("SEL1", "SEL2", "SEL3")
CSV_FIELDS = (
    "row_type", "execution_batch", "execution_index", "status", "reason", "contrast_id", "family",
    "config_id", "left_config", "right_config", "metric", "aggregation", "strict_or_backfill",
    "fallback_policy", "n_groups", "expected_n_groups", "delta", "delta_pp", "sign", "stable",
    "left_recall@10", "right_recall@10", "left_ndcg@10", "right_ndcg@10", "ranking_overlap@10",
    "mean_spearman", "changed_selection_fraction", "changed_group_fraction", "changed_selection_group_fraction",
    "questions_total", "run_rows", "groups_total", "answerable_rows", "answerable_unique_queries",
    "expected_answerable_rows", "expected_answerable_groups", "rows_without_qrels", "complete_group_coverage",
    "recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10", "hit_rate@5", "hit_rate@10",
    "recall@10_spread", "ndcg@10_spread", "delta_spread", "repair_groups", "repair_rate", "fallback_groups",
    "fallback_rate", "fallback_stages", "prompt_tokens_total", "completion_tokens_total", "llm_tokens_total",
    "cached_reused_selector_tokens_total", "selector_latency_p50_ms", "selector_latency_p95_ms",
    "jina_latency_p50_ms", "jina_latency_p95_ms", "reused_from_execution_batch", "reuse_reason",
    "missing_executions", "missing_groups",
)


def _read_json(path: Path, *, required: bool = True) -> Any:
    if not path.is_file():
        if required:
            raise ValueError(f"Missing stability input: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read stability input {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Stability input must be an object: {path}")
    return dict(value)


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _percentile(values: Sequence[Any], q: float) -> float | None:
    numbers = sorted(float(value) for value in (_number(value) for value in values) if value is not None)
    if not numbers:
        return None
    position = (len(numbers) - 1) * q / 100
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)


def _complete_sum(values: Sequence[Any]) -> int | float | None:
    if not values:
        return 0
    numbers = [_number(value) for value in values]
    if any(value is None for value in numbers):
        return None
    result = float(sum(numbers))
    return int(result) if result.is_integer() else result


def _has_fallback(record: Mapping[str, Any]) -> bool:
    return bool(record.get("fallback_stage") is not None or record.get("fallback_stages") or record.get("fallback_used"))


def _fallback_stages(record: Mapping[str, Any]) -> list[str]:
    stages = record.get("fallback_stages", [])
    if isinstance(stages, str):
        stages = [stages]
    result = [str(stage) for stage in stages] if isinstance(stages, Sequence) else []
    if record.get("fallback_stage") is not None:
        result.append(str(record["fallback_stage"]))
    return list(dict.fromkeys(result))


def _group_ids(value: Any) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result = []
    for item in value:
        group_id = item.get("group_id") if isinstance(item, Mapping) else item
        if not isinstance(group_id, str) or not group_id:
            return None
        result.append(group_id)
    return result


def _selected_group_ids(manifest: Mapping[str, Any], target: Mapping[str, Any], decision: Mapping[str, Any]) -> list[str]:
    candidates: list[list[str]] = []
    for document in (manifest, target, decision.get("stability_target", {})):
        if not isinstance(document, Mapping):
            continue
        for key in ("group_ids", "selected_group_ids", "groups", "selected_groups"):
            ids = _group_ids(document.get(key))
            if ids is not None:
                candidates.append(ids)
                break
    executions = manifest.get("executions", [])
    if isinstance(executions, Sequence):
        for execution in executions:
            if isinstance(execution, Mapping):
                ids = _group_ids(execution.get("group_ids"))
                if ids is not None:
                    candidates.append(ids)
    if not candidates:
        raise ValueError("Stability selected group IDs are unavailable")
    selected = candidates[0]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Stability selected group IDs must be nonempty and unique")
    if any(ids != selected for ids in candidates[1:]):
        raise ValueError("Stability selected group IDs differ between target and execution manifest")
    return selected


def _not_applicable(config_path: Path, experiment_dir: Path, decision: Mapping[str, Any], target: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    reason = str(target.get("reason") or "no LLM stability target")
    result = {
        "protocol_version": "coordinated_mechanism_v1",
        "status": "not_applicable",
        "reason": reason,
        "branch": decision.get("branch"),
        "route": decision.get("route"),
        "target": dict(target),
        "manifest_status": manifest.get("status") if manifest else "missing",
        "selected_group_ids": [],
        "runs": [],
        "total_executions": 0,
        "required_executions": [],
        "best_repeat_selected": False,
        "stable_contrasts": {},
        "contrast_stability": {},
        "not_repeated": [],
        "reuse": [],
        "metrics": {},
        "spreads": {},
        "ranking_stability": [],
        "smoke_excluded": True,
    }
    _write_outputs(experiment_dir, result, [_status_row(result)])
    return result


def _target_plan(decision: Mapping[str, Any], target: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    branch = str(decision.get("branch") or manifest.get("target_kind") or "")
    resolved = str(manifest.get("resolved_target") or branch)
    contrast_name = str(target.get("contrast") or "").casefold().replace(" ", "")
    if branch == "inconclusive" and resolved in {"", "inconclusive"}:
        if "r2np/r2p" in contrast_name:
            resolved = "coordinated_communication"
        elif "e/s/np" in contrast_name:
            resolved = "coordinated_candidate_generation"
        elif "e/np" in contrast_name or "e/s" in contrast_name:
            resolved = "simpler_candidate_generation"
        else:
            resolved = "inconclusive_full_selector"
    kind = resolved if branch == "inconclusive" else branch
    route = decision.get("route")
    configs = target.get("configs", [])
    config_text = " ".join(str(value) for value in configs) if isinstance(configs, Sequence) else ""
    if kind == "coordinated_communication":
        contrasts = COMMUNICATION_CONTRASTS
        fixed_pools = {"E"}
        regenerated: tuple[str, ...] = ()
        allow_r1 = True
    elif kind == "iterative_selection_without_exchange":
        contrasts = ITERATIVE_CONTRASTS
        fixed_pools = {"E"}
        regenerated = ()
        allow_r1 = True
    elif kind == "coordinated_candidate_generation":
        contrasts = CANDIDATE_CONTRASTS
        fixed_pools = {"B"}
        regenerated = ("E", "S", "NP", "W")
        allow_r1 = False
    elif kind == "simpler_candidate_generation":
        simple = str(route or ("NP" if "NP__JINA" in config_text or "e/np" in contrast_name else "S"))
        if simple not in {"S", "NP"}:
            raise ValueError(f"Unsupported simpler-candidate stability route: {simple}")
        route = simple
        contrasts = ("CG2", "CG4", "CG6") if simple == "S" else ("CG2", "CG5", "CG7")
        fixed_pools = {"B"}
        regenerated = ("E", simple, "W")
        allow_r1 = False
    elif kind == "generic_llm_reranking":
        simple = str(route or ("W" if "W__R0__ARB" in config_text else "B"))
        if simple not in {"B", "W"}:
            raise ValueError(f"Unsupported generic-arbiter stability route: {simple}")
        route = simple
        contrasts = ("SEL3", "GEN1", "GEN3") if simple == "B" else ("SEL3", "GEN2", "GEN4")
        fixed_pools = {"E", simple}
        regenerated = ()
        allow_r1 = False
    elif kind in {"inconclusive_full_selector", "selector_core"}:
        contrasts = FULL_SELECTOR_CONTRASTS
        fixed_pools = {"E"}
        regenerated = ()
        allow_r1 = False
    else:
        raise ValueError(f"Unsupported stability target: branch={branch!r}, resolved={resolved!r}")
    for contrast_id in contrasts:
        if contrast_id not in analysis.CONTRAST_REGISTRY:
            raise ValueError(f"Unknown stability contrast: {contrast_id}")
    return {
        "branch": branch,
        "kind": kind,
        "route": str(route) if route is not None else None,
        "contrast_ids": list(contrasts),
        "fixed_pools": fixed_pools,
        "regenerated_pools": regenerated,
        "allow_r1_reuse": allow_r1,
    }


def _generated_document(experiment_dir: Path, batch: str, group_id: str) -> dict[str, Any] | None:
    path = experiment_dir / "generated_pools" / batch / f"{group_id}.json"
    value = _read_json(path, required=False)
    if value is None:
        return None
    if value.get("group_id", group_id) != group_id or value.get("execution_batch", batch) != batch:
        raise ValueError(f"Generated pool identity mismatch: {path}")
    if not isinstance(value.get("pools"), Mapping):
        raise ValueError(f"Generated pools are missing: {path}")
    return value


def _generated_pool(document: Mapping[str, Any], batch: str, group_id: str, pool_id: str) -> dict[str, Any]:
    artifact = document["pools"].get(pool_id)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("pool_record"), Mapping):
        raise ValueError(f"Missing regenerated pool {pool_id} for {batch}/{group_id}")
    if artifact.get("execution_batch", batch) != batch:
        raise ValueError(f"Regenerated pool {pool_id} for {batch}/{group_id} comes from another execution")
    record = dict(artifact["pool_record"])
    assert_same_pool(record)
    return record


def _batch_snapshots(
    experiment_dir: Path,
    snapshots: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    batch: str,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    regenerated = set(plan["regenerated_pools"])
    views = [dict(snapshot) for snapshot in snapshots]
    by_id = {str(snapshot["group_id"]): snapshot for snapshot in views}
    valid_ids = {
        str(chunk_id)
        for snapshot in snapshots
        for pool in snapshot.get("pools", {}).values()
        if isinstance(pool, Mapping)
        for chunk_id in pool.get("chunk_ids", [])
    }
    provenance: dict[str, Any] = {}
    if not regenerated:
        return views, valid_ids, provenance
    for group_id in sorted(selected_ids):
        snapshot = by_id[group_id]
        pools = dict(snapshot.get("pools", {}))
        document = _generated_document(experiment_dir, batch, group_id)
        required = regenerated if batch != "main" else regenerated - {"E", "W"}
        loaded: dict[str, str] = {}
        for pool_id in sorted(required):
            if batch == "main" and isinstance(pools.get(pool_id), Mapping):
                continue
            if document is None:
                if batch == "main":
                    continue
                raise ValueError(f"Missing regenerated pools for {batch}/{group_id}")
            record = _generated_pool(document, batch, group_id, pool_id)
            pools[pool_id] = record
            valid_ids.update(record["chunk_ids"])
            loaded[pool_id] = str(record["pool_sha256"])
        if batch != "main" and "W" in required and document is not None:
            wide = document["pools"].get("W", {})
            expanded_hash = wide.get("expanded_pool_sha256") if isinstance(wide, Mapping) else None
            if expanded_hash is not None and expanded_hash != pools["E"]["pool_sha256"]:
                raise ValueError(f"Wide pool is not bound to repeat E for {batch}/{group_id}")
        snapshot["pools"] = pools
        provenance[group_id] = loaded
    return views, valid_ids, provenance


def _validate_regenerated_selectors(
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    snapshots: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    batch: str,
    plan: Mapping[str, Any],
) -> None:
    if batch == "main" or not plan["regenerated_pools"]:
        return
    regenerated = set(plan["regenerated_pools"])
    by_id = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    for config_id, records in collected.items():
        expected_pool = analysis._config_details(config_id)["pool_id"]
        if expected_pool not in regenerated:
            continue
        for group_id, record in records.items():
            if group_id not in selected_ids:
                continue
            if record.get("pool_id") != expected_pool:
                raise ValueError(f"Regenerated selector pool identity mismatch: {batch}/{group_id}/{config_id}")
            descriptor = record.get("cache_descriptor")
            if isinstance(descriptor, Mapping):
                if descriptor.get("execution_batch", batch) != batch or descriptor.get("pool_sha256", record["pool_sha256"]) != record["pool_sha256"]:
                    raise ValueError(f"Regenerated selector descriptor mismatch: {batch}/{group_id}/{config_id}")
            source_batch = record.get("source_execution_batch")
            source_config = record.get("reused_from")
            if source_batch in {None, batch} and source_config is None:
                continue
            pool_id = str(record["pool_id"])
            base_hash = by_id[group_id]["pools"]["B"]["pool_sha256"]
            inactive_wide_alias = (
                pool_id == "W" and isinstance(source_config, str) and source_config.startswith("B__")
                and record.get("pool_sha256") == base_hash
            )
            if not inactive_wide_alias:
                raise ValueError(f"Regenerated {pool_id} selector comes from another execution: {batch}/{group_id}/{config_id}")


def _reuse_allowed(base: str, plan: Mapping[str, Any]) -> tuple[bool, str]:
    if base == "E__R1__ARB" and plan["allow_r1_reuse"]:
        return True, "fixed_cached_R1"
    details = analysis._config_details(base)
    if details["selector"] in {"JINA", "RRF"} and details["pool_id"] in plan["fixed_pools"]:
        return True, f"deterministic_same_pool_{details['selector']}"
    return False, ""


def _reuse_record(record: Mapping[str, Any], source_batch: str, reason: str) -> dict[str, Any]:
    value = dict(record)
    value["_stability_reuse"] = {
        "source_execution_batch": source_batch,
        "source_config_id": record.get("config_id"),
        "reason": reason,
    }
    return value


def _actual_reuse(record: Mapping[str, Any], batch: str) -> dict[str, Any] | None:
    source_batch = record.get("source_execution_batch")
    source_config = record.get("reused_from")
    if source_batch is None and source_config is None:
        return None
    if source_batch is None:
        source_batch = batch
    return {
        "source_execution_batch": str(source_batch),
        "source_config_id": source_config or record.get("config_id"),
        "reason": "repeat_artifact_alias",
    }


def _resolve_side(
    base: str,
    batch: str,
    available: dict[str, dict[str, dict[str, Any]]],
    main: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selected_ids: set[str],
    plan: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    config_id = analysis._resolve_config_id(base, available, "strict")
    if config_id is not None:
        records = available[config_id]
        reuse = next((_actual_reuse(record, batch) for record in records.values() if _actual_reuse(record, batch) is not None), None)
        return config_id, reuse
    if batch == "main":
        return None, None
    allowed, reason = _reuse_allowed(base, plan)
    if not allowed:
        return None, None
    main_id = analysis._resolve_config_id(base, main, "strict")
    if main_id is None:
        return None, None
    reused_records = {
        group_id: _reuse_record(record, "main", reason)
        for group_id, record in main[main_id].items()
        if group_id in selected_ids
    }
    available[main_id] = reused_records
    return main_id, {"source_execution_batch": "main", "source_config_id": main_id, "reason": reason}


def _score_config(
    config_id: str,
    records: Mapping[str, Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    questions: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    valid_ids: set[str],
    selected_rows: int,
    expected_answerable_rows: int,
    expected_answerable_groups: int,
    selected_rows_without_qrels: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_snapshot = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    ordered = [records[str(snapshot["group_id"])] for snapshot in snapshots if str(snapshot["group_id"]) in records]
    question_ids = {str(qid) for record in ordered for qid in record["question_ids"]}
    unknown = question_ids - set(questions)
    if unknown:
        raise ValueError(f"Unknown benchmark question IDs for {config_id}: {sorted(unknown)[:5]}")
    subset_questions = {qid: query for qid, query in questions.items() if qid in question_ids}
    subset_qrels = {qid: values for qid, values in qrels.items() if qid in question_ids}
    summary = score_group_records(ordered, subset_questions, subset_qrels, valid_ids)
    metrics = summary["unique_query_macro"]
    expected_keys = {str(snapshot["query_group"]) for snapshot in snapshots if any(str(qid) in qrels for qid in snapshot["question_ids"])}
    actual_keys = {str(row["query_group"]) for row in summary["per_group"]}
    missing_groups = sorted(set(by_snapshot) - set(records))
    reused = [record for record in ordered if "_stability_reuse" in record]
    current = [record for record in ordered if "_stability_reuse" not in record]
    source_tokens = [_number(record.get("total_tokens")) for record in reused]
    fallback_stages = Counter(stage for record in ordered for stage in _fallback_stages(record))
    result = {
        "config_id": config_id,
        "aggregation": "unique_query_macro",
        "strict_or_backfill": "strict",
        "questions_total": len(subset_questions),
        "selected_questions_total": selected_rows,
        "run_rows": summary["run_rows"],
        "groups_total": len(ordered),
        "selected_groups_total": len(snapshots),
        "answerable_rows": summary["answerable_rows"],
        "answerable_unique_queries": summary["answerable_unique_queries"],
        "expected_answerable_rows": expected_answerable_rows,
        "expected_answerable_groups": expected_answerable_groups,
        "rows_without_qrels": summary["rows_without_qrels"],
        "selected_rows_without_qrels": selected_rows_without_qrels,
        "complete_group_coverage": not missing_groups and actual_keys == expected_keys,
        "missing_groups": missing_groups,
        "missing_answerable_groups": sorted(expected_keys - actual_keys),
        **{key: metrics.get(key) for key in analysis.ALL_METRICS},
        "repair_groups": sum(bool(record.get("repair_used")) for record in ordered),
        "fallback_groups": sum(_has_fallback(record) for record in ordered),
        "repair_rate": sum(bool(record.get("repair_used")) for record in ordered) / len(ordered) if ordered else None,
        "fallback_rate": sum(_has_fallback(record) for record in ordered) / len(ordered) if ordered else None,
        "fallback_stages": dict(fallback_stages),
        "error_groups": sum(bool(record.get("error") or record.get("errors")) for record in ordered),
        "prompt_tokens_total": _complete_sum([record.get("prompt_tokens") for record in current]),
        "completion_tokens_total": _complete_sum([record.get("completion_tokens") for record in current]),
        "llm_tokens_total": _complete_sum([record.get("total_tokens") for record in current]),
        "cached_reused_selector_tokens_total": _complete_sum(source_tokens) if source_tokens else 0,
        "selector_latency_p50_ms": _percentile([record.get("selector_latency_ms") for record in current], 50),
        "selector_latency_p95_ms": _percentile([record.get("selector_latency_ms") for record in current], 95),
        "jina_latency_p50_ms": _percentile([record.get("jina_latency_ms", record.get("selector_latency_ms")) for record in current], 50),
        "jina_latency_p95_ms": _percentile([record.get("jina_latency_ms", record.get("selector_latency_ms")) for record in current], 95),
        "reused_groups": len(reused),
        "reused_from_execution_batch": sorted({str(record["_stability_reuse"]["source_execution_batch"]) for record in reused}),
        "reuse_reasons": sorted({str(record["_stability_reuse"]["reason"]) for record in reused}),
    }
    per_group = {str(row["query_group"]): dict(row) for row in summary["per_group"]}
    return result, per_group


def _ranking_comparison(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]], group_ids: Sequence[str]) -> dict[str, Any]:
    if any(group_id not in left or group_id not in right for group_id in group_ids):
        return {
            "status": "missing",
            "n_groups": 0,
            "ranking_overlap@10": None,
            "mean_overlap@10": None,
            "mean_spearman": None,
            "mean_rank_correlation": None,
            "changed_selection_fraction": None,
            "changed_group_fraction": None,
            "changed_selection_group_fraction": None,
        }
    comparisons = [
        rank_overlap_and_correlation(left[group_id]["ranked_chunk_ids"], right[group_id]["ranked_chunk_ids"], k=10)
        for group_id in group_ids
    ]
    changed_groups = []
    changed_sets = []
    for group_id in group_ids:
        left_top = list(dict.fromkeys(left[group_id]["ranked_chunk_ids"]))[:10]
        right_top = list(dict.fromkeys(right[group_id]["ranked_chunk_ids"]))[:10]
        changed_groups.append(left_top != right_top)
        changed_sets.append(set(left_top) != set(right_top))
    overlap = _mean([row["overlap@10"] for row in comparisons])
    spearman = _mean([row["rank_correlation"] for row in comparisons])
    return {
        "status": "ok",
        "n_groups": len(group_ids),
        "ranking_overlap@10": overlap,
        "mean_overlap@10": overlap,
        "mean_spearman": spearman,
        "mean_rank_correlation": spearman,
        "changed_selection_fraction": 1.0 - overlap if overlap is not None else None,
        "changed_group_fraction": sum(changed_groups) / len(changed_groups) if changed_groups else None,
        "changed_selection_group_fraction": sum(changed_sets) / len(changed_sets) if changed_sets else None,
    }


def _missing_contrast(contrast_id: str, spec: Mapping[str, Any], left: str | None, right: str | None, expected_n: int, status: str, missing_groups: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "family": spec["family"],
        "contrast_id": contrast_id,
        "left_config": left,
        "right_config": right,
        "metric": "recall@10",
        "aggregation": "unique_query_macro",
        "strict_or_backfill": "strict",
        "fallback_policy": "all_groups",
        "same_pool": spec["same_pool"],
        "n_groups": 0,
        "expected_n_groups": expected_n,
        "delta": None,
        "delta_pp": None,
        "sign": None,
        "status": status,
        "missing_groups": list(missing_groups),
        "ranking_overlap@10": None,
        "mean_spearman": None,
        "changed_selection_fraction": None,
    }


def _contrast(
    contrast_id: str,
    left_id: str,
    right_id: str,
    available: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    answerable_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = analysis.CONTRAST_REGISTRY[contrast_id]
    expected_keys = [str(snapshot["query_group"]) for snapshot in answerable_groups]
    missing_left = sorted(set(expected_keys) - set(metrics.get(left_id, {})))
    missing_right = sorted(set(expected_keys) - set(metrics.get(right_id, {})))
    if missing_left or missing_right:
        return _missing_contrast(
            contrast_id, spec, left_id, right_id, len(expected_keys), "missing_groups",
            [f"left:{key}" for key in missing_left] + [f"right:{key}" for key in missing_right],
        )
    if spec["same_pool"]:
        analysis._assert_contrast_pools(contrast_id, left_id, right_id, available)
    left_values = [float(metrics[left_id][key]["recall@10"]) for key in expected_keys]
    right_values = [float(metrics[right_id][key]["recall@10"]) for key in expected_keys]
    delta = sum(left_values) / len(left_values) - sum(right_values) / len(right_values)
    sign = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
    group_ids = [str(snapshot["group_id"]) for snapshot in snapshots]
    ranking = _ranking_comparison(available[left_id], available[right_id], group_ids)
    return {
        "family": spec["family"],
        "contrast_id": contrast_id,
        "left_config": left_id,
        "right_config": right_id,
        "metric": "recall@10",
        "aggregation": "unique_query_macro",
        "strict_or_backfill": "strict",
        "fallback_policy": "all_groups",
        "same_pool": spec["same_pool"],
        "n_groups": len(expected_keys),
        "expected_n_groups": len(expected_keys),
        "delta": delta,
        "delta_pp": delta * 100,
        "sign": sign,
        "status": "ok",
        "missing_groups": [],
        "left_recall@10": summaries[left_id]["recall@10"],
        "right_recall@10": summaries[right_id]["recall@10"],
        "left_ndcg@10": summaries[left_id]["ndcg@10"],
        "right_ndcg@10": summaries[right_id]["ndcg@10"],
        **{key: value for key, value in ranking.items() if key not in {"status", "n_groups"}},
        "ranking_status": ranking["status"],
        "ranking_n_groups": ranking["n_groups"],
    }


def _spread(values: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"values": dict(zip(EXECUTION_BATCHES, values))}
    numbers = [_number(value) for value in values]
    result["available_executions"] = sum(value is not None for value in numbers)
    if any(value is None for value in numbers):
        result.update({"min": None, "max": None, "spread": None})
    else:
        numeric = [float(value) for value in numbers]
        result.update({"min": min(numeric), "max": max(numeric), "spread": max(numeric) - min(numeric)})
    return result


def _status_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_type": "status",
        "status": result["status"],
        "reason": result.get("reason"),
        "execution_batch": "",
        "best_repeat_selected": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(CSV_FIELDS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        output = {}
        for key in columns:
            value = row.get(key)
            if isinstance(value, (Mapping, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif value is None:
                value = ""
            output[key] = value
        writer.writerow(output)
    atomic_write_text(path, stream.getvalue())


def _write_outputs(experiment_dir: Path, result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(experiment_dir / "analysis" / "stability.csv", rows)
    atomic_write_json(experiment_dir / "analysis" / "stability.json", result)


def analyze_stability(
    config_path: str | os.PathLike[str],
    experiment_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    config_path, experiment_dir = Path(config_path), Path(experiment_dir)
    config = load_config(config_path)
    if config.get("protocol_version") != "coordinated_mechanism_v1":
        raise ValueError("Unexpected protocol_version")
    decision = _read_json(experiment_dir / "analysis" / "decision_pre_stability.json")
    target = _read_json(experiment_dir / "analysis" / "stability_target.json")
    no_target = target.get("status") == "not_applicable" or decision.get("branch") == "simple_hybrid_is_sufficient"
    manifest = _read_json(experiment_dir / "stability" / "execution_manifest.json", required=not no_target)
    if no_target:
        return _not_applicable(config_path, experiment_dir, decision, target, manifest)
    if target.get("status") not in {None, "selected", "ok"} or manifest.get("status") == "not_applicable":
        raise ValueError("Stability target and execution manifest disagree about applicability")
    if manifest.get("best_repeat_selected") is True or target.get("best_repeat_selected") is True:
        raise ValueError("Best-repeat selection is forbidden")
    selected_ids = _selected_group_ids(manifest, target, decision)
    plan = _target_plan(decision, target, manifest)
    snapshots = load_snapshots(experiment_dir)
    by_id = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown stability group IDs: {unknown[:5]}")
    selected = [by_id[group_id] for group_id in selected_ids]
    if len({str(snapshot["query_group"]) for snapshot in selected}) != len(selected):
        raise ValueError("Stability selection contains duplicate query groups")
    questions_path, qrels_path = analysis._benchmark_paths(config_path, config, experiment_dir)
    all_questions = load_questions(questions_path)
    all_qrels = load_qrels(qrels_path)
    selected_qids = {str(qid) for snapshot in selected for qid in snapshot["question_ids"]}
    unknown_qids = selected_qids - set(all_questions)
    if unknown_qids:
        raise ValueError(f"Unknown stability question IDs: {sorted(unknown_qids)[:5]}")
    questions = {qid: query for qid, query in all_questions.items() if qid in selected_qids}
    qrels = {qid: values for qid, values in all_qrels.items() if qid in selected_qids}
    answerable = [snapshot for snapshot in selected if any(str(qid) in qrels for qid in snapshot["question_ids"])]
    if not answerable:
        raise ValueError("Stability selection has no answerable groups")
    selected_rows = len(selected_qids)
    expected_answerable_rows = sum(
        sum(str(qid) in qrels for qid in snapshot["question_ids"]) for snapshot in selected
    )
    expected_answerable_groups = len(answerable)
    selected_rows_without_qrels = selected_rows - expected_answerable_rows
    batches: dict[str, dict[str, Any]] = {}
    valid_by_batch: dict[str, set[str]] = {}
    pool_provenance: dict[str, Any] = {}
    for batch in EXECUTION_BATCHES:
        views, valid_ids, provenance = _batch_snapshots(
            experiment_dir, snapshots, set(selected_ids), batch, plan,
        )
        collected = analysis.collect_selector_results(
            experiment_dir, snapshots=views, execution_batch=batch,
        )
        _validate_regenerated_selectors(collected, snapshots, set(selected_ids), batch, plan)
        batches[batch] = collected
        valid_by_batch[batch] = valid_ids
        pool_provenance[batch] = provenance
    runs: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    records_by_batch: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for execution_index, batch in enumerate(EXECUTION_BATCHES, 1):
        collected = batches[batch]
        available = {
            config_id: {group_id: record for group_id, record in records.items() if group_id in selected_ids}
            for config_id, records in collected.items()
        }
        resolved: dict[str, tuple[str | None, str | None]] = {}
        reuse: list[dict[str, Any]] = []
        for contrast_id in plan["contrast_ids"]:
            spec = analysis.CONTRAST_REGISTRY[contrast_id]
            left_id, left_reuse = _resolve_side(spec["left"], batch, available, batches["main"], set(selected_ids), plan)
            right_id, right_reuse = _resolve_side(spec["right"], batch, available, batches["main"], set(selected_ids), plan)
            resolved[contrast_id] = (left_id, right_id)
            for side, config_id, metadata in (("left", left_id, left_reuse), ("right", right_id, right_reuse)):
                if metadata is not None:
                    item = {"execution_batch": batch, "contrast_id": contrast_id, "side": side, "config_id": config_id, **metadata}
                    if item not in reuse:
                        reuse.append(item)
        needed = {config_id for pair in resolved.values() for config_id in pair if config_id is not None}
        available = {config_id: available[config_id] for config_id in sorted(needed)}
        if any(record.get("smoke_only") for records in available.values() for record in records.values()):
            raise ValueError(f"Smoke artifacts are forbidden in stability analysis: {batch}")
        summaries: dict[str, dict[str, Any]] = {}
        metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for config_id in sorted(available):
            summary, per_group = _score_config(
                config_id, available[config_id], selected, questions, qrels,
                valid_by_batch[batch] | {chunk_id for values in qrels.values() for chunk_id in values},
                selected_rows, expected_answerable_rows, expected_answerable_groups, selected_rows_without_qrels,
            )
            summaries[config_id] = summary
            metrics[config_id] = per_group
            csv_rows.append({"row_type": "config", "execution_batch": batch, "execution_index": execution_index, "status": "ok" if summary["complete_group_coverage"] else "missing", **summary})
        contrasts = []
        for contrast_id in plan["contrast_ids"]:
            spec = analysis.CONTRAST_REGISTRY[contrast_id]
            left_id, right_id = resolved[contrast_id]
            if left_id is None or right_id is None:
                row = _missing_contrast(contrast_id, spec, left_id, right_id, expected_answerable_groups, "missing_config")
            else:
                row = _contrast(contrast_id, left_id, right_id, available, metrics, summaries, selected, answerable)
            contrasts.append(row)
            csv_rows.append({"row_type": "contrast", "execution_batch": batch, "execution_index": execution_index, **row})
        run = {
            "execution_batch": batch,
            "execution_index": execution_index,
            "role": "predeclared_main" if batch == "main" else "independent_repeat",
            "group_ids": selected_ids,
            "selected_groups": len(selected),
            "answerable_groups": expected_answerable_groups,
            "answerable_rows": expected_answerable_rows,
            "rows_without_qrels": selected_rows_without_qrels,
            "metrics": summaries,
            "contrasts": contrasts,
            "signs": {row["contrast_id"]: row["sign"] for row in contrasts},
            "repair_groups": sum(summary["repair_groups"] for summary in summaries.values()),
            "fallback_groups": sum(summary["fallback_groups"] for summary in summaries.values()),
            "prompt_tokens_total": _complete_sum([summary["prompt_tokens_total"] for summary in summaries.values()]),
            "completion_tokens_total": _complete_sum([summary["completion_tokens_total"] for summary in summaries.values()]),
            "llm_tokens_total": _complete_sum([summary["llm_tokens_total"] for summary in summaries.values()]),
            "reuse": reuse,
            "regenerated_pool_sha256": pool_provenance[batch],
            "best_repeat_selected": False,
        }
        runs.append(run)
        records_by_batch[batch] = available
    result: dict[str, Any] = {
        "protocol_version": "coordinated_mechanism_v1",
        "status": "ok",
        "branch": plan["branch"],
        "target_kind": plan["kind"],
        "route": plan["route"],
        "target": dict(target),
        "target_contrasts": list(plan["contrast_ids"]),
        "selected_group_ids": selected_ids,
        "selected_groups": len(selected),
        "answerable_groups": expected_answerable_groups,
        "answerable_rows": expected_answerable_rows,
        "selected_rows": selected_rows,
        "rows_without_qrels": selected_rows_without_qrels,
        "runs": runs,
        "total_executions": 3,
        "required_executions": list(EXECUTION_BATCHES),
        "predeclared_execution": "main",
        "best_repeat_selected": False,
        "smoke_excluded": True,
        "reuse": [item for run in runs for item in run["reuse"]],
        "selection_rule": "all three predeclared executions; no best-repeat selection; STABLE requires three strictly positive Recall@10 deltas",
        "changed_selection_definition": "mean(1 - overlap@10); changed_group_fraction and changed_selection_group_fraction are reported separately",
    }
    stability: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    not_repeated = []
    for contrast_id in plan["contrast_ids"]:
        rows = [next(row for row in run["contrasts"] if row["contrast_id"] == contrast_id) for run in runs]
        deltas = [row["delta"] for row in rows]
        missing_batches = [batch for batch, row in zip(EXECUTION_BATCHES, rows) if _number(row["delta"]) is None]
        if missing_batches:
            missing[contrast_id] = missing_batches
        if all(_number(row["delta"]) is None for row in rows[1:]):
            not_repeated.append(contrast_id)
        stable = STABLE(result, contrast_id)
        stability[contrast_id] = {
            "stable": stable,
            "status": "missing" if missing_batches else "stable" if stable else "unstable",
            "deltas": dict(zip(EXECUTION_BATCHES, deltas)),
            "signs": {batch: row["sign"] for batch, row in zip(EXECUTION_BATCHES, rows)},
            "missing_executions": missing_batches,
            "positive_executions": sum(_number(value) is not None and float(value) > 0 for value in deltas),
            "delta_spread": _spread(deltas)["spread"],
        }
    result["stable_contrasts"] = {key: value["stable"] for key, value in stability.items()}
    result["contrast_stability"] = stability
    result["missing_contrast_executions"] = missing
    result["not_repeated"] = not_repeated
    if missing:
        result["status"] = "missing"
    spreads: dict[str, Any] = {}
    config_ids = sorted({config_id for run in runs for config_id in run["metrics"]})
    for config_id in config_ids:
        recall = [run["metrics"].get(config_id, {}).get("recall@10") for run in runs]
        ndcg = [run["metrics"].get(config_id, {}).get("ndcg@10") for run in runs]
        spreads[config_id] = {"recall@10": _spread(recall), "ndcg@10": _spread(ndcg)}
        csv_rows.append({
            "row_type": "spread", "config_id": config_id, "status": "ok",
            "recall@10_spread": spreads[config_id]["recall@10"]["spread"],
            "ndcg@10_spread": spreads[config_id]["ndcg@10"]["spread"],
        })
    result["spreads"] = spreads
    ranking_stability = []
    for config_id in config_ids:
        for left_batch, right_batch in (("main", "repeat_1"), ("main", "repeat_2"), ("repeat_1", "repeat_2")):
            comparison = _ranking_comparison(
                records_by_batch[left_batch].get(config_id, {}),
                records_by_batch[right_batch].get(config_id, {}),
                selected_ids,
            )
            row = {
                "config_id": config_id,
                "left_execution_batch": left_batch,
                "right_execution_batch": right_batch,
                **comparison,
            }
            ranking_stability.append(row)
            csv_rows.append({"row_type": "ranking_stability", "execution_batch": f"{left_batch}/{right_batch}", "config_id": config_id, **row})
    result["ranking_stability"] = ranking_stability
    result["metrics"] = {run["execution_batch"]: run["metrics"] for run in runs}
    _write_outputs(experiment_dir, result, csv_rows)
    return result


__all__ = ["EXECUTION_BATCHES", "analyze_stability"]
