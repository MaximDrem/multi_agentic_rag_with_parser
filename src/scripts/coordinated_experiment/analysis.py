import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.coordinated_experiment.legacy_reuse import load_snapshots
from scripts.coordinated_experiment.metrics import pool_quality_metrics, score_group_records
from scripts.coordinated_experiment.pools import multi_event_rrf
from scripts.coordinated_experiment.schema import atomic_write_json, atomic_write_text, load_config
from scripts.coordinated_experiment.statistics import apply_holm, classify_effect, make_bootstrap_indices, paired_contrast
from scripts.score_retrieval import load_qrels, load_questions

PRIMARY_METRIC = "recall@10"
SECONDARY_METRICS = ("recall@5", "hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10", "mrr@5", "mrr@10")
ALL_METRICS = (PRIMARY_METRIC, *SECONDARY_METRICS)
OUTPUT_SUFFIXES = ("__strict", "__common_backfill", "__legacy_backfill")
PRIMARY_FAMILIES = {
    "candidate_core": ("CG0", "CG1", "CG2"),
    "selector_core": ("SEL1", "SEL2", "SEL3"),
    "communication": ("MECH1", "MECH2", "MECH3", "MECH4", "MECH5", "SEL4", "SEL5"),
    "candidate_mechanism": ("CG4", "CG5", "CG6", "CG7"),
    "generic_arbiter": ("GEN1", "GEN2"),
}
STAGE_CONFIGS = {
    "candidate_core": (
        "B__RRF", "E__RRF", "W__RRF", "B__JINA", "E__JINA", "W__JINA",
        "E__R2P__ARB__LEGACY__strict", "E__R2P__ARB__LEGACY__common_backfill",
        "E__R2P__ARB__LEGACY__legacy_backfill",
    ),
    "selector_core": (
        "E__R0__ARB__strict", "E__R0__ARB__common_backfill",
        "E__LEGACY_R2P__ARB_REPLAY__strict", "E__LEGACY_R2P__ARB_REPLAY__common_backfill",
    ),
    "communication": (
        "E__R1__ARB__strict", "E__R1__ARB__common_backfill",
        "E__R2NP__ARB__strict", "E__R2NP__ARB__common_backfill",
        "E__R2P__ARB_CLEAN__strict", "E__R2P__ARB_CLEAN__common_backfill",
        "E__R2NP__WRRF", "E__R2P__WRRF",
    ),
    "candidate_mechanism": ("S__JINA", "NP__JINA"),
    "generic_arbiter": (
        "B__R0__ARB__strict", "B__R0__ARB__common_backfill",
        "W__R0__ARB__strict", "W__R0__ARB__common_backfill",
    ),
}


def _spec(family: str, left: str, right: str, same_pool: bool, primary: bool = True) -> dict[str, Any]:
    return {"family": family, "left": left, "right": right, "same_pool": same_pool, "primary": primary}


CONTRAST_REGISTRY = {
    "CG0": _spec("candidate_core", "W__JINA", "B__JINA", False),
    "CG1": _spec("candidate_core", "E__JINA", "B__JINA", False),
    "CG2": _spec("candidate_core", "E__JINA", "W__JINA", False),
    "CG3": _spec("candidate_core", "E__RRF", "W__RRF", False, False),
    "SCREEN1": _spec("screening", "E__R2P__ARB__LEGACY", "E__JINA", True, False),
    "SEL1": _spec("selector_core", "E__LEGACY_R2P__ARB_REPLAY", "E__JINA", True),
    "SEL2": _spec("selector_core", "E__LEGACY_R2P__ARB_REPLAY", "E__R0__ARB", True),
    "SEL3": _spec("selector_core", "E__R0__ARB", "E__JINA", True),
    "MECH1": _spec("communication", "E__R2NP__ARB", "E__R1__ARB", True),
    "MECH2": _spec("communication", "E__R2P__ARB_CLEAN", "E__R2NP__ARB", True),
    "MECH3": _spec("communication", "E__R2P__ARB_CLEAN", "E__R2P__WRRF", True),
    "MECH4": _spec("communication", "E__R2P__WRRF", "E__JINA", True),
    "MECH5": _spec("communication", "E__R2P__WRRF", "E__R2NP__WRRF", True),
    "SEL4": _spec("communication", "E__R2P__ARB_CLEAN", "E__JINA", True),
    "SEL5": _spec("communication", "E__R2NP__ARB", "E__JINA", True),
    "GEN1": _spec("generic_arbiter", "B__R0__ARB", "B__JINA", True),
    "GEN2": _spec("generic_arbiter", "W__R0__ARB", "W__JINA", True),
    "GEN3": _spec("generic_arbiter", "E__R0__ARB", "B__R0__ARB", False, False),
    "GEN4": _spec("generic_arbiter", "E__R0__ARB", "W__R0__ARB", False, False),
    "CG4": _spec("candidate_mechanism", "E__JINA", "S__JINA", False),
    "CG5": _spec("candidate_mechanism", "E__JINA", "NP__JINA", False),
    "CG6": _spec("candidate_mechanism", "S__JINA", "W__JINA", False),
    "CG7": _spec("candidate_mechanism", "NP__JINA", "W__JINA", False),
}

MAIN_FIELDS = [
    "config_id", "aggregation", "pool_id", "pool_size_mean", "selector", "report_state", "output_policy",
    "answerable_rows", "answerable_unique_queries", "fallback_groups", "repair_groups",
    "hit_rate@5", "recall@5", "ndcg@5", "mrr@5", "hit_rate@10", "recall@10", "ndcg@10", "mrr@10",
    "prompt_tokens_total", "completion_tokens_total", "llm_tokens_total",
    "selector_latency_p50_ms", "selector_latency_p95_ms", "groups_total", "run_rows", "rows_without_qrels",
    "subset", "execution_batches",
]
POOL_FIELDS = [
    "pool_id", "unique_candidates_mean", "raw_hits_mean", "candidate_recall_all",
    "candidate_recall_rrf@10", "candidate_recall_rrf@20", "candidate_recall_rrf@30",
    "new_ids_vs_base", "new_relevant_vs_base", "groups_expansion_active", "groups_total",
    "answerable_groups", "pool_size_min", "pool_size_p50", "pool_size_p95", "pool_size_max", "search_calls_mean",
]
CONTRAST_FIELDS = [
    "family", "contrast_id", "left_config", "right_config", "metric", "aggregation", "n_groups",
    "delta", "ci95_low", "ci95_high", "delta_pp", "ci95_low_pp", "ci95_high_pp", "p_randomization",
    "p_holm", "effect_interpretation", "practical_interpretation", "analysis_role", "confirmed",
    "strict_or_backfill", "fallback_policy", "bootstrap_seed", "bootstrap_samples", "randomization_samples",
    "observed_statistic", "same_pool", "sign_reversal_vs_strict", "status",
]
COST_FIELDS = [
    "config_id", "planner_tokens", "round1_tokens", "round2_tokens", "arbiter_tokens",
    "cached_generation_tokens", "new_selector_tokens", "prompt_tokens_total", "completion_tokens_total",
    "llm_tokens_total", "total_estimated_system_tokens", "estimated_system_tokens", "raw_search_hits",
    "search_calls", "jina_latency_p50_ms", "jina_latency_p95_ms", "selector_latency_p50_ms",
    "selector_latency_p95_ms", "measurement_notes",
]
FAILURE_FIELDS = [
    "row_type", "config_id", "contrast_id", "groups", "fallback_groups", "fallback_rate", "fallback_stages",
    "repair_groups", "repair_rate", "error_groups", "strict_output_size_mean", "strict_output_size_min",
    "strict_output_size_max", "strict_short_groups", "dropped_unknown_aliases", "dropped_duplicate_aliases",
    "strict_delta", "common_backfill_delta", "fallback_excluded_delta", "common_backfill_sign_reversal",
    "fallback_excluded_sign_reversal", "common_backfill_status", "fallback_excluded_status",
]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    columns = list(fields)
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
            if isinstance(value, (dict, list, tuple)):
                value = _json_text(value)
            elif isinstance(value, bool):
                value = str(value).lower()
            elif value is None:
                value = ""
            output[key] = value
        writer.writerow(output)
    atomic_write_text(path, stream.getvalue())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_text(path, "".join(_json_text(row) + "\n" for row in rows))


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _mean(values: Sequence[Any]) -> float | None:
    values = [float(value) for value in (_number(value) for value in values) if value is not None]
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[Any], q: float) -> float | None:
    values = [float(value) for value in (_number(value) for value in values) if value is not None]
    return float(np.percentile(values, q)) if values else None


def _complete_sum(values: Sequence[Any]) -> int | float | None:
    if not values:
        return None
    values = [_number(value) for value in values]
    if any(value is None for value in values):
        return None
    result = float(sum(values))
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


def _config_details(config_id: str) -> dict[str, str]:
    base = config_id
    policy = "strict"
    for suffix in OUTPUT_SUFFIXES:
        if base.endswith(suffix):
            policy = suffix[2:]
            base = base[:-len(suffix)]
            break
    parts = base.split("__")
    index = next((i for i, part in enumerate(parts) if part in {"RRF", "JINA", "WRRF"} or part.startswith("ARB")), None)
    return {
        "base_config_id": base,
        "pool_id": parts[0],
        "selector": parts[index] if index is not None else "UNKNOWN",
        "report_state": "__".join(parts[1:index]) if index is not None else "",
        "output_policy": policy,
    }


def _resolve_config_id(base: str, available: Mapping[str, Any], policy: str = "strict") -> str | None:
    if _config_details(base)["selector"] in {"RRF", "JINA", "WRRF"}:
        return base if base in available else None
    candidates = [f"{base}__{policy}"]
    if policy == "strict":
        candidates.append(base)
    return next((candidate for candidate in candidates if candidate in available), None)


def collect_selector_results(
    experiment_dir: str | os.PathLike[str],
    snapshots: Sequence[Mapping[str, Any]] | None = None,
    execution_batch: str | None = "main",
) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(experiment_dir)
    snapshots = list(snapshots) if snapshots is not None else load_snapshots(root)
    by_snapshot = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    collected: dict[str, dict[str, dict[str, Any]]] = {}
    origins: dict[tuple[str, str], str] = {}
    result_root = root / "selector_results"
    if not result_root.exists():
        return collected
    for path in sorted(result_root.glob("*/*/*.json")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read selector result {relative}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Selector result must be an object: {relative}")
        batch, group_id, filename = path.relative_to(result_root).parts
        if execution_batch is not None and batch != execution_batch:
            continue
        config_id = filename.removesuffix(".json")
        if record.get("config_id") != config_id or record.get("group_id") != group_id or record.get("execution_batch") != batch:
            raise ValueError(f"Result path identity mismatch: {relative}")
        if group_id not in by_snapshot:
            raise ValueError(f"Unknown snapshot group: {relative}")
        snapshot = by_snapshot[group_id]
        if record.get("query_group") != snapshot.get("query_group") or record.get("query") != snapshot.get("query"):
            raise ValueError(f"Query identity mismatch: {relative}")
        if not isinstance(record.get("question_ids"), list) or set(record["question_ids"]) != set(snapshot["question_ids"]):
            raise ValueError(f"question_ids mismatch: {relative}")
        if record.get("input_snapshot_sha256", snapshot.get("snapshot_sha256")) != snapshot.get("snapshot_sha256"):
            raise ValueError(f"Snapshot hash mismatch: {relative}")
        pool_id, pool_hash = record.get("pool_id"), record.get("pool_sha256")
        if not isinstance(pool_id, str) or not pool_id or not isinstance(pool_hash, str) or not pool_hash:
            raise ValueError(f"Missing pool identity: {relative}")
        pool = snapshot.get("pools", {}).get(pool_id)
        if pool is not None and pool.get("pool_sha256") != pool_hash:
            raise ValueError(f"Pool hash mismatch: {relative}")
        ranking = record.get("ranked_chunk_ids")
        if not isinstance(ranking, list) or not all(isinstance(item, str) for item in ranking):
            raise ValueError(f"Invalid ranked_chunk_ids: {relative}")
        if len(ranking) > 10 or len(set(ranking)) != len(ranking):
            raise ValueError(f"Ranking output contract violation: {relative}")
        if pool is not None:
            if set(ranking) - set(pool["chunk_ids"]):
                raise ValueError(f"Ranking contains IDs outside pool: {relative}")
            size = _number(record.get("pool_size"))
            if size is not None and size != len(pool["chunk_ids"]):
                raise ValueError(f"pool_size mismatch: {relative}")
        if _number(record.get("output_size")) not in (None, len(ranking)):
            raise ValueError(f"output_size mismatch: {relative}")
        key = (config_id, group_id)
        if key in origins:
            raise ValueError(f"Duplicate selector result for {config_id}/{group_id}: {origins[key]} and {relative}")
        value = dict(record)
        value["_source"] = str(relative)
        collected.setdefault(config_id, {})[group_id] = value
        origins[key] = str(relative)
    return collected


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _status_documents(experiment_dir: Path) -> list[Any]:
    paths = (
        "stage_status.json", "status.json", "run_status.json", "analysis/stage_status.json",
        "analysis/gates.json", "gates.json", "validation/run_integrity.json",
    )
    return [value for name in paths if (value := _read_json(experiment_dir / name)) is not None]


def _expected_from_document(value: Any) -> tuple[set[str], set[str], set[str]]:
    configs: set[str] = set()
    completed: set[str] = set()
    triggered: set[str] = set()
    if isinstance(value, Mapping):
        stage_name = value.get("stage", value.get("name"))
        stage_status = value.get("status")
        if stage_name in STAGE_CONFIGS and stage_status in {"complete", "completed", "done", "ok"}:
            completed.add(str(stage_name))
        for key, item in value.items():
            if key in {"expected_configs", "required_configs", "completed_configs"} and isinstance(item, list):
                configs.update(map(str, item))
            if key in {"completed_stages", "complete_stages"} and isinstance(item, list):
                completed.update(map(str, item))
            if key == "stages" and isinstance(item, Mapping):
                for stage, state in item.items():
                    status = state if isinstance(state, str) else state.get("status") if isinstance(state, Mapping) else None
                    if status in {"complete", "completed", "done", "ok"}:
                        completed.add(str(stage))
            if key in {"A", "B", "C"} and isinstance(item, Mapping):
                stage = {"A": "selector_core", "B": "communication", "C": "candidate_mechanism"}[key]
                if any(bool(item.get(field)) for field in ("run", "passed", "run_next_stage", "should_run", "run_selector_core", "run_communication", "run_S_NP")):
                    triggered.add(stage)
            if key == "gates" and isinstance(item, Mapping):
                for gate, stage in (("A", "selector_core"), ("B", "communication"), ("C", "candidate_mechanism")):
                    gate_value = item.get(gate, item.get(f"gate_{gate}"))
                    if gate_value is None:
                        alias = {"A": "A_screen_full_vs_E_jina", "B": "B_reports_package", "C": "C_expanded_vs_wide"}[gate]
                        gate_value = item.get(alias)
                    if isinstance(gate_value, Mapping) and any(bool(gate_value.get(field)) for field in ("run", "passed", "run_next_stage", "should_run", "run_selector_core", "run_communication", "run_S_NP")):
                        triggered.add(stage)
            subconfigs, substages, subtriggered = _expected_from_document(item)
            configs |= subconfigs
            completed |= substages
            triggered |= subtriggered
    elif isinstance(value, list):
        for item in value:
            subconfigs, substages, subtriggered = _expected_from_document(item)
            configs |= subconfigs
            completed |= substages
            triggered |= subtriggered
    return configs, completed, triggered


def expected_gated_configs(experiment_dir: str | os.PathLike[str], available: Sequence[str] = ()) -> tuple[set[str], dict[str, Any]]:
    documents = _status_documents(Path(experiment_dir))
    expected, completed, triggered = set(available), set(), set()
    for document in documents:
        values = _expected_from_document(document)
        expected |= values[0]
        completed |= values[1]
        triggered |= values[2]
    for stage in completed | triggered:
        expected.update(STAGE_CONFIGS.get(stage, ()))
    return expected, {"documents_found": len(documents), "completed_stages": sorted(completed), "gate_triggered_stages": sorted(triggered)}


def _benchmark_paths(config_path: Path, config: Mapping[str, Any], experiment_dir: Path) -> tuple[Path, Path]:
    benchmark = config.get("benchmark", {})
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    questions_name = str(benchmark.get("questions", "questions.jsonl"))
    qrels_name = str(benchmark.get("qrels", "qrels.jsonl"))
    if Path(questions_name).is_absolute() and Path(qrels_name).is_absolute():
        return Path(questions_name), Path(qrels_name)
    directories = []
    explicit = config.get("benchmark_dir", benchmark.get("dir", benchmark.get("path")))
    if explicit:
        path = Path(str(explicit))
        directories.append(path if path.is_absolute() else config_path.parent / path)
    for directory in (experiment_dir, *experiment_dir.parents, config_path.parent, *config_path.parent.parents):
        if directory not in directories:
            directories.append(directory)
    for directory in directories:
        questions, qrels = directory / questions_name, directory / qrels_name
        if questions.is_file() and qrels.is_file():
            return questions, qrels
    raise ValueError(f"Cannot locate benchmark questions/qrels for {experiment_dir.name}")


def _relevant(snapshot: Mapping[str, Any], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for qid in snapshot["question_ids"]:
        for chunk_id, relevance in qrels.get(str(qid), {}).items():
            result[str(chunk_id)] = max(int(relevance), result.get(str(chunk_id), 0))
    return result


def _score_configs(
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    snapshots: Sequence[Mapping[str, Any]],
    questions: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    config: Mapping[str, Any],
    require_complete: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    by_snapshot = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    all_groups = set(by_snapshot)
    valid_ids = {chunk_id for values in qrels.values() for chunk_id in values}
    for snapshot in snapshots:
        for pool in snapshot.get("pools", {}).values():
            if isinstance(pool, Mapping):
                valid_ids.update(pool["chunk_ids"])
    benchmark = config.get("benchmark", {})
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    expected = {
        "groups_total": int(benchmark.get("expected_groups", 556)),
        "run_rows": int(benchmark.get("expected_rows", 630)),
        "answerable_rows": int(benchmark.get("expected_answerable_rows", 617)),
        "answerable_unique_queries": int(benchmark.get("expected_answerable_groups", 544)),
    }
    summaries, metrics = {}, {}
    for config_id in sorted(collected):
        records_by_group = collected[config_id]
        selected = [by_snapshot[group_id] for group_id in sorted(records_by_group)]
        records = [records_by_group[str(snapshot["group_id"])] for snapshot in selected]
        qids = {str(qid) for snapshot in selected for qid in snapshot["question_ids"]}
        if qids - set(questions):
            raise ValueError(f"Unknown benchmark question IDs for {config_id}: {sorted(qids - set(questions))[:5]}")
        subset_questions = {qid: query for qid, query in questions.items() if qid in qids}
        subset_qrels = {qid: values for qid, values in qrels.items() if qid in qids}
        summary = score_group_records(records, subset_questions, subset_qrels, valid_ids)
        summary.update({
            "config_id": config_id,
            "groups_total": len(records_by_group),
            "subset": set(records_by_group) != all_groups,
            "benchmark_questions_total": len(questions),
            "complete_group_coverage": set(records_by_group) == all_groups,
            "missing_groups": sorted(all_groups - set(records_by_group)),
        })
        if summary["complete_group_coverage"]:
            actual = {key: summary[key] for key in expected}
            if actual != expected:
                raise ValueError(f"Full benchmark denominator mismatch for {config_id}: {actual} != {expected}")
        elif require_complete:
            raise ValueError(f"Config {config_id} is incomplete: {len(summary['missing_groups'])} groups missing")
        summaries[config_id] = summary
        metrics[config_id] = {str(row["query_group"]): dict(row) for row in summary["per_group"]}
    return summaries, metrics


def _assert_contrast_pools(
    contrast_id: str,
    left: str,
    right: str,
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for group_id in sorted(set(collected[left]) & set(collected[right])):
        a, b = collected[left][group_id], collected[right][group_id]
        if a.get("pool_id") != b.get("pool_id") or a.get("pool_sha256") != b.get("pool_sha256"):
            raise ValueError(
                f"Same-pool contrast {contrast_id} has pool mismatch in {group_id}: "
                f"{a.get('pool_id')}/{a.get('pool_sha256')} != {b.get('pool_id')}/{b.get('pool_sha256')}"
            )


def _external_pools(experiment_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for dirname in ("pools", "pool_records", "generated_pools"):
        root = experiment_dir / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            value = _read_json(path)
            if not isinstance(value, Mapping):
                continue
            pool = value.get("pool_record", value.get("pool", value))
            if not isinstance(pool, Mapping) or "chunk_ids" not in pool:
                continue
            group_id, pool_id = value.get("group_id"), value.get("pool_id")
            parts = path.relative_to(root).parts
            if group_id is None and len(parts) >= 2:
                group_id = parts[-2]
            if pool_id is None:
                pool_id = path.stem
            if group_id is None or pool_id is None:
                continue
            record = dict(pool)
            for field in ("pool_sha256", "retrieval_events", "raw_hit_counts", "rrf_ranking", "generation_usage", "search_calls"):
                if field in value and field not in record:
                    record[field] = value[field]
            key = (str(group_id), str(pool_id))
            if key in result and result[key] != record:
                raise ValueError(f"Duplicate or conflicting pool record: {key[0]}/{key[1]}")
            result[key] = record
    return result


def _pool_events(snapshot: Mapping[str, Any], pool_id: str, pool: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(pool.get("retrieval_events"), list):
        return [event for event in pool["retrieval_events"] if isinstance(event, Mapping)]
    events = [event for event in snapshot.get("retrieval_events", []) if isinstance(event, Mapping)]
    if pool_id == "B":
        return [event for event in events if event.get("stage", "initial") == "initial"]
    if pool_id == "E":
        return events
    if pool_id == "W":
        return [event for event in snapshot.get("wide_retrieval_events", []) if isinstance(event, Mapping)]
    return []


def _raw_hits(snapshot: Mapping[str, Any], pool_id: str, pool: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    counts = pool.get("raw_hit_counts")
    if not isinstance(counts, Mapping) and pool_id in {"B", "E"}:
        counts = snapshot.get("raw_hit_counts")
    raw = _number(counts.get("total")) if isinstance(counts, Mapping) else None
    if pool_id == "B" and isinstance(counts, Mapping):
        raw = _number(counts.get("initial", raw))
    if raw is None and events:
        raw = sum(len(event.get("hits", [])) for event in events)
    calls = _number(pool.get("search_calls"))
    if calls is None and isinstance(counts, Mapping):
        calls = _number(counts.get("search_calls"))
    if calls is None and events:
        calls = len(events)
    return raw, calls


def _rrf(pool: Mapping[str, Any], events: Sequence[Mapping[str, Any]], rrf_k: int) -> list[str] | None:
    for key in ("rrf_ranking", "ranked_chunk_ids"):
        if isinstance(pool.get(key), list):
            return [str(item["chunk_id"]) if isinstance(item, Mapping) else str(item) for item in pool[key]]
    if isinstance(pool.get("rrf_rows"), list):
        return [str(item["chunk_id"]) for item in pool["rrf_rows"]]
    return [row["chunk_id"] for row in multi_event_rrf(events, rrf_k, limit=30)] if events else None


def _difference(left: Sequence[str], right: Sequence[str], relevant: Mapping[str, int]) -> dict[str, Any]:
    right_set = set(right)
    ids = [chunk_id for chunk_id in dict.fromkeys(left) if chunk_id not in right_set]
    relevant_ids = [chunk_id for chunk_id in ids if chunk_id in relevant]
    return {"new_ids": ids, "new_relevant_ids": relevant_ids, "new_id_count": len(ids), "new_relevant_count": len(relevant_ids)}


def _build_pool_metrics(
    experiment_dir: Path,
    snapshots: Sequence[Mapping[str, Any]],
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    qrels: Mapping[str, Mapping[str, int]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    external = _external_pools(experiment_dir)
    retrieval = config.get("retrieval", {})
    rrf_k = int(retrieval.get("rrf_k", 60)) if isinstance(retrieval, Mapping) else 60
    result_records: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for records in collected.values():
        for group_id, record in records.items():
            result_records.setdefault((group_id, str(record["pool_id"])), []).append(record)
    rows: list[dict[str, Any]] = []
    pools: dict[tuple[str, str], Mapping[str, Any]] = {}
    relevants: dict[str, dict[str, int]] = {}
    for snapshot in snapshots:
        group_id = str(snapshot["group_id"])
        relevant = _relevant(snapshot, qrels)
        relevants[group_id] = relevant
        pool_ids = set(snapshot.get("pools", {}))
        pool_ids.update(pool for group, pool in result_records if group == group_id)
        pool_ids.update(pool for group, pool in external if group == group_id)
        for pool_id in sorted(pool_ids):
            pool = snapshot.get("pools", {}).get(pool_id) or external.get((group_id, pool_id))
            records = result_records.get((group_id, pool_id), [])
            if pool_id in snapshot.get("pools", {}) and snapshot["pools"][pool_id] is None and pool is None and not records:
                continue
            if not isinstance(pool, Mapping):
                size = next((value for value in (_number(record.get("pool_size")) for record in records) if value is not None), None)
                rows.append({
                    "group_id": group_id, "query_group": snapshot["query_group"], "question_ids": list(snapshot["question_ids"]),
                    "pool_id": pool_id, "pool_sha256": records[0].get("pool_sha256") if records else None,
                    "answerable": bool(relevant), "expansion_active": bool(snapshot.get("expansion_active")),
                    "unique_candidates": size, "pool_size": size, "raw_hits": None, "search_calls": None,
                    "candidate_recall_all": None,
                    **{f"candidate_recall_rrf@{cutoff}": None for cutoff in (10, 20, 30)},
                    "measurement_status": "pool_ids_unavailable",
                })
                continue
            pools[(group_id, pool_id)] = pool
            ids = list(pool["chunk_ids"])
            events = _pool_events(snapshot, pool_id, pool)
            raw, calls = _raw_hits(snapshot, pool_id, pool, events)
            ranking = _rrf(pool, events, rrf_k)
            if relevant:
                quality = pool_quality_metrics(ids, relevant, ranking)
            else:
                quality = {"unique_candidates": len(set(ids)), "candidate_recall_all": None,
                           **{f"candidate_recall_rrf@{cutoff}": None for cutoff in (10, 20, 30)}}
            rows.append({
                "group_id": group_id, "query_group": snapshot["query_group"], "question_ids": list(snapshot["question_ids"]),
                "pool_id": pool_id, "pool_sha256": pool.get("pool_sha256"), "answerable": bool(relevant),
                "expansion_active": bool(snapshot.get("expansion_active")), "pool_size": len(ids), **quality,
                "raw_hits": raw, "search_calls": calls,
                "raw_hits_by_channel": pool.get("raw_hit_counts", snapshot.get("raw_hit_counts") if pool_id in {"B", "E"} else None),
                "measurement_status": "ok",
            })
    lookup = {(row["group_id"], row["pool_id"]): row for row in rows}
    for row in rows:
        group_id, pool_id = row["group_id"], row["pool_id"]
        differences = {}
        for left, right, label in (("E", "B", "E-B"), ("E", "W", "E-W"), ("W", "E", "W-E")):
            if pool_id == left:
                a, b = pools.get((group_id, left)), pools.get((group_id, right))
                differences[label] = _difference(a["chunk_ids"], b["chunk_ids"], relevants[group_id]) if a and b else None
        row["differences"] = differences
        base_pool = pools.get((group_id, "B"))
        current_pool = pools.get((group_id, pool_id))
        base_difference = _difference(current_pool["chunk_ids"], base_pool["chunk_ids"], relevants[group_id]) if current_pool and base_pool else None
        row["base_difference"] = base_difference
        row["new_ids_vs_base"] = base_difference["new_id_count"] if base_difference else None
        row["new_relevant_vs_base"] = base_difference["new_relevant_count"] if base_difference else None
        for label, difference in differences.items():
            for field in ("new_ids", "new_relevant_ids", "new_id_count", "new_relevant_count"):
                row[f"{label}_{field}"] = difference.get(field) if difference else None
        for other in ("B", "W"):
            other_row = lookup.get((group_id, other))
            row[f"pool_size_delta_vs_{other}"] = (
                row["pool_size"] - other_row["pool_size"]
                if other_row and row.get("pool_size") is not None and other_row.get("pool_size") is not None else None
            )
    summaries = []
    for pool_id in sorted({row["pool_id"] for row in rows}):
        group_rows = [row for row in rows if row["pool_id"] == pool_id]
        answerable = [row for row in group_rows if row["answerable"]]
        differences = [row["base_difference"] for row in group_rows if row.get("base_difference")]
        sizes = [row["pool_size"] for row in group_rows if row.get("pool_size") is not None]
        summaries.append({
            "pool_id": pool_id,
            "unique_candidates_mean": _mean([row.get("unique_candidates") for row in group_rows]),
            "raw_hits_mean": _mean([row.get("raw_hits") for row in group_rows]),
            "candidate_recall_all": _mean([row.get("candidate_recall_all") for row in answerable]),
            **{f"candidate_recall_rrf@{cutoff}": _mean([row.get(f"candidate_recall_rrf@{cutoff}") for row in answerable]) for cutoff in (10, 20, 30)},
            "new_ids_vs_base": _mean([value["new_id_count"] for value in differences]),
            "new_relevant_vs_base": _mean([value["new_relevant_count"] for value in differences]),
            "groups_expansion_active": sum(bool(row["expansion_active"]) for row in group_rows),
            "groups_total": len(group_rows), "answerable_groups": len(answerable),
            "pool_size_min": min(sizes, default=None), "pool_size_p50": _percentile(sizes, 50),
            "pool_size_p95": _percentile(sizes, 95), "pool_size_max": max(sizes, default=None),
            "search_calls_mean": _mean([row.get("search_calls") for row in group_rows]),
        })
    rows.sort(key=lambda row: (row["group_id"], row["pool_id"]))
    return summaries, rows


def _generation_usage(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("generation_usage", "cached_generation_usage", "worker_generation_usage"):
        if isinstance(record.get(key), Mapping):
            return record[key]
    return {}


def _stage(usage: Mapping[str, Any], *names: str) -> Any:
    stages = usage.get("stage_tokens", {})
    stages = stages if isinstance(stages, Mapping) else {}
    for name in names:
        if name in stages:
            return _number(stages[name])
        if name in usage:
            return _number(usage[name])
    return None


def _route_cost(config_id: str, record: Mapping[str, Any], snapshot: Mapping[str, Any], pool: Mapping[str, Any] | None) -> dict[str, Any]:
    details = _config_details(config_id)
    base, pool_id, selector = details["base_config_id"], details["pool_id"], details["selector"]
    report = details["report_state"]
    old_usage = snapshot.get("generation_usage", {})
    old_usage = old_usage if isinstance(old_usage, Mapping) else {}
    usage = _generation_usage(record)
    cached_arb = "__ARB__LEGACY" in base
    new_arb = selector.startswith("ARB") and not cached_arb
    needs_planner = pool_id == "E"
    needs_round1 = pool_id in {"E", "NP"}
    has_r2 = "R2" in report
    needs_round2 = has_r2 and bool(snapshot.get("expansion_active"))
    planner = _stage(usage, "planner") if needs_planner else 0
    if needs_planner and planner is None:
        planner = _stage(old_usage, "planner")
    round1 = _stage(usage, "round_1", "round1") if needs_round1 else 0
    if pool_id == "E" and round1 is None:
        round1 = _stage(old_usage, "round_1", "round1")
    if has_r2 and not needs_round2:
        round2 = 0
    elif needs_round2:
        round2 = _stage(usage, "round_2", "round2")
        if round2 is None and (cached_arb or "LEGACY" in base):
            round2 = _stage(old_usage, "round_2", "round2")
    else:
        round2 = 0
    arbiter = _stage(old_usage, "arbiter") if cached_arb else 0
    new_tokens = _number(record.get("total_tokens")) if new_arb else 0
    prompt = _number(record.get("prompt_tokens")) if new_arb else 0
    completion = _number(record.get("completion_tokens")) if new_arb else 0
    other = _stage(usage, "single_llm", "candidate_generation", "query_generation") if pool_id == "S" else 0
    components = [planner, round1, round2, arbiter, other, new_tokens]
    missing = [name for name, value, needed in (
        ("planner", planner, needs_planner), ("round1", round1, needs_round1),
        ("round2", round2, needs_round2), ("cached_arbiter", arbiter, cached_arb),
        ("new_selector", new_tokens, new_arb), ("single_llm", other, pool_id == "S"),
    ) if needed and value is None]
    return {
        "config_id": config_id, "group_id": snapshot["group_id"], "query_group": snapshot["query_group"],
        "planner_tokens": planner, "round1_tokens": round1, "round2_tokens": round2, "arbiter_tokens": arbiter,
        "cached_generation_tokens": _complete_sum([planner, round1, round2, arbiter, other]),
        "new_selector_tokens": new_tokens, "other_generation_tokens": other,
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_estimated_system_tokens": _complete_sum(components),
        "raw_search_hits": pool.get("raw_hits") if pool else None,
        "search_calls": pool.get("search_calls") if pool else None,
        "selector_latency_ms": _number(record.get("selector_latency_ms")),
        "jina_latency_ms": _number(record.get("jina_latency_ms")) if record.get("jina_latency_ms") is not None else (_number(record.get("selector_latency_ms")) if selector == "JINA" else None),
        "repair_used": bool(record.get("repair_used")), "fallback_stages": _fallback_stages(record),
        "measurement_status": "ok" if not missing else "unavailable:" + ",".join(missing),
    }


def _build_costs(
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    snapshots: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_snapshot = {str(snapshot["group_id"]): snapshot for snapshot in snapshots}
    pools = {(row["group_id"], row["pool_id"]): row for row in pool_rows}
    per_group, summaries = [], []
    for config_id in sorted(collected):
        rows = []
        for group_id in sorted(collected[config_id]):
            record = collected[config_id][group_id]
            row = _route_cost(config_id, record, by_snapshot[group_id], pools.get((group_id, record["pool_id"])))
            rows.append(row)
            per_group.append(row)
        records = list(collected[config_id].values())
        estimated = _complete_sum([row["total_estimated_system_tokens"] for row in rows])
        summaries.append({
            "config_id": config_id,
            **{field: _complete_sum([row[field] for row in rows]) for field in ("planner_tokens", "round1_tokens", "round2_tokens", "arbiter_tokens", "cached_generation_tokens", "new_selector_tokens")},
            "prompt_tokens_total": _complete_sum([row["prompt_tokens"] for row in rows]),
            "completion_tokens_total": _complete_sum([row["completion_tokens"] for row in rows]),
            "llm_tokens_total": _complete_sum([row["new_selector_tokens"] for row in rows]),
            "total_estimated_system_tokens": estimated, "estimated_system_tokens": estimated,
            "raw_search_hits": _complete_sum([row["raw_search_hits"] for row in rows]),
            "search_calls": _complete_sum([row["search_calls"] for row in rows]),
            "jina_latency_p50_ms": _percentile([row["jina_latency_ms"] for row in rows], 50),
            "jina_latency_p95_ms": _percentile([row["jina_latency_ms"] for row in rows], 95),
            "selector_latency_p50_ms": _percentile([record.get("selector_latency_ms") for record in records], 50),
            "selector_latency_p95_ms": _percentile([record.get("selector_latency_ms") for record in records], 95),
            "measurement_notes": {"unavailable_groups": sum(row["measurement_status"] != "ok" for row in rows), "missing_usage_is_null": True, "latency_scope": "replay selector only"},
        })
    return summaries, per_group


def _effect_text(effect: str) -> str:
    return {
        "POS": "положительный сигнал: дельта не меньше 1 п.п., нижняя граница CI выше нуля",
        "BELOW_MARGIN": "дополнительный выигрыш меньше 1 п.п. по верхней границе CI; это не доказательство эквивалентности",
        "HARM": "верхняя граница CI ниже нуля: деградация левой стороны",
        "UNCERTAIN": "интервал не позволяет классифицировать практический эффект",
    }[effect]


def _contrast_row(
    contrast_id: str,
    spec: Mapping[str, Any],
    left: str,
    right: str,
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
    metric: str,
    policy: str,
    fallback_policy: str,
    indices: np.ndarray,
    seed: int,
    bootstrap: int,
    randomization: int,
) -> dict[str, Any]:
    primary = metric == PRIMARY_METRIC
    result = paired_contrast(
        left_rows, right_rows, metric=metric, group_keys=keys, bootstrap_indices=indices,
        n_randomization=randomization if primary else 1, seed=seed,
    )
    if not primary:
        result["p_randomization"] = None
    effect = classify_effect(result)
    is_primary = bool(spec["primary"]) and primary and policy == "strict" and fallback_policy == "all_groups"
    if is_primary:
        analysis_role = "primary"
    elif policy == "strict" and fallback_policy == "all_groups" and not primary:
        analysis_role = "secondary"
    else:
        analysis_role = "exploratory"
    return {
        "family": spec["family"], "contrast_id": contrast_id, "left_config": left, "right_config": right,
        "metric": metric, "aggregation": "unique_query_macro", **result,
        "delta_pp": result["delta"] * 100, "ci95_low_pp": result["ci95_low"] * 100, "ci95_high_pp": result["ci95_high"] * 100,
        "p_holm": None, "effect_interpretation": effect, "practical_interpretation": _effect_text(effect),
        "analysis_role": analysis_role,
        "confirmed": False, "strict_or_backfill": policy, "fallback_policy": fallback_policy,
        "bootstrap_seed": seed, "bootstrap_samples": bootstrap, "randomization_samples": randomization if primary else None,
        "same_pool": spec["same_pool"], "sign_reversal_vs_strict": None, "status": "ok",
    }


def _empty_contrast(
    contrast_id: str, spec: Mapping[str, Any], left: str, right: str, policy: str,
    fallback_policy: str, seed: int, bootstrap: int, randomization: int,
) -> dict[str, Any]:
    return {
        "family": spec["family"], "contrast_id": contrast_id, "left_config": left, "right_config": right,
        "metric": PRIMARY_METRIC, "aggregation": "unique_query_macro", "n_groups": 0,
        "delta": None, "ci95_low": None, "ci95_high": None, "delta_pp": None, "ci95_low_pp": None,
        "ci95_high_pp": None, "p_randomization": None, "p_holm": None, "effect_interpretation": "UNCERTAIN",
        "practical_interpretation": "нет paired groups после fallback exclusion", "analysis_role": "exploratory",
        "confirmed": False, "strict_or_backfill": policy, "fallback_policy": fallback_policy,
        "bootstrap_seed": seed, "bootstrap_samples": bootstrap, "randomization_samples": randomization,
        "observed_statistic": None, "same_pool": spec["same_pool"], "sign_reversal_vs_strict": None,
        "status": "no_eligible_groups",
    }


def _build_contrasts(
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    snapshots: Sequence[Mapping[str, Any]],
    bootstrap: int,
    randomization: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_for_query = {str(snapshot["query_group"]): str(snapshot["group_id"]) for snapshot in snapshots}
    rows, primary_rows, sensitivity = [], [], []
    cache: dict[tuple[str, ...], np.ndarray] = {}
    digests: dict[str, str] = {}

    def indices(keys: Sequence[str]) -> np.ndarray:
        key = tuple(keys)
        if key not in cache:
            cache[key] = make_bootstrap_indices(key, n_samples=bootstrap, seed=seed)
            group_digest = hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()
            digests[group_digest] = hashlib.sha256(np.asarray(cache[key], dtype=np.int64).tobytes()).hexdigest()
        return cache[key]

    for contrast_id, spec in CONTRAST_REGISTRY.items():
        left = _resolve_config_id(spec["left"], collected)
        right = _resolve_config_id(spec["right"], collected)
        if left is None or right is None:
            continue
        if spec["same_pool"]:
            _assert_contrast_pools(contrast_id, left, right, collected)
        keys = sorted(set(metrics[left]) & set(metrics[right]))
        variants = [("strict", left, right, "all_groups", keys)]
        left_common = _resolve_config_id(spec["left"], collected, "common_backfill")
        right_common = _resolve_config_id(spec["right"], collected, "common_backfill")
        common_status = "not_available"
        if left_common and right_common:
            if spec["same_pool"]:
                _assert_contrast_pools(contrast_id, left_common, right_common, collected)
            variants.append(("common_backfill", left_common, right_common, "all_groups", sorted(set(metrics[left_common]) & set(metrics[right_common]))))
            common_status = "ok"
        eligible = []
        for key in keys:
            group_id = group_for_query[key]
            if not _has_fallback(collected[left][group_id]) and not _has_fallback(collected[right][group_id]):
                eligible.append(key)
        variants.append(("strict", left, right, "exclude_fallback_pairs", eligible))
        deltas = {"strict": None, "common_backfill": None, "exclude_fallback_pairs": None}
        for policy, left_id, right_id, fallback_policy, paired_keys in variants:
            if not paired_keys:
                if fallback_policy == "exclude_fallback_pairs":
                    rows.append(_empty_contrast(contrast_id, spec, left_id, right_id, policy, fallback_policy, seed, bootstrap, randomization))
                continue
            shared = indices(paired_keys)
            metric_names = ALL_METRICS if policy == "strict" and fallback_policy == "all_groups" else (PRIMARY_METRIC,)
            for metric in metric_names:
                row = _contrast_row(
                    contrast_id, spec, left_id, right_id, metrics[left_id], metrics[right_id], paired_keys,
                    metric, policy, fallback_policy, shared, seed, bootstrap, randomization,
                )
                if metric == PRIMARY_METRIC:
                    key = "strict" if fallback_policy == "all_groups" and policy == "strict" else ("exclude_fallback_pairs" if fallback_policy == "exclude_fallback_pairs" else "common_backfill")
                    deltas[key] = row["delta"]
                if row["analysis_role"] == "primary":
                    primary_rows.append(row)
                else:
                    rows.append(row)
        sensitivity.append({
            "row_type": "contrast_sensitivity", "config_id": "", "contrast_id": contrast_id,
            "strict_delta": deltas["strict"], "common_backfill_delta": deltas["common_backfill"],
            "fallback_excluded_delta": deltas["exclude_fallback_pairs"],
            "common_backfill_sign_reversal": deltas["strict"] is not None and deltas["common_backfill"] is not None and deltas["strict"] * deltas["common_backfill"] < 0,
            "fallback_excluded_sign_reversal": deltas["strict"] is not None and deltas["exclude_fallback_pairs"] is not None and deltas["strict"] * deltas["exclude_fallback_pairs"] < 0,
            "common_backfill_status": common_status,
            "fallback_excluded_status": "ok" if eligible else "no_eligible_groups",
        })
    primary_rows = apply_holm(primary_rows, PRIMARY_FAMILIES)
    for row in primary_rows:
        row["confirmed"] = row["effect_interpretation"] == "POS" and row["p_holm"] is not None and row["p_holm"] < 0.05
    primary_by_id = {row["contrast_id"]: row for row in primary_rows}
    for row in rows:
        if row["metric"] == PRIMARY_METRIC and row["contrast_id"] in primary_by_id and row.get("delta") is not None:
            delta = primary_by_id[row["contrast_id"]]["delta"]
            row["sign_reversal_vs_strict"] = delta is not None and delta * row["delta"] < 0
    rows = primary_rows + rows
    order = {key: index for index, key in enumerate(CONTRAST_REGISTRY)}
    metric_order = {key: index for index, key in enumerate(ALL_METRICS)}
    variant_order = {("strict", "all_groups"): 0, ("common_backfill", "all_groups"): 1, ("strict", "exclude_fallback_pairs"): 2}
    rows.sort(key=lambda row: (order.get(row["contrast_id"], 999), variant_order.get((row["strict_or_backfill"], row["fallback_policy"]), 9), metric_order.get(row["metric"], 99)))
    metadata = {
        "bootstrap_indices_by_group_set_sha256": digests,
        "shared_group_sets": [list(keys) for keys in cache],
        "sensitivity": sensitivity,
    }
    return rows, metadata


def _pool_contrasts(pool_rows: Sequence[Mapping[str, Any]], bootstrap: int, randomization: int, seed: int) -> list[dict[str, Any]]:
    lookup = {(row["group_id"], row["pool_id"]): row for row in pool_rows}
    output = []
    for left, right in (("E", "B"), ("E", "W"), ("W", "E")):
        keys = sorted(
            group for group, pool in lookup if pool == left and lookup[(group, pool)].get("answerable")
            and lookup[(group, pool)].get("candidate_recall_all") is not None
            and lookup.get((group, right), {}).get("candidate_recall_all") is not None
        )
        if not keys:
            continue
        a = {key: {"candidate_recall_all": lookup[(key, left)]["candidate_recall_all"]} for key in keys}
        b = {key: {"candidate_recall_all": lookup[(key, right)]["candidate_recall_all"]} for key in keys}
        stats = paired_contrast(a, b, metric="candidate_recall_all", group_keys=keys,
                                bootstrap_indices=make_bootstrap_indices(keys, bootstrap, seed),
                                n_randomization=randomization, seed=seed)
        output.append({
            "family": "candidate_pool", "contrast_id": f"POOL_{left}_MINUS_{right}",
            "left_config": left, "right_config": right, "metric": "candidate_recall_all",
            "aggregation": "unique_query_macro", **stats, "delta_pp": stats["delta"] * 100,
            "ci95_low_pp": stats["ci95_low"] * 100, "ci95_high_pp": stats["ci95_high"] * 100,
            "p_holm": None, "effect_interpretation": classify_effect(stats),
            "practical_interpretation": "Candidate recall — верхняя граница возможной пользы, не доказательство улучшения итогового ranking.",
            "analysis_role": "exploratory", "confirmed": False, "strict_or_backfill": "pool",
            "fallback_policy": "all_groups", "bootstrap_seed": seed, "bootstrap_samples": bootstrap,
            "randomization_samples": randomization, "same_pool": False, "sign_reversal_vs_strict": None, "status": "ok",
        })
    return output


def _main_results(
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cost_lookup = {row["config_id"]: row for row in costs}
    output = []
    for config_id in sorted(collected):
        records = list(collected[config_id].values())
        summary = summaries[config_id]
        details = _config_details(config_id)
        cost = cost_lookup[config_id]
        common = {
            "config_id": config_id, "pool_id": details["pool_id"],
            "pool_size_mean": _mean([record.get("pool_size") for record in records]),
            "selector": details["selector"], "report_state": details["report_state"], "output_policy": details["output_policy"],
            "answerable_rows": summary["answerable_rows"], "answerable_unique_queries": summary["answerable_unique_queries"],
            "fallback_groups": sum(_has_fallback(record) for record in records),
            "repair_groups": sum(bool(record.get("repair_used")) for record in records),
            "prompt_tokens_total": cost["prompt_tokens_total"], "completion_tokens_total": cost["completion_tokens_total"],
            "llm_tokens_total": cost["llm_tokens_total"],
            "selector_latency_p50_ms": cost["selector_latency_p50_ms"], "selector_latency_p95_ms": cost["selector_latency_p95_ms"],
            "groups_total": summary["groups_total"], "run_rows": summary["run_rows"],
            "rows_without_qrels": summary["rows_without_qrels"], "subset": summary["subset"],
            "execution_batches": sorted({record["execution_batch"] for record in records}),
        }
        for aggregation in ("row_macro", "unique_query_macro"):
            output.append({**common, "aggregation": aggregation, **summary[aggregation]})
    return output


def _quality_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for config_id in sorted(summaries):
        for row in summaries[config_id]["per_group"]:
            output.append({"config_id": config_id, "aggregation": "unique_query_macro", **_config_details(config_id), **row})
    return output


def _failure_rows(collected: Mapping[str, Mapping[str, Mapping[str, Any]]], sensitivity: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(sensitivity)
    for config_id in sorted(collected):
        records = list(collected[config_id].values())
        sizes = [len(record["ranked_chunk_ids"]) for record in records]
        stages = Counter(stage for record in records for stage in _fallback_stages(record))
        rows.append({
            "row_type": "config_failures", "config_id": config_id, "groups": len(records),
            "fallback_groups": sum(_has_fallback(record) for record in records),
            "fallback_rate": sum(_has_fallback(record) for record in records) / len(records),
            "fallback_stages": dict(stages), "repair_groups": sum(bool(record.get("repair_used")) for record in records),
            "repair_rate": sum(bool(record.get("repair_used")) for record in records) / len(records),
            "error_groups": sum(bool(record.get("error") or record.get("errors")) for record in records),
            "strict_output_size_mean": _mean(sizes), "strict_output_size_min": min(sizes, default=None),
            "strict_output_size_max": max(sizes, default=None), "strict_short_groups": sum(size < 10 for size in sizes),
            "dropped_unknown_aliases": sum(len(record.get("dropped_unknown_aliases", [])) for record in records),
            "dropped_duplicate_aliases": sum(len(record.get("dropped_duplicate_aliases", [])) for record in records),
        })
    return rows


def analyze_experiment(
    config_path: str | os.PathLike[str],
    experiment_dir: str | os.PathLike[str],
    bootstrap: int = 10000,
    randomization: int = 10000,
    seed: int = 20261127,
    require_complete: bool = False,
) -> dict[str, Any]:
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap <= 0:
        raise ValueError("bootstrap must be a positive integer")
    if isinstance(randomization, bool) or not isinstance(randomization, int) or randomization <= 0:
        raise ValueError("randomization must be a positive integer")
    config_path, experiment_dir = Path(config_path), Path(experiment_dir)
    config = load_config(config_path)
    if config.get("protocol_version") != "coordinated_mechanism_v1":
        raise ValueError("Unexpected protocol_version")
    snapshots = load_snapshots(experiment_dir)
    collected = collect_selector_results(experiment_dir, snapshots)
    expected_configs, stage_status = expected_gated_configs(experiment_dir, list(collected))
    missing_configs = sorted(expected_configs - set(collected))
    if require_complete and missing_configs:
        raise ValueError(f"Expected gated configs are missing: {missing_configs}")
    questions_path, qrels_path = _benchmark_paths(config_path, config, experiment_dir)
    questions, qrels = load_questions(questions_path), load_qrels(qrels_path)
    summaries, metrics = _score_configs(collected, snapshots, questions, qrels, config, require_complete)
    contrasts, contrast_meta = _build_contrasts(collected, metrics, snapshots, bootstrap, randomization, seed)
    pool_summaries, pool_rows = _build_pool_metrics(experiment_dir, snapshots, collected, qrels, config)
    contrasts.extend(_pool_contrasts(pool_rows, bootstrap, randomization, seed))
    costs, cost_rows = _build_costs(collected, snapshots, pool_rows)
    main = _main_results(collected, summaries, costs)
    quality = _quality_rows(summaries)
    failures = _failure_rows(collected, contrast_meta["sensitivity"])
    incomplete_groups = {config_id: summary["missing_groups"] for config_id, summary in summaries.items() if summary["missing_groups"]}
    complete = not missing_configs and not incomplete_groups
    if require_complete and not complete:
        raise ValueError(f"Experiment is incomplete: {incomplete_groups}")
    root = experiment_dir
    _write_csv(root / "configs" / "main_results.csv", main, MAIN_FIELDS)
    _write_jsonl(root / "per_group" / "quality_metrics.jsonl", quality)
    _write_csv(root / "configs" / "pool_metrics.csv", pool_summaries, POOL_FIELDS)
    _write_jsonl(root / "per_group" / "pool_metrics.jsonl", pool_rows)
    _write_csv(root / "configs" / "costs.csv", costs, COST_FIELDS)
    _write_jsonl(root / "per_group" / "costs.jsonl", cost_rows)
    _write_csv(root / "analysis" / "paired_contrasts.csv", contrasts, CONTRAST_FIELDS)
    contrast_document = {
        "protocol_version": "coordinated_mechanism_v1", "primary_metric": PRIMARY_METRIC,
        "aggregation": "unique_query_macro", "seed": seed, "bootstrap_samples": bootstrap,
        "randomization_samples": randomization, "primary_families": PRIMARY_FAMILIES,
        "bootstrap_indices_by_group_set_sha256": contrast_meta["bootstrap_indices_by_group_set_sha256"],
        "contrasts": contrasts,
    }
    atomic_write_json(root / "analysis" / "paired_contrasts.json", contrast_document)
    _write_csv(root / "analysis" / "failures_and_sensitivity.csv", failures, FAILURE_FIELDS)
    return {
        "protocol_version": "coordinated_mechanism_v1", "complete": complete, "require_complete": require_complete,
        "configs": main, "summaries": summaries, "quality_metrics": quality, "pool_metrics": pool_summaries,
        "per_group_pool_metrics": pool_rows, "costs": costs, "per_group_costs": cost_rows,
        "contrasts": contrasts, "paired_contrasts": contrasts, "failures_and_sensitivity": failures,
        "expected_configs": sorted(expected_configs), "missing_configs": missing_configs,
        "missing_groups": incomplete_groups, "stage_status": stage_status,
        "statistics": {"seed": seed, "bootstrap": bootstrap, "randomization": randomization, **contrast_meta},
        "denominators": {"questions": len(questions), "snapshots": len(snapshots)},
    }


__all__ = [
    "ALL_METRICS", "CONTRAST_REGISTRY", "PRIMARY_FAMILIES", "PRIMARY_METRIC", "STAGE_CONFIGS",
    "analyze_experiment", "collect_selector_results", "expected_gated_configs",
]
