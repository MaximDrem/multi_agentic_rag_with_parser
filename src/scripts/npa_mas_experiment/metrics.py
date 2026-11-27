from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    CUTOFFS,
    Benchmark,
    QueryGroup,
    deduplicate_ranking,
    ranking_ids,
)

METRIC_NAMES = tuple(f"{metric}@{cutoff}" for cutoff in CUTOFFS for metric in ("hit_rate", "recall", "ndcg", "mrr"))


def query_metrics(
    ranking: Any,
    relevant: Mapping[str, int],
    *,
    cutoffs: Sequence[int] = CUTOFFS,
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, float]:
    positive = {str(chunk_id): int(value) for chunk_id, value in relevant.items() if int(value) > 0}
    if not positive:
        raise ValueError("Empty qrel set")
    maximum = max(cutoffs)
    unique_ranking = deduplicate_ranking(
        ranking,
        valid_chunk_ids=valid_chunk_ids,
        max_items=maximum,
        reject_duplicates=True,
    )
    ideal = sorted(positive.values(), reverse=True)
    result = {}
    for cutoff in cutoffs:
        top = unique_ranking[:cutoff]
        hits = [chunk_id for chunk_id in top if chunk_id in positive]
        dcg = sum(
            (2 ** positive[chunk_id] - 1) / math.log2(rank + 1)
            for rank, chunk_id in enumerate(top, 1)
            if chunk_id in positive
        )
        idcg = sum(
            (2 ** relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(ideal[:cutoff], 1)
        )
        first = next((rank for rank, chunk_id in enumerate(top, 1) if chunk_id in positive), None)
        result[f"hit_rate@{cutoff}"] = float(bool(hits))
        result[f"recall@{cutoff}"] = len(set(hits)) / len(positive)
        result[f"ndcg@{cutoff}"] = dcg / idcg if idcg else 0.0
        result[f"mrr@{cutoff}"] = 1.0 / first if first is not None else 0.0
    return result


def average_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    materialized = list(rows)
    if not materialized:
        return {name: None for name in METRIC_NAMES}
    return {
        name: math.fsum(float(row[name]) for row in materialized) / len(materialized)
        for name in METRIC_NAMES
    }


def macro_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    return average_metrics(rows)


def _failed_artifact(artifact: Any) -> bool:
    if not isinstance(artifact, Mapping):
        return False
    status = str(artifact.get("status", artifact.get("group_status", "success")))
    return bool(artifact.get("fallback_used")) or bool(artifact.get("strict_failed")) or status in {"fallback", "failed", "error"}


def _ranking_for_policy(artifact: Any, policy: str) -> list[str]:
    if policy in {"strict", "strict_failure_as_empty"} and _failed_artifact(artifact):
        return []
    return ranking_ids(artifact)


def score_benchmark(
    benchmark: Benchmark,
    rankings: Mapping[str, Any],
    *,
    output_policy: str = "complete_with_declared_fallback",
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if output_policy not in {"complete_with_declared_fallback", "practical", "strict", "strict_failure_as_empty"}:
        raise ValueError(f"Unknown output policy: {output_policy}")
    missing = sorted(set(benchmark.group_by_id) - set(rankings))
    unknown = sorted(set(rankings) - set(benchmark.group_by_id))
    if missing:
        raise ValueError(f"Missing group rankings: {missing[:5]}")
    if unknown:
        raise ValueError(f"Unknown group rankings: {unknown[:5]}")
    per_group = []
    per_query = []
    all_groups = []
    for group in benchmark.groups:
        ranking = deduplicate_ranking(
            _ranking_for_policy(rankings[group.query_group], output_policy),
            valid_chunk_ids=valid_chunk_ids,
            max_items=10,
            reject_duplicates=True,
        )
        base = {
            "query_group": group.query_group,
            "representative_question_id": group.representative_question_id,
            "question_ids": list(group.question_ids),
            "query": group.query,
            "answerable": group.answerable,
            "retrieved": len(ranking),
            "ranked_chunk_ids": ranking,
            "fallback_used": _failed_artifact(rankings[group.query_group]),
        }
        if group.answerable:
            group_row = {
                **base,
                "qrels": len(group.relevant),
                **query_metrics(ranking, group.relevant, valid_chunk_ids=valid_chunk_ids),
            }
            per_group.append(group_row)
        all_groups.append(base)
        for question_id in group.question_ids:
            relevant = benchmark.qrels.get(question_id, {})
            if not relevant:
                continue
            per_query.append(
                {
                    "question_id": question_id,
                    "query_group": group.query_group,
                    "query": benchmark.question_by_id[question_id].query,
                    "qrels": len(relevant),
                    "retrieved": len(ranking),
                    **query_metrics(ranking, relevant, valid_chunk_ids=valid_chunk_ids),
                }
            )
    summary = {
        "questions_total": len(benchmark.questions),
        "groups_total": len(benchmark.groups),
        "run_rows": len(benchmark.questions),
        "answerable_rows": len(per_query),
        "answerable_unique_queries": len(per_group),
        "answerable_groups": len(per_group),
        "rows_without_qrels": len(benchmark.questions) - len(per_query),
        "groups_without_qrels": len(benchmark.groups) - len(per_group),
        "row_macro": average_metrics(per_query),
        "unique_query_macro": average_metrics(per_group),
        "output_policy": output_policy,
        "ndcg_gain": "2^relevance-1",
        "duplicate_policy": "normalized_group_representative_min_source_row",
        "group_qrels_policy": "union_max_relevance",
        "empty_qrel_policy": "excluded_from_quality_denominators",
        "empty_ranking_policy": "explicit_empty_ranking_scores_zero_and_remains_in_denominator",
    }
    return {
        "per_query": per_query,
        "per_group": per_group,
        "all_groups": all_groups,
        "row_macro": summary["row_macro"],
        "unique_query_macro": summary["unique_query_macro"],
        "summary": summary,
    }


def score_run(
    benchmark: Benchmark,
    rankings: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    return score_benchmark(benchmark, rankings, **kwargs)


def score_variants(
    benchmark: Benchmark,
    variant_rankings: Mapping[str, Mapping[str, Any]],
    *,
    output_policy: str = "complete_with_declared_fallback",
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        variant: score_benchmark(
            benchmark,
            rankings,
            output_policy=output_policy,
            valid_chunk_ids=valid_chunk_ids,
        )
        for variant, rankings in variant_rankings.items()
    }


def pool_recall(
    candidates: Any,
    relevant: Mapping[str, int],
    *,
    cutoff: int = 30,
) -> float:
    if not relevant:
        raise ValueError("Empty qrel set")
    if isinstance(candidates, Mapping) and "candidates" in candidates:
        candidates = candidates["candidates"]
    pool_ids = set(deduplicate_ranking(candidates, max_items=cutoff))
    return len(pool_ids & set(relevant)) / len(relevant)


def _channel_ranking_ids(ranking: Any) -> list[str]:
    if isinstance(ranking, Mapping) and "candidates" in ranking:
        ranking = ranking["candidates"]
    return deduplicate_ranking(ranking, max_items=None)


def _provenance_records(raw: Any, channel: str | None = None) -> list[dict[str, Any]]:
    records = []
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("rank", "raw_rank", "first_raw_rank", "query", "query_index")):
            record = dict(raw)
            if channel is not None:
                record.setdefault("channel", channel)
            records.append(record)
        else:
            for key, value in raw.items():
                records.extend(_provenance_records(value, str(key)))
    elif isinstance(raw, (list, tuple)):
        for value in raw:
            records.extend(_provenance_records(value, channel))
    elif isinstance(raw, (int, float)) and channel is not None:
        records.append({"channel": channel, "rank": int(raw)})
    return records


def candidate_provenance(
    pool: Mapping[str, Any],
    relevant_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = list(pool.get("candidates", []))
    wanted = set(relevant_ids) if relevant_ids is not None else None
    pool_provenance = pool.get("provenance", {})
    raw_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if isinstance(pool_provenance, Mapping):
        raw_hits = pool_provenance.get("raw_hits", {})
        if isinstance(raw_hits, Mapping):
            for channel, query_rows in raw_hits.items():
                if not isinstance(query_rows, list):
                    continue
                for query_index, query_row in enumerate(query_rows):
                    if not isinstance(query_row, Mapping):
                        continue
                    for hit in query_row.get("hits", []):
                        if isinstance(hit, Mapping) and hit.get("chunk_id") is not None:
                            raw_by_chunk[str(hit["chunk_id"])].append(
                                {
                                    "channel": channel,
                                    "query": query_row.get("query"),
                                    "query_index": query_row.get("query_key", query_index),
                                    "rank": hit.get("rank"),
                                }
                            )
    result = []
    for index, raw_candidate in enumerate(candidates, 1):
        candidate = dict(raw_candidate)
        chunk_id = str(candidate["chunk_id"])
        if wanted is not None and chunk_id not in wanted:
            continue
        raw_records = list(raw_by_chunk.get(chunk_id, []))
        source_records = _provenance_records(candidate.get("source_ranks", candidate.get("provenance", {})))
        observed_channels = {record.get("channel") for record in raw_records}
        raw_records.extend(record for record in source_records if record.get("channel") not in observed_channels)
        if isinstance(pool_provenance, Mapping) and chunk_id in pool_provenance:
            raw_records.extend(_provenance_records(pool_provenance[chunk_id]))
        if not raw_records:
            raw_records = [{}]
        seen_records = set()
        for record in raw_records:
            rank = record.get("first_raw_rank", record.get("raw_rank", record.get("rank")))
            output = {
                "chunk_id": chunk_id,
                "channel": record.get("channel"),
                "query": record.get("query"),
                "query_index": record.get("query_index"),
                "first_raw_rank": int(rank) if rank is not None else None,
                "fusion_rank": int(candidate.get("pool_rank", index)),
            }
            signature = repr(sorted(output.items(), key=lambda item: item[0]))
            if signature not in seen_records:
                result.append(output)
                seen_records.add(signature)
    return result


def _sum_named_numbers(value: Any, names: set[str]) -> int:
    total = 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names and isinstance(child, (int, float)) and not isinstance(child, bool):
                total += int(child)
            else:
                total += _sum_named_numbers(child, names)
    elif isinstance(value, (list, tuple)):
        for child in value:
            total += _sum_named_numbers(child, names)
    return total


def _budget_totals(value: Any) -> dict[str, int]:
    ledgers = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            if {"requested_raw_hits", "actual_raw_hits"} <= set(item):
                ledgers.append(item)
                return
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    return {
        "target_raw_hits": sum(int(ledger.get("target_raw_hits", 0)) for ledger in ledgers),
        "requested_raw_hits": sum(int(ledger["requested_raw_hits"]) for ledger in ledgers),
        "actual_raw_hits": sum(int(ledger["actual_raw_hits"]) for ledger in ledgers),
        "unspent_raw_hits": sum(
            int(ledger.get("unspent_raw_hits", int(ledger["requested_raw_hits"]) - int(ledger["actual_raw_hits"])))
            for ledger in ledgers
        ),
        "saturated_ledgers": sum(ledger.get("channel_status", ledger.get("status")) == "saturated" for ledger in ledgers),
    }


def summarize_candidate_pool(
    pool: Mapping[str, Any],
    relevant: Mapping[str, int],
    *,
    channel_rankings: Mapping[str, Any] | None = None,
    baseline_pool: Any = None,
    cutoff: int = 30,
) -> dict[str, Any]:
    if not relevant:
        raise ValueError("Empty qrel set")
    candidates = list(pool.get("candidates", []))
    pool_ids = deduplicate_ranking(candidates, max_items=cutoff)
    pool_set = set(pool_ids)
    relevant_set = set(relevant)
    provenance = pool.get("provenance", {})
    channels = channel_rankings or pool.get("channel_rankings") or provenance.get("channel_rankings", {})
    channel_rows = {}
    channel_sets = {}
    for channel, ranking in sorted(channels.items()):
        ids = set(_channel_ranking_ids(ranking)[:cutoff])
        channel_sets[channel] = ids
        channel_rows[channel] = {
            "unique_ids": len(ids),
            "relevant_ids": len(ids & relevant_set),
            "recall@30": len(ids & relevant_set) / len(relevant_set),
        }
    for channel, ids in channel_sets.items():
        other_ids = set().union(*(values for other, values in channel_sets.items() if other != channel)) if len(channel_sets) > 1 else set()
        channel_rows[channel]["relevant_unique_contribution"] = sorted((ids & relevant_set) - other_ids)
        channel_rows[channel]["unique_contribution_count"] = len((ids & relevant_set) - other_ids)
    intersection = set.intersection(*channel_sets.values()) if channel_sets else set()
    union = set.union(*channel_sets.values()) if channel_sets else set()
    baseline_ids = set(_channel_ranking_ids(baseline_pool)) if baseline_pool is not None else set()
    generated = pool.get("generated_queries", {})
    generated_counts = {}
    for channel, queries in generated.items():
        if isinstance(queries, Mapping):
            queries = queries.get("queries", queries.get("generated", []))
        cleaned = [str(query).strip() for query in queries if str(query).strip()]
        generated_counts[channel] = {
            "generated": len(cleaned),
            "unique": len(dict.fromkeys(cleaned)),
            "duplicates_removed": len(cleaned) - len(dict.fromkeys(cleaned)),
        }
    truncated = pool.get("truncated_queries", provenance.get("truncated_queries", 0))
    if isinstance(truncated, Mapping):
        truncated_count = sum(len(value) if isinstance(value, (list, tuple)) else int(value) for value in truncated.values())
    elif isinstance(truncated, (list, tuple)):
        truncated_count = len(truncated)
    else:
        truncated_count = int(truncated)
    duplicate_queries = provenance.get("duplicate_queries", {})
    duplicate_query_count = sum(len(value) for value in duplicate_queries.values()) if isinstance(duplicate_queries, Mapping) else 0
    duplicate_raw_hits = 0
    raw_hits = provenance.get("raw_hits", {}) if isinstance(provenance, Mapping) else {}
    if isinstance(raw_hits, Mapping):
        for query_rows in raw_hits.values():
            if not isinstance(query_rows, list):
                continue
            for query_row in query_rows:
                hits = query_row.get("hits", []) if isinstance(query_row, Mapping) else []
                hit_ids = [str(hit["chunk_id"]) for hit in hits if isinstance(hit, Mapping) and "chunk_id" in hit]
                duplicate_raw_hits += len(hit_ids) - len(set(hit_ids))
    result = {
        "query_group": pool.get("query_group"),
        "pool_sha256": pool.get("pool_sha256"),
        "pool_recall@30": len(pool_set & relevant_set) / len(relevant_set),
        "relevant_ids": len(pool_set & relevant_set),
        "unique_ids": len(pool_set),
        "qrels": len(relevant_set),
        "channels": channel_rows,
        "channel_intersection_ids": len(intersection),
        "channel_union_ids": len(union),
        "channel_relevant_intersection_ids": len(intersection & relevant_set),
        "new_relevant_ids_vs_baseline": sorted((pool_set & relevant_set) - baseline_ids),
        "new_relevant_count_vs_baseline": len((pool_set & relevant_set) - baseline_ids),
        "generated_queries": generated_counts,
        "generated_queries_total": sum(row["generated"] for row in generated_counts.values()),
        "unique_generated_queries_total": sum(row["unique"] for row in generated_counts.values()),
        "truncated_queries": truncated_count,
        "duplicates_removed": _sum_named_numbers(pool, {"duplicates_removed", "duplicate_hits_removed"}) + duplicate_raw_hits,
        "duplicate_queries": duplicate_query_count,
        "duplicate_raw_hits": duplicate_raw_hits,
        "pool_size_before_cap": int(pool.get("pool_size_before_cap", pool.get("candidate_count_before_cap", len(candidates)))),
        "pool_size_after_cap": len(pool_set),
        "provenance": candidate_provenance(pool, relevant_set),
        **_budget_totals(pool.get("budget", {})),
    }
    return result


def aggregate_candidate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    numeric_keys = (
        "pool_recall@30",
        "relevant_ids",
        "unique_ids",
        "qrels",
        "new_relevant_count_vs_baseline",
        "generated_queries_total",
        "unique_generated_queries_total",
        "truncated_queries",
        "duplicates_removed",
        "pool_size_before_cap",
        "pool_size_after_cap",
        "target_raw_hits",
        "requested_raw_hits",
        "actual_raw_hits",
        "unspent_raw_hits",
        "saturated_ledgers",
    )
    result: dict[str, Any] = {"groups": len(materialized)}
    for key in numeric_keys:
        values = [float(row[key]) for row in materialized if key in row and row[key] is not None]
        result[f"{key}_mean"] = math.fsum(values) / len(values) if values else None
        if key not in {"pool_recall@30"}:
            result[f"{key}_sum"] = int(math.fsum(values)) if values else 0
    return result
