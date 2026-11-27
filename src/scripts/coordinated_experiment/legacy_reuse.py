import copy
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.coordinated_experiment.pools import apply_backfill, build_pool_record
from scripts.coordinated_experiment.schema import (
    atomic_write_json,
    file_sha256,
    load_config,
    snapshot_hash,
    text_sha256,
)

CHANNELS = ("dense", "bm25")
DEFAULT_PROTOCOL_VERSION = "coordinated_mechanism_v1"
DEFAULT_CONTROLS = {
    "groups": 556,
    "rows": 630,
    "expansion_active": 382,
    "expansion_inactive": 174,
    "fallback_groups": 4,
    "strict_short_groups": 72,
}


class LegacyReuseAuditError(ValueError):
    def __init__(self, report: Mapping[str, Any], report_path: Path):
        self.report = dict(report)
        self.report_path = report_path
        error_count = report.get("summary", {}).get("error_count", 0)
        super().__init__(f"Legacy reuse audit failed with {error_count} errors: {report_path}")


class _GroupError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise _GroupError(code, message)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise _GroupError("invalid_integer", f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    raise _GroupError("invalid_integer", f"{field} must be an integer")


def _token_value(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _config_mapping(config: Mapping[str, Any] | str | os.PathLike[str] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (str, os.PathLike)):
        return load_config(config)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping, path, or None")
    return dict(config)


def _expected_controls(config: Mapping[str, Any]) -> dict[str, int]:
    expected = dict(DEFAULT_CONTROLS)
    benchmark = config.get("benchmark", {})
    if isinstance(benchmark, Mapping):
        expected["groups"] = int(benchmark.get("expected_groups", expected["groups"]))
        expected["rows"] = int(benchmark.get("expected_rows", expected["rows"]))
    reuse = config.get("reuse_audit", {})
    if isinstance(reuse, Mapping):
        overrides = reuse.get("expected_controls", {})
        if isinstance(overrides, Mapping):
            for key in expected:
                if key in overrides:
                    expected[key] = int(overrides[key])
        aliases = {
            "groups": "expected_groups",
            "rows": "expected_rows",
            "expansion_active": "expected_expansion_active",
            "expansion_inactive": "expected_expansion_inactive",
            "fallback_groups": "expected_fallback_groups",
            "strict_short_groups": "expected_strict_short_groups",
        }
        for key, option in aliases.items():
            if option in reuse:
                expected[key] = int(reuse[option])
    return expected


def _chunk_value(chunk: Any, key: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk[key]
    return getattr(chunk, key)


def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk.get("id", chunk.get("chunk_id")))
    return str(getattr(chunk, "id", getattr(chunk, "chunk_id", None)))


def _load_chunks(path: Path) -> tuple[list[Any], dict[str, Any], str]:
    with path.open("rb") as stream:
        database = pickle.load(stream)
    if isinstance(database, Mapping) and "chunks" in database:
        chunks = list(database["chunks"])
    elif hasattr(database, "chunks"):
        chunks = list(database.chunks)
    elif isinstance(database, Sequence) and not isinstance(database, (str, bytes)):
        chunks = list(database)
    else:
        raise ValueError("chunks_pkl must contain a chunk database or chunk sequence")
    chunk_map: dict[str, Any] = {}
    for chunk in chunks:
        chunk_id = _chunk_id(chunk)
        if chunk_id in chunk_map:
            raise ValueError(f"Duplicate chunk ID in chunks_pkl: {chunk_id}")
        chunk_map[chunk_id] = chunk
    payload = json.dumps(
        [[_chunk_id(chunk), str(_chunk_value(chunk, "text"))] for chunk in chunks],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return chunks, chunk_map, hashlib.sha256(payload).hexdigest()


def _benchmark_hashes(
    benchmark_dir: Path,
    config: Mapping[str, Any],
    chunks: Sequence[Any],
    corpus_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark_config = config.get("benchmark", {})
    if not isinstance(benchmark_config, Mapping):
        benchmark_config = {}
    result: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for name in ("questions", "qrels", "evidence", "coverage"):
        filename = str(benchmark_config.get(name, f"{name}.jsonl"))
        expected = manifest.get(f"{name}_sha256")
        actual = file_sha256(benchmark_dir / filename)
        valid = isinstance(expected, str) and actual == expected
        result[name] = {"expected": expected, "actual": actual, "valid": valid}
        if not valid:
            errors.append({
                "code": "benchmark_hash_mismatch",
                "message": f"{name} hash mismatch: expected {expected}, got {actual}",
            })
    expected_corpus = manifest.get("corpus_sha256")
    corpus_valid = isinstance(expected_corpus, str) and corpus_sha256 == expected_corpus
    result["corpus"] = {
        "expected": expected_corpus,
        "actual": corpus_sha256,
        "valid": corpus_valid,
    }
    if not corpus_valid:
        errors.append({
            "code": "corpus_hash_mismatch",
            "message": f"Corpus hash mismatch: expected {expected_corpus}, got {corpus_sha256}",
        })
    expected_chunks = manifest.get("chunks")
    count_valid = expected_chunks is None or int(expected_chunks) == len(chunks)
    result["chunk_count"] = {
        "expected": expected_chunks,
        "actual": len(chunks),
        "valid": count_valid,
    }
    if not count_valid:
        errors.append({
            "code": "chunk_count_mismatch",
            "message": f"Chunk count mismatch: expected {expected_chunks}, got {len(chunks)}",
        })
    return result, errors


def _read_legacy_jsonl(path: Path) -> tuple[list[tuple[dict[str, Any], str]], str]:
    rows: list[tuple[dict[str, Any], str]] = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            raw = line.rstrip(b"\r\n")
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Each legacy detailed.jsonl row must be an object")
            rows.append((value, hashlib.sha256(raw).hexdigest()))
    return rows, digest.hexdigest()


def _reports(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "invalid_reports", f"{field} must be a mapping")
    reports = dict(value)
    for channel in CHANNELS:
        report = reports.get(channel)
        _require(isinstance(report, Mapping), "invalid_reports", f"{field}.{channel} must be a mapping")
        assessments = report.get("assessments", [])
        queries = report.get("search_queries", [])
        _require(isinstance(assessments, list), "invalid_reports", f"{field}.{channel}.assessments must be a list")
        _require(isinstance(queries, list), "invalid_reports", f"{field}.{channel}.search_queries must be a list")
        _require(all(isinstance(query, str) for query in queries), "invalid_reports", f"{field}.{channel} queries must be strings")
    return reports


def _validate_report_aliases(
    reports: Mapping[str, Any],
    observations: Mapping[str, list[str]],
    alias_to_id: Mapping[str, str],
    field: str,
) -> None:
    for channel in CHANNELS:
        observed = set(observations[channel])
        for assessment in reports[channel].get("assessments", []):
            _require(isinstance(assessment, Mapping), "invalid_reports", f"{field}.{channel} assessment must be an object")
            alias = str(assessment.get("alias", ""))
            chunk_id = alias_to_id.get(alias)
            _require(chunk_id in observed, "report_observation_mismatch", f"{field}.{channel} uses unobserved alias {alias}")


def _generation_usage(trace: Mapping[str, Any], result: Mapping[str, Any], active: bool) -> dict[str, Any]:
    agent_meta = trace.get("agent_meta", {})
    _require(isinstance(agent_meta, Mapping), "invalid_agent_meta", "agent_meta must be a mapping")
    atomic_stages = ["planner", "dense_round_1", "bm25_round_1", "dense_round_2", "bm25_round_2", "arbiter"]
    stage_tokens: dict[str, int | float | None] = {}
    statuses: dict[str, str] = {}
    for stage in atomic_stages:
        executed = active or not stage.endswith("_round_2")
        meta = agent_meta.get(stage)
        value = _token_value(meta.get("tokens")) if isinstance(meta, Mapping) else None
        if not executed:
            stage_tokens[stage] = 0
            statuses[stage] = "not_executed"
        elif value is not None:
            stage_tokens[stage] = value
            statuses[stage] = "legacy_total_only"
        else:
            stage_tokens[stage] = None
            statuses[stage] = "unavailable"
    for stage, parts in (
        ("round_1", ("dense_round_1", "bm25_round_1")),
        ("round_2", ("dense_round_2", "bm25_round_2")),
    ):
        values = [stage_tokens[part] for part in parts]
        if not active and stage == "round_2":
            stage_tokens[stage] = 0
            statuses[stage] = "not_executed"
        elif all(value is not None for value in values):
            stage_tokens[stage] = sum(values)
            statuses[stage] = "legacy_total_only"
        else:
            stage_tokens[stage] = None
            statuses[stage] = "unavailable"
    token_usage = result.get("token_usage", {})
    total = _token_value(token_usage.get("total_tokens")) if isinstance(token_usage, Mapping) else None
    if total is not None and all(stage_tokens[stage] is not None for stage in atomic_stages):
        computed = sum(stage_tokens[stage] for stage in atomic_stages)
        _require(total == computed, "token_total_mismatch", f"Legacy total tokens {total} != stage sum {computed}")
    statuses["total_tokens"] = "legacy_total_only" if total is not None else "unavailable"
    statuses["prompt_tokens"] = "unavailable_raw_legacy_usage"
    statuses["completion_tokens"] = "unavailable_raw_legacy_usage"
    return {
        "stage_tokens": stage_tokens,
        "measurement_status": statuses,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": total,
        "token_measurement_status": statuses["total_tokens"],
    }


def _build_snapshot(
    record: Mapping[str, Any],
    source_trace_sha256: str,
    group_index: int,
    chunk_map: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result = record.get("result")
    trace = result.get("trace") if isinstance(result, Mapping) else None
    _require(isinstance(result, Mapping) and isinstance(trace, Mapping), "invalid_trace", "result.trace must be a mapping")
    query_group = record.get("query_group")
    query = record.get("query")
    _require(isinstance(query_group, str) and isinstance(query, str), "invalid_query", "query_group and query must be strings")
    _require(trace.get("query") == query, "query_mismatch", "record.query differs from trace.query")
    questions = record.get("questions")
    _require(isinstance(questions, list) and bool(questions), "invalid_questions", "questions must be a nonempty list")
    question_ids: list[str] = []
    for question in questions:
        _require(isinstance(question, Mapping), "invalid_questions", "question entries must be objects")
        question_id = question.get("question_id")
        _require(isinstance(question_id, str) and bool(question_id), "invalid_questions", "question_id must be a nonempty string")
        question_ids.append(question_id)
    _require(len(set(question_ids)) == len(question_ids), "duplicate_question_id", "question IDs must be unique within a group")
    plan = trace.get("plan")
    _require(isinstance(plan, Mapping), "invalid_plan", "trace.plan must be a mapping")

    ledger = trace.get("candidate_ledger")
    _require(isinstance(ledger, Mapping), "invalid_ledger", "candidate_ledger must be a mapping")
    raw_unique = _integer(trace.get("raw_unique_candidates"), "raw_unique_candidates")
    _require(len(ledger) == raw_unique, "ledger_count_mismatch", f"Ledger has {len(ledger)} entries, raw_unique_candidates={raw_unique}")
    entries: list[tuple[int, str, str, Mapping[str, Any]]] = []
    aliases_seen: set[str] = set()
    ids_seen: set[str] = set()
    for map_key, entry in ledger.items():
        _require(isinstance(entry, Mapping), "invalid_ledger", f"Ledger entry {map_key} must be an object")
        chunk_id = entry.get("chunk_id")
        alias = entry.get("alias")
        _require(chunk_id == map_key, "ledger_id_mismatch", f"Ledger key {map_key} differs from entry chunk_id {chunk_id}")
        _require(isinstance(chunk_id, str) and chunk_id in chunk_map, "missing_chunk", f"Missing chunk in chunks_pkl: {chunk_id}")
        _require(isinstance(alias, str) and re.fullmatch(r"C\d{3}", alias) is not None, "invalid_alias", f"Invalid alias: {alias}")
        _require(alias not in aliases_seen and chunk_id not in ids_seen, "duplicate_alias_or_id", "Ledger aliases and chunk IDs must be unique")
        aliases_seen.add(alias)
        ids_seen.add(chunk_id)
        entries.append((int(alias[1:]), alias, chunk_id, entry))
    entries.sort(key=lambda item: item[0])
    _require([item[0] for item in entries] == list(range(1, len(entries) + 1)), "alias_sequence_mismatch", "Aliases must be exactly C001 through C%03d" % len(entries))
    expanded_ids = [item[2] for item in entries]
    alias_to_id = {alias: chunk_id for _, alias, chunk_id, _ in entries}

    rounds = trace.get("rounds")
    _require(isinstance(rounds, list), "invalid_rounds", "trace.rounds must be a list")
    first_rounds = [item for item in rounds if isinstance(item, Mapping) and "reports" in item and _integer(item.get("round"), "round") == 1]
    expansion_rounds = [item for item in rounds if isinstance(item, Mapping) and "expansion" in item]
    second_rounds = [item for item in rounds if isinstance(item, Mapping) and "reports" in item and _integer(item.get("round"), "round") == 2]
    _require(len(first_rounds) == 1, "round1_mismatch", "Exactly one round-1 reports entry is required")
    first = _reports(first_rounds[0]["reports"], "R1")
    expansion_queries = {channel: list(first[channel]["search_queries"]) for channel in CHANNELS}
    expansion_active = any(expansion_queries.values())
    _require(len(expansion_rounds) == int(expansion_active), "expansion_round_mismatch", "Expansion entry presence must match nonempty W1 queries")
    _require(len(second_rounds) == int(expansion_active), "round2_mismatch", "Round 2 must exist iff W1 queries exist")
    if expansion_active:
        expansion = expansion_rounds[0]["expansion"]
        _require(isinstance(expansion, Mapping), "invalid_expansion", "expansion must be a mapping")
        for channel in CHANNELS:
            _require(isinstance(expansion.get(channel), Mapping), "invalid_expansion", f"expansion.{channel} must be a mapping")
            _require(list(expansion[channel].get("queries", [])) == expansion_queries[channel], "expansion_query_mismatch", f"expansion.{channel}.queries differs from W1 queries")
        second = _reports(second_rounds[0]["reports"], "legacy_R2P")
        _require(second_rounds[0].get("shared_previous_reports") == first, "shared_reports_mismatch", "Round-2 shared_previous_reports must equal R1")
        for channel in CHANNELS:
            _require(second[channel].get("search_queries") == [], "round2_queries_forbidden", f"legacy_R2P.{channel} contains search queries")
    else:
        expansion = {channel: {"queries": [], "raw_new_ids": [], "new_observed_ids": []} for channel in CHANNELS}
        second = copy.deepcopy(first)

    retrieval_by_event: dict[tuple[str, int, str], list[tuple[int, str, str, int]]] = {}
    for alias_number, alias, chunk_id, entry in entries:
        retrievals = entry.get("retrievals")
        _require(isinstance(retrievals, list), "invalid_retrievals", f"{chunk_id}.retrievals must be a list")
        for retrieval_index, retrieval in enumerate(retrievals):
            _require(isinstance(retrieval, Mapping), "invalid_retrievals", f"{chunk_id} retrieval must be an object")
            channel = retrieval.get("channel")
            event_query = retrieval.get("query")
            _require(channel in CHANNELS, "invalid_retrieval_channel", f"Invalid retrieval channel: {channel}")
            _require(isinstance(event_query, str), "invalid_retrieval_query", "Retrieval query must be a string")
            call_index = _integer(retrieval.get("call_index"), "call_index")
            rank = _integer(retrieval.get("rank"), "rank")
            _require(call_index in (0, 1) and rank >= 1, "invalid_retrieval", "call_index must be 0/1 and rank must be positive")
            key = (str(channel), call_index, event_query)
            retrieval_by_event.setdefault(key, []).append((alias_number, alias, chunk_id, rank))

    event_specs: list[tuple[str, int, str, str, int]] = [
        ("dense", 0, query, "dense_initial", 0),
        ("bm25", 0, query, "bm25_initial", 0),
    ]
    for channel in CHANNELS:
        _require(len(set(expansion_queries[channel])) == len(expansion_queries[channel]), "duplicate_expansion_query", f"Duplicate {channel} expansion query")
        for query_index, expansion_query in enumerate(expansion_queries[channel], 1):
            event_specs.append((channel, 1, expansion_query, f"{channel}_expansion_{query_index}", query_index))
    events: list[dict[str, Any]] = []
    event_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for channel, call_index, event_query, event_id, _ in event_specs:
        key = (channel, call_index, event_query)
        raw_hits = retrieval_by_event.pop(key, [])
        _require(len({item[2] for item in raw_hits}) == len(raw_hits), "duplicate_event_chunk", f"Duplicate chunk in event {event_id}")
        raw_hits.sort(key=lambda item: (item[3], item[0]))
        _require([item[3] for item in raw_hits] == list(range(1, len(raw_hits) + 1)), "event_rank_mismatch", f"Ranks in {event_id} must be contiguous from 1")
        hits = []
        event_text_hashes: dict[str, str] = {}
        for _, _, chunk_id, rank in raw_hits:
            chunk_text_hash = text_sha256(str(_chunk_value(chunk_map[chunk_id], "text")))
            hits.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "text_sha256": chunk_text_hash,
            })
            event_text_hashes[chunk_id] = chunk_text_hash
        event = {
            "event_id": event_id,
            "channel": channel,
            "stage": "initial" if call_index == 0 else "expansion",
            "query": event_query,
            "call_index": call_index,
            "hits": hits,
            "text_sha256": event_text_hashes,
        }
        events.append(event)
        event_lookup[key] = event
    _require(not retrieval_by_event, "unexpected_retrieval_events", f"Retrievals do not match initial/expansion queries: {sorted(retrieval_by_event)[:3]}")

    initial = {
        channel: [hit["chunk_id"] for hit in event_lookup[(channel, 0, query)]["hits"] if hit["rank"] <= 10]
        for channel in CHANNELS
    }
    base_from_events = {
        item[2]
        for item in entries
        if any(_integer(retr.get("call_index"), "call_index") == 0 for retr in item[3].get("retrievals", []))
    }
    base_ids = [chunk_id for chunk_id in expanded_ids if chunk_id in base_from_events]
    initial_event_ids = {hit["chunk_id"] for channel in CHANNELS for hit in event_lookup[(channel, 0, query)]["hits"]}
    _require(set(base_ids) == initial_event_ids, "base_pool_mismatch", "P0 must exactly equal call-0 hit IDs")
    _require(set(base_ids) <= set(expanded_ids), "pool_subset_mismatch", "P0 must be a subset of P1")

    final_observations = {channel: list(initial[channel]) for channel in CHANNELS}
    expected_observed_by = {chunk_id: [] for chunk_id in expanded_ids}
    for channel in CHANNELS:
        for chunk_id in initial[channel]:
            expected_observed_by[chunk_id].append(channel)
    known_ids = set(base_ids)
    for channel in CHANNELS:
        expected_new_observed: list[str] = []
        expected_raw_new: list[str] = []
        for event in events:
            if event["channel"] != channel or event["stage"] != "expansion":
                continue
            for hit in event["hits"]:
                chunk_id = hit["chunk_id"]
                if chunk_id not in known_ids:
                    known_ids.add(chunk_id)
                    expected_raw_new.append(chunk_id)
                if chunk_id not in final_observations[channel] and len(final_observations[channel]) < 15:
                    final_observations[channel].append(chunk_id)
                    expected_new_observed.append(chunk_id)
        _require(list(expansion[channel].get("new_observed_ids", [])) == expected_new_observed, "observation_mismatch", f"expansion.{channel}.new_observed_ids is inconsistent with retrieval order")
        _require(list(expansion[channel].get("raw_new_ids", [])) == expected_raw_new, "raw_new_mismatch", f"expansion.{channel}.raw_new_ids is inconsistent with pool insertion")
        for chunk_id in expected_new_observed:
            expected_observed_by[chunk_id].append(channel)
    _require(known_ids == set(expanded_ids), "expanded_pool_mismatch", "P1 must equal initial plus expansion IDs")
    for chunk_id, expected in expected_observed_by.items():
        entry = next(item[3] for item in entries if item[2] == chunk_id)
        _require(list(entry.get("observed_by", [])) == expected, "observed_by_mismatch", f"observed_by mismatch for {chunk_id}: expected {expected}")
    _validate_report_aliases(first, initial, alias_to_id, "R1")
    _validate_report_aliases(second, final_observations, alias_to_id, "legacy_R2P")

    base_pool = build_pool_record(base_ids, chunk_map)
    expanded_pool = build_pool_record(expanded_ids, chunk_map)
    retrieval_config = config.get("retrieval", {})
    output_k = int(retrieval_config.get("output_k", 10)) if isinstance(retrieval_config, Mapping) else 10
    _require(0 < output_k <= 10, "invalid_output_k", "output_k must be in 1..10")
    strict_rows: list[tuple[int, str]] = []
    for _, _, chunk_id, entry in entries:
        arbiter = entry.get("arbiter", {})
        _require(isinstance(arbiter, Mapping), "invalid_arbiter", f"arbiter for {chunk_id} must be an object")
        if arbiter.get("selected_by_arbiter") is True:
            strict_rows.append((_integer(arbiter.get("final_rank"), "final_rank"), chunk_id))
    strict_rows.sort(key=lambda item: item[0])
    _require(len({rank for rank, _ in strict_rows}) == len(strict_rows), "arbiter_rank_mismatch", "Strict arbiter final_rank values must be unique")
    _require([rank for rank, _ in strict_rows] == list(range(1, len(strict_rows) + 1)), "arbiter_rank_mismatch", "Strict arbiter ranks must be contiguous from 1")
    strict_ids = [chunk_id for _, chunk_id in strict_rows]
    _require(len(strict_ids) <= output_k and len(set(strict_ids)) == len(strict_ids), "arbiter_selection_mismatch", "Strict arbiter IDs must be unique and at most output_k")
    _require(set(strict_ids) <= set(expanded_ids), "arbiter_selection_mismatch", "Strict arbiter IDs must belong to P1")
    production_ids = apply_backfill(strict_ids, expanded_pool, [], output_k=output_k, reports=second, policy="legacy")
    result_chunks = result.get("chunks")
    _require(isinstance(result_chunks, list), "invalid_production", "result.chunks must be a list")
    actual_production_ids = [item.get("chunk_id") if isinstance(item, Mapping) else None for item in result_chunks]
    _require(actual_production_ids == production_ids, "production_mismatch", f"Legacy production reconstruction mismatch: expected {production_ids}, got {actual_production_ids}")
    actual_ranks = [_integer(item.get("rank"), "result rank") for item in result_chunks]
    _require(actual_ranks == list(range(1, len(result_chunks) + 1)), "production_rank_mismatch", "result.chunks ranks must be contiguous from 1")
    backfill_count = len(production_ids) - len(strict_ids)
    _require(_integer(trace.get("arbiter_backfill_count"), "arbiter_backfill_count") == backfill_count, "backfill_count_mismatch", f"arbiter_backfill_count must equal {backfill_count}")
    entry_by_id = {chunk_id: entry for _, _, chunk_id, entry in entries}
    strict_set = set(strict_ids)
    for rank, chunk_id in enumerate(production_ids, 1):
        arbiter = entry_by_id[chunk_id].get("arbiter", {})
        _require(_integer(arbiter.get("final_rank"), "final_rank") == rank, "arbiter_final_rank_mismatch", f"final_rank mismatch for {chunk_id}")
        _require((arbiter.get("selected_by_arbiter") is True) == (chunk_id in strict_set), "arbiter_flag_mismatch", f"selected_by_arbiter mismatch for {chunk_id}")
        _require((arbiter.get("backfilled") is True) == (chunk_id not in strict_set), "arbiter_backfill_flag_mismatch", f"backfilled mismatch for {chunk_id}")

    raw_hit_counts: dict[str, Any] = {}
    for channel in CHANNELS:
        initial_count = sum(len(event["hits"]) for event in events if event["channel"] == channel and event["stage"] == "initial")
        expansion_count = sum(len(event["hits"]) for event in events if event["channel"] == channel and event["stage"] == "expansion")
        raw_hit_counts[channel] = {
            "initial": initial_count,
            "expansion": expansion_count,
            "total": initial_count + expansion_count,
        }
    raw_hit_counts["initial"] = sum(raw_hit_counts[channel]["initial"] for channel in CHANNELS)
    raw_hit_counts["expansion"] = sum(raw_hit_counts[channel]["expansion"] for channel in CHANNELS)
    raw_hit_counts["total"] = sum(raw_hit_counts[channel]["total"] for channel in CHANNELS)
    budget = trace.get("information_budget")
    if isinstance(budget, Mapping):
        channel_hits = budget.get("channel_hits", {})
        if isinstance(channel_hits, Mapping):
            for channel in CHANNELS:
                if channel in channel_hits:
                    _require(_integer(channel_hits[channel], f"{channel} budget hits") == raw_hit_counts[channel]["total"], "raw_hit_count_mismatch", f"information_budget {channel} hits differ from ledger")
        if "raw_index_hits" in budget:
            _require(_integer(budget["raw_index_hits"], "raw_index_hits") == raw_hit_counts["total"], "raw_hit_count_mismatch", "information_budget raw_index_hits differs from ledger")

    fallback_stages = trace.get("fallback_stages", [])
    _require(isinstance(fallback_stages, list) and all(isinstance(stage, str) for stage in fallback_stages), "invalid_fallback", "fallback_stages must be a list of strings")
    if "fallback_used" in result:
        _require(bool(result["fallback_used"]) == bool(fallback_stages), "fallback_flag_mismatch", "fallback_used differs from fallback_stages")
    fallback_provenance = []
    for stage in fallback_stages:
        if stage == "planner":
            state = "plan"
        elif stage.endswith("_round_1"):
            state = "R1"
        elif stage.endswith("_round_2"):
            state = "legacy_R2P"
        elif stage == "arbiter":
            state = "legacy_arbiter"
        else:
            state = "unknown"
        fallback_provenance.append({"stage": stage, "consumed_state": state, "source": "legacy_trace"})
    generation_usage = _generation_usage(trace, result, expansion_active)
    protocol_version = str(config.get("protocol_version", DEFAULT_PROTOCOL_VERSION))
    snapshot = {
        "protocol_version": protocol_version,
        "group_id": f"group_{group_index:04d}",
        "group_index": group_index,
        "query_group": query_group,
        "question_ids": question_ids,
        "query": query,
        "source_trace_sha256": source_trace_sha256,
        "plan": copy.deepcopy(dict(plan)),
        "reports": {"R1": copy.deepcopy(first), "legacy_R2P": copy.deepcopy(second)},
        "observations": {"initial": initial, "final": final_observations},
        "pools": {"B": base_pool, "E": expanded_pool, "W": None},
        "retrieval_events": events,
        "expansion_active": expansion_active,
        "raw_hit_counts": raw_hit_counts,
        "generation_usage": generation_usage,
        "legacy": {
            "strict_arbiter_ids": strict_ids,
            "production_ids": production_ids,
            "fallback_stages": list(fallback_stages),
            "backfill_count": backfill_count,
        },
        "fallback_provenance": fallback_provenance,
    }
    snapshot["snapshot_sha256"] = snapshot_hash(snapshot)
    return snapshot


def _control_values(records: Sequence[Mapping[str, Any]], output_k: int) -> dict[str, int]:
    active = 0
    fallback_groups = 0
    strict_short = 0
    rows = 0
    for record in records:
        questions = record.get("questions", [])
        if isinstance(questions, list):
            rows += len(questions)
        result = record.get("result", {})
        trace = result.get("trace", {}) if isinstance(result, Mapping) else {}
        if not isinstance(trace, Mapping):
            continue
        rounds = trace.get("rounds", [])
        has_queries = False
        if isinstance(rounds, list):
            for item in rounds:
                if not isinstance(item, Mapping) or "reports" not in item or item.get("round") not in (1, 1.0, "1"):
                    continue
                reports = item.get("reports", {})
                if isinstance(reports, Mapping):
                    has_queries = any(
                        isinstance(reports.get(channel), Mapping) and bool(reports[channel].get("search_queries"))
                        for channel in CHANNELS
                    )
                break
        active += int(has_queries)
        fallback = trace.get("fallback_stages", [])
        fallback_groups += int(isinstance(fallback, list) and bool(fallback))
        ledger = trace.get("candidate_ledger", {})
        if isinstance(ledger, Mapping):
            strict_count = sum(
                isinstance(entry, Mapping)
                and isinstance(entry.get("arbiter"), Mapping)
                and entry["arbiter"].get("selected_by_arbiter") is True
                for entry in ledger.values()
            )
            strict_short += int(strict_count < output_k)
    return {
        "groups": len(records),
        "rows": rows,
        "expansion_active": active,
        "expansion_inactive": len(records) - active,
        "fallback_groups": fallback_groups,
        "strict_short_groups": strict_short,
    }


def _commit_snapshots(output_dir: Path, snapshots: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "snapshots"
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Snapshot destination is not a directory: {destination}")
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".snapshots-staging-"))
    backup: Path | None = None
    try:
        for snapshot in snapshots:
            atomic_write_json(staging / f"{snapshot['group_id']}.json", snapshot)
        if destination.exists():
            backup = output_dir / f".snapshots-backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def audit_legacy_run(
    benchmark_dir: str | os.PathLike[str],
    legacy_run: str | os.PathLike[str],
    chunks_pkl: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> dict[str, Any]:
    config_value = _config_mapping(config)
    benchmark_path = Path(benchmark_dir)
    legacy_path = Path(legacy_run)
    if legacy_path.is_dir():
        legacy_path = legacy_path / "detailed.jsonl"
    chunks_path = Path(chunks_pkl)
    output_path = Path(output_dir)
    validation_path = output_path / "validation" / "reuse_audit.json"
    chunks, chunk_map, corpus_sha256 = _load_chunks(chunks_path)
    benchmark_hashes, global_errors = _benchmark_hashes(benchmark_path, config_value, chunks, corpus_sha256)
    raw_records, legacy_sha256 = _read_legacy_jsonl(legacy_path)
    retrieval_config = config_value.get("retrieval", {})
    output_k = int(retrieval_config.get("output_k", 10)) if isinstance(retrieval_config, Mapping) else 10
    actual_controls = _control_values([record for record, _ in raw_records], output_k)
    expected_controls = _expected_controls(config_value)
    for key, expected in expected_controls.items():
        actual = actual_controls[key]
        if actual != expected:
            global_errors.append({
                "code": "control_mismatch",
                "message": f"{key}: expected {expected}, got {actual}",
            })
    snapshots: list[dict[str, Any]] = []
    group_audits: list[dict[str, Any]] = []
    all_question_ids: set[str] = set()
    for group_index, (record, source_hash) in enumerate(raw_records):
        group_audit: dict[str, Any] = {
            "group_id": f"group_{group_index:04d}",
            "group_index": group_index,
            "query_group": record.get("query_group"),
            "source_trace_sha256": source_hash,
            "errors": [],
        }
        try:
            snapshot = _build_snapshot(record, source_hash, group_index, chunk_map, config_value)
            duplicates = all_question_ids.intersection(snapshot["question_ids"])
            if duplicates:
                raise _GroupError("duplicate_question_id", f"Question IDs occur in multiple legacy groups: {sorted(duplicates)}")
            all_question_ids.update(snapshot["question_ids"])
            snapshots.append(snapshot)
            group_audit.update({
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "expansion_active": snapshot["expansion_active"],
                "raw_unique_candidates": len(snapshot["pools"]["E"]["chunk_ids"]),
                "strict_size": len(snapshot["legacy"]["strict_arbiter_ids"]),
                "backfill_count": snapshot["legacy"]["backfill_count"],
                "fallback_stages": snapshot["legacy"]["fallback_stages"],
            })
        except _GroupError as exc:
            group_audit["errors"].append({"code": exc.code, "message": str(exc)})
        except Exception as exc:
            group_audit["errors"].append({
                "code": "group_reconstruction_error",
                "message": f"{type(exc).__name__}: {exc}",
            })
        group_audits.append(group_audit)
    group_error_count = sum(bool(group["errors"]) for group in group_audits)
    error_count = len(global_errors) + sum(len(group["errors"]) for group in group_audits)
    valid = error_count == 0
    summary = {
        **actual_controls,
        "valid_groups": len(snapshots),
        "group_errors": group_error_count,
        "global_errors": len(global_errors),
        "error_count": error_count,
        "snapshots_written": 0,
    }
    report = {
        "protocol_version": str(config_value.get("protocol_version", DEFAULT_PROTOCOL_VERSION)),
        "status": "ok" if valid else "error",
        "valid": valid,
        "legacy_run_sha256": legacy_sha256,
        "chunks_pkl_sha256": file_sha256(chunks_path),
        "benchmark_hashes": benchmark_hashes,
        "controls": {"expected": expected_controls, "actual": actual_controls},
        "summary": summary,
        "errors": global_errors,
        "groups": group_audits,
    }
    if not valid:
        atomic_write_json(validation_path, report)
        raise LegacyReuseAuditError(report, validation_path)
    _commit_snapshots(output_path, snapshots)
    summary["snapshots_written"] = len(snapshots)
    atomic_write_json(validation_path, report)
    return report


def load_snapshots(experiment_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    snapshot_dir = Path(experiment_dir) / "snapshots"
    paths = sorted(snapshot_dir.glob("group_*.json"))
    if not paths:
        raise ValueError(f"No snapshots found in {snapshot_dir}")
    snapshots: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read snapshot {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Snapshot must be an object: {path}")
        stored_hash = value.get("snapshot_sha256")
        computed_hash = snapshot_hash(value)
        if not isinstance(stored_hash, str) or stored_hash != computed_hash:
            raise ValueError(f"Snapshot hash mismatch: {path}")
        expected_id = f"group_{index:04d}"
        if path.name != f"{expected_id}.json" or value.get("group_id") != expected_id or value.get("group_index") != index:
            raise ValueError(f"Snapshot order/group identity mismatch: {path}")
        snapshots.append(value)
    return snapshots


def _questions_mapping(questions: Any) -> dict[str, str]:
    if isinstance(questions, (str, os.PathLike)):
        rows = [json.loads(line) for line in Path(questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif isinstance(questions, Mapping):
        result = {}
        for question_id, value in questions.items():
            result[str(question_id)] = str(value.get("query")) if isinstance(value, Mapping) else str(value)
        return result
    else:
        rows = list(questions)
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Question rows must be mappings")
        question_id = str(row["question_id"])
        if question_id in result:
            raise ValueError(f"Duplicate question_id: {question_id}")
        result[question_id] = str(row["query"])
    return result


def _qrels_mapping(qrels: Any) -> dict[str, dict[str, int]]:
    if isinstance(qrels, (str, os.PathLike)):
        rows = [json.loads(line) for line in Path(qrels).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif isinstance(qrels, Mapping):
        result: dict[str, dict[str, int]] = {}
        for question_id, values in qrels.items():
            if isinstance(values, Mapping):
                relevant = {str(chunk_id): int(relevance) for chunk_id, relevance in values.items() if int(relevance) > 0}
            else:
                relevant = {str(chunk_id): 1 for chunk_id in values}
            if relevant:
                result[str(question_id)] = relevant
        return result
    else:
        rows = list(qrels)
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Qrel rows must be mappings")
        relevance = int(row["relevance"])
        if relevance <= 0:
            continue
        question_id = str(row["question_id"])
        chunk_id = str(row["chunk_id"])
        values = result.setdefault(question_id, {})
        values[chunk_id] = max(values.get(chunk_id, 0), relevance)
    return result


def _select_groups(
    snapshots: Sequence[Mapping[str, Any]],
    questions: Any,
    qrels: Any,
    active_count: int,
    inactive_count: int,
    salt: str,
    required_multi: int,
) -> list[dict[str, Any]]:
    if active_count < 0 or inactive_count < 0 or required_multi < 0:
        raise ValueError("Selection counts must be non-negative")
    _questions_mapping(questions)
    qrels_map = _qrels_mapping(qrels)
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    candidates: list[tuple[dict[str, Any], bool, int]] = []
    for value in snapshots:
        snapshot = dict(value)
        group_id = str(snapshot.get("group_id"))
        query_group = snapshot.get("query_group")
        if not isinstance(query_group, str):
            raise ValueError(f"Snapshot {group_id} has no query_group")
        if group_id in seen_ids or query_group in seen_groups:
            raise ValueError(f"Duplicate snapshot group: {group_id} / {query_group}")
        seen_ids.add(group_id)
        seen_groups.add(query_group)
        question_ids = snapshot.get("question_ids")
        if not isinstance(question_ids, list) or not question_ids:
            raise ValueError(f"Snapshot {group_id} has no question_ids")
        relevant: set[str] = set()
        for question_id in question_ids:
            relevant.update(qrels_map.get(str(question_id), {}))
        if not relevant:
            continue
        candidates.append((snapshot, bool(snapshot.get("expansion_active")), len(relevant)))
    def order(item: tuple[dict[str, Any], bool, int]) -> tuple[str, str, int]:
        snapshot = item[0]
        digest = hashlib.sha256((str(snapshot["query_group"]) + salt).encode("utf-8")).hexdigest()
        return digest, str(snapshot["query_group"]), int(snapshot.get("group_index", 0))
    active = sorted((item for item in candidates if item[1]), key=order)
    inactive = sorted((item for item in candidates if not item[1]), key=order)
    if len(active) < active_count or len(inactive) < inactive_count:
        raise ValueError(f"Insufficient answerable groups: active={len(active)}/{active_count}, inactive={len(inactive)}/{inactive_count}")
    selected_active = active[:active_count]
    if sum(item[2] >= 2 for item in selected_active) < required_multi:
        for candidate in active[active_count:]:
            if sum(item[2] >= 2 for item in selected_active) >= required_multi:
                break
            if candidate[2] < 2:
                continue
            for index in range(len(selected_active) - 1, -1, -1):
                if selected_active[index][2] < 2:
                    selected_active[index] = candidate
                    break
    if sum(item[2] >= 2 for item in selected_active) < required_multi:
        raise ValueError(f"At least {required_multi} active multi-relevant groups are required")
    selected_inactive = inactive[:inactive_count]
    return [item[0] for item in selected_active + selected_inactive]


def select_smoke_groups(
    snapshots: Sequence[Mapping[str, Any]],
    questions: Any,
    qrels: Any,
    config: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> list[dict[str, Any]]:
    config_value = _config_mapping(config)
    smoke = config_value.get("smoke", {})
    if not isinstance(smoke, Mapping):
        smoke = {}
    active_count = int(smoke.get("expansion_active", 8))
    inactive_count = int(smoke.get("expansion_inactive", 4))
    required_multi = int(smoke.get("min_multi_relevant", 2))
    return _select_groups(snapshots, questions, qrels, active_count, inactive_count, ":20261127", required_multi)


def select_stability_groups(
    snapshots: Sequence[Mapping[str, Any]],
    questions: Any,
    qrels: Any,
    config: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> list[dict[str, Any]]:
    config_value = _config_mapping(config)
    stability = config_value.get("stability", {})
    if not isinstance(stability, Mapping):
        stability = {}
    active_count = int(stability.get("expansion_active", 20))
    inactive_count = int(stability.get("expansion_inactive", 20))
    return _select_groups(snapshots, questions, qrels, active_count, inactive_count, ":stability:20261127", 0)


def run_audit(args: Any) -> dict[str, Any]:
    output = getattr(args, "output", None)
    if output is None:
        output = args.output_dir
    return audit_legacy_run(
        benchmark_dir=args.benchmark_dir,
        legacy_run=args.legacy_run,
        chunks_pkl=args.chunks_pkl,
        output_dir=output,
        config=args.config,
    )


__all__ = [
    "LegacyReuseAuditError",
    "audit_legacy_run",
    "load_snapshots",
    "run_audit",
    "select_smoke_groups",
    "select_stability_groups",
]
