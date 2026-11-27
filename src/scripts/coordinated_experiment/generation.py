import asyncio
import copy
import json
import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from scripts.coordinated_experiment.pools import build_pool_record
from scripts.coordinated_experiment.schema import canonical_sha256, text_sha256
from scripts.coordinated_experiment.selectors import (
    ReplayLLMCaller,
    build_worker_prompts,
    make_worker_validator,
    planner_validator,
    prompt_sha256,
    reconstruct_pool,
)

CHANNELS = ("dense", "bm25")

PLANNER_SYSTEM_PROMPT = (
    "Ты Planner multi-agent поиска по нормативным документам РФ. "
    "Разбей вопрос на проверяемые юридические аспекты: объект, орган, действие, условия, даты, исключения и связанные нормы. "
    "Dense должен искать смысловые формулировки, BM25 — точные названия, номера и редкие термины. "
    "Не выдумывай факты. Верни только JSON: "
    '{"aspects":[{"id":"A1","need":"..."}],"assignments":{"dense":["..."],"bm25":["..."]}}.'
)

SINGLE_EXPANSION_SYSTEM_PROMPT = (
    "Ты единый агент расширения поискового запроса по нормативным документам РФ. "
    "Проанализируй вопрос и обе размеченные начальные выдачи. "
    "Предложи не более двух новых подзапросов для dense и не более двух для BM25: "
    "dense должен покрывать смысловые переформулировки и неявные условия, BM25 — точные названия, номера, даты и редкие термины. "
    "Не повторяй исходный вопрос и не дублируй подзапросы после нормализации. "
    "Верни только JSON: {\"dense_queries\":[\"...\"],\"bm25_queries\":[\"...\"]}. "
    "Никаких пояснений, оценок чанков, плана и reports."
)

ReplayCall = Callable[..., Any]


def normalize_query_text(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", value.casefold()).strip()


def generic_plan(query: str) -> dict[str, Any]:
    return {
        "aspects": [{"id": "A1", "need": query}],
        "assignments": {"dense": [], "bm25": []},
    }


def build_single_expansion_prompts(
    query: str,
    observations: Mapping[str, Sequence[str]],
    pool_or_record: Any,
    chunk_map: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    pool = reconstruct_pool(pool_or_record, chunk_map) if isinstance(pool_or_record, Mapping) else pool_or_record
    human = (
        f"Вопрос: {query}\n"
        f"DENSE initial observations:\n{pool.format(list(observations['dense']))}\n"
        f"BM25 initial observations:\n{pool.format(list(observations['bm25']))}"
    )
    return SINGLE_EXPANSION_SYSTEM_PROMPT, human


def single_expansion_validator(
    value: Mapping[str, Any],
    query: str,
    query_cache: MutableMapping[str, set[str]] | None = None,
    max_queries: int = 2,
) -> tuple[dict[str, list[str]] | None, str, dict[str, Any]]:
    cache = query_cache or {channel: {normalize_query_text(query)} for channel in CHANNELS}
    result: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {"dense": [], "bm25": []}
    for channel in CHANNELS:
        raw = value.get(f"{channel}_queries", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, Sequence):
            return None, f"{channel}_queries должен быть списком", {"dropped": dropped}
        selected: list[str] = []
        for item in raw[:max_queries]:
            query_value = str(item).strip()
            normalized = normalize_query_text(query_value)
            if not query_value or normalized in cache[channel]:
                if query_value:
                    dropped[channel].append(query_value)
                continue
            cache[channel].add(normalized)
            selected.append(query_value)
        result[f"{channel}_queries"] = selected
    return result, "", {"dropped": dropped}


def empty_worker_report() -> dict[str, Any]:
    return {"assessments": [], "search_queries": [], "coverage": {"covered": [], "missing": []}}


def _usage_from_metas(metas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stage_usage: dict[str, Any] = {}
    for stage, meta in metas.items():
        stage_usage[stage] = {
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "total_tokens": meta.get("total_tokens"),
            "token_measurement_status": meta.get("token_measurement_status", "unavailable"),
            "attempts": meta.get("attempts", 0),
            "repair_used": bool(meta.get("repair_used", False)),
        }
    usages = list(stage_usage.values())
    result: dict[str, Any] = {"stages": stage_usage}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [usage[key] for usage in usages]
        result[key] = sum(values) if values and all(value is not None for value in values) else None
    result["token_measurement_status"] = (
        "provider_usage"
        if usages and all(result[key] is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
        else "unavailable"
    )
    return result


async def _default_replay_call(
    llm: Any,
    timeout_seconds: float,
    role: str,
    system: str,
    human: str,
    required: tuple[str, ...],
    validator: Callable[[Mapping[str, Any]], tuple[Any, str, Any]],
) -> tuple[Any, dict[str, Any], str | None]:
    value, meta = await ReplayLLMCaller(llm, timeout_seconds=timeout_seconds).call(
        role, system, human, required, validator,
    )
    return value, meta, None


async def _invoke(
    replay_call: ReplayCall | None,
    llm: Any,
    timeout_seconds: float,
    role: str,
    system: str,
    human: str,
    required: tuple[str, ...],
    validator: Callable[[Mapping[str, Any]], tuple[Any, str, Any]],
) -> tuple[Any, dict[str, Any], str | None]:
    caller = replay_call or _default_replay_call
    return await caller(llm, timeout_seconds, role, system, human, required, validator)


async def planner_call(
    llm: Any,
    query: str,
    timeout_seconds: float = 180,
    replay_call: ReplayCall | None = None,
) -> dict[str, Any]:
    system = PLANNER_SYSTEM_PROMPT
    human = f"Вопрос: {query}"
    plan, meta, raw_path = await _invoke(
        replay_call, llm, timeout_seconds, "planner", system, human,
        ("aspects", "assignments"), planner_validator,
    )
    return {
        **meta,
        "plan": plan,
        "system_prompt": system,
        "human_prompt": human,
        "prompt_sha256": prompt_sha256(system, human),
        "raw_model_response_path": raw_path,
    }


async def original_worker_call(
    llm: Any,
    channel: str,
    query: str,
    plan: Mapping[str, Any],
    observations: Sequence[str],
    pool_record: Mapping[str, Any],
    chunk_map: Mapping[str, Any],
    query_cache: MutableMapping[str, set[str]],
    timeout_seconds: float = 180,
    replay_call: ReplayCall | None = None,
) -> dict[str, Any]:
    pool = reconstruct_pool(pool_record, chunk_map)
    system, human = build_worker_prompts(
        channel, query, plan, 1, observations, pool, None, clean=False,
    )
    validator = make_worker_validator(channel, 1, observations, pool, query_cache)
    report, meta, raw_path = await _invoke(
        replay_call, llm, timeout_seconds, f"{channel}_round_1", system, human,
        ("assessments", "search_queries", "coverage"), validator,
    )
    if isinstance(report, Mapping):
        for item in report.get("search_queries", []):
            query_cache[channel].add(normalize_query_text(str(item)))
    return {
        **meta,
        "report": report,
        "system_prompt": system,
        "human_prompt": human,
        "prompt_sha256": prompt_sha256(system, human),
        "pool_sha256": pool_record["pool_sha256"],
        "raw_model_response_path": raw_path,
    }


def _hit_parts(hit: Any) -> tuple[Any, str, int, float | None]:
    if isinstance(hit, Mapping):
        chunk = hit.get("chunk")
        if chunk is None:
            chunk = hit
        chunk_id = str(hit.get("chunk_id", chunk.get("id", chunk.get("chunk_id"))))
        rank = int(hit.get("rank", 0))
        score = hit.get("score")
        return chunk, chunk_id, rank, float(score) if score is not None else None
    chunk = hit.chunk
    chunk_id = str(getattr(chunk, "id", getattr(chunk, "chunk_id", None)))
    return chunk, chunk_id, int(hit.rank), float(hit.score) if hit.score is not None else None


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk["text"])
    return str(chunk.text)


def _initial_events(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(event) for event in snapshot.get("retrieval_events", [])
        if event.get("stage") == "initial"
    ]


def _raw_hit_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for channel in CHANNELS:
        initial = sum(
            len(event.get("hits", [])) for event in events
            if event.get("channel") == channel and event.get("stage") == "initial"
        )
        expansion = sum(
            len(event.get("hits", [])) for event in events
            if event.get("channel") == channel and event.get("stage") == "expansion"
        )
        result[channel] = {"initial": initial, "expansion": expansion, "total": initial + expansion}
    result["initial"] = sum(result[channel]["initial"] for channel in CHANNELS)
    result["expansion"] = sum(result[channel]["expansion"] for channel in CHANNELS)
    result["total"] = sum(result[channel]["total"] for channel in CHANNELS)
    return result


async def expand_pool_from_queries(
    engine: Any,
    snapshot: Mapping[str, Any],
    queries: Mapping[str, Sequence[str]],
    chunk_map: Mapping[str, Any],
    quota_per_channel: int = 10,
) -> dict[str, Any]:
    from retrieval.contracts import SearchBudget

    budget = SearchBudget()
    budgeted_engine = engine.with_budget(budget)
    base = snapshot["pools"]["B"]
    base_ids = list(base["chunk_ids"])
    normalized_queries = {
        channel: [str(query).strip() for query in queries.get(channel, []) if str(query).strip()]
        for channel in CHANNELS
    }

    async def search(channel: str) -> tuple[str, Mapping[str, Sequence[Any]]]:
        if not normalized_queries[channel]:
            return channel, {}
        results = await asyncio.to_thread(
            budgeted_engine.search_many, channel, normalized_queries[channel], quota_per_channel,
        )
        return channel, results

    searched = await asyncio.gather(search("dense"), search("bm25"))
    results_by_channel = dict(searched)
    initial_events = _initial_events(snapshot)
    events = copy.deepcopy(initial_events)
    ids = list(base_ids)
    known = set(ids)
    observations = {
        channel: list(snapshot["observations"]["initial"][channel]) for channel in CHANNELS
    }
    expansion: dict[str, Any] = {}
    for channel in CHANNELS:
        channel_results = results_by_channel[channel]
        raw_new: list[str] = []
        observed_new: list[str] = []
        for query_index, query in enumerate(normalized_queries[channel], 1):
            hits = list(channel_results.get(query, []))
            event_hits: list[dict[str, Any]] = []
            event_text_hashes: dict[str, str] = {}
            seen: set[str] = set()
            for hit_index, hit in enumerate(hits, 1):
                chunk, chunk_id, raw_rank, score = _hit_parts(hit)
                rank = raw_rank or hit_index
                if chunk_id in seen:
                    raise ValueError(f"Duplicate chunk {chunk_id} in generated event {channel}_expansion_{query_index}")
                seen.add(chunk_id)
                if chunk_id not in chunk_map:
                    raise KeyError(f"Generated retrieval returned missing chunk: {chunk_id}")
                text_hash = text_sha256(_chunk_text(chunk_map[chunk_id]))
                if text_sha256(_chunk_text(chunk)) != text_hash:
                    raise ValueError(f"Generated retrieval text mismatch: {chunk_id}")
                event_hits.append({
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": score,
                    "text_sha256": text_hash,
                })
                event_text_hashes[chunk_id] = text_hash
                if chunk_id not in known:
                    known.add(chunk_id)
                    ids.append(chunk_id)
                    raw_new.append(chunk_id)
                if chunk_id not in observations[channel] and len(observations[channel]) < 15:
                    observations[channel].append(chunk_id)
                    observed_new.append(chunk_id)
            events.append({
                "event_id": f"{channel}_expansion_{query_index}",
                "channel": channel,
                "stage": "expansion",
                "query": query,
                "call_index": 1,
                "hits": event_hits,
                "text_sha256": event_text_hashes,
            })
        expansion[channel] = {
            "queries": list(normalized_queries[channel]),
            "raw_new_ids": raw_new,
            "new_observed_ids": observed_new,
        }
    pool_record = build_pool_record(ids, chunk_map)
    if ids[: len(base_ids)] != base_ids:
        raise ValueError("Generated pool does not preserve B insertion prefix")
    expansion_active = any(normalized_queries.values())
    return {
        "pool_record": pool_record,
        "retrieval_events": events,
        "observations": {"initial": copy.deepcopy(snapshot["observations"]["initial"]), "final": observations},
        "expansion": expansion,
        "expansion_active": expansion_active,
        "raw_hit_counts": _raw_hit_counts(events),
        "search_budget": budget.snapshot(),
        "reused_from": "B" if not expansion_active else None,
    }


def _artifact_base(snapshot: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "group_id": snapshot["group_id"],
        "query_group": snapshot["query_group"],
        "query": snapshot["query"],
        "question_ids": list(snapshot["question_ids"]),
        "base_pool_sha256": snapshot["pools"]["B"]["pool_sha256"],
        "input_snapshot_sha256": snapshot["snapshot_sha256"],
    }


async def generate_single(
    engine: Any,
    snapshot: Mapping[str, Any],
    chunk_map: Mapping[str, Any],
    llm: Any,
    timeout_seconds: float = 180,
    quota_per_channel: int = 10,
    max_queries: int = 2,
    replay_call: ReplayCall | None = None,
) -> dict[str, Any]:
    base = snapshot["pools"]["B"]
    observations = snapshot["observations"]["initial"]
    system, human = build_single_expansion_prompts(snapshot["query"], observations, base, chunk_map)
    cache = {channel: {normalize_query_text(snapshot["query"])} for channel in CHANNELS}
    value, meta, raw_path = await _invoke(
        replay_call, llm, timeout_seconds, "single_expansion", system, human,
        ("dense_queries", "bm25_queries"),
        lambda item: single_expansion_validator(item, snapshot["query"], cache, max_queries),
    )
    if value is None:
        value = {"dense_queries": [], "bm25_queries": []}
        fallback_stages = ["single_expansion"]
    else:
        fallback_stages = []
    queries = {channel: list(value[f"{channel}_queries"]) for channel in CHANNELS}
    expanded = await expand_pool_from_queries(
        engine, snapshot, queries, chunk_map, quota_per_channel=quota_per_channel,
    )
    notes = meta.get("notes") if isinstance(meta.get("notes"), Mapping) else {}
    generation = {
        **_artifact_base(snapshot, "S"),
        **expanded,
        "queries": queries,
        "planner": None,
        "reports": None,
        "generation_usage": _usage_from_metas({"single_expansion": meta}),
        "generation_meta": {
            "single_expansion": {
                **meta,
                "prompt_sha256": prompt_sha256(system, human),
                "raw_model_response_path": raw_path,
            },
        },
        "raw_model_response_paths": [raw_path] if raw_path else [],
        "fallback_stages": fallback_stages,
        "fallback_provenance": [
            {"stage": stage, "consumed_state": "queries", "source": "new_generation"}
            for stage in fallback_stages
        ],
        "dropped_queries": notes.get("dropped", {}),
    }
    generation["generation_sha256"] = canonical_sha256(generation)
    return generation


async def _generate_worker_pool(
    engine: Any,
    snapshot: Mapping[str, Any],
    chunk_map: Mapping[str, Any],
    dense_llm: Any,
    bm25_llm: Any,
    plan: Mapping[str, Any],
    planner_meta: Mapping[str, Any] | None,
    kind: str,
    timeout_seconds: float,
    quota_per_channel: int,
    replay_call: ReplayCall | None,
) -> dict[str, Any]:
    base = snapshot["pools"]["B"]
    observations = snapshot["observations"]["initial"]
    query_cache = {channel: {normalize_query_text(snapshot["query"])} for channel in CHANNELS}

    async def run(channel: str, llm: Any) -> tuple[str, dict[str, Any]]:
        result = await original_worker_call(
            llm, channel, snapshot["query"], plan, observations[channel], base, chunk_map,
            query_cache, timeout_seconds=timeout_seconds, replay_call=replay_call,
        )
        return channel, result

    worker_results = await asyncio.gather(run("dense", dense_llm), run("bm25", bm25_llm))
    metas: dict[str, Mapping[str, Any]] = {}
    if planner_meta is not None:
        metas["planner"] = planner_meta
    reports: dict[str, Any] = {}
    generation_meta: dict[str, Any] = {}
    fallback_stages: list[str] = []
    if planner_meta is not None and planner_meta.get("plan") is None:
        fallback_stages.append("planner")
    for channel, result in worker_results:
        metas[f"{channel}_round_1"] = result
        generation_meta[f"{channel}_round_1"] = result
        report = result.get("report")
        if report is None:
            fallback_stages.append(f"{channel}_round_1")
            report = empty_worker_report()
        reports[channel] = report
    if planner_meta is not None:
        generation_meta["planner"] = planner_meta
    queries = {channel: list(reports[channel]["search_queries"]) for channel in CHANNELS}
    expanded = await expand_pool_from_queries(
        engine, snapshot, queries, chunk_map, quota_per_channel=quota_per_channel,
    )
    artifact = {
        **_artifact_base(snapshot, kind),
        **expanded,
        "plan": copy.deepcopy(dict(plan)),
        "queries": queries,
        "reports": {"R1": reports},
        "generation_usage": _usage_from_metas(metas),
        "generation_meta": generation_meta,
        "raw_model_response_paths": [
            result["raw_model_response_path"] for result in generation_meta.values()
            if result.get("raw_model_response_path")
        ],
        "fallback_stages": fallback_stages,
        "fallback_provenance": [
            {
                "stage": stage,
                "consumed_state": "plan" if stage == "planner" else "R1",
                "source": "new_generation",
            }
            for stage in fallback_stages
        ],
    }
    artifact["generation_sha256"] = canonical_sha256(artifact)
    return artifact


async def generate_no_planner(
    engine: Any,
    snapshot: Mapping[str, Any],
    chunk_map: Mapping[str, Any],
    dense_llm: Any,
    bm25_llm: Any,
    timeout_seconds: float = 180,
    quota_per_channel: int = 10,
    replay_call: ReplayCall | None = None,
) -> dict[str, Any]:
    return await _generate_worker_pool(
        engine, snapshot, chunk_map, dense_llm, bm25_llm,
        generic_plan(snapshot["query"]), None, "NP", timeout_seconds,
        quota_per_channel, replay_call,
    )


async def generate_coordinated(
    engine: Any,
    snapshot: Mapping[str, Any],
    chunk_map: Mapping[str, Any],
    dense_llm: Any,
    bm25_llm: Any,
    planner_llm: Any,
    timeout_seconds: float = 180,
    quota_per_channel: int = 10,
    replay_call: ReplayCall | None = None,
) -> dict[str, Any]:
    planner = await planner_call(
        planner_llm, snapshot["query"], timeout_seconds=timeout_seconds, replay_call=replay_call,
    )
    plan = planner.get("plan") or generic_plan(snapshot["query"])
    return await _generate_worker_pool(
        engine, snapshot, chunk_map, dense_llm, bm25_llm,
        plan, planner, "E", timeout_seconds, quota_per_channel, replay_call,
    )


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "SINGLE_EXPANSION_SYSTEM_PROMPT",
    "build_single_expansion_prompts",
    "empty_worker_report",
    "expand_pool_from_queries",
    "generate_coordinated",
    "generate_no_planner",
    "generate_single",
    "generic_plan",
    "normalize_query_text",
    "original_worker_call",
    "planner_call",
    "single_expansion_validator",
]
