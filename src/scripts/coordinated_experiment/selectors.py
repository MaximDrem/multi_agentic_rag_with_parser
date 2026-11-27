import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from common.models import Chunk
from scripts.coordinated_experiment.pools import assert_same_pool
from scripts.coordinated_experiment.schema import canonical_sha256, text_sha256

AGENT_TIMEOUT_SECONDS = 180
MAX_OUTPUT_CHUNKS = 10
PEER_REPORT_INSTRUCTION = "Используй report другого агента, чтобы не дублировать его работу. "

ARBITER_SYSTEM_PROMPT = (
    "Ты Arbiter кооперативного multi-agent retrieval по НПА РФ. "
    "Получаешь общий candidate pool, план аспектов и reports двух агентов. "
    "Самостоятельно оцени юридическую применимость, фактические условия, исключения и полноту покрытия. "
    "Используй worker relevance/aspect как рекомендации, но проверяй текст. "
    "Сформируй один глобальный ranking: сначала прямые ответы, затем необходимые дополнительные доказательства, "
    "покрой все аспекты без дубликатов по смыслу. Верни ровно 10 элементов при наличии 10 candidates. "
    "Только JSON: {\"selected_chunks\":[{\"alias\":\"C001\",\"relevance\":2,\"aspect\":\"A1\"}]}. "
    "Никаких рассуждений, новых aliases или повторов."
)

Validator = Callable[[Mapping[str, Any]], tuple[Any, str, Any]]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", value.casefold()).strip()


def _json_object(text: str, required: tuple[str, ...]) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(key in value for key in required):
            return value
    return None


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_usage(response: Any) -> dict[str, int | None]:
    usage_metadata = getattr(response, "usage_metadata", None)
    response_metadata = getattr(response, "response_metadata", None)
    if not isinstance(usage_metadata, Mapping):
        usage_metadata = {}
    if not isinstance(response_metadata, Mapping):
        response_metadata = {}
    candidates = [
        usage_metadata,
        response_metadata.get("token_usage", {}),
        response_metadata.get("usage", {}),
        response_metadata,
    ]
    mappings = [value for value in candidates if isinstance(value, Mapping)]

    def first(keys: tuple[str, ...]) -> int | None:
        for mapping in mappings:
            for key in keys:
                if key in mapping and mapping[key] is not None:
                    return _integer(mapping[key])
        return None

    prompt = first(("input_tokens", "prompt_tokens"))
    completion = first(("output_tokens", "completion_tokens"))
    total = first(("total_tokens",))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _aggregate_usage(usages: Sequence[Mapping[str, int | None]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [usage[key] for usage in usages]
        result[key] = sum(values) if values and all(value is not None for value in values) else None
    complete = bool(usages) and all(result[key] is not None for key in result)
    result["token_measurement_status"] = "provider_usage" if complete else "unavailable"
    return result


def prompt_sha256(system: str, human: str) -> str:
    return canonical_sha256({"system": system, "human": human})


class ReplayLLMCaller:
    def __init__(self, llm: Any, timeout_seconds: float = AGENT_TIMEOUT_SECONDS):
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    async def call(
        self,
        role: str,
        system: str,
        human: str,
        required: tuple[str, ...],
        validator: Validator,
    ) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter()
        messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=human)]
        raw_responses: list[dict[str, Any]] = []
        usages: list[dict[str, int | None]] = []
        errors: list[dict[str, Any]] = []
        notes: Any = []
        repair_used = False
        repair_validation_error: str | None = None
        attempts = 0
        final_error: str | None = None
        normalized: Any = None
        for attempt in range(2):
            attempts += 1
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                final_error = f"{type(exc).__name__}: {exc}"
                errors.append({"attempt": attempt + 1, "error": final_error})
                normalized = None
                break
            content = str(getattr(response, "content", ""))
            usage = _response_usage(response)
            usages.append(usage)
            raw_responses.append({
                "attempt": attempt + 1,
                "content": content,
                "usage": usage,
            })
            value = _json_object(content, required)
            validation_error = "Ответ не содержит требуемый JSON"
            if value is not None:
                try:
                    normalized, validation_error, notes = validator(value)
                except Exception as exc:
                    normalized = None
                    validation_error = f"Ошибка валидации: {type(exc).__name__}: {exc}"
                    notes = []
                if not validation_error:
                    final_error = None
                    break
            normalized = None
            final_error = validation_error
            errors.append({"attempt": attempt + 1, "error": validation_error})
            if attempt == 0:
                repair_used = True
                repair_validation_error = validation_error
                messages = messages + [
                    response,
                    HumanMessage(content=(
                        f"Ошибка: {validation_error}. Верни только исправленный JSON с теми же полями. "
                        "Используй только существующие aliases, без пояснений и без повторов."
                    )),
                ]
        meta = {
            "role": role,
            "attempts": attempts,
            "raw_responses": raw_responses,
            "repair_used": repair_used,
            "repair_metadata": {
                "attempt": 2 if repair_used else None,
                "validation_error": repair_validation_error,
            },
            "error": final_error,
            "errors": errors,
            "notes": notes,
            "selector_latency_ms": int((time.perf_counter() - started) * 1000),
            **_aggregate_usage(usages),
        }
        meta["tokens"] = meta["total_tokens"]
        return normalized, meta


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk["text"])
    return str(chunk.text)


def _coerce_chunk(chunk_id: str, chunk: Any) -> Any:
    if not isinstance(chunk, Mapping):
        return chunk
    value = dict(chunk)
    value.setdefault("id", chunk_id)
    value.setdefault("doc_path", "")
    value.setdefault("title", "")
    value.setdefault("text", "")
    value.setdefault("token_count", 0)
    return Chunk(**value)


def reconstruct_pool(pool_record: Mapping[str, Any], chunk_map: Mapping[str, Any]) -> Any:
    assert_same_pool(pool_record)
    from retrieval.coordinated import SharedCandidatePool

    pool = SharedCandidatePool()
    aliases = pool_record.get("aliases") or [
        f"C{index:03d}" for index in range(1, len(pool_record["chunk_ids"]) + 1)
    ]
    for chunk_id, alias in zip(pool_record["chunk_ids"], aliases, strict=True):
        if chunk_id not in chunk_map:
            raise KeyError(f"Missing chunk: {chunk_id}")
        chunk = chunk_map[chunk_id]
        if text_sha256(_chunk_text(chunk)) != pool_record["text_sha256"][chunk_id]:
            raise ValueError(f"Chunk text hash mismatch: {chunk_id}")
        chunk = _coerce_chunk(chunk_id, chunk)
        pool.ids.append(chunk_id)
        pool.aliases[chunk_id] = alias
        pool.chunks[chunk_id] = chunk
        pool.ledger[chunk_id] = {
            "alias": alias,
            "chunk_id": chunk_id,
            "retrievals": [],
            "observed_by": [],
            "worker_assessments": {},
            "arbiter": {},
        }
    pool.pool_sha256 = pool_record["pool_sha256"]
    pool.pool_record = dict(pool_record)
    return pool


def _resolve_pool(pool_or_record: Any, chunk_map: Mapping[str, Any] | None) -> Any:
    if isinstance(pool_or_record, Mapping):
        if chunk_map is None:
            raise ValueError("chunk_map is required with pool_record")
        return reconstruct_pool(pool_or_record, chunk_map)
    return pool_or_record


def build_arbiter_prompts(
    query: str,
    plan: Mapping[str, Any],
    reports: Mapping[str, Any] | None,
    pool: Any,
) -> tuple[str, str]:
    human = (
        f"Вопрос: {query}\nПлан: {json.dumps(plan, ensure_ascii=False)}\n"
        f"Worker reports: {json.dumps(reports, ensure_ascii=False)}\n"
        f"Общий pool:\n{pool.format(pool.ids)}"
    )
    return ARBITER_SYSTEM_PROMPT, human


def build_worker_prompts(
    channel: str,
    query: str,
    plan: Mapping[str, Any],
    round_index: int,
    observations: Sequence[str],
    pool: Any,
    other_report: Mapping[str, Any] | None,
    clean: bool = False,
) -> tuple[str, str]:
    if channel not in {"dense", "bm25"}:
        raise ValueError("channel must be 'dense' or 'bm25'")
    dense = channel == "dense"
    role = (
        "смысловые синонимы, определения и неявные условия" if dense
        else "точные названия документов, номера, даты, аббревиатуры и редкие термины"
    )
    system = (
        f"Ты {channel}-agent в кооперативном retrieval. Твоя специализация: {role}. "
        "Оценивай только существующие aliases и указывай, какой аспект покрывает каждый чанк. "
        "Запрашивай поиск только для непокрытых аспектов; обычно один точный подзапрос. "
        f"{PEER_REPORT_INSTRUCTION}"
        "Верни только JSON: {\"assessments\":[{\"alias\":\"C001\",\"relevance\":2,"
        "\"aspect\":\"A1\",\"evidence\":\"короткая опорная фраза\"}],"
        "\"search_queries\":[\"...\"],\"coverage\":{\"covered\":[\"A1\"],\"missing\":[\"A2\"]}}. "
        "Во втором раунде search_queries должен быть пустым."
    )
    if clean:
        system = system.replace(PEER_REPORT_INSTRUCTION, "")
    human = (
        f"Вопрос: {query}\nОбщий план: {json.dumps(plan, ensure_ascii=False)}\n"
        f"Твои assignments: {json.dumps(plan['assignments'][channel], ensure_ascii=False)}\n"
        f"Раунд: {round_index}. Доступные чанки:\n{pool.format(observations)}\n"
        f"Report другого агента предыдущего раунда: {json.dumps(other_report, ensure_ascii=False)}"
    )
    return system, human


def planner_validator(value: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str, list[Any]]:
    aspects = []
    for index, item in enumerate(value.get("aspects", []), 1):
        if isinstance(item, str) and item.strip():
            aspects.append({"id": f"A{len(aspects) + 1}", "need": item.strip()})
        elif isinstance(item, dict) and str(item.get("need", "")).strip():
            aspects.append({"id": str(item.get("id") or f"A{index}"), "need": str(item["need"]).strip()})
    assignments = {}
    for channel in ("dense", "bm25"):
        raw = value.get("assignments", {}).get(channel, [])
        if isinstance(raw, str):
            raw = [raw]
        assignments[channel] = [str(item).strip() for item in raw[:2] if str(item).strip()]
    if not aspects:
        return None, "aspects пуст", []
    return {"aspects": aspects[:6], "assignments": assignments}, "", []


def arbiter_validator(value: Mapping[str, Any], pool: Any) -> tuple[list[dict[str, Any]] | None, str, dict[str, Any]]:
    notes: dict[str, Any] = {"dropped_unknown_aliases": [], "dropped_duplicate_aliases": []}
    selected = []
    seen = set()
    for item in value.get("selected_chunks", []):
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias", ""))
        chunk_id = pool.chunk_id(alias)
        if chunk_id is None:
            notes["dropped_unknown_aliases"].append(alias)
            continue
        if chunk_id in seen:
            notes["dropped_duplicate_aliases"].append(alias)
            continue
        relevance = item.get("relevance")
        aspect = str(item.get("aspect", "")).strip()
        if relevance not in (1, 2) or not aspect:
            continue
        seen.add(chunk_id)
        selected.append({"chunk_id": chunk_id, "alias": alias, "relevance": relevance, "aspect": aspect})
    if not selected and pool.ids:
        return None, "После валидации selection пуст", notes
    if len(selected) > MAX_OUTPUT_CHUNKS:
        return None, "Нужно не более 10 чанков", notes
    return selected, "", notes


def worker_validator(
    value: Mapping[str, Any],
    channel: str,
    round_index: int,
    observations: Sequence[str],
    pool: Any,
    query_cache: MutableMapping[str, set[str]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    notes: dict[str, Any] = {"dropped_unknown_aliases": [], "dropped_duplicate_aliases": []}
    selected = []
    seen = set()
    observed_aliases = {pool.alias(chunk_id) for chunk_id in observations}
    for item in value.get("assessments", [])[:15]:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias", ""))
        if alias not in observed_aliases:
            notes["dropped_unknown_aliases"].append(alias)
            continue
        if alias in seen:
            notes["dropped_duplicate_aliases"].append(alias)
            continue
        relevance = item.get("relevance")
        aspect = str(item.get("aspect", "")).strip()
        if relevance not in (1, 2) or not aspect:
            continue
        seen.add(alias)
        selected.append({
            "alias": alias,
            "relevance": relevance,
            "aspect": aspect,
            "evidence": str(item.get("evidence", ""))[:1000],
        })
    queries = []
    if round_index == 1:
        raw_queries = value.get("search_queries", [])
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        for item in raw_queries[:2]:
            item = str(item).strip()
            normalized = _normalize(item)
            if item and normalized not in query_cache[channel]:
                queries.append(item)
                query_cache[channel].add(normalized)
    coverage = value.get("coverage", {})
    return {
        "assessments": selected,
        "search_queries": queries,
        "coverage": {
            "covered": [str(item) for item in coverage.get("covered", [])],
            "missing": [str(item) for item in coverage.get("missing", [])],
        },
    }, "", notes


def make_worker_validator(
    channel: str,
    round_index: int,
    observations: Sequence[str],
    pool: Any,
    query_cache: MutableMapping[str, set[str]],
) -> Validator:
    def validate(value: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
        return worker_validator(value, channel, round_index, observations, pool, query_cache)

    return validate


async def arbiter_call(
    llm: Any,
    query: str,
    plan: Mapping[str, Any],
    reports: Mapping[str, Any] | None,
    pool_or_record: Any,
    chunk_map: Mapping[str, Any] | None = None,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    pool = _resolve_pool(pool_or_record, chunk_map)
    system, human = build_arbiter_prompts(query, plan, reports, pool)
    selected, meta = await ReplayLLMCaller(llm, timeout_seconds).call(
        "arbiter",
        system,
        human,
        ("selected_chunks",),
        lambda value: arbiter_validator(value, pool),
    )
    notes = meta["notes"] if isinstance(meta["notes"], Mapping) else {}
    return {
        **meta,
        "selected_chunks": selected,
        "strict_ids": [item["chunk_id"] for item in selected] if selected is not None else [],
        "ranked_chunk_ids": [item["chunk_id"] for item in selected] if selected is not None else [],
        "system_prompt": system,
        "human_prompt": human,
        "prompt_sha256": prompt_sha256(system, human),
        "pool_sha256": getattr(pool, "pool_sha256", None),
        "dropped_unknown_aliases": list(notes.get("dropped_unknown_aliases", [])),
        "dropped_duplicate_aliases": list(notes.get("dropped_duplicate_aliases", [])),
    }


async def worker_call(
    llm: Any,
    channel: str,
    query: str,
    plan: Mapping[str, Any],
    round_index: int,
    observations: Sequence[str],
    pool_or_record: Any,
    chunk_map: Mapping[str, Any] | None = None,
    other_report: Mapping[str, Any] | None = None,
    clean: bool = False,
    query_cache: MutableMapping[str, set[str]] | None = None,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    pool = _resolve_pool(pool_or_record, chunk_map)
    if query_cache is None:
        query_cache = {"dense": {_normalize(query)}, "bm25": {_normalize(query)}}
    system, human = build_worker_prompts(
        channel,
        query,
        plan,
        round_index,
        observations,
        pool,
        other_report,
        clean,
    )
    report, meta = await ReplayLLMCaller(llm, timeout_seconds).call(
        f"{channel}_round_{round_index}",
        system,
        human,
        ("assessments", "search_queries", "coverage"),
        make_worker_validator(channel, round_index, observations, pool, query_cache),
    )
    notes = meta["notes"] if isinstance(meta["notes"], Mapping) else {}
    return {
        **meta,
        "report": report,
        "system_prompt": system,
        "human_prompt": human,
        "prompt_sha256": prompt_sha256(system, human),
        "pool_sha256": getattr(pool, "pool_sha256", None),
        "dropped_unknown_aliases": list(notes.get("dropped_unknown_aliases", [])),
        "dropped_duplicate_aliases": list(notes.get("dropped_duplicate_aliases", [])),
    }
