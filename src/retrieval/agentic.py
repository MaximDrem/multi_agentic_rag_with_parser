import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common.benchmark_logger import sum_tokens_from_messages
from retrieval.contracts import (
    MAX_OBSERVED_CHUNKS,
    MAX_OBSERVED_TOKENS,
    ObservationBudget,
    chunk_dict,
)
from retrieval.engine import RetrievalEngine
from retrieval.rrf import rrf_merge, weighted_rrf

logger = logging.getLogger(__name__)

AGENTIC_MAX_CALLS = 3
AGENTIC_RECURSION_LIMIT = 20
AGENTIC_TIMEOUT_SECONDS = 180


def _normalize(query: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", query.casefold()).strip()


def _json_selection(text: str) -> dict | None:
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
        if isinstance(value, dict) and "selected_chunks" in value:
            return value
    return None


class AgenticSearchRuntime:
    def __init__(
        self,
        llm,
        engine: RetrievalEngine,
        mode: str,
        dynamic_alpha: bool = False,
        max_observed_chunks: int = MAX_OBSERVED_CHUNKS,
        max_observed_tokens: int = MAX_OBSERVED_TOKENS,
        selection_limit: int = 10,
        reject_tool_copy: bool = True,
    ):
        self.llm = llm
        self.engine = engine
        self.mode = mode
        self.dynamic_alpha = dynamic_alpha
        self.max_observed_chunks = max_observed_chunks
        self.max_observed_tokens = max_observed_tokens
        self.selection_limit = selection_limit
        self.reject_tool_copy = reject_tool_copy

    def _system_prompt(self) -> str:
        alpha = (
            " Можешь дополнительно передавать alpha в [0,1]: большие значения усиливают dense, меньшие — BM25; начни с 0.5."
            if self.dynamic_alpha else ""
        )
        return (
            "Ты — агент поиска доказательств по нормативным документам РФ. "
            "У тебя есть только инструмент search; нельзя выбирать отдельный retrieval-метод. "
            "Формулируй 1-3 новых подзапроса через ; только для непокрытых аспектов, сохраняя объекты, органы, даты и ограничения."
            + alpha +
            " Индексные scores и RRF ranks скрыты: оценивай содержимое чанков самостоятельно. "
            "Не копируй порядок или первые элементы observation. Выбери лучшие самостоятельные доказательства с учётом покрытия аспектов и разнообразия. "
            "В финальном ответе верни только JSON: "
            '{"selected_chunks":[{"chunk_id":"...","relevance":2,"answer_aspect":"..."}]}. '
            "Порядок элементов — твой итоговый ranking. Relevance 2 — прямой ответ или важная часть, 1 — необходимый косвенный факт."
        )

    async def retrieve(
        self,
        query: str,
        initial_hits: list | None = None,
        initial_observed_limit: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        observation = ObservationBudget(self.max_observed_chunks, self.max_observed_tokens)
        candidates: dict[str, Any] = {}
        cache: dict[str, bool] = {}
        fallback_rankings: list[list[str]] = []
        trace = {"searches": [], "tool_calls": [], "alpha_values": [], "errors": [], "selection_sanitization": []}
        lock = threading.Lock()
        call_count = 0
        last_observation_ids: list[str] = []
        messages = []
        repair_used = False

        def perform(queries: list[str], alpha: float, call_index: int) -> list:
            new_queries = []
            for subquery in queries:
                normalized = _normalize(subquery)
                cache_key = f"{normalized}|alpha={alpha:.6f}" if self.dynamic_alpha and self.mode == "hybrid" else normalized
                if cache_key in cache:
                    trace["searches"].append({"query": subquery, "cached": True, "call_index": call_index})
                    continue
                cache[cache_key] = True
                new_queries.append(subquery)
            if not new_queries:
                return []
            if self.mode == "hybrid":
                dense = self.engine.search_many("dense", new_queries, 10)
                lexical = self.engine.search_many("bm25", new_queries, 10)
            elif self.mode == "dense":
                dense = self.engine.search_many("dense", new_queries, 10)
                lexical = {subquery: [] for subquery in new_queries}
            else:
                dense = {subquery: [] for subquery in new_queries}
                lexical = self.engine.search_many("bm25", new_queries, 10)
            new_chunks = []
            for subquery in new_queries:
                dense_hits = dense.get(subquery, [])
                lexical_hits = lexical.get(subquery, [])
                for hit in list(dense_hits) + list(lexical_hits):
                    candidates[hit.chunk.id] = hit.chunk
                if self.mode == "hybrid":
                    merged = weighted_rrf(
                        [hit.chunk.id for hit in dense_hits],
                        [hit.chunk.id for hit in lexical_hits],
                        alpha=alpha,
                        limit=30,
                    )
                    ranking = [item["chunk_id"] for item in merged]
                else:
                    hits = dense_hits if self.mode == "dense" else lexical_hits
                    ranking = [hit.chunk.id for hit in hits]
                fallback_rankings.append(ranking)
                ranked_ids = set(ranking)
                new_chunks.extend(self.engine.chunk_map[chunk_id] for chunk_id in ranking)
                for hit in list(dense_hits) + list(lexical_hits):
                    if hit.chunk.id not in ranked_ids:
                        new_chunks.append(hit.chunk)
                trace["searches"].append({
                    "query": subquery,
                    "cached": False,
                    "call_index": call_index,
                    "alpha": alpha if self.mode == "hybrid" else None,
                    "dense_hits": len(dense_hits),
                    "bm25_hits": len(lexical_hits),
                    "ranking_ids": ranking,
                })
            if self.mode == "hybrid":
                trace["alpha_values"].append(alpha)
            unique = {chunk.id: chunk for chunk in new_chunks}
            selected = observation.observe(list(unique.values()))
            return sorted(selected, key=lambda chunk: hashlib.sha256((query + chunk.id).encode()).hexdigest())

        def render(chunks: list) -> str:
            nonlocal last_observation_ids
            last_observation_ids = [chunk.id for chunk in chunks]
            if not chunks:
                return "Новых чанков не добавлено: подзапросы повторяются, бюджет исчерпан или поиск ничего не нашёл."
            lines = [
                f"Показаны {len(chunks)} новых полных чанков в порядке, не связанном со scores или RRF. "
                "Оцени их содержимое и выбери доказательства самостоятельно."
            ]
            for chunk in chunks:
                lines.append(
                    f"chunk_id={chunk.id}\n"
                    f"file={chunk.doc_path}\n"
                    f"title={chunk.title}\n"
                    f"text={chunk.text}"
                )
            return "\n---\n".join(lines)

        def execute_search(keywords: str, alpha: float) -> str:
            nonlocal call_count
            with lock:
                if call_count >= AGENTIC_MAX_CALLS:
                    return f"Достигнут лимит {AGENTIC_MAX_CALLS} поисковых вызовов. Заверши и верни собственный JSON ranking."
                if self.dynamic_alpha and not 0.0 <= alpha <= 1.0:
                    return "alpha должен быть в [0,1]. Исправь вызов."
                call_count += 1
                alpha = alpha if self.dynamic_alpha else 0.5
                queries = [item.strip() for item in keywords.split(";") if item.strip()][:3]
                if not queries:
                    return "Пустой подзапрос. Передай 1-3 непустые формулировки через ;."
                try:
                    chunks = perform(queries, alpha, call_count)
                except Exception as exc:
                    logger.exception("Agentic search failed")
                    trace["errors"].append({"error": str(exc), "call_index": call_count})
                    return "Поиск завершился ошибкой. Попробуй другую формулировку или заверши работу."
                return render(chunks)

        if self.dynamic_alpha:
            @tool
            def search(keywords: str, alpha: float = 0.5) -> str:
                """Единственный hybrid-поиск: 1-3 подзапроса через ; и alpha Weighted RRF в [0,1]."""
                return execute_search(keywords, alpha)
        else:
            @tool
            def search(keywords: str) -> str:
                """Единственный поиск в системе: 1-3 новых подзапроса через ; без выбора retrieval-метода."""
                return execute_search(keywords, 0.5)

        initial_display = []
        try:
            if initial_hits is None:
                initial = await asyncio.to_thread(perform, [query], 0.5, 1)
                initial_display = initial
            else:
                initial = list(initial_hits)
                with lock:
                    for hit in initial:
                        candidates[hit.chunk.id] = hit.chunk
                    ranking = [hit.chunk.id for hit in initial]
                    fallback_rankings.append(ranking)
                    visible = initial[:initial_observed_limit] if initial_observed_limit is not None else initial
                    accepted = observation.observe([hit.chunk for hit in visible])
                    initial_display = sorted(
                        accepted,
                        key=lambda chunk: hashlib.sha256((query + chunk.id).encode()).hexdigest(),
                    )
                    trace["searches"].append({
                        "query": query,
                        "cached": False,
                        "call_index": 1,
                        "alpha": None,
                        "dense_hits": len(initial) if self.mode == "dense" else 0,
                        "bm25_hits": len(initial) if self.mode == "bm25" else 0,
                        "ranking_ids": ranking,
                    })
            call_count = 1
        except Exception as exc:
            return {
                "chunks": [],
                "retriever": self.mode,
                "fallback_used": True,
                "error": str(exc),
                "trace": trace,
                "token_usage": {"total_tokens": 0},
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

        agent = create_agent(
            self.llm,
            [search],
            system_prompt=SystemMessage(content=self._system_prompt()),
        )
        human = HumanMessage(content=(
            f"Исходный вопрос: {query}\n\n"
            f"Исходный поиск уже выполнен. Результаты:\n{render(initial_display)}\n\n"
            "При необходимости используй search для новых подзапросов. "
            f"В конце выбери до {self.selection_limit} чанков и верни только JSON с собственным ranking."
        ))

        def validate_selection(value: dict) -> tuple[list[dict], str, dict]:
            sanitization = {"dropped_unknown_ids": [], "dropped_duplicate_ids": []}
            rows = value.get("selected_chunks")
            if not isinstance(rows, list):
                return [], "selected_chunks должен быть списком", sanitization
            if not rows and candidates:
                return [], "Пустой selection при наличии кандидатов", sanitization
            if len(rows) > self.selection_limit:
                return [], f"Максимум {self.selection_limit} чанков", sanitization
            selected = []
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    return [], "Некорректный элемент selection", sanitization
                chunk_id = row.get("chunk_id")
                relevance = row.get("relevance")
                aspect = row.get("answer_aspect")
                if not isinstance(chunk_id, str) or chunk_id not in observation.observed_set:
                    sanitization["dropped_unknown_ids"].append(chunk_id)
                    continue
                if chunk_id in seen:
                    sanitization["dropped_duplicate_ids"].append(chunk_id)
                    continue
                if relevance not in (1, 2) or not isinstance(aspect, str) or not aspect.strip():
                    return [], "relevance должен быть 1/2, answer_aspect непустым", sanitization
                seen.add(chunk_id)
                selected.append({"chunk_id": chunk_id, "relevance": relevance, "answer_aspect": aspect.strip()})
            if not selected and candidates:
                return [], "После удаления неизвестных и повторных ID selection пуст", sanitization
            ids = [row["chunk_id"] for row in selected]
            if self.reject_tool_copy and len(ids) == self.selection_limit and len(candidates) > len(ids) and ids == last_observation_ids[:len(ids)]:
                return [], "Selection копирует порядок последнего observation; выполни самостоятельный отбор", sanitization
            return selected, "", sanitization

        async def invoke(payload):
            return await asyncio.wait_for(
                agent.ainvoke(payload, config={"recursion_limit": AGENTIC_RECURSION_LIMIT}),
                timeout=AGENTIC_TIMEOUT_SECONDS,
            )

        selected = []
        selection_error = ""
        try:
            invocation = await invoke({"messages": [human]})
            messages = invocation["messages"]
            for message in messages:
                for tool_call in getattr(message, "tool_calls", None) or []:
                    trace["tool_calls"].append({"name": tool_call.get("name"), "args": tool_call.get("args", {})})
            content = str(getattr(messages[-1], "content", "")) if messages else ""
            value = _json_selection(content)
            if value is None:
                selection_error = "Финальный ответ не содержит JSON selected_chunks"
            else:
                selected, selection_error, sanitization = validate_selection(value)
                if any(sanitization.values()):
                    trace["selection_sanitization"].append({"attempt": "initial", **sanitization})
            if selection_error:
                repair_used = True
                repair = HumanMessage(content=(
                    f"Ошибка selection: {selection_error}. "
                    "Верни только исправленный JSON selected_chunks с собственным ranking, без пояснений."
                ))
                invocation = await invoke({"messages": messages + [repair]})
                messages = invocation["messages"]
                content = str(getattr(messages[-1], "content", "")) if messages else ""
                value = _json_selection(content)
                selected = []
                if value is None:
                    selection_error = "После repair JSON отсутствует"
                else:
                    selected, selection_error, sanitization = validate_selection(value)
                    if any(sanitization.values()):
                        trace["selection_sanitization"].append({"attempt": "repair", **sanitization})
        except Exception as exc:
            logger.exception("Agentic ReAct failed")
            selection_error = f"{type(exc).__name__}: {exc}"
            trace["errors"].append({"error": str(exc), "type": type(exc).__name__})

        fallback_used = bool(selection_error)
        if selection_error:
            trace["errors"].append({"selection_error": selection_error})
            merged = rrf_merge(
                {f"search_{index}": ranking for index, ranking in enumerate(fallback_rankings)},
                limit=30,
            )
            selected = [
                {"chunk_id": item["chunk_id"], "relevance": None, "answer_aspect": "", "score": item["score"]}
                for item in merged[:self.selection_limit]
            ]
        chunks = []
        for rank, row in enumerate(selected, 1):
            chunk = self.engine.chunk_map.get(row["chunk_id"])
            if chunk:
                chunks.append(chunk_dict(chunk, rank, row.get("score"), row.get("relevance"), row.get("answer_aspect", "")))
        return {
            "chunks": chunks,
            "retriever": self.mode,
            "fallback_used": fallback_used,
            "error": None,
            "trace": {
                **trace,
                "query": query,
                "mode": self.mode,
                "dynamic_alpha": self.dynamic_alpha,
                "selection_source": "backend_fallback" if fallback_used else ("agent_sanitized" if trace["selection_sanitization"] else "agent"),
                "selection_error": selection_error,
                "repair_used": repair_used,
                "search_calls": call_count,
                "unique_candidates": len(candidates),
                "observation": observation.snapshot(),
                "budget": self.engine.budget.snapshot(),
            },
            "token_usage": {"total_tokens": sum_tokens_from_messages(messages)},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
