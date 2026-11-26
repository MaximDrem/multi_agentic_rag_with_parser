import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from common.benchmark_logger import sum_tokens_from_messages
from common.config import (
    RETRIEVER_BM25_ENABLE_THINKING,
    RETRIEVER_BM25_LLM_BASE_URL,
    RETRIEVER_BM25_LLM_MAX_TOKENS,
    RETRIEVER_BM25_LLM_MODEL,
    RETRIEVER_BM25_LLM_TEMPERATURE,
    RETRIEVER_EMB_ENABLE_THINKING,
    RETRIEVER_EMB_LLM_BASE_URL,
    RETRIEVER_EMB_LLM_MAX_TOKENS,
    RETRIEVER_EMB_LLM_MODEL,
    RETRIEVER_EMB_LLM_TEMPERATURE,
    create_llm,
)
from retrieval.contracts import MAX_OUTPUT_CHUNKS, SearchBudget, chunk_dict
from retrieval.engine import RetrievalEngine
from retrieval.rrf import weighted_rrf

AGENT_TIMEOUT_SECONDS = 180


def _normalize(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", value.casefold()).strip()


def _json_object(text: str, required: tuple[str, ...]) -> dict | None:
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


class SharedCandidatePool:
    def __init__(self):
        self.ids = []
        self.aliases = {}
        self.chunks = {}
        self.ledger = {}

    def add(self, chunk, channel: str, query: str, call_index: int, rank: int) -> str:
        chunk_id = chunk.id
        if chunk_id not in self.aliases:
            alias = f"C{len(self.ids) + 1:03d}"
            self.ids.append(chunk_id)
            self.aliases[chunk_id] = alias
            self.chunks[chunk_id] = chunk
            self.ledger[chunk_id] = {
                "alias": alias,
                "chunk_id": chunk_id,
                "retrievals": [],
                "observed_by": [],
                "worker_assessments": {},
                "arbiter": {},
            }
        self.ledger[chunk_id]["retrievals"].append({
            "channel": channel,
            "query": query,
            "call_index": call_index,
            "rank": rank,
        })
        return self.aliases[chunk_id]

    def alias(self, chunk_id: str) -> str:
        return self.aliases[chunk_id]

    def chunk_id(self, alias: str) -> str | None:
        index = int(alias[1:]) - 1 if re.fullmatch(r"C\d{3}", alias or "") else -1
        return self.ids[index] if 0 <= index < len(self.ids) else None

    def format(self, chunk_ids: list[str]) -> str:
        blocks = []
        for chunk_id in chunk_ids:
            chunk = self.chunks[chunk_id]
            blocks.append(
                f"alias={self.alias(chunk_id)}\nfile={chunk.doc_path}\ntitle={chunk.title}\ntext={chunk.text}"
            )
        return "\n---\n".join(blocks) if blocks else "Нет доступных чанков."


class CoordinatedMultiAgentRetriever:
    def __init__(self, engine: RetrievalEngine | None = None, dense_llm=None, bm25_llm=None, planner_llm=None, arbiter_llm=None):
        self.engine = engine or RetrievalEngine()
        self.dense_llm = dense_llm if dense_llm is not None else self._llm("dense")
        self.bm25_llm = bm25_llm if bm25_llm is not None else self._llm("bm25")
        self.planner_llm = planner_llm if planner_llm is not None else self._llm("dense")
        self.arbiter_llm = arbiter_llm if arbiter_llm is not None else self._llm("dense")

    @staticmethod
    def _llm(channel: str):
        dense = channel == "dense"
        return create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL if dense else RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL if dense else RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE if dense else RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS if dense else RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING if dense else RETRIEVER_BM25_ENABLE_THINKING,
        )

    async def _call(self, role, llm, system, human, required, validator):
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        tokens = 0
        repair_used = False
        notes = []
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(llm.ainvoke(messages), timeout=AGENT_TIMEOUT_SECONDS)
            except Exception as exc:
                return None, {
                    "tokens": tokens,
                    "repair_used": repair_used,
                    "error": f"{type(exc).__name__}: {exc}",
                    "notes": notes,
                }
            tokens += sum_tokens_from_messages([response])
            value = _json_object(str(getattr(response, "content", "")), required)
            error = "Ответ не содержит требуемый JSON"
            if value is not None:
                try:
                    normalized, error, notes = validator(value)
                except Exception as exc:
                    normalized = None
                    error = f"Ошибка валидации: {type(exc).__name__}: {exc}"
                    notes = []
                if not error:
                    return normalized, {"tokens": tokens, "repair_used": repair_used, "error": None, "notes": notes}
            if attempt == 0:
                repair_used = True
                messages = messages + [response, HumanMessage(content=(
                    f"Ошибка: {error}. Верни только исправленный JSON с теми же полями. "
                    "Используй только существующие aliases, без пояснений и без повторов."
                ))]
        return None, {"tokens": tokens, "repair_used": repair_used, "error": error, "notes": notes}

    @staticmethod
    def _planner_validator(value):
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

    async def retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine = self.engine.with_budget(SearchBudget())
        pool = SharedCandidatePool()
        agent_meta = {}
        fallback_stages = []
        initial_dense, initial_bm25 = await asyncio.gather(
            asyncio.to_thread(engine.search, "dense", query, 15),
            asyncio.to_thread(engine.search, "bm25", query, 15),
        )
        for rank in range(15):
            if rank < len(initial_dense):
                pool.add(initial_dense[rank].chunk, "dense", query, 0, rank + 1)
            if rank < len(initial_bm25):
                pool.add(initial_bm25[rank].chunk, "bm25", query, 0, rank + 1)
        observations = {
            "dense": [hit.chunk.id for hit in initial_dense[:10]],
            "bm25": [hit.chunk.id for hit in initial_bm25[:10]],
        }
        for channel in observations:
            for chunk_id in observations[channel]:
                pool.ledger[chunk_id]["observed_by"].append(channel)
        planner_system = (
            "Ты Planner multi-agent поиска по нормативным документам РФ. "
            "Разбей вопрос на проверяемые юридические аспекты: объект, орган, действие, условия, даты, исключения и связанные нормы. "
            "Dense должен искать смысловые формулировки, BM25 — точные названия, номера и редкие термины. "
            "Не выдумывай факты. Верни только JSON: "
            '{"aspects":[{"id":"A1","need":"..."}],"assignments":{"dense":["..."],"bm25":["..."]}}.'
        )
        plan, agent_meta["planner"] = await self._call(
            "planner", self.planner_llm, planner_system, f"Вопрос: {query}",
            ("aspects", "assignments"), self._planner_validator,
        )
        if plan is None:
            fallback_stages.append("planner")
            plan = {"aspects": [{"id": "A1", "need": query}], "assignments": {"dense": [], "bm25": []}}
        query_cache = {"dense": {_normalize(query)}, "bm25": {_normalize(query)}}
        rounds = []

        def worker_validator(channel, round_index):
            def validate(value):
                notes = {"dropped_unknown_aliases": [], "dropped_duplicate_aliases": []}
                selected = []
                seen = set()
                observed_aliases = {pool.alias(chunk_id) for chunk_id in observations[channel]}
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
            return validate

        async def worker(channel, round_index, other_report):
            dense = channel == "dense"
            role = (
                "смысловые синонимы, определения и неявные условия" if dense
                else "точные названия документов, номера, даты, аббревиатуры и редкие термины"
            )
            system = (
                f"Ты {channel}-agent в кооперативном retrieval. Твоя специализация: {role}. "
                "Оценивай только существующие aliases и указывай, какой аспект покрывает каждый чанк. "
                "Запрашивай поиск только для непокрытых аспектов; обычно один точный подзапрос. "
                "Используй report другого агента, чтобы не дублировать его работу. "
                "Верни только JSON: {\"assessments\":[{\"alias\":\"C001\",\"relevance\":2,"
                "\"aspect\":\"A1\",\"evidence\":\"короткая опорная фраза\"}],"
                "\"search_queries\":[\"...\"],\"coverage\":{\"covered\":[\"A1\"],\"missing\":[\"A2\"]}}. "
                "Во втором раунде search_queries должен быть пустым."
            )
            human = (
                f"Вопрос: {query}\nОбщий план: {json.dumps(plan, ensure_ascii=False)}\n"
                f"Твои assignments: {json.dumps(plan['assignments'][channel], ensure_ascii=False)}\n"
                f"Раунд: {round_index}. Доступные чанки:\n{pool.format(observations[channel])}\n"
                f"Report другого агента предыдущего раунда: {json.dumps(other_report, ensure_ascii=False)}"
            )
            llm = self.dense_llm if dense else self.bm25_llm
            result, meta = await self._call(
                f"{channel}_round_{round_index}", llm, system, human,
                ("assessments", "search_queries", "coverage"), worker_validator(channel, round_index),
            )
            return result, meta

        first_dense, first_bm25 = await asyncio.gather(worker("dense", 1, None), worker("bm25", 1, None))
        first = {"dense": first_dense[0], "bm25": first_bm25[0]}
        agent_meta["dense_round_1"] = first_dense[1]
        agent_meta["bm25_round_1"] = first_bm25[1]
        for channel, result in first.items():
            if result is None:
                fallback_stages.append(f"{channel}_round_1")
                first[channel] = {"assessments": [], "search_queries": [], "coverage": {"covered": [], "missing": []}}
        rounds.append({"round": 1, "reports": first})
        expansion_queries = {channel: first[channel]["search_queries"] for channel in ("dense", "bm25")}
        if any(expansion_queries.values()):
            search_results = await asyncio.gather(
                asyncio.to_thread(engine.search_many, "dense", expansion_queries["dense"], 10),
                asyncio.to_thread(engine.search_many, "bm25", expansion_queries["bm25"], 10),
            )
            expansion = {}
            for channel, results in zip(("dense", "bm25"), search_results):
                raw_new = []
                observed_new = []
                for subquery, hits in results.items():
                    for hit in hits:
                        chunk_id = hit.chunk.id
                        is_new = chunk_id not in pool.aliases
                        pool.add(hit.chunk, channel, subquery, 1, hit.rank)
                        if is_new:
                            raw_new.append(chunk_id)
                        if chunk_id not in observations[channel] and len(observations[channel]) < 15:
                            observations[channel].append(chunk_id)
                            pool.ledger[chunk_id]["observed_by"].append(channel)
                            observed_new.append(chunk_id)
                expansion[channel] = {"queries": expansion_queries[channel], "raw_new_ids": raw_new, "new_observed_ids": observed_new}
            second_dense, second_bm25 = await asyncio.gather(
                worker("dense", 2, first["bm25"]), worker("bm25", 2, first["dense"]),
            )
            second = {"dense": second_dense[0], "bm25": second_bm25[0]}
            agent_meta["dense_round_2"] = second_dense[1]
            agent_meta["bm25_round_2"] = second_bm25[1]
            for channel, result in second.items():
                if result is None:
                    fallback_stages.append(f"{channel}_round_2")
                    second[channel] = first[channel]
            rounds.append({"round": 1, "expansion": expansion})
            rounds.append({"round": 2, "reports": second, "shared_previous_reports": first})
        else:
            second = first
        for channel in ("dense", "bm25"):
            for assessment in second[channel]["assessments"]:
                chunk_id = pool.chunk_id(assessment["alias"])
                if chunk_id:
                    pool.ledger[chunk_id]["worker_assessments"][channel] = assessment
        arbiter_system = (
            "Ты Arbiter кооперативного multi-agent retrieval по НПА РФ. "
            "Получаешь общий candidate pool, план аспектов и reports двух агентов. "
            "Самостоятельно оцени юридическую применимость, фактические условия, исключения и полноту покрытия. "
            "Используй worker relevance/aspect как рекомендации, но проверяй текст. "
            "Сформируй один глобальный ranking: сначала прямые ответы, затем необходимые дополнительные доказательства, "
            "покрой все аспекты без дубликатов по смыслу. Верни ровно 10 элементов при наличии 10 candidates. "
            "Только JSON: {\"selected_chunks\":[{\"alias\":\"C001\",\"relevance\":2,\"aspect\":\"A1\"}]}. "
            "Никаких рассуждений, новых aliases или повторов."
        )

        def arbiter_validator(value):
            notes = {"dropped_unknown_aliases": [], "dropped_duplicate_aliases": []}
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

        arbiter_human = (
            f"Вопрос: {query}\nПлан: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Worker reports: {json.dumps(second, ensure_ascii=False)}\n"
            f"Общий pool:\n{pool.format(pool.ids)}"
        )
        arbiter, agent_meta["arbiter"] = await self._call(
            "arbiter", self.arbiter_llm, arbiter_system, arbiter_human,
            ("selected_chunks",), arbiter_validator,
        )
        if arbiter is None:
            fallback_stages.append("arbiter")
            arbiter = []
        worker_rankings = {}
        assessment_map = {}
        for channel in ("dense", "bm25"):
            ordered = sorted(
                enumerate(second[channel]["assessments"]),
                key=lambda item: (-item[1]["relevance"], item[0]),
            )
            aliases = []
            for _, assessment in ordered:
                chunk_id = pool.chunk_id(assessment["alias"])
                if chunk_id:
                    aliases.append(chunk_id)
                    assessment_map.setdefault(chunk_id, assessment)
            worker_rankings[channel] = aliases
        worker_merged = weighted_rrf(
            worker_rankings["dense"], worker_rankings["bm25"], alpha=0.5, limit=30,
        )
        selected_ids = [item["chunk_id"] for item in arbiter]
        backfill_ids = [item["chunk_id"] for item in worker_merged if item["chunk_id"] not in selected_ids]
        backfill_ids += [chunk_id for chunk_id in pool.ids if chunk_id not in selected_ids and chunk_id not in backfill_ids]
        backfill_count = 0
        for chunk_id in backfill_ids:
            if len(selected_ids) >= MAX_OUTPUT_CHUNKS:
                break
            if len(arbiter) < MAX_OUTPUT_CHUNKS or not arbiter:
                selected_ids.append(chunk_id)
                backfill_count += 1
        chunks = []
        arbiter_by_id = {item["chunk_id"]: item for item in arbiter}
        for rank, chunk_id in enumerate(selected_ids[:MAX_OUTPUT_CHUNKS], 1):
            selected = arbiter_by_id.get(chunk_id)
            assessment = assessment_map.get(chunk_id, {})
            relevance = selected["relevance"] if selected else assessment.get("relevance")
            aspect = selected["aspect"] if selected else assessment.get("aspect", "")
            chunks.append(chunk_dict(pool.chunks[chunk_id], rank, relevance=relevance, answer_aspect=aspect))
            pool.ledger[chunk_id]["arbiter"] = {
                "final_rank": rank,
                "selected_by_arbiter": selected is not None,
                "relevance": relevance,
                "aspect": aspect,
                "backfilled": selected is None,
            }
        total_tokens = sum(meta["tokens"] for meta in agent_meta.values())
        return {
            "chunks": chunks,
            "retriever": "multiagent_coordinated",
            "fallback_used": bool(fallback_stages),
            "error": None,
            "trace": {
                "query": query,
                "fusion": "planner_shared_blackboard_workers_arbiter",
                "plan": plan,
                "rounds": rounds,
                "agent_meta": agent_meta,
                "fallback_stages": fallback_stages,
                "arbiter_backfill_count": backfill_count,
                "raw_unique_candidates": len(pool.ids),
                "candidate_ledger": pool.ledger,
                "information_budget": engine.budget.snapshot(),
            },
            "token_usage": {"total_tokens": total_tokens},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
