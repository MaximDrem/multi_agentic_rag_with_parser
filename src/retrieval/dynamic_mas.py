import asyncio
import itertools
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field

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
from retrieval.coordinated import SharedCandidatePool, _json_object
from retrieval.engine import RetrievalEngine

AGENT_TIMEOUT_SECONDS = 180
MAX_AGENT_TURNS = 32
MAX_CONTRACTS = 3


def _norm(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", str(value).casefold()).strip()


def _aliases(text: str) -> set[str]:
    return set(re.findall(r"C\d{3}", text))


@dataclass
class AgentMemory:
    received: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    known_aliases: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class Contract:
    contract_id: str
    aspect: str
    need: str
    kind: str
    participants: list[str]
    origin: str
    bids: dict = field(default_factory=dict)
    status: str = "open"
    awarded_to: str | None = None


class DynamicMASAgent:
    def __init__(self, name: str, channel: str | None, llm):
        self.name = name
        self.channel = channel
        self.llm = llm
        self.memory = AgentMemory()

    async def handle(self, message: dict, runtime) -> list[dict]:
        self.memory.received.append(message["message_id"])
        if self.name in ("dense", "bm25") or self.name.startswith("specialist_"):
            if message["type"] == "cfp":
                return await self._bid(message, runtime)
            if message["type"] == "stop_vote":
                return await self._stop_vote(message, runtime)
            if message["type"] in ("initial_evidence", "award", "peer_report"):
                return await self._observe(message, runtime)
            return []
        if self.name == "critic":
            if message["type"] == "stop_vote":
                return await self._stop_vote(message, runtime)
            return await self._critic(message, runtime)
        if self.name == "synthesizer":
            return await self._synthesize(message, runtime)
        return []

    async def _json(self, runtime, system: str, human: str, required: tuple[str, ...], validator=None):
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        tokens = 0
        repair = False
        error = "invalid_json"
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=AGENT_TIMEOUT_SECONDS)
            except Exception as exc:
                runtime.llm_tokens += tokens
                return None, {"tokens": tokens, "repair": repair, "error": f"{type(exc).__name__}: {exc}"}
            tokens += sum_tokens_from_messages([response])
            value = _json_object(str(getattr(response, "content", "")), required)
            if value is not None and validator is not None:
                error = validator(value)
                if error:
                    value = None
            if value is not None:
                runtime.llm_tokens += tokens
                return value, {"tokens": tokens, "repair": repair, "error": None}
            if attempt == 0:
                repair = True
                messages += [response, HumanMessage(content=f"Ошибка: {error}. Верни только исправленный JSON с требуемыми полями, без пояснений.")]
        runtime.llm_tokens += tokens
        return None, {"tokens": tokens, "repair": repair, "error": error}

    def _known_text(self, runtime) -> str:
        ids = [runtime.pool.chunk_id(alias) for alias in self.memory.known_aliases]
        return runtime.pool.format([chunk_id for chunk_id in ids if chunk_id])

    async def _bid(self, message, runtime):
        gap = message["payload"]
        system = (
            f"Ты автономный {self.name}-agent в Contract Net retrieval MAS. "
            "Оцени свою пригодность для непокрытого аспекта по локальным наблюдениям. "
            "Можешь отказаться. Dense отвечает за смысловые связи, BM25 за точные термины/номера, специалисты — за свой тип зависимости. "
            "Bid должен предлагать один конкретный запрос с ожидаемой новизной. Только JSON: "
            '{"decision":"bid|refuse","query":"...","coverage":0.0,"confidence":0.0,"cost":1.0,"reason":"..."}.'
        )
        human = (
            f"Вопрос: {runtime.query}\nГлобальная задача: {gap['aspect']} — {gap['need']}\n"
            f"Тип: {gap['kind']}. Локальные наблюдения:\n{self._known_text(runtime)}"
        )
        value, meta = await self._json(runtime, system, human, ("decision", "reason"))
        self.memory.actions.append(f"bid:{gap['contract_id']}")
        if value is None:
            runtime.failures.append({"agent": self.name, "stage": "bid", "error": meta["error"]})
            return [runtime.message("contract_error", self.name, "blackboard", {"contract_id": gap["contract_id"], "error": meta["error"]}, message["message_id"])]
        if value.get("decision") != "bid":
            return [runtime.message("refuse", self.name, "blackboard", {"contract_id": gap["contract_id"], "reason": value.get("reason", "")}, message["message_id"])]
        query = str(value.get("query", "")).strip() or gap["need"]
        coverage = min(1.0, max(0.0, float(value.get("coverage", 0.0))))
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
        cost = min(5.0, max(0.1, float(value.get("cost", 1.0))))
        utility = coverage * confidence / cost
        return [runtime.message("bid", self.name, "blackboard", {
            "contract_id": gap["contract_id"], "query": query, "coverage": coverage,
            "confidence": confidence, "cost": cost, "utility": utility,
            "reason": str(value.get("reason", "")),
        }, message["message_id"], tokens=meta["tokens"], repair=meta["repair"])]

    async def _observe(self, message, runtime):
        payload = message["payload"]
        new_ids = []
        if message["type"] == "award":
            contract = runtime.contracts[payload["contract_id"]]
            query = payload["query"]
            try:
                hits = await asyncio.to_thread(runtime.engine.search, self.channel, query, 10)
            except Exception as exc:
                runtime.failures.append({"agent": self.name, "stage": "search", "error": str(exc)})
                return [runtime.message("contract_error", self.name, "blackboard", {"contract_id": contract.contract_id, "error": str(exc)}, message["message_id"])]
            for hit in hits:
                is_new = hit.chunk.id not in runtime.pool.aliases
                alias = runtime.pool.add(hit.chunk, self.channel, query, int(contract.contract_id[1:]), hit.rank)
                if is_new:
                    new_ids.append(hit.chunk.id)
                if alias not in self.memory.known_aliases:
                    self.memory.known_aliases.append(alias)
            runtime.message_new_chunks[message["message_id"]] = new_ids
        available = [runtime.pool.chunk_id(alias) for alias in self.memory.known_aliases]
        available = [chunk_id for chunk_id in available if chunk_id]
        system = (
            f"Ты stateful {self.name}-agent в event-driven retrieval MAS. "
            "Оценивай свои локальные чанки, создавай EvidenceClaims и при необходимости CoverageGap. "
            "Можешь адресно запросить другого агента, если нужна его компетенция. Не повторяй уже покрытые зависимости и не создавай "
            "гипотетические уточнения сверх исходного вопроса, если доступные evidence уже содержат ответ или допустимые условные варианты. "
            "Только JSON: {\"claims\":[{\"alias\":\"C001\",\"relevance\":2,\"aspect\":\"точный аспект\",\"evidence\":\"дословная короткая цитата\"}],"
            "\"gaps\":[{\"aspect\":\"непокрытый аспект\",\"need\":\"что найти\",\"kind\":\"semantic|lexical|exception|cross_reference|date\"}],"
            "\"peer_request\":{\"target\":\"dense|bm25\",\"aspect\":\"...\",\"need\":\"...\",\"kind\":\"...\"},"
            "\"stop\":false}. peer_request может быть null. Для второго анализа учитывай peer report и измени решение при необходимости."
        )
        human = (
            f"Вопрос: {runtime.query}\nСобытие: {message['type']}\n"
            f"Новые chunk IDs: {json.dumps(new_ids, ensure_ascii=False)}\n"
            f"Локальные чанки:\n{runtime.pool.format(available)}\n"
            f"Входящий report: {json.dumps(payload.get('report', {}), ensure_ascii=False)}\n"
            f"Evidence graph: {json.dumps(runtime.aspects, ensure_ascii=False)}"
        )
        value, meta = await self._json(runtime, system, human, ("claims", "gaps", "peer_request", "stop"))
        self.memory.actions.append(f"observe:{message['type']}")
        if value is None:
            runtime.failures.append({"agent": self.name, "stage": "observe", "error": meta["error"]})
            return [runtime.message("agent_error", self.name, "critic", {"error": meta["error"]}, message["message_id"])]
        claims = []
        allowed = set(self.memory.known_aliases)
        for item in value.get("claims", [])[:15]:
            if not isinstance(item, dict):
                continue
            alias = str(item.get("alias", ""))
            relevance = item.get("relevance")
            aspect = str(item.get("aspect", "")).strip()
            if alias not in allowed or relevance not in (1, 2) or not aspect:
                continue
            claims.append({
                "alias": alias, "relevance": relevance, "aspect": aspect,
                "evidence": str(item.get("evidence", ""))[:1000],
            })
        gaps = []
        for item in value.get("gaps", [])[:3]:
            if isinstance(item, dict) and str(item.get("need", "")).strip():
                gaps.append({
                    "aspect": str(item.get("aspect") or item["need"]),
                    "need": str(item["need"]).strip(),
                    "kind": str(item.get("kind", "semantic")),
                })
        report = {"agent": self.name, "claims": claims, "gaps": gaps, "stop": bool(value.get("stop"))}
        messages = [runtime.message("evidence_report", self.name, "blackboard", {
            "report": report, "phase": "initial" if message["type"] == "initial_evidence" else "expansion",
            "source_type": message["type"], "contract_id": payload.get("contract_id"),
            "new_chunk_ids": new_ids,
        }, message["message_id"], tokens=meta["tokens"], repair=meta["repair"])]
        peer = value.get("peer_request")
        if isinstance(peer, dict) and peer.get("target") in ("dense", "bm25") and peer.get("target") != self.name:
            target = peer["target"]
            messages.append(runtime.message("peer_request", self.name, target, {
                "aspect": str(peer.get("aspect") or peer.get("need", "peer dependency")),
                "need": str(peer.get("need", "")),
                "kind": str(peer.get("kind", "semantic")),
            }, message["message_id"]))
        return messages

    async def _critic(self, message, runtime):
        report = message["payload"].get("report", {})
        system = (
            "Ты Legal Critic в retrieval MAS. Управляешь зависимостями доказательств, а не выполняешь заранее заданный этап. "
            "Проверяй claims по тексту, оспаривай нерелевантные/неприменимые evidence, определяй покрытые и открытые аспекты. "
            "Scout gaps являются предложениями: открывай только зависимости, без которых нельзя ответить на исходный вопрос. "
            "Не требуй гипотетического точного вида акта, единственного универсального названия или дополнительных юридических свойств, "
            "если текст уже даёт ответ, включая условные варианты. Можешь создать новый CoverageGap после ошибки контракта. "
            "Завершай только при покрытии всех обязательных аспектов. "
            "Только JSON: {\"accepted_aliases\":[\"C001\"],\"challenged\":[{\"alias\":\"C002\",\"reason\":\"...\"}],"
            "\"aspects\":[{\"aspect\":\"...\",\"status\":\"covered|open\",\"covered_by\":[\"C001\"]}],"
            "\"gaps\":[{\"aspect\":\"...\",\"need\":\"...\",\"kind\":\"semantic|lexical|exception|cross_reference|date\"}],\"stop\":false}."
        )
        review_aliases = list(dict.fromkeys(
            [item.get("alias") for item in report.get("claims", []) if isinstance(item, dict)]
            + [alias for alias, claim in runtime.claims.items() if claim.get("accepted")]
            + list(runtime.challenged)
        ))
        review_ids = [runtime.pool.chunk_id(alias) for alias in review_aliases if runtime.pool.chunk_id(alias)][:20]
        human = (
            f"Вопрос: {runtime.query}\nВходящий report: {json.dumps(report, ensure_ascii=False)}\n"
            f"Текущий граф: {json.dumps(runtime.aspects, ensure_ascii=False)}\n"
            f"Evidence under review:\n{runtime.pool.format(review_ids)}\n"
            f"Ошибка: {json.dumps(message['payload'].get('error'), ensure_ascii=False)}"
        )
        value, meta = await self._json(runtime, system, human, ("accepted_aliases", "challenged", "aspects", "gaps", "stop"))
        self.memory.actions.append(f"critic:{message['type']}")
        if value is None:
            runtime.failures.append({"agent": self.name, "stage": "critic", "error": meta["error"]})
            open_contracts = any(contract.status in ("open", "awarded") for contract in runtime.contracts.values())
            if not open_contracts and not runtime.stop_in_progress():
                return [runtime.message(
                    "stop_proposal", self.name, "blackboard",
                    {"proposal_id": f"S{next(runtime.proposal_ids):03d}", "reason": "critic_error_best_effort"},
                    message["message_id"],
                )]
            return []
        result = dict(value)
        result["meta"] = meta
        result["initial_review"] = message["payload"].get("phase") == "initial"
        result["report_agent"] = report.get("agent")
        return [runtime.message("critic_verdict", self.name, "blackboard", result, message["message_id"], tokens=meta["tokens"], repair=meta["repair"])]

    async def _stop_vote(self, message, runtime):
        system = (
            f"Ты {self.name}-agent, голосующий за завершение retrieval. "
            "Принимай завершение только если по локальным наблюдениям покрыты обязательные аспекты. "
            "Если аспект имеет status=blocked после исчерпания контрактов, поддержи best-effort завершение и не повторяй тот же gap. "
            "При отказе обязан назвать конкретный непокрытый gap. Только JSON: "
            '{"accept":true,"reason":"...","gap":{"aspect":"...","need":"...","kind":"..."}}.'
        )
        human = f"Вопрос: {runtime.query}\nГраф: {json.dumps(runtime.aspects, ensure_ascii=False)}\nЛокальные чанки:\n{self._known_text(runtime)}"
        value, meta = await self._json(runtime, system, human, ("accept", "reason"))
        if value is None:
            runtime.failures.append({"agent": self.name, "stage": "stop_vote", "error": meta["error"]})
            value = {"accept": False, "reason": meta["error"], "gap": None, "abstain": True}
        return [runtime.message("stop_ballot", self.name, "blackboard", {
            "proposal_id": message["payload"]["proposal_id"],
            "accept": bool(value.get("accept")),
            "abstain": bool(value.get("abstain", False)),
            "reason": str(value.get("reason", "")),
            "gap": value.get("gap"),
        }, message["message_id"], tokens=meta["tokens"], repair=meta["repair"])]

    async def _synthesize(self, message, runtime):
        system = (
            "Ты Synthesizer в event-driven MAS. Построй глобальный top-10 по evidence dependency graph. "
            "Сначала прямые доказательства, затем необходимые зависимости; покрой все accepted аспекты и не выбирай challenged aliases. "
            "Если граф неполон, верни gaps, но всё равно укажи 10 текущих лучших candidates. Не создавай gaps для уже покрытых аспектов, "
            "гипотетических уточнений или вопросов вне исходного запроса; условные варианты ответа не требуют единственного универсального документа. "
            "Всегда возвращай ровно 10 ranked_aliases при наличии 10 candidates. "
            "Только JSON: {\"ranked_aliases\":[\"C001\",\"C002\"],"
            "\"gaps\":[{\"aspect\":\"...\",\"need\":\"...\",\"kind\":\"...\"}],\"reason\":\"...\"}."
        )
        human = (
            f"Вопрос: {runtime.query}\nEvidence graph: {json.dumps(runtime.aspects, ensure_ascii=False)}\n"
            f"Claims: {json.dumps(runtime.claims, ensure_ascii=False)}\n"
            f"Challenged: {json.dumps(runtime.challenged, ensure_ascii=False)}\n"
            f"Pool:\n{runtime.pool.format(runtime.pool.ids)}"
        )
        def validate(value):
            valid = set()
            for alias in value.get("ranked_aliases", []):
                alias = str(alias)
                chunk_id = runtime.pool.chunk_id(alias)
                if chunk_id and alias not in runtime.challenged:
                    valid.add(chunk_id)
            expected = min(MAX_OUTPUT_CHUNKS, len(runtime.pool.ids))
            if expected and not valid:
                return "Нужен хотя бы один валидный уникальный ranked_alias"
            return ""

        value, meta = await self._json(
            runtime, system, human, ("ranked_aliases", "gaps", "reason"), validator=validate,
        )
        if value is None:
            runtime.failures.append({"agent": self.name, "stage": "synthesizer", "error": meta["error"]})
            return [runtime.message("synth_error", self.name, "blackboard", {"error": meta["error"]}, message["message_id"])]
        value["meta"] = meta
        return [runtime.message("final_selection", self.name, "blackboard", value, message["message_id"], tokens=meta["tokens"], repair=meta["repair"])]


class DynamicMASRuntime:
    def __init__(self, query, engine, agents):
        self.query = query
        self.engine = engine
        self.pool = SharedCandidatePool()
        self.agents = agents
        self.queue = deque()
        self.messages = []
        self.message_log = []
        self.execution_trace = []
        self.message_new_chunks = {}
        self.aspects = {}
        self.contracts = {}
        self.claims = {}
        self.challenged = {}
        self.failures = []
        self.stop_votes = {}
        self.stop_decisions = {}
        self.stop_expected = {}
        self.llm_tokens = 0
        self.initial_reviewed = set()
        self.final_ids = None
        self.abort = False
        self.final_gaps = []
        self.ignored_final_gaps = []
        self.synthesis_revisions = 0
        self.termination = "event_queue_empty"
        self.ids = itertools.count(1)
        self.contract_ids = itertools.count(1)
        self.proposal_ids = itertools.count(1)

    def message(self, type_, sender, recipient, payload, caused_by=None, tokens=0, repair=False):
        message = {
            "message_id": f"M{next(self.ids):04d}", "type": type_, "sender": sender,
            "recipient": recipient, "payload": payload, "caused_by": caused_by,
            "tokens": tokens, "repair": repair,
        }
        self.message_log.append(message)
        return message

    def post(self, message):
        self.messages.append(message)
        if message["sender"] in self.agents:
            self.agents[message["sender"]].memory.sent.append(message["message_id"])
        type_ = message["type"]
        if type_ == "peer_request":
            self.open_gap(message["payload"], message["sender"], message["message_id"], preferred=message["recipient"])
        elif type_ == "evidence_report":
            report = message["payload"]["report"]
            contract_id = message["payload"].get("contract_id")
            if contract_id in self.contracts:
                self.contracts[contract_id].status = "completed"
            for claim in report["claims"]:
                claim["agent"] = report["agent"]
                self.claims.setdefault(claim["alias"], claim)
            if message["payload"].get("source_type") in ("initial_evidence", "award") or report["claims"] or report["gaps"]:
                self.queue.append(self.message("evidence_report", "blackboard", "critic", message["payload"], message["message_id"]))
            other = "bm25" if report["agent"] == "dense" else "dense"
            if report["agent"] in ("dense", "bm25") and message["payload"].get("source_type") == "initial_evidence":
                self.queue.append(self.message("peer_report", report["agent"], other, {"report": report}, message["message_id"]))
        elif type_ == "critic_verdict":
            self.apply_critic(message)
        elif type_ in ("bid", "refuse"):
            self.apply_bid(message)
        elif type_ == "synth_error":
            self.abort = True
            self.termination = "synthesis_failed"
        elif type_ in ("contract_error", "agent_error"):
            if type_ == "contract_error":
                contract = self.contracts.get(message["payload"].get("contract_id"))
                if contract:
                    contract.status = "failed"
            self.queue.append(self.message(type_, "blackboard", "critic", message["payload"], message["message_id"]))
        elif type_ == "stop_proposal":
            self.start_stop_vote(message)
        elif type_ == "stop_ballot":
            self.apply_ballot(message)
        elif type_ == "final_selection":
            self.apply_final(message)

    def has_valid_coverage(self, state) -> bool:
        return state.get("status") == "covered" and any(
            alias in self.claims
            and self.claims[alias].get("accepted")
            and alias not in self.challenged
            for alias in state.get("covered_by", [])
        )

    def open_gap(self, gap, origin, caused_by, preferred=None):
        aspect = str(gap.get("aspect") or gap.get("need") or "dependency")
        key = _norm(aspect)
        if not key:
            return
        state = self.aspects.get(key, {"aspect": aspect, "need": gap.get("need", aspect), "kind": gap.get("kind", "semantic"), "status": "open", "covered_by": []})
        if self.has_valid_coverage(state):
            return
        state["status"] = "open"
        state["need"] = gap.get("need", state["need"])
        self.aspects[key] = state
        for contract in self.contracts.values():
            if _norm(contract.aspect) == key and contract.status in ("open", "awarded"):
                return
        if len(self.contracts) >= MAX_CONTRACTS:
            state["status"] = "blocked"
            state["blocked_reason"] = "global_contract_budget"
            return
        aspect_contracts = [contract for contract in self.contracts.values() if _norm(contract.aspect) == key]
        if len(aspect_contracts) >= 2:
            if not self.has_valid_coverage(state):
                state["status"] = "blocked"
                state["blocked_reason"] = "contract_limit_without_valid_coverage"
            return
        kind = str(gap.get("kind", "semantic"))
        if kind in ("exception", "cross_reference"):
            specialist = f"specialist_{kind}"
            if specialist not in self.agents:
                self.agents[specialist] = DynamicMASAgent(specialist, "dense" if kind == "cross_reference" else "bm25", self.agents["dense"].llm)
        participants = [preferred] if preferred else [name for name, agent in self.agents.items() if name in ("dense", "bm25") or name.startswith("specialist_")]
        contract_id = f"K{next(self.contract_ids):03d}"
        contract = Contract(contract_id, aspect, str(gap.get("need", aspect)), kind, participants, origin)
        self.contracts[contract_id] = contract
        payload = {"contract_id": contract_id, "aspect": aspect, "need": contract.need, "kind": kind, "origin": origin}
        for participant in participants:
            self.queue.append(self.message("cfp", "blackboard", participant, payload, caused_by))

    def stop_in_progress(self) -> bool:
        incomplete = any(
            len(self.stop_votes[proposal_id]) < len(self.stop_expected.get(proposal_id, set()))
            for proposal_id in self.stop_votes
        )
        return incomplete or any(message["type"] == "synthesize" for message in self.queue)

    def reconcile_aspect_contracts(self, caused_by):
        for key, state in list(self.aspects.items()):
            if state["status"] != "open":
                continue
            contracts = [contract for contract in self.contracts.values() if _norm(contract.aspect) == key]
            if any(contract.status in ("open", "awarded") for contract in contracts):
                continue
            if len(contracts) >= 2:
                state["status"] = "blocked"
                state["blocked_reason"] = "no_valid_coverage_after_two_contracts"
            else:
                self.open_gap({
                    "aspect": state["aspect"], "need": state["need"],
                    "kind": state.get("kind", "semantic"),
                }, "critic_reconciliation", caused_by)

    def apply_critic(self, message):
        value = message["payload"]
        if value.get("initial_review") and value.get("report_agent"):
            self.initial_reviewed.add(value["report_agent"])
        for alias in value.get("accepted_aliases", []):
            if alias in self.claims:
                self.claims[alias]["accepted"] = True
        for item in value.get("challenged", []):
            if isinstance(item, dict) and item.get("alias"):
                self.challenged[item["alias"]] = str(item.get("reason", ""))
                if item["alias"] in self.claims:
                    self.claims[item["alias"]]["accepted"] = False
                for state in self.aspects.values():
                    if item["alias"] in state.get("covered_by", []):
                        state["covered_by"] = [alias for alias in state["covered_by"] if alias != item["alias"]]
                        if not state["covered_by"]:
                            state["status"] = "open"
        for item in value.get("aspects", []):
            if not isinstance(item, dict) or not item.get("aspect"):
                continue
            key = _norm(item["aspect"])
            state = self.aspects.setdefault(key, {"aspect": item["aspect"], "need": item["aspect"], "status": "open", "covered_by": []})
            existing = list(dict.fromkeys(state.get("covered_by", [])))
            for alias in item.get("covered_by", []):
                if alias in self.claims and alias not in self.challenged:
                    self.claims[alias]["accepted"] = True
                    if alias not in existing:
                        existing.append(alias)
            state["covered_by"] = [
                alias for alias in existing
                if alias in self.claims and self.claims[alias].get("accepted") and alias not in self.challenged
            ]
            if item.get("status") == "covered" and state["covered_by"]:
                state["status"] = "covered"
            elif item.get("status") == "open" and not self.has_valid_coverage(state):
                state["status"] = "open"
            elif not state["covered_by"]:
                state["status"] = "open"
        for gap in value.get("gaps", []):
            if isinstance(gap, dict) and gap.get("need"):
                self.open_gap(gap, "critic", message["message_id"])
        self.reconcile_aspect_contracts(message["message_id"])
        terminal_graph = bool(self.aspects) and all(
            state["status"] in ("covered", "blocked") for state in self.aspects.values()
        )
        if value.get("stop") or terminal_graph:
            open_contracts = any(contract.status in ("open", "awarded") for contract in self.contracts.values())
            if not open_contracts and not self.open_aspects() and self.initial_reviewed >= {"dense", "bm25"} and not self.stop_in_progress():
                proposal = self.message("stop_proposal", "critic", "blackboard", {"proposal_id": f"S{next(self.proposal_ids):03d}"}, message["message_id"])
                self.post(proposal)

    def apply_bid(self, message):
        payload = message["payload"]
        contract = self.contracts[payload["contract_id"]]
        if contract.status != "open":
            return
        contract.bids[message["sender"]] = payload
        if set(contract.bids) != set(contract.participants):
            return
        bids = [payload | {"agent": name} for name, payload in contract.bids.items() if "utility" in payload]
        if not bids:
            contract.status = "failed"
            self.queue.append(self.message("contract_error", "blackboard", "critic", {"contract_id": contract.contract_id, "error": "all_agents_refused"}, message["message_id"]))
            return
        winner = max(bids, key=lambda bid: (bid["utility"], bid["coverage"], bid["confidence"], -bid["cost"]))
        contract.status = "awarded"
        contract.awarded_to = winner["agent"]
        self.queue.append(self.message("award", "blackboard", winner["agent"], {
            "contract_id": contract.contract_id, "query": winner["query"], "bid": winner,
        }, message["message_id"]))

    def start_stop_vote(self, message):
        proposal_id = message["payload"]["proposal_id"]
        self.stop_votes[proposal_id] = {}
        expected = {name for name in self.agents if name in ("dense", "bm25") or name.startswith("specialist_")}
        self.stop_expected[proposal_id] = expected
        for name in sorted(expected):
                self.queue.append(self.message("stop_vote", "blackboard", name, {"proposal_id": proposal_id}, message["message_id"]))

    def apply_ballot(self, message):
        payload = message["payload"]
        proposal = self.stop_votes[payload["proposal_id"]]
        proposal[message["sender"]] = payload
        if payload.get("gap"):
            self.open_gap(payload["gap"], message["sender"], message["message_id"])
        expected = self.stop_expected[payload["proposal_id"]]
        if set(proposal) != expected:
            return
        decisive = [ballot for ballot in proposal.values() if not ballot.get("abstain")]
        accept_votes = sum(ballot["accept"] for ballot in decisive)
        reject_votes = len(decisive) - accept_votes
        open_contracts = any(contract.status in ("open", "awarded") for contract in self.contracts.values())
        accepted = accept_votes >= reject_votes and not open_contracts
        self.stop_decisions[payload["proposal_id"]] = {
            "accept_votes": accept_votes, "reject_votes": reject_votes,
            "abstentions": len(proposal) - len(decisive),
            "proposer_vote": "accept", "accepted": accepted,
            "rule": "critic proposer breaks scout tie; new active contract blocks synthesis",
        }
        if accepted:
            self.queue.append(self.message("synthesize", "blackboard", "synthesizer", {"proposal_id": payload["proposal_id"]}, message["message_id"]))

    def apply_final(self, message):
        value = message["payload"]
        self.final_gaps = []
        allow_revision = self.synthesis_revisions < 1 and len(self.contracts) < MAX_CONTRACTS
        for gap in value.get("gaps", []):
            if isinstance(gap, dict) and gap.get("need"):
                aspect = str(gap.get("aspect") or gap["need"])
                state = self.aspects.get(_norm(aspect), {})
                if self.has_valid_coverage(state) or state.get("status") == "blocked":
                    continue
                if not allow_revision:
                    self.ignored_final_gaps.append(gap)
                    continue
                self.final_gaps.append(gap)
                self.open_gap(gap, "synthesizer", message["message_id"])
        if self.final_gaps:
            self.synthesis_revisions += 1
            return
        selected = []
        seen = set()
        challenged_ids = {self.pool.chunk_id(alias) for alias in self.challenged}
        for alias in value.get("ranked_aliases", []):
            alias = str(alias)
            chunk_id = self.pool.chunk_id(alias)
            if not chunk_id or chunk_id in seen or chunk_id in challenged_ids:
                continue
            seen.add(chunk_id)
            claim = self.claims.get(alias, {})
            aspect = str(claim.get("aspect", "")).strip()
            if not aspect:
                aspect = next(
                    (state["aspect"] for state in self.aspects.values() if alias in state.get("covered_by", [])),
                    "evidence",
                )
            relevance = claim.get("relevance") if claim.get("relevance") in (1, 2) else 1
            selected.append({"chunk_id": chunk_id, "alias": alias, "relevance": relevance, "aspect": aspect})
        selected = selected[:MAX_OUTPUT_CHUNKS]
        if not selected and self.pool.ids:
            self.failures.append({"agent": "synthesizer", "stage": "validation", "error": "empty_selection"})
            return
        self.final_ids = selected

    def open_aspects(self):
        return [value for value in self.aspects.values() if value["status"] == "open"]

    async def run(self):
        while self.queue and not self.final_ids and not self.abort and len(self.execution_trace) < MAX_AGENT_TURNS:
            message = self.queue.popleft()
            batch = [message]
            parallel_types = {"initial_evidence", "cfp", "peer_report", "stop_vote"}
            while self.queue and message["type"] in parallel_types:
                candidate = self.queue[0]
                if candidate["type"] != message["type"] or candidate["recipient"] in {item["recipient"] for item in batch}:
                    break
                batch.append(self.queue.popleft())
            results = await asyncio.gather(*(self.agents[item["recipient"]].handle(item, self) for item in batch))
            for incoming, outgoing in zip(batch, results):
                self.execution_trace.append({
                    "agent": incoming["recipient"], "incoming_message": incoming["message_id"],
                    "outgoing_messages": [item["message_id"] for item in outgoing],
                    "tokens": sum(item.get("tokens", 0) for item in outgoing),
                })
                for item in outgoing:
                    self.post(item)
        if self.abort and self.final_ids is None:
            self.termination = "synthesis_failed"
        elif self.final_ids is not None:
            self.termination = "final_selection"
        elif self.queue:
            self.termination = "max_agent_turns"
        else:
            self.termination = "event_queue_without_final"
        return self.final_ids


class DynamicMASRetriever:
    def __init__(self, engine: RetrievalEngine | None = None, dense_llm=None, bm25_llm=None, critic_llm=None, synth_llm=None):
        self.engine = engine or RetrievalEngine()
        self.dense_llm = dense_llm if dense_llm is not None else self._llm("dense")
        self.bm25_llm = bm25_llm if bm25_llm is not None else self._llm("bm25")
        self.critic_llm = critic_llm if critic_llm is not None else self._llm("dense")
        self.synth_llm = synth_llm if synth_llm is not None else self._llm("dense")

    @staticmethod
    def _llm(channel):
        dense = channel == "dense"
        return create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL if dense else RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL if dense else RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE if dense else RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS if dense else RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING if dense else RETRIEVER_BM25_ENABLE_THINKING,
        )

    async def retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine = self.engine.with_budget(SearchBudget())
        agents = {
            "dense": DynamicMASAgent("dense", "dense", self.dense_llm),
            "bm25": DynamicMASAgent("bm25", "bm25", self.bm25_llm),
            "critic": DynamicMASAgent("critic", None, self.critic_llm),
            "synthesizer": DynamicMASAgent("synthesizer", None, self.synth_llm),
        }
        runtime = DynamicMASRuntime(query, engine, agents)
        try:
            dense_hits, lexical_hits = await asyncio.gather(
                asyncio.to_thread(engine.search, "dense", query, 15),
                asyncio.to_thread(engine.search, "bm25", query, 15),
            )
        except Exception as exc:
            return {
                "chunks": [], "retriever": "dynamic_mas", "fallback_used": True,
                "error": f"Initial retrieval failed: {exc}", "trace": {}, "token_usage": {"total_tokens": 0},
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        for rank in range(15):
            if rank < len(dense_hits):
                runtime.pool.add(dense_hits[rank].chunk, "dense", query, 0, rank + 1)
            if rank < len(lexical_hits):
                runtime.pool.add(lexical_hits[rank].chunk, "bm25", query, 0, rank + 1)
        for name, hits in (("dense", dense_hits), ("bm25", lexical_hits)):
            for hit in hits[:10]:
                alias = runtime.pool.alias(hit.chunk.id)
                if alias not in agents[name].memory.known_aliases:
                    agents[name].memory.known_aliases.append(alias)
                runtime.pool.ledger[hit.chunk.id]["observed_by"].append(name)
        runtime.queue.append(runtime.message("initial_evidence", "environment", "dense", {"query": query}))
        runtime.queue.append(runtime.message("initial_evidence", "environment", "bm25", {"query": query}))
        selected = await runtime.run()
        selected = selected or []
        selected_ids = [item["chunk_id"] for item in selected]
        accepted_ranking = sorted(
            (claim for claim in runtime.claims.values() if claim.get("accepted") and claim["alias"] not in runtime.challenged),
            key=lambda claim: (-claim["relevance"], claim["alias"]),
        )
        worker_ids = [runtime.pool.chunk_id(claim["alias"]) for claim in accepted_ranking]
        backfill = [chunk_id for chunk_id in worker_ids if chunk_id and chunk_id not in selected_ids]
        backfill += [chunk_id for chunk_id in runtime.pool.ids if chunk_id not in selected_ids and chunk_id not in backfill]
        backfill_count = 0
        for chunk_id in backfill:
            if len(selected_ids) >= MAX_OUTPUT_CHUNKS:
                break
            selected_ids.append(chunk_id)
            backfill_count += 1
        selected_by_id = {item["chunk_id"]: item for item in selected}
        chunks = []
        for rank, chunk_id in enumerate(selected_ids[:MAX_OUTPUT_CHUNKS], 1):
            item = selected_by_id.get(chunk_id, {})
            alias = runtime.pool.alias(chunk_id)
            claim = runtime.claims.get(alias, {})
            chunks.append(chunk_dict(
                runtime.pool.chunks[chunk_id], rank,
                relevance=item.get("relevance", claim.get("relevance")),
                answer_aspect=item.get("aspect", claim.get("aspect", "")),
            ))
            runtime.pool.ledger[chunk_id]["dynamic_mas"] = {
                "final_rank": rank, "selected_by_synthesizer": chunk_id in selected_by_id,
                "backfilled": chunk_id not in selected_by_id,
            }
        total_tokens = runtime.llm_tokens
        active_edges = sorted({(message["sender"], message["recipient"]) for message in runtime.message_log})
        return {
            "chunks": chunks,
            "retriever": "dynamic_mas",
            "fallback_used": runtime.final_ids is None or not chunks,
            "error": None if chunks else f"Synthesis did not complete: {runtime.termination}",
            "trace": {
                "query": query,
                "protocol": "event_driven_contract_net_evidence_graph",
                "termination": runtime.termination,
                "synthesis_completed": runtime.final_ids is not None,
                "synthesis_revisions": runtime.synthesis_revisions,
                "ignored_final_gaps": runtime.ignored_final_gaps,
                "evidence_graph": runtime.aspects,
                "contracts": {
                    key: {
                        "aspect": value.aspect, "kind": value.kind, "participants": value.participants,
                        "origin": value.origin, "bids": value.bids, "status": value.status,
                        "awarded_to": value.awarded_to,
                    } for key, value in runtime.contracts.items()
                },
                "claims": runtime.claims,
                "challenged": runtime.challenged,
                "stop_votes": runtime.stop_votes,
                "stop_decisions": runtime.stop_decisions,
                "messages": runtime.message_log,
                "posted_agent_messages": runtime.messages,
                "execution_trace": runtime.execution_trace,
                "message_new_chunks": runtime.message_new_chunks,
                "active_agents": list(runtime.agents),
                "active_edges": active_edges,
                "failures": runtime.failures,
                "degraded": bool(runtime.failures),
                "recovered_errors": bool(runtime.failures) and runtime.final_ids is not None,
                "initial_reviewed": sorted(runtime.initial_reviewed),
                "backfill_count": backfill_count,
                "selection_sanitized": backfill_count > 0,
                "partial_coverage": any(state["status"] == "blocked" for state in runtime.aspects.values()),
                "unresolved_aspects": [state for state in runtime.aspects.values() if state["status"] == "blocked"],
                "candidate_ledger": runtime.pool.ledger,
                "agent_memory": {
                    name: {
                        "received": agent.memory.received, "sent": agent.memory.sent,
                        "known_aliases": agent.memory.known_aliases, "actions": agent.memory.actions,
                    } for name, agent in runtime.agents.items()
                },
                "information_budget": engine.budget.snapshot(),
            },
            "token_usage": {"total_tokens": total_tokens},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
