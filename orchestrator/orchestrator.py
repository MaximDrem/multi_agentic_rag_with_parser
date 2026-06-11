import json
import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext
from a2a.helpers.proto_helpers import new_data_artifact_update_event

from common.config import (
    ORCHESTRATOR_LLM_BASE_URL,
    ORCHESTRATOR_LLM_MODEL,
    ORCHESTRATOR_LLM_TEMPERATURE,
    ORCHESTRATOR_LLM_MAX_TOKENS,
    ORCHESTRATOR_ENABLE_THINKING,
    create_llm,
    get_reasoning,
)
from common.a2a_client import A2AClientManager
from orchestrator.discovery import AgentRegistry, AgentDiscoverer, TaskTracker

logger = logging.getLogger(__name__)

_DYNAMIC_PROMPT_TEMPLATE = (
    "Ты — оркестратор. Твоя задача — построить цепочку агентов от запроса пользователя до финального ответа. "
    "Ты можешь обрабатывать несколько вопросов/задач последовательно в рамках одного сообщения. "
    "Как только агент выполнен, ты видишь тип его Output. "
    "Если Output не финальный — НАЙДИ агента с подходящим Input среди доступных и вызови его. "
    "Когда результат готов — отправь его пользователю. "
    "Затем можешь перейти к следующему вопросу или завершить работу.\n\n"
    "Когда отвечать напрямую без агентов:\n"
    "- приветствие, вопрос о твоих возможностях, мета-вопрос о системе\n"
    "- запрос не по нормативно-правовым документам РФ\n"
    "- простой вопрос не требующий поиска\n\n"
    "Агенты с тегом 'shared-corpus' работают с одним корпусом — вызывай обоих для полноты.\n"
    "Агенты с тегом 'intermediate' выдают промежуточный артефакт — не останавливайся на них.\n"
    "Агенты с тегом 'final' выдают готовый ответ — это терминальная точка цепочки.\n\n"
    "Агенты:\n{agent_cards_summary}"
)


class AgentInput(BaseModel):
    query: str = Field(description="Запрос к агенту")
    data_from: list[str] | None = Field(default=None, description="Имена агентов, чьи данные нужно передать этому агенту. Пример: ['retriever_embedding', 'retriever_bm25']. Передавай только те данные, которые реально нужны агенту. None или [] — ничего не передавать")


class ForwardAgentArtifactsInput(BaseModel):
    agent_name: str = Field(description="Имя агента, чьи артефакты нужно отправить пользователю")


class SendTextInput(BaseModel):
    message: str = Field(description="Текст, который нужно отправить пользователю")


class OrchestratorAgent:
    def __init__(self):
        self.llm = create_llm(
            base_url=ORCHESTRATOR_LLM_BASE_URL,
            model=ORCHESTRATOR_LLM_MODEL,
            temperature=ORCHESTRATOR_LLM_TEMPERATURE,
            max_tokens=ORCHESTRATOR_LLM_MAX_TOKENS,
            enable_thinking=ORCHESTRATOR_ENABLE_THINKING,
        )
        self.client_manager = A2AClientManager()
        self.registry = AgentRegistry()
        self.discoverer = AgentDiscoverer(self.client_manager, self.registry)
        self._agent: Any = None
        self._agent_tool_count: int = 0
        self._initialized = False
        self._session_tracker: TaskTracker | None = None
        self._agent_contexts: dict[str, str] = {}
        self._session_context_id: str | None = None
        self._outbox: list[dict] = []
        self._stream_eq: EventQueue | None = None
        self._stream_task_id: str = ""
        self._stream_ctx_id: str = ""
        self._stream_seq: int = 0
        self._text_seq: int = 0

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info("Discovering agents via A2A agent cards...")
        await self.discoverer.discover()
        self._build_agent()
        self._agent_tool_count = len(self.registry.cards)
        await self.discoverer.start_health_check(interval=60.0)
        self._initialized = True
        logger.info(f"Discovered {len(self.registry.cards)} agents, orchestrator ready")

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def _build_cards_summary(self) -> str:
        lines = []
        for url, card in self.registry.cards.items():
            safe_name = self._safe_name(card.name)
            lines.append(f"  Инструмент: {safe_name}")
            lines.append(f"    Название: {card.name}")
            lines.append(f"    Описание: {card.description}")
            for skill in card.skills:
                lines.append(f"    Скилл: {skill.id} — {skill.name}")
                lines.append(f"      Назначение: {skill.description}")
                if skill.input_modes:
                    lines.append(f"      Принимает: {', '.join(skill.input_modes)}")
                if skill.output_modes:
                    lines.append(f"      Отдаёт: {', '.join(skill.output_modes)}")
                if skill.examples:
                    lines.append(f"      Примеры запросов: {', '.join(skill.examples[:3])}")
                if skill.tags:
                    lines.append(f"      Тэги: {', '.join(skill.tags)}")
            lines.append("")
        return "\n".join(lines) if lines else "(нет доступных агентов)"

    def _build_prompt(self) -> str:
        return _DYNAMIC_PROMPT_TEMPLATE.format(
            agent_cards_summary=self._build_cards_summary()
        )

    def _build_agent(self) -> None:
        tools: list = []

        forward_tool = StructuredTool.from_function(
            name="forward_agent_artifacts",
            description="Переслать пользователю артефакты указанного агента. Можно вызывать многократно. Вернёт 'SENT: ...' — продолжай или напиши 'OK' для завершения.",
            coroutine=self._forward_agent_artifacts,
            args_schema=ForwardAgentArtifactsInput,
        )
        tools.append(forward_tool)

        send_text_tool = StructuredTool.from_function(
            name="send_text",
            description="Отправить пользователю текст оркестратора. Можно вызывать многократно. Вернёт 'SENT: ...' — продолжай или напиши 'OK' для завершения.",
            coroutine=self._send_text,
            args_schema=SendTextInput,
        )
        tools.append(send_text_tool)

        for url, card in self.registry.cards.items():
            safe_name = self._safe_name(card.name)
            agent_url = url
            skill_desc = card.skills[0].description if card.skills else card.description

            def make_tool(_url: str, _safe_name: str, _desc: str):
                async def invoke_impl(query: str, data_from: list[str] | None = None) -> str:
                    try:
                        return await self._call_agent(_url, _safe_name, query, data_from=data_from)
                    except Exception as e:
                        logger.error(f"Agent {_safe_name} failed: {e}")
                        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

                return StructuredTool.from_function(
                    name=_safe_name, description=_desc,
                    coroutine=invoke_impl, args_schema=AgentInput,
                )

            tool_instance = make_tool(agent_url, safe_name, skill_desc)
            tools.append(tool_instance)
            logger.info(f"Tool '{safe_name}' → {agent_url}")

        system_prompt = self._build_prompt()
        self._agent = create_react_agent(
            self.llm, tools,
            prompt=SystemMessage(content=system_prompt),
        )
        logger.info(f"Orchestrator built with {len(tools)} tool(s)")

    async def _call_agent(self, url: str, safe_name: str, query: str, data_from: list[str] | None = None) -> str:
        data_payload: dict[str, Any] = {"query": query}
        ref_task_ids: list[str] = []

        if self._session_tracker and data_from:
            all_chunks: list[dict] = []
            for entry in self._session_tracker.tasks.values():
                entry_name = entry.get("safe_name", "")
                for df in data_from:
                    if self._safe_name(df) == entry_name or df.lower() in entry_name:
                        all_chunks.extend(A2AClientManager.extract_chunks_from_artifacts(entry["artifact_dicts"]))
                        break
            if all_chunks:
                data_payload["chunks"] = all_chunks
                logger.info(f"→ {safe_name}: passing {len(all_chunks)} chunks from {data_from}")

        if self._session_tracker:
            for entry in self._session_tracker.tasks.values():
                if entry["agent_url"] != url:
                    ref_task_ids.append(entry["task_id"])

        context_id = self._agent_contexts.get(url) or self._session_context_id
        logger.info(f"→ {safe_name}: sending A2A request (context={context_id}, ref_tasks={ref_task_ids})")

        task_id, new_context_id, artifacts = await self.client_manager.send_message(
            url, text=query, data=data_payload, context_id=context_id, reference_task_ids=ref_task_ids,
        )
        self._agent_contexts[url] = new_context_id
        logger.info(f"← {safe_name}: task_id={task_id}, context={new_context_id}, {len(artifacts)} artifacts")

        if self._session_tracker:
            self._session_tracker.record(task_id, new_context_id, url, artifacts, safe_name)

        result = self._build_metadata_result(safe_name, artifacts)
        logger.info(f"← {safe_name} result: {result}")
        return result

    async def _forward_agent_artifacts(self, agent_name: str) -> str:
        safe_name = self._safe_name(agent_name)
        logger.info(f"▲ forward_agent_artifacts called for agent '{safe_name}'")
        if not self._session_tracker or not self._session_tracker.tasks:
            return "Nothing to send — no tasks in session."
        for entry in reversed(self._session_tracker.tasks.values()):
            entry_name = entry.get("safe_name", "")
            if entry_name == safe_name or agent_name.lower() in entry_name:
                artifacts = OrchestratorAgent._task_artifacts_to_response(entry["artifact_dicts"])
                for art in artifacts:
                    await self._stream_artifact(art)
                self._outbox.extend(artifacts)
                n = len(artifacts)
                logger.info(f"▲ forward_agent_artifacts: {n} artifacts from '{safe_name}'")
                return f"SENT: {n} artifact(s) from '{safe_name}'. Continue or write 'OK' to finish."
        known = [e.get("safe_name", "?") for e in self._session_tracker.tasks.values()]
        return f"Agent '{safe_name}' not found. Known: {known}."

    async def _send_text(self, message: str) -> str:
        self._text_seq += 1
        logger.info(f"▲ send_text #{self._text_seq}: {message[:100]}...")
        art = {"artifactId": f"orchestrator_text_{self._text_seq}", "data": {"answer": message}, "name": "orchestrator_text"}
        await self._stream_artifact(art)
        self._outbox.append(art)
        logger.info(f"▲ send_text #{self._text_seq}: ({len(message)} chars)")
        return "SENT: text delivered. Continue or write 'OK' to finish."

    async def _stream_artifact(self, art: dict) -> None:
        if self._stream_eq is None:
            return
        self._stream_seq += 1
        art_id = f"{art.get('artifactId', 'artifact')}_{self._stream_seq}"
        try:
            await self._stream_eq.enqueue_event(
                new_data_artifact_update_event(
                    task_id=self._stream_task_id,
                    context_id=self._stream_ctx_id,
                    name=art.get("name", art.get("artifactId", "result")),
                    data=art.get("data", {}),
                    media_type="application/json",
                    last_chunk=True,
                    artifact_id=art_id,
                )
            )
            logger.debug(f"  streamed artifact {art_id}")
        except Exception as e:
            logger.warning(f"  stream failed for {art_id}: {e}")

    @staticmethod
    def _build_metadata_result(safe_name: str, artifacts: list[dict]) -> str:
        for art in artifacts:
            for p in art.get("parts", []):
                data = p.get("data", {})
                if not isinstance(data, dict):
                    continue
                if "answer" in data:
                    return f"[FINAL from {safe_name}] answer ready ({len(str(data['answer']))} chars)"
                if "chunks" in data:
                    return f"[from {safe_name}] {len(data['chunks'])} chunks ready"
                return f"[from {safe_name}] artifact with keys: {list(data.keys())}"
        return f"[from {safe_name}] {len(artifacts)} artifact(s)"

    def _maybe_rebuild_tools(self) -> None:
        current_count = len(self.registry.cards)
        if current_count != self._agent_tool_count or self._agent is None:
            logger.info(f"Agents changed ({self._agent_tool_count} → {current_count}), rebuilding tools")
            self._build_agent()
            self._agent_tool_count = current_count

    async def process_query(self, query: str, user_data: dict | None = None, context_id: str | None = None) -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        self._maybe_rebuild_tools()
        self._session_tracker = TaskTracker()
        self._outbox = []
        self._stream_seq = 0
        self._text_seq = 0
        self._agent_contexts.clear()
        if context_id:
            self._session_context_id = context_id
        logger.info(f"=== ORCHESTRATOR QUERY: {query[:100]} ===")

        try:
            logger.info("→ LLM: invoking ReAct agent")
            result = await self._agent.ainvoke({"messages": [HumanMessage(content=query)]})
            messages = result.get("messages", [])
            for i, msg in enumerate(messages):
                mtype = type(msg).__name__
                content = str(getattr(msg, "content", ""))
                tool_calls = getattr(msg, "tool_calls", None)
                reasoning = get_reasoning(msg)
                if reasoning:
                    logger.info(f"  LLM msg[{i}] {mtype} REASONING:\n{reasoning}")
                if tool_calls:
                    for tc in tool_calls:
                        logger.info(f"  LLM msg[{i}] {mtype}: tool_call={tc.get('name','?')}({str(tc.get('args',{}))})")
                elif content:
                    logger.info(f"  LLM msg[{i}] {mtype}:\n{content}")

            if self._outbox:
                if self._stream_eq:
                    logger.info(f"Streamed {len(self._outbox)} artifacts live, returning empty")
                    last_artifacts = []
                else:
                    last_artifacts = self._outbox
                    logger.info(f"Outbox: {len(last_artifacts)} artifacts (no streaming)")
            elif self._session_tracker and self._session_tracker.tasks:
                logger.warning("Sub-agent tasks but no forward_agent_artifacts/send_text")
                last_artifacts = [{"artifactId": "err", "data": {"answer": "Оркестратор не вызвал forward_agent_artifacts или send_text."}}]
            else:
                text = self._extract_llm_text(result) or "Не удалось сформировать ответ."
                logger.info(f"LLM direct answer: {text[:200]}")
                self._text_seq += 1
                last_artifacts = [{"artifactId": f"llm_answer_{self._text_seq}", "data": {"answer": text}, "name": "llm_answer"}]

            return {"status": "completed", "message": "Completed", "artifacts": last_artifacts}
        except Exception as e:
            logger.exception("Orchestrator failed")
            return {"status": "failed", "message": str(e), "artifacts": []}

    @staticmethod
    def _task_artifacts_to_response(artifact_dicts: list[dict]) -> list[dict]:
        result: list[dict] = []
        for art_dict in artifact_dicts:
            a_data: dict = {}
            a_name = art_dict.get("name", "")
            for p in art_dict.get("parts", []):
                if "data" in p and isinstance(p["data"], dict):
                    a_data.update(p["data"])
                elif "text" in p and p.get("text"):
                    a_data["text"] = p["text"]
            item = {"artifactId": art_dict.get("artifactId", art_dict.get("artifact_id", "unknown")), "data": a_data}
            if a_name:
                item["name"] = a_name
            if art_dict.get("description"):
                item["description"] = art_dict["description"]
            result.append(item)
        return result

    @staticmethod
    def _extract_llm_text(langgraph_result: dict) -> str:
        for msg in reversed(langgraph_result.get("messages", [])):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return msg.content
        return ""

    async def handle_a2a_request(self, user_text: str, user_data: dict | None, context: RequestContext, event_queue: EventQueue) -> dict[str, Any]:
        self._stream_eq = event_queue
        self._stream_task_id = context.task_id
        self._stream_ctx_id = context.context_id
        query = user_text
        if isinstance(user_data, dict):
            query = user_data.get("query", user_text)
        return await self.process_query(query, user_data=user_data, context_id=context.context_id)

    async def shutdown(self):
        await self.discoverer.stop_health_check()
        await self.client_manager.close()
