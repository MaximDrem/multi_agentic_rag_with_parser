import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from a2a.types import AgentSkill
from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext

from common.config import (
    SUMMARIZER_PORT, SUMMARIZER_URL,
    SUMMARIZER_LLM_BASE_URL,
    SUMMARIZER_LLM_MODEL,
    SUMMARIZER_LLM_TEMPERATURE,
    SUMMARIZER_LLM_MAX_TOKENS,
    SUMMARIZER_ENABLE_THINKING,
    CHUNKS_PATH,
    create_llm,
    get_reasoning,
)
from common.chunk_store import ChunkDatabase
from common.a2a_server import A2AServer, A2AExecutor
from common.benchmark_logger import tokens_from_usage_metadata
from common.logging_utils import setup_logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — экспертный помощник по нормативно-правовым документам РФ в сфере госпрограмм и нацпроектов. "
    "Тебе предоставлены релевантные фрагменты документов и вопрос пользователя. "
    "Дай развёрнутый, точный ответ на основе предоставленных фрагментов. "
    "Не пиши смайлики, Отвечай ТОЛЬКО на русском."
    "Если в фрагментах есть ссылки на конкретные статьи, пункты, постановления — обязательно укажи их. "
    "В конце ответа перечисли использованные источники в формате: [1] Название документа. "
    "Если информации недостаточно, честно скажи об этом."
)

MAX_CONTEXT_CHUNKS = 50

_agent_instance: "SummarizerAgent | None" = None


class SummarizerAgent:
    def __init__(self):
        logger.info("Loading chunk database...")
        self.db = ChunkDatabase.load(CHUNKS_PATH)
        self.llm = create_llm(
            base_url=SUMMARIZER_LLM_BASE_URL,
            model=SUMMARIZER_LLM_MODEL,
            temperature=SUMMARIZER_LLM_TEMPERATURE,
            max_tokens=SUMMARIZER_LLM_MAX_TOKENS,
            enable_thinking=SUMMARIZER_ENABLE_THINKING,
        )
        logger.info(f"Summarizer ready, {len(self.db.chunks)} chunks in DB")

    def _enrich_chunks(self, chunks_data: list[dict]) -> list:
        enriched = []
        seen = set()
        for cd in chunks_data:
            cid = cd.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            chunk = self.db.get_by_id(cid)
            if chunk:
                enriched.append({
                    "chunk_id": chunk.id,
                    "doc_path": chunk.doc_path,
                    "title": chunk.title,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                })
                for nb in self.db.get_neighbors_recursive(cid, depth=1):
                    if nb.id not in seen:
                        seen.add(nb.id)
                        enriched.append({
                            "chunk_id": nb.id,
                            "doc_path": nb.doc_path,
                            "title": nb.title,
                            "text": nb.text,
                            "token_count": nb.token_count,
                        })
        return enriched

    async def run(self, query: str, chunks: list[dict]) -> dict:
        logger.info(f"=== SUMMARIZER: query={query[:80]}, chunks={len(chunks)} ===")
        enriched = self._enrich_chunks(chunks)
        if len(enriched) > MAX_CONTEXT_CHUNKS:
            enriched = enriched[:MAX_CONTEXT_CHUNKS]
        logger.info(f"  enriched to {len(enriched)} chunks")

        if not enriched:
            logger.info("  no chunks → returning empty answer")
            return {
                "answer": "Не найдено релевантных документов для ответа на вопрос.",
                "sources": [],
                "chunks_used": [],
                "token_usage": {"total_tokens": 0},
            }

        context_texts = []
        sources = []
        for i, chunk in enumerate(enriched, start=1):
            context_texts.append(
                f"[{i}] Источник: {chunk['title']} ({chunk['doc_path']})\n{chunk['text']}"
            )
            if chunk["title"] not in sources:
                sources.append(chunk["title"])

        context_block = "\n\n---\n\n".join(context_texts)
        prompt = (
            f"Вопрос пользователя:\n{query}\n\n"
            f"Релевантные фрагменты документов:\n\n{context_block}\n\n"
            f"Дай развёрнутый ответ на вопрос на основе предоставленных фрагментов. "
            f"Укажи конкретные нормативные акты, статьи и пункты, если они присутствуют в фрагментах. "
            f"В конце обязательно приведи список источников."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        logger.info(f"→ LLM: summarization (prompt len={len(prompt)})")
        response = await self.llm.ainvoke(messages)
        answer = response.content
        reasoning = get_reasoning(response)
        if reasoning:
            logger.info(f"  SUMMARIZER REASONING:\n{reasoning}")
        logger.info(f"← LLM: answer len={len(answer)}")

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": [c["chunk_id"] for c in enriched],
            "token_usage": {"total_tokens": tokens_from_usage_metadata(getattr(response, "usage_metadata", None))},
        }


def get_agent() -> SummarizerAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SummarizerAgent()
    return _agent_instance


async def handle_request(
    user_text: str,
    user_data: dict | None,
    context: RequestContext,
    event_queue: EventQueue,
) -> dict:
    agent = get_agent()

    chunks_data: list[dict] = []
    query = user_text

    if isinstance(user_data, dict):
        query = user_data.get("query", user_text)
        chunks_data = user_data.get("chunks", [])

    result = await agent.run(query, chunks_data)
    return {
        "status": "completed",
        "message": "Summarized",
        "artifacts": [{
            "artifactId": "final_answer",
            "data": result,
        }],
    }


def create_server():
    executor = A2AExecutor(handle_request)
    skills = [
        AgentSkill(
            id="summarize",
            name="Synthesis and answer generation",
            description="Формирует финальный ответ на основе запроса и чанков из Part.data. Возвращает ответ со ссылками на источники. Ответ готов к пересылке пользователю.",
            input_modes=["text", "data"],
            output_modes=["text", "data"],
            examples=["составь ответ по найденным фрагментам со ссылками на нормативные акты", "суммаризируй документы по вопросу с указанием источников"],
            tags=["summarization", "synthesis", "final", "user-facing"],
        )
    ]
    return A2AServer(
        name="Summarizer",
        url=SUMMARIZER_URL,
        executor=executor,
        skills=skills,
        description="Document summarization agent that produces comprehensive answers from retrieved chunks.",
    )


if __name__ == "__main__":
    import uvicorn
    setup_logging("summarizer")
    agent = create_server()
    logger.info(f"Starting {agent.name} on port {SUMMARIZER_PORT}")
    uvicorn.run(agent.app, host="0.0.0.0", port=SUMMARIZER_PORT, log_config=None)
