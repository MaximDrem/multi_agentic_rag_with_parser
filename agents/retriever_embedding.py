import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from a2a.types import AgentSkill
from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext

from common.config import (
    RETRIEVER_EMB_PORT, RETRIEVER_EMB_URL,
    CHUNKS_PATH, EMB_TOP_K, SCORE_THRESHOLD,
    RETRIEVER_EMB_LLM_BASE_URL,
    RETRIEVER_EMB_LLM_MODEL,
    RETRIEVER_EMB_LLM_TEMPERATURE,
    RETRIEVER_EMB_LLM_MAX_TOKENS,
    RETRIEVER_EMB_ENABLE_THINKING,
    create_llm,
    get_reasoning,
)
from common.chunk_store import ChunkDatabase
from common.a2a_server import A2AServer, A2AExecutor
from common.benchmark_logger import sum_tokens_from_messages
from common.logging_utils import setup_logging
from indexing.faiss_indexer import FaissIndexer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — агент итеративного семантического поиска по базе нормативно-правовых документов РФ. "
    "Твоя задача: найти чанки, которые помогут ответить на вопрос пользователя. "
    "Можешь вызывать search несколько раз (не более 5), "
    "каждый раз формулируя 1-3 варианта запроса через ; "
    "(суть вопроса, синонимы, смежные темы). "
    "Анализируй найденные чанки и при необходимости уточняй запросы, "
    "чтобы собрать наиболее полную и релевантную подборку. "
    "Когда чанков достаточно или лимит исчерпан — заверши работу."
)

_agent_instance: "EmbeddingRetrieverAgent | None" = None


class EmbeddingRetrieverAgent:
    def __init__(self):
        faiss_index_path = CHUNKS_PATH.parent / "faiss_index"
        if faiss_index_path.is_dir() or str(faiss_index_path).endswith("/"):
            faiss_index_path = faiss_index_path / "index.faiss"

        logger.info("Loading chunks and FAISS index...")
        self.db = ChunkDatabase.load(CHUNKS_PATH)
        self.indexer = FaissIndexer()
        self.indexer.load(faiss_index_path)
        logger.info(f"Loaded {len(self.db.chunks)} chunks, FAISS ready")

        self.llm = create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING,
        )

    async def run(self, query: str) -> dict:
        logger.info(f"=== EMB RETRIEVER: {query[:100]} ===")
        indexer = self.indexer
        db = self.db

        call_count = [0]
        scores: dict[str, float] = {}
        MAX_CALLS = 5

        @tool
        def search(keywords: str) -> str:
            """Семантический поиск документов (FAISS). Принимает 1-3 запроса через ; например 'соглашение; бюджетный кодекс; НПА'. Возвращает фрагменты с текстом, заголовком и путём к документу."""
            call_count[0] += 1
            if call_count[0] > MAX_CALLS:
                return f"Достигнут лимит вызовов ({MAX_CALLS}). Больше нельзя вызывать search. Проанализируй уже найденные чанки и заверши работу."
            logger.info(f"EMB search tool called (#{call_count[0]}/{MAX_CALLS}) with: {keywords}")
            queries = [q.strip() for q in keywords.split(";") if q.strip()]
            if not queries:
                queries = [keywords]
            all_chunks: dict[str, Any] = {}
            for q in queries:
                results = indexer.search(q, top_k=EMB_TOP_K, score_threshold=SCORE_THRESHOLD)
                logger.info(f"  sub-query '{q}': {len(results)} hits")
                for idx, score in results:
                    chunk = db.chunks[idx]
                    if chunk.id not in all_chunks:
                        all_chunks[chunk.id] = chunk
                    if chunk.id not in scores or score > scores[chunk.id]:
                        scores[chunk.id] = score
            logger.info(f"  unique chunks after dedup: {len(all_chunks)}")
            if not all_chunks:
                return f"По '{keywords}' ничего не найдено. Попробуй другие формулировки."
            chunks_list = list(all_chunks.values())[:25]
            lines = [f"Найдено {len(chunks_list)} фрагментов. Насколько они соответствуют запросу пользователя?"]
            for chunk in chunks_list:
                lines.append(
                    f"chunk_id={chunk.id}\n"
                    f"file={chunk.doc_path}\n"
                    f"title={chunk.title}\n"
                    f"text={chunk.text[:800]}"
                )
            return "\n---\n".join(lines)

        agent = create_agent(
            self.llm,
            [search],
            system_prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

        logger.info("→ LLM: invoking ReAct agent for search planning")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Найди документы для ответа на вопрос: {query}")]
        })

        messages = result["messages"]
        for i, msg in enumerate(messages):
            mtype = type(msg).__name__
            content = str(getattr(msg, "content", ""))
            tool_calls = getattr(msg, "tool_calls", None)
            reasoning = get_reasoning(msg)
            if reasoning:
                logger.info(f"  EMB LLM msg[{i}] {mtype} REASONING:\n{reasoning}")
            if tool_calls:
                for tc in tool_calls:
                    logger.info(f"  EMB LLM msg[{i}] {mtype}: tool_call={tc.get('name','?')}({str(tc.get('args',{}))})")
            elif content:
                logger.info(f"  EMB LLM msg[{i}] {mtype}:\n{content}")

        result = self._extract_chunks(messages, scores)
        result["token_usage"] = {"total_tokens": sum_tokens_from_messages(messages)}
        logger.info(f"EMB final: {len(result['chunks'])} chunks extracted")
        return result

    def _extract_chunks(self, messages: list, scores: dict[str, float]) -> dict:
        seen: set[str] = set()
        chunks: list[dict] = []

        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search":
                content = msg.content if hasattr(msg, "content") else str(msg)
                for line in content.split("\n"):
                    if line.startswith("chunk_id="):
                        cid = line.split("chunk_id=", 1)[1].strip()
                        if cid not in seen:
                            seen.add(cid)
                            chunk = self.db.get_by_id(cid)
                            if chunk:
                                chunks.append({
                                    "chunk_id": chunk.id,
                                    "doc_path": chunk.doc_path,
                                    "title": chunk.title,
                                    "text": chunk.text,
                                    "token_count": chunk.token_count,
                                    "score": scores.get(cid),
                                })
        return {"chunks": chunks[:10], "retriever": "embedding"}


def get_agent() -> EmbeddingRetrieverAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = EmbeddingRetrieverAgent()
    return _agent_instance


async def handle_request(
    user_text: str,
    user_data: dict | None,
    context: RequestContext,
    event_queue: EventQueue,
) -> dict:
    agent = get_agent()

    query = user_text
    if isinstance(user_data, dict) and "query" in user_data:
        query = user_data["query"]

    chunks_data = await agent.run(query)
    n = len(chunks_data["chunks"])
    return {
        "status": "completed",
        "message": f"Retrieved {n} chunks",
        "artifacts": [{
            "artifactId": "retrieved_chunks",
            "data": {"chunks": chunks_data["chunks"], "retriever": "embedding"},
        }],
    }


def create_server():
    executor = A2AExecutor(handle_request)
    skills = [
        AgentSkill(
            id="retrieve_embedding",
            name="Dense semantic retrieval (FAISS)",
            description="Выполняет семантический поиск (FAISS) по единому корпусу нормативных документов. Принимает query в Part.data или text. Возвращает массив чанков в Part.data.",
            input_modes=["text", "data"],
            output_modes=["data"],
            examples=["порядок финансирования инвестиционных проектов", "как организовать контроль за расходами бюджетных средств", "требования к отчётности при получении субсидий"],
            tags=["retrieval", "semantic", "faiss", "embeddings", "intermediate", "shared-corpus"],
        )
    ]
    return A2AServer(
        name="Retriever-Embedding",
        url=RETRIEVER_EMB_URL,
        executor=executor,
        skills=skills,
        description="Semantic document retrieval agent using FAISS embeddings and LLM reasoning.",
    )


if __name__ == "__main__":
    import uvicorn
    setup_logging("retriever_embedding")
    agent = create_server()
    logger.info(f"Starting {agent.name} on port {RETRIEVER_EMB_PORT}")
    uvicorn.run(agent.app, host="0.0.0.0", port=RETRIEVER_EMB_PORT, log_config=None)
