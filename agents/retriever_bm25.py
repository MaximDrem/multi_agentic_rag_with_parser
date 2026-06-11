import json
import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from a2a.types import AgentSkill
from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext

from common.config import (
    RETRIEVER_BM25_PORT, RETRIEVER_BM25_URL,
    CHUNKS_PATH, BM25_INDEX_PATH, BM25_TOP_K, SCORE_THRESHOLD,
    RETRIEVER_BM25_LLM_BASE_URL,
    RETRIEVER_BM25_LLM_MODEL,
    RETRIEVER_BM25_LLM_TEMPERATURE,
    RETRIEVER_BM25_LLM_MAX_TOKENS,
    RETRIEVER_BM25_ENABLE_THINKING,
    create_llm,
    get_reasoning,
)
from common.chunk_store import ChunkDatabase
from common.a2a_server import A2AServer, A2AExecutor
from common.models import Chunk
from indexing.bm25_indexer import BM25Indexer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — агент итеративного поиска по базе нормативно-правовых документов РФ. "
    "Твоя задача: найти чанки, которые помогут ответить на вопрос пользователя. "
    "Можешь вызывать search несколько раз (не более 5), "
    "каждый раз формулируя 1-3 варианта запроса через ; "
    "(ключевые слова, синонимы, смежные термины). "
    "Анализируй найденные чанки и при необходимости уточняй запросы, "
    "чтобы собрать наиболее полную и релевантную подборку. "
    "Когда чанков достаточно или лимит исчерпан — заверши работу."
)

_agent_instance: "BM25RetrieverAgent | None" = None


class BM25RetrieverAgent:
    def __init__(self):
        logger.info("Loading BM25 index...")
        self.db = ChunkDatabase.load(CHUNKS_PATH)
        self.indexer = BM25Indexer()
        self.indexer.load(BM25_INDEX_PATH)
        logger.info(f"Loaded {len(self.db.chunks)} chunks, BM25 ready")

        self.llm = create_llm(
            base_url=RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_BM25_ENABLE_THINKING,
        )

    async def run(self, query: str) -> dict:
        logger.info(f"=== BM25 RETRIEVER: {query[:100]} ===")
        indexer = self.indexer

        call_count = [0]
        MAX_CALLS = 5

        @tool
        def search(keywords: str) -> str:
            """Поиск документов по ключевым словам (BM25). Принимает 1-3 запроса через ; например 'соглашение; бюджетный кодекс; НПА'. Возвращает фрагменты с текстом, заголовком и путём к документу."""
            call_count[0] += 1
            if call_count[0] > MAX_CALLS:
                return f"Достигнут лимит вызовов ({MAX_CALLS}). Больше нельзя вызывать search. Проанализируй уже найденные чанки и заверши работу."
            logger.info(f"BM25 search tool called (#{call_count[0]}/{MAX_CALLS}) with: {keywords}")
            queries = [q.strip() for q in keywords.split(";") if q.strip()]
            if not queries:
                queries = [keywords]
            all_chunks: dict[str, Chunk] = {}
            for q in queries:
                results = indexer.search(q, top_k=BM25_TOP_K, score_threshold=SCORE_THRESHOLD)
                logger.info(f"  sub-query '{q}': {len(results)} hits")
                for idx, score in results:
                    chunk = indexer.chunks[idx]
                    if chunk.id not in all_chunks:
                        all_chunks[chunk.id] = chunk
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

        agent = create_react_agent(
            self.llm,
            [search],
            prompt=SystemMessage(content=SYSTEM_PROMPT),
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
                logger.info(f"  BM25 LLM msg[{i}] {mtype} REASONING:\n{reasoning}")
            if tool_calls:
                for tc in tool_calls:
                    logger.info(f"  BM25 LLM msg[{i}] {mtype}: tool_call={tc.get('name','?')}({str(tc.get('args',{}))})")
            elif content:
                logger.info(f"  BM25 LLM msg[{i}] {mtype}:\n{content}")

        result = self._extract_chunks(messages)
        logger.info(f"BM25 final: {len(result['chunks'])} chunks extracted")
        return result

    def _extract_chunks(self, messages: list) -> dict:
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
                            idx = self._find_chunk_index(cid)
                            if idx is not None:
                                chunk = self.indexer.chunks[idx]
                                chunks.append({
                                    "chunk_id": chunk.id,
                                    "doc_path": chunk.doc_path,
                                    "title": chunk.title,
                                    "text": chunk.text,
                                    "token_count": chunk.token_count,
                                })
        return {"chunks": chunks[:10], "retriever": "bm25"}

    def _find_chunk_index(self, chunk_id: str) -> int | None:
        for i, c in enumerate(self.indexer.chunks):
            if c.id == chunk_id:
                return i
        return None


def get_agent() -> BM25RetrieverAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = BM25RetrieverAgent()
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
            "data": {"chunks": chunks_data["chunks"], "retriever": "bm25"},
        }],
    }


def create_server():
    executor = A2AExecutor(handle_request)
    skills = [
        AgentSkill(
            id="retrieve_bm25",
            name="BM25 lexical retrieval",
            description="Выполняет лексический поиск (BM25) по единому корпусу нормативных документов. Принимает query в Part.data или text. Возвращает массив чанков в Part.data.",
            input_modes=["text", "data"],
            output_modes=["data"],
            examples=["статья 14 бюджетного кодекса рф", "финансовое соглашение муниципального образования", "методические рекомендации по проектному финансированию"],
            tags=["retrieval", "lexical", "bm25", "intermediate", "shared-corpus"],
        )
    ]
    return A2AServer(
        name="Retriever-BM25",
        url=RETRIEVER_BM25_URL,
        executor=executor,
        skills=skills,
        description="Keyword document retrieval agent using BM25 index and LLM reasoning.",
    )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    agent = create_server()
    logger.info(f"Starting {agent.name} on port {RETRIEVER_BM25_PORT}")
    uvicorn.run(agent.app, host="0.0.0.0", port=RETRIEVER_BM25_PORT)
