"""Классический (baseline) RAG: FAISS + BM25 → RRF → реранк → генерация.

Паритет с мультиагентной системой (см. план бенчмарка): те же корпус и чанкинг
(chunks.pkl), тот же FAISS-индекс, тот же BM25-индекс, тот же генератор
(SUMMARIZER_* LLM + SYSTEM_PROMPT из agents.summarizer), те же параметры
каналов кандидат-генерации (EMB_TOP_K, BM25_TOP_K, SCORE_THRESHOLD).
Baseline-специфичны только RRF-слияние (RRF_K, HYBRID_CANDIDATES) и реранкер.
"""

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.summarizer import SYSTEM_PROMPT
from baseline_rag.reranker import JinaReranker
from common.benchmark_logger import tokens_from_usage_metadata
from common.chunk_store import ChunkDatabase
from common.config import (
    BM25_INDEX_PATH,
    BM25_TOP_K,
    CHUNKS_PATH,
    EMB_TOP_K,
    HYBRID_CANDIDATES,
    RRF_K,
    RERANK_TOP_N,
    SCORE_THRESHOLD,
    SUMMARIZER_LLM_BASE_URL,
    SUMMARIZER_ENABLE_THINKING,
    SUMMARIZER_LLM_MAX_TOKENS,
    SUMMARIZER_LLM_MODEL,
    SUMMARIZER_LLM_TEMPERATURE,
    create_llm,
    get_reasoning,
)
from indexing.bm25_indexer import BM25Indexer
from indexing.faiss_indexer import FaissIndexer

logger = logging.getLogger(__name__)

EMPTY_ANSWER = "Не найдено релевантных документов для ответа на вопрос."


def _chunk_dict(chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "doc_path": chunk.doc_path,
        "title": chunk.title,
        "text": chunk.text,
        "token_count": chunk.token_count,
    }


class BaselineRAG:
    def __init__(self):
        faiss_index_path = CHUNKS_PATH.parent / "faiss_index"
        if faiss_index_path.is_dir() or str(faiss_index_path).endswith("/"):
            faiss_index_path = faiss_index_path / "index.faiss"

        logger.info("Loading chunks and FAISS index...")
        self.db = ChunkDatabase.load(CHUNKS_PATH)
        self.emb_indexer = FaissIndexer()
        self.emb_indexer.load(faiss_index_path)
        logger.info(f"Loaded {len(self.db.chunks)} chunks, FAISS ready")

        logger.info("Loading BM25 index...")
        self.bm25_indexer = BM25Indexer()
        self.bm25_indexer.load(BM25_INDEX_PATH)
        logger.info(f"BM25 ready, {len(self.bm25_indexer.chunks)} chunks")

        self.reranker = JinaReranker()

        self.llm = create_llm(
            base_url=SUMMARIZER_LLM_BASE_URL,
            model=SUMMARIZER_LLM_MODEL,
            temperature=SUMMARIZER_LLM_TEMPERATURE,
            max_tokens=SUMMARIZER_LLM_MAX_TOKENS,
            enable_thinking=SUMMARIZER_ENABLE_THINKING,
        )
        logger.info("BaselineRAG ready")

    def retrieve_detailed(self, query: str) -> dict[str, Any]:
        """Полный retrieval-конвейер baseline.

        Возвращает {"candidates": RRF-слитые кандидаты (tool_score=RRF-скор),
        "chunks": топ-RERANK_TOP_N после реранка (score=rerank_score)}.
        """
        emb_results = self.emb_indexer.search(
            query, top_k=EMB_TOP_K, score_threshold=SCORE_THRESHOLD
        )
        bm25_results = self.bm25_indexer.search(
            query, top_k=BM25_TOP_K, score_threshold=SCORE_THRESHOLD
        )
        logger.info(f"  FAISS hits: {len(emb_results)}, BM25 hits: {len(bm25_results)}")

        candidates = self._rrf_fuse(emb_results, bm25_results)
        logger.info(f"  RRF candidates: {len(candidates)}")

        reranked = self.reranker.rerank(query, candidates)
        logger.info(f"  after rerank: {len(reranked)} chunks")
        return {"candidates": candidates, "chunks": reranked}

    def _rrf_fuse(
        self,
        emb_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion: score = Σ 1/(RRF_K + rank) по chunk_id."""
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for source, results in (("embedding", emb_results), ("bm25", bm25_results)):
            chunks = self.db.chunks if source == "embedding" else self.bm25_indexer.chunks
            for rank, (idx, tool_score) in enumerate(results, start=1):
                if idx >= len(chunks):
                    continue
                chunk = chunks[idx]
                if chunk.id not in by_id:
                    entry = _chunk_dict(chunk)
                    entry["emb_score"] = None
                    entry["bm25_score"] = None
                    by_id[chunk.id] = entry
                if source == "embedding":
                    by_id[chunk.id]["emb_score"] = tool_score
                else:
                    by_id[chunk.id]["bm25_score"] = tool_score
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)

        ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:HYBRID_CANDIDATES]
        candidates = []
        for cid, rrf_score in ordered:
            entry = dict(by_id[cid])
            entry["rrf_score"] = rrf_score
            entry["score"] = rrf_score
            candidates.append(entry)
        return candidates

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Топ-RERANK_TOP_N чанков в формате ретриверов агентной системы."""
        return self.retrieve_detailed(query)["chunks"]

    async def answer(self, query: str, question_id: str = "") -> dict[str, Any]:
        """Генерация ответа по top-N чанкам (без соседей — паритет с A0).

        Промпт и SYSTEM_PROMPT — 1:1 как в SummarizerAgent.run.
        """
        start = time.perf_counter()
        retrieval = self.retrieve_detailed(query)
        chunks = retrieval["chunks"][:RERANK_TOP_N]

        if not chunks:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "question_id": question_id,
                "answer": EMPTY_ANSWER,
                "sources": [],
                "chunks": [],
                "candidates": retrieval["candidates"],
                "token_usage": {"total_tokens": 0},
                "latency_ms": latency_ms,
            }

        context_texts = []
        sources: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
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
        logger.info(f"→ LLM: baseline generation (prompt len={len(prompt)})")
        response = await self.llm.ainvoke(messages)
        answer = response.content
        reasoning = get_reasoning(response)
        if reasoning:
            logger.info(f"  BASELINE REASONING:\n{reasoning}")
        logger.info(f"← LLM: answer len={len(answer)}")

        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "question_id": question_id,
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "candidates": retrieval["candidates"],
            "token_usage": {
                "total_tokens": tokens_from_usage_metadata(
                    getattr(response, "usage_metadata", None)
                )
            },
            "latency_ms": latency_ms,
        }
