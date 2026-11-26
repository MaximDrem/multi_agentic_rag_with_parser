import asyncio
import threading
import time

from common.config import RERANKER_DEVICE, RERANKER_MODEL
from retrieval.base import BaseRetriever
from retrieval.contracts import MAX_OUTPUT_CHUNKS, chunk_dict
from retrieval.engine import RetrievalEngine
from retrieval.reranker import JinaReranker


class BaselineRerankerRetriever(BaseRetriever):
    def __init__(self, engine: RetrievalEngine | None = None):
        super().__init__(engine)
        self.reranker = JinaReranker()
        self._rerank_lock = threading.Lock()

    def _rerank(self, query, candidates):
        with self._rerank_lock:
            return self.reranker.rerank(query, candidates)

    async def retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine, merged, raw_unique_candidates = await asyncio.to_thread(self._candidates, query)
        candidates = []
        for item in merged:
            candidate = chunk_dict(engine.chunk_map[item["chunk_id"]], item["rank"], item["score"])
            candidate["rrf_score"] = item["score"]
            candidate["rrf_rank"] = item["rank"]
            candidates.append(candidate)
        reranked = await asyncio.to_thread(self._rerank, query, candidates)
        return {
            "chunks": reranked[:MAX_OUTPUT_CHUNKS],
            "retriever": "baseline_reranker",
            "fallback_used": False,
            "error": None,
            "trace": {
                "query": query,
                "fusion": "rrf+jina_reranker",
                "alpha": 0.5,
                "raw_unique_candidates": raw_unique_candidates,
                "unique_candidates_post_cap": len(merged),
                "reranker_model": RERANKER_MODEL,
                "reranker_device": RERANKER_DEVICE,
                "source_ranks": {item["chunk_id"]: item["source_ranks"] for item in merged},
                "budget": engine.budget.snapshot(),
            },
            "token_usage": {"total_tokens": 0},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
