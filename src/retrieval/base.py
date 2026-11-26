import asyncio
import time

from retrieval.contracts import MAX_OUTPUT_CHUNKS, SearchBudget, chunk_dict
from retrieval.engine import RetrievalEngine
from retrieval.rrf import weighted_rrf


class BaseRetriever:
    def __init__(self, engine: RetrievalEngine | None = None):
        self.engine = engine or RetrievalEngine()

    async def retrieve(self, query: str) -> dict:
        return await asyncio.to_thread(self._retrieve, query)

    def _candidates(self, query: str):
        engine = self.engine.with_budget(SearchBudget())
        dense = engine.search("dense", query, 15)
        lexical = engine.search("bm25", query, 15)
        merged = weighted_rrf(
            [hit.chunk.id for hit in dense],
            [hit.chunk.id for hit in lexical],
            alpha=0.5,
            limit=30,
        )
        raw_unique_candidates = len({hit.chunk.id for hit in dense} | {hit.chunk.id for hit in lexical})
        return engine, merged, raw_unique_candidates

    def _retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine, merged, raw_unique_candidates = self._candidates(query)
        chunks = []
        for item in merged[:MAX_OUTPUT_CHUNKS]:
            chunk = engine.chunk_map[item["chunk_id"]]
            chunks.append(chunk_dict(chunk, item["rank"], item["score"], answer_aspect=""))
        return {
            "chunks": chunks,
            "retriever": "base",
            "fallback_used": False,
            "error": None,
            "trace": {
                "query": query,
                "fusion": "rrf",
                "alpha": 0.5,
                "raw_unique_candidates": raw_unique_candidates,
                "unique_candidates_post_cap": len(merged),
                "source_ranks": {item["chunk_id"]: item["source_ranks"] for item in merged[:MAX_OUTPUT_CHUNKS]},
                "budget": engine.budget.snapshot(),
            },
            "token_usage": {"total_tokens": 0},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
