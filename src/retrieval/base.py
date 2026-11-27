import asyncio
import time

from retrieval.contracts import MAX_OUTPUT_CHUNKS
from retrieval.engine import RetrievalEngine
from retrieval.npa_mas.pipeline import NpaMasPipeline


class BaseRetriever:
    variant = "B0_RRF"
    retriever_name = "base"
    fusion_name = "rrf"

    def __init__(self, engine: RetrievalEngine | None = None, pipeline: NpaMasPipeline | None = None):
        self.engine = engine or (pipeline.engine if pipeline is not None else None) or RetrievalEngine()
        self.pipeline = pipeline or NpaMasPipeline(engine=self.engine)

    async def retrieve(self, query: str) -> dict:
        return await asyncio.to_thread(self._retrieve_sync, query)

    def _retrieve_sync(self, query: str) -> dict:
        started = time.perf_counter()
        result = asyncio.run(self.pipeline.run(query, self.variant, query_group=query))
        chunks = []
        for item in result.chunks[:MAX_OUTPUT_CHUNKS]:
            chunk = self.engine.chunk_map[item.chunk_id]
            chunks.append({
                "chunk_id": item.chunk_id,
                "doc_path": item.doc_path,
                "title": item.title,
                "text": item.text,
                "token_count": chunk.token_count,
                "score": item.score,
                "rank": item.rank,
                "relevance": None,
                "answer_aspect": "",
            })
        pool = result.pool
        channel_rankings = pool.provenance.get("channel_rankings", {}) if pool else {}
        raw_unique_candidates = len({chunk_id for ranking in channel_rankings.values() for chunk_id in ranking})
        return {
            "chunks": chunks,
            "retriever": self.retriever_name,
            "fallback_used": result.fallback_used,
            "error": result.error,
            "trace": {
                "query": query,
                "fusion": self.fusion_name,
                "alpha": self.pipeline.config.retrieval.final_alpha_dense,
                "raw_unique_candidates": raw_unique_candidates,
                "unique_candidates_post_cap": len(pool.candidates) if pool else 0,
                "pool_sha256": pool.pool_sha256 if pool else None,
                "source_ranks": {candidate.chunk_id: candidate.source_ranks for candidate in (pool.candidates[:MAX_OUTPUT_CHUNKS] if pool else [])},
                "budget": {channel: value.model_dump(mode="json") for channel, value in pool.budget.items()} if pool else {},
            },
            "token_usage": {"total_tokens": 0},
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    def _retrieve(self, query: str) -> dict:
        return self._retrieve_sync(query)
