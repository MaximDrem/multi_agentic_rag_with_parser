import asyncio
import threading
from types import SimpleNamespace

from common.models import Chunk
from retrieval.base import BaseRetriever
from retrieval.npa_mas.schemas import Candidate, CandidatePool, PipelineResult, RankedChunk


def test_production_adapter_offloads_pipeline_from_a2a_event_loop():
    chunk = Chunk(id="chunk-1", doc_path="doc.md", title="Норма", text="Текст", token_count=2)
    candidate = Candidate(
        chunk_id=chunk.id,
        pool_rank=1,
        selector_text=chunk.text,
        text_sha256="0" * 64,
        doc_path=chunk.doc_path,
        title=chunk.title,
        source_ranks={"dense": 1},
        rrf_score=1.0,
        text=chunk.text,
    )
    pool = CandidatePool.create(
        query_group="q",
        candidates=[candidate],
        budget={},
        generated_queries={},
        provenance={"channel_rankings": {"dense": [chunk.id]}, "observations": {"dense": []}},
    )
    ranked = RankedChunk(
        chunk_id=chunk.id,
        rank=1,
        score=1.0,
        text=chunk.text,
        selector_text=chunk.text,
        doc_path=chunk.doc_path,
        title=chunk.title,
    )

    class FakePipeline:
        config = SimpleNamespace(retrieval=SimpleNamespace(final_alpha_dense=0.5))

        def __init__(self):
            self.thread_id = None

        async def run(self, query, variant, *, query_group, **kwargs):
            self.thread_id = threading.get_ident()
            return PipelineResult(
                variant=variant,
                query=query,
                query_group=query_group,
                status="success",
                strict_failed=False,
                fallback_used=False,
                fallback_stage=None,
                fallback_reason=None,
                fallbacks=[],
                ranked_chunk_ids=[chunk.id],
                full_ranked_chunk_ids=[chunk.id],
                chunks=[ranked],
                pool=pool,
                plan=None,
                worker_queries={},
                reports={},
                calls=[],
                token_usage=[],
                stage_timings_ms={},
                latency_ms=1.0,
                error=None,
            )

    class FakeEngine:
        chunk_map = {chunk.id: chunk}

    async def invoke():
        pipeline = FakePipeline()
        retriever = BaseRetriever(engine=FakeEngine(), pipeline=pipeline)
        result = await retriever.retrieve("q")
        assert pipeline.thread_id != threading.get_ident()
        return result

    result = asyncio.run(invoke())
    assert result["chunks"][0]["chunk_id"] == chunk.id
    assert result["trace"]["pool_sha256"] == pool.pool_sha256
