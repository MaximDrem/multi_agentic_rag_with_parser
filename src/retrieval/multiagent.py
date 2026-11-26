import asyncio
import time

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
from retrieval.agentic import AgenticSearchRuntime
from retrieval.contracts import (
    MULTI_OBSERVED_CHUNKS,
    MULTI_OBSERVED_TOKENS,
    MULTI_SELECTION_LIMIT,
    SearchBudget,
    chunk_dict,
)
from retrieval.engine import RetrievalEngine
from retrieval.rrf import weighted_rrf


class MultiAgentRetriever:
    def __init__(self, engine: RetrievalEngine | None = None):
        self.engine = engine or RetrievalEngine()
        self.dense_llm = create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING,
        )
        self.bm25_llm = create_llm(
            base_url=RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_BM25_ENABLE_THINKING,
        )

    async def retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine = self.engine.with_budget(SearchBudget())
        dense_runtime = AgenticSearchRuntime(
            self.dense_llm,
            engine,
            mode="dense",
            max_observed_chunks=MULTI_OBSERVED_CHUNKS,
            max_observed_tokens=MULTI_OBSERVED_TOKENS,
            selection_limit=MULTI_SELECTION_LIMIT,
            reject_tool_copy=False,
        )
        bm25_runtime = AgenticSearchRuntime(
            self.bm25_llm,
            engine,
            mode="bm25",
            max_observed_chunks=MULTI_OBSERVED_CHUNKS,
            max_observed_tokens=MULTI_OBSERVED_TOKENS,
            selection_limit=MULTI_SELECTION_LIMIT,
            reject_tool_copy=False,
        )
        dense, lexical = await asyncio.gather(dense_runtime.retrieve(query), bm25_runtime.retrieve(query))
        dense_ids = [chunk["chunk_id"] for chunk in dense["chunks"]]
        bm25_ids = [chunk["chunk_id"] for chunk in lexical["chunks"]]
        merged = weighted_rrf(dense_ids, bm25_ids, alpha=0.5, limit=30)
        observed_ids = list(dict.fromkeys(
            dense["trace"]["observation"]["observed_chunk_ids"]
            + lexical["trace"]["observation"]["observed_chunk_ids"]
        ))
        global_observation = {
            "llm_observed_unique_chunks": len(observed_ids),
            "llm_observed_unique_tokens": sum(engine.chunk_map[chunk_id].token_count for chunk_id in observed_ids),
            "observed_chunk_ids": observed_ids,
        }
        chunks = [
            chunk_dict(engine.chunk_map[item["chunk_id"]], item["rank"], item["score"])
            for item in merged[:10]
        ]
        dense_error = dense.get("error")
        bm25_error = lexical.get("error")
        error = None
        if dense_error and bm25_error:
            error = f"dense: {dense_error}; bm25: {bm25_error}"
        return {
            "chunks": chunks,
            "retriever": "multiagent",
            "fallback_used": dense["fallback_used"] or lexical["fallback_used"],
            "error": error,
            "trace": {
                "query": query,
                "fusion": "rrf_agent_selected_ranks",
                "alpha": 0.5,
                "fusion_input_count": len(merged),
                "source_ranks": {item["chunk_id"]: item["source_ranks"] for item in merged[:10]},
                "dense": {key: value for key, value in dense.items() if key not in ("chunks", "retriever")},
                "bm25": {key: value for key, value in lexical.items() if key not in ("chunks", "retriever")},
                "information_budget": engine.budget.snapshot(),
                "global_observation": global_observation,
            },
            "token_usage": {
                "total_tokens": dense["token_usage"]["total_tokens"] + lexical["token_usage"]["total_tokens"]
            },
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
