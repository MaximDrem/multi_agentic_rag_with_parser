import asyncio
import threading
import time

import torch

from common.config import (
    RERANKER_DEVICE,
    RERANKER_MODEL,
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
    MAX_OUTPUT_CHUNKS,
    MULTI_OBSERVED_CHUNKS,
    MULTI_OBSERVED_TOKENS,
    MULTI_SELECTION_LIMIT,
    SearchBudget,
    chunk_dict,
)
from retrieval.engine import RetrievalEngine
from retrieval.reranker import JinaReranker
from retrieval.rrf import weighted_rrf


class MultiAgentV2Retriever:
    def __init__(self, engine: RetrievalEngine | None = None, dense_llm=None, bm25_llm=None, reranker=None):
        self.engine = engine or RetrievalEngine()
        self.dense_llm = dense_llm if dense_llm is not None else create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING,
        )
        self.bm25_llm = bm25_llm if bm25_llm is not None else create_llm(
            base_url=RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_BM25_ENABLE_THINKING,
        )
        self.reranker = reranker if reranker is not None else JinaReranker()
        if reranker is None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._rerank_lock = threading.Lock()
        self.rerank_batch_size = 8

    def _rerank(self, query, candidates, events=None):
        with self._rerank_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def run_batch(batch):
                if not batch:
                    return []
                try:
                    return self.reranker.rerank(query, batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(batch) == 1:
                        raise
                    middle = len(batch) // 2
                    if events is not None:
                        events.append({"batch_size": len(batch), "split_into": [middle, len(batch) - middle]})
                    return run_batch(batch[:middle]) + run_batch(batch[middle:])

            try:
                if len(candidates) <= self.rerank_batch_size:
                    return run_batch(candidates)
                pool = []
                for start in range(0, len(candidates), self.rerank_batch_size):
                    pool.extend(run_batch(candidates[start:start + self.rerank_batch_size]))
                return sorted(pool, key=lambda chunk: chunk["rerank_score"], reverse=True)[:MAX_OUTPUT_CHUNKS]
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    async def retrieve(self, query: str) -> dict:
        started = time.perf_counter()
        engine = self.engine.with_budget(SearchBudget())
        dense_hits, lexical_hits = await asyncio.gather(
            asyncio.to_thread(engine.search, "dense", query, 15),
            asyncio.to_thread(engine.search, "bm25", query, 15),
        )
        base_merged = weighted_rrf(
            [hit.chunk.id for hit in dense_hits],
            [hit.chunk.id for hit in lexical_hits],
            alpha=0.5,
            limit=30,
        )
        base_ranks = {item["chunk_id"]: item for item in base_merged}
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
        dense, lexical = await asyncio.gather(
            dense_runtime.retrieve(query, initial_hits=dense_hits, initial_observed_limit=10),
            bm25_runtime.retrieve(query, initial_hits=lexical_hits, initial_observed_limit=10),
        )
        ledger = {}

        def entry(chunk_id):
            if chunk_id not in ledger:
                ledger[chunk_id] = {
                    "base_rrf_rank": None,
                    "base_rrf_score": None,
                    "retrievals": [],
                    "observed_by": [],
                    "selected": {},
                }
            return ledger[chunk_id]

        for chunk_id, item in base_ranks.items():
            row = entry(chunk_id)
            row["base_rrf_rank"] = item["rank"]
            row["base_rrf_score"] = item["score"]
        for channel, result in (("dense", dense), ("bm25", lexical)):
            observed = set(result["trace"]["observation"]["observed_chunk_ids"])
            for search in result["trace"]["searches"]:
                if search.get("cached"):
                    continue
                for rank, chunk_id in enumerate(search.get("ranking_ids", []), 1):
                    entry(chunk_id)["retrievals"].append({
                        "channel": channel,
                        "query": search["query"],
                        "call_index": search["call_index"],
                        "rank": rank,
                    })
            for chunk_id in observed:
                entry(chunk_id)["observed_by"].append(channel)
        candidate_ids = list(ledger)
        selections = {}
        for channel, result in (("dense", dense), ("bm25", lexical)):
            selections[channel] = []
            for rank, chunk in enumerate(result["chunks"], 1):
                chunk_id = chunk["chunk_id"]
                selection = {
                    "rank": rank,
                    "relevance": chunk.get("relevance"),
                    "answer_aspect": chunk.get("answer_aspect", ""),
                }
                selections[channel].append({"chunk_id": chunk_id, **selection})
                entry(chunk_id)["selected"][channel] = selection
        candidates = []
        for chunk_id in candidate_ids:
            base = base_ranks.get(chunk_id)
            candidate = chunk_dict(
                engine.chunk_map[chunk_id],
                base["rank"] if base else None,
                base["score"] if base else None,
            )
            candidate["rrf_rank"] = base["rank"] if base else None
            candidate["rrf_score"] = base["score"] if base else None
            candidates.append(candidate)
        fallback_used = False
        reranker_error = None
        reranker_adaptive_events = []
        try:
            reranked = await asyncio.to_thread(self._rerank, query, candidates, reranker_adaptive_events)
        except Exception as exc:
            fallback_used = True
            reranker_error = f"{type(exc).__name__}: {exc}"
            reranked = sorted(
                candidates,
                key=lambda chunk: (
                    chunk["rrf_rank"] is None,
                    chunk["rrf_rank"] if chunk["rrf_rank"] is not None else 0,
                ),
            )[:MAX_OUTPUT_CHUNKS]
        chunks = reranked[:MAX_OUTPUT_CHUNKS]
        observed_ids = list(dict.fromkeys(
            dense["trace"]["observation"]["observed_chunk_ids"]
            + lexical["trace"]["observation"]["observed_chunk_ids"]
        ))
        base_ids = set(base_ranks)
        source_ranks = {
            chunk_id: base_ranks[chunk_id]["source_ranks"]
            for chunk_id in base_ranks
        }
        return {
            "chunks": chunks,
            "retriever": "multiagent_v2",
            "fallback_used": fallback_used,
            "error": None,
            "trace": {
                "query": query,
                "fusion": "shared_base_15_15+agent_expansion+jina",
                "reranker_model": RERANKER_MODEL,
                "reranker_device": RERANKER_DEVICE,
                "reranker_batch_size": self.rerank_batch_size,
                "reranker_adaptive_events": reranker_adaptive_events,
                "reranker_error": reranker_error,
                "base_unique_candidates": len(base_ids),
                "agent_new_unique_candidates": len(set(candidate_ids) - base_ids),
                "raw_unique_candidates": len(candidate_ids),
                "source_ranks": source_ranks,
                "candidate_ledger": ledger,
                "dense": {
                    **{key: value for key, value in dense.items() if key not in ("chunks", "retriever")},
                    "selected_chunks": selections["dense"],
                },
                "bm25": {
                    **{key: value for key, value in lexical.items() if key not in ("chunks", "retriever")},
                    "selected_chunks": selections["bm25"],
                },
                "subagent_fallback_used": dense["fallback_used"] or lexical["fallback_used"],
                "information_budget": engine.budget.snapshot(),
                "global_observation": {
                    "llm_observed_unique_chunks": len(observed_ids),
                    "llm_observed_unique_tokens": sum(engine.chunk_map[chunk_id].token_count for chunk_id in observed_ids),
                    "observed_chunk_ids": observed_ids,
                },
            },
            "token_usage": {
                "total_tokens": dense["token_usage"]["total_tokens"] + lexical["token_usage"]["total_tokens"]
            },
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
