from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from retrieval.npa_mas.llm import JsonLLMGateway, OutputValidationError, PromptLoader, mark_fallback, render_prompt
from retrieval.npa_mas.schemas import (
    CallTrace,
    Candidate,
    CandidatePool,
    RankedChunk,
    SelectorPermutation,
    WorkerReport,
)


def candidate_reference(pool: CandidatePool, candidate: Candidate | str) -> str:
    chunk_id = candidate if isinstance(candidate, str) else candidate.chunk_id
    return f"{pool.pool_id}::{chunk_id}"


def validate_permutation(permutation: SelectorPermutation, pool: CandidatePool) -> SelectorPermutation:
    expected = [candidate_reference(pool, candidate) for candidate in pool.candidates]
    actual = permutation.ranked_ids
    if len(actual) != len(set(actual)):
        raise OutputValidationError("duplicate_candidate_id", "selector returned duplicate candidate IDs")
    unknown = [item for item in actual if item not in set(expected)]
    if unknown:
        raise OutputValidationError("invalid_candidate_id", "selector returned extra or out-of-pool candidate IDs")
    missing = [item for item in expected if item not in set(actual)]
    if missing:
        raise OutputValidationError("missing_candidate_id", "selector returned an incomplete permutation")
    return permutation


def ranked_chunk(candidate: Candidate, rank: int, score: float | None) -> RankedChunk:
    return RankedChunk(
        chunk_id=candidate.chunk_id,
        rank=rank,
        score=score,
        text=candidate.text,
        selector_text=candidate.selector_text,
        doc_path=candidate.doc_path,
        title=candidate.title,
    )


class JinaSelector:
    def __init__(self, reranker: Any | None = None, *, output_chunks: int = 10):
        self.reranker = reranker
        self.output_chunks = output_chunks
        self._lock = threading.Lock()

    def rank(self, query: str, pool: CandidatePool) -> list[RankedChunk]:
        return self.rank_full(query, pool)[: self.output_chunks]

    def rank_full(self, query: str, pool: CandidatePool) -> list[RankedChunk]:
        if not pool.candidates:
            return []
        if self.reranker is None:
            from retrieval.reranker import JinaReranker

            self.reranker = JinaReranker()
        payload = [
            {
                "chunk_id": candidate.chunk_id,
                "text": candidate.selector_text,
            }
            for candidate in pool.candidates
        ]
        with self._lock:
            previous_top_n = getattr(self.reranker, "top_n", None)
            if previous_top_n is not None:
                self.reranker.top_n = len(payload)
            try:
                reranked = self.reranker.rerank(query, payload)
            finally:
                if previous_top_n is not None:
                    self.reranker.top_n = previous_top_n
        by_id = {candidate.chunk_id: candidate for candidate in pool.candidates}
        scored = []
        seen = set()
        for item in reranked:
            chunk_id = item.get("chunk_id")
            if chunk_id not in by_id or chunk_id in seen:
                raise ValueError("Jina returned an invalid or duplicate candidate ID")
            seen.add(chunk_id)
            score = float(item.get("rerank_score", item.get("score")))
            scored.append((score, by_id[chunk_id]))
        if len(scored) != len(pool.candidates):
            raise ValueError("Jina returned an incomplete candidate permutation")
        scored.sort(key=lambda item: (-item[0], item[1].pool_rank, item[1].chunk_id))
        return [
            ranked_chunk(candidate, index, score)
            for index, (score, candidate) in enumerate(scored, 1)
        ]


@dataclass
class LLMSelection:
    chunks: list[RankedChunk]
    trace: CallTrace
    fallback_reason: str | None
    full_ranked_chunk_ids: list[str]


class LLMSelector:
    stage = "llm_selector"

    def __init__(
        self,
        gateway: JsonLLMGateway,
        prompts: PromptLoader,
        *,
        output_chunks: int = 10,
        max_output_tokens: int = 2000,
    ):
        self.gateway = gateway
        self.prompts = prompts
        self.output_chunks = output_chunks
        self.max_output_tokens = max_output_tokens

    async def rank(
        self,
        query: str,
        pool: CandidatePool,
        *,
        reports: dict[str, WorkerReport] | None = None,
    ) -> LLMSelection:
        candidates = [
            {
                "pool_id": pool.pool_id,
                "chunk_id": candidate.chunk_id,
                "pool_rank": candidate.pool_rank,
                "selector_text": candidate.selector_text,
                "doc_path": candidate.doc_path,
                "title": candidate.title,
            }
            for candidate in pool.candidates
        ]
        report_payload = {
            channel: report.model_dump(mode="json")
            for channel, report in (reports or {}).items()
        }
        prompt = render_prompt(
            self.prompts.load("llm_selector.txt"),
            query=query,
            candidates_json=candidates,
            reports_json=report_payload if reports is not None else None,
        )
        result = await self.gateway.generate_json(
            stage=self.stage,
            prompt=prompt,
            response_model=SelectorPermutation,
            max_output_tokens=self.max_output_tokens,
            validator=lambda value: validate_permutation(value, pool),
        )
        if result.value is None:
            return self._fallback(pool, result.trace, result.error or "llm_selector_failed")
        try:
            permutation = validate_permutation(result.value, pool)
        except OutputValidationError as exc:
            trace = mark_fallback(
                result.trace,
                str(exc),
                event=exc.event,
                error_class=type(exc).__name__,
            )
            return self._fallback(pool, trace, exc.event, mark=False)
        by_reference = {
            candidate_reference(pool, candidate): candidate
            for candidate in pool.candidates
        }
        chunks = [
            ranked_chunk(by_reference[reference], index, None)
            for index, reference in enumerate(permutation.ranked_ids[: self.output_chunks], 1)
        ]
        return LLMSelection(
            chunks=chunks,
            trace=result.trace,
            fallback_reason=None,
            full_ranked_chunk_ids=[by_reference[reference].chunk_id for reference in permutation.ranked_ids],
        )

    def _fallback(
        self,
        pool: CandidatePool,
        trace: CallTrace,
        reason: str,
        *,
        mark: bool = True,
    ) -> LLMSelection:
        fallback_trace = mark_fallback(trace, reason) if mark else trace
        chunks = [
            ranked_chunk(candidate, index, candidate.rrf_score)
            for index, candidate in enumerate(pool.candidates[: self.output_chunks], 1)
        ]
        return LLMSelection(
            chunks=chunks,
            trace=fallback_trace,
            fallback_reason=reason,
            full_ranked_chunk_ids=pool.candidate_ids,
        )
