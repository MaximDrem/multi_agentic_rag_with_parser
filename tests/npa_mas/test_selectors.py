from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from retrieval.npa_mas.candidates import CandidateBuilder
from retrieval.npa_mas.llm import OutputValidationError, PromptLoader
from retrieval.npa_mas.pipeline import SamePoolError, assert_same_pool
from retrieval.npa_mas.schemas import CandidatePool, SelectorPermutation
from retrieval.npa_mas.selectors import JinaSelector, LLMSelector, candidate_reference, validate_permutation


@pytest.fixture
def pool(backend):
    return CandidateBuilder(backend).build(query_group="g", query="q", channels=("dense", "bm25"))


def valid_ranking(pool):
    return [candidate_reference(pool, candidate) for candidate in pool.candidates]


def test_full_permutation_is_accepted(pool):
    ranking = list(reversed(valid_ranking(pool)))
    value = SelectorPermutation(ranking=ranking)
    assert validate_permutation(value, pool).ranking == ranking


@pytest.mark.parametrize("kind", ["extra", "missing", "duplicate", "out_of_pool"])
def test_selector_rejects_invalid_permutations(pool, kind):
    ranking = valid_ranking(pool)
    if kind == "extra":
        ranking.append(f"{pool.pool_id}::extra")
    elif kind == "missing":
        ranking = ranking[:-1]
    elif kind == "duplicate":
        ranking[-1] = ranking[0]
    else:
        ranking[0] = f"wrong-pool::{pool.candidates[0].chunk_id}"
    with pytest.raises(OutputValidationError):
        validate_permutation(SelectorPermutation(ranking=ranking), pool)


def test_partial_permutation_falls_back_to_rrf_with_trace(pool, gateway, prompts_dir):
    gateway.set_response("llm_selector", {"ranking": valid_ranking(pool)[:-1]})
    selector = LLMSelector(gateway, PromptLoader(prompts_dir))
    selection = asyncio.run(selector.rank("q", pool))
    assert selection.fallback_reason == "missing_candidate_id"
    assert selection.trace.status == "fallback"
    assert selection.trace.fallback_used
    assert selection.trace.events[-1].event == "missing_candidate_id"
    assert [chunk.chunk_id for chunk in selection.chunks] == pool.candidate_ids[:10]
    assert selection.full_ranked_chunk_ids == pool.candidate_ids


def test_valid_llm_permutation_returns_at_most_ten_unique_ids(pool, gateway, prompts_dir):
    selector = LLMSelector(gateway, PromptLoader(prompts_dir))
    selection = asyncio.run(selector.rank("q", pool))
    assert selection.fallback_reason is None
    assert len(selection.chunks) == 10
    assert len({chunk.chunk_id for chunk in selection.chunks}) == 10
    assert selection.chunks[0].chunk_id == pool.candidates[-1].chunk_id
    assert len(selection.full_ranked_chunk_ids) == len(pool.candidates)
    assert selection.full_ranked_chunk_ids[0] == pool.candidates[-1].chunk_id


def test_selector_prompt_has_no_raw_scores_and_uses_selector_text(pool, gateway, prompts_dir):
    selector = LLMSelector(gateway, PromptLoader(prompts_dir))
    asyncio.run(selector.rank("q", pool))
    prompt = gateway.calls[-1]["prompt"]
    assert pool.candidates[0].selector_text in prompt
    assert "rrf_score" not in prompt
    assert '"source_ranks"' not in prompt
    assert '"hits"' not in prompt


def test_jina_full_permutation_restores_configured_top_n(pool):
    class FakeReranker:
        top_n = 15

        def rerank(self, query, chunks):
            self.requested_top_n = self.top_n
            return [
                {"chunk_id": chunk["chunk_id"], "rerank_score": float(index)}
                for index, chunk in enumerate(chunks)
            ]

    reranker = FakeReranker()
    selector = JinaSelector(reranker)
    full = selector.rank_full("q", pool)
    assert len(full) == len(pool.candidates)
    assert reranker.requested_top_n == len(pool.candidates)
    assert reranker.top_n == 15
    assert len(selector.rank("q", pool)) == 10
    assert reranker.top_n == 15


def test_lazy_jina_initialization_is_lock_serialized(pool, monkeypatch):
    constructed = []

    class SlowReranker:
        top_n = 10

        def __init__(self):
            time.sleep(0.02)
            constructed.append(threading.get_ident())

        def rerank(self, query, chunks):
            return [{"chunk_id": chunk["chunk_id"], "rerank_score": 1.0} for chunk in chunks]

    monkeypatch.setattr("retrieval.reranker.JinaReranker", SlowReranker)
    selector = JinaSelector()
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _: selector.rank("q", pool), range(2)))
    assert len(constructed) == 1
    assert all(len(output) == 10 for output in outputs)


def test_same_pool_guard_detects_text_mismatch(pool):
    changed = pool.candidates[0].model_copy(update={"selector_text": "другой текст"})
    other = CandidatePool.create(
        query_group=pool.query_group,
        candidates=[changed, *pool.candidates[1:]],
        budget=pool.budget,
        generated_queries=pool.generated_queries,
        provenance=pool.provenance,
    )
    with pytest.raises(SamePoolError):
        assert_same_pool(pool, other)
