from __future__ import annotations

from types import SimpleNamespace

from retrieval.npa_mas.candidates import CandidateBuilder, EngineSearchAdapter
from retrieval.npa_mas.schemas import RetrievalSettings, compute_pool_sha256


def test_original_plus_three_unique_generated_queries_and_fixed_budgets(backend):
    builder = CandidateBuilder(backend)
    generated = ["  q1  ", "q1", "q2", "", "q3", "q4"]
    pool = builder.build(
        query_group="g",
        query="исходный",
        channels=("dense", "bm25"),
        generated_queries={"dense": generated, "bm25": generated},
    )
    for channel in ("dense", "bm25"):
        calls = [call for call in backend.calls if call[0] == channel]
        assert [call[2] for call in calls] == [30, 10, 10, 10]
        assert [call[1] for call in calls] == ["исходный", "q1", "q2", "q3"]
        assert pool.generated_queries[channel] == ["q1", "q2", "q3"]
        assert pool.provenance["duplicate_queries"][channel] == ["q1"]
        assert pool.provenance["truncated_queries"][channel] == ["q4"]
        assert pool.budget[channel].target_raw_hits == 60
        assert pool.budget[channel].requested_raw_hits == 60
        assert pool.budget[channel].actual_raw_hits == 60
    assert len(pool.candidates) <= 30
    assert len(pool.candidate_ids) == len(set(pool.candidate_ids))


def test_fewer_queries_leave_unrequested_budget_without_compensation(backend):
    builder = CandidateBuilder(backend)
    pool = builder.build(
        query_group="g",
        query="исходный",
        channels=("dense", "bm25"),
        generated_queries={"dense": [], "bm25": ["один"]},
    )
    assert pool.budget["dense"].requested_raw_hits == 30
    assert pool.budget["dense"].unrequested_raw_hits == 30
    assert pool.budget["bm25"].requested_raw_hits == 40
    assert pool.budget["bm25"].unrequested_raw_hits == 20
    assert len(backend.calls) == 3


def test_saturation_does_not_trigger_backfill_or_cross_channel_transfer(backend):
    backend.actual_overrides["dense", "исходный"] = 1
    settings = RetrievalSettings(
        baseline_target_raw_hits_per_channel=11,
        wide_target_raw_hits_per_channel=11,
        expanded_target_raw_hits_per_channel=11,
        expanded_original_query_target_raw_hits=11,
    )
    pool = CandidateBuilder(backend, settings).build(
        query_group="g",
        query="исходный",
        channels=("dense", "bm25"),
    )
    assert pool.budget["dense"].status == "saturated"
    assert pool.budget["dense"].actual_raw_hits == 1
    assert pool.budget["dense"].requested_raw_hits == 11
    assert pool.budget["bm25"].requested_raw_hits == 11
    assert len(backend.calls) == 2


def test_wide_budget_uses_only_original_query(backend):
    pool = CandidateBuilder(backend).build(
        query_group="g",
        query="исходный",
        channels=("dense", "bm25"),
        wide=True,
    )
    assert [call[2] for call in backend.calls] == [60, 60]
    assert all(call[1] == "исходный" for call in backend.calls)
    assert all(budget.requested_raw_hits == 60 for budget in pool.budget.values())
    assert all(not queries for queries in pool.generated_queries.values())


def test_duplicate_raw_hits_never_create_duplicate_pool_candidates(chunks):
    class DuplicateBackend:
        def search(self, channel, query, requested_hits):
            from retrieval.npa_mas.candidates import RawSearchHit

            return [RawSearchHit(chunks[0], 2.0, 1), RawSearchHit(chunks[0], 1.0, 2)]

    pool = CandidateBuilder(DuplicateBackend()).build(
        query_group="g",
        query="q",
        channels=("bm25",),
        generated_queries={"bm25": ["q1"]},
    )
    assert pool.candidate_ids == ["chunk-000"]
    assert pool.budget["bm25"].actual_raw_hits == 4
    assert len(pool.provenance["raw_hits"]["bm25"][0]["hits"]) == 2


def test_pool_hash_covers_order_text_title_doc_path_and_ignores_trace_metadata(backend):
    pool = CandidateBuilder(backend).build(query_group="g", query="q", channels=("bm25",))
    candidates = pool.candidates
    original = compute_pool_sha256(candidates)
    assert compute_pool_sha256(list(reversed(candidates))) != original
    for update in (
        {"selector_text": "изменённый текст"},
        {"title": "Другой заголовок"},
        {"doc_path": "other.md"},
    ):
        changed = [candidates[0].model_copy(update=update), *candidates[1:]]
        assert compute_pool_sha256(changed) != original
    metadata_changed = [
        candidates[0].model_copy(update={"rrf_score": 999.0, "source_ranks": {"other": 99}}),
        *candidates[1:],
    ]
    assert compute_pool_sha256(metadata_changed) == original


def test_selector_text_is_identical_bounded_prefix(chunks, backend):
    chunks[0].text = "я" * 5000
    pool = CandidateBuilder(backend).build(query_group="g", query="q", channels=("bm25",))
    candidate = next(candidate for candidate in pool.candidates if candidate.chunk_id == chunks[0].id)
    assert candidate.selector_text == chunks[0].text[:2000]
    assert len(candidate.selector_text) == 2000
    assert candidate.text == chunks[0].text


def test_engine_adapter_uses_existing_indices_and_explicit_threshold(chunks):
    calls = []

    class Index:
        def __init__(self):
            self.chunks = chunks

        def search(self, query, top_k, score_threshold):
            calls.append((query, top_k, score_threshold))
            return [(0, 0.75)]

    engine = SimpleNamespace(
        embedding=Index(),
        bm25=Index(),
        db=SimpleNamespace(chunks=chunks),
    )
    adapter = EngineSearchAdapter(engine, score_threshold=0.25)
    dense = adapter.search("dense", "q", 1)
    lexical = adapter.search("bm25", "q", 1)
    assert dense[0].chunk.id == chunks[0].id
    assert lexical[0].chunk.id == chunks[0].id
    assert calls == [("q", 1, 0.25), ("q", 1, 0.25)]
