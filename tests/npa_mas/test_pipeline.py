from __future__ import annotations

import asyncio
import json

import pytest

from retrieval.npa_mas.pipeline import NpaMasPipeline, SamePoolError
from retrieval.npa_mas.variants import CORE_VARIANT_ORDER


def make_pipeline(backend, gateway, jina, prompts_dir):
    return NpaMasPipeline(
        search_backend=backend,
        llm_gateway=gateway,
        jina_reranker=jina,
        prompts_dir=prompts_dir,
    )


def run(pipeline, variant, query="Что регулирует статья 5?", **kwargs):
    return asyncio.run(pipeline.run(query, variant, query_group="group-1", **kwargs))


def test_all_variants_call_counts_and_output_contract(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = None
    expected_calls = {
        "B0_BM25": [],
        "B0_DENSE": [],
        "B0_RRF": [],
        "B1": [],
        "B1W": [],
        "S": ["single_rewriter"],
        "M1": ["bm25_worker", "dense_worker"],
        "M2": ["planner", "bm25_worker", "dense_worker"],
        "M3": ["llm_selector"],
        "M4": ["worker_report", "worker_report", "llm_selector"],
    }
    assert tuple(expected_calls) == CORE_VARIANT_ORDER
    for variant in CORE_VARIANT_ORDER:
        gateway.calls.clear()
        kwargs = {"m2_artifact": m2} if variant in {"M3", "M4"} else {}
        result = run(pipeline, variant, **kwargs)
        assert [call["stage"] for call in gateway.calls] == expected_calls[variant]
        assert result.status == "success"
        assert not result.strict_failed
        assert len(result.ranked_chunk_ids) <= 10
        assert len(result.ranked_chunk_ids) == len(set(result.ranked_chunk_ids))
        assert set(result.ranked_chunk_ids) <= set(result.pool.candidate_ids)
        assert result.model_dump(mode="json")["variant"] == variant
        if variant == "M2":
            m2 = result
        if variant == "M3":
            assert result.reports == {}
        if variant == "M4":
            assert set(result.reports) == {"bm25", "dense"}
    assert m2.pool.pool_sha256 == result.pool.pool_sha256


def test_m2_m3_m4_consume_identical_frozen_pool(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    original_hash = m2.pool.pool_sha256
    search_calls = len(backend.calls)
    m3 = run(pipeline, "M3", m2_artifact=m2)
    m4 = run(pipeline, "M4", m2_artifact=m2.model_dump(mode="json"))
    assert len(backend.calls) == search_calls
    assert m2.pool.pool_sha256 == m3.pool.pool_sha256 == m4.pool.pool_sha256 == original_hash
    assert m2.pool.candidate_ids == m3.pool.candidate_ids == m4.pool.candidate_ids
    assert m2.pool.selector_texts == m3.pool.selector_texts == m4.pool.selector_texts
    assert m2.pool.pool_sha256 == original_hash
    assert [call.stage for call in m4.calls] == ["worker_report", "worker_report", "llm_selector"]


def test_jina_and_llm_receive_identical_selector_text(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    gateway.calls.clear()
    m3 = run(pipeline, "M3", m2_artifact=m2)
    jina_texts = [item["text"] for item in jina.calls[-1]["chunks"]]
    prompt = gateway.calls[-1]["prompt"]
    raw = prompt.split("Ordered candidates:\n", 1)[1].split("\n\nOptional worker reports:", 1)[0]
    llm_candidates = json.loads(raw)
    llm_texts = [item["selector_text"] for item in llm_candidates]
    assert jina_texts == llm_texts == m2.pool.selector_texts == m3.pool.selector_texts


def test_generic_rewriter_fallback_uses_original_only_and_strict_empty(backend, gateway, jina, prompts_dir):
    gateway.set_response("single_rewriter", RuntimeError("provider down"))
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    result = run(pipeline, "S")
    assert result.status == "complete_with_fallback"
    assert result.fallback_used
    assert result.strict_failed
    assert result.fallback_stage == "single_rewriter"
    assert len(backend.calls) == 2
    assert [call[2] for call in backend.calls] == [30, 30]
    assert result.ranked_chunk_ids
    assert result.model_dump(mode="json")["calls"][0]["status"] == "fallback"
    strict_result = run(pipeline, "S", strict=True)
    assert strict_result.status == "failed"
    assert strict_result.ranked_chunk_ids == []
    assert strict_result.strict_failed


def test_planner_fallback_runs_workers_without_assignments(backend, gateway, jina, prompts_dir):
    gateway.set_response("planner", RuntimeError("planner down"))
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    result = run(pipeline, "M2")
    assert result.fallback_stage == "planner"
    assert result.plan.fallback_used
    worker_prompts = [call["prompt"] for call in gateway.calls if call["stage"].endswith("_worker")]
    assert len(worker_prompts) == 2
    assert all("QueryPlan или null:\nnull" in prompt for prompt in worker_prompts)
    assert result.ranked_chunk_ids


def test_worker_failure_compensates_neither_queries_nor_budget(backend, gateway, jina, prompts_dir):
    gateway.set_response("bm25_worker", RuntimeError("worker down"))
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    result = run(pipeline, "M1")
    bm25_calls = [call for call in backend.calls if call[0] == "bm25"]
    dense_calls = [call for call in backend.calls if call[0] == "dense"]
    assert len(bm25_calls) == 1
    assert bm25_calls[0][2] == 30
    assert len(dense_calls) == 3
    assert [call[2] for call in dense_calls] == [30, 10, 10]
    assert result.pool.budget["bm25"].requested_raw_hits == 30
    assert result.pool.budget["dense"].requested_raw_hits == 50
    assert result.fallback_used


def test_declared_worker_fallback_is_original_only_in_pipeline(backend, gateway, jina, prompts_dir):
    gateway.set_response("bm25_worker", {
        "channel": "bm25",
        "queries": ["запрещённый дополнительный запрос"],
        "aspects_addressed": [],
        "fallback_used": True,
    })
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    result = run(pipeline, "M1")
    bm25_calls = [call for call in backend.calls if call[0] == "bm25"]
    assert len(bm25_calls) == 1
    assert bm25_calls[0][2] == 30
    assert result.pool.budget["bm25"].requested_raw_hits == 30
    assert result.worker_queries["bm25"].queries == []
    assert result.fallback_used
    assert all("запрещённый" not in call[1] for call in bm25_calls)


def test_llm_selector_failure_uses_rrf_pool_order(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    gateway.set_response("llm_selector", RuntimeError("selector down"))
    m3 = run(pipeline, "M3", m2_artifact=m2)
    assert m3.status == "complete_with_fallback"
    assert m3.strict_failed
    assert m3.ranked_chunk_ids == m2.pool.candidate_ids[:10]
    assert m3.fallback_stage == "llm_selector"


def test_failed_report_is_explicit_and_cannot_change_pool(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    gateway.set_response("worker_report", [RuntimeError("report down"), None])
    m4 = run(pipeline, "M4", m2_artifact=m2)
    assert m4.pool.pool_sha256 == m2.pool.pool_sha256
    assert m4.reports["bm25"].status == "failed"
    assert m4.reports["bm25"].fallback_used
    assert m4.reports["dense"].status == "success"
    assert m4.fallback_used
    assert set(m4.ranked_chunk_ids) <= set(m2.pool.candidate_ids)


def test_report_with_new_id_is_rejected_without_adding_candidate(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    original_default = gateway._default

    def bad_report(*, stage, prompt, response_model):
        report = original_default(stage, prompt)
        report.observed_chunk_ids.append("new-id")
        report.supporting_chunk_ids.append("new-id")
        return report

    gateway.set_response("worker_report", bad_report)
    m4 = run(pipeline, "M4", m2_artifact=m2)
    assert "new-id" not in m4.pool.candidate_ids
    assert "new-id" not in m4.ranked_chunk_ids
    assert all(report.status == "failed" for report in m4.reports.values())
    assert m4.pool.pool_sha256 == m2.pool.pool_sha256


def test_search_failure_is_not_faked_as_successful_fallback(backend, gateway, jina, prompts_dir):
    backend.failures["dense", "Что регулирует статья 5?"] = "index unavailable"
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    result = run(pipeline, "B1")
    assert result.status == "failed"
    assert result.strict_failed
    assert not result.fallback_used
    assert result.ranked_chunk_ids == []
    assert result.error == "search_error"


def test_m2_search_failure_preserves_pool_for_failed_m3_m4(backend, gateway, jina, prompts_dir):
    backend.failures["dense", "Что регулирует статья 5?"] = "index unavailable"
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    assert m2.status == "failed"
    assert m2.pool is not None
    for variant in ("M3", "M4"):
        gateway.calls.clear()
        result = run(pipeline, variant, m2_artifact=m2)
        assert result.status == "failed"
        assert result.strict_failed
        assert result.error == "search_error"
        assert result.ranked_chunk_ids == []
        assert result.full_ranked_chunk_ids == []
        assert result.pool.pool_sha256 == m2.pool.pool_sha256
        assert gateway.calls == []


def test_mismatched_frozen_artifact_is_protocol_error(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    with pytest.raises(SamePoolError):
        asyncio.run(
            pipeline.run(
                "другой запрос",
                "M3",
                query_group="group-1",
                m2_artifact=m2,
            )
        )
    with pytest.raises(SamePoolError):
        asyncio.run(
            pipeline.run(
                "Что регулирует статья 5?",
                "M4",
                query_group="other-group",
                m2_artifact=m2,
            )
        )


def test_planner_component_has_no_search_capability(gateway, prompts_dir):
    from retrieval.npa_mas.llm import PromptLoader
    from retrieval.npa_mas.planner import PlannerAgent

    planner = PlannerAgent(gateway, PromptLoader(prompts_dir))
    assert not hasattr(planner, "search")
    assert not hasattr(planner, "engine")
    plan, trace, reason = asyncio.run(planner.plan("запрос"))
    assert reason is None
    assert plan.aspects[0].aspect_id == "a1"
    assert trace.status == "success"


def test_m3_inherits_m2_generation_fallback_without_replaying_calls(backend, gateway, jina, prompts_dir):
    gateway.set_response("planner", RuntimeError("planner down"))
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    gateway.calls.clear()
    m3 = run(pipeline, "M3", m2_artifact=m2)
    assert [call["stage"] for call in gateway.calls] == ["llm_selector"]
    assert m3.fallback_used
    assert m3.strict_failed
    assert m3.fallback_stage == "planner"
    assert m3.status == "complete_with_fallback"
    strict_m3 = run(pipeline, "M3", m2_artifact=m2, strict=True)
    assert strict_m3.status == "failed"
    assert strict_m3.ranked_chunk_ids == []


def test_m3_reports_block_is_null_and_m4_has_two_reports(backend, gateway, jina, prompts_dir):
    pipeline = make_pipeline(backend, gateway, jina, prompts_dir)
    m2 = run(pipeline, "M2")
    gateway.calls.clear()
    run(pipeline, "M3", m2_artifact=m2)
    m3_prompt = gateway.calls[-1]["prompt"]
    assert m3_prompt.strip().endswith("Optional worker reports:\nnull")
    gateway.calls.clear()
    run(pipeline, "M4", m2_artifact=m2)
    report_prompts = [call["prompt"] for call in gateway.calls if call["stage"] == "worker_report"]
    observations = [json.loads(prompt.split("Observations одного worker:\n", 1)[1]) for prompt in report_prompts]
    channels = [observation["channel"] for observation in observations]
    assert channels == ["bm25", "dense"]
    assert all("score" not in hit for observation in observations for hit in observation["hits"])
    selector_prompt = gateway.calls[-1]["prompt"]
    reports = json.loads(selector_prompt.split("Optional worker reports:\n", 1)[1])
    assert set(reports) == {"bm25", "dense"}
