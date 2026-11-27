import asyncio

from retrieval.npa_mas.llm import PromptLoader
from retrieval.npa_mas.planner import PlannerAgent
from retrieval.npa_mas.reports import WorkerReportGenerator
from retrieval.npa_mas.schemas import QueryPlan, WorkerQueries, WorkerReport
from retrieval.npa_mas.workers import BM25Worker


def test_planner_preserves_declared_fallback(gateway, prompts_dir):
    gateway.set_response("planner", QueryPlan(aspects=[], planner_notes="Недостаточно данных", fallback_used=True))
    plan, trace, reason = asyncio.run(PlannerAgent(gateway, PromptLoader(prompts_dir)).plan("запрос"))
    assert plan.fallback_used
    assert plan.planner_notes == ""
    assert reason == "planner_declared_fallback"
    assert trace.fallback_used
    assert trace.status == "fallback"
    assert trace.events[-1].event == "declared_fallback"


def test_worker_normalizes_declared_fallback_to_original_only(gateway, prompts_dir):
    gateway.set_response("bm25_worker", WorkerQueries(
        channel="bm25",
        queries=["точный реквизит"],
        aspects_addressed=[],
        fallback_used=True,
    ))
    queries, trace, reason = asyncio.run(BM25Worker(gateway, PromptLoader(prompts_dir)).generate("запрос"))
    assert queries.fallback_used
    assert queries.queries == []
    assert reason == "bm25_worker_declared_fallback"
    assert trace.fallback_used
    assert trace.status == "fallback"


def test_report_preserves_declared_failure(gateway, prompts_dir):
    worker_queries = WorkerQueries(channel="bm25", queries=["точный реквизит"], aspects_addressed=[])
    gateway.set_response("worker_report", WorkerReport(
        channel="bm25",
        aspects_addressed=[],
        generated_queries=["точный реквизит"],
        observed_chunk_ids=["chunk-1"],
        coverage="",
        gaps=["Нет подтверждения"],
        supporting_chunk_ids=[],
        fallback_used=True,
        status="failed",
        error="declared_worker_failure",
    ))
    report, trace, reason = asyncio.run(WorkerReportGenerator(gateway, PromptLoader(prompts_dir)).generate(
        query="запрос",
        channel="bm25",
        worker_queries=worker_queries,
        observations=[{"chunk_id": "chunk-1", "text": "Текст"}],
    ))
    assert report.fallback_used
    assert report.status == "failed"
    assert report.error == "declared_worker_failure"
    assert report.observed_chunk_ids == []
    assert reason == "declared_worker_failure"
    assert trace.fallback_used
    assert trace.status == "fallback"
