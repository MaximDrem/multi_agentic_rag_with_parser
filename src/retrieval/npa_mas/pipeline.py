from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from retrieval.npa_mas.candidates import CandidateBuilder, EngineSearchAdapter
from retrieval.npa_mas.llm import AsyncJsonLLMGateway, JsonLLMGateway, PromptLoader
from retrieval.npa_mas.planner import PlannerAgent
from retrieval.npa_mas.reports import WorkerReportGenerator
from retrieval.npa_mas.schemas import (
    CallTrace,
    CandidatePool,
    FallbackTrace,
    NpaMasConfig,
    PipelineResult,
    QueryPlan,
    RankedChunk,
    TokenUsage,
    WorkerQueries,
    WorkerReport,
)
from retrieval.npa_mas.selectors import JinaSelector, LLMSelector, ranked_chunk
from retrieval.npa_mas.variants import CORE_VARIANT_ORDER, get_variant
from retrieval.npa_mas.workers import BM25Worker, DenseWorker, GenericSingleRewriter


class SamePoolError(ValueError):
    pass


def assert_same_pool(reference: CandidatePool, candidate: CandidatePool) -> None:
    try:
        reference.assert_same_pool(candidate)
    except ValueError as exc:
        raise SamePoolError(str(exc)) from exc


class NpaMasPipeline:
    def __init__(
        self,
        engine: Any | None = None,
        *,
        search_backend: Any | None = None,
        llm_gateway: JsonLLMGateway | None = None,
        jina_reranker: Any | None = None,
        config: NpaMasConfig | None = None,
        prompts_dir: str | Path = Path("prompts/npa_mas_component_v1"),
        score_threshold: float | None = None,
        strict: bool = False,
    ):
        if engine is not None and search_backend is not None:
            raise ValueError("pass either engine or search_backend, not both")
        if search_backend is None:
            if engine is None:
                from retrieval.engine import RetrievalEngine

                engine = RetrievalEngine()
            search_backend = EngineSearchAdapter(engine, score_threshold=score_threshold)
        self.engine = engine
        self.search_backend = search_backend
        self.llm_gateway = llm_gateway
        self.config = config or NpaMasConfig()
        self.prompts = PromptLoader(prompts_dir)
        self.strict = strict
        self.candidate_builder = CandidateBuilder(search_backend, self.config.retrieval)
        self.jina_selector = JinaSelector(jina_reranker, output_chunks=self.config.retrieval.output_chunks)

    async def run(
        self,
        query: str,
        variant: str,
        *,
        query_group: str | None = None,
        m2_artifact: PipelineResult | CandidatePool | Mapping[str, Any] | None = None,
        strict: bool | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        spec = get_variant(variant)
        group = query_group or query
        strict_mode = self.strict if strict is None else strict
        calls: list[CallTrace] = []
        fallbacks: list[FallbackTrace] = []
        stage_timings: dict[str, float] = {}
        operation_usage: list[TokenUsage] = []
        plan: QueryPlan | None = None
        worker_queries: dict[str, WorkerQueries] = {}
        reports: dict[str, WorkerReport] = {}
        pool: CandidatePool | None = None
        frozen_reference: CandidatePool | None = None
        chunks: list[RankedChunk] = []
        full_ranked_ids: list[str] = []
        error: str | None = None
        strict_failed = False

        if spec.requires_m2_pool:
            pool, worker_queries, frozen_reference, plan = self._frozen_m2_pool(
                m2_artifact,
                query=query,
                query_group=group,
            )
            assert_same_pool(frozen_reference, pool)
            fallbacks.extend(self._inherited_m2_fallbacks(m2_artifact))
            if pool.has_search_errors:
                error = "search_error"
                strict_failed = True
            stage_timings["pool_reuse"] = 0.0
        else:
            generation_started = time.perf_counter()
            if spec.query_generation == "single":
                gateway = self._gateway()
                rewriter = GenericSingleRewriter(
                    gateway,
                    self.prompts,
                    self.config.llm.max_output_tokens["single_rewriter"],
                )
                rewritten, trace, reason = await rewriter.generate(query)
                worker_queries.update(rewritten)
                calls.append(trace)
                self._record_fallback(fallbacks, "single_rewriter", reason)
            elif spec.query_generation == "specialized":
                gateway = self._gateway()
                if spec.planner:
                    planner = PlannerAgent(
                        gateway,
                        self.prompts,
                        max_aspects=self.config.planner.max_aspects,
                        max_output_tokens=self.config.llm.max_output_tokens["planner"],
                    )
                    plan, trace, reason = await planner.plan(query)
                    calls.append(trace)
                    self._record_fallback(fallbacks, "planner", reason)
                for channel, worker_class in (("bm25", BM25Worker), ("dense", DenseWorker)):
                    worker = worker_class(
                        gateway,
                        self.prompts,
                        self.config.llm.max_output_tokens[f"{channel}_worker"],
                    )
                    worker_plan = plan if plan is not None and not plan.fallback_used else None
                    queries, trace, reason = await worker.generate(query, worker_plan)
                    worker_queries[channel] = queries
                    calls.append(trace)
                    self._record_fallback(fallbacks, f"{channel}_worker", reason)
            stage_timings["query_generation"] = (time.perf_counter() - generation_started) * 1000

            retrieval_started = time.perf_counter()
            generated = None
            if spec.query_generation in {"single", "specialized"}:
                generated = {
                    channel: worker_queries[channel].queries
                    for channel in spec.channels
                }
            pool = self.candidate_builder.build(
                query_group=group,
                query=query,
                channels=spec.channels,
                generated_queries=generated,
                wide=spec.budget == "wide",
            )
            stage_timings["candidate_generation"] = (time.perf_counter() - retrieval_started) * 1000
            operation_usage.extend(
                TokenUsage.not_applicable(
                    stage=f"{channel}_search",
                    latency_ms=stage_timings["candidate_generation"],
                )
                for channel in spec.channels
            )
            operation_usage.append(
                TokenUsage.not_applicable(
                    stage="rrf_fusion",
                    latency_ms=stage_timings["candidate_generation"],
                )
            )
            if pool.has_search_errors:
                error = "search_error"
                strict_failed = True

        if error is None and pool is not None:
            selector_started = time.perf_counter()
            if spec.selector in {"raw", "rrf"}:
                full_ranked_ids = pool.candidate_ids
                chunks = [
                    ranked_chunk(candidate, index, candidate.rrf_score)
                    for index, candidate in enumerate(pool.candidates[: self.config.retrieval.output_chunks], 1)
                ]
            elif spec.selector == "jina":
                try:
                    full_chunks = await asyncio.to_thread(self.jina_selector.rank_full, query, pool)
                    full_ranked_ids = [chunk.chunk_id for chunk in full_chunks]
                    chunks = full_chunks[: self.config.retrieval.output_chunks]
                except Exception as exc:
                    error = f"jina_error:{type(exc).__name__}"
                    chunks = []
                    full_ranked_ids = []
                    strict_failed = True
                finally:
                    stage_timings["jina_selector"] = (time.perf_counter() - selector_started) * 1000
                    operation_usage.append(
                        TokenUsage.not_applicable(
                            stage="jina_selector",
                            latency_ms=stage_timings["jina_selector"],
                        )
                    )
            else:
                gateway = self._gateway()
                if spec.reports:
                    reports_started = time.perf_counter()
                    report_generator = WorkerReportGenerator(
                        gateway,
                        self.prompts,
                        self.config.reports,
                        max_output_tokens=self.config.llm.max_output_tokens["worker_report"],
                    )
                    observations = pool.provenance.get("observations", {})
                    for channel in ("bm25", "dense"):
                        queries = worker_queries.get(channel) or WorkerQueries(
                            channel=channel,
                            queries=pool.generated_queries.get(channel, []),
                            aspects_addressed=[],
                        )
                        report, trace, reason = await report_generator.generate(
                            query=query,
                            channel=channel,
                            worker_queries=queries,
                            observations=observations.get(channel, []),
                            plan=plan,
                        )
                        reports[channel] = report
                        calls.append(trace)
                        self._record_fallback(fallbacks, f"{channel}_worker_report", reason)
                    if frozen_reference is not None:
                        assert_same_pool(frozen_reference, pool)
                    stage_timings["worker_reports"] = (time.perf_counter() - reports_started) * 1000
                llm_selector_started = time.perf_counter()
                selector = LLMSelector(
                    gateway,
                    self.prompts,
                    output_chunks=self.config.retrieval.output_chunks,
                    max_output_tokens=self.config.llm.max_output_tokens["llm_selector"],
                )
                selection = await selector.rank(query, pool, reports=reports if spec.reports else None)
                chunks = selection.chunks
                full_ranked_ids = selection.full_ranked_chunk_ids
                calls.append(selection.trace)
                self._record_fallback(fallbacks, "llm_selector", selection.fallback_reason)
                stage_timings["llm_selector"] = (time.perf_counter() - llm_selector_started) * 1000

        if frozen_reference is not None:
            assert_same_pool(frozen_reference, pool)

        if fallbacks:
            strict_failed = True
        if strict_mode and strict_failed:
            chunks = []
            if error is None:
                error = "strict_failure_as_empty"
        ranked_ids = [chunk.chunk_id for chunk in chunks]
        if len(ranked_ids) > self.config.retrieval.output_chunks or len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("final ranking violates output size or uniqueness constraints")
        if pool is not None and not set(ranked_ids) <= set(pool.candidate_ids):
            raise ValueError("final ranking contains an out-of-pool candidate")
        fatal_failure = error is not None and error != "strict_failure_as_empty"
        if fatal_failure:
            status = "failed"
            chunks = []
            ranked_ids = []
        elif strict_mode and strict_failed:
            status = "failed"
        elif fallbacks:
            status = "complete_with_fallback"
        else:
            status = "success"
        token_usage = operation_usage + [usage for call in calls for usage in call.token_usage]
        return PipelineResult(
            variant=spec.variant,
            query=query,
            query_group=group,
            status=status,
            strict_failed=strict_failed,
            fallback_used=bool(fallbacks),
            fallback_stage=fallbacks[0].stage if fallbacks else None,
            fallback_reason=fallbacks[0].reason if fallbacks else None,
            fallbacks=fallbacks,
            ranked_chunk_ids=ranked_ids,
            full_ranked_chunk_ids=full_ranked_ids,
            chunks=chunks,
            pool=pool,
            plan=plan,
            worker_queries=worker_queries,
            reports=reports,
            calls=calls,
            token_usage=token_usage,
            stage_timings_ms=stage_timings,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )

    def run_sync(self, query: str, variant: str, **kwargs: Any) -> PipelineResult:
        return asyncio.run(self.run(query, variant, **kwargs))

    def _gateway(self) -> JsonLLMGateway:
        if self.llm_gateway is None:
            if not os.environ.get(self.config.llm.model_env):
                raise RuntimeError(f"{self.config.llm.model_env} is required for LLM variants")
            self.llm_gateway = AsyncJsonLLMGateway.from_env(
                timeout_s=self.config.llm.timeout_s,
                max_transient_retries=self.config.llm.max_transient_retries,
                json_repair_attempts=self.config.llm.json_repair_attempts,
                temperature=self.config.llm.temperature,
            )
        return self.llm_gateway

    @staticmethod
    def _record_fallback(
        fallbacks: list[FallbackTrace],
        stage: str,
        reason: str | None,
    ) -> None:
        if reason is not None:
            fallbacks.append(FallbackTrace(stage=stage, reason=reason))

    @staticmethod
    def _inherited_m2_fallbacks(
        artifact: PipelineResult | CandidatePool | Mapping[str, Any] | None,
    ) -> list[FallbackTrace]:
        if isinstance(artifact, PipelineResult):
            return [fallback.model_copy(deep=True) for fallback in artifact.fallbacks]
        if isinstance(artifact, Mapping):
            try:
                return [FallbackTrace.model_validate(fallback) for fallback in artifact.get("fallbacks", [])]
            except ValidationError as exc:
                raise SamePoolError("frozen M2 fallback metadata failed schema validation") from exc
        return []

    @staticmethod
    def _frozen_m2_pool(
        artifact: PipelineResult | CandidatePool | Mapping[str, Any] | None,
        *,
        query: str,
        query_group: str,
    ) -> tuple[CandidatePool, dict[str, WorkerQueries], CandidatePool, QueryPlan | None]:
        if artifact is None:
            raise SamePoolError("M3/M4 require a frozen M2 artifact")
        try:
            if isinstance(artifact, PipelineResult):
                if artifact.variant != "M2" or artifact.pool is None:
                    raise SamePoolError("frozen artifact must be an M2 result with a pool")
                pool = artifact.pool
                workers = artifact.worker_queries
                plan = artifact.plan
                artifact_query = artifact.query
            elif isinstance(artifact, CandidatePool):
                pool = artifact
                workers = {}
                plan = None
                artifact_query = query
            else:
                if artifact.get("variant", "M2") != "M2" or "pool" not in artifact:
                    raise SamePoolError("frozen artifact must be an M2 result with a pool")
                pool = CandidatePool.model_validate(artifact["pool"])
                workers = {
                    channel: WorkerQueries.model_validate(value)
                    for channel, value in artifact.get("worker_queries", {}).items()
                }
                raw_plan = artifact.get("plan")
                plan = QueryPlan.model_validate(raw_plan) if raw_plan is not None else None
                artifact_query = artifact.get("query", query)
        except ValidationError as exc:
            raise SamePoolError("frozen M2 artifact failed schema validation") from exc
        if artifact_query != query or pool.query_group != query_group:
            raise SamePoolError("frozen M2 query/query_group does not match the requested group")
        reference = pool.model_copy(deep=True)
        working = pool.model_copy(deep=True)
        assert_same_pool(reference, working)
        return working, dict(workers), reference, plan


__all__ = ["CORE_VARIANT_ORDER", "NpaMasPipeline", "SamePoolError", "assert_same_pool"]
