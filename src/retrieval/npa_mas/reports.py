from __future__ import annotations

from typing import Any, Sequence

from retrieval.npa_mas.llm import JsonLLMGateway, PromptLoader, mark_fallback, render_prompt
from retrieval.npa_mas.schemas import CallTrace, Channel, QueryPlan, ReportSettings, WorkerQueries, WorkerReport


class WorkerReportGenerator:
    stage = "worker_report"

    def __init__(
        self,
        gateway: JsonLLMGateway,
        prompts: PromptLoader,
        settings: ReportSettings | None = None,
        *,
        max_output_tokens: int = 1200,
    ):
        self.gateway = gateway
        self.prompts = prompts
        self.settings = settings or ReportSettings()
        self.max_output_tokens = max_output_tokens

    async def generate(
        self,
        *,
        query: str,
        channel: Channel,
        worker_queries: WorkerQueries,
        observations: Sequence[dict[str, Any]],
        plan: QueryPlan | None = None,
    ) -> tuple[WorkerReport, CallTrace, str | None]:
        sanitized = []
        observed_ids = []
        seen = set()
        for observation in observations:
            chunk_id = str(observation["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if len(sanitized) >= self.settings.max_observed_chunks_per_worker:
                continue
            observed_ids.append(chunk_id)
            sanitized.append(
                {
                    "chunk_id": chunk_id,
                    "text": str(observation.get("text", ""))[: self.settings.text_max_chars],
                }
            )
        observation_payload = {
            "channel": channel,
            "generated_queries": worker_queries.queries,
            "aspects_addressed": worker_queries.aspects_addressed,
            "status": "fallback" if worker_queries.fallback_used else "success",
            "hits": sanitized,
        }
        prompt = render_prompt(
            self.prompts.load("worker_report.txt"),
            query=query,
            plan_json=plan.model_dump(mode="json") if plan is not None else None,
            observations_json=observation_payload,
        )
        result = await self.gateway.generate_json(
            stage=self.stage,
            prompt=prompt,
            response_model=WorkerReport,
            max_output_tokens=self.max_output_tokens,
        )
        if result.value is None:
            reason = result.error or "worker_report_failed"
            return self._fallback(channel, reason), mark_fallback(result.trace, reason), reason
        value = result.value
        allowed = set(observed_ids)
        invalid_ids = (
            value.channel != channel
            or len(value.observed_chunk_ids) != len(set(value.observed_chunk_ids))
            or len(value.supporting_chunk_ids) != len(set(value.supporting_chunk_ids))
            or value.observed_chunk_ids != observed_ids
            or not set(value.supporting_chunk_ids) <= allowed
            or value.generated_queries != worker_queries.queries
            or value.aspects_addressed != worker_queries.aspects_addressed
            or len(value.gaps) > self.settings.max_gap_items
        )
        if invalid_ids:
            reason = "invalid_worker_report"
            trace = mark_fallback(
                result.trace,
                reason,
                event="invalid_candidate_id",
                error_class="WorkerReportConstraintError",
            )
            return self._fallback(channel, reason), trace, reason
        declared_failure = value.fallback_used or value.status == "failed" or bool(value.error)
        if declared_failure:
            reason = value.error or (
                "worker_report_declared_failed" if value.status == "failed" else "worker_report_declared_fallback"
            )
            return self._fallback(channel, reason), mark_fallback(result.trace, reason, event="declared_fallback"), reason
        return value, result.trace, None

    @staticmethod
    def _fallback(channel: Channel, reason: str) -> WorkerReport:
        return WorkerReport(
            channel=channel,
            aspects_addressed=[],
            generated_queries=[],
            observed_chunk_ids=[],
            coverage="",
            gaps=[],
            supporting_chunk_ids=[],
            fallback_used=True,
            status="failed",
            error=reason,
        )
