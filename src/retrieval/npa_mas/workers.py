from __future__ import annotations

from retrieval.npa_mas.llm import JsonLLMGateway, PromptLoader, mark_fallback, render_prompt
from retrieval.npa_mas.schemas import (
    CallTrace,
    Channel,
    QueryPlan,
    SingleRewriterOutput,
    WorkerQueries,
)


class GenericSingleRewriter:
    stage = "single_rewriter"

    def __init__(self, gateway: JsonLLMGateway, prompts: PromptLoader, max_output_tokens: int = 1200):
        self.gateway = gateway
        self.prompts = prompts
        self.max_output_tokens = max_output_tokens

    async def generate(self, query: str) -> tuple[dict[str, WorkerQueries], CallTrace, str | None]:
        prompt = render_prompt(self.prompts.load("single_rewriter.txt"), query=query)
        result = await self.gateway.generate_json(
            stage=self.stage,
            prompt=prompt,
            response_model=SingleRewriterOutput,
            max_output_tokens=self.max_output_tokens,
        )
        if result.value is None:
            reason = result.error or "single_rewriter_failed"
            return self._fallback_queries(), mark_fallback(result.trace, reason), reason
        value = result.value
        return (
            {
                "bm25": WorkerQueries(
                    channel="bm25",
                    queries=value.lexical_queries,
                    aspects_addressed=[],
                ),
                "dense": WorkerQueries(
                    channel="dense",
                    queries=value.semantic_queries,
                    aspects_addressed=[],
                ),
            },
            result.trace,
            None,
        )

    @staticmethod
    def _fallback_queries() -> dict[str, WorkerQueries]:
        return {
            channel: WorkerQueries(channel=channel, queries=[], aspects_addressed=[], fallback_used=True)
            for channel in ("bm25", "dense")
        }


class SpecializedWorker:
    def __init__(
        self,
        *,
        channel: Channel,
        gateway: JsonLLMGateway,
        prompts: PromptLoader,
        max_output_tokens: int = 800,
    ):
        self.channel = channel
        self.gateway = gateway
        self.prompts = prompts
        self.max_output_tokens = max_output_tokens
        self.stage = f"{channel}_worker"
        self.prompt_filename = f"{channel}_worker.txt"

    async def generate(
        self,
        query: str,
        plan: QueryPlan | None = None,
    ) -> tuple[WorkerQueries, CallTrace, str | None]:
        prompt = render_prompt(
            self.prompts.load(self.prompt_filename),
            query=query,
            plan_json=plan.model_dump(mode="json") if plan is not None else None,
        )
        result = await self.gateway.generate_json(
            stage=self.stage,
            prompt=prompt,
            response_model=WorkerQueries,
            max_output_tokens=self.max_output_tokens,
        )
        if result.value is None:
            reason = result.error or f"{self.stage}_failed"
            return self._fallback(), mark_fallback(result.trace, reason), reason
        if result.value.channel != self.channel:
            reason = "invalid_worker_channel"
            trace = mark_fallback(
                result.trace,
                reason,
                event="schema_violation",
                error_class="WorkerChannelMismatch",
            )
            return self._fallback(), trace, reason
        value = result.value
        if value.fallback_used:
            reason = f"{self.stage}_declared_fallback"
            return self._fallback(), mark_fallback(result.trace, reason, event="declared_fallback"), reason
        return value, result.trace, None

    def _fallback(self) -> WorkerQueries:
        return WorkerQueries(
            channel=self.channel,
            queries=[],
            aspects_addressed=[],
            fallback_used=True,
        )


class BM25Worker(SpecializedWorker):
    def __init__(self, gateway: JsonLLMGateway, prompts: PromptLoader, max_output_tokens: int = 800):
        super().__init__(
            channel="bm25",
            gateway=gateway,
            prompts=prompts,
            max_output_tokens=max_output_tokens,
        )


class DenseWorker(SpecializedWorker):
    def __init__(self, gateway: JsonLLMGateway, prompts: PromptLoader, max_output_tokens: int = 800):
        super().__init__(
            channel="dense",
            gateway=gateway,
            prompts=prompts,
            max_output_tokens=max_output_tokens,
        )
