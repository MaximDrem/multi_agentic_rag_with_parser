from __future__ import annotations

from retrieval.npa_mas.llm import JsonLLMGateway, PromptLoader, mark_fallback, render_prompt
from retrieval.npa_mas.schemas import CallTrace, QueryPlan


class PlannerAgent:
    stage = "planner"

    def __init__(
        self,
        gateway: JsonLLMGateway,
        prompts: PromptLoader,
        *,
        max_aspects: int = 4,
        max_output_tokens: int = 800,
    ):
        self.gateway = gateway
        self.prompts = prompts
        self.max_aspects = max_aspects
        self.max_output_tokens = max_output_tokens

    async def plan(self, query: str) -> tuple[QueryPlan, CallTrace, str | None]:
        prompt = render_prompt(
            self.prompts.load("planner.txt"),
            query=query,
            max_aspects=str(self.max_aspects),
        )
        result = await self.gateway.generate_json(
            stage="planner",
            prompt=prompt,
            response_model=QueryPlan,
            max_output_tokens=self.max_output_tokens,
        )
        if result.value is None:
            reason = result.error or "planner_failed"
            return self._fallback(), mark_fallback(result.trace, reason), reason
        aspect_ids = [aspect.aspect_id for aspect in result.value.aspects]
        if len(aspect_ids) > self.max_aspects or len(aspect_ids) != len(set(aspect_ids)):
            reason = "invalid_query_plan"
            trace = mark_fallback(
                result.trace,
                reason,
                event="schema_violation",
                error_class="QueryPlanConstraintError",
            )
            return self._fallback(), trace, reason
        value = result.value
        if value.fallback_used:
            reason = "planner_declared_fallback"
            return value, mark_fallback(result.trace, reason, event="declared_fallback"), reason
        return value, result.trace, None

    @staticmethod
    def _fallback() -> QueryPlan:
        return QueryPlan(aspects=[], planner_notes="", fallback_used=True)
