from __future__ import annotations

import asyncio
from types import SimpleNamespace

from retrieval.npa_mas.llm import AsyncJsonLLMGateway, OutputValidationError, PromptLoader
from retrieval.npa_mas.schemas import QueryPlan, SelectorPermutation, SingleRewriterOutput
from retrieval.npa_mas.workers import GenericSingleRewriter


def response(content, usage="reported"):
    if usage == "reported":
        usage = SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9)
    elif usage == "missing":
        usage = None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


class Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client_for(responses):
    completions = Completions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def gateway_for(responses):
    client, completions = client_for(responses)
    gateway = AsyncJsonLLMGateway(
        model="fake-model",
        client=client,
        backoff_s=0,
        timeout_s=180,
    )
    return gateway, completions


def test_json_success_and_token_usage():
    gateway, completions = gateway_for([response('{"lexical_queries":["q"],"semantic_queries":[]}')])
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="prompt",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    assert result.value.lexical_queries == ["q"]
    assert result.trace.status == "success"
    assert result.trace.token_usage[0].total_tokens == 9
    assert completions.calls[0]["temperature"] == 0.0
    assert completions.calls[0]["max_tokens"] == 100


def test_missing_usage_is_not_zero():
    gateway, _ = gateway_for([response('{"lexical_queries":[],"semantic_queries":[]}', usage="missing")])
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="prompt",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    usage = result.trace.token_usage[0]
    assert usage.token_usage_status == "missing"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_empty_usage_object_is_missing_not_reported():
    empty = SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    gateway, _ = gateway_for([response('{"lexical_queries":[],"semantic_queries":[]}', usage=empty)])
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="q",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    usage = result.trace.token_usage[0]
    assert usage.token_usage_status == "missing"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_two_transient_retries_then_success():
    gateway, completions = gateway_for(
        [asyncio.TimeoutError(), asyncio.TimeoutError(), response('{"lexical_queries":[],"semantic_queries":[]}')]
    )
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="prompt",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    assert result.trace.status == "success_after_transient_retry"
    assert result.trace.attempts == 3
    assert len(completions.calls) == 3
    assert [event.event for event in result.trace.events] == ["timeout", "timeout"]


def test_malformed_json_gets_exactly_one_repair():
    gateway, completions = gateway_for(
        [response("not-json"), response('{"lexical_queries":[],"semantic_queries":[]}')]
    )
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="prompt",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    assert result.trace.status == "success_after_json_repair"
    assert result.trace.attempts == 2
    assert result.trace.events[0].event == "malformed_json"
    assert "Исправь ответ" in completions.calls[1]["messages"][0]["content"]


def test_second_malformed_response_fails_after_one_repair():
    gateway, completions = gateway_for([response("bad"), response("still-bad")])
    result = asyncio.run(
        gateway.generate_json(
            stage="single_rewriter",
            prompt="prompt",
            response_model=SingleRewriterOutput,
            max_output_tokens=100,
        )
    )
    assert result.value is None
    assert result.trace.status == "failed"
    assert result.trace.attempts == 2
    assert len(completions.calls) == 2
    assert [event.event for event in result.trace.events] == ["malformed_json", "malformed_json"]


def test_schema_violation_uses_repair_call():
    gateway, _ = gateway_for(
        [response('{"aspects":[],"fallback_used":false}'), response('{"aspects":[],"planner_notes":"x","fallback_used":false}')]
    )
    result = asyncio.run(
        gateway.generate_json(
            stage="planner",
            prompt="prompt",
            response_model=QueryPlan,
            max_output_tokens=100,
        )
    )
    assert result.trace.status == "success_after_json_repair"
    assert result.trace.events[0].event == "schema_violation"


def test_semantic_selector_validation_is_repaired():
    gateway, _ = gateway_for([response('{"ranking":["missing"]}'), response('{"ranking":["ok"]}')])

    def validator(value):
        if value.ranking != ["ok"]:
            raise OutputValidationError("missing_candidate_id", "incomplete permutation")
        return value

    result = asyncio.run(
        gateway.generate_json(
            stage="llm_selector",
            prompt="prompt",
            response_model=SelectorPermutation,
            max_output_tokens=100,
            validator=validator,
        )
    )
    assert result.trace.status == "success_after_json_repair"
    assert result.trace.events[0].event == "missing_candidate_id"
    assert result.value.ranking == ["ok"]


def test_failed_repair_becomes_declared_rewriter_fallback(prompts_dir):
    llm_gateway, _ = gateway_for([response("bad"), response("worse")])
    rewriter = GenericSingleRewriter(llm_gateway, PromptLoader(prompts_dir))
    queries, trace, reason = asyncio.run(rewriter.generate("запрос"))
    assert reason == "malformed_json"
    assert trace.status == "fallback"
    assert trace.fallback_used
    assert trace.attempts == 2
    assert [event.event for event in trace.events] == ["malformed_json", "malformed_json"]
    assert all(item.fallback_used and item.queries == [] for item in queries.values())
