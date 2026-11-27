from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from common.models import Chunk
from retrieval.npa_mas.candidates import RawSearchHit
from retrieval.npa_mas.llm import OutputValidationError
from retrieval.npa_mas.schemas import (
    CallTrace,
    LLMResult,
    QueryAspect,
    QueryPlan,
    ReliabilityEvent,
    SelectorPermutation,
    SingleRewriterOutput,
    TokenUsage,
    WorkerQueries,
    WorkerReport,
)


class FakeBackend:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.calls: list[tuple[str, str, int]] = []
        self.actual_overrides: dict[tuple[str, str], int] = {}
        self.failures: dict[tuple[str, str], str] = {}

    def search(self, channel: str, query: str, requested_hits: int) -> list[RawSearchHit]:
        self.calls.append((channel, query, requested_hits))
        if (channel, query) in self.failures:
            raise RuntimeError(self.failures[channel, query])
        offset = sum(ord(char) for char in query) % len(self.chunks)
        if channel == "bm25":
            offset = (offset + 7) % len(self.chunks)
        ordered = self.chunks[offset:] + self.chunks[:offset]
        actual = self.actual_overrides.get((channel, query), requested_hits)
        return [
            RawSearchHit(chunk, float(len(ordered) - index), index + 1)
            for index, chunk in enumerate(ordered[:actual])
        ]


class FakeJina:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "chunks": chunks})
        return [
            {**chunk, "rerank_score": float(index + 1)}
            for index, chunk in enumerate(reversed(chunks))
        ]


class FakeGateway:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.missing_usage = False

    def set_response(self, stage: str, response: Any) -> None:
        self.responses[stage] = response

    async def generate_json(
        self,
        *,
        stage: str,
        prompt: str,
        response_model: type,
        max_output_tokens: int,
        validator: Any = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "response_model": response_model,
                "max_output_tokens": max_output_tokens,
                "validator": validator,
            }
        )
        response = self.responses.get(stage)
        if isinstance(response, list):
            response = response.pop(0)
        if callable(response):
            response = response(stage=stage, prompt=prompt, response_model=response_model)
        if response is None:
            response = self._default(stage, prompt)
        if isinstance(response, Exception):
            trace = self._trace(stage, "failed", "provider_error", type(response).__name__)
            return LLMResult(value=None, trace=trace, error="provider_error")
        try:
            value = response if isinstance(response, response_model) else response_model.model_validate(response)
            if validator is not None:
                value = validator(value)
        except OutputValidationError as exc:
            trace = self._trace(stage, "failed", exc.event, type(exc).__name__)
            return LLMResult(value=None, trace=trace, error=exc.event)
        return LLMResult(value=value, trace=self._trace(stage, "success"))

    def _trace(
        self,
        stage: str,
        status: str,
        event: str | None = None,
        error_class: str = "FakeError",
    ) -> CallTrace:
        usage = (
            TokenUsage.missing(stage=stage, model="fake", latency_ms=1.0, attempt=1)
            if self.missing_usage
            else TokenUsage(
                stage=stage,
                model="fake",
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
                token_usage_status="reported",
                latency_ms=1.0,
                attempt=1,
            )
        )
        events = []
        if event is not None:
            events.append(ReliabilityEvent(stage=stage, event=event, attempt=1, error_class=error_class))
        return CallTrace(
            stage=stage,
            model="fake",
            status=status,
            attempts=1,
            token_usage=[usage],
            events=events,
            latency_ms=1.0,
            error=event,
        )

    @staticmethod
    def _default(stage: str, prompt: str) -> Any:
        if stage == "single_rewriter":
            return SingleRewriterOutput(
                lexical_queries=["точное название", "номер статьи"],
                semantic_queries=["смысловая потребность", "правовые последствия"],
            )
        if stage == "planner":
            return QueryPlan(
                aspects=[
                    QueryAspect(
                        aspect_id="a1",
                        legal_issue="Применимая норма",
                        bm25_instruction="Сохранить реквизиты",
                        dense_instruction="Найти смысловые синонимы",
                    )
                ],
                planner_notes="Один правовой аспект.",
            )
        if stage in {"bm25_worker", "dense_worker"}:
            channel = "bm25" if stage == "bm25_worker" else "dense"
            has_plan = "QueryPlan или null:\nnull" not in prompt
            queries = ["точная норма", "реквизиты акта"] if channel == "bm25" else ["смысл нормы", "права и обязанности"]
            return WorkerQueries(
                channel=channel,
                queries=queries,
                aspects_addressed=["a1"] if has_plan else [],
            )
        if stage == "worker_report":
            raw = prompt.split("Observations одного worker:\n", 1)[1]
            observations = json.loads(raw)
            observed = [hit["chunk_id"] for hit in observations["hits"]]
            return WorkerReport(
                channel=observations["channel"],
                aspects_addressed=observations["aspects_addressed"],
                generated_queries=observations["generated_queries"],
                observed_chunk_ids=observed,
                coverage="Покрытие по сохранённым наблюдениям.",
                gaps=[],
                supporting_chunk_ids=observed[:1],
            )
        if stage == "llm_selector":
            raw = prompt.split("Ordered candidates:\n", 1)[1].split("\n\nOptional worker reports:", 1)[0]
            candidates = json.loads(raw)
            ranking = [f"{item['pool_id']}::{item['chunk_id']}" for item in reversed(candidates)]
            return SelectorPermutation(ranking=ranking)
        raise AssertionError(f"unexpected LLM stage: {stage}")


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{index:03d}",
            doc_path=f"doc-{index % 4}.md",
            title=f"Норма {index % 4}",
            text=f"Текст нормативного фрагмента {index} " + "x" * (index % 5),
            token_count=10,
        )
        for index in range(70)
    ]


@pytest.fixture
def backend(chunks: list[Chunk]) -> FakeBackend:
    return FakeBackend(chunks)


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def jina() -> FakeJina:
    return FakeJina()


@pytest.fixture
def prompts_dir() -> Path:
    return Path("prompts/npa_mas_component_v1")
