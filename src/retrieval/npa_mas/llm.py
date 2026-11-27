from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from retrieval.npa_mas.schemas import CallTrace, LLMResult, ReliabilityEvent, TokenUsage

T = TypeVar("T", bound=BaseModel)
ResponseValidator = Callable[[T], T]


class OutputValidationError(ValueError):
    def __init__(self, event: str, message: str):
        super().__init__(message)
        self.event = event


class JsonLLMGateway(Protocol):
    async def generate_json(
        self,
        *,
        stage: str,
        prompt: str,
        response_model: type[T],
        max_output_tokens: int,
        validator: ResponseValidator[T] | None = None,
    ) -> LLMResult[T]: ...


def mark_fallback(
    trace: CallTrace,
    reason: str,
    *,
    event: str | None = None,
    error_class: str = "DeclaredFallback",
) -> CallTrace:
    updated = trace.model_copy(deep=True)
    updated.status = "fallback"
    updated.fallback_used = True
    updated.error = reason
    if event is not None:
        updated.events.append(
            ReliabilityEvent(
                stage=updated.stage,
                event=event,
                attempt=max(1, updated.attempts),
                error_class=error_class,
                message=reason,
            )
        )
    return updated


class PromptLoader:
    def __init__(self, root: str | Path = Path("prompts/npa_mas_component_v1")):
        self.root = Path(root)

    def load(self, filename: str) -> str:
        if filename not in {
            "single_rewriter.txt",
            "planner.txt",
            "bm25_worker.txt",
            "dense_worker.txt",
            "worker_report.txt",
            "llm_selector.txt",
        }:
            raise ValueError(f"unknown prompt file: {filename}")
        return (self.root / filename).read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    rendered_values = {
        key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        for key, value in values.items()
    }
    return template.format(**rendered_values)


class AsyncJsonLLMGateway:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | Any | None = None,
        timeout_s: float = 180.0,
        max_transient_retries: int = 2,
        json_repair_attempts: int = 1,
        temperature: float = 0.0,
        seed: int = 20261127,
        backoff_s: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if not 0 <= max_transient_retries <= 2:
            raise ValueError("at most two transient retries are allowed")
        if not 0 <= json_repair_attempts <= 1:
            raise ValueError("at most one JSON/schema repair is allowed")
        if temperature != 0.0:
            raise ValueError("npa_mas_component_v1 requires temperature=0")
        self.model = model
        self.timeout_s = timeout_s
        self.max_transient_retries = max_transient_retries
        self.json_repair_attempts = json_repair_attempts
        self.temperature = temperature
        self.seed = seed
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.client = client or AsyncOpenAI(api_key=api_key or "unused", base_url=base_url, timeout=timeout_s)

    @classmethod
    def from_env(cls, **overrides: Any) -> AsyncJsonLLMGateway:
        return cls(
            model=os.environ.get("NPA_LLM_MODEL", ""),
            api_key=os.environ.get("NPA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("NPA_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
            **overrides,
        )

    async def generate_json(
        self,
        *,
        stage: str,
        prompt: str,
        response_model: type[T],
        max_output_tokens: int,
        validator: ResponseValidator[T] | None = None,
    ) -> LLMResult[T]:
        started = time.perf_counter()
        usages: list[TokenUsage] = []
        events: list[ReliabilityEvent] = []
        attempt = 0
        transient_retries = 0
        repairs = 0
        current_prompt = prompt
        retried = False
        repaired = False
        last_error = "empty_output"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        rng = random.Random(f"{self.seed}:{stage}:{prompt_hash}")

        while True:
            attempt += 1
            call_started = time.perf_counter()
            content = ""
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=0.0,
                        max_tokens=max_output_tokens,
                        response_format={"type": "json_object"},
                    ),
                    timeout=self.timeout_s,
                )
                latency_ms = (time.perf_counter() - call_started) * 1000
                usages.append(self._usage(stage, response, latency_ms, attempt))
                content = self._content(response)
                if not content.strip():
                    raise OutputValidationError("empty_output", "LLM returned empty content")
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise OutputValidationError("malformed_json", str(exc)) from exc
                try:
                    value = response_model.model_validate(payload)
                    if validator is not None:
                        value = validator(value)
                except OutputValidationError:
                    raise
                except (ValidationError, ValueError) as exc:
                    raise OutputValidationError("schema_violation", str(exc)) from exc
                if repaired:
                    status = "success_after_json_repair"
                elif retried:
                    status = "success_after_transient_retry"
                else:
                    status = "success"
                trace = CallTrace(
                    stage=stage,
                    model=self.model,
                    status=status,
                    attempts=attempt,
                    token_usage=usages,
                    events=events,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                return LLMResult[T](value=value, trace=trace)
            except OutputValidationError as exc:
                last_error = exc.event
                events.append(
                    ReliabilityEvent(
                        stage=stage,
                        event=exc.event,
                        attempt=attempt,
                        error_class=type(exc).__name__,
                        message=str(exc)[:1000],
                    )
                )
                if repairs < self.json_repair_attempts:
                    repairs += 1
                    repaired = True
                    current_prompt = self._repair_prompt(prompt, response_model, content, str(exc))
                    continue
                break
            except Exception as exc:
                latency_ms = (time.perf_counter() - call_started) * 1000
                if not usages or usages[-1].attempt != attempt:
                    usages.append(TokenUsage.missing(stage=stage, model=self.model, latency_ms=latency_ms, attempt=attempt))
                event, transient = self._classify_exception(exc)
                last_error = event
                events.append(
                    ReliabilityEvent(
                        stage=stage,
                        event=event,
                        attempt=attempt,
                        error_class=type(exc).__name__,
                        message=str(exc)[:1000],
                    )
                )
                if transient and transient_retries < self.max_transient_retries:
                    transient_retries += 1
                    retried = True
                    delay = self.backoff_s * (2 ** (transient_retries - 1)) * (0.5 + rng.random())
                    if delay > 0:
                        await self.sleep(delay)
                    continue
                break

        trace = CallTrace(
            stage=stage,
            model=self.model,
            status="failed",
            attempts=attempt,
            token_usage=usages,
            events=events,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=last_error,
        )
        return LLMResult[T](value=None, trace=trace, error=last_error)

    def _usage(self, stage: str, response: Any, latency_ms: float, attempt: int) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage.missing(stage=stage, model=self.model, latency_ms=latency_ms, attempt=attempt)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return TokenUsage(
            stage=stage,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_usage_status="reported",
            cached=False,
            latency_ms=latency_ms,
            attempt=attempt,
        )

    @staticmethod
    def _content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise OutputValidationError("empty_output", "LLM response has no message content") from exc
        if not isinstance(content, str):
            raise OutputValidationError("empty_output", "LLM message content is not text")
        return content

    @staticmethod
    def _classify_exception(exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError)):
            return "timeout", True
        if isinstance(exc, openai.RateLimitError):
            return "rate_limit", True
        if isinstance(exc, openai.APIConnectionError):
            return "provider_error", True
        message = str(exc).lower()
        if "context length" in message or "context_length" in message:
            return "context_length_exceeded", False
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code == 429:
                return "rate_limit", True
            return "provider_error", exc.status_code >= 500
        return "provider_error", False

    @staticmethod
    def _repair_prompt(original_prompt: str, response_model: type[BaseModel], content: str, error: str) -> str:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, sort_keys=True)
        return (
            "Исправь ответ строго по JSON Schema. Верни только JSON без пояснений.\n"
            f"Ошибка: {error}\nJSON Schema: {schema}\n"
            f"Исходная задача:\n{original_prompt}\nНекорректный ответ:\n{content}"
        )
