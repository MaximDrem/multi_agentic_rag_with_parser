from __future__ import annotations

import hashlib
import json
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

Channel = Literal["bm25", "dense"]
BudgetStatus = Literal["exact_match", "saturated", "empty", "error"]
CallStatus = Literal[
    "success",
    "success_after_transient_retry",
    "success_after_json_repair",
    "fallback",
    "failed",
]
TokenUsageStatus = Literal["reported", "missing", "not_applicable"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class QueryAspect(StrictModel):
    aspect_id: str
    legal_issue: str
    bm25_instruction: str
    dense_instruction: str


class QueryPlan(StrictModel):
    aspects: list[QueryAspect]
    planner_notes: str
    fallback_used: bool = False


class SingleRewriterOutput(StrictModel):
    lexical_queries: list[str]
    semantic_queries: list[str]


class WorkerQueries(StrictModel):
    channel: Channel
    queries: list[str]
    aspects_addressed: list[str]
    fallback_used: bool = False


class WorkerReport(StrictModel):
    channel: Channel
    aspects_addressed: list[str]
    generated_queries: list[str]
    observed_chunk_ids: list[str]
    coverage: str
    gaps: list[str]
    supporting_chunk_ids: list[str]
    fallback_used: bool = False
    status: Literal["success", "empty", "failed"] = "success"
    error: str | None = None


class Candidate(StrictModel):
    chunk_id: str
    pool_rank: int = Field(ge=1)
    selector_text: str
    text_sha256: str
    doc_path: str
    title: str
    source_ranks: dict[str, int]
    rrf_score: float
    text: str = ""


class QueryBudget(StrictModel):
    channel: Channel
    query: str
    query_key: str
    target_raw_hits: int = Field(ge=0)
    requested_raw_hits: int = Field(ge=0)
    actual_raw_hits: int = Field(ge=0)
    unspent_raw_hits: int = Field(ge=0)
    status: BudgetStatus
    error: str | None = None

    @model_validator(mode="after")
    def validate_invariants(self) -> QueryBudget:
        if not self.actual_raw_hits <= self.requested_raw_hits <= self.target_raw_hits:
            raise ValueError("actual_raw_hits <= requested_raw_hits <= target_raw_hits is required")
        if self.unspent_raw_hits != self.requested_raw_hits - self.actual_raw_hits:
            raise ValueError("unspent_raw_hits must equal requested_raw_hits - actual_raw_hits")
        expected = budget_status(self.requested_raw_hits, self.actual_raw_hits, self.error)
        if self.status != expected:
            raise ValueError(f"status must be {expected}")
        return self


class ChannelBudget(StrictModel):
    channel: Channel
    target_raw_hits: int = Field(ge=0)
    requested_raw_hits: int = Field(ge=0)
    actual_raw_hits: int = Field(ge=0)
    unspent_raw_hits: int = Field(ge=0)
    unrequested_raw_hits: int = Field(ge=0)
    status: BudgetStatus
    queries: list[QueryBudget]

    @model_validator(mode="after")
    def validate_invariants(self) -> ChannelBudget:
        if not self.actual_raw_hits <= self.requested_raw_hits <= self.target_raw_hits:
            raise ValueError("actual_raw_hits <= requested_raw_hits <= target_raw_hits is required")
        if self.requested_raw_hits != sum(item.requested_raw_hits for item in self.queries):
            raise ValueError("channel requested hits must equal the sum of query reservations")
        if self.actual_raw_hits != sum(item.actual_raw_hits for item in self.queries):
            raise ValueError("channel actual hits must equal the sum of query hits")
        if self.unspent_raw_hits != self.requested_raw_hits - self.actual_raw_hits:
            raise ValueError("invalid channel unspent hits")
        if self.unrequested_raw_hits != self.target_raw_hits - self.requested_raw_hits:
            raise ValueError("invalid channel unrequested hits")
        expected = aggregate_budget_status(self.queries)
        if self.status != expected:
            raise ValueError(f"status must be {expected}")
        return self


def budget_status(requested: int, actual: int, error: str | None = None) -> BudgetStatus:
    if error is not None:
        return "error"
    if actual == 0:
        return "empty"
    if actual == requested:
        return "exact_match"
    return "saturated"


def aggregate_budget_status(queries: list[QueryBudget]) -> BudgetStatus:
    if any(item.status == "error" for item in queries):
        return "error"
    requested = sum(item.requested_raw_hits for item in queries)
    actual = sum(item.actual_raw_hits for item in queries)
    return budget_status(requested, actual)


def selector_visible_record(candidate: Candidate) -> dict[str, Any]:
    return {
        "pool_rank": candidate.pool_rank,
        "chunk_id": candidate.chunk_id,
        "selector_text": candidate.selector_text,
        "doc_path": candidate.doc_path,
        "title": candidate.title,
    }


def canonical_pool_json(candidates: list[Candidate]) -> str:
    return json.dumps(
        [selector_visible_record(candidate) for candidate in candidates],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_pool_sha256(candidates: list[Candidate]) -> str:
    return hashlib.sha256(canonical_pool_json(candidates).encode("utf-8")).hexdigest()


class CandidatePool(StrictModel):
    query_group: str
    pool_id: str
    pool_sha256: str
    candidates: list[Candidate]
    budget: dict[str, ChannelBudget]
    generated_queries: dict[str, list[str]]
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def validate_pool(self) -> CandidatePool:
        expected_hash = compute_pool_sha256(self.candidates)
        if self.pool_sha256 != expected_hash:
            raise ValueError("pool_sha256 does not match selector-visible candidate fields")
        if self.pool_id != self.pool_sha256:
            raise ValueError("pool_id must equal pool_sha256")
        if [candidate.pool_rank for candidate in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise ValueError("pool_rank must be consecutive and ordered")
        ids = [candidate.chunk_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate pool IDs must be unique")
        return self

    @classmethod
    def create(
        cls,
        *,
        query_group: str,
        candidates: list[Candidate],
        budget: dict[str, ChannelBudget],
        generated_queries: dict[str, list[str]],
        provenance: dict[str, Any],
    ) -> CandidatePool:
        pool_hash = compute_pool_sha256(candidates)
        return cls(
            query_group=query_group,
            pool_id=pool_hash,
            pool_sha256=pool_hash,
            candidates=candidates,
            budget=budget,
            generated_queries=generated_queries,
            provenance=provenance,
        )

    @property
    def candidate_ids(self) -> list[str]:
        return [candidate.chunk_id for candidate in self.candidates]

    @property
    def selector_texts(self) -> list[str]:
        return [candidate.selector_text for candidate in self.candidates]

    @property
    def has_search_errors(self) -> bool:
        return any(budget.status == "error" for budget in self.budget.values())

    def assert_same_pool(self, other: CandidatePool) -> None:
        mismatches = []
        if self.pool_sha256 != other.pool_sha256:
            mismatches.append("pool_sha256")
        if self.candidate_ids != other.candidate_ids:
            mismatches.append("candidate_ids")
        if self.selector_texts != other.selector_texts:
            mismatches.append("selector_texts")
        if mismatches:
            raise ValueError("same-pool mismatch: " + ", ".join(mismatches))


class SelectorPermutation(StrictModel):
    ranking: list[str]

    @property
    def ranked_ids(self) -> list[str]:
        return self.ranking


class TokenUsage(StrictModel):
    stage: str
    model: str | None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_usage_status: TokenUsageStatus
    cached: bool = False
    latency_ms: float = Field(ge=0)
    attempt: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_usage_status(self) -> TokenUsage:
        if self.token_usage_status in {"missing", "not_applicable"}:
            if any(value is not None for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens)):
                raise ValueError("missing/not_applicable token usage must retain null values")
        return self

    @classmethod
    def missing(cls, *, stage: str, model: str | None, latency_ms: float, attempt: int) -> TokenUsage:
        return cls(
            stage=stage,
            model=model,
            token_usage_status="missing",
            latency_ms=latency_ms,
            attempt=attempt,
        )

    @classmethod
    def not_applicable(cls, *, stage: str, latency_ms: float = 0.0) -> TokenUsage:
        return cls(
            stage=stage,
            model=None,
            token_usage_status="not_applicable",
            latency_ms=latency_ms,
            attempt=1,
        )


class ReliabilityEvent(StrictModel):
    stage: str
    event: str
    attempt: int = Field(ge=1)
    error_class: str
    message: str | None = None


class CallTrace(StrictModel):
    stage: str
    model: str | None
    status: CallStatus
    attempts: int = Field(ge=0)
    token_usage: list[TokenUsage]
    events: list[ReliabilityEvent]
    latency_ms: float = Field(ge=0)
    fallback_used: bool = False
    error: str | None = None


class FallbackTrace(StrictModel):
    stage: str
    reason: str


class RankedChunk(StrictModel):
    chunk_id: str
    rank: int = Field(ge=1)
    score: float | None
    text: str
    selector_text: str
    doc_path: str
    title: str


class PipelineResult(StrictModel):
    protocol_version: Literal["npa_mas_component_v1"] = "npa_mas_component_v1"
    variant: str
    query: str
    query_group: str
    status: Literal["success", "complete_with_fallback", "failed"]
    strict_failed: bool
    fallback_used: bool
    fallback_stage: str | None
    fallback_reason: str | None
    fallbacks: list[FallbackTrace]
    ranked_chunk_ids: list[str]
    full_ranked_chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[RankedChunk]
    pool: CandidatePool | None
    plan: QueryPlan | None
    worker_queries: dict[str, WorkerQueries]
    reports: dict[str, WorkerReport]
    calls: list[CallTrace]
    token_usage: list[TokenUsage]
    stage_timings_ms: dict[str, float]
    latency_ms: float = Field(ge=0)
    error: str | None


T = TypeVar("T")


class LLMResult(StrictModel, Generic[T]):
    value: T | None
    trace: CallTrace
    error: str | None = None


class RetrievalSettings(StrictModel):
    baseline_target_raw_hits_per_channel: int = Field(default=30, gt=0)
    expanded_target_raw_hits_per_channel: int = Field(default=60, gt=0)
    expanded_original_query_target_raw_hits: int = Field(default=30, gt=0)
    expanded_each_generated_query_target_raw_hits: int = Field(default=10, gt=0)
    wide_target_raw_hits_per_channel: int = Field(default=60, gt=0)
    max_generated_queries_per_channel: int = Field(default=3, ge=0, le=3)
    max_generated_query_chars: int = Field(default=300, gt=0)
    pool_size: int = Field(default=30, gt=0, le=30)
    output_chunks: int = Field(default=10, gt=0, le=10)
    rrf_k: int = Field(default=60, gt=0)
    final_alpha_dense: float = Field(default=0.5, ge=0.0, le=1.0)
    selector_text_max_chars: int = Field(default=2000, gt=0)
    redistribute_unused_query_budget: Literal[False] = False
    allow_cross_channel_compensation: Literal[False] = False
    allow_saturation_backfill: Literal[False] = False


class PlannerSettings(StrictModel):
    max_aspects: int = Field(default=4, gt=0)


class ReportSettings(StrictModel):
    max_observed_chunks_per_worker: int = Field(default=10, gt=0)
    text_max_chars: int = Field(default=2000, gt=0)
    max_gap_items: int = Field(default=4, gt=0)


class LLMSettings(StrictModel):
    model_env: str = "NPA_LLM_MODEL"
    temperature: float = Field(default=0.0, strict=True)
    timeout_s: float = Field(default=180.0, gt=0)
    max_transient_retries: int = Field(default=2, ge=0, le=2)
    json_repair_attempts: int = Field(default=1, ge=0, le=1)
    max_output_tokens: dict[str, int] = {
        "single_rewriter": 1200,
        "planner": 800,
        "bm25_worker": 800,
        "dense_worker": 800,
        "worker_report": 1200,
        "llm_selector": 2000,
    }


class NpaMasConfig(StrictModel):
    protocol_version: Literal["npa_mas_component_v1"] = "npa_mas_component_v1"
    retrieval: RetrievalSettings = RetrievalSettings()
    planner: PlannerSettings = PlannerSettings()
    reports: ReportSettings = ReportSettings()
    llm: LLMSettings = LLMSettings()

    @model_validator(mode="after")
    def validate_fixed_protocol(self) -> NpaMasConfig:
        if self.llm.temperature != 0.0:
            raise ValueError("npa_mas_component_v1 requires temperature=0")
        settings = self.retrieval
        maximum_reservation = (
            settings.expanded_original_query_target_raw_hits
            + settings.max_generated_queries_per_channel * settings.expanded_each_generated_query_target_raw_hits
        )
        if maximum_reservation > settings.expanded_target_raw_hits_per_channel:
            raise ValueError("expanded query reservations exceed the channel target")
        if settings.final_alpha_dense != 0.5:
            raise ValueError("npa_mas_component_v1 requires final_alpha_dense=0.5")
        return self
