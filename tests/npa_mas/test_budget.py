from __future__ import annotations

import pytest
from pydantic import ValidationError

from retrieval.npa_mas.budget import BudgetError, BudgetLedger
from retrieval.npa_mas.schemas import QueryBudget, TokenUsage


def test_saturation_target_11_actual_1_is_valid_and_not_refunded():
    ledger = BudgetLedger({"bm25": 11})
    item = ledger.record(
        channel="bm25",
        query="запрос",
        query_key="original",
        target=11,
        requested=11,
        actual=1,
    )
    snapshot = ledger.channel_budget("bm25")
    assert item.status == "saturated"
    assert item.unspent_raw_hits == 10
    assert snapshot.requested_raw_hits == 11
    assert snapshot.actual_raw_hits == 1
    assert snapshot.unrequested_raw_hits == 0
    assert not hasattr(ledger, "refund")


@pytest.mark.parametrize(
    ("actual", "error", "expected"),
    [(11, None, "exact_match"), (1, None, "saturated"), (0, None, "empty"), (0, "RuntimeError", "error")],
)
def test_query_budget_statuses(actual, error, expected):
    item = QueryBudget(
        channel="dense",
        query="q",
        query_key="original",
        target_raw_hits=11,
        requested_raw_hits=11,
        actual_raw_hits=actual,
        unspent_raw_hits=11 - actual,
        status=expected,
        error=error,
    )
    assert item.status == expected


def test_budget_rejects_actual_above_requested():
    ledger = BudgetLedger({"dense": 11})
    with pytest.raises(BudgetError):
        ledger.record(
            channel="dense",
            query="q",
            query_key="original",
            target=11,
            requested=10,
            actual=11,
        )


def test_channel_cannot_transfer_or_exceed_reservation_target():
    ledger = BudgetLedger({"bm25": 11, "dense": 11})
    ledger.record(
        channel="bm25",
        query="q",
        query_key="original",
        target=11,
        requested=11,
        actual=1,
    )
    with pytest.raises(BudgetError):
        ledger.record(
            channel="bm25",
            query="backfill",
            query_key="generated_001",
            target=10,
            requested=10,
            actual=0,
        )
    dense = ledger.record(
        channel="dense",
        query="q",
        query_key="original",
        target=11,
        requested=11,
        actual=11,
    )
    assert dense.requested_raw_hits == 11


def test_missing_and_non_llm_token_usage_remain_null():
    missing = TokenUsage.missing(stage="planner", model="m", latency_ms=2, attempt=1)
    retrieval = TokenUsage.not_applicable(stage="bm25_search", latency_ms=3)
    assert missing.prompt_tokens is None
    assert missing.completion_tokens is None
    assert missing.total_tokens is None
    assert missing.token_usage_status == "missing"
    assert retrieval.total_tokens is None
    assert retrieval.token_usage_status == "not_applicable"
    with pytest.raises(ValidationError):
        TokenUsage(
            stage="planner",
            model="m",
            prompt_tokens=0,
            token_usage_status="missing",
            latency_ms=0,
            attempt=1,
        )
