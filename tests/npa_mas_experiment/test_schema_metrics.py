import math
from pathlib import Path

import pytest

from scripts.npa_mas_experiment.metrics import query_metrics, score_benchmark, summarize_candidate_pool
from scripts.npa_mas_experiment.schema import (
    atomic_write_json,
    audit_benchmark,
    build_benchmark,
    candidate_pool_sha256,
    canonical_json,
    project_rankings,
    read_json,
    sha256_file,
)


def fixture_rows():
    questions = [
        {"question_id": "row_0002", "source_row": 2, "query": "Кто подписывает Отчет?"},
        {"question_id": "row_0001", "source_row": 1, "query": "Первый вопрос"},
        {"question_id": "row_0003", "source_row": 3, "query": " кто   подписывает отчет?! "},
        {"question_id": "row_0004", "source_row": 4, "query": "Третий вопрос"},
    ]
    qrels = [
        {"question_id": "row_0001", "chunk_id": "c", "relevance": 2},
        {"question_id": "row_0002", "chunk_id": "a", "relevance": 1},
        {"question_id": "row_0003", "chunk_id": "a", "relevance": 2},
        {"question_id": "row_0003", "chunk_id": "b", "relevance": 2},
        {"question_id": "row_0004", "chunk_id": "ignored", "relevance": 0},
    ]
    return questions, qrels


def test_grouping_union_and_projection():
    questions, qrels = fixture_rows()
    benchmark = build_benchmark(questions, qrels)
    duplicate = next(group for group in benchmark.groups if len(group.question_ids) == 2)
    assert duplicate.representative_question_id == "row_0002"
    assert duplicate.relevant == {"a": 2, "b": 2}
    assert [group.query_group for group in benchmark.groups] == ["group_0000", "group_0001", "group_0002"]
    assert build_benchmark(list(reversed(questions)), qrels).model_dump() == benchmark.model_dump()
    audit = audit_benchmark(
        benchmark,
        expected_counts={
            "questions_total": 4,
            "groups_total": 3,
            "answerable_rows": 3,
            "answerable_groups": 2,
            "rows_without_qrels": 1,
        },
    )
    assert audit["valid"]
    rankings = {group.query_group: ["a", "b"] for group in benchmark.groups}
    projected = project_rankings(benchmark, rankings)
    assert len(projected) == 4
    duplicate_rows = [row for row in projected if row["query_group"] == duplicate.query_group]
    assert len(duplicate_rows) == 2
    assert duplicate_rows[0]["ranked_chunk_ids"] == duplicate_rows[1]["ranked_chunk_ids"] == ["a", "b"]


def test_real_frozen_benchmark_read_only_audit():
    root = Path(__file__).parents[2] / "var" / "benchmark" / "final"
    questions = root / "questions.jsonl"
    qrels = root / "qrels.jsonl"
    if not questions.is_file() or not qrels.is_file():
        pytest.skip("Frozen benchmark is unavailable")
    before = (sha256_file(questions), sha256_file(qrels))
    benchmark = build_benchmark(questions, qrels)
    report = audit_benchmark(benchmark, strict=True)
    assert report["actual"] == {
        "questions_total": 630,
        "groups_total": 556,
        "answerable_rows": 617,
        "answerable_groups": 544,
        "rows_without_qrels": 13,
    }
    assert before == (sha256_file(questions), sha256_file(qrels))


def test_hand_calculated_metrics_and_empty_ranking():
    relevant = {"a": 2, "b": 1, "c": 2}
    ranking = ["x", "a", "b", "y", "z", "c"]
    metrics = query_metrics(ranking, relevant)
    dcg5 = 3 / math.log2(3) + 1 / math.log2(4)
    idcg5 = 3 + 3 / math.log2(3) + 1 / math.log2(4)
    dcg10 = dcg5 + 3 / math.log2(7)
    assert metrics["hit_rate@5"] == 1
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["ndcg@5"] == pytest.approx(dcg5 / idcg5)
    assert metrics["mrr@5"] == 0.5
    assert metrics["recall@10"] == 1
    assert metrics["ndcg@10"] == pytest.approx(dcg10 / idcg5)
    assert metrics["mrr@10"] == 0.5
    empty = query_metrics([], relevant)
    assert set(empty.values()) == {0.0}


def test_group_row_macro_and_strict_failed_empty():
    questions, qrels = fixture_rows()
    benchmark = build_benchmark(questions, qrels)
    practical = {group.query_group: {"ranked_chunk_ids": ["a", "b", "c"], "fallback_used": True} for group in benchmark.groups}
    practical_score = score_benchmark(benchmark, practical)
    strict_score = score_benchmark(benchmark, practical, output_policy="strict_failure_as_empty")
    assert practical_score["summary"]["answerable_rows"] == 3
    assert practical_score["summary"]["answerable_unique_queries"] == 2
    assert practical_score["summary"]["groups_total"] == 3
    assert practical_score["unique_query_macro"]["recall@10"] > 0
    assert set(strict_score["unique_query_macro"].values()) == {0.0}
    assert set(strict_score["row_macro"].values()) == {0.0}
    assert len(strict_score["per_group"]) == 2


def test_canonical_atomic_json_and_pool_hash(tmp_path):
    path = tmp_path / "nested" / "artifact.json"
    atomic_write_json(path, {"я": 1, "a": [2, 3]})
    assert path.read_text(encoding="utf-8") == '{"a":[2,3],"я":1}\n'
    assert canonical_json(read_json(path)) == '{"a":[2,3],"я":1}'
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {}, overwrite=False)
    candidates = [
        {
            "pool_rank": 1,
            "chunk_id": "a",
            "selector_text": "text",
            "doc_path": "doc.md",
            "title": "title",
            "rrf_score": 10,
        }
    ]
    changed_metadata = [{**candidates[0], "rrf_score": 20}]
    assert candidate_pool_sha256(candidates) == candidate_pool_sha256(changed_metadata)
    assert candidate_pool_sha256(candidates) != candidate_pool_sha256([{**candidates[0], "title": "other"}])


def test_candidate_pool_metrics_and_provenance():
    pool = {
        "query_group": "group_0000",
        "candidates": [
            {"pool_rank": 1, "chunk_id": "a", "source_ranks": {"bm25": {"query": "q", "rank": 2}}},
            {"pool_rank": 2, "chunk_id": "b", "source_ranks": {"dense": 1}},
            {"pool_rank": 3, "chunk_id": "x", "source_ranks": {}},
        ],
        "channel_rankings": {"bm25": ["a", "x"], "dense": ["b", "c"]},
        "generated_queries": {"bm25": [" q ", "q", ""], "dense": ["semantic"]},
        "truncated_queries": {"bm25": 1},
        "budget": {
            "bm25": {
                "target_raw_hits": 11,
                "requested_raw_hits": 11,
                "actual_raw_hits": 1,
                "channel_status": "saturated",
                "queries": [
                    {"target_raw_hits": 11, "requested_raw_hits": 11, "actual_raw_hits": 1, "channel_status": "saturated"}
                ],
            }
        },
        "provenance": {
            "raw_hits": {
                "bm25": [
                    {"query_key": "original", "query": "q", "hits": [{"chunk_id": "a", "rank": 2}]}
                ]
            }
        },
        "pool_size_before_cap": 4,
    }
    result = summarize_candidate_pool(pool, {"a": 2, "c": 2}, baseline_pool={"candidates": [{"chunk_id": "a"}]})
    assert result["pool_recall@30"] == 0.5
    assert result["relevant_ids"] == 1
    assert result["channels"]["bm25"]["relevant_unique_contribution"] == ["a"]
    assert result["channels"]["dense"]["relevant_unique_contribution"] == ["c"]
    assert result["new_relevant_count_vs_baseline"] == 0
    assert result["generated_queries"]["bm25"] == {
        "generated": 2,
        "unique": 1,
        "duplicates_removed": 1,
        "normalization_removed": 0,
        "pre_normalization_recorded": False,
    }
    assert result["requested_raw_hits"] == 11
    assert result["actual_raw_hits"] == 1
    assert result["saturated_ledgers"] == 1
    assert result["pool_size_before_cap"] == 4
    assert result["pool_size_after_cap"] == 3
    assert result["provenance"][0] == {
        "chunk_id": "a",
        "channel": "bm25",
        "query": "q",
        "query_index": "original",
        "first_raw_rank": 2,
        "fusion_rank": 1,
    }
