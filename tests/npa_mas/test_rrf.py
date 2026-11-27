from __future__ import annotations

import pytest

from retrieval.rrf import rrf_merge, weighted_rrf


def test_rrf_score_limit_and_explicit_tie_registration():
    result = rrf_merge(
        {"dense": ["a", "b"], "bm25": ["b", "a"]},
        {"dense": 0.5, "bm25": 0.5},
        60,
        1,
        ("dense", "bm25"),
    )
    assert result == [
        {
            "chunk_id": "a",
            "score": pytest.approx(0.5 / 61 + 0.5 / 62),
            "rank": 1,
            "source_ranks": {"dense": 1, "bm25": 2},
        }
    ]


def test_source_order_invariance_for_dictionary_iteration():
    left = {"dense": ["a", "b"], "bm25": ["b", "a"]}
    right = {"bm25": ["b", "a"], "dense": ["a", "b"]}
    first = rrf_merge(left, k=60, source_order=("dense", "bm25"))
    second = rrf_merge(right, k=60, source_order=("dense", "bm25"))
    assert first == second
    assert [item["chunk_id"] for item in first] == ["a", "b"]


def test_duplicate_ids_do_not_add_votes_or_change_ranks():
    with_duplicates = rrf_merge({"q": ["a", "a", "b", "b"]}, k=60, source_order=("q",))
    without_duplicates = rrf_merge({"q": ["a", "b"]}, k=60, source_order=("q",))
    assert with_duplicates == without_duplicates
    assert with_duplicates[1]["source_ranks"] == {"q": 2}


def test_source_order_must_be_complete_and_unique():
    with pytest.raises(ValueError):
        rrf_merge({"a": ["1"], "b": ["2"]}, source_order=("a",))
    with pytest.raises(ValueError):
        rrf_merge({"a": ["1"]}, source_order=("a", "a"))


def test_weighted_rrf_alpha_and_legacy_positional_api():
    result = weighted_rrf(["d"], ["b"], 0.5, 60, 30)
    assert len(result) == 2
    with pytest.raises(ValueError):
        weighted_rrf([], [], alpha=1.1)
