import pytest

from scripts.npa_mas_experiment.analysis import (
    aggregate_cost_reliability,
    compare_stability_artifacts,
    kendall_tau_common,
    rank_movement,
    render_report,
    same_pool_group_analysis,
    summarize_stability,
    top10_overlap,
)
from scripts.npa_mas_experiment.schema import QUERY_LABELS, candidate_pool_sha256
from scripts.npa_mas_experiment.statistics import (
    confirmatory_contrasts,
    frozen_label_summary,
    holm_adjust,
    paired_bootstrap,
    paired_test,
    practical_comparisons,
    sign_flip_randomization,
    subgroup_effects,
    subgroup_metric_summary,
)


def score(value):
    return {"recall@10": value}


def test_statistics_reproducibility_and_counts():
    left = {"g1": 1.0, "g2": 0.0, "g3": 1.0, "g4": 0.0}
    right = {"g1": 0.0, "g2": 0.0, "g3": 1.0, "g4": 1.0}
    first = paired_test(
        left,
        right,
        contrast="C1",
        bootstrap_iterations=300,
        randomization_iterations=300,
        seed=20261127,
        fallback_left={key: key == "g1" for key in left},
        fallback_right={key: False for key in right},
    )
    second = paired_test(
        left,
        right,
        contrast="C1",
        bootstrap_iterations=300,
        randomization_iterations=300,
        seed=20261127,
        fallback_left={key: key == "g1" for key in left},
        fallback_right={key: False for key in right},
    )
    assert first == second
    assert first["wins"] == 1
    assert first["losses"] == 1
    assert first["ties"] == 2
    assert first["mean_delta"] == 0
    assert first["fallback_groups_left"] == 1
    assert first["fallback_groups_right"] == 0
    bootstrap_one = paired_bootstrap([0, 1, 0.5, -0.5], iterations=300, stream="one")
    bootstrap_two = paired_bootstrap([0, 1, 0.5, -0.5], iterations=300, stream="two")
    assert bootstrap_one != bootstrap_two
    randomization_one = sign_flip_randomization([1, 0.5, -0.25], iterations=300, stream="one")
    randomization_two = sign_flip_randomization([1, 0.5, -0.25], iterations=300, stream="one")
    assert randomization_one == randomization_two


def test_holm_and_contrast_families():
    assert holm_adjust({"a": 0.01, "b": 0.02, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})
    scores = {
        variant: {f"g{index}": score((index + variant_index) % 2) for index in range(4)}
        for variant_index, variant in enumerate(("B1", "B1W", "S", "M1", "M2", "M3", "M4"))
    }
    config = {
        "seed": 20261127,
        "statistics": {"bootstrap_iterations": 40, "randomization_iterations": 40, "primary_metric": "recall@10"},
    }
    confirmatory = confirmatory_contrasts(scores, config=config)
    practical = practical_comparisons(scores, config=config)
    assert [row["contrast"] for row in confirmatory] == ["C1", "C2", "C3", "C4"]
    assert all(row["family"] == "confirmatory" for row in confirmatory)
    assert [row["contrast"] for row in practical] == ["M4-B1", "M4-B1W", "M4-S"]
    assert all(row["family"] == "secondary_practical" for row in practical)


def annotations():
    def labels(**values):
        return {label: values.get(label, False) for label in QUERY_LABELS}

    return [
        {"query_group": "g0", "labels": labels(exact_reference=True)},
        {"query_group": "g1", "labels": labels(exact_reference=True, multi_issue=True)},
        {"query_group": "g2", "labels": labels()},
        {"query_group": "g3", "labels": labels()},
    ]


def test_frozen_subgroups_and_interactions():
    scores = {
        "M1": {f"g{index}": score(value) for index, value in enumerate((1, 1, 0, 0))},
        "S": {f"g{index}": score(value) for index, value in enumerate((0, 1, 1, 0))},
    }
    metric_rows = subgroup_metric_summary({"M1": scores["M1"], "S": scores["S"]}, annotations())
    exact_true = next(row for row in metric_rows if row["variant"] == "M1" and row["label"] == "exact_reference" and row["label_value"])
    assert exact_true["n_groups"] == 2
    assert exact_true["recall@10"] == 1
    effects = subgroup_effects(
        scores,
        annotations(),
        contrasts={"C1": ("M1", "S")},
        bootstrap_iterations=30,
        randomization_iterations=30,
    )
    exact = next(row for row in effects if row["label"] == "exact_reference")
    assert exact["n_label_true"] == 2
    assert exact["n_label_false"] == 2
    assert exact["mean_delta_label_true"] == 0.5
    assert exact["mean_delta_label_false"] == -0.5
    assert exact["interaction_delta"] == 1
    assert exact["descriptive_only"]
    summary = frozen_label_summary(annotations())
    assert summary["label_counts"]["exact_reference"] == 2
    assert summary["multi_label_groups"] == 1


def usage_call(prompt, completion, total, status):
    return {
        "stage": "planner",
        "model": "test-model",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "token_usage_status": status,
        "latency_ms": 100,
        "attempt": 1,
    }


def test_missing_usage_stays_null_and_jina_is_separate():
    artifacts = {
        "g1": {
            "ranked_chunk_ids": ["a"],
            "calls": [usage_call(10, 5, 15, "reported")],
            "latency_ms": 150,
        },
        "g2": {
            "ranked_chunk_ids": [],
            "calls": [usage_call(None, None, None, "missing")],
            "fallback_used": True,
            "latency_ms": 200,
        },
        "g3": {
            "ranked_chunk_ids": ["a"],
            "calls": [
                {
                    "stage": "jina_selector",
                    "model": "jina-test",
                    "device": "cpu",
                    "token_usage_status": "not_applicable",
                    "latency_ms": 30,
                }
            ],
        },
    }
    result = aggregate_cost_reliability(artifacts, variant="M2")
    cost = result["cost"]
    assert cost["llm_calls_total"] == 2
    assert cost["total_tokens_sum"] is None
    assert cost["total_tokens_observed_sum"] == 15
    assert cost["tokens"]["prompt_tokens"]["mean"] == 10
    assert cost["tokens"]["prompt_tokens"]["coverage"] == 0.5
    assert cost["token_usage_group_counts"] == {"missing": 1, "not_applicable": 1, "reported": 1}
    assert cost["token_usage_group_coverage"]["reported"] == 0.5
    assert cost["monetary_cost"] is None
    assert cost["jina_local_compute"]["calls"] == 1
    assert cost["jina_local_compute"]["devices"] == ["cpu"]
    assert cost["jina_local_compute"]["llm_api_tokens"] is None
    assert result["reliability"]["fallback_groups"] == 1
    assert result["reliability"]["empty_output_groups"] == 1


def test_no_llm_usage_is_not_zero():
    result = aggregate_cost_reliability({"g1": {"ranked_chunk_ids": ["a"], "calls": []}})
    assert result["cost"]["total_tokens_sum"] is None
    assert result["cost"]["total_tokens_observed_sum"] is None
    assert result["cost"]["token_usage_group_coverage"]["reported"] is None


def test_pipeline_nested_usage_and_channel_budget_shape():
    first = usage_call(10, 5, 15, "reported")
    first["attempt"] = 1
    second = usage_call(20, 10, 30, "reported")
    second["attempt"] = 2
    call = {
        "stage": "planner",
        "model": "test-model",
        "status": "success_after_transient_retry",
        "attempts": 2,
        "token_usage": [first, second],
        "events": [{"stage": "planner", "event": "timeout", "error_class": "Timeout"}],
        "latency_ms": 250,
        "fallback_used": False,
    }
    artifact = {
        "ranked_chunk_ids": ["a"],
        "status": "success",
        "calls": [call],
        "token_usage": [
            {"stage": "jina_selector", "token_usage_status": "not_applicable", "latency_ms": 20, "attempt": 1},
            first,
            second,
        ],
        "pool": {
            "budget": {
                "bm25": {
                    "target_raw_hits": 11,
                    "requested_raw_hits": 11,
                    "actual_raw_hits": 1,
                    "unspent_raw_hits": 10,
                    "status": "saturated",
                    "queries": [
                        {"target_raw_hits": 11, "requested_raw_hits": 11, "actual_raw_hits": 1, "unspent_raw_hits": 10, "status": "saturated"}
                    ],
                }
            }
        },
    }
    result = aggregate_cost_reliability({"g1": artifact})
    assert result["cost"]["llm_calls_total"] == 2
    assert result["cost"]["total_tokens_sum"] == 45
    assert result["cost"]["jina_local_compute"]["calls"] == 1
    assert result["cost"]["requested_raw_hits"] == 11
    assert result["cost"]["actual_raw_hits"] == 1
    assert result["reliability"]["retry_calls"] == 1
    assert result["reliability"]["timeout_groups"] == 1
    assert result["reliability"]["saturated_groups"] == 1


def pool_artifact(ranking, candidates, reports=None):
    return {
        "pool_sha256": candidate_pool_sha256(candidates),
        "candidate_pool": {"candidates": candidates},
        "ranked_chunk_ids": ranking,
        "reports": reports or {},
    }


def test_same_pool_movement_overlap_and_kendall():
    candidates = [
        {"pool_rank": index + 1, "chunk_id": chunk_id, "selector_text": chunk_id, "doc_path": "d", "title": "t"}
        for index, chunk_id in enumerate(("a", "b", "c"))
    ]
    left = pool_artifact(["a", "b", "c"], candidates)
    right = pool_artifact(["b", "a", "c"], candidates, {"dense": {"supporting_chunk_ids": ["b"]}})
    result = same_pool_group_analysis(left, right, {"a": 2, "b": 1}, contrast="M3-M2")
    assert result["jaccard"] == 1
    assert result["ordered_overlap"] == pytest.approx(1 / 3)
    assert result["kendall_tau_common"] == pytest.approx(1 / 3)
    assert result["pool_recall_constant"]
    assert result["report_supported_moved_up"] == ["b"]
    assert kendall_tau_common(["a", "b", "c"], ["c", "b", "a"]) == -1
    assert top10_overlap(["a"], [])["jaccard"] == 0
    movement = rank_movement(["a"], ["b", "a"], {"a": 1})
    assert movement[0]["rank_improvement"] == -1


def test_stability_summary_and_report():
    base = {
        "query_group": "g1",
        "generated_queries": {"bm25": {"queries": ["Query"]}},
        "ranked_chunk_ids": ["a", "b"],
        "recall@10": 0.5,
        "calls": [usage_call(1, 1, 2, "reported")],
    }
    repeat = {
        "query_group": "g1",
        "generated_queries": {"bm25": {"queries": ["query", "new"]}},
        "ranked_chunk_ids": ["b", "a"],
        "recall@10": 1.0,
        "calls": [usage_call(2, 1, 3, "reported")],
    }
    comparison = compare_stability_artifacts(base, repeat)
    assert not comparison["exact_query_set_match"]
    assert comparison["top10_jaccard"] == 1
    summary = summarize_stability({"S": [{"g1": base}, {"g1": repeat}]})
    assert summary["variants"][0]["metric_range_max"] == 0.5
    assert summary["variants"][0]["total_tokens_observed_range"] == 1
    assert summary["variants"][0]["independent_primary_n_added"] == 0
    report = render_report({f"T{index}": [] for index in range(1, 9)})
    assert "T1" in report
    assert "T8" in report
    assert "Limitations" in report
