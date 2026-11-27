import asyncio
import json
from pathlib import Path

import pytest

from retrieval.npa_mas.schemas import Candidate, CandidatePool
from scripts.npa_mas_experiment.runner import ExperimentRunner, ProtocolError
from scripts.npa_mas_experiment.schema import build_benchmark

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "npa_mas_component_v1.yaml"


def small_inputs():
    questions = [
        {"question_id": "row_0001", "source_row": 1, "query": "Что такое ОКС?"},
        {"question_id": "row_0002", "source_row": 2, "query": "Что такое дашборд?"},
    ]
    qrels = [{"question_id": "row_0001", "chunk_id": "chunk-1", "relevance": 2}]
    benchmark = build_benchmark(questions, qrels)
    groups = [
        {
            "query_group": group.query_group,
            "normalized_query": group.normalized_query,
            "query": group.query,
            "representative_question_id": group.representative_question_id,
            "question_ids": list(group.question_ids),
            "answerable_question_ids": [qid for qid in group.question_ids if benchmark.qrels.get(qid)],
            "questions": [question for question in questions if question["question_id"] in group.question_ids],
            "relevant": dict(group.relevant),
            "answerable": group.answerable,
        }
        for group in benchmark.groups
    ]
    return benchmark, groups


def make_pool(group_id):
    candidates = [
        Candidate(
            chunk_id=chunk_id,
            pool_rank=index,
            selector_text=f"Текст {chunk_id}",
            text_sha256=f"{index:064x}",
            doc_path="doc.md",
            title="Документ",
            source_ranks={"dense": index, "bm25": index},
            rrf_score=1.0 / (60 + index),
            text=f"Текст {chunk_id}",
        )
        for index, chunk_id in enumerate(("chunk-1", "chunk-2"), 1)
    ]
    return CandidatePool.create(
        query_group=group_id,
        candidates=candidates,
        budget={},
        generated_queries={"dense": [], "bm25": []},
        provenance={
            "channel_rankings": {"dense": ["chunk-1", "chunk-2"], "bm25": ["chunk-1", "chunk-2"]},
            "observations": {"dense": [], "bm25": []},
            "raw_hits": {"dense": [], "bm25": []},
        },
    )


class FakePipeline:
    async def run(self, query, variant, *, query_group, m2_artifact=None):
        if variant in {"M3", "M4"}:
            assert m2_artifact["query_group"] == query_group
        pool = CandidatePool.model_validate(m2_artifact["pool"]) if m2_artifact else make_pool(query_group)
        return {
            "variant": variant,
            "query": query,
            "query_group": query_group,
            "status": "success",
            "strict_failed": False,
            "fallback_used": False,
            "ranked_chunk_ids": ["chunk-1"],
            "full_ranked_chunk_ids": pool.candidate_ids,
            "chunks": [],
            "pool": pool.model_dump(mode="json"),
            "plan": None,
            "worker_queries": {},
            "reports": {},
            "calls": [],
            "token_usage": [],
            "stage_timings_ms": {"query_generation": 1.0, "candidate_generation": 1.0},
            "latency_ms": 2.0,
            "error": None,
        }


def frozen_manifest():
    return {
        "protocol_version": "npa_mas_component_v1",
        "run_id": "core_001",
        "config_sha256": "a" * 64,
        "prompt_bundle_sha256": "b" * 64,
        "annotations_sha256": "c" * 64,
        "annotations_bundle_sha256": "e" * 64,
        "source_code_sha256": "d" * 64,
        "benchmark_hashes": {},
        "chunk_index_manifest": {},
        "model_manifest": {},
        "retrieval_manifest": {},
        "variants": ["B0_BM25", "B0_DENSE", "B0_RRF", "B1", "B1W", "S", "M1", "M2", "M3", "M4"],
    }


def test_score_analysis_and_stability_integration(tmp_path, monkeypatch):
    benchmark, groups = small_inputs()
    instance = ExperimentRunner(CONFIG, pipeline=FakePipeline())
    instance.output_root = tmp_path
    instance._groups = groups
    monkeypatch.setattr(instance, "_benchmark", lambda: benchmark)
    monkeypatch.setattr(instance, "_valid_chunk_ids", lambda: {"chunk-1", "chunk-2"})
    monkeypatch.setattr(instance, "validate_run", lambda *args, **kwargs: {"valid": True, "error_count": 0})
    instance.config["benchmark"].update({
        "expected_rows": 2,
        "expected_groups": 2,
        "expected_answerable_rows": 1,
        "expected_answerable_groups": 1,
        "expected_rows_without_qrels": 1,
    })
    instance.config["statistics"]["bootstrap_iterations"] = 20
    instance.config["statistics"]["randomization_iterations"] = 20
    instance.config["stability"]["groups"] = 1
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text("\n".join(
        json.dumps({
            "query_group": group["query_group"],
            "labels": {label: False for label in (
                "exact_reference", "lay_paraphrase", "multi_issue", "applicability_conditions",
                "cross_reference_multi_document", "other_unclear",
            )},
        })
        for group in groups
    ) + "\n")
    instance.config["annotations"]["labels"] = str(annotation_path)
    manifest = frozen_manifest()
    monkeypatch.setattr(instance, "_assert_frozen_postprocessing_inputs", lambda run_id: manifest)
    asyncio.run(instance._execute_matrix("core_001", groups, manifest, concurrency=1))
    scored = instance.score("core_001")
    assert scored["valid"]
    assert scored["primary"]["M2"]["recall@10"] == 1.0
    analyzed = instance.analyze("core_001")
    assert analyzed["valid"]
    for filename in (
        "core_metrics.csv", "candidate_generation.csv", "confirmatory_contrasts.csv",
        "same_pool_ranking.csv", "subgroup_effects.csv", "cost.csv", "reliability.csv",
        "report.md", "trace_examples.md", "limitations.md",
    ):
        assert (instance._run_dir("core_001") / "analysis" / filename).is_file()
    score_manifest_path = instance._run_dir("core_001") / "scores" / "manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text())
    original_scoring_hash = score_manifest["scoring_code_sha256"]
    score_manifest["scoring_code_sha256"] = "0" * 64
    score_manifest_path.write_text(json.dumps(score_manifest))
    with pytest.raises(ProtocolError):
        instance._assert_scores_fresh("core_001")
    with pytest.raises(ProtocolError):
        instance.analyze("core_001")
    score_manifest["scoring_code_sha256"] = original_scoring_hash
    score_manifest_path.write_text(json.dumps(score_manifest))
    monkeypatch.setattr(instance, "_assert_frozen_current", lambda run_id: manifest)
    monkeypatch.setattr(instance, "analyze", lambda run_id: {"valid": True})
    stability = asyncio.run(instance.run_stability("stability_001", core_run_id="core_001", concurrency=1))
    assert stability["valid"]
    summary = json.loads((instance._run_dir("stability_001") / "summary.json").read_text())
    assert summary["additional_repeats"] == 2
    assert len(summary["variants"]) == 5
    assert all(row["groups"] == 1 for row in summary["variants"])
    assert all(row["independent_primary_n_added"] == 0 for row in summary["variants"])
    (instance._run_dir("core_001") / "validation.json").write_text(json.dumps({"valid": True, "errors": []}))
    packed = instance.pack("core_001")
    assert packed["valid"]
    assert (instance.output_root / "results_pack.zip").is_file()
    assert packed["core_validation"]
    assert packed["files"] > 0
    selected_group = summary["selected_groups"][0]
    core_pool_path = instance._group_path("core_001", "M2", selected_group)
    core_artifact = json.loads(core_pool_path.read_text())
    core_artifact["pool"]["pool_sha256"] = "0" * 64
    core_pool_path.write_text(json.dumps(core_artifact))
    linkage = instance._validate_stability_linkage("core_001", "stability_001")
    assert not linkage["valid"]
    assert any("core_m2_pool_hashes" in error or "original core M2 pool" in error for error in linkage["errors"])
    with pytest.raises(ProtocolError):
        instance.pack("core_001")
