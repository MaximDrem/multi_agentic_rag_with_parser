import asyncio
import json
from pathlib import Path

import pytest

from retrieval.npa_mas.schemas import Candidate, CandidatePool
from scripts.npa_mas_experiment.runner import ExperimentRunner, ProtocolError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "npa_mas_component_v1.yaml"


def manifest():
    return {
        "protocol_version": "npa_mas_component_v1",
        "run_id": "matrix",
        "engineering_only": True,
        "config_sha256": "a",
        "prompt_bundle_sha256": "b",
        "annotations_sha256": "c",
        "source_code_sha256": "d" * 64,
        "benchmark_hashes": {},
        "chunk_index_manifest": {},
        "model_manifest": {},
        "variants": ["B0_BM25", "B0_DENSE", "B0_RRF", "B1", "B1W", "S", "M1", "M2", "M3", "M4"],
    }


def pool(group_id):
    candidates = [
        Candidate(
            chunk_id=f"{group_id}-chunk-{index}",
            pool_rank=index + 1,
            selector_text=f"Текст {index}",
            text_sha256=f"{index:064x}",
            doc_path="doc.md",
            title="Документ",
            source_ranks={"dense": index + 1},
            rrf_score=1.0 / (60 + index + 1),
            text=f"Текст {index}",
        )
        for index in range(2)
    ]
    return CandidatePool.create(
        query_group=group_id,
        candidates=candidates,
        budget={},
        generated_queries={},
        provenance={"observations": {"bm25": [], "dense": []}},
    )


class FakePipeline:
    def __init__(self):
        self.calls = []

    async def run(self, query, variant, *, query_group, m2_artifact=None):
        self.calls.append((variant, query_group))
        if variant in {"M3", "M4"}:
            assert m2_artifact is not None
            assert m2_artifact["variant"] == "M2"
            assert m2_artifact["query_group"] == query_group
        candidate_pool = pool(query_group)
        if m2_artifact is not None:
            candidate_pool = CandidatePool.model_validate(m2_artifact["pool"])
        return {
            "variant": variant,
            "query": query,
            "query_group": query_group,
            "status": "success",
            "strict_failed": False,
            "fallback_used": False,
            "ranked_chunk_ids": candidate_pool.candidate_ids[:1],
            "chunks": [],
            "pool": candidate_pool.model_dump(mode="json"),
            "plan": None,
            "worker_queries": {},
            "reports": {},
            "calls": [],
            "token_usage": [],
            "stage_timings_ms": {},
            "latency_ms": 1.0,
            "error": None,
        }


def test_full_matrix_same_pool_resume_and_invalidation(tmp_path):
    fake = FakePipeline()
    instance = ExperimentRunner(CONFIG, pipeline=fake)
    instance.output_root = tmp_path
    groups = instance._load_groups()[:2]
    frozen = manifest()
    result = asyncio.run(instance._execute_matrix("matrix", groups, frozen, concurrency=1))
    assert result["complete"]
    assert [variant for variant, _ in fake.calls[::2]] == instance.config["run"]["variants"]
    assert len(fake.calls) == 20
    for variant in instance.config["run"]["variants"]:
        assert len(list((instance._variant_dir("matrix", variant) / "groups").glob("*.json"))) == 2
        assert (instance._variant_dir("matrix", variant) / "run.jsonl").is_file()
    for group in groups:
        pools = [
            json.loads(instance._group_path("matrix", variant, group["query_group"]).read_text())["pool"]["pool_sha256"]
            for variant in ("M2", "M3", "M4")
        ]
        assert len(set(pools)) == 1
        assert (instance._run_dir("matrix") / "pools" / "M2" / f"{group['query_group']}.json").is_file()
    resumed_fake = FakePipeline()
    instance.pipeline = resumed_fake
    asyncio.run(instance._execute_matrix("matrix", groups, frozen, concurrency=1))
    assert resumed_fake.calls == []
    changed = dict(frozen)
    changed["config_sha256"] = "changed"
    with pytest.raises(ProtocolError):
        asyncio.run(instance._execute_matrix("matrix", groups, changed, concurrency=1))


def test_unhandled_pipeline_error_is_serialized_and_does_not_skip_matrix(tmp_path):
    class FailingPipeline(FakePipeline):
        async def run(self, query, variant, *, query_group, m2_artifact=None):
            if variant == "S":
                raise RuntimeError("synthetic failure")
            return await super().run(query, variant, query_group=query_group, m2_artifact=m2_artifact)

    instance = ExperimentRunner(CONFIG, pipeline=FailingPipeline())
    instance.output_root = tmp_path
    groups = instance._load_groups()[:1]
    result = asyncio.run(instance._execute_matrix("matrix", groups, manifest(), concurrency=1))
    assert result["complete"]
    artifact = instance._load_group_result("matrix", "S", groups[0]["query_group"])
    assert artifact["status"] == "failed"
    assert artifact["strict_failed"]
    assert "synthetic failure" in artifact["error"]
    assert (instance._variant_dir("matrix", "M4") / "run.jsonl").is_file()
