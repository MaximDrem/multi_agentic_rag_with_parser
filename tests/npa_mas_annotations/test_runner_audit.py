import json
from pathlib import Path

import pytest
import yaml

from scripts.npa_mas_experiment.runner import ExperimentRunner, ProtocolError
from scripts.npa_mas_experiment.schema import load_benchmark
from scripts.npa_mas_experiment.annotations import annotation_template, select_double_annotation, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "npa_mas_component_v1.yaml"


def runner():
    return ExperimentRunner(CONFIG)


def test_real_benchmark_structural_counts_and_protected_hashes(monkeypatch):
    monkeypatch.setenv("NPA_LLM_BASE_URL", "http://localhost:8070/v1")
    monkeypatch.setenv("NPA_LLM_MODEL", "test-model")
    report = runner().audit_inputs(require_annotations=False)
    assert report["valid"]
    assert report["counts"] == {
        "rows": 630,
        "groups": 556,
        "answerable_rows": 617,
        "answerable_groups": 544,
        "rows_without_qrels": 13,
    }
    assert report["annotations"]["status"] == "missing_allowed_for_structural_audit"
    assert report["chunk_index_manifest"]


def test_templates_cover_all_groups_and_exact_control_sample(tmp_path):
    report = runner().prepare_annotation_template(tmp_path)
    assert report["groups"] == 556
    assert report["double_annotation_groups"] == 112
    primary = (tmp_path / "npa_query_types_v1_human_a.template.jsonl").read_text().splitlines()
    secondary = (tmp_path / "npa_query_types_v1_human_b.template.jsonl").read_text().splitlines()
    assert len(primary) == 556
    assert len(secondary) == 112
    selection = json.loads((tmp_path / "npa_query_types_v1.double_annotation_ids.json").read_text())
    assert len(selection["query_groups"]) == 112
    review_text = (tmp_path / "npa_query_types_v1_human_a.review_pack.jsonl").read_text()
    assert "relevant" not in review_text
    assert "qrels" not in review_text
    assert "ranked_chunk_ids" not in review_text
    assert not report["final_labels_created"]


def test_freeze_cannot_bypass_missing_human_annotations(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_LLM_BASE_URL", "http://user:secret@localhost:8070/v1")
    monkeypatch.setenv("NPA_LLM_MODEL", "test-model")
    instance = runner()
    instance.output_root = tmp_path / "evaluation"
    with pytest.raises(ProtocolError):
        instance.freeze("core_001")
    assert not instance.output_root.exists()


def test_valid_synthetic_human_annotations_can_freeze_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_LLM_BASE_URL", "http://user:secret@localhost:8070/v1")
    monkeypatch.setenv("NPA_LLM_MODEL", "test-model")
    instance = runner()
    groups = instance._load_groups()
    rows = annotation_template(groups)
    selected = select_double_annotation(groups, 20261127)
    by_id = {row["query_group"]: row for row in rows}
    for row in rows:
        row["labels"] = {label: False for label in row["labels"]}
        row["annotation_confidence"] = "high"
        row["annotator"] = "human_a"
        row["review_status"] = "human_reviewed"
    secondary = []
    for group_id in selected:
        row = dict(by_id[group_id])
        row["labels"] = {label: False for label in row["labels"]}
        row["annotation_confidence"] = "high"
        row["annotator"] = "human_b"
        row["review_status"] = "human_reviewed"
        secondary.append(row)
    paths = {key: tmp_path / f"{key}.jsonl" for key in ("labels", "primary_independent", "secondary_independent", "adjudications")}
    paths["reliability"] = tmp_path / "reliability.json"
    write_jsonl(paths["labels"], rows)
    write_jsonl(paths["primary_independent"], rows)
    write_jsonl(paths["secondary_independent"], secondary)
    write_jsonl(paths["adjudications"], [])
    for key, path in paths.items():
        instance.config["annotations"][key] = str(path)
    instance.output_root = tmp_path / "evaluation"
    instance.audit_root = tmp_path / "audit"
    assert instance.validate_annotation_inputs()["valid"]
    result = instance.freeze("core_001")
    assert result["frozen"]
    freeze_dir = instance.output_root / "core_001" / "freeze"
    assert (freeze_dir / "annotations_reliability.json").is_file()
    serialized = (freeze_dir / "model_manifest.json").read_text() + (freeze_dir / "freeze_manifest.json").read_text()
    assert "secret" not in serialized
    assert "localhost:8070" not in serialized
    assert json.loads((freeze_dir / "model_manifest.json").read_text())["base_url_recorded"] is False
    original_config_path = instance.config_path
    changed_config = tmp_path / "changed.yaml"
    changed_config.write_text(yaml.safe_dump(instance.config, allow_unicode=True, sort_keys=True))
    instance.config_path = changed_config
    with pytest.raises(ProtocolError):
        instance._assert_frozen_current("core_001")
    instance.config_path = original_config_path
    monkeypatch.setenv("NPA_LLM_MODEL", "different-model")
    with pytest.raises(ProtocolError):
        instance._assert_frozen_current("core_001")
    monkeypatch.setenv("NPA_LLM_MODEL", "test-model")
    rows[0]["notes"] = "Editorial clarification without a label change"
    write_jsonl(paths["primary_independent"], rows)
    stale = instance._annotation_audit(True)
    assert not stale["valid"]
    assert any("stale" in error for error in stale["errors"])
    with pytest.raises(ProtocolError):
        instance._assert_frozen_current("core_001")


def test_quality_independent_group_selection_and_resume_keys():
    instance = runner()
    pilot = instance._select_groups(24)
    assert pilot == instance._select_groups(24)
    assert len({group["query_group"] for group in pilot}) == 24
    stability = instance._select_groups(50, answerable_only=True)
    assert len(stability) == 50
    assert all(group["answerable"] for group in stability)
    manifest = {
        key: value
        for key, value in (
            ("protocol_version", "npa_mas_component_v1"),
            ("config_sha256", "a"),
            ("prompt_bundle_sha256", "b"),
            ("annotations_sha256", "c"),
            ("source_code_sha256", {"x": "d"}),
            ("benchmark_hashes", {}),
            ("chunk_index_manifest", {}),
            ("model_manifest", {}),
        )
    }
    key = instance._resume_key(manifest, "M2", "group_0000")
    assert key == instance._resume_key(manifest, "M2", "group_0000")
    assert key != instance._resume_key(manifest, "M3", "group_0000")
    changed = dict(manifest)
    changed["config_sha256"] = "different"
    assert key != instance._resume_key(changed, "M2", "group_0000")
    linked = {
        **manifest,
        "core_run_id": "core_001",
        "core_m2_pool_hashes": {"group_0000": "a" * 64},
    }
    linked_key = instance._resume_key(linked, "M3", "group_0000")
    changed_pool = {
        **linked,
        "core_m2_pool_hashes": {"group_0000": "b" * 64},
    }
    assert linked_key != instance._resume_key(changed_pool, "M3", "group_0000")
    changed_core = {**linked, "core_run_id": "other_core"}
    assert linked_key != instance._resume_key(changed_core, "M3", "group_0000")


def test_group_artifacts_resume_and_duplicate_row_projection(tmp_path):
    instance = runner()
    instance.output_root = tmp_path
    manifest = {
        "protocol_version": "npa_mas_component_v1",
        "config_sha256": "a",
        "prompt_bundle_sha256": "b",
        "annotations_sha256": "c",
        "source_code_sha256": "d" * 64,
        "benchmark_hashes": {},
        "chunk_index_manifest": {},
        "model_manifest": {},
    }
    groups = instance._load_groups()[:3]
    for group in groups:
        result = {
            "ranked_chunk_ids": [f"chunk-{group['query_group']}"],
            "fallback_used": False,
            "strict_failed": False,
        }
        instance._write_group_result("run", "B0_RRF", group, result, manifest)
    summary = instance._finalize_variant("run", "B0_RRF", groups, manifest)
    expected_rows = sum(len(group["question_ids"]) for group in groups)
    assert summary["rows"] == expected_rows
    rows = (instance._variant_dir("run", "B0_RRF") / "run.jsonl").read_text().splitlines()
    assert len(rows) == expected_rows
    for group in groups:
        projected = [json.loads(row) for row in rows if json.loads(row)["query_group"] == group["query_group"]]
        assert {row["question_id"] for row in projected} == set(group["question_ids"])
        assert len({tuple(row["ranked_chunk_ids"]) for row in projected}) == 1
    with pytest.raises(ProtocolError):
        instance._write_group_result("run", "B1", groups[0], {"ranked_chunk_ids": ["x", "x"]}, manifest)
    changed = dict(manifest)
    changed["config_sha256"] = "new"
    with pytest.raises(ProtocolError):
        instance._write_group_result("run", "B0_RRF", groups[0], {"ranked_chunk_ids": ["x"]}, changed)


def test_runner_group_ids_match_canonical_scoring_benchmark():
    instance = runner()
    benchmark = load_benchmark(
        ROOT / "var" / "benchmark" / "final" / "questions.jsonl",
        ROOT / "var" / "benchmark" / "final" / "qrels.jsonl",
        config=instance.config,
        strict_audit=True,
    )
    groups = instance._load_groups()
    assert len(groups) == len(benchmark.groups)
    for actual, expected in zip(groups, benchmark.groups, strict=True):
        assert actual["query_group"] == expected.query_group
        assert actual["representative_question_id"] == expected.representative_question_id
        assert tuple(actual["question_ids"]) == expected.question_ids
        assert actual["query"] == expected.query
        assert actual["relevant"] == dict(expected.relevant)
