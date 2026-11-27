import json
import zipfile
from pathlib import Path

import pytest

from scripts.npa_mas_experiment.packaging import (
    ANALYSIS_FILES,
    canonical_relative_path,
    create_results_pack,
    validate_budget_invariants,
    validate_protected_hashes,
    validate_results_pack,
    validate_zip_archive,
)
from scripts.npa_mas_experiment.schema import (
    VARIANTS,
    assert_artifact_clean,
    atomic_write_json,
    candidate_pool_sha256,
    read_json,
    scan_artifact,
    sha256_canonical,
    sha256_directory,
    sha256_file,
    verify_file_hashes,
)

COUNTS = {
    "questions_total": 1,
    "groups_total": 1,
    "answerable_rows": 1,
    "answerable_groups": 1,
    "rows_without_qrels": 0,
}
def candidate():
    return {
        "pool_rank": 1,
        "chunk_id": "chunk-a",
        "selector_text": "text",
        "doc_path": "doc.md",
        "title": "title",
    }


def group_artifact(variant):
    artifact = {
        "query_group": "group_0000",
        "ranked_chunk_ids": ["chunk-a"],
        "full_ranked_chunk_ids": ["chunk-a"],
        "candidate_pool": {"candidates": [candidate()]},
        "budget": {
            "target_raw_hits": 11,
            "requested_raw_hits": 11,
            "actual_raw_hits": 1,
            "unspent_raw_hits": 10,
            "channel_status": "saturated",
        },
    }
    if variant in {"M2", "M3", "M4"}:
        pool_hash = candidate_pool_sha256(artifact["candidate_pool"]["candidates"])
        artifact["pool_sha256"] = pool_hash
        artifact["candidate_pool"]["pool_sha256"] = pool_hash
    return artifact


def make_run(root):
    for variant in VARIANTS:
        variant_root = root / "variants" / variant
        atomic_write_json(variant_root / "groups" / "group_0000.json", group_artifact(variant))
        atomic_write_json(
            variant_root / "run.jsonl",
            [{"question_id": "row_0001", "query_group": "group_0000", "ranked_chunk_ids": ["chunk-a"]}],
        )
        (variant_root / "run.jsonl").write_text(
            '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":["chunk-a"]}\n',
            encoding="utf-8",
        )
        atomic_write_json(variant_root / "detailed.jsonl", [{"query_group": "group_0000"}])
        (variant_root / "detailed.jsonl").write_text('{"query_group":"group_0000"}\n', encoding="utf-8")
        atomic_write_json(variant_root / "manifest.json", {"variant": variant})
        (variant_root / "errors.jsonl").write_bytes(b"")
    return root


def validate(root):
    return validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a"},
        require_complete=False,
    )


def test_minimal_validation_success(tmp_path):
    root = make_run(tmp_path / "run")
    report = validate(root)
    assert report["valid"], report["errors"]


def test_complete_hash_and_protected_validation(tmp_path):
    root = make_run(tmp_path / "run")
    freeze = root / "freeze"
    (freeze / "prompt_bundle").mkdir(parents=True)
    (freeze / "protocol.yaml").write_text("protocol_version: npa_mas_component_v1\n", encoding="utf-8")
    (freeze / "prompt_bundle" / "planner.txt").write_text("planner\n", encoding="utf-8")
    (freeze / "annotations.jsonl").write_text('{"query_group":"group_0000","labels":{}}\n', encoding="utf-8")
    (freeze / "annotations_codebook.md").write_text("# Codebook\n", encoding="utf-8")
    (freeze / "annotations_human_a.jsonl").write_text('{"query_group":"group_0000","labels":{}}\n', encoding="utf-8")
    (freeze / "annotations_human_b.jsonl").write_text('{"query_group":"group_0000","labels":{}}\n', encoding="utf-8")
    (freeze / "annotations_adjudications.jsonl").write_text("", encoding="utf-8")
    atomic_write_json(freeze / "annotations_reliability.json", {"valid": True, "control_groups": 1})
    for name in ("model_manifest.json", "retrieval_manifest.json", "chunk_index_manifest.json", "literature.json"):
        atomic_write_json(freeze / name, {"recorded": True})
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("value = 1\n", encoding="utf-8")
    source_hashes = {"module.py": sha256_file(source_root / "module.py")}
    atomic_write_json(freeze / "source_code_hashes.json", source_hashes)
    annotation_files = {
        "codebook": "annotations_codebook.md",
        "labels": "annotations.jsonl",
        "primary_independent": "annotations_human_a.jsonl",
        "secondary_independent": "annotations_human_b.jsonl",
        "adjudications": "annotations_adjudications.jsonl",
        "reliability": "annotations_reliability.json",
    }
    annotation_hashes = {key: sha256_file(freeze / filename) for key, filename in annotation_files.items()}
    atomic_write_json(freeze / "annotations_bundle_hashes.json", annotation_hashes)
    annotation_bundle_hash = sha256_canonical(annotation_hashes)
    (freeze / "annotations_bundle_sha256.txt").write_text(annotation_bundle_hash + "\n", encoding="utf-8")
    hashes = {
        "config_sha256": sha256_file(freeze / "protocol.yaml"),
        "prompt_bundle_sha256": sha256_directory(freeze / "prompt_bundle"),
        "source_code_sha256": sha256_canonical(source_hashes),
        "annotations_sha256": sha256_file(freeze / "annotations.jsonl"),
        "annotations_bundle_sha256": annotation_bundle_hash,
    }
    (freeze / "config_sha256.txt").write_text(hashes["config_sha256"] + "\n", encoding="utf-8")
    (freeze / "prompt_bundle_sha256.txt").write_text(hashes["prompt_bundle_sha256"] + "\n", encoding="utf-8")
    (freeze / "annotations_sha256.txt").write_text(hashes["annotations_sha256"] + "\n", encoding="utf-8")
    atomic_write_json(root / "run_manifest.json", hashes)
    atomic_write_json(freeze / "freeze_manifest.json", hashes)
    protected = tmp_path / "protected.json"
    atomic_write_json(protected, {"fixed": True})
    atomic_write_json(freeze / "benchmark_hashes_before.json", {"protected.json": sha256_file(protected)})
    for variant in VARIANTS:
        variant_root = root / "variants" / variant
        atomic_write_json(variant_root / "manifest.json", {"variant": variant, **hashes})
        group_path = variant_root / "groups" / "group_0000.json"
        atomic_write_json(group_path, {**read_json(group_path), **hashes})
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a"},
        source_root=source_root,
        protected_base_dir=tmp_path,
        expected_hashes=hashes,
    )
    assert report["valid"], report["errors"]


def test_validation_detects_missing_variant(tmp_path):
    root = make_run(tmp_path / "run")
    variant = root / "variants" / "M4"
    for path in sorted(variant.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    variant.rmdir()
    report = validate(root)
    assert not report["valid"]
    assert "variant_matrix" in {error["code"] for error in report["errors"]}


def test_validation_detects_duplicate_and_unknown_ids(tmp_path):
    root = make_run(tmp_path / "run")
    run_path = root / "variants" / "B1" / "run.jsonl"
    run_path.write_text(
        '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":["chunk-a","chunk-a"]}\n',
        encoding="utf-8",
    )
    report = validate(root)
    assert not report["valid"]
    assert "invalid_final_ranking" in {error["code"] for error in report["errors"]}
    run_path.write_text(
        '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":["unknown"]}\n',
        encoding="utf-8",
    )
    report = validate(root)
    assert "invalid_final_ranking" in {error["code"] for error in report["errors"]}


def test_validation_rejects_more_than_ten_ids(tmp_path):
    root = make_run(tmp_path / "run")
    ids = [f"chunk-{index}" for index in range(11)]
    run_path = root / "variants" / "B1" / "run.jsonl"
    run_path.write_text(
        json.dumps({"question_id": "row_0001", "query_group": "group_0000", "ranked_chunk_ids": ids}) + "\n",
        encoding="utf-8",
    )
    group_path = root / "variants" / "B1" / "groups" / "group_0000.json"
    atomic_write_json(group_path, {**read_json(group_path), "ranked_chunk_ids": ids})
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids=set(ids) | {"chunk-a"},
        require_complete=False,
    )
    assert "ranking_too_long" in {error["code"] for error in report["errors"]}


def test_validation_detects_budget_and_same_pool_failures(tmp_path):
    root = make_run(tmp_path / "run")
    budget_path = root / "variants" / "S" / "groups" / "group_0000.json"
    artifact = read_json(budget_path)
    artifact["budget"]["actual_raw_hits"] = 12
    atomic_write_json(budget_path, artifact)
    pool_path = root / "variants" / "M3" / "groups" / "group_0000.json"
    artifact = read_json(pool_path)
    artifact["pool_sha256"] = "b" * 64
    atomic_write_json(pool_path, artifact)
    report = validate(root)
    codes = {error["code"] for error in report["errors"]}
    assert "budget_actual_exceeds_requested" in codes
    assert "budget_unspent_mismatch" in codes
    assert "same_pool_hash_mismatch" in codes


def test_failed_none_same_pool_is_validation_error_not_crash(tmp_path):
    root = make_run(tmp_path / "run")
    path = root / "variants" / "M3" / "groups" / "group_0000.json"
    artifact = read_json(path)
    artifact.pop("candidate_pool")
    artifact["pool"] = None
    artifact["ranked_chunk_ids"] = []
    artifact["full_ranked_chunk_ids"] = []
    artifact["status"] = "failed"
    artifact["error"] = "search_error"
    atomic_write_json(path, artifact)
    (root / "variants" / "M3" / "run.jsonl").write_text(
        '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":[]}\n',
        encoding="utf-8",
    )
    report = validate(root)
    assert not report["valid"]
    assert "same_pool_hash_mismatch" in {error["code"] for error in report["errors"]}


def test_valid_store_id_outside_pool_is_rejected(tmp_path):
    root = make_run(tmp_path / "run")
    path = root / "variants" / "B1" / "groups" / "group_0000.json"
    atomic_write_json(path, {**read_json(path), "ranked_chunk_ids": ["chunk-b"]})
    (root / "variants" / "B1" / "run.jsonl").write_text(
        '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":["chunk-b"]}\n',
        encoding="utf-8",
    )
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a", "chunk-b"},
        require_complete=False,
    )
    assert not report["valid"]
    assert "unknown_final_id" in {error["code"] for error in report["errors"]}


def test_full_ranking_must_be_exact_pool_permutation(tmp_path):
    root = make_run(tmp_path / "run")
    path = root / "variants" / "B1" / "groups" / "group_0000.json"
    artifact = read_json(path)
    second = {**candidate(), "pool_rank": 2, "chunk_id": "chunk-b"}
    artifact["candidate_pool"]["candidates"].append(second)
    artifact["full_ranked_chunk_ids"] = ["chunk-a"]
    atomic_write_json(path, artifact)
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a", "chunk-b"},
        require_complete=False,
    )
    assert not report["valid"]
    assert "invalid_full_permutation" in {error["code"] for error in report["errors"]}


def test_declared_failed_empty_selector_output_is_scoreable(tmp_path):
    root = make_run(tmp_path / "run")
    path = root / "variants" / "B1" / "groups" / "group_0000.json"
    artifact = read_json(path)
    artifact.update({
        "status": "failed",
        "error": "jina_error:Timeout",
        "ranked_chunk_ids": [],
        "full_ranked_chunk_ids": [],
    })
    atomic_write_json(path, artifact)
    (root / "variants" / "B1" / "run.jsonl").write_text(
        '{"question_id":"row_0001","query_group":"group_0000","ranked_chunk_ids":[]}\n',
        encoding="utf-8",
    )
    report = validate(root)
    assert report["valid"], report["errors"]


def test_validation_detects_secrets_and_old_snapshots(tmp_path):
    root = make_run(tmp_path / "run")
    atomic_write_json(root / "variants" / "M1" / "groups" / "group_0000.json", {**group_artifact("M1"), "notes": "sk-live_123456789012345"})
    (root / "coordinated_mechanism_v1").mkdir()
    (root / "coordinated_mechanism_v1" / "old.json").write_text("{}", encoding="utf-8")
    report = validate(root)
    codes = {error["code"] for error in report["errors"]}
    assert any(code.startswith("artifact_openai_like_key") for code in codes)
    assert "old_snapshot_forbidden" in codes


def test_scored_denominator_validation_failure(tmp_path):
    root = make_run(tmp_path / "run")
    atomic_write_json(
        root / "scores" / "M1" / "summary.json",
        {"answerable_unique_queries": 0, "answerable_rows": 0},
    )
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a"},
        require_complete=False,
        require_scored=True,
    )
    codes = {error["code"] for error in report["errors"]}
    assert "answerable_group_denominator" in codes
    assert "answerable_row_denominator" in codes


def test_complete_scored_analysis_inventory_is_accepted(tmp_path):
    root = make_run(tmp_path / "run")
    atomic_write_json(root / "scores" / "manifest.json", {"complete": True})
    for variant in VARIANTS:
        for score_root in (root / "scores" / variant, root / "scores" / variant / "strict"):
            atomic_write_json(score_root / "summary.json", {"answerable_unique_queries": 1, "answerable_rows": 1})
            (score_root / "per_unique_query.jsonl").write_text('{"query_group":"group_0000"}\n', encoding="utf-8")
            (score_root / "per_query.jsonl").write_text('{"question_id":"row_0001"}\n', encoding="utf-8")
            (score_root / "all_groups.jsonl").write_text('{"query_group":"group_0000"}\n', encoding="utf-8")
    analysis = root / "analysis"
    analysis.mkdir()
    all_variant_tables = {"core_metrics.csv", "cost.csv", "reliability.csv", "candidate_generation_summary.csv", "output_policy_sensitivity.csv"}
    for filename in ANALYSIS_FILES:
        path = analysis / filename
        if filename in all_variant_tables:
            path.write_text("variant,value\n" + "".join(f"{variant},0\n" for variant in VARIANTS), encoding="utf-8")
        elif filename in {"confirmatory_contrasts.csv", "confirmatory_contrasts_strict.csv"}:
            path.write_text("contrast,value\n" + "".join(f"C{index},0\n" for index in range(1, 5)), encoding="utf-8")
        elif filename == "stability.csv":
            path.write_text("variant,value\n" + "".join(f"{variant},0\n" for variant in ("S", "M1", "M2", "M3", "M4")), encoding="utf-8")
        elif filename.endswith(".csv"):
            path.write_text("value\n1\n", encoding="utf-8")
        elif filename.endswith(".json"):
            atomic_write_json(path, {"complete": True})
        else:
            path.write_text("# Evidence\n", encoding="utf-8")
    report = validate_results_pack(
        root,
        expected_counts=COUNTS,
        valid_chunk_ids={"chunk-a"},
        require_complete=False,
        require_scored=True,
    )
    assert report["valid"], report["errors"]


def test_budget_saturation_and_hash_validation(tmp_path):
    valid = {"target_raw_hits": 11, "requested_raw_hits": 11, "actual_raw_hits": 1, "channel_status": "saturated"}
    assert validate_budget_invariants(valid) == []
    invalid = {**valid, "actual_raw_hits": 0}
    assert validate_budget_invariants(invalid)[0]["code"] == "budget_status_mismatch"
    protected = tmp_path / "protected.json"
    atomic_write_json(protected, {"fixed": True})
    digest = sha256_file(protected)
    assert validate_protected_hashes({"protected.json": digest}, base_dir=tmp_path)["valid"]
    assert not validate_protected_hashes({"protected.json": digest}, {"protected.json": "0" * 64})["valid"]
    assert verify_file_hashes({"protected": protected}, {"protected": digest})["valid"]


def test_secret_scanner_does_not_echo_values():
    secret = "sk-test_123456789012345"
    artifact = {
        "api_key": secret,
        "nested": {"text": f"prefix {secret}"},
        "environment": {"HOME": "/private/home"},
        "prompt_tokens": 0,
        "token_usage_status": "reported",
    }
    findings = scan_artifact(artifact, secret_values={"test": secret})
    kinds = {finding["kind"] for finding in findings}
    assert "sensitive_key" in kinds
    assert "raw_secret" in kinds
    assert "environment_dump" in kinds
    assert "absolute_path" in kinds
    assert secret not in json.dumps(findings)
    with pytest.raises(ValueError):
        assert_artifact_clean(artifact, secret_values={"test": secret})
    assert scan_artifact({"prompt_tokens": 0, "completion_tokens": None}) == []


def test_safe_deterministic_zip_and_secret_rejection(tmp_path):
    source = tmp_path / "source"
    atomic_write_json(source / "nested" / "result.json", {"value": 1})
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    result = create_results_pack(source, first)
    create_results_pack(source, second)
    assert result["files"] == 1
    assert sha256_file(first) == sha256_file(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["nested/result.json"]
    assert validate_zip_archive(first)["valid"]
    atomic_write_json(source / "nested" / "secret.json", {"api_key": "sk-test_123456789012345"})
    with pytest.raises(ValueError):
        create_results_pack(source, tmp_path / "unsafe.zip")
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("/absolute.json", "{}")
    assert not validate_zip_archive(malicious)["valid"]
    assert canonical_relative_path("nested/result.json") == "nested/result.json"
    with pytest.raises(ValueError):
        canonical_relative_path("../escape.json")
