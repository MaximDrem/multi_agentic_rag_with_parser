from __future__ import annotations

import csv
import json
import math
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .schema import (
    DEFAULT_EXPECTED_COUNTS,
    OLD_SNAPSHOT_MARKERS,
    SAME_POOL_VARIANTS,
    VARIANTS,
    Benchmark,
    candidate_pool_sha256,
    deduplicate_ranking,
    read_json,
    read_jsonl,
    read_yaml,
    scan_artifact,
    sha256_canonical,
    sha256_directory,
    sha256_file,
)

HASH_FIELDS = ("config_sha256", "prompt_bundle_sha256", "source_code_sha256", "annotations_sha256")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ANALYSIS_FILES = (
    "core_metrics.csv",
    "candidate_generation.csv",
    "candidate_generation_summary.csv",
    "confirmatory_contrasts.csv",
    "confirmatory_contrasts_strict.csv",
    "practical_comparisons.csv",
    "same_pool_ranking.csv",
    "same_pool_ranking_summary.csv",
    "subgroup_effects.csv",
    "subgroup_metrics.csv",
    "cost.csv",
    "reliability.csv",
    "output_policy_sensitivity.csv",
    "stability.csv",
    "trace_examples.md",
    "limitations.md",
    "manifest.json",
    "tables.json",
    "report.md",
)

FROZEN_ANNOTATION_FILES = {
    "codebook": "annotations_codebook.md",
    "labels": "annotations.jsonl",
    "primary_independent": "annotations_human_a.jsonl",
    "secondary_independent": "annotations_human_b.jsonl",
    "adjudications": "annotations_adjudications.jsonl",
    "reliability": "annotations_reliability.json",
}

COMPLETE_FREEZE_FILES = (
    "freeze_manifest.json",
    "model_manifest.json",
    "retrieval_manifest.json",
    "chunk_index_manifest.json",
    "annotations_bundle_hashes.json",
    "annotations_bundle_sha256.txt",
    "literature.json",
    *FROZEN_ANNOTATION_FILES.values(),
)


def _config_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (str, os.PathLike)):
        return read_yaml(config)
    return dict(config)


def _expected_counts(config: Mapping[str, Any], override: Mapping[str, int] | None) -> dict[str, int]:
    if override is not None:
        source = dict(override)
        aliases = {
            "expected_rows": "questions_total",
            "expected_groups": "groups_total",
            "expected_answerable_rows": "answerable_rows",
            "expected_answerable_groups": "answerable_groups",
            "expected_rows_without_qrels": "rows_without_qrels",
        }
        return {aliases.get(key, key): int(value) for key, value in source.items()}
    benchmark = config.get("benchmark", {})
    return {
        "questions_total": int(benchmark.get("expected_rows", DEFAULT_EXPECTED_COUNTS["questions_total"])),
        "groups_total": int(benchmark.get("expected_groups", DEFAULT_EXPECTED_COUNTS["groups_total"])),
        "answerable_rows": int(benchmark.get("expected_answerable_rows", DEFAULT_EXPECTED_COUNTS["answerable_rows"])),
        "answerable_groups": int(benchmark.get("expected_answerable_groups", DEFAULT_EXPECTED_COUNTS["answerable_groups"])),
        "rows_without_qrels": int(benchmark.get("expected_rows_without_qrels", DEFAULT_EXPECTED_COUNTS["rows_without_qrels"])),
    }


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def validate_budget_invariants(value: Any, *, path: str = "budget") -> list[dict[str, str]]:
    errors = []

    def walk(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            if any(key in item for key in ("target_raw_hits", "requested_raw_hits", "actual_raw_hits")):
                required = ("target_raw_hits", "requested_raw_hits", "actual_raw_hits")
                missing = [key for key in required if key not in item]
                if missing:
                    errors.append(_issue("budget_fields_missing", location, "Budget ledger is incomplete"))
                else:
                    numbers = {}
                    for key in required:
                        raw = item[key]
                        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                            errors.append(_issue("budget_not_numeric", location, f"Invalid {key}"))
                        else:
                            numbers[key] = float(raw)
                    if len(numbers) == 3:
                        target = numbers["target_raw_hits"]
                        requested = numbers["requested_raw_hits"]
                        actual = numbers["actual_raw_hits"]
                        if min(target, requested, actual) < 0:
                            errors.append(_issue("budget_negative", location, "Budget values must be non-negative"))
                        if requested > target:
                            errors.append(_issue("budget_requested_exceeds_target", location, "requested_raw_hits exceeds target_raw_hits"))
                        if actual > requested:
                            errors.append(_issue("budget_actual_exceeds_requested", location, "actual_raw_hits exceeds requested_raw_hits"))
                        if "unspent_raw_hits" in item:
                            unspent = item["unspent_raw_hits"]
                            if isinstance(unspent, bool) or not isinstance(unspent, (int, float)) or unspent != requested - actual:
                                errors.append(_issue("budget_unspent_mismatch", location, "unspent_raw_hits does not match requested minus actual"))
                        status = item.get("channel_status", item.get("status"))
                        if status == "exact_match" and actual != requested:
                            errors.append(_issue("budget_status_mismatch", location, "exact_match status has unequal hits"))
                        if status == "saturated" and not 0 < actual < requested:
                            errors.append(_issue("budget_status_mismatch", location, "saturated status has invalid hits"))
                        if status == "empty" and actual != 0:
                            errors.append(_issue("budget_status_mismatch", location, "empty status has non-zero hits"))
            for key, child in item.items():
                walk(child, f"{location}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")

    walk(value, path)
    return errors


def _read_hash_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    token = text.split()[0] if text.split() else ""
    return token if HASH_PATTERN.match(token) else None


def _declared_hash(manifest: Mapping[str, Any], field: str) -> str | None:
    value = manifest.get(field)
    if value is None and isinstance(manifest.get("hashes"), Mapping):
        value = manifest["hashes"].get(field)
    return str(value) if value is not None else None


def validate_protected_hashes(
    before: Mapping[str, str] | str | os.PathLike[str],
    after: Mapping[str, str] | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    expected = read_json(before) if isinstance(before, (str, os.PathLike)) else dict(before)
    if "protected_hashes" in expected:
        expected = expected["protected_hashes"]
    if after is not None and "protected_hashes" in after:
        after = after["protected_hashes"]
    root = Path(base_dir) if base_dir is not None else None
    errors = []
    actual = {}
    for name in sorted(expected):
        if after is not None:
            digest = after.get(name)
            if digest is None:
                errors.append(_issue("protected_hash_missing_after", name, "After hash is missing"))
                continue
        else:
            path = Path(name) if root is None else root / name
            if not path.is_file():
                errors.append(_issue("protected_file_missing", name, "Protected file is missing"))
                continue
            digest = sha256_file(path)
        actual[name] = str(digest)
        if digest != expected[name]:
            errors.append(_issue("protected_hash_mismatch", name, "Protected file hash changed"))
    if after is not None:
        for name in sorted(set(after) - set(expected)):
            errors.append(_issue("protected_hash_unexpected", name, "Unexpected after hash"))
    return {"valid": not errors, "actual_hashes": actual, "errors": errors}


def _validate_freeze_hashes(
    root: Path,
    *,
    source_root: Path | None,
    expected_hashes: Mapping[str, str] | None,
    errors: list[dict[str, str]],
    require_complete: bool = False,
) -> None:
    freeze = root / "freeze"
    if not freeze.is_dir():
        errors.append(_issue("freeze_missing", "freeze", "Freeze directory is missing"))
        return
    if require_complete:
        for name in COMPLETE_FREEZE_FILES:
            if not (freeze / name).is_file():
                errors.append(_issue("freeze_file_missing", f"freeze/{name}", "Required freeze evidence is missing"))
    protocol = freeze / "protocol.yaml"
    if not protocol.is_file():
        errors.append(_issue("config_missing", "freeze/protocol.yaml", "Frozen config is missing"))
        config_hash = None
    else:
        config_hash = sha256_file(protocol)
        recorded = _read_hash_text(freeze / "config_sha256.txt")
        if recorded != config_hash:
            errors.append(_issue("config_hash_mismatch", "freeze/config_sha256.txt", "Config hash does not match protocol.yaml"))
    prompt_bundle = freeze / "prompt_bundle"
    if not prompt_bundle.is_dir():
        errors.append(_issue("prompts_missing", "freeze/prompt_bundle", "Frozen prompt bundle is missing"))
        prompt_hash = None
    else:
        try:
            prompt_hash = sha256_directory(prompt_bundle)
        except ValueError:
            prompt_hash = None
            errors.append(_issue("prompt_hash_failed", "freeze/prompt_bundle", "Prompt bundle cannot be hashed safely"))
        recorded = _read_hash_text(freeze / "prompt_bundle_sha256.txt")
        if recorded != prompt_hash:
            errors.append(_issue("prompt_hash_mismatch", "freeze/prompt_bundle_sha256.txt", "Prompt bundle hash mismatch"))
    annotations = freeze / "annotations.jsonl"
    if not annotations.is_file():
        errors.append(_issue("annotations_missing", "freeze/annotations.jsonl", "Frozen annotations are missing"))
        annotations_hash = None
    else:
        annotations_hash = sha256_file(annotations)
        recorded = _read_hash_text(freeze / "annotations_sha256.txt")
        if recorded != annotations_hash:
            errors.append(_issue("annotations_hash_mismatch", "freeze/annotations_sha256.txt", "Annotations hash mismatch"))
    source_hashes = None
    source_hashes_path = freeze / "source_code_hashes.json"
    if not source_hashes_path.is_file():
        errors.append(_issue("source_hashes_missing", "freeze/source_code_hashes.json", "Source hashes are missing"))
        source_manifest = None
    else:
        source_manifest = read_json(source_hashes_path)
        source_hashes = source_manifest.get("files", source_manifest.get("hashes", source_manifest))
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            errors.append(_issue("source_hashes_empty", "freeze/source_code_hashes.json", "Source hashes are empty"))
        else:
            for name, digest in source_hashes.items():
                if not HASH_PATTERN.match(str(digest)):
                    errors.append(_issue("source_hash_invalid", "freeze/source_code_hashes.json", "Invalid source hash"))
                if Path(str(name)).is_absolute() or ".." in PurePosixPath(str(name)).parts:
                    errors.append(_issue("source_path_unsafe", "freeze/source_code_hashes.json", "Source paths must be relative and contained"))
                if source_root is not None:
                    source_path = source_root / str(name)
                    if not source_path.is_file() or sha256_file(source_path) != digest:
                        errors.append(_issue("source_hash_mismatch", str(name), "Source file hash mismatch"))
    source_aggregate = sha256_canonical(source_hashes) if isinstance(source_hashes, Mapping) and source_hashes else None
    run_manifest_path = root / "run_manifest.json"
    if not run_manifest_path.is_file():
        errors.append(_issue("run_manifest_missing", "run_manifest.json", "Run manifest is missing"))
        run_manifest = {}
    else:
        run_manifest = read_json(run_manifest_path)
    computed = {
        "config_sha256": config_hash,
        "prompt_bundle_sha256": prompt_hash,
        "annotations_sha256": annotations_hash,
    }
    for field, digest in computed.items():
        declared = _declared_hash(run_manifest, field)
        if declared is None:
            errors.append(_issue("manifest_hash_missing", "run_manifest.json", f"Missing {field}"))
        elif digest is not None and declared != digest:
            errors.append(_issue("manifest_hash_mismatch", "run_manifest.json", f"Mismatching {field}"))
    declared_source = _declared_hash(run_manifest, "source_code_sha256")
    if declared_source is None:
        errors.append(_issue("manifest_hash_missing", "run_manifest.json", "Missing source_code_sha256"))
    elif source_aggregate is not None and declared_source != source_aggregate:
        errors.append(_issue("source_aggregate_mismatch", "run_manifest.json", "source_code_sha256 does not match per-file source hashes"))
    if require_complete:
        bundle_hashes = {}
        for logical_name, filename in FROZEN_ANNOTATION_FILES.items():
            path = freeze / filename
            if path.is_file():
                bundle_hashes[logical_name] = sha256_file(path)
        if len(bundle_hashes) == len(FROZEN_ANNOTATION_FILES):
            recorded_bundle = read_json(freeze / "annotations_bundle_hashes.json") if (freeze / "annotations_bundle_hashes.json").is_file() else {}
            if recorded_bundle != bundle_hashes:
                errors.append(_issue("annotations_bundle_file_mismatch", "freeze/annotations_bundle_hashes.json", "Frozen annotation files differ from their bundle hashes"))
            bundle_hash = sha256_canonical(bundle_hashes)
            if _read_hash_text(freeze / "annotations_bundle_sha256.txt") != bundle_hash:
                errors.append(_issue("annotations_bundle_hash_mismatch", "freeze/annotations_bundle_sha256.txt", "Annotation bundle hash mismatch"))
            if _declared_hash(run_manifest, "annotations_bundle_sha256") != bundle_hash:
                errors.append(_issue("annotations_bundle_manifest_mismatch", "run_manifest.json", "Run manifest has a different annotation bundle"))
        freeze_manifest_path = freeze / "freeze_manifest.json"
        freeze_manifest = read_json(freeze_manifest_path) if freeze_manifest_path.is_file() else {}
        for field in (*HASH_FIELDS, "annotations_bundle_sha256"):
            if freeze_manifest.get(field) != run_manifest.get(field):
                errors.append(_issue("freeze_manifest_mismatch", "freeze/freeze_manifest.json", f"Freeze and run manifests differ for {field}"))
        reliability_path = freeze / "annotations_reliability.json"
        if reliability_path.is_file() and not read_json(reliability_path).get("valid"):
            errors.append(_issue("annotation_reliability_invalid", "freeze/annotations_reliability.json", "Human annotation reliability is not valid"))
        for name in ("model_manifest.json", "retrieval_manifest.json", "chunk_index_manifest.json"):
            path = freeze / name
            if path.is_file() and not isinstance(read_json(path), Mapping):
                errors.append(_issue("execution_manifest_invalid", f"freeze/{name}", "Execution manifest must be a JSON object"))
    for field, expected in (expected_hashes or {}).items():
        declared = _declared_hash(run_manifest, field)
        if declared != expected:
            errors.append(_issue("expected_hash_mismatch", "run_manifest.json", f"Unexpected {field}"))


def _validate_variant_manifest(
    manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    path: str,
    errors: list[dict[str, str]],
) -> None:
    for field in HASH_FIELDS:
        digest = _declared_hash(manifest, field)
        if digest is None or not HASH_PATTERN.match(digest):
            errors.append(_issue("variant_hash_missing", path, f"Missing or invalid {field}"))
        run_digest = _declared_hash(run_manifest, field)
        if digest is not None and run_digest is not None and digest != run_digest:
            errors.append(_issue("variant_hash_mismatch", path, f"Variant {field} differs from run manifest"))


def _validate_scores(
    root: Path,
    variants: Iterable[str],
    counts: Mapping[str, int],
    errors: list[dict[str, str]],
) -> None:
    if not (root / "scores" / "manifest.json").is_file():
        errors.append(_issue("score_manifest_missing", "scores/manifest.json", "Scoring manifest is missing"))
    for variant in variants:
        score_root = root / "scores" / variant
        summary_path = score_root / "summary.json"
        relative = f"scores/{variant}/summary.json"
        if not summary_path.is_file():
            errors.append(_issue("score_summary_missing", relative, "Score summary is missing"))
            continue
        summary = read_json(summary_path)
        if summary.get("answerable_unique_queries", summary.get("answerable_groups")) != counts["answerable_groups"]:
            errors.append(_issue("answerable_group_denominator", relative, "Incorrect answerable group denominator"))
        if summary.get("answerable_rows") != counts["answerable_rows"]:
            errors.append(_issue("answerable_row_denominator", relative, "Incorrect answerable row denominator"))
        per_group_path = score_root / "per_unique_query.jsonl"
        if not per_group_path.is_file() or len(read_jsonl(per_group_path)) != counts["answerable_groups"]:
            errors.append(_issue("score_group_rows", f"scores/{variant}/per_unique_query.jsonl", "Incorrect scored group rows"))
        per_row_path = score_root / "per_query.jsonl"
        if not per_row_path.is_file() or len(read_jsonl(per_row_path)) != counts["answerable_rows"]:
            errors.append(_issue("score_row_rows", f"scores/{variant}/per_query.jsonl", "Incorrect scored row rows"))
        all_groups_path = score_root / "all_groups.jsonl"
        if not all_groups_path.is_file() or len(read_jsonl(all_groups_path)) != counts["groups_total"]:
            errors.append(_issue("score_all_group_rows", f"scores/{variant}/all_groups.jsonl", "all_groups must include answerable and no-GT rows"))
        strict_root = score_root / "strict"
        strict_summary_path = strict_root / "summary.json"
        if not strict_summary_path.is_file():
            errors.append(_issue("strict_score_summary_missing", f"scores/{variant}/strict/summary.json", "Strict score summary is missing"))
        else:
            strict_summary = read_json(strict_summary_path)
            if strict_summary.get("answerable_unique_queries", strict_summary.get("answerable_groups")) != counts["answerable_groups"] or strict_summary.get("answerable_rows") != counts["answerable_rows"]:
                errors.append(_issue("strict_score_denominator", f"scores/{variant}/strict/summary.json", "Strict sensitivity has incorrect denominators"))
        for filename, expected in (("per_unique_query.jsonl", counts["answerable_groups"]), ("per_query.jsonl", counts["answerable_rows"]), ("all_groups.jsonl", counts["groups_total"])):
            path = strict_root / filename
            if not path.is_file() or len(read_jsonl(path)) != expected:
                errors.append(_issue("strict_score_rows", f"scores/{variant}/strict/{filename}", "Strict sensitivity row coverage is incomplete"))
    for filename in ANALYSIS_FILES:
        path = root / "analysis" / filename
        if not path.is_file():
            errors.append(_issue("analysis_file_missing", f"analysis/{filename}", "Required analysis output is missing"))
        elif filename.endswith(".csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                errors.append(_issue("analysis_table_empty", f"analysis/{filename}", "Required analysis table is empty"))
                continue
            if filename in {"core_metrics.csv", "cost.csv", "reliability.csv", "candidate_generation_summary.csv", "output_policy_sensitivity.csv"}:
                variants_in_table = {row.get("variant") for row in rows}
                if variants_in_table != set(VARIANTS):
                    errors.append(_issue("analysis_variant_coverage", f"analysis/{filename}", "Analysis table must cover all ten variants"))
            if filename in {"confirmatory_contrasts.csv", "confirmatory_contrasts_strict.csv"} and {row.get("contrast") for row in rows} != {"C1", "C2", "C3", "C4"}:
                errors.append(_issue("analysis_contrast_coverage", f"analysis/{filename}", "Confirmatory table must contain C1-C4"))
            if filename == "stability.csv" and {row.get("variant") for row in rows} != {"S", "M1", "M2", "M3", "M4"}:
                errors.append(_issue("stability_analysis_coverage", "analysis/stability.csv", "Stability table must cover S, M1, M2, M3 and M4"))


def validate_results_pack(
    root: str | os.PathLike[str],
    config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    *,
    benchmark: Benchmark | None = None,
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
    expected_counts: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    protected_hashes: Mapping[str, str] | str | os.PathLike[str] | None = None,
    protected_hashes_after: Mapping[str, str] | None = None,
    protected_base_dir: str | os.PathLike[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
    secret_values: Mapping[str, str] | Iterable[str] | None = None,
    require_scored: bool = False,
    require_complete: bool = True,
    require_hashes: bool | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    run_root = Path(root)
    config_mapping = _config_mapping(config)
    counts = _expected_counts(config_mapping, expected_counts)
    check_hashes = require_complete if require_hashes is None else require_hashes
    errors: list[dict[str, str]] = []
    if not run_root.is_dir():
        errors.append(_issue("run_root_missing", ".", "Run root is missing"))
        report = {"valid": False, "errors": errors, "expected_counts": counts}
        if raise_on_error:
            raise ValueError("Results pack validation failed")
        return report
    variants_root = run_root / "variants"
    actual_variants = sorted(path.name for path in variants_root.iterdir() if path.is_dir()) if variants_root.is_dir() else []
    if set(actual_variants) != set(VARIANTS):
        errors.append(_issue("variant_matrix", "variants", "Exactly ten frozen variants are required"))
    group_hashes: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in SAME_POOL_VARIANTS}
    run_manifest = read_json(run_root / "run_manifest.json") if (run_root / "run_manifest.json").is_file() else {}
    expected_question_ids = set(benchmark.question_by_id) if benchmark else None
    expected_group_ids = {f"group_{index:04d}" for index in range(counts["groups_total"])}
    for variant in VARIANTS:
        variant_root = variants_root / variant
        relative_root = f"variants/{variant}"
        if not variant_root.is_dir():
            continue
        manifest_path = variant_root / "manifest.json"
        errors_path = variant_root / "errors.jsonl"
        detailed_path = variant_root / "detailed.jsonl"
        run_path = variant_root / "run.jsonl"
        if not manifest_path.is_file():
            errors.append(_issue("variant_manifest_missing", f"{relative_root}/manifest.json", "Variant manifest is missing"))
            variant_manifest = {}
        else:
            variant_manifest = read_json(manifest_path)
            if check_hashes:
                _validate_variant_manifest(variant_manifest, run_manifest, f"{relative_root}/manifest.json", errors)
        if not errors_path.is_file():
            errors.append(_issue("errors_missing", f"{relative_root}/errors.jsonl", "errors.jsonl is required even when empty"))
        else:
            read_jsonl(errors_path)
        if not detailed_path.is_file() or len(read_jsonl(detailed_path)) != counts["groups_total"]:
            errors.append(_issue("detailed_group_count", f"{relative_root}/detailed.jsonl", "Incorrect detailed group count"))
        if not run_path.is_file():
            errors.append(_issue("run_missing", f"{relative_root}/run.jsonl", "run.jsonl is missing"))
            run_rows = []
        else:
            run_rows = read_jsonl(run_path)
        if len(run_rows) != counts["questions_total"]:
            errors.append(_issue("run_row_count", f"{relative_root}/run.jsonl", "Incorrect source row count"))
        seen_questions = set()
        run_by_group = {}
        for index, row in enumerate(run_rows):
            row_path = f"{relative_root}/run.jsonl:{index + 1}"
            question_id = str(row.get("question_id", ""))
            if question_id in seen_questions:
                errors.append(_issue("duplicate_question_id", row_path, "Duplicate source question ID"))
            seen_questions.add(question_id)
            if expected_question_ids is not None and question_id not in expected_question_ids:
                errors.append(_issue("unknown_question_id", row_path, "Unknown source question ID"))
            try:
                ranking = deduplicate_ranking(
                    row,
                    valid_chunk_ids=valid_chunk_ids,
                    max_items=None,
                    reject_duplicates=True,
                )
                if len(ranking) > 10:
                    errors.append(_issue("ranking_too_long", row_path, "More than ten final IDs"))
                group = str(row.get("query_group", ""))
                if group in run_by_group and run_by_group[group] != ranking:
                    errors.append(_issue("duplicate_projection_mismatch", row_path, "Duplicate rows do not share a group ranking"))
                run_by_group[group] = ranking
            except (ValueError, KeyError, TypeError):
                errors.append(_issue("invalid_final_ranking", row_path, "Unknown, duplicate, null, or invalid final IDs"))
        if expected_question_ids is not None and seen_questions != expected_question_ids:
            errors.append(_issue("question_coverage", f"{relative_root}/run.jsonl", "Run rows do not match benchmark question IDs"))
        groups_root = variant_root / "groups"
        group_files = sorted(groups_root.glob("*.json")) if groups_root.is_dir() else []
        if len(group_files) != counts["groups_total"]:
            errors.append(_issue("group_file_count", f"{relative_root}/groups", "Incorrect group JSON count"))
        actual_group_ids = set()
        for group_file in group_files:
            relative = f"{relative_root}/groups/{group_file.name}"
            artifact = read_json(group_file)
            group = str(artifact.get("query_group", group_file.stem))
            if group in actual_group_ids:
                errors.append(_issue("duplicate_query_group", relative, "Duplicate query group"))
            actual_group_ids.add(group)
            if group != group_file.stem or group not in expected_group_ids:
                errors.append(_issue("group_id_mismatch", relative, "Group file name and query_group are invalid"))
            errors.extend(validate_budget_invariants(artifact, path=relative))
            try:
                group_ranking = deduplicate_ranking(
                    artifact,
                    valid_chunk_ids=valid_chunk_ids,
                    max_items=None,
                    reject_duplicates=True,
                )
                if len(group_ranking) > 10:
                    errors.append(_issue("ranking_too_long", relative, "More than ten final group IDs"))
                pool = artifact.get("candidate_pool", artifact.get("pool", {}))
                if isinstance(pool, Mapping) and isinstance(pool.get("candidates"), list):
                    pool_ids = {str(candidate["chunk_id"]) for candidate in pool["candidates"]}
                    if set(group_ranking) - pool_ids:
                        errors.append(_issue("unknown_final_id", relative, "Final IDs are outside the candidate pool"))
                    full_ranking = artifact.get("full_ranked_chunk_ids")
                    declared_failed = artifact.get("status") == "failed" and bool(artifact.get("error"))
                    failed_empty = declared_failed and full_ranking == [] and not group_ranking
                    if not failed_empty:
                        valid_full = (
                            isinstance(full_ranking, list)
                            and all(isinstance(chunk_id, str) and chunk_id for chunk_id in full_ranking)
                            and len(full_ranking) == len(pool["candidates"])
                            and len(set(full_ranking)) == len(full_ranking)
                            and set(full_ranking) == pool_ids
                            and group_ranking == full_ranking[:10]
                        )
                        if not valid_full:
                            errors.append(_issue("invalid_full_permutation", relative, "Full ranking must be an exact pool permutation, except declared failed empty outputs"))
                if group in run_by_group and run_by_group[group] != group_ranking:
                    errors.append(_issue("group_run_ranking_mismatch", relative, "Group and projected rankings differ"))
            except (ValueError, KeyError, TypeError):
                errors.append(_issue("invalid_group_ranking", relative, "Invalid final group ranking"))
            if check_hashes:
                _validate_variant_manifest(artifact, run_manifest, relative, errors)
            if variant in SAME_POOL_VARIANTS:
                group_hashes[variant][group] = artifact
        if actual_group_ids != expected_group_ids:
            errors.append(_issue("group_coverage", f"{relative_root}/groups", "Group IDs do not match the complete frozen set"))
    for group in sorted(expected_group_ids):
        artifacts = [group_hashes[variant].get(group) for variant in SAME_POOL_VARIANTS]
        if any(artifact is None for artifact in artifacts):
            continue
        pools = []
        for artifact in artifacts:
            pool = artifact.get("candidate_pool", artifact.get("pool", {}))
            pools.append(pool if isinstance(pool, Mapping) else {})
        for artifact, pool in zip(artifacts, pools, strict=True):
            top_hash = artifact.get("pool_sha256")
            nested_hash = pool.get("pool_sha256") if pool else None
            if top_hash and nested_hash and top_hash != nested_hash:
                errors.append(_issue("same_pool_hash_mismatch", group, "Top-level and nested pool hashes differ"))
        hashes = [str((pool.get("pool_sha256") or artifact.get("pool_sha256") or "") if pool else "") for artifact, pool in zip(artifacts, pools, strict=True)]
        if any(not value for value in hashes) or len(set(hashes)) != 1:
            errors.append(_issue("same_pool_hash_mismatch", group, "M2, M3, and M4 pool hashes differ or are missing"))
            continue
        if all(isinstance(pool.get("candidates"), list) for pool in pools):
            from .schema import candidate_pool_sha256

            visible_hashes = [candidate_pool_sha256(pool["candidates"]) for pool in pools]
            if len(set(visible_hashes)) != 1:
                errors.append(_issue("same_pool_visible_mismatch", group, "M2, M3, and M4 selector-visible pools differ"))
            if set(hashes) != set(visible_hashes):
                errors.append(_issue("pool_hash_content_mismatch", group, "Declared pool hashes do not match selector-visible content"))
    if check_hashes:
        _validate_freeze_hashes(
            run_root,
            source_root=Path(source_root) if source_root is not None else None,
            expected_hashes=expected_hashes,
            errors=errors,
            require_complete=require_complete,
        )
        if require_complete:
            for filename in ("annotations.jsonl", "annotations_human_a.jsonl"):
                path = run_root / "freeze" / filename
                if not path.is_file() or len(read_jsonl(path)) != counts["groups_total"]:
                    errors.append(_issue("annotation_group_coverage", f"freeze/{filename}", "Final and primary annotations must cover every group"))
            secondary_path = run_root / "freeze" / "annotations_human_b.jsonl"
            expected_secondary = int((config_mapping.get("annotations") or {}).get("double_annotation_groups", min(112, counts["groups_total"])))
            if not secondary_path.is_file() or len(read_jsonl(secondary_path)) != expected_secondary:
                errors.append(_issue("annotation_control_coverage", "freeze/annotations_human_b.jsonl", "Independent secondary annotations have incorrect control coverage"))
    before = protected_hashes
    if before is None and (run_root / "freeze" / "benchmark_hashes_before.json").is_file():
        before = run_root / "freeze" / "benchmark_hashes_before.json"
    if before is not None:
        protected_report = validate_protected_hashes(
            before,
            protected_hashes_after,
            base_dir=protected_base_dir,
        )
        errors.extend(protected_report["errors"])
    elif require_complete:
        errors.append(_issue("protected_hashes_missing", "freeze/benchmark_hashes_before.json", "Protected benchmark hashes are missing"))
    if require_scored:
        _validate_scores(run_root, VARIANTS, counts, errors)
    for path in sorted(run_root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(run_root).as_posix()
        if path.is_symlink():
            errors.append(_issue("symlink_forbidden", relative, "Symlinks are not allowed"))
            continue
        if any(marker in relative.casefold() for marker in OLD_SNAPSHOT_MARKERS):
            errors.append(_issue("old_snapshot_forbidden", relative, "Legacy runs and snapshots are not allowed"))
        if path.is_file():
            if path.name.startswith(".env") or path.suffix == ".env":
                errors.append(_issue("env_file_forbidden", relative, "Environment files are not allowed"))
            for finding in scan_artifact(path, secret_values=secret_values):
                errors.append(_issue(f"artifact_{finding['kind']}", relative, "Artifact safety scan failed"))
    report = {
        "valid": not errors,
        "errors": errors,
        "error_count": len(errors),
        "expected_counts": counts,
        "variants": actual_variants,
        "same_pool_variants": list(SAME_POOL_VARIANTS),
        "scored_validation": require_scored,
    }
    if raise_on_error and errors:
        raise ValueError(f"Results pack validation failed with {len(errors)} errors")
    return report


def validate_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validate_results_pack(*args, **kwargs)


def canonical_relative_path(path: str | os.PathLike[str]) -> str:
    raw = str(path)
    pure = PurePosixPath(raw)
    if not raw or raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw) or "\\" in raw:
        raise ValueError("Archive paths must be canonical relative POSIX paths")
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or pure.as_posix() != raw:
        raise ValueError("Archive paths must be canonical relative POSIX paths")
    return pure.as_posix()


def create_results_pack(
    source_root: str | os.PathLike[str],
    destination_zip: str | os.PathLike[str],
    *,
    include_paths: Iterable[str | os.PathLike[str]] | None = None,
    secret_values: Mapping[str, str] | Iterable[str] | None = None,
    validate: bool = False,
    validation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(source_root)
    destination = Path(destination_zip)
    if not root.is_dir():
        raise ValueError("Pack source root is not a directory")
    if validate:
        report = validate_results_pack(root, **dict(validation_kwargs or {}))
        if not report["valid"]:
            raise ValueError(f"Cannot pack invalid run: {report['error_count']} validation errors")
    files = []
    if include_paths is None:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    else:
        candidates = [root / canonical_relative_path(path) for path in include_paths]
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Pack inputs must be regular files without symlinks")
        relative = canonical_relative_path(path.relative_to(root).as_posix())
        if destination.exists() and path.resolve() == destination.resolve():
            continue
        if any(marker in relative.casefold() for marker in OLD_SNAPSHOT_MARKERS):
            raise ValueError("Legacy snapshots are not allowed in results pack")
        findings = scan_artifact(path, secret_values=secret_values)
        if findings:
            kinds = sorted({finding["kind"] for finding in findings})
            raise ValueError(f"Artifact safety scan failed for {relative}: {kinds}")
        files.append((relative, path))
    files.sort(key=lambda item: item[0])
    if len({relative for relative, _ in files}) != len(files):
        raise ValueError("Duplicate archive paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, path in files:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "zip": destination.name,
        "files": len(files),
        "sha256": sha256_file(destination),
        "canonical_paths": True,
        "secrets_scanned": True,
    }


def pack_results(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return create_results_pack(*args, **kwargs)


def validate_zip_archive(
    archive_path: str | os.PathLike[str],
    *,
    secret_values: Mapping[str, str] | Iterable[str] | None = None,
) -> dict[str, Any]:
    errors = []
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = info.filename
            try:
                canonical_relative_path(name)
            except ValueError:
                errors.append(_issue("zip_path_unsafe", name, "Archive path is absolute or non-canonical"))
                continue
            if name in seen:
                errors.append(_issue("zip_duplicate_path", name, "Duplicate archive path"))
            seen.add(name)
            if PurePosixPath(name).name.startswith(".env") or PurePosixPath(name).suffix == ".env":
                errors.append(_issue("zip_env_file", name, "Environment files are not allowed"))
            if info.flag_bits & 0x1:
                errors.append(_issue("zip_encrypted", name, "Encrypted entries are not allowed"))
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                errors.append(_issue("zip_symlink", name, "Archive symlinks are not allowed"))
            if any(marker in name.casefold() for marker in OLD_SNAPSHOT_MARKERS):
                errors.append(_issue("zip_old_snapshot", name, "Legacy snapshots are not allowed"))
            data = archive.read(info)
            for finding in scan_artifact(data, secret_values=secret_values):
                errors.append(_issue(f"zip_{finding['kind']}", name, "Archive content safety scan failed"))
    return {"valid": not errors, "errors": errors, "files": len(seen)}
