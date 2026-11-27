import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.coordinated_experiment.analysis import (
    CONTRAST_REGISTRY,
    _assert_contrast_pools,
    _resolve_config_id,
    collect_selector_results,
    expected_gated_configs,
)
from scripts.coordinated_experiment.legacy_reuse import load_snapshots
from scripts.coordinated_experiment.schema import atomic_write_json, canonical_sha256

PACK_FILES = (
    "README.md",
    "protocol.yaml",
    "manifest.json",
    "validation/reuse_audit.json",
    "validation/index_replay.json",
    "validation/test_results.json",
    "validation/run_integrity.json",
    "validation/smoke_report.json",
    "validation/protocol_deviations.jsonl",
    "configs/main_results.csv",
    "configs/pool_metrics.csv",
    "configs/costs.csv",
    "per_group/pool_metrics.jsonl",
    "per_group/selector_results.jsonl",
    "per_group/quality_metrics.jsonl",
    "per_group/costs.jsonl",
    "analysis/paired_contrasts.csv",
    "analysis/paired_contrasts.json",
    "analysis/decision_pre_stability.json",
    "analysis/stability_target.json",
    "analysis/stability.csv",
    "analysis/failures_and_sensitivity.csv",
    "analysis/decision.json",
    "analysis/decision.md",
)
REQUIRED_COMPLETE_FILES = tuple(name for name in PACK_FILES if name not in {"README.md", "protocol.yaml", "manifest.json"})
SELECTOR_FIELDS = (
    "config_id", "group_id", "query_group", "question_ids", "query", "pool_id", "pool_sha256",
    "input_snapshot_sha256", "ranked_chunk_ids", "fallback_stage", "fallback_stages", "fallback_used",
    "repair_used", "prompt_tokens", "completion_tokens", "total_tokens", "token_measurement_status",
    "selector_latency_ms", "jina_latency_ms", "pool_size", "output_size", "execution_batch",
    "dropped_unknown_aliases", "dropped_duplicate_aliases", "error", "errors", "output_policy",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(?<![a-z0-9_])[\"']?(?:(?:OPENAI|ANTHROPIC|JINA|KILO|HF|HUGGINGFACE)_)?API[_-]?KEY[\"']?\s*[:=]\s*[\"']?[^\s,}\"']+"),
    re.compile(r"(?i)(?<![a-z0-9_])[\"']?(?:password|passwd|secret|access[_-]?token|refresh[_-]?token|authorization|private[_-]?key)[\"']?\s*[:=]\s*[\"']?[^\s,}\"']+"),
    re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[a-z0-9_\-.=+/]{16,}"),
    re.compile(r"\b(?:sk-[a-z0-9_\-]{16,}|ghp_[a-z0-9]{20,}|xox[baprs]-[a-z0-9-]{12,}|AIza[a-z0-9_\-]{20,})\b"),
)
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:file://|(?<![a-z0-9_.:-])/(?:home|users|root|tmp|var|etc|mnt|media|opt|private|workspace|workspaces|srv|data|volumes)(?:/|\\)|\b[a-z]:[\\/][^/\s])"
)
RAW_BULK_PATTERN = re.compile(r"(?i)[\"'](?:raw_responses|raw_model_response|system_prompt|human_prompt|chunk_texts)[\"']\s*:")


def _safe_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scan_text(name: str, text: str, experiment_dir: Path) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Secret-like content rejected in lightweight pack: {name}")
    if RAW_BULK_PATTERN.search(text):
        raise ValueError(f"Raw model bulk rejected in lightweight pack: {name}")
    if ABSOLUTE_PATH_PATTERN.search(text):
        raise ValueError(f"Absolute local path rejected in lightweight pack: {name}")
    for local in (str(experiment_dir), str(experiment_dir.parent), str(Path.home())):
        if local and local != "/" and local in text:
            raise ValueError(f"Absolute local path rejected in lightweight pack: {name}")


def _selector_public(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in SELECTOR_FIELDS if key in record and not key.startswith("_")}


def _public_selector_rows(collected: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for config_id in sorted(collected):
        for group_id in sorted(collected[config_id]):
            rows.append(_selector_public(collected[config_id][group_id]))
    return rows


def _load_protocol(experiment_dir: Path) -> tuple[dict[str, Any], str]:
    for name in ("protocol.yaml", "config.yaml", "experiment_config.yaml"):
        path = experiment_dir / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            value = yaml.safe_load(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"Protocol config must be a mapping: {name}")
            return dict(value), text
    return {"protocol_version": "coordinated_mechanism_v1"}, "protocol_version: coordinated_mechanism_v1\n"


def _protocol_deviations(experiment_dir: Path) -> int:
    path = experiment_dir / "validation" / "protocol_deviations.jsonl"
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def validate_run_integrity(experiment_dir: str | os.PathLike[str], write: bool = True) -> dict[str, Any]:
    root = Path(experiment_dir)
    errors: list[dict[str, str]] = []
    snapshots = []
    collected: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        snapshots = load_snapshots(root)
    except Exception as exc:
        errors.append({"code": "snapshot_integrity", "message": f"{type(exc).__name__}: {exc}"})
    if snapshots:
        try:
            collected = collect_selector_results(root, snapshots)
        except Exception as exc:
            errors.append({"code": "selector_integrity", "message": f"{type(exc).__name__}: {exc}"})
            collected = {}
    all_groups = {str(snapshot["group_id"]) for snapshot in snapshots}
    missing_groups: dict[str, list[str]] = {}
    duplicate_question_rows: dict[str, list[str]] = {}
    counts: dict[str, dict[str, Any]] = {}
    for config_id in sorted(collected):
        records = collected[config_id]
        missing = sorted(all_groups - set(records))
        if missing:
            missing_groups[config_id] = missing
        seen: set[str] = set()
        duplicates: set[str] = set()
        for record in records.values():
            for qid in record.get("question_ids", []):
                if qid in seen:
                    duplicates.add(str(qid))
                seen.add(str(qid))
        if duplicates:
            duplicate_question_rows[config_id] = sorted(duplicates)
            errors.append({"code": "duplicate_question_rows", "message": f"{config_id}: {sorted(duplicates)[:5]}"})
        counts[config_id] = {
            "groups": len(records),
            "rows": len(seen),
            "errors": sum(bool(record.get("error") or record.get("errors")) for record in records.values()),
            "fallback_groups": sum(bool(record.get("fallback_stage") or record.get("fallback_stages") or record.get("fallback_used")) for record in records.values()),
            "repair_groups": sum(bool(record.get("repair_used")) for record in records.values()),
            "execution_batches": sorted({str(record.get("execution_batch")) for record in records.values()}),
        }
    checked_contrasts = []
    if collected:
        for contrast_id, spec in CONTRAST_REGISTRY.items():
            if not spec["same_pool"]:
                continue
            for policy in ("strict", "common_backfill"):
                left = _resolve_config_id(spec["left"], collected, policy)
                right = _resolve_config_id(spec["right"], collected, policy)
                if left and right:
                    try:
                        _assert_contrast_pools(contrast_id, left, right, collected)
                        checked_contrasts.append(f"{contrast_id}:{policy}")
                    except Exception as exc:
                        errors.append({"code": "same_pool_mismatch", "message": str(exc)})
    for snapshot in snapshots:
        wide = snapshot.get("pools", {}).get("W")
        if not isinstance(wide, Mapping):
            continue
        expanded_counts = snapshot.get("raw_hit_counts", {})
        wide_counts = wide.get("raw_hit_counts", {})
        if not isinstance(expanded_counts, Mapping) or not isinstance(wide_counts, Mapping):
            continue
        for channel in ("dense", "bm25"):
            expected_channel = expanded_counts.get(channel, {})
            actual_channel = wide_counts.get(channel, {})
            expected_total = expected_channel.get("total") if isinstance(expected_channel, Mapping) else expected_channel
            actual_total = actual_channel.get("total") if isinstance(actual_channel, Mapping) else actual_channel
            if expected_total is not None and actual_total is not None and int(expected_total) != int(actual_total):
                errors.append({"code": "wide_budget_mismatch", "message": f"{snapshot['group_id']}:{channel} E={expected_total} W={actual_total}"})
    for name in ("reuse_audit.json", "index_replay.json"):
        value = _read_json(root / "validation" / name)
        if isinstance(value, Mapping) and (value.get("valid") is False or value.get("status") == "error"):
            errors.append({"code": "validation_failed", "message": f"validation/{name} reports failure"})
    expected, stage_status = expected_gated_configs(root, list(collected))
    missing_configs = sorted(expected - set(collected))
    protocol, _ = _load_protocol(root)
    benchmark = protocol.get("benchmark", {})
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    expected_groups = int(benchmark.get("expected_groups", 556))
    expected_rows = int(benchmark.get("expected_rows", 630))
    denominator_errors = []
    if snapshots and len(snapshots) != expected_groups:
        denominator_errors.append(f"snapshots={len(snapshots)} expected={expected_groups}")
    for config_id, count in counts.items():
        if not missing_groups.get(config_id) and count["groups"] == expected_groups and count["rows"] != expected_rows:
            denominator_errors.append(f"{config_id}: rows={count['rows']} expected={expected_rows}")
    if denominator_errors:
        errors.append({"code": "denominator_mismatch", "message": "; ".join(denominator_errors[:10])})
    deviations = _protocol_deviations(root)
    complete = bool(collected) and not errors and not missing_configs and not missing_groups and deviations == 0
    report = {
        "protocol_version": "coordinated_mechanism_v1",
        "valid": not errors,
        "complete": complete,
        "protocol_complete": complete and deviations == 0,
        "groups": len(snapshots),
        "configs": counts,
        "missing_configs": missing_configs,
        "missing_groups": missing_groups,
        "duplicate_question_rows": duplicate_question_rows,
        "same_pool_contrasts_checked": checked_contrasts,
        "stage_status": stage_status,
        "protocol_deviations": deviations,
        "errors": errors,
    }
    if write:
        atomic_write_json(root / "validation" / "run_integrity.json", report)
    return report


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    scopes = ["src/scripts/coordinated_experiment", "experiments/coordinated_mechanism_v1.yaml", "tests/coordinated_experiment"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD", "--", *scopes], cwd=root, check=True, capture_output=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *scopes], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
        digest = hashlib.sha256(diff)
        code_hashes = {}
        for name in sorted(untracked):
            path = root / name
            if path.is_file():
                content = path.read_bytes()
                digest.update(name.encode("utf-8") + b"\0" + content)
                code_hashes[name] = hashlib.sha256(content).hexdigest()
        return {"git_commit": commit, "git_diff_sha256": digest.hexdigest(), "untracked_code_sha256": code_hashes}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_diff_sha256": None, "untracked_code_sha256": {}}


def _without_paths(value: Any, key: str = "") -> Any:
    secret_keys = {"api_key", "apikey", "password", "secret", "access_token", "refresh_token", "authorization", "private_key"}
    if key.casefold() in secret_keys:
        return None
    if isinstance(value, Mapping):
        return {str(k): _without_paths(v, str(k)) for k, v in value.items() if _without_paths(v, str(k)) is not None}
    if isinstance(value, list):
        return [_without_paths(item, key) for item in value]
    if isinstance(value, str):
        if ABSOLUTE_PATH_PATTERN.search(value):
            return "[removed-local-path]"
        if key in {"path", "raw_model_response_path"} or key.endswith("_dir"):
            return Path(value).name
    return value


def _model_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("model", "models", "llm", "jina", "jina_model", "generation", "retrieval")
    return _without_paths({key: config[key] for key in keys if key in config})


def _build_manifest(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_text: str,
    integrity: Mapping[str, Any],
    collected: Mapping[str, Mapping[str, Mapping[str, Any]]],
    missing_pack_files: Sequence[str],
) -> dict[str, Any]:
    reuse = _read_json(root / "validation" / "reuse_audit.json") or {}
    replay = _read_json(root / "validation" / "index_replay.json") or {}
    stage = _read_json(root / "stage_status.json") or _read_json(root / "status.json") or {}
    batches = sorted({str(record.get("execution_batch")) for records in collected.values() for record in records.values()})
    git = _git_metadata()
    return {
        "protocol_version": protocol.get("protocol_version", "coordinated_mechanism_v1"),
        **git,
        "config_sha256": canonical_sha256(protocol),
        "protocol_file_sha256": hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
        "benchmark_hashes": _without_paths(reuse.get("benchmark_hashes", {})),
        "legacy_run_sha256": reuse.get("legacy_run_sha256"),
        "model_configuration": _model_metadata(protocol),
        "index_and_chunk_store_hashes": _without_paths({
            "chunks_pkl_sha256": reuse.get("chunks_pkl_sha256"),
            "index_replay": {key: replay.get(key) for key in ("index_hashes", "indices", "chunk_store", "chunks_pkl_sha256") if key in replay},
        }),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamps": _without_paths({key: stage.get(key) for key in ("started_at", "completed_at", "updated_at") if isinstance(stage, Mapping) and key in stage}),
        "execution_batches": batches,
        "stages": {
            "completed": integrity.get("stage_status", {}).get("completed_stages", []),
            "gate_triggered": integrity.get("stage_status", {}).get("gate_triggered_stages", []),
            "raw_status": _without_paths(stage),
        },
        "configs": integrity.get("configs", {}),
        "errors": integrity.get("errors", []),
        "fallback_groups": sum(value.get("fallback_groups", 0) for value in integrity.get("configs", {}).values()),
        "repair_groups": sum(value.get("repair_groups", 0) for value in integrity.get("configs", {}).values()),
        "protocol_deviations": integrity.get("protocol_deviations", 0),
        "secrets_present": False,
        "secret_and_path_scan": "passed",
        "raw_model_bulk_included": False,
        "integrity_valid": integrity.get("valid", False),
        "missing_pack_files": list(missing_pack_files),
        "complete": bool(integrity.get("complete")) and not missing_pack_files,
    }


def _source_paths(root: Path) -> dict[str, Path | None]:
    sources: dict[str, Path | None] = {name: root / name for name in PACK_FILES}
    sources["protocol.yaml"] = next((root / name for name in ("protocol.yaml", "config.yaml", "experiment_config.yaml") if (root / name).is_file()), None)
    sources["README.md"] = root / "README.md" if (root / "README.md").is_file() else None
    sources["manifest.json"] = None
    return sources


def pack_experiment(experiment_dir: str | os.PathLike[str], output_zip: str | os.PathLike[str]) -> dict[str, Any]:
    root, output = Path(experiment_dir), Path(output_zip)
    if not root.is_dir():
        raise ValueError("experiment_dir must be a directory")
    protocol, protocol_text = _load_protocol(root)
    preliminary_sources = _source_paths(root)
    for name, path in preliminary_sources.items():
        if name in {"manifest.json", "validation/run_integrity.json", "per_group/selector_results.jsonl"} or path is None or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Pack files must be UTF-8: {name}") from exc
        _scan_text(name, text, root)
    _scan_text("protocol.yaml", protocol_text, root)
    integrity = validate_run_integrity(root)
    if not integrity["valid"]:
        raise ValueError(f"Run integrity validation failed: {integrity['errors'][:3]}")
    snapshots = load_snapshots(root)
    collected = collect_selector_results(root, snapshots)
    selector_text = "".join(_safe_text(row) + "\n" for row in _public_selector_rows(collected))
    sources = _source_paths(root)
    generated: dict[str, str] = {
        "protocol.yaml": protocol_text,
        "per_group/selector_results.jsonl": selector_text,
        "validation/run_integrity.json": json.dumps(integrity, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
    }
    if sources["README.md"] is None:
        generated["README.md"] = "# Coordinated mechanism v1\n\nОблегчённый пакет содержит конфигурации, per-group rankings, метрики, проверки целостности и машинное решение. Raw model responses, секреты и локальные абсолютные пути исключены.\n"
    missing = []
    payloads: dict[str, bytes] = {}
    for name in PACK_FILES:
        if name in {"manifest.json", "validation/run_integrity.json"}:
            continue
        if name in generated:
            text = generated[name]
        else:
            path = sources.get(name)
            if path is None or not path.is_file():
                missing.append(name)
                continue
            if any(part == ".env" or part.startswith(".env.") for part in path.parts):
                raise ValueError(f".env file rejected: {name}")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Pack files must be UTF-8: {name}") from exc
        _scan_text(name, text, root)
        payloads[name] = text.encode("utf-8")
    payloads["validation/run_integrity.json"] = generated["validation/run_integrity.json"].encode("utf-8")
    manifest = _build_manifest(root, protocol, protocol_text, integrity, collected, missing)
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _scan_text("manifest.json", manifest_text, root)
    payloads["manifest.json"] = manifest_text.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in PACK_FILES:
                if name in payloads:
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, payloads[name])
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP integrity failure: {bad}")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in archive.namelist()):
                raise ValueError("ZIP contains a non-relative or unsafe path")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return manifest


__all__ = ["PACK_FILES", "pack_experiment", "validate_run_integrity"]
