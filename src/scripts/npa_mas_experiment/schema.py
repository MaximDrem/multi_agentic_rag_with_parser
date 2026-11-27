from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict

PROTOCOL_VERSION = "npa_mas_component_v1"
DEFAULT_SEED = 20261127
CUTOFFS = (5, 10)
VARIANTS = ("B0_BM25", "B0_DENSE", "B0_RRF", "B1", "B1W", "S", "M1", "M2", "M3", "M4")
SAME_POOL_VARIANTS = ("M2", "M3", "M4")
QUERY_LABELS = (
    "exact_reference",
    "lay_paraphrase",
    "multi_issue",
    "applicability_conditions",
    "cross_reference_multi_document",
    "other_unclear",
)
CONTRASTS = {"C1": ("M1", "S"), "C2": ("M2", "M1"), "C3": ("M3", "M2"), "C4": ("M4", "M3")}
PRACTICAL_CONTRASTS = {"M4-B1": ("M4", "B1"), "M4-B1W": ("M4", "B1W"), "M4-S": ("M4", "S")}
DEFAULT_EXPECTED_COUNTS = {
    "questions_total": 630,
    "groups_total": 556,
    "answerable_rows": 617,
    "answerable_groups": 544,
    "rows_without_qrels": 13,
}
SELECTOR_VISIBLE_FIELDS = ("pool_rank", "chunk_id", "selector_text", "doc_path", "title")
SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "proxy_authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer_token",
    "private_key",
    "credential",
    "credentials",
    "cookie",
    "set_cookie",
    "session_token",
    "auth_token",
}
SECRET_PATTERNS = {
    "bearer_credential": re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    "openai_like_key": re.compile(r"\b(?:sk|pk)-[a-z0-9_-]{12,}\b", re.IGNORECASE),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{16,}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"),
}
ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:/|file://|[A-Za-z]:[\\/]|\\\\)")
ENV_ASSIGNMENT_PATTERN = re.compile(r"(?m)^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
OLD_SNAPSHOT_MARKERS = (
    "coordinated_mechanism_v1",
    "run_coordinated_experiment",
    "coordinated_experiment",
    "legacy_snapshot",
)


class Question(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    question_id: str
    source_row: int
    query: str


class Qrel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    question_id: str
    chunk_id: str
    relevance: int


class QueryGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_group: str
    normalized_query: str
    representative_question_id: str
    representative_source_row: int
    query: str
    question_ids: tuple[str, ...]
    source_rows: tuple[int, ...]
    relevant: dict[str, int]

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    questions: tuple[Question, ...]
    groups: tuple[QueryGroup, ...]
    qrels: dict[str, dict[str, int]]

    @property
    def question_by_id(self) -> dict[str, Question]:
        return {question.question_id: question for question in self.questions}

    @property
    def group_by_id(self) -> dict[str, QueryGroup]:
        return {group.query_group: group for group in self.groups}


def normalize_query(text: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", text.casefold()).strip()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_compatible(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_compatible(item) for item in value), key=lambda item: canonical_json(item))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_compatible(value.model_dump(mode="json"))
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_compatible(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _json_compatible(value.item())
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(value)
    return rows


def read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping")
    return value


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    overwrite: bool = True,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(str(destination))
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    overwrite: bool = True,
    trailing_newline: bool = True,
) -> Path:
    data = canonical_json_bytes(value) + (b"\n" if trailing_newline else b"")
    return atomic_write_bytes(path, data, overwrite=overwrite)


def atomic_write_jsonl(
    path: str | os.PathLike[str],
    rows: Iterable[Any],
    *,
    overwrite: bool = True,
) -> Path:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return atomic_write_bytes(path, data, overwrite=overwrite)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | os.PathLike[str]) -> str:
    root = Path(path)
    if not root.is_dir():
        raise ValueError("Directory hash input is not a directory")
    entries = []
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise ValueError("Symlinks are not allowed in hashed directories")
        if item.is_file():
            entries.append({"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)})
    return sha256_canonical(entries)


def verify_file_hashes(
    files: Mapping[str, str | os.PathLike[str]],
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    errors = []
    actual = {}
    for name in sorted(set(files) | set(expected_hashes)):
        if name not in files:
            errors.append({"name": name, "error": "missing_path"})
            continue
        if name not in expected_hashes:
            errors.append({"name": name, "error": "missing_expected_hash"})
            continue
        path = Path(files[name])
        if not path.is_file():
            errors.append({"name": name, "error": "missing_file"})
            continue
        actual[name] = sha256_file(path)
        if actual[name] != expected_hashes[name]:
            errors.append({"name": name, "error": "hash_mismatch"})
    return {"valid": not errors, "actual_hashes": actual, "errors": errors}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Expected a mapping")


def load_questions(questions: Any) -> list[Question]:
    if isinstance(questions, (str, os.PathLike)):
        raw_rows: Any = read_jsonl(questions)
    elif isinstance(questions, Benchmark):
        return list(questions.questions)
    elif isinstance(questions, Mapping):
        raw_rows = []
        for question_id, value in questions.items():
            if isinstance(value, str):
                match = re.search(r"(\d+)$", question_id)
                if match is None:
                    raise ValueError("source_row is required when it cannot be derived")
                raw_rows.append({"question_id": question_id, "source_row": int(match.group(1)), "query": value})
            else:
                raw_rows.append({"question_id": question_id, **_mapping(value)})
    else:
        raw_rows = list(questions)
    result = []
    seen = set()
    for raw in raw_rows:
        question = raw if isinstance(raw, Question) else Question.model_validate(raw)
        if question.question_id in seen:
            raise ValueError(f"Duplicate question_id: {question.question_id}")
        if not question.query.strip():
            raise ValueError(f"Empty query: {question.question_id}")
        seen.add(question.question_id)
        result.append(question)
    return sorted(result, key=lambda item: (item.source_row, item.question_id))


def load_qrels(qrels: Any) -> dict[str, dict[str, int]]:
    if isinstance(qrels, (str, os.PathLike)):
        raw_rows: Any = read_jsonl(qrels)
    elif isinstance(qrels, Benchmark):
        return {qid: dict(values) for qid, values in qrels.qrels.items()}
    elif isinstance(qrels, Mapping):
        raw_rows = []
        for question_id, values in qrels.items():
            if isinstance(values, Mapping):
                for chunk_id, relevance in values.items():
                    raw_rows.append({"question_id": question_id, "chunk_id": chunk_id, "relevance": relevance})
            else:
                for value in values:
                    raw_rows.append({"question_id": question_id, **_mapping(value)})
    else:
        raw_rows = list(qrels)
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for raw in raw_rows:
        qrel = raw if isinstance(raw, Qrel) else Qrel.model_validate(raw)
        if qrel.relevance <= 0:
            continue
        previous = result[qrel.question_id].get(qrel.chunk_id, 0)
        result[qrel.question_id][qrel.chunk_id] = max(previous, qrel.relevance)
    return {qid: dict(values) for qid, values in result.items() if values}


def union_qrels(question_ids: Iterable[str], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    relevant: dict[str, int] = {}
    for question_id in question_ids:
        for chunk_id, relevance in qrels.get(question_id, {}).items():
            if relevance > 0:
                relevant[chunk_id] = max(relevant.get(chunk_id, 0), int(relevance))
    return relevant


def build_benchmark(questions: Any, qrels: Any) -> Benchmark:
    loaded_questions = load_questions(questions)
    loaded_qrels = load_qrels(qrels)
    known_questions = {question.question_id for question in loaded_questions}
    unknown = sorted(set(loaded_qrels) - known_questions)
    if unknown:
        raise ValueError(f"Unknown question IDs in qrels: {unknown[:5]}")
    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in loaded_questions:
        normalized = normalize_query(question.query)
        if not normalized:
            raise ValueError(f"Query normalizes to empty: {question.question_id}")
        grouped[normalized].append(question)
    groups = []
    for index, normalized in enumerate(sorted(grouped)):
        members = sorted(grouped[normalized], key=lambda item: (item.source_row, item.question_id))
        representative = members[0]
        question_ids = tuple(question.question_id for question in members)
        groups.append(
            QueryGroup(
                query_group=f"group_{index:04d}",
                normalized_query=normalized,
                representative_question_id=representative.question_id,
                representative_source_row=representative.source_row,
                query=representative.query,
                question_ids=question_ids,
                source_rows=tuple(question.source_row for question in members),
                relevant=union_qrels(question_ids, loaded_qrels),
            )
        )
    return Benchmark(questions=tuple(loaded_questions), groups=tuple(groups), qrels=loaded_qrels)


def load_benchmark(
    questions_path: str | os.PathLike[str],
    qrels_path: str | os.PathLike[str],
    *,
    config: Mapping[str, Any] | None = None,
    strict_audit: bool = False,
) -> Benchmark:
    benchmark = build_benchmark(questions_path, qrels_path)
    if strict_audit:
        audit_benchmark(benchmark, config=config, strict=True)
    return benchmark


def _expected_counts(
    config: Mapping[str, Any] | None,
    expected_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    if expected_counts is not None:
        return dict(expected_counts)
    if config is None:
        return dict(DEFAULT_EXPECTED_COUNTS)
    benchmark_config = config.get("benchmark", config)
    aliases = {
        "questions_total": "expected_rows",
        "groups_total": "expected_groups",
        "answerable_rows": "expected_answerable_rows",
        "answerable_groups": "expected_answerable_groups",
        "rows_without_qrels": "expected_rows_without_qrels",
    }
    return {
        actual: int(benchmark_config[expected])
        for actual, expected in aliases.items()
        if expected in benchmark_config
    }


def audit_benchmark(
    questions: Any,
    qrels: Any = None,
    *,
    config: Mapping[str, Any] | None = None,
    expected_counts: Mapping[str, int] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    benchmark = questions if isinstance(questions, Benchmark) else build_benchmark(questions, qrels)
    answerable_groups = sum(group.answerable for group in benchmark.groups)
    actual = {
        "questions_total": len(benchmark.questions),
        "groups_total": len(benchmark.groups),
        "answerable_rows": sum(qid in benchmark.qrels and bool(benchmark.qrels[qid]) for qid in benchmark.question_by_id),
        "answerable_groups": answerable_groups,
        "rows_without_qrels": len(benchmark.questions)
        - sum(qid in benchmark.qrels and bool(benchmark.qrels[qid]) for qid in benchmark.question_by_id),
    }
    expected = _expected_counts(config, expected_counts)
    mismatches = [
        {"check": name, "expected": expected[name], "actual": value}
        for name, value in actual.items()
        if name in expected and expected[name] != value
    ]
    report = {
        "valid": not mismatches,
        "actual": actual,
        "expected": expected,
        "mismatches": mismatches,
        "normalization": "casefold_non_alphanumeric_to_space",
        "representative_policy": "min_source_row",
        "group_qrels_policy": "union_max_relevance",
    }
    if strict and mismatches:
        raise ValueError(f"Benchmark audit failed: {mismatches}")
    return report


def ranking_ids(value: Any) -> list[str]:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        raw = None
        for key in ("ranked_chunk_ids", "final_ranked_ids", "ranking", "chunks"):
            if key in value:
                raw = value[key]
                break
        if raw is None:
            raise ValueError("Ranking field is missing")
    else:
        raw = value
    if raw is None:
        raise ValueError("Ranking cannot be null; use an explicit empty list")
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, BaseModel):
            result.append(str(item.model_dump()["chunk_id"]))
        elif isinstance(item, Mapping):
            result.append(str(item["chunk_id"]))
        else:
            result.append(str(item.chunk_id))
    return result


def deduplicate_ranking(
    ranking: Any,
    *,
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
    max_items: int | None = 10,
    reject_duplicates: bool = False,
) -> list[str]:
    result = []
    seen = set()
    for chunk_id in ranking_ids(ranking):
        if not chunk_id:
            raise ValueError("Empty chunk_id in ranking")
        if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
            raise ValueError(f"Unknown chunk_id: {chunk_id}")
        if chunk_id in seen:
            if reject_duplicates:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            continue
        seen.add(chunk_id)
        result.append(chunk_id)
        if max_items is not None and len(result) >= max_items:
            break
    return result


def project_rankings(
    benchmark: Benchmark,
    rankings: Mapping[str, Any],
    *,
    valid_chunk_ids: set[str] | frozenset[str] | None = None,
    require_all_groups: bool = True,
) -> list[dict[str, Any]]:
    missing = sorted(set(benchmark.group_by_id) - set(rankings))
    if require_all_groups and missing:
        raise ValueError(f"Missing group rankings: {missing[:5]}")
    unknown = sorted(set(rankings) - set(benchmark.group_by_id))
    if unknown:
        raise ValueError(f"Unknown query groups: {unknown[:5]}")
    rows = []
    for group in benchmark.groups:
        if group.query_group not in rankings:
            continue
        ranking = deduplicate_ranking(rankings[group.query_group], valid_chunk_ids=valid_chunk_ids, max_items=10)
        for question_id, source_row in zip(group.question_ids, group.source_rows):
            rows.append(
                {
                    "question_id": question_id,
                    "source_row": source_row,
                    "query": benchmark.question_by_id[question_id].query,
                    "query_group": group.query_group,
                    "ranked_chunk_ids": list(ranking),
                }
            )
    return rows


def selector_visible_pool(candidates: Sequence[Any]) -> list[dict[str, Any]]:
    visible = []
    for index, raw in enumerate(candidates, 1):
        candidate = _mapping(raw)
        visible.append(
            {
                "pool_rank": int(candidate.get("pool_rank", index)),
                "chunk_id": str(candidate["chunk_id"]),
                "selector_text": str(candidate.get("selector_text", "")),
                "doc_path": str(candidate.get("doc_path", "")),
                "title": str(candidate.get("title", "")),
            }
        )
    return visible


def candidate_pool_sha256(candidates: Sequence[Any]) -> str:
    return sha256_canonical(selector_visible_pool(candidates))


def secret_values_from_environ(environ: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in environ.items() if value and _is_sensitive_key(key)}


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return (
        normalized in SENSITIVE_KEY_NAMES
        or normalized.endswith(("_api_key", "_secret", "_password", "_private_key", "_access_token", "_refresh_token", "_access_key"))
    )


def _is_redacted(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in {"", "<redacted>", "redacted", "***", "null"}


def scan_artifact(
    artifact: Any,
    *,
    secret_values: Mapping[str, str] | Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    findings: set[tuple[str, str]] = set()
    source_name = "artifact"
    value = artifact
    artifact_path = None
    if isinstance(artifact, os.PathLike):
        artifact_path = Path(artifact)
    elif isinstance(artifact, str) and "\n" not in artifact and "\x00" not in artifact and len(artifact) < 4096:
        try:
            candidate_path = Path(artifact)
            artifact_path = candidate_path if candidate_path.is_file() else None
        except OSError:
            artifact_path = None
    if artifact_path is not None:
        path = artifact_path
        source_name = path.name
        value = path.read_bytes().decode("utf-8", errors="replace")
        if path.name.startswith(".env") or path.suffix == ".env":
            findings.add(("env_file", source_name))
        if path.suffix == ".json":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif path.suffix == ".jsonl":
            try:
                value = [json.loads(line) for line in value.splitlines() if line.strip()]
            except json.JSONDecodeError:
                pass
    if isinstance(value, (str, bytes)):
        raw_text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        if raw_text.lstrip().startswith(("{", "[")):
            try:
                value = json.loads(raw_text)
            except json.JSONDecodeError:
                pass
    secrets = secret_values_from_environ(environ or {})
    if secret_values is not None:
        if isinstance(secret_values, Mapping):
            secrets.update({str(key): item for key, item in secret_values.items() if isinstance(item, str)})
        else:
            secrets.update({f"secret_{index}": item for index, item in enumerate(secret_values) if isinstance(item, str)})
    secrets = {key: item for key, item in secrets.items() if len(item) >= 4}

    def walk(item: Any, location: str) -> None:
        if isinstance(item, BaseModel):
            walk(item.model_dump(mode="json"), location)
        elif isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_location = f"{location}.{key}"
                if _is_sensitive_key(key) and not _is_redacted(child):
                    findings.add(("sensitive_key", child_location))
                if key.casefold() in {"env", "environ", "environment", "environment_variables"} and isinstance(child, Mapping) and child:
                    findings.add(("environment_dump", child_location))
                walk(child, child_location)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")
        elif isinstance(item, (str, bytes, Path)):
            text = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            for key, secret in secrets.items():
                if secret in text:
                    findings.add(("raw_secret", f"{location}:{key}"))
            for kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.add((kind, location))
            for match in ENV_ASSIGNMENT_PATTERN.finditer(text):
                variable = match.group(1)
                if _is_sensitive_key(variable) or variable.startswith(("OPENAI_", "ANTHROPIC_", "NPA_LLM_", "JINA_")) or source_name.startswith(".env"):
                    findings.add(("environment_assignment", f"{location}:{variable}"))
            if ABSOLUTE_PATH_PATTERN.match(text.strip()):
                findings.add(("absolute_path", location))

    walk(value, source_name)
    return [
        {"kind": kind, "location": location}
        for kind, location in sorted(findings)
    ]


def redact_artifact(
    artifact: Any,
    *,
    secret_values: Mapping[str, str] | Iterable[str] | None = None,
) -> Any:
    if isinstance(secret_values, Mapping):
        secrets = [value for value in secret_values.values() if isinstance(value, str)]
    else:
        secrets = [value for value in (secret_values or []) if isinstance(value, str)]
    secrets = [value for value in secrets if len(value) >= 4]

    def redact_text(text: str) -> str:
        result = text
        for secret in secrets:
            result = result.replace(secret, "<redacted>")
        for pattern in SECRET_PATTERNS.values():
            result = pattern.sub("<redacted>", result)
        return result

    def walk(item: Any, key: str = "") -> Any:
        if isinstance(item, BaseModel):
            return walk(item.model_dump(mode="json"), key)
        if isinstance(item, Mapping):
            return {
                str(child_key): "<redacted>"
                if _is_sensitive_key(str(child_key)) and not _is_redacted(child)
                else walk(child, str(child_key))
                for child_key, child in item.items()
            }
        if isinstance(item, list):
            return [walk(child, key) for child in item]
        if isinstance(item, tuple):
            return tuple(walk(child, key) for child in item)
        if isinstance(item, str):
            return redact_text(item)
        return item

    return walk(artifact)


def assert_artifact_clean(artifact: Any, *, secret_values: Any = None) -> None:
    findings = scan_artifact(artifact, secret_values=secret_values)
    if findings:
        kinds = sorted({finding["kind"] for finding in findings})
        raise ValueError(f"Artifact safety scan failed: {kinds}")
