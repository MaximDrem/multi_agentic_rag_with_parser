import asyncio
import copy
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.coordinated_experiment.generation import (
    PLANNER_SYSTEM_PROMPT,
    generate_coordinated,
    generate_no_planner,
    generate_single,
    generic_plan,
    normalize_query_text,
)
from scripts.coordinated_experiment.legacy_reuse import (
    load_snapshots,
    select_smoke_groups,
    select_stability_groups,
)
from scripts.coordinated_experiment.pools import (
    apply_backfill,
    assert_same_pool,
    build_pool_record,
    multi_event_rrf,
    worker_rrf,
)
from scripts.coordinated_experiment.schema import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_config,
    snapshot_hash,
    text_sha256,
)
from scripts.coordinated_experiment.selectors import (
    ReplayLLMCaller,
    arbiter_validator,
    build_arbiter_prompts,
    build_worker_prompts,
    make_worker_validator,
    prompt_sha256,
    reconstruct_pool,
)

CHANNELS = ("dense", "bm25")
DERIVED_POLICIES = ("strict", "common_backfill")
CACHE_SCHEMA_VERSION = 1


class IndexReplayError(ValueError):
    def __init__(self, report: Mapping[str, Any], report_path: Path):
        self.report = dict(report)
        self.report_path = report_path
        super().__init__(f"Index replay failed: {report_path}")


class CacheIntegrityError(ValueError):
    pass


def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk.get("id", chunk.get("chunk_id")))
    return str(getattr(chunk, "id", getattr(chunk, "chunk_id", None)))


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk["text"])
    return str(chunk.text)


def _chunk_value(chunk: Any, key: str, default: Any = None) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _hit_parts(hit: Any, implicit_rank: int) -> tuple[Any, str, int, float | None]:
    if isinstance(hit, Mapping):
        chunk = hit.get("chunk", hit)
        if isinstance(chunk, Mapping):
            chunk_id = str(hit.get("chunk_id", chunk.get("id", chunk.get("chunk_id"))))
        else:
            chunk_id = str(hit.get("chunk_id", _chunk_id(chunk)))
        rank = int(hit.get("rank") or implicit_rank)
        score = hit.get("score")
        return chunk, chunk_id, rank, float(score) if score is not None else None
    chunk = hit.chunk
    score = getattr(hit, "score", None)
    return chunk, _chunk_id(chunk), int(getattr(hit, "rank", implicit_rank)), float(score) if score is not None else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ExperimentRunner:
    def __init__(
        self,
        config_path: str | os.PathLike[str],
        experiment_dir: str | os.PathLike[str],
        concurrency: int = 2,
        execution_batch: str = "main",
        smoke_only: bool = False,
    ):
        if isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", execution_batch):
            raise ValueError("execution_batch contains unsafe characters")
        self.config_path = Path(config_path)
        self.experiment_dir = Path(experiment_dir)
        self.config = load_config(config_path)
        self.concurrency = int(concurrency)
        self.smoke_only = bool(smoke_only)
        self.execution_batch = "smoke" if self.smoke_only and execution_batch == "main" else execution_batch
        protocol = str(self.config.get("protocol_version", "coordinated_mechanism_v1"))
        if protocol != "coordinated_mechanism_v1":
            raise ValueError(f"Unsupported protocol: {protocol}")
        self.protocol_version = protocol
        self.engine: Any = None
        self.reranker: Any = None
        self.llm: Any = None
        self.dense_llm: Any = None
        self.bm25_llm: Any = None
        self.planner_llm: Any = None
        self.arbiter_llm: Any = None
        self.chunk_map: dict[str, Any] | None = None
        self.chunks: list[Any] | None = None
        self._snapshots: list[dict[str, Any]] | None = None
        self._questions: dict[str, str] | None = None
        self._qrels: dict[str, dict[str, int]] | None = None
        self._benchmark_hashes: dict[str, Any] | None = None
        self._llm_cache: dict[str, Any] = {}
        self._rerank_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._reranker_config: dict[str, Any] | None = None
        execution = self.config.get("execution", {})
        self.timeout_seconds = float(execution.get("agent_timeout_seconds", 180))
        retrieval = self.config.get("retrieval", {})
        self.initial_top_k = int(retrieval.get("initial_top_k_per_channel", 15))
        self.expansion_quota = int(retrieval.get("expansion_quota_per_channel", 10))
        self.max_queries = int(retrieval.get("max_queries_per_worker", 2))
        self.output_k = int(retrieval.get("output_k", 10))
        self.rrf_k = int(retrieval.get("rrf_k", os.environ.get("RRF_K", 60)))
        self._git_commit = self._resolve_git_commit()
        self._implementation_sha256 = self._compute_implementation_hash()

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        if self._snapshots is None:
            self._snapshots = load_snapshots(self.experiment_dir)
        return self._snapshots

    @property
    def questions(self) -> dict[str, str]:
        if self._questions is None:
            self._questions = self._load_questions()
        return self._questions

    @property
    def qrels(self) -> dict[str, dict[str, int]]:
        if self._qrels is None:
            self._qrels = self._load_qrels()
        return self._qrels

    def _resolve_git_commit(self) -> str:
        configured = self.config.get("git_commit")
        if configured:
            return str(configured)
        root = Path(__file__).resolve().parents[3]
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _compute_implementation_hash(self) -> str:
        directory = Path(__file__).resolve().parent
        values: dict[str, str] = {}
        for name in (
            "schema.py", "pools.py", "selectors.py", "legacy_reuse.py",
            "metrics.py", "statistics.py", "generation.py", "runner.py",
        ):
            path = directory / name
            if path.exists():
                values[name] = file_sha256(path)
        return canonical_sha256(values)

    def _paths(self) -> Mapping[str, Any]:
        value = self.config.get("paths", {})
        return value if isinstance(value, Mapping) else {}

    def _benchmark_dir(self) -> Path:
        benchmark = self.config.get("benchmark", {})
        value = (
            self.config.get("benchmark_dir")
            or self._paths().get("benchmark_dir")
            or (benchmark.get("dir") if isinstance(benchmark, Mapping) else None)
        )
        if value:
            return Path(value)
        return Path(__file__).resolve().parents[3] / "var" / "benchmark" / "final"

    def _chunks_path(self) -> Path:
        value = (
            self.config.get("chunks_pkl")
            or self._paths().get("chunks_pkl")
            or os.environ.get("CHUNKS_PATH")
        )
        if value:
            return Path(value)
        return Path(__file__).resolve().parents[3] / "var" / "data" / "chunks.pkl"

    def _benchmark_input(self, name: str) -> Any:
        benchmark = self.config.get("benchmark", {})
        if isinstance(benchmark, Mapping) and name in benchmark and not isinstance(benchmark[name], str):
            return benchmark[name]
        if name in self.config and not isinstance(self.config[name], (str, os.PathLike)):
            return self.config[name]
        filename = str(benchmark.get(name, f"{name}.jsonl")) if isinstance(benchmark, Mapping) else f"{name}.jsonl"
        return self._benchmark_dir() / filename

    def _load_questions(self) -> dict[str, str]:
        source = self._benchmark_input("questions")
        rows = _jsonl(Path(source)) if isinstance(source, (str, os.PathLike)) else source
        if isinstance(rows, Mapping):
            return {
                str(question_id): str(value.get("query")) if isinstance(value, Mapping) else str(value)
                for question_id, value in rows.items()
            }
        result: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("Question rows must be mappings")
            question_id = str(row["question_id"])
            if question_id in result:
                raise ValueError(f"Duplicate question_id: {question_id}")
            result[question_id] = str(row["query"])
        return result

    def _load_qrels(self) -> dict[str, dict[str, int]]:
        source = self._benchmark_input("qrels")
        rows = _jsonl(Path(source)) if isinstance(source, (str, os.PathLike)) else source
        if isinstance(rows, Mapping):
            result: dict[str, dict[str, int]] = {}
            for question_id, values in rows.items():
                if isinstance(values, Mapping):
                    relevant = {
                        str(chunk_id): int(relevance) for chunk_id, relevance in values.items()
                        if int(relevance) > 0
                    }
                else:
                    relevant = {str(chunk_id): 1 for chunk_id in values}
                if relevant:
                    result[str(question_id)] = relevant
            return result
        result = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("Qrel rows must be mappings")
            relevance = int(row["relevance"])
            if relevance <= 0:
                continue
            question_id = str(row["question_id"])
            chunk_id = str(row["chunk_id"])
            target = result.setdefault(question_id, {})
            target[chunk_id] = max(target.get(chunk_id, 0), relevance)
        return result

    def _ensure_chunks(self) -> dict[str, Any]:
        if self.chunk_map is not None:
            return self.chunk_map
        path = self._chunks_path()
        if not path.exists() and self.engine is not None and hasattr(self.engine, "chunk_map"):
            self.chunk_map = dict(self.engine.chunk_map)
            self.chunks = list(self.chunk_map.values())
            return self.chunk_map
        with path.open("rb") as stream:
            database = pickle.load(stream)
        if isinstance(database, Mapping) and "chunks" in database:
            chunks = list(database["chunks"])
        elif hasattr(database, "chunks"):
            chunks = list(database.chunks)
        elif isinstance(database, Sequence) and not isinstance(database, (str, bytes)):
            chunks = list(database)
        else:
            raise ValueError("chunks_pkl must contain a chunk database or chunk sequence")
        chunk_map: dict[str, Any] = {}
        for chunk in chunks:
            chunk_id = _chunk_id(chunk)
            if chunk_id in chunk_map:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            chunk_map[chunk_id] = chunk
        self.chunks = chunks
        self.chunk_map = chunk_map
        return chunk_map

    def _get_engine(self) -> Any:
        if self.engine is None:
            with self._init_lock:
                if self.engine is None:
                    from retrieval.engine import RetrievalEngine

                    self.engine = RetrievalEngine()
                    embedding_device = os.environ.get("RETRIEVAL_EMBEDDING_DEVICE", "cpu")
                    if embedding_device != "cpu":
                        self.engine.embedding.model.to(embedding_device)
                    print(f"Embedding device: {self.engine.embedding.model.device}", flush=True)
        return self.engine

    def _dense_device(self) -> str:
        embedding = getattr(self._get_engine(), "embedding", None)
        model = getattr(embedding, "model", None)
        return str(getattr(model, "device", os.environ.get("RETRIEVAL_EMBEDDING_DEVICE", "cpu")))

    def _get_reranker(self) -> Any:
        if self.reranker is None:
            with self._init_lock:
                if self.reranker is None:
                    from retrieval.reranker import JinaReranker

                    self.reranker = JinaReranker()
        return self.reranker

    def _make_llm(self, role: str) -> Any:
        from common.config import (
            RETRIEVER_BM25_ENABLE_THINKING,
            RETRIEVER_BM25_LLM_BASE_URL,
            RETRIEVER_BM25_LLM_MAX_TOKENS,
            RETRIEVER_BM25_LLM_MODEL,
            RETRIEVER_BM25_LLM_TEMPERATURE,
            RETRIEVER_EMB_ENABLE_THINKING,
            RETRIEVER_EMB_LLM_BASE_URL,
            RETRIEVER_EMB_LLM_MAX_TOKENS,
            RETRIEVER_EMB_LLM_MODEL,
            RETRIEVER_EMB_LLM_TEMPERATURE,
            create_llm,
        )

        dense = role != "bm25"
        return create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL if dense else RETRIEVER_BM25_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL if dense else RETRIEVER_BM25_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE if dense else RETRIEVER_BM25_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS if dense else RETRIEVER_BM25_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING if dense else RETRIEVER_BM25_ENABLE_THINKING,
        )

    def _get_llm(self, role: str) -> Any:
        explicit = {
            "dense": self.dense_llm,
            "bm25": self.bm25_llm,
            "planner": self.planner_llm,
            "arbiter": self.arbiter_llm,
        }
        if self.llm is not None:
            return self.llm
        if explicit.get(role) is not None:
            return explicit[role]
        canonical = "bm25" if role == "bm25" else "dense"
        if canonical not in self._llm_cache:
            with self._init_lock:
                if canonical not in self._llm_cache:
                    self._llm_cache[canonical] = self._make_llm(canonical)
        return self._llm_cache[canonical]

    def _safe_model_config(self, llm: Any, role: str) -> dict[str, Any]:
        models = self.config.get("models", {})
        configured = models.get(role) if isinstance(models, Mapping) else None
        if configured is None and isinstance(models, Mapping):
            configured = models.get("bm25" if role == "bm25" else "dense")
        result: dict[str, Any] = {}
        if isinstance(configured, Mapping):
            for key in (
                "model", "model_name", "revision", "temperature", "max_tokens", "top_p",
                "seed", "enable_thinking", "timeout",
            ):
                if key in configured:
                    result[key] = _safe_scalar(configured[key])
        attributes = {
            "model": ("model_name", "model"),
            "temperature": ("temperature",),
            "max_tokens": ("max_tokens",),
            "top_p": ("top_p",),
            "seed": ("seed",),
        }
        for output_key, names in attributes.items():
            if output_key in result:
                continue
            for name in names:
                value = getattr(llm, name, None)
                if value is not None and isinstance(value, (str, int, float, bool)):
                    result[output_key] = value
                    break
        if "model" not in result and "model_name" in result:
            result["model"] = result["model_name"]
        extra_body = getattr(llm, "extra_body", {})
        if "enable_thinking" not in result and isinstance(extra_body, Mapping):
            template = extra_body.get("chat_template_kwargs", {})
            if isinstance(template, Mapping) and "enable_thinking" in template:
                result["enable_thinking"] = bool(template["enable_thinking"])
        result["role"] = role
        return result

    def _get_reranker_metadata(self) -> dict[str, Any]:
        if self._reranker_config is not None:
            return copy.deepcopy(self._reranker_config)
        reranker = self._get_reranker()
        configured = self.config.get("reranker", {})
        result: dict[str, Any] = {}
        if isinstance(configured, Mapping):
            for key in ("model", "model_name", "device", "revision", "top_n", "dtype"):
                if key in configured:
                    result[key] = _safe_scalar(configured[key])
        model = getattr(reranker, "model", None)
        model_config = getattr(model, "config", None)
        defaults = {
            "model": getattr(model, "_name_or_path", None),
            "revision": getattr(model_config, "_commit_hash", None),
            "device": getattr(model, "device", None),
            "top_n": getattr(reranker, "top_n", None),
        }
        for key, value in defaults.items():
            if key not in result and value is not None:
                result[key] = _safe_scalar(value)
        if "model" not in result:
            result["model"] = os.environ.get("RERANKER_MODEL", type(reranker).__name__)
        if "device" not in result:
            result["device"] = os.environ.get("RERANKER_DEVICE", "cpu")
        result["top_n"] = int(result.get("top_n", self.output_k))
        if "dtype" not in result:
            result["dtype"] = "float16" if str(result.get("device", "cpu")).startswith("cuda") else "float32"
        self._reranker_config = result
        return copy.deepcopy(result)

    def _corpus_hash(self) -> str:
        chunk_map = self._ensure_chunks()
        chunks = self.chunks if self.chunks is not None else list(chunk_map.values())
        payload = json.dumps(
            [[_chunk_id(chunk), _chunk_text(chunk)] for chunk in chunks],
            ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def benchmark_hashes(self) -> dict[str, Any]:
        if self._benchmark_hashes is not None:
            return copy.deepcopy(self._benchmark_hashes)
        result: dict[str, Any] = {}
        benchmark_dir = self._benchmark_dir()
        benchmark = self.config.get("benchmark", {})
        for name in ("questions", "qrels", "evidence", "coverage"):
            filename = str(benchmark.get(name, f"{name}.jsonl")) if isinstance(benchmark, Mapping) else f"{name}.jsonl"
            path = benchmark_dir / filename
            if path.exists():
                result[name] = file_sha256(path)
        result["corpus_sha256"] = self._corpus_hash()
        manifest = benchmark_dir / "manifest.json"
        if manifest.exists():
            result["manifest_sha256"] = file_sha256(manifest)
        self._benchmark_hashes = result
        return copy.deepcopy(result)

    def _base_descriptor(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        pool_record: Mapping[str, Any],
        selector_id: str,
        report_state: str | None,
        output_policy: str,
        batch: str,
    ) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "protocol_version": self.protocol_version,
            "config_sha256": canonical_sha256(self.config),
            "git_commit": self._git_commit,
            "implementation_sha256": self._implementation_sha256,
            "benchmark_hashes": self.benchmark_hashes(),
            "execution_batch": batch,
            "group_id": snapshot["group_id"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "pool_id": pool_id,
            "pool_sha256": pool_record["pool_sha256"],
            "selector_id": selector_id,
            "report_state": report_state,
            "output_policy": output_policy,
        }

    def _read_cache(self, path: Path) -> tuple[dict[str, Any] | None, str | None]:
        if not path.exists():
            return None, None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Corrupted cache {path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("cache_key"), str):
            raise CacheIntegrityError(f"Incomplete cache descriptor: {path}")
        if not isinstance(value.get("cache_descriptor"), Mapping):
            raise CacheIntegrityError(f"Missing cache descriptor: {path}")
        if value["cache_key"] != canonical_sha256(value["cache_descriptor"]):
            raise CacheIntegrityError(f"Cache key/descriptor mismatch: {path}")
        return value, str(value.get("invalidated_previous_cache_key")) if value.get("invalidated_previous_cache_key") else None

    def _write_cache(self, path: Path, descriptor: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
        key = canonical_sha256(descriptor)
        previous, _ = self._read_cache(path)
        previous_key = previous.get("cache_key") if previous is not None else None
        output = {
            **dict(value),
            "cache_descriptor": copy.deepcopy(dict(descriptor)),
            "cache_key": key,
        }
        if previous_key is not None and previous_key != key:
            output["invalidated_previous_cache_key"] = previous_key
        atomic_write_json(path, output)
        return output

    def _matching_cache(self, path: Path, descriptor: Mapping[str, Any]) -> dict[str, Any] | None:
        value, _ = self._read_cache(path)
        if value is None:
            return None
        if value["cache_key"] != canonical_sha256(descriptor) or value["cache_descriptor"] != descriptor:
            return None
        return value

    def _selector_path(self, batch: str, group_id: str, config_id: str) -> Path:
        return self.experiment_dir / "selector_results" / batch / group_id / f"{config_id}.json"

    async def _cached_replay(
        self,
        llm: Any,
        timeout_seconds: float,
        role: str,
        system: str,
        human: str,
        required: tuple[str, ...],
        validator: Callable[[Mapping[str, Any]], tuple[Any, str, Any]],
        *,
        snapshot: Mapping[str, Any],
        pool_id: str,
        pool_record: Mapping[str, Any],
        selector_id: str,
        report_state: str | None,
        batch: str,
        report_payload_sha256: str | None = None,
    ) -> tuple[Any, dict[str, Any], str | None]:
        prompt_hash = prompt_sha256(system, human)
        model_config = self._safe_model_config(llm, role)
        descriptor = self._base_descriptor(
            snapshot, pool_id, pool_record, selector_id, report_state, "raw_llm", batch,
        )
        raw_config = copy.deepcopy(self.config)
        raw_config.pop("outputs", None)
        descriptor["config_sha256"] = canonical_sha256(raw_config)
        descriptor.update({
            "role": role,
            "prompt_sha256": prompt_hash,
            "model_config": model_config,
            "model_config_sha256": canonical_sha256(model_config),
            "report_payload_sha256": report_payload_sha256,
            "required_json_fields": list(required),
        })
        key = canonical_sha256(descriptor)
        safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", role)
        path = self.experiment_dir / "llm_calls" / batch / snapshot["group_id"] / f"{safe_role}-{key[:24]}.json"
        cached = self._matching_cache(path, descriptor)
        if cached is not None:
            if "value" not in cached or not isinstance(cached.get("meta"), Mapping):
                raise CacheIntegrityError(f"Incomplete raw LLM cache: {path}")
            meta = copy.deepcopy(dict(cached["meta"]))
            meta["cache_status"] = "hit"
            meta["raw_cache_key"] = key
            return copy.deepcopy(cached["value"]), meta, path.relative_to(self.experiment_dir).as_posix()
        value, meta = await ReplayLLMCaller(llm, timeout_seconds=timeout_seconds).call(
            role, system, human, required, validator,
        )
        meta["cache_status"] = "miss"
        meta["raw_cache_key"] = key
        raw_value = {"value": value, "meta": meta}
        self._write_cache(path, descriptor, raw_value)
        return value, meta, path.relative_to(self.experiment_dir).as_posix()

    def _generation_replay(
        self,
        snapshot: Mapping[str, Any],
        kind: str,
        batch: str,
    ) -> Callable[..., Awaitable[tuple[Any, dict[str, Any], str | None]]]:
        base = snapshot["pools"]["B"]

        async def call(
            llm: Any,
            timeout_seconds: float,
            role: str,
            system: str,
            human: str,
            required: tuple[str, ...],
            validator: Callable[[Mapping[str, Any]], tuple[Any, str, Any]],
        ) -> tuple[Any, dict[str, Any], str | None]:
            return await self._cached_replay(
                llm, timeout_seconds, role, system, human, required, validator,
                snapshot=snapshot, pool_id="B", pool_record=base,
                selector_id=f"generation_{kind}", report_state=role, batch=batch,
            )

        return call

    def _main_pool_artifact(self, snapshot: Mapping[str, Any], pool_id: str) -> dict[str, Any]:
        record = snapshot.get("pools", {}).get(pool_id)
        if record is None:
            raise ValueError(f"Pool {pool_id} is not built for {snapshot['group_id']}")
        if pool_id in {"B", "E"}:
            events = [
                copy.deepcopy(event) for event in snapshot.get("retrieval_events", [])
                if pool_id == "E" or event.get("stage") == "initial"
            ]
            return {
                "pool_record": record,
                "retrieval_events": events,
                "raw_hit_counts": snapshot.get("raw_hit_counts"),
                "expansion_active": snapshot.get("expansion_active", False),
                "fallback_provenance": snapshot.get("fallback_provenance", []),
                "fallback_stages": snapshot.get("legacy", {}).get("fallback_stages", []),
            }
        wide = snapshot.get("wide", {})
        events = wide.get("retrieval_events")
        if events is None:
            events = [
                copy.deepcopy(event) for event in snapshot.get("retrieval_events", [])
                if event.get("stage") == "initial"
            ]
        return {
            "pool_record": record,
            "retrieval_events": copy.deepcopy(events),
            "raw_hit_counts": wide.get("raw_hit_counts"),
            "expansion_active": snapshot.get("expansion_active", False),
            "reused_from": record.get("reused_from") or wide.get("reused_from"),
        }

    def _generated_path(self, batch: str, group_id: str) -> Path:
        return self.experiment_dir / "generated_pools" / batch / f"{group_id}.json"

    def _read_generated(self, batch: str, group_id: str) -> dict[str, Any] | None:
        path = self._generated_path(batch, group_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Corrupted generated pool cache {path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("pools"), Mapping):
            raise CacheIntegrityError(f"Incomplete generated pool cache: {path}")
        return value

    def _pool_artifact(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        batch: str,
        prefer_generated: bool = True,
    ) -> dict[str, Any]:
        if prefer_generated:
            generated = self._read_generated(batch, snapshot["group_id"])
            if generated is not None and pool_id in generated["pools"]:
                artifact = generated["pools"][pool_id]
                if not isinstance(artifact, Mapping) or not isinstance(artifact.get("pool_record"), Mapping):
                    raise CacheIntegrityError(f"Invalid generated pool {pool_id} for {snapshot['group_id']}")
                assert_same_pool(artifact["pool_record"])
                return copy.deepcopy(dict(artifact))
        return self._main_pool_artifact(snapshot, pool_id)

    def load_pool(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        execution_batch: str = "main",
    ) -> dict[str, Any]:
        artifact = self._pool_artifact(snapshot, pool_id, execution_batch)
        return copy.deepcopy(artifact["pool_record"])

    def _events_for(self, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
        return copy.deepcopy(list(artifact.get("retrieval_events", [])))

    def _rrf_rows(self, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
        return multi_event_rrf(self._events_for(artifact), self.rrf_k, limit=30)

    def _base_fallbacks(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        artifact: Mapping[str, Any],
        report_state: str | None = None,
        cached_legacy_reports: bool = False,
        cached_legacy_selector: bool = False,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if pool_id in {"S", "NP"} or (pool_id in {"E", "W"} and "generation_meta" in artifact):
            stages = list(artifact.get("fallback_stages", []))
            provenance = copy.deepcopy(list(artifact.get("fallback_provenance", [])))
            if pool_id == "W" and artifact.get("reused_from") == "B":
                return [], []
            return stages, provenance
        if pool_id not in {"E"}:
            return [], []
        legacy = snapshot.get("legacy", {})
        all_stages = [str(stage) for stage in legacy.get("fallback_stages", [])]
        selected: list[str] = []
        for stage in all_stages:
            if stage == "planner" or stage.endswith("_round_1"):
                selected.append(stage)
            elif cached_legacy_reports and stage.endswith("_round_2"):
                selected.append(stage)
            elif cached_legacy_selector:
                selected.append(stage)
        provenance = [
            copy.deepcopy(item) for item in snapshot.get("fallback_provenance", [])
            if isinstance(item, Mapping) and item.get("stage") in selected
        ]
        known = {item.get("stage") for item in provenance}
        provenance.extend(
            {"stage": stage, "consumed_state": "legacy", "source": "legacy_trace"}
            for stage in selected if stage not in known
        )
        return selected, provenance

    def _selector_artifact(
        self,
        snapshot: Mapping[str, Any],
        config_id: str,
        pool_id: str,
        pool_record: Mapping[str, Any],
        selector_id: str,
        report_state: str | None,
        output_policy: str,
        ranking: Sequence[str],
        batch: str,
        descriptor: Mapping[str, Any],
        **extra: Any,
    ) -> dict[str, Any]:
        from scripts.coordinated_experiment.pools import validate_ranking

        ranked = validate_ranking(list(ranking), pool_record, max_k=self.output_k)
        value = {
            "config_id": config_id,
            "base_config_id": config_id.rsplit("__", 1)[0] if output_policy in DERIVED_POLICIES + ("legacy_backfill",) and config_id.endswith(f"__{output_policy}") else config_id,
            "group_id": snapshot["group_id"],
            "query_group": snapshot["query_group"],
            "query": snapshot["query"],
            "question_ids": list(snapshot["question_ids"]),
            "pool_id": pool_id,
            "pool_sha256": pool_record["pool_sha256"],
            "input_snapshot_sha256": snapshot["snapshot_sha256"],
            "selector_id": selector_id,
            "report_state": report_state,
            "output_policy": output_policy,
            "ranked_chunk_ids": ranked,
            "pool_size": len(pool_record["chunk_ids"]),
            "output_size": len(ranked),
            "expansion_active": bool(snapshot.get("expansion_active")),
            "execution_batch": batch,
            "smoke_only": bool(self.smoke_only or batch == "smoke"),
            **extra,
        }
        path = self._selector_path(batch, snapshot["group_id"], config_id)
        return self._write_cache(path, descriptor, value)

    def _descriptor_for_selector(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        pool_record: Mapping[str, Any],
        selector_id: str,
        report_state: str | None,
        output_policy: str,
        batch: str,
        prompt_hash: str | None,
        model_config: Mapping[str, Any],
        report_payload_sha256: str | None = None,
    ) -> dict[str, Any]:
        descriptor = self._base_descriptor(
            snapshot, pool_id, pool_record, selector_id, report_state, output_policy, batch,
        )
        descriptor.update({
            "prompt_sha256": prompt_hash,
            "model_config": copy.deepcopy(dict(model_config)),
            "model_config_sha256": canonical_sha256(model_config),
            "report_payload_sha256": report_payload_sha256,
        })
        return descriptor

    async def _run_rrf(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        batch: str,
        config_id: str | None = None,
    ) -> dict[str, Any]:
        config_id = config_id or f"{pool_id}__RRF"
        artifact = self._pool_artifact(snapshot, pool_id, batch)
        if pool_id == "W" and (
            artifact.get("reused_from") == "B"
            or (not artifact.get("expansion_active") and artifact["pool_record"]["pool_sha256"] == snapshot["pools"]["B"]["pool_sha256"])
        ):
            source_batch = self.execution_batch if batch.startswith("repeat_") else batch
            source = await self._run_rrf(snapshot, "B", source_batch)
            return self._alias_selector(
                snapshot, source, config_id, pool_id, batch,
                target_pool_record=artifact["pool_record"],
            )
        pool = artifact["pool_record"]
        rows = self._rrf_rows(artifact)
        ranking = [row["chunk_id"] for row in rows[: self.output_k]]
        model_config = {"selector": "multi_event_rrf", "rrf_k": self.rrf_k, "alpha": 0.5, "limit": 30}
        descriptor = self._descriptor_for_selector(
            snapshot, pool_id, pool, "RRF", None, "strict", batch, None, model_config,
        )
        cached = self._matching_cache(self._selector_path(batch, snapshot["group_id"], config_id), descriptor)
        if cached is not None:
            return cached
        stages, provenance = self._base_fallbacks(snapshot, pool_id, artifact)
        return self._selector_artifact(
            snapshot, config_id, pool_id, pool, "RRF", None, "strict", ranking, batch, descriptor,
            raw_model_response_path=None,
            selector_latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens=0,
            token_measurement_status="not_applicable",
            repair_used=False,
            fallback_stage=None,
            fallback_stages=stages,
            fallback_provenance=provenance,
            dropped_unknown_aliases=[],
            dropped_duplicate_aliases=[],
            rrf_rows=rows,
            search_calls=len(self._events_for(artifact)),
            raw_search_hits=sum(len(event.get("hits", [])) for event in self._events_for(artifact)),
            model_config=model_config,
        )

    def _rerank_sync(self, query: str, candidates: list[dict[str, Any]]) -> Any:
        with self._rerank_lock:
            return self._get_reranker().rerank(query, candidates)

    async def _run_jina(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        batch: str,
        config_id: str | None = None,
    ) -> dict[str, Any]:
        config_id = config_id or f"{pool_id}__JINA"
        artifact = self._pool_artifact(snapshot, pool_id, batch)
        if pool_id == "W" and (
            artifact.get("reused_from") == "B"
            or (not artifact.get("expansion_active") and artifact["pool_record"]["pool_sha256"] == snapshot["pools"]["B"]["pool_sha256"])
        ):
            source_batch = self.execution_batch if batch.startswith("repeat_") else batch
            source = await self._run_jina(snapshot, "B", source_batch)
            return self._alias_selector(
                snapshot, source, config_id, pool_id, batch,
                target_pool_record=artifact["pool_record"],
            )
        pool = artifact["pool_record"]
        model_config = self._get_reranker_metadata()
        descriptor = self._descriptor_for_selector(
            snapshot, pool_id, pool, "JINA", None, "strict", batch, None, model_config,
        )
        path = self._selector_path(batch, snapshot["group_id"], config_id)
        cached = self._matching_cache(path, descriptor)
        if cached is not None:
            return cached
        chunk_map = self._ensure_chunks()
        rrf_rows = self._rrf_rows(artifact)
        candidates: list[dict[str, Any]] = []
        for row in rrf_rows:
            chunk_id = row["chunk_id"]
            chunk = chunk_map[chunk_id]
            candidates.append({
                "chunk_id": chunk_id,
                "id": chunk_id,
                "text": _chunk_text(chunk),
                "doc_path": _chunk_value(chunk, "doc_path", ""),
                "title": _chunk_value(chunk, "title", ""),
                "token_count": _chunk_value(chunk, "token_count", 0),
                "rrf_rank": row["rank"],
                "rrf_score": row["score"],
            })
        started = time.perf_counter()
        reranked = await asyncio.to_thread(self._rerank_sync, snapshot["query"], candidates)
        latency = int((time.perf_counter() - started) * 1000)
        ranking: list[str] = []
        scores: dict[str, float] = {}
        for item in list(reranked)[: self.output_k]:
            if isinstance(item, str):
                chunk_id = item
                score = None
            elif isinstance(item, Mapping):
                chunk_id = str(item.get("chunk_id", item.get("id")))
                raw_score = item.get("rerank_score", item.get("score"))
                score = float(raw_score) if raw_score is not None else None
            else:
                chunk_id = _chunk_id(getattr(item, "chunk", item))
                raw_score = getattr(item, "score", None)
                score = float(raw_score) if raw_score is not None else None
            if chunk_id not in pool["chunk_ids"]:
                raise ValueError(f"Jina returned chunk outside pool: {chunk_id}")
            if chunk_id in ranking:
                raise ValueError(f"Jina returned duplicate chunk: {chunk_id}")
            ranking.append(chunk_id)
            if score is not None:
                scores[chunk_id] = score
        stages, provenance = self._base_fallbacks(snapshot, pool_id, artifact)
        return self._selector_artifact(
            snapshot, config_id, pool_id, pool, "JINA", None, "strict", ranking, batch, descriptor,
            raw_model_response_path=None,
            selector_latency_ms=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens=0,
            token_measurement_status="not_applicable",
            repair_used=False,
            fallback_stage=None,
            fallback_stages=stages,
            fallback_provenance=provenance,
            dropped_unknown_aliases=[],
            dropped_duplicate_aliases=[],
            rrf_rows=rrf_rows,
            jina_scores=scores,
            search_calls=len(self._events_for(artifact)),
            raw_search_hits=sum(len(event.get("hits", [])) for event in self._events_for(artifact)),
            model_config=model_config,
            jina_metadata=model_config,
        )

    def _alias_selector(
        self,
        snapshot: Mapping[str, Any],
        source: Mapping[str, Any],
        target_config_id: str,
        target_pool_id: str,
        batch: str,
        target_report_state: str | None = None,
        target_pool_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pool_record = target_pool_record or snapshot["pools"][target_pool_id]
        assert_same_pool({"pool_sha256": source["pool_sha256"]}, pool_record)
        output_policy = str(source.get("output_policy", "strict"))
        model_config = copy.deepcopy(source.get("model_config", {"selector": source.get("selector_id")}))
        descriptor = self._descriptor_for_selector(
            snapshot, target_pool_id, pool_record, str(source["selector_id"]),
            target_report_state if target_report_state is not None else source.get("report_state"),
            output_policy, batch, source.get("prompt_sha256"), model_config,
            source.get("report_payload_sha256"),
        )
        descriptor["reused_from"] = source["config_id"]
        descriptor["reused_from_cache_key"] = source.get("cache_key")
        path = self._selector_path(batch, snapshot["group_id"], target_config_id)
        cached = self._matching_cache(path, descriptor)
        if cached is not None:
            return cached
        excluded = {
            "cache_descriptor", "cache_key", "invalidated_previous_cache_key",
            "config_id", "base_config_id", "group_id", "query_group", "query", "question_ids",
            "pool_id", "pool_sha256", "input_snapshot_sha256", "selector_id", "report_state",
            "output_policy", "ranked_chunk_ids", "pool_size", "output_size", "expansion_active",
            "execution_batch", "smoke_only",
        }
        value = {key: copy.deepcopy(item) for key, item in source.items() if key not in excluded}
        value.update({
            "selector_latency_ms": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens": 0,
            "token_measurement_status": "not_executed_reused",
            "repair_used": False,
            "source_repair_used": source.get("repair_used"),
            "new_llm_call": False,
            "new_llm_calls": 0,
            "reused_from": source["config_id"],
            "source_execution_batch": source.get("execution_batch"),
        })
        return self._selector_artifact(
            snapshot, target_config_id, target_pool_id, pool_record, str(source["selector_id"]),
            target_report_state if target_report_state is not None else source.get("report_state"),
            output_policy, source["ranked_chunk_ids"], batch, descriptor, **value,
        )

    async def _run_legacy(
        self,
        snapshot: Mapping[str, Any],
        batch: str,
    ) -> list[dict[str, Any]]:
        pool_id = "E"
        pool = snapshot["pools"][pool_id]
        strict = list(snapshot["legacy"]["strict_arbiter_ids"])
        reports = snapshot["reports"]["legacy_R2P"]
        rows = self._rrf_rows(self._main_pool_artifact(snapshot, "E"))
        common = apply_backfill(strict, pool, rows, output_k=self.output_k, policy="common")
        legacy = apply_backfill(
            strict, pool, rows, output_k=self.output_k, reports=reports, policy="legacy",
        )
        if legacy != list(snapshot["legacy"]["production_ids"]):
            raise ValueError(f"Legacy backfill mismatch for {snapshot['group_id']}")
        model_config = {"selector": "cached_legacy_arbiter", "raw_usage_available": False}
        stages, provenance = self._base_fallbacks(
            snapshot, pool_id, self._main_pool_artifact(snapshot, pool_id),
            cached_legacy_reports=True, cached_legacy_selector=True,
        )
        results = []
        for policy, ranking in (
            ("strict", strict), ("common_backfill", common), ("legacy_backfill", legacy),
        ):
            config_id = f"E__R2P__ARB__LEGACY__{policy}"
            descriptor = self._descriptor_for_selector(
                snapshot, pool_id, pool, "ARB_LEGACY", "legacy_R2P", policy, batch,
                None, model_config, canonical_sha256(reports),
            )
            cached = self._matching_cache(self._selector_path(batch, snapshot["group_id"], config_id), descriptor)
            if cached is not None:
                results.append(cached)
                continue
            results.append(self._selector_artifact(
                snapshot, config_id, pool_id, pool, "ARB_LEGACY", "legacy_R2P", policy,
                ranking, batch, descriptor,
                raw_model_response_path=None,
                selector_latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                tokens=None,
                token_measurement_status="unavailable_raw_legacy_selector",
                repair_used=None,
                fallback_stage="legacy_arbiter" if "arbiter" in stages else None,
                fallback_stages=stages,
                fallback_provenance=provenance,
                dropped_unknown_aliases=[],
                dropped_duplicate_aliases=[],
                strict_ids=strict,
                model_config=model_config,
                cached_generation_tokens=snapshot.get("generation_usage", {}).get("total_tokens"),
                legacy_wall_time_ms=None,
                search_calls=len(snapshot.get("retrieval_events", [])),
                raw_search_hits=snapshot.get("raw_hit_counts", {}).get("total"),
            ))
        return results

    def _report_path(self, batch: str, group_id: str, state: str) -> Path:
        return self.experiment_dir / "reports" / batch / group_id / f"{state}.json"

    def _load_clean_reports(self, snapshot: Mapping[str, Any], state: str, batch: str) -> dict[str, Any]:
        path = self._report_path(batch, snapshot["group_id"], state)
        if not path.exists():
            raise ValueError(f"Missing report state {state} for {snapshot['group_id']}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Corrupted report state {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
            raise ValueError(f"Report snapshot mismatch: {path}")
        if value.get("pool_sha256") != snapshot["pools"]["E"]["pool_sha256"]:
            raise ValueError(f"Report pool mismatch: {path}")
        if not isinstance(value.get("reports"), Mapping):
            raise ValueError(f"Report payload missing: {path}")
        return value

    async def _run_clean_reports(
        self,
        snapshot: Mapping[str, Any],
        state: str,
        batch: str,
    ) -> dict[str, Any]:
        if state not in {"R2NP", "R2P"}:
            raise ValueError("Clean report state must be R2NP or R2P")
        report_path = self._report_path(batch, snapshot["group_id"], state)
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CacheIntegrityError(f"Corrupted report state {report_path}: {exc}") from exc
            if not isinstance(existing, dict) or not isinstance(existing.get("reports"), Mapping):
                raise CacheIntegrityError(f"Incomplete report state: {report_path}")
        if not snapshot.get("expansion_active"):
            value = {
                "protocol_version": self.protocol_version,
                "group_id": snapshot["group_id"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "pool_sha256": snapshot["pools"]["E"]["pool_sha256"],
                "report_state": state,
                "reports": copy.deepcopy(snapshot["reports"]["R1"]),
                "worker_meta": {},
                "raw_model_response_paths": [],
                "fallback_stages": [],
                "fallback_provenance": [],
                "expansion_active": False,
                "reused_from": "R1",
                "new_llm_calls": 0,
            }
            atomic_write_json(self._report_path(batch, snapshot["group_id"], state), value)
            return value
        pool_record = snapshot["pools"]["E"]
        pool = reconstruct_pool(pool_record, self._ensure_chunks())
        query_cache = {channel: {normalize_query_text(snapshot["query"])} for channel in CHANNELS}

        async def call(channel: str) -> tuple[str, dict[str, Any]]:
            other = snapshot["reports"]["R1"]["bm25" if channel == "dense" else "dense"] if state == "R2P" else None
            system, human = build_worker_prompts(
                channel, snapshot["query"], snapshot["plan"], 2,
                snapshot["observations"]["final"][channel], pool, other, clean=True,
            )
            validator = make_worker_validator(
                channel, 2, snapshot["observations"]["final"][channel], pool, query_cache,
            )
            llm = self._get_llm(channel)
            report, meta, raw_path = await self._cached_replay(
                llm, self.timeout_seconds, f"{channel}_round_2", system, human,
                ("assessments", "search_queries", "coverage"), validator,
                snapshot=snapshot, pool_id="E", pool_record=pool_record,
                selector_id=f"{state}_worker", report_state=state, batch=batch,
                report_payload_sha256=canonical_sha256(other),
            )
            return channel, {
                **meta,
                "report": report,
                "system_prompt": system,
                "human_prompt": human,
                "prompt_sha256": prompt_sha256(system, human),
                "raw_model_response_path": raw_path,
                "other_report_sha256": canonical_sha256(other),
            }

        results = await asyncio.gather(call("dense"), call("bm25"))
        reports: dict[str, Any] = {}
        worker_meta: dict[str, Any] = {}
        fallback_stages: list[str] = []
        for channel, result in results:
            worker_meta[channel] = result
            report = result.get("report")
            if report is None:
                fallback_stages.append(f"{channel}_round_2")
                report = copy.deepcopy(snapshot["reports"]["R1"][channel])
            reports[channel] = report
        base_stages, base_provenance = self._base_fallbacks(
            snapshot, "E", self._main_pool_artifact(snapshot, "E"),
        )
        all_stages = base_stages + fallback_stages
        provenance = base_provenance + [
            {"stage": stage, "consumed_state": state, "source": "new_clean_worker", "execution_batch": batch}
            for stage in fallback_stages
        ]
        value = {
            "protocol_version": self.protocol_version,
            "group_id": snapshot["group_id"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "pool_sha256": pool_record["pool_sha256"],
            "report_state": state,
            "reports": reports,
            "worker_meta": worker_meta,
            "raw_model_response_paths": [
                result["raw_model_response_path"] for _, result in results
                if result.get("raw_model_response_path")
            ],
            "fallback_stages": all_stages,
            "fallback_provenance": provenance,
            "new_worker_fallback_stages": fallback_stages,
            "expansion_active": True,
            "reused_from": None,
            "clean_system_prompt": True,
            "peer_payload": "opposing_R1" if state == "R2P" else None,
            "new_llm_calls": sum(result.get("cache_status") == "miss" for _, result in results),
        }
        value["report_sha256"] = canonical_sha256(reports)
        atomic_write_json(self._report_path(batch, snapshot["group_id"], state), value)
        return value

    async def _report_payload(
        self,
        snapshot: Mapping[str, Any],
        state: str,
        batch: str,
        clean: bool,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        if state == "R0":
            stages, provenance = self._base_fallbacks(
                snapshot, "E", self._main_pool_artifact(snapshot, "E"),
            )
            return None, stages, provenance
        if state == "R1":
            reports = snapshot["reports"]["R1"]
        elif state == "legacy_R2P":
            reports = snapshot["reports"]["legacy_R2P"]
        elif state in {"R2NP", "R2P"} and clean:
            if not snapshot.get("expansion_active"):
                reports = snapshot["reports"]["R1"]
                stages, provenance = self._base_fallbacks(
                    snapshot, "E", self._main_pool_artifact(snapshot, "E"),
                )
                return reports, stages, provenance
            artifact = self._load_clean_reports(snapshot, state, batch)
            reports = artifact["reports"]
            return (
                reports,
                list(artifact.get("fallback_stages", [])),
                copy.deepcopy(artifact.get("fallback_provenance", [])),
            )
        else:
            raise ValueError(f"Unsupported report state: {state}")
        artifact = self._main_pool_artifact(snapshot, "E")
        stages, provenance = self._base_fallbacks(
            snapshot, "E", artifact, cached_legacy_reports=state == "legacy_R2P",
        )
        return reports, stages, provenance

    async def _run_arbiter(
        self,
        snapshot: Mapping[str, Any],
        pool_id: str,
        state: str,
        batch: str,
        config_id: str,
        clean: bool = False,
        alias_inactive: bool = True,
    ) -> list[dict[str, Any]]:
        if (
            alias_inactive and clean and state in {"R2NP", "R2P"}
            and not snapshot.get("expansion_active")
        ):
            sources = await self._run_arbiter(
                snapshot, "E", "R1", self.execution_batch, "E__R1__ARB", clean=False,
            )
            results = []
            for source in sources:
                policy = source["output_policy"]
                results.append(self._alias_selector(
                    snapshot, source, f"{config_id}__{policy}", pool_id, batch,
                    target_report_state=state, target_pool_record=snapshot["pools"][pool_id],
                ))
            return results
        artifact = self._pool_artifact(snapshot, pool_id, batch, prefer_generated=pool_id in {"S", "NP"})
        pool_record = artifact["pool_record"]
        reports, stages, provenance = await self._report_payload(snapshot, state, batch, clean)
        if pool_id != "E":
            stages, provenance = self._base_fallbacks(snapshot, pool_id, artifact)
        pool = reconstruct_pool(pool_record, self._ensure_chunks())
        system, human = build_arbiter_prompts(snapshot["query"], snapshot["plan"], reports, pool)
        prompt_hash = prompt_sha256(system, human)
        llm = self._get_llm("arbiter")
        model_config = self._safe_model_config(llm, "arbiter")
        selector_id = "ARB_CLEAN" if clean and state == "R2P" else "ARB"
        if config_id == "E__LEGACY_R2P__ARB_REPLAY":
            selector_id = "ARB_REPLAY"
        report_hash = canonical_sha256(reports)
        value, meta, retained_raw_path = await self._cached_replay(
            llm, self.timeout_seconds, "arbiter", system, human,
            ("selected_chunks",), lambda item: arbiter_validator(item, pool),
            snapshot=snapshot, pool_id=pool_id, pool_record=pool_record,
            selector_id=selector_id, report_state=state, batch=batch,
            report_payload_sha256=report_hash,
        )
        notes = meta.get("notes") if isinstance(meta.get("notes"), Mapping) else {}
        raw_strict = [item["chunk_id"] for item in value] if value is not None else []
        failed = value is None
        rrf_rows = self._rrf_rows(artifact)
        rrf_ids = [row["chunk_id"] for row in rrf_rows[: self.output_k]]
        if failed:
            strict = rrf_ids
            common = list(rrf_ids)
            failure = {
                "config_id": config_id,
                "group_id": snapshot["group_id"],
                "query_group": snapshot["query_group"],
                "question_ids": list(snapshot["question_ids"]),
                "pool_id": pool_id,
                "pool_sha256": pool_record["pool_sha256"],
                "input_snapshot_sha256": snapshot["snapshot_sha256"],
                "ranked_chunk_ids": [],
                "strict_ids": [],
                "error": meta.get("error"),
                "errors": meta.get("errors", []),
                "raw_responses": meta.get("raw_responses", []),
                "raw_model_response_path": retained_raw_path,
                "execution_batch": batch,
                "smoke_only": bool(self.smoke_only or batch == "smoke"),
            }
            atomic_write_json(
                self.experiment_dir / "raw_failures" / batch / snapshot["group_id"] / f"{config_id}.json",
                failure,
            )
        else:
            strict = raw_strict
            common = apply_backfill(
                strict, pool_record, rrf_rows, output_k=self.output_k, policy="common",
            )
        if failed:
            stages = list(stages) + ["arbiter"]
            provenance = list(provenance) + [{
                "stage": "arbiter", "consumed_state": state, "source": "new_selector",
                "execution_batch": batch,
            }]
        results = []
        for policy, ranking in (("strict", strict), ("common_backfill", common)):
            derived_config = f"{config_id}__{policy}"
            descriptor = self._descriptor_for_selector(
                snapshot, pool_id, pool_record, selector_id, state, policy, batch,
                prompt_hash, model_config, report_hash,
            )
            descriptor["raw_llm_cache_key"] = meta["raw_cache_key"]
            path = self._selector_path(batch, snapshot["group_id"], derived_config)
            cached = self._matching_cache(path, descriptor)
            if cached is not None:
                results.append(cached)
                continue
            results.append(self._selector_artifact(
                snapshot, derived_config, pool_id, pool_record, selector_id, state, policy,
                ranking, batch, descriptor,
                raw_model_response_path=retained_raw_path,
                selector_latency_ms=meta.get("selector_latency_ms") if meta.get("cache_status") == "miss" else None,
                prompt_tokens=meta.get("prompt_tokens"),
                completion_tokens=meta.get("completion_tokens"),
                total_tokens=meta.get("total_tokens"),
                tokens=meta.get("total_tokens"),
                token_measurement_status=meta.get("token_measurement_status", "unavailable"),
                repair_used=bool(meta.get("repair_used", False)),
                repair_metadata=copy.deepcopy(meta.get("repair_metadata")),
                fallback_stage="arbiter" if failed else None,
                fallback_stages=stages,
                fallback_provenance=provenance,
                dropped_unknown_aliases=list(notes.get("dropped_unknown_aliases", [])),
                dropped_duplicate_aliases=list(notes.get("dropped_duplicate_aliases", [])),
                strict_ids=raw_strict,
                raw_strict_ids=raw_strict,
                attempts=meta.get("attempts"),
                error=meta.get("error"),
                prompt_sha256=prompt_hash,
                model_config=model_config,
                report_payload_sha256=report_hash,
                raw_llm_cache_key=meta["raw_cache_key"],
                new_llm_call=meta.get("cache_status") == "miss",
                search_calls=0,
                raw_search_hits=0,
                rrf_rows=rrf_rows,
                cached_generation_tokens=snapshot.get("generation_usage", {}).get("total_tokens") if pool_id == "E" else 0,
            ))
        return results

    async def _run_wrrf(
        self,
        snapshot: Mapping[str, Any],
        state: str,
        batch: str,
        config_id: str,
    ) -> dict[str, Any]:
        clean = state in {"R2NP", "R2P"}
        reports, stages, provenance = await self._report_payload(snapshot, state, batch, clean)
        pool = snapshot["pools"]["E"]
        rows = worker_rrf(reports, pool["alias_to_id"], self.rrf_k, limit=30)
        ranking = [row["chunk_id"] for row in rows[: self.output_k]]
        model_config = {"selector": "worker_rrf", "rrf_k": self.rrf_k, "alpha": 0.5, "limit": 30}
        descriptor = self._descriptor_for_selector(
            snapshot, "E", pool, "WRRF", state, "strict", batch, None,
            model_config, canonical_sha256(reports),
        )
        path = self._selector_path(batch, snapshot["group_id"], config_id)
        cached = self._matching_cache(path, descriptor)
        if cached is not None:
            return cached
        return self._selector_artifact(
            snapshot, config_id, "E", pool, "WRRF", state, "strict", ranking, batch, descriptor,
            raw_model_response_path=None,
            selector_latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens=0,
            token_measurement_status="not_applicable",
            repair_used=False,
            fallback_stage=None,
            fallback_stages=stages,
            fallback_provenance=provenance,
            dropped_unknown_aliases=[],
            dropped_duplicate_aliases=[],
            model_config=model_config,
            report_payload_sha256=canonical_sha256(reports),
            reused_from="R1" if not snapshot.get("expansion_active") and clean else None,
            new_llm_calls=0,
            search_calls=0,
            raw_search_hits=0,
            worker_rrf_rows=rows,
        )

    async def _map_groups(
        self,
        snapshots: Sequence[Mapping[str, Any]],
        function: Callable[[Mapping[str, Any]], Awaitable[Any]],
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0

        async def run(snapshot: Mapping[str, Any]) -> Any:
            nonlocal completed
            async with semaphore:
                result = await function(snapshot)
                completed += 1
                if completed % 25 == 0 or completed == len(snapshots):
                    print(f"[{self.execution_batch}] {function.__name__}: {completed}/{len(snapshots)}", flush=True)
                return result

        return list(await asyncio.gather(*(run(snapshot) for snapshot in snapshots)))

    def _validate_benchmark_counts(self) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        benchmark = self.config.get("benchmark", {})
        if not isinstance(benchmark, Mapping):
            return errors
        checks = (
            ("expected_rows", len(self.questions), "questions"),
            ("expected_groups", len(self.snapshots), "groups"),
        )
        for key, actual, label in checks:
            if key in benchmark and int(benchmark[key]) != actual:
                errors.append({
                    "code": "benchmark_count_mismatch",
                    "message": f"{label}: expected {benchmark[key]}, got {actual}",
                })
        answerable_rows = sum(question_id in self.qrels for question_id in self.questions)
        answerable_groups = sum(
            any(question_id in self.qrels for question_id in snapshot["question_ids"])
            for snapshot in self.snapshots
        )
        for key, actual, label in (
            ("expected_answerable_rows", answerable_rows, "answerable rows"),
            ("expected_answerable_groups", answerable_groups, "answerable groups"),
        ):
            if key in benchmark and int(benchmark[key]) != actual:
                errors.append({
                    "code": "benchmark_count_mismatch",
                    "message": f"{label}: expected {benchmark[key]}, got {actual}",
                })
        return errors

    def _event_identity(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "channel": event.get("channel"),
            "stage": event.get("stage"),
            "query": event.get("query"),
            "call_index": event.get("call_index"),
            "hits": [
                {"chunk_id": hit.get("chunk_id"), "rank": hit.get("rank")}
                for hit in event.get("hits", [])
            ],
        }

    def _actual_event(
        self,
        expected: Mapping[str, Any],
        hits: Sequence[Any],
        chunk_map: Mapping[str, Any],
    ) -> dict[str, Any]:
        event_hits: list[dict[str, Any]] = []
        text_hashes: dict[str, str] = {}
        seen: set[str] = set()
        for index, raw_hit in enumerate(hits, 1):
            chunk, chunk_id, rank, score = _hit_parts(raw_hit, index)
            if chunk_id in seen:
                raise ValueError(f"Duplicate chunk {chunk_id} in {expected['event_id']}")
            seen.add(chunk_id)
            if chunk_id not in chunk_map:
                raise KeyError(f"Missing replay chunk {chunk_id}")
            text_hash = text_sha256(_chunk_text(chunk_map[chunk_id]))
            if text_sha256(_chunk_text(chunk)) != text_hash:
                raise ValueError(f"Replay/chunk-store text mismatch: {chunk_id}")
            event_hits.append({
                "chunk_id": chunk_id, "rank": rank, "score": score, "text_sha256": text_hash,
            })
            text_hashes[chunk_id] = text_hash
        return {
            "event_id": expected["event_id"],
            "channel": expected["channel"],
            "stage": expected["stage"],
            "query": expected["query"],
            "call_index": expected["call_index"],
            "hits": event_hits,
            "text_sha256": text_hashes,
        }

    async def _replay_group(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        from retrieval.contracts import SearchBudget

        chunk_map = self._ensure_chunks()
        budget = SearchBudget()
        engine = self._get_engine().with_budget(budget)
        expected_events = list(snapshot.get("retrieval_events", []))
        initial = {
            channel: [event for event in expected_events if event.get("channel") == channel and event.get("stage") == "initial"]
            for channel in CHANNELS
        }
        for channel in CHANNELS:
            if len(initial[channel]) != 1 or initial[channel][0].get("query") != snapshot["query"]:
                raise ValueError(f"Invalid initial event in {snapshot['group_id']}:{channel}")
        actual_events: list[dict[str, Any]] = []

        async def initial_search(channel: str) -> tuple[str, Any]:
            result = await asyncio.to_thread(engine.search, channel, snapshot["query"], self.initial_top_k)
            return channel, result

        initial_results = dict(await asyncio.gather(initial_search("dense"), initial_search("bm25")))
        for channel in CHANNELS:
            actual_events.append(self._actual_event(initial[channel][0], initial_results[channel], chunk_map))
        expansion_queries = {
            channel: [
                str(event["query"]) for event in expected_events
                if event.get("channel") == channel and event.get("stage") == "expansion"
            ]
            for channel in CHANNELS
        }

        async def expansion_search(channel: str) -> tuple[str, Mapping[str, Sequence[Any]]]:
            if not expansion_queries[channel]:
                return channel, {}
            results = await asyncio.to_thread(
                engine.search_many, channel, expansion_queries[channel], self.expansion_quota,
            )
            return channel, results

        expansion_results = dict(await asyncio.gather(expansion_search("dense"), expansion_search("bm25")))
        for channel in CHANNELS:
            expected = [
                event for event in expected_events
                if event.get("channel") == channel and event.get("stage") == "expansion"
            ]
            results = expansion_results[channel]
            for event in expected:
                if event["query"] not in results:
                    raise ValueError(f"Missing search_many result for {event['event_id']}")
                actual_events.append(self._actual_event(event, results[event["query"]], chunk_map))
        actual_by_id = {event["event_id"]: event for event in actual_events}
        if len(actual_by_id) != len(actual_events):
            raise ValueError(f"Duplicate replay event ID in {snapshot['group_id']}")
        mismatches: list[dict[str, Any]] = []
        for expected in expected_events:
            event_id = expected["event_id"]
            actual = actual_by_id.get(event_id)
            if actual is None:
                mismatches.append({"event_id": event_id, "code": "missing_event"})
                continue
            if self._event_identity(expected) != self._event_identity(actual):
                mismatches.append({
                    "event_id": event_id,
                    "code": "event_identity_mismatch",
                    "expected": self._event_identity(expected),
                    "actual": self._event_identity(actual),
                })
            for hit in expected.get("hits", []):
                chunk_id = hit["chunk_id"]
                actual_text = text_sha256(_chunk_text(chunk_map[chunk_id]))
                if hit.get("text_sha256") != actual_text or expected.get("text_sha256", {}).get(chunk_id) != actual_text:
                    mismatches.append({
                        "event_id": event_id, "chunk_id": chunk_id, "code": "text_hash_mismatch",
                    })
        if len(actual_events) != len(expected_events):
            mismatches.append({"code": "event_count_mismatch", "expected": len(expected_events), "actual": len(actual_events)})
        if mismatches:
            raise ValueError(json.dumps({"code": "index_replay_mismatch", "mismatches": mismatches}, ensure_ascii=False))
        return {
            "group_id": snapshot["group_id"],
            "valid": True,
            "event_count": len(actual_events),
            "event_ids": [event["event_id"] for event in actual_events],
            "event_sha256": {
                event["event_id"]: canonical_sha256(self._event_identity(event))
                for event in actual_events
            },
            "text_hashes_valid": True,
            "raw_hit_counts": snapshot.get("raw_hit_counts"),
            "budget_after_replay": budget.snapshot(),
            "snapshot_sha256_before_wide": snapshot["snapshot_sha256"],
        }

    def _wide_targets(self, snapshot: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, int]:
        events = artifact.get("retrieval_events", [])
        targets: dict[str, int] = {}
        for channel in CHANNELS:
            initial = sum(
                len(event.get("hits", [])) for event in events
                if event.get("channel") == channel and event.get("stage") == "initial"
            )
            expansion = sum(
                len(event.get("hits", [])) for event in events
                if event.get("channel") == channel and event.get("stage") == "expansion"
            )
            targets[channel] = initial + expansion
        return targets

    async def _build_wide_artifact(
        self,
        snapshot: Mapping[str, Any],
        expanded: Mapping[str, Any],
    ) -> dict[str, Any]:
        from retrieval.contracts import SearchBudget

        base = snapshot["pools"]["B"]
        active = bool(expanded.get("expansion_active"))
        if not active:
            record = copy.deepcopy(base)
            record["reused_from"] = "B"
            targets = self._wide_targets(snapshot, expanded)
            return {
                "pool_record": record,
                "retrieval_events": [
                    copy.deepcopy(event) for event in snapshot["retrieval_events"]
                    if event.get("stage") == "initial"
                ],
                "raw_hit_counts": {
                    channel: {"initial": targets[channel], "expansion": 0, "total": targets[channel]}
                    for channel in CHANNELS
                },
                "targets": targets,
                "actual_raw_hits": dict(targets),
                "new_raw_hits": {"dense": 0, "bm25": 0},
                "prefix_checks": {"dense": True, "bm25": True},
                "reused_from": "B",
                "new_search_calls": 0,
            }
        targets = self._wide_targets(snapshot, expanded)
        budget = SearchBudget()
        engine = self._get_engine().with_budget(budget)

        async def search(channel: str) -> tuple[str, list[Any]]:
            hits = await asyncio.to_thread(engine.search, channel, snapshot["query"], targets[channel])
            return channel, hits

        results = dict(await asyncio.gather(search("dense"), search("bm25")))
        chunk_map = self._ensure_chunks()
        events: list[dict[str, Any]] = []
        by_channel: dict[str, list[dict[str, Any]]] = {}
        prefix_checks: dict[str, bool] = {}
        errors: list[str] = []
        for channel in CHANNELS:
            expected_initial = next(
                event for event in snapshot["retrieval_events"]
                if event.get("channel") == channel and event.get("stage") == "initial"
            )
            pseudo = {
                "event_id": f"{channel}_wide", "channel": channel, "stage": "wide",
                "query": snapshot["query"], "call_index": 0,
            }
            event = self._actual_event(pseudo, results[channel], chunk_map)
            events.append(event)
            hits = event["hits"]
            by_channel[channel] = hits
            expected_ids = [hit["chunk_id"] for hit in expected_initial["hits"]]
            actual_prefix = [hit["chunk_id"] for hit in hits[: len(expected_ids)]]
            prefix_checks[channel] = (
                actual_prefix == expected_ids
                and [hit["rank"] for hit in hits[: len(expected_ids)]] == list(range(1, len(expected_ids) + 1))
            )
            if not prefix_checks[channel]:
                errors.append(f"{channel} initial hits are not a W prefix")
            if len(hits) != targets[channel]:
                errors.append(f"{channel} W raw hits {len(hits)} != E raw hits {targets[channel]}")
        if errors:
            raise ValueError("; ".join(errors))
        ids: list[str] = []
        seen: set[str] = set()
        for rank in range(max(targets.values(), default=0)):
            for channel in CHANNELS:
                if rank >= len(by_channel[channel]):
                    continue
                chunk_id = by_channel[channel][rank]["chunk_id"]
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    ids.append(chunk_id)
        record = build_pool_record(ids, chunk_map)
        raw_counts = {
            channel: {
                "initial": len(by_channel[channel]), "expansion": 0,
                "total": len(by_channel[channel]),
            }
            for channel in CHANNELS
        }
        return {
            "pool_record": record,
            "retrieval_events": events,
            "raw_hit_counts": raw_counts,
            "targets": targets,
            "actual_raw_hits": {channel: len(by_channel[channel]) for channel in CHANNELS},
            "prefix_checks": prefix_checks,
            "reused_from": None,
            "new_search_calls": 2,
            "search_budget": budget.snapshot(),
        }

    def _commit_snapshots(self, snapshots: Sequence[Mapping[str, Any]]) -> None:
        destination = self.experiment_dir / "snapshots"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.experiment_dir, prefix=".runner-snapshots-"))
        backup: Path | None = None
        try:
            for snapshot in snapshots:
                atomic_write_json(staging / f"{snapshot['group_id']}.json", snapshot)
            if destination.exists():
                backup = self.experiment_dir / f".runner-snapshots-backup-{time.time_ns()}"
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                if backup is not None and backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                    backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        self._snapshots = [copy.deepcopy(dict(snapshot)) for snapshot in snapshots]

    async def validate_indices_and_build_wide(self) -> dict[str, Any]:
        snapshots = copy.deepcopy(self.snapshots)
        report_path = self.experiment_dir / "validation" / "index_replay.json"
        global_errors = self._validate_benchmark_counts()
        if global_errors:
            report = {
                "protocol_version": self.protocol_version,
                "git_commit": self._git_commit,
                "status": "error",
                "valid": False,
                "benchmark_hashes": self.benchmark_hashes(),
                "errors": global_errors,
                "groups": [],
                "summary": {"groups": len(snapshots), "replayed": 0, "wide_written": 0, "error_count": len(global_errors)},
            }
            atomic_write_json(report_path, report)
            raise IndexReplayError(report, report_path)
        async def replay_safe(snapshot: Mapping[str, Any]) -> dict[str, Any]:
            try:
                return await self._replay_group(snapshot)
            except Exception as exc:
                return {
                    "group_id": snapshot["group_id"], "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        replay_results = await self._map_groups(snapshots, replay_safe)
        replay_errors = [result for result in replay_results if not result.get("valid")]
        if replay_errors:
            report = {
                "protocol_version": self.protocol_version,
                "git_commit": self._git_commit,
                "status": "error",
                "valid": False,
                "benchmark_hashes": self.benchmark_hashes(),
                "errors": [],
                "groups": replay_results,
                "summary": {
                    "groups": len(snapshots),
                    "replayed": len(replay_results) - len(replay_errors),
                    "wide_written": 0,
                    "error_count": len(replay_errors),
                },
            }
            atomic_write_json(report_path, report)
            raise IndexReplayError(report, report_path)
        async def wide_safe(snapshot: Mapping[str, Any]) -> dict[str, Any]:
            try:
                wide = await self._build_wide_artifact(snapshot, self._main_pool_artifact(snapshot, "E"))
                return {"group_id": snapshot["group_id"], "valid": True, "wide": wide}
            except Exception as exc:
                return {
                    "group_id": snapshot["group_id"], "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        wide_results = await self._map_groups(snapshots, wide_safe)
        wide_errors = [result for result in wide_results if not result.get("valid")]
        if wide_errors:
            groups = []
            for replay, wide in zip(replay_results, wide_results, strict=True):
                groups.append({**replay, "wide": {key: value for key, value in wide.items() if key != "wide"}})
            report = {
                "protocol_version": self.protocol_version,
                "git_commit": self._git_commit,
                "status": "error",
                "valid": False,
                "benchmark_hashes": self.benchmark_hashes(),
                "errors": [],
                "groups": groups,
                "summary": {
                    "groups": len(snapshots), "replayed": len(snapshots), "wide_written": 0,
                    "error_count": len(wide_errors),
                },
            }
            atomic_write_json(report_path, report)
            raise IndexReplayError(report, report_path)
        group_reports: list[dict[str, Any]] = []
        for snapshot, replay, wide_result in zip(snapshots, replay_results, wide_results, strict=True):
            wide = wide_result["wide"]
            snapshot["pools"]["W"] = wide["pool_record"]
            snapshot["wide"] = {key: value for key, value in wide.items() if key != "pool_record"}
            snapshot["snapshot_sha256"] = snapshot_hash(snapshot)
            group_reports.append({
                **replay,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "wide": {
                    "pool_sha256": wide["pool_record"]["pool_sha256"],
                    "pool_size": len(wide["pool_record"]["chunk_ids"]),
                    "targets": wide["targets"],
                    "actual_raw_hits": wide["actual_raw_hits"],
                    "prefix_checks": wide["prefix_checks"],
                    "reused_from": wide["reused_from"],
                    "new_search_calls": wide["new_search_calls"],
                },
            })
        self._commit_snapshots(snapshots)
        report = {
            "protocol_version": self.protocol_version,
            "git_commit": self._git_commit,
            "implementation_sha256": self._implementation_sha256,
            "retrieval_devices": {"dense": self._dense_device(), "bm25": "cpu"},
            "status": "ok",
            "valid": True,
            "benchmark_hashes": self.benchmark_hashes(),
            "errors": [],
            "groups": group_reports,
            "snapshot_sha256": {snapshot["group_id"]: snapshot["snapshot_sha256"] for snapshot in snapshots},
            "summary": {
                "groups": len(snapshots),
                "replayed": len(snapshots),
                "wide_written": len(snapshots),
                "expansion_active": sum(bool(snapshot["expansion_active"]) for snapshot in snapshots),
                "expansion_inactive": sum(not bool(snapshot["expansion_active"]) for snapshot in snapshots),
                "error_count": 0,
            },
        }
        atomic_write_json(report_path, report)
        return report

    def _wide_is_validated(self) -> bool:
        if any(snapshot.get("pools", {}).get("W") is None for snapshot in self.snapshots):
            return False
        path = self.experiment_dir / "validation" / "index_replay.json"
        if not path.exists():
            return False
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(report, Mapping) or not report.get("valid"):
            return False
        hashes = report.get("snapshot_sha256", {})
        current_device = self._dense_device()
        recorded_device = report.get("retrieval_devices", {}).get("dense")
        return recorded_device == current_device and all(
            hashes.get(snapshot["group_id"]) == snapshot["snapshot_sha256"] for snapshot in self.snapshots
        )

    async def _ensure_wide(self) -> None:
        if not self._wide_is_validated():
            await self.validate_indices_and_build_wide()

    async def _run_candidate_group(self, snapshot: Mapping[str, Any], batch: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pool_id in ("B", "E", "W"):
            results.append(await self._run_rrf(snapshot, pool_id, batch))
        for pool_id in ("B", "E", "W"):
            results.append(await self._run_jina(snapshot, pool_id, batch))
        results.extend(await self._run_legacy(snapshot, batch))
        return results

    async def run_candidate_core(self) -> dict[str, Any]:
        await self._ensure_wide()
        snapshots = list(self.snapshots)
        results = await self._map_groups(snapshots, lambda snapshot: self._run_candidate_group(snapshot, self.execution_batch))
        return {
            "stage": "candidate_core",
            "execution_batch": self.execution_batch,
            "groups": len(snapshots),
            "artifacts": sum(len(group) for group in results),
            "new_planner_calls": 0,
            "new_worker_calls": 0,
            "new_arbiter_calls": 0,
            "config_ids": [
                "B__RRF", "E__RRF", "W__RRF", "B__JINA", "E__JINA", "W__JINA",
                "E__R2P__ARB__LEGACY__strict", "E__R2P__ARB__LEGACY__common_backfill",
                "E__R2P__ARB__LEGACY__legacy_backfill",
            ],
        }

    async def _run_selector_group(self, snapshot: Mapping[str, Any], batch: str) -> list[dict[str, Any]]:
        results = await self._run_arbiter(snapshot, "E", "R0", batch, "E__R0__ARB")
        results.extend(await self._run_arbiter(
            snapshot, "E", "legacy_R2P", batch, "E__LEGACY_R2P__ARB_REPLAY",
        ))
        return results

    async def run_selector_core(self) -> dict[str, Any]:
        snapshots = list(self.snapshots)
        results = await self._map_groups(snapshots, lambda snapshot: self._run_selector_group(snapshot, self.execution_batch))
        return {
            "stage": "selector_core", "execution_batch": self.execution_batch,
            "groups": len(snapshots), "artifacts": sum(len(group) for group in results),
            "config_ids": [
                "E__R0__ARB__strict", "E__R0__ARB__common_backfill",
                "E__LEGACY_R2P__ARB_REPLAY__strict",
                "E__LEGACY_R2P__ARB_REPLAY__common_backfill",
            ],
        }

    async def _run_communication_group(
        self,
        snapshot: Mapping[str, Any],
        batch: str,
        include_r1: bool = True,
        include_r2p: bool = True,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if include_r1:
            results.extend(await self._run_arbiter(snapshot, "E", "R1", batch, "E__R1__ARB"))
        if snapshot.get("expansion_active"):
            await self._run_clean_reports(snapshot, "R2NP", batch)
            if include_r2p:
                await self._run_clean_reports(snapshot, "R2P", batch)
        else:
            await self._run_clean_reports(snapshot, "R2NP", batch)
            if include_r2p:
                await self._run_clean_reports(snapshot, "R2P", batch)
        results.extend(await self._run_arbiter(
            snapshot, "E", "R2NP", batch, "E__R2NP__ARB", clean=True,
        ))
        if include_r2p:
            results.extend(await self._run_arbiter(
                snapshot, "E", "R2P", batch, "E__R2P__ARB_CLEAN", clean=True,
            ))
        results.append(await self._run_wrrf(snapshot, "R2NP", batch, "E__R2NP__WRRF"))
        if include_r2p:
            results.append(await self._run_wrrf(snapshot, "R2P", batch, "E__R2P__WRRF"))
        return results

    async def run_communication(self) -> dict[str, Any]:
        snapshots = list(self.snapshots)
        results = await self._map_groups(
            snapshots, lambda snapshot: self._run_communication_group(snapshot, self.execution_batch),
        )
        return {
            "stage": "communication", "execution_batch": self.execution_batch,
            "groups": len(snapshots), "artifacts": sum(len(group) for group in results),
            "inactive_groups_reused": sum(not bool(snapshot["expansion_active"]) for snapshot in snapshots),
            "config_ids": [
                "E__R1__ARB__strict", "E__R1__ARB__common_backfill",
                "E__R2NP__ARB__strict", "E__R2NP__ARB__common_backfill",
                "E__R2P__ARB_CLEAN__strict", "E__R2P__ARB_CLEAN__common_backfill",
                "E__R2NP__WRRF", "E__R2P__WRRF",
            ],
        }

    def _generation_descriptor(
        self,
        snapshot: Mapping[str, Any],
        kind: str,
        batch: str,
        pool_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        descriptor = self._base_descriptor(
            snapshot, kind, pool_record, f"generation_{kind}", None, "generated_pool", batch,
        )
        models = {
            role: self._safe_model_config(self._get_llm(role), role)
            for role in (("dense", "bm25") if kind == "NP" else ("dense", "bm25", "planner") if kind == "E" else ("dense",))
        }
        descriptor.update({
            "generation_kind": kind,
            "model_configs": models,
            "model_config_sha256": canonical_sha256(models),
            "quota_per_channel": self.expansion_quota,
            "max_queries_per_worker": self.max_queries,
            "timeout_seconds": self.timeout_seconds,
            "planner_prompt_sha256": canonical_sha256(PLANNER_SYSTEM_PROMPT) if kind == "E" else None,
            "generic_plan_sha256": canonical_sha256(generic_plan(snapshot["query"])) if kind == "NP" else None,
            "prompt_sha256": canonical_sha256({
                "kind": kind, "query": snapshot["query"], "B": snapshot["pools"]["B"]["pool_sha256"],
                "implementation": self._implementation_sha256,
            }),
        })
        return descriptor

    def _save_generated(
        self,
        snapshot: Mapping[str, Any],
        batch: str,
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        path = self._generated_path(batch, snapshot["group_id"])
        existing = self._read_generated(batch, snapshot["group_id"]) or {
            "protocol_version": self.protocol_version,
            "group_id": snapshot["group_id"],
            "query_group": snapshot["query_group"],
            "execution_batch": batch,
            "pools": {},
        }
        pools = copy.deepcopy(dict(existing["pools"]))
        for pool_id, artifact in artifacts.items():
            descriptor = artifact["cache_descriptor"]
            value = {key: copy.deepcopy(item) for key, item in artifact.items() if key not in {"cache_descriptor", "cache_key"}}
            previous = pools.get(pool_id)
            previous_key = previous.get("cache_key") if isinstance(previous, Mapping) else None
            saved = {
                **value,
                "cache_descriptor": copy.deepcopy(descriptor),
                "cache_key": canonical_sha256(descriptor),
            }
            if previous_key is not None and previous_key != saved["cache_key"]:
                saved["invalidated_previous_cache_key"] = previous_key
            pools[pool_id] = saved
        output = {
            **existing,
            "protocol_version": self.protocol_version,
            "group_id": snapshot["group_id"],
            "query_group": snapshot["query_group"],
            "execution_batch": batch,
            "pools": pools,
        }
        atomic_write_json(path, output)

    async def _generate_pool(
        self,
        snapshot: Mapping[str, Any],
        kind: str,
        batch: str,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if kind not in {"E", "S", "NP"}:
            raise ValueError(f"Unsupported generated pool kind: {kind}")
        existing = self._read_generated(batch, snapshot["group_id"])
        if use_cache and existing is not None and kind in existing["pools"]:
            artifact = existing["pools"][kind]
            descriptor = artifact.get("cache_descriptor")
            if not isinstance(descriptor, Mapping) or artifact.get("cache_key") != canonical_sha256(descriptor):
                raise CacheIntegrityError(f"Invalid generated cache descriptor: {snapshot['group_id']}:{kind}")
            expected = self._generation_descriptor(snapshot, kind, batch, artifact["pool_record"])
            if descriptor == expected and artifact["cache_key"] == canonical_sha256(expected):
                assert_same_pool(artifact["pool_record"])
                return copy.deepcopy(artifact)
        replay = self._generation_replay(snapshot, kind, batch)
        chunk_map = self._ensure_chunks()
        engine = self._get_engine()
        if kind == "S":
            generated = await generate_single(
                engine, snapshot, chunk_map, self._get_llm("dense"),
                timeout_seconds=self.timeout_seconds, quota_per_channel=self.expansion_quota,
                max_queries=self.max_queries, replay_call=replay,
            )
        elif kind == "NP":
            generated = await generate_no_planner(
                engine, snapshot, chunk_map, self._get_llm("dense"), self._get_llm("bm25"),
                timeout_seconds=self.timeout_seconds, quota_per_channel=self.expansion_quota,
                replay_call=replay,
            )
        else:
            generated = await generate_coordinated(
                engine, snapshot, chunk_map, self._get_llm("dense"), self._get_llm("bm25"),
                self._get_llm("planner"), timeout_seconds=self.timeout_seconds,
                quota_per_channel=self.expansion_quota, replay_call=replay,
            )
        descriptor = self._generation_descriptor(snapshot, kind, batch, generated["pool_record"])
        generated["cache_descriptor"] = descriptor
        generated["cache_key"] = canonical_sha256(descriptor)
        generated["execution_batch"] = batch
        generated["smoke_only"] = bool(self.smoke_only or batch == "smoke")
        self._save_generated(snapshot, batch, {kind: generated})
        return generated

    async def _generate_wide(
        self,
        snapshot: Mapping[str, Any],
        expanded: Mapping[str, Any],
        batch: str,
    ) -> dict[str, Any]:
        wide = await self._build_wide_artifact(snapshot, expanded)
        pool_id = "W"
        descriptor = self._base_descriptor(
            snapshot, pool_id, wide["pool_record"], "budget_matched_wide", None,
            "generated_pool", batch,
        )
        descriptor.update({
            "expanded_pool_sha256": expanded["pool_record"]["pool_sha256"],
            "expanded_generation_sha256": expanded.get("generation_sha256"),
            "prompt_sha256": None,
            "model_config": {"selector": "original_query_search"},
            "model_config_sha256": canonical_sha256({"selector": "original_query_search"}),
        })
        wide.update({
            "kind": "W",
            "group_id": snapshot["group_id"],
            "query_group": snapshot["query_group"],
            "query": snapshot["query"],
            "question_ids": list(snapshot["question_ids"]),
            "input_snapshot_sha256": snapshot["snapshot_sha256"],
            "expanded_pool_sha256": expanded["pool_record"]["pool_sha256"],
            "expanded_generation_sha256": expanded.get("generation_sha256"),
            "expansion_active": expanded.get("expansion_active", False),
            "fallback_stages": [],
            "fallback_provenance": [],
            "cache_descriptor": descriptor,
            "cache_key": canonical_sha256(descriptor),
            "execution_batch": batch,
        })
        self._save_generated(snapshot, batch, {"W": wide})
        return wide

    async def _run_simplification_group(
        self,
        snapshot: Mapping[str, Any],
        candidate: bool,
        generic: bool,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if candidate:
            for kind in ("S", "NP"):
                await self._generate_pool(snapshot, kind, self.execution_batch)
                results.append(await self._run_jina(snapshot, kind, self.execution_batch))
        if generic:
            for pool_id in ("B", "W"):
                results.extend(await self._run_arbiter(
                    snapshot, pool_id, "R0", self.execution_batch, f"{pool_id}__R0__ARB",
                ))
        return results

    async def run_simplification(
        self,
        candidate: bool = True,
        generic: bool = False,
    ) -> dict[str, Any]:
        if not candidate and not generic:
            return {
                "stage": "simplification", "execution_batch": self.execution_batch,
                "groups": len(self.snapshots), "artifacts": 0, "skipped": True,
            }
        if generic:
            await self._ensure_wide()
        snapshots = list(self.snapshots)
        results = await self._map_groups(
            snapshots,
            lambda snapshot: self._run_simplification_group(snapshot, candidate, generic),
        )
        config_ids: list[str] = []
        if candidate:
            config_ids.extend(["S__JINA", "NP__JINA"])
        if generic:
            config_ids.extend([
                "B__R0__ARB__strict", "B__R0__ARB__common_backfill",
                "W__R0__ARB__strict", "W__R0__ARB__common_backfill",
            ])
        return {
            "stage": "simplification", "execution_batch": self.execution_batch,
            "groups": len(snapshots), "artifacts": sum(len(group) for group in results),
            "candidate": candidate, "generic": generic, "config_ids": config_ids,
        }

    def _smoke_selection(self) -> list[dict[str, Any]]:
        path = self.experiment_dir / "smoke_groups.json"
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Corrupted smoke group file: {exc}") from exc
            ids = value if isinstance(value, list) else value.get("group_ids") if isinstance(value, Mapping) else None
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise ValueError("smoke_groups.json must contain group_ids")
            by_id = {snapshot["group_id"]: snapshot for snapshot in self.snapshots}
            if len(set(ids)) != len(ids) or any(group_id not in by_id for group_id in ids):
                raise ValueError("smoke_groups.json contains duplicate or unknown groups")
            return [by_id[group_id] for group_id in ids]
        selected = select_smoke_groups(self.snapshots, self.questions, self.qrels, self.config)
        value = {
            "protocol_version": self.protocol_version,
            "selection": "sha256(query_group + ':20261127')",
            "group_ids": [snapshot["group_id"] for snapshot in selected],
            "groups": [{
                "group_id": snapshot["group_id"],
                "query_group": snapshot["query_group"],
                "question_ids": list(snapshot["question_ids"]),
                "expansion_active": snapshot["expansion_active"],
            } for snapshot in selected],
        }
        value["selection_sha256"] = canonical_sha256(value)
        atomic_write_json(path, value)
        return selected

    async def run_smoke(self) -> dict[str, Any]:
        await self.validate_indices_and_build_wide()
        selected = self._smoke_selection()
        batch = "smoke" if self.execution_batch == "main" else self.execution_batch
        candidate_results = await self._map_groups(
            selected, lambda snapshot: self._run_candidate_group(snapshot, batch),
        )
        selector_results = await self._map_groups(
            selected, lambda snapshot: self._run_selector_group(snapshot, batch),
        )
        report = {
            "protocol_version": self.protocol_version,
            "status": "ok",
            "smoke_only": True,
            "execution_batch": batch,
            "group_ids": [snapshot["group_id"] for snapshot in selected],
            "groups": len(selected),
            "expansion_active": sum(bool(snapshot["expansion_active"]) for snapshot in selected),
            "expansion_inactive": sum(not bool(snapshot["expansion_active"]) for snapshot in selected),
            "candidate_artifacts": sum(len(group) for group in candidate_results),
            "selector_artifacts": sum(len(group) for group in selector_results),
            "metrics_used_for_tuning": False,
        }
        atomic_write_json(self.experiment_dir / "validation" / "smoke_report.json", report)
        return report

    def _target_kind(self, target_spec: Mapping[str, Any]) -> str:
        value = (
            target_spec.get("target_kind") or target_spec.get("branch")
            or target_spec.get("kind") or target_spec.get("target")
            or target_spec.get("status")
        )
        if isinstance(value, Mapping):
            value = value.get("kind") or value.get("branch") or value.get("target")
        if not isinstance(value, str) or not value:
            raise ValueError("Stability target spec is missing target_kind")
        return value

    def _stability_groups(self, target_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = target_spec.get("group_ids") or target_spec.get("selected_group_ids")
        if ids is None:
            ids = target_spec.get("groups") or target_spec.get("selected_groups")
        if ids is not None and not isinstance(ids, (str, bytes, Mapping)):
            ids = [
                item.get("group_id") if isinstance(item, Mapping) else item
                for item in ids
            ]
        if ids is None:
            return select_stability_groups(self.snapshots, self.questions, self.qrels, self.config)
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ValueError("Stability group_ids must be a sequence")
        by_id = {snapshot["group_id"]: snapshot for snapshot in self.snapshots}
        if len(set(ids)) != len(ids) or any(group_id not in by_id for group_id in ids):
            raise ValueError("Stability groups contain duplicate or unknown IDs")
        return [by_id[str(group_id)] for group_id in ids]

    def _load_required_selector(self, batch: str, group_id: str, config_id: str) -> dict[str, Any]:
        path = self._selector_path(batch, group_id, config_id)
        value = self._matching_cache(path, json.loads(path.read_text(encoding="utf-8")).get("cache_descriptor", {})) if path.exists() else None
        if value is None:
            raise ValueError(f"Required main selector is missing or invalid: {path}")
        return value

    async def _alias_main_arbiter(
        self,
        snapshot: Mapping[str, Any],
        source_base: str,
        target_base: str,
        target_state: str,
        batch: str,
    ) -> list[dict[str, Any]]:
        results = []
        for policy in DERIVED_POLICIES:
            source = self._load_required_selector(
                self.execution_batch, snapshot["group_id"], f"{source_base}__{policy}",
            )
            results.append(self._alias_selector(
                snapshot, source, f"{target_base}__{policy}", "E", batch,
                target_report_state=target_state, target_pool_record=snapshot["pools"]["E"],
            ))
        return results

    async def _stability_candidate_group(
        self,
        snapshot: Mapping[str, Any],
        batch: str,
        kinds: Sequence[str],
    ) -> list[str]:
        generated: dict[str, Any] = {}
        for kind in kinds:
            generated[kind] = await self._generate_pool(snapshot, kind, batch)
        await self._generate_wide(snapshot, generated["E"], batch)
        config_ids = []
        for kind in kinds:
            await self._run_jina(snapshot, kind, batch)
            config_ids.append(f"{kind}__JINA")
        await self._run_jina(snapshot, "W", batch)
        config_ids.append("W__JINA")
        return config_ids

    async def _stability_selector_group(
        self,
        snapshot: Mapping[str, Any],
        batch: str,
        kind: str,
        target_spec: Mapping[str, Any],
    ) -> list[str]:
        config_ids: list[str] = []
        if kind == "coordinated_communication":
            await self._run_communication_group(snapshot, batch, include_r1=False, include_r2p=True)
            return [
                "E__R2NP__ARB__strict", "E__R2P__ARB_CLEAN__strict",
                "E__R2NP__WRRF", "E__R2P__WRRF",
            ]
        if kind == "iterative_selection_without_exchange":
            await self._alias_main_arbiter(
                snapshot, "E__R1__ARB", "E__R1__ARB", "R1", batch,
            )
            await self._run_communication_group(snapshot, batch, include_r1=False, include_r2p=False)
            return ["E__R1__ARB__strict", "E__R2NP__ARB__strict"]
        if kind == "generic_llm_reranking":
            simple = str(
                target_spec.get("simple_pool") or target_spec.get("selected_pool")
                or target_spec.get("selected_route") or target_spec.get("simplest_pool") or "B"
            ).replace("__R0__ARB", "")
            if simple not in {"B", "W"}:
                raise ValueError("Generic stability simple pool must be B or W")
            await self._run_arbiter(snapshot, "E", "R0", batch, "E__R0__ARB")
            await self._run_arbiter(snapshot, simple, "R0", batch, f"{simple}__R0__ARB")
            return ["E__R0__ARB__strict", f"{simple}__R0__ARB__strict"]
        if kind == "inconclusive_full_selector":
            await self._run_arbiter(snapshot, "E", "R0", batch, "E__R0__ARB")
            await self._run_arbiter(
                snapshot, "E", "legacy_R2P", batch, "E__LEGACY_R2P__ARB_REPLAY",
            )
            return ["E__LEGACY_R2P__ARB_REPLAY__strict", "E__R0__ARB__strict"]
        raise ValueError(f"Unsupported stability selector target: {kind}")

    async def run_stability(
        self,
        target_spec: Mapping[str, Any] | str | os.PathLike[str],
    ) -> dict[str, Any]:
        if isinstance(target_spec, (str, os.PathLike)):
            target_spec = json.loads(Path(target_spec).read_text(encoding="utf-8"))
        if not isinstance(target_spec, Mapping):
            raise TypeError("target_spec must be a mapping or JSON path")
        if isinstance(target_spec.get("target"), Mapping):
            target_spec = {**target_spec, **target_spec["target"]}
        kind = self._target_kind(target_spec)
        if kind in {"not_applicable", "simple_hybrid_is_sufficient"}:
            return {
                "stage": "stability", "status": "not_applicable", "target_kind": kind,
                "executions": [{"execution_batch": self.execution_batch, "execution_index": 1}],
                "repeats": [],
            }
        original_kind = kind
        normalized_kind = kind.casefold().replace(" ", "_")
        aliases = {
            "communication": "coordinated_communication",
            "r2np/r2p": "coordinated_communication",
            "iterative": "iterative_selection_without_exchange",
            "r1/r2np": "iterative_selection_without_exchange",
            "candidate": "coordinated_candidate_generation",
            "e/s/np": "coordinated_candidate_generation",
            "simpler": "simpler_candidate_generation",
            "generic": "generic_llm_reranking",
            "full_cached_r2p/r0": "inconclusive_full_selector",
        }
        kind = aliases.get(normalized_kind, kind)
        candidate_kinds: tuple[str, ...] | None = None
        if kind == "coordinated_candidate_generation":
            candidate_kinds = ("E", "S", "NP")
        elif kind == "simpler_candidate_generation":
            simple = str(
                target_spec.get("simplest_system") or target_spec.get("simple_pool")
                or target_spec.get("selected_candidate") or "S"
            ).replace("__JINA", "")
            if simple not in {"S", "NP"}:
                raise ValueError("Simpler candidate stability target must select S or NP")
            candidate_kinds = ("E", simple)
        elif kind == "inconclusive":
            advanced = str(
                target_spec.get("most_advanced") or target_spec.get("advanced_target")
                or target_spec.get("stability_contrast") or target_spec.get("contrast")
                or "full_cached_R2P/R0"
            )
            normalized = advanced.casefold().replace(" ", "")
            if "r2np/r2p" in normalized or "communication" in normalized:
                kind = "coordinated_communication"
            elif "r1/r2np" in normalized:
                kind = "iterative_selection_without_exchange"
            elif "e/s/np" in normalized:
                kind = "coordinated_candidate_generation"
                candidate_kinds = ("E", "S", "NP")
            elif "e/np" in normalized:
                kind = "simpler_candidate_generation"
                candidate_kinds = ("E", "NP")
            elif "e/s" in normalized:
                kind = "simpler_candidate_generation"
                candidate_kinds = ("E", "S")
            else:
                kind = "inconclusive_full_selector"
        if candidate_kinds is not None or kind == "generic_llm_reranking":
            await self._ensure_wide()
        groups = self._stability_groups(target_spec)
        repeat_results: list[dict[str, Any]] = []
        for repeat_index, batch in enumerate(("repeat_1", "repeat_2"), 2):
            async def run(snapshot: Mapping[str, Any]) -> list[str]:
                if candidate_kinds is not None:
                    return await self._stability_candidate_group(snapshot, batch, candidate_kinds)
                return await self._stability_selector_group(snapshot, batch, kind, target_spec)

            configs = await self._map_groups(groups, run)
            config_ids = sorted({config_id for group in configs for config_id in group})
            repeat_results.append({
                "execution_batch": batch,
                "execution_index": repeat_index,
                "groups": len(groups),
                "group_ids": [snapshot["group_id"] for snapshot in groups],
                "config_ids": config_ids,
                "selector_artifacts": len(self.collect_results(batch, group_ids=[snapshot["group_id"] for snapshot in groups])),
                "best_repeat_selected": False,
            })
        result = {
            "stage": "stability",
            "status": "ok",
            "target_kind": original_kind,
            "resolved_target": kind,
            "groups": len(groups),
            "group_ids": [snapshot["group_id"] for snapshot in groups],
            "executions": [{
                "execution_batch": self.execution_batch,
                "execution_index": 1,
                "role": "predeclared_main",
            }, *repeat_results],
            "total_executions": 3,
            "best_repeat_selected": False,
        }
        atomic_write_json(self.experiment_dir / "stability" / "execution_manifest.json", result)
        return result

    def collect_results(
        self,
        execution_batch: str = "main",
        config_ids: Sequence[str] | None = None,
        group_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        root = self.experiment_dir / "selector_results" / execution_batch
        if not root.exists():
            return []
        groups = set(group_ids) if group_ids is not None else None
        requested = list(config_ids) if config_ids is not None else None
        paths: set[Path] = set()
        if requested is None:
            candidates = root.glob("group_*/*.json")
            for path in candidates:
                if groups is None or path.parent.name in groups:
                    paths.add(path)
        else:
            candidate_groups = sorted(groups) if groups is not None else [path.name for path in root.glob("group_*") if path.is_dir()]
            for group_id in candidate_groups:
                for config_id in requested:
                    exact = self._selector_path(execution_batch, group_id, config_id)
                    if exact.exists():
                        paths.add(exact)
                        continue
                    strict = self._selector_path(execution_batch, group_id, f"{config_id}__strict")
                    if not config_id.endswith(("__strict", "__common_backfill", "__legacy_backfill")) and strict.exists():
                        paths.add(strict)
        results: list[dict[str, Any]] = []
        for path in sorted(paths, key=lambda item: (item.parent.name, item.name)):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CacheIntegrityError(f"Corrupted selector result {path}: {exc}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("ranked_chunk_ids"), list):
                raise CacheIntegrityError(f"Incomplete selector result: {path}")
            descriptor = value.get("cache_descriptor")
            if not isinstance(descriptor, Mapping) or value.get("cache_key") != canonical_sha256(descriptor):
                raise CacheIntegrityError(f"Selector cache identity mismatch: {path}")
            results.append(value)
        return results


__all__ = ["CacheIntegrityError", "ExperimentRunner", "IndexReplayError"]
