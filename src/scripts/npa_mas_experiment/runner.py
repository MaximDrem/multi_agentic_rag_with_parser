import asyncio
import copy
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from scripts.npa_mas_experiment.annotations import (
    annotation_review_pack,
    annotation_template,
    read_jsonl,
    select_double_annotation,
    validate_independent_annotation_protocol,
    write_jsonl,
)
from scripts.score_retrieval import normalize_query

load_dotenv()

PROTECTED_HASHES = {
    "benchmark_v3_review.xlsx": "141dadb4210cdd5d4685898467fd0440ee9679a854a6ab82b67089ab2d2671b7",
    "coverage.jsonl": "2ee6c4673410c9e17dafeadbcd949527884e64f4a129e75cd9410eb63a33036a",
    "evidence.jsonl": "7a9c6704e6e39428c15f2e36e68adfb9efe4b1c7c1bfc1834f44dcaef26f874f",
    "manifest.json": "bf39fb64fea3bb8e186f15ca4c814953d277fdf32405fac7acf20d7f8e293a36",
    "questions.jsonl": "aff110752180fcbad3323dea034f659fc8f46ce628f714eee7b34e5e0d7c4d23",
    "qrels.jsonl": "a162f4be821ee5c7deb26217fd6d867a6f5f07bc1906e834f222044cd4985212",
    "unresolved.jsonl": "07ff328d358378620cb8d2524c777f0f5fe105a207dc4027fc26a8d52f9960c7",
    "validation.json": "fd7b88d8770cffc9a895e3d417e82524ee6531ba92fa740b560c5ed477f38758",
}


class ProtocolError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


class ExperimentRunner:
    def __init__(self, config_path: Path, pipeline: Any | None = None):
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent.parent
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("protocol_version") != "npa_mas_component_v1":
            raise ProtocolError("Unexpected protocol_version")
        if self.config.get("run", {}).get("adaptive_gates") is not False:
            raise ProtocolError("Adaptive gates are forbidden")
        self.output_root = self._path(self.config["output"]["root"])
        self.audit_root = self.root / "var" / "audit" / "npa_mas_component_v1"
        self.pipeline = pipeline
        self._groups = None

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_groups(self) -> list[dict[str, Any]]:
        if self._groups is not None:
            return self._groups
        benchmark = self.config["benchmark"]
        benchmark_root = self._path(benchmark["root"])
        questions = read_jsonl(benchmark_root / benchmark["questions"])
        qrel_rows = read_jsonl(benchmark_root / benchmark["qrels"])
        question_ids = {row["question_id"] for row in questions}
        if len(question_ids) != len(questions):
            raise ProtocolError("Duplicate question_id")
        grouped = defaultdict(list)
        for row in questions:
            grouped[normalize_query(row["query"])].append(row)
        qrels = defaultdict(dict)
        for row in qrel_rows:
            question_id = row["question_id"]
            if question_id not in question_ids:
                raise ProtocolError(f"Unknown qrel question_id: {question_id}")
            relevance = int(row["relevance"])
            if relevance > 0:
                chunk_id = row["chunk_id"]
                previous = qrels[question_id].get(chunk_id)
                if previous is not None and previous != relevance:
                    raise ProtocolError(f"Conflicting qrel for {question_id}/{chunk_id}")
                qrels[question_id][chunk_id] = relevance
        groups = []
        ordered = sorted(grouped.items())
        for index, (normalized, rows) in enumerate(ordered):
            rows = sorted(rows, key=lambda row: row["source_row"])
            relevant = {}
            for row in rows:
                for chunk_id, relevance in qrels.get(row["question_id"], {}).items():
                    relevant[chunk_id] = max(relevant.get(chunk_id, 0), relevance)
            groups.append({
                "query_group": f"group_{index:04d}",
                "normalized_query": normalized,
                "query": rows[0]["query"],
                "representative_question_id": rows[0]["question_id"],
                "question_ids": [row["question_id"] for row in rows],
                "answerable_question_ids": [row["question_id"] for row in rows if qrels.get(row["question_id"])],
                "questions": rows,
                "relevant": relevant,
                "answerable": bool(relevant),
            })
        self._groups = groups
        return groups

    def prepare_annotation_template(self, output_dir: Path) -> dict[str, Any]:
        groups = self._load_groups()
        output_dir = self._path(output_dir)
        rows = annotation_template(groups)
        selected = select_double_annotation(groups, int(self.config["seed"]))
        by_id = {row["query_group"]: row for row in rows}
        secondary = [by_id[group_id] for group_id in selected]
        write_jsonl(output_dir / "npa_query_types_v1_human_a.template.jsonl", rows)
        write_jsonl(output_dir / "npa_query_types_v1_human_b.template.jsonl", secondary)
        write_jsonl(output_dir / "npa_query_types_v1.final.template.jsonl", rows)
        write_jsonl(output_dir / "npa_query_types_v1_adjudications.template.jsonl", [])
        write_jsonl(output_dir / "npa_query_types_v1_human_a.review_pack.jsonl", annotation_review_pack(groups, rows))
        write_jsonl(output_dir / "npa_query_types_v1_human_b.review_pack.jsonl", annotation_review_pack(groups, secondary))
        atomic_json(output_dir / "npa_query_types_v1.double_annotation_ids.json", {
            "seed": self.config["seed"],
            "groups": len(groups),
            "double_annotation_count": len(selected),
            "query_groups": selected,
        })
        return {
            "valid": True,
            "groups": len(groups),
            "double_annotation_groups": len(selected),
            "output_dir": str(output_dir.relative_to(self.root)) if output_dir.is_relative_to(self.root) else output_dir.name,
            "final_labels_created": False,
            "human_review_required": True,
        }

    def _annotation_paths(self) -> dict[str, Path]:
        values = self.config["annotations"]
        return {
            key: self._path(values[key])
            for key in ("labels", "primary_independent", "secondary_independent", "adjudications", "reliability")
        }

    def _annotation_hashes(self, paths: dict[str, Path]) -> dict[str, str]:
        return {
            key: sha256_file(paths[key])
            for key in ("labels", "primary_independent", "secondary_independent", "adjudications")
        }

    def _annotation_bundle_hashes(self) -> dict[str, str]:
        paths = self._annotation_paths()
        files = {
            "codebook": self._path(self.config["annotations"]["codebook"]),
            "labels": paths["labels"],
            "primary_independent": paths["primary_independent"],
            "secondary_independent": paths["secondary_independent"],
            "adjudications": paths["adjudications"],
            "reliability": paths["reliability"],
        }
        return {key: sha256_file(path) for key, path in files.items()}

    @staticmethod
    def _bundle_sha256(hashes: dict[str, str]) -> str:
        return sha256_bytes(canonical_json(hashes).encode())

    def validate_annotation_inputs(self, second_override: Path | None = None, write_reliability: bool = True) -> dict[str, Any]:
        groups = self._load_groups()
        paths = self._annotation_paths()
        if second_override is not None:
            paths["secondary_independent"] = self._path(second_override)
        required = ("labels", "primary_independent", "secondary_independent", "adjudications")
        missing = [str(paths[key].relative_to(self.root)) for key in required if not paths[key].is_file()]
        if missing:
            return {
                "valid": False,
                "errors": [f"Missing required human annotation file: {path}" for path in missing],
                "final_labels_created": paths["labels"].is_file(),
                "human_review_required": True,
            }
        final_rows = read_jsonl(paths["labels"])
        primary_rows = read_jsonl(paths["primary_independent"])
        secondary_rows = read_jsonl(paths["secondary_independent"])
        adjudications = read_jsonl(paths["adjudications"])
        selected = select_double_annotation(groups, int(self.config["seed"]))
        report = validate_independent_annotation_protocol(
            final_rows,
            primary_rows,
            secondary_rows,
            adjudications,
            groups,
            selected,
            int(self.config["seed"]),
        )
        report["input_hashes"] = self._annotation_hashes(paths)
        report["codebook_sha256"] = sha256_file(self._path(self.config["annotations"]["codebook"]))
        report["human_review_required"] = False
        if report["valid"] and write_reliability:
            atomic_json(paths["reliability"], report)
        return report

    def _annotation_audit(self, require_annotations: bool) -> dict[str, Any]:
        paths = self._annotation_paths()
        if not require_annotations and not paths["labels"].is_file():
            return {"valid": True, "status": "missing_allowed_for_structural_audit", "human_review_required": True}
        report = self.validate_annotation_inputs(write_reliability=False)
        if not report.get("valid"):
            return report
        if not paths["reliability"].is_file():
            report["valid"] = False
            report.setdefault("errors", []).append("Reliability file missing; run annotations-validate")
            return report
        stored = json.loads(paths["reliability"].read_text(encoding="utf-8"))
        if stored != report:
            report["valid"] = False
            report.setdefault("errors", []).append("Reliability file is stale or differs from independent inputs")
        return report

    def _path_manifest(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ProtocolError(f"Missing protected data path: {path}")
        if path.is_file():
            return {"type": "file", "size": path.stat().st_size, "sha256": sha256_file(path)}
        files = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            files.append({
                "path": child.relative_to(path).as_posix(),
                "size": child.stat().st_size,
                "sha256": sha256_file(child),
            })
        return {"type": "directory", "files": files}

    def _chunk_index_manifest(self) -> dict[str, Any]:
        from common.config import BM25_INDEX_PATH, CHUNKS_PATH, FAISS_INDEX_PATH

        return {
            key: self._path_manifest(self._path(path))
            for key, path in (
                ("chunks", CHUNKS_PATH),
                ("faiss_index", FAISS_INDEX_PATH),
                ("bm25_index", BM25_INDEX_PATH),
            )
        }

    def _chunk_store_audit(self) -> dict[str, Any]:
        from common.chunk_store import ChunkDatabase
        from common.config import CHUNKS_PATH

        database = ChunkDatabase.load(self._path(CHUNKS_PATH))
        ids = [chunk.id for chunk in database.chunks]
        if len(ids) != len(set(ids)):
            raise ProtocolError("Duplicate chunk IDs in chunk store")
        known = set(ids)
        referenced = {chunk_id for group in self._load_groups() for chunk_id in group["relevant"]}
        unknown = sorted(referenced - known)
        if unknown:
            raise ProtocolError(f"Qrels reference unknown chunks: {unknown[:5]}")
        return {"chunks": len(ids), "unique_chunk_ids": len(known), "qrel_chunk_ids": len(referenced), "unknown_qrel_chunk_ids": 0}

    def _source_code_hashes(self) -> dict[str, str]:
        paths = [
            self.root / "src" / "retrieval" / "rrf.py",
            self.root / "src" / "retrieval" / "engine.py",
            self.root / "src" / "retrieval" / "reranker.py",
            self.root / "src" / "retrieval" / "contracts.py",
            self.root / "src" / "indexing" / "faiss_indexer.py",
            self.root / "src" / "indexing" / "bm25_indexer.py",
            self.root / "src" / "indexing" / "lemma_bm25.py",
            self.root / "src" / "indexing" / "chunker.py",
            self.root / "src" / "common" / "config.py",
            self.root / "src" / "common" / "models.py",
            self.root / "src" / "common" / "chunk_store.py",
            self.root / "src" / "scripts" / "score_retrieval.py",
        ]
        for directory in (self.root / "src" / "retrieval" / "npa_mas", self.root / "src" / "scripts" / "npa_mas_experiment"):
            if directory.is_dir():
                paths.extend(sorted(directory.rglob("*.py")))
        return {path.relative_to(self.root).as_posix(): sha256_file(path) for path in sorted(set(paths)) if path.is_file()}

    def _prompt_hashes(self) -> dict[str, str]:
        prompt_config = self.config["prompts"]
        prompt_root = self._path(prompt_config["root"])
        return {
            name: sha256_file(prompt_root / filename)
            for name, filename in sorted(prompt_config.items())
            if name != "root"
        }

    def _model_manifest(self) -> dict[str, Any]:
        llm = self.config["llm"]
        model = os.environ.get(llm["model_env"])
        base_url = os.environ.get(llm["base_url_env"])
        if not model or not base_url:
            raise ProtocolError(f"Required LLM environment: {llm['model_env']}, {llm['base_url_env']}")
        return {
            "model": model,
            "model_env": llm["model_env"],
            "base_url_env": llm["base_url_env"],
            "base_url_recorded": False,
            "revision": os.environ.get("NPA_LLM_REVISION"),
            "temperature": llm["temperature"],
            "timeout_s": llm["timeout_s"],
            "max_transient_retries": llm["max_transient_retries"],
            "json_repair_attempts": llm["json_repair_attempts"],
            "max_output_tokens": llm["max_output_tokens"],
        }

    def _retrieval_manifest(self) -> dict[str, Any]:
        from importlib.metadata import PackageNotFoundError, version

        import faiss
        from common.config import RERANKER_DEVICE, RERANKER_MODEL, RERANK_TOP_N, SCORE_THRESHOLD

        packages = {}
        for distribution in ("sentence-transformers", "transformers", "torch", "numpy", "rank-bm25", "pymorphy3"):
            try:
                packages[distribution] = version(distribution)
            except PackageNotFoundError:
                packages[distribution] = None
        return {
            "embedding_model": "jinaai/jina-embeddings-v5-text-small",
            "embedding_provider": "sentence-transformers",
            "embedding_revision": os.environ.get("EMBEDDING_MODEL_REVISION"),
            "embedding_device": "cpu",
            "embedding_query_prompt": "query",
            "embedding_document_prompt": "document",
            "embedding_normalized": True,
            "score_threshold": SCORE_THRESHOLD,
            "bm25_tokenizer": "indexing.lemma_bm25.tokenize_lemma",
            "bm25_positive_scores_only": True,
            "reranker_model": RERANKER_MODEL,
            "reranker_provider": "sentence-transformers-cross-encoder",
            "reranker_revision": os.environ.get("RERANKER_MODEL_REVISION"),
            "reranker_device": RERANKER_DEVICE,
            "reranker_configured_top_n": RERANK_TOP_N,
            "reranker_protocol_permutation": "full_pool_with_configured_top_n_restored",
            "faiss_version": getattr(faiss, "__version__", None),
            "package_versions": packages,
            "urls_recorded": False,
        }

    def audit_inputs(self, require_annotations: bool = True) -> dict[str, Any]:
        groups = self._load_groups()
        benchmark = self.config["benchmark"]
        answerable_rows = sum(len(group["answerable_question_ids"]) for group in groups)
        answerable_groups = sum(group["answerable"] for group in groups)
        rows_without_qrels = sum(
            len(group["question_ids"]) - len(group["answerable_question_ids"])
            for group in groups
        )
        counts = {
            "rows": sum(len(group["question_ids"]) for group in groups),
            "groups": len(groups),
            "answerable_rows": answerable_rows,
            "answerable_groups": answerable_groups,
            "rows_without_qrels": rows_without_qrels,
        }
        expected_counts = {
            "rows": benchmark["expected_rows"],
            "groups": benchmark["expected_groups"],
            "answerable_rows": benchmark["expected_answerable_rows"],
            "answerable_groups": benchmark["expected_answerable_groups"],
            "rows_without_qrels": benchmark["expected_rows_without_qrels"],
        }
        errors = [f"Count {key}={counts[key]} != {expected_counts[key]}" for key in counts if counts[key] != expected_counts[key]]
        benchmark_root = self._path(benchmark["root"])
        protected = {}
        for name, expected in PROTECTED_HASHES.items():
            actual = sha256_file(benchmark_root / name)
            protected[name] = actual
            if actual != expected:
                errors.append(f"Protected hash mismatch: {name}")
        annotations = self._annotation_audit(require_annotations)
        if not annotations.get("valid"):
            errors.extend(annotations.get("errors", ["Annotation audit failed"]))
        annotation_bundle_hashes = self._annotation_bundle_hashes() if annotations.get("valid") and require_annotations else None
        try:
            chunk_index = self._chunk_index_manifest()
            chunk_store = self._chunk_store_audit()
        except Exception as exc:
            chunk_index = None
            chunk_store = None
            errors.append(f"Chunk/index audit failed: {type(exc).__name__}: {exc}")
        try:
            model = self._model_manifest()
        except Exception as exc:
            model = None
            errors.append(str(exc))
        try:
            retrieval_manifest = self._retrieval_manifest()
        except Exception as exc:
            retrieval_manifest = None
            errors.append(f"Retrieval manifest failed: {type(exc).__name__}: {exc}")
        source_hashes = self._source_code_hashes()
        report = {
            "protocol_version": self.config["protocol_version"],
            "valid": not errors,
            "counts": counts,
            "expected_counts": expected_counts,
            "protected_hashes": protected,
            "chunk_index_manifest": chunk_index,
            "chunk_store": chunk_store,
            "model_manifest": model,
            "retrieval_manifest": retrieval_manifest,
            "annotations": annotations,
            "annotations_bundle_hashes": annotation_bundle_hashes,
            "annotations_bundle_sha256": self._bundle_sha256(annotation_bundle_hashes) if annotation_bundle_hashes else None,
            "config_sha256": sha256_file(self.config_path),
            "prompt_hashes": self._prompt_hashes(),
            "source_code_hashes": source_hashes,
            "source_code_sha256": sha256_bytes(canonical_json(source_hashes).encode()),
            "errors": errors,
        }
        atomic_json(self.audit_root / "latest_input_audit.json", report)
        return report

    def _require_audit(self) -> dict[str, Any]:
        report = self.audit_inputs(require_annotations=True)
        if not report["valid"]:
            raise ProtocolError("Input audit failed; see var/audit/npa_mas_component_v1/latest_input_audit.json")
        return report

    def freeze(self, run_id: str) -> dict[str, Any]:
        audit = self._require_audit()
        run_dir = self.output_root / run_id
        freeze_dir = run_dir / "freeze"
        if run_dir.exists():
            raise ProtocolError(f"Run directory already exists: {run_dir}")
        freeze_dir.mkdir(parents=True)
        shutil.copy2(self.config_path, freeze_dir / "protocol.yaml")
        (freeze_dir / "config_sha256.txt").write_text(audit["config_sha256"] + "\n", encoding="utf-8")
        prompt_root = self._path(self.config["prompts"]["root"])
        bundle = freeze_dir / "prompt_bundle"
        shutil.copytree(prompt_root, bundle)
        from scripts.npa_mas_experiment.schema import sha256_directory

        prompt_bundle_sha256 = sha256_directory(bundle)
        (freeze_dir / "prompt_bundle_sha256.txt").write_text(prompt_bundle_sha256 + "\n", encoding="utf-8")
        atomic_json(freeze_dir / "source_code_hashes.json", audit["source_code_hashes"])
        atomic_json(freeze_dir / "benchmark_hashes_before.json", audit["protected_hashes"])
        atomic_json(freeze_dir / "chunk_index_manifest.json", audit["chunk_index_manifest"])
        atomic_json(freeze_dir / "model_manifest.json", audit["model_manifest"])
        atomic_json(freeze_dir / "retrieval_manifest.json", audit["retrieval_manifest"])
        literature_path = self.root / "experiments" / "npa_mas_component_v1_literature.json"
        if literature_path.is_file():
            shutil.copy2(literature_path, freeze_dir / "literature.json")
        annotations = self.config["annotations"]
        shutil.copy2(self._path(annotations["codebook"]), freeze_dir / "annotations_codebook.md")
        shutil.copy2(self._path(annotations["labels"]), freeze_dir / "annotations.jsonl")
        for source_name, target_name in (
            ("primary_independent", "annotations_human_a.jsonl"),
            ("secondary_independent", "annotations_human_b.jsonl"),
            ("adjudications", "annotations_adjudications.jsonl"),
            ("reliability", "annotations_reliability.json"),
        ):
            shutil.copy2(self._path(annotations[source_name]), freeze_dir / target_name)
        annotations_sha256 = sha256_file(self._path(annotations["labels"]))
        (freeze_dir / "annotations_sha256.txt").write_text(annotations_sha256 + "\n", encoding="utf-8")
        annotations_bundle_sha256 = audit["annotations_bundle_sha256"]
        atomic_json(freeze_dir / "annotations_bundle_hashes.json", audit["annotations_bundle_hashes"])
        (freeze_dir / "annotations_bundle_sha256.txt").write_text(annotations_bundle_sha256 + "\n", encoding="utf-8")
        manifest = {
            "protocol_version": self.config["protocol_version"],
            "run_id": run_id,
            "config_sha256": audit["config_sha256"],
            "prompt_bundle_sha256": prompt_bundle_sha256,
            "annotations_sha256": annotations_sha256,
            "annotations_bundle_sha256": annotations_bundle_sha256,
            "source_code_sha256": audit["source_code_sha256"],
            "benchmark_hashes": audit["protected_hashes"],
            "chunk_index_manifest": audit["chunk_index_manifest"],
            "model_manifest": audit["model_manifest"],
            "retrieval_manifest": audit["retrieval_manifest"],
            "variants": self.config["run"]["variants"],
            "groups": len(self._load_groups()),
            "engineering_only": False,
        }
        atomic_json(freeze_dir / "freeze_manifest.json", manifest)
        atomic_json(run_dir / "run_manifest.json", {**manifest, "status": "frozen", "complete": False})
        return {"valid": True, "run_id": run_id, "frozen": True, "groups": len(self._load_groups()), "variants": len(manifest["variants"])}

    def _current_prompt_bundle_sha256(self) -> str:
        from scripts.npa_mas_experiment.schema import sha256_directory

        prompt_root = self._path(self.config["prompts"]["root"])
        return sha256_directory(prompt_root)

    def _freeze_manifest(self, run_id: str) -> dict[str, Any]:
        path = self.output_root / run_id / "freeze" / "freeze_manifest.json"
        if not path.is_file():
            raise ProtocolError(f"Missing freeze manifest for {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_frozen_current(self, run_id: str) -> dict[str, Any]:
        frozen = self._freeze_manifest(run_id)
        audit = self._require_audit()
        actual = {
            "protocol_version": self.config["protocol_version"],
            "config_sha256": audit["config_sha256"],
            "prompt_bundle_sha256": self._current_prompt_bundle_sha256(),
            "annotations_sha256": sha256_file(self._path(self.config["annotations"]["labels"])),
            "annotations_bundle_sha256": self._bundle_sha256(self._annotation_bundle_hashes()),
            "source_code_sha256": audit["source_code_sha256"],
            "benchmark_hashes": audit["protected_hashes"],
            "chunk_index_manifest": audit["chunk_index_manifest"],
            "model_manifest": audit["model_manifest"],
            "retrieval_manifest": audit["retrieval_manifest"],
        }
        mismatches = [key for key, value in actual.items() if frozen.get(key) != value]
        if mismatches:
            raise ProtocolError(f"Frozen resume mismatch: {', '.join(mismatches)}; use a new run ID")
        return frozen

    def _assert_frozen_postprocessing_inputs(self, run_id: str) -> dict[str, Any]:
        frozen = self._freeze_manifest(run_id)
        audit = self._require_audit()
        actual = {
            "config_sha256": audit["config_sha256"],
            "prompt_bundle_sha256": self._current_prompt_bundle_sha256(),
            "annotations_sha256": sha256_file(self._path(self.config["annotations"]["labels"])),
            "annotations_bundle_sha256": self._bundle_sha256(self._annotation_bundle_hashes()),
            "benchmark_hashes": audit["protected_hashes"],
            "chunk_index_manifest": audit["chunk_index_manifest"],
            "model_manifest": audit["model_manifest"],
            "retrieval_manifest": audit["retrieval_manifest"],
        }
        mismatches = [key for key, value in actual.items() if frozen.get(key) != value]
        if mismatches:
            raise ProtocolError(f"Frozen scoring input mismatch: {', '.join(mismatches)}")
        return frozen

    def _select_groups(self, count: int, answerable_only: bool = False) -> list[dict[str, Any]]:
        groups = [group for group in self._load_groups() if group["answerable"] or not answerable_only]
        if count > len(groups):
            raise ProtocolError(f"Cannot select {count} groups from {len(groups)}")
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{group['query_group']}{self.config['seed']}".encode()).hexdigest(),
        )
        return sorted(ordered[:count], key=lambda group: group["query_group"])

    def _run_dir(self, run_id: str) -> Path:
        return self.output_root / run_id

    def _variant_dir(self, run_id: str, variant: str) -> Path:
        return self._run_dir(run_id) / "variants" / variant

    def _group_path(self, run_id: str, variant: str, group_id: str) -> Path:
        return self._variant_dir(run_id, variant) / "groups" / f"{group_id}.json"

    def _resume_key(self, manifest: dict[str, Any], variant: str, group_id: str) -> str:
        value = {
            key: manifest[key]
            for key in (
                "protocol_version",
                "config_sha256",
                "prompt_bundle_sha256",
                "annotations_sha256",
                "annotations_bundle_sha256",
                "source_code_sha256",
                "benchmark_hashes",
                "chunk_index_manifest",
                "model_manifest",
                "retrieval_manifest",
            )
        }
        value.update({"variant": variant, "query_group": group_id})
        for key in ("core_run_id", "stability_repeat", "core_m2_pool_hashes", "selected_groups"):
            if key in manifest:
                value[key] = manifest[key]
        return sha256_bytes(canonical_json(value).encode())

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if isinstance(result, dict):
            return result
        raise ProtocolError(f"Unsupported group result type: {type(result).__name__}")

    def _ranked_ids(self, result: dict[str, Any]) -> list[str]:
        value = result.get("ranked_chunk_ids", result.get("final_ranking", result.get("chunks", [])))
        ranking = []
        for item in value:
            chunk_id = item if isinstance(item, str) else item.get("chunk_id")
            if not chunk_id:
                raise ProtocolError("Final ranking contains an invalid chunk ID")
            if chunk_id in ranking:
                raise ProtocolError(f"Duplicate final chunk ID: {chunk_id}")
            ranking.append(chunk_id)
        if len(ranking) > self.config["retrieval"]["output_chunks"]:
            raise ProtocolError("Final ranking exceeds output cap")
        return ranking

    def _write_group_result(
        self,
        run_id: str,
        variant: str,
        group: dict[str, Any],
        result: Any,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._group_path(run_id, variant, group["query_group"])
        resume_key = self._resume_key(manifest, variant, group["query_group"])
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("resume_key") != resume_key:
                raise ProtocolError(f"Resume key mismatch for {variant}/{group['query_group']}; use a new run ID")
            return stored
        serialized = self._serialize_result(result)
        artifact = {
            **serialized,
            "protocol_version": self.config["protocol_version"],
            "variant": variant,
            "query_group": group["query_group"],
            "representative_question_id": group["representative_question_id"],
            "question_ids": group["question_ids"],
            "representative_query": group["query"],
            "ranked_chunk_ids": self._ranked_ids(serialized),
            "resume_key": resume_key,
            "config_sha256": manifest["config_sha256"],
            "prompt_bundle_sha256": manifest["prompt_bundle_sha256"],
            "annotations_sha256": manifest["annotations_sha256"],
            "source_code_sha256": manifest["source_code_sha256"],
            "model_manifest": manifest["model_manifest"],
            "cache_status": "computed",
        }
        atomic_json(path, artifact)
        return artifact

    def _load_group_result(self, run_id: str, variant: str, group_id: str) -> dict[str, Any]:
        path = self._group_path(run_id, variant, group_id)
        if not path.is_file():
            raise ProtocolError(f"Missing group artifact: {variant}/{group_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _finalize_variant(self, run_id: str, variant: str, groups: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
        rows = []
        errors = []
        fallback_groups = 0
        failed_groups = 0
        detailed_count = 0
        variant_dir = self._variant_dir(run_id, variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        detailed_path = variant_dir / "detailed.jsonl"
        temporary = detailed_path.with_suffix(".jsonl.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for group in groups:
                    artifact = self._load_group_result(run_id, variant, group["query_group"])
                    detailed_count += 1
                    handle.write(canonical_json(artifact) + "\n")
                    fallback_groups += bool(artifact.get("fallback_used"))
                    failed = bool(artifact.get("error")) or artifact.get("status") == "failed"
                    failed_groups += failed
                    ranking = self._ranked_ids(artifact)
                    for question in group["questions"]:
                        rows.append({
                            "question_id": question["question_id"],
                            "source_row": question["source_row"],
                            "query": question["query"],
                            "query_group": group["query_group"],
                            "ranked_chunk_ids": ranking,
                            "fallback_used": bool(artifact.get("fallback_used")),
                            "strict_failed": bool(artifact.get("strict_failed")),
                            "error": artifact.get("error"),
                        })
                    if failed:
                        errors.append({
                            "query_group": group["query_group"],
                            "variant": variant,
                            "error": artifact.get("error", "failed"),
                        })
            temporary.replace(detailed_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        write_jsonl(variant_dir / "run.jsonl", rows)
        write_jsonl(variant_dir / "errors.jsonl", errors)
        summary = {
            "protocol_version": self.config["protocol_version"],
            "run_id": run_id,
            "variant": variant,
            "groups": detailed_count,
            "rows": len(rows),
            "expected_groups": len(groups),
            "expected_rows": sum(len(group["question_ids"]) for group in groups),
            "fallback_groups": fallback_groups,
            "failed_groups": failed_groups,
            "config_sha256": manifest["config_sha256"],
            "prompt_bundle_sha256": manifest["prompt_bundle_sha256"],
            "annotations_sha256": manifest["annotations_sha256"],
            "source_code_sha256": manifest["source_code_sha256"],
            "complete": detailed_count == len(groups),
        }
        atomic_json(variant_dir / "manifest.json", summary)
        return summary

    def _pipeline_config(self) -> Any:
        from retrieval.npa_mas.schemas import (
            LLMSettings,
            NpaMasConfig,
            PlannerSettings,
            ReportSettings,
            RetrievalSettings,
        )

        retrieval_values = {
            key: value
            for key, value in self.config["retrieval"].items()
            if key in RetrievalSettings.model_fields
        }
        retrieval_values["selector_text_max_chars"] = self.config["selector"]["text_max_chars"]
        llm_values = {
            key: value
            for key, value in self.config["llm"].items()
            if key in LLMSettings.model_fields
        }
        return NpaMasConfig(
            protocol_version=self.config["protocol_version"],
            retrieval=RetrievalSettings.model_validate(retrieval_values),
            planner=PlannerSettings.model_validate(self.config["planner"]),
            reports=ReportSettings.model_validate(self.config["reports"]),
            llm=LLMSettings.model_validate(llm_values),
        )

    def _get_pipeline(self) -> Any:
        if self.pipeline is None:
            from retrieval.npa_mas.pipeline import NpaMasPipeline

            self.pipeline = NpaMasPipeline(
                config=self._pipeline_config(),
                prompts_dir=self._path(self.config["prompts"]["root"]),
            )
        return self.pipeline

    def _runtime_group(self, group: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in group.items()
            if key not in {"relevant", "answerable", "answerable_question_ids"}
        }

    def _existing_group_result(
        self,
        run_id: str,
        variant: str,
        group_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any] | None:
        path = self._group_path(run_id, variant, group_id)
        if not path.exists():
            return None
        stored = json.loads(path.read_text(encoding="utf-8"))
        expected = self._resume_key(manifest, variant, group_id)
        if stored.get("resume_key") != expected:
            raise ProtocolError(f"Resume key mismatch for {variant}/{group_id}; use a new run ID")
        return stored

    def _failure_result(self, variant: str, group: dict[str, Any], exc: Exception) -> dict[str, Any]:
        return {
            "protocol_version": self.config["protocol_version"],
            "variant": variant,
            "query": group["query"],
            "query_group": group["query_group"],
            "status": "failed",
            "strict_failed": True,
            "fallback_used": False,
            "fallback_stage": None,
            "fallback_reason": None,
            "fallbacks": [],
            "ranked_chunk_ids": [],
            "full_ranked_chunk_ids": [],
            "chunks": [],
            "pool": None,
            "plan": None,
            "worker_queries": {},
            "reports": {},
            "calls": [],
            "token_usage": [],
            "stage_timings_ms": {},
            "latency_ms": 0.0,
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "trace_loss": "unhandled_exception",
        }

    async def _execute_variant(
        self,
        run_id: str,
        variant: str,
        groups: list[dict[str, Any]],
        manifest: dict[str, Any],
        pipeline: Any,
        concurrency: int,
        m2_run_id: str | None = None,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        completion_lock = asyncio.Lock()

        async def execute(group: dict[str, Any]) -> None:
            nonlocal completed
            existing = self._existing_group_result(run_id, variant, group["query_group"], manifest)
            if existing is not None:
                async with completion_lock:
                    completed += 1
                return
            async with semaphore:
                started = time.perf_counter()
                try:
                    m2_artifact = None
                    if variant in {"M3", "M4"}:
                        m2_artifact = self._load_group_result(m2_run_id or run_id, "M2", group["query_group"])
                    result = await pipeline.run(
                        group["query"],
                        variant,
                        query_group=group["query_group"],
                        m2_artifact=m2_artifact,
                    )
                except Exception as exc:
                    result = self._failure_result(variant, group, exc)
                    result["latency_ms"] = (time.perf_counter() - started) * 1000
                artifact = self._write_group_result(run_id, variant, group, result, manifest)
                if variant == "M2" and artifact.get("pool") is not None:
                    pool_path = self._run_dir(run_id) / "pools" / "M2" / f"{group['query_group']}.json"
                    atomic_json(pool_path, artifact["pool"])
            async with completion_lock:
                completed += 1
                if completed % 10 == 0 or completed == len(groups):
                    print(f"[{run_id}] {variant}: {completed}/{len(groups)}", flush=True)

        await asyncio.gather(*(execute(self._runtime_group(group)) for group in groups))
        return self._finalize_variant(run_id, variant, groups, manifest)

    async def _execute_matrix(
        self,
        run_id: str,
        groups: list[dict[str, Any]],
        manifest: dict[str, Any],
        concurrency: int | None,
    ) -> dict[str, Any]:
        if concurrency is None:
            concurrency = int(self.config["llm"]["concurrency"])
        if concurrency < 1:
            raise ProtocolError("Concurrency must be positive")
        pipeline = self._get_pipeline()
        summaries = {}
        for variant in self.config["run"]["variants"]:
            summaries[variant] = await self._execute_variant(
                run_id,
                variant,
                groups,
                manifest,
                pipeline,
                concurrency,
            )
        run_manifest = {
            **manifest,
            "status": "complete",
            "complete": True,
            "groups": len(groups),
            "rows": sum(len(group["question_ids"]) for group in groups),
            "variant_summaries": summaries,
            "quality_results_opened": False,
        }
        atomic_json(self._run_dir(run_id) / "run_manifest.json", run_manifest)
        return run_manifest

    @staticmethod
    def _complete_usage_sum(usages: list[dict[str, Any]], field: str) -> int | None:
        if not usages or any(usage.get(field) is None for usage in usages):
            return None
        return sum(int(usage[field]) for usage in usages)

    def _technical_reliability(self, run_id: str, groups: list[dict[str, Any]]) -> dict[str, Any]:
        variants = {}
        for variant in self.config["run"]["variants"]:
            artifacts = [self._load_group_result(run_id, variant, group["query_group"]) for group in groups]
            usages = [usage for artifact in artifacts for usage in artifact.get("token_usage", [])]
            llm_usages = [usage for usage in usages if usage.get("token_usage_status") != "not_applicable"]
            reported = [usage for usage in llm_usages if usage.get("token_usage_status") == "reported"]
            budgets = [
                budget
                for artifact in artifacts
                for budget in (artifact.get("pool") or {}).get("budget", {}).values()
            ]
            query_budgets = [query for budget in budgets for query in budget.get("queries", [])]
            variants[variant] = {
                "groups": len(artifacts),
                "failed_groups": sum(item.get("status") == "failed" for item in artifacts),
                "fallback_groups": sum(bool(item.get("fallback_used")) for item in artifacts),
                "llm_stage_traces": sum(len(item.get("calls", [])) for item in artifacts),
                "llm_calls": sum(
                    max(int(call.get("attempts", 0)), len(call.get("token_usage", [])))
                    for item in artifacts for call in item.get("calls", [])
                ),
                "json_repairs": sum(
                    call.get("status") == "success_after_json_repair"
                    for item in artifacts for call in item.get("calls", [])
                ),
                "token_usage_reported_calls": len(reported),
                "token_usage_partial_calls": sum(
                    any(usage.get(field) is None for field in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    for usage in reported
                ),
                "token_usage_missing_calls": sum(usage.get("token_usage_status") == "missing" for usage in llm_usages),
                "prompt_tokens": self._complete_usage_sum(reported, "prompt_tokens"),
                "completion_tokens": self._complete_usage_sum(reported, "completion_tokens"),
                "total_tokens": self._complete_usage_sum(reported, "total_tokens"),
                "requested_raw_hits": sum(budget["requested_raw_hits"] for budget in budgets),
                "actual_raw_hits": sum(budget["actual_raw_hits"] for budget in budgets),
                "saturated_queries": sum(query["status"] == "saturated" for query in query_budgets),
                "empty_queries": sum(query["status"] == "empty" for query in query_budgets),
                "latency_ms_sum": sum(item.get("latency_ms", 0.0) for item in artifacts),
                "monetary_cost": None,
            }
        return {
            "protocol_version": self.config["protocol_version"],
            "run_id": run_id,
            "engineering_only": True,
            "scoring_enabled": False,
            "quality_metrics_computed": False,
            "groups": len(groups),
            "variants": variants,
        }

    async def run_pilot(
        self,
        run_id: str,
        group_ids: list[str] | None = None,
        concurrency: int | None = None,
    ) -> dict[str, Any]:
        audit = self._require_audit()
        if group_ids:
            by_id = {group["query_group"]: group for group in self._load_groups()}
            unknown = sorted(set(group_ids) - set(by_id))
            if unknown:
                raise ProtocolError(f"Unknown pilot groups: {unknown}")
            groups = [by_id[group_id] for group_id in dict.fromkeys(group_ids)]
        else:
            groups = self._select_groups(int(self.config["pilot"]["groups"]))
        run_dir = self._run_dir(run_id)
        manifest = {
            "protocol_version": self.config["protocol_version"],
            "run_id": run_id,
            "engineering_only": True,
            "scoring_enabled": False,
            "config_sha256": audit["config_sha256"],
            "prompt_bundle_sha256": self._current_prompt_bundle_sha256(),
            "annotations_sha256": sha256_file(self._path(self.config["annotations"]["labels"])),
            "annotations_bundle_sha256": self._bundle_sha256(self._annotation_bundle_hashes()),
            "source_code_sha256": audit["source_code_sha256"],
            "benchmark_hashes": audit["protected_hashes"],
            "chunk_index_manifest": audit["chunk_index_manifest"],
            "model_manifest": audit["model_manifest"],
            "retrieval_manifest": audit["retrieval_manifest"],
            "variants": self.config["run"]["variants"],
            "selected_groups": [group["query_group"] for group in groups],
            "selection_override": bool(group_ids),
        }
        if run_dir.exists() and not (run_dir / "run_manifest.json").is_file():
            raise ProtocolError("Pilot directory exists without a run manifest")
        if (run_dir / "run_manifest.json").is_file():
            existing_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            mismatches = [key for key, value in manifest.items() if existing_manifest.get(key) != value]
            if mismatches:
                raise ProtocolError(f"Pilot resume mismatch: {', '.join(mismatches)}; use a new run ID")
        atomic_json(run_dir / "run_manifest.json", {**manifest, "status": "running", "complete": False})
        result = await self._execute_matrix(run_id, groups, manifest, concurrency)
        reliability = self._technical_reliability(run_id, groups)
        atomic_json(run_dir / "reliability.json", reliability)
        result["reliability"] = "reliability.json"
        result["valid"] = True
        return result

    async def run_core(self, run_id: str, concurrency: int | None = None) -> dict[str, Any]:
        manifest = self._assert_frozen_current(run_id)
        groups = self._load_groups()
        atomic_json(self._run_dir(run_id) / "run_manifest.json", {
            **manifest,
            "status": "running",
            "complete": False,
            "engineering_only": False,
            "quality_results_opened": False,
        })
        result = await self._execute_matrix(run_id, groups, manifest, concurrency)
        result["valid"] = True
        return result

    async def run_stability(
        self,
        run_id: str,
        core_run_id: str = "core_001",
        concurrency: int | None = None,
    ) -> dict[str, Any]:
        from scripts.npa_mas_experiment.analysis import summarize_stability, write_csv
        from scripts.npa_mas_experiment.metrics import query_metrics

        manifest = self._assert_frozen_current(core_run_id)
        core_manifest = json.loads((self._run_dir(core_run_id) / "run_manifest.json").read_text(encoding="utf-8"))
        if not core_manifest.get("scored"):
            raise ProtocolError("Stability runs after the complete validated core matrix and scoring")
        groups = self._select_groups(int(self.config["stability"]["groups"]), answerable_only=True)
        selected_group_ids = [group["query_group"] for group in groups]
        core_m2_pool_hashes = {}
        for group in groups:
            m2 = self._load_group_result(core_run_id, "M2", group["query_group"])
            pool = m2.get("pool")
            pool_hash = pool.get("pool_sha256") if isinstance(pool, dict) else None
            if not pool_hash:
                raise ProtocolError(f"Core M2 pool is missing for stability group {group['query_group']}")
            core_m2_pool_hashes[group["query_group"]] = pool_hash
        pipeline = self._get_pipeline()
        concurrency = concurrency if concurrency is not None else int(self.config["llm"]["concurrency"])
        if concurrency < 1:
            raise ProtocolError("Concurrency must be positive")
        repeat_count = int(self.config["stability"]["additional_repeats"])
        variants = ("S", "M1", "M2", "M3", "M4")
        run_dir = self._run_dir(run_id)
        atomic_json(run_dir / "run_manifest.json", {
            **manifest,
            "run_id": run_id,
            "core_run_id": core_run_id,
            "stability": True,
            "selected_groups": selected_group_ids,
            "core_m2_pool_hashes": core_m2_pool_hashes,
            "additional_repeats": repeat_count,
            "selection": "deterministic_hash_before_quality_results",
            "cold_end_to_end_latency_claimed": False,
            "status": "running",
            "complete": False,
        })
        for repeat_index in range(1, repeat_count + 1):
            repeat_id = f"{run_id}/repeat_{repeat_index}"
            repeat_manifest = {
                **manifest,
                "run_id": repeat_id,
                "stability_repeat": repeat_index,
                "core_run_id": core_run_id,
                "selected_groups": selected_group_ids,
                "core_m2_pool_hashes": core_m2_pool_hashes,
            }
            atomic_json(self._run_dir(repeat_id) / "run_manifest.json", {
                **repeat_manifest,
                "status": "running",
                "complete": False,
            })
            for variant in variants:
                await self._execute_variant(
                    repeat_id,
                    variant,
                    groups,
                    repeat_manifest,
                    pipeline,
                    concurrency,
                    m2_run_id=core_run_id,
                )
            atomic_json(self._run_dir(repeat_id) / "run_manifest.json", {
                **repeat_manifest,
                "status": "complete",
                "complete": True,
            })

        runs = {}
        valid_chunk_ids = self._valid_chunk_ids()
        for variant in variants:
            runs[variant] = []
            for current_run in (core_run_id, *(f"{run_id}/repeat_{index}" for index in range(1, repeat_count + 1))):
                values = {}
                for group in groups:
                    artifact = self._slim_artifact(self._load_group_result(current_run, variant, group["query_group"]))
                    artifact["metrics"] = query_metrics(
                        artifact["ranked_chunk_ids"],
                        group["relevant"],
                        valid_chunk_ids=valid_chunk_ids,
                    )
                    values[group["query_group"]] = artifact
                runs[variant].append(values)
        summary = summarize_stability(runs, metric=self.config["statistics"]["primary_metric"])
        summary.update({
            "protocol_version": self.config["protocol_version"],
            "run_id": run_id,
            "core_run_id": core_run_id,
            "selected_groups": selected_group_ids,
            "core_m2_pool_hashes": core_m2_pool_hashes,
            "additional_repeats": repeat_count,
            "independent_primary_n_added": 0,
            "cold_end_to_end_latency_claimed": False,
            "valid": True,
        })
        atomic_json(run_dir / "summary.json", summary)
        write_jsonl(run_dir / "detailed.jsonl", summary["rows"])
        write_csv(run_dir / "stability.csv", summary["variants"])
        atomic_json(run_dir / "run_manifest.json", {
            **manifest,
            "run_id": run_id,
            "core_run_id": core_run_id,
            "stability": True,
            "selected_groups": selected_group_ids,
            "core_m2_pool_hashes": core_m2_pool_hashes,
            "additional_repeats": repeat_count,
            "status": "complete",
            "complete": True,
            "valid": True,
        })
        self.analyze(core_run_id)
        return {
            "valid": True,
            "run_id": run_id,
            "groups": len(groups),
            "additional_repeats": repeat_count,
            "variants": list(variants),
            "independent_primary_n_added": 0,
        }

    def _benchmark(self) -> Any:
        from scripts.npa_mas_experiment.schema import load_benchmark

        benchmark = self.config["benchmark"]
        root = self._path(benchmark["root"])
        return load_benchmark(
            root / benchmark["questions"],
            root / benchmark["qrels"],
            config=self.config,
            strict_audit=True,
        )

    def _valid_chunk_ids(self) -> set[str]:
        from common.chunk_store import ChunkDatabase
        from common.config import CHUNKS_PATH

        return {chunk.id for chunk in ChunkDatabase.load(self._path(CHUNKS_PATH)).chunks}

    def validate_run(
        self,
        run_id: str,
        require_complete: bool = False,
        require_scored: bool = False,
        check_current_source: bool = False,
    ) -> dict[str, Any]:
        from scripts.npa_mas_experiment.packaging import validate_results_pack
        from scripts.npa_mas_experiment.schema import secret_values_from_environ

        benchmark_root = self._path(self.config["benchmark"]["root"])
        after = {name: sha256_file(benchmark_root / name) for name in PROTECTED_HASHES}
        report = validate_results_pack(
            self._run_dir(run_id),
            self.config,
            benchmark=self._benchmark(),
            valid_chunk_ids=self._valid_chunk_ids(),
            protected_hashes_after=after,
            protected_base_dir=benchmark_root,
            source_root=self.root if check_current_source else None,
            secret_values=secret_values_from_environ(os.environ),
            require_complete=require_complete,
            require_scored=require_scored,
        )
        if require_complete:
            frozen = self._freeze_manifest(run_id)
            current_chunk_index = self._chunk_index_manifest()
            chunk_index_valid = current_chunk_index == frozen.get("chunk_index_manifest")
            if not chunk_index_valid:
                report["errors"].append({
                    "code": "chunk_index_hash_mismatch",
                    "path": "freeze/chunk_index_manifest.json",
                    "message": "Chunk or index files differ from the execution freeze",
                })
            report["chunk_index_valid"] = chunk_index_valid
            report["valid"] = not report["errors"]
            report["error_count"] = len(report["errors"])
        if require_scored:
            score_manifest_path = self._run_dir(run_id) / "scores" / "manifest.json"
            analysis_manifest_path = self._run_dir(run_id) / "analysis" / "manifest.json"
            score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8")) if score_manifest_path.is_file() else {}
            analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8")) if analysis_manifest_path.is_file() else {}
            if score_manifest.get("scoring_code_sha256") != self._scoring_code_sha256():
                report["errors"].append({
                    "code": "scoring_code_stale",
                    "path": "scores/manifest.json",
                    "message": "Scores must be recomputed with the current scoring code",
                })
            if analysis_manifest.get("analysis_code_sha256") != self._analysis_code_sha256():
                report["errors"].append({
                    "code": "analysis_code_stale",
                    "path": "analysis/manifest.json",
                    "message": "Analysis must be regenerated with the current analysis code",
                })
            report["valid"] = not report["errors"]
            report["error_count"] = len(report["errors"])
        report["run_id"] = run_id
        report["require_complete"] = require_complete
        atomic_json(self._run_dir(run_id) / "validation.json", report)
        return report

    def _scoring_code_hashes(self) -> dict[str, str]:
        paths = [
            self.root / "src" / "scripts" / "npa_mas_experiment" / "schema.py",
            self.root / "src" / "scripts" / "npa_mas_experiment" / "metrics.py",
            self.root / "src" / "scripts" / "npa_mas_experiment" / "runner.py",
            self.root / "src" / "scripts" / "score_retrieval.py",
        ]
        return {path.relative_to(self.root).as_posix(): sha256_file(path) for path in paths}

    def _analysis_code_hashes(self) -> dict[str, str]:
        paths = [
            self.root / "src" / "scripts" / "npa_mas_experiment" / "statistics.py",
            self.root / "src" / "scripts" / "npa_mas_experiment" / "analysis.py",
            self.root / "src" / "scripts" / "npa_mas_experiment" / "runner.py",
        ]
        return {path.relative_to(self.root).as_posix(): sha256_file(path) for path in paths}

    def _scoring_code_sha256(self) -> str:
        return self._bundle_sha256(self._scoring_code_hashes())

    def _analysis_code_sha256(self) -> str:
        return self._bundle_sha256(self._analysis_code_hashes())

    def _assert_scores_fresh(self, run_id: str) -> None:
        path = self._run_dir(run_id) / "scores" / "manifest.json"
        if not path.is_file():
            raise ProtocolError("Scores manifest is missing; run score first")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("scoring_code_sha256") != self._scoring_code_sha256():
            raise ProtocolError("Scoring code changed; rerun score before analyze or pack")

    def _compute_scores(self, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        from scripts.npa_mas_experiment.metrics import score_variants

        groups = self._load_groups()
        rankings = {}
        for variant in self.config["run"]["variants"]:
            rankings[variant] = {}
            for group in groups:
                artifact = self._load_group_result(run_id, variant, group["query_group"])
                rankings[variant][group["query_group"]] = {
                    key: artifact.get(key)
                    for key in ("ranked_chunk_ids", "status", "fallback_used", "strict_failed", "error")
                }
        benchmark = self._benchmark()
        valid_ids = self._valid_chunk_ids()
        practical = score_variants(
            benchmark,
            rankings,
            output_policy="complete_with_declared_fallback",
            valid_chunk_ids=valid_ids,
        )
        strict = score_variants(
            benchmark,
            rankings,
            output_policy="strict_failure_as_empty",
            valid_chunk_ids=valid_ids,
        )
        return practical, strict

    def _write_score_set(self, run_id: str, scores: dict[str, Any], strict: dict[str, Any]) -> None:
        for variant in self.config["run"]["variants"]:
            primary = scores[variant]
            sensitivity = strict[variant]
            score_dir = self._run_dir(run_id) / "scores" / variant
            write_jsonl(score_dir / "per_query.jsonl", primary["per_query"])
            write_jsonl(score_dir / "per_unique_query.jsonl", primary["per_group"])
            write_jsonl(score_dir / "all_groups.jsonl", primary["all_groups"])
            atomic_json(score_dir / "summary.json", {
                **primary["summary"],
                "strict_sensitivity": {
                    "row_macro": sensitivity["row_macro"],
                    "unique_query_macro": sensitivity["unique_query_macro"],
                },
            })
            strict_dir = score_dir / "strict"
            write_jsonl(strict_dir / "per_query.jsonl", sensitivity["per_query"])
            write_jsonl(strict_dir / "per_unique_query.jsonl", sensitivity["per_group"])
            write_jsonl(strict_dir / "all_groups.jsonl", sensitivity["all_groups"])
            atomic_json(strict_dir / "summary.json", sensitivity["summary"])

    def _load_score_set(self, run_id: str, policy: str = "practical") -> dict[str, Any]:
        scores = {}
        for variant in self.config["run"]["variants"]:
            score_dir = self._run_dir(run_id) / "scores" / variant
            if policy == "strict":
                score_dir = score_dir / "strict"
            summary_path = score_dir / "summary.json"
            if not summary_path.is_file():
                raise ProtocolError(f"Scores missing for {variant}; run score first")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scores[variant] = {
                "summary": summary,
                "per_query": read_jsonl(score_dir / "per_query.jsonl"),
                "per_group": read_jsonl(score_dir / "per_unique_query.jsonl"),
                "all_groups": read_jsonl(score_dir / "all_groups.jsonl"),
                "row_macro": summary["row_macro"],
                "unique_query_macro": summary["unique_query_macro"],
            }
        return scores

    def score(self, run_id: str) -> dict[str, Any]:
        self._assert_frozen_postprocessing_inputs(run_id)
        validation = self.validate_run(run_id, require_complete=True)
        if not validation["valid"]:
            raise ProtocolError(f"Complete raw run validation failed with {validation['error_count']} errors; quality remains closed")
        practical, strict = self._compute_scores(run_id)
        self._write_score_set(run_id, practical, strict)
        manifest_path = self._run_dir(run_id) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"scored": True, "quality_results_opened": True, "status": "scored"})
        atomic_json(manifest_path, manifest)
        atomic_json(self._run_dir(run_id) / "scores" / "manifest.json", {
            "variants": self.config["run"]["variants"],
            "primary_policy": "complete_with_declared_fallback",
            "sensitivity_policy": "strict_failure_as_empty",
            "answerable_groups": practical[self.config["run"]["variants"][0]]["summary"]["answerable_unique_queries"],
            "scoring_code_hashes": self._scoring_code_hashes(),
            "scoring_code_sha256": self._scoring_code_sha256(),
            "complete": True,
        })
        return {
            "valid": True,
            "run_id": run_id,
            "variants": self.config["run"]["variants"],
            "primary": {variant: score["unique_query_macro"] for variant, score in practical.items()},
            "strict_sensitivity": {variant: score["unique_query_macro"] for variant, score in strict.items()},
        }

    @staticmethod
    def _slim_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
        slim = copy.deepcopy(artifact)
        slim["chunks"] = []
        pool = slim.get("pool")
        if isinstance(pool, dict):
            for candidate in pool.get("candidates", []):
                candidate.pop("text", None)
            provenance = pool.get("provenance")
            if isinstance(provenance, dict):
                provenance.pop("observations", None)
        return slim

    def _artifact_map(self, run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        groups = self._load_groups()
        return {
            variant: {
                group["query_group"]: self._slim_artifact(self._load_group_result(run_id, variant, group["query_group"]))
                for group in groups
            }
            for variant in self.config["run"]["variants"]
        }

    @staticmethod
    def _reconstructed_cost_fields(variant: str, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if variant not in {"M3", "M4"}:
            return {}
        usages = []
        calls = 0
        for group, artifact in artifacts.items():
            source_calls = list(artifact.get("calls", []))
            calls += sum(max(int(call.get("attempts", 0)), len(call.get("token_usage", []))) for call in source_calls)
            usages.extend(
                usage
                for call in source_calls
                for usage in call.get("token_usage", [])
                if usage.get("token_usage_status") != "not_applicable"
            )
        result = {"reconstructed_full_system_llm_calls": calls}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result[f"reconstructed_full_system_{field}"] = (
                sum(int(usage[field]) for usage in usages)
                if usages and all(usage.get("token_usage_status") == "reported" and usage.get(field) is not None for usage in usages)
                else None
            )
        result["reconstructed_full_system_usage_coverage"] = (
            sum(usage.get("token_usage_status") == "reported" for usage in usages) / len(usages)
            if usages else None
        )
        return result

    def analyze(self, run_id: str) -> dict[str, Any]:
        from scripts.npa_mas_experiment.analysis import (
            DEFAULT_LIMITATIONS,
            aggregate_cost_reliability,
            core_metrics_table,
            same_pool_ranking_analysis,
            write_analysis_report,
            write_csv,
        )
        from scripts.npa_mas_experiment.metrics import aggregate_candidate_metrics, summarize_candidate_pool
        from scripts.npa_mas_experiment.statistics import (
            confirmatory_contrasts,
            practical_comparisons,
            subgroup_effects,
            subgroup_metric_summary,
        )

        self._assert_frozen_postprocessing_inputs(run_id)
        raw_validation = self.validate_run(run_id, require_complete=True)
        if not raw_validation["valid"]:
            raise ProtocolError("Raw run is invalid; analysis forbidden")
        self._assert_scores_fresh(run_id)
        scores = self._load_score_set(run_id)
        strict_scores = self._load_score_set(run_id, "strict")
        artifacts = self._artifact_map(run_id)
        benchmark = self._benchmark()
        groups = self._load_groups()
        for variant in ("M3", "M4"):
            for group_id, artifact in artifacts[variant].items():
                m2 = artifacts["M2"][group_id]
                candidate_latency = sum(
                    float(m2.get("stage_timings_ms", {}).get(stage, 0.0))
                    for stage in ("query_generation", "candidate_generation")
                )
                artifact["reconstructed_full_system_latency_ms"] = candidate_latency + float(artifact.get("latency_ms", 0.0))
                combined = copy.deepcopy(artifact)
                combined["calls"] = list(m2.get("calls", [])) + list(artifact.get("calls", []))
                artifacts[variant][group_id]["reconstructed_calls_for_cost"] = combined["calls"]

        analysis_dir = self._run_dir(run_id) / "analysis"
        t1 = core_metrics_table(scores)
        candidate_rows = []
        candidate_aggregates = []
        for variant in self.config["run"]["variants"]:
            variant_rows = []
            for group in groups:
                if not group["answerable"]:
                    continue
                group_id = group["query_group"]
                artifact = artifacts[variant][group_id]
                pool = artifact.get("pool")
                if pool is None:
                    row = {
                        "query_group": group_id,
                        "pool_recall@30": 0.0,
                        "relevant_ids": 0,
                        "unique_ids": 0,
                        "qrels": len(group["relevant"]),
                        "error": artifact.get("error", "missing_pool"),
                    }
                else:
                    baseline_pool = (artifacts["B1"][group_id].get("pool") or {}).get("candidates", [])
                    row = summarize_candidate_pool(pool, group["relevant"], baseline_pool=baseline_pool)
                row["variant"] = variant
                variant_rows.append(row)
                candidate_rows.append(row)
            candidate_aggregates.append({"variant": variant, **aggregate_candidate_metrics(variant_rows)})

        t3 = confirmatory_contrasts(scores, config=self.config)
        for row in t3:
            row["output_policy"] = "complete_with_declared_fallback"
        t3_strict = confirmatory_contrasts(strict_scores, config=self.config)
        for row in t3_strict:
            row["output_policy"] = "strict_failure_as_empty"
        practical = practical_comparisons(scores, config=self.config)
        same_pool = same_pool_ranking_analysis(artifacts, benchmark)
        annotations = read_jsonl(self._path(self.config["annotations"]["labels"]))
        stats = self.config["statistics"]
        t5 = subgroup_effects(
            scores,
            annotations,
            metric=stats["primary_metric"],
            bootstrap_iterations=int(stats["bootstrap_iterations"]),
            randomization_iterations=int(stats["randomization_iterations"]),
            confidence_level=float(stats["confidence_level"]),
            seed=int(self.config["seed"]),
        )
        subgroup_metrics = subgroup_metric_summary(scores, annotations)
        cost_rows = []
        reliability_rows = []
        for variant in self.config["run"]["variants"]:
            aggregate = aggregate_cost_reliability(artifacts[variant], variant=variant)
            cost = aggregate["cost"]
            if variant in {"M3", "M4"}:
                combined_artifacts = {}
                for group_id, artifact in artifacts[variant].items():
                    combined_artifacts[group_id] = {
                        **artifact,
                        "calls": artifact["reconstructed_calls_for_cost"],
                    }
                cost.update(self._reconstructed_cost_fields(variant, combined_artifacts))
            cost_rows.append(cost)
            reliability_rows.append({"variant": variant, **aggregate["reliability"]})

        sensitivity_rows = []
        for variant in self.config["run"]["variants"]:
            practical_macro = scores[variant]["unique_query_macro"]
            strict_macro = strict_scores[variant]["unique_query_macro"]
            sensitivity_rows.append({
                "variant": variant,
                **{f"practical_{key}": value for key, value in practical_macro.items()},
                **{f"strict_{key}": value for key, value in strict_macro.items()},
                "strict_recall@10_delta": strict_macro["recall@10"] - practical_macro["recall@10"],
            })

        stability_path = self.output_root / "stability_001" / "summary.json"
        t8 = []
        if stability_path.is_file():
            stability_summary = json.loads(stability_path.read_text(encoding="utf-8"))
            t8 = stability_summary.get("variants", stability_summary.get("rows", []))
            if isinstance(t8, dict):
                t8 = [{"variant": key, **value} for key, value in t8.items()]

        write_csv(analysis_dir / "core_metrics.csv", t1)
        write_csv(analysis_dir / "candidate_generation.csv", candidate_rows)
        write_csv(analysis_dir / "candidate_generation_summary.csv", candidate_aggregates)
        write_csv(analysis_dir / "confirmatory_contrasts.csv", t3)
        write_csv(analysis_dir / "confirmatory_contrasts_strict.csv", t3_strict)
        write_csv(analysis_dir / "practical_comparisons.csv", practical)
        write_csv(analysis_dir / "same_pool_ranking.csv", same_pool["rows"])
        write_csv(analysis_dir / "same_pool_ranking_summary.csv", same_pool["summary"])
        write_csv(analysis_dir / "subgroup_effects.csv", t5)
        write_csv(analysis_dir / "subgroup_metrics.csv", subgroup_metrics)
        write_csv(analysis_dir / "cost.csv", cost_rows)
        write_csv(analysis_dir / "reliability.csv", reliability_rows)
        write_csv(analysis_dir / "output_policy_sensitivity.csv", sensitivity_rows)
        write_csv(analysis_dir / "stability.csv", t8)

        per_group_scores = {
            variant: {row["query_group"]: row for row in score["per_group"]}
            for variant, score in scores.items()
        }
        trace_examples = []
        used_groups = set()
        for row in candidate_rows:
            if row["variant"] == "M2" and row.get("new_relevant_count_vs_baseline", 0) > 0:
                trace_examples.append(f"{row['query_group']}: planner/worker pool добавил новые relevant IDs относительно B1.")
                used_groups.add(row["query_group"])
                break
        for row in candidate_rows:
            channels = row.get("channels", {})
            dense = channels.get("dense", {}) if isinstance(channels, dict) else {}
            if row["query_group"] not in used_groups and dense.get("unique_contribution_count", 0) > 0:
                trace_examples.append(f"{row['query_group']}: Dense channel внёс уникальный relevant candidate.")
                used_groups.add(row["query_group"])
                break
        for group in groups:
            group_id = group["query_group"]
            if not group["answerable"] or group_id in used_groups:
                continue
            delta = per_group_scores["M3"][group_id]["recall@10"] - per_group_scores["M2"][group_id]["recall@10"]
            if delta > 0:
                trace_examples.append(f"{group_id}: LLM-selector улучшил Recall@10 на том же M2 pool.")
                used_groups.add(group_id)
                break
        for group in groups:
            group_id = group["query_group"]
            if not group["answerable"] or group_id in used_groups:
                continue
            delta = per_group_scores["M3"][group_id]["recall@10"] - per_group_scores["M2"][group_id]["recall@10"]
            if delta < 0:
                trace_examples.append(f"{group_id}: LLM-selector ухудшил Recall@10 на том же M2 pool.")
                used_groups.add(group_id)
                break
        for group in groups:
            group_id = group["query_group"]
            if group_id in used_groups:
                continue
            if artifacts["M3"][group_id]["ranked_chunk_ids"] == artifacts["M4"][group_id]["ranked_chunk_ids"]:
                trace_examples.append(f"{group_id}: worker reports не изменили top-10 относительно M3.")
                break

        tables = {
            "T1": t1,
            "T2": candidate_aggregates,
            "T3": [*t3, *t3_strict],
            "T4": same_pool["summary"],
            "T5": t5,
            "T6": cost_rows,
            "T7": reliability_rows,
            "T8": t8,
        }
        write_analysis_report(analysis_dir, tables, trace_examples=trace_examples, limitations=DEFAULT_LIMITATIONS)
        atomic_text(analysis_dir / "trace_examples.md", "# Trace examples\n\n" + "\n".join(f"- {item}" for item in trace_examples) + "\n")
        atomic_text(analysis_dir / "limitations.md", "# Limitations\n\n" + "\n".join(f"- {item}" for item in DEFAULT_LIMITATIONS) + "\n")
        atomic_json(analysis_dir / "tables.json", tables)
        atomic_json(analysis_dir / "manifest.json", {
            "analysis_code_hashes": self._analysis_code_hashes(),
            "analysis_code_sha256": self._analysis_code_sha256(),
            "scoring_code_sha256": self._scoring_code_sha256(),
            "complete": True,
        })
        manifest_path = self._run_dir(run_id) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"analyzed": True, "status": "analyzed"})
        atomic_json(manifest_path, manifest)
        return {"valid": True, "run_id": run_id, "analysis": "analysis", "tables": sorted(tables), "trace_examples": len(trace_examples)}

    def _validate_stability_linkage(self, core_run_id: str, stability_id: str) -> dict[str, Any]:
        errors = []
        stability_dir = self._run_dir(stability_id)
        summary_path = stability_dir / "summary.json"
        if not summary_path.is_file():
            return {"valid": False, "errors": ["Stability summary is missing"], "error_count": 1}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        groups = self._select_groups(int(self.config["stability"]["groups"]), answerable_only=True)
        expected_group_ids = [group["query_group"] for group in groups]
        expected_rows = sum(len(group["question_ids"]) for group in groups)
        expected_core_hashes = {}
        for group in groups:
            pool = self._load_group_result(core_run_id, "M2", group["query_group"]).get("pool")
            expected_core_hashes[group["query_group"]] = pool.get("pool_sha256") if isinstance(pool, dict) else None
        repeat_count = int(self.config["stability"]["additional_repeats"])
        expected_fields = {
            "core_run_id": core_run_id,
            "selected_groups": expected_group_ids,
            "core_m2_pool_hashes": expected_core_hashes,
            "additional_repeats": repeat_count,
            "independent_primary_n_added": 0,
            "cold_end_to_end_latency_claimed": False,
        }
        for key, expected in expected_fields.items():
            if summary.get(key) != expected:
                errors.append(f"Stability summary mismatch: {key}")
        if not summary.get("valid"):
            errors.append("Stability summary is not valid")
        variant_summaries = summary.get("variants", [])
        if len(variant_summaries) != 5:
            errors.append("Stability must summarize S, M1, M2, M3 and M4")
        for row in variant_summaries:
            if row.get("groups") != len(groups) or row.get("additional_repeats") != repeat_count or row.get("independent_primary_n_added") != 0:
                errors.append(f"Invalid stability variant summary: {row.get('variant')}")
        repeat_root = stability_dir
        actual_repeats = sorted(path.name for path in repeat_root.glob("repeat_*") if path.is_dir())
        expected_repeats = [f"repeat_{index}" for index in range(1, repeat_count + 1)]
        if actual_repeats != expected_repeats:
            errors.append("Stability repeat directories do not match the frozen repeat count")
        for repeat_index in range(1, repeat_count + 1):
            repeat_id = f"{stability_id}/repeat_{repeat_index}"
            repeat_dir = self._run_dir(repeat_id)
            manifest_path = repeat_dir / "run_manifest.json"
            if not manifest_path.is_file():
                errors.append(f"Missing repeat manifest: {repeat_id}")
                continue
            repeat_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, expected in (
                ("core_run_id", core_run_id),
                ("stability_repeat", repeat_index),
                ("selected_groups", expected_group_ids),
                ("core_m2_pool_hashes", expected_core_hashes),
                ("complete", True),
            ):
                if repeat_manifest.get(key) != expected:
                    errors.append(f"Repeat {repeat_index} mismatch: {key}")
            for variant in ("S", "M1", "M2", "M3", "M4"):
                variant_dir = self._variant_dir(repeat_id, variant)
                group_files = sorted((variant_dir / "groups").glob("*.json"))
                if {path.stem for path in group_files} != set(expected_group_ids) or len(group_files) != len(groups):
                    errors.append(f"Repeat {repeat_index}/{variant}: group coverage mismatch")
                for required in ("manifest.json", "detailed.jsonl", "run.jsonl", "errors.jsonl"):
                    if not (variant_dir / required).is_file():
                        errors.append(f"Repeat {repeat_index}/{variant}: missing {required}")
                if (variant_dir / "detailed.jsonl").is_file() and len(read_jsonl(variant_dir / "detailed.jsonl")) != len(groups):
                    errors.append(f"Repeat {repeat_index}/{variant}: detailed count mismatch")
                if (variant_dir / "run.jsonl").is_file() and len(read_jsonl(variant_dir / "run.jsonl")) != expected_rows:
                    errors.append(f"Repeat {repeat_index}/{variant}: source row count mismatch")
                for group_id in expected_group_ids:
                    try:
                        artifact = self._load_group_result(repeat_id, variant, group_id)
                    except ProtocolError as exc:
                        errors.append(str(exc))
                        continue
                    expected_key = self._resume_key(repeat_manifest, variant, group_id)
                    if artifact.get("resume_key") != expected_key:
                        errors.append(f"Repeat {repeat_index}/{variant}/{group_id}: resume linkage mismatch")
                    if variant in {"M3", "M4"}:
                        pool = artifact.get("pool")
                        actual_hash = pool.get("pool_sha256") if isinstance(pool, dict) else None
                        if actual_hash != expected_core_hashes[group_id]:
                            errors.append(f"Repeat {repeat_index}/{variant}/{group_id}: not the original core M2 pool")
        return {
            "valid": not errors,
            "errors": errors,
            "error_count": len(errors),
            "core_run_id": core_run_id,
            "stability_run_id": stability_id,
            "groups": len(groups),
            "additional_repeats": repeat_count,
        }

    def pack(self, run_id: str) -> dict[str, Any]:
        from scripts.npa_mas_experiment.packaging import create_results_pack, validate_zip_archive
        from scripts.npa_mas_experiment.schema import secret_values_from_environ

        stability_id = "stability_001"
        stability_dir = self._run_dir(stability_id)
        stability_summary = stability_dir / "summary.json"
        linkage = self._validate_stability_linkage(run_id, stability_id)
        if not linkage["valid"]:
            atomic_json(stability_dir / "validation.json", linkage)
            raise ProtocolError(f"Stability linkage validation failed with {linkage['error_count']} errors")
        atomic_json(stability_dir / "validation.json", linkage)
        self.analyze(run_id)
        validation = self.validate_run(run_id, require_complete=True, require_scored=True)
        if not validation["valid"]:
            raise ProtocolError(f"Scored results validation failed with {validation['error_count']} errors")
        secrets = secret_values_from_environ(os.environ)
        include_paths = []
        for root in (self._run_dir(run_id), stability_dir):
            include_paths.extend(path.relative_to(self.output_root).as_posix() for path in root.rglob("*") if path.is_file())
        pack_manifest_path = self._run_dir(run_id) / "results_pack_manifest.json"
        atomic_json(pack_manifest_path, {
            "protocol_version": self.config["protocol_version"],
            "core_run_id": run_id,
            "stability_run_id": stability_id,
            "core_validation_sha256": sha256_file(self._run_dir(run_id) / "validation.json"),
            "stability_summary_sha256": sha256_file(stability_summary),
            "stability_validation_sha256": sha256_file(stability_dir / "validation.json"),
            "old_evaluation_included": False,
            "pilot_included": False,
        })
        include_paths.append(pack_manifest_path.relative_to(self.output_root).as_posix())
        include_paths = sorted(set(include_paths))
        destination = self.output_root / "results_pack.zip"
        packed = create_results_pack(
            self.output_root,
            destination,
            include_paths=include_paths,
            secret_values=secrets,
        )
        archive_validation = validate_zip_archive(destination, secret_values=secrets)
        if not archive_validation.get("valid"):
            raise ProtocolError("ZIP validation failed")
        report = {
            "valid": True,
            "run_id": run_id,
            "stability_run_id": stability_id,
            "core_validation": validation["valid"],
            "archive_validation": archive_validation,
            **packed,
        }
        atomic_json(self.output_root / "results_pack.validation.json", report)
        return report
