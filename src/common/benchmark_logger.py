"""Логирование запросов для бенчмарк-тестирования RAG.

Для каждого запроса создаётся отдельная папка в logs_benchmark/
(experiment_<дата>_<время>_<pipeline>/) с тремя CSV-файлами:
  requests.csv               — метаданные всего запроса (пишется в конце)
  chunks_raw.csv             — все чанки, извлечённые агентами за каждую итерацию
  orchestrator_decisions.csv — решения оркестратора о продолжении цикла
"""

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parents[2] / "var" / "logs_benchmark"

REQUESTS_CSV_FIELDS = [
    "request_id",
    "question_id",
    "question",
    "pipeline_type",
    "iterations_done",
    "total_chunks_collected",
    "final_answer",
    "total_latency_ms",
    "total_tokens",
    "generator_model",
    "timestamp",
]

CHUNKS_CSV_FIELDS = [
    "request_id",
    "iteration",
    "chunk_id",
    "chunk_text",
    "source_agent",
    "tool_score",
    "was_selected_final",
    "final_rank",
]

DECISIONS_CSV_FIELDS = [
    "request_id",
    "iteration",
    "unique_chunks_so_far",
    "decision",
    "reason",
]

_req_counter = 0
_req_counter_lock = threading.Lock()


def next_request_id() -> str:
    global _req_counter
    with _req_counter_lock:
        _req_counter += 1
        return f"req_{_req_counter:03d}"


def tokens_from_usage_metadata(usage_metadata: Any) -> int:
    """Суммарное число токенов из usage_metadata (langchain / OpenAI-формат)."""
    if not usage_metadata:
        return 0
    if isinstance(usage_metadata, dict):
        total = usage_metadata.get("total_tokens")
        if total:
            return int(total)
        inp = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0)) or 0
        out = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0)) or 0
        return int(inp) + int(out)
    return 0


def sum_tokens_from_messages(messages: Iterable[Any]) -> int:
    """Сумма токенов по всем сообщениям (берётся usage_metadata каждого)."""
    total = 0
    for msg in messages:
        total += tokens_from_usage_metadata(getattr(msg, "usage_metadata", None))
    return total


class BenchmarkLogger:
    """Записывает бенчмарк-логи одного запроса в отдельную папку logs_benchmark/."""

    def __init__(
        self,
        request_id: str,
        question_id: str = "",
        question: str = "",
        pipeline_type: str = "agent",
    ):
        self.request_id = request_id
        self.question_id = question_id
        self.question = question
        self.pipeline_type = pipeline_type
        self.start_time = datetime.now()

        folder = self.start_time.strftime(
            f"experiment_%Y-%m-%d_%-H-%M-%S_{pipeline_type}"
        )
        self.dir = LOGS_DIR / folder
        self.dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._requests_path = self.dir / "requests.csv"
        self._chunks_path = self.dir / "chunks_raw.csv"
        self._decisions_path = self.dir / "orchestrator_decisions.csv"
        self._write_header(self._requests_path, REQUESTS_CSV_FIELDS)
        self._write_header(self._chunks_path, CHUNKS_CSV_FIELDS)
        self._write_header(self._decisions_path, DECISIONS_CSV_FIELDS)

        self._chunk_rows: list[dict[str, Any]] = []
        self._unique_chunk_ids: set[str] = set()

    @staticmethod
    def _write_header(path: Path, fields: list[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_rows(self, path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for row in rows:
                    writer.writerow(row)

    @property
    def unique_chunk_count(self) -> int:
        return len(self._unique_chunk_ids)

    @staticmethod
    def _chunk_csv_row(
        request_id: str,
        iteration: int,
        chunk_id: str,
        chunk_text: str,
        source_agent: str,
        tool_score: Any,
        was_selected_final: bool,
        final_rank: int | None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "iteration": iteration,
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "source_agent": source_agent,
            "tool_score": tool_score if tool_score is not None else "NULL",
            "was_selected_final": "TRUE" if was_selected_final else "FALSE",
            "final_rank": final_rank if final_rank is not None else "NULL",
        }

    def log_chunks(self, iteration: int, source_agent: str, chunks: Iterable[dict[str, Any]]) -> None:
        """Все чанки, извлечённые одним агентом за одну итерацию (FALSE, rank=NULL)."""
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            self._unique_chunk_ids.add(cid)
            rows.append({
                "request_id": self.request_id,
                "iteration": iteration,
                "chunk_id": cid,
                "chunk_text": str(chunk.get("text", ""))[:200],
                "source_agent": source_agent,
                "tool_score": chunk.get("score"),
            })
        if rows:
            self._chunk_rows.extend(rows)
            self._append_rows(self._chunks_path, CHUNKS_CSV_FIELDS, [
                self._chunk_csv_row(**row, was_selected_final=False, final_rank=None)
                for row in rows
            ])

    def log_final_selection(self, chunks: Iterable[dict[str, Any]]) -> None:
        """После финального отбора: дописывает те же чанки с was_selected_final=TRUE и рангом."""
        selected_ids: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                selected_ids.append(cid)
        if not selected_ids:
            return

        by_id: dict[str, list[dict[str, Any]]] = {}
        for row in self._chunk_rows:
            by_id.setdefault(row["chunk_id"], []).append(row)

        rows: list[dict[str, Any]] = []
        for rank, cid in enumerate(selected_ids, start=1):
            occurrences = by_id.get(cid)
            if not occurrences:
                occurrences = [{
                    "request_id": self.request_id,
                    "iteration": 0,
                    "chunk_id": cid,
                    "chunk_text": "",
                    "source_agent": "unknown",
                    "tool_score": None,
                }]
            for occ in occurrences:
                rows.append(self._chunk_csv_row(
                    request_id=occ["request_id"],
                    iteration=occ["iteration"],
                    chunk_id=cid,
                    chunk_text=occ["chunk_text"],
                    source_agent=occ["source_agent"],
                    tool_score=occ["tool_score"],
                    was_selected_final=True,
                    final_rank=rank,
                ))
        self._append_rows(self._chunks_path, CHUNKS_CSV_FIELDS, rows)

    def log_decision(self, iteration: int, unique_chunks_so_far: int, decision: str, reason: str = "") -> None:
        self._append_rows(self._decisions_path, DECISIONS_CSV_FIELDS, [{
            "request_id": self.request_id,
            "iteration": iteration,
            "unique_chunks_so_far": unique_chunks_so_far,
            "decision": decision,
            "reason": reason,
        }])

    def finalize_request(
        self,
        iterations_done: int,
        total_chunks_collected: int,
        final_answer: str,
        total_latency_ms: int,
        total_tokens: int,
        generator_model: str,
    ) -> None:
        self._append_rows(self._requests_path, REQUESTS_CSV_FIELDS, [{
            "request_id": self.request_id,
            "question_id": self.question_id,
            "question": self.question,
            "pipeline_type": self.pipeline_type,
            "iterations_done": iterations_done,
            "total_chunks_collected": total_chunks_collected,
            "final_answer": final_answer,
            "total_latency_ms": total_latency_ms,
            "total_tokens": total_tokens,
            "generator_model": generator_model,
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }])
        logger.info(f"Benchmark request logged: {self.dir}")
