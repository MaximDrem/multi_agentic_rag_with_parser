import threading
from dataclasses import dataclass
from typing import Any

from common.models import Chunk
from common.config import HYBRID_CANDIDATES, RRF_K

MAX_INDEX_HITS = 60
MAX_CHANNEL_HITS = 30
MAX_OBSERVED_CHUNKS = 30
MAX_OBSERVED_TOKENS = 20000
MAX_FUSION_CANDIDATES = HYBRID_CANDIDATES
MAX_OUTPUT_CHUNKS = 10
MULTI_OBSERVED_CHUNKS = 15
MULTI_OBSERVED_TOKENS = 10000
MULTI_SELECTION_LIMIT = 15


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int


class SearchBudget:
    def __init__(self, total: int = MAX_INDEX_HITS, per_channel: int = MAX_CHANNEL_HITS):
        self.total = total
        self.per_channel = per_channel
        self.remaining_total = total
        self.remaining_channels = {"dense": per_channel, "bm25": per_channel}
        self.used_total = 0
        self.used_channels = {"dense": 0, "bm25": 0}
        self._lock = threading.Lock()

    def reserve(self, channel: str, requested: int) -> int:
        with self._lock:
            allowed = max(0, min(requested, self.remaining_total, self.remaining_channels[channel]))
            self.remaining_total -= allowed
            self.remaining_channels[channel] -= allowed
            return allowed

    def refund(self, channel: str, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self.remaining_total += count
            self.remaining_channels[channel] += count

    def record(self, channel: str, reserved: int, actual: int) -> None:
        with self._lock:
            self.used_total += actual
            self.used_channels[channel] += actual
        self.refund(channel, reserved - actual)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_limit": self.total,
                "per_channel_limit": self.per_channel,
                "raw_index_hits": self.used_total,
                "channel_hits": dict(self.used_channels),
                "remaining_total": self.remaining_total,
                "remaining_channels": dict(self.remaining_channels),
            }


class ObservationBudget:
    def __init__(self, max_chunks: int, max_tokens: int):
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
        self.observed_ids: list[str] = []
        self.observed_set: set[str] = set()
        self.observed_tokens = 0
        self._lock = threading.Lock()

    def observe(self, chunks: list[Chunk]) -> list[Chunk]:
        accepted = []
        with self._lock:
            for chunk in chunks:
                if chunk.id in self.observed_set:
                    continue
                if len(self.observed_ids) >= self.max_chunks:
                    continue
                if self.observed_tokens + chunk.token_count > self.max_tokens:
                    continue
                self.observed_ids.append(chunk.id)
                self.observed_set.add(chunk.id)
                self.observed_tokens += chunk.token_count
                accepted.append(chunk)
        return accepted

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunk_limit": self.max_chunks,
                "token_limit": self.max_tokens,
                "llm_observed_unique_chunks": len(self.observed_ids),
                "llm_observed_tokens": self.observed_tokens,
                "observed_chunk_ids": list(self.observed_ids),
            }


def chunk_dict(chunk: Chunk, rank: int, score: float | None = None, relevance: int | None = None, answer_aspect: str = "") -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "doc_path": chunk.doc_path,
        "title": chunk.title,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "score": score,
        "rank": rank,
        "relevance": relevance,
        "answer_aspect": answer_aspect,
    }
