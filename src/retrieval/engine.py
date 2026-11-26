import copy
import threading
from pathlib import Path

from common.chunk_store import ChunkDatabase
from common.config import BM25_INDEX_PATH, CHUNKS_PATH, FAISS_INDEX_PATH, SCORE_THRESHOLD
from indexing.bm25_indexer import BM25Indexer
from indexing.faiss_indexer import FaissIndexer
from retrieval.contracts import SearchBudget, SearchHit


class RetrievalEngine:
    def __init__(self, budget: SearchBudget | None = None):
        self.db = ChunkDatabase.load(CHUNKS_PATH)
        self.chunk_map = {chunk.id: chunk for chunk in self.db.chunks}
        self.embedding = FaissIndexer()
        faiss_path = Path(FAISS_INDEX_PATH)
        if faiss_path.is_dir():
            faiss_path = faiss_path / "index.faiss"
        self.embedding.load(faiss_path)
        if self.embedding.model.device.type != "cpu":
            self.embedding.model.to("cpu")
        self.bm25 = BM25Indexer()
        self.bm25.load(BM25_INDEX_PATH)
        self.budget = budget or SearchBudget()
        self._dense_lock = threading.Lock()
        self._bm25_lock = threading.Lock()

    def with_budget(self, budget: SearchBudget) -> "RetrievalEngine":
        engine = copy.copy(self)
        engine.budget = budget
        return engine

    def search(self, channel: str, query: str, requested_hits: int) -> list[SearchHit]:
        reserved = self.budget.reserve(channel, requested_hits)
        if reserved <= 0:
            return []
        if channel == "dense":
            with self._dense_lock:
                raw = self.embedding.search(query, top_k=reserved, score_threshold=SCORE_THRESHOLD)
            chunks = self.db.chunks
        else:
            with self._bm25_lock:
                raw = self.bm25.search(query, top_k=reserved, score_threshold=SCORE_THRESHOLD)
            chunks = self.bm25.chunks
        hits = [SearchHit(chunks[index], float(score), rank) for rank, (index, score) in enumerate(raw, 1)]
        self.budget.record(channel, reserved, len(hits))
        return hits

    def search_many(self, channel: str, queries: list[str], total_quota: int) -> dict[str, list[SearchHit]]:
        if not queries:
            return {}
        base, remainder = divmod(max(0, total_quota), len(queries))
        result = {}
        for index, query in enumerate(queries):
            requested = base + int(index < remainder)
            result[query] = self.search(channel, query, requested) if requested else []
        return result
