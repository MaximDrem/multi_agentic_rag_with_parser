import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from common.models import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]+", text.lower())


class BM25Indexer:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10, score_threshold: float = 0.0) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(
            [(i, float(s)) for i, s in enumerate(scores)],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [(i, s) for i, s in top_idx if s >= score_threshold]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self.tokenized_corpus = [tokenize(c.text) for c in self.chunks]
