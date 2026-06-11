import pickle
from pathlib import Path
from typing import Optional
from common.models import Chunk


class ChunkDatabase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._by_id: dict[str, Chunk] = {c.id: c for c in chunks}

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self._by_id.get(chunk_id)

    def get_neighbors(self, chunk_id: str) -> tuple[Optional[Chunk], Optional[Chunk]]:
        chunk = self._by_id.get(chunk_id)
        if not chunk:
            return None, None
        prev_chunk = self._by_id.get(chunk.prev_chunk_id) if chunk.prev_chunk_id else None
        next_chunk = self._by_id.get(chunk.next_chunk_id) if chunk.next_chunk_id else None
        return prev_chunk, next_chunk

    def get_neighbors_recursive(
        self, chunk_id: str, depth: int = 1
    ) -> list[Chunk]:
        result: list[Chunk] = []
        visited = {chunk_id}
        queue = [(chunk_id, 0)]
        while queue:
            current_id, dist = queue.pop(0)
            if dist >= depth:
                continue
            prev_c, next_c = self.get_neighbors(current_id)
            for c in (prev_c, next_c):
                if c and c.id not in visited:
                    visited.add(c.id)
                    result.append(c)
                    queue.append((c.id, dist + 1))
        return result

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "ChunkDatabase":
        with open(path, "rb") as f:
            return pickle.load(f)
