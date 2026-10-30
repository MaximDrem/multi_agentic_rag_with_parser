import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.models import Chunk


class FaissIndexer:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.model: Optional[SentenceTransformer] = None
        self.chunks: list[Chunk] = []

    def load_model(self, model_name: str = "jinaai/jina-embeddings-v5-text-small") -> None:
        print(f"Loading embedding model {model_name} ...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        with torch.no_grad():
            test_emb = self.model.encode(["dimension probe"], task="retrieval")
            self.dim = test_emb.shape[1]
        print(f"Model loaded. Dimension: {self.dim}")

    def build(self, chunks: list[Chunk], batch_size: int = 32) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks in batches of {batch_size} ...")

        self.model.to("cuda")
        embeddings_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(
                batch,
                task="retrieval",
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings_list.append(emb)
        self.model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        embeddings = np.vstack(embeddings_list).astype("float32")

        n = len(chunks)
        if n > 100_000:
            nlist = min(int(np.sqrt(n)), 4096)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            print("Training IVF index ...")
            self.index.train(embeddings)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        print("Adding vectors to FAISS index ...")
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query: str, top_k: int = 10, score_threshold: float = 0.0) -> list[tuple[int, float]]:
        embedding = self.model.encode(
            [query],
            task="retrieval",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding = np.asarray(embedding, dtype="float32")
        scores, indices = self.index.search(embedding, top_k)
        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and score >= score_threshold:
                results.append((int(idx), float(score)))
        return results

    def save(self, index_path: str | Path, meta_path: Optional[str | Path] = None) -> None:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta = {
            "dim": self.dim,
            "ntotal": int(self.index.ntotal),
            "model": getattr(self, "model_name", None),
        }
        meta_path = meta_path or index_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str | Path, model_name: str = "jinaai/jina-embeddings-v5-text-small") -> None:
        self.index = faiss.read_index(str(index_path))
        if self.model is None:
            self.load_model(model_name)
