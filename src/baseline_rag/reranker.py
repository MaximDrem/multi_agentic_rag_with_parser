"""Реранкер для baseline RAG (jina-reranker-v3.5, transformers + trust_remote_code)."""

import logging
from typing import Any

import torch
from transformers import AutoModel

from common.config import RERANKER_DEVICE, RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


class JinaReranker:
    """Обёртка над jina-reranker-v3.5: model.rerank(query, documents, top_n).

    Возвращает входные чанки, дополненные полями rerank_score и rank
    (позиция по убыванию relevance_score, поле index ответа модели).
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = RERANKER_DEVICE,
        top_n: int = RERANK_TOP_N,
    ):
        self.top_n = top_n
        logger.info(f"Loading reranker {model_name} on {device}...")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device,
            dtype=dtype,
        )
        self.model.eval()
        logger.info("Reranker ready")

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Реранжирует чанки по релевантности запросу, возвращает топ top_n."""
        if not chunks:
            return []
        documents = [c["text"] for c in chunks]
        results = self.model.rerank(query, documents, top_n=self.top_n)
        reranked: list[dict[str, Any]] = []
        for rank, res in enumerate(results, start=1):
            src = chunks[res["index"]]
            out = dict(src)
            out["rerank_score"] = float(res["relevance_score"])
            out["rank"] = rank
            out["score"] = float(res["relevance_score"])
            reranked.append(out)
        return reranked
