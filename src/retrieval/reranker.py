import logging
from typing import Any

import torch
from transformers import AutoModel

from common.config import RERANKER_DEVICE, RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


class JinaReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE, top_n: int = RERANK_TOP_N):
        self.top_n = top_n
        logger.info(f"Loading reranker {model_name} on {device}...")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device, dtype=dtype)
        self.model.eval()
        logger.info("Reranker ready")

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunks:
            return []
        results = self.model.rerank(query, [chunk["text"] for chunk in chunks], top_n=self.top_n)
        reranked = []
        for rank, result in enumerate(results, 1):
            output = dict(chunks[result["index"]])
            output["rerank_score"] = float(result["relevance_score"])
            output["rank"] = rank
            output["score"] = float(result["relevance_score"])
            reranked.append(output)
        return reranked
