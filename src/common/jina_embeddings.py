"""Адаптер jina-эмбеддингов для ragas (современный интерфейс BaseRagasEmbedding).

jina-embeddings-v5 требует task="retrieval" при encode — обычные langchain-обёртки
это не передают, поэтому здесь собственный нативный для ragas адаптер.
Легаси-алиасы embed_query/embed_documents оставлены для кода, ожидающего
langchain-стиль.
"""

import logging
from typing import Any

from ragas.embeddings import BaseRagasEmbedding
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"


class JinaRagasEmbeddings(BaseRagasEmbedding):
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
    ):
        super().__init__()
        logger.info(f"Loading embedding model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        logger.info("Embedding model ready")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            task="retrieval",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in embeddings]

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self._encode([text])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._encode(texts)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts, **kwargs)

    # легаси-алиасы (langchain-стиль)
    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)
