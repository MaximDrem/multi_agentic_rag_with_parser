from retrieval.base import BaseRetriever
from retrieval.engine import RetrievalEngine
from retrieval.npa_mas.pipeline import NpaMasPipeline
from retrieval.reranker import JinaReranker


class BaselineRerankerRetriever(BaseRetriever):
    variant = "B1"
    retriever_name = "baseline_reranker"
    fusion_name = "rrf+jina_reranker"

    def __init__(self, engine: RetrievalEngine | None = None, pipeline: NpaMasPipeline | None = None):
        super().__init__(engine=engine, pipeline=pipeline)
        self.reranker = JinaReranker()
        self.pipeline.jina_selector.reranker = self.reranker
        self._rerank_lock = self.pipeline.jina_selector._lock

    def _rerank(self, query, candidates):
        with self._rerank_lock:
            return self.reranker.rerank(query, candidates)
