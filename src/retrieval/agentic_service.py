from common.config import (
    RETRIEVER_EMB_ENABLE_THINKING,
    RETRIEVER_EMB_LLM_BASE_URL,
    RETRIEVER_EMB_LLM_MAX_TOKENS,
    RETRIEVER_EMB_LLM_MODEL,
    RETRIEVER_EMB_LLM_TEMPERATURE,
    create_llm,
)
from retrieval.agentic import AgenticSearchRuntime
from retrieval.contracts import SearchBudget
from retrieval.engine import RetrievalEngine


class AgenticRetriever:
    def __init__(self, engine: RetrievalEngine | None = None, dynamic_alpha: bool = False):
        self.engine = engine or RetrievalEngine()
        self.dynamic_alpha = dynamic_alpha
        self.llm = create_llm(
            base_url=RETRIEVER_EMB_LLM_BASE_URL,
            model=RETRIEVER_EMB_LLM_MODEL,
            temperature=RETRIEVER_EMB_LLM_TEMPERATURE,
            max_tokens=RETRIEVER_EMB_LLM_MAX_TOKENS,
            enable_thinking=RETRIEVER_EMB_ENABLE_THINKING,
        )

    async def retrieve(self, query: str) -> dict:
        runtime = AgenticSearchRuntime(
            self.llm,
            self.engine.with_budget(SearchBudget()),
            mode="hybrid",
            dynamic_alpha=self.dynamic_alpha,
            selection_limit=10,
            reject_tool_copy=True,
        )
        result = await runtime.retrieve(query)
        result["retriever"] = "agentic_alpha" if self.dynamic_alpha else "agentic"
        return result
