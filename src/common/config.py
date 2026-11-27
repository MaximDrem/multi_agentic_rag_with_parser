import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGeneration

load_dotenv()

A2A_VERSION = "1.0"
A2A_EXTENSIONS = os.environ.get("A2A_EXTENSIONS", "")


class ChatOpenAIReasoning(ChatOpenAI):
    """ChatOpenAI that preserves reasoning_content from vLLM responses."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            choices = response.choices
        except AttributeError:
            choices = response.get("choices", []) if isinstance(response, dict) else []
        for i, choice in enumerate(choices):
            msg = choice.message if hasattr(choice, "message") else choice.get("message", {})
            reasoning = (
                (msg.reasoning if hasattr(msg, "reasoning") else None)
                or (msg.reasoning_content if hasattr(msg, "reasoning_content") else None)
                or (msg.get("reasoning") if isinstance(msg, dict) else None)
                or (msg.get("reasoning_content") if isinstance(msg, dict) else None)
            )
            if reasoning and i < len(result.generations):
                gen = result.generations[i]
                if isinstance(gen, ChatGeneration):
                    gen.message.additional_kwargs["reasoning"] = reasoning
        return result


def get_reasoning(msg) -> str:
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "thinking", None)
        or (msg.additional_kwargs or {}).get("reasoning", "")
        or (msg.additional_kwargs or {}).get("reasoning_content", "")
        or (msg.additional_kwargs or {}).get("thinking", "")
        or (getattr(msg, "response_metadata", {}) or {}).get("reasoning_content", "")
    )
    return reasoning or ""


def create_llm(
    base_url: str,
    model: str,
    api_key: str = "dummy",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    enable_thinking: bool = False,
) -> ChatOpenAI:
    kwargs: dict = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    if enable_thinking:
        return ChatOpenAIReasoning(**kwargs)
    return ChatOpenAI(**kwargs)


ORCHESTRATOR_PORT = int(os.environ["ORCHESTRATOR_PORT"])
ORCHESTRATOR_URL = os.environ["ORCHESTRATOR_URL"]
ORCHESTRATOR_LLM_BASE_URL = os.environ["ORCHESTRATOR_LLM_BASE_URL"]
ORCHESTRATOR_LLM_MODEL = os.environ["ORCHESTRATOR_LLM_MODEL"]
ORCHESTRATOR_LLM_TEMPERATURE = float(os.environ["ORCHESTRATOR_LLM_TEMPERATURE"])
ORCHESTRATOR_LLM_MAX_TOKENS = int(os.environ["ORCHESTRATOR_LLM_MAX_TOKENS"])
ORCHESTRATOR_ENABLE_THINKING = os.environ["ORCHESTRATOR_ENABLE_THINKING"].lower() == "true"

RETRIEVER_PORT = int(os.environ.get("RETRIEVER_PORT", "8102"))
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", f"http://localhost:{RETRIEVER_PORT}")

SUMMARIZER_PORT = int(os.environ["SUMMARIZER_PORT"])
SUMMARIZER_URL = os.environ["SUMMARIZER_URL"]
SUMMARIZER_LLM_BASE_URL = os.environ["SUMMARIZER_LLM_BASE_URL"]
SUMMARIZER_LLM_MODEL = os.environ["SUMMARIZER_LLM_MODEL"]
SUMMARIZER_LLM_TEMPERATURE = float(os.environ["SUMMARIZER_LLM_TEMPERATURE"])
SUMMARIZER_LLM_MAX_TOKENS = int(os.environ["SUMMARIZER_LLM_MAX_TOKENS"])
SUMMARIZER_ENABLE_THINKING = os.environ["SUMMARIZER_ENABLE_THINKING"].lower() == "true"

SUB_AGENT_URLS: list[str] = [
    u.strip() for u in os.environ["SUB_AGENT_URLS"].split(",") if u.strip()
]

CHUNKS_PATH = Path(os.environ["CHUNKS_PATH"])
FAISS_INDEX_PATH = Path(os.environ["FAISS_INDEX_PATH"])
BM25_INDEX_PATH = Path(os.environ["BM25_INDEX_PATH"])

SCORE_THRESHOLD = float(os.environ["SCORE_THRESHOLD"])

REQUEST_TIMEOUT = float(os.environ["REQUEST_TIMEOUT"])
TASK_POLL_INTERVAL = float(os.environ["TASK_POLL_INTERVAL"])

MARKDOWN_DIR = os.environ["MARKDOWN_DIR"]

RERANKER_MODEL = os.environ["RERANKER_MODEL"]
RERANKER_DEVICE = os.environ["RERANKER_DEVICE"]
RERANK_TOP_N = int(os.environ["RERANK_TOP_N"])
HYBRID_CANDIDATES = int(os.environ["HYBRID_CANDIDATES"])
RRF_K = int(os.environ["RRF_K"])
