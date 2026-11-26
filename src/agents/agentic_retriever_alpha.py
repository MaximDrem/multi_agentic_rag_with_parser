import uvicorn

from common.config import RETRIEVER_PORT, RETRIEVER_URL
from common.logging_utils import setup_logging
from retrieval.a2a import create_retriever_server
from retrieval.agentic_service import AgenticRetriever


def create_server():
    return create_retriever_server("agentic_retriever_alpha", RETRIEVER_URL, AgenticRetriever(dynamic_alpha=True))


if __name__ == "__main__":
    setup_logging("agentic_retriever_alpha")
    server = create_server()
    uvicorn.run(server.app, host="0.0.0.0", port=RETRIEVER_PORT, log_config=None)
