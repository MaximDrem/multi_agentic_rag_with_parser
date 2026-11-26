import uvicorn

from common.config import RETRIEVER_PORT, RETRIEVER_URL
from common.logging_utils import setup_logging
from retrieval.a2a import create_retriever_server
from retrieval.dynamic_mas import DynamicMASRetriever


def create_server():
    return create_retriever_server("dynamic_mas", RETRIEVER_URL, DynamicMASRetriever())


if __name__ == "__main__":
    setup_logging("dynamic_mas")
    server = create_server()
    uvicorn.run(server.app, host="0.0.0.0", port=RETRIEVER_PORT, log_config=None)
