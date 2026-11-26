import uvicorn

from common.config import RETRIEVER_PORT, RETRIEVER_URL
from common.logging_utils import setup_logging
from retrieval.a2a import create_retriever_server
from retrieval.base import BaseRetriever


def create_server():
    return create_retriever_server("base_retriever", RETRIEVER_URL, BaseRetriever())


if __name__ == "__main__":
    setup_logging("base_retriever")
    server = create_server()
    uvicorn.run(server.app, host="0.0.0.0", port=RETRIEVER_PORT, log_config=None)
