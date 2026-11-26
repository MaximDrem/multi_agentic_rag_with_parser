import uvicorn

from common.config import RETRIEVER_PORT, RETRIEVER_URL
from common.logging_utils import setup_logging
from retrieval.a2a import create_retriever_server
from retrieval.coordinated import CoordinatedMultiAgentRetriever


def create_server():
    return create_retriever_server("multiagent_coordinated", RETRIEVER_URL, CoordinatedMultiAgentRetriever())


if __name__ == "__main__":
    setup_logging("multiagent_coordinated")
    server = create_server()
    uvicorn.run(server.app, host="0.0.0.0", port=RETRIEVER_PORT, log_config=None)
