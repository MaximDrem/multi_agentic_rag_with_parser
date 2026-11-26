import uvicorn

from common.config import RETRIEVER_PORT, RETRIEVER_URL
from common.logging_utils import setup_logging
from retrieval.a2a import create_retriever_server
from retrieval.multiagent_v2 import MultiAgentV2Retriever


def create_server():
    return create_retriever_server("multiagent_retriever_v2", RETRIEVER_URL, MultiAgentV2Retriever())


if __name__ == "__main__":
    setup_logging("multiagent_retriever_v2")
    server = create_server()
    uvicorn.run(server.app, host="0.0.0.0", port=RETRIEVER_PORT, log_config=None)
