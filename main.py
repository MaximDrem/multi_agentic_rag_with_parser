#!/usr/bin/env python3
"""Entry point for the A2A multi-agent RAG system (A2A v1.0 compliant).

Starts the orchestrator agent server. Sub-agents (retrievers, summarizer)
should be started separately via scripts/start_agents.sh.
"""
import argparse
import asyncio
import logging
import signal
import sys

import uvicorn
from a2a.types import AgentSkill

from common.config import ORCHESTRATOR_PORT, ORCHESTRATOR_URL
from common.a2a_server import A2AServer, A2AExecutor
from orchestrator.orchestrator import OrchestratorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")


async def create_orchestrator_server():
    orch = OrchestratorAgent()
    await orch.initialize()

    executor = A2AExecutor(orch.handle_a2a_request)
    skills = [
        AgentSkill(
            id="orchestrate_rag",
            name="Orchestrate multi-agent RAG pipeline",
            description="Receives user queries, discovers sub-agents via A2A agent cards, delegates to retrievers in parallel, forwards results to summarizer via Part.data, and delivers the final answer.",
            input_modes=["text", "data"],
            output_modes=["text", "data"],
            examples=["Найди информацию о бюджетном кодексе", "Какие НПА регулируют финансовые соглашения?", "Подготовь ответ о порядке отчетности"],
            tags=["orchestration", "routing", "multi-agent", "rag", "final", "user-facing"],
        )
    ]
    return A2AServer(
        name="Orchestrator",
        url=ORCHESTRATOR_URL,
        executor=executor,
        skills=skills,
        description="Multi-agent RAG orchestrator (A2A v1.0): coordinates semantic + keyword retrieval and summarization via agent cards.",
    ), orch


def main():
    parser = argparse.ArgumentParser(description="A2A Multi-Agent RAG Orchestrator")
    parser.add_argument("--port", type=int, default=ORCHESTRATOR_PORT, help="Orchestrator server port")
    args = parser.parse_args()

    orch = asyncio.run(create_orchestrator_server())

    logger.info(f"Starting Orchestrator on port {args.port}")
    logger.info(f"Health check: {ORCHESTRATOR_URL}/health")
    logger.info(f"Agent card: {ORCHESTRATOR_URL}/.well-known/agent-card.json")

    agent, orch_instance = orch
    config = uvicorn.Config(agent.app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        server.should_exit = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.run_until_complete(orch_instance.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
