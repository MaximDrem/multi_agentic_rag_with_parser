from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentSkill

from common.a2a_server import A2AExecutor, A2AServer


def create_retriever_server(name: str, url: str, service) -> A2AServer:
    async def handle_request(
        user_text: str,
        user_data: dict | None,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict:
        query = user_text
        if isinstance(user_data, dict) and "query" in user_data:
            query = user_data["query"]
        result = await service.retrieve(query)
        if result.get("error"):
            return {"status": "failed", "message": result["error"], "artifacts": []}
        return {
            "status": "completed",
            "message": f"Retrieved {len(result['chunks'])} chunks",
            "artifacts": [{"artifactId": "retrieved_chunks", "data": result}],
        }

    executor = A2AExecutor(handle_request)
    skills = [AgentSkill(
        id="retrieve",
        name="Unified retrieval",
        description="Возвращает ranked standalone чанки по единому корпусу через A2A. Принимает query в text или Part.data.",
        input_modes=["text", "data"],
        output_modes=["data"],
        examples=["порядок финансирования инвестиционных проектов", "требования к отчётности при получении субсидий"],
        tags=["retrieval", "intermediate", "shared-corpus"],
    )]
    return A2AServer(
        name=name,
        url=url,
        executor=executor,
        skills=skills,
        description="Unified A2A retrieval service returning ranked chunks.",
    )
