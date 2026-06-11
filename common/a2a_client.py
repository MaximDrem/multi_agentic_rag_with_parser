import asyncio
import logging
import uuid
from typing import Any, AsyncIterator

import httpx
from google.protobuf.json_format import MessageToDict

from a2a.client import create_client, Client, ClientConfig
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageConfiguration,
    StreamResponse,
    TaskState,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    GetTaskRequest,
)
from a2a.utils.errors import A2AError
from a2a.client.errors import A2AClientError
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from a2a.helpers.proto_helpers import (
    new_text_part,
    new_data_part,
)

from common.config import REQUEST_TIMEOUT, A2A_VERSION

logger = logging.getLogger(__name__)


def make_user_message(
    text: str,
    data: dict[str, Any] | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    reference_task_ids: list[str] | None = None,
    extensions: list[str] | None = None,
) -> Message:
    parts: list[Part] = [new_text_part(text, media_type="text/plain")]
    if data:
        parts.append(new_data_part(data, media_type="application/json"))
    return Message(
        role=Role.ROLE_USER,
        parts=parts,
        message_id=f"msg-{uuid.uuid4().hex[:12]}",
        context_id=context_id,
        task_id=task_id,
        reference_task_ids=reference_task_ids,
        extensions=extensions,
    )


async def collect_responses(
    stream: AsyncIterator[StreamResponse],
) -> list[StreamResponse]:
    responses = []
    async for resp in stream:
        responses.append(resp)
    return responses


async def stream_to_dicts(
    stream: AsyncIterator[StreamResponse],
) -> AsyncIterator[dict[str, Any]]:
    async for resp in stream:
        yield MessageToDict(resp)


class A2AClientManager:
    """Manages A2A client connections to remote agents using the a2a-sdk Client."""

    def __init__(self):
        self._clients: dict[str, Client] = {}
        self._agent_cards: dict[str, AgentCard] = {}

    async def get_client(self, agent_url: str) -> Client:
        agent_url = agent_url.rstrip("/")
        if agent_url not in self._clients:
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT),
                headers={"A2A-Version": A2A_VERSION},
            )
            config = ClientConfig(httpx_client=http_client)
            client = await create_client(agent_url, client_config=config)
            self._clients[agent_url] = client
        return self._clients[agent_url]

    async def fetch_agent_card(self, agent_url: str) -> AgentCard:
        agent_url = agent_url.rstrip("/")
        if agent_url not in self._agent_cards:
            import httpx
            from a2a.client.card_resolver import A2ACardResolver
            async with httpx.AsyncClient(timeout=10.0, headers={"A2A-Version": A2A_VERSION}) as http_client:
                resolver = A2ACardResolver(http_client, agent_url)
                card = await resolver.get_agent_card()
                self._verify_protocol_version(card, agent_url)
                self._agent_cards[agent_url] = card
        return self._agent_cards[agent_url]

    @staticmethod
    def _verify_protocol_version(card: AgentCard, agent_url: str) -> None:
        supported_versions = set()
        for iface in card.supported_interfaces:
            supported_versions.add(iface.protocol_version)
        if supported_versions and PROTOCOL_VERSION_1_0 not in supported_versions:
            raise A2AClientError(
                f"Agent at {agent_url} does not support A2A v1.0; "
                f"supported versions: {supported_versions}. Refusing to interact."
            )

    async def send_message(
        self,
        agent_url: str,
        text: str,
        data: dict[str, Any] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        reference_task_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        return await self._send_and_collect(
            agent_url, text, data=data,
            context_id=context_id, task_id=task_id,
            reference_task_ids=reference_task_ids,
            metadata=metadata,
            extensions=extensions,
        )

    async def send_message_and_wait(
        self,
        agent_url: str,
        text: str,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        reference_task_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        t = timeout or REQUEST_TIMEOUT
        return await asyncio.wait_for(
            self.send_message(
                agent_url, text, data=data,
                context_id=context_id, task_id=task_id,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=extensions,
            ),
            timeout=t,
        )

    async def get_task(self, agent_url: str, task_id: str) -> dict[str, Any]:
        """Get task status and artifacts (spec 3.1.3)."""
        client = await self.get_client(agent_url)
        request = GetTaskRequest(id=task_id)
        result = await client.get_task(request)
        return MessageToDict(result)

    async def list_tasks(
        self,
        agent_url: str,
        context_id: str | None = None,
        limit: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List tasks with optional filtering (spec 3.1.4)."""
        client = await self.get_client(agent_url)
        from a2a.types import ListTasksRequest
        request = ListTasksRequest(
            context_id=context_id,
            page_size=limit,
            page_token=page_token,
        )
        result = await client.list_tasks(request)
        return MessageToDict(result)

    async def subscribe_to_task(
        self,
        agent_url: str,
        task_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to streaming updates for an existing task (spec 3.1.6)."""
        client = await self.get_client(agent_url)
        from a2a.types import SubscribeToTaskRequest
        request = SubscribeToTaskRequest(id=task_id)
        async for resp in client.subscribe_to_task(request):
            yield MessageToDict(resp)

    async def _send_and_collect(
        self,
        agent_url: str,
        text: str,
        data: dict[str, Any] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        reference_task_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        client = await self.get_client(agent_url)
        message = make_user_message(
            text, data=data,
            context_id=context_id, task_id=task_id,
            reference_task_ids=reference_task_ids,
            extensions=extensions,
        )
        request = SendMessageRequest(
            message=message,
            configuration=SendMessageConfiguration(return_immediately=False),
            metadata=metadata,
        )

        task_id_result = ""
        context_id_result = ""
        artifact_dicts: list[dict[str, Any]] = []

        async for resp in client.send_message(request):
            if resp.HasField("task"):
                task = resp.task
                task_id_result = task.id
                context_id_result = task.context_id
                state = task.status.state
                if state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED):
                    msg_text = ""
                    if task.status.message and task.status.message.parts:
                        for p in task.status.message.parts:
                            if p.text:
                                msg_text += p.text
                    raise A2AError(f"Agent task terminated (state={TaskState.Name(state)}): {msg_text}")
                if state == TaskState.TASK_STATE_COMPLETED:
                    for art in task.artifacts:
                        artifact_dicts.append(MessageToDict(art))
            elif resp.HasField("message"):
                msg = resp.message
                task_id_result = msg.message_id or "msg"
                context_id_result = msg.context_id or ""
                artifact_dicts.append(MessageToDict(msg))
            elif resp.HasField("artifact_update"):
                art = resp.artifact_update.artifact
                artifact_dicts.append(MessageToDict(art))
            elif resp.HasField("status_update"):
                update = resp.status_update
                state = update.status.state
                if state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED):
                    msg_text = ""
                    if update.status.message and update.status.message.parts:
                        for p in update.status.message.parts:
                            if p.text:
                                msg_text += p.text
                    raise A2AError(f"Agent task terminated: {msg_text}")
                if state == TaskState.TASK_STATE_COMPLETED:
                    pass

        if not task_id_result and not artifact_dicts:
            raise A2AError(f"No task or artifacts received from {agent_url}")

        return task_id_result, context_id_result, artifact_dicts

    async def close(self):
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._agent_cards.clear()

    @staticmethod
    def extract_artifact_data(artifact_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for art in artifact_dicts:
            parts = art.get("parts", [])
            for part in parts:
                if "data" in part and part["data"]:
                    data = part["data"]
                    if isinstance(data, dict):
                        results.append(data)
        return results

    @staticmethod
    def extract_chunks_from_artifacts(artifact_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks = []
        data_list = A2AClientManager.extract_artifact_data(artifact_dicts)
        for d in data_list:
            if "chunks" in d:
                all_chunks.extend(d["chunks"])
        return all_chunks

    @staticmethod
    def extract_answer_from_artifacts(artifact_dicts: list[dict[str, Any]]) -> str:
        data_list = A2AClientManager.extract_artifact_data(artifact_dicts)
        for d in data_list:
            if "answer" in d:
                return d["answer"]
        return ""
