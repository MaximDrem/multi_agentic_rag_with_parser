import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Awaitable

from fastapi import FastAPI, Request, Response as FastAPIResponse
from google.protobuf.json_format import MessageToDict
from google.rpc.error_details_pb2 import ErrorInfo
from starlette.types import ASGIApp, Receive, Scope, Send

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    Message,
    Part,
    SecurityScheme,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from a2a.utils.errors import (
    A2AError,
    ContentTypeNotSupportedError,
    UnsupportedOperationError,
    TaskNotFoundError,
)
from a2a.helpers.proto_helpers import (
    new_text_part,
    new_data_part,
    new_text_message,
    new_text_artifact,
    new_data_artifact,
    new_text_status_update_event,
    new_data_artifact_update_event,
    get_message_text,
    get_data_parts,
)

from common.config import A2A_VERSION

logger = logging.getLogger(__name__)

_A2A_EXEMPT_PATHS = {"/.well-known/agent-card.json", "/agent-card", "/health"}


class A2AVersionASGIMiddleware:
    """Pure ASGI middleware: validates A2A-Version on requests (spec 3.6.2)
    and adds it to responses (spec 14.2.1).

    Per spec 3.6.2: if client version is incompatible, the request MUST be
    rejected with HTTP 400 and the server's supported version in the header.
    """

    def __init__(self, app: ASGIApp, a2a_extensions: list[str] | None = None):
        self.app = app
        self._a2a_extensions_header: str | None = None
        if a2a_extensions:
            self._a2a_extensions_header = ", ".join(e.uri for e in a2a_extensions)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            version_rejected = False

            if path not in _A2A_EXEMPT_PATHS:
                headers = dict(scope.get("headers", []))
                client_version = headers.get(b"a2a-version", b"").decode()
                if client_version and client_version != A2A_VERSION:
                    logger.warning(
                        f"Client A2A-Version mismatch: client={client_version}, server={A2A_VERSION} — rejecting request (spec 3.6.2)"
                    )
                    version_rejected = True

            if version_rejected:
                await self._send_version_error(scope, receive, send)
                return

            async def wrapped_send(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    existing = {k.lower(): True for k, _ in headers}
                    if b"a2a-version" not in existing:
                        headers.append((b"A2A-Version", A2A_VERSION.encode()))
                    if self._a2a_extensions_header and b"a2a-extensions" not in existing:
                        headers.append((b"A2A-Extensions", self._a2a_extensions_header.encode()))
                    message["headers"] = headers
                await send(message)

            await self.app(scope, receive, wrapped_send)
        else:
            await self.app(scope, receive, send)

    async def _send_version_error(self, scope: Scope, receive: Receive, send: Send) -> None:
        error_info = MessageToDict(ErrorInfo(
            reason="VERSION_NOT_SUPPORTED",
            domain="a2a-protocol.org",
        ), preserving_proto_field_name=True)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32009,
                "message": f"Incompatible A2A protocol version. Server supports {A2A_VERSION}.",
                "data": [{
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    **error_info,
                }],
            },
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [
                (b"Content-Type", b"application/json"),
                (b"A2A-Version", A2A_VERSION.encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


def _collect_skill_modes(skills: list[AgentSkill], attr: str, fallback: list[str]) -> list[str]:
    modes: set[str] = set()
    for skill in skills:
        skill_modes = getattr(skill, attr, None)
        if skill_modes:
            for m in skill_modes:
                modes.add(m)
    return sorted(modes) if modes else fallback


def make_agent_card(
    name: str,
    url: str,
    skills: list[AgentSkill],
    description: str | None = None,
    provider_organization: str | None = None,
    provider_url: str | None = None,
    security_schemes: list[SecurityScheme] | None = None,
    extensions: list[AgentExtension] | None = None,
    default_input_modes: list[str] | None = None,
    default_output_modes: list[str] | None = None,
) -> AgentCard:
    desc = description or f"A2A agent: {name}"
    provider = None
    if provider_organization or provider_url:
        provider = AgentProvider(
            organization=provider_organization or name,
            url=provider_url or url,
        )
    if default_input_modes is None:
        default_input_modes = _collect_skill_modes(skills, "input_modes", ["text/plain"])
    if default_output_modes is None:
        default_output_modes = _collect_skill_modes(skills, "output_modes", ["text/plain"])
    return AgentCard(
        name=name,
        description=desc,
        version="1.0.0",
        provider=provider,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            extended_agent_card=False,
            extensions=extensions or [],
        ),
        skills=skills,
        security_schemes=security_schemes or [],
        default_input_modes=default_input_modes or ["text/plain"],
        default_output_modes=default_output_modes or ["text/plain"],
        supported_interfaces=[
            AgentInterface(
                protocol_binding=TransportProtocol.JSONRPC,
                url=url.rstrip("/"),
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
    )


class A2AExecutor(AgentExecutor):
    """A2A AgentExecutor that delegates to an async handler function.

    Handler signature:
        async def handler(
            user_text: str,
            user_data: dict | None,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> dict:
            return {"status": "completed", "artifacts": [...], "message": "..."}

    Cancellation: when the framework calls cancel(), the cancel_event is set,
    causing the execute() race to short-circuit. The framework's cancel()
    already publishes TASK_STATE_CANCELED, so execute() just returns early.
    Framework-initiated CancelledError is re-raised (not caught by except Exception).
    """

    def __init__(self, handler: Callable[..., Awaitable[dict]]):
        self.handler = handler
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        tid = context.task_id
        cid = context.context_id

        cancel_event = asyncio.Event()
        self._cancel_events[tid] = cancel_event

        task = Task(
            id=tid,
            context_id=cid,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
        )
        if context.message is not None:
            task.history.append(context.message)
        await event_queue.enqueue_event(task)

        try:
            user_text = ""
            user_data: dict[str, Any] = {}
            msg = context.message
            if msg is not None:
                user_text = get_message_text(msg)
                data_parts = get_data_parts(msg.parts)
                if data_parts:
                    for dp in data_parts:
                        if isinstance(dp, dict):
                            user_data.update(dp)

            logger.info(
                f"A2A task START: task_id={tid}, "
                f"context={cid}, query={user_text[:100]}"
            )

            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=TaskState.TASK_STATE_WORKING,
                    text="Working",
                )
            )

            handler_task = asyncio.ensure_future(
                self.handler(
                    user_text=user_text,
                    user_data=user_data if user_data else None,
                    context=context,
                    event_queue=event_queue,
                )
            )
            cancel_task = asyncio.ensure_future(cancel_event.wait())

            done, pending = await asyncio.wait(
                [handler_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_task in done:
                handler_task.cancel()
                try:
                    await handler_task
                except asyncio.CancelledError:
                    pass
                logger.info(f"A2A task CANCELLED: task_id={tid}")
                return

            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            result = handler_task.result()

            status_str = result.get("status", "completed")
            artifacts = result.get("artifacts", [])
            message_text = result.get("message", "")

            final_state = TaskState.TASK_STATE_COMPLETED
            if status_str == "failed":
                final_state = TaskState.TASK_STATE_FAILED
            elif status_str == "rejected":
                final_state = TaskState.TASK_STATE_REJECTED

            if artifacts and status_str not in ("failed", "rejected"):
                for art in artifacts:
                    await event_queue.enqueue_event(
                        new_data_artifact_update_event(
                            task_id=tid,
                            context_id=cid,
                            name=art.get("name", art.get("artifactId", "result")),
                            data=art.get("data", {}),
                            media_type="application/json",
                            last_chunk=True,
                            artifact_id=art.get("artifactId"),
                        )
                    )

            status_msg = message_text or (
                "Completed" if status_str == "completed" else
                "Failed" if status_str == "failed" else
                "Rejected"
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=final_state,
                    text=status_msg,
                )
            )
            logger.info(
                f"A2A task DONE: task_id={tid}, "
                f"status={status_str}, artifacts={len(artifacts)}"
            )
        except ContentTypeNotSupportedError:
            logger.warning(f"A2A task UNSUPPORTED TYPE: task_id={tid}")
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=TaskState.TASK_STATE_FAILED,
                    text="Content type not supported",
                )
            )
        except UnsupportedOperationError:
            logger.warning(f"A2A task UNSUPPORTED OP: task_id={tid}")
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=TaskState.TASK_STATE_FAILED,
                    text="Operation not supported",
                )
            )
        except TaskNotFoundError:
            logger.warning(f"A2A task NOT FOUND: task_id={tid}")
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=TaskState.TASK_STATE_FAILED,
                    text="Task not found",
                )
            )
        except Exception:
            logger.exception(
                f"A2A task ERROR: task_id={tid}, "
                f"context={cid}"
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=tid,
                    context_id=cid,
                    state=TaskState.TASK_STATE_FAILED,
                    text="Internal error",
                )
            )
        finally:
            self._cancel_events.pop(tid, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        tid = context.task_id
        event = self._cancel_events.get(tid)
        if event is not None:
            event.set()
            logger.info(f"A2A cancel signal sent: task_id={tid}")
        if context.current_task is not None:
            await event_queue.enqueue_event(context.current_task)
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=context.task_id,
                context_id=context.context_id,
                state=TaskState.TASK_STATE_CANCELED,
                text="Canceled",
            )
        )


class A2AServer:
    """A single A2A-compliant agent server with FastAPI + DefaultRequestHandler."""

    def __init__(
        self,
        name: str,
        url: str,
        executor: A2AExecutor,
        skills: list[AgentSkill] | None = None,
        description: str | None = None,
    ):
        self.name = name
        self.url = url.rstrip("/")
        self.executor = executor
        self.agent_card = make_agent_card(name, self.url, skills or [], description)
        self.task_store = InMemoryTaskStore()
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
            agent_card=self.agent_card,
        )
        self.app = FastAPI(title=name)

        extensions_list = list(self.agent_card.capabilities.extensions) if self.agent_card.capabilities else None
        self.app.add_middleware(A2AVersionASGIMiddleware, a2a_extensions=extensions_list)

        add_a2a_routes_to_fastapi(
            self.app,
            agent_card_routes=create_agent_card_routes(self.agent_card),
            jsonrpc_routes=create_jsonrpc_routes(self.request_handler, rpc_url="/"),
            rest_routes=create_rest_routes(self.request_handler),
        )
        self.app.get("/health")(self._health)

        self._setup_agent_card_caching()

    def _setup_agent_card_caching(self):
        """Per spec 8.6: add Cache-Control, ETag, and conditional request headers
        to agent card endpoint. ETag is derived from the card's version field."""

        import email.utils
        import datetime

        etag_value = f'"{self.agent_card.name}-{self.agent_card.version}"'
        last_modified = email.utils.format_datetime(datetime.datetime.now(datetime.timezone.utc), usegmt=True)

        @self.app.middleware("http")
        async def _agent_card_cache_middleware(request: Request, call_next):
            if request.url.path == "/.well-known/agent-card.json":
                if_none_match = request.headers.get("if-none-match", "")
                if if_none_match == etag_value:
                    response = FastAPIResponse(status_code=304)
                    response.headers["ETag"] = etag_value
                    response.headers["Cache-Control"] = "public, max-age=300"
                    return response
                if_modified_since = request.headers.get("if-modified-since", "")
                if if_modified_since and if_modified_since == last_modified:
                    response = FastAPIResponse(status_code=304)
                    response.headers["ETag"] = etag_value
                    response.headers["Cache-Control"] = "public, max-age=300"
                    return response
            response = await call_next(request)
            if request.url.path == "/.well-known/agent-card.json":
                response.headers["Cache-Control"] = "public, max-age=300"
                response.headers["ETag"] = etag_value
                response.headers["Last-Modified"] = last_modified
            return response

    async def _health(self):
        return {"status": "ok", "agent": self.name}
