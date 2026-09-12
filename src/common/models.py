from pydantic import BaseModel
from typing import Optional, Any


class Chunk(BaseModel):
    id: str
    doc_path: str
    title: str
    text: str
    token_count: int
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    metadata: dict[str, Any] = {}


class AgentCardInfo(BaseModel):
    name: str
    description: str
    url: str
    skills: list[dict[str, Any]]


class TaskResult(BaseModel):
    task_id: str
    agent_name: str
    agent_url: str
    status: str
    artifacts: list[dict[str, Any]]
