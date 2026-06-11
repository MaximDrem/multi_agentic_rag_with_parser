import asyncio
import logging
from typing import Any

from a2a.types import AgentCard

from common.config import SUB_AGENT_URLS
from common.a2a_client import A2AClientManager

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Tracks discovered agents and their capabilities."""

    def __init__(self):
        self.cards: dict[str, AgentCard] = {}
        self.url_by_name: dict[str, str] = {}

    def register(self, url: str, card: AgentCard) -> None:
        url = url.rstrip("/")
        self.cards[url] = card
        name = card.name.lower().replace(" ", "_").replace("-", "_")
        self.url_by_name[name] = url
        self.url_by_name[card.name.lower()] = url

    def get_url(self, name: str) -> str | None:
        key = name.lower().replace(" ", "_").replace("-", "_")
        return self.url_by_name.get(key) or self.url_by_name.get(name.lower())

    def get_card(self, url: str) -> AgentCard | None:
        return self.cards.get(url.rstrip("/"))


class AgentDiscoverer:
    """Discovers A2A agents by fetching their agent cards via A2A protocol.

    Supports initial discovery and periodic health-check / re-discovery.
    """

    def __init__(self, client_manager: A2AClientManager, registry: AgentRegistry):
        self._client = client_manager
        self.registry = registry
        self._base_urls: list[str] = []
        self._health_task: asyncio.Task | None = None

    async def discover(self, urls: list[str] | None = None) -> AgentRegistry:
        urls = urls or SUB_AGENT_URLS
        self._base_urls = list(urls)
        for url in urls:
            card = await self._discover_with_retry(url, max_retries=10, delay=6.0)
            if card:
                self.registry.register(url, card)
                skill_ids = [s.id for s in card.skills]
                logger.info(
                    f"Discovered agent '{card.name}' at {url} "
                    f"with {len(card.skills)} skills: {skill_ids}"
                )
            else:
                logger.warning(f"Failed to discover agent at {url} after retries")
        return self.registry

    async def refresh(self) -> bool:
        """Re-fetch agent cards for all known URLs. Returns True if any were updated."""
        updated = False
        for url in self._base_urls:
            try:
                card = await self._discover_with_retry(url, max_retries=2, delay=2.0)
                if card:
                    self.registry.register(url, card)
                    updated = True
            except Exception as e:
                logger.warning(f"Refresh failed for {url}: {e}")
        if updated:
            logger.info(f"Agent registry refreshed: {len(self.registry.cards)} agents")
        return updated

    async def start_health_check(self, interval: float = 60.0):
        """Start periodic health-check loop for dynamic re-discovery."""
        if self._health_task is not None:
            return

        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.refresh()
                except Exception as e:
                    logger.warning(f"Health check error: {e}")

        self._health_task = asyncio.create_task(_loop())
        logger.info(f"Agent health check started (interval={interval}s)")

    async def stop_health_check(self):
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _discover_with_retry(self, url: str, max_retries: int, delay: float) -> AgentCard | None:
        for attempt in range(1, max_retries + 1):
            try:
                card = await self._client.fetch_agent_card(url)
                return card
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Discovery attempt {attempt}/{max_retries} for {url}: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Failed to discover agent at {url}: {e}")
        return None


class TaskTracker:
    """In-process registry of A2A task results for a single orchestration session."""

    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}

    def record(self, task_id: str, context_id: str, agent_url: str, artifact_dicts: list[dict], safe_name: str = ""):
        self.tasks[task_id] = {
            "task_id": task_id,
            "context_id": context_id,
            "agent_url": agent_url,
            "artifact_dicts": artifact_dicts,
            "safe_name": safe_name,
        }

    def get_all_artifact_dicts(self) -> list[dict]:
        result = []
        for entry in self.tasks.values():
            result.extend(entry["artifact_dicts"])
        return result
