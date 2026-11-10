"""Dynamic MCP discovery, authorization, execution, health, and audit logging."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .config import MCPServerConfig, load_server_configs
from .permissions import authorize

LOGGER = logging.getLogger("mindmate.mcp")


@dataclass
class ServerHealth:
    name: str
    status: str
    tool_count: int = 0
    error: str | None = None
    checked_at: float | None = None


@dataclass(frozen=True)
class DiscoveredTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    tool: Any

    @property
    def qualified_name(self) -> str:
        return self.name if self.name.startswith(f"{self.server}_") else f"{self.server}_{self.name}"


class MCPManager:
    """Own the registry; MCP tools themselves create short-lived sessions per call."""

    def __init__(
        self,
        configs: dict[str, MCPServerConfig] | None = None,
        *,
        adapter_factory: Callable[[dict[str, Any]], Any] | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.configs = configs or load_server_configs()
        self._adapter_factory = adapter_factory
        self._tools: dict[str, DiscoveredTool] = {}
        self._health = {
            name: ServerHealth(name, "disabled" if not config.enabled else "not_connected")
            for name, config in self.configs.items()
        }
        self._semaphore = asyncio.Semaphore(max_concurrency or int(os.getenv("MCP_MAX_CONCURRENT_TOOL_CALLS", "5")))
        self._connect_timeout = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "30"))

    async def connect(self) -> dict[str, ServerHealth]:
        if self._adapter_factory is None:
            try:
                from langchain.mcp import MCPAdapter
            except ImportError as exc:
                raise RuntimeError('Install backend/requirements.txt to enable MCP') from exc
            self._adapter_factory = MCPAdapter
        await asyncio.gather(*(self._discover_one(config) for config in self.configs.values()))
        return self.health()

    async def _discover_one(self, config: MCPServerConfig) -> None:
        health = self._health[config.name]
        health.checked_at = time.time()
        if not config.enabled:
            health.status = "disabled"
            return
        if config.missing_environment:
            credential = any(marker in name for name in config.missing_environment for marker in ("KEY", "TOKEN", "SECRET"))
            health.status = "authentication_required" if credential else "configuration_required"
            health.error = "Missing: " + ", ".join(config.missing_environment)
            return
        try:
            target = {"mcpServers": {config.name: config.adapter_entry()}}
            adapter = self._adapter_factory(target)
            async def discover() -> list[Any]:
                async with adapter as connected:
                    return await connected.list_tools()
            tools = await asyncio.wait_for(discover(), timeout=self._connect_timeout)
            for tool in tools:
                schema = getattr(tool, "args_schema", None)
                if hasattr(schema, "model_json_schema"):
                    schema = schema.model_json_schema()
                elif not isinstance(schema, dict):
                    schema = {}
                item = DiscoveredTool(
                    server=config.name,
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=schema,
                    tool=tool,
                )
                self._tools[item.qualified_name] = item
            health.status = "connected"
            health.tool_count = len(tools)
            health.error = None
        except Exception as exc:  # One optional server must never take down chat.
            health.status = "unavailable"
            health.error = f"{type(exc).__name__}: {exc}"[:500]
            LOGGER.warning("MCP discovery failed for %s: %s", config.name, type(exc).__name__)

    def tools_for_agent(self, agent: str) -> list[DiscoveredTool]:
        return [tool for tool in self._tools.values() if agent in self.configs[tool.server].allowed_agents]

    async def execute(self, agent: str, qualified_name: str, arguments: dict[str, Any], *, user_id: str | None = None) -> Any:
        try:
            item = self._tools[qualified_name]
        except KeyError as exc:
            raise KeyError(f"Unknown or unavailable MCP tool: {qualified_name}") from exc
        config = self.configs[item.server]
        authorize(config, agent, item.name, arguments)
        if item.server == "memory":
            if not user_id:
                raise PermissionError("Memory calls require an authenticated user namespace")
            namespace = f"user:{user_id}:"
            if namespace not in json.dumps(arguments, ensure_ascii=False):
                raise PermissionError("Memory arguments must use the current user's namespace")
        clean_arguments = {key: value for key, value in arguments.items() if key != "user_approved"}
        started = time.monotonic()
        LOGGER.info("MCP call agent=%s server=%s tool=%s", agent, item.server, item.name)
        async with self._semaphore:
            result = await asyncio.wait_for(item.tool.ainvoke(clean_arguments), timeout=config.timeout_seconds)
        LOGGER.info("MCP complete server=%s tool=%s elapsed_ms=%d", item.server, item.name, (time.monotonic() - started) * 1000)
        return result

    def health(self) -> dict[str, ServerHealth]:
        return {name: ServerHealth(**asdict(value)) for name, value in self._health.items()}

    async def reconnect(self, name: str) -> ServerHealth:
        if name not in self.configs:
            raise KeyError(f"Unknown MCP server: {name}")
        self._tools = {key: value for key, value in self._tools.items() if value.server != name}
        await self._discover_one(self.configs[name])
        return self.health()[name]

    async def close(self) -> None:
        self._tools.clear()
