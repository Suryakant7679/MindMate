"""Validated, environment-aware MCP server configuration."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("servers.json")
ENV_DEFAULTS = {"DEFAULT_TIMEZONE": "Asia/Kolkata"}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    enabled: bool
    transport: str
    allowed_agents: frozenset[str]
    permissions: frozenset[str]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    required_env: tuple[str, ...] = ()
    timeout_seconds: float = 60.0

    @property
    def missing_environment(self) -> tuple[str, ...]:
        return tuple(name for name in self.required_env if not (os.getenv(name) or ENV_DEFAULTS.get(name)))

    @property
    def available(self) -> bool:
        return self.enabled and not self.missing_environment

    def adapter_entry(self) -> dict[str, Any]:
        """Return standard MCPConfig, resolving secrets only at runtime."""
        if not self.available:
            raise RuntimeError(f"MCP server {self.name!r} is disabled or missing credentials")
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"stdio server {self.name!r} has no command")
            entry: dict[str, Any] = {
                "command": self.command,
                "args": [self._expand(value) for value in self.args],
            }
            environment = {key: self._expand(value) for key, value in self.env.items()}
            if environment:
                entry["env"] = environment
            return entry
        if self.transport == "streamable_http":
            if not self.url or not self.url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
                raise ValueError(f"Unsafe MCP URL for {self.name!r}")
            entry = {"url": self._expand(self.url)}
            headers = {key: self._expand(value) for key, value in self.headers.items()}
            if headers:
                entry["headers"] = headers
            return entry
        raise ValueError(f"Unsupported MCP transport: {self.transport}")

    @staticmethod
    def _expand(value: str) -> str:
        return re.sub(
            r"\$\{([A-Z][A-Z0-9_]*)\}",
            lambda match: os.environ.get(match.group(1), ENV_DEFAULTS.get(match.group(1), "")),
            value,
        )


def load_server_configs(path: Path = CONFIG_PATH) -> dict[str, MCPServerConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, MCPServerConfig] = {}
    for name, raw in data["servers"].items():
        result[name] = MCPServerConfig(
            name=name,
            enabled=bool(raw.get("enabled", True)),
            transport=raw["transport"],
            allowed_agents=frozenset(raw["allowed_agents"]),
            permissions=frozenset(raw.get("permissions", ["read"])),
            command=raw.get("command"),
            args=tuple(raw.get("args", ())),
            url=raw.get("url"),
            env=dict(raw.get("env", {})),
            headers=dict(raw.get("headers", {})),
            required_env=tuple(raw.get("required_env", ())),
            timeout_seconds=float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", raw.get("timeout_seconds", 60))),
        )
    return result
