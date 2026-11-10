"""Least-privilege checks applied before every MCP tool execution."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from .config import MCPServerConfig


MUTATION_MARKERS = (
    "create", "update", "delete", "write", "push", "merge", "add", "commit", "checkout", "reset", "remote",
    "click", "submit", "upload", "type", "fill", "press", "select", "drag", "dialog", "evaluate",
)


class MCPPermissionError(PermissionError):
    pass


def authorize(config: MCPServerConfig, agent: str, tool_name: str, arguments: Mapping[str, object]) -> None:
    if agent not in config.allowed_agents:
        raise MCPPermissionError(f"Agent {agent!r} may not use MCP server {config.name!r}")
    lowered = tool_name.lower()
    mutation = bool(set(re.findall(r"[a-z0-9]+", lowered)) & set(MUTATION_MARKERS))
    if "write" not in config.permissions and mutation:
        raise MCPPermissionError(f"Tool {tool_name!r} is blocked by read-only policy")
    if "approval" in config.permissions and mutation:
        if arguments.get("user_approved") is not True:
            raise MCPPermissionError(f"Tool {tool_name!r} requires explicit user approval")
    _reject_private_urls(arguments)


def _reject_private_urls(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_private_urls(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_private_urls(nested)
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        host = urlsplit(value).hostname
        if not host:
            raise MCPPermissionError("Malformed URL")
        if host in {"localhost", "localhost.localdomain"}:
            raise MCPPermissionError("Private network URLs are blocked")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return
        if not address.is_global:
            raise MCPPermissionError("Private network URLs are blocked")
