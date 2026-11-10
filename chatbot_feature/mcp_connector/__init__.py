"""MCP server discovery, connection lifecycle, and tool/resource adapters."""
"""Maintained LangChain MCP integration for MindMate."""

from .config import MCPServerConfig, load_server_configs
from .manager import DiscoveredTool, MCPManager, ServerHealth
from .permissions import MCPPermissionError, authorize

__all__ = ["DiscoveredTool", "MCPManager", "MCPPermissionError", "MCPServerConfig", "ServerHealth", "authorize", "load_server_configs"]
