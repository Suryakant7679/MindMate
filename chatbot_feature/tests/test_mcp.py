from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from chatbot_feature.mcp_connector.config import MCPServerConfig, load_server_configs
from chatbot_feature.mcp_connector.manager import MCPManager
from chatbot_feature.mcp_connector.permissions import MCPPermissionError, authorize


class FakeTool:
    name = "lookup"
    description = "Look up a value"
    args_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    async def ainvoke(self, arguments):
        return {"value": arguments["query"]}


class FakeAdapter:
    def __init__(self, target):
        self.target = target

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def list_tools(self):
        return [FakeTool()]


def config(**overrides):
    values = dict(
        name="test", enabled=True, transport="stdio", command="test-server",
        allowed_agents=frozenset({"research"}), permissions=frozenset({"read"}),
    )
    values.update(overrides)
    return MCPServerConfig(**values)


class MCPConfigTests(unittest.TestCase):
    def test_all_eleven_servers_are_defined(self):
        configs = load_server_configs()
        self.assertEqual(len(configs), 11)
        self.assertEqual(
            set(configs),
            {"brave_search", "filesystem", "memory", "playwright", "github", "context7", "firecrawl", "sequential_thinking", "fetch", "git", "time"},
        )

    def test_missing_credentials_are_reported_without_connecting(self):
        secured = config(required_env=("MISSING_TEST_SECRET",))
        with patch.dict(os.environ, {}, clear=True):
            manager = MCPManager({"test": secured}, adapter_factory=FakeAdapter)
            health = asyncio.run(manager.connect())
        self.assertEqual(health["test"].status, "authentication_required")

    def test_environment_references_expand_inside_header_values(self):
        secured = config(
            transport="streamable_http", command=None, url="https://example.com/mcp",
            headers={"Authorization": "Bearer ${TEST_MCP_TOKEN}"}, required_env=("TEST_MCP_TOKEN",),
        )
        with patch.dict(os.environ, {"TEST_MCP_TOKEN": "secret"}):
            self.assertEqual(secured.adapter_entry()["headers"]["Authorization"], "Bearer secret")

    def test_dynamic_discovery_and_execution(self):
        manager = MCPManager({"test": config()}, adapter_factory=FakeAdapter)

        async def run():
            await manager.connect()
            tools = manager.tools_for_agent("research")
            result = await manager.execute("research", tools[0].qualified_name, {"query": "hello"})
            return tools, result

        tools, result = asyncio.run(run())
        self.assertEqual(tools[0].input_schema["type"], "object")
        self.assertEqual(result, {"value": "hello"})

    def test_agent_permissions_and_ssrf_are_enforced(self):
        secured = config()
        with self.assertRaises(MCPPermissionError):
            authorize(secured, "coding", "lookup", {})
        with self.assertRaises(MCPPermissionError):
            authorize(secured, "research", "lookup", {"url": "http://127.0.0.1/private"})

    def test_read_only_blocks_mutation_tools(self):
        for tool_name in ("delete_repository", "git_add", "git_commit", "browser_type"):
            with self.subTest(tool=tool_name), self.assertRaises(MCPPermissionError):
                authorize(config(), "research", tool_name, {})


if __name__ == "__main__":
    unittest.main()
