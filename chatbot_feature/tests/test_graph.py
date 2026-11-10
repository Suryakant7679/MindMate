from __future__ import annotations

import asyncio
import unittest

from langchain_core.messages import HumanMessage

from chatbot_feature.graph import build_agent_graph, classify_task
from chatbot_feature.mcp_connector.config import MCPServerConfig
from chatbot_feature.mcp_connector.manager import DiscoveredTool, MCPManager


class Tool:
    name = "search"
    description = "Search current sources"
    args_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    async def ainvoke(self, arguments):
        return "evidence for " + arguments["query"]


class Model:
    async def complete(self, prompt):
        if "Authorized tools:" in prompt:
            return '{"calls":[{"tool":"brave_search_search","arguments":{"query":"LangGraph"}}]}'
        return "Synthesized answer"


class GraphTests(unittest.TestCase):
    def test_supervisor_routes_single_and_multi_domain_tasks(self):
        self.assertEqual(classify_task("Search the latest AI news"), ["research"])
        routes = classify_task("Research current docs, inspect my repository, and create report")
        self.assertEqual(routes, ["research", "coding", "document"])

    def test_graph_executes_authorized_tool_and_synthesizes(self):
        config = MCPServerConfig(
            name="brave_search", enabled=True, transport="stdio", command="unused",
            allowed_agents=frozenset({"research"}), permissions=frozenset({"read"}),
        )
        manager = MCPManager({"brave_search": config}, adapter_factory=lambda _: None)
        item = DiscoveredTool("brave_search", "search", Tool.description, Tool.args_schema, Tool())
        manager._tools[item.qualified_name] = item
        graph = build_agent_graph(Model(), manager)

        state = asyncio.run(graph.ainvoke({
            "messages": [HumanMessage(content="Search the latest LangGraph docs")],
            "task": "Search the latest LangGraph docs",
            "user_id": "user-1",
        }))
        self.assertEqual(state["final_response"], "Synthesized answer")
        self.assertEqual(state["tool_activity"][-1]["status"], "completed")
        self.assertEqual(state["intermediate_results"][0]["server"], "brave_search")
        self.assertLessEqual(state["delegation_count"], 4)


if __name__ == "__main__":
    unittest.main()
