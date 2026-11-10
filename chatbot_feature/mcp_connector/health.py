"""CLI health probe: connect, dynamically discover, and print safe status JSON."""

from __future__ import annotations

import asyncio
import argparse
import json
from dataclasses import asdict

from .config import load_server_configs
from .manager import MCPManager


async def probe(server: str | None = None, tool_name: str | None = None, arguments: str = "{}") -> int:
    configs = load_server_configs()
    if server:
        if server not in configs:
            raise SystemExit(f"Unknown MCP server: {server}")
        configs = {server: configs[server]}
    manager = MCPManager(configs)
    try:
        statuses = await manager.connect()
        print(json.dumps({
            "servers": {name: asdict(status) for name, status in statuses.items()},
            "discovered_tools": sorted(manager._tools),
        }, indent=2))
        if tool_name:
            matches = [tool for tool in manager._tools.values() if tool.name == tool_name or tool.qualified_name == tool_name]
            if len(matches) != 1:
                raise SystemExit(f"Expected one discovered tool named {tool_name!r}, found {len(matches)}")
            agent = next(iter(manager.configs[matches[0].server].allowed_agents))
            result = await manager.execute(agent, matches[0].qualified_name, json.loads(arguments))
            print(json.dumps({"tool": matches[0].qualified_name, "result": str(result)}, indent=2))
        return 0 if any(status.status == "connected" for status in statuses.values()) else 1
    finally:
        await manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", help="Probe only one configured server")
    parser.add_argument("--call-tool", help="Optionally execute one discovered read-only tool")
    parser.add_argument("--arguments", default="{}", help="JSON object for --call-tool")
    options = parser.parse_args()
    raise SystemExit(asyncio.run(probe(options.server, options.call_tool, options.arguments)))
