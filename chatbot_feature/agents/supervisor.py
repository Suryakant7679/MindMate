"""Supervisor node: classify, select, and bound agent delegation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:   # runtime import would be circular (graph.builder imports this package)
    from ..graph.state import AgentState


async def supervisor_node(state: AgentState) -> dict[str, object]:
    from ..graph.routing import classify_task   # deferred: graph.builder imports this package

    routes = classify_task(state["task"])
    return {"pending_agents": routes, "current_agent": "supervisor", "delegation_count": 0}
