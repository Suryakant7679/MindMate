"""Build the bounded asynchronous LangGraph multi-agent workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents import ResponseAgent, SpecialistAgent, supervisor_node
from ..mcp_connector import MCPManager
from .routing import AGENT_ORDER, next_node
from .state import AgentState


def build_agent_graph(model: Any, manager: MCPManager, *, checkpointer: Any = None, emotion_reasoner: Any = None):
    """``emotion_reasoner`` is an optional node run first; without it the graph is unchanged."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    for name in AGENT_ORDER:
        graph.add_node(name, SpecialistAgent(name, model, manager))
    graph.add_node("response", ResponseAgent(model))
    if emotion_reasoner is not None:
        graph.add_node("emotion_reasoner", emotion_reasoner)
        graph.add_edge(START, "emotion_reasoner")
        graph.add_edge("emotion_reasoner", "supervisor")
    else:
        graph.add_edge(START, "supervisor")
    destinations = {**{name: name for name in AGENT_ORDER}, "response": "response"}
    graph.add_conditional_edges("supervisor", next_node, destinations)
    for name in AGENT_ORDER:
        graph.add_conditional_edges(name, next_node, destinations)
    graph.add_edge("response", END)
    return graph.compile(checkpointer=checkpointer)
