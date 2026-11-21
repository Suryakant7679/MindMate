"""Build the bounded asynchronous LangGraph multi-agent workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents import ResponseAgent, SpecialistAgent, supervisor_node
from ..mcp_connector import MCPManager
from .routing import AGENT_ORDER, next_node
from .state import AgentState
from ..agents.safety import SafetyNodes
from ..safety import SafetyAgent


def build_agent_graph(model: Any, manager: MCPManager, *, checkpointer: Any = None, emotion_reasoner: Any = None, safety_agent: Any = None):
    """Safety gates both branches; Emotion Reasoner remains optional on normal turns."""
    safety = SafetyNodes(safety_agent or SafetyAgent(model))
    graph = StateGraph(AgentState)
    graph.add_node("safety_assessment", safety.assess)
    graph.add_node("crisis_response", safety.crisis)
    graph.add_node("outgoing_safety", safety.validate)
    graph.add_edge(START, "safety_assessment")
    async def after_safety(state):
        if state["safety"]["policy"]["route"] != "normal":
            return "crisis_response"
        return "emotion_reasoner" if emotion_reasoner is not None else "supervisor"

    graph.add_conditional_edges("safety_assessment", after_safety)
    graph.add_edge("crisis_response", "outgoing_safety")
    graph.add_node("supervisor", supervisor_node)
    for name in AGENT_ORDER:
        graph.add_node(name, SpecialistAgent(name, model, manager))
    graph.add_node("response", ResponseAgent(model))
    if emotion_reasoner is not None:
        graph.add_node("emotion_reasoner", emotion_reasoner)
        graph.add_edge("emotion_reasoner", "supervisor")
    destinations = {**{name: name for name in AGENT_ORDER}, "response": "response", "crisis_response": "crisis_response"}
    graph.add_conditional_edges("supervisor", next_node, destinations)
    for name in AGENT_ORDER:
        graph.add_conditional_edges(name, next_node, destinations)
    graph.add_edge("response", "outgoing_safety")
    graph.add_edge("outgoing_safety", END)
    return graph.compile(checkpointer=checkpointer)
