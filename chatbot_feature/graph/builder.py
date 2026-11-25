"""Build the bounded asynchronous LangGraph multi-agent workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..agents import ResponseAgent, SpecialistAgent, supervisor_node
from ..mcp_connector import MCPManager
from .routing import AGENT_ORDER, next_node
from .state import AgentState


async def begin_turn(state: AgentState) -> dict[str, Any]:
    """Per-turn fields start empty; the conversation (messages, emotion history) carries over between turns."""
    return {"pending_agents": [], "draft_response": "", "final_response": "", "emotion_plan": {}, "steering": {},
            "current_agent": "start"}


async def finish_turn(state: AgentState) -> dict[str, Any]:
    """The reply becomes the turn's final response and enters the conversation history."""
    reply = state.get("draft_response", "")
    return {"messages": [AIMessage(content=reply)], "draft_response": "", "final_response": reply}


def build_agent_graph(model: Any, manager: MCPManager, *, checkpointer: Any = None, emotion_reasoner: Any = None):
    """start -> (Emotion Reasoner) -> supervisor -> specialists -> response -> finish."""
    graph = StateGraph(AgentState)
    graph.add_node("begin", begin_turn)
    graph.add_edge(START, "begin")
    graph.add_node("supervisor", supervisor_node)
    for name in AGENT_ORDER:
        graph.add_node(name, SpecialistAgent(name, model, manager))
    graph.add_node("response", ResponseAgent(model))
    graph.add_node("finish", finish_turn)
    if emotion_reasoner is not None:
        graph.add_node("emotion_reasoner", emotion_reasoner)
        graph.add_edge("begin", "emotion_reasoner")
        graph.add_edge("emotion_reasoner", "supervisor")
    else:
        graph.add_edge("begin", "supervisor")
    destinations = {**{name: name for name in AGENT_ORDER}, "response": "response"}
    graph.add_conditional_edges("supervisor", next_node, destinations)
    for name in AGENT_ORDER:
        graph.add_conditional_edges(name, next_node, destinations)
    graph.add_edge("response", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
