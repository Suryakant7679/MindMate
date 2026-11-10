"""Shared LangGraph state and reducers for the chatbot agents."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    task: str
    user_id: str
    run_id: str
    pending_agents: list[str]
    current_agent: str
    intermediate_results: Annotated[list[dict[str, Any]], operator.add]
    tool_activity: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, str]], operator.add]
    delegation_count: int
    final_response: str
    emotion_plan: dict[str, Any]
    emotion_history: Annotated[list[dict[str, Any]], operator.add]
    steering: dict[str, Any]
