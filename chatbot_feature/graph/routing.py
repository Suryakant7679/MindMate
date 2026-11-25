"""Bounded supervisor classification and conditional-edge helpers."""

from __future__ import annotations

from .state import AgentState

AGENT_ORDER = ("research", "coding", "browser", "memory", "document", "planning")
KEYWORDS = {
    "research": ("research", "search", "latest", "current", "news", "internet", "web", "source"),
    "coding": ("code", "debug", "repository", "github", "git ", "documentation", "library", "api"),
    "browser": ("browser", "website", "click", "navigate", "fill form", "screenshot"),
    "memory": ("remember", "forget", "what do you know about me", "memory"),
    "document": ("write file", "save report", "create report", "document", "export"),
    "planning": ("plan", "schedule", "timezone", "what time", "strategy", "steps"),
}


def classify_task(task: str, *, limit: int = 4) -> list[str]:
    lowered = f" {task.lower()} "
    routes = [agent for agent in AGENT_ORDER if any(word in lowered for word in KEYWORDS[agent])]
    return routes[:limit]


async def next_node(state: AgentState) -> str:
    pending = state.get("pending_agents", [])
    return pending[0] if pending else "response"
