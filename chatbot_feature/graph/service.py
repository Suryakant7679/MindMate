"""Lifecycle facade with bounded, expiring conversation state (no risk checkpoint on disk)."""
from __future__ import annotations
import os
import uuid
import threading
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage
from ..mcp_connector import MCPManager
from ..safety import SafetyAgent
from ..safety.session import SessionStore, session_key
from .builder import build_agent_graph


class AgentChatService:
    def __init__(self, model: Any, manager: MCPManager, checkpoint_path: str | Path | None = None,
                 emotion_reasoner: Any = None, safety_agent: Any = None) -> None:
        self.model, self.manager, self.emotion_reasoner = model, manager, emotion_reasoner
        self.safety_agent = safety_agent
        # Kept as a compatibility attribute only. Never persist new safety/chat state here.
        self.checkpoint_path = checkpoint_path
        self.sessions = SessionStore(float(os.getenv("MINDMATE_SAFETY_RETENTION_SECONDS", "1800")))
        self._connected = False
        self._run_lock = threading.Lock()

    async def start(self) -> None:
        if not self._connected:
            await self.manager.connect()
            self._connected = True

    async def run(self, message: str, *, session_id: str, user_id: str = "anonymous", country=None) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Agent service busy")
        try:
            key = session_key(user_id, session_id)
            prior = self.sessions.get(key) or {}
            run_id = uuid.uuid4().hex
            # Connecting MCP is delayed until a normal-risk assessment has authorized tools.
            agent = self.safety_agent or SafetyAgent(self.model)
            graph = build_agent_graph(self.model, self.manager, emotion_reasoner=self.emotion_reasoner, safety_agent=agent)
            state = await graph.ainvoke({**prior, "messages": [*prior.get("messages", [])[-24:], HumanMessage(content=message)],
                "task": message, "user_id": user_id, "run_id": run_id, "country": country}, {"recursion_limit": 24})
            self.sessions.put(key, {"messages": state.get("messages", [])[-24:], "safety": state["safety"],
                                    "emotion_history": state.get("emotion_history", [])[-3:]})
            return {"response": state["final_response"],
                "activities": [x for x in state.get("tool_activity", []) if x.get("run_id") == run_id],
                "errors": [x["message"] for x in state.get("errors", []) if x.get("run_id") == run_id],
                "active_agent": state.get("current_agent", "response"),
                "emotion_plan": state.get("emotion_plan") or None, "steering": state.get("steering") or None,
                "safety": {"risk_level": state["safety"]["policy"]["risk_level"],
                           "route": state["safety"]["policy"]["route"], "validation": state["safety_validation"]}}
        finally:
            self._run_lock.release()

    async def close(self) -> None:
        await self.manager.close()
        self._connected = False
