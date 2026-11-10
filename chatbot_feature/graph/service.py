"""Lifecycle facade used by the existing backend and tests."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from ..mcp_connector import MCPManager
from .builder import build_agent_graph


class AgentChatService:
    def __init__(self, model: Any, manager: MCPManager, checkpoint_path: str | Path | None = None,
                 emotion_reasoner: Any = None) -> None:
        self.model = model
        self.manager = manager
        self.emotion_reasoner = emotion_reasoner
        self.checkpoint_path = Path(checkpoint_path or os.getenv("LANGGRAPH_CHECKPOINT_PATH", "chatbot_feature/data/checkpoints.sqlite"))
        self._connected = False

    async def start(self) -> None:
        if not self._connected:
            await self.manager.connect()
            self._connected = True

    async def run(self, message: str, *, session_id: str, user_id: str = "anonymous") -> dict[str, Any]:
        await self.start()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path)) as saver:
            graph = build_agent_graph(self.model, self.manager, checkpointer=saver,
                                      emotion_reasoner=self.emotion_reasoner)
            run_id = uuid.uuid4().hex
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=message)], "task": message, "user_id": user_id, "run_id": run_id},
                {"configurable": {"thread_id": f"{user_id}:{session_id}"}, "recursion_limit": 16},
            )
        return {
            "response": state["final_response"],
            "activities": [item for item in state.get("tool_activity", []) if item.get("run_id") == run_id],
            "errors": [item["message"] for item in state.get("errors", []) if item.get("run_id") == run_id],
            "active_agent": state.get("current_agent", "response"),
            "emotion_plan": plan if (plan := state.get("emotion_plan")) and plan.get("run_id") == run_id else None,
            "steering": steering if (steering := state.get("steering")) and steering.get("run_id") == run_id else None,
        }

    async def close(self) -> None:
        await self.manager.close()
        self._connected = False
