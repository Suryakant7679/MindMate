"""LangGraph node that runs the Emotion Reasoner Agent before delegation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..emotion_reasoner import EmotionReasonerAgent
if TYPE_CHECKING:   # runtime import would be circular (graph.builder imports this package)
    from ..graph.state import AgentState

HISTORY_TURNS = 3


class EmotionReasonerNode:
    """Plans the emotional response for the current user turn.

    ``probe`` maps a user message to the frozen probe's emotion probabilities.
    Earlier turns of the thread come from the checkpointed ``emotion_history``,
    so the plan reflects how the user's emotion has moved across the conversation.
    """

    def __init__(self, agent: EmotionReasonerAgent, probe: Callable[[str], Mapping[str, float]]) -> None:
        self.agent = agent
        self.probe = probe

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        try:
            history = list(state.get("emotion_history", []))[-HISTORY_TURNS:]
            plan = self.agent.plan(state["task"], self.probe(state["task"]), history).to_dict()
        except Exception as exc:   # the reasoner is optional: never block the reply
            return {"current_agent": self.agent.name,
                    "errors": [{"run_id": run_id, "message": f"emotion_reasoner: {type(exc).__name__}"}]}
        return {
            "emotion_plan": {**plan, "run_id": run_id},
            "emotion_history": [{"emotion": plan["user_emotion"], "valence": plan["valence"], "run_id": run_id}],
            "current_agent": self.agent.name,
        }
