"""LangGraph nodes: assessment precedes routing; only validated replies enter history."""
from langchain_core.messages import AIMessage
from ..safety import SafetyAgent, SafetyResult


class SafetyNodes:
    def __init__(self, agent: SafetyAgent):
        self.agent = agent

    async def assess(self, state):
        history = list(state.get("messages", []))
        if history and getattr(history[-1], "type", "") == "human":
            history = history[:-1]
        previous = state.get("safety", {}).get("policy")
        result = await self.agent.assess(state["task"], history, previous)
        return {"safety": result.model_dump(), "pending_agents": [], "draft_response": "",
                "final_response": "", "safety_validation": {}, "emotion_plan": {}, "steering": {},
                "current_agent": "safety"}

    async def crisis(self, state):
        result = SafetyResult.model_validate(state["safety"])
        reply, _ = await self.agent.response(state["task"], result, state.get("country"))
        return {"draft_response": reply, "current_agent": "safety", "pending_agents": [],
                "steering": {"run_id": state.get("run_id", ""), "applied_alpha": 0.0}}

    async def validate(self, state):
        result = SafetyResult.model_validate(state["safety"])
        reply, validation = await self.agent.finalize(state["task"], state.get("draft_response", ""), result,
            country=state.get("country"), trusted_template=result.policy.route != "normal")
        update = {}
        if not validation.approved:
            # Drafts also occur in steering comparison metadata; discard it on rejection.
            update = {"steering": {}, "emotion_plan": {}}
        elif state.get("steering", {}).get("effect"):
            steering = dict(state["steering"])
            steering["effect"] = dict(steering["effect"])
            for candidate in ("unsteered_reply", "steered_reply"):
                text = steering["effect"].get(candidate)
                if text and text != reply:
                    _, check = await self.agent.finalize(state["task"], text, result, country=state.get("country"))
                    if not check.approved:
                        steering["effect"][candidate] = "[Safety-blocked research draft]"
                        steering["comparison"] = None
            update["steering"] = steering
        return {**update, "messages": [AIMessage(content=reply)], "draft_response": "", "final_response": reply,
                "safety_validation": validation.model_dump()}
