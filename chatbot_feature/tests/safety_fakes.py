"""Scripted schemas for control-flow tests only; never used to report model quality."""
import json
from chatbot_feature.safety.schemas import Assessment, Validation


def assessment(level="R0", **changes):
    values = dict(risk_level=level, confidence=0.9, uncertain=False, subject="self", timeframe="current",
        intent="not_disclosed", plan="not_disclosed", means_access="unknown", immediacy="unknown",
        behavior="none_disclosed", violence=False, safety_answer="unanswered", language="en", evidence=[])
    return Assessment(**(values | changes))


class SafetyModel:
    def __init__(self, value=None, validation=None):
        self.value = value or assessment()
        self.validation = validation or Validation(approved=True, severity="none", violations=[], action="approve")
        self.inputs = []

    async def complete_safety(self, prompt):
        self.inputs.append(json.loads(prompt.split("\nDATA:\n")[1]))
        value = self.validation if prompt.startswith("MINDMATE_SAFETY_VALIDATION") else self.value
        return value.model_dump_json()
