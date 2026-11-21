"""Engineering triage types. These are neither diagnoses nor C-SSRS scores."""
from __future__ import annotations

from enum import IntEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Risk(IntEnum):
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Assessment(StrictModel):
    risk_level: Literal["R0", "R1", "R2", "R3", "R4"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertain: bool
    subject: Literal["self", "other", "fictional", "unknown"]
    timeframe: Literal["current", "historical", "hypothetical", "unknown"]
    intent: Literal["present", "denied", "not_disclosed"]
    plan: Literal["present", "denied", "not_disclosed"]
    means_access: Literal["present", "denied", "unknown"]
    immediacy: Literal["immediate", "soon", "unknown"]
    behavior: Literal["attempt", "injury", "preparation", "none_disclosed"]
    violence: bool
    safety_answer: Literal["safe_with_support", "unsafe", "unanswered"]
    language: Literal["en", "hi", "romanized_hi", "hinglish"]
    # Codes, never free text or quotations in retained state / diagnostics.
    evidence: list[Literal["distress", "death_wish", "ideation", "intent", "plan", "means",
                           "immediacy", "behavior", "violence", "context", "negation",
                           "historical", "third_person", "figurative", "resolution", "failure"]] = Field(max_length=16)
    protective_factors: list[Literal["human_support", "seeking_help", "reasons_for_living"]] = Field(default_factory=list, max_length=3)

    @property
    def level(self) -> Risk:
        return Risk[self.risk_level]


class Screen(StrictModel):
    floor: Risk = Risk.R0
    evidence: list[str] = Field(default_factory=list)
    intent: bool = False
    immediate: bool = False
    means: bool = False
    behavior: bool = False
    resolved: bool = False
    subject: Literal["self", "other", "fictional", "unknown"] = "self"
    language: Literal["en", "hi", "romanized_hi", "hinglish"] = "en"


class Policy(StrictModel):
    risk_level: Literal["R0", "R1", "R2", "R3", "R4"]
    route: Literal["normal", "clarification", "safety", "emergency"]
    steering_allowed: bool
    tools_allowed: bool
    requires_clarification: bool
    unresolved: bool
    safety_question_answered: bool
    uncertain: bool
    reason: Literal["assessed", "conflict", "unresolved", "failure", "low_confidence"]


class Validation(StrictModel):
    approved: bool
    severity: Literal["none", "concerning", "critical"]
    violations: list[Literal["harm_instructions", "encouragement", "minimization", "shaming", "diagnosis",
                             "discourages_care", "clinical_authority", "missing_urgent_help",
                             "false_promise", "unsafe_steering", "validation_failure"]] = Field(max_length=12)
    action: Literal["approve", "replace_with_safe_response"]


class SafetyResult(StrictModel):
    assessment: Assessment
    policy: Policy
    fallback: bool = False
    timeout: bool = False
    assessment_ms: float = 0
    llm_used: bool = False
    llm_ms: float = 0
    screening_ms: float = 0
