"""Deterministic authority: model classifications never authorize their own bypass."""
from .schemas import Assessment, Policy, Risk, Screen


def decide(assessment: Assessment, screen: Screen, previous: Policy | None = None,
           *, failed: bool = False, confidence_threshold: float = 0.65) -> Policy:
    level = max(assessment.level, screen.floor)
    reason = "failure" if failed else "conflict" if screen.floor > assessment.level else "assessed"
    if assessment.timeframe == "current" and assessment.subject in ("self", "other"):
        if assessment.behavior in ("attempt", "injury") or (
            assessment.intent == "present" and (assessment.plan == "present" or assessment.means_access == "present")
            and assessment.immediacy in ("immediate", "soon")):
            level = Risk.R4
    uncertain = failed or assessment.uncertain or assessment.confidence < confidence_threshold
    if uncertain and level == Risk.R0:
        level, reason = Risk.R1, "failure" if failed else "low_confidence"
    answered = (assessment.safety_answer == "safe_with_support" and screen.resolved
                and assessment.level <= Risk.R1 and screen.floor <= Risk.R1 and not failed
                and "human_support" in assessment.protective_factors)
    if previous and previous.unresolved and not answered:
        level = max(level, Risk[previous.risk_level])
        reason = "failure" if failed else "unresolved"
    route = "normal" if level == Risk.R0 else "clarification" if level == Risk.R1 else "emergency" if level == Risk.R4 else "safety"
    return Policy(risk_level=level.name, route=route, steering_allowed=level == Risk.R0,
                  tools_allowed=level == Risk.R0, requires_clarification=level in (Risk.R1, Risk.R2, Risk.R3),
                  unresolved=level >= Risk.R1, safety_question_answered=answered,
                  uncertain=uncertain, reason=reason)
