"""Context-aware orchestration, with no persistent risk records or raw diagnostic logs."""
import asyncio
import os
import re
import time
from .assessor import ContextualAssessor
from .policy import decide
from .responses import safety_response
from .schemas import Assessment, Policy, Risk, SafetyResult, Screen
from .screening import ProvisionalScreen
from .validator import ResponseValidator


class SafetyAgent:
    def __init__(self, model=None, *, screen=None, timeout_seconds=None, context_turns=None,
                 confidence_threshold=0.65, directory=None):
        self.screen = screen or ProvisionalScreen()
        self.assessor = ContextualAssessor(model, timeout_seconds if timeout_seconds is not None else float(os.getenv("MINDMATE_SAFETY_TIMEOUT", "30")))
        self.validator = ResponseValidator(self.assessor)
        self.context_turns = context_turns if context_turns is not None else int(os.getenv("MINDMATE_SAFETY_CONTEXT_TURNS", "12"))
        if self.context_turns < 1:
            raise ValueError("Safety context must include at least one turn")
        self.confidence_threshold, self.directory = confidence_threshold, directory

    async def assess(self, message, history=(), previous=None):
        started = time.perf_counter()
        failed = timed_out = False
        try:
            screen = self.screen.screen(message)
            if not isinstance(screen, Screen):
                raise ValueError("Invalid screen output")
        except Exception:
            screen, failed = Screen(floor=Risk.R1, evidence=["failure"]), True
        screening_ms = (time.perf_counter() - started) * 1000
        context = []
        for turn in history[-self.context_turns:]:
            role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "type", None)
            content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", None)
            if role in ("user", "human", "assistant", "ai") and isinstance(content, str):
                context.append({"role": "user" if role in ("user", "human") else "assistant", "content": content[:8000]})
        previous = Policy.model_validate(previous) if isinstance(previous, dict) else previous
        # Explicit elliptical escalation needs a recent risk anchor; no topic-wide keyword matching.
        anchor = Risk.R0
        try:
            if previous is not None:
                anchor = Risk[previous.risk_level] if previous.unresolved else Risk.R0
            else:
                for turn in context:
                    if turn["role"] == "user":
                        cue = self.screen.screen(turn["content"])
                        anchor = cue.floor if cue.resolved else max(anchor, cue.floor)
        except Exception:
            failed = True
        current = message.lower().strip(" .!।")
        if anchor >= Risk.R1 and re.fullmatch(r"i have already decided|i've decided|maine decide kar liya", current):
            screen.floor = max(screen.floor, Risk.R3)
            screen.evidence = list(dict.fromkeys(screen.evidence + ["context", "intent"]))
        if anchor >= Risk.R3 and re.fullmatch(r"tonight|right now|aaj raat|abhi|आज रात|अभी", current):
            screen.floor = Risk.R4
            screen.evidence = list(dict.fromkeys(screen.evidence + ["context", "immediacy"]))
        llm_start = time.perf_counter()
        try:
            if screen.floor == Risk.R4:
                assessment = Assessment(risk_level="R4", confidence=0.0, uncertain=True,
                    subject=screen.subject, timeframe="current", intent="present" if screen.intent else "not_disclosed",
                    plan="not_disclosed", means_access="present" if screen.means else "unknown",
                    immediacy="immediate", behavior="attempt" if screen.behavior else "none_disclosed",
                    violence="violence" in screen.evidence, safety_answer="unanswered", language=screen.language,
                    evidence=screen.evidence)
            else:
                assessment = await self.assessor.assess(message, context, previous.model_dump() if previous else None)
        except Exception as exc:
            failed, timed_out = True, isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            assessment = Assessment(risk_level=max(screen.floor, Risk.R1).name, confidence=0.0, uncertain=True,
                subject=screen.subject, timeframe="unknown", intent="present" if screen.intent else "not_disclosed",
                plan="not_disclosed", means_access="present" if screen.means else "unknown",
                immediacy="immediate" if screen.immediate else "unknown", behavior="attempt" if screen.behavior else "none_disclosed",
                violence="violence" in screen.evidence, safety_answer="unanswered", language=screen.language,
                evidence=list(dict.fromkeys(screen.evidence + ["failure"])))
        policy = decide(assessment, screen, previous, failed=failed, confidence_threshold=self.confidence_threshold)
        return SafetyResult(assessment=assessment, policy=policy, fallback=failed, timeout=timed_out,
            assessment_ms=(time.perf_counter()-started)*1000,
            llm_used=self.assessor.model is not None and screen.floor != Risk.R4, llm_ms=(time.perf_counter()-llm_start)*1000,
            screening_ms=screening_ms)

    async def finalize(self, message, draft, result, *, country=None, trusted_template=False, steering_alpha=0):
        # The trusted path is limited to exact internally constructed templates.
        template = safety_response(result, country, self.directory)
        trusted_template = trusted_template and draft == template
        validation = await self.validator.validate(message, draft, result,
            trusted_template=trusted_template, steering_alpha=steering_alpha)
        if validation.approved:
            return draft, validation
        replacement_check = await self.validator.validate(message, template, result, trusted_template=True)
        if not replacement_check.approved:
            raise RuntimeError("Safety fallback failed validation")
        return template, validation

    async def response(self, message, result, country=None):
        return await self.finalize(message, safety_response(result, country, self.directory), result,
                                   country=country, trusted_template=True)
