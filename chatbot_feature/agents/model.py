"""Adapter from the existing synchronous Hugging Face runtime to async agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LocalRuntimeModel:
    """``emotion_service`` is the backend's EmotionSteeringService.  With it, conversational
    replies go through the same steering selection and safeguards as the direct chat endpoints;
    without it the model never steers.
    """

    def __init__(self, runtime: Any, emotion_service: Any = None) -> None:
        self.runtime = runtime
        self.emotion_service = emotion_service
        if emotion_service is not None:
            self.complete_planned = self._complete_planned

    async def complete(self, prompt: str) -> str:
        # The existing API already runs each request in its own handler thread;
        # the runtime lock serializes access to the single local GPU model.
        result = self.runtime.generate(
            [{"role": "user", "content": prompt}],
            1024,
            True,
            0.2,
        )
        return str(result["response"])

    async def _complete_planned(self, prompt: str, plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Reply for an Emotion Reasoner plan; returns the text and the turn analysis
        (decision, measured intervention, emotion estimates), as in direct chat."""
        from backend.config import SYSTEM_PROMPT   # the prompt ``complete`` uses by default

        analysis = self.emotion_service.respond_with_plan(
            [{"role": "user", "content": prompt}], SYSTEM_PROMPT, plan, 1024)
        return str(analysis["reply"]), analysis


class CallableModel:
    """Small test/host adapter for an asynchronous completion callable."""

    def __init__(self, complete: Callable[[str], Any]) -> None:
        self._complete = complete

    async def complete(self, prompt: str) -> str:
        value = self._complete(prompt)
        if hasattr(value, "__await__"):
            value = await value
        return str(value)
