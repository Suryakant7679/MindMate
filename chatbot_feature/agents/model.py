"""Adapter from the existing synchronous Hugging Face runtime to async agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LocalRuntimeModel:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

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

    async def complete_steered(self, prompt: str, plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Generate with the Emotion Reasoner's steering plan; also return the hook trace."""
        trace: dict[str, Any] = {}
        result = self.runtime.generate(
            [{"role": "user", "content": prompt}], 1024, True, 0.2,
            steering_alpha=float(plan["steering_alpha"]), steering_emotion=plan["response_emotion"],
            steering_trace=trace,
        )
        return str(result["response"]), trace


class CallableModel:
    """Small test/host adapter for an asynchronous completion callable."""

    def __init__(self, complete: Callable[[str], Any]) -> None:
        self._complete = complete

    async def complete(self, prompt: str) -> str:
        value = self._complete(prompt)
        if hasattr(value, "__await__"):
            value = await value
        return str(value)
