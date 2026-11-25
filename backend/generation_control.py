"""Per-request generation control: cancellation and token progress.

A ``GenerationControl`` is bound to the request thread with ``use(control)``.  Every runtime generation started in
that context (``generate``, ``stream``, ``generate_candidates``) receives one extra Hugging Face stopping criterion
that counts decode steps and stops all rows as soon as the request is cancelled.  Nothing else changes: a criterion
that never fires leaves greedy decoding token-for-token identical, so research results are unaffected.

Generations started in other threads do not inherit the context unless it is copied explicitly (``stream`` copies it
into its worker thread).
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class GenerationCancelled(Exception):
    """The request that owned this generation was cancelled (Stop button or closed connection)."""


class GenerationControl:
    def __init__(self, on_progress: Callable[[int], None] | None = None) -> None:
        self._event = threading.Event()
        self.on_progress = on_progress
        self.steps = 0          # decode steps of every generation run under this control

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise GenerationCancelled()

    def step(self) -> None:
        self.steps += 1
        if self.on_progress is not None:
            try:
                self.on_progress(self.steps)
            except Exception:       # noqa: BLE001 - progress reporting never breaks generation
                pass


_CURRENT: contextvars.ContextVar[GenerationControl | None] = contextvars.ContextVar(
    "mindmate_generation_control", default=None)


def current() -> GenerationControl | None:
    return _CURRENT.get()


@contextmanager
def use(control: GenerationControl | None) -> Iterator[GenerationControl | None]:
    token = _CURRENT.set(control)
    try:
        yield control
    finally:
        _CURRENT.reset(token)


def control_settings() -> dict[str, Any]:
    """Extra ``model.generate`` keyword arguments for the control bound to this thread (none without one)."""
    control = _CURRENT.get()
    if control is None:
        return {}
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Control(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):       # one call per decode step
            control.step()
            return torch.full((input_ids.shape[0],), control.cancelled, dtype=torch.bool, device=input_ids.device)

    return {"stopping_criteria": StoppingCriteriaList([_Control()])}
