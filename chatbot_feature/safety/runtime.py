"""Local-only assessment adapter. A bounded worker survives request timeouts safely.

Timeout stops waiting, not GPU execution. No new work is queued while a timed-out
inference is still running. Runtime's existing lock keeps steering hooks serialized.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindmate-safety")
_SLOT = threading.BoundedSemaphore(1)


class SafetyRuntimeModel:
    def __init__(self, runtime):
        self.runtime = runtime

    async def complete_safety(self, prompt):
        if self.runtime is None or not _SLOT.acquire(blocking=False):
            raise RuntimeError("Safety inference unavailable or busy")
        def generate():
            # Only arguments every runtime's generate() accepts: SarvamRuntime.generate has no steering
            # parameters (passing steering_alpha raised TypeError there, so every Sarvam assessment failed).
            # Steering stays off because no runtime steers unless a strength is passed.
            try:
                return self.runtime.generate(
                    [{"role": "user", "content": prompt}], max_new_tokens=1024,
                    deterministic=True, temperature=0.0,
                    system_prompt="You are a safety classification component. Follow the supplied schema and return JSON only.")["response"]
            finally:
                _SLOT.release()
        try:
            future = _POOL.submit(generate)
        except Exception:
            _SLOT.release()
            raise
        return await asyncio.shield(asyncio.wrap_future(future))
