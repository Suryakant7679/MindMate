"""One chat turn of a stored conversation: Emotion Reasoner plan, generation and streaming.

1. the Emotion Reasoner plans strategy, response emotion and strength from the model's calibrated policy;
2. steering is applied only where that policy steers and the run-time gate accepts the steered reply;
3. the reply is streamed to the client as it is generated.

Streaming: without planned steering the reply is streamed token by token.  With planned steering the unsteered reply
and the steered candidate are decoded together, the gate picks one, and that reply is then sent; text never shows
a candidate the gate rejected.  A stopped reply keeps the text the user has already seen.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

from chatbot_feature.emotion_reasoner.trajectory import reply_state

from .config import STEERING_GAIN, ModelSpec
from .generation_control import GenerationCancelled, GenerationControl, use

Emit = Callable[[str, dict[str, Any]], Any]


def steering_metadata(runtime: Any, service: Any, plan: dict[str, Any] | None, analysis: dict[str, Any] | None,
                      steering_mode: str) -> dict[str, Any]:
    """What steering did for one reply: settings, decision, panel size and layers (stored with the message)."""
    panels = getattr(runtime, "_emotion_panels", None) or {}
    directions = getattr(runtime, "_emotion_directions", None) or {}
    meta: dict[str, Any] = {"mode": steering_mode, "supported": bool(panels),
                            "policy_enabled": bool(getattr(service, "steering_enabled", False)) if service else False,
                            "applied": False, "planned_alpha": 0.0, "applied_alpha": 0.0}
    if plan is None or analysis is None:
        meta.update(status="not_run", reason="this model has no Emotion Reasoner pipeline: unsteered reply")
        return meta
    decision = analysis.get("decision") or {}
    planned = float(decision.get("planned_alpha") or 0.0)
    applied = float(decision.get("applied_alpha") or 0.0)
    shadow = decision.get("shadow_panel")
    panel = (plan.get("steering_panel") or plan.get("response_emotion")) if planned else shadow
    meta.update(status=decision.get("status"), reason=decision.get("reason"), disabled_by=decision.get("disabled_by"),
                planned_alpha=planned, applied_alpha=applied, applied=bool(applied), panel=panel,
                response_emotion=plan.get("response_emotion"), strategy=plan.get("strategy"),
                prompt_style=plan.get("prompt_style"), state_level=plan.get("state_level"),
                policy_alpha=plan.get("policy_alpha"),
                factors=[{"factor": item.get("factor"), "effect": item.get("effect")}
                         for item in plan.get("strength_factors") or []])
    if panel:
        layers = panels.get(panel) or {}
        if layers:
            meta.update(neurons=sum(len(neurons) for neurons, _ in layers.values()), layers=sorted(int(k) for k in layers))
        if panel in directions:
            meta["direction_layer"] = int(directions[panel][0])
        meta.update(gain=STEERING_GAIN, location=getattr(runtime, "steering_location", None))
    if shadow:
        meta.update(shadow=True, shadow_alpha=decision.get("shadow_alpha"))
    measured = (analysis.get("steering") or {}).get("decode_steps")
    if measured is not None:
        meta["decode_steps"] = measured
    return meta


def tone_record(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Emotional tone of the reply before steering (unsteered version) and after steering (steered version).

    Both versions are written in one batched pass; the GoEmotions classifier scores each text.  ``after`` is None
    when no steered version was generated.  The numbers describe the reply text, not the user.
    """
    effect = (analysis or {}).get("effect") or {}
    if not effect.get("unsteered_scores"):
        return None
    target = effect.get("target")

    def describe(text: str | None, scores: dict[str, float] | None) -> dict[str, Any] | None:
        if not text or not scores:
            return None
        state = reply_state(text, scores, target)
        top = sorted(state["distribution"].items(), key=lambda item: -item[1])[:3]
        return {"emotion": state["emotion"], "expected_valence": state["expected_valence"],
                "target_score": state["target_score"], "top": [{"emotion": k, "share": v} for k, v in top],
                "distribution": state["distribution"]}
    before = describe(effect.get("unsteered_reply"), effect.get("unsteered_scores"))
    after = describe(effect.get("steered_reply"), effect.get("steered_scores"))
    change = {}
    if before and after:
        if before["target_score"] is not None and after["target_score"] is not None:
            change["target_score"] = round(after["target_score"] - before["target_score"], 4)
        change["expected_valence"] = round(after["expected_valence"] - before["expected_valence"], 4)
    return {"method": effect.get("method"), "target": target, "sent": effect.get("sent"),
            "shadow": bool(effect.get("shadow")), "before": before, "after": after, "change": change or None}


class TurnEngine:
    """Runs turns against a leased runtime.  ``resource_owners`` is the server's turn-ownership store."""

    def __init__(self, resource_owners: Any) -> None:
        self.resource_owners = resource_owners

    def run(self, *, runtime: Any, service: Any, spec: ModelSpec, owner: str, conversation_key: str,
            messages: list[dict[str, str]], steering_mode: str, control: GenerationControl,
            emit: Emit) -> dict[str, Any]:
        """Generate and stream one reply.  ``result["cancelled"]`` is True when the user stopped it; ``reply`` is
        then the text already sent (possibly empty)."""
        started = time.perf_counter()
        released: list[str] = []

        def on_text(piece: str) -> None:
            if control.cancelled:
                raise GenerationCancelled()      # stop streaming at once; the research record is not written
            if not released:
                piece = piece.lstrip()
            if piece:
                released.append(piece)
                emit("delta", {"text": piece})

        emit("status", {"phase": "generating"})
        plan = analysis = turn_id = None
        cancelled = False
        try:
            with use(control):
                if service is not None:
                    turn_id, plan = service.begin_turn(messages, conversation_id=conversation_key,
                                                       steering_mode=steering_mode)
                    self.resource_owners.put(turn_id, owner)
                    self.resource_owners.put(conversation_key, owner)
                    analysis = service.respond(turn_id, max_new_tokens=spec.max_new_tokens, on_text=on_text)
                    reply = analysis["reply"]
                else:
                    for piece in runtime.stream(messages, max_new_tokens=spec.max_new_tokens,
                                                deterministic=spec.deterministic, temperature=spec.temperature):
                        on_text(piece)
                    reply = "".join(released).strip()
                    if not reply:
                        raise RuntimeError("The model returned an empty response.")
            cancelled = control.cancelled
        except GenerationCancelled:
            cancelled = True
        except RuntimeError:
            if not control.cancelled:          # a reply stopped before its first token reads as "empty"
                raise
            cancelled = True
        if cancelled:
            reply = "".join(released).strip()
        return {"reply": reply, "cancelled": cancelled, "turn_id": None if cancelled else turn_id, "plan": plan,
                "decision": (analysis or {}).get("decision"),
                "steering": (steering_metadata(runtime, service, plan, analysis, steering_mode) if not cancelled
                             else {"mode": steering_mode, "status": "cancelled", "applied": False}),
                "tone": None if cancelled else tone_record(analysis),
                "latency_seconds": round(time.perf_counter() - started, 3), "generated_tokens": control.steps}


# ----------------------------------------------------------------------------- titles

TITLE_PROMPT = ("Write a short, neutral title of 3 to 6 words for a conversation that begins with the message below. "
                "Reply with the title only, without quotes.\n\nMessage:\n{text}")
TITLE_SYSTEM = "You write short conversation titles."
NEUTRAL_TITLE = "Support conversation"
_REFUSAL = re.compile(r"\b(sorry|cannot|can't|unable|as an ai|i am an ai|i'm an ai)\b", re.I)


def clean_model_title(raw: str) -> str | None:
    line = next((part for part in str(raw).splitlines() if part.strip()), "")
    line = re.sub(r"^\s*(title|conversation title)\s*[:\-]\s*", "", line, flags=re.I)
    line = line.strip().strip("\"'“”‘’*#`_ ").strip()
    line = re.sub(r"[.!?:;,]+$", "", line).strip()
    words = line.split()
    if not 1 <= len(words) <= 10 or len(line) > 80 or _REFUSAL.search(line) or "<" in line:
        return None
    return line[:1].upper() + line[1:]


def fallback_title(text: str) -> str:
    first = re.split(r"(?<=[.!?।])\s+", text.strip(), maxsplit=1)[0]
    words = first.split()
    title = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    return (title[:1].upper() + title[1:]) if title else "New chat"


def make_title(runtime: Any, user_text: str, sensitive: bool = False) -> tuple[str, str]:
    """``(title, source)`` for a new conversation from its first message: the model's short summary, a neutral title
    when the Emotion Reasoner read the first message as a crisis, the opening words when the model fails."""
    if sensitive:
        return NEUTRAL_TITLE, "fallback"
    try:
        output = runtime.generate([{"role": "user", "content": TITLE_PROMPT.format(text=user_text[:500])}],
                                  max_new_tokens=20, deterministic=True, temperature=0.0, system_prompt=TITLE_SYSTEM)
        title = clean_model_title(output.get("response", ""))
        if title:
            return title, "auto"
    except Exception as exc:        # noqa: BLE001 - a title never fails a turn
        print(f"Title generation failed ({type(exc).__name__}); using the opening words.")
    return fallback_title(user_text), "fallback"


# ----------------------------------------------------------------------------- server-sent events

class EventStream:
    """Server-sent events over a ``BaseHTTPRequestHandler`` response, with keep-alive comments.

    A failed write means the client went away: ``on_disconnect`` is called once (it cancels the generation).
    """

    def __init__(self, handler: Any, on_disconnect: Callable[[], None] | None = None, heartbeat: float = 10.0) -> None:
        self.handler, self.on_disconnect, self.heartbeat = handler, on_disconnect, heartbeat
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self.closed = False
        self.started = False

    def start(self, headers: dict[str, str] | None = None) -> None:
        handler = self.handler
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Accel-Buffering", "no")
        handler.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.end_headers()
        self.started = True
        if self.heartbeat > 0:
            threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self) -> None:
        while not self._stopped.wait(self.heartbeat):
            self._write(b": keep-alive\n\n")

    def _write(self, payload: bytes) -> bool:
        with self._lock:
            if self.closed:
                return False
            try:
                self.handler.wfile.write(payload)
                self.handler.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                self.closed = True
        if self.on_disconnect:
            self.on_disconnect()
        return False

    def send(self, event: str, data: dict[str, Any]) -> bool:
        body = json.dumps(data, ensure_ascii=False, default=str)
        return self._write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))

    def close(self) -> None:
        self._stopped.set()
        with self._lock:
            self.closed = True


__all__ = ["EventStream", "GenerationCancelled", "TurnEngine", "make_title", "steering_metadata"]
