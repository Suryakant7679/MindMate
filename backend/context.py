"""Conversation history sent to a model: normalised, then fitted to that model's context window.

Stored conversations can contain things a chat template rejects: a user message whose reply failed or was stopped
(two user turns in a row), or nothing but whitespace.  ``normalize`` turns the stored history into strict
user/assistant alternation that starts with the user and ends with the new user message; consecutive user messages
are joined so nothing the user wrote is lost.

``fit`` keeps the newest complete turns that fit the selected model's prompt budget, counted with that model's own
tokenizer (Llama 3.1 and Sarvam: 16,384-token app window; Llama-2: 4,096; EmoLLaMA: 2,048).  When a conversation
moves to a model with a shorter window, older turns are left out of that model's prompt, not deleted.  The runtime
still checks the exact templated length and drops further turns if needed.
"""
from __future__ import annotations

from typing import Any, Callable

MAX_HISTORY_MESSAGES = 200          # upper bound before token counting (a 16k window never holds more)
PER_MESSAGE_OVERHEAD = 8            # role markers / separators each turn adds in a chat template
SYSTEM_RESERVE = 384                # strategy guidance and the check-in prompt the Emotion Reasoner may add


def normalize(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()
        if not out and role == "assistant":
            continue                                   # a model turn cannot open the conversation
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": out[-1]["content"] + "\n\n" + content}
        else:
            out.append({"role": role, "content": content})
    while out and out[-1]["role"] != "user":
        out.pop()
    return out


def token_counter(runtime: Any) -> Callable[[str], int]:
    """Count tokens with the runtime's tokenizer; a length estimate when it has none (tests, fakes)."""
    tokenizer = getattr(runtime, "tokenizer", None)

    def count(text: str) -> int:
        if tokenizer is not None:
            try:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
            except Exception:       # noqa: BLE001 - fall back to an estimate
                pass
        return max(1, len(text) // 3)
    return count


def prompt_budget(runtime: Any, max_new_tokens: int) -> int:
    """Prompt tokens the runtime allows for a reply of ``max_new_tokens`` (see ``HuggingFaceRuntime._context_budget``)."""
    budget = getattr(runtime, "_context_budget", None)
    if callable(budget):
        try:
            return int(budget(max_new_tokens))
        except Exception:           # noqa: BLE001
            pass
    window = int(getattr(runtime, "context_tokens", 4096) or 4096)
    return max(window - max_new_tokens, window // 4)


def fit(messages: list[dict[str, str]], *, count: Callable[[str], int], budget: int,
        system_prompt: str | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Newest suffix of ``messages`` (normalised) that fits ``budget`` tokens, starting with a user turn.

    The last user message is always kept; whether it fits on its own is for the runtime to decide (it raises a
    clear error).  Returns the messages and a description of what was left out.
    """
    total = len(messages)
    window = messages[-MAX_HISTORY_MESSAGES:]
    available = budget - (count(system_prompt) if system_prompt else 0) - SYSTEM_RESERVE
    kept: list[dict[str, str]] = []
    used = 0
    for message in reversed(window):
        cost = count(message["content"]) + PER_MESSAGE_OVERHEAD
        if kept and used + cost > available:
            break
        kept.append(message)
        used += cost
    kept.reverse()
    while len(kept) > 1 and kept[0]["role"] != "user":
        used -= count(kept[0]["content"]) + PER_MESSAGE_OVERHEAD
        kept.pop(0)
    info = {"messages_total": total, "messages_sent": len(kept), "messages_left_out": total - len(kept),
            "estimated_prompt_tokens": max(used, 0), "prompt_budget": budget, "trimmed": len(kept) < total}
    return kept, info
