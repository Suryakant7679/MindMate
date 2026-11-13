"""Experimental response-tone layer for the Emotion Reasoner (prompt version 3).

The knowledge graph and the nine strategies are unchanged.  What changes is
the last step of the mapping

    user emotion -> emotional need -> response strategy -> desired response tone

In version 2 the last step was one of three "response emotions" (love for five
of the nine strategies, admiration, joy), which doubled as the steering target.
Here every strategy has its own tone, written for the reply and adjusted for
how strongly the user feels; the steering target stays a separate, optional
decision that calibration has to support.

``strategy_prompt_v3`` is used by experiments only.  The deployed agent keeps
``strategy_prompt`` until held-out results support a change.
"""
from __future__ import annotations

from typing import Any, Mapping

from .knowledge_graph import is_hinglish

# strategy: (tone label, tone asked of the reply)
TONES = {
    "crisis_support": ("calm_serious", "calm, serious and direct; no cheerfulness, no long explanations"),
    "compassionate_comfort": ("tender_sympathy", "tender and sympathetic: quietly sorry with them, not cheerful, not affectionate "
                                                 "beyond what a caring listener would say"),
    "warm_connection": ("companionable_warmth", "warm and companionable: glad they reached out, present, unhurried"),
    "reflective_validation": ("steady_validation", "steady and on their side: take the frustration seriously, without anger of "
                                                   "your own and without sweetness"),
    "grounding_reassurance": ("calm_reassurance", "calm and reassuring: steady, unhurried, quietly confident, never excited or alarmed"),
    "strengths_affirmation": ("sincere_respect", "sincere and respectful: honest encouragement in measured words, no gushing praise"),
    "celebratory_sharing": ("shared_gladness", "glad and lively: share the happiness at the user's own level of energy, not above it"),
    "appreciative_mirroring": ("warm_appreciation", "warmly appreciative: meet the fondness the user expresses without overstating it"),
    "gentle_exploration": ("friendly_curiosity", "friendly, relaxed and curious; lightly warm, not emotional"),
}

STYLE = ("Write two to four sentences in plain, natural words. Respond to what the user actually said; do not give advice, "
         "tips or techniques unless they ask. Do not repeat a phrase, restate the user's message at length, or use "
         "exaggerated or flowery emotional language. Ask at most one question, at the end.")

HINGLISH = ("The user writes Hinglish (Hindi in Roman letters mixed with English). Reply in the same natural Hinglish: "
            "Roman script, the same casual mix of Hindi and English words they use, no Devanagari, and do not switch to "
            "full English or to formal Hindi.")


def response_tone(plan: Mapping[str, Any]) -> dict[str, str]:
    """Tone label and wording for a plan (strategy, user emotion intensity and valence)."""
    label, text = TONES[plan["strategy"]]
    intensity, valence = float(plan.get("intensity") or 0.0), float(plan.get("valence") or 0.0)
    if plan["strategy"] == "crisis_support":
        modifier = ""
    elif intensity >= 0.7 and valence < 0:
        modifier = " The feeling is strong: keep the reply short and steady, and stay with the feeling."
    elif intensity >= 0.7:
        modifier = " The feeling is strong: let the reply show real energy."
    elif intensity < 0.4:
        modifier = " The feeling is mild: keep the reply light and do not dramatise it."
    else:
        modifier = ""
    return {"label": label, "text": text + "." + modifier}


def strategy_prompt_v3(base_prompt: str | None, plan: Mapping[str, Any], user_text: str, *, tone: bool = True,
                       style: bool = True, language: bool | None = None) -> str:
    """Version-3 system prompt: strategy guidance, then tone, then style and language rules.

    ``tone``, ``style`` and ``language`` (the Hinglish rule; follows ``style`` unless given) switch the
    additions off separately (ablations); with all off this is the version-2 prompt.
    """
    parts = [base_prompt] if base_prompt else []
    parts.append(f"Response strategy for this turn ({plan['strategy_title']}): {plan['guidance']}")
    if tone:
        parts.append(f"Tone: {response_tone(plan)['text']}")
    if style and plan["strategy"] != "crisis_support":
        parts.append(STYLE)
    if (style if language is None else language) and is_hinglish(user_text):
        parts.append(HINGLISH)
    return " ".join(parts)


def degenerate_reply(reply: str) -> bool:
    """A reply that is not a conversational answer: empty, a bare number or label, or a sentence fragment.

    EmoLLaMA sometimes answers a long instruction with an emotion-intensity score such as "0.896".
    """
    words = reply.strip().split()
    if len(words) < 4:
        return True
    letters = sum(character.isalpha() for character in reply)
    return letters < 0.5 * len(reply.replace(" ", ""))
