"""E5: quality-constrained adaptive selective emotion-neuron steering (decision logic only, no model dependency).

E4 (steering mode ``always``) steers nearly every reply with a strength set by the user's emotional state.  The
calibrated ``policy`` mode (E5-old) steers almost nothing, mostly because its calibration cells were set to 0 before
generation.  E5 decides per turn, in four steps, and records the reason for every decision:

1. **Safety** (before generation): the risk screen (``risk_screen.screen``) and the knowledge graph's crisis flag.
   Any risk category except ``none`` / ``figurative`` means no intervention.  Safety is never traded for tone.
2. **Context** (before generation): the frozen, model-specific policy cell for (strategy, language) - else
   (response emotion, language) - gives the strength multipliers to try, applied to E4's state-scaled strength
   (``alpha_E4 = low + (high - low) x state level``).  No cell, or multipliers ``()``, means no steering.  Cells come
   from development data only (``generation_analysis.e5_fit``); a context without development evidence is not
   steered.
3. **Emotional deficit** (after generation, response-aware): the unsteered (knowledge-graph-only) reply and the
   steered candidates are decoded together in one batch, as in E4.  If the policy enables the deficit gate and the
   unsteered reply already expresses the target emotion (gate-classifier score at or above the frozen threshold),
   the unsteered reply is kept.  A low score alone never triggers steering: steps 1 and 2 must already allow it.
4. **Acceptance** (after generation): each candidate, strongest first, must pass E4's safeguards (conversational,
   repetition, length, language) plus E5's: same language/script as the unsteered reply, bounded length inflation,
   and no risky phrasing that the unsteered reply did not have (familiarity / dependency, unsupported diagnosis,
   false reassurance).  The first candidate that passes is sent; otherwise the unsteered reply is sent.

None of this is a response critic (E6): no model judges or rewrites a reply.  The checks are the lightweight run-time
safeguards E4 already had, extended with lexical checks whose thresholds are frozen with the policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .knowledge_graph import is_hinglish
from .tone import degenerate_reply

POLICY_SCHEMA = 1
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LANGUAGES = ("english", "hinglish", "hindi")


def language_of(text: str) -> str:
    """``hindi`` (mostly Devanagari), ``hinglish`` (romanised Hindi, or Devanagari mixed into Latin text) or
    ``english``."""
    letters = [character for character in text or "" if character.isalpha()]
    if letters:
        devanagari = sum(bool(DEVANAGARI.match(character)) for character in letters) / len(letters)
        if devanagari >= 0.5:
            return "hindi"
    return "hinglish" if is_hinglish(text or "") else "english"


def repeated_bigram_rate(text: str) -> float:
    words = (text or "").lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


# ------------------------------------------------------------------------------------------- lexical markers
# Phrases that show emotional validation (logged as a deficit feature; adopted as a gate only if calibration supports
# it) and risky phrasing a steered candidate must not *add* (acceptance).  English, Hinglish, Hindi.
MARKERS: dict[str, tuple[str, ...]] = {
    "validation": (
        r"makes (complete |total |so much |perfect )?sense", r"(completely |totally |so )?understandable",
        r"it'?s (okay|ok|normal|natural|valid) to (feel|be)", r"(anyone|anybody) would (feel|be)",
        r"of course you('re| are| feel| would)", r"i hear you", r"no wonder",
        r"that sounds (really |so |incredibly |very )?(hard|tough|painful|difficult|frustrating|scary|lonely|overwhelming|exhausting|upsetting)",
        r"your feelings are valid", r"it'?s not your fault", r"you have every right",
        r"samajh (sakta|sakti|sakte|aata|aati) (hoon|hu|hai|hain)", r"bilkul (normal|theek|samajh)",
        r"aisa (feel|mehsoos) (karna|hona)", r"(feelings|feel karna) valid", r"tumhari galti nahi", r"aapki galti nahi",
        r"समझ (सकता|सकती|सकते)", r"स्वाभाविक", r"आपकी ग़लती नहीं", r"आपकी गलती नहीं", r"ऐसा महसूस करना",
    ),
    "familiarity": (
        r"\bi love you\b", r"\blove you\b", r"sweetheart", r"sweetie", r"\bhoney\b", r"my dear", r"\bdarling\b", r"\bbabe\b",
        r"\bdear friend\b", r"big (warm )?hug", r"(warm |tight )?hugs? (to you|for you)", r"sending (you )?(lots of )?(love|hugs)",
        r"\bjaanu\b", r"pyaar karta", r"pyaar karti", r"प्यार करता", r"प्यार करती",
    ),
    "dependency": (
        r"i('?ll| will) always be (here|there) for you", r"i'?m always (here|there) for you", r"always here for you",
        r"you (only|just) need me", r"you don'?t need anyone else", r"i'?m all you need", r"you can always count on me",
        r"\byou have me\b", r"i('?ll| will) never leave you", r"hamesha (tumhare|aapke) saath", r"main hamesha",
        r"हमेशा (तुम्हारे|आपके) साथ", r"मैं हमेशा",
    ),
    "diagnosis": (
        r"you (have|might have|may have|probably have|are suffering from|suffer from) (clinical )?"
        r"(depression|anxiety|ptsd|bipolar|ocd|adhd|an? (anxiety|mental|panic|eating) disorder|a disorder)",
        r"sounds like (clinical )?(depression|an? (anxiety|panic) disorder|ptsd|bipolar)", r"you are depressed",
        r"you'?re depressed", r"you should (take|start) (medication|medicine|pills|antidepressants)",
        r"(depression|anxiety) (hai|ho sakta) (aapko|tumhe)", r"डिप्रेशन है",
    ),
    "false_reassurance": (
        r"everything (will|is going to|'ll) be (fine|okay|ok|alright|all right|perfect)", r"nothing to worry about",
        r"i promise", r"it will all work out", r"it'?ll all work out", r"you will definitely", r"you'?ll definitely",
        r"sab (theek|thik) ho jayega", r"main promise", r"सब ठीक हो जाएगा", r"वादा करता", r"वादा करती",
    ),
}
RISKY = ("familiarity", "dependency", "diagnosis", "false_reassurance")
_COMPILED = {name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] for name, patterns in MARKERS.items()}


def markers(text: str) -> dict[str, list[str]]:
    """Marker categories found in ``text`` (category -> matched phrases)."""
    found: dict[str, list[str]] = {}
    for name, patterns in _COMPILED.items():
        hits = [match.group(0).lower() for pattern in patterns for match in [pattern.search(text or "")] if match]
        if hits:
            found[name] = hits
    return found


# ----------------------------------------------------------------------------------------------- the policy

@dataclass
class PolicyCell:
    """Strength multipliers to try, strongest first (``()`` = do not steer), and the evidence behind them."""
    multipliers: tuple[float, ...] = ()
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectivePolicy:
    model: str
    version: str = "e5-draft"
    frozen: bool = False
    alpha_range: tuple[float, float] = (0.25, 0.75)
    cells: dict[str, PolicyCell] = field(default_factory=dict)
    deficit: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "signal": "gate_target",
                                                             "threshold": None, "languages": ["english"]})
    acceptance: dict[str, Any] = field(default_factory=lambda: {
        "max_repetition_increase": 0.05, "min_length_ratio": 0.4, "max_word_ratio": None, "keep_language": True,
        "reject_added": list(RISKY)})
    safety: dict[str, Any] = field(default_factory=lambda: {"allowed_categories": ["none", "figurative"],
                                                            "respect_crisis_flag": True})
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = POLICY_SCHEMA
        data["alpha_range"] = list(self.alpha_range)
        data["cells"] = {key: {**asdict(cell), "multipliers": list(cell.multipliers)} for key, cell in self.cells.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectivePolicy":
        if data.get("schema", POLICY_SCHEMA) != POLICY_SCHEMA:
            raise ValueError(f"Unsupported E5 policy schema {data.get('schema')!r}.")
        cells = {key: PolicyCell(multipliers=tuple(float(m) for m in cell.get("multipliers", ())),
                                 reason=cell.get("reason", ""), evidence=dict(cell.get("evidence") or {}))
                 for key, cell in (data.get("cells") or {}).items()}
        defaults = cls(model=data["model"])
        return cls(model=data["model"], version=str(data.get("version", "e5-draft")), frozen=bool(data.get("frozen")),
                   alpha_range=tuple(data.get("alpha_range", defaults.alpha_range)), cells=cells,
                   deficit={**defaults.deficit, **(data.get("deficit") or {})},
                   acceptance={**defaults.acceptance, **(data.get("acceptance") or {})},
                   safety={**defaults.safety, **(data.get("safety") or {})},
                   provenance=dict(data.get("provenance") or {}))

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "SelectivePolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    # ---------------------------------------------------------------- lookup
    def cell_for(self, strategy: str | None, language: str, target: str | None) -> tuple[str | None, PolicyCell | None]:
        """Most specific cell: (strategy, language), else (response emotion, language), else (*, language)."""
        for key in (f"strategy:{strategy}|{language}", f"{target}|{language}", f"*|{language}"):
            if key in self.cells:
                return key, self.cells[key]
        return None, None

    def alpha_for(self, state_level: float | None) -> float:
        """E4's state-scaled strength (the base every multiplier applies to), rounded as
        ``EmotionSteeringService._steer_from_state`` rounds it."""
        low, high = self.alpha_range
        level = min(max(float(state_level or 0.0), 0.0), 1.0)
        return round(low + (high - low) * level, 2)


# ------------------------------------------------------------------------------------------------- decisions

def pre_decision(policy: SelectivePolicy, plan: Mapping[str, Any], user_text: str, risk: Mapping[str, Any],
                 *, panel: str | None) -> dict[str, Any]:
    """Steps 1-2: whether to generate steered candidates for this turn, and at which strengths.

    ``plan`` is the Emotion Reasoner's plan as steering mode ``always`` builds it (``state_level``,
    ``response_emotion``, ``strategy``, ``safety_flag``); ``panel`` is the model's panel for it (None: none).
    """
    language = language_of(user_text)
    base = {"language": language, "risk_category": risk.get("category"), "risk_cues": list(risk.get("cues") or []),
            "cell": None, "multipliers": [], "alphas": [], "alpha_e4": None}
    if policy.safety.get("respect_crisis_flag", True) and (plan.get("safety_flag") or plan.get("strategy") == "crisis_support"):
        return {**base, "eligible": False, "stage": "safety",
                "reason": "the Emotion Reasoner flagged a crisis; experimental steering is disabled on safety turns"}
    if risk.get("category") not in policy.safety.get("allowed_categories", ["none", "figurative"]):
        return {**base, "eligible": False, "stage": "safety",
                "reason": f"risk screen: {risk.get('category')} ({risk.get('reason')}); steering is disabled"}
    target = plan.get("response_emotion")
    if not target or not panel:
        return {**base, "eligible": False, "stage": "panel",
                "reason": f"this model has no neuron panel for the planned response emotion ({target})"}
    key, cell = policy.cell_for(plan.get("strategy"), language, target)
    if cell is None or not cell.multipliers:
        why = (cell.reason if cell is not None and cell.reason else
               "no development evidence for this context, so it is not steered")
        return {**base, "eligible": False, "stage": "context", "cell": key, "reason": why}
    # E4's own strength for this turn when the plan carries it (mode "always" planning), else the same formula.
    proposed = float(plan.get("steering_alpha") or 0.0)
    alpha_e4 = proposed if proposed > 0 else policy.alpha_for(plan.get("state_level"))
    alphas = [round(m * alpha_e4, 4) for m in cell.multipliers if m > 0]
    return {**base, "eligible": bool(alphas), "stage": "eligible" if alphas else "context", "cell": key,
            "multipliers": [m for m in cell.multipliers if m > 0], "alphas": alphas, "alpha_e4": alpha_e4,
            "reason": cell.reason or "the policy steers this context"}


def deficit_decision(policy: SelectivePolicy, target: str, unsteered_reply: str,
                     unsteered_scores: Mapping[str, float], language: str = "english") -> dict[str, Any]:
    """Step 3: does the knowledge-graph-only reply leave an emotional deficit worth steering?

    The gate applies only in the languages the policy names (``deficit["languages"]``): the audit found the gate
    classifier tracks the judges' tone on English replies (Spearman 0.52-0.69) but not on Hinglish (-0.02 to 0.55).
    """
    score = float(unsteered_scores.get(target, 0.0)) if target else 0.0
    found = markers(unsteered_reply)
    config = dict(policy.deficit or {})
    languages = config.get("languages", ["english"])
    if config.get("enabled") and language not in languages:
        config["enabled"] = False
    threshold = config.get("threshold")
    info = {"signal": config.get("signal", "gate_target"), "alignment": round(score, 4), "deficit": round(1 - score, 4),
            "threshold": threshold, "validation_markers": found.get("validation", []), "enabled": bool(config.get("enabled")),
            "language": language}
    if policy.deficit.get("enabled") and not config["enabled"]:
        return {**info, "steer": True, "reason": f"deficit gate not used for {language} replies (classifier unreliable there)"}
    if config.get("enabled") and threshold is not None and score >= float(threshold):
        return {**info, "steer": False,
                "reason": f"the knowledge-graph reply already expresses {target} (classifier {score:.2f} >= {float(threshold):.2f})"}
    if config.get("enabled") and config.get("skip_if_validated") and found.get("validation"):
        return {**info, "steer": False, "reason": "the knowledge-graph reply already validates the user's feelings"}
    return {**info, "steer": True, "reason": "emotional deficit: the knowledge-graph reply under-expresses the target"
            if config.get("enabled") else "deficit gate off for this model"}


def accept(policy: SelectivePolicy, unsteered: str, candidate: str, *, user_text: str = "") -> tuple[bool, str, dict[str, Any]]:
    """Step 4: E4's safeguards plus E5's checks for one candidate; ``(accepted, reason, checks)``."""
    rules = policy.acceptance
    checks: dict[str, Any] = {
        "degenerate": degenerate_reply(candidate) and not degenerate_reply(unsteered),
        "repetition_increase": round(repeated_bigram_rate(candidate) - repeated_bigram_rate(unsteered), 4),
        "length_ratio": round(len(candidate) / max(len(unsteered), 1), 3),
        "word_ratio": round(len(candidate.split()) / max(len(unsteered.split()), 1), 3),
        "language_unsteered": language_of(unsteered), "language_candidate": language_of(candidate),
        "user_language": language_of(user_text) if user_text else None,
    }
    before, after = markers(unsteered), markers(candidate)
    checks["added_markers"] = {name: after[name] for name in RISKY if name in after and name not in before}
    if checks["degenerate"]:
        return False, "not a conversational answer", checks
    if checks["repetition_increase"] > float(rules.get("max_repetition_increase", 0.05)):
        return False, "more repetitive than the unsteered reply", checks
    if checks["length_ratio"] < float(rules.get("min_length_ratio", 0.4)):
        return False, "much shorter than the unsteered reply", checks
    if rules.get("max_word_ratio") and checks["word_ratio"] > float(rules["max_word_ratio"]):
        return False, f"much longer than the unsteered reply ({checks['word_ratio']:.2f}x words)", checks
    # E4's rule: never leave the user's Hinglish for English (applies whatever ``keep_language`` says).
    if is_hinglish(unsteered) and not is_hinglish(candidate):
        return False, "left the user's Hinglish for English", checks
    if rules.get("keep_language", True) and checks["language_candidate"] != checks["language_unsteered"]:
        return False, f"changed language ({checks['language_unsteered']} -> {checks['language_candidate']})", checks
    rejected = [name for name in rules.get("reject_added", RISKY) if name in checks["added_markers"]]
    if rejected:
        return False, "added risky phrasing: " + ", ".join(f"{name} ({checks['added_markers'][name][0]})" for name in rejected), checks
    return True, "passed every safeguard", checks


def choose(policy: SelectivePolicy, unsteered: str, candidates: Sequence[tuple[float, str]], *,
           user_text: str = "") -> dict[str, Any]:
    """Step 4 over the candidates (``(alpha, reply)``, strongest first): the first accepted one is sent."""
    tried = []
    for alpha, reply in candidates:
        ok, reason, checks = accept(policy, unsteered, reply, user_text=user_text)
        tried.append({"alpha": alpha, "accepted": ok, "reason": reason, "checks": checks})
        if ok:
            return {"sent": "steered", "alpha": alpha, "reply": reply, "tried": tried}
    return {"sent": "unsteered", "alpha": 0.0, "reply": unsteered, "tried": tried}


__all__ = ["LANGUAGES", "MARKERS", "RISKY", "PolicyCell", "SelectivePolicy", "accept", "choose", "deficit_decision",
           "language_of", "markers", "pre_decision", "repeated_bigram_rate"]
