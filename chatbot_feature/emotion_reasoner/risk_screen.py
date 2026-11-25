"""Steering-eligibility risk screen (E5): may this user turn receive an experimental neuron intervention?

This is *not* a safety agent and it does not judge or repair replies.  It answers one narrow question before any
steering: is there a sign of risk that makes an experimental change to the model's activations inappropriate?  A
false positive only costs coverage (the reply is generated unsteered, exactly as the knowledge-graph-only reply),
so the screen is deliberately over-inclusive.

Categories (first match wins, most serious first):

  disclosure        first-person self-harm or suicidal intent, plan or act (the knowledge graph's crisis cause,
                    negation-aware, plus act / plan phrases)
  harm_to_others    intent to hurt someone else
  abuse_or_danger   the user describes being hurt, threatened or unsafe
  medical_emergency an acute physical emergency
  indirect          hopelessness, burden, finality or disappearance language without an explicit crisis phrase
  mention           a self-harm / suicide word that is negated, about someone else, or reported (news, a show)
  figurative        a death idiom with no other risk cue ("this exam is killing me")
  none              no risk cue

Steering is disallowed for every category except ``none`` and ``figurative``.  The distinction between a mention
and a disclosure is recorded for analysis (and matters for the reply itself), but for steering eligibility both
lead to the same decision: no intervention.

English, romanised Hindi (Hinglish) and Devanagari Hindi cues.  Lexicons are heuristics written before any E5
held-out message was drawn; their recall on held-out safety messages is reported, not assumed.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .knowledge_graph import CAUSES, NEGATIONS, matched_cues

ALLOWED = ("none", "figurative")
ORDER = ("disclosure", "harm_to_others", "abuse_or_danger", "medical_emergency", "indirect", "mention", "figurative",
         "none")

CRISIS_CUES = CAUSES["crisis"][1]
# Acts and plans that the knowledge graph's crisis cues do not name.
ACT_CUES = (
    "cut myself", "cutting myself", "cut my wrist", "slit my", "hang myself", "hanging myself", "jump off",
    "jump in front of", "overdose", "od on", "took pills", "take all my pills", "take all the pills", "pills to sleep forever",
    "starve myself", "burn myself", "kill me now", "not wake up", "never wake up", "sleep forever",
    "wrote a suicide note", "suicide note", "goodbye letter", "plan to end", "end things", "ending things",
    "khud ko nuksan", "khud ko hurt", "nas kaat", "zeher", "zehar", "phaansi", "fansi", "kood jaun", "kud jaun",
    "आत्महत्या", "ख़ुदकुशी", "खुदकुशी", "मरना चाहता", "मरना चाहती", "मर जाना चाहता", "मर जाना चाहती",
    "जीना नहीं चाहता", "जीना नहीं चाहती", "ख़ुद को नुकसान", "खुद को नुकसान", "नस काट", "ज़हर", "फांसी", "फाँसी",
)
INDIRECT_CUES = (
    "no point in living", "no point living", "no point anymore", "no point to anything", "what's the point of living",
    "whats the point of living", "nothing to live for", "can't go on", "cant go on", "cannot go on",
    "can't do this anymore", "cant do this anymore", "can't take it anymore", "cant take it anymore",
    "better off without me", "better off if i wasn't", "better off if i was gone", "burden to everyone",
    "burden on everyone", "i am a burden", "i'm a burden", "im a burden", "nobody would miss me", "no one would miss me",
    "nobody would notice if i", "no one would notice if i", "won't be around", "wont be around", "won't be here much longer",
    "not be here much longer", "saying goodbye to everyone", "give away my things", "giving away my things",
    "giving away my stuff", "tired of living", "tired of life", "tired of being alive", "want it all to stop",
    "want everything to stop", "want to disappear", "wish i could disappear", "wish i was never born",
    "wish i wasn't born", "wish i were dead", "wish i was dead", "rather be dead", "rather not exist",
    "don't want to exist", "dont want to exist", "don't see a future", "dont see a future", "no way out",
    "trapped with no way out", "everyone would be happier without me",
    "jeene ka koi matlab nahi", "jeene ka koi fayda nahi", "jeene ka mann nahi", "jeena bekaar", "sab khatam",
    "sab khatam kar", "koi fayda nahi jeene", "bojh hoon", "bojh hu", "sab par bojh", "gayab ho jaun", "gayab ho jana",
    "kabhi na uthun", "kabhi na uthu", "mere bina sab", "mar jaun toh", "mar jau toh", "zindagi se thak",
    "jeene se thak", "ab aur nahi", "aur nahi seh", "koi raasta nahi", "koi rasta nahi",
    "जीने का कोई मतलब नहीं", "जीने का मन नहीं", "सब ख़त्म", "सब खत्म", "बोझ हूँ", "बोझ हूं", "गायब हो जाऊं",
    "गायब हो जाऊँ", "कभी न उठूं", "कभी न उठूँ", "मेरे बिना सब", "ज़िंदगी से थक", "जिंदगी से थक", "अब और नहीं",
    "कोई रास्ता नहीं",
)
# Families of indirect risk language (regular expressions over the lower-cased message).  Written from the E5
# development messages' *kinds* of phrasing, not their wording; the held-out recall is measured, never tuned.
INDIRECT_PATTERNS = (
    r"(fall asleep|go to sleep|sleep) and (just )?(stay asleep|never wake|not wake|don'?t wake)",
    r"\b(never|not|don'?t) (want to )?wake up\b",
    r"\bno point (in )?(anything|anymore|living|going on|trying|pretending|carrying on|being here)\b",
    r"\b(if|when|once) i('?m| am| was| were) (gone|not here|not around|dead)\b",
    r"\bif i (wasn'?t|weren'?t|didn'?t exist|disappeared|was not|were not)\b",
    r"\b(disappear|vanish) (for good|forever)\b",
    r"\b(won'?t|wont|will not|not going to|don'?t think i'?ll|dont think ill) be (around|here) (much )?(longer|anymore|for long)\b",
    r"\b(i'?m|i am|feel like|feeling like) (just )?(a|such a) (burden|weight|drain|waste of space)\b",
    r"\b(dragging|pulling|weighing) (everyone|my (whole )?(family|friends|parents))( \w+)? down\b",
    r"\b(something|someone) (will|would|could|to) (just )?end it\b",
    r"\bmake (it|everything|this) (all )?stop\b",
    r"\b(taking up space|waste of oxygen|using up oxygen)\b",
    r"\bmere bina (sab|sabko|ghar|sabki|duniya)\b", r"\b(main|mai) na (rahoon|rahu|rahun)\b",
    r"\bkabhi na (uthun|uthu|uthoon)\b", r"\b(khatam|gayab) ho (jaun|jau|jaaun|jana chahta|jana chahti)\b",
    r"मेरे बिना", r"न रहूँ", r"न रहूं", r"गायब हो", r"ख़त्म हो जाऊँ", r"खत्म हो जाऊं", r"कभी न उठ",
)
_INDIRECT_RE = [re.compile(pattern) for pattern in INDIRECT_PATTERNS]
HARM_OTHERS_CUES = (
    "kill him", "kill her", "kill them", "want to hurt him", "want to hurt her", "want to hurt them",
    "going to hurt him", "going to hurt her", "going to hurt them", "gonna hurt him", "gonna hurt her",
    "i'll hurt him", "i'll hurt her", "beat him up", "beat her up",
    "make them pay", "get revenge", "take revenge", "shoot him", "shoot her", "stab him", "stab her", "stab them",
    "use a knife on",
    "maar dunga", "maar dungi", "maar daalunga", "maar daalungi", "jaan se maar", "badla lunga", "badla lungi",
    "मार दूंगा", "मार दूँगा", "मार दूंगी", "जान से मार", "बदला लूंगा", "बदला लूँगा",
)
ABUSE_CUES = (
    "hits me", "beats me", "beat me up", "abuses me", "abusive", "abused", "threatens me", "threatened me",
    "threatening me", "hurts me", "touched me", "forced me", "not safe at home", "unsafe at home", "afraid to go home",
    "scared to go home", "being followed", "stalking me", "stalker", "locked me in", "raped", "assaulted", "molested",
    "harassed", "harassing", "harassment", "harasses", "blackmail", "blackmailing", "blackmailed",
    "maarta hai", "maarti hai", "marta hai mujhe", "peetta", "peet-ta", "pitai", "dhamki", "zabardasti",
    "ghar pe safe nahi",
    "ghar mein safe nahi", "chhedta", "chhedti", "blackmail kar",
    "मारता है", "मारती है", "पीटता", "धमकी", "ज़बरदस्ती", "जबरदस्ती", "घर पर सुरक्षित नहीं", "घर में सुरक्षित नहीं",
)
MEDICAL_CUES = (
    "chest pain", "can't breathe", "cant breathe", "cannot breathe", "trouble breathing", "fainted", "passed out",
    "seizure", "bleeding a lot", "won't stop bleeding", "heart attack", "having a stroke", "overdosed", "poisoned",
    "seene mein dard", "saans nahi", "saans lene mein", "behosh", "khoon ruk nahi",
    "सीने में दर्द", "सांस नहीं", "साँस नहीं", "बेहोश", "खून रुक नहीं",
)
# Words that make a crisis cue a mention (someone else, media, the past) rather than a first-person disclosure.
THIRD_PERSON = ("my friend", "my brother", "my sister", "my mom", "my dad", "my mother", "my father", "my cousin",
                "my uncle", "my aunt", "my classmate", "my colleague", "a friend", "someone i know", "he ", "she ",
                "they ", "his ", "her ", "their ", "in the movie", "in the show", "in the news", "documentary",
                "article", "series", "book", "character", "mera dost", "meri dost", "mere dost", "uska", "uski",
                "usne", "unhone", "मेरा दोस्त", "मेरी दोस्त", "उसने", "उन्होंने")
FIGURATIVE = ("killing me", "is killing me", "are killing me", "dying to", "dying of laughter", "die laughing",
              "died laughing", "die of embarrassment", "died of embarrassment", "dead tired", "scared to death",
              "bored to death", "to die for", "dying for", "kill for", "could kill for", "mar gaya hasse",
              "has has ke mar", "hasi se mar", "thak ke mar", "jaan nikal gayi", "जान निकल गई")
SELF_HARM_WORDS = ("suicide", "suicidal", "self harm", "self-harm", "kill myself", "kill herself", "kill himself",
                   "kill themselves", "killed himself", "killed herself", "killed themselves", "want to die",
                   "wants to die", "wanted to die", "end his life", "end her life", "ended his life", "ended her life",
                   "overdose", "khudkushi", "aatmahatya", "आत्महत्या", "ख़ुदकुशी", "खुदकुशी")


def _normalise(text: str) -> str:
    return " " + re.sub(r"\s+", " ", (text or "").lower().replace("’", "'")) + " "


def _found(text: str, cues: Iterable[str]) -> list[str]:
    """Cue phrases present on word boundaries (Latin) or as substrings (Devanagari, which has no case or spacing
    conventions the boundary pattern could rely on)."""
    lowered = _normalise(text)
    hits = []
    for cue in cues:
        if re.search(r"[ऀ-ॿ]", cue):
            if cue in lowered:
                hits.append(cue)
        elif re.search(r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])", lowered):
            hits.append(cue)
    return hits


FIRST_PERSON = re.compile(r"(?<![a-z])(i|i'm|im|i've|i'd|i'll|me|myself|main|mai|mujhe|mujhse|khud)(?![a-z])"
                          r"|मैं|मुझे|मुझसे|ख़ुद|खुद")


def _first_person_near(lowered: str, cue: str, window: int = 5) -> bool:
    """The cue itself is first person ("kill myself") or a first-person word is among the ``window`` words before it."""
    if FIRST_PERSON.search(cue):
        return True
    position = lowered.find(cue)
    if position < 0:
        return False
    before = lowered[:position].split()[-window:]
    return any(FIRST_PERSON.fullmatch(word.strip(".,!?;:'\"")) for word in before)


def _negated_everywhere(text: str, cues: Sequence[str]) -> bool:
    """Every occurrence of every cue follows a negation ("I don't want to die")."""
    return bool(cues) and not matched_cues(text, tuple(cues), negatable=True)


def screen(text: str, history: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    """Risk category of the latest user message (earlier user turns raise the category, never lower it).

    Returns ``{"category", "cues", "steering_allowed", "reason", "source"}``; ``source`` is ``"message"`` or
    ``"history"`` (an earlier user turn of the same conversation carried the risk).
    """
    own = _classify(text)
    earlier = [_classify(turn.get("content", "")) for turn in history if turn.get("role") == "user"]
    serious = [item for item in earlier if item["category"] not in ALLOWED]
    if serious and ORDER.index(serious[0]["category"]) < ORDER.index(own["category"]):
        worst = min(serious, key=lambda item: ORDER.index(item["category"]))
        own = {**worst, "source": "history",
               "reason": worst["reason"] + " (in an earlier user turn of this conversation)"}
    return own


def _classify(text: str) -> dict[str, Any]:
    def result(category: str, cues: list[str], reason: str) -> dict[str, Any]:
        return {"category": category, "cues": cues, "steering_allowed": category in ALLOWED, "reason": reason,
                "source": "message"}
    lowered = _normalise(text)
    crisis = matched_cues(text, CRISIS_CUES, negatable=True)       # knowledge-graph crisis cue, not negated
    acts = _found(text, ACT_CUES)
    if crisis or acts:
        third = any(marker in lowered for marker in THIRD_PERSON)
        if not third or any(_first_person_near(lowered, cue) for cue in crisis + acts):
            return result("disclosure", crisis + acts, "first-person self-harm or suicide language")
    if _found(text, HARM_OTHERS_CUES):
        return result("harm_to_others", _found(text, HARM_OTHERS_CUES), "intent to hurt another person")
    if _found(text, ABUSE_CUES):
        return result("abuse_or_danger", _found(text, ABUSE_CUES), "the user describes being hurt, threatened or unsafe")
    if _found(text, MEDICAL_CUES):
        return result("medical_emergency", _found(text, MEDICAL_CUES), "an acute physical emergency")
    indirect = _found(text, INDIRECT_CUES) + [m.group(0) for p in _INDIRECT_RE for m in [p.search(lowered)] if m]
    if indirect:
        return result("indirect", indirect, "hopelessness, burden or finality language")
    words = _found(text, SELF_HARM_WORDS) + _found(text, CRISIS_CUES) + acts
    if words:
        third = any(marker in lowered for marker in THIRD_PERSON)
        why = ("negated" if _negated_everywhere(text, _found(text, CRISIS_CUES)) else
               "about someone else or reported" if third else "a self-harm word without first-person intent")
        return result("mention", sorted(set(words)), f"self-harm or suicide mention ({why})")
    figurative = _found(text, FIGURATIVE)
    if figurative:
        return result("figurative", figurative, "a death idiom with no other risk cue")
    return result("none", [], "no risk cue")


__all__ = ["ALLOWED", "ORDER", "screen", "NEGATIONS"]
