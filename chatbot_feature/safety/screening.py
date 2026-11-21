"""Provisional compositional screen; NOT a trained or validated risk classifier.

Scope, negation, behavior and time are separate signals. Every message still
receives semantic LLM assessment, including screen negatives. Replace via screen().
"""
import re
import unicodedata
from .schemas import Risk, Screen


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    for source, target in ((r"\bnahin\b|\bnhi\b", "nahi"), (r"\bhu\b", "hoon"),
                           (r"\bjeena\b", "jina"), (r"\bmarnaa\b", "marna")):
        text = re.sub(source, target, text)
    return text


def language(text: str) -> str:
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"\b(main|mujhe|mera|kaash|nahi|yaar|sab|jeene|mujhse|jaan|hoon|hai)\b", text.lower()):
        return "hinglish" if re.search(r"\b(i|everyone|exist|assignment|feel|without)\b", text.lower()) else "romanized_hi"
    return "en"


class ProvisionalScreen:
    def __init__(self, multilingual_normalization: bool = True):
        self.multilingual_normalization = multilingual_normalization

    def screen(self, text: str) -> Screen:
        t = normalize(text) if self.multilingual_normalization else text.lower()
        result = Screen(language=language(text))
        resolved_now = bool(re.search(r"safe now|ab safe|अभी सुरक्षित", t))
        # Clause-local exclusions: a denial or a quotation must not erase a later disclosure.
        clauses = re.split(r"[.!?;\n]|\bbut\b|\blekin\b|लेकिन", t)
        for clause in clauses:
            c = clause.strip()
            if not c:
                continue
            other = bool(re.search(r"\b(my friend|my sister|my brother|he said|she said|they said)\b|मेरा दोस्त", c))
            fictional = bool(re.search(r"writing a story|fictional|a character|in a novel|quote from|song lyric", c))
            historical = bool(re.search(r"used to|years ago|last year|in the past|i remember thinking", c))
            denied = bool(re.search(r"\b(don't|do not|never|not going to|no longer) (want to die|want to hurt|want to kill|feel suicidal|hurt myself|kill myself)|\bnot suicidal\b|marna nahi|खुद को.*नहीं", c))
            resolved = bool(re.search(r"safe now|safe with|with my (sister|friend|family)|ab safe|ab surakshit|अभी सुरक्षित", c))
            result.resolved |= resolved
            if fictional:
                result.subject = "fictional"
                continue
            if denied:
                continue
            passive = bool(re.search(r"better off without me|fine without me|wish.*(wouldn't wake|not wake|never wake|disappear|not exist)|disappear forever|jeene ka mann nahi|jina nahi|kaash.*(na uth|exist.*nahi)|जीने का मन नहीं|काश.*न उठ", c))
            active = bool(re.search(r"\b(kill myself|end(ing)? my life|want to die|wanna die|hurt myself|harm myself|self.harm|feel suicidal|thinking about suicide)\b|apni jaan|marna chaht|khatam kar (lun|doon)|खुदकुशी|आत्महत्या|अपनी जान|खुद को चोट", c))
            distress = bool(re.search(r"cannot handle|can't handle|can't keep going|cannot keep going|don't think i can keep going|pointless|hopeless|no reason to continue|don't see any reason|aur nahi (ho|sah)|aur nahi saha|sab khatam|kisi ko zarurat nahi|और नहीं सह|निराश", c))
            behavior = bool(re.search(r"(i('ve| have)?|maine|मैंने).*(already (taken|swallowed|overdosed|cut)|just (taken|swallowed|cut)|took an overdose)|attempt in progress|bleeding.*(won't|not) stop|गोलियां.*खा ली|ज़हर.*पी लिया", c))
            violence = bool(re.search(r"(i will|i am going to|i'm going to|going to).*(kill|stab|shoot) (him|her|them|my)|उसे मार दूंगा", c))
            immediate = bool(re.search(r"tonight|right now|today|in an hour|abhi|aaj raat|अभी|आज रात", c))
            intent = bool(re.search(r"i will|i'm going to|i am going to|decided|intend|ready to|karunga|karungi|लूंगा|लूँगा", c))
            means = bool(re.search(r"(have|holding|got|ready|access).*(pills|knife|gun|rope)|मेरे पास.*(गोलियां|चाकू)", c))
            floor = Risk.R4 if behavior or ((active or violence) and immediate and (intent or means)) else (
                Risk.R3 if active or violence else Risk.R2 if passive else Risk.R1 if distress else Risk.R0)
            if historical and not (resolved or resolved_now):
                floor = min(floor, Risk.R2)
            elif historical and (resolved or resolved_now):
                floor = Risk.R0
            if other:
                result.subject = "other"  # Still act on a real other person's emergency.
            if floor >= result.floor:
                result.floor = floor
                result.intent, result.means, result.immediate, result.behavior = intent, means, immediate, behavior
            if floor:
                result.evidence.extend([code for flag, code in ((passive, "death_wish"), (active, "ideation"),
                    (distress, "distress"), (behavior, "behavior"), (violence, "violence"),
                    (immediate, "immediacy"), (historical, "historical"), (other, "third_person")) if flag])
        result.evidence = list(dict.fromkeys(result.evidence))[:16]
        return result
