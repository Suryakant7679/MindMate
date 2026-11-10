"""Emotion knowledge graph used by the Emotion Reasoner Agent.

Node kinds
----------
``emotion``   the ten labels of the frozen recognition probe (with valence/arousal)
``cause``     a life context that can be recognised from cue phrases
``need``      what a person in that state usually needs from a listener
``strategy``  a therapist-inspired response strategy
``response``  an emotion whose neuron panel can be steered in the reply

Edge relations (all weighted 0..1)
----------------------------------
``cause  -evokes->        emotion``
``emotion -needs->        need``
``cause  -needs->         need``
``need   -addressed_by->  strategy``
``strategy -expresses->   response``

The graph is plain data so that every decision of the agent can be shown as a
path through it.  It encodes supportive-listening heuristics, not clinical
knowledge, and the agent is not a therapist.
"""
from __future__ import annotations

import re

import networkx as nx

# label: (valence, arousal)
EMOTIONS = {
    "admiration": (0.72, 0.48), "anger": (-0.75, 0.82), "disapproval": (-0.55, 0.55),
    "disgust": (-0.70, 0.65), "fear": (-0.70, 0.80), "joy": (0.85, 0.70), "love": (0.90, 0.55),
    "neutral": (0.05, 0.20), "sadness": (-0.80, 0.35), "surprise": (0.10, 0.75),
}

# cause: (description, cue phrases, {emotion: weight}, {need: weight})
CAUSES = {
    "crisis": ("thoughts of self-harm or not wanting to live",
               ("kill myself", "suicide", "end my life", "want to die", "wanna die", "hurt myself", "self harm",
                "self-harm", "no reason to live", "better off without me", "marna chahta", "marna chahti",
                "mar jaun", "mar jana chahta", "jeene ka mann nahi", "jeena nahi chahta", "khatam kar lun"),
               {"sadness": 1.0, "fear": 0.5}, {"immediate_safety": 1.0}),
    "loss_grief": ("losing someone or something loved",
                   ("passed away", "died", "death", "funeral", "lost my", "is no more", "guzar gaye", "guzar gayi",
                    "nahi rahe", "nahi rahi", "chal base"),
                   {"sadness": 1.0}, {"comfort": 1.0, "being_heard": 0.6}),
    "loneliness": ("feeling alone or unseen",
                   ("lonely", "alone", "akela", "akeli", "invisible", "nobody", "no one", "koi nahi", "left out",
                    "stopped talking to me", "ignore"),
                   {"sadness": 0.9}, {"connection": 1.0, "comfort": 0.5}),
    "self_worth": ("doubting one's own worth",
                   ("worthless", "disappointment", "failure", "not good enough", "useless", "loser", "hate myself",
                    "comparing myself", "peeche reh gaya", "peeche reh gayi", "kuch nahi kar sakta", "nalayak"),
                   {"sadness": 0.9, "disapproval": 0.3}, {"recognition": 1.0, "validation": 0.6}),
    "academic_pressure": ("study, exam or placement pressure",
                          ("exam", "failed", "marks", "grades", "semester", "placement", "study", "college",
                           "padhai", "result", "backlog", "assignment", "class"),
                          {"sadness": 0.5, "fear": 0.5}, {"reassurance": 0.7, "recognition": 0.6}),
    "work_stress": ("pressure or unfairness at work",
                    ("manager", "boss", "office", "deadline", "colleague", "took credit", "workload", "fired",
                     "laid off", "overtime"),
                    {"anger": 0.7, "fear": 0.3}, {"validation": 1.0, "being_heard": 0.5}),
    "relationship_conflict": ("conflict with someone close",
                              ("fight with", "argument", "breakup", "broke up", "partner", "girlfriend", "boyfriend",
                               "husband", "wife", "cheated", "jhagda", "ladai", "said things i regret"),
                              {"sadness": 0.6, "anger": 0.6}, {"validation": 0.8, "being_heard": 0.8}),
    "family_concern": ("worry about family",
                       ("mummy", "mom", "mother", "papa", "dad", "father", "parents", "family", "ghar se door",
                        "tabiyat", "hospital", "bhai", "behen", "sister", "brother"),
                       {"fear": 0.6, "sadness": 0.5}, {"reassurance": 0.7, "comfort": 0.6}),
    "anticipation_anxiety": ("fear of an upcoming event",
                             ("tomorrow", "presentation", "interview", "can't stop shaking", "nervous", "anxious",
                              "panic", "scared", "darr", "dar lag", "tension", "ghabrahat", "what if"),
                             {"fear": 1.0}, {"safety_calm": 1.0, "reassurance": 0.6}),
    "sleep_health": ("poor sleep or feeling unwell",
                     ("can't sleep", "cannot sleep", "insomnia", "mind keeps racing", "neend nahi", "exhausted",
                      "tired", "sick", "headache", "thak gaya", "thak gayi"),
                     {"fear": 0.5, "sadness": 0.4}, {"safety_calm": 0.8, "comfort": 0.5}),
    "numbness": ("feeling flat or empty",
                 ("don't feel anything", "do not feel anything", "numb", "empty", "nothing matters",
                  "everything is the same", "kuch feel nahi", "khali"),
                 {"sadness": 0.7, "neutral": 0.4}, {"gentle_exploration": 1.0, "connection": 0.5}),
    "achievement": ("a success or step forward",
                    ("got the job", "job offer", "promoted", "promotion", "passed my", "passed the", "pass ho gaya", "won", "selected", "cleared",
                     "first time", "finally", "achieved", "mil gaya", "mil gayi", "ho gaya", "started going"),
                    {"joy": 1.0, "admiration": 0.3}, {"shared_joy": 1.0, "recognition": 0.5}),
    "everyday_question": ("a practical, everyday question",
                          ("what should i", "how do i", "how to", "can you tell me", "can you help me", "recipe",
                           "cook", "schedule", "recommend", "suggest", "kaise karu", "kya banau"),
                          {"neutral": 1.0}, {"gentle_exploration": 1.0}),
    "celebration": ("a happy occasion",
                    ("birthday", "party", "baby", "wedding", "shaadi", "festival", "surprise dene", "celebrate",
                     "anniversary"),
                    {"joy": 1.0, "love": 0.4}, {"shared_joy": 1.0}),
}

# emotion: {need: weight}
EMOTION_NEEDS = {
    "sadness": {"comfort": 1.0, "being_heard": 0.7},
    "fear": {"safety_calm": 1.0, "reassurance": 0.8},
    "anger": {"validation": 1.0, "being_heard": 0.8},
    "disapproval": {"validation": 0.9, "being_heard": 0.7},
    "disgust": {"validation": 0.9, "being_heard": 0.6},
    "joy": {"shared_joy": 1.0},
    "admiration": {"shared_appreciation": 1.0},
    "love": {"shared_appreciation": 0.9, "connection": 0.5},
    "surprise": {"gentle_exploration": 0.8, "shared_joy": 0.3},
    "neutral": {"gentle_exploration": 1.0},
}

NEEDS = {
    "immediate_safety": "to be safe right now",
    "comfort": "to be comforted",
    "being_heard": "to feel heard without being judged",
    "connection": "to feel that someone is with them",
    "validation": "to have the feeling treated as understandable",
    "safety_calm": "to feel steadier and safe",
    "reassurance": "to hear that this is manageable",
    "recognition": "to have their worth and effort seen",
    "shared_joy": "to have the good moment shared",
    "shared_appreciation": "to have the warm feeling met",
    "gentle_exploration": "room to say more",
}

# need: {strategy: weight}
NEED_STRATEGIES = {
    "immediate_safety": {"crisis_support": 1.0},
    "comfort": {"compassionate_comfort": 1.0},
    "being_heard": {"compassionate_comfort": 0.5, "reflective_validation": 0.6},
    "connection": {"warm_connection": 1.0},
    "validation": {"reflective_validation": 1.0},
    "safety_calm": {"grounding_reassurance": 1.0},
    "reassurance": {"grounding_reassurance": 0.7, "strengths_affirmation": 0.4},
    "recognition": {"strengths_affirmation": 1.0},
    "shared_joy": {"celebratory_sharing": 1.0},
    "shared_appreciation": {"appreciative_mirroring": 1.0},
    "gentle_exploration": {"gentle_exploration": 1.0},
}

# strategy: (title, technique it borrows from, guidance for the reply, response emotion, base strength)
STRATEGIES = {
    "crisis_support": (
        "Crisis support", "safety-first response",
        "The user may be in danger. Respond calmly and directly, take what they said seriously, encourage "
        "contacting local emergency services or a crisis helpline and a trusted person nearby right now, and "
        "stay with them in the conversation.", None, 0.0),
    "compassionate_comfort": (
        "Compassionate comfort", "empathic reflection",
        "Acknowledge the pain before anything else and name the feeling in the user's own words. Stay with it; "
        "do not rush to advice or silver linings. Offer warm presence and one gentle open question.",
        "love", 0.8),
    "warm_connection": (
        "Warm connection", "unconditional positive regard",
        "Let the user feel accompanied: say plainly that you are here with them and glad they reached out. "
        "Reflect the loneliness without minimising it and invite them to share more.", "love", 0.7),
    "reflective_validation": (
        "Reflective validation", "validation and reflective listening",
        "Reflect the frustration as understandable given what happened. Do not argue, take sides against other "
        "people or tell the user to calm down. Ask what felt most unfair or what they need now.", "love", 0.6),
    "grounding_reassurance": (
        "Grounding reassurance", "normalising and grounding",
        "Normalise the worry in a slow, steady tone. Keep sentences short. Offer one small concrete grounding "
        "step, such as a slow breath, and remind them they do not have to face it alone.", "love", 0.7),
    "strengths_affirmation": (
        "Strengths affirmation", "strengths-based reframing",
        "Separate the event from the user's worth. Acknowledge the disappointment, then name a real strength or "
        "effort that is visible in what they shared. Do not exaggerate or dismiss the setback.",
        "admiration", 0.7),
    "celebratory_sharing": (
        "Celebratory sharing", "active-constructive responding",
        "Share the user's happiness openly, say what makes the moment meaningful and ask them to tell you more "
        "about it.", "joy", 0.8),
    "appreciative_mirroring": (
        "Appreciative mirroring", "affirming reflection",
        "Meet the warm feeling the user expresses, appreciate what they value and invite them to say more.",
        "admiration", 0.8),
    "gentle_exploration": (
        "Gentle exploration", "open-ended inquiry",
        "Be warm and curious. Answer what was asked, then leave room with one open question about how the user "
        "is doing.", "love", 0.4),
}

INTENSIFIERS = ("really", "so much", "very", "extremely", "can't", "cannot", "never", "always", "bahut", "bohot",
                "bilkul", "itna", "!!")
HINGLISH_WORDS = {"hai", "hain", "nahi", "nahin", "mera", "meri", "mujhe", "main", "kya", "aur", "yaar", "ho",
                  "raha", "rahi", "hoon", "tha", "thi", "kar", "ka", "ki", "ke", "se", "mein", "aaj", "kal", "bahut",
                  "mann", "sab", "koi", "kuch", "toh", "bhi", "gaya", "gayi", "ek"}


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for emotion, (valence, arousal) in EMOTIONS.items():
        graph.add_node(f"emotion:{emotion}", kind="emotion", label=emotion, valence=valence, arousal=arousal)
    for need, description in NEEDS.items():
        graph.add_node(f"need:{need}", kind="need", label=need, description=description)
    for strategy, (title, technique, guidance, response, strength) in STRATEGIES.items():
        graph.add_node(f"strategy:{strategy}", kind="strategy", label=strategy, title=title, technique=technique,
                       guidance=guidance, base_strength=strength)
        if response:
            graph.add_node(f"response:{response}", kind="response", label=response)
            graph.add_edge(f"strategy:{strategy}", f"response:{response}", relation="expresses", weight=1.0)
    for cause, (description, cues, emotions, needs) in CAUSES.items():
        graph.add_node(f"cause:{cause}", kind="cause", label=cause, description=description, cues=cues)
        for emotion, weight in emotions.items():
            graph.add_edge(f"cause:{cause}", f"emotion:{emotion}", relation="evokes", weight=weight)
        for need, weight in needs.items():
            graph.add_edge(f"cause:{cause}", f"need:{need}", relation="needs", weight=weight)
    for emotion, needs in EMOTION_NEEDS.items():
        for need, weight in needs.items():
            graph.add_edge(f"emotion:{emotion}", f"need:{need}", relation="needs", weight=weight)
    for need, strategies in NEED_STRATEGIES.items():
        for strategy, weight in strategies.items():
            graph.add_edge(f"need:{need}", f"strategy:{strategy}", relation="addressed_by", weight=weight)
    return graph


def matched_cues(text: str, cues: tuple[str, ...]) -> list[str]:
    """Cue phrases found in ``text`` on word boundaries."""
    lowered = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return [cue for cue in cues if re.search(r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])", lowered)]


def is_hinglish(text: str) -> bool:
    """Romanised Hindi or Devanagari, where steering loses fluency sooner."""
    if re.search(r"[ऀ-ॿ]", text):
        return True
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and sum(word in HINGLISH_WORDS for word in words) / len(words) >= 0.2
