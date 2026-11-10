"""Small English/Hinglish word lists used to label generated words.

The lexicon is deliberately conservative.  It is one of two signals for an
"emotional" word (the other is the matched-prompt log-probability ratio) and
the only signal that excludes a word from the "semantic" class.
"""
from __future__ import annotations

import re

EMOTION_LEXICON: dict[str, frozenset[str]] = {
    "admiration": frozenset("""
        admire admiration admirable amazing awesome brilliant impressive impressed incredible
        inspiring inspired inspirational remarkable respect respectable outstanding excellent
        wonderful fantastic talented praise proud commendable kamaal zabardast shandaar
        superb stellar legend legendary bravo kudos""".split()),
    "anger": frozenset("""
        angry anger furious fury rage raging enraged mad outraged outrage irritated irritating
        annoyed annoying infuriating livid frustrated frustrating frustration hate hatred
        resent resentment gussa gussaa naraz pissed fuming unacceptable""".split()),
    "disapproval": frozenset("""
        disapprove disapproval disappointing disappointed disagree wrong unfair inappropriate
        unacceptable shameful irresponsible careless bad poor criticize criticise disapproving
        objection objectionable galat bura""".split()),
    "disgust": frozenset("""
        disgust disgusting disgusted gross revolting repulsive repulsed nasty vile sick sickening
        nauseating yuck eww ew horrible awful appalling ghinn chee chhi""".split()),
    "fear": frozenset("""
        fear afraid scared scary frightened frightening terrified terrifying anxious anxiety
        worried worry worrying nervous panic panicked dread dreadful alarming uneasy darr dar
        dara ghabrahat tension""".split()),
    "joy": frozenset("""
        joy joyful happy happiness glad delighted delightful thrilled excited exciting cheerful
        yay wonderful fantastic great fun awesome ecstatic overjoyed elated khush khushi maza
        mazaa celebrate celebration""".split()),
    "love": frozenset("""
        love loving loved lovely adore adorable affection affectionate care caring cherish
        cherished dear darling sweetheart heart hearts warmth tender pyaar pyar mohabbat ishq
        jaan hug hugs""".split()),
    "sadness": frozenset("""
        sad sadness sorrow sorry unhappy heartbroken heartbreaking depressed depressing gloomy
        miserable lonely grief grieving tears crying cry hurt painful pain upset down dukh
        udaas udas""".split()),
    "surprise": frozenset("""
        surprise surprised surprising shocked shocking astonished astonishing amazed stunned
        unexpected unbelievable wow whoa omg really seriously incredible sach achanak""".split()),
    "neutral": frozenset(),
}
ALL_EMOTION_WORDS = frozenset().union(*EMOTION_LEXICON.values())

STOPWORDS = frozenset("""
    a an the and or but if then so than that this these those there here it its it's is are was
    were be been being am do does did done have has had having i me my mine we us our ours you
    your yours he him his she her hers they them their theirs what which who whom whose when
    where why how all any both each few more most other some such no nor not only own same too
    very can will just should would could may might must shall of at by for with about against
    between into through during before after above below to from up down in out on off over
    under again further once also as like get got go going let lets make really s t don't can't
    i'm you're it's that's there's i've you've we're they're yes okay ok oh well
    hai hain ho tha thi the ka ki ke ko se me mein main mai tu tum aap hum yeh ye woh wo kya
    kyun kaise toh to bhi aur ya nahi na hi par pe jo ab bas kuch koi
""".split())

WORD_PATTERN = re.compile(r"[a-z][a-z'’]*")


def normalize_word(text: str) -> str:
    """Lowercase and strip a generated word to its alphabetic core."""
    match = WORD_PATTERN.search(text.lower().replace("’", "'"))
    return match.group(0).strip("'") if match else ""


def is_emotion_word(word: str) -> bool:
    return word in ALL_EMOTION_WORDS


def is_content_word(word: str) -> bool:
    return len(word) >= 3 and word not in STOPWORDS


def content_words(text: str) -> set[str]:
    """Content words with emotion-lexicon words removed (for semantic overlap)."""
    words = {normalize_word(token) for token in text.split()}
    return {word for word in words if is_content_word(word) and not is_emotion_word(word)}
