"""Outgoing safety only. Semantic review supplements deterministic constraints."""
import re
from .schemas import Validation


def rejected(*violations):
    return Validation(approved=False, severity="critical", violations=list(violations), action="replace_with_safe_response")


# A chatbot cannot keep secrets: conversations are processed and may be stored, and a risk to life overrides
# confidentiality. Promises of secrecy in English, Hindi, romanized Hindi and Hinglish are false promises.
CONFIDENTIALITY_PROMISE = re.compile("|".join((
    r"\b(stays?|remains?|kept|keep (it|this|that|everything))\b.{0,20}\bbetween (us|you and me|the two of us)\b",
    r"\bbetween us (rahega|rahegi|rahenge|rahengi)\b",
    r"\bi(?:'ll| will| would|'d)? never (share|tell|repeat|reveal|disclose)\b.{0,30}\b(anyone|anybody|a soul)\b",
    r"\bi(?:'ll| will)? never (share|tell|repeat|reveal|disclose) (it|this|that|anything|any of (it|this)|your secrets?|what you)\b",
    r"\bi (won't|will not|wouldn't|would not) (share|tell|repeat|reveal|disclose)\b.{0,30}\b(anyone|anybody|a soul)\b",
    r"\b(your )?secrets? (is|are|will be) safe with me\b",
    r"\b(i'll|i will) keep (it|this|that|everything|your secrets?|what you (say|tell me))\b.{0,15}\b(secret|to myself|private|confidential)\b",
    r"\b(100%|totally|completely|fully|entirely|strictly) (private|confidential|secret|anonymous)\b",
    r"\b(no one|nobody|no-one) (else )?will (ever )?(know|find out|see|read)\b",
    r"हमारे बीच (ही )?(रहेगा|रहेगी|रहेंगी|रहेंगे)",
    r"किसी (और )?को (भी )?(कभी )?(नहीं|ना) (बताऊंगा|बताऊँगा|बताऊंगी|बताऊँगी)",
    r"किसी को (कभी )?पता (नहीं|ना) चलेगा",
    r"\b(hamare|humare|hamaare|humaare) (beech|bich) (hi |mein |me )?(rahega|rahegi|rahenge|rahengi)\b",
    r"\bkisi (aur )?ko (bhi )?(kabhi )?(nahi|nahin|nhi|na) (bataunga|bataungi|bataoonga|bataoongi|btaunga|btaungi)\b",
    r"\bkisi ko (kabhi )?pata (nahi|nahin|nhi|na) chalega\b",
)))
# Honest limits ("I can't promise to keep this secret if you're in danger") are not promises.
NO_PROMISE = re.compile(r"\b(can't|cannot|can not|couldn't|not able to|unable to) (promise|guarantee)\b.{0,50}"
                        r"\b(secret|confidential|private|between us|tell|share)"
                        r"|वादा नहीं कर सकत|\b(promise|guarantee|wada|vaada) (nahi|nahin|nhi) kar (sakta|sakti)\b")

# The chatbot cannot contact anyone, so claims that it has (or is about to) alert police, an ambulance or family,
# or that help is on its way, are false promises.  ("If I called ..." is hypothetical and not a claim.)
FALSE_ACTION_CLAIM = re.compile("|".join((
    r"(?<!if )(?<!when )\bi(?:'ve|'ll|'m| have| will| am)?(?: just| already)?\s+(alerted|notified|informed|called|"
    r"contacted|phoned|messaged|texted|dispatched|reached out to|sent (for|a message to|someone to)|alerting|notifying|"
    r"informing|calling|contacting|alert|notify|inform|call|contact|phone|message|text|send for)\b.{0,30}\b(police|cops|ambulance|emergency services|emergency|paramedics|"
    r"authorities|your (family|parents|mom|mum|mother|dad|father|friends?|school|college|doctor|therapist))\b",
    r"\b(help|the police|police|an ambulance|the ambulance|ambulance|paramedics|emergency services)"
    r"( is| are|'s|'re) (already )?(on (its|their|the) way|being sent|coming (to you|for you|now))\b",
    r"\bmaine\b.{0,40}\b(police|ambulance|emergency|helpline|family|parents|ghar ?wal\w*)\b.{0,30}"
    r"\b(bula|bulaa|bulwa|inform|call|phone|bata|khabar|message|suchit)\w*\b.{0,15}\b(di|diya|liya|li|hai|kar)\b",
    r"\b(police|ambulance|madad|help)\b.{0,15}\b(aa rahi|aa raha|aa rahe|raste (mein|me) (hai|hain)|"
    r"pahunch (rahi|raha|rahe)|nikal (chuki|chuka|chuke))\b",
    r"मैंने.{0,40}(पुलिस|एम्बुलेंस|एंबुलेंस|आपातकालीन|परिवार|घरवाल).{0,30}(बुला|सूचित|सूचना|फ़ोन|फोन|कॉल|बता)",
    r"(पुलिस|एम्बुलेंस|एंबुलेंस|मदद).{0,15}(आ रही|आ रहे|आ रहा|रास्ते में|पहुँच रह|पहुंच रह)",
)))

# Offers or framing of a way to die ("if you want it to be painless, there are ways ...") are harm instructions.
METHOD_OFFER = re.compile("|".join((
    r"\bpainless(ly)? (ways?|methods?|death|suicide|options?|exit)\b",
    r"\b(want|need|like|prefer) (it|this|that|death|dying) to be (quick (and|or) |fast (and|or) )?painless\b",
    r"\bif you want it (to be )?(quick (and|or) )?painless\b",
    r"\b(quickest|fastest|easiest|surest|most (effective|reliable|painless)) (way|method|means) to "
    r"(die|end (it|it all|your life|things)|kill yourself)\b",
    r"\bhow many (pills|tablets)\b.{0,30}\b(to die|would kill|is fatal|are fatal|is lethal|would be lethal|it takes to)\b",
    r"\b(bina dard|dard rahit|dard ke bina|painless)\b.{0,30}\b(tarik|tareek)\w*",
    r"(बिना दर्द|दर्द रहित|दर्दरहित|दर्द के बिना).{0,30}(तरीक|ढंग|उपाय)",
)))
# A refusal just before the match ("I won't describe painless ways ...") is not an offer.
REFUSAL = re.compile(r"\b(can't|cannot|can not|won't|will not|not going to|unable to|never|refuse to)\b|नहीं बता|nahi bata")


# Violations that can only be judged with the user's message: minimizing or shaming what they said, and missing
# urgent help when what they said needs it.  The in-context review decides only these.  Everything else is judged
# on the draft alone (and by the rules above): in live tests the in-context review was swayed by the user's
# wording, and on safe replies it named encouragement / harm instructions on most of them, often while also saying
# "approved".  Shaming is included because the models use it and minimization interchangeably for dismissive
# replies (Sarvam labelled "you're being too dramatic" in Hindi as shaming); it rejected no safe test draft.
CONTEXT_DEPENDENT = frozenset({"minimization", "shaming", "missing_urgent_help"})


def context_flags(review):
    """The context-dependent violations an in-context review names (its approved flag is not used)."""
    return [code for code in review.violations if code in CONTEXT_DEPENDENT]


def _offers_method(text):
    return any(not REFUSAL.search(text[max(0, m.start() - 40):m.start()]) for m in METHOD_OFFER.finditer(text))


def deterministic_check(draft, result, steering_alpha=0):
    t = draft.lower().replace("’", "'")
    violations = []
    patterns = {
        "encouragement": r"you should (kill|hurt) yourself|go ahead and (kill|hurt)|खुद को मार लो",
        "harm_instructions": r"to (kill|harm) yourself.{0,30}(take|use|mix)|lethal dose|घातक खुराक",
        "minimization": r"just get over it|stop being dramatic|others have it worse",
        "shaming": r"you are (a burden|worthless)|you're (a burden|worthless)",
        "diagnosis": r"you (definitely have|have been diagnosed by me)|you have (bipolar disorder|clinical depression|schizophrenia)",
        "discourages_care": r"(don't|do not|never) (see|contact|call|trust) (a doctor|a therapist|emergency|a professional)",
        "clinical_authority": r"as your (licensed therapist|psychiatrist|doctor)",
        "false_promise": r"(i have|i've|i will) (called|call|contacted|contact) (emergency|the police|an ambulance)|completely confidential|nobody will ever know",
    }
    for code, pattern in patterns.items():
        if re.search(pattern, t):
            violations.append(code)
    if "false_promise" not in violations and (FALSE_ACTION_CLAIM.search(t)
                                              or (CONFIDENTIALITY_PROMISE.search(t) and not NO_PROMISE.search(t))):
        violations.append("false_promise")
    if "harm_instructions" not in violations and _offers_method(t):
        violations.append("harm_instructions")
    if result.policy.route == "emergency" and not re.search(r"emergency|आपातकालीन", t):
        violations.append("missing_urgent_help")
    if not result.policy.steering_allowed and steering_alpha:
        violations.append("unsafe_steering")
    if not draft.strip():
        violations.append("validation_failure")
    return rejected(*violations) if violations else Validation(approved=True, severity="none", violations=[], action="approve")


class ResponseValidator:
    def __init__(self, assessor):
        self.assessor = assessor

    async def validate(self, message, draft, result, *, trusted_template=False, steering_alpha=0):
        check = deterministic_check(draft, result, steering_alpha)
        if not check.approved or trusted_template:
            return check
        try:
            # Two semantic checks.  First the draft alone, for every violation, strictly (an inconsistent answer
            # rejects): in live tests the user's own words ("... but I'm not thinking of hurting myself") led the
            # in-context check to approve self-harm encouragement and harm instructions.  Then in context, only
            # for the violations that depend on what the user said (CONTEXT_DEPENDENT).  A failed call or output
            # that does not parse still rejects.  Sequential on purpose: the safety model has one inference slot.
            alone = await self.assessor.validate_draft(draft, result.policy)
            if not alone.approved:
                return alone
            flags = context_flags(await self.assessor.review_in_context(message, draft, result.policy))
            return rejected(*flags) if flags else alone
        except Exception:
            return rejected("validation_failure")
