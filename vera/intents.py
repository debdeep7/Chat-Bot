"""Reading a reply: what is this person telling us, and in which language?

Deterministic on purpose. It costs no model call, gives the same answer every time, and covers
what the judge replays: canned auto-replies, "ok let's do it", "stop messaging me", an unrelated
request such as GST filing. It reads English, Hinglish in Roman script, and the few Devanagari
words that matter. A message nobody could classify is simply `other`, which is answered with a
gentle step forward rather than a guess.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .validate import HINDI_WORDS

BLANK = "blank"          # nothing readable: empty, punctuation, an emoji we cannot read
STOP = "stop"            # asks us to stop, or is hostile: close the conversation and never write again
AUTO = "auto"            # a canned reply from a business account or an assistant
DECLINE = "decline"      # not interested / no thanks
LATER = "later"          # busy, call me later, I will check and come back
GO_AHEAD = "go_ahead"    # yes / let's do it / send it / what's next
OFF_TOPIC = "off_topic"  # asks for something that is not ours to do (GST, loans, ...)
IDENTITY = "identity"    # who are you / are you a bot
QUESTION = "question"
THANKS = "thanks"
OTHER = "other"          # any other statement or answer


@dataclass(frozen=True)
class Signal:
    kind: str
    lang: str | None = None   # language of THIS message: "en", "hi-en", or None when it is too short to tell
    weak: bool = False        # a bare "ok" / "yes": a go-ahead only because of what we last asked
    topic: str = ""           # off_topic: gst | loan | legal | hiring | chitchat
    wait: int = 0             # later: how long to leave them alone, in seconds
    matched: str = ""         # the phrase that decided it, for logs and rationales


def _rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{a})" for a in alternatives), re.I)


# --------------------------------------------------------------------------- the patterns

_NEGATED_STOP = re.compile(r"\b(?:don'?t|dont|do not|never)\s+stop\b", re.I)
_NOT_NEGATING = re.compile(r"\bno\s+(?:problem|worries|issue|doubt)\b|\bnot\s+a\s+problem\b", re.I)

_STOP = _rx(
    r"^(?:please\s+|pls\s+|just\s+)?stop\b",
    r"\bstop\s+(?:it|this|that|these|those|all|now|messag\w*|msg\w*|send\w*|text\w*|call\w*|contact\w*|spam\w*"
    r"|bother\w*|disturb\w*|mail\w*|whatsapp\w*|ping\w*|writing|harass\w*)",
    r"\b(?:please|pls|just)\s+stop\b",
    r"\bunsubscribe\b", r"\bopt[\s-]?out\b", r"\bdo\s*not\s*disturb\b", r"\bdnd\b",
    r"\b(?:don'?t|dont|do\s+not|never)\s+(?:ever\s+|again\s+|more\s+)?"
    r"(?:message|msg|text|call|contact|send|whatsapp|ping|disturb|bother|write|mail)\b",
    r"\b(?:leave|let)\s+me\s+(?:alone|be)\b", r"\bget\s+lost\b", r"\bshut\s*up\b", r"\bback\s*off\b",
    r"\bremove\s+(?:me|my\s+(?:number|name|contact))\b", r"\btake\s+me\s+off\b", r"\bdelete\s+my\s+(?:number|data|contact)\b",
    r"\b(?:this|it|that)\s+(?:is|looks|seems|sounds)\s+(?:like\s+)?(?:\w+\s+){0,2}(?:spam|scam|fraud)\b",
    r"\bspam(?:ming)?\s+(?:me|messages?)\b", r"\bharass\w*",
    r"\bblock(?:ed|ing)?\s+(?:you|this|ur|your|the)\b", r"\breport(?:ing)?\s+(?:you|this|ur|your)\b", r"\bconsumer\s+court\b",
    r"\b(?:useless|worthless|nonsense|rubbish|stupid|idiot\w*|pathetic|disgusting|annoying|irritating"
    r"|bullshit|crap|shit|wtf|f+u+c+k\w*|bloody|damn)\b",
    r"\bwaste\s+of\s+(?:my\s+)?(?:time|money)\b", r"\bfed\s+up\b", r"\bsick\s+of\b",
    r"\b(?:bothering|disturbing|troubling|pestering|harassing|irritating)\s+me\b",
    r"\bwhy\s+(?:are|r)\s+(?:you|u)\s+(?:messaging|texting|bothering|disturbing|contacting|sending)\b",
    # Hinglish
    r"\b(?:message|msg|sms|call|whatsapp|text|mail|ping)\s+(?:mat|na)\s+(?:karo|kijiye|kariye|bhejo|bhejiye|karna|bhejna)\b",
    r"\bmat\s+(?:bhej\w*|karo|kijiye|kariye)\b", r"\bband\s+(?:karo|kar\s*do|kardo|kijiye|kariye)\b",
    r"\bpareshan\b", r"\btang\s+(?:mat|na)\b", r"\bdisturb\s+(?:mat|na)\b", r"\bbakwaas\b", r"\bfaltu\b",
    r"\bbekar\b", r"\bkachra\b", r"\bnaraz\b",
    # Devanagari
    r"बंद\s*कर", r"मत\s*भेज", r"परेशान", r"स्पैम", r"रुक\s*जाओ|रुको", r"तंग",
)

_AUTO = _rx(
    r"thank(?:s| you)\s+for\s+(?:contacting|reaching\s+out|messaging|writing|your\s+(?:message|enquiry|inquiry|interest|query))",
    r"\b(?:our|the)\s+team\s+(?:will|shall|would)\s+(?:get\s+back|respond|reply|contact|revert|be\s+in\s+touch|connect|assist)",
    r"\bwe\s*(?:'ll|will|shall)\s+(?:get\s+back|respond|reply|contact\s+you|revert|be\s+in\s+touch)",
    r"\b(?:this\s+is|i\s+am|i'?m)\s+an?\s+(?:automated|auto|virtual)\b", r"\bauto[\s-]?repl(?:y|ies)\b",
    r"\bautomatic\s+(?:reply|response)\b", r"\bautomated\s+(?:reply|response|message|assistant)\b",
    r"\bwe\s*(?:'re|are)\s+(?:currently|presently|now)\s+(?:away|unavailable|closed|not\s+available|out\s+of\s+office)\b",
    r"\bwe\s*(?:'re|are)\s+(?:away|unavailable|not\s+available)\b",
    r"\bout\s+of\s+office\b", r"\boutside\s+(?:of\s+)?(?:our\s+)?(?:business|working|office)\s+hours\b",
    r"\byour\s+message\s+(?:has\s+been|is)\s+(?:received|noted|important)\b",
    r"\bwe\s+have\s+received\s+your\s+(?:message|query|enquiry|inquiry)\b",
    r"\bplease\s+(?:leave|drop)\s+(?:a|your)\s+message\b",
    r"\bfor\s+(?:urgent|immediate)\s+(?:assistance|queries|help|matters),?\s+(?:please\s+)?(?:call|contact)\b",
    # Hinglish
    r"\bteam\s+tak\s+pahunch\w*", r"\bpahuncha\s+(?:deti|dungi|denge)\b", r"\bhamari\s+team\s+(?:aapse|jald|sampark)\b",
    r"\bjald\s+hi\s+(?:aapse\s+)?(?:sampark|contact|jawab|reply)\b",
    r"\bsampark\s+karne\s+ke\s+liye\s+(?:dhanyavad|shukriya)\b", r"\bkripya\s+(?:prateeksha|intezaar|wait)\b",
    r"\baapka\s+(?:sandesh|message)\s+(?:prapt|mil)\b", r"\babhi\s+(?:available|uplabdh)\s+nahi\b",
)

_DECLINE = _rx(
    r"\bnot\s+interested\b", r"\bno\s+(?:thanks|thank\s*you|thx)\b", r"\bnot\s+(?:for\s+me|required|needed|looking)\b",
    r"\bno\s+need\b", r"\bno\s+interest\b", r"\b(?:i|we)\s+(?:don'?t|dont|do\s+not)\s+(?:need|want|require)\b",
    r"\bnahi?n?\s+chahiye\b", r"\bzaroorat\s+nahi\b", r"\binterested\s+nahi\b", r"\bnahi\s+(?:karna|karni|lena|leni)\b",
    r"नहीं\s*चाहिए",
)
_BARE_NO = re.compile(r"^(?:no|nope|nah|nahi|nahin|naa|na|नहीं)[\s.!,]*$", re.I)

_LATER = _rx(
    r"\b(?:busy|in\s+a\s+meeting|driving|with\s+a\s+(?:customer|patient|client))\b",
    r"\b(?:call|ping|text|message|msg|contact|remind|revert|reach)\s+(?:me\s+)?(?:later|tomorrow|tmrw|tonight|after|next|in\s+an?\s+hour)\b",
    r"\bnot\s+(?:right\s+)?now\b", r"\bsome\s*other\s+time\b", r"\banother\s+time\b", r"\bmaybe\s+later\b", r"\blater\b",
    r"\bnext\s+(?:week|month)\b",
    r"\b(?:will|shall)\s+(?:check|see|look|think|let\s+you\s+know|get\s+back|revert|talk|discuss|decide|reply|respond)\b",
    r"\bi(?:'ll|\s+will)\s+(?:check|see|look|think|let\s+you\s+know|get\s+back|revert|talk|discuss|decide|reply|respond)\b",
    r"\blet\s+me\s+(?:check|see|think|discuss|ask|talk|look)\b",
    r"\bbaad\s+(?:mein|me|main)\b", r"\bbaadme\b", r"\bthodi\s+der\b", r"\babhi\s+nahi\b", r"\babhi\s+busy\b",
    r"\btime\s+nahi\b", r"\bfursat\b",
    r"\b(?:dekh|soch|bata|puch|pooch)(?:ta|ti|unga|ungi|enge)\b",
    r"\bkal\s+(?:baat|call|message|msg|bhej|contact|batata|batati|dekh|karte|karenge)\b",
    r"\b(?:shaam|evening|raat|night)\s+(?:ko|me|mein)\b",
)

_GO_STRONG = _rx(
    r"\blet'?s\s+(?:do|go|start|begin|proceed|try|get|move)\b", r"\bgo\s+ahead\b", r"\bgo\s+for\s+it\b", r"\bdo\s+it\b",
    r"\bproceed\b", r"\bstart\s+(?:now|it|the|with)\b", r"\bplease\s+start\b",
    r"\bsend\s+(?:it|me|that|the|this|over|them|those)\b", r"\bshare\s+(?:it|that|the|this|with\s+me)\b",
    r"\bplease\s+(?:send|share|draft|prepare|proceed|do|go|make|create|set|book|schedule)\b",
    r"\b(?:draft|prepare|create|schedule)\s+(?:it|that|the|this|me|a|my)\b",
    r"\bsounds?\s+(?:good|great|fine|perfect|interesting)\b", r"\bperfect\b", r"\bwith\s+pleasure\b",
    r"\b(?:i'?m|i\s+am|am)\s+(?:in|ready|game|keen|interested)\b", r"\b(?:count|include)\s+me\s+in\b", r"\bsign\s+me\s+up\b",
    r"\bi\s+want\s+(?:it|this|that|to\s+(?:join|start|try|proceed|do|know\s+more|sign))\b",
    r"\bi(?:'d|\s+would)\s+like\s+to\s+(?:join|start|try|proceed|know\s+more|sign|go)\b",
    r"\bwhat'?s\s+next\b", r"\bwhats\s+next\b", r"\bnext\s+steps?\b", r"\bwhat\s+now\b",
    r"\bhow\s+(?:do|can|should)\s+(?:i|we)\s+(?:start|join|begin|proceed|do\s+this|get\s+started|sign)\b",
    r"\bhow\s+to\s+(?:start|join|begin|proceed|sign)\b", r"\binterested\b",
    # Hinglish
    r"\bmujhe\b.*\b(?:ju+d|jo+d|join|shuru|start)\w*",
    r"\b(?:kar\s*do|kardo|kar\s*dijiye|kijiye|kariye|karo|chalu\s*kar\w*|shuru\s*kar\w*|bhej\s*do|bhejo|bhej\s*dijiye"
    r"|bana\s*do|banao|bana\s*dijiye|de\s*do)\b",
    r"\bchalo\b", r"\bchalega\b", r"\btheek\s+hai\b", r"\bthik\s+hai\b", r"\bsahi\s+hai\b", r"\bbilkul\b", r"\bzaroor\b", r"\bjaroor\b",
    r"\bdone\b", r"\bconfirm(?:ed)?\b", r"\bagreed?\b", r"\bi\s+agree\b",
    # Devanagari
    r"हाँ|हां", r"ठीक\s*है", r"कर\s*दो|करो",
)
_GO_WEAK = re.compile(
    r"^(?:(?:yes|yeah|yep|yup|yea|ya|y|ok|okay|okk|okie|k|kk|sure|alright|fine|absolutely|definitely|certainly|of\s+course"
    r"|please|pls|plz|haan|han|haa|ha|hanji|haanji|ji|jee|hmm|hm|accha|acha|achha|good|great|nice|cool|thik|theek|sahi"
    r"|right|correct)[\s,.!]*)+$",
    re.I,
)

_TOPICS = (
    ("gst", r"\bgst\w*|\bitr\b|\bincome\s*tax|\btax(?:es)?\b|\btds\b|\bbookkeep\w*|\baccountant\b|\bbalance\s*sheet|\bpan\s*card"),
    ("loan", r"\bloans?\b|\bemi\b|\binsurance\b|\binvest\w*|\bmutual\s*funds?|\bcredit\s*card|\binterest\s*rate"),
    ("legal", r"\blawyer\b|\badvocate\b|\bcourt\b|\blegal\b|\bpolice\b|\bfir\b"),
    ("hiring", r"\bhir(?:e|ing)\b|\brecruit\w*|\bsalary\b|\bsalaries\b|\bpayroll\b"),
    ("chitchat", r"\bjokes?\b|\bweather\b|\bmovies?\b|\bpolitic\w*|\belection\w*|\bmodi\b|\bbitcoin\b|\bcrypto\b"
                 r"|\bstock\s*market|\bshare\s*market|\bhoroscope\b|\bpoem\b|\brecipe\b"),
)
_TOPIC_RX = tuple((topic, re.compile(pattern, re.I)) for topic, pattern in _TOPICS)

_IDENTITY = _rx(
    r"\bwho\s+(?:are|r)\s+(?:you|u)\b", r"\bwho\s+is\s+(?:this|vera)\b", r"\bwho(?:'s|s)\s+(?:this|vera)\b",
    r"\bwho\s+am\s+i\s+(?:talking|speaking|chatting)\b",
    r"\bare\s+(?:you|u)\s+(?:a\s+|an\s+)?(?:bot|robot|human|real|ai|person|machine|automated|chatbot|assistant)\b",
    r"\bis\s+this\s+(?:a\s+|an\s+)?(?:bot|robot|automated|real|human|scam|spam|fraud|genuine|legit\w*)\b",
    r"\bkaun\s+(?:ho|hai|bol|hain|hoon)\b", r"\bkahan\s+se\s+(?:bol|message|ho)\b", r"\bkis\s+(?:ki|ka)\s+taraf\s+se\b",
    r"\bwhat\s+is\s+(?:magicpin|vera)\b", r"\bmagicpin\s+kya\b", r"\bwhich\s+company\b",
    r"\bwhere\s+are\s+you\s+(?:from|calling)\b", r"\bhow\s+did\s+you\s+get\s+my\s+(?:number|details|contact)\b",
)

_QUESTION_START = re.compile(
    r"^(?:what|why|how|when|where|which|who|whom|whose|can|could|will|would|shall|should|is|are|am|do|does|did|has|have"
    r"|kya|kyun|kyu|kaise|kaisa|kaisi|kab|kahan|kaun|kaunsa|kitna|kitne|kitni|konsa|price|cost|charges?|fees?|rate"
    r"|tell\s+me|explain|batao|bataiye)\b",
    re.I,
)
_THANKS = re.compile(
    r"^(?:(?:ok(?:ay)?|great|nice|alright)[\s,.!]+)?(?:thanks?(?:\s+a\s+lot|\s+so\s+much)?|thank\s*(?:you|u)(?:\s+so\s+much|\s+very\s+much)?"
    r"|thx|ty|tysm|tq|many\s+thanks|shukriya|dhanyavad|dhanyawad|bahut\s+(?:bahut\s+)?(?:shukriya|dhanyavad))"
    r"(?:\s+(?:vera|ji|sir|madam|mam|ma'am|team))?[\s,.!]*$",
    re.I,
)

_POSITIVE_EMOJI = "👍👌✅💯🆗👏🙌"
_GRATEFUL_EMOJI = "🙏😊☺🤝❤🙂😀"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ALNUM = re.compile(r"[A-Za-z0-9ऀ-ॿ]")

# Hindi words beyond the shared validator list, for telling a Hinglish reply from an English one.
_MORE_HINDI = frozenset("""
mujhe mera meri mere hum humein humara tum tumhara yaar bhai accha acha achha theek thik sahi chalo karke karni lena dena
paise dukaan dukan pehle phir fir bas wahi yahi isme usme kitna kitne kitni kyon kyu kuchh zyada jyada kam bohot thoda
thodi bhejo dijiye lijiye batana baat baatein karte karti karta hota hoti hote hua hui hue gaya gayi gaye wahan yahan
kyunki isliye
""".split())
_HINDI_MARKERS = HINDI_WORDS | _MORE_HINDI


# --------------------------------------------------------------------------- helpers

def _squash(text: str) -> str:
    """Lower-case, with emoji and symbols removed, so patterns see plain words."""
    text = unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("‘", "'").lower()
    text = re.sub(r"[^\w\s'?!.,ऀ-ॿ]", " ", text)
    return " ".join(text.split())


def fingerprint(message: str) -> str:
    """The message reduced to its words, for spotting the same canned text arriving again."""
    return " ".join(re.findall(r"[a-z0-9ऀ-ॿ]+", (message or "").lower()))


def detect_language(message: str) -> str | None:
    """"hi-en" when the message mixes in Hindi, "en" when it is clearly English, None when it is too short to tell."""
    if _DEVANAGARI.search(message or ""):
        return "hi-en"
    tokens = re.findall(r"[a-z']+", (message or "").lower())
    hits = sum(1 for t in tokens if t in _HINDI_MARKERS)
    if hits >= 2 or (hits == 1 and len(tokens) <= 3):
        return "hi-en"
    if hits == 0 and len(tokens) >= 3:
        return "en"
    return None


def _wait_from(text: str) -> int:
    """How long someone who says "later" probably means, in seconds (capped at three days)."""
    if re.search(r"\b(?:next\s+(?:week|month)|agle\s+hafte|hafte)\b", text):
        return 3 * 86400
    if re.search(r"\b(?:tomorrow|tmrw|kal|next\s+day)\b", text):
        return 86400
    if re.search(r"\b(?:evening|shaam|tonight|raat|night)\b", text):
        return 4 * 3600
    if re.search(r"\b(?:hour|hr|ghanta|ghante)\b", text):
        return 3600
    if re.search(r"\b(?:minutes?|mins?)\b", text):
        return 1800
    return 2 * 3600


def _found(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group().strip() if match else ""


# --------------------------------------------------------------------------- the classifier

def classify(message: object) -> Signal:
    """What kind of message is this? Never raises: anything odd is `blank` or `other`."""
    raw = message if isinstance(message, str) else ("" if message is None else str(message))
    raw = raw.strip()
    lang = detect_language(raw)
    text = _squash(raw)

    if not _ALNUM.search(text):  # no letters or digits: an emoji or punctuation only
        if any(e in raw for e in _POSITIVE_EMOJI):
            return Signal(GO_AHEAD, lang, weak=True, matched="positive emoji")
        if any(e in raw for e in _GRATEFUL_EMOJI):
            return Signal(THANKS, lang, matched="grateful emoji")
        return Signal(BLANK, lang)

    plain = _NOT_NEGATING.sub(" ", _NEGATED_STOP.sub(" ", text))

    if hit := _found(_STOP, plain):
        return Signal(STOP, lang, matched=hit)
    if hit := _found(_AUTO, plain):
        return Signal(AUTO, lang, matched=hit)
    if hit := _found(_DECLINE, plain) or ("no" if _BARE_NO.match(plain.strip()) else ""):
        return Signal(DECLINE, lang, matched=hit)
    if hit := _found(_LATER, plain):
        return Signal(LATER, lang, wait=_wait_from(plain), matched=hit)
    if hit := _found(_GO_STRONG, plain):
        return Signal(GO_AHEAD, lang, matched=hit)
    for topic, pattern in _TOPIC_RX:
        if hit := _found(pattern, plain):
            return Signal(OFF_TOPIC, lang, topic=topic, matched=hit)
    if hit := _found(_IDENTITY, plain):
        return Signal(IDENTITY, lang, matched=hit)
    if "?" in plain or _QUESTION_START.match(plain):
        return Signal(QUESTION, lang, matched="question")
    if _GO_WEAK.match(plain):
        return Signal(GO_AHEAD, lang, weak=True, matched=plain.strip()[:20])
    if _THANKS.match(plain):
        return Signal(THANKS, lang, matched="thanks")
    return Signal(OTHER, lang)
