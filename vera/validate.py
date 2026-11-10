"""Checks every message must pass before it is sent, whoever wrote it (the model or a template).

The judge penalises a link (-3), a repeated body (-2), invented data (-2) and internal jargon (-1),
and scores down anything that ignores the language preference, buries the ask or asks twice.
Each rule here maps to one of those, so a message that passes is safe to send. A failed check never
raises: it returns Problems, which the composer turns into a retry or a template.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .prompt import business_name, grounding_lines, max_chars, slot_labels, taboo_words

if TYPE_CHECKING:
    from .signals import Brief


@dataclass(frozen=True)
class Problem:
    code: str
    detail: str   # written as an instruction, so it can be fed straight back to the model

    def __str__(self) -> str:
        return self.detail


# --------------------------------------------------------------------------- numbers

# Digits glued to a letter on the left ("Studio11", "AT2024", "B12") belong to a name, not a quantity.
_NUMBER = re.compile(r"(?<![A-Za-z\d])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?")
_DURATION = re.compile(r"(?<![A-Za-z\d])(\d+)\s*-?\s*(min|mins|minute|minutes|sec|secs|second|seconds)\b", re.I)
_DAYS = re.compile(r"(\d+)\s*-?\s*days?\b", re.I)
TRIVIAL = frozenset({"1", "2"})  # "Reply 1 or 2", "2 slots"


def _canon(token: str) -> str:
    token = token.replace(",", "")
    try:
        value = float(token)
    except ValueError:
        return token
    return str(int(value)) if value.is_integer() else repr(value)


def numbers_of(text: str) -> list[str]:
    """Numbers written in `text`, in order of appearance, as plain canonical strings."""
    return [_canon(m.group()) for m in _NUMBER.finditer(text)]


def _licensed(lines: Iterable[str]) -> set[str]:
    """Numbers the model was shown, plus unit conversions of them ('57 days' also licenses 8 weeks)."""
    out: set[str] = set()
    for line in lines:
        out.update(numbers_of(line))
        for m in _DAYS.finditer(line):
            days = int(m.group(1))
            out.update({str(round(days / 7)), str(round(days / 30))})
        if line.startswith("slots available:"):  # the menu numbers of "Reply 1 for ...; 2 for ...; 3 for ..."
            out.update(str(n) for n in range(1, line.count("|") + 2))
    return out


def unlicensed_numbers(body: str, allowed: set[str]) -> list[str]:
    """Numbers in `body` that are not in `allowed` (and are not trivial or a small effort estimate)."""
    effort = set()  # "a 5-min setup" is an effort estimate, not a claim about the business
    for m in _DURATION.finditer(body):
        n, unit = int(m.group(1)), m.group(2).lower()
        if (unit.startswith("min") and n <= 15) or (unit.startswith("sec") and n <= 60):
            effort.add(str(n))
    unknown = [n for n in numbers_of(body) if n not in allowed and n not in effort and n not in TRIVIAL]
    return list(dict.fromkeys(unknown))


def invented_numbers(body: str, brief: "Brief") -> list[str]:
    return unlicensed_numbers(body, _licensed(grounding_lines(brief)))


# --------------------------------------------------------------------------- patterns

_URL = re.compile(
    r"(?i)\bhttps?://|\bwww\.|\b[a-z0-9-]{2,}\.(?:com|in|co|org|net|io|app|me|ly|info|biz|store|shop)\b"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<![\d.,])(?:\+?91[\s-]?)?\d{10}(?!\d)|(?<!\d)\d{5}[\s-]\d{5}(?!\d)")
_JARGON = re.compile(
    r"(?i)\b(?:triggers?|payload|suppression|send_as|binary_yes_no|open_ended|multi_choice\w*|confirm_cancel"
    r"|json|llm|placeholder|degraded|template_params|cta)\b"
    r"|\b(?:m|c|trg|d|o)_[a-z0-9_]*\d[a-z0-9_]*\b|\bnull\b|\bundefined\b|\bnan\b"
    r"|\{[^}]*\}|\[[A-Za-z_ ]{1,25}\]|<[A-Za-z_ ]{1,20}>"
)
_INTRO = re.compile(
    r"(?i)\b(?:this is|i am|i'm|its|it's|main|mai)\s+vera\b|\bvera\s+(?:here|from magicpin|se bol|bol rahi)\b"
)
# Any way of asking for a reply, in English or Hinglish ("YES bhej dijiye", "time bata dijiye").
_ASK = re.compile(
    r"(?i)\?|\breply\b|\bsend\b|\btell (?:me|us)\b|\blet (?:me|us) know\b|\bsay yes\b"
    r"|\bbhej\w*|\bbata\w*|\bbol\w*|\blikh\w*|\bkijiye\b|\bkariye\b"
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Hindi function words that are not English words, for telling Hinglish from plain English.
HINDI_WORDS = frozenset("""
aap aapka aapke aapki aapko hai hain hoon hun hoga hogi honge tha thi ka ke ki ko se mein pe kya kyun kaise
kab kahan kaun kuch koi sab bahut bhi toh nahi nahin haan ji ek yeh woh abhi aaj kal parso hafte hafta mahine
mahina saal raat subah shaam liye taaki lekin magar aur ya agar jab jaise zaroor bilkul sirf sakte sakta sakti
chahiye chahte chahta chahenge chahein karein karna kar karo kijiye karenge dein diya diye bhej bhejein bhejiye
bana banao banayein banaiye batao bataiye batayein bata dekh dekhein dekhiye dekho milega milegi rahe raha rahi
rahega namaste dhanyavad shukriya naya nayi naye wala wali wale doon dun denge karunga karungi
dungi karungi rakhungi bhejungi banungi lagta lag hamare hamari hamara aapne mujhe isliye jaldi sandesh
""".split())

_GENERIC_NAME_WORDS = frozenset({
    "clinic", "dental", "salon", "restaurant", "pharmacy", "gym", "fitness", "family", "studio", "care",
    "centre", "center", "hair", "beauty", "health", "the", "and", "cafe", "dr",
})


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _first_name(salutation: str) -> str:
    stripped = re.sub(r"(?i)^(?:dr|mr|mrs|ms|shri|smt)\.?\s+", "", salutation.strip())
    parts = stripped.split()
    return parts[0].strip(".,'") if parts else ""


def _business_tokens(name: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", name))
    return [t for t in tokens if len(t) >= 3 and t.lower() not in _GENERIC_NAME_WORDS]


def _repeats(body: str, previous: Iterable[str]) -> bool:
    mine, mine_words = _norm(body), _words(body)
    for old in previous:
        if not old:
            continue
        if _norm(old) == mine:
            return True
        theirs = _words(old)
        if len(mine_words) >= 6 and len(theirs) >= 6 and len(mine_words & theirs) / len(mine_words | theirs) >= 0.85:
            return True
    return False


# --------------------------------------------------------------------------- the check

def check(body: str, brief: "Brief", seen: Iterable[str] = ()) -> list[Problem]:
    """Every problem with `body` as the message for `brief` (empty list = safe to send)."""
    body = body or ""
    if len(body.strip()) < 20:
        return [Problem("too_short", "The message is empty or far too short.")]
    problems: list[Problem] = []

    def add(code: str, detail: str) -> None:
        problems.append(Problem(code, detail))

    limit = max_chars(brief)
    if len(body) > limit:
        add("too_long", f"The message is {len(body)} characters; keep it under {limit}.")

    for pattern, what in ((_URL, "a link or website name"), (_EMAIL, "an e-mail address"), (_PHONE, "a phone number")):
        found = pattern.search(body)
        if found:
            add("url", f"The message contains {what} ('{found.group().strip()}'). Remove it; no links, sites, e-mails or phone numbers.")

    unknown = invented_numbers(body, brief)
    if unknown:
        add("number", f"These numbers are not in the facts: {', '.join(unknown)}. Use only numbers given under FACTS and OFFERS, and calculate nothing.")

    banned = [w for w in taboo_words(brief.category) if re.search(r"(?i)(?<!\w)" + re.escape(w) + r"(?!\w)", body)]
    if banned:
        add("taboo", f"Uses forbidden wording: {', '.join(banned)}.")

    jargon = _JARGON.search(body)
    if jargon:
        add("jargon", f"Uses internal wording or an unfilled placeholder ('{jargon.group()}'). Write plain words a shop owner would use.")

    if _repeats(body, [*brief.prior_bodies, *seen]):
        add("repeat", "It repeats or closely paraphrases a message that was already sent. Say something new.")

    problems += _address(body, brief)
    problems += _ask(body, brief)
    problems += _language(body, brief)
    return problems


def _address(body: str, brief: "Brief") -> list[Problem]:
    out = []
    lowered = body.lower()
    name = _first_name(brief.salutation)
    if len(name) >= 2 and name.lower() not in lowered:
        out.append(Problem("no_name", f"Address the recipient by name ('{brief.salutation}')."))
    if brief.about and brief.about.lower() not in lowered:
        out.append(Problem("no_name", f"Say who the message is about ('{brief.about}')."))
    if brief.send_as == "merchant_on_behalf":
        if re.search(r"(?i)\bvera\b", body):
            out.append(Problem("vera", "This message is from the business to its customer: never mention Vera."))
        tokens = _business_tokens(business_name(brief))
        if tokens and not any(t.lower() in lowered for t in tokens):
            out.append(Problem("business", f"Say which business it is from ('{business_name(brief)}')."))
    elif any("re-introduce" in n for n in brief.notes) and _INTRO.search(body):
        out.append(Problem("reintro", "There has been an earlier conversation: do not introduce yourself again."))
    return out


def _ask(body: str, brief: "Brief") -> list[Problem]:
    cta = brief.cta
    questions = body.count("?")
    if cta == "none":
        return [Problem("ask", "This message needs no reply: remove the question.")] if questions else []
    out = []
    if questions > (2 if cta == "multi_choice_slot" else 1):
        out.append(Problem("ask", "Ask only ONE thing: keep a single question or instruction to reply."))
    if not _ASK.search(body[-170:]):
        out.append(Problem("ask", "End with the call to action as the last sentence."))
    if cta == "multi_choice_slot":
        labels = slot_labels(brief)
        if labels and not any(_norm(label) in _norm(body) for label in labels):
            out.append(Problem("slots", f"Offer the given slots with their wording unchanged: {' | '.join(labels)}."))
    if cta == "binary_confirm_cancel" and not re.search(r"(?i)confirm|reschedule|cancel|another time|different time|better time", body):
        out.append(Problem("ask", "Ask them to reply CONFIRM, or to name a better time."))
    return out


def _language(body: str, brief: "Brief") -> list[Problem]:
    return language_problems(body, brief.language)


def language_problems(body: str, code: str) -> list[Problem]:
    """Is `body` written in the language `code` (en | hi-en | hi)? Other codes are not checked."""
    hindi_words = len(_words(body) & HINDI_WORDS)
    devanagari = bool(_DEVANAGARI.search(body))
    if code == "en" and (devanagari or hindi_words >= 3):
        return [Problem("language", "Write in English only; this recipient prefers English.")]
    if code == "hi-en" and not devanagari and hindi_words < 2:
        return [Problem("language", "Write in natural Hinglish (Hindi-English mix in Roman script); this recipient prefers it.")]
    if code == "hi" and not devanagari and hindi_words < 3:
        return [Problem("language", "Write in Hindi (Roman script is fine); this recipient prefers Hindi.")]
    return []


# --------------------------------------------------------------------------- replies

# From the judge's own scenario check: after the merchant says "let's do it", a reply that keeps
# qualifying loses points, and one that does the work uses words like these.
QUALIFYING = ("would you", "do you", "can you tell", "what if", "how about")
ACTION_WORDS = ("done", "sending", "draft", "here", "confirm", "proceed", "next")


def check_reply(
    body: str,
    *,
    lang: str,
    grounding: Iterable[str],
    taboo: Iterable[str] = (),
    seen: Iterable[str] = (),
    customer_facing: bool = False,
    limit: int = 900,
    max_questions: int = 1,
    action: bool = False,
) -> list[Problem]:
    """Every problem with `body` as a reply in a live conversation (empty list = safe to send).

    A reply is judged more lightly than an opening message (no greeting or call-to-action position
    rules) but by the same hard rules: nothing invented, no links, nothing forbidden, no repeat,
    the right language. `grounding` lists every line whose numbers may be quoted; `action=True` is
    for a reply to "let's do it", which must start doing the work rather than ask more questions.
    """
    body = body or ""
    if len(body.strip()) < 12:
        return [Problem("too_short", "The reply is empty or far too short.")]
    problems: list[Problem] = []

    def add(code: str, detail: str) -> None:
        problems.append(Problem(code, detail))

    if len(body) > limit:
        add("too_long", f"The reply is {len(body)} characters; keep it under {limit}.")
    for pattern, what in ((_URL, "a link or website name"), (_EMAIL, "an e-mail address"), (_PHONE, "a phone number")):
        found = pattern.search(body)
        if found:
            add("url", f"The reply contains {what} ('{found.group().strip()}'). Remove it.")
    unknown = unlicensed_numbers(body, _licensed(grounding))
    if unknown:
        add("number", f"These numbers are not in the facts or the conversation: {', '.join(unknown)}. Do not invent or calculate numbers.")
    banned = [w for w in taboo if re.search(r"(?i)(?<!\w)" + re.escape(w) + r"(?!\w)", body)]
    if banned:
        add("taboo", f"Uses forbidden wording: {', '.join(banned)}.")
    jargon = _JARGON.search(body)
    if jargon:
        add("jargon", f"Uses internal wording or an unfilled placeholder ('{jargon.group()}'). Write plain words.")
    if _repeats(body, seen):
        add("repeat", "It repeats or closely paraphrases something already sent in this conversation. Say something new.")
    if body.count("?") > max_questions:
        add("ask", "Ask at most one question.")
    if customer_facing and re.search(r"(?i)\bvera\b|\bmagicpin\b", body):
        add("vera", "This reply is from the business to its customer: never mention Vera or magicpin.")
    problems += language_problems(body, "hi-en" if lang == "hi" else lang)
    if action:
        lowered = body.lower()
        asked = [phrase for phrase in QUALIFYING if phrase in lowered]
        if asked:
            add("qualifying", f"They already said yes: do not ask qualifying questions ('{asked[0]}'). Do the work.")
        if not any(word in lowered for word in ACTION_WORDS):
            add("no_action", "Start doing the work: say what is done, being drafted or sent, and the one next step.")
    return problems
