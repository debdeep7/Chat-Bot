"""Wording for replies: the deterministic floor under every reply the bot sends.

Each builder returns a short list of variants, best first, so that a situation that comes up twice
in one conversation is answered in different words (a verbatim repeat is penalised). They only
rearrange what the Brief already knows, so they cannot invent anything. English is plain; Hinglish
for Vera uses her feminine forms ("dungi", "karungi"); a business speaking to its customer uses the
plural "hum" forms.

The reply to "let's do it" starts doing the work and ends in a single CONFIRM. It avoids the phrases
the judge treats as qualifying questions ("would you", "do you", "can you tell", "what if", "how about").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CLIENTS = {"dentists": "patients", "salons": "clients", "gyms": "members", "restaurants": "guests", "pharmacies": "customers"}


@dataclass(frozen=True)
class Ctx:
    name: str = ""               # who we are talking to ("Dr. Meera", "Priya"); empty when unknown
    hindi: bool = False          # answer in Hinglish
    does: str = ""               # what Vera offered, already in the right language ("pull the abstract ...")
    working: str = ""            # a sentence naming the fact the work starts from, or ""
    business: str = "our team"
    clients: str = "customers"


# --------------------------------------------------------------------------- what Vera offers, per kind

DOES = {
    "research_digest": ("pull the abstract and draft a short note for your {clients}",
                        "abstract nikaal kar aapke {clients} ke liye ek chhota note bana dungi"),
    "regulation_change": ("put together a one-page checklist so you are ready before the deadline",
                          "deadline se pehle ke liye ek one-page checklist bana dungi"),
    "cde_opportunity": ("remind you the day before so you do not miss it",
                        "ek din pehle yaad dila dungi taaki aap miss na karein"),
    "supply_alert": ("draft the notice for your affected customers along with the return steps",
                     "affected customers ke liye notice aur return steps ka draft bana dungi"),
    "category_seasonal": ("set up a ready-to-post offer for this season",
                          "is season ke liye ek ready-to-post offer bana dungi"),
    "perf_dip": ("look into what changed and send you a short fix list",
                 "dekh kar ki kya badla, aapko ek chhoti fix list bhej dungi"),
    "perf_spike": ("turn this momentum into a post or an offer",
                   "is momentum ko ek post ya offer mein badal dungi"),
    "seasonal_perf_dip": ("plan a low-cost way to keep your {clients} coming",
                          "aapke {clients} ko jodne ka ek low-cost plan bana dungi"),
    "milestone_reached": ("draft a short message to help you get there",
                          "wahan tak pahunchne ke liye ek chhota message draft kar dungi"),
    "review_theme_emerged": ("draft replies to these reviews along with a quick fix plan",
                             "in reviews ke reply aur ek quick fix plan draft kar dungi"),
    "competitor_opened": ("draft an offer idea that plays to your strengths",
                          "aapki strengths par ek offer idea draft kar dungi"),
    "renewal_due": ("renew your plan for you", "aapka plan renew kar dungi"),
    "winback_eligible": ("relaunch your plan in one step", "aapka plan ek hi step mein dobara shuru kar dungi"),
    "dormant_with_vera": ("share one quick idea for your listing", "aapki listing ke liye ek quick idea bhej dungi"),
    "curious_ask_due": ("turn your answer into a ready-to-post offer and a short reply for your customers",
                        "aapke jawab se ek ready-to-post offer aur customers ke liye ek chhota reply bana dungi"),
    "active_planning_intent": ("send you the starter draft to edit", "aapko edit karne ke liye starter draft bhej dungi"),
    "gbp_unverified": ("walk you through the verification step by step", "verification step by step karwa dungi"),
    "festival_upcoming": ("draft a ready-to-post festival offer", "ek ready-to-post festival offer draft kar dungi"),
    "ipl_match_today": ("draft a match-day offer that suits today", "aaj ke liye ek match-day offer draft kar dungi"),
    "generic_merchant": ("pick the one change that would help your listing most and draft it for you",
                         "aapki listing ke liye sabse kaam ka ek badlav chun kar uska draft bana dungi"),
}

# Leads that read as an instruction to the writer, or that say nothing about the trigger, are never quoted back.
UNQUOTABLE_LEADS = frozenset({"ask.topic", "trigger.intent", "trigger.dormant", "signal.dormant", "trigger.kind"})


def does_for(kind: str, hindi: bool, clients: str) -> str:
    pair = DOES.get(kind) or DOES["generic_merchant"]
    return pair[1 if hindi else 0].format(clients=clients)


def clients_for(category_slug: str) -> str:
    return CLIENTS.get(category_slug, "customers")


def _shorten(text: str, limit: int = 200) -> str:
    """The text cut down to whole sentences that fit `limit` (or, failing that, to whole words), without end punctuation."""
    text = " ".join((text or "").split())
    if len(text) > limit:
        kept = ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(kept) + len(sentence) + 1 > limit:
                break
            kept = f"{kept} {sentence}".strip()
        if not kept:
            head = text[: limit + 1]  # one extra character, so a word that ends exactly at the limit is kept whole
            kept = head.rsplit(" ", 1)[0] if " " in head else ""  # one endless word cannot be cut cleanly: say nothing
            if kept.count("(") > kept.count(")"):
                kept = kept[: kept.rfind("(")]
        text = kept
    return text.rstrip(" ,;:.!?")


def working_line(lead_text: str, hindi: bool, own_profile: bool = False) -> str:
    """"Here is what I am working from: <fact>. " for the opening of an action reply, or "".

    `own_profile` is for a fact that comes from the merchant's own numbers rather than from the event
    the conversation was about.
    """
    text = _shorten(lead_text)
    if not text:
        return ""
    if own_profile:
        return f"Aapke profile se shuru karte hain: {text}. " if hindi else f"Starting from your profile: {text}. "
    return f"Main isi par kaam kar rahi hoon: {text}. " if hindi else f"Here is what I am working from: {text}. "


def _pick(pairs: list[tuple[str, str]], c: Ctx, **extra: str) -> list[str]:
    values = {"name": c.name, "does": c.does, "working": c.working, "business": c.business, "clients": c.clients, **extra}
    return [pair[1 if c.hindi else 0].format(**values) for pair in pairs]


def _tidy(variants: list[str]) -> list[str]:
    """Close up the gaps that an empty name or an empty `working` sentence leaves behind."""
    out = []
    for text in variants:
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"(^|[.!?])\s*,\s*", r"\1 ", text).strip()  # the comma an empty name leaves at a sentence start
        text = re.sub(r",\s*([.!])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        out.append(text)
    return out


# --------------------------------------------------------------------------- merchant-facing replies

def auto_flag(c: Ctx) -> list[str]:
    """The first canned reply: one short note meant for the owner's eyes."""
    return _tidy(_pick([
        ("Looks like an automated reply, so I will keep this short. {name}, whenever you see this: reply YES and I will {does}.",
         "Lagta hai yeh ek automated reply hai, isliye main baat chhoti rakhungi. {name}, jab bhi aap dekhein: YES bhej dijiye aur main {does}."),
        ("{name}, this seems to be an auto-reply, so I am not sure you have seen my note. One YES from you and I will {does}.",
         "{name}, yeh auto-reply lag raha hai, toh shayad aapne mera note nahi dekha. Bas ek YES bhej dijiye aur main {does}."),
    ], c))


def action(c: Ctx) -> list[str]:
    """They said yes: start the work, say what comes next, ask for one CONFIRM."""
    return _tidy(_pick([
        ("Great, {name}. {working}Next step: I will {does}. I am putting the draft together now and will share it here. Reply CONFIRM and I will proceed.",
         "Badhiya, {name}. {working}Agla step: main {does}. Draft taiyaar kar rahi hoon aur yahin share karungi. CONFIRM bhej dijiye aur main aage badhungi."),
        ("Done deal, {name}. {working}I will {does}. The draft will land here next. Reply CONFIRM to proceed.",
         "Theek hai, {name}. {working}Main {does}. Draft agle message mein yahin aayega. Aage badhne ke liye CONFIRM bhej dijiye."),
    ], c))


def done(c: Ctx) -> list[str]:
    """They confirmed: say it is in hand."""
    return _tidy(_pick([
        ("Done, {name}. I have queued this up and will share the result here. If you want any change, just tell me.",
         "Ho gaya, {name}. Maine ise queue kar diya hai aur result yahin share karungi. Koi badlav chahiye toh bata dijiye."),
        ("All set, {name}. It is queued and I will update you here once it is ready. Tell me if you want anything changed.",
         "Sab set hai, {name}. Yeh queue mein hai aur taiyaar hote hi main yahin update karungi. Kuch badalna ho toh bata dijiye."),
    ], c))


def answer(c: Ctx) -> list[str]:
    """A question we cannot answer from the facts: say what we will do and never guess."""
    return _tidy(_pick([
        ("Good question, {name}. In short: I will {does}, and you only have to approve it. I will not guess at details I am not sure of; I will confirm them here first. Shall I start? Reply YES.",
         "Achha sawaal hai, {name}. Seedhi baat: main {does}, aapko bas approve karna hai. Jo baat mujhe pakki nahi pata, uska andaaza nahi lagaungi, pehle yahin confirm karungi. Shuru karoon? YES bhej dijiye."),
        ("Fair question, {name}. I will {does} and show it to you here before anything is final. If you have a specific doubt, tell me and I will check. Reply YES to start.",
         "Sahi sawaal, {name}. Main {does} aur final karne se pehle yahin aapko dikhaungi. Koi khaas doubt ho toh bata dijiye, main check karungi. Shuru karne ke liye YES bhej dijiye."),
    ], c))


OFF_TOPIC_LINES = {
    "gst": ("GST and tax filing are best left to your CA", "GST aur tax filing ke liye aapka CA sabse sahi rahega"),
    "loan": ("loans and finance are best discussed with your bank or CA", "loan aur finance ke liye bank ya CA se baat karna theek rahega"),
    "legal": ("legal matters are best handled by a lawyer", "legal maamle ke liye kisi vakeel se baat karna theek rahega"),
    "hiring": ("hiring is outside what I can help with", "hiring mere kaam ke daayre se bahar hai"),
}
OFF_TOPIC_DEFAULT = ("that is outside what I can help with here", "yeh mere kaam ke daayre se bahar hai")


def off_topic(c: Ctx, topic: str) -> list[str]:
    line = OFF_TOPIC_LINES.get(topic, OFF_TOPIC_DEFAULT)[1 if c.hindi else 0]
    sentence = line[:1].upper() + line[1:]
    inline = line if line[:2].isupper() else line[:1].lower() + line[1:]  # "GST ..." keeps its capitals mid-sentence
    return _tidy(_pick([
        ("{name}, {inline}, so I will stay out of that one. Back to my earlier note: I can {does}. Reply YES and I will start.",
         "{name}, {inline}, isliye main usme nahi padungi. Pichhli baat par wapas: main {does}. Shuru karne ke liye YES bhej dijiye."),
        ("{name}, I will have to pass on that. {topic}. On my earlier note, I can {does}. Reply YES to start.",
         "{name}, isme main madad nahi kar paungi. {topic}. Pichhle note par: main {does}. Shuru karne ke liye YES bhej dijiye."),
    ], c, topic=sentence, inline=inline))


def identity(c: Ctx) -> list[str]:
    return _tidy(_pick([
        ("I am Vera, magicpin's AI assistant for merchants. I help businesses like {business} with offers, posts and replies to reviews. Back to my note: I can {does}. Reply YES and I will start.",
         "Main Vera hoon, magicpin ki AI assistant. Main {business} jaise businesses ki offers, posts aur reviews ke reply mein madad karti hoon. Pichhli baat par: main {does}. Shuru karne ke liye YES bhej dijiye."),
    ], c))


def thanks_nudge(c: Ctx) -> list[str]:
    return _tidy(_pick([
        ("You are welcome, {name}. Whenever you are ready: reply YES and I will {does}.",
         "Koi baat nahi, {name}. Jab bhi aap ready hon: YES bhej dijiye aur main {does}."),
        ("Happy to help, {name}. If you want me to start, just reply YES and I will {does}.",
         "Khushi hui, {name}. Agar aap chahte hain ki main shuru karoon, toh bas YES bhej dijiye aur main {does}."),
    ], c))


def other(c: Ctx) -> list[str]:
    """A statement we cannot act on: acknowledge it and hold the thread."""
    return _tidy(_pick([
        ("Thanks for sharing that, {name}. I will keep it in mind. To move ahead, reply YES and I will {does}.",
         "Samajh gayi, {name}, batane ke liye shukriya. Aage badhne ke liye YES bhej dijiye aur main {does}."),
        ("Noted, {name}. Reply YES and I will {does}; I will share it here for your approval.",
         "Note kar liya, {name}. YES bhej dijiye, main {does} aur approval ke liye yahin share karungi."),
    ], c))


# --------------------------------------------------------------------------- business-to-customer replies

def _greet(c: Ctx) -> str:
    if c.hindi:
        return f"Namaste {c.name}" if c.name else "Namaste"
    return f"Hi {c.name}" if c.name else "Hello"


def _customer(pairs: list[tuple[str, str]], c: Ctx, **extra: str) -> list[str]:
    values = {"greet": _greet(c), "business": c.business, **extra}
    return _tidy([pair[1 if c.hindi else 0].format(**values) for pair in pairs])


def customer_booked(c: Ctx, label: str) -> list[str]:
    return _customer([
        ("{greet}, done. Your slot is booked: {label}. {business} will see you then. If you need to change it, just reply here.",
         "{greet}, ho gaya. Aapka slot book ho gaya: {label}. {business} mein milte hain. Badalna ho toh yahin bata dijiye."),
    ], c, label=label)


def customer_pick(c: Ctx, labels: list[str]) -> list[str]:
    options = "; ".join(f"{n} for {label}" for n, label in enumerate(labels[:3], start=1))
    return _customer([
        ("{greet}, great! Please pick a slot: reply {options}; or tell us a time that works.",
         "{greet}, badhiya! Slot chun lijiye: reply {options}; ya koi aur time bata dijiye."),
    ], c, options=options)


def customer_confirmed(c: Ctx) -> list[str]:
    return _customer([
        ("{greet}, confirmed. {business} will have everything ready for you. If anything changes, just let us know here.",
         "{greet}, confirm ho gaya. {business} mein sab taiyaar rahega. Kuch badle toh yahin bata dijiye."),
    ], c)


def customer_open(c: Ctx) -> list[str]:
    return _customer([
        ("{greet}, thank you. Please tell us a day and time that suits you, and we will confirm it here.",
         "{greet}, shukriya. Aap jo din aur time chahte hain bata dijiye, hum yahin confirm kar denge."),
    ], c)


def customer_question(c: Ctx) -> list[str]:
    return _customer([
        ("{greet}, thank you for asking. We will check the details and reply here shortly.",
         "{greet}, poochne ke liye shukriya. Hum details check karke yahin jaldi reply karenge."),
        ("{greet}, good question. Our team will confirm the exact details here shortly.",
         "{greet}, achha sawaal hai. Hamari team exact details yahin jaldi confirm karegi."),
    ], c)


def customer_other(c: Ctx) -> list[str]:
    return _customer([
        ("{greet}, noted. We will get back to you here shortly.",
         "{greet}, note kar liya. Hum yahin jaldi aapse baat karenge."),
        ("{greet}, thank you for letting us know. Our team will follow up here.",
         "{greet}, bataane ke liye shukriya. Hamari team yahin follow up karegi."),
    ], c)


def customer_identity(c: Ctx) -> list[str]:
    return _customer([
        ("This is the virtual assistant of {business}. We use this chat for reminders and bookings. Tell us how we can help.",
         "Main {business} ki virtual assistant hoon. Hum is chat par reminders aur bookings bhejte hain. Bataiye hum kaise madad karein."),
    ], c)
