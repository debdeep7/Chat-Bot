"""Deterministic message writer: the floor the bot never falls below.

It runs when there is no API key, when the model is rate-limited or too slow, and when the model's
draft fails validation. It only rearranges facts the Brief already holds, so it cannot invent
anything, and it follows the same rules the validator enforces: name first, the reason for writing
now, one supporting line, one ask as the last sentence, in the recipient's language.

Every wording table below holds (English, Hinglish) pairs. Hinglish for Vera uses forms that sound
right in her voice ("bana doon"); for a business speaking to its customer it uses "hum" forms.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .facts import Fact
from .prompt import REASON, business_name, max_chars, slot_labels

if TYPE_CHECKING:
    from .signals import Brief

CLIENTS = {"dentists": "patients", "salons": "clients", "gyms": "members", "restaurants": "guests", "pharmacies": "customers"}

# --------------------------------------------------------------------------- merchant-facing wording

# How each kind introduces its lead; the lead itself is quoted as the Brief holds it.
LEAD_IN = {
    "research_digest": ("a finding worth your time: {lead}", "ek finding aapke kaam ki: {lead}"),
    "regulation_change": ("heads-up on a rule change: {lead}", "ek rule change ki jaankari: {lead}"),
    "cde_opportunity": ("an event worth your time: {lead}", "ek event jo aapke kaam ka hai: {lead}"),
    "supply_alert": ("urgent product alert: {lead}", "zaroori product alert: {lead}"),
    "category_seasonal": ("what is moving this season: {lead}", "is season mein kya chal raha hai: {lead}"),
    "perf_dip": ("a quick flag on your numbers: {lead}", "aapke numbers par ek quick flag: {lead}"),
    "perf_spike": ("good news on your numbers: {lead}", "aapke numbers par achhi khabar: {lead}"),
    "seasonal_perf_dip": ("on your numbers: {lead}", "aapke numbers par: {lead}"),
    "milestone_reached": ("milestone check: {lead}", "milestone update: {lead}"),
    "review_theme_emerged": ("in your recent reviews: {lead}", "aapke recent reviews mein: {lead}"),
    "competitor_opened": ("heads-up: {lead}", "dhyan dene wali baat: {lead}"),
    "renewal_due": ("your plan: {lead}", "aapka plan: {lead}"),
    "winback_eligible": ("since your plan lapsed: {lead}", "plan lapse hone ke baad: {lead}"),
    "gbp_unverified": ("{lead}", "{lead}"),
    "festival_upcoming": ("a festival is coming up: {lead}", "ek tyohaar aane wala hai: {lead}"),
    "ipl_match_today": ("match day: {lead}", "match day: {lead}"),
    "generic_merchant": ("quick note: {lead}", "ek quick note: {lead}"),
}
# Used when the lead is not about the trigger itself but the merchant's own numbers.
OWN_PROFILE_LEAD_IN = ("I looked at your profile: {lead}", "maine aapka profile dekha: {lead}")

# One ask per kind, always the last sentence, always a single instruction to reply.
ASKS = {
    "research_digest": ("Want me to pull the abstract and draft a short note for your {clients}? Reply YES.",
                        "Kya main abstract nikaal kar aapke {clients} ke liye ek chhota note bana doon? YES bhej dijiye."),
    "regulation_change": ("Want me to send a one-page checklist so you are ready before the deadline? Reply YES.",
                          "Kya main deadline se pehle ke liye ek one-page checklist bhej doon? YES bhej dijiye."),
    "cde_opportunity": ("Shall I remind you the day before so you do not miss it? Reply YES.",
                        "Kya main ek din pehle yaad dila doon taaki aap miss na karein? YES bhej dijiye."),
    "supply_alert": ("Want me to draft the notice for your affected customers and the return steps? Reply YES.",
                     "Kya main affected customers ke liye notice aur return steps ka draft bana doon? YES bhej dijiye."),
    "category_seasonal": ("Want me to set up a ready-to-post offer for this? Reply YES.",
                          "Kya main iske liye ek ready-to-post offer bana doon? YES bhej dijiye."),
    "perf_dip": ("Want me to look into what changed and send you a short fix list? Reply YES.",
                 "Kya main dekh loon ki kya badla aur aapko ek chhoti fix list bhej doon? YES bhej dijiye."),
    "perf_spike": ("Want me to turn this into a post or an offer so the momentum keeps going? Reply YES.",
                   "Kya main ise ek post ya offer mein badal doon taaki yeh momentum chalta rahe? YES bhej dijiye."),
    "seasonal_perf_dip": ("Want me to plan a low-cost way to keep your {clients} coming through this? Reply YES.",
                          "Kya main is dip mein aapke {clients} ko jodne ka ek low-cost plan bana doon? YES bhej dijiye."),
    "milestone_reached": ("Want me to draft a short message you can send to get there? Reply YES.",
                          "Kya main wahan tak pahunchne ke liye ek chhota message draft kar doon? YES bhej dijiye."),
    "review_theme_emerged": ("Want me to draft a reply to these reviews and a quick fix plan? Reply YES.",
                             "Kya main in reviews ka reply aur ek quick fix plan draft kar doon? YES bhej dijiye."),
    "competitor_opened": ("Want me to draft an offer idea that plays to your strengths? Reply YES.",
                          "Kya main aapki strengths par ek offer idea draft kar doon? YES bhej dijiye."),
    "renewal_due": ("Want me to renew it for you now? Reply YES.",
                    "Kya main ise abhi aapke liye renew kar doon? YES bhej dijiye."),
    "winback_eligible": ("Want me to relaunch your plan in one step? Reply YES.",
                         "Kya main aapka plan ek hi step mein dobara shuru kar doon? YES bhej dijiye."),
    "dormant_with_vera": ("Is there one thing I can help with this week?",
                          "Is hafte main aapki kis ek cheez mein madad kar sakti hoon?"),
    "gbp_unverified": ("Want me to walk you through the verification now? It takes a few minutes. Reply YES.",
                       "Kya main abhi verification mein aapki madad kar doon? Bas kuch minute lagenge. YES bhej dijiye."),
    "festival_upcoming": ("Want me to draft a ready-to-post festival offer? Reply YES.",
                          "Kya main ek ready-to-post festival offer draft kar doon? YES bhej dijiye."),
    "ipl_match_today": ("Want me to draft a match-day offer that suits today? Reply YES.",
                        "Kya main aaj ke liye ek match-day offer draft kar doon? YES bhej dijiye."),
    "generic_merchant": ("Want me to look into this and suggest one next step? Reply YES.",
                         "Kya main ise dekh kar ek next step suggest kar doon? YES bhej dijiye."),
}

# When the trigger carried no detail, only the kind of event is known.
PLAIN_MERCHANT = {"competitor_opened": ("I noticed that a new competitor has opened near you",
                                        "maine dekha ki aapke paas ek naya competitor khula hai")}
NO_DETAIL = ("I wanted to flag something about your business", "aapke business ke baare mein ek baat batani thi")
# The lead reads better addressed to the merchant than described from outside.
REPHRASE = {"gbp_unverified": (("the Google business profile is not verified", "your Google business profile is not verified"),)}

# --------------------------------------------------------------------------- customer-facing wording
# {your}, {ki}, {ka}: "your" / "aapki" / "aapka", or "<Name>'s" / "<Name> ki" / "<Name> ka" for a third person.
# {for_}: "" or " for <Name>" when the message is about someone other than the reader.

PLAIN_CUSTOMER = {  # the trigger carried no detail, so say only what the kind itself says
    "appointment_tomorrow": ("{your} appointment with us is tomorrow", "{ki} appointment kal hai"),
    "recall_due": ("it is time for {your} next regular visit", "{ka} agla regular visit due hai"),
    "chronic_refill_due": ("{your} regular medicines are due for a refill", "{ki} regular dawaiyon ka refill due hai"),
    "trial_followup": ("thank you for trying a session with us", "hamare saath ek session try karne ke liye shukriya"),
    "customer_lapsed_soft": ("it has been a while since {your} last visit{when}", "{ki} last visit{when} ke baad kaafi time ho gaya hai"),
    "customer_lapsed_hard": ("we have missed seeing you", "hum aapko miss kar rahe hain"),
    "wedding_package_followup": ("we are following up on {your} bridal package", "hum {ka} bridal package ke baare mein follow up kar rahe hain"),
    "generic_customer": ("we wanted to check in", "hum bas check in karna chahte the"),
}
CUSTOMER_LEAD_IN = {  # the trigger carried real detail
    "recall_due": ("Reminder{for_}: {lead}.", "Yaad dilana tha{for_}: {lead}."),
    "appointment_tomorrow": ("Reminder{for_}: {lead}.", "Yaad dilana tha{for_}: {lead}."),
    "chronic_refill_due": ("Reminder{for_}: {lead}.", "Yaad dilana tha{for_}: {lead}."),
    "trial_followup": ("Thank you for coming for your {lead}{for_}.", "Aapke {lead} ke liye shukriya{for_}."),
    "wedding_package_followup": ("Following up on the bridal package{for_}: {lead}.", "Bridal package ke baare mein{for_}: {lead}."),
    "generic_customer": ("{lead}.", "{lead}."),
}
CUSTOMER_ASKS = {
    "recall_due": ("Shall we book a slot for you? Reply YES and tell us a day that suits you.",
                   "Kya hum ek slot book kar dein? YES bhej dijiye aur apna pasandida din bata dijiye."),
    "appointment_tomorrow": ("Reply CONFIRM to keep it, or tell us a better time.",
                             "Pakka karne ke liye CONFIRM bhejiye, ya koi aur time bata dijiye."),
    "chronic_refill_due": ("Reply CONFIRM and we will keep the refill ready, or tell us if anything has changed.",
                           "Refill ready rakhne ke liye CONFIRM bhejiye, ya kuch badla ho toh bata dijiye."),
    "trial_followup": ("Would you like to book the next session? Reply YES.",
                       "Kya aap agla session book karna chahenge? YES bhej dijiye."),
    "customer_lapsed_soft": ("Would you like us to hold a spot for you this week? Reply YES.",
                             "Kya hum is hafte aapke liye ek spot hold kar dein? YES bhej dijiye."),
    "customer_lapsed_hard": ("No pressure at all. Would you like us to hold a spot for you? Reply YES.",
                             "Bilkul koi pressure nahin. Kya hum aapke liye ek spot hold kar dein? YES bhej dijiye."),
    "wedding_package_followup": ("Shall we set up the next step? Reply YES.",
                                 "Kya hum agla step set up kar dein? YES bhej dijiye."),
    "generic_customer": ("Is there anything we can help with? Reply YES and we will get in touch.",
                         "Kya hum kisi cheez mein madad kar sakte hain? YES bhej dijiye."),
}
# Promotions only go into messages whose purpose is bringing someone back; never into a refill or a reminder.
OFFER_KINDS = frozenset({"recall_due", "trial_followup", "customer_lapsed_soft", "customer_lapsed_hard",
                         "wedding_package_followup", "generic_customer"})

_LABELS = {
    "suggested action": "Suggested next step",
    "active offer": "Your live offer",
    "offer you could run (from the catalogue)": "An offer you could run",
    "likely driver": "Likely driver",
    "past services": "Past services",
}
_LABEL = re.compile(r"^([a-z][a-z ()]{2,45}): (.+)$", re.S)
_DORMANT_KEYS = frozenset({"trigger.dormant", "signal.dormant"})


# --------------------------------------------------------------------------- small helpers

def _plain(text: str) -> str:
    return re.sub(r"\bCTR\b", "click-through rate", " ".join(text.split()))


def _end(text: str) -> str:
    text = text.strip()
    return text if text[-1:] in ".!?" else text + "."


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lower_first(text: str) -> str:
    return text if text[:2] in ("I ", "I'") else text[:1].lower() + text[1:]


def _support_sentence(fact: Fact) -> str:
    text = _plain(fact.text)
    match = _LABEL.match(text)
    if match:
        label, value = match.groups()
        return f"{_LABELS.get(label, _cap(label))}: {_end(value)}"
    return f"For context: {_end(text)}"


def _value(brief: "Brief", key: str, prefix: str) -> str:
    """The text of the fact with this key (lead or support), without its label."""
    for fact in (brief.lead, *brief.support):
        if fact is not None and fact.key == key:
            return fact.text.removeprefix(prefix).strip()
    return ""


def _fit(parts: list[str], limit: int) -> str | None:
    """Join the sentences; drop optional ones, then shorten the opening, until the body fits."""
    parts = [p for p in parts if p]
    while len(" ".join(parts)) > limit and len(parts) > 2:
        del parts[-2]  # the sentence just before the ask
    body = " ".join(parts)
    if len(body) <= limit:
        return body
    ask, head = parts[-1], parts[0]
    room = limit - len(ask) - 1
    kept = ""
    for sentence in re.split(r"(?<=[.!?])\s+", head):
        if len(kept) + len(sentence) + 1 > room:
            break
        kept = f"{kept} {sentence}".strip()
    return f"{kept} {ask}" if kept else None


def _topic_of(lead_text: str) -> str:
    match = re.search(r"merchant is planning: (.+?)(?:; their last message|$)", lead_text)
    return match.group(1).strip() if match else ""


# --------------------------------------------------------------------------- merchant-facing

def _merchant_message(brief: "Brief", hindi: bool, variant: int) -> list[str]:
    i = 1 if hindi else 0
    lead, kind = brief.lead, brief.kind
    name = f"{brief.salutation},"
    support = [f for f in brief.support if f.text != lead.text]
    clients = CLIENTS.get(brief.category_slug, "customers")
    # A kind-specific ask only makes sense when the lead really is about that kind of event.
    about_the_trigger = not brief.degraded or lead.key == "trigger.kind"
    ask = ASKS[kind if about_the_trigger and kind in ASKS else "generic_merchant"][i].format(clients=clients)

    if kind == "curious_ask_due":
        question = _cap(lead.text.removeprefix("question to ask:").strip()).rstrip("?") + "?"
        promise = ("Aapke jawab se main ek ready-to-post offer aur customers ke liye ek chhota reply bana dungi."
                   if hindi else "I will turn your answer into a ready-to-post offer and a short reply for customers.")
        return [f"{name} {'ek quick sawaal.' if hindi else 'quick question.'} {promise}", question]

    if kind == "active_planning_intent":
        topic = _topic_of(lead.text)
        if hindi:
            return [f"{name} badhiya. Main{f' aapke {topic} ka' if topic else ' iska'} ek starter draft bana kar yahin bhej dungi, aap use edit kar sakte hain.",
                    "Kya main pehla version abhi bhej doon? YES bhej dijiye."]
        return [f"{name} great. I will put together a starter draft{f' for your {topic}' if topic else ''} and send it here for your edits.",
                "Shall I send the first version now? Reply YES."]

    if lead.key in _DORMANT_KEYS:  # never say "we have not heard from you": open with something useful instead
        if support:
            return [f"{name} {_end(_plain(OWN_PROFILE_LEAD_IN[i].format(lead=support[0].text)))}", ask]
        return [f"{name} {'bas ek quick check-in.' if hindi else 'a quick check-in from my side.'}", ask]

    if lead.key == "trigger.kind":
        sentence = _end(PLAIN_MERCHANT.get(kind, NO_DETAIL)[i])
    elif brief.degraded:
        sentence = _end(OWN_PROFILE_LEAD_IN[i].format(lead=_plain(lead.text)))
    else:
        text = _plain(lead.text)
        for old, new in REPHRASE.get(kind, ()):
            text = text.replace(old, new)
        sentence = _end(LEAD_IN.get(kind, LEAD_IN["generic_merchant"])[i].format(lead=text))
    parts = [f"{name} {sentence}"]
    if support:
        parts.append(_support_sentence(support[variant % len(support)]))
    parts.append(ask)
    return parts


# --------------------------------------------------------------------------- customer-facing

def _lapsed_sentence(lead_text: str, your: str) -> str:
    days = re.search(r"(\d+) days since the last visit", lead_text)
    goal = re.search(r"earlier goal: ([^;]+)", lead_text)
    first = f"It has been {days.group(1)} days since {your} last visit" if days else f"It has been a while since {your} last visit"
    return _end(first + (f", and the goal then was {goal.group(1).strip()}" if goal else ""))


def _customer_message(brief: "Brief", hindi: bool, variant: int) -> list[str]:
    i = 1 if hindi else 0
    lead, kind = brief.lead, brief.kind
    business = business_name(brief)
    name = brief.salutation
    about = brief.about or ""
    # A relative's phone (a son, a parent) receives a message about the customer, who is not the reader.
    relative = any("third person" in n for n in brief.notes) and bool(name) and not about

    if hindi:
        greeting = f"Namaste {name}, {business} ki taraf se." if name and not relative else f"Namaste, {business} ki taraf se."
    else:
        greeting = f"Hi {name}, {business} here." if name and not relative else f"Hello, {business} here."

    who = about or (name if relative else "")  # whom the message is about, when that is not the reader
    your, ki, ka = (f"{who}'s", f"{who} ki", f"{who} ka") if who else ("your", "aapki", "aapka")
    for_ = (f" for {who}" if not hindi else f" ({who} ke liye)") if who else ""
    last_visit = _value(brief, "cust.last_visit", "last visit:")
    text = _plain(lead.text)

    if lead.key == "trigger.kind":
        when = (f" on {last_visit}" if not hindi else f" ({last_visit})") if last_visit and kind.startswith("customer_lapsed") else ""
        frame = PLAIN_CUSTOMER.get(kind, PLAIN_CUSTOMER["generic_customer"])[i]
        opening = _end(_cap(frame.format(your=your, ki=ki, ka=ka, when=when)))
    elif lead.key == "cust.last_visit":
        opening = f"{_cap(your)} last visit with us was on {last_visit}." if not hindi else f"{_cap(ki)} last visit {last_visit} ko thi."
    elif lead.key == "trigger.lapsed":
        opening = _lapsed_sentence(text, your) if not hindi else f"{_cap(ki)} last visit ke baad kaafi time ho gaya hai: {text}."
    else:
        frame = CUSTOMER_LEAD_IN.get(kind, CUSTOMER_LEAD_IN["generic_customer"])[i]
        opening = frame.format(lead=text.rstrip("."), for_=for_)
    if who and who.lower() not in opening.lower():  # the reader must see whom this is about
        opening = f"For {who}: {_lower_first(opening)}" if not hindi else f"{who} ke liye: {_lower_first(opening)}"
    parts = [f"{greeting} {opening}"]

    # One middle sentence at most: the saved delivery address, or the live offer (slots go into the ask).
    if kind == "chronic_refill_due" and _value(brief, "refill.delivery", ""):
        parts.append("We can also deliver it to the saved address." if not hindi else "Hum ise saved address par deliver bhi kar sakte hain.")
    elif kind in OFFER_KINDS:
        offer = _value(brief, "offer.active", "active offer:").split("; ")[0]
        if offer:
            parts.append(f"We currently have {offer}." if not hindi else f"Abhi hamare paas {offer} chal raha hai.")
        elif last_visit and lead.key == "trigger.kind" and kind == "recall_due":
            parts.append(f"{_cap(your)} last visit was on {last_visit}." if not hindi else f"{_cap(ki)} last visit {last_visit} ko thi.")

    ask = CUSTOMER_ASKS.get(kind, CUSTOMER_ASKS["generic_customer"])[i]
    labels = slot_labels(brief) if brief.cta == "multi_choice_slot" else []
    if labels:  # slot labels contain commas themselves, so the choices are separated by semicolons
        numbered = "; ".join(f"{n} for {label}" for n, label in enumerate(labels[:3], start=1))
        ask = f"Reply {numbered}; or tell us a time that works." if not hindi else f"Reply {numbered}; ya koi aur time bata dijiye."
    parts.append(ask)
    return parts


# --------------------------------------------------------------------------- public

def render(brief: "Brief", variant: int = 0) -> str | None:
    """The fallback message for this Brief, or None when there is nothing safe to say.

    `variant` picks a different supporting fact when the first choice repeated an earlier message.
    """
    if brief.lead is None:
        return None
    hindi = brief.language in ("hi", "hi-en")
    parts = _customer_message(brief, hindi, variant) if brief.send_as == "merchant_on_behalf" else _merchant_message(brief, hindi, variant)
    return _fit(parts, max_chars(brief))


_CTA_PHRASE = {
    "binary_yes_no": "yes/no", "open_ended": "open-ended", "binary_confirm_cancel": "confirm-or-reschedule",
    "multi_choice_slot": "slot-choice", "none": "no-reply-needed",
}


def rationale(brief: "Brief") -> str:
    reason = REASON.get(brief.kind, REASON["generic_merchant"])
    lead = brief.lead.text if brief.lead else ""
    lead = lead if len(lead) <= 110 else lead[:107].rstrip() + "..."
    return f'Why now: {reason}. Leads with "{lead}" and closes with a single {_CTA_PHRASE.get(brief.cta, "simple")} ask.'
