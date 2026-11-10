"""What the model is told.

The Brief already fixes what the message is about (the lead, the supporting facts, the call to
action, the cautions), so the prompt only has to say how to phrase it well and what is forbidden.
`grounding_lines` lists every piece of data the model is shown; the validator uses the same list to
check that no number in the reply came from anywhere else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .facts import active_offers

if TYPE_CHECKING:
    from .signals import Brief

SYSTEM = """You write ONE WhatsApp message, either from Vera (magicpin's assistant) to a merchant, or from a local business to one of its own customers. A busy person reads it on a phone. It is scored on five things: specificity, category fit, merchant fit, why-now relevance, and whether the reader will want to reply.

HOW A STRONG MESSAGE IS BUILT
1. Open with the recipient's name, then go straight to the reason for writing now (the LEAD). No small talk ("hope you are well"), no "I am reaching out", no self-introduction.
2. Anchor on concrete facts: quote numbers, dates, names and sources exactly as given under FACTS.
3. Say in one short line why it matters to THIS business, using one supporting fact. Never add generic claims.
4. Lower the effort: offer to do the work yourself ("I'll draft...", "I can pull...") instead of asking them to do it.
5. The last sentence is the single call to action, shaped as the CALL TO ACTION line says. One ask only, never two.
Tone: a sharp, friendly colleague from the same trade, never an advertiser. At most one exclamation mark. No hype, no capital-letter shouting.

HARD RULES (a message that breaks one is thrown away)
- Use ONLY the FACTS, OFFERS and CAUTIONS you are given. Never invent or estimate a number, name, price, offer, date, study, competitor, review or customer count. Never calculate new numbers (no new percentages, differences or totals). If something is not given, leave it out.
- No links, website names, e-mail addresses or phone numbers.
- Never use internal words: trigger, payload, lead, CTA, template, JSON, ids, "CTR" (write "click-through rate" if you need it).
- Obey every CAUTION literally.
- Do not promise outcomes ("guaranteed", "will double", "best in city") and never use a word from NEVER SAY.
- Do not repeat or lightly rephrase anything listed under ALREADY SENT.
- If there has been an earlier conversation, do not introduce yourself again.
- A message from a business to its customer is written as that business ("<name> here"), warm and short. It never mentions Vera or magicpin, offers only what the facts show, gives no medical advice, and uses the customer's name. Emoji: none to professionals; at most one to a customer.

OUTPUT: a JSON object and nothing else:
{"body": "<the WhatsApp message>", "rationale": "<one or two plain sentences: why this message now, and which facts and levers it uses>"}

TWO EXAMPLES OF THE SHAPE (other businesses and facts; never reuse their facts or wording)
Facts: LEAD 3 reviews this week mention long waiting time (quote: "waited 40 minutes for a haircut"). Recipient Anjali, a salon owner, language Hinglish, ask yes/no.
{"body": "Anjali, is hafte 3 reviews mein waiting time ka zikr hai - ek ne likha \\"waited 40 minutes for a haircut\\". Main teeno reviews ka polite reply aur peak hours ke liye ek 'slot book karein' message draft kar doon? Reply YES.", "rationale": "Review theme with an exact quote; offers the drafting work; single yes/no ask."}
Facts: LEAD hair spa due, last done 3 Mar. Slots: Thu 5pm | Sat 11am. Offer: Hair Spa @ ₹499. Recipient Kavita, a salon customer, language English, ask: pick a slot.
{"body": "Hi Kavita, Glow Salon here. Your last hair spa was on 3 Mar, so the next one is due. We have Thu 5pm and Sat 11am open, and Hair Spa @ ₹499 is on. Reply 1 for Thu, 2 for Sat, or tell us a time that suits you.", "rationale": "Recall for a specific service with the real slots and the real offer; one reply to book."}
"""

LANGUAGE = {
    "en": "English only.",
    "hi-en": "Hinglish: natural Hindi-English code-mix in Roman script, the way an Indian shop owner texts (aapka, kar doon, hafte). Keep numbers, brand names and technical terms in English.",
    "hi": "Hindi in Roman script (Hinglish), respectful 'aap' register, simple English words where natural.",
}
_INDIAN = {"ta": "Tamil", "te": "Telugu", "kn": "Kannada", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam"}

CTA_RULES = {
    "binary_yes_no": "End with ONE question answerable with a single word, e.g. 'Want me to ...? Reply YES.' Nothing after it.",
    "open_ended": "End with ONE short, easy question (what / which / how) they can answer in a line. Nothing after it.",
    "binary_confirm_cancel": "End by asking them to reply CONFIRM to go ahead, or to tell you a better time. One ask only.",
    "multi_choice_slot": "Offer exactly the slots listed under FACTS, each slot's wording unchanged, as the LAST sentence in this shape: 'Reply 1 for <first slot>; 2 for <second slot>; or tell us a time that works.' (in Hinglish end with 'ya koi aur time bata dijiye' instead of 'or tell us a time that works'; semicolons between choices, because slot wording contains commas). Never write the words 'slots available' or a list like '1. ..., 2. ...'. This is the only ask.",
    "none": "No question and no ask; close with a plain statement.",
}

LEVERS = {
    "specificity": "specificity (exact numbers, dates, names, sources)",
    "loss_aversion": "loss aversion (what waiting costs, only as far as the facts show)",
    "social_proof": "social proof (a comparison or peer fact, only if one is given)",
    "effort_externalization": "effort externalization (offer to draft, pull or set it up)",
    "curiosity": "curiosity (a hook that makes them want to see more)",
    "reciprocity": "reciprocity (give something useful first)",
    "asking_the_merchant": "asking the merchant (one easy question about their business)",
    "reassurance": "reassurance (calm, factual reframing)",
    "warmth": "warmth (personal, no pressure, no guilt)",
}

# One plain sentence per kind so the model understands "why now" even when the trigger carried no detail.
REASON = {
    "research_digest": "a new research or digest item relevant to the merchant's field has been published",
    "regulation_change": "a rule or regulation relevant to the merchant's field has changed",
    "cde_opportunity": "a continuing-education event is coming up",
    "supply_alert": "a product recall or supply alert affects the merchant's stock",
    "category_seasonal": "a seasonal demand shift is under way in the merchant's field",
    "perf_dip": "the merchant's performance numbers have dipped",
    "perf_spike": "the merchant's performance numbers have risen",
    "seasonal_perf_dip": "the merchant's numbers dipped for a normal seasonal reason",
    "milestone_reached": "the merchant is at or near a milestone",
    "review_theme_emerged": "a theme has emerged in the merchant's recent reviews",
    "competitor_opened": "a new competitor has opened nearby",
    "renewal_due": "the merchant's subscription is up for renewal",
    "winback_eligible": "the merchant's lapsed subscription can be won back",
    "dormant_with_vera": "the merchant has not talked to Vera for a while",
    "curious_ask_due": "it is time for Vera's regular question to the merchant about their business",
    "active_planning_intent": "the merchant said yes to a plan and is waiting for the draft",
    "gbp_unverified": "the merchant's Google business profile is not verified",
    "festival_upcoming": "a festival is coming up",
    "ipl_match_today": "an IPL match is on today",
    "recall_due": "the customer is due for a repeat visit",
    "appointment_tomorrow": "the customer has an appointment tomorrow",
    "chronic_refill_due": "the customer's regular medicines are due for a refill",
    "trial_followup": "the customer recently tried a session",
    "customer_lapsed_soft": "the customer has not visited for a while",
    "customer_lapsed_hard": "the customer has not visited for a long time",
    "wedding_package_followup": "the customer's wedding is coming up",
    "generic_merchant": "something happened that the merchant should hear about",
    "generic_customer": "the business wants to check in with the customer",
}

# Always forbidden, on top of the category's own list.
UNIVERSAL_TABOO = ("guaranteed", "miracle", "100% safe", "best in city")


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


REPLY_SYSTEM = """You write ONE WhatsApp reply inside an ongoing conversation, either from Vera (magicpin's assistant) to a merchant, or from a local business to one of its own customers. A busy person reads it on a phone. It is scored on specificity, fit to the business, relevance to what they just said, and whether it moves the conversation forward.

HOW A STRONG REPLY IS BUILT
1. Answer THEIR LATEST MESSAGE first, in their language. Do not repeat your earlier message.
2. Use concrete facts from FACTS and from the conversation (exact names, numbers, dates, offers). Never invent anything that is not given.
3. Do the work yourself (a short draft, the next step) instead of asking them to do it.
4. End with one clear next step, shaped as WHAT TO DO says. One ask only.
Tone: a sharp, friendly colleague. Short. At most one exclamation mark. No hype.

HARD RULES (a reply that breaks one is thrown away)
- Use ONLY the FACTS, OFFERS, CAUTIONS and the conversation. Never invent, estimate or calculate a number, price, date, offer, study or customer count.
- No links, website names, e-mail addresses or phone numbers.
- Never use internal words: trigger, payload, lead, CTA, template, JSON, ids.
- Do not promise outcomes ("guaranteed", "will double") and never use a word from NEVER SAY.
- Do not repeat or lightly rephrase anything under ALREADY SENT.
- Do not introduce yourself again unless they ask who you are.
- A reply from a business to its customer is written as that business ("we"), never mentions Vera or magicpin, offers only what the facts show and gives no medical advice.

OUTPUT: a JSON object and nothing else:
{"body": "<the WhatsApp reply>", "rationale": "<one or two plain sentences: what they meant and why this reply>"}
"""

# What the reply must do, by what the other person just said.
REPLY_TASKS = {
    "go_ahead": "They said yes. Start the work NOW instead of asking anything: say what you are doing, give a short first draft or the concrete content built only from the FACTS, then end with ONE line asking them to reply CONFIRM to go ahead. Do not ask qualifying questions and never use the phrases 'would you', 'do you', 'can you tell', 'what if', 'how about'.",
    "question": "They asked a question. Answer it directly in one or two sentences using only the FACTS. If the facts do not cover it, say plainly that you will check and come back; never guess. Then end with one easy next step (reply YES).",
    "off_topic": "They asked for something outside what you do ({topic}). Decline in one friendly sentence and point to the right person (for GST, their CA), then steer back to the earlier topic and end with one ask.",
    "other": "They said something that is neither a yes nor a question. Acknowledge it in one sentence using their own words, then move the earlier topic forward and end with one ask.",
    "customer_go_ahead": "The customer agreed. Write as the business. If they chose one of the slots under FACTS, confirm exactly that slot; if they did not choose, list the slots under FACTS (numbered, wording unchanged) and ask them to pick one. Promise nothing else.",
    "customer_question": "The customer asked something. Answer only from the FACTS; if the facts do not cover it, say the team will confirm shortly. Write as the business, warm and short, with no medical advice.",
    "customer_other": "The customer said something that is not a clear yes or a question. Acknowledge it warmly as the business and say the team will follow up, using only the FACTS.",
}


def reply_max_chars(customer_facing: bool) -> int:
    return 650 if customer_facing else 900


# --------------------------------------------------------------------------- small shared helpers

def language_rule(code: str) -> str:
    if code in LANGUAGE:
        return LANGUAGE[code]
    base = code.split("-")[0]
    if code.endswith("-en") and base in _INDIAN:
        return f"English mixed lightly with {_INDIAN[base]} in Roman script."
    return LANGUAGE["en"]


def max_chars(brief: "Brief") -> int:
    if brief.kind == "active_planning_intent":
        return 1300  # a usable draft needs room
    return 650 if brief.send_as == "merchant_on_behalf" else 800


def _length_hint(brief: "Brief") -> str:
    if brief.kind == "active_planning_intent":
        return "80 to 170 words: a usable draft the merchant can edit, then the ask."
    return "25 to 65 words." if brief.send_as == "merchant_on_behalf" else "40 to 85 words."


def slot_labels(brief: "Brief") -> list[str]:
    """The slot wording the trigger supplied (the Brief carries it as a support fact)."""
    for fact in brief.support:
        if fact.key == "slots":
            return [part.strip() for part in fact.text.removeprefix("slots available:").split("|") if part.strip()]
    return []


def taboo_words(category: dict) -> list[str]:
    """Phrases this category must never use; a bracketed qualifier like '(without disclosure)' is dropped."""
    voice = category.get("voice") or {}
    listed = voice.get("vocab_taboo") or voice.get("taboos") or []
    cleaned = [re.sub(r"\(.*?\)", "", str(item)).strip().lower() for item in listed]
    return [*dict.fromkeys([*(w for w in cleaned if w), *UNIVERSAL_TABOO])]


def business_name(brief: "Brief") -> str:
    return str((brief.merchant.get("identity") or {}).get("name") or "the business")


def grounding_lines(brief: "Brief") -> list[str]:
    """Every piece of data the model is shown. A number that is in none of these is invented."""
    identity = brief.merchant.get("identity") or {}
    lines = [t for t in (
        brief.lead.text if brief.lead else "", *(f.text for f in brief.support), *brief.notes,
        *(o["title"] for o in active_offers(brief.merchant)),
        business_name(brief), str(identity.get("locality") or ""), str(identity.get("city") or ""),
        brief.salutation, brief.about or "",
    ) if t]
    return lines


# --------------------------------------------------------------------------- the prompt

def _clip(text: str, n: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_prompt(
    brief: "Brief", *, avoid: Iterable[str] = (), feedback: Iterable[str] = (), previous: str = ""
) -> Prompt:
    """The user turn for one Brief. `feedback` and `previous` are set when a first draft was rejected."""
    identity = brief.merchant.get("identity") or {}
    customer_facing = brief.send_as == "merchant_on_behalf"
    voice = brief.category.get("voice") or {}
    place = ", ".join(str(x) for x in (identity.get("locality"), identity.get("city")) if x)
    recipient = brief.salutation or "(name not known: greet plainly)"
    if brief.about:
        recipient += f" (writing about {brief.about})"

    sender = (
        f'the business "{business_name(brief)}" writing to its own customer (use "we"; never mention Vera or magicpin)'
        if customer_facing else "Vera, magicpin's assistant, writing to the merchant"
    )
    lines = [
        f"SENDER: {sender}",
        f"RECIPIENT: {recipient}",
        f"BUSINESS: {business_name(brief)}" + (f", {place}" if place else "") + f" ({brief.category.get('display_name') or brief.category_slug})",
        f"LANGUAGE: {language_rule(brief.language)}",
        f"WHY NOW: {REASON.get(brief.kind, REASON['generic_merchant'])}",
        f"GOAL: {brief.strategy.goal}",
        "LEVERS TO USE (only those the facts support): " + "; ".join(LEVERS.get(x, x) for x in brief.strategy.levers),
        f"CALL TO ACTION: {CTA_RULES.get(brief.cta, CTA_RULES['open_ended'])}",
        "",
        "FACTS",
        f"LEAD (open on this): {brief.lead.text if brief.lead else '(none)'}",
    ]
    lines += [f"SUPPORT: {fact.text}" for fact in brief.support]
    offers = [o["title"] for o in active_offers(brief.merchant)]
    if offers:
        lines.append("OFFERS (quote exactly if you mention one): " + "; ".join(offers))
    if brief.notes:
        lines.append("")
        lines.append("CAUTIONS")
        lines += [f"- {note}" for note in brief.notes]

    lines += [
        "",
        f"VOICE: tone {voice.get('tone', 'peer')}, register {voice.get('register', 'friendly')}; natural vocabulary: "
        + ", ".join(str(w) for w in (voice.get("vocab_allowed") or [])[:8]),
        "NEVER SAY: " + ", ".join(taboo_words(brief.category)),
    ]
    sent = [*brief.prior_bodies[-2:], *list(avoid)[-3:]]
    if sent:
        lines += ["", "ALREADY SENT (do not repeat or rephrase):"] + [f'- "{_clip(b)}"' for b in sent]
    lines += ["", f"LENGTH: {_length_hint(brief)}"]

    problems = [p for p in feedback]
    if problems:
        lines += ["", "YOUR PREVIOUS DRAFT WAS REJECTED"]
        if previous:
            lines.append(f'Draft: "{_clip(previous, 500)}"')
        lines += [f"- {p}" for p in problems]
        lines.append("Write a new version that fixes every problem and still follows all rules above.")
    return Prompt(SYSTEM, "\n".join(lines))


def build_reply_prompt(
    brief: "Brief", *, lang: str, transcript: Iterable[tuple[str, str]], message: str, task: str, avoid: Iterable[str] = (),
) -> Prompt:
    """The user turn for one reply. `transcript` is [(speaker, text), ...], oldest first, ending before `message`."""
    identity = brief.merchant.get("identity") or {}
    customer_facing = brief.send_as == "merchant_on_behalf"
    voice = brief.category.get("voice") or {}
    place = ", ".join(str(x) for x in (identity.get("locality"), identity.get("city")) if x)
    recipient = brief.salutation or "(name not known: greet plainly)"
    if brief.about:
        recipient += f" (writing about {brief.about})"
    sender = (
        f'the business "{business_name(brief)}" writing to its own customer (use "we"; never mention Vera or magicpin)'
        if customer_facing else "Vera, magicpin's assistant, writing to the merchant"
    )
    lines = [
        f"SENDER: {sender}",
        f"RECIPIENT: {recipient}",
        f"BUSINESS: {business_name(brief)}" + (f", {place}" if place else "") + f" ({brief.category.get('display_name') or brief.category_slug})",
        f"LANGUAGE (match their latest message): {language_rule(lang)}",
        f"WHAT THE EARLIER MESSAGES WERE ABOUT: {REASON.get(brief.kind, REASON['generic_merchant'])}",
        "",
        "FACTS",
        f"MAIN FACT: {brief.lead.text if brief.lead else '(none)'}",
    ]
    lines += [f"SUPPORT: {fact.text}" for fact in brief.support]
    offers = [o["title"] for o in active_offers(brief.merchant)]
    if offers:
        lines.append("OFFERS (quote exactly if you mention one): " + "; ".join(offers))
    if brief.notes:
        lines += ["", "CAUTIONS"] + [f"- {note}" for note in brief.notes]
    lines += ["", "CONVERSATION SO FAR (oldest first)"]
    lines += [f"{speaker}: {_clip(text, 320)}" for speaker, text in transcript]
    lines += [
        "",
        f'THEIR LATEST MESSAGE: "{_clip(message, 400)}"',
        f"WHAT TO DO: {task}",
        "",
        f"VOICE: tone {voice.get('tone', 'peer')}, register {voice.get('register', 'friendly')}",
        "NEVER SAY: " + ", ".join(taboo_words(brief.category)),
    ]
    sent = list(avoid)[-4:]
    if sent:
        lines += ["", "ALREADY SENT (do not repeat or rephrase):"] + [f'- "{_clip(b)}"' for b in sent]
    lines += ["", "LENGTH: 25 to 80 words" + (" (a short draft may take up to 130)." if not customer_facing else ".")]
    return Prompt(REPLY_SYSTEM, "\n".join(lines))
