"""Lead-signal builders: for each trigger kind, pick the ONE fact the message leads with.

The lead is the "why now" the judge looks for. It is chosen here, deterministically, before
any language model runs, so the model only has to phrase a decision that has already been made.
Each builder also returns a few supporting facts and short notes (traps for the composer).

Two rules hold for every builder:
  * only use what is in the contexts; never compute anything from the clock
  * if the trigger carries a placeholder payload (no real details), fall back to what the
    merchant's own data supports and say so in the notes, instead of inventing a story
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .facts import (
    Fact, active_offers, aggregate_facts, beat_for, catalog_offer, ctr_vs_peer, customer_facts,
    digest_action, digest_fact, digest_of_kind, find_digest, money, move_fact, num, offer_fact, pct,
    perf_snapshot, pick_digest, review_facts, search_digest, signal_facts, subscription_fact, when, words,
)


@dataclass
class Draft:
    lead: Fact | None = None
    support: list[Fact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Ctx:
    trigger: dict
    merchant: dict
    category: dict
    customer: dict | None

    @property
    def payload(self) -> dict:
        return self.trigger.get("payload") or {}

    @property
    def placeholder(self) -> bool:
        return bool(self.payload.get("placeholder"))


_BUILDERS: dict[str, Callable[[Ctx], Draft]] = {}


def _for(*kinds: str):
    def register(fn: Callable[[Ctx], Draft]) -> Callable[[Ctx], Draft]:
        for kind in kinds:
            _BUILDERS[kind] = fn
        return fn
    return register


# When a trigger carries only a placeholder payload, the kind alone still says why we are writing.
# These statements contain no invented specifics (no names, dates, numbers), so they are safe leads.
KIND_STATEMENT = {
    "appointment_tomorrow": "the customer has an appointment tomorrow (time and service not specified)",
    "recall_due": "the customer is due for a regular visit (service and date not specified)",
    "chronic_refill_due": "the customer's regular medicines are due for a refill (medicines not specified)",
    "trial_followup": "the customer recently tried a session and a follow-up is due",
    "customer_lapsed_soft": "the customer has not visited for a while",
    "customer_lapsed_hard": "the customer has not visited for a long time",
    "wedding_package_followup": "the customer's wedding is coming up and a follow-up on the bridal package is due",
    "competitor_opened": "a new competitor has opened near this merchant (no details available)",
}


def build_draft(kind: str, ctx: Ctx) -> Draft:
    """The draft for a canonical kind; unknown kinds get the generic payload-based draft."""
    draft = _BUILDERS.get(kind, _generic)(ctx)
    if draft.lead is None and ctx.placeholder and kind in KIND_STATEMENT:
        draft.lead = Fact("trigger.kind", KIND_STATEMENT[kind])
    return draft


# --------------------------------------------------------------------------- small helpers

def _opt(*items: Fact | None) -> list[Fact]:
    return [item for item in items if item is not None]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _span(window: Any) -> str:
    """'7d' -> 'the last 7 days'."""
    if not window:
        return "the recent window"
    match = re.fullmatch(r"(\d+)\s*([dw])", str(window).strip().lower())
    if not match:
        return words(window)
    n, unit = int(match.group(1)), "day" if match.group(2) == "d" else "week"
    return f"the last {n} {unit}{'' if n == 1 else 's'}"


def _slots_fact(options: Any) -> Fact | None:
    """Slot labels are used exactly as given; weekday names are never recomputed from dates."""
    labels = [
        str(s.get("label") or when(s.get("iso")))
        for s in options or [] if isinstance(s, dict) and (s.get("label") or when(s.get("iso")))
    ]
    return Fact("slots", "slots available: " + " | ".join(labels)) if labels else None


def _matching_offer(merchant: dict, text: str) -> Fact | None:
    """The merchant's own active offer that relates to `text` (e.g. the service being recalled)."""
    wanted = set(re.findall(r"[a-z]{4,}", text.lower()))
    for offer in active_offers(merchant):
        if wanted & set(re.findall(r"[a-z]{4,}", offer["title"].lower())):
            return Fact("offer.active", f"active offer: {offer['title']}")
    return None


def _suggest(c: Ctx, **kwargs: Any) -> Fact | None:
    """A catalogue offer the merchant is not already running."""
    return catalog_offer(c.category, exclude=[o["title"] for o in active_offers(c.merchant)], **kwargs)


_METRIC_NAME = {"review_count": "reviews"}
_ASK_PHRASE = {"what_service_in_demand_this_week": "which service is most in demand this week"}


def _month_of(date: Any) -> str | None:
    try:
        return datetime.fromisoformat(str(date)[:10]).strftime("%b").lower()
    except ValueError:
        return None


def _payload_facts(payload: dict) -> list[Fact]:
    """Readable facts from any payload, for kinds we have no dedicated builder for."""
    out = []
    for key, value in payload.items():
        if key in ("placeholder", "metric_or_topic") or value in (None, "", [], {}):
            continue
        if isinstance(value, bool):
            text = f"{words(key)}: {'yes' if value else 'no'}"
        elif isinstance(value, float) and key.endswith("_pct"):
            text = f"{words(key[:-4])}: {pct(value)}"
        elif isinstance(value, (int, float, str)):
            text = f"{words(key)}: {value}"
        elif isinstance(value, list) and all(isinstance(v, (int, float, str)) for v in value):
            text = f"{words(key)}: {', '.join(str(v) for v in value)}"
        else:
            continue
        out.append(Fact(f"payload.{key}", text))
    return out


def _generic(c: Ctx) -> Draft:
    facts = _payload_facts(c.payload)
    return Draft(facts[0] if facts else None, facts[1:4])


# --------------------------------------------------------------------------- knowledge and compliance

@_for("research_digest")
def _research(c: Ctx) -> Draft:
    item = find_digest(c.category, c.payload.get("top_item_id"))
    notes = []
    if item is None:
        item = pick_digest(c.category, c.merchant, avoid_kinds=("compliance", "cde", "alert"))
        if item:
            notes.append("The trigger named no digest item, so this one was picked as the best match for this merchant.")
    if item is None:
        return Draft(notes=["No digest item is available; do not cite any research."])
    segment = None
    if "high_risk" in str(item.get("patient_segment", "")):
        segment = next((f for f in aggregate_facts(c.merchant) if f.key == "agg.high_risk_adult_count"), None)
    return Draft(digest_fact(item), _opt(digest_action(item), segment), notes)


@_for("regulation_change")
def _regulation(c: Ctx) -> Draft:
    item = find_digest(c.category, c.payload.get("top_item_id")) or digest_of_kind(c.category, "compliance")
    deadline = when(c.payload.get("deadline_iso"))
    deadline_fact = Fact("reg.deadline", f"deadline: {deadline}") if deadline else None
    if item is None:
        lead = Fact("reg.deadline", f"a regulation change applies from {deadline}") if deadline else None
        return Draft(lead, notes=["The rule text is not available; do not describe its details."])
    return Draft(digest_fact(item), _opt(deadline_fact, digest_action(item)))


@_for("cde_opportunity")
def _cde(c: Ctx) -> Draft:
    item = find_digest(c.category, c.payload.get("digest_item_id")) or digest_of_kind(c.category, "cde")
    if item is None:
        return Draft(notes=["No event details are available; do not name an event."])
    bits = [item.get("title", "CDE event")]
    if when(item.get("date")):
        bits.append(when(item["date"]))
    credits = c.payload.get("credits", item.get("credits"))
    if credits:
        bits.append(f"{num(credits)} CDE credits")
    if c.payload.get("fee"):
        bits.append(f"fee: {words(c.payload['fee'])}")
    lead = Fact("cde.event", ", ".join(bits) + f" (source: {item.get('source', 'event calendar')})")
    summary = Fact("cde.summary", item["summary"]) if item.get("summary") else None
    return Draft(lead, _opt(summary, digest_action(item)))


@_for("supply_alert")
def _supply(c: Ctx) -> Draft:
    p = c.payload
    item = find_digest(c.category, p.get("alert_id")) or digest_of_kind(c.category, "alert")
    parts = [f"{p['molecule']} recall" if p.get("molecule") else "product recall"]
    if p.get("affected_batches"):
        parts.append("batches " + ", ".join(str(b) for b in p["affected_batches"]))
    if p.get("manufacturer"):
        parts.append(f"from {p['manufacturer']}")
    text = ", ".join(parts) + (f" (source: {item['source']})" if item and item.get("source") else "")
    detail = Fact("alert.detail", item["summary"]) if item and item.get("summary") else None
    count = next((f for f in aggregate_facts(c.merchant) if f.key == "agg.chronic_rx_count"), None)
    return Draft(Fact("alert.recall", text), _opt(detail, digest_action(item) if item else None, count))


_TREND = re.compile(r"(?P<name>.+?)_demand_(?P<delta>[+-]\d+)")


def _trend_text(raw: Any) -> str:
    match = _TREND.fullmatch(str(raw))
    return f"{words(match['name'])} demand {match['delta']}%" if match else words(raw)


@_for("category_seasonal")
def _category_seasonal(c: Ctx) -> Draft:
    season = words(c.payload.get("season") or "this season")
    trends = [_trend_text(t) for t in c.payload.get("trends") or []]
    lead = Fact("season.trends", f"{season}: " + ", ".join(trends)) if trends else None
    item = search_digest(c.category, season.split()[0])
    return Draft(lead, _opt(digest_action(item) if item else None, beat_for(c.category, tokens=season.split())))


# --------------------------------------------------------------------------- performance

def _perf_lead(c: Ctx, direction: str) -> tuple[Fact | None, list[str]]:
    p = c.payload
    delta = _number(p.get("delta_pct"))
    if not c.placeholder and p.get("metric") and delta is not None:
        base = f", against a baseline of {num(p['vs_baseline'])}" if p.get("vs_baseline") is not None else ""
        word = "down" if delta < 0 else "up"
        text = f"{p['metric']} {word} {pct(delta)} over {_span(p.get('window'))}{base}"
        return Fact(f"trigger.perf_{direction}", text), []
    lead = move_fact(c.merchant, direction, minimum=0.05 if direction == "down" else 0.10)
    if lead is None:
        return None, [f"The trigger gave no numbers and the merchant's own data shows no clear {direction} move: do not claim one."]
    return lead, ["The trigger gave no numbers; the lead is the merchant's own 7-day change."]


@_for("perf_dip")
def _perf_dip(c: Ctx) -> Draft:
    lead, notes = _perf_lead(c, "down")
    expired = (c.merchant.get("subscription") or {}).get("status") == "expired"
    support = _opt(move_fact(c.merchant, "up"), ctr_vs_peer(c.merchant, c.category), subscription_fact(c.merchant) if expired else None)
    return Draft(lead, support, notes)


@_for("perf_spike")
def _perf_spike(c: Ctx) -> Draft:
    lead, notes = _perf_lead(c, "up")
    driver = c.payload.get("likely_driver")
    support = _opt(Fact("trigger.driver", f"likely driver: {words(driver)}") if driver else None, offer_fact(c.merchant))
    return Draft(lead, support, notes)


@_for("seasonal_perf_dip")
def _seasonal_dip(c: Ctx) -> Draft:
    p = c.payload
    delta = _number(p.get("delta_pct"))
    lead = None
    if p.get("metric") and delta is not None:
        expected = " (an expected seasonal pattern)" if p.get("is_expected_seasonal") else ""
        lead = Fact("trigger.seasonal_dip", f"{p['metric']} {'down' if delta < 0 else 'up'} {pct(delta)} over {_span(p.get('window'))}{expected}")
    lead = lead or move_fact(c.merchant, "down")
    item = digest_of_kind(c.category, "seasonal")
    season_words = words(p.get("season_note") or "").split()
    return Draft(lead, _opt(beat_for(c.category, tokens=season_words), digest_action(item) if item else None))


@_for("milestone_reached")
def _milestone(c: Ctx) -> Draft:
    p = c.payload
    value, goal = _number(p.get("value_now")), _number(p.get("milestone_value"))
    if not c.placeholder and value is not None and goal is not None:
        metric = _METRIC_NAME.get(str(p.get("metric")), words(p.get("metric") or "count"))
        text = f"{num(value)} {metric}, {num(goal - value)} away from the {num(goal)} milestone" if goal > value else f"reached {num(goal)} {metric}"
        return Draft(Fact("trigger.milestone", text), _opt(*review_facts(c.merchant)[:1]))
    return Draft(None, _opt(*review_facts(c.merchant)[:1]), ["No milestone details were provided: do not state a milestone or a number for one."])


def _theme_fact(theme: dict) -> Fact:
    occurrences = _number(theme.get("occurrences_30d"))
    text = f"{num(occurrences)} reviews in 30 days mention {words(theme.get('theme'))}" if occurrences is not None else f"reviews mention {words(theme.get('theme'))}"
    if theme.get("trend"):
        text += f" ({theme['trend']} trend)"
    if theme.get("common_quote"):
        text += f'; typical comment: "{theme["common_quote"]}"'
    return Fact("trigger.review_theme", text)


@_for("review_theme_emerged")
def _review_theme(c: Ctx) -> Draft:
    strengths = [f for f in review_facts(c.merchant) if "praised" in f.text][:1]
    if not c.placeholder and c.payload.get("theme"):
        return Draft(_theme_fact(c.payload), strengths)
    complaints = sorted(
        (t for t in c.merchant.get("review_themes") or [] if t.get("sentiment") == "neg"),
        key=lambda t: -(_number(t.get("occurrences_30d")) or 0),
    )
    if complaints:
        return Draft(_theme_fact(complaints[0]), strengths, ["The trigger gave no theme; the lead is the merchant's most-mentioned complaint."])
    return Draft(None, strengths, ["No review theme is available: do not claim one."])


@_for("competitor_opened")
def _competitor(c: Ctx) -> Draft:
    p = c.payload
    strengths = [f for f in review_facts(c.merchant) if "praised" in f.text][:1]
    if not c.placeholder and p.get("competitor_name"):
        text = f"{p['competitor_name']} opened"
        if _number(p.get("distance_km")) is not None:
            text += f" {num(p['distance_km'])} km away"
        if when(p.get("opened_date")):
            text += f" on {when(p['opened_date'])}"
        if p.get("their_offer"):
            text += f", offering {p['their_offer']}"
        return Draft(Fact("trigger.competitor", text), _opt(offer_fact(c.merchant), *strengths),
                     ["Mention the competitor only as stated, and never disparage them."])
    return Draft(None, _opt(offer_fact(c.merchant), *strengths),
                 ["No competitor details were provided: do not name a competitor or invent a distance or an offer."])


# --------------------------------------------------------------------------- account and engagement

@_for("renewal_due")
def _renewal(c: Ctx) -> Draft:
    p = c.payload
    days = _number(p.get("days_remaining"))
    if not c.placeholder and days is not None:
        plan = p.get("plan") or (c.merchant.get("subscription") or {}).get("plan") or "plan"
        text = f"{plan} plan renews in {num(days)} days"
        if _number(p.get("renewal_amount")) is not None:
            text += f" ({money(p['renewal_amount'])})"
        lead = Fact("trigger.renewal", text)
    else:
        lead = subscription_fact(c.merchant)
    return Draft(lead, _opt(perf_snapshot(c.merchant), move_fact(c.merchant, "up")))


@_for("winback_eligible")
def _winback(c: Ctx) -> Draft:
    p = c.payload
    bits = []
    if _number(p.get("days_since_expiry")) is not None:
        bits.append(f"plan expired {num(p['days_since_expiry'])} days ago")
    dip = _number(p.get("perf_dip_pct"))
    if dip is not None:
        bits.append(f"performance {'down' if dip < 0 else 'up'} {pct(dip)} since")
    if _number(p.get("lapsed_customers_added_since_expiry")) is not None:
        bits.append(f"{num(p['lapsed_customers_added_since_expiry'])} more customers have lapsed")
    lead = Fact("trigger.winback", "; ".join(bits)) if bits else subscription_fact(c.merchant)
    return Draft(lead, _opt(perf_snapshot(c.merchant)))


@_for("dormant_with_vera")
def _dormant(c: Ctx) -> Draft:
    p = c.payload
    days = _number(p.get("days_since_last_merchant_message"))
    if not c.placeholder and days is not None:
        text = f"{num(days)} days since the last conversation"
        if p.get("last_topic"):
            text += f" (last topic: {words(p['last_topic'])})"
        lead = Fact("trigger.dormant", text)
    else:
        lead = next((f for f in signal_facts(c.merchant) if f.key == "signal.dormant"), None)
    return Draft(lead, notes=["Re-engage gently: no guilt and no 'we haven't heard from you'. Lead with something useful."])


_ASK_NOUN = {"dentists": "treatment", "salons": "service", "restaurants": "dish", "gyms": "class", "pharmacies": "product category"}


@_for("curious_ask_due")
def _curious(c: Ctx) -> Draft:
    template = c.payload.get("ask_template")
    if template and not c.placeholder:
        lead = Fact("ask.topic", f"question to ask: {_ASK_PHRASE.get(str(template), words(template))}")
    else:
        noun = _ASK_NOUN.get(c.category.get("slug"), "service")
        lead = Fact("ask.topic", f"question to ask: which {noun} customers are asking for most this week")
    return Draft(lead, notes=["Promise one concrete thing you will make from their answer (a post, an offer, a reply template)."])


@_for("active_planning_intent")
def _planning(c: Ctx) -> Draft:
    p = c.payload
    topic = words(p.get("intent_topic") or "")
    parts = ([f"merchant is planning: {topic}"] if topic else []) + ([f'their last message: "{p["merchant_last_message"]}"'] if p.get("merchant_last_message") else [])
    notes = ["The merchant already agreed. Deliver the draft now. Do not invent addresses, buildings, counts or prices that are not in the data, and present any price as a proposal."]
    support = _opt(offer_fact(c.merchant), _suggest(c, match=topic.split(), require_match=True))
    return Draft(Fact("trigger.intent", "; ".join(parts)) if parts else None, support, notes)


@_for("gbp_unverified")
def _gbp(c: Ctx) -> Draft:
    p = c.payload
    bits = ["the Google business profile is not verified"]
    if _number(p.get("estimated_uplift_pct")) is not None:
        bits.append(f"verifying is estimated to lift performance by about {pct(p['estimated_uplift_pct'])}")
    if p.get("verification_path"):
        bits.append(f"verification path: {words(p['verification_path'])}")
    return Draft(Fact("trigger.gbp", "; ".join(bits)), _opt(perf_snapshot(c.merchant)))


@_for("festival_upcoming")
def _festival(c: Ctx) -> Draft:
    p = c.payload
    name = p.get("festival")
    if c.placeholder or not name:
        return Draft(None, notes=["No festival is named in the trigger: do not name one or invent a date."])
    days = _number(p.get("days_until"))
    text = name + (f" on {when(p['date'])}" if when(p.get("date")) else "") + (f" ({num(days)} days away)" if days is not None else "")
    notes = []
    if days is not None and days > 45:
        notes.append(f"{name} is {num(days)} days away: use plan-ahead framing and no false urgency.")
    relevance = p.get("category_relevance")
    if relevance and c.category.get("slug") not in relevance:
        notes.append(f"{name} is not marked relevant for this category: keep the link modest.")
    month = _month_of(p.get("date"))
    return Draft(Fact("trigger.festival", text), _opt(beat_for(c.category, months=[month] if month else [], tokens=[name])), notes)


@_for("ipl_match_today")
def _ipl(c: Ctx) -> Draft:
    p = c.payload
    if not p.get("match"):
        return Draft(notes=["No match details were provided: do not name a match."])
    text = p["match"] + (f" at {p['venue']}" if p.get("venue") else "") + (f", {when(p['match_time_iso'])}" if when(p.get("match_time_iso")) else "")
    item = search_digest(c.category, "ipl")
    notes = []
    if p.get("is_weeknight") is False:
        notes.append("This match is NOT on a weeknight. If the digest says such matches cut dine-in covers, say so honestly and lean on delivery or watch-party combos instead of promising a packed house.")
    elif p.get("is_weeknight") is True:
        notes.append("This is a weeknight match, which the digest links to higher covers: a match-night offer is justified.")
    summary = Fact("ipl.digest", item["summary"]) if item and item.get("summary") else None
    support = _opt(summary, digest_action(item) if item else None, offer_fact(c.merchant),
                   _suggest(c, match=["match", "night", "combo"], require_match=True))
    return Draft(Fact("trigger.ipl", text), support, notes)


# --------------------------------------------------------------------------- customer-facing

@_for("recall_due")
def _recall(c: Ctx) -> Draft:
    p = c.payload
    service = words(p["service_due"]) if p.get("service_due") else None
    if service or when(p.get("due_date")):
        text = (service or "check-up") + " due" + (f" {when(p['due_date'])}" if when(p.get("due_date")) else "")
        if when(p.get("last_service_date")):
            text += f"; last done {when(p['last_service_date'])}"
        lead, notes = Fact("trigger.recall", text), []
    else:
        lead, notes = None, ["No service or due date was provided: invite them back for their regular visit without naming a service, a date or a price."]
    offer = _matching_offer(c.merchant, service or "") or offer_fact(c.merchant)
    return Draft(lead, _opt(_slots_fact(p.get("available_slots")), offer, *customer_facts(c.customer)[:2]), notes)


@_for("appointment_tomorrow")
def _appointment(c: Ctx) -> Draft:
    details = _payload_facts(c.payload)
    notes = [] if details else ["The trigger does not say the appointment time or service: do not invent either. Ask them to confirm tomorrow's appointment or to reschedule."]
    support = details[1:] + _opt(offer_fact(c.merchant), *customer_facts(c.customer)[:2])
    return Draft(details[0] if details else None, support, notes)


@_for("chronic_refill_due")
def _refill(c: Ctx) -> Draft:
    p = c.payload
    molecules = [str(m) for m in p.get("molecule_list") or []]
    lead = None
    if molecules:
        text = ", ".join(molecules)
        if when(p.get("last_refill")):
            text += f": last refilled {when(p['last_refill'])}"
        if when(p.get("stock_runs_out_iso")):
            text += f", supply runs out {when(p['stock_runs_out_iso'])}"
        lead = Fact("trigger.refill", text)
    senior = bool(((c.customer or {}).get("identity") or {}).get("senior_citizen"))
    support = _opt(
        Fact("refill.delivery", "a delivery address is saved, so home delivery is possible") if p.get("delivery_address_saved") else None,
        offer_fact(c.merchant),
        _suggest(c, match=["delivery"], require_match=True),
        _suggest(c, audience="senior", match=["senior", "citizen"], require_match=True) if senior else None,
    )
    notes = [] if molecules else ["The medicines are not listed: do not name any."]
    return Draft(lead, support, notes)


@_for("trial_followup")
def _trial(c: Ctx) -> Draft:
    p = c.payload
    lead = Fact("trigger.trial", f"trial session on {when(p['trial_date'])}") if when(p.get("trial_date")) else None
    return Draft(lead, _opt(_slots_fact(p.get("next_session_options")), offer_fact(c.merchant), *customer_facts(c.customer)[:2]))


@_for("customer_lapsed_soft", "customer_lapsed_hard")
def _lapsed(c: Ctx) -> Draft:
    p = c.payload
    bits = []
    if _number(p.get("days_since_last_visit")) is not None:
        bits.append(f"{num(p['days_since_last_visit'])} days since the last visit")
    if p.get("previous_focus"):
        bits.append(f"earlier goal: {words(p['previous_focus'])}")
    if _number(p.get("previous_membership_months")):
        bits.append(f"was a member for {num(p['previous_membership_months'])} months")
    trial = _suggest(c, match=["trial"], require_match=True)
    return Draft(Fact("trigger.lapsed", "; ".join(bits)) if bits else None, _opt(offer_fact(c.merchant), trial, *customer_facts(c.customer)[:2]))


@_for("wedding_package_followup")
def _wedding(c: Ctx) -> Draft:
    p = c.payload
    bits = []
    if when(p.get("wedding_date")):
        days = _number(p.get("days_to_wedding"))
        bits.append(f"wedding on {when(p['wedding_date'])}" + (f" ({num(days)} days away)" if days is not None else ""))
    if when(p.get("trial_completed")):
        bits.append(f"bridal trial done on {when(p['trial_completed'])}")
    if p.get("next_step_window_open"):
        bits.append(f"next step now open: {words(p['next_step_window_open'])}")
    support = _opt(offer_fact(c.merchant), _suggest(c, match=["bridal"], require_match=True), *customer_facts(c.customer)[:1])
    notes = ["Only claim programs and prices that appear in the merchant's offers; otherwise offer them as options to set up."]
    return Draft(Fact("trigger.wedding", "; ".join(bits)) if bits else None, support, notes)
