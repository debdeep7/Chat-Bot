"""Assemble a Brief: everything the composer needs to write one message.

A Brief is the result of every deterministic decision made before a language model is involved:
who the message is for and in which language, what it leads with (the "why now"), which facts
back the lead up, what call-to-action it ends on, and which traps to avoid. The model's job is
then only to phrase it well; it cannot change what the message is about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .facts import Fact, aggregate_facts, customer_facts, offer_fact, peer_comparisons, perf_snapshot, review_facts
from .leads import Ctx, build_draft
from .router import GENERIC_CUSTOMER, GENERIC_MERCHANT, Strategy, fits_category, route
from .store import ContextStore

MAX_SUPPORT = 3


@dataclass(frozen=True)
class Brief:
    trigger_id: str
    kind: str                    # canonical kind, or generic_merchant / generic_customer
    raw_kind: str                # the kind exactly as it arrived
    strategy: Strategy
    merchant_id: str
    customer_id: str | None      # set only for customer-facing messages
    category_slug: str
    send_as: str                 # "vera" or "merchant_on_behalf"
    salutation: str              # who the message is addressed to ("Dr. Meera", "Priya", "Sneha")
    about: str | None            # who it is about when that differs (a child, "Aanya")
    language: str                # en | hi | hi-en | ta-en | te-en | kn-en
    cta: str
    lead: Fact | None            # the "why now"; None only when nothing at all could be found
    support: tuple[Fact, ...]
    notes: tuple[str, ...]       # warnings for the composer
    urgency: int
    suppression_key: str
    degraded: bool               # the lead did not come from the trigger itself: stay modest
    placeholder: bool            # the trigger payload held no real details
    prior_bodies: tuple[str, ...]  # what Vera already told this merchant (for anti-repetition)
    trigger: dict
    merchant: dict
    category: dict
    customer: dict | None

    @property
    def facts(self) -> tuple[Fact, ...]:
        return ((self.lead,) if self.lead else ()) + self.support


# --------------------------------------------------------------------------- who and how

def merchant_salutation(merchant: dict, category_slug: str | None) -> str:
    identity = merchant.get("identity") or {}
    owner = str(identity.get("owner_first_name") or "").strip()
    if not owner:
        return f"{identity.get('name') or 'there'} team"
    # Some owner names already carry the title ("Dr. Sameer"); never produce "Dr. Dr.".
    if category_slug != "dentists" or re.match(r"dr\.?\s", owner, re.I):
        return owner
    return f"Dr. {owner}"


def customer_address(customer: dict | None) -> tuple[str, str | None]:
    """(who to address, who it is about when different). A child's record is addressed to the parent."""
    if not customer:
        return "", None
    name = str((customer.get("identity") or {}).get("name") or "").strip()
    child_of = re.fullmatch(r"(.+?)\s*\(\s*parent:\s*(.+?)\s*\)", name, re.I)
    if child_of:
        return child_of.group(2), child_of.group(1)
    if name.startswith("("):  # e.g. "(walk-in, no profile)"
        return "", None
    return name, None


def merchant_language(merchant: dict) -> str:
    languages = [str(x).lower() for x in (merchant.get("identity") or {}).get("languages") or []]
    return "hi-en" if "hi" in languages else "en"


def customer_language(customer: dict | None, fallback: str) -> str:
    if not customer:
        return fallback
    pref = str((customer.get("identity") or {}).get("language_pref") or "").strip().lower()
    if pref in ("en", "english"):
        return "en"
    if pref == "hi":
        return "hi"
    if pref.startswith("hi"):
        return "hi-en"
    if "-en" in pref:  # "ta-en mix"
        return pref.split()[0]
    return fallback


# --------------------------------------------------------------------------- notes and support

def _merchant_notes(merchant: dict) -> list[str]:
    notes = []
    if merchant.get("conversation_history"):
        notes.append("There is earlier conversation with this merchant: do not re-introduce Vera and do not repeat earlier messages.")
    if {"engaged_in_last_24h", "engaged_in_last_48h"} & {str(s) for s in merchant.get("signals") or []}:
        notes.append("The merchant engaged very recently: skip pleasantries and get to the point.")
    calls = (merchant.get("performance") or {}).get("calls")
    if isinstance(calls, (int, float)) and calls < 8:
        notes.append(f"Only {calls} calls in 30 days: do not work out new percentages from the call counts.")
    return notes


def _customer_notes(customer: dict) -> list[str]:
    notes = []
    channel = str((customer.get("preferences") or {}).get("channel") or "")
    if channel.startswith("whatsapp_via_"):
        notes.append(f"This number belongs to the customer's {channel.removeprefix('whatsapp_via_')}: address them respectfully and refer to the customer in the third person.")
    if (customer.get("identity") or {}).get("senior_citizen"):
        notes.append("Senior customer: use a formal, respectful tone.")
    return notes


# Kinds about a problem or a threat are better opened with a gap against peers; the rest with a strength.
_PREFER_GAP = frozenset({
    "perf_dip", "competitor_opened", "dormant_with_vera", "renewal_due", "winback_eligible",
    "review_theme_emerged", "gbp_unverified",
})
# Generic "your offer" support is noise next to a research finding or a rule change.
_NO_OFFER_SUPPORT = frozenset({"research_digest", "regulation_change", "cde_opportunity"})


def _fallback_lead(strategy: Strategy, merchant: dict, category: dict, customer: dict | None) -> Fact | None:
    """A lead for when the trigger itself gave nothing usable: the merchant's own clearest fact."""
    if strategy.audience == "customer":
        return next(iter(customer_facts(customer)), None)
    want_gap = strategy.kind in _PREFER_GAP
    ranked = sorted(peer_comparisons(merchant, category), key=lambda item: item[0] if want_gap else -item[0])
    for gap, fact in ranked:  # a clear difference in the wanted direction
        if abs(gap) >= 0.05 and (gap < 0) == want_gap:
            return fact
    for gap, fact in ranked:  # otherwise any clear difference
        if abs(gap) >= 0.05:
            return fact
    return perf_snapshot(merchant)


def _pick_support(
    draft_support: list[Fact], lead: Fact | None, strategy: Strategy, merchant: dict, category: dict, customer: dict | None
) -> tuple[Fact, ...]:
    """The kind-specific facts first, then generic ones, de-duplicated and capped."""
    if strategy.audience == "customer":
        extras = [offer_fact(merchant), *customer_facts(customer)]
    else:
        # Only peer comparisons with a clear gap are worth quoting; "in line with peers" says nothing.
        clear_gaps = [fact for gap, fact in peer_comparisons(merchant, category) if abs(gap) >= 0.05]
        extras = [
            None if strategy.kind in _NO_OFFER_SUPPORT else offer_fact(merchant),
            *clear_gaps, *aggregate_facts(merchant)[:1], *review_facts(merchant)[:1], perf_snapshot(merchant),
        ]
    seen = {lead.key} if lead else set()
    chosen: list[Fact] = []
    for fact in [*draft_support, *extras]:
        if fact is None or fact.key in seen:
            continue
        seen.add(fact.key)
        chosen.append(fact)
        if len(chosen) == MAX_SUPPORT:
            break
    return tuple(chosen)


# --------------------------------------------------------------------------- the entry point

def build_brief(trigger: dict, store: ContextStore, trigger_id: str | None = None) -> tuple[Brief | None, str | None]:
    """Turn one trigger plus the stored contexts into a Brief.

    Returns (brief, None), or (None, reason) when the message cannot be written at all.
    """
    merchant = store.get("merchant", trigger.get("merchant_id")) if trigger.get("merchant_id") else None
    if merchant is None:
        return None, "merchant_context_missing"
    category = store.get("category", merchant.get("category_slug"))
    if category is None:
        return None, "category_context_missing"

    slug = str(category.get("slug") or merchant.get("category_slug"))
    strategy = route(trigger)
    notes: list[str] = []
    placeholder = bool((trigger.get("payload") or {}).get("placeholder"))

    mismatch = not fits_category(strategy.kind, slug)
    if mismatch:
        notes.append(f"A '{strategy.kind}' trigger does not fit a {slug} merchant: ignore its details and write an honest, general check-in.")
        strategy = GENERIC_CUSTOMER if strategy.audience == "customer" else GENERIC_MERCHANT
        trigger = {**trigger, "payload": {}}

    customer = customer_id = None
    if strategy.audience == "customer":
        customer_id = trigger.get("customer_id")
        if not customer_id:
            return None, "customer_id_missing"
        customer = store.get("customer", customer_id)
        if customer is None:
            notes.append("The customer's details were not provided: greet plainly and do not invent anything about them.")

    draft = build_draft(strategy.kind, Ctx(trigger, merchant, category, customer))
    notes += draft.notes
    # What we know about the merchant only matters when Vera is the one writing to them.
    notes += _customer_notes(customer) if customer else _merchant_notes(merchant) if strategy.audience == "merchant" else []

    lead = draft.lead or _fallback_lead(strategy, merchant, category, customer)
    if draft.lead is None and lead is not None:
        notes.append("The trigger gave no usable detail, so the lead comes from the merchant's own profile: connect the message to the trigger kind only in general terms.")
    # Degraded = the lead is not backed by the trigger's own details, so the composer must stay modest.
    degraded = (
        draft.lead is None or draft.lead.key == "trigger.kind" or mismatch
        or (strategy.audience == "customer" and customer is None)
    )

    cta = strategy.cta
    if cta == "multi_choice_slot" and not any(f.key == "slots" for f in draft.support):
        cta = "binary_yes_no"

    if strategy.audience == "customer":
        salutation, about = customer_address(customer)
        language = customer_language(customer, merchant_language(merchant))
    else:
        salutation, about, language = merchant_salutation(merchant, slug), None, merchant_language(merchant)

    urgency = trigger.get("urgency")
    trigger_id = str(trigger.get("id") or trigger_id or "")
    merchant_id = str(trigger["merchant_id"])
    brief = Brief(
        trigger_id=trigger_id,
        kind=strategy.kind,
        raw_kind=str(trigger.get("kind") or ""),
        strategy=strategy,
        merchant_id=merchant_id,
        customer_id=str(customer_id) if customer_id else None,
        category_slug=slug,
        send_as=strategy.send_as,
        salutation=salutation,
        about=about,
        language=language,
        cta=cta,
        lead=lead,
        support=_pick_support(draft.support, lead, strategy, merchant, category, customer),
        notes=tuple(notes),
        urgency=int(urgency) if isinstance(urgency, (int, float)) and not isinstance(urgency, bool) else 1,
        suppression_key=str(trigger.get("suppression_key") or f"{strategy.kind}:{merchant_id}:{customer_id or ''}"),
        degraded=degraded,
        placeholder=placeholder,
        # Vera's earlier messages went to the merchant, so they only matter for a message to the merchant.
        prior_bodies=tuple(
            str(turn["body"]) for turn in merchant.get("conversation_history") or []
            if strategy.audience == "merchant" and turn.get("from") == "vera" and turn.get("body")
        ),
        trigger=trigger,
        merchant=merchant,
        category=category,
        customer=customer,
    )
    return brief, None
