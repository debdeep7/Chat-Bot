"""Trigger router: trigger kind -> message strategy.

Every trigger kind calls for a different kind of message (a recall reminder is nothing like a
regulation alert), so the kind picks a `Strategy`: who the message is for, which engagement
levers fit, what the single call-to-action should look like, and what the message must achieve.
The table is data, not branching code: supporting a new kind means adding one row.

Unknown kinds never crash the bot. They are matched by keyword ("recall" in the name ->
recall_due) and, failing that, fall back to a generic strategy chosen by the trigger's scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    kind: str                # canonical trigger kind
    audience: str            # "merchant" (Vera writes) or "customer" (written for the merchant)
    cta: str                 # open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none
    levers: tuple[str, ...]  # engagement levers from the challenge brief that fit this kind
    goal: str                # what the message must achieve, as an instruction to the composer

    @property
    def send_as(self) -> str:
        return "merchant_on_behalf" if self.audience == "customer" else "vera"


def _s(kind: str, audience: str, cta: str, levers: str, goal: str) -> Strategy:
    return Strategy(kind, audience, cta, tuple(levers.split()), goal)


_ALL = (
    # ---- merchant-facing: knowledge and compliance
    _s("research_digest", "merchant", "binary_yes_no", "specificity curiosity reciprocity",
       "Share the one digest finding that matters to this merchant: headline number, source, and why it fits their patients or customers. Offer to pull or draft something."),
    _s("regulation_change", "merchant", "binary_yes_no", "loss_aversion specificity effort_externalization",
       "State the rule change, its deadline and what it means for this merchant's setup. Offer a ready checklist or draft."),
    _s("cde_opportunity", "merchant", "binary_yes_no", "specificity reciprocity",
       "Name the event, its date, credits and fee. Offer to register or remind them."),
    _s("supply_alert", "merchant", "binary_yes_no", "loss_aversion specificity effort_externalization",
       "Flag the recalled product with molecule, batch numbers and manufacturer. Offer to draft the customer notice and the return steps."),
    _s("category_seasonal", "merchant", "binary_yes_no", "specificity loss_aversion",
       "Explain the seasonal demand shift with its numbers and give one concrete shelf, menu or offer action."),
    # ---- merchant-facing: performance
    _s("perf_dip", "merchant", "binary_yes_no", "loss_aversion specificity effort_externalization",
       "Name the metric, the size of the drop and the window. Give one likely cause from the data and one concrete fix. No alarmism."),
    _s("perf_spike", "merchant", "binary_yes_no", "specificity reciprocity",
       "Celebrate the rise with its number, name the likely driver if known, and propose how to build on it."),
    _s("seasonal_perf_dip", "merchant", "binary_yes_no", "reassurance specificity",
       "Reassure: the dip is a normal seasonal pattern. Give the number, reframe it, and suggest a cheaper, smarter action than panic-spending."),
    _s("milestone_reached", "merchant", "binary_yes_no", "specificity curiosity",
       "Name the milestone and how close it is. Propose one small action to cross it."),
    _s("review_theme_emerged", "merchant", "binary_yes_no", "specificity effort_externalization",
       "Name the review theme, how often it appears and a typical quote. Propose a fix plus a drafted reply."),
    _s("competitor_opened", "merchant", "open_ended", "loss_aversion social_proof specificity",
       "Mention the new competitor factually (name, distance, their offer). Never disparage them. Point to this merchant's own strengths and propose a response."),
    # ---- merchant-facing: account and engagement
    _s("renewal_due", "merchant", "binary_yes_no", "loss_aversion specificity",
       "Say how many days the plan has left and what the merchant keeps by renewing, using their own numbers. Make renewing one reply."),
    _s("winback_eligible", "merchant", "binary_yes_no", "loss_aversion specificity",
       "Show what changed since the plan lapsed (dip, lapsed customers) and offer to relaunch in one step."),
    _s("dormant_with_vera", "merchant", "open_ended", "curiosity reciprocity",
       "Re-engage gently with no guilt. Lead with one useful fact and a light question."),
    _s("curious_ask_due", "merchant", "open_ended", "asking_the_merchant reciprocity",
       "Ask one low-stakes question about their business and promise a concrete output made from the answer."),
    _s("active_planning_intent", "merchant", "binary_yes_no", "effort_externalization specificity",
       "The merchant already said yes. Deliver a concrete draft now and ask for a go-ahead. Do not ask another qualifying question."),
    _s("gbp_unverified", "merchant", "binary_yes_no", "loss_aversion specificity",
       "Explain the profile is unverified, the estimated uplift from verifying, and the quickest verification path. Offer to walk them through it."),
    _s("festival_upcoming", "merchant", "binary_yes_no", "specificity effort_externalization",
       "Tie the festival date to this category's demand. Propose one offer or creative that is ready to go."),
    _s("ipl_match_today", "merchant", "binary_yes_no", "specificity loss_aversion",
       "Use the match, the time and what the category data says about this kind of match day to recommend the right promotion."),
    # ---- customer-facing (written on the merchant's behalf)
    _s("recall_due", "customer", "multi_choice_slot", "specificity warmth",
       "Remind the customer their service is due, name the last visit and offer the available slots. Make replying easy."),
    _s("appointment_tomorrow", "customer", "binary_confirm_cancel", "specificity warmth",
       "Remind the customer of tomorrow's appointment and ask them to confirm or reschedule."),
    _s("chronic_refill_due", "customer", "binary_confirm_cancel", "specificity warmth",
       "Tell the customer their regular medicines run out soon, name them, and offer to prepare the refill and delivery."),
    _s("trial_followup", "customer", "multi_choice_slot", "specificity warmth",
       "Follow up on the trial, mention what happened and offer the next session options."),
    _s("customer_lapsed_soft", "customer", "binary_yes_no", "warmth specificity",
       "A friendly, no-pressure nudge to come back. Mention the last visit and one reason to return."),
    _s("customer_lapsed_hard", "customer", "binary_yes_no", "warmth specificity",
       "A warm, no-shame invitation back. Reference their earlier goal and offer a low-commitment way to return."),
    _s("wedding_package_followup", "customer", "binary_yes_no", "specificity warmth",
       "Follow up on the bridal trial: the wedding date, the next step that is now open, and one simple way to book it."),
)

STRATEGIES: dict[str, Strategy] = {s.kind: s for s in _ALL}

GENERIC_MERCHANT = _s(
    "generic_merchant", "merchant", "open_ended", "specificity curiosity",
    "Lead with the single most specific fact in the trigger payload; if it has none, use one of the merchant's own numbers against its peers. Ask one simple question.",
)
GENERIC_CUSTOMER = _s(
    "generic_customer", "customer", "binary_yes_no", "warmth specificity",
    "A short, friendly message from the merchant that refers to the customer's last visit and asks one simple question.",
)

# A few kinds only make sense for some categories. A chronic-refill reminder from a dental
# clinic is a data glitch, so we say less rather than invent a story for it.
KIND_ONLY_FOR: dict[str, frozenset[str]] = {
    "chronic_refill_due": frozenset({"pharmacies"}),
    "supply_alert": frozenset({"pharmacies"}),
    "ipl_match_today": frozenset({"restaurants"}),
    "wedding_package_followup": frozenset({"salons"}),
}

_ALIASES = {
    "research_digest_release": "research_digest",
    "digest": "research_digest",
    "compliance": "regulation_change",
    "recall": "recall_due",
    "refill_due": "chronic_refill_due",
    "winback": "winback_eligible",
}

# Keyword fallback for unseen kinds, checked in order (more specific words first).
_INFER = (
    ("supply", "supply_alert"), ("refill", "chronic_refill_due"), ("appointment", "appointment_tomorrow"),
    ("wedding", "wedding_package_followup"), ("bridal", "wedding_package_followup"),
    ("trial", "trial_followup"), ("hard", "customer_lapsed_hard"), ("lapsed", "customer_lapsed_soft"),
    ("winback", "winback_eligible"), ("recall", "recall_due"), ("digest", "research_digest"),
    ("research", "research_digest"), ("regulat", "regulation_change"), ("compliance", "regulation_change"),
    ("cde", "cde_opportunity"), ("webinar", "cde_opportunity"), ("seasonal_perf", "seasonal_perf_dip"),
    ("seasonal", "category_seasonal"), ("dip", "perf_dip"), ("drop", "perf_dip"), ("spike", "perf_spike"),
    ("surge", "perf_spike"), ("renew", "renewal_due"), ("milestone", "milestone_reached"),
    ("review", "review_theme_emerged"), ("competitor", "competitor_opened"), ("dormant", "dormant_with_vera"),
    ("curious", "curious_ask_due"), ("planning", "active_planning_intent"), ("unverified", "gbp_unverified"),
    ("gbp", "gbp_unverified"), ("festival", "festival_upcoming"), ("ipl", "ipl_match_today"),
)


def canonical_kind(raw: object) -> str | None:
    """Map a trigger kind to one of ours, or None when we have no idea what it is."""
    kind = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower())
    if kind in STRATEGIES:
        return kind
    if kind in _ALIASES:
        return _ALIASES[kind]
    for needle, target in _INFER:
        # match at the start of a word, so "ipl" finds ipl_match_today but not "triple"
        if re.search(r"(?:^|_)" + re.escape(needle), kind):
            return target
    return None


def route(trigger: dict) -> Strategy:
    """Pick the strategy for a trigger; never raises."""
    kind = canonical_kind(trigger.get("kind"))
    if kind:
        return STRATEGIES[kind]
    customer_scoped = trigger.get("scope") == "customer" or bool(trigger.get("customer_id"))
    return GENERIC_CUSTOMER if customer_scoped else GENERIC_MERCHANT


def fits_category(kind: str, category_slug: str | None) -> bool:
    allowed = KIND_ONLY_FOR.get(kind)
    return allowed is None or category_slug in allowed
