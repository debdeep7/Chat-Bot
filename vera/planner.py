"""Tick planner: from the triggers the judge calls active, choose what to act on right now.

The judge's `available_triggers` is a hint, not an order, and "restraint is rewarded, spam is
penalized". The policy is deliberately small and explainable:

  1. never act twice on the same trigger or suppression key, and stop retrying a trigger that
     keeps producing no sendable message
  2. never message someone who opted out or cannot be reached
  3. one merchant-facing message per merchant per tick, and one per customer per tick
  4. after messaging a merchant, wait MERCHANT_COOLDOWN before the next one (urgent triggers skip the wait)
  5. best first: higher urgency, then triggers with real details, then triggers that are not degraded
  6. never more than the judge's cap of 20 actions per tick

Triggers are not dropped for being "expired": dataset timestamps and the judge's clock do not
line up, so `available_triggers` is trusted instead. A skipped trigger can come back on a later
tick, because nothing is marked as sent until the caller says a message really went out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .ledger import Ledger
from .signals import Brief, build_brief
from .store import ContextStore

log = logging.getLogger("vera.planner")

MAX_ACTIONS = 20
MERCHANT_COOLDOWN = timedelta(minutes=15)
URGENT = 4  # urgency at or above this skips the cooldown
MAX_SNOOZE = timedelta(days=8)  # the longest pause the reply handler sets is 7 days

# These customer messages are reminders, so a customer who turned reminders off must not get them.
REMINDER_KINDS = frozenset({
    "recall_due", "appointment_tomorrow", "chronic_refill_due", "trial_followup", "wedding_package_followup",
})


@dataclass(frozen=True)
class Skip:
    trigger_id: str
    reason: str


@dataclass
class TickPlan:
    briefs: list[Brief] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)


def parse_time(value: Any) -> datetime | None:
    """ISO timestamp -> aware datetime (naive input is taken as UTC); junk -> None."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def priority(brief: Brief) -> int:
    """Higher goes first: urgency dominates, real details beat placeholders, degraded briefs lose ties."""
    return brief.urgency * 10 + (0 if brief.placeholder else 5) - (3 if brief.degraded else 0)


def blocked_reason(brief: Brief, ledger: Ledger, now: datetime | None = None) -> str | None:
    """Why this message must not be sent to this person right now, or None."""
    if ledger.merchant_block(brief.merchant_id):
        return "merchant_opted_out"
    if brief.customer_id is None:
        # A merchant who was busy, auto-replied or said "later" is left alone until the pause ends. A pause
        # far longer than any we set means the judge's clock was reset (a new run), so it no longer applies.
        until = ledger.snoozed_until(brief.merchant_id)
        if now is not None and until is not None and timedelta(0) < until - now <= MAX_SNOOZE:
            return "merchant_snoozed"
        return None
    if ledger.customer_block(brief.customer_id):
        return "customer_opted_out"
    prefs = (brief.customer or {}).get("preferences") or {}
    if prefs.get("channel") == "none_recorded":
        return "customer_unreachable"
    # Consent scopes are deliberately not enforced: nearly every customer lists only
    # "promotional_offers", so a strict check would silence all service reminders.
    if brief.kind in REMINDER_KINDS and prefs.get("reminder_opt_in") is False:
        return "customer_declined_reminders"
    return None


def plan_tick(
    store: ContextStore,
    ledger: Ledger,
    trigger_ids: Iterable[str],
    now: datetime | None = None,
    *,
    max_actions: int = MAX_ACTIONS,
    cooldown: timedelta = MERCHANT_COOLDOWN,
) -> TickPlan:
    plan = TickPlan()
    candidates: list[Brief] = []
    seen: set[str] = set()

    for trigger_id in trigger_ids:
        if trigger_id in seen:
            continue
        seen.add(trigger_id)
        trigger = store.get("trigger", trigger_id)
        if trigger is None:
            plan.skipped.append(Skip(trigger_id, "trigger_not_pushed"))
            continue
        if ledger.already_sent(trigger_id, trigger.get("merchant_id"), trigger.get("customer_id"), trigger.get("suppression_key")):
            plan.skipped.append(Skip(trigger_id, "already_sent"))
            continue
        if ledger.gave_up(trigger_id):
            plan.skipped.append(Skip(trigger_id, "gave_up"))
            continue
        try:
            brief, why = build_brief(trigger, store, trigger_id)
        except Exception:  # a malformed record must cost one trigger, never the whole tick
            log.exception("could not build a brief for %s", trigger_id)
            plan.skipped.append(Skip(trigger_id, "brief_error"))
            continue
        if brief is None:
            plan.skipped.append(Skip(trigger_id, why or "cannot_build_brief"))
            continue
        blocked = blocked_reason(brief, ledger, now)
        if blocked:
            plan.skipped.append(Skip(trigger_id, blocked))
            continue
        candidates.append(brief)

    candidates.sort(key=lambda b: (-priority(b), b.trigger_id))  # deterministic order

    merchants_used: set[str] = set()
    customers_used: set[str] = set()
    for brief in candidates:
        if len(plan.briefs) >= max_actions:
            plan.skipped.append(Skip(brief.trigger_id, "tick_cap_reached"))
        elif brief.customer_id is not None:
            if brief.customer_id in customers_used:
                plan.skipped.append(Skip(brief.trigger_id, "customer_already_messaged_this_tick"))
                continue
            customers_used.add(brief.customer_id)
            plan.briefs.append(brief)
        elif brief.merchant_id in merchants_used:
            plan.skipped.append(Skip(brief.trigger_id, "merchant_already_messaged_this_tick"))
        elif _cooling_down(ledger, brief, now, cooldown):
            plan.skipped.append(Skip(brief.trigger_id, "merchant_cooldown"))
        else:
            merchants_used.add(brief.merchant_id)
            plan.briefs.append(brief)
    return plan


def _cooling_down(ledger: Ledger, brief: Brief, now: datetime | None, cooldown: timedelta) -> bool:
    last = ledger.last_merchant_send(brief.merchant_id)
    if now is None or last is None or brief.urgency >= URGENT:
        return False
    # A negative gap means the clock was reset (a new test run), not that the merchant is cooling down.
    return timedelta(0) <= now - last < cooldown
