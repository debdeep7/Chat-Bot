"""The reply state machine: one incoming message in, one action (send / wait / end) out.

A reply can arrive on any conversation id, including ones the bot never opened (the judge replays
scenarios on fresh ids), so the first thing done is to find or adopt the conversation and rebuild
its context from the stored merchant. Then:

    classify the message (intents.py)  ->  update what we know about this sender  ->  decide

The decisions follow what the judge rewards:
  * a canned auto-reply is flagged once for the owner, then waited out (a day), then the
    conversation is closed; the count is kept per merchant because the judge changes the
    conversation id between repeats
  * "let's do it" switches from asking to doing: the reply starts the work and asks for one CONFIRM
  * an opt-out or hostile message ends the conversation and stops all outreach to that person
  * "not now" waits, "no thanks" ends, an unrelated request is declined politely and redirected
  * a message in Hindi or English is answered in that language, whatever the profile says

The words come from templates (replytext.py), so the machine works with no model. When a model is
configured it drafts the replies that benefit from specifics (the action, an answer, a redirect);
its draft is checked like every other message and replaced by the template if it fails.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from . import intents, replytext
from .composer import Composer
from .conversations import Conversation, ConversationBook
from .intents import Signal
from .ledger import Ledger
from .models import ReplyRequest
from .planner import parse_time
from .prompt import REPLY_TASKS, UNIVERSAL_TABOO, build_reply_prompt, grounding_lines, reply_max_chars, slot_labels, taboo_words
from .signals import Brief, build_brief, customer_address, customer_language, merchant_language, merchant_salutation
from .store import ContextStore
from .validate import check_reply

log = logging.getLogger("vera.dialogue")

MAX_BOT_TURNS = 6          # messages the bot may send in one conversation, the opening one included
MAX_TURN_NUMBER = 12       # the judge stops at 5; a counter far past that means something is looping
AUTO_WAIT = 86400          # second canned reply in a row: the owner is away, try again tomorrow
AUTO_END_SNOOZE = 3 * 86400
DECLINE_SNOOZE = 7 * 86400
BLANK_WAIT = 1800
UNCLEAR_WAIT = 3600
QUIET_WAIT = 6 * 3600      # after "thanks" once the work is in hand


@dataclass
class Decision:
    action: str                      # send | wait | end
    rationale: str
    body: str = ""
    cta: str = "none"
    wait: int = 0
    close: str = ""                  # a reason closes the conversation
    block: str = ""                  # a reason stops all outreach to this person
    snooze: int = 0                  # seconds without outreach to this merchant
    phase: str = ""                  # new conversation phase, if it changes
    source: str = "rules"            # rules | llm


@dataclass
class Memory:
    """What we know about one sender across conversations (the judge changes the id between repeats)."""
    seen: Counter = field(default_factory=Counter)   # message fingerprint -> times received
    auto_streak: int = 0                             # canned replies in a row


@dataclass
class Thread:
    conv: Conversation
    role: str                        # merchant | customer
    sig: Signal
    brief: Brief | None
    lang: str                        # the language to answer in
    message: str
    now: datetime
    name: str
    business: str
    clients: str

    @property
    def hindi(self) -> bool:
        return self.lang in ("hi", "hi-en")


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- slot choice (customers)

def _slot_tokens(text: str) -> set[str]:
    text = re.sub(r"(\d)\s*(am|pm)\b", r"\1\2", text.lower())
    return {t for t in re.findall(r"[a-z0-9]+", text) if not (t.isdigit() and len(t) <= 2)}


def pick_slot(message: str, labels: list[str]) -> str | None:
    """The slot label the customer chose ("2", "Thu 5pm works"), or None when it is not clear."""
    text = message.strip().lower()
    numbers = re.findall(r"\b([1-9])\b", text)
    if labels and len(numbers) == 1 and len(text) <= 40 and int(numbers[0]) <= len(labels) and len(_slot_tokens(text)) <= 3:
        return labels[int(numbers[0]) - 1]
    chosen = _slot_tokens(text)
    scored = sorted(((len(chosen & _slot_tokens(label)), label) for label in labels), reverse=True)
    if scored and scored[0][0] >= 1 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


# --------------------------------------------------------------------------- the machine

class Replier:
    def __init__(
        self, store: ContextStore, ledger: Ledger, conversations: ConversationBook, composer: Composer,
        *, reply_timeout: float = 6.0,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.book = conversations
        self.composer = composer
        self.reply_timeout = reply_timeout
        self.stats: Counter[str] = Counter()
        self._memory: dict[str, Memory] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def clear(self) -> None:
        with self._guard:
            self._memory.clear()
            self._locks.clear()
            self.stats.clear()

    # ------------------------------------------------------------------ entry point

    def handle(self, req: ReplyRequest) -> dict:
        """Always returns a valid send / wait / end object, whatever arrives."""
        began = time.monotonic()
        try:
            result = self._handle(req)
        except Exception:
            log.exception("reply on %s failed; answering with a wait", req.conversation_id)
            return {"action": "wait", "wait_seconds": UNCLEAR_WAIT,
                    "rationale": "Could not read this reply safely, so staying quiet rather than guessing."}
        log.info("reply %s turn %s -> %s (%.2fs)", req.conversation_id, req.turn_number, result["action"], time.monotonic() - began)
        return result

    def _handle(self, req: ReplyRequest) -> dict:
        conv_id = str(req.conversation_id or "")
        with self._lock_for(conv_id):
            now = parse_time(req.received_at) or datetime.now(timezone.utc)
            raw = req.message
            message = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
            role = "customer" if str(req.from_role or "").lower() == "customer" else "merchant"
            conv = self._conversation(req, conv_id, role, now)
            sig = intents.classify(message)
            memory = self._memory_for(conv, role)
            sig = self._note_canned(sig, message, memory)
            if sig.lang:
                conv.language = sig.lang  # a switch of language sticks until the next clear message
            thread = self._thread(conv, role, sig, message, now)

            if conv.status == "closed":
                if conv.closed_reason == "auto_reply" and sig.kind not in (intents.AUTO, intents.BLANK, intents.STOP):
                    self.book.reopen(conv.id)  # the closure was about canned replies; a human has now written
                else:
                    decision = Decision("end", f"This conversation is closed ({conv.closed_reason or 'closed'}); not writing again.")
                    return self._finish(thread, decision)
            if _as_int(req.turn_number) > MAX_TURN_NUMBER:
                return self._finish(thread, Decision("end", "Far beyond the judge's turn limit; closing.", close="turn_cap"))

            decision = self._customer(thread, memory) if role == "customer" else self._merchant(thread, memory)
            self.stats[f"{sig.kind}->{decision.action}"] += 1
            return self._finish(thread, decision)

    # ------------------------------------------------------------------ finding the conversation

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def _memory_for(self, conv: Conversation, role: str) -> Memory:
        key = f"{conv.merchant_id or conv.id}/{conv.customer_id or ''}/{role}"
        with self._guard:
            return self._memory.setdefault(key, Memory())

    def _merchant_in(self, conversation_id: str) -> str:
        """Our own conversation ids look like conv_<merchant>_<trigger>; recover the merchant from one."""
        for merchant_id in sorted(self.store.ids("merchant"), key=len, reverse=True):
            if conversation_id.startswith(f"conv_{merchant_id}_"):
                return merchant_id
        return ""

    def _conversation(self, req: ReplyRequest, conv_id: str, role: str, now: datetime) -> Conversation:
        conv = self.book.get(conv_id)
        if conv is not None:
            return conv
        merchant_id = str(req.merchant_id or "") or self._merchant_in(conv_id)
        customer_id = str(req.customer_id) if req.customer_id else None
        merchant = self.store.get("merchant", merchant_id) if merchant_id else None
        customer = self.store.get("customer", customer_id) if customer_id else None
        linked = self.book.latest_for(merchant_id, customer_id) if merchant_id else None
        if customer_id or role == "customer":
            salutation = customer_address(customer)[0]
            language = customer_language(customer, merchant_language(merchant or {}))
        else:
            slug = (merchant or {}).get("category_slug")
            salutation = merchant_salutation(merchant, slug) if merchant else ""
            language = merchant_language(merchant) if merchant else "en"
        return self.book.adopt(
            conv_id, merchant_id=merchant_id, customer_id=customer_id,
            send_as="merchant_on_behalf" if (customer_id or role == "customer") else "vera",
            language=language, salutation=salutation,
            trigger_id=linked.trigger_id if linked else "", kind=linked.kind if linked else "unknown",
            lead=linked.lead if linked else "", cta=linked.cta if linked else "open_ended", at=now,
        )

    def _brief(self, conv: Conversation) -> Brief | None:
        """Rebuild the context of the conversation: its own trigger if it has one, else a generic one for the merchant."""
        if not conv.merchant_id:
            return None
        trigger = self.store.get("trigger", conv.trigger_id) if conv.trigger_id else None
        if trigger is None:
            trigger = {
                "id": f"reply:{conv.id}", "kind": "unknown", "scope": "customer" if conv.customer_id else "merchant",
                "merchant_id": conv.merchant_id, "customer_id": conv.customer_id, "payload": {},
                "urgency": 1, "suppression_key": f"reply:{conv.id}",
            }
        try:
            brief, _ = build_brief(trigger, self.store, conv.trigger_id or None)
        except Exception:
            log.exception("could not rebuild the context of %s", conv.id)
            return None
        return brief

    def _thread(self, conv: Conversation, role: str, sig: Signal, message: str, now: datetime) -> Thread:
        brief = self._brief(conv)
        merchant = self.store.get("merchant", conv.merchant_id) if conv.merchant_id else None
        identity = (merchant or {}).get("identity") or {}
        slug = brief.category_slug if brief else str((merchant or {}).get("category_slug") or "")
        name = conv.salutation
        if brief is not None and any("third person" in note for note in brief.notes):
            name = ""  # this number is a relative's: greet them without the customer's name
        return Thread(
            conv=conv, role=role, sig=sig, brief=brief, lang=sig.lang or conv.language or "en", message=message, now=now,
            name=name, business=str(identity.get("name") or "our team"), clients=replytext.clients_for(slug),
        )

    def _note_canned(self, sig: Signal, message: str, memory: Memory) -> Signal:
        """Count canned replies. The same text again and again counts too, even when no pattern knows it."""
        fingerprint = intents.fingerprint(message)
        if fingerprint:
            memory.seen[fingerprint] += 1
        times = memory.seen[fingerprint] if fingerprint else 0
        words = len(fingerprint.split())
        repeated = (times >= 3 and words >= 3) or (times >= 2 and words >= 8)
        if sig.kind in (intents.STOP, intents.DECLINE, intents.GO_AHEAD, intents.BLANK):
            repeated = False
        if sig.kind == intents.AUTO:
            memory.auto_streak += 1
            return sig
        if repeated:
            memory.auto_streak = max(memory.auto_streak + 1, times)
            return replace(sig, kind=intents.AUTO, matched="same text again")
        if sig.kind != intents.BLANK:
            memory.auto_streak = 0
        return sig

    # ------------------------------------------------------------------ deciding: merchants

    def _merchant(self, t: Thread, memory: Memory) -> Decision:
        c, sig = t.conv, t.sig
        if c.merchant_id and sig.kind in (intents.GO_AHEAD, intents.QUESTION, intents.OTHER, intents.OFF_TOPIC, intents.IDENTITY):
            self.ledger.release_merchant(c.merchant_id)  # they wrote to us: an earlier pause or block no longer applies

        if sig.kind == intents.STOP:
            return Decision("end", f"The merchant asked us to stop or was hostile ({sig.matched!r}). Closing without a reply and "
                                   "stopping all outreach to them.", close="opted_out", block="asked_to_stop")
        if sig.kind == intents.AUTO:
            return self._auto(t, memory)
        if sig.kind == intents.DECLINE:
            return Decision("end", "The merchant is not interested. Closing politely and leaving them alone for a week.",
                            close="declined", snooze=DECLINE_SNOOZE)
        if sig.kind == intents.LATER:
            return Decision("wait", f"The merchant is busy or will come back ({sig.matched!r}). Leaving them alone until then.",
                            wait=sig.wait, snooze=sig.wait)
        if sig.kind == intents.BLANK:
            return Decision("wait", "Nothing readable in the message; not answering a blank.", wait=BLANK_WAIT)
        if len(c.bot_bodies) >= MAX_BOT_TURNS:
            return Decision("end", "Turn limit for one conversation reached; closing.", close="turn_cap")

        c.counts[sig.kind] = c.counts.get(sig.kind, 0) + 1
        kind = sig.kind
        asked_before = bool(self.book.bodies_to(c.merchant_id, c.customer_id))
        if kind == intents.GO_AHEAD and sig.weak and not asked_before:
            kind = intents.OTHER  # a bare "ok" to a question nobody asked
        if kind == intents.OTHER and c.phase == "opened" and (t.brief.kind if t.brief else c.kind) == "curious_ask_due":
            kind = intents.GO_AHEAD  # they answered our question: that answer is the go-ahead

        if kind == intents.GO_AHEAD:
            if c.phase == "done":
                return Decision("wait", "The work is already in hand and they only acknowledged it; waiting.", wait=QUIET_WAIT)
            if c.phase == "acted":
                return self._speak(t, "done", cta="none", phase="done",
                                   why="They confirmed. Closing the loop: it is queued and the result will follow here.")
            return self._speak(t, "action", cta="binary_confirm_cancel", phase="acted", llm="go_ahead",
                               why=f"They said yes ({sig.matched!r}). Switching from asking to doing: says what happens next "
                                   "and asks for one CONFIRM.")
        if kind == intents.THANKS:
            if c.phase in ("acted", "done") or c.counts.get(intents.THANKS, 0) > 1:
                return Decision("wait", "They only said thanks; nothing to add.", wait=QUIET_WAIT)
            return self._speak(t, "thanks_nudge", cta="binary_yes_no", why="They said thanks without answering; one easy way to say yes.")
        if kind == intents.IDENTITY:
            return self._speak(t, "identity", cta="binary_yes_no", why="They asked who we are: answer honestly in one line and return to the note.")
        if kind == intents.OFF_TOPIC and not self._on_topic(t):
            if c.counts.get(kind, 0) > 2:
                return Decision("wait", "They keep asking about something we do not do; waiting.", wait=UNCLEAR_WAIT)
            return self._speak(t, "off_topic", cta="binary_yes_no", llm="off_topic",
                               why=f"Out-of-scope request ({sig.topic}): declined politely and steered back to the earlier note.")
        if kind in (intents.QUESTION, intents.OFF_TOPIC):
            return self._speak(t, "answer", cta="binary_yes_no", llm="question",
                               why="They asked a question: answers from what we know, never guesses, and ends on one easy next step.")
        if c.counts.get(intents.OTHER, 0) > 2:
            return Decision("wait", "Several messages that call for no action; waiting.", wait=UNCLEAR_WAIT)
        return self._speak(t, "other", cta="binary_yes_no", llm="other",
                           why="A statement rather than an answer: acknowledged it and held the thread with one easy next step.")

    def _on_topic(self, t: Thread) -> bool:
        """An unrelated topic is fine when the conversation itself is about it (a GST rule change, say)."""
        if t.brief is None or not t.sig.matched:
            return False
        return t.sig.matched.lower() in " ".join(grounding_lines(t.brief)).lower()

    def _auto(self, t: Thread, memory: Memory) -> Decision:
        streak = memory.auto_streak
        if streak >= 3:
            return Decision("end", "A canned reply three times in a row and no human answer: no engagement signal, so closing.",
                            close="auto_reply", snooze=AUTO_END_SNOOZE)
        if streak == 2:
            return Decision("wait", "The same canned reply twice in a row: the owner is not at the phone. Waiting a day.",
                            wait=AUTO_WAIT, snooze=AUTO_WAIT)
        if t.conv.counts.get("auto_flagged"):
            return Decision("wait", "Another canned reply after the note for the owner; waiting.", wait=AUTO_WAIT, snooze=AUTO_WAIT)
        decision = self._speak(t, "auto_flag", cta="binary_yes_no",
                               why="A canned auto-reply: one short note meant for the owner, with a single easy way to answer.")
        if decision.action == "send":
            t.conv.counts["auto_flagged"] = 1
            return decision
        return Decision("wait", "A canned auto-reply; backing off until the owner can see it.", wait=AUTO_WAIT, snooze=AUTO_WAIT)

    # ------------------------------------------------------------------ deciding: customers

    def _customer(self, t: Thread, memory: Memory) -> Decision:
        c, sig = t.conv, t.sig
        if sig.kind == intents.STOP:
            return Decision("end", "The customer asked us to stop. Closing and not contacting them again.", close="opted_out", block="asked_to_stop")
        if sig.kind == intents.DECLINE:
            return Decision("end", "The customer is not interested. Closing politely.", close="declined")
        if sig.kind == intents.AUTO:
            if memory.auto_streak >= 2:
                return Decision("end", "Canned replies twice in a row; closing.", close="auto_reply")
            return Decision("wait", "A canned reply from the customer's number; waiting for a person.", wait=AUTO_WAIT)
        if sig.kind == intents.LATER:
            return Decision("wait", "The customer will come back later.", wait=sig.wait)
        if sig.kind == intents.BLANK:
            return Decision("wait", "Nothing readable in the message.", wait=BLANK_WAIT)
        if len(c.bot_bodies) >= MAX_BOT_TURNS:
            return Decision("end", "Turn limit for one conversation reached; closing.", close="turn_cap")
        c.counts[sig.kind] = c.counts.get(sig.kind, 0) + 1

        labels = slot_labels(t.brief) if t.brief is not None else []
        chosen = pick_slot(t.message, labels) if labels else None
        if chosen and c.phase != "done":
            return self._speak(t, "customer_booked", cta="none", phase="done", extra=chosen,
                               why="The customer chose one of the offered slots: confirmed exactly that slot.")
        if sig.kind == intents.THANKS:
            return Decision("wait", "The customer only said thanks.", wait=QUIET_WAIT)
        if sig.kind == intents.GO_AHEAD:
            if c.phase == "done":
                return Decision("wait", "Already confirmed; nothing more to add.", wait=QUIET_WAIT)
            if len(labels) == 1:  # one slot was offered and they said yes to it
                return self._speak(t, "customer_booked", cta="none", phase="done", extra=labels[0],
                                   why="The customer said yes to the only slot offered: confirmed that slot.")
            if labels and c.cta == "multi_choice_slot":
                return self._speak(t, "customer_pick", cta="multi_choice_slot", llm="customer_go_ahead", extra=labels,
                                   why="The customer agreed but did not pick a slot: listed the offered slots.")
            if c.cta == "binary_confirm_cancel":
                return self._speak(t, "customer_confirmed", cta="none", phase="done", llm="customer_go_ahead",
                                   why="The customer confirmed: acknowledged and kept it short.")
            return self._speak(t, "customer_open", cta="open_ended", llm="customer_go_ahead",
                               why="The customer agreed: asked for a day and time that suits them.")
        if sig.kind == intents.IDENTITY:
            return self._speak(t, "customer_identity", cta="none", why="The customer asked who is writing: answered as the business's assistant.")
        if sig.kind in (intents.QUESTION, intents.OFF_TOPIC):
            return self._speak(t, "customer_question", cta="none", llm="customer_question",
                               why="The customer asked something: promised the team will confirm the details here.")
        if c.counts.get(intents.OTHER, 0) > 2:
            return Decision("wait", "Several messages that call for no action; waiting.", wait=UNCLEAR_WAIT)
        return self._speak(t, "customer_other", cta="none", llm="customer_other",
                           why="A statement from the customer: acknowledged it and said the team will follow up.")

    # ------------------------------------------------------------------ writing the reply

    def _variants(self, t: Thread, what: str, extra: object) -> list[str]:
        """The template wordings for this situation, best first."""
        kind = (t.brief.kind if t.brief else t.conv.kind) or "generic_merchant"
        does = replytext.does_for(kind, t.hindi, t.clients)
        working = ""
        brief = t.brief
        if what == "action" and brief is not None and brief.lead is not None and brief.lead.key not in replytext.UNQUOTABLE_LEADS:
            working = replytext.working_line(brief.lead.text, t.hindi, own_profile=brief.degraded)
        ctx = replytext.Ctx(name=t.name, hindi=t.hindi, does=does, working=working, business=t.business, clients=t.clients)
        plain = replace(ctx, working="")
        if what == "action":
            return [*replytext.action(ctx), *replytext.action(plain)]
        if what == "off_topic":
            return replytext.off_topic(ctx, t.sig.topic)
        if what == "customer_booked":
            return replytext.customer_booked(ctx, str(extra))
        if what == "customer_pick":
            return replytext.customer_pick(ctx, list(extra or []))  # type: ignore[arg-type]
        return getattr(replytext, what)(ctx)  # every other situation is a builder of the same name

    def _grounding(self, t: Thread) -> list[str]:
        lines = [t.message, *(turn.body for turn in t.conv.turns)]
        if t.brief is not None:
            lines += grounding_lines(t.brief)
        return lines

    def _problems(self, t: Thread, body: str, seen: list[str], *, action: bool) -> list:
        customer = t.role == "customer"
        return check_reply(
            body, lang=t.lang, grounding=self._grounding(t),
            taboo=taboo_words(t.brief.category) if t.brief is not None else UNIVERSAL_TABOO,
            seen=seen, customer_facing=customer, limit=reply_max_chars(customer),
            max_questions=2 if t.conv.cta == "multi_choice_slot" else 1, action=action,
        )

    def _model(self, t: Thread, task_key: str, seen: list[str], *, action: bool) -> tuple[str, str] | None:
        if t.brief is None or not self.composer.uses_llm:
            return None
        who = {"vera": "VERA", "merchant": "MERCHANT", "customer": "CUSTOMER"}
        transcript = [(who.get(turn.role, turn.role.upper()), turn.body) for turn in t.conv.turns][-6:]
        task = REPLY_TASKS[task_key].replace("{topic}", t.sig.topic or "not our field")
        prompt = build_reply_prompt(t.brief, lang=t.lang, transcript=transcript, message=t.message, task=task, avoid=seen)
        got = self.composer.respond(prompt.system, prompt.user, timeout=self.reply_timeout)
        if got is None:
            return None
        problems = self._problems(t, got[0], seen, action=action)
        if problems:
            log.info("model reply for %s rejected: %s", t.conv.id, ", ".join(p.code for p in problems))
            self.stats["model_reply_rejected"] += 1
            return None
        return got

    def _speak(
        self, t: Thread, what: str, *, cta: str, why: str, phase: str = "", llm: str = "", extra: object = None,
    ) -> Decision:
        """The model's draft if there is a good one, else the first template that passes the checks, else a wait."""
        conv = t.conv
        # A verbatim repeat is penalised within one conversation, so that is what we avoid (plus the merchant's old history).
        seen = [*conv.bot_bodies, *(t.brief.prior_bodies if t.brief else ())]
        action = what == "action"
        if llm:
            drafted = self._model(t, llm, seen, action=action)
            if drafted is not None:
                body, rationale = drafted
                return Decision("send", rationale or why, body=body, cta=cta, phase=phase, source="llm")
        for body in self._variants(t, what, extra):
            problems = self._problems(t, body, seen, action=action)
            if not problems:
                return Decision("send", why, body=body, cta=cta, phase=phase)
            log.debug("template reply for %s rejected: %s", conv.id, ", ".join(p.code for p in problems))
        return Decision("wait", "Nothing new and safe to say right now, so waiting instead of repeating myself.", wait=UNCLEAR_WAIT)

    # ------------------------------------------------------------------ applying the decision

    def _finish(self, t: Thread, d: Decision) -> dict:
        conv = t.conv
        self.book.append(conv.id, t.role, t.message, t.now)
        if d.action == "send" and not d.body.strip():
            d = Decision("wait", "Had nothing to say.", wait=UNCLEAR_WAIT)
        if d.action == "send":
            self.book.append(conv.id, "vera", d.body, t.now)
        if d.close:
            self.book.close(conv.id, d.close)
        if d.phase:
            conv.phase = d.phase
        if t.role == "merchant" and conv.merchant_id:
            pause = d.snooze or (d.wait if d.action == "wait" else 0)
            if d.block:
                self.ledger.block_merchant(conv.merchant_id, d.block)
            if pause:
                self.ledger.snooze_merchant(conv.merchant_id, t.now + timedelta(seconds=pause))
        elif t.role == "customer" and d.block and conv.customer_id:
            self.ledger.block_customer(conv.customer_id, d.block)

        if d.action == "send":
            return {"action": "send", "body": d.body, "cta": d.cta, "rationale": d.rationale}
        if d.action == "end":
            return {"action": "end", "rationale": d.rationale}
        return {"action": "wait", "wait_seconds": int(d.wait or UNCLEAR_WAIT), "rationale": d.rationale}
