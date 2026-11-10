"""Conversation book: what has been said, to whom, in each conversation the bot started.

Two jobs. Before a message is written it tells the composer what this person already received, so
nothing is repeated. When a reply arrives (a later step) it gives the reply handler the thread it
belongs to. Like the other stores it lives in process memory, so the app runs as a single worker.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .signals import Brief


@dataclass
class Turn:
    role: str            # vera | merchant | customer
    body: str
    at: datetime | None = None


@dataclass
class Conversation:
    id: str
    merchant_id: str
    customer_id: str | None
    trigger_id: str
    kind: str
    send_as: str
    language: str
    cta: str
    salutation: str
    lead: str                        # what the opening message was about
    started_at: datetime | None = None
    status: str = "open"             # open | closed
    closed_reason: str = ""
    turns: list[Turn] = field(default_factory=list)
    # State of the reply conversation, owned by the reply handler.
    phase: str = "opened"            # opened -> acted (we started the work, asked CONFIRM) -> done
    counts: dict[str, int] = field(default_factory=dict)   # how often each kind of message has arrived here

    @property
    def bot_bodies(self) -> list[str]:
        return [t.body for t in self.turns if t.role == "vera"]

    @property
    def last_bot_body(self) -> str:
        bodies = self.bot_bodies
        return bodies[-1] if bodies else ""


class ConversationBook:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Conversation] = {}

    def open(self, conversation_id: str, brief: "Brief", body: str, at: datetime | None = None) -> Conversation:
        """Record the opening message of a conversation (idempotent: a repeat adds nothing)."""
        with self._lock:
            existing = self._by_id.get(conversation_id)
            if existing is not None:
                return existing
            conversation = Conversation(
                id=conversation_id, merchant_id=brief.merchant_id, customer_id=brief.customer_id,
                trigger_id=brief.trigger_id, kind=brief.kind, send_as=brief.send_as, language=brief.language,
                cta=brief.cta, salutation=brief.salutation, lead=brief.lead.text if brief.lead else "",
                started_at=at, turns=[Turn("vera", body, at)],
            )
            self._by_id[conversation_id] = conversation
            return conversation

    def adopt(
        self, conversation_id: str, *, merchant_id: str, customer_id: str | None = None, send_as: str = "vera",
        language: str = "en", salutation: str = "", trigger_id: str = "", kind: str = "unknown", lead: str = "",
        cta: str = "open_ended", at: datetime | None = None,
    ) -> Conversation:
        """Start tracking a conversation the bot did not open (the judge may reply on any id)."""
        with self._lock:
            existing = self._by_id.get(conversation_id)
            if existing is not None:
                return existing
            conversation = Conversation(
                id=conversation_id, merchant_id=merchant_id, customer_id=customer_id, trigger_id=trigger_id,
                kind=kind, send_as=send_as, language=language, cta=cta, salutation=salutation, lead=lead, started_at=at,
            )
            self._by_id[conversation_id] = conversation
            return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._by_id.get(conversation_id)

    def latest_for(self, merchant_id: str, customer_id: str | None) -> Conversation | None:
        """The newest conversation the bot itself opened with this merchant (or this customer of the merchant)."""
        with self._lock:
            found = [
                c for c in self._by_id.values()
                if c.merchant_id == merchant_id and c.customer_id == customer_id and c.trigger_id and c.kind != "unknown"
            ]
            return found[-1] if found else None

    def append(self, conversation_id: str, role: str, body: str, at: datetime | None = None) -> Conversation | None:
        with self._lock:
            conversation = self._by_id.get(conversation_id)
            if conversation is not None:
                conversation.turns.append(Turn(role, body, at))
            return conversation

    def close(self, conversation_id: str, reason: str = "") -> None:
        with self._lock:
            conversation = self._by_id.get(conversation_id)
            if conversation is not None:
                conversation.status, conversation.closed_reason = "closed", reason

    def reopen(self, conversation_id: str) -> None:
        with self._lock:
            conversation = self._by_id.get(conversation_id)
            if conversation is not None:
                conversation.status, conversation.closed_reason = "open", ""

    def bodies_to(self, merchant_id: str, customer_id: str | None) -> list[str]:
        """Everything the bot has sent to this merchant (or this customer of the merchant), oldest first."""
        with self._lock:
            found = [c for c in self._by_id.values() if c.merchant_id == merchant_id and c.customer_id == customer_id]
            found.sort(key=lambda c: (c.started_at is None, c.started_at))
            return [body for c in found for body in c.bot_bodies]

    def bodies_for(self, brief: "Brief") -> list[str]:
        return self.bodies_to(brief.merchant_id, brief.customer_id)

    def for_merchant(self, merchant_id: str) -> list[Conversation]:
        with self._lock:
            return [c for c in self._by_id.values() if c.merchant_id == merchant_id]

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)
