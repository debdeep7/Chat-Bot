"""Ledger: what the bot has already done, so a repeated tick never repeats a message.

The judge keeps listing a trigger as "active" tick after tick, so without a memory the bot would
send the same thing every five minutes. The ledger records sent triggers and suppression keys,
when each merchant was last messaged, and who has opted out. Like the context store it lives in
process memory, which is why the app must run as a single worker.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .signals import Brief


MAX_SILENT_TRIES = 2  # a trigger that produced no valid message this many times is not tried again


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sent_triggers: set[str] = set()
        self._sent_keys: set[tuple[str, str, str]] = set()
        self._last_merchant_send: dict[str, datetime] = {}
        self._blocked_merchants: dict[str, str] = {}
        self._blocked_customers: dict[str, str] = {}
        self._snoozed: dict[str, datetime] = {}
        self._silent: dict[str, int] = {}

    @staticmethod
    def _key(merchant_id: str | None, customer_id: str | None, suppression_key: str | None) -> tuple[str, str, str]:
        # A suppression key can be shared by many merchants (a category-wide digest uses one key),
        # so it only suppresses repeats for the same merchant and customer.
        return (merchant_id or "", customer_id or "", suppression_key or "")

    def already_sent(
        self, trigger_id: str, merchant_id: str | None, customer_id: str | None, suppression_key: str | None
    ) -> bool:
        with self._lock:
            if trigger_id in self._sent_triggers:
                return True
            return bool(suppression_key) and self._key(merchant_id, customer_id, suppression_key) in self._sent_keys

    def mark_sent(self, brief: "Brief", now: datetime | None = None) -> None:
        """Call after a message for this brief has actually been composed and returned."""
        with self._lock:
            self._sent_triggers.add(brief.trigger_id)
            if brief.suppression_key:
                self._sent_keys.add(self._key(brief.merchant_id, brief.customer_id, brief.suppression_key))
            if now is not None and brief.customer_id is None:
                self._last_merchant_send[brief.merchant_id] = now

    def note_silence(self, trigger_id: str) -> int:
        """Composing this trigger produced no valid message. Returns how many times that has happened."""
        with self._lock:
            self._silent[trigger_id] = self._silent.get(trigger_id, 0) + 1
            return self._silent[trigger_id]

    def gave_up(self, trigger_id: str) -> bool:
        """True once retrying is pointless: the same facts keep giving a message that cannot be sent."""
        with self._lock:
            return self._silent.get(trigger_id, 0) >= MAX_SILENT_TRIES

    def last_merchant_send(self, merchant_id: str) -> datetime | None:
        with self._lock:
            return self._last_merchant_send.get(merchant_id)

    def block_merchant(self, merchant_id: str, reason: str = "opted_out") -> None:
        with self._lock:
            self._blocked_merchants[merchant_id] = reason

    def merchant_block(self, merchant_id: str) -> str | None:
        with self._lock:
            return self._blocked_merchants.get(merchant_id)

    def snooze_merchant(self, merchant_id: str, until: datetime) -> None:
        """Leave this merchant alone until `until` (they were busy, auto-replied, or said "later")."""
        with self._lock:
            current = self._snoozed.get(merchant_id)
            if current is None or until > current:
                self._snoozed[merchant_id] = until

    def snoozed_until(self, merchant_id: str) -> datetime | None:
        with self._lock:
            return self._snoozed.get(merchant_id)

    def release_merchant(self, merchant_id: str) -> None:
        """The merchant wrote to us themselves: whatever pause or block we had on them no longer applies."""
        with self._lock:
            self._snoozed.pop(merchant_id, None)
            self._blocked_merchants.pop(merchant_id, None)

    def block_customer(self, customer_id: str, reason: str = "opted_out") -> None:
        with self._lock:
            self._blocked_customers[customer_id] = reason

    def customer_block(self, customer_id: str) -> str | None:
        with self._lock:
            return self._blocked_customers.get(customer_id)

    def clear(self) -> None:
        with self._lock:
            self._sent_triggers.clear()
            self._sent_keys.clear()
            self._last_merchant_send.clear()
            self._blocked_merchants.clear()
            self._blocked_customers.clear()
            self._snoozed.clear()
            self._silent.clear()
