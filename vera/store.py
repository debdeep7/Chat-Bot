"""In-memory, versioned context store.

The judge pushes four kinds of context (category / merchant / customer / trigger) and may
re-push or upgrade any of them mid-test. The rules come from the testing brief:

* same (scope, context_id, version) again -> no-op (idempotent), still "accepted"
* higher version                          -> replaces the old payload atomically
* lower version                           -> rejected as stale

State lives in process memory, so the app must run as a single worker process.
Payloads are shared, not copied: callers must treat what `get` returns as read-only.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

SCOPES: tuple[str, ...] = ("category", "merchant", "customer", "trigger")

Outcome = Literal["stored", "noop", "stale"]


@dataclass(frozen=True)
class PutResult:
    outcome: Outcome
    version: int  # the version the store holds after the call


class ContextStore:
    """scope -> context_id -> (version, payload). Lookups are O(1) dict hits."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {s: {} for s in SCOPES}
        self._lock = threading.Lock()

    def put(self, scope: str, context_id: str, version: int, payload: dict[str, Any]) -> PutResult:
        if scope not in self._data:
            raise ValueError(f"unknown scope: {scope!r}")
        with self._lock:
            bucket = self._data[scope]
            current = bucket.get(context_id)
            if current is not None:
                if version == current[0]:
                    return PutResult("noop", current[0])
                if version < current[0]:
                    return PutResult("stale", current[0])
            # One tuple assignment: readers see the old entry or the new one, never a mix.
            bucket[context_id] = (version, payload)
            return PutResult("stored", version)

    def get(self, scope: str, context_id: str) -> dict[str, Any] | None:
        entry = self._data.get(scope, {}).get(context_id)
        return entry[1] if entry else None

    def version(self, scope: str, context_id: str) -> int | None:
        entry = self._data.get(scope, {}).get(context_id)
        return entry[0] if entry else None

    def ids(self, scope: str) -> list[str]:
        """All context ids in a scope, sorted so iteration order is deterministic."""
        with self._lock:
            return sorted(self._data.get(scope, {}))

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {scope: len(bucket) for scope, bucket in self._data.items()}

    def clear(self) -> None:
        with self._lock:
            for bucket in self._data.values():
                bucket.clear()
