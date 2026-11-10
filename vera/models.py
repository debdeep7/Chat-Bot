"""Request bodies for the judge's calls.

Deliberately lenient: unknown fields are kept, optional fields default. A 422 here would
look like a broken bot to the harness, so we only reject what we truly cannot use.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class ContextPush(_Lenient):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickRequest(_Lenient):
    now: str | None = None
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(_Lenient):
    """Every field is optional and loosely typed: a reply must always get an answer, even to an odd request."""
    conversation_id: Any = ""
    merchant_id: Any = None
    customer_id: Any = None
    from_role: Any = "merchant"
    message: Any = ""
    received_at: Any = None
    turn_number: Any = 1
