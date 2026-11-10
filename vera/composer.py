"""Composer: turn Briefs into messages, preferring the model and never falling below the templates.

For every Brief, in the planner's order:
    1. ask Gemini, if it is switched on, the rate gate has a free call and the tick has time left
    2. validate the draft; if it fails, ask once more with the problems listed
    3. otherwise write the template, validated the same way
    4. otherwise send nothing for this trigger: silence is cheaper than a bad message

Model calls for one tick run side by side in a small thread pool, and the tick waits for them only
as long as its budget allows, so a slow or failing model costs quality, never a timeout.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from . import templates
from .llm import GeminiClient, LLMError, RateGate
from .prompt import build_prompt, business_name
from .validate import check

if TYPE_CHECKING:
    from .config import Settings
    from .signals import Brief

log = logging.getLogger("vera.composer")

RESERVE = 2              # model calls per minute kept free for replies, which matter more than new messages
MIN_RETRY_SECONDS = 3.0  # a second model call is only worth starting when this much of the tick is left
RATIONALE_LIMIT = 420
NO_HISTORY: Callable[["Brief"], Sequence[str]] = lambda brief: ()  # noqa: E731


@dataclass(frozen=True)
class Message:
    body: str
    rationale: str
    source: str                           # llm | llm_retry | template
    rejected: tuple[str, ...] = ()        # why earlier drafts were thrown away, for the logs


@dataclass
class _Draft:
    body: str | None = None
    rationale: str = ""
    retried: bool = False
    rejected: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- reading the model's reply

def _json_candidates(text: str):
    yield text
    braces = re.search(r"\{.*\}", text, re.S)  # chatter before or after the object
    if braces:
        yield braces.group()


def parse_reply(raw: str) -> tuple[str, str] | None:
    """(body, rationale) from the model's text, tolerating code fences and chatter around the JSON."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (raw or "").strip(), flags=re.I)
    data = None
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    fields = {str(k).lower(): v for k, v in data.items()}
    body = fields.get("body") or fields.get("message") or fields.get("text")
    if isinstance(body, list):
        body = "\n".join(str(part) for part in body)
    if not isinstance(body, str) or not body.strip():
        return None
    body = body.strip()
    if len(body) > 1 and body[0] == body[-1] == '"':  # the whole message wrapped in quotes
        body = body[1:-1].strip()
    rationale = fields.get("rationale") or fields.get("reason") or ""
    return body, " ".join(str(rationale).split())[:RATIONALE_LIMIT]


# --------------------------------------------------------------------------- the composer

class Composer:
    """With no client (no key, or USE_LLM=0) it runs on templates alone."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        gate: RateGate | None = None,
        *,
        workers: int = 6,
        call_timeout: float = 8.0,
        tick_budget: float = 8.0,
        reserve: int = RESERVE,
    ) -> None:
        self.client = client
        self.gate = gate
        self.call_timeout = call_timeout
        self.tick_budget = tick_budget
        self.reserve = reserve
        self.patience = 0.0  # seconds a caller may wait for a free model call; only offline scripts raise it
        self.stats: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vera-llm") if client and gate else None

    @property
    def uses_llm(self) -> bool:
        return self._pool is not None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        if self.client is not None:
            self.client.close()

    # ------------------------------------------------------------------ public

    def compose_batch(
        self, briefs: Sequence["Brief"], avoid: Callable[["Brief"], Sequence[str]] = NO_HISTORY
    ) -> list[Message | None]:
        """One Message (or None = stay silent) per Brief, in the same order.

        `avoid(brief)` lists what this person has already been sent, so nothing is repeated.
        """
        deadline = time.monotonic() + self.tick_budget
        drafts: dict[int, Future] = {}
        if self._pool is not None and self.gate is not None:
            for index, brief in enumerate(briefs):
                if not self.gate.acquire(reserve=self.reserve, wait=self.patience):
                    self._count("llm_budget_spent")
                    break  # the rest of the tick is written from templates
                drafts[index] = self._pool.submit(self._draft, brief, list(avoid(brief)), deadline)
            wait(drafts.values(), timeout=max(0.0, deadline - time.monotonic()))

        sent_now: dict[tuple[str, str | None], list[str]] = {}
        messages: list[Message | None] = []
        for index, brief in enumerate(briefs):
            person = (brief.merchant_id, brief.customer_id)
            earlier = [*avoid(brief), *sent_now.get(person, [])]
            future = drafts.get(index)
            draft = future.result() if future is not None and future.done() and not future.exception() else None
            if future is not None and draft is None:
                self._count("llm_late" if not future.done() else "llm_crashed")
            try:
                message = self._finalize(brief, draft, earlier)
            except Exception:  # one odd record must not cost the whole tick
                log.exception("composing %s failed", brief.trigger_id)
                message = None
            if message is not None:
                sent_now.setdefault(person, []).append(message.body)
            messages.append(message)
        return messages

    def compose(self, brief: "Brief", avoid: Sequence[str] = ()) -> Message | None:
        return self.compose_batch([brief], lambda _brief: avoid)[0]

    def respond(self, system: str, user: str, *, timeout: float | None = None) -> tuple[str, str] | None:
        """One model call for a reply, made on the caller's own thread: (body, rationale), or None.

        None means "write it yourself": no model, no free call in the rate window, or no usable
        answer. Replies may use the calls that new messages leave in reserve.
        """
        if self._pool is None or self.gate is None or self.client is None:
            return None
        if not self.gate.acquire(reserve=0, wait=0.0):
            self._count("reply_budget_spent")
            return None
        try:
            raw = self.client.generate(system, user, timeout=timeout or self.call_timeout)
        except LLMError as exc:
            self._model_failed(exc)
            return None
        self.gate.succeeded()
        parsed = parse_reply(raw)
        self._count("reply_llm" if parsed else "reply_unreadable")
        return parsed

    # ------------------------------------------------------------------ the model path (worker threads)

    def _draft(self, brief: "Brief", avoid: list[str], deadline: float) -> _Draft:
        assert self.client is not None and self.gate is not None
        result = _Draft()
        feedback: list[str] = []
        previous = ""
        for attempt in range(2):
            if attempt:
                enough_time = deadline - time.monotonic() >= MIN_RETRY_SECONDS
                if not enough_time or not self.gate.acquire(reserve=self.reserve):
                    return result
                result.retried = True
            prompt = build_prompt(brief, avoid=avoid, feedback=feedback, previous=previous)
            timeout = max(1.0, min(self.call_timeout, deadline - time.monotonic()))
            try:
                raw = self.client.generate(prompt.system, prompt.user, timeout=timeout)
            except LLMError as exc:
                self._model_failed(exc)
                return result
            self.gate.succeeded()
            parsed = parse_reply(raw)
            if parsed is None:
                result.rejected.append("unreadable")
                feedback, previous = ['Reply with only a JSON object: {"body": "...", "rationale": "..."}.'], ""
                continue
            body, rationale = parsed
            problems = check(body, brief, avoid)
            if not problems:
                result.body, result.rationale = body, rationale
                return result
            result.rejected += [p.code for p in problems]
            feedback, previous = [p.detail for p in problems], body
        return result

    def _model_failed(self, exc: LLMError) -> None:
        self._count(f"llm_{exc.kind}")
        if exc.kind == "rate_limited":
            self.gate.penalize(exc.retry_after)
        elif exc.kind in ("auth", "model_not_found", "bad_request", "timeout", "network", "server"):
            self.gate.penalize(None)  # stop calling for a while instead of failing on every message
        if exc.kind in ("auth", "model_not_found", "bad_request"):
            log.error("model call rejected (check GEMINI_API_KEY and GEMINI_MODEL): %s", exc)
        else:
            log.warning("model call failed: %s", exc)

    # ------------------------------------------------------------------ choosing what to send

    def _finalize(self, brief: "Brief", draft: _Draft | None, earlier: list[str]) -> Message | None:
        rejected = list(draft.rejected) if draft else []
        if draft is not None and draft.body:
            problems = check(draft.body, brief, earlier)  # again, now that earlier messages of this tick are known
            if not problems:
                source = "llm_retry" if draft.retried else "llm"
                self._count(source)
                return Message(draft.body, draft.rationale or templates.rationale(brief), source, tuple(rejected))
            rejected += [p.code for p in problems]

        for variant in (0, 1):
            body = templates.render(brief, variant)
            if body is None:
                break
            problems = check(body, brief, earlier)
            if not problems:
                self._count("template")
                return Message(body, templates.rationale(brief), "template", tuple(rejected))
            rejected += [f"template:{p.code}" for p in problems]
            if {p.code for p in problems} != {"repeat"}:
                break  # only a repeat can be cured by wording the same facts differently
        self._count("silent")
        log.info("staying silent on %s (%s): %s", brief.trigger_id, brief.kind, ", ".join(rejected) or "no lead")
        return None

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1


def build_composer(settings: "Settings") -> Composer:
    """The composer the app should use: Gemini when a key is present and switched on, else templates only."""
    if not settings.llm_enabled:
        return Composer(tick_budget=settings.tick_budget)
    client = GeminiClient(settings.gemini_api_key, settings.gemini_model, timeout=settings.llm_timeout)
    return Composer(
        client, RateGate(settings.llm_per_minute),
        workers=settings.llm_workers, call_timeout=settings.llm_timeout, tick_budget=settings.tick_budget,
    )


# --------------------------------------------------------------------------- the action the judge receives

def conversation_id(brief: "Brief") -> str:
    return f"conv_{brief.merchant_id}_{brief.trigger_id}"


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def build_action(brief: "Brief", message: Message, conv_id: str) -> dict:
    """The action object for /v1/tick. Every field the judge requires is always present."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", message.body) if s]
    owner = "merchant" if brief.send_as == "merchant_on_behalf" else "vera"
    return {
        "conversation_id": conv_id,
        "merchant_id": brief.merchant_id,
        "customer_id": brief.customer_id,
        "send_as": brief.send_as,
        "trigger_id": brief.trigger_id,
        "template_name": f"{owner}_{brief.kind}_v1",
        "template_params": [
            brief.salutation or business_name(brief),
            _clip(brief.lead.text if brief.lead else "", 120),
            _clip(sentences[-1] if sentences else message.body, 160),
        ],
        "body": message.body,
        "cta": brief.cta,
        "suppression_key": brief.suppression_key,
        "rationale": message.rationale,
    }
