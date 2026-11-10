"""Gemini over plain HTTPS, plus the guard rails a free tier needs.

    GeminiClient   one method, `generate(system, prompt)`: returns the model's text or raises LLMError.
    RateGate       says whether a call may go out right now. It counts calls in a sliding one-minute
                   window and, after a 429, shuts the door for a while (growing on repeated failures)
                   so the bot stops hammering a quota it has already used up.

Neither class ever sleeps on the request path unless asked to. The caller's fallback is a
deterministic template, so "no model right now" is a normal, cheap outcome rather than an error.
The API key travels only in a header; it is never put in a URL, a log line or an exception message.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable

import httpx

API_ROOT = "https://generativelanguage.googleapis.com"
WINDOW = 60.0            # seconds covered by the per-minute budget
MIN_BACKOFF = 20.0       # shortest pause after a 429
MAX_BACKOFF = 900.0      # longest pause (15 minutes)


class LLMError(Exception):
    """A model call that produced no usable text. `kind` says why, so callers can react to it."""

    def __init__(self, kind: str, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(f"{kind}: {message}" if message else kind)
        # rate_limited | timeout | network | auth | model_not_found | bad_request | server | blocked | empty | bad_response
        self.kind = kind
        self.retry_after = retry_after  # seconds the API asked us to wait, when it said


class RateGate:
    """Sliding-window call budget with a circuit breaker. Thread-safe."""

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.per_minute = max(1, int(per_minute))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()
        self._blocked_until = 0.0
        self._strikes = 0

    def acquire(self, *, reserve: int = 0, wait: float = 0.0) -> bool:
        """Take one call from the budget. `reserve` leaves that many calls for someone more important.

        Returns False at once when the budget is spent or the breaker is open, unless `wait` allows
        sleeping until a slot frees up (used by offline scripts, never by the request path).
        """
        deadline = self._clock() + wait
        while True:
            with self._lock:
                now = self._clock()
                self._trim(now)
                if now >= self._blocked_until and len(self._calls) + reserve < self.per_minute:
                    self._calls.append(now)
                    return True
                pause = max(self._blocked_until - now, self._calls[0] + WINDOW - now if self._calls else 0.0)
            if now + pause > deadline:
                return False
            self._sleep(max(pause, 0.05))

    def penalize(self, retry_after: float | None = None) -> None:
        """The API failed (a 429, a timeout, a bad key). Pause for what it asked, at least MIN_BACKOFF,
        doubling per consecutive failure. Several calls failing together count as one strike."""
        with self._lock:
            now = self._clock()
            if now >= self._blocked_until:
                self._strikes += 1
            pause = max(retry_after or 0.0, MIN_BACKOFF) * (2 ** (self._strikes - 1))
            self._blocked_until = max(self._blocked_until, now + min(pause, MAX_BACKOFF))

    def succeeded(self) -> None:
        with self._lock:
            self._strikes = 0

    @property
    def blocked_for(self) -> float:
        """Seconds until calls are allowed again because of a 429 (0 when open)."""
        with self._lock:
            return max(0.0, self._blocked_until - self._clock())

    def _trim(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= WINDOW:
            self._calls.popleft()


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        http: httpx.Client | None = None,
        timeout: float = 8.0,
        max_output_tokens: int = 900,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiClient needs an API key")
        self.model = model
        self._key = api_key
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self._http = http or httpx.Client(base_url=API_ROOT, timeout=timeout)

    def generate(self, system: str, prompt: str, *, timeout: float | None = None) -> str:
        """One request, one answer. The model is asked for JSON; parsing it is the caller's job.

        No temperature or other sampling settings are sent: current Gemini models ignore them and
        newer ones reject them, so behaviour is steered by the instructions alone.
        """
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": self._max_output_tokens,
            },
        }
        try:
            response = self._http.post(
                f"/v1beta/models/{self.model}:generateContent",
                json=payload,
                headers={"x-goog-api-key": self._key},
                timeout=timeout or self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMError("timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__) from exc
        return self._text_of(response)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ reading the reply

    def _text_of(self, response: httpx.Response) -> str:
        status = response.status_code
        if status != 200:
            detail = self._error_message(response)
            if status == 429:
                raise LLMError("rate_limited", detail, self._retry_after(response))
            if status in (401, 403):
                raise LLMError("auth", detail)
            if status == 404:
                raise LLMError("model_not_found", f"{self.model}: {detail}")
            if status == 400:
                raise LLMError("bad_request", detail)  # an invalid key is reported as 400 too
            raise LLMError("server" if status >= 500 else "bad_response", f"HTTP {status} {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("bad_response", "not JSON") from exc

        candidates = (data.get("candidates") or []) if isinstance(data, dict) else []
        if not candidates:
            blocked = (data.get("promptFeedback") or {}).get("blockReason") if isinstance(data, dict) else None
            raise LLMError("blocked" if blocked else "empty", str(blocked or ""))
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            raise LLMError("empty", str(candidates[0].get("finishReason") or ""))
        return text

    def _error_message(self, response: httpx.Response) -> str:
        try:
            message = str((response.json().get("error") or {}).get("message") or "")
        except (ValueError, AttributeError):
            message = response.text
        return message.replace(self._key, "***")[:240]

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """Seconds the API asked us to wait, from the Retry-After header or the error details."""
        header = response.headers.get("retry-after")
        if header and re.fullmatch(r"\d+(?:\.\d+)?", header.strip()):
            return float(header)
        try:
            details = (response.json().get("error") or {}).get("details") or []
        except (ValueError, AttributeError):
            return None
        for item in details:
            delay = item.get("retryDelay") if isinstance(item, dict) else None
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(delay or ""))
            if match:
                return float(match.group(1))
        return None
