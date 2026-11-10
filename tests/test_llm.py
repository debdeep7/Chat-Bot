"""The Gemini client and the rate gate, tested with a fake clock and a fake HTTP transport (no network)."""
import json

import httpx
import pytest

from vera.llm import API_ROOT, MAX_BACKOFF, MIN_BACKOFF, GeminiClient, LLMError, RateGate

KEY = "secret-key-123"


class Clock:
    """A clock the test moves by hand; sleeping just advances it."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


def gate(clock, per_minute=3):
    return RateGate(per_minute, clock=clock, sleep=clock.sleep)


# --------------------------------------------------------------------------- the rate gate

def test_the_gate_allows_its_budget_per_minute_then_refuses(clock):
    g = gate(clock, 3)
    assert [g.acquire() for _ in range(4)] == [True, True, True, False]
    clock.now += 59
    assert g.acquire() is False
    clock.now += 2  # the first call is now more than a minute old
    assert g.acquire() is True


def test_a_reserve_keeps_calls_free_for_someone_more_important(clock):
    g = gate(clock, 4)
    assert [g.acquire(reserve=2) for _ in range(3)] == [True, True, False]
    assert g.acquire() is True and g.acquire() is True  # replies may use the reserved calls


def test_waiting_gate_sleeps_until_a_call_frees_up(clock):
    g = gate(clock, 1)
    assert g.acquire()
    assert g.acquire() is False
    assert g.acquire(wait=120) is True
    assert clock.now >= 1060


def test_a_failure_closes_the_gate_for_at_least_the_minimum_pause(clock):
    g = gate(clock, 10)
    g.penalize(None)
    assert g.acquire() is False and g.blocked_for == MIN_BACKOFF
    clock.now += MIN_BACKOFF + 1
    assert g.acquire() is True


def test_the_pause_follows_what_the_api_asked_for(clock):
    g = gate(clock, 10)
    g.penalize(75)
    assert g.blocked_for == 75


def test_repeated_failures_double_the_pause_up_to_a_ceiling(clock):
    g = gate(clock, 10)
    pauses = []
    for _ in range(12):
        g.penalize(None)
        pauses.append(g.blocked_for)
        clock.now += g.blocked_for + 0.1
    assert pauses[:3] == pytest.approx([MIN_BACKOFF, 2 * MIN_BACKOFF, 4 * MIN_BACKOFF])
    assert max(pauses) == pytest.approx(MAX_BACKOFF)


def test_a_burst_of_simultaneous_failures_counts_once(clock):
    g = gate(clock, 10)
    for _ in range(6):  # six parallel calls all hit the same limit
        g.penalize(None)
    assert g.blocked_for == MIN_BACKOFF


def test_a_success_resets_the_escalation(clock):
    g = gate(clock, 10)
    g.penalize(None)
    clock.now += MIN_BACKOFF + 1
    g.succeeded()
    g.penalize(None)
    assert g.blocked_for == MIN_BACKOFF


# --------------------------------------------------------------------------- the client

def client_for(handler):
    http = httpx.Client(base_url=API_ROOT, transport=httpx.MockTransport(handler))
    return GeminiClient(KEY, "gemini-test", http=http)


def reply(text="hello"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def test_a_request_carries_the_key_only_in_a_header_and_no_sampling_settings():
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json=reply('{"body": "hi"}'))

    assert client_for(handler).generate("be brief", "say hi") == '{"body": "hi"}'
    assert seen["url"] == f"{API_ROOT}/v1beta/models/gemini-test:generateContent"
    assert KEY not in seen["url"] and seen["headers"]["x-goog-api-key"] == KEY
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert body["contents"][0]["parts"][0]["text"] == "say hi"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    flat = json.dumps(body).lower()
    assert "temperature" not in flat and "topp" not in flat and "topk" not in flat  # ignored or rejected by current models


def test_reasoning_parts_are_not_part_of_the_answer():
    parts = [{"text": "thinking...", "thought": True}, {"text": "the answer"}]
    client = client_for(lambda request: httpx.Response(200, json={"candidates": [{"content": {"parts": parts}}]}))
    assert client.generate("s", "p") == "the answer"


@pytest.mark.parametrize("status, kind", [
    (401, "auth"), (403, "auth"), (404, "model_not_found"), (400, "bad_request"),
    (429, "rate_limited"), (500, "server"), (503, "server"), (418, "bad_response"),
])
def test_http_errors_become_typed_errors(status, kind):
    client = client_for(lambda request: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(LLMError) as caught:
        client.generate("s", "p")
    assert caught.value.kind == kind


def test_retry_hints_are_read_from_the_header_or_the_error_details():
    by_header = client_for(lambda r: httpx.Response(429, headers={"retry-after": "42"}, json={"error": {"message": "slow down"}}))
    with pytest.raises(LLMError) as caught:
        by_header.generate("s", "p")
    assert caught.value.retry_after == 42

    details = {"error": {"message": "quota", "details": [{"@type": "RetryInfo", "retryDelay": "31s"}]}}
    by_details = client_for(lambda r: httpx.Response(429, json=details))
    with pytest.raises(LLMError) as caught:
        by_details.generate("s", "p")
    assert caught.value.retry_after == 31


@pytest.mark.parametrize("payload, kind", [
    ({"candidates": []}, "empty"),
    ({}, "empty"),
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"),
    ({"candidates": [{"content": {"parts": [{"text": "   "}]}, "finishReason": "MAX_TOKENS"}]}, "empty"),
])
def test_answers_without_text_are_reported(payload, kind):
    client = client_for(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(LLMError) as caught:
        client.generate("s", "p")
    assert caught.value.kind == kind


def test_a_body_that_is_not_json_is_a_bad_response():
    client = client_for(lambda request: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(LLMError) as caught:
        client.generate("s", "p")
    assert caught.value.kind == "bad_response"


def test_timeouts_and_network_failures_are_typed():
    def slow(request):
        raise httpx.ReadTimeout("too slow", request=request)

    def down(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(LLMError) as caught:
        client_for(slow).generate("s", "p")
    assert caught.value.kind == "timeout"
    with pytest.raises(LLMError) as caught:
        client_for(down).generate("s", "p")
    assert caught.value.kind == "network"


def test_the_key_never_appears_in_an_error_message():
    leaky = {"error": {"message": f"API key not valid: {KEY}"}}
    client = client_for(lambda request: httpx.Response(400, json=leaky))
    with pytest.raises(LLMError) as caught:
        client.generate("s", "p")
    assert KEY not in str(caught.value) and "***" in str(caught.value)


def test_a_client_needs_a_key():
    with pytest.raises(ValueError):
        GeminiClient("", "gemini-test")
