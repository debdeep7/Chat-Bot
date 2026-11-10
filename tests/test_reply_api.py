"""/v1/reply and /v1/teardown through the real HTTP layer: the judge's replay scenarios, response shapes, robustness."""
import pytest
from fastapi.testclient import TestClient

from vera.app import create_app
from vera.config import Settings
from vera.validate import ACTION_WORDS, QUALIFYING

DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
SLOTS = [{"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"}, {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"}]
RECALL = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
AUTO = "Thank you for contacting us! Our team will respond shortly."
NOW = "2026-11-06T10:00:00Z"
CTAS = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}
SHAPES = {
    "send": {"action", "body", "cta", "rationale"},
    "wait": {"action", "wait_seconds", "rationale"},
    "end": {"action", "rationale"},
}


@pytest.fixture
def app(store, add_trigger):
    add_trigger("t_dip", "perf_dip", "m_den", DIP, urgency=4)
    add_trigger("t_recall", "recall_due", "m_den", RECALL, customer_id="c_priya", urgency=3)
    application = create_app(Settings())
    for scope in ("category", "merchant", "customer", "trigger"):
        for context_id in store.ids(scope):
            application.state.store.put(scope, context_id, 1, store.get(scope, context_id))
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def reply(client, conversation_id, message, turn=2, merchant="m_den", customer=None, role=None, received_at=NOW):
    response = client.post("/v1/reply", json={
        "conversation_id": conversation_id, "merchant_id": merchant, "customer_id": customer,
        "from_role": role or ("customer" if customer else "merchant"),
        "message": message, "received_at": received_at, "turn_number": turn,
    })
    assert response.status_code == 200, response.text
    return response.json()


def assert_valid(result):
    """Exactly the fields the contract names, nothing missing, nothing extra."""
    assert result["action"] in SHAPES
    assert set(result) == SHAPES[result["action"]], result
    assert isinstance(result["rationale"], str) and result["rationale"].strip()
    if result["action"] == "send":
        assert isinstance(result["body"], str) and result["body"].strip()
        assert result["cta"] in CTAS
        assert "http" not in result["body"].lower() and "www." not in result["body"].lower()
    if result["action"] == "wait":
        assert isinstance(result["wait_seconds"], int) and result["wait_seconds"] > 0


# --------------------------------------------------------------------------- the judge's replay scenarios

def test_auto_reply_scenario(client):
    """The same canned text on a new conversation id each turn: flag it once, wait a day, then stop."""
    results = [reply(client, f"conv_auto_{i}", AUTO, turn=i + 1) for i in range(1, 5)]
    for result in results:
        assert_valid(result)
    assert [r["action"] for r in results] == ["send", "wait", "end", "end"]
    assert results[1]["wait_seconds"] == 86400
    assert len(results[0]["body"]) < 400  # a short note for the owner, not a pitch


def test_intent_scenario(client):
    result = reply(client, "conv_intent_1", "Ok lets do it. Whats next?")
    assert_valid(result)
    assert result["action"] == "send"
    body = result["body"].lower()
    assert any(word in body for word in ACTION_WORDS)
    assert not any(phrase in body for phrase in QUALIFYING)


def test_hostile_scenario(client):
    result = reply(client, "conv_hostile", "Stop messaging me. This is useless spam.")
    assert_valid(result)
    assert result["action"] == "end"


def test_the_full_judge_flow_in_one_run_on_the_same_merchant(client):
    """The simulator's `_all` uses one merchant for every scenario; an earlier scenario must not break a later one."""
    assert [reply(client, f"conv_auto_{i}", AUTO, turn=i + 1)["action"] for i in range(1, 4)] == ["send", "wait", "end"]
    intent = reply(client, "conv_intent_1", "Ok lets do it. Whats next?")
    assert intent["action"] == "send" and any(word in intent["body"].lower() for word in ACTION_WORDS)
    assert reply(client, "conv_hostile", "Stop messaging me. This is useless spam.")["action"] == "end"


# --------------------------------------------------------------------------- replies to conversations the bot opened

def test_a_reply_continues_the_conversation_the_tick_opened(client):
    tick = client.post("/v1/tick", json={"now": NOW, "available_triggers": ["t_dip"]}).json()["actions"][0]
    result = reply(client, tick["conversation_id"], "Yes please, go ahead")
    assert_valid(result)
    assert result["action"] == "send" and "calls" in result["body"]  # it knows what the conversation was about
    assert result["body"] != tick["body"]


def test_a_customer_reply_stays_in_the_businesss_voice(client):
    tick = client.post("/v1/tick", json={"now": NOW, "available_triggers": ["t_recall"]}).json()["actions"][0]
    result = reply(client, tick["conversation_id"], "2", customer="c_priya")
    assert_valid(result)
    assert "Thu 6 Nov, 5pm" in result["body"] and "Vera" not in result["body"]


def test_blocking_a_merchant_stops_the_next_tick(client):
    reply(client, "conv_x", "Stop messaging me")
    assert client.post("/v1/tick", json={"now": NOW, "available_triggers": ["t_dip"]}).json()["actions"] == []


# --------------------------------------------------------------------------- the shapes, whatever arrives

@pytest.mark.parametrize("message", [
    AUTO, "ok lets do it", "Stop messaging me", "not interested", "busy, call me tomorrow", "how much?", "who are you?",
    "thanks", "Btw can you help with my GST filing?", "my clinic has two chairs", "", "👍", "haan kar do", "ठीक है",
])
def test_every_kind_of_message_gets_a_valid_answer(client, message):
    assert_valid(reply(client, "conv_shapes", message))


@pytest.mark.parametrize("payload", [
    {},
    {"conversation_id": "c"},
    {"conversation_id": "c", "message": "hello"},
    {"conversation_id": None, "message": None, "turn_number": None, "from_role": None},
    {"conversation_id": "c", "message": "yes", "turn_number": "3", "received_at": "not a date"},
    {"conversation_id": 7, "merchant_id": 9, "message": 12345},
    {"conversation_id": "c", "message": "yes", "merchant_id": "nobody", "customer_id": "nobody", "from_role": "customer"},
    {"conversation_id": "c", "message": "x" * 50000},
    {"conversation_id": "c", "message": "yes", "unexpected": {"nested": [1, 2, 3]}},
])
def test_odd_requests_still_get_a_valid_answer(client, payload):
    response = client.post("/v1/reply", json=payload)
    assert response.status_code == 200, response.text
    assert_valid(response.json())


def test_a_flood_of_replies_from_many_threads_is_answered(client):
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        return reply(client, f"conv_{i % 5}", "ok lets do it" if i % 2 else "how much will it cost?", turn=2 + i % 3)

    with ThreadPoolExecutor(8) as pool:
        for result in pool.map(one, range(40)):
            assert_valid(result)


# --------------------------------------------------------------------------- teardown

def test_teardown_wipes_every_piece_of_state(app, client):
    client.post("/v1/tick", json={"now": NOW, "available_triggers": ["t_dip"]})
    reply(client, "conv_hostile", "Stop messaging me")
    reply(client, "conv_a", AUTO)
    reply(client, "conv_b", AUTO)
    assert len(app.state.conversations) >= 2 and app.state.ledger.merchant_block("m_den")

    response = client.post("/v1/teardown")
    assert response.status_code == 200 and response.json() == {"status": "cleared"}
    assert sum(client.get("/v1/healthz").json()["contexts_loaded"].values()) == 0
    assert len(app.state.conversations) == 0
    assert app.state.ledger.merchant_block("m_den") is None and app.state.ledger.snoozed_until("m_den") is None
    assert not app.state.ledger.already_sent("t_dip", "m_den", None, None)
    # the canned-reply count is gone too: this is a first canned reply again, and the merchant is simply unknown
    assert_valid(reply(client, "conv_c", AUTO))
    assert reply(client, "conv_d", AUTO)["action"] == "wait"  # (second in a row after the wipe)


def test_teardown_can_be_called_twice_and_the_service_keeps_working(client):
    assert client.post("/v1/teardown").status_code == 200
    assert client.post("/v1/teardown").status_code == 200
    assert client.get("/v1/healthz").status_code == 200
    assert_valid(reply(client, "conv_after", "ok lets do it"))
