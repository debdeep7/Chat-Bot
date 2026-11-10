"""/v1/tick end to end: contexts in, actions out, through the real HTTP layer."""
import json

import pytest
from fastapi.testclient import TestClient

from vera.app import create_app
from vera.composer import Composer
from vera.config import Settings
from vera.llm import RateGate
from vera.validate import check

DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}
SLOTS = [{"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"}, {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"}]
RECALL = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
REQUIRED = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
            "template_params", "body", "cta", "suppression_key", "rationale"}


def load(app, store):
    """Copy the fixture world into the app's own store."""
    for scope in ("category", "merchant", "customer", "trigger"):
        for context_id in store.ids(scope):
            app.state.store.put(scope, context_id, 1, store.get(scope, context_id))


@pytest.fixture
def world(store, add_trigger):
    add_trigger("t_dip", "perf_dip", "m_den", DIP, urgency=4)
    add_trigger("t_recall", "recall_due", "m_den", RECALL, customer_id="c_priya", urgency=3)
    add_trigger("t_digest", "research_digest", "m_pha", PLACEHOLDER, urgency=2)
    add_trigger("t_compete", "competitor_opened", "m_res", PLACEHOLDER, urgency=2)
    return store


@pytest.fixture
def app(world):
    application = create_app(Settings())
    load(application, world)
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def tick(client, triggers, now="2026-11-06T09:00:00Z"):
    response = client.post("/v1/tick", json={"now": now, "available_triggers": triggers})
    assert response.status_code == 200
    return response.json()["actions"]


def test_a_tick_turns_active_triggers_into_complete_actions(client):
    actions = tick(client, ["t_dip", "t_recall", "t_digest", "t_compete"])
    assert {a["trigger_id"] for a in actions} == {"t_dip", "t_recall", "t_digest", "t_compete"}
    for action in actions:
        assert set(action) == REQUIRED
        assert action["body"].strip() and action["rationale"].strip()
        assert action["conversation_id"] == f"conv_{action['merchant_id']}_{action['trigger_id']}"
        assert isinstance(action["template_params"], list)
    by_trigger = {a["trigger_id"]: a for a in actions}
    assert by_trigger["t_dip"]["send_as"] == "vera" and by_trigger["t_dip"]["customer_id"] is None
    assert by_trigger["t_recall"]["send_as"] == "merchant_on_behalf" and by_trigger["t_recall"]["customer_id"] == "c_priya"


def test_every_message_that_leaves_passes_the_validator(app, client):
    triggers = ["t_dip", "t_recall", "t_digest", "t_compete"]
    for action in tick(client, triggers):
        from vera.signals import build_brief
        brief, _ = build_brief(app.state.store.get("trigger", action["trigger_id"]), app.state.store, action["trigger_id"])
        assert check(action["body"], brief, brief.prior_bodies) == [], action["body"]


def test_the_same_triggers_are_not_acted_on_twice(client):
    assert len(tick(client, ["t_dip", "t_recall"])) == 2
    assert tick(client, ["t_dip", "t_recall"], now="2026-11-06T09:05:00Z") == []


def test_a_merchant_gets_one_message_per_tick(store, add_trigger, client, app):
    add_trigger("t_second", "perf_spike", "m_den", PLACEHOLDER, urgency=1)
    load(app, store)
    actions = tick(client, ["t_dip", "t_second"])
    assert [a["trigger_id"] for a in actions] == ["t_dip"]  # the more urgent one


def test_sent_messages_are_remembered_in_the_conversation_book(app, client):
    action = tick(client, ["t_dip"])[0]
    conversation = app.state.conversations.get(action["conversation_id"])
    assert conversation.bot_bodies == [action["body"]] and conversation.merchant_id == "m_den"


def test_unknown_triggers_and_empty_ticks_are_harmless(client):
    assert tick(client, []) == []
    assert tick(client, ["never_pushed"]) == []
    assert client.post("/v1/tick", json={}).json() == {"actions": []}


def test_one_malformed_trigger_does_not_spoil_the_tick(store, add_trigger, app, client):
    add_trigger("t_broken", "perf_dip", "m_pha", ["not", "a", "dict"], urgency=5)
    load(app, store)
    actions = tick(client, ["t_broken", "t_dip"])
    assert [a["trigger_id"] for a in actions] == ["t_dip"]


def test_a_duplicate_trigger_is_not_repeated_and_is_eventually_dropped(store, add_trigger, app, client):
    # Two triggers that would produce the identical message for the same customer.
    add_trigger("t_first", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya", urgency=2)
    add_trigger("t_again", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya", urgency=2)
    load(app, store)
    assert [a["trigger_id"] for a in tick(client, ["t_first"])] == ["t_first"]
    assert tick(client, ["t_again"]) == []  # would repeat the first message word for word
    assert tick(client, ["t_again"]) == []
    assert app.state.ledger.gave_up("t_again")  # the third tick does not even try


def test_a_crash_inside_the_composer_answers_with_no_actions_not_a_500(world):
    class Broken(Composer):
        def compose_batch(self, briefs, avoid=None):
            raise RuntimeError("boom")

    application = create_app(Settings(), composer=Broken())
    load(application, world)
    with TestClient(application) as test_client:
        assert test_client.post("/v1/tick", json={"available_triggers": ["t_dip"]}).json() == {"actions": []}
        # nothing was marked as sent, so the trigger can still be handled on a later tick
        assert not application.state.ledger.already_sent("t_dip", "m_den", None, None)


def test_with_a_model_the_actions_carry_its_wording_and_rationale(world):
    body = ("Dr. Meera, aapke calls pichhle 7 days mein 50% neeche hain, baseline 12 tha. "
            "Kya main dekh loon ki kya badla aur ek chhoti fix list bhej doon? YES bhej dijiye.")

    class Model:
        def generate(self, system, prompt, *, timeout=None):
            return json.dumps({"body": body, "rationale": "Calls halved in a week; offers a fix list."})

        def close(self):
            pass

    composer = Composer(Model(), RateGate(30), workers=2, call_timeout=2, tick_budget=5)
    application = create_app(Settings(gemini_model="gemini-test"), composer=composer)
    load(application, world)
    with TestClient(application) as test_client:
        action = tick(test_client, ["t_dip"])[0]
        assert action["body"] == body and action["rationale"] == "Calls halved in a week; offers a fix list."
        assert test_client.get("/v1/metadata").json()["model"] == "gemini-test"


def test_metadata_is_honest_when_no_model_is_used(client):
    assert client.get("/v1/metadata").json()["model"] == "deterministic-templates"
