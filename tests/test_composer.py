"""The composer's decisions, with a fake model so no network is involved."""
import json
import threading
import time

import pytest

from vera import templates
from vera.composer import Composer, build_action, conversation_id, parse_reply
from vera.llm import LLMError, RateGate

DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}
GOOD = ("Dr. Sameer, your calls are down 50% over the last 7 days, against a baseline of 12. "
        "Want me to look into what changed and send you a short fix list? Reply YES.")


def model_says(body=GOOD, rationale="Numbers dipped; offers a fix list."):
    return json.dumps({"body": body, "rationale": rationale})


class FakeClient:
    """Plays back scripted replies (the last one repeats); an Exception is raised instead of returned."""

    def __init__(self, *replies, hold=None):
        self.replies, self.calls, self.hold = list(replies), [], hold

    def generate(self, system, prompt, *, timeout=None):
        self.calls.append(prompt)
        if self.hold is not None:
            self.hold.wait(10)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        pass


def composer_with(client, per_minute=20, **kwargs):
    return Composer(client, RateGate(per_minute), workers=3, call_timeout=2, tick_budget=kwargs.pop("tick_budget", 5), **kwargs)


@pytest.fixture
def sam(add_trigger, brief_for):
    return brief_for(add_trigger("t_sam", "perf_dip", "m_sam", DIP, urgency=4))


# --------------------------------------------------------------------------- reading the model's reply

@pytest.mark.parametrize("raw", [
    '{"body": "Hello Dr. Sameer", "rationale": "why"}',
    '```json\n{"body": "Hello Dr. Sameer", "rationale": "why"}\n```',
    'Sure! Here you go: {"body": "Hello Dr. Sameer", "rationale": "why"} Hope that helps.',
    '[{"body": "Hello Dr. Sameer", "rationale": "why"}]',
    '{"Body": "Hello Dr. Sameer", "Rationale": "why"}',
    '{"body": "\\"Hello Dr. Sameer\\"", "rationale": "why"}',
])
def test_replies_are_read_leniently(raw):
    assert parse_reply(raw) == ("Hello Dr. Sameer", "why")


@pytest.mark.parametrize("raw", ["", "no json here", '{"rationale": "only a reason"}', '{"body": ""}', '{"body": 12}', "[]"])
def test_unusable_replies_are_rejected(raw):
    assert parse_reply(raw) is None


def test_a_long_rationale_is_clipped():
    assert len(parse_reply(json.dumps({"body": "x y z", "rationale": "word " * 300}))[1]) <= 420


# --------------------------------------------------------------------------- the model path

def test_a_valid_draft_from_the_model_is_used(sam):
    client = FakeClient(model_says())
    message = composer_with(client).compose(sam)
    assert (message.source, message.body) == ("llm", GOOD)
    assert message.rationale == "Numbers dipped; offers a fix list."
    assert len(client.calls) == 1 and "50%" in client.calls[0]


def test_a_rejected_draft_gets_one_retry_with_the_problems_listed(sam):
    client = FakeClient(model_says(GOOD.replace("50%", "70%")), model_says())
    message = composer_with(client).compose(sam)
    assert message.source == "llm_retry" and message.body == GOOD
    assert len(client.calls) == 2
    assert "REJECTED" in client.calls[1] and "70" in client.calls[1]


def test_two_bad_drafts_fall_back_to_the_template(sam):
    client = FakeClient(model_says(GOOD.replace("50%", "70%")))
    message = composer_with(client).compose(sam)
    assert message.source == "template" and "70" not in message.body
    assert "number" in message.rejected


def test_unreadable_replies_fall_back_to_the_template(sam):
    message = composer_with(FakeClient("I cannot do that.")).compose(sam)
    assert message.source == "template" and "unreadable" in message.rejected


def test_a_rate_limit_closes_the_gate_and_later_ticks_skip_the_model(sam):
    client = FakeClient(LLMError("rate_limited", "quota", retry_after=30))
    composer = composer_with(client)
    assert composer.compose(sam).source == "template"
    assert composer.gate.blocked_for > 25  # the 30 seconds the API asked for, minus the time already passed
    assert composer.compose(sam).source == "template"
    assert len(client.calls) == 1  # the second message never reached the model
    assert composer.stats["llm_rate_limited"] == 1 and composer.stats["llm_budget_spent"] >= 1


@pytest.mark.parametrize("kind", ["timeout", "network", "server", "auth", "model_not_found", "bad_request"])
def test_persistent_failures_also_pause_the_model(sam, kind):
    composer = composer_with(FakeClient(LLMError(kind)))
    assert composer.compose(sam).source == "template"
    assert composer.gate.blocked_for > 0


def test_a_one_off_empty_answer_does_not_pause_the_model(sam):
    composer = composer_with(FakeClient(LLMError("empty")))
    assert composer.compose(sam).source == "template"
    assert composer.gate.blocked_for == 0


def test_calls_beyond_the_budget_use_templates_and_replies_keep_a_reserve(add_trigger, brief_for):
    briefs = [brief_for(add_trigger(f"t{i}", "perf_dip", m, DIP, urgency=3)) for i, m in enumerate(["m_sam", "m_den", "m_res"])]
    client = FakeClient(model_says())
    composer = composer_with(client, per_minute=3)  # reserve of 2 leaves one call for new messages
    messages = composer.compose_batch(briefs)
    assert len(client.calls) == 1
    assert [m.source for m in messages].count("template") >= 2


def test_a_slow_model_cannot_hold_the_tick_past_its_budget(sam):
    release = threading.Event()
    composer = composer_with(FakeClient(model_says(), hold=release), tick_budget=0.3)
    started = time.monotonic()
    message = composer.compose(sam)
    elapsed = time.monotonic() - started
    release.set()
    assert message.source == "template" and elapsed < 2.0
    assert composer.stats["llm_late"] == 1


def test_without_a_client_everything_is_written_from_templates(sam):
    composer = Composer()
    assert not composer.uses_llm
    assert composer.compose(sam).source == "template"


# --------------------------------------------------------------------------- never repeating

def test_a_template_that_repeats_is_reworded_with_another_fact_or_dropped(add_trigger, brief_for):
    with_support = brief_for(add_trigger("t_a", "perf_dip", "m_den", DIP, urgency=3))
    first = Composer().compose(with_support)
    second = Composer().compose(with_support, avoid=[first.body])
    assert second is not None and second.body != first.body

    appointment = brief_for(add_trigger("t_b", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    only = Composer().compose(appointment)
    composer = Composer()
    assert composer.compose(appointment, avoid=[only.body]) is None  # nothing new to say: stay silent
    assert composer.stats["silent"] == 1


def test_two_messages_to_the_same_person_in_one_tick_differ(add_trigger, brief_for):
    first = brief_for(add_trigger("t_1", "perf_dip", "m_den", DIP, urgency=3))
    second = brief_for(add_trigger("t_2", "perf_dip", "m_den", DIP, urgency=3))  # same facts, different trigger
    messages = Composer().compose_batch([first, second])
    assert messages[0] is not None
    assert messages[1] is None or messages[1].body != messages[0].body


def test_a_model_draft_that_repeats_an_earlier_message_is_replaced(sam):
    client = FakeClient(model_says())  # the model keeps producing the message that was already sent
    message = composer_with(client).compose(sam, avoid=[GOOD])
    assert message.source == "template" and message.body != GOOD
    assert "repeat" in message.rejected and len(client.calls) == 2  # it was asked once more before giving up


def test_one_broken_record_does_not_cost_the_others(sam, add_trigger, brief_for, monkeypatch):
    other = brief_for(add_trigger("t_other", "perf_dip", "m_den", DIP, urgency=3))
    real = templates.render

    def flaky(brief, variant=0):
        if brief.trigger_id == "t_sam":
            raise RuntimeError("bad record")
        return real(brief, variant)

    monkeypatch.setattr(templates, "render", flaky)
    messages = Composer().compose_batch([sam, other])
    assert messages[0] is None and messages[1].source == "template"


# --------------------------------------------------------------------------- the action

REQUIRED = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
            "template_params", "body", "cta", "suppression_key", "rationale"}


def test_a_merchant_action_has_every_field_the_judge_requires(sam):
    message = Composer().compose(sam)
    action = build_action(sam, message, conversation_id(sam))
    assert set(action) == REQUIRED
    assert action["conversation_id"] == "conv_m_sam_t_sam"
    assert (action["send_as"], action["customer_id"], action["template_name"]) == ("vera", None, "vera_perf_dip_v1")
    assert action["body"] == message.body and action["rationale"]
    assert action["cta"] == sam.cta and action["suppression_key"] == sam.suppression_key
    assert len(action["template_params"]) == 3 and all(isinstance(p, str) and p for p in action["template_params"])


def test_a_customer_action_is_sent_on_the_merchants_behalf(add_trigger, brief_for):
    brief = brief_for(add_trigger("t_c", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    action = build_action(brief, Composer().compose(brief), conversation_id(brief))
    assert (action["send_as"], action["customer_id"], action["cta"]) == ("merchant_on_behalf", "c_priya", "binary_confirm_cancel")
    assert action["template_name"] == "merchant_appointment_tomorrow_v1"
