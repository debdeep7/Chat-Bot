"""The reply state machine: auto-replies, intent, opt-outs, off-topic asks, languages, customers, the model path."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vera.composer import Composer, conversation_id
from vera.conversations import ConversationBook
from vera.dialogue import MAX_BOT_TURNS, Replier, pick_slot
from vera.llm import LLMError, RateGate
from vera.ledger import Ledger
from vera.models import ReplyRequest
from vera.planner import plan_tick
from vera.validate import ACTION_WORDS, QUALIFYING

NOW = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
AT = "2026-11-06T10:00:00Z"
DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}
SLOTS = [{"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"}, {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"}]
RECALL = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
AUTO = "Thank you for contacting us! Our team will respond shortly."


@pytest.fixture
def p(store, add_trigger, brief_for):
    """A reply machine over the fixture world, plus a helper that opens a conversation the way /v1/tick does."""
    add_trigger("t_dip", "perf_dip", "m_den", DIP, urgency=4)
    add_trigger("t_recall", "recall_due", "m_den", RECALL, customer_id="c_priya")
    add_trigger("t_appt", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_kid")
    add_trigger("t_refill", "chronic_refill_due", "m_pha", PLACEHOLDER, customer_id="c_sharma")
    add_trigger("t_curious", "curious_ask_due", "m_den", {"ask_template": "what_service_in_demand_this_week"})
    add_trigger("t_gst", "category_seasonal", "m_pha", {"season": "GST filing season", "trends": ["GST_returns_due"]})
    ledger, book, composer = Ledger(), ConversationBook(), Composer()
    replier = Replier(store, ledger, book, composer)

    def open_conversation(trigger_id):
        brief = brief_for(trigger_id)
        message = composer.compose(brief)
        assert message is not None
        conv_id = conversation_id(brief)
        book.open(conv_id, brief, message.body, NOW)
        ledger.mark_sent(brief, NOW)
        return conv_id

    def say(conv, message, turn=2, merchant="m_den", customer=None, role=None, at=AT):
        role = role or ("customer" if customer else "merchant")
        return replier.handle(ReplyRequest(
            conversation_id=conv, merchant_id=merchant, customer_id=customer, from_role=role,
            message=message, received_at=at, turn_number=turn,
        ))

    return SimpleNamespace(replier=replier, ledger=ledger, book=book, composer=composer, store=store,
                           open=open_conversation, say=say, add_trigger=add_trigger)


def is_send(reply):
    return reply["action"] == "send" and reply["body"].strip() and reply["cta"] and reply["rationale"]


# --------------------------------------------------------------------------- auto-replies

def test_a_canned_reply_is_flagged_once_then_waited_out_then_closed(p):
    first = p.say("conv_a", AUTO, 2)
    assert is_send(first) and "auto" in first["body"].lower() and first["cta"] == "binary_yes_no"
    second = p.say("conv_a", AUTO, 3)
    assert second["action"] == "wait" and second["wait_seconds"] == 86400
    third = p.say("conv_a", AUTO, 4)
    assert third["action"] == "end"
    assert p.say("conv_a", AUTO, 5)["action"] == "end"  # never writes again on a closed conversation


def test_the_count_follows_the_merchant_when_the_conversation_id_changes_every_turn(p):
    # the judge's simulator replays the same canned text on conv_auto_1 .. conv_auto_4
    actions = [p.say(f"conv_auto_{i}", AUTO, i + 1)["action"] for i in range(1, 4)]
    assert actions == ["send", "wait", "end"]


@pytest.mark.parametrize("text", [
    "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari team tak pahuncha deti hoon.",
    "Hello! Main ek automated assistant hoon. Aapka message hamari team tak pahunch gaya hai.",
    "We are currently away. We will get back to you soon.",
])
def test_the_canned_replies_seen_in_production_are_recognised(p, text):
    actions = [p.say(f"c{i}", text, i + 1)["action"] for i in (1, 2, 3)]
    assert actions == ["send", "wait", "end"]


def test_the_same_unknown_text_arriving_again_and_again_is_treated_as_canned(p):
    text = "Your satisfaction is our priority and a representative will be along soon to assist you"
    kinds = [p.say(f"c{i}", text, i + 1)["action"] for i in range(1, 4)]
    assert kinds[0] == "send"            # first time it is just a message
    assert kinds[1:] == ["wait", "end"]  # then it is clearly a script


def test_a_real_message_resets_the_canned_count(p):
    assert p.say("c1", AUTO)["action"] == "send"
    assert p.say("c2", AUTO)["action"] == "wait"
    assert is_send(p.say("c3", "yes please send it"))        # a human answered
    assert p.say("c4", AUTO)["action"] == "send"             # so the next canned reply is a first one again


def test_a_conversation_closed_for_canned_replies_reopens_when_a_person_writes(p):
    for turn in (2, 3, 4):
        p.say("c1", AUTO, turn)
    assert p.book.get("c1").status == "closed"
    assert is_send(p.say("c1", "Hi, this is Meera. Yes please send it", 5))


# --------------------------------------------------------------------------- going ahead

def test_lets_do_it_switches_to_action_with_no_qualifying_question(p):
    reply = p.say("conv_intent_1", "Ok lets do it. Whats next?", 2)
    body = reply["body"].lower()
    assert is_send(reply) and reply["cta"] == "binary_confirm_cancel"
    assert any(word in body for word in ACTION_WORDS)
    assert not any(phrase in body for phrase in QUALIFYING)
    assert "confirm" in body


def test_the_action_reply_names_what_the_conversation_was_about(p):
    conv = p.open("t_dip")
    reply = p.say(conv, "Yes please, go ahead")
    assert is_send(reply)
    assert "calls" in reply["body"] and "fix list" in reply["body"]  # the fact behind the trigger and the thing offered


def test_confirm_after_the_action_closes_the_loop_and_a_thank_you_after_that_waits(p):
    conv = p.open("t_dip")
    assert "confirm" in p.say(conv, "Ok lets do it", 2)["body"].lower()
    done = p.say(conv, "CONFIRM", 3)
    assert is_send(done) and done["cta"] == "none" and "?" not in done["body"]
    assert p.say(conv, "Thanks!", 4)["action"] == "wait"
    assert p.say(conv, "ok", 5)["action"] == "wait"


def test_a_merchants_answer_to_the_curious_question_is_taken_as_the_go_ahead(p):
    conv = p.open("t_curious")
    reply = p.say(conv, "Haircuts and facials mostly")
    assert is_send(reply) and "confirm" in reply["body"].lower()


def test_a_bare_ok_to_a_question_nobody_asked_is_not_a_commitment(p):
    reply = p.say("c_new", "ok", merchant="m_sam")
    assert is_send(reply) and "confirm" not in reply["body"].lower()


def test_the_action_reply_never_repeats_itself_in_one_conversation(p):
    conv = p.open("t_dip")
    sent = []
    for turn in range(2, 8):
        p.book.get(conv).phase = "opened"  # force a fresh go-ahead each time
        reply = p.say(conv, "ok lets do it", turn)
        if reply["action"] == "send":
            sent.append(reply["body"])
    assert len(sent) >= 2                 # there is more than one way to say it
    assert len(sent) == len(set(sent))    # and none is used twice


# --------------------------------------------------------------------------- stopping

def test_a_hostile_message_ends_the_conversation_and_stops_all_outreach(p, add_trigger):
    reply = p.say("conv_hostile", "Stop messaging me. This is useless spam.")
    assert reply["action"] == "end" and "body" not in reply
    assert p.ledger.merchant_block("m_den") == "asked_to_stop"
    assert p.say("conv_hostile", "hello?", 3)["action"] == "end"  # the conversation stays closed
    add_trigger("t_later", "perf_dip", "m_den", DIP, urgency=2)
    assert plan_tick(p.store, p.ledger, ["t_later"], NOW).skipped[0].reason == "merchant_opted_out"


def test_a_merchant_who_wrote_to_us_again_is_answered_and_unblocked(p):
    p.say("conv_hostile", "Stop messaging me")
    reply = p.say("conv_fresh", "Ok lets do it. Whats next?")
    assert is_send(reply) and p.ledger.merchant_block("m_den") is None


def test_not_interested_ends_politely_and_pauses_outreach_for_a_week(p, add_trigger):
    assert p.say("c1", "Not interested, thanks")["action"] == "end"
    assert p.ledger.merchant_block("m_den") is None
    add_trigger("t_next", "perf_dip", "m_den", DIP, urgency=2)
    assert plan_tick(p.store, p.ledger, ["t_next"], NOW + timedelta(days=1)).skipped[0].reason == "merchant_snoozed"
    assert [b.trigger_id for b in plan_tick(p.store, p.ledger, ["t_next"], NOW + timedelta(days=8)).briefs] == ["t_next"]


def test_busy_waits_for_as_long_as_they_said_and_nobody_is_messaged_meanwhile(p, add_trigger):
    reply = p.say("c1", "I'm in a meeting, call me tomorrow")
    assert reply["action"] == "wait" and reply["wait_seconds"] == 86400
    add_trigger("t_next", "perf_dip", "m_den", DIP, urgency=2)
    assert plan_tick(p.store, p.ledger, ["t_next"], NOW + timedelta(hours=2)).skipped[0].reason == "merchant_snoozed"
    assert plan_tick(p.store, p.ledger, ["t_next"], NOW + timedelta(hours=30)).briefs


def test_a_snooze_never_blocks_messages_to_the_merchants_customers(p, add_trigger):
    p.say("c1", "busy right now")
    assert plan_tick(p.store, p.ledger, ["t_recall"], NOW + timedelta(minutes=5)).briefs


def test_a_wild_clock_does_not_leave_a_merchant_snoozed_forever(p, add_trigger):
    p.say("c1", "Not interested", at="2030-01-01T00:00:00Z")  # the judge's clock jumped far ahead, then reset
    add_trigger("t_next", "perf_dip", "m_den", DIP, urgency=2)
    assert plan_tick(p.store, p.ledger, ["t_next"], NOW).briefs


# --------------------------------------------------------------------------- off-topic, questions, identity, thanks

def test_an_unrelated_request_is_declined_politely_and_steered_back(p):
    conv = p.open("t_dip")
    reply = p.say(conv, "Btw can you also help me with my GST filing this month?")
    assert is_send(reply)
    assert "CA" in reply["body"] and "YES" in reply["body"]
    assert "GST" in reply["body"]  # names what is being declined


def test_asking_again_gets_new_words_and_eventually_silence(p):
    conv = p.open("t_dip")
    bodies = []
    for turn in (2, 3, 4):
        reply = p.say(conv, "and my GST filing?", turn)
        if reply["action"] == "send":
            bodies.append(reply["body"])
    assert len(bodies) == len(set(bodies))
    assert len(bodies) <= 2


def test_gst_is_on_topic_when_the_conversation_is_about_gst(p):
    conv = p.open("t_gst")
    reply = p.say(conv, "What about GST on this?", merchant="m_pha")
    assert is_send(reply) and "outside" not in reply["body"].lower() and "pass on" not in reply["body"].lower()


def test_a_question_is_answered_without_guessing_and_ends_on_one_easy_step(p):
    conv = p.open("t_dip")
    reply = p.say(conv, "How much will this cost me?")
    assert is_send(reply) and reply["body"].count("?") <= 1 and "YES" in reply["body"]
    assert not any(ch.isdigit() for ch in reply["body"])  # nothing invented


def test_who_are_you_gets_an_honest_one_line_answer(p):
    reply = p.say("c1", "Who are you? Are you a bot?")
    assert is_send(reply) and "Vera" in reply["body"] and "AI" in reply["body"]


def test_thanks_without_an_answer_gets_one_nudge_then_silence(p):
    conv = p.open("t_dip")
    first = p.say(conv, "Thanks", 2)
    assert is_send(first) and "YES" in first["body"]
    assert p.say(conv, "thank you", 3)["action"] == "wait"


def test_an_unreadable_message_is_not_answered(p):
    assert p.say("c1", "🤔")["action"] == "wait"
    assert p.say("c1", "")["action"] == "wait"


def test_a_statement_is_acknowledged_twice_then_left_alone(p):
    conv = p.open("t_dip")
    outcomes = [p.say(conv, text, turn)["action"] for turn, text in enumerate(
        ["My clinic has two chairs", "We are closed on Sundays", "The waiting room is small"], start=2)]
    assert outcomes == ["send", "send", "wait"]


# --------------------------------------------------------------------------- language

def test_a_hindi_message_is_answered_in_hinglish(p):
    reply = p.say("c1", "Haan kar do, mujhe yeh chahiye")
    hindi = {"aap", "main", "bhej", "dijiye", "agla", "yahin", "karungi", "badhungi", "taiyaar", "kar"}
    assert is_send(reply) and len(set(reply["body"].lower().replace(",", " ").replace(".", " ").split()) & hindi) >= 2


def test_a_clearly_english_message_is_answered_in_english_even_for_a_hinglish_profile(p):
    # m_den's profile says Hindi + English; this message is plain English
    reply = p.say("c1", "Ok lets do it. Whats next?")
    assert "Agla" not in reply["body"] and "bhej" not in reply["body"]


def test_the_language_follows_the_latest_message(p):
    conv = p.open("t_dip")
    english = p.say(conv, "yes please go ahead with it", 2)["body"]
    hinglish = p.say(conv, "haan theek hai, kar do aap", 3)["body"]
    assert "CONFIRM" in english and "bhej dijiye" not in english
    assert "bhej dijiye" in hinglish or "badhungi" in hinglish or "bata dijiye" in hinglish


# --------------------------------------------------------------------------- limits and robustness

def test_the_bot_stops_after_its_turn_limit(p):
    conv = p.open("t_dip")
    script = ["ok lets do it", "CONFIRM", "how much does it cost?", "what does it include exactly?", "who are you?", "how long will it take?"]
    actions = [p.say(conv, text, turn)["action"] for turn, text in enumerate(script, start=2)]
    assert actions == ["send"] * 5 + ["end"]
    assert len(p.book.get(conv).bot_bodies) == MAX_BOT_TURNS
    assert p.book.get(conv).status == "closed"


def test_a_runaway_turn_counter_ends_the_conversation(p):
    assert p.say("c1", "ok lets do it", 99)["action"] == "end"


def test_a_merchant_we_have_never_heard_of_still_gets_a_valid_answer(p):
    for message in ("Ok lets do it", "How much?", "stop", AUTO, "hmm"):
        reply = p.say("conv_x", message, merchant="m_ghost")
        assert reply["action"] in ("send", "wait", "end")
        if reply["action"] == "send":
            assert reply["body"].strip() and ", ," not in reply["body"] and not reply["body"].startswith(",")


def test_a_reply_on_an_unknown_id_borrows_the_merchants_latest_conversation(p):
    p.open("t_dip")
    reply = p.say("conv_other_id", "Ok lets do it. Whats next?")
    assert "calls" in reply["body"]  # it found the perf-dip conversation of this merchant


def test_odd_requests_never_crash(p):
    for req in (
        ReplyRequest(),
        ReplyRequest(conversation_id=None, message=None, turn_number="abc", received_at="garbage", from_role=None),
        ReplyRequest(conversation_id=123, message=42, merchant_id=7),
        ReplyRequest(conversation_id="c", message="x" * 20000),
    ):
        reply = p.replier.handle(req)
        assert reply["action"] in ("send", "wait", "end") and reply["rationale"]
        if reply["action"] == "send":
            assert reply["body"].strip()
        if reply["action"] == "wait":
            assert isinstance(reply["wait_seconds"], int) and reply["wait_seconds"] > 0


def test_clear_forgets_everything_the_replier_learned(p):
    p.say("c1", AUTO)
    p.say("c2", AUTO)
    p.replier.clear()
    p.book.clear()
    assert p.say("c3", AUTO)["action"] == "send"  # a first canned reply again


# --------------------------------------------------------------------------- customers

def test_a_customer_picking_a_slot_by_number_gets_exactly_that_slot_confirmed(p):
    conv = p.open("t_recall")
    reply = p.say(conv, "2", customer="c_priya")
    assert is_send(reply) and "Thu 6 Nov, 5pm" in reply["body"] and "Wed 5 Nov" not in reply["body"]
    assert "Vera" not in reply["body"] and "magicpin" not in reply["body"].lower()
    assert reply["cta"] == "none"


def test_a_customer_picking_a_slot_in_words_gets_it_confirmed(p):
    conv = p.open("t_recall")
    assert "Wed 5 Nov, 6pm" in p.say(conv, "Wed 6pm works for me", customer="c_priya")["body"]


def test_a_customer_saying_yes_is_asked_to_pick_a_slot(p):
    conv = p.open("t_recall")
    reply = p.say(conv, "yes please", customer="c_priya")
    assert is_send(reply) and "1 for Wed 5 Nov, 6pm" in reply["body"] and reply["cta"] == "multi_choice_slot"


def test_a_customer_confirming_an_appointment_is_thanked_in_the_businesss_voice(p):
    conv = p.open("t_appt")
    reply = p.say(conv, "CONFIRM", customer="c_kid")
    assert is_send(reply) and "Dr. Meera's Dental Clinic" in reply["body"] and "Vera" not in reply["body"]


def test_a_customer_question_is_promised_an_answer_from_the_team(p):
    conv = p.open("t_recall")
    reply = p.say(conv, "Can I come in the morning instead?", customer="c_priya")
    assert is_send(reply) and "Vera" not in reply["body"] and "?" not in reply["body"]


def test_a_customer_who_says_stop_is_never_contacted_again(p, add_trigger):
    conv = p.open("t_recall")
    assert p.say(conv, "STOP", customer="c_priya")["action"] == "end"
    assert p.ledger.customer_block("c_priya") == "asked_to_stop"
    assert p.ledger.merchant_block("m_den") is None  # the merchant is not punished for a customer's opt-out
    assert p.say(conv, "hello", 3, customer="c_priya")["action"] == "end"


def test_a_relatives_phone_is_not_greeted_with_the_customers_name(p):
    conv = p.open("t_refill")
    reply = p.say(conv, "Ok", merchant="m_pha", customer="c_sharma")
    assert is_send(reply) and "Mr. Sharma" not in reply["body"].split(",")[0]


def test_pick_slot():
    labels = ["Wed 5 Nov, 6pm", "Thu 6 Nov, 5pm"]
    assert pick_slot("1", labels) == labels[0] and pick_slot("2 please", labels) == labels[1]
    assert pick_slot("Thu works", labels) == labels[1] and pick_slot("5 pm thursday", labels) == labels[1]
    assert pick_slot("yes", labels) is None and pick_slot("Nov", labels) is None and pick_slot("3", labels) is None
    assert pick_slot("anything", []) is None


# --------------------------------------------------------------------------- the model path

class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, system, prompt, *, timeout=None):
        self.calls += 1
        item = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    replies_default = ""

    def close(self):
        pass


@pytest.fixture
def m(store, add_trigger, brief_for):
    """The same world, with a model behind the composer."""
    add_trigger("t_dip", "perf_dip", "m_den", DIP, urgency=4)

    def build(replies, per_minute=30):
        model = FakeModel(replies)
        composer = Composer(model, RateGate(per_minute), workers=2, call_timeout=2, tick_budget=5)
        book, ledger = ConversationBook(), Ledger()
        brief = brief_for("t_dip")
        book.open(conversation_id(brief), brief, "opening message about calls", NOW)
        replier = Replier(store, ledger, book, composer, reply_timeout=2)
        conv = conversation_id(brief)

        def say(message, turn=2):
            return replier.handle(ReplyRequest(conversation_id=conv, merchant_id="m_den", from_role="merchant",
                                               message=message, received_at=AT, turn_number=turn))
        return SimpleNamespace(model=model, say=say, composer=composer, replier=replier, book=book)
    return build


GOOD = ("Great, Dr. Meera. Here is the first draft: your calls fell 50% in 7 days against a baseline of 12, so the likely fix is a "
        "fresh post plus a reminder to past patients. I will send both here next. Reply CONFIRM and I will proceed.")


def test_a_valid_model_draft_is_used_and_its_rationale_kept(m):
    t = m([{"body": GOOD, "rationale": "Merchant said yes; drafts the fix using the stored numbers."}])
    reply = t.say("Ok lets do it. Whats next?")
    assert reply["body"] == GOOD and reply["rationale"].startswith("Merchant said yes")
    assert t.model.calls == 1


def test_a_model_draft_that_invents_a_number_is_replaced_by_the_template(m):
    bad = GOOD.replace("12", "17")  # 17 is not in any fact
    t = m([{"body": bad, "rationale": "x"}])
    reply = t.say("Ok lets do it. Whats next?")
    assert reply["body"] != bad and "17" not in reply["body"] and is_send(reply)
    assert t.replier.stats["model_reply_rejected"] == 1


def test_a_model_draft_that_keeps_qualifying_is_replaced(m):
    t = m([{"body": "Great. Would you like the draft as a post or as a message? Reply POST or MESSAGE and the draft goes here.", "rationale": "x"}])
    reply = t.say("Ok lets do it. Whats next?")
    assert "would you" not in reply["body"].lower() and "confirm" in reply["body"].lower()


def test_a_failing_or_garbled_model_falls_back_to_the_template(m):
    for failure in (LLMError("timeout", "slow"), LLMError("rate_limited", "429", retry_after=30), "not json at all"):
        t = m([failure])
        reply = t.say("Ok lets do it. Whats next?")
        assert is_send(reply) and "confirm" in reply["body"].lower()


def test_no_free_model_call_means_a_template_not_a_delay(m):
    t = m([{"body": GOOD, "rationale": "x"}], per_minute=1)
    assert t.composer.gate.acquire()  # spend the only call
    reply = t.say("Ok lets do it. Whats next?")
    assert is_send(reply) and reply["body"] != GOOD and t.model.calls == 0


def test_opt_outs_and_canned_replies_never_cost_a_model_call(m):
    t = m([{"body": GOOD, "rationale": "x"}] * 5)
    for message in ("Stop messaging me", AUTO, "Not interested", "busy, later", "🙏"):
        t.say(message)
    assert t.model.calls == 0


def test_the_model_sees_the_conversation_and_the_latest_message(m):
    seen = {}

    class Spy(FakeModel):
        def generate(self, system, prompt, *, timeout=None):
            seen["prompt"] = prompt
            return json.dumps({"body": GOOD, "rationale": "x"})

    model = Spy([])
    t = m([])
    t.composer.client = model
    t.say("Ok lets do it. Whats next?")
    assert "opening message about calls" in seen["prompt"] and "Ok lets do it" in seen["prompt"]
    assert "perf" in seen["prompt"].lower() or "calls" in seen["prompt"]
