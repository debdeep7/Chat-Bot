from datetime import datetime, timezone

from vera.conversations import ConversationBook

DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}


def at(hour):
    return datetime(2026, 11, 6, hour, 0, tzinfo=timezone.utc)


def test_opening_a_conversation_records_the_first_message(add_trigger, brief_for):
    book = ConversationBook()
    brief = brief_for(add_trigger("t1", "perf_dip", "m_den", DIP))
    conversation = book.open("conv_1", brief, "Dr. Meera, hello", at(9))
    assert (conversation.merchant_id, conversation.kind, conversation.language) == ("m_den", "perf_dip", "hi-en")
    assert conversation.lead == brief.lead.text and conversation.status == "open"
    assert conversation.bot_bodies == ["Dr. Meera, hello"] and len(book) == 1
    assert book.get("conv_1") is conversation and book.get("nope") is None


def test_opening_twice_adds_nothing(add_trigger, brief_for):
    book = ConversationBook()
    brief = brief_for(add_trigger("t1", "perf_dip", "m_den", DIP))
    book.open("conv_1", brief, "first", at(9))
    book.open("conv_1", brief, "second", at(10))
    assert book.get("conv_1").bot_bodies == ["first"]


def test_replies_extend_a_thread_and_conversations_can_be_closed(add_trigger, brief_for):
    book = ConversationBook()
    brief = brief_for(add_trigger("t1", "perf_dip", "m_den", DIP))
    book.open("conv_1", brief, "hello", at(9))
    book.append("conv_1", "merchant", "yes please", at(10))
    book.append("conv_1", "vera", "done", at(11))
    assert [t.role for t in book.get("conv_1").turns] == ["vera", "merchant", "vera"]
    assert book.get("conv_1").bot_bodies == ["hello", "done"]
    assert book.append("missing", "merchant", "hi") is None
    book.close("conv_1", "merchant opted out")
    assert (book.get("conv_1").status, book.get("conv_1").closed_reason) == ("closed", "merchant opted out")


def test_what_a_person_was_already_sent_is_listed_oldest_first(add_trigger, brief_for):
    book = ConversationBook()
    dip = brief_for(add_trigger("t1", "perf_dip", "m_den", DIP))
    spike = brief_for(add_trigger("t2", "perf_spike", "m_den", PLACEHOLDER))
    customer = brief_for(add_trigger("t3", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    book.open("conv_late", spike, "second message", at(12))
    book.open("conv_early", dip, "first message", at(9))
    book.open("conv_customer", customer, "to the customer", at(10))
    assert book.bodies_for(dip) == ["first message", "second message"]  # merchant-facing only
    assert book.bodies_for(customer) == ["to the customer"]
    assert len(book.for_merchant("m_den")) == 3


def test_a_conversation_the_bot_never_opened_can_be_adopted():
    book = ConversationBook()
    conversation = book.adopt("conv_x", merchant_id="m_den", language="hi-en", salutation="Dr. Meera", at=at(9))
    assert (conversation.kind, conversation.trigger_id, conversation.status, conversation.phase) == ("unknown", "", "open", "opened")
    assert conversation.turns == [] and conversation.counts == {} and conversation.last_bot_body == ""
    assert book.adopt("conv_x", merchant_id="somebody_else") is conversation  # idempotent: the first one wins
    assert book.get("conv_x") is conversation and len(book) == 1


def test_latest_for_finds_the_newest_conversation_the_bot_itself_opened(add_trigger, brief_for):
    book = ConversationBook()
    dip = brief_for(add_trigger("t1", "perf_dip", "m_den", DIP))
    spike = brief_for(add_trigger("t2", "perf_spike", "m_den", PLACEHOLDER))
    recall = brief_for(add_trigger("t3", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    book.open("older", dip, "first", at(9))
    book.open("newer", spike, "second", at(10))
    book.open("to_priya", recall, "reminder", at(11))
    book.adopt("adopted", merchant_id="m_den", at=at(12))  # no trigger behind it, so it is never "the one we opened"
    assert book.latest_for("m_den", None).id == "newer"
    assert book.latest_for("m_den", "c_priya").id == "to_priya"
    assert book.latest_for("m_den", "c_kid") is None and book.latest_for("m_ghost", None) is None


def test_a_closed_conversation_can_be_reopened(add_trigger, brief_for):
    book = ConversationBook()
    book.open("conv_1", brief_for(add_trigger("t1", "perf_dip", "m_den", DIP)), "hello", at(9))
    book.close("conv_1", "auto_reply")
    assert book.get("conv_1").status == "closed"
    book.reopen("conv_1")
    assert (book.get("conv_1").status, book.get("conv_1").closed_reason) == ("open", "")
    book.reopen("missing")  # nothing to reopen, nothing to fail


def test_bodies_to_covers_adopted_conversations_too_and_keeps_customers_apart(add_trigger, brief_for):
    book = ConversationBook()
    book.open("later", brief_for(add_trigger("t1", "perf_spike", "m_den", PLACEHOLDER)), "second", at(12))
    book.open("earlier", brief_for(add_trigger("t2", "perf_dip", "m_den", DIP)), "first", at(9))
    book.adopt("adopted", merchant_id="m_den", at=at(10))
    book.append("adopted", "vera", "in between", at(10))
    book.open("customer", brief_for(add_trigger("t3", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya")), "hi", at(11))
    assert book.bodies_to("m_den", None) == ["first", "in between", "second"]
    assert book.bodies_to("m_den", "c_priya") == ["hi"] and book.bodies_to("m_ghost", None) == []


def test_last_bot_body_is_the_latest_thing_vera_said(add_trigger, brief_for):
    book = ConversationBook()
    conversation = book.open("conv_1", brief_for(add_trigger("t1", "perf_dip", "m_den", DIP)), "hello", at(9))
    book.append("conv_1", "merchant", "yes", at(10))
    assert conversation.last_bot_body == "hello"
    book.append("conv_1", "vera", "done", at(11))
    assert conversation.last_bot_body == "done"


def test_clear_forgets_every_conversation(add_trigger, brief_for):
    book = ConversationBook()
    book.open("conv_1", brief_for(add_trigger("t1", "perf_dip", "m_den", DIP)), "hello", at(9))
    book.adopt("conv_2", merchant_id="m_sam")
    book.clear()
    assert len(book) == 0 and book.get("conv_1") is None and book.for_merchant("m_den") == []
