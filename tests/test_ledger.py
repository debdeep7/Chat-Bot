from datetime import datetime, timedelta, timezone

from vera.ledger import Ledger

T0 = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)


def test_a_sent_trigger_is_remembered(add_trigger, brief_for):
    ledger = Ledger()
    brief = brief_for(add_trigger("t1", "curious_ask_due", "m_den", {"ask_template": "x"}))
    assert not ledger.already_sent("t1", "m_den", None, brief.suppression_key)
    ledger.mark_sent(brief, T0)
    assert ledger.already_sent("t1", "m_den", None, None)


def test_a_suppression_key_only_suppresses_the_same_merchant(add_trigger, brief_for):
    # Category-wide triggers share one key across many merchants; one merchant's send must not hide the rest.
    ledger = Ledger()
    first = brief_for(add_trigger("t1", "research_digest", "m_den", PLACEHOLDER_DIGEST, suppression_key="research:2026-W17"))
    ledger.mark_sent(first, T0)
    assert ledger.already_sent("another_id", "m_den", None, "research:2026-W17")
    assert not ledger.already_sent("another_id", "m_sam", None, "research:2026-W17")


def test_only_merchant_facing_sends_start_a_merchant_cooldown(add_trigger, brief_for):
    ledger = Ledger()
    to_customer = brief_for(add_trigger("t1", "recall_due", "m_den", {"placeholder": True}, customer_id="c_priya"))
    ledger.mark_sent(to_customer, T0)
    assert ledger.last_merchant_send("m_den") is None

    to_merchant = brief_for(add_trigger("t2", "curious_ask_due", "m_den", {"ask_template": "x"}))
    ledger.mark_sent(to_merchant, T0)
    assert ledger.last_merchant_send("m_den") == T0


def test_a_trigger_that_keeps_producing_nothing_is_given_up_on():
    ledger = Ledger()
    assert not ledger.gave_up("t1")
    assert ledger.note_silence("t1") == 1 and not ledger.gave_up("t1")  # one more chance
    assert ledger.note_silence("t1") == 2 and ledger.gave_up("t1")
    assert not ledger.gave_up("t2")  # other triggers are unaffected


def test_opt_outs_are_recorded():
    ledger = Ledger()
    assert ledger.merchant_block("m_den") is None and ledger.customer_block("c_priya") is None
    ledger.block_merchant("m_den", "asked_to_stop")
    ledger.block_customer("c_priya")
    assert ledger.merchant_block("m_den") == "asked_to_stop"
    assert ledger.customer_block("c_priya") == "opted_out"


def test_a_snooze_never_shortens_and_is_per_merchant():
    ledger = Ledger()
    assert ledger.snoozed_until("m_den") is None
    ledger.snooze_merchant("m_den", T0 + timedelta(hours=5))
    ledger.snooze_merchant("m_den", T0 + timedelta(hours=1))  # a shorter pause never cuts a longer one short
    assert ledger.snoozed_until("m_den") == T0 + timedelta(hours=5)
    ledger.snooze_merchant("m_den", T0 + timedelta(hours=9))
    assert ledger.snoozed_until("m_den") == T0 + timedelta(hours=9)
    assert ledger.snoozed_until("m_sam") is None


def test_a_merchant_who_writes_to_us_is_released_from_any_pause_or_block():
    ledger = Ledger()
    ledger.snooze_merchant("m_den", T0 + timedelta(days=3))
    ledger.block_merchant("m_den", "asked_to_stop")
    ledger.block_customer("c_priya")
    ledger.release_merchant("m_den")
    assert ledger.snoozed_until("m_den") is None and ledger.merchant_block("m_den") is None
    assert ledger.customer_block("c_priya") == "opted_out"  # a customer's opt-out is theirs alone
    ledger.release_merchant("m_ghost")  # nothing to release, nothing to fail


def test_clear_forgets_everything(add_trigger, brief_for):
    ledger = Ledger()
    brief = brief_for(add_trigger("t1", "curious_ask_due", "m_den", {"ask_template": "x"}))
    ledger.mark_sent(brief, T0)
    ledger.note_silence("t2")
    ledger.note_silence("t2")
    ledger.block_merchant("m_den")
    ledger.block_customer("c_priya")
    ledger.snooze_merchant("m_den", T0 + timedelta(days=1))
    ledger.clear()
    assert not ledger.already_sent("t1", "m_den", None, brief.suppression_key)
    assert ledger.last_merchant_send("m_den") is None and not ledger.gave_up("t2")
    assert ledger.merchant_block("m_den") is None and ledger.customer_block("c_priya") is None
    assert ledger.snoozed_until("m_den") is None


PLACEHOLDER_DIGEST = {"category": "dentists", "top_item_id": "d_fluoride"}
