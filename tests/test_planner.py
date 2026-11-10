from datetime import datetime, timedelta, timezone

from vera.ledger import Ledger
from vera.planner import Skip, parse_time, plan_tick

T0 = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)
DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
ASK = {"ask_template": "what_service_in_demand_this_week"}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}


def ids(plan):
    return [brief.trigger_id for brief in plan.briefs]


def test_higher_urgency_goes_first(store, add_trigger):
    add_trigger("low", "curious_ask_due", "m_den", ASK, urgency=1)
    add_trigger("high", "perf_dip", "m_sam", DIP, urgency=4)
    add_trigger("mid", "category_seasonal", "m_pha", {"season": "summer_2026", "trends": ["ORS_demand_+40"]}, urgency=2)
    assert ids(plan_tick(store, Ledger(), ["low", "high", "mid"])) == ["high", "mid", "low"]


def test_real_details_beat_a_placeholder_at_equal_urgency(store, add_trigger):
    add_trigger("placeholder", "dormant_with_vera", "m_pha", PLACEHOLDER, urgency=2)
    add_trigger("real", "curious_ask_due", "m_den", ASK, urgency=2)
    assert ids(plan_tick(store, Ledger(), ["placeholder", "real"])) == ["real", "placeholder"]


def test_one_merchant_facing_message_per_merchant_per_tick(store, add_trigger):
    add_trigger("a", "perf_dip", "m_den", DIP, urgency=3)
    add_trigger("b", "curious_ask_due", "m_den", ASK, urgency=2)
    plan = plan_tick(store, Ledger(), ["a", "b"])
    assert ids(plan) == ["a"]
    assert plan.skipped == [Skip("b", "merchant_already_messaged_this_tick")]


def test_customer_messages_are_separate_from_the_merchants_own(store, add_trigger):
    add_trigger("to_merchant", "curious_ask_due", "m_den", ASK)
    add_trigger("to_priya", "recall_due", "m_den", PLACEHOLDER, customer_id="c_priya", urgency=3)
    add_trigger("to_kid", "recall_due", "m_den", PLACEHOLDER, customer_id="c_kid")
    add_trigger("priya_again", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya", urgency=2)
    plan = plan_tick(store, Ledger(), ["to_merchant", "to_priya", "to_kid", "priya_again"])
    # the merchant and two different customers can all be messaged in one tick; a second message to Priya cannot
    assert sorted(ids(plan)) == ["to_kid", "to_merchant", "to_priya"]
    assert plan.skipped == [Skip("priya_again", "customer_already_messaged_this_tick")]


def test_merchant_cooldown_applies_unless_the_trigger_is_urgent(store, add_trigger, brief_for):
    ledger = Ledger()
    ledger.mark_sent(brief_for(add_trigger("first", "curious_ask_due", "m_den", ASK)), T0)
    add_trigger("normal", "perf_dip", "m_den", DIP, urgency=3)
    add_trigger("urgent", "perf_dip", "m_den", DIP, urgency=4, suppression_key="other")

    soon = T0 + timedelta(minutes=5)
    assert plan_tick(store, ledger, ["normal"], soon).skipped == [Skip("normal", "merchant_cooldown")]
    assert ids(plan_tick(store, ledger, ["urgent"], soon)) == ["urgent"]
    assert ids(plan_tick(store, ledger, ["normal"], T0 + timedelta(minutes=16))) == ["normal"]
    assert ids(plan_tick(store, ledger, ["normal"], T0 - timedelta(hours=1))) == ["normal"]  # clock reset
    assert ids(plan_tick(store, ledger, ["normal"], None)) == ["normal"]  # no clock, no cooldown


def test_a_trigger_is_never_acted_on_twice(store, add_trigger, brief_for):
    ledger = Ledger()
    brief = brief_for(add_trigger("t", "curious_ask_due", "m_den", ASK))
    assert ids(plan_tick(store, ledger, ["t"])) == ["t"]  # planning alone does not mark anything as sent
    ledger.mark_sent(brief, T0)
    assert plan_tick(store, ledger, ["t"]).skipped == [Skip("t", "already_sent")]


def test_a_trigger_that_kept_producing_nothing_is_no_longer_planned(store, add_trigger):
    ledger = Ledger()
    add_trigger("t", "curious_ask_due", "m_den", ASK)
    ledger.note_silence("t")
    assert ids(plan_tick(store, ledger, ["t"])) == ["t"]  # one silent attempt is forgiven
    ledger.note_silence("t")
    assert plan_tick(store, ledger, ["t"]).skipped == [Skip("t", "gave_up")]


def test_a_shared_suppression_key_does_not_block_other_merchants(store, add_trigger, brief_for):
    ledger = Ledger()
    ledger.mark_sent(brief_for(add_trigger("a", "curious_ask_due", "m_den", ASK, suppression_key="shared")), T0)
    add_trigger("b", "curious_ask_due", "m_pha", ASK, suppression_key="shared")
    assert ids(plan_tick(store, ledger, ["b"])) == ["b"]


def test_unusable_triggers_are_skipped_with_a_reason(store):
    store.put("trigger", "orphan", 1, {"id": "orphan", "kind": "perf_dip", "merchant_id": "m_ghost", "payload": {}})
    plan = plan_tick(store, Ledger(), ["never_pushed", "orphan"])
    assert plan.briefs == []
    assert plan.skipped == [Skip("never_pushed", "trigger_not_pushed"), Skip("orphan", "merchant_context_missing")]


def test_opted_out_people_are_left_alone(store, add_trigger):
    ledger = Ledger()
    add_trigger("merchant", "curious_ask_due", "m_den", ASK)
    add_trigger("declined", "recall_due", "m_den", PLACEHOLDER, customer_id="c_optout")
    add_trigger("promo", "customer_lapsed_soft", "m_den", PLACEHOLDER, customer_id="c_optout")  # not a reminder
    add_trigger("unreachable", "recall_due", "m_den", PLACEHOLDER, customer_id="c_noreach")
    add_trigger("blocked", "recall_due", "m_den", PLACEHOLDER, customer_id="c_priya")
    ledger.block_customer("c_priya")

    plan = plan_tick(store, ledger, ["merchant", "declined", "promo", "unreachable", "blocked"])
    assert sorted(ids(plan)) == ["merchant", "promo"]
    assert {s.trigger_id: s.reason for s in plan.skipped} == {
        "declined": "customer_declined_reminders",
        "unreachable": "customer_unreachable",
        "blocked": "customer_opted_out",
    }

    ledger.block_merchant("m_den")
    assert plan_tick(store, ledger, ["merchant"]).skipped == [Skip("merchant", "merchant_opted_out")]


def test_the_tick_is_capped(store, add_trigger):
    for urgency, merchant in enumerate(["m_den", "m_sam", "m_pha", "m_res"], start=1):
        add_trigger(f"t{urgency}", "curious_ask_due", merchant, ASK, urgency=urgency)
    plan = plan_tick(store, Ledger(), ["t1", "t2", "t3", "t4"], max_actions=2)
    assert ids(plan) == ["t4", "t3"]
    assert plan.skipped == [Skip("t2", "tick_cap_reached"), Skip("t1", "tick_cap_reached")]


def test_repeated_ids_are_handled_once(store, add_trigger):
    add_trigger("t", "curious_ask_due", "m_den", ASK)
    assert ids(plan_tick(store, Ledger(), ["t", "t", "t"])) == ["t"]


def test_parse_time():
    assert parse_time("2026-04-26T10:00:00Z") == T0
    assert parse_time("2026-04-26T10:00:00").tzinfo is not None  # naive input is taken as UTC
    assert parse_time("garbage") is None and parse_time(None) is None


def test_a_snoozed_merchant_is_left_alone_until_the_pause_ends(store, add_trigger):
    ledger = Ledger()
    ledger.snooze_merchant("m_den", T0 + timedelta(hours=2))
    add_trigger("t", "perf_dip", "m_den", DIP, urgency=4)  # even an urgent trigger waits
    assert plan_tick(store, ledger, ["t"], T0).skipped == [Skip("t", "merchant_snoozed")]
    assert plan_tick(store, ledger, ["t"], T0 + timedelta(hours=1, minutes=59)).skipped == [Skip("t", "merchant_snoozed")]
    assert ids(plan_tick(store, ledger, ["t"], T0 + timedelta(hours=2, minutes=1))) == ["t"]
    assert ids(plan_tick(store, ledger, ["t"], None)) == ["t"]  # no clock, no way to tell how long ago: carry on


def test_a_snooze_covers_the_merchant_only_not_their_customers_or_other_merchants(store, add_trigger):
    ledger = Ledger()
    ledger.snooze_merchant("m_den", T0 + timedelta(days=1))
    add_trigger("to_merchant", "curious_ask_due", "m_den", ASK)
    add_trigger("to_customer", "recall_due", "m_den", PLACEHOLDER, customer_id="c_priya", urgency=3)
    add_trigger("other_merchant", "curious_ask_due", "m_sam", ASK)
    plan = plan_tick(store, ledger, ["to_merchant", "to_customer", "other_merchant"], T0)
    assert sorted(ids(plan)) == ["other_merchant", "to_customer"]
    assert plan.skipped == [Skip("to_merchant", "merchant_snoozed")]


def test_a_snooze_set_by_a_wild_clock_does_not_last_forever(store, add_trigger):
    ledger = Ledger()
    ledger.snooze_merchant("m_den", T0 + timedelta(days=400))  # the judge's clock jumped ahead and was later reset
    add_trigger("t", "perf_dip", "m_den", DIP)
    assert ids(plan_tick(store, ledger, ["t"], T0)) == ["t"]


def test_an_opt_out_outranks_a_snooze(store, add_trigger):
    ledger = Ledger()
    ledger.snooze_merchant("m_den", T0 + timedelta(hours=1))
    ledger.block_merchant("m_den", "asked_to_stop")
    add_trigger("t", "perf_dip", "m_den", DIP)
    assert plan_tick(store, ledger, ["t"], T0).skipped == [Skip("t", "merchant_opted_out")]
