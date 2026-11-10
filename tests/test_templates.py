"""The template writer is the floor of quality: whatever it writes must pass the same validator as the model."""
from dataclasses import replace

import pytest

from vera.facts import Fact
from vera.prompt import max_chars
from vera.templates import rationale, render
from vera.validate import check

PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}
SLOTS = [
    {"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"},
    {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"},
]
DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
RECALL = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
REFILL = {"molecule_list": ["metformin", "atorvastatin"], "last_refill": "2026-03-26",
          "stock_runs_out_iso": "2026-04-28T00:00:00+05:30", "delivery_address_saved": True}

# (kind, payload, merchant, customer): every shape of brief the fixture world can produce
CASES = [
    ("research_digest", {"category": "dentists", "top_item_id": "d_fluoride"}, "m_den", None),
    ("research_digest", PLACEHOLDER, "m_den", None),
    ("regulation_change", {"category": "dentists", "top_item_id": "d_radiograph", "deadline_iso": "2026-12-15"}, "m_den", None),
    ("cde_opportunity", {"category": "dentists", "top_item_id": "d_webinar"}, "m_den", None),
    ("supply_alert", {"alert_id": "d_recall", "molecule": "atorvastatin", "affected_batches": ["AT2024-1102"], "manufacturer": "MfrZ"}, "m_pha", None),
    ("perf_dip", DIP, "m_den", None),
    ("perf_dip", PLACEHOLDER, "m_sam", None),
    ("perf_spike", PLACEHOLDER, "m_den", None),
    ("competitor_opened", {"competitor_name": "Smile Studio", "distance_km": 1.3, "their_offer": "Dental Cleaning @ ₹199", "opened_date": "2026-04-08"}, "m_den", None),
    ("competitor_opened", PLACEHOLDER, "m_den", None),
    ("ipl_match_today", {"match": "DC vs MI", "venue": "Arun Jaitley Stadium", "match_time_iso": "2026-04-26T19:30:00+05:30", "is_weeknight": False}, "m_res", None),
    ("festival_upcoming", {"festival": "Diwali", "date": "2026-10-31", "days_until": 188}, "m_den", None),
    ("festival_upcoming", PLACEHOLDER, "m_den", None),
    ("mystery_event", {"headline": "Road closed near clinic", "days": 3}, "m_den", None),
    ("curious_ask_due", {"ask_template": "x"}, "m_den", None),
    ("dormant_with_vera", {"days_since_last_merchant_message": 38, "last_topic": "subscription expiry"}, "m_den", None),
    ("dormant_with_vera", PLACEHOLDER, "m_sam", None),
    ("active_planning_intent", {"intent_topic": "kids yoga camp", "merchant_last_message": "Yes, what would it look like"}, "m_den", None),
    ("recall_due", RECALL, "m_den", "c_priya"),
    ("recall_due", PLACEHOLDER, "m_den", "c_kid"),
    ("recall_due", PLACEHOLDER, "m_den", "c_ghost"),
    ("appointment_tomorrow", PLACEHOLDER, "m_den", "c_priya"),
    ("chronic_refill_due", REFILL, "m_pha", "c_sharma"),
    ("chronic_refill_due", {"molecule_list": ["metformin"]}, "m_den", "c_priya"),
    ("customer_lapsed_soft", PLACEHOLDER, "m_den", "c_priya"),
    ("customer_lapsed_hard", {"days_since_last_visit": 57, "previous_focus": "weight_loss", "previous_membership_months": 5}, "m_den", "c_priya"),
]


@pytest.fixture
def make(add_trigger, brief_for):
    counter = iter(range(1000))

    def build(kind, payload, merchant, customer=None):
        return brief_for(add_trigger(f"t{next(counter)}", kind, merchant, payload, customer_id=customer))
    return build


@pytest.mark.parametrize("kind, payload, merchant, customer", CASES, ids=[f"{c[0]}-{c[3] or c[2]}" for c in CASES])
def test_every_template_passes_the_same_checks_as_the_model(make, kind, payload, merchant, customer):
    brief = make(kind, payload, merchant, customer)
    body = render(brief)
    assert body, f"no template message for {kind}"
    assert check(body, brief) == []
    assert len(body) <= max_chars(brief)


def test_the_language_follows_the_recipient(make):
    english = render(make("perf_dip", DIP, "m_sam"))
    hinglish = render(make("perf_dip", DIP, "m_den"))
    assert "Kya main" not in english and "Kya main" in hinglish


def test_customer_messages_speak_as_the_business_and_never_mention_vera(make):
    body = render(make("recall_due", RECALL, "m_den", "c_priya"))
    assert body.startswith("Namaste Priya, Dr. Meera's Dental Clinic")
    assert "Vera" not in body and "magicpin" not in body.lower()


def test_slots_are_offered_verbatim_with_the_live_offer(make):
    body = render(make("recall_due", RECALL, "m_den", "c_priya"))
    assert "Wed 5 Nov, 6pm" in body and "Thu 6 Nov, 5pm" in body
    assert "Dental Cleaning @ ₹299" in body
    assert body.rstrip(".").endswith("bata dijiye")


def test_three_slots_become_a_numbered_menu_that_still_passes_the_checks(make):
    three = [*SLOTS, {"iso": "2026-11-07T11:00:00+05:30", "label": "Fri 7 Nov, 11am"}]
    brief = make("recall_due", {**RECALL, "available_slots": three}, "m_den", "c_priya")
    body = render(brief)
    assert "1 for Wed 5 Nov, 6pm" in body and "3 for Fri 7 Nov, 11am" in body
    assert check(body, brief) == []


def test_a_relatives_phone_gets_a_respectful_third_person_message(make):
    body = render(make("chronic_refill_due", REFILL, "m_pha", "c_sharma"))
    assert "Mr. Sharma" in body and "metformin, atorvastatin" in body
    assert "CONFIRM" in body


def test_a_childs_message_is_addressed_to_the_parent_and_names_the_child(make):
    body = render(make("recall_due", PLACEHOLDER, "m_den", "c_kid"))
    assert "Sneha" in body and "Aanya" in body


def test_refill_reminders_carry_no_promotion(make):
    body = render(make("chronic_refill_due", REFILL, "m_pha", "c_sharma"))
    assert "currently have" not in body and "Senior Citizen" not in body


def test_a_trigger_without_details_does_not_claim_any(make):
    competitor = render(make("competitor_opened", PLACEHOLDER, "m_den"))
    assert "Smile Studio" not in competitor and "km" not in competitor
    # The milestone-specific ask ("get there") would refer to a milestone nobody named.
    real = render(make("milestone_reached", {"metric": "reviews", "value_now": 145, "milestone_value": 150}, "m_sam"))
    vague = render(make("milestone_reached", PLACEHOLDER, "m_sam"))
    assert "get there" in real and "145" in real
    assert "get there" not in vague


def test_jargon_in_a_fact_is_translated(make):
    body = render(make("perf_dip", PLACEHOLDER, "m_den"))
    assert "CTR" not in body


def test_dormant_merchants_are_not_guilt_tripped(make):
    body = render(make("dormant_with_vera", {"days_since_last_merchant_message": 38, "last_topic": "x"}, "m_den"))
    assert "38" not in body and "days since" not in body


def test_a_different_variant_picks_a_different_supporting_fact(make):
    brief = make("perf_dip", DIP, "m_den")
    assert len(brief.support) >= 2
    assert render(brief, 0) != render(brief, 1)
    assert render(brief, 0) == render(brief, 0)  # deterministic


def test_nothing_is_written_without_a_lead(make):
    assert render(replace(make("perf_dip", DIP, "m_den"), lead=None)) is None


def test_an_overlong_lead_is_cut_at_a_sentence_and_keeps_the_ask(make):
    brief = make("research_digest", {"category": "dentists", "top_item_id": "d_fluoride"}, "m_den")
    long_lead = " ".join(f"Sentence number {n} is here." for n in range(60))
    body = render(replace(brief, lead=Fact("digest.item", long_lead)))
    assert len(body) <= max_chars(brief)
    assert body.endswith("YES bhej dijiye.") and "Sentence number 0 is here." in body


def test_rationale_says_why_now_and_stays_short(make):
    brief = make("perf_dip", DIP, "m_den")
    text = rationale(brief)
    assert text.startswith("Why now:") and "calls down 50%" in text and len(text) < 420
