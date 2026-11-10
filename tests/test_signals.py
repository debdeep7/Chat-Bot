from pathlib import Path

import pytest

from vera.dataset import load_store
from vera.signals import build_brief, customer_address, customer_language, merchant_language, merchant_salutation

PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}
SLOTS = [
    {"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"},
    {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"},
]
DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}


# --------------------------------------------------------------------------- merchant-facing leads

def test_research_digest_leads_with_the_named_item(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "research_digest", "m_den", {"category": "dentists", "top_item_id": "d_fluoride"}))
    assert brief.kind == "research_digest" and brief.send_as == "vera" and brief.customer_id is None
    assert (brief.salutation, brief.language, brief.cta) == ("Dr. Meera", "hi-en", "binary_yes_no")
    assert "fluoride varnish" in brief.lead.text and "JIDA Oct 2026, p.14" in brief.lead.text
    assert {"digest.action", "agg.high_risk_adult_count"} <= {f.key for f in brief.support}
    assert not brief.degraded and not brief.placeholder


def test_placeholder_research_digest_picks_the_most_relevant_item(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "research_digest", "m_den", PLACEHOLDER))
    assert "fluoride" in brief.lead.text
    assert brief.placeholder and not brief.degraded
    assert any("named no digest item" in note for note in brief.notes)


def test_regulation_change_carries_its_deadline(add_trigger, brief_for):
    payload = {"category": "dentists", "top_item_id": "d_radiograph", "deadline_iso": "2026-12-15"}
    brief = brief_for(add_trigger("t", "regulation_change", "m_den", payload, urgency=4))
    assert "radiograph" in brief.lead.text and brief.urgency == 4
    assert "deadline: 15 Dec" in [f.text for f in brief.support]


def test_supply_alert_names_molecule_batches_and_manufacturer(add_trigger, brief_for):
    payload = {"alert_id": "d_recall", "molecule": "atorvastatin", "affected_batches": ["AT2024-1102", "AT2024-1108"], "manufacturer": "MfrZ"}
    brief = brief_for(add_trigger("t", "supply_alert", "m_pha", payload, urgency=5))
    assert brief.lead.text == "atorvastatin recall, batches AT2024-1102, AT2024-1108, from MfrZ (source: CDSCO alert Apr 2026)"
    assert "60 chronic-prescription customers on your records" in [f.text for f in brief.support]


def test_perf_dip_with_numbers_leads_with_them(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "perf_dip", "m_den", DIP, urgency=4))
    assert brief.lead.text == "calls down 50% over the last 7 days, against a baseline of 12"
    assert not brief.degraded


def test_perf_dip_without_evidence_does_not_claim_a_dip(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "perf_dip", "m_sam", PLACEHOLDER))
    assert brief.degraded
    assert brief.lead.key == "peer.calls"  # its clearest gap against peers, stated as a gap
    assert any("do not claim one" in note for note in brief.notes)
    assert "Pro subscription expired 39 days ago" in [f.text for f in brief.support]


def test_perf_spike_without_numbers_uses_the_merchants_own_rise(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "perf_spike", "m_den", PLACEHOLDER))
    assert brief.lead.text == "views up 18% over the last 7 days"
    assert not brief.degraded


def test_competitor_with_details_is_stated_factually(add_trigger, brief_for):
    payload = {"competitor_name": "Smile Studio", "distance_km": 1.3, "their_offer": "Dental Cleaning @ ₹199", "opened_date": "2026-04-08"}
    brief = brief_for(add_trigger("t", "competitor_opened", "m_den", payload))
    assert brief.lead.text == "Smile Studio opened 1.3 km away on 8 Apr, offering Dental Cleaning @ ₹199"
    assert brief.cta == "open_ended"
    assert any("never disparage" in note for note in brief.notes)


def test_competitor_without_details_names_no_one(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "competitor_opened", "m_den", PLACEHOLDER))
    assert brief.lead.key == "trigger.kind" and brief.degraded
    assert any("do not name a competitor" in note for note in brief.notes)


def test_ipl_off_weeknight_warns_against_a_dine_in_rush(add_trigger, brief_for):
    payload = {"match": "DC vs MI", "venue": "Arun Jaitley Stadium", "city": "Delhi",
               "match_time_iso": "2026-04-26T19:30:00+05:30", "is_weeknight": False}
    brief = brief_for(add_trigger("t", "ipl_match_today", "m_res", payload, urgency=3))
    assert brief.lead.text == "DC vs MI at Arun Jaitley Stadium, 26 Apr, 7:30pm"
    assert any("NOT on a weeknight" in note for note in brief.notes)
    assert any("12%" in f.text for f in brief.support)


def test_far_off_festival_warns_against_false_urgency(add_trigger, brief_for):
    payload = {"festival": "Diwali", "date": "2026-10-31", "days_until": 188, "category_relevance": ["salons"]}
    brief = brief_for(add_trigger("t", "festival_upcoming", "m_den", payload))
    assert brief.lead.text == "Diwali on 31 Oct (188 days away)"
    notes = " ".join(brief.notes)
    assert "188 days away" in notes and "not marked relevant" in notes
    assert any(f.key == "category.season" for f in brief.support)


def test_festival_without_a_name_names_no_festival(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "festival_upcoming", "m_den", PLACEHOLDER))
    assert brief.degraded
    assert any("do not name one" in note for note in brief.notes)


def test_unknown_kind_still_gets_a_brief_from_its_payload(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "mystery_event", "m_den", {"headline": "Road closed near clinic", "days": 3}))
    assert brief.kind == "generic_merchant" and brief.raw_kind == "mystery_event"
    assert brief.lead.text == "headline: Road closed near clinic"


# --------------------------------------------------------------------------- customer-facing leads

def test_recall_due_offers_the_given_slots_verbatim(add_trigger, brief_for):
    payload = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
    brief = brief_for(add_trigger("t", "recall_due", "m_den", payload, customer_id="c_priya", urgency=3))
    assert brief.send_as == "merchant_on_behalf" and brief.customer_id == "c_priya"
    assert (brief.salutation, brief.language, brief.cta) == ("Priya", "hi-en", "multi_choice_slot")
    assert brief.lead.text == "6-month cleaning due 12 Nov; last done 12 May"
    texts = [f.text for f in brief.support]
    assert texts[0] == "slots available: Wed 5 Nov, 6pm | Thu 6 Nov, 5pm"
    assert "active offer: Dental Cleaning @ ₹299" in texts


def test_recall_without_slots_falls_back_to_a_yes_no_ask(add_trigger, brief_for):
    payload = {"service_due": "6_month_cleaning", "due_date": "2026-11-12"}
    assert brief_for(add_trigger("t", "recall_due", "m_den", payload, customer_id="c_priya")).cta == "binary_yes_no"


def test_placeholder_customer_trigger_leads_with_the_kind_statement(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    assert brief.lead.key == "trigger.kind" and brief.degraded and brief.placeholder
    assert brief.cta == "binary_confirm_cancel"
    assert any("do not invent" in note for note in brief.notes)


def test_child_records_are_addressed_to_the_parent(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "recall_due", "m_den", PLACEHOLDER, customer_id="c_kid"))
    assert (brief.salutation, brief.about) == ("Sneha", "Aanya")


def test_elder_reached_through_family_gets_respectful_notes(add_trigger, brief_for):
    payload = {"molecule_list": ["metformin", "atorvastatin"], "last_refill": "2026-03-26",
               "stock_runs_out_iso": "2026-04-28T00:00:00+05:30", "delivery_address_saved": True}
    brief = brief_for(add_trigger("t", "chronic_refill_due", "m_pha", payload, customer_id="c_sharma", urgency=3))
    assert brief.lead.text == "metformin, atorvastatin: last refilled 26 Mar, supply runs out 28 Apr"
    assert (brief.salutation, brief.language) == ("Mr. Sharma", "hi")
    assert any("customer's son" in note for note in brief.notes)
    assert any("Senior" in note for note in brief.notes)
    assert "a delivery address is saved, so home delivery is possible" in [f.text for f in brief.support]


def test_kind_that_does_not_fit_the_category_becomes_a_general_check_in(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "chronic_refill_due", "m_den", {"molecule_list": ["metformin"]}, customer_id="c_priya"))
    assert brief.kind == "generic_customer" and brief.degraded
    assert "metformin" not in " ".join(f.text for f in brief.facts)
    assert any("does not fit" in note for note in brief.notes)


def test_notes_are_scoped_to_the_audience(add_trigger, brief_for):
    to_merchant = brief_for(add_trigger("t1", "curious_ask_due", "m_den", {"ask_template": "x"}))
    to_customer = brief_for(add_trigger("t2", "recall_due", "m_den", PLACEHOLDER, customer_id="c_priya"))
    assert any("earlier conversation" in note for note in to_merchant.notes)
    assert not any("earlier conversation" in note for note in to_customer.notes)
    assert to_merchant.prior_bodies == ("Profile audit done.",)
    assert to_customer.prior_bodies == ()  # Vera's messages to the merchant were never sent to the customer


# --------------------------------------------------------------------------- when a brief cannot be built

def test_missing_contexts_are_reported_not_raised(store, add_trigger):
    assert build_brief({"id": "t", "kind": "perf_dip", "merchant_id": "m_unknown"}, store) == (None, "merchant_context_missing")
    store.put("merchant", "m_orphan", 1, {"merchant_id": "m_orphan", "category_slug": "no_such_category", "identity": {}})
    assert build_brief({"id": "t", "kind": "perf_dip", "merchant_id": "m_orphan"}, store) == (None, "category_context_missing")
    add_trigger("t2", "recall_due", "m_den", PLACEHOLDER)  # customer-facing kind, but no customer id
    assert build_brief(store.get("trigger", "t2"), store, "t2") == (None, "customer_id_missing")


def test_a_customer_who_was_never_pushed_still_gets_a_modest_brief(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "recall_due", "m_den", PLACEHOLDER, customer_id="c_ghost"))
    assert brief.customer is None and brief.degraded and brief.salutation == ""
    assert brief.language == "hi-en"  # falls back to the merchant's language
    assert any("details were not provided" in note for note in brief.notes)


# --------------------------------------------------------------------------- who and how

@pytest.mark.parametrize("identity, slug, expected", [
    ({"owner_first_name": "Meera"}, "dentists", "Dr. Meera"),
    ({"owner_first_name": "Dr. Sameer"}, "dentists", "Dr. Sameer"),  # never "Dr. Dr."
    ({"owner_first_name": "Drew"}, "dentists", "Dr. Drew"),
    ({"owner_first_name": "Karim"}, "salons", "Karim"),
    ({"name": "Glow Salon"}, "salons", "Glow Salon team"),
])
def test_merchant_salutation(identity, slug, expected):
    assert merchant_salutation({"identity": identity}, slug) == expected


@pytest.mark.parametrize("name, expected", [
    ("Aanya (parent: Sneha)", ("Sneha", "Aanya")),
    ("(walk-in, no profile)", ("", None)),
    ("Mr. Sharma", ("Mr. Sharma", None)),
    ("Priya", ("Priya", None)),
])
def test_customer_address(name, expected):
    assert customer_address({"identity": {"name": name}}) == expected
    assert customer_address(None) == ("", None)


@pytest.mark.parametrize("pref, expected", [
    ("en", "en"), ("english", "en"), ("hi", "hi"), ("hi-en mix", "hi-en"), ("ta-en mix", "ta-en"),
    ("", "FALLBACK"), ("fr", "FALLBACK"),
])
def test_customer_language(pref, expected):
    assert customer_language({"identity": {"language_pref": pref}}, "FALLBACK") == expected


def test_merchant_language():
    assert merchant_language({"identity": {"languages": ["en", "hi", "mr"]}}) == "hi-en"
    assert merchant_language({"identity": {"languages": ["en"]}}) == "en"
    assert merchant_language({"identity": {}}) == "en"


# --------------------------------------------------------------------------- the real dataset

DATA = Path(__file__).resolve().parent.parent / "data" / "expanded"


@pytest.mark.skipif(not DATA.exists(), reason="run generate_dataset.py first (data/ is not in git)")
def test_every_trigger_in_the_generated_dataset_builds_a_sane_brief():
    store = load_store(DATA)
    assert store.counts()["trigger"] == 100
    for trigger_id in store.ids("trigger"):
        brief, why = build_brief(store.get("trigger", trigger_id), store, trigger_id)
        assert brief is not None, (trigger_id, why)
        assert brief.lead is not None, trigger_id
        assert len(brief.support) <= 3
        assert brief.send_as in ("vera", "merchant_on_behalf")
        assert (brief.customer_id is not None) == (brief.send_as == "merchant_on_behalf")
        assert brief.cta in {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}
        assert "Dr. Dr." not in brief.salutation
