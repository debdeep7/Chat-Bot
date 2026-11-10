import pytest

from vera.facts import (
    active_offers, catalog_offer, customer_facts, months_in, move_fact, moves, numbers_in, offer_fact,
    peer_comparisons, pct, pick_digest, subscription_fact, when, words,
)


def test_pct_formats_fractions_as_percentages():
    assert pct(0.18) == "18%"
    assert pct(0.021) == "2.1%"
    assert pct(0.03) == "3%"
    assert pct(-0.5, signed=True) == "-50%"
    assert pct(0.15, signed=True) == "+15%"


@pytest.mark.parametrize("iso, expected", [
    ("2026-05-02T19:00:00+05:30", "2 May, 7pm"),
    ("2026-05-02T19:30:00+05:30", "2 May, 7:30pm"),
    ("2026-05-02T12:00:00+05:30", "2 May, 12pm"),
    ("2026-05-02", "2 May"),
    ("2026-04-28T00:00:00+05:30", "28 Apr"),  # midnight is a plain date, not "12am"
    ("not a date", None),
    (None, None),
])
def test_when_formats_dates_without_shifting_time_zones(iso, expected):
    assert when(iso) == expected


def test_words_makes_snake_case_readable():
    assert words("delivery_late") == "delivery late"
    assert words("6_month_cleaning") == "6-month cleaning"


def test_months_in_expands_ranges_including_year_wrap():
    assert months_in("Oct-Dec") == {"oct", "nov", "dec"}
    assert months_in("Nov-Feb") == {"nov", "dec", "jan", "feb"}
    assert months_in("Jan") == {"jan"}
    assert months_in("Feb 14") == {"feb"}


def test_numbers_in_finds_every_number_as_plain_digits():
    assert numbers_in("₹2,410 views, CTR 2.1%, 4 calls") == {"2410", "2.1", "4"}


def test_percentage_swings_on_a_tiny_base_are_ignored():
    merchant = {"performance": {"views": 756, "calls": 2, "delta_7d": {"views_pct": 0.26, "calls_pct": 0.23}}}
    assert [label for label, _ in moves(merchant)] == ["views"]
    assert move_fact(merchant, "up").text == "views up 26% over the last 7 days"
    assert move_fact(merchant, "down") is None


def test_small_moves_are_not_worth_mentioning():
    merchant = {"performance": {"views": 900, "calls": 30, "delta_7d": {"views_pct": 0.02, "calls_pct": -0.03}}}
    assert move_fact(merchant, "up") is None
    assert move_fact(merchant, "down") is None


def test_only_active_offers_can_be_claimed():
    merchant = {"offers": [{"title": "Cleaning @ ₹299", "status": "active"}, {"title": "Old deal", "status": "expired"}]}
    assert [o["title"] for o in active_offers(merchant)] == ["Cleaning @ ₹299"]
    assert offer_fact(merchant).text == "active offer: Cleaning @ ₹299"
    assert offer_fact({"offers": []}) is None


CATALOG = {"offer_catalog": [
    {"title": "3 FREE Trial Classes", "audience": "new_user"},
    {"title": "Haircut @ ₹99", "audience": "new_user"},
    {"title": "Senior Citizen 15% OFF", "audience": "senior"},
]}


def test_catalog_offer_ranks_by_match_and_can_require_one():
    assert catalog_offer(CATALOG, match=["haircut"]).text.endswith("Haircut @ ₹99")
    assert catalog_offer(CATALOG, match=["bridal"], require_match=True) is None
    assert catalog_offer(CATALOG, audience="senior").text.endswith("Senior Citizen 15% OFF")


def test_catalog_offer_skips_what_the_merchant_already_runs():
    fact = catalog_offer(CATALOG, match=["trial"], require_match=True, exclude=["3 free trial classes"])
    assert fact is None


def test_peer_comparisons_report_direction_and_gap():
    merchant = {"performance": {"ctr": 0.021, "calls": 18}}
    category = {"peer_stats": {"scope": "metro_solo_practices_2026", "avg_ctr": 0.03, "avg_calls_30d": 12}}
    by_key = {fact.key: (gap, fact) for gap, fact in peer_comparisons(merchant, category)}
    assert by_key["peer.ctr"][0] < 0 and "below peers" in by_key["peer.ctr"][1].text
    assert by_key["peer.calls"][0] > 0 and "above peers" in by_key["peer.calls"][1].text
    assert by_key["peer.ctr"][1].text == "CTR 2.1% vs 3% for metro solo practices (below peers)"


def test_subscription_fact_covers_active_trial_and_expired():
    assert subscription_fact({"subscription": {"status": "active", "plan": "Pro", "days_remaining": 82}}).text == "Pro subscription: 82 days remaining"
    assert subscription_fact({"subscription": {"status": "trial", "plan": "Trial", "days_remaining": 5}}).text == "free trial: 5 days left"
    assert subscription_fact({"subscription": {"status": "expired", "plan": "Pro", "days_since_expiry": 39}}).text == "Pro subscription expired 39 days ago"


def test_pick_digest_prefers_items_that_overlap_the_merchant_and_honours_avoid_kinds():
    category = {"digest": [
        {"id": "a", "kind": "tech", "title": "Crown lab pricing", "summary": "labs", "actionable": ""},
        {"id": "b", "kind": "trend", "title": "Whitening searches rise", "summary": "whitening demand", "actionable": ""},
        {"id": "c", "kind": "compliance", "title": "Whitening rule", "summary": "whitening whitening", "actionable": ""},
    ]}
    merchant = {"offers": [{"title": "Teeth Whitening @ ₹1,499"}]}
    assert pick_digest(category, merchant, avoid_kinds=("compliance",))["id"] == "b"


def test_customer_facts_use_plain_wording():
    customer = {
        "relationship": {"last_visit": "2026-04-01", "visits_total": 1, "services_received": ["chronic_rx_metformin", "..."]},
        "preferences": {"preferred_slots": "morning_delivery"},
    }
    texts = [f.text for f in customer_facts(customer)]
    assert texts == ["last visit: 1 Apr", "1 visit so far", "past services: chronic rx metformin", "prefers morning delivery slots"]
    assert customer_facts(None) == []
