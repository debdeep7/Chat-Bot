import pytest

from vera.leads import _BUILDERS
from vera.router import (
    GENERIC_CUSTOMER, GENERIC_MERCHANT, STRATEGIES, canonical_kind, fits_category, route,
)

CTA_TYPES = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}


def test_every_strategy_is_well_formed():
    for kind, strategy in STRATEGIES.items():
        assert strategy.kind == kind
        assert strategy.audience in ("merchant", "customer")
        assert strategy.cta in CTA_TYPES
        assert strategy.levers and strategy.goal


def test_every_kind_has_a_lead_builder():
    # Guards against adding a router row and forgetting its builder.
    assert set(STRATEGIES) <= set(_BUILDERS)


@pytest.mark.parametrize("kind, audience, send_as", [
    ("research_digest", "merchant", "vera"),
    ("perf_dip", "merchant", "vera"),
    ("recall_due", "customer", "merchant_on_behalf"),
    ("chronic_refill_due", "customer", "merchant_on_behalf"),
    ("customer_lapsed_hard", "customer", "merchant_on_behalf"),
])
def test_known_kinds_pick_the_right_audience(kind, audience, send_as):
    strategy = route({"kind": kind})
    assert (strategy.audience, strategy.send_as) == (audience, send_as)


def test_slot_and_confirm_kinds_get_matching_ctas():
    assert route({"kind": "recall_due"}).cta == "multi_choice_slot"
    assert route({"kind": "appointment_tomorrow"}).cta == "binary_confirm_cancel"
    assert route({"kind": "curious_ask_due"}).cta == "open_ended"


@pytest.mark.parametrize("raw, expected", [
    ("research_digest_release", "research_digest"),  # the name used in the challenge brief
    ("Perf Dip", "perf_dip"),
    ("PERF-SPIKE", "perf_spike"),
    ("customer_recall_reminder", "recall_due"),
    ("supply_recall_notice", "supply_alert"),
    ("gst_compliance_update", "regulation_change"),
    ("weekly_triple_offer", None),  # "ipl" inside "triple" must not match
    ("", None),
    (None, None),
])
def test_kind_names_are_normalised_aliased_and_inferred(raw, expected):
    assert canonical_kind(raw) == expected


def test_unknown_kinds_fall_back_by_scope():
    assert route({"kind": "mystery_event", "scope": "customer"}) is GENERIC_CUSTOMER
    assert route({"kind": "mystery_event", "customer_id": "c1"}) is GENERIC_CUSTOMER
    assert route({"kind": "mystery_event", "scope": "merchant"}) is GENERIC_MERCHANT
    assert route({}) is GENERIC_MERCHANT


def test_category_bound_kinds():
    assert fits_category("chronic_refill_due", "pharmacies")
    assert not fits_category("chronic_refill_due", "dentists")
    assert not fits_category("ipl_match_today", "salons")
    assert fits_category("perf_dip", "dentists")
    assert fits_category("recall_due", None)
