import pytest
from fastapi.testclient import TestClient

from vera.app import create_app
from vera.config import Settings
from vera.signals import build_brief
from vera.store import ContextStore


@pytest.fixture
def app():
    # Settings() uses plain defaults, so tests never depend on a developer's .env or API key.
    return create_app(Settings())


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- a small world
# Just enough of each context, in the real shapes, to exercise the brief and planner logic.

def _dentists() -> dict:
    return {
        "slug": "dentists",
        "peer_stats": {"scope": "metro_solo_practices_2026", "avg_ctr": 0.030, "avg_calls_30d": 12},
        "offer_catalog": [
            {"id": "den_001", "title": "Dental Cleaning @ ₹299", "audience": "new_user"},
            {"id": "den_002", "title": "Free Consultation", "audience": "new_user"},
        ],
        "digest": [
            {"id": "d_fluoride", "kind": "research", "title": "3-month fluoride varnish recall beats 6-month for high-risk adults",
             "source": "JIDA Oct 2026, p.14", "trial_n": 2100, "patient_segment": "high_risk_adults",
             "summary": "38% lower caries recurrence in a 2,100-patient trial.",
             "actionable": "Reassess recall interval for high-risk adults"},
            {"id": "d_radiograph", "kind": "compliance", "title": "DCI revised radiograph dose limits effective 2026-12-15",
             "source": "DCI circular", "summary": "Dose limit drops from 1.5 mSv to 1.0 mSv.",
             "actionable": "Audit your X-ray setup before Dec 15"},
            {"id": "d_webinar", "kind": "cde", "title": "IDA Delhi: Digital impressions", "source": "IDA Delhi chapter calendar",
             "date": "2026-05-02T19:00:00+05:30", "credits": 2, "summary": "Covers CAD/CAM workflow ROI.",
             "actionable": "Free for IDA members; ₹500 for non-members"},
        ],
        "seasonal_beats": [{"month_range": "Oct-Dec", "note": "wedding whitening peak, bookings 2x baseline"}],
    }


def _pharmacies() -> dict:
    return {
        "slug": "pharmacies",
        "peer_stats": {"scope": "metro_neighbourhood_pharmacies_2026", "avg_ctr": 0.038, "avg_calls_30d": 22},
        "offer_catalog": [
            {"id": "phr_002", "title": "Free Home Delivery > ₹499", "audience": "new_user"},
            {"id": "phr_005", "title": "Senior Citizen 15% OFF (60+ age)", "audience": "senior"},
        ],
        "digest": [
            {"id": "d_recall", "kind": "alert", "title": "Voluntary recall: specific atorvastatin batches",
             "source": "CDSCO alert Apr 2026", "summary": "Two batches flagged for sub-potency.",
             "actionable": "Pull the batches and tell affected customers"},
        ],
        "seasonal_beats": [],
    }


def _restaurants() -> dict:
    return {
        "slug": "restaurants",
        "peer_stats": {"scope": "metro_casual_dining_2026", "avg_ctr": 0.025, "avg_calls_30d": 38},
        "offer_catalog": [{"id": "res_005", "title": "Match-night Combo @ ₹399 (food + drink)", "audience": "new_user"}],
        "digest": [
            {"id": "d_ipl", "kind": "seasonal", "title": "IPL home-match Saturdays underperformed weeknight matches",
             "source": "magicpin order data, Apr 2026",
             "summary": "Saturday matches: covers down 12%. Weeknight matches drive +18% covers.",
             "actionable": "Push match-night combos on Tue/Wed/Thu match dates only"},
        ],
        "seasonal_beats": [],
    }


def _merchant(merchant_id: str, slug: str, owner: str, name: str, **overrides) -> dict:
    merchant = {
        "merchant_id": merchant_id,
        "category_slug": slug,
        "identity": {"name": name, "city": "Delhi", "locality": "Lajpat Nagar", "owner_first_name": owner, "languages": ["en", "hi"]},
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {"window_days": 30, "views": 1200, "calls": 30, "directions": 60, "ctr": 0.04,
                        "delta_7d": {"views_pct": 0.02, "calls_pct": 0.0}},
        "offers": [], "conversation_history": [], "customer_aggregate": {}, "signals": [], "review_themes": [],
    }
    merchant.update(overrides)
    return merchant


def _customer(customer_id: str, merchant_id: str, name: str, language: str = "en", **overrides) -> dict:
    customer = {
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "identity": {"name": name, "language_pref": language, "age_band": "30-40"},
        "relationship": {"last_visit": "2026-05-12", "visits_total": 4, "services_received": ["cleaning"]},
        "state": "active",
        "preferences": {"channel": "whatsapp", "reminder_opt_in": True},
        "consent": {"scope": ["promotional_offers"]},
    }
    for key, value in overrides.items():
        customer[key] = {**customer[key], **value} if isinstance(value, dict) and key in customer else value
    return customer


@pytest.fixture
def store() -> ContextStore:
    s = ContextStore()
    for category in (_dentists(), _pharmacies(), _restaurants()):
        s.put("category", category["slug"], 1, category)

    s.put("merchant", "m_den", 1, _merchant(
        "m_den", "dentists", "Meera", "Dr. Meera's Dental Clinic",
        performance={"window_days": 30, "views": 2410, "calls": 18, "directions": 45, "ctr": 0.021,
                     "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05}},
        offers=[{"id": "o1", "title": "Dental Cleaning @ ₹299", "status": "active"},
                {"id": "o2", "title": "Deep Cleaning @ ₹499", "status": "expired"}],
        conversation_history=[{"ts": "2026-04-24T10:12:00Z", "from": "vera", "body": "Profile audit done."},
                              {"ts": "2026-04-24T10:18:00Z", "from": "merchant", "body": "Yes please"}],
        customer_aggregate={"high_risk_adult_count": 124, "lapsed_180d_plus": 78},
        signals=["stale_posts:22d", "engaged_in_last_48h"],
        review_themes=[{"theme": "doctor_manner", "sentiment": "pos", "occurrences_30d": 5,
                        "common_quote": "Dr. Meera explains everything patiently"}],
    ))
    # Sparse merchant: owner name already carries the title, tiny call count, plan expired.
    s.put("merchant", "m_sam", 1, _merchant(
        "m_sam", "dentists", "Dr. Sameer", "Smile Care",
        identity={"name": "Smile Care", "owner_first_name": "Dr. Sameer", "languages": ["en"]},
        subscription={"status": "expired", "plan": "Pro", "days_remaining": 0, "days_since_expiry": 39},
        performance={"window_days": 30, "views": 756, "calls": 2, "directions": 31, "ctr": 0.038,
                     "delta_7d": {"views_pct": 0.26, "calls_pct": 0.23}},
        customer_aggregate={"total_unique_ytd": 1509},
    ))
    s.put("merchant", "m_pha", 1, _merchant(
        "m_pha", "pharmacies", "Ramesh", "Apollo Pharmacy",
        offers=[{"id": "o3", "title": "Free Home Delivery > ₹499", "status": "active"}],
        customer_aggregate={"chronic_rx_count": 60},
    ))
    s.put("merchant", "m_res", 1, _merchant("m_res", "restaurants", "Suresh", "Pizza Junction"))

    s.put("customer", "c_priya", 1, _customer("c_priya", "m_den", "Priya", "hi-en mix"))
    s.put("customer", "c_kid", 1, _customer("c_kid", "m_den", "Aanya (parent: Sneha)", "en"))
    s.put("customer", "c_sharma", 1, _customer(
        "c_sharma", "m_pha", "Mr. Sharma", "hi",
        identity={"senior_citizen": True}, preferences={"channel": "whatsapp_via_son"},
    ))
    s.put("customer", "c_optout", 1, _customer("c_optout", "m_den", "Rohit", preferences={"reminder_opt_in": False}))
    s.put("customer", "c_noreach", 1, _customer("c_noreach", "m_den", "(walk-in, no profile)", preferences={"channel": "none_recorded"}))
    return s


@pytest.fixture
def add_trigger(store):
    """add_trigger("t1", "perf_dip", "m_den", {...}, customer_id=None, urgency=2) -> "t1"."""
    def add(trigger_id, kind, merchant_id, payload=None, customer_id=None, urgency=2, suppression_key=None, scope=None):
        store.put("trigger", trigger_id, 1, {
            "id": trigger_id,
            "scope": scope or ("customer" if customer_id else "merchant"),
            "kind": kind,
            "source": "internal",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "payload": {} if payload is None else payload,
            "urgency": urgency,
            "suppression_key": suppression_key or f"{kind}:{merchant_id}:{trigger_id}",
            "expires_at": "2026-06-30T00:00:00Z",
        })
        return trigger_id
    return add


@pytest.fixture
def brief_for(store):
    def make(trigger_id):
        brief, why = build_brief(store.get("trigger", trigger_id), store, trigger_id)
        assert brief is not None, why
        return brief
    return make
