"""The reply wording tables are the floor under every reply, so each variant must be safe on its own."""
import itertools

import pytest

from vera import replytext as rt
from vera.prompt import UNIVERSAL_TABOO
from vera.validate import ACTION_WORDS, QUALIFYING, check_reply

CATEGORIES = ["dentists", "salons", "gyms", "restaurants", "pharmacies", "something_new"]
TABOO = (*UNIVERSAL_TABOO, "cure", "whitening guarantee", "best in town", "hurry")
SLOTS = ["Wed 5 Nov, 6pm", "Thu 6 Nov, 5pm", "Fri 7 Nov, 11am"]


def ctx(slug, hindi, name="Dr. Meera", kind="perf_dip", working=""):
    clients = rt.clients_for(slug)
    return rt.Ctx(name=name, hindi=hindi, does=rt.does_for(kind, hindi, clients), working=working,
                  business="Dr. Meera's Dental Clinic", clients=clients)


def problems(body, hindi, *, customer=False, action=False, grounding=()):
    return check_reply(body, lang="hi-en" if hindi else "en", grounding=[*grounding, "Dr. Meera's Dental Clinic"], taboo=TABOO,
                       customer_facing=customer, action=action, max_questions=2 if customer else 1)


MERCHANT_BUILDERS = [rt.auto_flag, rt.action, rt.done, rt.answer, rt.identity, rt.thanks_nudge, rt.other]
CUSTOMER_BUILDERS = [rt.customer_confirmed, rt.customer_open, rt.customer_question, rt.customer_other, rt.customer_identity]


@pytest.mark.parametrize("builder", MERCHANT_BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("hindi", [False, True], ids=["en", "hi-en"])
@pytest.mark.parametrize("slug", CATEGORIES)
def test_every_merchant_variant_passes_the_reply_checks(builder, hindi, slug):
    variants = builder(ctx(slug, hindi))
    assert variants
    for body in variants:
        assert problems(body, hindi, action=builder is rt.action) == [], body
        assert "{" not in body and "}" not in body and "  " not in body


@pytest.mark.parametrize("hindi", [False, True], ids=["en", "hi-en"])
@pytest.mark.parametrize("topic", ["gst", "loan", "legal", "hiring", "chitchat", ""])
def test_every_off_topic_variant_passes_the_reply_checks(hindi, topic):
    variants = rt.off_topic(ctx("dentists", hindi), topic)
    assert len(variants) == 2
    for body in variants:
        assert problems(body, hindi) == [], body


@pytest.mark.parametrize("builder", CUSTOMER_BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("hindi", [False, True], ids=["en", "hi-en"])
@pytest.mark.parametrize("name", ["Priya", ""])
def test_every_customer_variant_is_safe_and_never_mentions_vera(builder, hindi, name):
    for body in builder(ctx("dentists", hindi, name=name)):
        assert problems(body, hindi, customer=True) == [], body
        assert "Vera" not in body and "magicpin" not in body.lower()


@pytest.mark.parametrize("hindi", [False, True], ids=["en", "hi-en"])
def test_slot_replies_quote_the_slots_exactly(hindi):
    booked = rt.customer_booked(ctx("dentists", hindi), SLOTS[1])
    pick = rt.customer_pick(ctx("dentists", hindi), SLOTS)
    assert all(SLOTS[1] in body for body in booked)
    assert all(label in body for body in pick for label in SLOTS)
    facts = ["slots available: " + " | ".join(SLOTS)]  # how the Brief carries the slots, so the menu numbers are licensed
    for body in (*booked, *pick):
        assert problems(body, hindi, customer=True, grounding=facts) == [], body
    assert "3 for Fri 7 Nov, 11am" in pick[0]


def test_a_slot_menu_offers_at_most_three_choices():
    body = rt.customer_pick(ctx("dentists", False), [f"slot {i}" for i in range(1, 8)])[0]
    assert "3 for slot 3" in body and "4 for" not in body


def test_the_action_reply_names_the_work_and_asks_for_one_confirm():
    for body in rt.action(ctx("dentists", False)):
        lowered = body.lower()
        assert "confirm" in lowered and body.count("?") == 0
        assert any(w in lowered for w in ACTION_WORDS) and not any(q in lowered for q in QUALIFYING)
        assert "look into what changed" in body  # the thing that was offered, not a generic line


def test_each_situation_has_more_than_one_wording_where_it_can_come_up_twice():
    c = ctx("dentists", False)
    for builder in (rt.auto_flag, rt.action, rt.done, rt.answer, rt.thanks_nudge, rt.other, rt.customer_question, rt.customer_other):
        variants = builder(c)
        assert len(variants) >= 2 and len(set(variants)) == len(variants), builder.__name__


def test_an_empty_name_leaves_no_stray_punctuation_or_lowercase_start():
    c = ctx("dentists", False, name="")
    for builder in MERCHANT_BUILDERS:
        for body in builder(c):
            assert not body.startswith(",") and ", ," not in body and " ," not in body and ",." not in body, body
            assert body[0].isupper(), body
            assert ". ," not in body and "!," not in body
    # ... and the Hinglish forms too
    for body in itertools.chain.from_iterable(b(ctx("dentists", True, name="")) for b in MERCHANT_BUILDERS):
        assert not body.startswith(",") and ", ," not in body and " ," not in body, body


def test_the_off_topic_line_reads_naturally_mid_sentence():
    first = rt.off_topic(ctx("dentists", False), "loan")[0]
    assert "Dr. Meera, loans and finance" in first          # lower-case after the comma
    assert "Dr. Meera, GST and tax filing" in rt.off_topic(ctx("dentists", False), "gst")[0]  # an acronym keeps its capitals
    second = rt.off_topic(ctx("dentists", False), "loan")[1]
    assert ". Loans and finance are best discussed" in second  # capitalised as a sentence of its own


def test_hinglish_uses_the_feminine_forms_vera_speaks_in():
    text = " ".join(rt.action(ctx("dentists", True)))
    assert "rahi hoon" in text and "dungi" in text
    assert "rahe hain" not in text and "dunga" not in text


# --------------------------------------------------------------------------- helpers

def test_does_for_falls_back_to_a_generic_offer_for_unknown_kinds():
    assert rt.does_for("never_heard_of_it", False, "patients") == rt.does_for("generic_merchant", False, "patients")
    assert "patients" in rt.does_for("research_digest", False, "patients")
    assert "aapke members" in rt.does_for("seasonal_perf_dip", True, "members")


def test_clients_for():
    assert rt.clients_for("gyms") == "members" and rt.clients_for("anything_else") == "customers" and rt.clients_for("") == "customers"


@pytest.mark.parametrize("text,limit,expected", [
    ("", 200, ""),
    ("short fact.", 200, "short fact"),
    ("calls down 50%!", 200, "calls down 50%"),
    ("  spaced   out   words  ", 200, "spaced out words"),
    ("First sentence is here. Second sentence is longer than the limit allows for sure.", 30, "First sentence is here"),
    ("one-long-sentence-without-any-stop but with words that go on and on and on", 30, ""),  # no clean cut: say nothing
    ("Some fact (with a bracket that never closes and keeps going on", 30, "Some fact"),
    ("exactly twenty chars! and then more words follow here", 20, "exactly twenty"),
    ("alpha beta gamma delta", 10, "alpha beta"),  # a word that ends exactly at the limit is kept whole
])
def test_shorten(text, limit, expected):
    assert rt._shorten(text, limit) == expected


def test_shorten_never_cuts_a_word_in_half():
    text = "D-speed film does not pass the new limit. " * 10
    short = rt._shorten(text, 60)
    assert short == "D-speed film does not pass the new limit" and len(short) <= 60
    for limit in range(10, 120, 7):
        out = rt._shorten("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda", limit)
        assert out == "" or out.split()[-1] in "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda".split()


def test_working_line():
    assert rt.working_line("calls down 50%", False) == "Here is what I am working from: calls down 50%. "
    assert rt.working_line("calls down 50%", True) == "Main isi par kaam kar rahi hoon: calls down 50%. "
    assert rt.working_line("calls down 50%", False, own_profile=True) == "Starting from your profile: calls down 50%. "
    assert rt.working_line("calls down 50%", True, own_profile=True) == "Aapke profile se shuru karte hain: calls down 50%. "
    assert rt.working_line("", False) == "" and rt.working_line("   ", True) == ""


def test_a_working_line_inside_an_action_reply_stays_one_clean_sentence():
    c = ctx("dentists", False, working=rt.working_line("calls down 50% over the last 7 days. Extra detail that is long " * 6, False))
    for body in rt.action(c):
        assert "..." not in body and ".." not in body.replace("...", "") and "  " not in body


def test_tidy_closes_the_gaps_an_empty_name_leaves():
    assert rt._tidy(["Great, . Next step."]) == ["Great. Next step."]
    assert rt._tidy([", whenever you see this: reply YES."]) == ["Whenever you see this: reply YES."]
    assert rt._tidy(["Done.  , thanks for waiting."]) == ["Done. Thanks for waiting."]
    assert rt._tidy(["ok, then"]) == ["Ok, then"]
    assert rt._tidy(["Fine  ,  really."]) == ["Fine, really."]


def test_unquotable_leads_cover_the_instruction_like_ones():
    assert {"ask.topic", "trigger.intent", "trigger.dormant"} <= rt.UNQUOTABLE_LEADS
