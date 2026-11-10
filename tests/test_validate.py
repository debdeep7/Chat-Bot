"""The validator is the last gate before a message leaves the bot, so each rule gets its own test."""
from dataclasses import replace

import pytest

from vera.validate import ACTION_WORDS, QUALIFYING, _licensed, check, check_reply, language_problems, numbers_of, unlicensed_numbers

SLOTS = [
    {"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"},
    {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"},
]
DIP = {"metric": "calls", "delta_pct": -0.5, "window": "7d", "vs_baseline": 12}
RECALL = {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12", "available_slots": SLOTS}
PLACEHOLDER = {"placeholder": True, "metric_or_topic": "x"}

# One message that is fine for each of the three briefs below.
EN_DIP = ("Dr. Sameer, your calls are down 50% over the last 7 days, against a baseline of 12. "
          "Want me to look into what changed and send you a short fix list? Reply YES.")
HINGLISH_DIP = ("Dr. Meera, aapke calls pichhle 7 days mein 50% neeche hain, baseline 12 tha. "
                "Kya main dekh loon ki kya badla aur ek chhoti fix list bhej doon? YES bhej dijiye.")
RECALL_BODY = ("Hi Priya, Dr. Meera's Dental Clinic here. Your 6-month cleaning is due 12 Nov; last done 12 May. "
               "Dental Cleaning @ ₹299 is on. Reply 1 for Wed 5 Nov, 6pm; 2 for Thu 6 Nov, 5pm; ya koi aur time bata dijiye.")


@pytest.fixture
def sam(add_trigger, brief_for):
    """An English-speaking merchant with no earlier conversation."""
    return brief_for(add_trigger("t_sam", "perf_dip", "m_sam", DIP, urgency=4))


@pytest.fixture
def meera(add_trigger, brief_for):
    """A Hinglish-speaking merchant who has talked to Vera before."""
    return brief_for(add_trigger("t_meera", "perf_dip", "m_den", DIP, urgency=4))


@pytest.fixture
def priya(add_trigger, brief_for):
    """A Hinglish-speaking customer being offered two slots."""
    return brief_for(add_trigger("t_priya", "recall_due", "m_den", RECALL, customer_id="c_priya", urgency=3))


def codes(body, brief, seen=()):
    return [p.code for p in check(body, brief, seen)]


def test_grounded_messages_pass(sam, meera, priya):
    assert check(EN_DIP, sam) == []
    assert check(HINGLISH_DIP, meera) == []
    assert check(RECALL_BODY, priya) == []


# --------------------------------------------------------------------------- numbers

def test_numbers_are_read_the_way_people_write_them():
    assert numbers_of("₹1,499/month, 3.0% and Studio11 on 12 Nov, batch AT2024-1102, vitamin B12") == ["1499", "3", "12", "1102"]


def test_a_number_that_is_not_in_the_facts_is_flagged(sam):
    problems = check(EN_DIP.replace("50%", "70%"), sam)
    assert [p.code for p in problems] == ["number"]
    assert "70" in problems[0].detail


def test_small_effort_estimates_are_not_claims_about_the_business(sam):
    assert check(EN_DIP.replace("Want me", "It is a 5-min job. Want me"), sam) == []
    assert codes(EN_DIP.replace("Want me", "It is a 45-min job. Want me"), sam) == ["number"]


def test_day_counts_license_the_same_span_in_weeks_and_months():
    assert {"90", "13", "3"} <= _licensed(["plan expired 90 days ago"])
    assert "13" not in _licensed(["plan expired 91 months ago"])


# --------------------------------------------------------------------------- patterns

@pytest.mark.parametrize("insert", [
    "Details at www.example.com.", "Visit https://x.in/offer.", "Mail help@shop.com.", "Call 98765 43210.", "Call 9876543210.",
])
def test_links_emails_and_phone_numbers_are_flagged(sam, insert):
    assert "url" in codes(EN_DIP.replace("Want me", f"{insert} Want me"), sam)


def test_forbidden_wording_is_flagged(sam):
    assert "taboo" in codes(EN_DIP.replace("Want me", "Results are guaranteed. Want me"), sam)
    strict = replace(sam, category={**sam.category, "voice": {"vocab_taboo": ["cure (without disclosure)"]}})
    assert "taboo" in codes(EN_DIP.replace("Want me", "A cure for slow weeks. Want me"), strict)
    assert "taboo" not in codes(EN_DIP, strict)


@pytest.mark.parametrize("word", ["trigger", "{name}", "[Merchant]", "m_den_1", "null", "CTA", "json"])
def test_internal_wording_is_flagged(sam, word):
    assert "jargon" in codes(EN_DIP.replace("Want me", f"Note: {word}. Want me"), sam)


def test_repeating_an_earlier_message_is_flagged(sam):
    assert "repeat" in codes(EN_DIP, sam, seen=[EN_DIP])
    nearly = EN_DIP.replace("look into what changed", "check what changed")
    assert "repeat" in codes(nearly, sam, seen=[EN_DIP])
    assert "repeat" in codes(EN_DIP, replace(sam, prior_bodies=(EN_DIP,)))
    assert "repeat" not in codes(EN_DIP, sam, seen=["Hello there, a completely different message about something else."])


# --------------------------------------------------------------------------- who it is for

def test_the_recipient_must_be_addressed_by_name(sam, priya):
    assert "no_name" in codes(EN_DIP.replace("Dr. Sameer, ", ""), sam)
    assert "business" in codes(RECALL_BODY.replace("Dr. Meera's Dental Clinic", "Our clinic"), priya)
    assert "vera" in codes(RECALL_BODY.replace("here.", "here, Vera's partner."), priya)


def test_a_message_about_a_child_names_the_child(add_trigger, brief_for):
    brief = brief_for(add_trigger("t", "recall_due", "m_den", PLACEHOLDER, customer_id="c_kid"))
    body = "Hi Sneha, Dr. Meera's Dental Clinic here. It is time for the next regular visit. Shall we book a slot? Reply YES."
    assert "no_name" in codes(body, brief)
    assert "no_name" not in codes(body.replace("the next", "Aanya's next"), brief)


def test_vera_does_not_introduce_herself_again(meera):
    assert "reintro" in codes(HINGLISH_DIP.replace("Dr. Meera,", "Dr. Meera, main Vera hoon."), meera)
    assert "reintro" not in codes(HINGLISH_DIP, meera)


# --------------------------------------------------------------------------- the ask

def test_there_must_be_exactly_one_ask_and_it_comes_last(sam):
    assert "ask" in codes("Dr. Sameer, your calls are down 50% over the last 7 days, against a baseline of 12.", sam)
    assert "ask" in codes(EN_DIP.replace("Want me", "Why is that? Want me"), sam)
    assert "ask" in codes(EN_DIP + " Thanks " + "x" * 200, sam)


def test_slot_and_confirm_requirements(priya, add_trigger, brief_for):
    wrong_slots = RECALL_BODY.replace("Wed 5 Nov, 6pm", "Wednesday evening").replace("Thu 6 Nov, 5pm", "Thursday")
    assert "slots" in codes(wrong_slots, priya)

    appointment = brief_for(add_trigger("t2", "appointment_tomorrow", "m_den", PLACEHOLDER, customer_id="c_priya"))
    good = "Hi Priya, Dr. Meera's Dental Clinic here. Aapki appointment kal hai. Reply CONFIRM ya koi aur time bata dijiye."
    assert check(good, appointment) == []
    assert "ask" in codes("Hi Priya, Dr. Meera's Dental Clinic here. Aapki appointment kal hai. Kya aap aa sakte hain?", appointment)


# --------------------------------------------------------------------------- language and length

def test_language_follows_the_recipients_preference(sam, meera):
    assert "language" in codes(HINGLISH_DIP.replace("Dr. Meera", "Dr. Sameer"), sam)   # Hinglish to an English reader
    assert "language" in codes(EN_DIP.replace("Dr. Sameer", "Dr. Meera"), meera)       # plain English to a Hinglish reader
    devanagari = "Dr. Meera, आपके calls 7 days में 50% कम हैं। क्या मैं मदद करूँ? YES भेजिए।"
    assert "language" not in codes(devanagari, meera)


def test_length_limits(sam):
    assert codes("Hi", sam) == ["too_short"]
    assert "too_long" in codes(EN_DIP + " " + "word " * 200, sam)


# --------------------------------------------------------------------------- slot menus

def test_a_menu_of_three_slots_may_number_them_one_to_three(add_trigger, brief_for):
    three = [*SLOTS, {"iso": "2026-11-07T11:00:00+05:30", "label": "Fri 7 Nov, 11am"}]
    brief = brief_for(add_trigger("t3", "recall_due", "m_den", {**RECALL, "available_slots": three}, customer_id="c_priya"))
    menu = RECALL_BODY.replace("; ya koi aur", "; 3 for Fri 7 Nov, 11am; ya koi aur")
    assert check(menu, brief) == []
    assert "number" in codes(menu.replace("3 for Fri", "4 for Fri"), brief)  # only as many numbers as there are slots


def test_a_three_slot_menu_is_not_a_licence_for_other_small_numbers(priya):
    assert "number" in codes(RECALL_BODY.replace("Dental Cleaning", "3 Dental Cleanings"), priya)


# --------------------------------------------------------------------------- numbers outside the brief

def test_unlicensed_numbers_ignores_menu_digits_and_small_effort_estimates():
    assert unlicensed_numbers("calls down 50%, baseline 12", {"50", "12"}) == []
    assert unlicensed_numbers("Reply 1 or 2", set()) == []
    assert unlicensed_numbers("a 5-min setup", set()) == [] and unlicensed_numbers("a 10 minute fix", set()) == []
    assert unlicensed_numbers("a 30-min setup", set()) == ["30"] and unlicensed_numbers("done in 40 seconds", set()) == []
    assert unlicensed_numbers("3 patients and 3 reviews", set()) == ["3"]  # each invented number is listed once


@pytest.mark.parametrize("body, code, clean", [
    ("Happy to help you with that today", "en", True),
    ("नमस्ते, मैं आपकी मदद कर सकती हूँ", "en", False),
    ("Aap bataiye, main aapke liye draft bhej dungi", "en", False),
    ("Aap bataiye, main aapke liye draft bhej dungi", "hi-en", True),
    ("Happy to help you with that today", "hi-en", False),
    ("Aap bataiye, main aapke liye draft bhej dungi", "hi", True),
    ("Happy to help you with that today", "hi", False),
    ("Anything at all", "fr", True),
])
def test_language_problems(body, code, clean):
    assert (language_problems(body, code) == []) is clean


# --------------------------------------------------------------------------- replies

REPLY_FACTS = ["calls down 50% over the last 7 days, against a baseline of 12", "Dr. Meera's Dental Clinic"]
GOOD_REPLY = ("Great. I am drafting the fix list for your 50% drop in calls now and will send it here next. "
              "Reply CONFIRM and I will proceed.")


def reply_codes(body, **kwargs):
    kwargs.setdefault("lang", "en")
    kwargs.setdefault("grounding", REPLY_FACTS)
    return [p.code for p in check_reply(body, **kwargs)]


def test_a_grounded_reply_passes_even_as_an_action_reply():
    assert reply_codes(GOOD_REPLY) == []
    assert reply_codes(GOOD_REPLY, action=True) == []


def test_a_blank_or_tiny_reply_is_only_ever_too_short():
    assert reply_codes("") == reply_codes("ok") == reply_codes(None) == ["too_short"]


def test_reply_length_is_capped():
    assert reply_codes(GOOD_REPLY + " word" * 300) == ["too_long"]
    assert reply_codes(GOOD_REPLY, limit=40) == ["too_long"]


@pytest.mark.parametrize("body", [
    GOOD_REPLY + " See https://example.org/offer",
    GOOD_REPLY + " See www.example.org",
    GOOD_REPLY + " Mail me at someone@example.org",
    GOOD_REPLY + " Call 9876543210 today",
])
def test_links_addresses_and_phone_numbers_never_leave(body):
    assert "url" in reply_codes(body)


def test_a_reply_may_quote_numbers_from_the_facts_or_from_what_the_other_person_wrote():
    assert "number" in reply_codes(GOOD_REPLY.replace("50%", "70%"))
    said = REPLY_FACTS + ["I have 40 patients waiting"]
    assert reply_codes(GOOD_REPLY.replace("your 50% drop", "your 40 waiting patients"), grounding=said) == []
    assert "number" in reply_codes(GOOD_REPLY.replace("your 50% drop", "your 40 waiting patients"))


def test_forbidden_wording_and_internal_jargon_are_caught():
    assert "taboo" in reply_codes(GOOD_REPLY + " A guaranteed result.", taboo=("guaranteed",))
    assert reply_codes(GOOD_REPLY + " A guaranteed result.") == []  # only what the category forbids
    assert "jargon" in reply_codes(GOOD_REPLY + " The payload says so.")
    assert "jargon" in reply_codes(GOOD_REPLY + " Hello {name}")


def test_a_reply_cannot_repeat_itself():
    assert "repeat" in reply_codes(GOOD_REPLY, seen=[GOOD_REPLY])
    assert "repeat" in reply_codes(GOOD_REPLY, seen=[GOOD_REPLY.upper() + "  "])  # case and spacing do not hide it
    assert "repeat" in reply_codes(GOOD_REPLY.replace("Great.", "Okay."), seen=[GOOD_REPLY])  # a near copy
    assert reply_codes(GOOD_REPLY, seen=["Something else entirely, about reviews and the weekend rush."]) == []


def test_question_limit_follows_the_conversation():
    two = "Shall I start with the fix list? Or the post? Reply CONFIRM and I will proceed with the draft here next."
    assert "ask" in reply_codes(two) and reply_codes(two, max_questions=2) == []


def test_customer_facing_replies_never_name_vera_or_magicpin():
    assert "vera" in reply_codes(GOOD_REPLY + " Vera will follow up.", customer_facing=True)
    assert "vera" in reply_codes(GOOD_REPLY + " Booked via magicpin.", customer_facing=True)
    assert "vera" not in reply_codes(GOOD_REPLY + " Vera will follow up.")  # fine when speaking to a merchant


def test_a_reply_has_to_be_in_the_right_language():
    assert "language" in reply_codes("Aap bataiye, main aapke liye draft bhej dungi aur yahin share karungi.")
    assert reply_codes("Aap bataiye, main aapke liye draft bhej dungi aur yahin share karungi.", lang="hi-en") == []
    assert reply_codes("Aap bataiye, main aapke liye draft bhej dungi aur yahin share karungi.", lang="hi") == []
    assert "language" in reply_codes(GOOD_REPLY, lang="hi-en")


def test_an_action_reply_must_do_the_work_and_not_qualify():
    asking = "Would you like the fix list as a post or as a message? Reply POST or MESSAGE and I will send it here."
    assert "qualifying" in reply_codes(asking, action=True) and "qualifying" not in reply_codes(asking)
    vague = "Okay, noted. Thanks for telling me about that, it is useful."
    assert "no_action" in reply_codes(vague, action=True) and "no_action" not in reply_codes(vague)
    for phrase in QUALIFYING:
        assert "qualifying" in reply_codes(f"Great. {phrase.capitalize()} want it now? Reply CONFIRM to proceed.", action=True), phrase
    for word in ACTION_WORDS:
        assert "no_action" not in reply_codes(f"Fine, {word} it goes for the fix list.", action=True), word
