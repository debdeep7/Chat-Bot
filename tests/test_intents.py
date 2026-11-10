import pytest

from vera.intents import (
    AUTO, BLANK, DECLINE, GO_AHEAD, IDENTITY, LATER, OFF_TOPIC, OTHER, QUESTION, STOP, THANKS,
    classify, detect_language, fingerprint,
)

# (message, expected kind). Many of these come straight from the challenge pack.
CASES = [
    # --- the judge's own scenarios
    ("Thank you for contacting us! Our team will respond shortly.", AUTO),
    ("Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.", AUTO),
    ("Ok lets do it. Whats next?", GO_AHEAD),
    ("Ok, let's do it. What's next?", GO_AHEAD),
    ("Stop messaging me. This is useless spam.", STOP),
    ("Why are you bothering me. This is useless. Stop sending these.", STOP),
    ("Not interested. Stop messaging me.", STOP),
    ("Btw can you also help me with my GST filing this month?", OFF_TOPIC),
    ("Yes please send the abstract. Also draft the patient WhatsApp.", GO_AHEAD),
    ("Mujhe magicpin judrna hai", GO_AHEAD),
    # --- canned replies seen in production
    ("Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari team tak pahuncha deti hoon.", AUTO),
    ("Hello! Main ek automated assistant hoon. Aapka message hamari team tak pahunch gaya hai.", AUTO),
    ("We are currently away. We will get back to you soon.", AUTO),
    ("This is an automated reply. Our office is closed.", AUTO),
    ("Thanks for reaching out! We'll get back to you shortly.", AUTO),
    # --- opting out and hostility
    ("STOP", STOP),
    ("stop", STOP),
    ("Please stop", STOP),
    ("unsubscribe", STOP),
    ("Don't message me again", STOP),
    ("Do not contact me", STOP),
    ("leave me alone", STOP),
    ("remove my number from your list", STOP),
    ("this is spam", STOP),
    ("You people are idiots", STOP),
    ("Mujhe message mat bhejo", STOP),
    ("band karo ye sab", STOP),
    ("bahut pareshan kar rahe ho", STOP),
    ("मुझे मैसेज मत भेजो", STOP),
    # --- declining
    ("Not interested", DECLINE),
    ("no thanks", DECLINE),
    ("No", DECLINE),
    ("nahi", DECLINE),
    ("Nahi chahiye", DECLINE),
    ("I don't need this", DECLINE),
    ("we are not interested in this", DECLINE),
    ("interested nahi hoon", DECLINE),
    # --- later
    ("I'm busy right now", LATER),
    ("Not now, maybe later", LATER),
    ("Call me tomorrow", LATER),
    ("will check and let you know", LATER),
    ("let me think about it", LATER),
    ("abhi busy hoon, baad mein baat karte hain", LATER),
    ("kal baat karte hain", LATER),
    ("dekhta hoon", LATER),
    # --- going ahead
    ("Yes", GO_AHEAD),
    ("yes please", GO_AHEAD),
    ("Ok", GO_AHEAD),
    ("okay sure", GO_AHEAD),
    ("haan", GO_AHEAD),
    ("Haan kar do", GO_AHEAD),
    ("theek hai", GO_AHEAD),
    ("go ahead", GO_AHEAD),
    ("Sounds good, send it", GO_AHEAD),
    ("I'm interested", GO_AHEAD),
    ("please proceed", GO_AHEAD),
    ("CONFIRM", GO_AHEAD),
    ("Done", GO_AHEAD),
    ("sign me up", GO_AHEAD),
    ("What's next?", GO_AHEAD),
    ("no problem, go ahead", GO_AHEAD),
    ("हाँ", GO_AHEAD),
    ("👍", GO_AHEAD),
    # --- unrelated requests
    ("Can you help me with my income tax return?", OFF_TOPIC),
    ("I need a loan for my clinic", OFF_TOPIC),
    ("tell me a joke", OFF_TOPIC),
    ("Do you know a good lawyer?", OFF_TOPIC),
    ("Can you hire a receptionist for me?", OFF_TOPIC),
    # --- who are you
    ("Who are you?", IDENTITY),
    ("Are you a bot?", IDENTITY),
    ("is this a scam?", IDENTITY),
    ("aap kaun ho?", IDENTITY),
    ("How did you get my number?", IDENTITY),
    # --- questions
    ("How much will it cost?", QUESTION),
    ("kitna charge lagega", QUESTION),
    ("What exactly will you do?", QUESTION),
    ("Is it free?", QUESTION),
    ("Ok but what is the price?", QUESTION),
    ("Can you explain how this works", QUESTION),
    # --- thanks
    ("Thanks", THANKS),
    ("thank you so much", THANKS),
    ("ok thanks", THANKS),
    ("shukriya", THANKS),
    ("🙏", THANKS),
    # --- anything else
    ("Haircuts and facials are in demand this week", OTHER),
    ("We are closed on Sundays", OTHER),
    ("My clinic has two chairs", OTHER),
    # --- nothing readable
    ("", BLANK),
    ("   ", BLANK),
    ("...", BLANK),
    ("🤔", BLANK),
]


@pytest.mark.parametrize("message,expected", CASES, ids=[f"{m[:40]!r}" for m, _ in CASES])
def test_a_message_is_classified(message, expected):
    assert classify(message).kind == expected, classify(message)


def test_a_bare_yes_is_weak_but_a_clear_commitment_is_not():
    assert classify("ok").weak and classify("yes please").weak and classify("haan ji").weak
    assert not classify("Ok lets do it. Whats next?").weak
    assert not classify("go ahead").weak


def test_later_says_how_long_to_leave_them_alone():
    assert classify("call me tomorrow").wait == 86400
    assert classify("I'm in a meeting, ping me in an hour").wait == 3600
    assert classify("busy right now").wait == 2 * 3600
    assert classify("let's talk next week").wait == 3 * 86400


def test_off_topic_names_the_topic():
    assert classify("what about my GST?").topic == "gst"
    assert classify("need a business loan").topic == "loan"


def test_the_language_of_each_message_is_detected():
    assert detect_language("Ok lets do it. Whats next?") == "en"
    assert detect_language("Haan kar do") == "hi-en"
    assert detect_language("mujhe yeh offer chahiye aur kitna lagega") == "hi-en"
    assert detect_language("मुझे जानकारी चाहिए") == "hi-en"
    assert detect_language("ok") is None and detect_language("yes please") is None


def test_dont_stop_is_not_a_request_to_stop():
    assert classify("please don't stop sending me the tips").kind != STOP


def test_anything_odd_is_survivable():
    for odd in (None, 42, ["x"], {"a": 1}, "\x00\x01", "a" * 5000):
        assert classify(odd).kind in {OTHER, BLANK, QUESTION, GO_AHEAD, DECLINE, LATER, STOP, AUTO}


def test_fingerprint_ignores_case_and_punctuation():
    a = fingerprint("Thank you for contacting us!  Our team will respond shortly.")
    b = fingerprint("thank you for contacting us, our team will respond shortly")
    assert a == b and a != fingerprint("something else")
