---
title: Vera Bot
emoji: 💬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Vera bot — magicpin AI Challenge

A merchant-messaging bot for the magicpin AI Challenge: it receives category / merchant /
customer / trigger context from the judge harness and decides what (and whether) to send.

**Status:** step 5 of 6 — `/v1/tick` writes and checks the opening messages, and `/v1/reply`
now runs a reply state machine (auto-reply, "let's do it", opt-out, off-topic, customers).
Next: deployment (step 6).

## How replies work

```
message ──► intents.classify ──► per-merchant memory ──► dialogue (decide) ──► replytext / Gemini ──► check_reply ──► send | wait | end
```

* **Works on any conversation id**, including ones the bot never opened: the merchant comes from
  the request, and the latest conversation the bot opened with them supplies the topic.
* **Canned auto-replies:** one short note for the owner, then wait a day, then end. The count is
  kept per merchant, because the judge changes the conversation id on every repeat.
* **"Let's do it":** switches to action (what happens next + one CONFIRM), never qualifying questions.
* **Opt-out / hostile:** end and stop all outreach. "Not interested" ends and pauses a week. "Busy"
  waits as long as they said. A later message from the merchant lifts the pause.
* **Off-topic** (GST, loans...) is declined politely and steered back; asking again gets new
  words, then silence. **Customers** are answered as the business and never hear about Vera.
* Replies are templates first (EN / Hinglish, several variants so nothing repeats); Gemini drafts
  only go-ahead / question / off-topic / other replies (`REPLY_TIMEOUT`, default 6 s), and a
  draft that fails `check_reply` is replaced by the template.
* `POST /v1/teardown` wipes all state.

## How a message is made

```
trigger ──► router ──► lead builder ──► Brief ──► planner ──► composer ──► validator ──► action
            kind →      one "why now"    facts,     which ones   Gemini,       numbers, links,
            Strategy    fact + ≤3        notes,     go out this  else a        language, one
                        supporting       who to     tick         template      ask, repeats
                        facts            address
```

* **Decisions are deterministic.** The Brief fixes what the message is about, who gets it, in
  which language, and what it asks. The model only phrases it; it cannot change the subject.
* **Nothing is invented.** The model is shown only the Brief's facts, and the validator rejects
  any number that is not in them, plus links, e-mails, phone numbers, forbidden wording,
  internal jargon, a missing name, a buried or doubled ask, the wrong language, and any repeat.
* **The bot never falls below the templates.** If there is no key, the rate limit is spent, the
  model is slow, or its draft fails validation (it gets one retry with the problems listed),
  the template writer produces a safe message from the same facts. If even that would be a
  repeat, the bot stays silent: restraint is rewarded, spam is penalised.
* **A slow model cannot cause a timeout.** Model calls for a tick run in parallel and the tick
  waits for them only `TICK_BUDGET` seconds (default 8).
* **The free tier is protected.** Calls are counted in a one-minute window (`LLM_PER_MINUTE`,
  default 8, two of them reserved for replies). After a 429 or a failure the bot stops calling
  for 20 s, doubling on repeated failures, and runs on templates meanwhile.

When a trigger carries no usable details (75 of the 100 sample triggers are bare placeholders),
the lead comes from the merchant's own data and the Brief carries a note saying what must not
be claimed.

## Layout

```
vera/
  app.py          the endpoints (/v1/context, /tick, /reply, /healthz, /metadata, /teardown)
  store.py        versioned in-memory context store (idempotent, thread-safe)
  models.py       lenient request bodies
  config.py       settings from environment / .env
  router.py       trigger kind -> Strategy (audience, call-to-action, levers, goal); 26 kinds + 2 generic fallbacks
  facts.py        small, checked statements about a merchant / category / customer
  leads.py        per-kind builders that pick the one "why now" fact plus supporting facts and cautions
  signals.py      assembles the Brief: recipient, salutation, language, call-to-action, facts, notes
  ledger.py       what has been sent / opted out / given up on, so nothing is sent twice
  planner.py      picks which triggers to act on in a tick (priority, one message per person, cooldown, cap of 20)
  llm.py          Gemini over HTTPS + the rate gate (per-minute budget and circuit breaker)
  prompt.py       what the model is told: rules, the Brief's facts, language and ask
  validate.py     the checks every message must pass, whoever wrote it
  templates.py    the deterministic writer (English and Hinglish) used as the fallback
  composer.py     model -> validate -> retry once -> template -> or silence; builds the action object
  conversations.py  what was said in each conversation (anti-repetition, reply threads)
  intents.py      classifies an incoming message (auto-reply, stop, go-ahead, off-topic...) in EN / Hinglish
  replytext.py    reply wording tables (merchant and customer, EN / Hinglish, several variants)
  dialogue.py     the reply state machine behind /v1/reply
  dataset.py      loads data/expanded straight into a store (used by the report and tests)
tests/            pytest: store, endpoints, facts, router, briefs, ledger, planner, validator, templates, model client, composer, conversations, tick, intents, dialogue, reply API, reply wording
scripts/
  load_dataset.py     push the expanded dataset into a running bot (warm-up rehearsal)
  decision_report.py  print what the bot decides (and, with --compose, writes) for the 30 canonical test pairs
  try_gemini.py       smoke-test your Gemini key and model, then write two real messages
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt     # Windows
cp .env.example .env                                                 # then fill in the values
```

Get the organizer's pack from the challenge page into `challenge-zip/`, then build the dataset:

```bash
.venv/Scripts/python.exe -I challenge-zip/dataset/generate_dataset.py --seed-dir challenge-zip/dataset --out data/expanded
```

### Gemini (optional but recommended)

Create a free key at https://aistudio.google.com/ and paste it after `GEMINI_API_KEY=` in `.env`
(never in chat, never in git). The default model is `gemini-3.5-flash-lite`; the older
`gemini-2.5-*` models are closed to new projects. Your own free-tier limits are listed at
https://aistudio.google.com/rate-limit; raise `LLM_PER_MINUTE` in `.env` if they allow it.

```bash
.venv/Scripts/python.exe -I -X utf8 scripts/try_gemini.py        # checks the key and model, writes two messages
```

Without a key the bot runs on templates alone and says so in `/v1/metadata` (`model`).

## Run

```bash
.venv/Scripts/python.exe -m uvicorn vera.app:app --host 0.0.0.0 --port 8080
.venv/Scripts/python.exe scripts/load_dataset.py --url http://localhost:8080      # in another terminal
```

Run **one worker only**: all context and conversation state lives in process memory.

## See the decisions and the messages

```bash
.venv/Scripts/python.exe -I -X utf8 scripts/decision_report.py --data data/expanded            # the 30 test pairs
.venv/Scripts/python.exe -I -X utf8 scripts/decision_report.py --data data/expanded --only T05,T24
.venv/Scripts/python.exe -I -X utf8 scripts/decision_report.py --data data/expanded --all       # all 100 triggers, summary
.venv/Scripts/python.exe -I -X utf8 scripts/decision_report.py --data data/expanded --compose   # + the template message for each
.venv/Scripts/python.exe -I -X utf8 scripts/decision_report.py --data data/expanded --compose --llm   # + Gemini (slow: waits on the rate limit)
```

## Test

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Secrets

`.env` holds the Gemini API key and is git-ignored. Never commit it or paste a key into code.
