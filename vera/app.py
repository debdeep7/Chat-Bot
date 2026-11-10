"""HTTP surface of the bot: the endpoints the judge harness calls.

    POST /v1/context   receive a context push (idempotent by scope + id + version)
    POST /v1/tick      periodic wake-up; the bot may start conversations
    POST /v1/reply     the merchant/customer answered; the bot sends, waits or ends
    GET  /v1/healthz   liveness (3 failures in a row = disqualified)
    GET  /v1/metadata  team identity for the leaderboard

`tick` plans which triggers to act on, composes the messages and returns them as actions.
`reply` hands the incoming message to the reply state machine (dialogue.py).
    POST /v1/teardown  optional end-of-test call: wipe every piece of state
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .composer import Composer, build_action, build_composer, conversation_id
from .config import Settings
from .conversations import ConversationBook
from .dialogue import Replier
from .ledger import Ledger
from .models import ContextPush, ReplyRequest, TickRequest
from .planner import parse_time, plan_tick
from .store import SCOPES, ContextStore

log = logging.getLogger("vera")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_app(settings: Settings | None = None, composer: Composer | None = None) -> FastAPI:
    """Build an app with its own empty state (tests get a fresh one each time).

    `composer` can be supplied by tests; by default it follows the settings (Gemini when a key is
    present, templates only otherwise).
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = settings or Settings.from_env()
    store = ContextStore()
    ledger = Ledger()
    conversations = ConversationBook()
    composer = composer or build_composer(settings)
    replier = Replier(store, ledger, conversations, composer, reply_timeout=settings.reply_timeout)
    started = time.monotonic()
    started_at = _utc_now()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        composer.close()

    app = FastAPI(title="Vera bot", version=settings.version, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.ledger = ledger
    app.state.conversations = conversations
    app.state.composer = composer
    app.state.replier = replier
    log.info("composer: %s", f"Gemini {settings.gemini_model}" if composer.uses_llm else "templates only (no model)")

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError):
        # The testing brief asks for a 400 with {"accepted": false, ...} on a bad context push.
        if request.url.path == "/v1/context":
            problems = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
            return JSONResponse(
                status_code=400,
                content={"accepted": False, "reason": "malformed_request", "details": problems},
            )
        return await request_validation_exception_handler(request, exc)

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def root():
        return {"service": "vera-bot", "status": "ok"}

    # HEAD is allowed because free uptime monitors (e.g. UptimeRobot) ping with HEAD by default.
    @app.api_route("/v1/healthz", methods=["GET", "HEAD"])
    async def healthz():
        return {
            "status": "ok",
            "uptime_seconds": int(time.monotonic() - started),
            "contexts_loaded": store.counts(),
        }

    @app.get("/v1/metadata")
    async def metadata():
        return {
            "team_name": settings.team_name,
            "team_members": settings.team_members,
            "model": settings.gemini_model if composer.uses_llm else "deterministic-templates",
            "approach": settings.approach,
            "contact_email": settings.contact_email,
            "version": settings.version,
            "submitted_at": started_at,
        }

    @app.post("/v1/context")
    async def push_context(body: ContextPush):
        if body.scope not in SCOPES:
            return JSONResponse(
                status_code=400,
                content={
                    "accepted": False,
                    "reason": "invalid_scope",
                    "details": f"scope must be one of {list(SCOPES)}, got {body.scope!r}",
                },
            )
        result = store.put(body.scope, body.context_id, body.version, body.payload)
        if result.outcome == "stale":
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": result.version},
            )
        log.debug("context %s %s v%s -> %s", body.scope, body.context_id, body.version, result.outcome)
        return {
            "accepted": True,
            "ack_id": f"ack_{body.scope}_{body.context_id}_v{result.version}",
            "stored_at": _utc_now(),
        }

    # A plain `def`, not `async def`: FastAPI runs it in a worker thread, so waiting for the model
    # never blocks /v1/healthz (three failed health checks in a row disqualify the bot).
    @app.post("/v1/tick")
    def tick(body: TickRequest):
        began = time.monotonic()
        try:
            now = parse_time(body.now)
            plan = plan_tick(store, ledger, body.available_triggers, now)
            messages = composer.compose_batch(plan.briefs, avoid=conversations.bodies_for)
            actions = []
            for brief, message in zip(plan.briefs, messages):
                if message is None:
                    # Nothing valid to say: stay silent. The trigger is tried again on a later tick,
                    # but not forever (the ledger gives up on it after a couple of silent attempts).
                    ledger.note_silence(brief.trigger_id)
                    continue
                conv_id = conversation_id(brief)
                actions.append(build_action(brief, message, conv_id))
                ledger.mark_sent(brief, now)
                conversations.open(conv_id, brief, message.body, now)
        except Exception:
            # An empty answer is a valid tick; a 500 would look like a broken bot.
            log.exception("tick failed; answering with no actions")
            return {"actions": []}
        sources = Counter(m.source for m in messages if m is not None)
        skipped = Counter(s.reason for s in plan.skipped)
        log.info(
            "tick: %d triggers offered, %d sent %s, %d skipped %s, %.1fs",
            len(body.available_triggers), len(actions), dict(sources), len(plan.skipped), dict(skipped),
            time.monotonic() - began,
        )
        return {"actions": actions}

    # Also a plain `def`: a reply may wait a few seconds for the model, and must not hold up /v1/healthz.
    @app.post("/v1/reply")
    def reply(body: ReplyRequest):
        return replier.handle(body)

    # Optional end-of-test call from the judge: the bot must not keep context data after the test.
    @app.post("/v1/teardown")
    def teardown():
        store.clear()
        ledger.clear()
        conversations.clear()
        replier.clear()
        log.info("teardown: all state wiped")
        return {"status": "cleared"}

    return app


app = create_app()
