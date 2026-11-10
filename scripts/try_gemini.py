"""Check that the Gemini setup works, then write two real messages with it.

    python scripts/try_gemini.py

Reads GEMINI_API_KEY and GEMINI_MODEL from .env (copy .env.example to .env and paste your key
after GEMINI_API_KEY=). The key is never printed. Exit code 0 means the model answered.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vera.composer import build_composer, parse_reply  # noqa: E402
from vera.config import Settings  # noqa: E402
from vera.dataset import load_pairs, load_store  # noqa: E402
from vera.llm import GeminiClient, LLMError  # noqa: E402
from vera.signals import build_brief  # noqa: E402

HINTS = {
    "auth": "The key was rejected. Create a new key at https://aistudio.google.com/ and paste it after GEMINI_API_KEY= in .env.",
    "bad_request": "The request was rejected; an invalid key is also reported this way. Re-check GEMINI_API_KEY in .env.",
    "model_not_found": "This model name is not available to your project. Set GEMINI_MODEL in .env (see .env.example for options).",
    "rate_limited": "You hit a free-tier limit. Wait a minute and retry, or look up your limits at https://aistudio.google.com/rate-limit.",
    "timeout": "The model took too long. Retry; if it keeps happening, set LLM_TIMEOUT higher in .env.",
    "network": "Could not reach Google. Check your internet connection or proxy.",
    "server": "Google's side failed (often 'overloaded'). Retry in a minute.",
    "blocked": "The prompt was blocked by a safety filter.",
    "empty": "The model returned no text.",
    "bad_response": "The reply was not in the expected format.",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = Settings.from_env()
    if not settings.gemini_api_key:
        print("No GEMINI_API_KEY found. Copy .env.example to .env and paste your key after GEMINI_API_KEY=.")
        return 1
    print(f"model: {settings.gemini_model}   (key found: yes, never printed)\n")

    client = GeminiClient(settings.gemini_api_key, settings.gemini_model, timeout=max(settings.llm_timeout, 15))
    started = time.monotonic()
    try:
        raw = client.generate('Answer with a JSON object only: {"body": "<one friendly greeting in Hinglish>", "rationale": "test"}', "Say hello to a salon owner called Anjali.")
    except LLMError as exc:
        print(f"FAILED ({exc.kind}): {exc}")
        print(HINTS.get(exc.kind, ""))
        return 2
    parsed = parse_reply(raw)
    print(f"1. the model answered in {time.monotonic() - started:.1f}s: {parsed[0] if parsed else raw[:200]!r}")
    print("   reply format understood: " + ("yes" if parsed else "NO (the composer will fall back to templates)") + "\n")

    data = Path(__file__).resolve().parent.parent / "data" / "expanded"
    if not data.exists():
        print("2. skipped: no dataset in data/expanded (run scripts/load_dataset.py first)")
        return 0
    store = load_store(data)
    composer = build_composer(settings)
    composer.call_timeout = max(composer.call_timeout, 15)
    pairs = {p["test_id"]: p for p in load_pairs(data)}
    print("2. two real messages through the full pipeline (validator and fallback included):\n")
    for test_id in ("T24", "T28"):
        pair = pairs.get(test_id)
        brief, _ = build_brief(store.get("trigger", pair["trigger_id"]), store, pair["trigger_id"]) if pair else (None, None)
        if brief is None:
            continue
        message = composer.compose(brief)
        print(f"   {test_id} {brief.kind} -> {message.source if message else 'silent'}")
        if message:
            print(f"   {message.body}")
            print(f"   why: {message.rationale}")
            if message.rejected:
                print(f"   (earlier drafts rejected for: {', '.join(message.rejected)})")
        print()
    print(f"model calls that failed: { {k: v for k, v in composer.stats.items() if k.startswith('llm_') and k != 'llm'} or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
