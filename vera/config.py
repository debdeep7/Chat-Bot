"""Settings, read from environment variables.

A `.env` file in the project root is loaded if it exists (real environment variables win).
`.env` is git-ignored: API keys must never be committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _get(name: str, default: str = "") -> str:
    """Environment value, or `default` when the variable is unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or default


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _number(name: str, default: float, minimum: float) -> float:
    """A numeric setting; junk or out-of-range values fall back to the default rather than crash startup."""
    try:
        value = float(_get(name, str(default)))
    except ValueError:
        return default
    return value if value >= minimum else default


def _flag(name: str, default: bool) -> bool:
    value = _get(name)
    return default if not value else value.lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class Settings:
    # Identity shown on /v1/metadata (the leaderboard reads this).
    team_name: str = "unnamed-team"
    team_members: list[str] = field(default_factory=list)
    contact_email: str = ""
    version: str = "0.1.0"
    approach: str = (
        "deterministic router and lead picker, Gemini composer constrained to the stored facts, "
        "fact validator, template fallback"
    )

    # LLM (Gemini free tier). With no key the bot runs on deterministic templates only.
    # gemini-2.5-* is closed to new projects, so the default is the current fast "lite" model.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    use_llm: bool = True           # USE_LLM=0 forces templates even when a key is present
    llm_per_minute: int = 8        # calls per minute we allow ourselves; free-tier limits are per project
    llm_timeout: float = 8.0       # seconds before one model call is abandoned
    tick_budget: float = 8.0       # seconds a /v1/tick may spend waiting for the model (judge allows 10-30)
    llm_workers: int = 6           # model calls in flight at once during a tick
    reply_timeout: float = 6.0     # seconds a /v1/reply may wait for the model; the judge's limit is 10-30

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            team_name=_get("TEAM_NAME", cls.team_name),
            team_members=_csv(_get("TEAM_MEMBERS")),
            contact_email=_get("CONTACT_EMAIL"),
            version=_get("BOT_VERSION", cls.version),
            approach=_get("BOT_APPROACH", cls.approach),
            gemini_api_key=_get("GEMINI_API_KEY"),
            gemini_model=_get("GEMINI_MODEL", cls.gemini_model),
            use_llm=_flag("USE_LLM", True),
            llm_per_minute=int(_number("LLM_PER_MINUTE", cls.llm_per_minute, 1)),
            llm_timeout=_number("LLM_TIMEOUT", cls.llm_timeout, 1),
            tick_budget=_number("TICK_BUDGET", cls.tick_budget, 1),
            llm_workers=int(_number("LLM_WORKERS", cls.llm_workers, 1)),
            reply_timeout=_number("REPLY_TIMEOUT", cls.reply_timeout, 1),
        )

    @property
    def llm_enabled(self) -> bool:
        return self.use_llm and bool(self.gemini_api_key)
