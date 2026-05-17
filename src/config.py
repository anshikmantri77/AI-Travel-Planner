"""Centralised configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    """Application settings derived from env vars with sensible defaults."""

    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "anthropic" | "openai" | "groq"
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile" if os.getenv("LLM_PROVIDER", "groq") == "groq" else (
            "claude-sonnet-4-20250514" if os.getenv("LLM_PROVIDER") == "anthropic" else "gpt-4o"
        ),
    )

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Session TTL in seconds (1 hour)
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

    # Maximum HITL revision cycles before auto-finalize
    MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "3"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
# Reload trigger 2
