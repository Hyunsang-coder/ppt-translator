"""Factory for creating LLM instances based on provider."""

from __future__ import annotations

import logging
import os
import hashlib
import threading
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from src.utils.config import get_settings

# Ensure .env is loaded
load_dotenv()

LOGGER = logging.getLogger(__name__)

Provider = Literal["openai", "anthropic"]
_RATE_LIMITERS: dict[tuple[str, str], InMemoryRateLimiter] = {}
_RATE_LIMITER_LOCK = threading.Lock()


def create_rate_limiter() -> InMemoryRateLimiter:
    """Create a rate limiter based on application settings.

    Returns:
        Configured InMemoryRateLimiter instance.
    """
    settings = get_settings()
    return InMemoryRateLimiter(
        requests_per_second=settings.rate_limit_requests_per_second,
        check_every_n_seconds=settings.rate_limit_check_interval,
        max_bucket_size=settings.rate_limit_max_bucket_size,
    )


def _rate_limiter_key(provider: Provider, api_key: Optional[str]) -> tuple[str, str]:
    """Build a stable non-secret key for provider/API-key scoped limiting."""
    key_material = api_key or "<missing>"
    fingerprint = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return provider, fingerprint


def get_shared_rate_limiter(provider: Provider, api_key: Optional[str]) -> InMemoryRateLimiter:
    """Return a process-wide limiter shared by provider and API key."""
    key = _rate_limiter_key(provider, api_key)
    with _RATE_LIMITER_LOCK:
        limiter = _RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = create_rate_limiter()
            _RATE_LIMITERS[key] = limiter
        return limiter


def get_models_for_provider(provider: Provider) -> list[str]:
    """Return available models for the given provider.

    Derived from the single source of truth in src/services/models.py.

    Args:
        provider: The LLM provider ("openai" or "anthropic").

    Returns:
        List of model identifiers available for the provider.
    """
    from src.services.models import model_ids

    return model_ids(provider)


def create_llm(
    provider: Provider,
    model_name: str,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    rate_limiter: Optional[InMemoryRateLimiter] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Create an LLM instance based on the provider.

    Args:
        provider: The LLM provider ("openai" or "anthropic").
        model_name: The model identifier.
        max_tokens: Maximum tokens for response (required for Anthropic).
        api_key: Optional API key. If not provided, reads from environment.
        rate_limiter: Optional rate limiter. If not provided, creates one from settings.
        temperature: Optional temperature for sampling. If None, uses provider default.

    Returns:
        Configured LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        _rate_limiter = (
            rate_limiter
            if rate_limiter is not None
            else get_shared_rate_limiter(provider, key)
        )
        LOGGER.debug("Creating ChatAnthropic with model=%s, max_tokens=%d", model_name, max_tokens)
        kwargs: dict = dict(
            model=model_name,
            max_tokens=max_tokens,
            api_key=key,
            rate_limiter=_rate_limiter,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        _rate_limiter = (
            rate_limiter
            if rate_limiter is not None
            else get_shared_rate_limiter(provider, key)
        )
        LOGGER.debug("Creating ChatOpenAI with model=%s", model_name)
        kwargs: dict = dict(
            model=model_name,
            api_key=key,
            rate_limiter=_rate_limiter,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
