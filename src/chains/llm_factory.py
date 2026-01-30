"""Factory for creating LLM instances based on provider."""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Ensure .env is loaded
load_dotenv()

LOGGER = logging.getLogger(__name__)

Provider = Literal["openai", "anthropic"]

OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5",
]

ANTHROPIC_MODEL_DISPLAY_NAMES = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


def get_models_for_provider(provider: Provider) -> list[str]:
    """Return available models for the given provider.

    Args:
        provider: The LLM provider ("openai" or "anthropic").

    Returns:
        List of model identifiers available for the provider.
    """
    if provider == "anthropic":
        return ANTHROPIC_MODELS.copy()
    return OPENAI_MODELS.copy()


def create_llm(
    provider: Provider,
    model_name: str,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Create an LLM instance based on the provider.

    Args:
        provider: The LLM provider ("openai" or "anthropic").
        model_name: The model identifier.
        max_tokens: Maximum tokens for response (required for Anthropic).
        api_key: Optional API key. If not provided, reads from environment.

    Returns:
        Configured LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        LOGGER.debug("Creating ChatAnthropic with model=%s, max_tokens=%d", model_name, max_tokens)
        return ChatAnthropic(model=model_name, max_tokens=max_tokens, api_key=key)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        LOGGER.debug("Creating ChatOpenAI with model=%s", model_name)
        return ChatOpenAI(model=model_name, api_key=key)

    raise ValueError(f"Unsupported provider: {provider}")
