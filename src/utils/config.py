"""Application configuration utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    """Container for runtime configuration values."""

    openai_api_key: Optional[str]
    max_upload_size_mb: int = 50
    batch_size: int = 400
    max_retries: int = 3
    min_batch_size: int = 40
    target_batch_count: int = 5
    max_concurrency: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and from `.env` if available.

    Returns:
        Loaded :class:`Settings` instance.
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    concurrency_raw = os.getenv("TRANSLATION_MAX_CONCURRENCY")
    max_concurrency = 1

    if concurrency_raw:
        try:
            max_concurrency = max(1, int(concurrency_raw))
        except ValueError:
            LOGGER.warning(
                "Invalid TRANSLATION_MAX_CONCURRENCY=%s; falling back to 1.",
                concurrency_raw,
            )

    if not openai_api_key:
        LOGGER.warning("OPENAI_API_KEY is not set. The app will fail when translating.")

    return Settings(openai_api_key=openai_api_key, max_concurrency=max_concurrency)
