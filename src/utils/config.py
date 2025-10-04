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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and from `.env` if available.

    Returns:
        Loaded :class:`Settings` instance.
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        LOGGER.warning("OPENAI_API_KEY is not set. The app will fail when translating.")

    return Settings(openai_api_key=openai_api_key)
