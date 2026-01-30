"""Application configuration utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    """Container for runtime configuration values."""

    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str] = None
    max_upload_size_mb: int = 200
    batch_size: int = 80
    max_retries: int = 3
    min_batch_size: int = 60
    max_batch_size: int = 100
    target_batch_count: int = 5
    max_concurrency: int = 8
    wave_multiplier: float = 1.2
    tpm_limit: int = 30_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment and from `.env` if available.

    Returns:
        Loaded :class:`Settings` instance.
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    base_settings = Settings(openai_api_key=openai_api_key, anthropic_api_key=anthropic_api_key)

    concurrency_raw = os.getenv("TRANSLATION_MAX_CONCURRENCY")
    max_concurrency = base_settings.max_concurrency

    if concurrency_raw:
        try:
            max_concurrency = max(1, int(concurrency_raw))
        except ValueError:
            LOGGER.warning(
                "Invalid TRANSLATION_MAX_CONCURRENCY=%s; falling back to %d.",
                concurrency_raw,
                max_concurrency,
            )

    batch_size = base_settings.batch_size
    batch_raw = os.getenv("TRANSLATION_BATCH_SIZE")
    if batch_raw:
        try:
            batch_size = max(1, int(batch_raw))
        except ValueError:
            LOGGER.warning("Invalid TRANSLATION_BATCH_SIZE=%s; using default %d.", batch_raw, batch_size)

    min_batch_size = base_settings.min_batch_size
    min_batch_raw = os.getenv("TRANSLATION_MIN_BATCH_SIZE")
    if min_batch_raw:
        try:
            min_batch_size = max(1, int(min_batch_raw))
        except ValueError:
            LOGGER.warning("Invalid TRANSLATION_MIN_BATCH_SIZE=%s; using default %d.", min_batch_raw, min_batch_size)

    max_batch_size = base_settings.max_batch_size
    max_batch_raw = os.getenv("TRANSLATION_MAX_BATCH_SIZE")
    if max_batch_raw:
        try:
            max_batch_size = max(min_batch_size, int(max_batch_raw))
        except ValueError:
            LOGGER.warning("Invalid TRANSLATION_MAX_BATCH_SIZE=%s; using default %d.", max_batch_raw, max_batch_size)

    if max_batch_size < min_batch_size:
        max_batch_size = min_batch_size

    batch_size = max(min_batch_size, min(max_batch_size, batch_size))

    wave_multiplier = base_settings.wave_multiplier
    wave_raw = os.getenv("TRANSLATION_WAVE_MULTIPLIER")
    if wave_raw:
        try:
            wave_multiplier = max(0.5, float(wave_raw))
        except ValueError:
            LOGGER.warning("Invalid TRANSLATION_WAVE_MULTIPLIER=%s; using default %.2f.", wave_raw, wave_multiplier)

    target_batch_count = base_settings.target_batch_count
    target_raw = os.getenv("TRANSLATION_TARGET_BATCH_COUNT")
    if target_raw:
        try:
            target_batch_count = max(1, int(target_raw))
        except ValueError:
            LOGGER.warning(
                "Invalid TRANSLATION_TARGET_BATCH_COUNT=%s; using default %d.",
                target_raw,
                target_batch_count,
            )

    tpm_limit_raw = os.getenv("TRANSLATION_TPM_LIMIT")
    tpm_limit = base_settings.tpm_limit

    if tpm_limit_raw:
        try:
            tpm_limit = max(1_000, int(tpm_limit_raw))
        except ValueError:
            LOGGER.warning(
                "Invalid TRANSLATION_TPM_LIMIT=%s; falling back to %d.",
                tpm_limit_raw,
                tpm_limit,
            )

    if not openai_api_key and not anthropic_api_key:
        LOGGER.warning("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set. The app will fail when translating.")

    return replace(
        base_settings,
        batch_size=batch_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        max_concurrency=max_concurrency,
        wave_multiplier=wave_multiplier,
        target_batch_count=target_batch_count,
        tpm_limit=tpm_limit,
    )
