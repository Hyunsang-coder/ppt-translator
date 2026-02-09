"""Tests for TranslationService._determine_batch_size edge cases."""

from __future__ import annotations

import pytest

from src.services.translation_service import TranslationService
from src.utils.config import Settings


def _make_service(**overrides) -> TranslationService:
    """Create a TranslationService with custom settings."""
    defaults = dict(
        openai_api_key="test",
        anthropic_api_key="test",
        max_concurrency=8,
        batch_size=80,
        min_batch_size=60,
        max_batch_size=100,
        target_batch_count=5,
        wave_multiplier=1.2,
        tpm_limit=30000,
        rate_limit_requests_per_second=1.0,
        rate_limit_check_interval=0.1,
        rate_limit_max_bucket_size=10,
        max_upload_size_mb=50,
    )
    defaults.update(overrides)
    settings = Settings(**defaults)
    return TranslationService(settings=settings)


class TestDetermineBatchSize:
    """Tests for batch size calculation edge cases."""

    def test_zero_paragraphs(self):
        """Zero paragraphs should return 1."""
        service = _make_service()
        assert service._determine_batch_size(0) == 1

    def test_one_paragraph(self):
        """Single paragraph should return 1."""
        service = _make_service()
        assert service._determine_batch_size(1) == 1

    def test_small_count_no_division_by_zero(self):
        """Small paragraph counts should not cause division by zero.

        When actual_batches=1, the remainder adjustment path
        divides by (actual_batches - 1) which would be 0.
        """
        service = _make_service(min_batch_size=10, max_batch_size=100)
        # This should not raise ZeroDivisionError
        result = service._determine_batch_size(5)
        assert result >= 1
        assert result <= 5

    def test_exact_batch_size_multiple(self):
        """Paragraphs exactly divisible by batch size."""
        service = _make_service(min_batch_size=10, max_batch_size=10)
        result = service._determine_batch_size(30)
        assert result == 10

    def test_result_never_exceeds_total(self):
        """Batch size should never exceed total paragraphs."""
        service = _make_service()
        for n in range(1, 50):
            result = service._determine_batch_size(n)
            assert result <= n, f"batch_size {result} > total {n}"

    def test_result_always_positive(self):
        """Batch size should always be >= 1."""
        service = _make_service()
        for n in range(0, 100):
            result = service._determine_batch_size(n)
            assert result >= 1, f"batch_size {result} < 1 for total {n}"

    def test_large_paragraph_count(self):
        """Large paragraph count should produce reasonable batch sizes."""
        service = _make_service()
        result = service._determine_batch_size(10000)
        assert 60 <= result <= 100  # within configured bounds
