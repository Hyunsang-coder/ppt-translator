"""Tests for provider/model-specific LLM configuration."""

import pytest
from langchain_core.rate_limiters import InMemoryRateLimiter

from src.chains.llm_factory import create_llm


@pytest.mark.parametrize("model_name", ["gpt-5.6-sol", "gpt-5.6-luna"])
def test_gpt_56_models_use_high_reasoning_without_temperature(model_name: str) -> None:
    """Both GPT-5.6 choices should be configured as high-reasoning models."""
    llm = create_llm(
        provider="openai",
        model_name=model_name,
        api_key="test-key",
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=100,
            check_every_n_seconds=0.01,
            max_bucket_size=100,
        ),
        temperature=0,
    )

    assert llm.model_name == model_name
    assert llm.reasoning_effort == "high"
    assert llm.temperature is None
