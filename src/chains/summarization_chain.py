"""Summarization chain for generating presentation context."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from langchain_core.prompts import PromptTemplate

from src.chains.llm_factory import Provider, create_llm

LOGGER = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """프레젠테이션 번역에 필요한 핵심 컨텍스트를 **300자 이내**로 추출하세요.

**프레젠테이션:**
{markdown}

**규칙:**
- 반드시 300자 이내
- 핵심만 간결하게

**출력 형식:**
- 문서 유형/주제
- 대상 독자
- 주요 용어나 고유명사 (있으면)

**예시:**
> 게임 이벤트 운영 전략 문서. 비수기 트래픽 유지를 위한 콘텐츠/이벤트 기획안. 대상: 이벤트 기획자, 마케터. 주요 용어: Binary Spot, Heist Royale.

**컨텍스트:**"""


def create_summarization_prompt() -> PromptTemplate:
    """Create the prompt template for summarization.

    Returns:
        PromptTemplate configured for presentation summarization.
    """
    return PromptTemplate(
        input_variables=["markdown"],
        template=SUMMARIZATION_PROMPT,
    )


async def summarize_presentation(
    markdown: str,
    provider: Provider = "openai",
    model: str = "gpt-5-mini",
    max_chars: int = 50000,
) -> str:
    """Summarize presentation content for translation context.

    Args:
        markdown: The extracted markdown content from the presentation.
        provider: LLM provider to use ("openai" or "anthropic").
        model: Model identifier to use.
        max_chars: Maximum characters to process (truncates if exceeded).

    Returns:
        Summary string suitable for use as translation context.

    Raises:
        ValueError: If the markdown is empty or provider is invalid.
    """
    if not markdown or not markdown.strip():
        raise ValueError("Markdown content is empty")

    # Truncate if too long to avoid token limits
    truncated = markdown[:max_chars] if len(markdown) > max_chars else markdown
    if len(markdown) > max_chars:
        LOGGER.warning(
            "Markdown truncated from %d to %d characters for summarization",
            len(markdown),
            max_chars,
        )

    LOGGER.info("Summarizing presentation with provider=%s, model=%s", provider, model)

    llm = create_llm(
        provider=provider,
        model_name=model,
        max_tokens=512,  # Limit for concise output (~300 chars)
    )

    prompt = create_summarization_prompt()
    chain = prompt | llm

    # Run in executor for async compatibility
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: chain.invoke({"markdown": truncated}),
    )

    summary = result.content if hasattr(result, "content") else str(result)
    summary = summary.strip()

    LOGGER.info("Generated summary: %d characters", len(summary))

    return summary
