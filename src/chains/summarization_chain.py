"""Summarization chain for generating presentation context."""

from __future__ import annotations

import logging

from langchain_core.prompts import PromptTemplate

from src.chains.llm_factory import Provider, create_llm
from src.services.models import DEFAULT_LIGHT_MODEL

LOGGER = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """프레젠테이션 번역 품질을 높이기 위한 핵심 컨텍스트를 **500자 이내**로 추출하세요.

**프레젠테이션:**
{markdown}

**규칙:**
- 반드시 500자 이내
- 번역자가 실제로 참고할 정보만 남기세요.
- 원문 고유명사, 제품명, 기능명, 캠페인명은 임의 번역하지 말고 원문 표기를 유지하세요.
- 추측이 필요한 항목은 쓰지 마세요.

**출력 형식:**
- 문서 유형/목적:
- 대상 독자:
- 핵심 용어/고유명사:
- 번역상 주의:

**예시:**
문서 유형/목적: 게임 이벤트 운영 전략 문서. 비수기 트래픽 유지를 위한 콘텐츠/이벤트 기획안.
대상 독자: 이벤트 기획자, 마케터.
핵심 용어/고유명사: Binary Spot, Heist Royale.
번역상 주의: 이벤트명과 게임 모드는 원문 유지, 운영 지표는 숫자/단위 보존.

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
    model: str = DEFAULT_LIGHT_MODEL["openai"],
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
        max_tokens=700,  # Limit for concise output (~500 chars)
        temperature=0.2,
    )

    prompt = create_summarization_prompt()
    chain = prompt | llm

    result = await chain.ainvoke({"markdown": truncated})

    summary = result.content if hasattr(result, "content") else str(result)
    summary = summary.strip()

    LOGGER.info("Generated summary: %d characters", len(summary))

    return summary
