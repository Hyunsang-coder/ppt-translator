"""Basic unit tests for non-LLM components."""

from __future__ import annotations

import asyncio
import types
import unittest

import pytest

from src.chains.translation_chain import (
    PROMPT_TEMPLATE,
    TranslationOutput,
    _batch_translate_with_retry,
    _force_match_expected,
    translate_with_progress_async,
    translate_with_progress,
)
from src.chains.context_manager import ContextManager
from src.chains.llm_factory import get_shared_rate_limiter
from src.chains.summarization_chain import SUMMARIZATION_PROMPT
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import (
    chunk_paragraphs,
    chunk_paragraphs_by_tokens,
    split_text_into_segments,
)
from src.utils.language_detector import LanguageDetector


def _fake_paragraph(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(original_text=text)


def _fake_context_paragraph(
    text: str,
    slide_index: int,
    title: str,
    *,
    is_note: bool = False,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        original_text=text,
        slide_index=slide_index,
        slide_title=title,
        is_note=is_note,
    )


class LanguageDetectorTestCase(unittest.TestCase):
    def test_map_lang_code_known_language(self) -> None:
        detector = LanguageDetector()
        self.assertEqual(detector.map_lang_code("ko"), "한국어")

    def test_map_lang_code_unknown_language(self) -> None:
        detector = LanguageDetector()
        self.assertEqual(detector.map_lang_code("xx"), "영어")


class GlossaryLoaderTestCase(unittest.TestCase):
    def test_apply_glossary_to_texts(self) -> None:
        glossary = {"PUBG": "배틀그라운드"}
        updated = GlossaryLoader.apply_glossary_to_texts(["I love PUBG"], glossary)
        self.assertEqual(updated[0], "I love 배틀그라운드")


class RateLimiterTestCase(unittest.TestCase):
    def test_shared_rate_limiter_is_scoped_by_provider_and_key(self) -> None:
        first = get_shared_rate_limiter("openai", "same-key")
        second = get_shared_rate_limiter("openai", "same-key")
        different_key = get_shared_rate_limiter("openai", "other-key")
        different_provider = get_shared_rate_limiter("anthropic", "same-key")

        self.assertIs(first, second)
        self.assertIsNot(first, different_key)
        self.assertIsNot(first, different_provider)


class PromptTemplateTestCase(unittest.TestCase):
    def test_translation_prompt_contains_ppt_preservation_rules(self) -> None:
        self.assertIn("Current Batch Context", PROMPT_TEMPLATE)
        self.assertIn("Glossary (highest priority terminology rules)", PROMPT_TEMPLATE)
        self.assertIn("Preserve numbers, units, dates", PROMPT_TEMPLATE)
        self.assertIn("Return exactly {expected_count}", PROMPT_TEMPLATE)
        self.assertIn("Do not add explanations", PROMPT_TEMPLATE)

    def test_summarization_prompt_requests_translation_relevant_context(self) -> None:
        self.assertIn("500자 이내", SUMMARIZATION_PROMPT)
        self.assertIn("핵심 용어/고유명사", SUMMARIZATION_PROMPT)
        self.assertIn("번역상 주의", SUMMARIZATION_PROMPT)
        self.assertIn("원문 표기를 유지", SUMMARIZATION_PROMPT)


class ContextManagerTestCase(unittest.TestCase):
    def test_build_batch_context_marks_current_and_nearby_text(self) -> None:
        paragraphs = [
            _fake_context_paragraph("Intro", 0, "Overview"),
            _fake_context_paragraph("Metric +20%", 1, "Results"),
            _fake_context_paragraph("Launch date: 2026-07-01", 1, "Results"),
            _fake_context_paragraph("Appendix", 2, "Appendix"),
        ]

        context = ContextManager(paragraphs).build_batch_context(
            1,
            3,
            window=1,
        )

        self.assertIn("Use nearby text only", context)
        self.assertIn("(nearby) Slide 1 | Overview: Intro", context)
        self.assertIn("(current batch) Slide 2 | Results: Metric +20%", context)
        self.assertIn("Launch date: 2026-07-01", context)


class HelperTestCase(unittest.TestCase):
    def test_chunk_paragraphs_preserves_order(self) -> None:
        paragraphs = [_fake_paragraph(f"Paragraph {idx}") for idx in range(3)]
        batches = chunk_paragraphs(paragraphs, batch_size=2, ppt_context="ctx", glossary_terms="glossary")
        self.assertEqual(len(batches), 2)
        self.assertIn("1. Paragraph 0", batches[0]["texts"])

    def test_chunk_paragraphs_by_tokens_splits_long_texts(self) -> None:
        paragraphs = [
            _fake_paragraph("short"),
            _fake_paragraph("x" * 200),
            _fake_paragraph("tail"),
        ]

        batches = chunk_paragraphs_by_tokens(
            paragraphs,
            max_items=10,
            max_tokens=420,
            ppt_context="ctx",
            glossary_terms="glossary",
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]["start_idx"], 1)
        self.assertEqual(batches[0]["end_idx"], 2)
        self.assertEqual(batches[1]["start_idx"], 3)

    def test_split_text_into_segments_respects_segment_count(self) -> None:
        segments = split_text_into_segments("abcdefghij", 3, weights=[10, 5, 5])
        self.assertEqual(len(segments), 3)
        self.assertEqual("".join(segments), "abcdefghij")


class BatchRetryTestCase(unittest.TestCase):
    def test_retry_only_resubmits_failed_batches(self) -> None:
        # Two batches; the first attempt yields only batch 0 (batch 1 missing,
        # which raises and triggers a tenacity retry). The retry must resubmit
        # ONLY batch 1 — batch 0 was already translated and must not re-run.
        batches = [
            {"paragraphs": [object()], "start_idx": 1, "end_idx": 1},
            {"paragraphs": [object()], "start_idx": 2, "end_idx": 2},
        ]
        submitted_sizes = []

        class FakeChain:
            def __init__(self):
                self.calls = 0

            def batch_as_completed(self, submitted, config=None):
                submitted_sizes.append(len(submitted))
                self.calls += 1
                if self.calls == 1:
                    # Only the first of two batches completes.
                    yield (0, TranslationOutput(translations=["b0"]))
                else:
                    # Retry receives only the still-pending batch.
                    yield (0, TranslationOutput(translations=["b1"]))

        accumulator = [None, None]
        results = _batch_translate_with_retry(
            FakeChain(), batches, config=None,
            total_batches=2, ordered_results=accumulator,
        )

        self.assertEqual([r.translations for r in results], [["b0"], ["b1"]])
        # First attempt submitted 2 batches, retry submitted only 1.
        self.assertEqual(submitted_sizes, [2, 1])

    def test_count_mismatch_retries_single_batch(self) -> None:
        batches = [
            {
                "paragraphs": [_fake_paragraph("a"), _fake_paragraph("b")],
                "start_idx": 1,
                "end_idx": 2,
            }
        ]

        class FakeChain:
            def __init__(self):
                self.invoke_calls = 0

            def batch_as_completed(self, submitted, config=None):
                yield (0, TranslationOutput(translations=["하나만"]))

            def invoke(self, batch, config=None):
                self.invoke_calls += 1
                return TranslationOutput(translations=["하나", "둘"])

        chain = FakeChain()
        result = translate_with_progress(chain, batches, max_concurrency=1)

        self.assertEqual(result, ["하나", "둘"])
        self.assertEqual(chain.invoke_calls, 1)

    def test_count_mismatch_retry_falls_back_to_originals(self) -> None:
        batches = [
            {
                "paragraphs": [_fake_paragraph("a"), _fake_paragraph("b")],
                "start_idx": 1,
                "end_idx": 2,
            }
        ]

        class FakeChain:
            def batch_as_completed(self, submitted, config=None):
                yield (0, TranslationOutput(translations=["하나만"]))

            def invoke(self, batch, config=None):
                return TranslationOutput(translations=["아직 하나만"])

        result = translate_with_progress(FakeChain(), batches, max_concurrency=1)

        self.assertEqual(result, ["하나만", "b"])


@pytest.mark.asyncio
async def test_async_count_mismatch_retries_single_batch() -> None:
    batches = [
        {
            "paragraphs": [_fake_paragraph("a"), _fake_paragraph("b")],
            "start_idx": 1,
            "end_idx": 2,
        }
    ]

    class FakeChain:
        def __init__(self):
            self.invoke_calls = 0

        async def abatch_as_completed(self, submitted, config=None):
            yield (0, TranslationOutput(translations=["하나만"]))

        async def ainvoke(self, batch, config=None):
            self.invoke_calls += 1
            return TranslationOutput(translations=["하나", "둘"])

    chain = FakeChain()
    result = await translate_with_progress_async(chain, batches, max_concurrency=1)

    assert result == ["하나", "둘"]
    assert chain.invoke_calls == 1


@pytest.mark.asyncio
async def test_async_translation_batch_can_be_cancelled() -> None:
    batches = [{"paragraphs": [_fake_paragraph("a")], "start_idx": 1, "end_idx": 1}]

    class SlowChain:
        async def abatch_as_completed(self, submitted, config=None):
            await asyncio.sleep(999)
            yield (0, TranslationOutput(translations=["never"]))

    task = asyncio.create_task(
        translate_with_progress_async(SlowChain(), batches, max_concurrency=1)
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


class ForceMatchExpectedTestCase(unittest.TestCase):
    def test_pads_missing_with_original_text_not_empty(self) -> None:
        # LLM returned 1 of 3 translations; the 2 missing must fall back to
        # the original text, never empty strings (which would erase the slide).
        result = _force_match_expected(["번역"], 3, originals=["a", "b", "c"])
        self.assertEqual(result, ["번역", "b", "c"])

    def test_pads_with_empty_when_no_originals(self) -> None:
        result = _force_match_expected(["x"], 3)
        self.assertEqual(result, ["x", "", ""])

    def test_trims_extras(self) -> None:
        result = _force_match_expected(["a", "b", "c"], 2, originals=["a", "b"])
        self.assertEqual(result, ["a", "b"])

    def test_exact_count_unchanged(self) -> None:
        result = _force_match_expected(["a", "b"], 2, originals=["x", "y"])
        self.assertEqual(result, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
