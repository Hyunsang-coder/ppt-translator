"""Basic unit tests for non-LLM components."""

from __future__ import annotations

import types
import unittest

from src.chains.translation_chain import (
    TranslationOutput,
    _batch_translate_with_retry,
    _force_match_expected,
    translate_with_progress,
)
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import chunk_paragraphs, split_text_into_segments
from src.utils.language_detector import LanguageDetector


def _fake_paragraph(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(original_text=text)


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


class HelperTestCase(unittest.TestCase):
    def test_chunk_paragraphs_preserves_order(self) -> None:
        paragraphs = [_fake_paragraph(f"Paragraph {idx}") for idx in range(3)]
        batches = chunk_paragraphs(paragraphs, batch_size=2, ppt_context="ctx", glossary_terms="glossary")
        self.assertEqual(len(batches), 2)
        self.assertIn("1. Paragraph 0", batches[0]["texts"])

    def test_split_text_into_segments_respects_segment_count(self) -> None:
        segments = split_text_into_segments("abcdefghij", 3, weights=[10, 5, 5])
        self.assertEqual(len(segments), 3)
        self.assertEqual("".join(segments), "abcdefghij")


class TeamRulesInjectionTestCase(unittest.TestCase):
    """WP-C1: the rules slice reaches the rendered prompt (no LLM call)."""

    def _render(self, team_rules: str) -> str:
        from langchain_core.prompts import PromptTemplate
        from src.chains.translation_chain import PROMPT_TEMPLATE

        return PromptTemplate.from_template(PROMPT_TEMPLATE).format(
            team_rules=team_rules,
            ppt_context="ctx",
            context="bg",
            glossary_terms="None",
            instructions="inst",
            length_constraint="",
            source_lang="영어",
            target_lang="한국어",
            expected_count=1,
            texts="1. Hello",
        )

    def test_rules_appear_in_prompt_when_connected(self) -> None:
        rendered = self._render("- 총기 사운드, not 사격음\n  avoid: 사격음 → use: 총기 사운드")
        self.assertIn("Team Translation Rules", rendered)
        self.assertIn("총기 사운드", rendered)

    def test_prompt_has_none_placeholder_when_not_connected(self) -> None:
        rendered = self._render("None")
        # The header is always present, but the slice is the "None" sentinel.
        self.assertIn("**Team Translation Rules", rendered)
        rules_block = rendered.split("**Team Translation Rules")[1].split("**Context")[0]
        self.assertIn("None", rules_block)
        self.assertNotIn("총기 사운드", rendered)


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
