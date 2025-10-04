"""Basic unit tests for non-LLM components."""

from __future__ import annotations

import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
