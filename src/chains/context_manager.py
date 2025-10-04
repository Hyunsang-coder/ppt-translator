"""Presentation-wide context helpers to keep translations consistent."""

from __future__ import annotations

from typing import Iterable, List

from src.core.ppt_parser import ParagraphInfo


class ContextManager:
    """Aggregate presentation context for the translation prompts."""

    def __init__(self, paragraphs: Iterable[ParagraphInfo]):
        """Initialise the context manager.

        Args:
            paragraphs: Iterable of paragraph metadata extracted from the PPT.
        """

        self.paragraphs: List[ParagraphInfo] = list(paragraphs)

    def build_global_context(self, max_paragraphs: int = 80, max_chars: int = 2000) -> str:
        """Construct a concise summary of the presentation for the LLM.

        Args:
            max_paragraphs: Maximum number of paragraphs to include.
            max_chars: Character limit for the final context string.

        Returns:
            Formatted multi-line string describing slide order, titles, and text.
        """

        context_lines: List[str] = []
        for paragraph in self.paragraphs[:max_paragraphs]:
            slide_label = f"Slide {paragraph.slide_index + 1}"
            title = paragraph.slide_title or "Untitled"
            text = paragraph.original_text.strip().replace("\n", " ")
            context_lines.append(f"{slide_label} - {title}: {text}")

        context = "\n".join(context_lines)
        return context[:max_chars]

    def build_context_window(self, center_index: int, window: int = 2) -> str:
        """Provide neighbouring paragraph context for more precise translations.

        Args:
            center_index: Index of the paragraph currently being translated.
            window: Number of neighbour paragraphs to include on each side.

        Returns:
            Multi-line string enumerating neighbouring paragraphs.
        """

        start = max(center_index - window, 0)
        end = min(center_index + window + 1, len(self.paragraphs))
        window_paragraphs = self.paragraphs[start:end]
        lines = []
        for idx, paragraph in enumerate(window_paragraphs, start=start + 1):
            lines.append(f"[{idx}] {paragraph.original_text.strip()}")
        return "\n".join(lines)
