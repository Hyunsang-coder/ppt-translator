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
        """Construct concise presentation context for the LLM.

        Args:
            max_paragraphs: Maximum number of paragraphs to include.
            max_chars: Character limit for the final context string.

        Returns:
            Formatted multi-line string describing slide order, titles, and text.
        """

        context_lines: List[str] = [
            "Use this as presentation context only. Translate only the requested batch texts.",
            "Slide outline:",
        ]
        for paragraph in self.paragraphs[:max_paragraphs]:
            if getattr(paragraph, "is_note", False):
                continue
            slide_label = f"Slide {paragraph.slide_index + 1}"
            title = paragraph.slide_title or "Untitled"
            text = paragraph.original_text.strip().replace("\n", " ")
            context_lines.append(f"- {slide_label} | title: {title} | text: {text}")

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
            marker = "current" if idx == center_index + 1 else "nearby"
            lines.append(self._format_context_line(idx, paragraph, marker=marker))
        return "\n".join(lines)

    def build_batch_context(
        self,
        start_index: int,
        end_index: int,
        *,
        window: int = 2,
        max_chars: int = 1200,
    ) -> str:
        """Provide neighbouring context around a translation batch.

        Args:
            start_index: Zero-based inclusive batch start index.
            end_index: Zero-based exclusive batch end index.
            window: Number of neighbouring paragraphs to include on each side.
            max_chars: Character limit for the returned context.

        Returns:
            Multi-line context containing nearby and current-batch paragraphs.
        """
        if not self.paragraphs:
            return "No nearby context."

        start_index = max(0, start_index)
        end_index = max(start_index + 1, min(end_index, len(self.paragraphs)))
        window_start = max(start_index - window, 0)
        window_end = min(end_index + window, len(self.paragraphs))

        lines = [
            "Use nearby text only to disambiguate terminology, pronouns, and tone.",
        ]
        for idx in range(window_start, window_end):
            paragraph = self.paragraphs[idx]
            marker = "current batch" if start_index <= idx < end_index else "nearby"
            lines.append(self._format_context_line(idx + 1, paragraph, marker=marker))

        context = "\n".join(lines)
        return context[:max_chars]

    @staticmethod
    def _format_context_line(index: int, paragraph: ParagraphInfo, *, marker: str) -> str:
        """Format one paragraph for prompt context."""
        slide_label = f"Slide {paragraph.slide_index + 1}"
        title = paragraph.slide_title or "Untitled"
        text = paragraph.original_text.strip().replace("\n", " ")
        note = " notes" if getattr(paragraph, "is_note", False) else ""
        return f"[{index}] ({marker}{note}) {slide_label} | {title}: {text}"
