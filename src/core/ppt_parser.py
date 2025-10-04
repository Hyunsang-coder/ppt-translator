"""Utilities for parsing PPT/PPTX files into structured paragraph data."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from pptx.text.text import Paragraph
else:
    Paragraph = Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ParagraphInfo:
    """Metadata for a single paragraph in the presentation."""

    slide_index: int
    shape_index: int
    paragraph_index: int
    original_text: str
    paragraph: Paragraph
    slide_title: str | None


class PPTParser:
    """Extract text content and structure from PPT presentations."""

    def extract_paragraphs(self, ppt_file: io.BytesIO) -> Tuple[List[ParagraphInfo], Presentation]:
        """Parse the uploaded PPT file and collect paragraphs.

        Args:
            ppt_file: In-memory PPT/PPTX file buffer.

        Returns:
            A tuple containing the list of paragraphs and the loaded presentation object.
        """

        ppt_file.seek(0)
        presentation = Presentation(ppt_file)
        paragraphs: List[ParagraphInfo] = []

        for slide_index, slide in enumerate(presentation.slides):
            slide_title = None
            if slide.shapes.title and slide.shapes.title.text:
                slide_title = slide.shapes.title.text

            for shape_index, shape in enumerate(slide.shapes):
                paragraphs.extend(
                    self._extract_from_shape(
                        shape=shape,
                        slide_index=slide_index,
                        shape_index=shape_index,
                        slide_title=slide_title,
                    )
                )

        LOGGER.info("Extracted %d paragraphs from %d slides.", len(paragraphs), len(presentation.slides))
        return paragraphs, presentation

    def _extract_from_shape(
        self,
        shape,
        slide_index: int,
        shape_index: int,
        slide_title: str | None,
    ) -> List[ParagraphInfo]:
        """Recursively extract paragraphs from a shape and its children."""

        collected: List[ParagraphInfo] = []

        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for child in shape.shapes:  # type: ignore[attr-defined]
                collected.extend(
                    self._extract_from_shape(
                        shape=child,
                        slide_index=slide_index,
                        shape_index=shape_index,
                        slide_title=slide_title,
                    )
                )
            return collected

        if getattr(shape, "has_table", False):
            table = shape.table
            collected.extend(
                self._extract_from_table(
                    table=table,
                    slide_index=slide_index,
                    shape_index=shape_index,
                    slide_title=slide_title,
                )
            )

        if getattr(shape, "has_text_frame", False):
            text_frame = shape.text_frame
            collected.extend(
                self._collect_paragraphs_from_text_frame(
                    text_frame.paragraphs,
                    slide_index,
                    shape_index,
                    slide_title,
                )
            )

        return collected

    def _extract_from_table(
        self,
        table,
        slide_index: int,
        shape_index: int,
        slide_title: str | None,
    ) -> List[ParagraphInfo]:
        """Extract paragraphs from a table shape."""

        collected: List[ParagraphInfo] = []
        for row in table.rows:
            for cell in row.cells:
                if not getattr(cell, "text_frame", None):
                    continue
                collected.extend(
                    self._collect_paragraphs_from_text_frame(
                        cell.text_frame.paragraphs,
                        slide_index,
                        shape_index,
                        slide_title,
                    )
                )
        return collected

    @staticmethod
    def _collect_paragraphs_from_text_frame(
        paragraphs: Sequence[Paragraph],
        slide_index: int,
        shape_index: int,
        slide_title: str | None,
    ) -> List[ParagraphInfo]:
        """Convert pptx paragraphs into ParagraphInfo instances."""

        collected: List[ParagraphInfo] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            text = "".join(run.text for run in paragraph.runs)
            if not text or not text.strip():
                continue

            collected.append(
                ParagraphInfo(
                    slide_index=slide_index,
                    shape_index=shape_index,
                    paragraph_index=paragraph_index,
                    original_text=text,
                    paragraph=paragraph,
                    slide_title=slide_title,
                )
            )
        return collected
