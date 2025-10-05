"""Write translated text back into the original PPT presentation."""

from __future__ import annotations

import io
import logging
from typing import Iterable, List

from pptx import Presentation

from src.utils.helpers import split_text_into_segments

LOGGER = logging.getLogger(__name__)


class PPTWriter:
    """Apply translated strings back into PPT paragraphs."""

    def apply_translations(
        self,
        paragraphs,
        translations: Iterable[str],
        presentation: Presentation,
    ) -> io.BytesIO:
        """Apply translated paragraphs and return the updated PPT as bytes.

        Args:
            paragraphs: Iterable of :class:`ParagraphInfo` objects sharing paragraph references.
            translations: Translated strings aligned to the provided paragraphs.
            presentation: Loaded presentation object to mutate in place.

        Returns:
            BytesIO buffer containing the updated PPTX file.
        """

        translation_list: List[str] = list(translations)
        total = len(translation_list)
        LOGGER.info("Starting to apply %d translated paragraphs to presentation.", total)

        for idx, (paragraph_info, translation) in enumerate(zip(paragraphs, translation_list), start=1):
            if idx % 100 == 0 or idx == total:
                LOGGER.info("Applying translation %d/%d...", idx, total)

            paragraph = paragraph_info.paragraph
            runs = list(paragraph.runs)

            if not runs:
                run = paragraph.add_run()
                runs = [run]

            weights = [max(len(run.text), 1) for run in runs]
            segments = split_text_into_segments(translation, len(runs), weights=weights)

            for run, segment in zip(runs, segments):
                run.text = segment

        LOGGER.info("Saving translated presentation to buffer...")
        buffer = io.BytesIO()
        presentation.save(buffer)
        buffer.seek(0)
        LOGGER.info("Applied %d translated paragraphs to presentation.", len(translation_list))
        return buffer
