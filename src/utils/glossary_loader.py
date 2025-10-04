"""Glossary loading and formatting utilities."""

from __future__ import annotations

import io
import logging
from typing import Dict, Iterable, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


class GlossaryLoader:
    """Load glossary definitions from Excel files for translation prompts."""

    REQUIRED_COLUMNS = 2

    def load_glossary(self, file_data: io.BytesIO) -> Dict[str, str]:
        """Load glossary entries from an uploaded Excel file.

        Args:
            file_data: Bytes originating from the uploaded Excel file.

        Returns:
            Mapping from source term to preferred translation.

        Raises:
            ValueError: If the uploaded file is invalid or empty.
        """

        try:
            file_data.seek(0)
            dataframe = pd.read_excel(file_data, dtype=str, header=None)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Failed to read glossary Excel file: %s", exc)
            raise ValueError("용어집 파일을 읽을 수 없습니다. 형식을 확인해주세요.") from exc

        if dataframe.empty or dataframe.shape[1] < self.REQUIRED_COLUMNS:
            raise ValueError("용어집 파일에 필요한 두 개의 열이 존재하지 않습니다.")

        glossary: Dict[str, str] = {}
        for _, row in dataframe.iterrows():
            source = str(row.iloc[0]).strip()
            target = str(row.iloc[1]).strip()
            if not source or not target:
                continue
            glossary[source] = target

        if not glossary:
            raise ValueError("유효한 용어집 항목을 찾을 수 없습니다.")

        LOGGER.info("Loaded %d glossary entries.", len(glossary))
        return glossary

    @staticmethod
    def format_glossary_terms(glossary: Dict[str, str] | None) -> str:
        """Format glossary entries into a prompt-friendly string.

        Args:
            glossary: Mapping of source terms to their preferred translations.

        Returns:
            Human-readable string representation used inside LLM prompts.
        """

        if not glossary:
            return "None"
        lines = [f"- {src} => {dest}" for src, dest in glossary.items()]
        return "\n".join(lines)

    @staticmethod
    def apply_glossary_to_texts(texts: Iterable[str], glossary: Dict[str, str] | None) -> List[str]:
        """Apply glossary replacements to an iterable of texts.

        Args:
            texts: Original texts to transform.
            glossary: Optional glossary mapping.

        Returns:
            A list of texts with glossary substitutions applied.
        """

        if not glossary:
            return list(texts)

        transformed_texts: List[str] = []
        for text in texts:
            updated = text
            for source, target in glossary.items():
                if source in updated:
                    updated = updated.replace(source, target)
            transformed_texts.append(updated)
        return transformed_texts

    @staticmethod
    def apply_glossary_to_translation(text: str, glossary: Dict[str, str] | None) -> str:
        """Ensure translated text respects glossary preferences.

        Args:
            text: Raw translated text from the LLM.
            glossary: Optional glossary mapping.

        Returns:
            Post-processed text with glossary substitutions enforced.
        """

        if not glossary or not text:
            return text
        updated = text
        for source, target in glossary.items():
            updated = updated.replace(source, target)
        return updated
