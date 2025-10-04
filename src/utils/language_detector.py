"""Language detection helpers built on top of langdetect."""

from __future__ import annotations

import logging
from langdetect import LangDetectException, detect

LOGGER = logging.getLogger(__name__)


class LanguageDetector:
    """Detects source language and infers target language recommendations."""

    _LANG_CODE_MAPPING = {
        "ko": "한국어",
        "en": "영어",
        "ja": "일본어",
        "zh-cn": "중국어",
        "zh": "중국어",
        "es": "스페인어",
        "fr": "프랑스어",
        "de": "독일어",
    }

    _DEFAULT_LANGUAGE = "영어"

    def detect_language(self, text: str) -> str:
        """Detect the language name (in Korean) for the given text.

        Args:
            text: Arbitrary text sample extracted from the presentation.

        Returns:
            Human-friendly language name recognised by the UI.
        """

        sample = (text or "").strip()
        if not sample:
            LOGGER.debug("No text provided for language detection; returning default.")
            return self._DEFAULT_LANGUAGE

        try:
            lang_code = detect(sample[:500])
        except LangDetectException as exc:
            LOGGER.warning("Language detection failed; falling back to default. Error: %s", exc)
            return self._DEFAULT_LANGUAGE

        language = self.map_lang_code(lang_code)
        LOGGER.debug("Detected language code '%s' resolved to '%s'.", lang_code, language)
        return language

    def map_lang_code(self, code: str) -> str:
        """Map a langdetect code into a display name.

        Args:
            code: Two-letter or locale code reported by langdetect.

        Returns:
            Language name compatible with the UI selection options.
        """

        if not code:
            return self._DEFAULT_LANGUAGE
        return self._LANG_CODE_MAPPING.get(code.lower(), self._DEFAULT_LANGUAGE)

    def infer_target_language(self, source_lang: str) -> str:
        """Infer an appropriate target language when 'Auto' is selected.

        Args:
            source_lang: The display name of the source language.

        Returns:
            The inferred target language display name.
        """

        if source_lang == "한국어":
            return "영어"
        if source_lang == "영어":
            return "한국어"
        return "한국어"
