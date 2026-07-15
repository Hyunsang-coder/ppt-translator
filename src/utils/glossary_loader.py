"""Glossary loading and formatting utilities."""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from src.utils.security import validate_excel_file

LOGGER = logging.getLogger(__name__)

# Header labels used by glossary_template.xlsx / OddEyes-style CSV exports.
_HEADER_SOURCE_ALIASES = frozenset({"원문", "source", "소스", "용어", "term"})
_HEADER_TARGET_ALIASES = frozenset({"번역", "target", "타겟", "번역어", "translation"})


class GlossaryLoader:
    """Load glossary definitions from Excel files or JSON for translation prompts."""

    REQUIRED_COLUMNS = 2
    MAX_FILE_SIZE_MB = 10  # Maximum Excel file size: 10MB
    MAX_ENTRIES = 5000
    MAX_JSON_CHARS = 1_000_000
    MAX_TERM_CHARS = 500

    def load_glossary(self, file_data: io.BytesIO) -> Dict[str, str]:
        """Load glossary entries from an uploaded Excel file.

        Args:
            file_data: Bytes originating from the uploaded Excel file.

        Returns:
            Mapping from source term to preferred translation.

        Raises:
            ValueError: If the uploaded file is invalid or empty.
        """

        # Validate file size
        file_data.seek(0, io.SEEK_END)
        file_size = file_data.tell()
        file_data.seek(0)
        
        size_mb = file_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(f"용어집 파일 크기가 {self.MAX_FILE_SIZE_MB}MB를 초과합니다. 더 작은 파일로 다시 시도해주세요.")

        # Validate file signature
        is_valid, error_msg = validate_excel_file(file_data)
        if not is_valid:
            raise ValueError(error_msg or "파일 형식이 올바르지 않습니다.")

        try:
            file_data.seek(0)
            dataframe = pd.read_excel(file_data, dtype=str, header=None)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Failed to read glossary Excel file: %s", exc)
            raise ValueError("용어집 파일을 읽을 수 없습니다. 형식을 확인해주세요.") from exc

        if dataframe.empty or dataframe.shape[1] < self.REQUIRED_COLUMNS:
            raise ValueError("용어집 파일에 필요한 두 개의 열이 존재하지 않습니다.")

        rows = [
            (str(row.iloc[0]).strip(), str(row.iloc[1]).strip())
            for _, row in dataframe.iterrows()
        ]
        glossary = self.from_pairs(rows, require_non_empty=True)

        LOGGER.info("Loaded %d glossary entries.", len(glossary))
        return glossary

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
        *,
        require_non_empty: bool = False,
    ) -> Dict[str, str]:
        """Build a glossary dict from (source, target) pairs.

        Skips a leading header row when the first pair looks like column labels.
        Duplicate sources upsert (last value wins). Enforces entry/term limits.
        """
        glossary: Dict[str, str] = {}
        started = False
        for source, target in pairs:
            source = (source or "").strip()
            target = (target or "").strip()
            if not source or not target:
                continue
            if not started and cls._is_header_row(source, target):
                started = True
                continue
            started = True
            if len(source) > cls.MAX_TERM_CHARS or len(target) > cls.MAX_TERM_CHARS:
                raise ValueError(
                    f"용어 길이는 {cls.MAX_TERM_CHARS}자를 초과할 수 없습니다."
                )
            glossary[source] = target
            if len(glossary) > cls.MAX_ENTRIES:
                raise ValueError(
                    f"용어집 항목은 최대 {cls.MAX_ENTRIES}개까지 지원합니다."
                )

        if require_non_empty and not glossary:
            raise ValueError("유효한 용어집 항목을 찾을 수 없습니다.")
        return glossary

    @classmethod
    def from_json(cls, raw: str | None) -> Dict[str, str] | None:
        """Parse a glossary JSON payload into a mapping.

        Accepted shapes:
        - ``{"PUBG": "배틀그라운드", ...}``
        - ``[{"source": "PUBG", "target": "배틀그라운드"}, ...]``

        Empty / whitespace / ``{}`` / ``[]`` return ``None`` (no glossary).
        """
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        if len(text) > cls.MAX_JSON_CHARS:
            raise ValueError(
                f"용어집 JSON이 {cls.MAX_JSON_CHARS}자를 초과합니다."
            )
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("용어집 JSON 형식이 올바르지 않습니다.") from exc

        if payload is None:
            return None
        if isinstance(payload, Mapping):
            pairs = [(str(k), str(v)) for k, v in payload.items()]
        elif isinstance(payload, list):
            pairs = []
            for item in payload:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        "용어집 배열 항목은 {source, target} 객체여야 합니다."
                    )
                source = item.get("source", item.get("src"))
                target = item.get("target", item.get("tgt", item.get("dest")))
                if source is None or target is None:
                    raise ValueError(
                        "용어집 배열 항목에 source/target이 필요합니다."
                    )
                pairs.append((str(source), str(target)))
        else:
            raise ValueError("용어집 JSON은 객체 또는 배열이어야 합니다.")

        glossary = cls.from_pairs(pairs, require_non_empty=False)
        return glossary or None

    @staticmethod
    def _is_header_row(source: str, target: str) -> bool:
        return (
            source.casefold() in _HEADER_SOURCE_ALIASES
            and target.casefold() in _HEADER_TARGET_ALIASES
        )

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
    def _is_word_char(ch: str) -> bool:
        """True when a character is a regex word character (\\w)."""
        return ch.isalnum() or ch == "_"

    @staticmethod
    def _term_pattern(source: str) -> str:
        """Build a match pattern for one glossary source term.

        A ``\\b`` boundary is only added on an edge whose adjacent character is a
        word character (ASCII letter/digit/underscore). This lets multi-word
        Latin phrases ("smart director") match while skipping boundaries for CJK
        terms, where Python's ``\\b`` never fires and would leave the term
        unmatched. Boundaries are anchored to the *term's* edge so a shorter term
        cannot match inside a longer one that was already consumed by the single
        pass.
        """
        escaped = re.escape(source)
        prefix = r"\b" if source and GlossaryLoader._is_word_char(source[0]) else ""
        suffix = r"\b" if source and GlossaryLoader._is_word_char(source[-1]) else ""
        return f"{prefix}{escaped}{suffix}"

    @staticmethod
    def _compile_glossary(glossary: Dict[str, str]) -> "tuple[re.Pattern[str], Dict[str, str]] | None":
        """Compile a glossary into a single alternation regex + lookup map.

        Terms are ordered longest-first so an overlapping longer term ("게임팀")
        wins over a shorter one ("게임") in the single pass, and so a shorter term
        never pollutes a longer term's substring ("공지" inside "공지사항").
        Returns ``None`` when no usable term exists.
        """
        sources = [s for s in glossary if s]
        if not sources:
            return None
        sources.sort(key=len, reverse=True)
        combined = "|".join(GlossaryLoader._term_pattern(s) for s in sources)
        try:
            pattern = re.compile(combined)
        except re.error:  # pragma: no cover - defensive; escaped input is safe
            LOGGER.exception("Failed to compile glossary pattern; skipping replacement.")
            return None
        return pattern, dict(glossary)

    @staticmethod
    def _apply_compiled(text: str, compiled: "tuple[re.Pattern[str], Dict[str, str]]") -> str:
        """Apply a compiled glossary pattern to one string in a single pass.

        The replacement is delivered via a callback so backslashes in target
        text are never interpreted as regex escapes (which would raise re.error).
        """
        pattern, lookup = compiled
        # ``m.group(0)`` is the matched source; map it back to its target. A
        # ``\b``-anchored match returns the exact source text, so the lookup hits.
        return pattern.sub(lambda m: lookup.get(m.group(0), m.group(0)), text)

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

        compiled = GlossaryLoader._compile_glossary(glossary)
        if compiled is None:
            return list(texts)

        return [GlossaryLoader._apply_compiled(text, compiled) for text in texts]

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
        compiled = GlossaryLoader._compile_glossary(glossary)
        if compiled is None:
            return text
        return GlossaryLoader._apply_compiled(text, compiled)
