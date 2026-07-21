"""Glossary loading and formatting utilities."""

from __future__ import annotations

import io
import json
import logging
import re
import unicodedata
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

        entries = self.load_glossary_entries(file_data)
        glossary = {entry["source"]: entry["target"] for entry in entries}
        LOGGER.info("Loaded %d glossary entries.", len(glossary))
        return glossary

    def load_glossary_entries(self, file_data: io.BytesIO) -> List[Dict[str, str]]:
        """Load source/target/notes rows for the in-app glossary editor."""

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

        entries: Dict[str, Dict[str, str]] = {}
        started = False
        for _, row in dataframe.iterrows():
            source = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
            target = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else ""
            notes = (
                str(row.iloc[2]).strip()
                if dataframe.shape[1] > 2 and not pd.isna(row.iloc[2])
                else ""
            )
            if not source or not target:
                continue
            if not started and self._is_header_row(source, target):
                started = True
                continue
            started = True
            if len(source) > self.MAX_TERM_CHARS or len(target) > self.MAX_TERM_CHARS:
                raise ValueError(
                    f"용어 길이는 {self.MAX_TERM_CHARS}자를 초과할 수 없습니다."
                )
            if len(notes) > 2000:
                raise ValueError("메모는 2000자를 초과할 수 없습니다.")
            key = unicodedata.normalize("NFKC", source).casefold()
            entries[key] = {"source": source, "target": target, "notes": notes}
            if len(entries) > self.MAX_ENTRIES:
                raise ValueError(
                    f"용어집 항목은 최대 {self.MAX_ENTRIES}개까지 지원합니다."
                )

        if not entries:
            raise ValueError("유효한 용어집 항목을 찾을 수 없습니다.")
        return list(entries.values())

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
        *,
        require_non_empty: bool = False,
        skip_header: bool = False,
    ) -> Dict[str, str]:
        """Build a glossary dict from (source, target) pairs.

        When ``skip_header`` is True (Excel/CSV file imports only), a leading
        pair that looks like column labels is ignored. JSON/PATCH paths must
        pass ``skip_header=False`` so legitimate terms like source→target are kept.
        Duplicate sources upsert (last value wins). Enforces entry/term limits.
        """
        glossary: Dict[str, str] = {}
        source_keys: Dict[str, str] = {}
        # When not skipping headers, treat the stream as already past the header.
        started = not skip_header
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
            normalized_key = unicodedata.normalize("NFKC", source).casefold()
            previous_source = source_keys.get(normalized_key)
            if previous_source is not None and previous_source != source:
                glossary.pop(previous_source, None)
            source_keys[normalized_key] = source
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

        glossary = cls.from_pairs(pairs, require_non_empty=False, skip_header=False)
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
    def filter_matching_terms(
        glossary: Dict[str, str] | None,
        texts: Iterable[str],
    ) -> Dict[str, str]:
        """Return only glossary entries whose source term appears in ``texts``.

        Uses the same boundary rules as ``apply_glossary_to_*`` so prompt
        injection stays aligned with PRE/POST replacement. Match against
        *original* source text (before PRE substitution), otherwise already
        replaced targets would hide the source terms.
        """
        if not glossary:
            return {}

        compiled = GlossaryLoader._compile_glossary(glossary)
        if compiled is None:
            return {}
        return GlossaryLoader._filter_matching_terms_compiled(compiled, texts)

    @staticmethod
    def _filter_matching_terms_compiled(
        compiled: "tuple[re.Pattern[str], Dict[str, str]]",
        texts: Iterable[str],
    ) -> Dict[str, str]:
        """Return matching terms while reusing an already compiled glossary."""
        pattern, lookup = compiled

        matched: Dict[str, str] = {}
        for text in texts:
            if not text:
                continue
            for match in pattern.finditer(text):
                source = match.group(0)
                target = lookup.get(source.casefold())
                if target is not None:
                    matched[source] = target
            if len(matched) == len(lookup):
                break
        return matched

    @staticmethod
    def format_matching_terms(
        glossary: Dict[str, str] | None,
        texts: Iterable[str],
    ) -> str:
        """Format the subset of glossary terms found in ``texts`` for prompts."""
        return GlossaryLoader.format_glossary_terms(
            GlossaryLoader.filter_matching_terms(glossary, texts)
        )

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
            pattern = re.compile(combined, re.IGNORECASE)
        except re.error:  # pragma: no cover - defensive; escaped input is safe
            LOGGER.exception("Failed to compile glossary pattern; skipping replacement.")
            return None
        return pattern, {source.casefold(): target for source, target in glossary.items()}

    @staticmethod
    def _apply_compiled(text: str, compiled: "tuple[re.Pattern[str], Dict[str, str]]") -> str:
        """Apply a compiled glossary pattern to one string in a single pass.

        The replacement is delivered via a callback so backslashes in target
        text are never interpreted as regex escapes (which would raise re.error).
        """
        pattern, lookup = compiled
        # ``m.group(0)`` is the matched source; map it back to its target. A
        # ``\b``-anchored match returns the exact source text, so the lookup hits.
        return pattern.sub(lambda m: lookup.get(m.group(0).casefold(), m.group(0)), text)

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

    @staticmethod
    def apply_matching_glossary_to_translations(
        source_texts: Iterable[str],
        translated_texts: Iterable[str],
        glossary: Dict[str, str] | None,
    ) -> List[str]:
        """Post-process each translation using only terms found in its source.

        Keeping the source-matched subset prevents opposite-direction entries
        (for example ``공지→Notice`` and ``Notice→공지``) from undoing one
        another in translated output. The full glossary is compiled once, then
        each usually-small matched subset is applied to its aligned target.
        """
        sources = list(source_texts)
        translations = list(translated_texts)
        if len(sources) != len(translations):
            raise ValueError("원문과 번역문 개수가 일치하지 않습니다.")
        if not glossary:
            return translations

        compiled = GlossaryLoader._compile_glossary(glossary)
        if compiled is None:
            return translations

        results: List[str] = []
        for source, translation in zip(sources, translations, strict=True):
            matching = GlossaryLoader._filter_matching_terms_compiled(
                compiled, [source]
            )
            matching_compiled = GlossaryLoader._compile_glossary(matching)
            results.append(
                GlossaryLoader._apply_compiled(translation, matching_compiled)
                if matching_compiled is not None
                else translation
            )
        return results
