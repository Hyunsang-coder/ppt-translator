"""Team translation-rules loading and prompt formatting.

Consumes the translation team's shared ``translation-rules.json`` (source of
truth lives in the trans_agent repo). This module only *parses and formats* the
rules for prompt injection and consistency checks — it never edits them. See
``consulting/ppt-translator-quality-design.md`` §3.6 for the consumption
contract.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

# Target-language display name -> rules bucket. Display names come from
# LanguageDetector (Korean labels), matching the rest of the pipeline.
_BUCKET_BY_TARGET: Dict[str, str] = {
    "한국어": "kr_target_rules",
    "영어": "en_target_rules",
}

_TOP_LEVEL_BUCKETS = ("kr_target_rules", "en_target_rules", "bidirectional")


class RulesLoader:
    """Load and format team translation rules for prompt injection."""

    MAX_FILE_SIZE_MB = 10  # Mirror GlossaryLoader's guard.

    def load_rules(self, file_data: io.BytesIO) -> dict:
        """Load and validate a ``translation-rules.json`` upload.

        Args:
            file_data: Bytes of the uploaded JSON file.

        Returns:
            The parsed rules document.

        Raises:
            ValueError: If the file is too large, unparseable, or missing the
                expected top-level buckets.
        """

        file_data.seek(0, io.SEEK_END)
        file_size = file_data.tell()
        file_data.seek(0)

        size_mb = file_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"규칙집 파일 크기가 {self.MAX_FILE_SIZE_MB}MB를 초과합니다. "
                "더 작은 파일로 다시 시도해주세요."
            )

        try:
            file_data.seek(0)
            rules = json.load(file_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOGGER.error("Failed to parse translation-rules JSON: %s", exc)
            raise ValueError(
                "규칙집 파일을 읽을 수 없습니다. JSON 형식을 확인해주세요."
            ) from exc

        if not isinstance(rules, dict) or not any(
            key in rules for key in _TOP_LEVEL_BUCKETS
        ):
            raise ValueError(
                "규칙집 파일 구조가 올바르지 않습니다. "
                "kr_target_rules / en_target_rules / bidirectional 중 하나 이상이 필요합니다."
            )

        counts = {b: len(rules.get(b, [])) for b in _TOP_LEVEL_BUCKETS}
        LOGGER.info("Loaded translation rules: %s", counts)
        return rules

    @staticmethod
    def slice_for_direction(rules: dict | None, target_lang: str) -> List[dict]:
        """Return the target-direction bucket plus bidirectional rules.

        Args:
            rules: Parsed rules document (or ``None``).
            target_lang: Target-language display name (e.g. ``"한국어"``).

        Returns:
            Combined list of rule entries applicable to this direction. Empty if
            no rules or the direction is unrecognized.
        """

        if not rules:
            return []
        bucket = _BUCKET_BY_TARGET.get(target_lang)
        combined: List[dict] = []
        if bucket:
            combined.extend(rules.get(bucket, []) or [])
        combined.extend(rules.get("bidirectional", []) or [])
        return combined

    @staticmethod
    def format_team_rules(rules: dict | None, target_lang: str) -> str:
        """Format the injection slice into a prompt-friendly string.

        Only ``summary``/``avoid``/``prefer``/``locked_term`` are injected; red
        rules also include one example pair. ``why``/``memory_ref``/promotion
        metadata are never injected (contract §3.6).

        Returns ``"None"`` when there is nothing to inject (same convention as
        ``GlossaryLoader.format_glossary_terms``).
        """

        entries = RulesLoader.slice_for_direction(rules, target_lang)
        if not entries:
            return "None"

        lines: List[str] = []
        for rule in entries:
            summary = str(rule.get("summary", "")).strip()
            if not summary:
                continue
            lines.append(f"- {summary}")

            avoid = [str(a).strip() for a in rule.get("avoid", []) if str(a).strip()]
            prefer = [str(p).strip() for p in rule.get("prefer", []) if str(p).strip()]
            if avoid or prefer:
                avoid_str = ", ".join(avoid) if avoid else "—"
                prefer_str = ", ".join(prefer) if prefer else "—"
                lines.append(f"  avoid: {avoid_str} → use: {prefer_str}")

            locked = rule.get("locked_term")
            if locked:
                lines.append(f"  [LOCKED TERM, use exactly: {locked}]")

            if rule.get("severity") == "red":
                examples = rule.get("examples") or []
                if examples and isinstance(examples[0], dict):
                    bad = str(examples[0].get("bad", "")).strip()
                    good = str(examples[0].get("good", "")).strip()
                    if bad and good:
                        lines.append(f'  e.g. "{bad}" → "{good}"')

        return "\n".join(lines) if lines else "None"

    @staticmethod
    def locked_terms(rules: dict | None, target_lang: str) -> Dict[str, str]:
        """Extract locked terms for the consistency sweep to enforce.

        Red rules carrying a ``locked_term`` are treated like glossary entries
        by WP-C3. The key is the term's own summary id-ish hint and the value is
        the exact target term that must appear.

        Returns a ``{hint: locked_term}`` mapping (``hint`` is the rule id, used
        only for reporting which rule fired).
        """

        entries = RulesLoader.slice_for_direction(rules, target_lang)
        locked: Dict[str, str] = {}
        for rule in entries:
            term = rule.get("locked_term")
            if term and rule.get("severity") == "red":
                locked[str(rule.get("id", term))] = str(term)
        return locked
