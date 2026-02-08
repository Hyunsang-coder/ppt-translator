"""Translation service layer for PPT translation."""

from __future__ import annotations

import io
import logging
import math
import time
from typing import Dict, List, Optional

from src.chains.color_distribution_chain import ColoredSegment, distribute_colors
from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
from src.core.ppt_parser import PPTParser
from src.core.ppt_writer import PPTWriter, _group_runs_by_format
from src.services.models import (
    ProgressCallback,
    TranslationProgress,
    TranslationRequest,
    TranslationResult,
    TranslationStatus,
)
from src.utils.config import Settings, get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import chunk_paragraphs
from src.utils.language_detector import LanguageDetector
from src.utils.repetition import build_repetition_plan, expand_translations

LOGGER = logging.getLogger(__name__)

# Characters treated as equivalent whitespace during validation
_WHITESPACE_CHARS = {"\u00a0", "\u2002", "\u2003", "\u2009", "\u200a", "\u3000"}


def _normalize_whitespace(text: str) -> str:
    """Normalize various whitespace characters to regular spaces."""
    for ch in _WHITESPACE_CHARS:
        text = text.replace(ch, " ")
    return text


def _validate_distribution(
    segments: list[str],
    translation: str,
    num_groups: int,
    para_idx: int,
) -> list[str] | None:
    """Validate and optionally correct a color distribution for a paragraph.

    Args:
        segments: List of text segments from the LLM.
        translation: The expected full translated text.
        num_groups: Number of format groups expected.
        para_idx: Paragraph index (for logging).

    Returns:
        Corrected segment list if valid, or None if validation fails.
    """
    # Check segment count
    if len(segments) != num_groups:
        LOGGER.warning(
            "Color distribution mismatch for paragraph %d: expected %d groups, got %d. Skipping.",
            para_idx, num_groups, len(segments),
        )
        return None

    # Normalize whitespace in segments
    normalized_segments = [_normalize_whitespace(s) for s in segments]
    normalized_translation = _normalize_whitespace(translation)

    joined = "".join(normalized_segments)

    # Exact match after normalization
    if joined == normalized_translation:
        return normalized_segments

    # Try stripped comparison and correct leading/trailing whitespace
    if joined.strip() == normalized_translation.strip():
        corrected = list(normalized_segments)

        # Fix leading whitespace
        excess_leading = (
            (len(joined) - len(joined.lstrip()))
            - (len(normalized_translation) - len(normalized_translation.lstrip()))
        )
        if excess_leading > 0:
            for i in range(len(corrected)):
                if corrected[i].strip():
                    corrected[i] = corrected[i][excess_leading:]
                    break
        elif excess_leading < 0:
            corrected[0] = (" " * (-excess_leading)) + corrected[0]

        # Fix trailing whitespace
        excess_trailing = (
            (len(joined) - len(joined.rstrip()))
            - (len(normalized_translation) - len(normalized_translation.rstrip()))
        )
        if excess_trailing > 0:
            for i in range(len(corrected) - 1, -1, -1):
                if corrected[i].strip() or i == 0:
                    corrected[i] = corrected[i][:-excess_trailing] if excess_trailing <= len(corrected[i]) else ""
                    break
        elif excess_trailing < 0:
            corrected[-1] = corrected[-1] + (" " * (-excess_trailing))

        if "".join(corrected) == normalized_translation:
            return corrected

    LOGGER.warning(
        "Color distribution concatenation mismatch for paragraph %d: "
        "expected %r, got %r. Skipping.",
        para_idx, normalized_translation, joined,
    )
    return None


def _validate_colored_segments(
    segments: list[ColoredSegment],
    translation: str,
    num_groups: int,
    para_idx: int,
) -> list[ColoredSegment] | None:
    """Validate a list of ColoredSegment objects for a paragraph.

    Args:
        segments: List of ColoredSegment from the LLM.
        translation: The expected full translated text.
        num_groups: Number of format groups in the original paragraph.
        para_idx: Paragraph index (for logging).

    Returns:
        Validated segment list, or None if validation fails.
    """
    if not segments:
        LOGGER.warning(
            "Color distribution for paragraph %d returned empty segments. Skipping.",
            para_idx,
        )
        return None

    # Validate group_index range
    for seg in segments:
        if seg.group_index < 0 or seg.group_index >= num_groups:
            LOGGER.warning(
                "Color distribution for paragraph %d has out-of-range group_index %d "
                "(expected 0..%d). Skipping.",
                para_idx, seg.group_index, num_groups - 1,
            )
            return None

    # Validate concatenation
    normalized_translation = _normalize_whitespace(translation)
    joined = _normalize_whitespace("".join(seg.text for seg in segments))

    if joined == normalized_translation:
        return segments

    # Try whitespace-corrected match
    if joined.strip() == normalized_translation.strip():
        corrected_texts = [_normalize_whitespace(seg.text) for seg in segments]

        excess_leading = (
            (len(joined) - len(joined.lstrip()))
            - (len(normalized_translation) - len(normalized_translation.lstrip()))
        )
        if excess_leading > 0:
            for i in range(len(corrected_texts)):
                if corrected_texts[i].strip():
                    corrected_texts[i] = corrected_texts[i][excess_leading:]
                    break
        elif excess_leading < 0:
            corrected_texts[0] = (" " * (-excess_leading)) + corrected_texts[0]

        excess_trailing = (
            (len(joined) - len(joined.rstrip()))
            - (len(normalized_translation) - len(normalized_translation.rstrip()))
        )
        if excess_trailing > 0:
            for i in range(len(corrected_texts) - 1, -1, -1):
                if corrected_texts[i].strip() or i == 0:
                    corrected_texts[i] = corrected_texts[i][:-excess_trailing] if excess_trailing <= len(corrected_texts[i]) else ""
                    break
        elif excess_trailing < 0:
            corrected_texts[-1] = corrected_texts[-1] + (" " * (-excess_trailing))

        if "".join(corrected_texts) == normalized_translation:
            return [
                ColoredSegment(text=t, group_index=seg.group_index)
                for t, seg in zip(corrected_texts, segments)
            ]

    LOGGER.warning(
        "Color distribution concatenation mismatch for paragraph %d: "
        "expected %r, got %r. Skipping.",
        para_idx, normalized_translation, joined,
    )
    return None


class ServiceProgressTracker:
    """Progress tracker adapter for the service layer.

    This class implements the same interface as ProgressTracker from src.ui.progress_tracker
    to be compatible with translate_with_progress function.
    """

    def __init__(
        self,
        total_batches: int,
        total_sentences: int,
        callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.total_batches = total_batches
        self.total_sentences = total_sentences
        self._callback = callback
        self._completed_batches = 0
        self._current_sentence = 0
        self._start_time = time.time()
        self.total_elapsed: float = 0.0

    def _calc_percent(self) -> int:
        """Calculate overall percent (translation phase spans 10-80%)."""
        if self.total_batches <= 0:
            return 10
        ratio = self._completed_batches / self.total_batches
        return 10 + int(ratio * 70)

    def reset(self, total_batches: int, total_sentences: int) -> None:
        """Reset tracker state for a new translation run."""
        self.total_batches = total_batches
        self.total_sentences = total_sentences
        self._completed_batches = 0
        self._current_sentence = 0
        self._start_time = time.time()

        if self._callback:
            self._callback(
                TranslationProgress(
                    status=TranslationStatus.TRANSLATING,
                    current_batch=0,
                    total_batches=total_batches,
                    current_sentence=0,
                    total_sentences=total_sentences,
                    percent=10,
                    message=f"번역 대기 중... (0/{total_batches} 배치)",
                )
            )

    def batch_completed(
        self, batch_start_idx: Optional[int] = None, batch_end_idx: Optional[int] = None
    ) -> None:
        """Record completion of a batch and notify callback."""
        self._completed_batches = min(self.total_batches, self._completed_batches + 1)

        # Estimate sentences completed based on batch indices
        if batch_end_idx is not None:
            self._current_sentence = batch_end_idx

        if self._callback:
            if self._completed_batches < self.total_batches:
                if batch_start_idx is not None and batch_end_idx is not None:
                    message = f"번역 중... 완료 {self._completed_batches}/{self.total_batches} ({batch_start_idx}~{batch_end_idx})"
                else:
                    message = f"번역 중... 완료 {self._completed_batches}/{self.total_batches}"
            else:
                message = "번역이 완료되었습니다. PPT 반영 중..."

            self._callback(
                TranslationProgress(
                    status=TranslationStatus.TRANSLATING,
                    current_batch=self._completed_batches,
                    total_batches=self.total_batches,
                    current_sentence=self._current_sentence,
                    total_sentences=self.total_sentences,
                    percent=self._calc_percent(),
                    message=message,
                )
            )

    def finish(self) -> float:
        """Mark translation as finished and return elapsed time."""
        self.total_elapsed = time.time() - self._start_time
        return self.total_elapsed

    def get_total_elapsed(self) -> float:
        """Return the measured total elapsed time for the translation run."""
        return self.total_elapsed


class TranslationService:
    """Service for translating PowerPoint presentations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the translation service.

        Args:
            settings: Configuration settings. Uses default if not provided.
            progress_callback: Optional callback for progress updates.
        """
        self._settings = settings or get_settings()
        self._progress_callback = progress_callback

    def _notify_progress(self, progress: TranslationProgress) -> None:
        """Send progress update to callback if available."""
        if self._progress_callback:
            self._progress_callback(progress)

    @staticmethod
    def _approximate_tokens(text: str) -> int:
        """Rudimentary character-based token estimate for heuristics."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _estimate_tokens_for_batch(batch: Dict[str, object]) -> int:
        """Estimate total prompt tokens for a single translation batch."""
        texts = str(batch.get("texts", ""))
        ppt_context = str(batch.get("ppt_context", ""))
        glossary_terms = str(batch.get("glossary_terms", ""))

        token_estimate = (
            TranslationService._approximate_tokens(texts)
            + TranslationService._approximate_tokens(ppt_context)
            + TranslationService._approximate_tokens(glossary_terms)
            + 200  # instructions + response padding
        )

        return max(1, token_estimate)

    def _determine_batch_size(self, total_paragraphs: int) -> int:
        """Calculate a batch size that balances latency and throughput."""
        settings = self._settings

        if total_paragraphs <= 0:
            return 1

        min_size = max(1, int(getattr(settings, "min_batch_size", 40)))
        max_size = max(
            min_size,
            int(getattr(settings, "max_batch_size", getattr(settings, "batch_size", min_size))),
        )
        default_size = max(
            min_size, min(max_size, int(getattr(settings, "batch_size", max_size)))
        )

        concurrency = max(1, int(getattr(settings, "max_concurrency", 1)))
        wave_multiplier = float(getattr(settings, "wave_multiplier", 1.2) or 1.2)
        wave_multiplier = max(1.0, wave_multiplier)

        target_batches = max(
            concurrency, int(math.ceil(concurrency * wave_multiplier * 2))
        )
        suggested_size = (
            math.ceil(total_paragraphs / target_batches)
            if target_batches > 0
            else default_size
        )

        batch_size = max(min_size, min(max_size, suggested_size))
        if batch_size < default_size:
            batch_size = max(batch_size, min(default_size, max_size))

        actual_batches = max(1, math.ceil(total_paragraphs / batch_size))
        if actual_batches > 1:
            remainder = total_paragraphs - (actual_batches - 1) * batch_size
            if 0 < remainder < max(1, int(min_size * 0.5)):
                adjusted = math.ceil(total_paragraphs / (actual_batches - 1))
                batch_size = max(min_size, min(max_size, adjusted))

        return max(1, min(total_paragraphs, batch_size))

    @staticmethod
    def _try_rule_based_distribution(
        group_texts: list[str],
        translation: str,
    ) -> list[ColoredSegment] | None:
        """Try to distribute translated text using deterministic string matching.

        Only handles the simple case where a numeric/symbol token from one group
        appears verbatim at the START or END of the translated text, so the split
        point is unambiguous.

        Returns:
            List of ColoredSegment if rule-based split succeeded, else None.
        """
        if len(group_texts) != 2:
            return None

        for anchor_idx in range(2):
            anchor = group_texts[anchor_idx].strip()
            if not anchor or len(anchor) < 2:
                continue
            if not any(ch.isdigit() or ch in "$%€£¥#@&+" for ch in anchor):
                continue

            other_idx = 1 - anchor_idx

            # Case 1: anchor at the end of translation
            if translation.endswith(anchor):
                before = translation[: len(translation) - len(anchor)]
                segments = [
                    ColoredSegment(text=before, group_index=other_idx),
                    ColoredSegment(text=anchor, group_index=anchor_idx),
                ]
                if "".join(s.text for s in segments) == translation:
                    return segments

            # Case 2: anchor at the start of translation
            if translation.startswith(anchor):
                after = translation[len(anchor):]
                segments = [
                    ColoredSegment(text=anchor, group_index=anchor_idx),
                    ColoredSegment(text=after, group_index=other_idx),
                ]
                if "".join(s.text for s in segments) == translation:
                    return segments

        return None

    @staticmethod
    def _fix_color_distributions(
        paragraphs,
        translated_texts: list[str],
        provider: str,
        model_name: str | None = None,
    ) -> dict[int, list[ColoredSegment]] | None:
        """Detect multi-color paragraphs and distribute translated text by meaning.

        Uses a three-tier approach:
        1. Rule-based matching for simple cases (no LLM needed)
        2. LLM-based distribution in batches
        3. One retry for paragraphs that failed validation

        Args:
            paragraphs: List of ParagraphInfo objects.
            translated_texts: Translated text for each paragraph.
            provider: LLM provider ("openai" or "anthropic").
            model_name: Model to use. Uses the same model as main translation.

        Returns:
            Mapping from paragraph index to list of ColoredSegment, or None if
            no multi-color paragraphs were found.
        """
        candidates: list[tuple[int, list[str], str]] = []  # (idx, group_texts, translation)

        for idx, (para, translation) in enumerate(zip(paragraphs, translated_texts)):
            if para.is_note:
                continue
            runs = list(para.paragraph.runs)
            if len(runs) <= 1:
                continue
            groups = _group_runs_by_format(runs)
            if len(groups) <= 1:
                continue
            group_texts = [
                "".join(run.text for run in group) for group in groups
            ]
            candidates.append((idx, group_texts, translation))

        if not candidates:
            LOGGER.info("No multi-color paragraphs found; skipping color distribution.")
            return None

        LOGGER.info("Found %d multi-color paragraphs for color distribution.", len(candidates))

        result: dict[int, list[ColoredSegment]] = {}

        # --- Tier 1: Rule-based distribution for simple cases ---
        llm_candidates: list[tuple[int, list[str], str]] = []
        rule_count = 0
        for para_idx, group_texts, translation in candidates:
            rule_result = TranslationService._try_rule_based_distribution(
                group_texts, translation,
            )
            if rule_result is not None:
                result[para_idx] = rule_result
                rule_count += 1
            else:
                llm_candidates.append((para_idx, group_texts, translation))

        if rule_count:
            LOGGER.info("Rule-based color distribution resolved %d/%d paragraphs.", rule_count, len(candidates))

        if not llm_candidates:
            return result if result else None

        # --- Tier 2: LLM-based distribution ---
        original_groups = [c[1] for c in llm_candidates]
        translations_for_dist = [c[2] for c in llm_candidates]

        distributions = distribute_colors(
            original_groups, translations_for_dist,
            provider=provider, model_name=model_name,
        )

        if distributions is None:
            LOGGER.warning("Color distribution chain returned None; using fallback for all.")
        else:
            for (para_idx, group_texts, translation), dist in zip(llm_candidates, distributions):
                if dist is None:
                    continue
                validated = _validate_colored_segments(
                    segments=dist,
                    translation=translation,
                    num_groups=len(group_texts),
                    para_idx=para_idx,
                )
                if validated is not None:
                    result[para_idx] = validated

        LOGGER.info(
            "Color distribution validated %d/%d paragraphs.",
            len(result), len(candidates),
        )
        return result if result else None

    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate a PowerPoint presentation.

        Args:
            request: Translation request containing the PPT file and options.

        Returns:
            TranslationResult with the translated file or error information.
        """
        start_time = time.time()

        try:
            return self._execute_translation(request, start_time)
        except Exception as exc:
            LOGGER.exception("Translation failed: %s", exc)
            return TranslationResult(
                success=False,
                error_message=f"번역 중 오류가 발생했습니다: {str(exc)}",
                elapsed_seconds=time.time() - start_time,
            )

    def _execute_translation(
        self, request: TranslationRequest, start_time: float
    ) -> TranslationResult:
        """Execute the translation workflow."""
        # Phase 1: Parse PPT
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.PARSING,
                percent=2,
                message="PPT 파일 분석 중...",
            )
        )

        parser = PPTParser()
        request.ppt_file.seek(0)
        paragraphs, presentation = parser.extract_paragraphs(
            request.ppt_file, translate_notes=request.translate_notes
        )

        if not paragraphs:
            return TranslationResult(
                success=False,
                error_message="번역할 텍스트를 찾을 수 없습니다.",
                elapsed_seconds=time.time() - start_time,
            )

        total_paragraphs = len(paragraphs)

        # Phase 2: Build context
        context_manager = ContextManager(paragraphs)
        ppt_context = context_manager.build_global_context()

        # Phase 3: Apply glossary
        glossary = request.glossary
        glossary_terms = "None"
        prepared_texts: List[str] = [info.original_text for info in paragraphs]

        if glossary:
            prepared_texts = GlossaryLoader.apply_glossary_to_texts(
                prepared_texts, glossary
            )
            glossary_terms = GlossaryLoader.format_glossary_terms(glossary)

        # Phase 4: Handle repetitions
        repetition_plan = None
        target_paragraphs = paragraphs
        target_prepared_texts = prepared_texts

        if request.preprocess_repetitions:
            repetition_plan = build_repetition_plan(paragraphs)
            target_paragraphs = [
                paragraphs[idx] for idx in repetition_plan.unique_indices
            ]
            target_prepared_texts = [
                prepared_texts[idx] for idx in repetition_plan.unique_indices
            ]

            LOGGER.info(
                "Repetition preprocessing: %d -> %d unique paragraphs",
                total_paragraphs,
                len(target_paragraphs),
            )

        if not target_paragraphs:
            return TranslationResult(
                success=False,
                error_message="반복 문구 사전 처리 결과 번역할 텍스트가 없습니다.",
                elapsed_seconds=time.time() - start_time,
            )

        # Phase 5: Detect languages
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.DETECTING_LANGUAGE,
                percent=5,
                message="언어 감지 중...",
            )
        )

        detector = LanguageDetector()
        sample_text = "\n".join(
            paragraph.original_text
            for paragraph in paragraphs[:50]
            if not paragraph.is_note
        )

        source_language = request.source_lang
        target_language = request.target_lang

        if source_language == "Auto":
            source_language = detector.detect_language(sample_text)
            LOGGER.info("Detected source language: %s", source_language)

        if target_language == "Auto":
            target_language = detector.infer_target_language(source_language)
            LOGGER.info("Inferred target language: %s", target_language)

        # Phase 6: Prepare batches
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.PREPARING_BATCHES,
                percent=8,
                message="번역 배치 준비 중...",
            )
        )

        batch_size = self._determine_batch_size(len(target_paragraphs))

        batches = chunk_paragraphs(
            target_paragraphs,
            batch_size=batch_size,
            ppt_context=ppt_context,
            glossary_terms=glossary_terms,
            prepared_texts=target_prepared_texts,
        )

        LOGGER.info(
            "Prepared %d batches (batch size %d, unique paragraphs %d of %d total).",
            len(batches),
            batch_size,
            len(target_paragraphs),
            total_paragraphs,
        )

        if not batches:
            return TranslationResult(
                success=False,
                error_message="번역할 배치를 생성하지 못했습니다.",
                elapsed_seconds=time.time() - start_time,
            )

        # Calculate safe concurrency
        estimated_tokens = self._estimate_tokens_for_batch(batches[0])
        safe_concurrency = max(
            1,
            min(
                int(self._settings.max_concurrency),
                max(1, self._settings.tpm_limit // max(estimated_tokens, 1)),
            ),
        )

        LOGGER.info(
            "Estimated %d tokens per batch; using concurrency=%d (config max=%d, TPM limit=%d).",
            estimated_tokens,
            safe_concurrency,
            self._settings.max_concurrency,
            self._settings.tpm_limit,
        )

        # Phase 7: Translate
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.TRANSLATING,
                current_batch=0,
                total_batches=len(batches),
                current_sentence=0,
                total_sentences=len(target_paragraphs),
                percent=10,
                message="번역 시작...",
            )
        )

        progress_tracker = ServiceProgressTracker(
            total_batches=len(batches),
            total_sentences=len(target_paragraphs),
            callback=self._progress_callback,
        )

        chain = create_translation_chain(
            model_name=request.model,
            source_lang=source_language,
            target_lang=target_language,
            context=request.context,
            instructions=request.instructions,
            provider=request.provider,
        )

        LOGGER.info(
            "Starting translation with concurrency=%d and model=%s.",
            safe_concurrency,
            request.model,
        )

        translated_unique = translate_with_progress(
            chain,
            batches,
            progress_tracker,
            max_concurrency=safe_concurrency,
        )

        # Phase 8: Expand repetitions
        if repetition_plan is not None:
            translated_texts = expand_translations(
                repetition_plan,
                translated_unique,
                total_paragraphs,
            )
        else:
            translated_texts = translated_unique

        # Phase 9: Apply glossary post-processing
        if glossary:
            translated_texts = [
                GlossaryLoader.apply_glossary_to_translation(text, glossary)
                for text in translated_texts
            ]

        # Phase 9.5: Fix color distributions for multi-color paragraphs
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.FIXING_COLORS,
                percent=80,
                message="다색 문단 서식 분석 중...",
            )
        )
        color_distributions = self._fix_color_distributions(
            paragraphs, translated_texts, request.provider,
            model_name=request.model,
        )
        if color_distributions:
            self._notify_progress(
                TranslationProgress(
                    status=TranslationStatus.FIXING_COLORS,
                    percent=90,
                    message=f"다색 문단 {len(color_distributions)}개 서식 보정 완료",
                )
            )
        else:
            self._notify_progress(
                TranslationProgress(
                    status=TranslationStatus.FIXING_COLORS,
                    percent=90,
                    message="다색 문단 없음 — 서식 보정 생략",
                )
            )

        # Phase 10: Write output
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.APPLYING_TRANSLATIONS,
                percent=95,
                message="번역 결과 적용 중...",
            )
        )

        writer = PPTWriter()
        output_buffer = writer.apply_translations(
            paragraphs,
            translated_texts,
            presentation,
            text_fit_mode=request.text_fit_mode.value,
            min_font_ratio=request.min_font_ratio,
            color_distributions=color_distributions,
        )

        elapsed = progress_tracker.finish()

        # Phase 11: Complete
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.COMPLETED,
                current_batch=len(batches),
                total_batches=len(batches),
                current_sentence=len(target_paragraphs),
                total_sentences=len(target_paragraphs),
                percent=100,
                message="번역 완료",
            )
        )

        LOGGER.info("Translation completed in %.1f seconds", elapsed)

        return TranslationResult(
            success=True,
            output_file=output_buffer,
            source_language_detected=source_language,
            target_language_used=target_language,
            total_paragraphs=total_paragraphs,
            unique_paragraphs=len(target_paragraphs),
            batch_count=len(batches),
            elapsed_seconds=elapsed,
        )
