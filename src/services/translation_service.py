"""Translation service layer for PPT translation."""

from __future__ import annotations

import io
import logging
import math
import time
from typing import Dict, List, Optional

from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
from src.core.ppt_parser import PPTParser
from src.core.ppt_writer import PPTWriter
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

        # Phase 10: Write output
        self._notify_progress(
            TranslationProgress(
                status=TranslationStatus.APPLYING_TRANSLATIONS,
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
