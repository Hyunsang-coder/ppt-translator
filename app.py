"""Streamlit entry point for the PPT translation prototype."""

from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
from src.core.image_optimizer import ImageOptimizer
from src.core.ppt_parser import PPTParser
from src.core.ppt_writer import PPTWriter
from src.ui.file_handler import get_cached_upload, handle_upload
from src.ui.progress_tracker import ProgressTracker
from src.ui.settings_panel import render_settings
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import chunk_paragraphs
from src.utils.language_detector import LanguageDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="PPT Translator", page_icon="📊", layout="wide")


def _load_glossary(glossary_file) -> Tuple[dict[str, str] | None, str]:
    """Load glossary data from the uploaded file.

    Args:
        glossary_file: Streamlit uploaded file containing glossary definitions.

    Returns:
        Tuple of glossary mapping (or ``None``) and formatted string for prompts.
    """

    if glossary_file is None:
        return None, "None"

    glossary_loader = GlossaryLoader()
    glossary_bytes = io.BytesIO(glossary_file.getvalue())

    try:
        glossary = glossary_loader.load_glossary(glossary_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return None, "None"

    st.success(f"📚 용어집 로드 완료: {len(glossary)}건")
    return glossary, GlossaryLoader.format_glossary_terms(glossary)


def _determine_batch_size(total_paragraphs: int, settings) -> int:
    """Calculate a user-friendly batch size for translation."""

    if total_paragraphs <= 0:
        return 1

    max_batch_size = max(1, settings.batch_size)
    min_batch_size = max(1, getattr(settings, "min_batch_size", 40))
    target_batches = max(1, getattr(settings, "target_batch_count", 5))

    suggested = math.ceil(total_paragraphs / target_batches)
    batch_size = min(max_batch_size, max(min_batch_size, suggested))
    return max(1, min(total_paragraphs, batch_size))


def _sanitize_for_filename(value: str, fallback: str) -> str:
    """Remove characters that are risky inside file names while keeping unicode."""

    sanitized = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
    return sanitized or fallback


def main() -> None:
    """Render the Streamlit UI and orchestrate the translation workflow."""

    settings = get_settings()
    if not settings.openai_api_key:
        st.warning("OPENAI_API_KEY가 설정되지 않았습니다. 번역 실행 시 에러가 발생할 수 있습니다.")

    st.title("📊 PowerPoint 번역기")
    st.markdown("LangChain + OpenAI GPT-5를 활용한 고품질 PPT 번역")

    uploaded_file = st.file_uploader("PPT 파일 업로드", type=["ppt", "pptx"])

    ppt_buffer = None
    if uploaded_file:
        ppt_buffer = handle_upload(uploaded_file, max_size_mb=settings.max_upload_size_mb)

    if ppt_buffer:
        settings_state = render_settings()

        if st.button("🚀 번역 시작", type="primary"):
            with st.spinner("번역 진행 중..."):
                cached_buffer = get_cached_upload()
                if cached_buffer is None:
                    st.error("업로드된 파일을 찾을 수 없습니다. 다시 업로드해주세요.")
                    return

                parser = PPTParser()
                paragraphs, presentation = parser.extract_paragraphs(cached_buffer)

                if not paragraphs:
                    st.warning("번역할 텍스트를 찾을 수 없습니다.")
                    return

                if len(presentation.slides) > 100:
                    st.warning("슬라이드가 100장을 초과합니다. 처리 시간이 길어질 수 있습니다.")

                context_manager = ContextManager(paragraphs)
                ppt_context = context_manager.build_global_context()

                glossary, glossary_terms = _load_glossary(settings_state.get("glossary_file"))
                prepared_texts: List[str] = [info.original_text for info in paragraphs]
                if glossary:
                    prepared_texts = GlossaryLoader.apply_glossary_to_texts(prepared_texts, glossary)

                detector = LanguageDetector()
                sample_text = "\n".join(paragraph.original_text for paragraph in paragraphs[:50])

                source_language = settings_state.get("source_lang")
                target_language = settings_state.get("target_lang")

                if source_language == "Auto":
                    source_language = detector.detect_language(sample_text)
                    st.info(f"🔍 소스 언어 감지: {source_language}")

                if target_language == "Auto":
                    target_language = detector.infer_target_language(source_language)
                    st.info(f"🔍 타겟 언어 추론: {target_language}")

                batch_size = _determine_batch_size(len(paragraphs), settings)
                st.caption(f"배치 크기: {batch_size} 문장 (총 {len(paragraphs)} 문장)")

                batches = chunk_paragraphs(
                    paragraphs,
                    batch_size=batch_size,
                    ppt_context=ppt_context,
                    glossary_terms=glossary_terms,
                    prepared_texts=prepared_texts,
                )

                if not batches:
                    st.warning("번역할 배치를 생성하지 못했습니다.")
                    return

                progress_tracker = ProgressTracker(
                    total_batches=len(batches), total_sentences=len(paragraphs)
                )

                chain = create_translation_chain(
                    model_name=settings_state.get("model", "gpt-5"),
                    source_lang=source_language,
                    target_lang=target_language,
                    user_prompt=settings_state.get("user_prompt"),
                )

                try:
                    translated_texts = translate_with_progress(chain, batches, progress_tracker)
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.exception("Translation failed: %s", exc)
                    st.error("번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                    return

                if glossary:
                    translated_texts = [
                        GlossaryLoader.apply_glossary_to_translation(text, glossary)
                        for text in translated_texts
                    ]

                image_optimizer = ImageOptimizer()
                presentation = image_optimizer.optimise(presentation)

                writer = PPTWriter()
                output_buffer = writer.apply_translations(paragraphs, translated_texts, presentation)

                total_elapsed = progress_tracker.get_total_elapsed()
                minutes, seconds = divmod(total_elapsed, 60)
                LOGGER.info("Translation completed in %d분 %.1f초", int(minutes), seconds)
                st.success(f"✅ 번역 완료! 총 소요 시간: {int(minutes)}분 {seconds:.1f}초")

                original_name = st.session_state.get("uploaded_ppt_name", "presentation")
                original_stem = Path(original_name).stem or "presentation"
                original_stem = _sanitize_for_filename(original_stem, "presentation")
                clean_model = _sanitize_for_filename(settings_state.get("model", "model"), "model")
                timestamp = datetime.now().strftime("%Y%m%d")
                safe_target_lang = _sanitize_for_filename(target_language, "target")
                download_name = f"{safe_target_lang}_{original_stem}_{clean_model}_{timestamp}.pptx"

                st.download_button(
                    label="📥 번역된 PPT 다운로드",
                    data=output_buffer.getvalue(),
                    file_name=download_name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )


if __name__ == "__main__":
    main()
