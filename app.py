"""Streamlit entry point for the PPT translation prototype."""

from __future__ import annotations

import io
import logging
import math
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import streamlit as st

from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
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

MAX_UI_LOG_LINES = 400
LOG_QUEUE_KEY = "ui_log_queue"
LOG_BUFFER_KEY = "ui_log_buffer"
LOG_DIRTY_KEY = "ui_log_dirty"


class StreamlitLogHandler(logging.Handler):
    """Thread-safe log handler that enqueues messages for the UI thread."""

    def __init__(self, target_queue: "queue.SimpleQueue[str]") -> None:
        super().__init__(level=logging.INFO)
        self._queue = target_queue
        self.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - see class docstring
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive
            return

        try:
            self._queue.put_nowait(message)
        except queue.Full:  # pragma: no cover - queue is unbounded but defensive check
            pass


def _initialise_log_state() -> tuple[queue.SimpleQueue[str], List[str]]:
    """Ensure session state contains queue and buffer for UI logs."""

    if LOG_QUEUE_KEY not in st.session_state:
        st.session_state[LOG_QUEUE_KEY] = queue.SimpleQueue()
    if LOG_BUFFER_KEY not in st.session_state:
        st.session_state[LOG_BUFFER_KEY] = []
    if LOG_DIRTY_KEY not in st.session_state:
        st.session_state[LOG_DIRTY_KEY] = True
    return st.session_state[LOG_QUEUE_KEY], st.session_state[LOG_BUFFER_KEY]


def _drain_log_queue(target_buffer: List[str]) -> None:
    """Transfer queued log messages into the render buffer on the main thread."""

    message_queue: queue.SimpleQueue[str] = st.session_state[LOG_QUEUE_KEY]

    while True:
        try:
            message = message_queue.get_nowait()
        except queue.Empty:
            break
        target_buffer.append(message)

    if len(target_buffer) > MAX_UI_LOG_LINES:
        del target_buffer[: len(target_buffer) - MAX_UI_LOG_LINES]

    st.session_state[LOG_DIRTY_KEY] = True


def _render_log_panel(placeholder: Any, log_buffer: List[str]) -> None:
    """Render buffered logs inside the provided placeholder."""

    if not log_buffer:
        placeholder.info("로그가 아직 없습니다.")
    else:
        placeholder.markdown("```\n" + "\n".join(log_buffer) + "\n```")
    st.session_state[LOG_DIRTY_KEY] = False


def _refresh_ui_logs(placeholder: Any, log_buffer: List[str]) -> None:
    """Drain queued logs and update the Streamlit panel if required."""

    _drain_log_queue(log_buffer)
    if st.session_state.get(LOG_DIRTY_KEY):
        _render_log_panel(placeholder, log_buffer)


def _approximate_tokens(text: str) -> int:
    """Rudimentary character-based token estimate for heuristics."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_tokens_for_batch(batch: dict[str, object]) -> int:
    """Estimate total prompt tokens for a single translation batch."""

    texts = str(batch.get("texts", ""))
    ppt_context = str(batch.get("ppt_context", ""))
    glossary_terms = str(batch.get("glossary_terms", ""))

    token_estimate = (
        _approximate_tokens(texts)
        + _approximate_tokens(ppt_context)
        + _approximate_tokens(glossary_terms)
        + 200  # instructions + response padding
    )

    return max(1, token_estimate)


def _attach_streamlit_log_handler(log_queue: "queue.SimpleQueue[str]") -> None:
    """Attach (or replace) the Streamlit log handler on the root logger."""

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, StreamlitLogHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(StreamlitLogHandler(log_queue))


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
    """Calculate a batch size that balances latency and throughput."""

    if total_paragraphs <= 0:
        return 1

    min_size = max(1, int(getattr(settings, "min_batch_size", 40)))
    max_size = max(min_size, int(getattr(settings, "max_batch_size", getattr(settings, "batch_size", min_size))))
    default_size = max(min_size, min(max_size, int(getattr(settings, "batch_size", max_size))))

    concurrency = max(1, int(getattr(settings, "max_concurrency", 1)))
    wave_multiplier = float(getattr(settings, "wave_multiplier", 1.2) or 1.2)
    wave_multiplier = max(1.0, wave_multiplier)

    target_batches = max(concurrency, int(math.ceil(concurrency * wave_multiplier * 2)))
    suggested_size = math.ceil(total_paragraphs / target_batches) if target_batches > 0 else default_size

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

    log_panel = st.expander("📜 실행 로그", expanded=True)
    log_placeholder = log_panel.empty()
    log_queue, log_buffer = _initialise_log_state()
    _attach_streamlit_log_handler(log_queue)
    _refresh_ui_logs(log_placeholder, log_buffer)

    uploaded_file = st.file_uploader("PPT 파일 업로드", type=["ppt", "pptx"])

    ppt_buffer = None
    if uploaded_file:
        ppt_buffer = handle_upload(uploaded_file, max_size_mb=settings.max_upload_size_mb)
        _refresh_ui_logs(log_placeholder, log_buffer)

    if ppt_buffer:
        settings_state = render_settings()
        _refresh_ui_logs(log_placeholder, log_buffer)

        if st.button("🚀 번역 시작", type="primary"):
            with st.spinner("번역 진행 중..."):
                log_buffer.clear()
                while True:
                    try:
                        log_queue.get_nowait()
                    except queue.Empty:
                        break
                st.session_state[LOG_DIRTY_KEY] = True
                _render_log_panel(log_placeholder, log_buffer)

                cached_buffer = get_cached_upload()
                if cached_buffer is None:
                    st.error("업로드된 파일을 찾을 수 없습니다. 다시 업로드해주세요.")
                    return

                parser = PPTParser()
                paragraphs, presentation = parser.extract_paragraphs(cached_buffer)
                _refresh_ui_logs(log_placeholder, log_buffer)

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

                batches = chunk_paragraphs(
                    paragraphs,
                    batch_size=batch_size,
                    ppt_context=ppt_context,
                    glossary_terms=glossary_terms,
                    prepared_texts=prepared_texts,
                )

                LOGGER.info(
                    "Prepared %d batches (batch size %d, total paragraphs %d).",
                    len(batches),
                    batch_size,
                    len(paragraphs),
                )
                _refresh_ui_logs(log_placeholder, log_buffer)

                if not batches:
                    st.warning("번역할 배치를 생성하지 못했습니다.")
                    return

                estimated_tokens = _estimate_tokens_for_batch(batches[0])
                safe_concurrency = max(
                    1,
                    min(
                        int(settings.max_concurrency),
                        max(1, settings.tpm_limit // max(estimated_tokens, 1)),
                    ),
                )

                LOGGER.info(
                    "Estimated %d tokens per batch; using concurrency=%d (config max=%d, TPM limit=%d).",
                    estimated_tokens,
                    safe_concurrency,
                    settings.max_concurrency,
                    settings.tpm_limit,
                )
                _refresh_ui_logs(log_placeholder, log_buffer)

                st.caption(
                    f"배치 크기: {batch_size} 문장 (총 {len(paragraphs)} 문장) | 동시 실행: {safe_concurrency}"
                )

                progress_tracker = ProgressTracker(
                    total_batches=len(batches),
                    total_sentences=len(paragraphs),
                    log_update_fn=lambda: _refresh_ui_logs(log_placeholder, log_buffer),
                )

                chain = create_translation_chain(
                    model_name=settings_state.get("model", "gpt-5"),
                    source_lang=source_language,
                    target_lang=target_language,
                    user_prompt=settings_state.get("user_prompt"),
                )

                try:
                    LOGGER.info(
                        "Starting translation with concurrency=%d and model=%s.",
                        safe_concurrency,
                        settings_state.get("model", "gpt-5"),
                    )
                    _refresh_ui_logs(log_placeholder, log_buffer)
                    translated_texts = translate_with_progress(
                        chain,
                        batches,
                        progress_tracker,
                        max_concurrency=safe_concurrency,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.exception("Translation failed: %s", exc)
                    st.error("번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                    return

                if glossary:
                    translated_texts = [
                        GlossaryLoader.apply_glossary_to_translation(text, glossary)
                        for text in translated_texts
                    ]

                writer = PPTWriter()
                output_buffer = writer.apply_translations(paragraphs, translated_texts, presentation)
                _refresh_ui_logs(log_placeholder, log_buffer)

                total_elapsed = progress_tracker.get_total_elapsed()
                minutes, seconds = divmod(total_elapsed, 60)
                LOGGER.info("Translation completed in %d분 %.1f초", int(minutes), seconds)
                _refresh_ui_logs(log_placeholder, log_buffer)
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
