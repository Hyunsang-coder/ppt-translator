"""Streamlit entry point for the PPT translation prototype."""

from __future__ import annotations

import io
import logging
import math
import queue
from itertools import islice
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from PIL import Image

from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
from src.core.ppt_parser import PPTParser
from src.core.ppt_writer import PPTWriter
from src.core.text_extractor import ExtractionOptions, docs_to_markdown, extract_pptx_to_docs
from src.ui.extraction_settings import render_extraction_settings
from src.ui.file_handler import get_cached_upload, handle_upload
from src.ui.progress_tracker import ProgressTracker
from src.ui.settings_panel import render_settings
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import chunk_paragraphs
from src.utils.repetition import build_repetition_plan, expand_translations
from src.utils.language_detector import LanguageDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

CAT_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "번역캣 회색.png"
try:
    resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow < 9 fallback
    resample = Image.LANCZOS

try:
    CAT_IMAGE = Image.open(CAT_IMAGE_PATH)
    CAT_IMAGE_SCALED = CAT_IMAGE.resize(
        (
            max(1, int(CAT_IMAGE.width * 0.7)),
            max(1, int(CAT_IMAGE.height * 0.7)),
        ),
        resample,
    )
except FileNotFoundError:  # pragma: no cover - asset expected to be present in prod
    CAT_IMAGE = None
    CAT_IMAGE_SCALED = None

st.set_page_config(page_title="PPT 번역캣", page_icon=CAT_IMAGE or "📊", layout="wide")

MAX_UI_LOG_LINES = 400
LOG_QUEUE_KEY = "ui_log_queue"
LOG_BUFFER_KEY = "ui_log_buffer"
LOG_DIRTY_KEY = "ui_log_dirty"
LOG_HANDLER_KEY = "ui_log_handler_attached"
EXTRACTION_STATE_KEY = "text_extraction_state"


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
    st.session_state.setdefault(LOG_HANDLER_KEY, False)
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


def _estimate_tokens_for_batch(batch: Dict[str, object]) -> int:
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
    """Attach the Streamlit log handler on the root logger once per session."""

    if st.session_state.get(LOG_HANDLER_KEY):
        return

    root_logger = logging.getLogger()
    root_logger.addHandler(StreamlitLogHandler(log_queue))
    st.session_state[LOG_HANDLER_KEY] = True


def _load_glossary(glossary_file) -> Tuple[dict[str, str] | None, str]:
    """Load glossary data from the uploaded file."""

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


def _get_extraction_state() -> Dict[str, Any]:
    state = st.session_state.setdefault(EXTRACTION_STATE_KEY, {})
    state.setdefault("markdown", "")
    state.setdefault("file_name", None)
    state.setdefault("options", None)
    state.setdefault("slides", 0)
    state.setdefault("blocks", 0)
    state.setdefault("stale", False)
    return state


def _render_text_extraction_page(settings, extraction_options: ExtractionOptions) -> None:
    """Render PPT text extraction workflow."""

    st.title("🧾 PPT 텍스트 추출")
    st.markdown("PPT 파일에서 텍스트를 추출하여 Markdown 형식으로 정리할 수 있습니다.")

    uploaded_file = st.file_uploader(
        "PPTX 파일 업로드",
        type=["ppt", "pptx"],
        key="text_extraction_uploader",
        help="최대 %dMB까지 업로드 가능합니다." % settings.max_upload_size_mb,
    )

    state = _get_extraction_state()
    current_signature = {
        "figures": extraction_options.figures,
        "charts": extraction_options.charts,
        "with_notes": extraction_options.with_notes,
    }

    if state["markdown"]:
        if uploaded_file and uploaded_file.name != state["file_name"]:
            state["stale"] = True
        elif state["options"] != current_signature:
            state["stale"] = True
        else:
            state["stale"] = False
    else:
        state["stale"] = False

    convert_clicked = st.button(
        "Markdown 변환",
        type="primary",
        disabled=uploaded_file is None,
    )

    if convert_clicked and uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            st.error(
                f"파일 크기가 {settings.max_upload_size_mb}MB를 초과합니다. 더 작은 파일로 다시 시도해주세요."
            )
        else:
            ppt_buffer = io.BytesIO(uploaded_file.getvalue())
            ppt_buffer.seek(0)
            try:
                docs = extract_pptx_to_docs(ppt_buffer, extraction_options)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Extraction failed: %s", exc)
                st.error("텍스트 추출 중 오류가 발생했습니다. 파일을 다시 확인해주세요.")
            else:
                markdown_text = docs_to_markdown(docs, extraction_options)
                total_blocks = sum(len(doc.blocks) for doc in docs)
                state.update(
                    {
                        "markdown": markdown_text,
                        "file_name": uploaded_file.name,
                        "options": current_signature,
                        "slides": len(docs),
                        "blocks": total_blocks,
                        "stale": False,
                    }
                )
                st.session_state["markdown_preview"] = markdown_text
                if markdown_text.strip():
                    st.success(f"총 {len(docs)}개의 슬라이드에서 {total_blocks}개의 블록을 추출했습니다.")
                else:
                    st.warning("추출된 텍스트가 없습니다.")

    if state["stale"]:
        st.info("옵션이나 파일이 변경되었습니다. 다시 변환을 실행하면 최신 결과를 확인할 수 있습니다.")

    markdown_value = state["markdown"]
    if "markdown_preview" not in st.session_state:
        st.session_state["markdown_preview"] = markdown_value

    st.text_area(
        "Markdown 미리보기",
        height=400,
        key="markdown_preview",
    )

    if markdown_value.strip():
        safe_name = _sanitize_for_filename(Path(state["file_name"] or "presentation").stem, "presentation")
        download_name = f"{safe_name}_extracted.md"
        st.download_button(
            "📥 Markdown 다운로드",
            data=markdown_value.encode("utf-8"),
            file_name=download_name,
            mime="text/markdown",
        )



def _render_translation_page(settings, settings_state: Dict[str, Any]) -> None:
    """Render PPT translation workflow."""

    st.title("🌐 번역된 PPT 생성")
    st.markdown("원본 PPT의 디자인을 유지하면서 내부 텍스트만 번역한 새 파일을 생성합니다.")

    if not settings.openai_api_key:
        st.warning("OPENAI_API_KEY가 설정되지 않았습니다. 번역 실행 시 에러가 발생할 수 있습니다.")

    preprocess_repetitions = bool(settings_state.get("preprocess_repetitions"))
    if preprocess_repetitions:
        st.info("반복 문구 사전 처리 옵션이 활성화되어 동일 문장을 한 번만 번역합니다.")

    log_panel = st.expander("📜 실행 로그", expanded=True)
    log_placeholder = log_panel.empty()
    log_queue, log_buffer = _initialise_log_state()
    _attach_streamlit_log_handler(log_queue)
    _refresh_ui_logs(log_placeholder, log_buffer)

    uploaded_file = st.file_uploader(
        "PPT 파일 업로드",
        type=["ppt", "pptx"],
        key="translation_uploader",
        help="최대 %dMB까지 업로드 가능합니다." % settings.max_upload_size_mb,
    )

    ppt_buffer = None
    if uploaded_file:
        ppt_buffer = handle_upload(uploaded_file, max_size_mb=settings.max_upload_size_mb)
        _refresh_ui_logs(log_placeholder, log_buffer)

    if not ppt_buffer:
        return

    if st.button("🚀 번역 시작", type="primary"):
        with st.spinner("번역 진행 중..."):
            log_buffer.clear()
            while True:
                try:
                    log_queue.get_nowait()
                except queue.Empty:
                    break
            st.session_state[LOG_DIRTY_KEY] = True
            if st.session_state.get(LOG_HANDLER_KEY, False) is False:
                _attach_streamlit_log_handler(log_queue)
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

            repetition_plan = None
            target_paragraphs = paragraphs
            target_prepared_texts = prepared_texts

            if preprocess_repetitions:
                repetition_plan = build_repetition_plan(paragraphs)
                target_paragraphs = [paragraphs[idx] for idx in repetition_plan.unique_indices]
                target_prepared_texts = [prepared_texts[idx] for idx in repetition_plan.unique_indices]

                duplicates_info = repetition_plan.duplicate_counts()
                reduced = len(paragraphs) - len(target_paragraphs)
                if duplicates_info:
                    st.caption(
                        f"반복 문구 {len(duplicates_info)}개 감지: 번역 문장 수 {len(paragraphs)} → {len(target_paragraphs)} (감소 {reduced})"
                    )
                    preview = sorted(duplicates_info.items(), key=lambda item: item[1], reverse=True)
                    with st.expander("반복 문구 미리보기", expanded=False):
                        for text, count in islice(preview, 5):
                            st.write(f"{count}×: {text}")
                else:
                    st.caption("반복 문구 사전 처리 결과 중복 문장이 발견되지 않았습니다.")

                if not target_paragraphs:
                    st.warning("반복 문구 사전 처리 결과 번역할 텍스트가 없습니다.")
                    return

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

            batch_size = _determine_batch_size(len(target_paragraphs), settings)

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
                f"배치 크기: {batch_size} 문장 (고유 {len(target_paragraphs)} / 전체 {len(paragraphs)}) | 최대 동시 실행: {safe_concurrency}"
            )

            progress_tracker = ProgressTracker(
                total_batches=len(batches),
                total_sentences=len(target_paragraphs),
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
                translated_unique = translate_with_progress(
                    chain,
                    batches,
                    progress_tracker,
                    max_concurrency=safe_concurrency,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Translation failed: %s", exc)
                st.error("번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                return

            if repetition_plan is not None:
                translated_texts = expand_translations(
                    repetition_plan,
                    translated_unique,
                    len(paragraphs),
                )
            else:
                translated_texts = translated_unique

            if glossary:
                translated_texts = [
                    GlossaryLoader.apply_glossary_to_translation(text, glossary)
                    for text in translated_texts
                ]

            writer = PPTWriter()
            output_buffer = writer.apply_translations(paragraphs, translated_texts, presentation)
            _refresh_ui_logs(log_placeholder, log_buffer)

            total_elapsed = progress_tracker.finish()
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


def main() -> None:
    """Render the Streamlit UI and orchestrate workflows."""

    settings = get_settings()

    st.sidebar.markdown(
        """
        <div style="text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
            PPT 번역캣
        </div>
        """,
        unsafe_allow_html=True,
    )

    if CAT_IMAGE_SCALED is not None:
        st.sidebar.image(CAT_IMAGE_SCALED, use_container_width=True)
    elif CAT_IMAGE is not None:
        st.sidebar.image(CAT_IMAGE, use_container_width=True)

    st.sidebar.markdown("### 기능 선택")
    feature = st.sidebar.radio(
        "기능 선택",
        options=("PPT 번역", "텍스트 추출"),
        index=0,
        label_visibility="collapsed",
    )

    if feature == "텍스트 추출":
        extraction_options = render_extraction_settings(st.sidebar)
        _render_text_extraction_page(settings, extraction_options)
    else:
        translation_settings = render_settings(st.sidebar)
        _render_translation_page(settings, translation_settings)


if __name__ == "__main__":
    main()
