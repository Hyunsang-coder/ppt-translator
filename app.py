"""Streamlit entry point for the PPT translation prototype."""

from __future__ import annotations

import html
import io
import json
import logging
import math
import queue
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.chains.context_manager import ContextManager
from src.chains.translation_chain import create_translation_chain, translate_with_progress
from src.core.ppt_parser import PPTParser
from src.core.ppt_writer import PPTWriter
from src.core.text_extractor import ExtractionOptions, docs_to_markdown, extract_pptx_to_docs
from src.core.pdf_processor import PDFProcessor
from src.core.pdf_to_ppt_writer import PDFToPPTWriter, TextBoxStyle
from src.ui.extraction_settings import render_extraction_settings
from src.ui.file_handler import handle_upload
from src.ui.progress_tracker import ProgressTracker
from src.ui.settings_panel import render_settings
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.helpers import chunk_paragraphs
from src.utils.repetition import build_repetition_plan, expand_translations
from src.utils.language_detector import LanguageDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

APP_ROOT_PATH = Path(__file__).resolve()
APP_BASE_DIR = APP_ROOT_PATH.parent
APP_TIMEZONE = ZoneInfo("Asia/Seoul")


def _compute_last_updated_date() -> str:
    """Resolve last updated date from Git, GitHub API, or file modification time."""

    # 1. Try Git
    try:
        git_output = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=short"],
            cwd=APP_BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        last_commit_date = git_output.stdout.strip()
        if last_commit_date:
            return last_commit_date
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. Try GitHub API (Fallback for cloud environments without .git)
    try:
        import urllib.request
        repo_api_url = "https://api.github.com/repos/Hyunsang-coder/ppt-translator"
        headers = {"User-Agent": "Streamlit-App-Metadata-Fetcher"}
        req = urllib.request.Request(repo_api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            pushed_at = data.get("pushed_at")
            if pushed_at:
                return pushed_at[:10]  # Extract YYYY-MM-DD
    except Exception:
        pass

    # 3. Try File Modification Time (Fallback for offline/no-git environments)
    try:
        mtime = APP_ROOT_PATH.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=APP_TIMEZONE).strftime("%Y-%m-%d")
    except Exception:
        pass

    return datetime.now(APP_TIMEZONE).strftime("%Y-%m-%d")


APP_LAST_UPDATED = _compute_last_updated_date()

CAT_IMAGE_PATH = APP_BASE_DIR / "assets" / "번역캣 회색.png"
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
PDF_CONVERSION_STATE_KEY = "pdf_conversion_state"


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
    elif len(st.session_state[LOG_BUFFER_KEY]) > MAX_UI_LOG_LINES * 2:
        # Prevent excessive memory usage by clearing buffer if it exceeds 2x limit
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

    from src.utils.security import sanitize_filename

    return sanitize_filename(value, fallback=fallback)


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
            
            # Validate file signature
            from src.utils.security import validate_pptx_file, sanitize_filename
            is_valid, error_msg = validate_pptx_file(ppt_buffer)
            
            if not is_valid:
                st.error(error_msg or "파일 형식이 올바르지 않습니다. PPT 또는 PPTX 파일만 업로드 가능합니다.")
                ppt_buffer.close()
            else:
                ppt_buffer.seek(0)
                try:
                    docs = extract_pptx_to_docs(ppt_buffer, extraction_options)
                    markdown_text = docs_to_markdown(docs, extraction_options)
                    total_blocks = sum(len(doc.blocks) for doc in docs)
                    sanitized_name = sanitize_filename(uploaded_file.name)
                    state.update(
                        {
                            "markdown": markdown_text,
                            "file_name": sanitized_name,
                            "options": current_signature,
                            "slides": len(docs),
                            "blocks": total_blocks,
                            "stale": False,
                        }
                    )
                    if markdown_text.strip():
                        st.success(f"총 {len(docs)}개의 슬라이드에서 {total_blocks}개의 블록을 추출했습니다.")
                    else:
                        st.warning("추출된 텍스트가 없습니다.")
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.exception("Extraction failed: %s", exc)
                    st.error("텍스트 추출 중 오류가 발생했습니다. 파일을 다시 확인해주세요.")
                finally:
                    # Explicitly close buffer to free memory
                    ppt_buffer.close()

    if state["stale"]:
        st.info("옵션이나 파일이 변경되었습니다. 다시 변환을 실행하면 최신 결과를 확인할 수 있습니다.")

    markdown_value = state["markdown"]

    # 버튼을 먼저 표시
    if markdown_value.strip():
        safe_name = _sanitize_for_filename(Path(state["file_name"] or "presentation").stem, "presentation")
        download_name = f"{safe_name}_extracted.md"
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "📥 Markdown 다운로드",
                data=markdown_value.encode("utf-8"),
                file_name=download_name,
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            # JavaScript를 사용한 클립보드 복사
            # JSON으로 직렬화하여 안전하게 JavaScript로 전달
            # 길이 제한을 추가하여 XSS 및 DoS 방지
            max_markdown_length = 10 * 1024 * 1024  # 10MB 제한
            safe_markdown = markdown_value[:max_markdown_length] if len(markdown_value) > max_markdown_length else markdown_value
            escaped_markdown = json.dumps(safe_markdown)
            
            # JSON 문자열 길이 확인 (과도한 길이 방지)
            if len(escaped_markdown) > 15 * 1024 * 1024:  # 15MB 제한 (JSON 이스케이프 고려)
                st.warning("내용이 너무 커서 클립보드 복사 기능을 사용할 수 없습니다. 다운로드 버튼을 사용해주세요.")
                copy_html = """
                <div style="padding: 0.375rem 0.75rem; text-align: center; color: #6b7280;">
                    내용이 너무 커서 복사할 수 없습니다
                </div>
                """
            else:
                copy_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{
                            margin: 0;
                            padding: 0;
                            font-family: 'Source Sans Pro', sans-serif;
                        }}
                        button {{
                            width: 100%;
                            padding: 0.375rem 0.75rem;
                            background-color: rgb(255, 255, 255);
                            color: rgb(49, 51, 63);
                            border: 1px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            font-family: 'Source Sans Pro', sans-serif;
                            font-size: 1rem;
                            font-weight: 400;
                            line-height: 1.6;
                            cursor: pointer;
                            transition: all 0.2s;
                        }}
                        button:hover {{
                            border-color: rgb(255, 75, 75);
                            color: rgb(255, 75, 75);
                        }}
                    </style>
                </head>
                <body>
                    <button onclick="copyToClipboard()">📋 클립보드 복사</button>
                    <script>
                        (function() {{
                            const text = {escaped_markdown};
                            
                            function copyToClipboard() {{
                                if (typeof navigator !== 'undefined' && navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        const btn = document.querySelector('button');
                                        if (btn) {{
                                            btn.textContent = '✅ 복사 완료!';
                                            setTimeout(function() {{
                                                btn.textContent = '📋 클립보드 복사';
                                            }}, 2000);
                                        }}
                                    }}).catch(function(err) {{
                                        alert('복사에 실패했습니다.');
                                    }});
                                }} else {{
                                    alert('클립보드 API를 사용할 수 없습니다.');
                                }}
                            }}
                            
                            // 전역 함수로 노출
                            window.copyToClipboard = copyToClipboard;
                        }})();
                    </script>
                </body>
                </html>
                """
            components.html(copy_html, height=50)

    # 미리보기를 버튼 아래에 표시
    st.subheader("Markdown 미리보기")
    st.code(
        markdown_value,
        language="markdown",
        line_numbers=False,
    )


def _get_pdf_conversion_state() -> Dict[str, Any]:
    """Get or initialize PDF conversion state."""
    state = st.session_state.setdefault(PDF_CONVERSION_STATE_KEY, {})
    state.setdefault("result_buffer", None)
    state.setdefault("file_name", None)
    state.setdefault("pages_processed", 0)
    state.setdefault("text_blocks_count", 0)
    return state


def _render_pdf_conversion_settings(sidebar) -> Dict[str, Any]:
    """Render PDF to PPT conversion settings in sidebar."""
    sidebar.markdown("### PDF 변환 설정")
    
    sidebar.info("🤖 OpenAI Vision API를 사용하여 PDF를 분석합니다. API 비용이 발생합니다.")

    sidebar.markdown("#### 텍스트 박스 스타일")

    use_auto_color = sidebar.checkbox(
        "자동 색상 매칭 (Adaptive Style)",
        value=True,
        help="원본 이미지의 배경색을 분석하여 텍스트 박스 색상을 자동으로 맞춥니다.",
    )

    if not use_auto_color:
        bg_color = sidebar.color_picker(
            "배경색",
            value="#FFFFFF",
            help="텍스트 박스의 배경색을 선택합니다. 원본 텍스트를 덮습니다.",
        )

        text_color = sidebar.color_picker(
            "글자색",
            value="#000000",
            help="텍스트 색상을 선택합니다.",
        )
    else:
        bg_color = None
        text_color = None

    font_name = sidebar.selectbox(
        "폰트",
        options=["맑은 고딕", "Arial", "나눔고딕", "굴림"],
        index=0,
        help="텍스트 박스에 사용할 폰트를 선택합니다.",
    )

    sidebar.markdown("#### 이미지 설정")

    include_background = sidebar.checkbox(
        "원본 배경 이미지 포함",
        value=False,
        help="체크하면 PPT 슬라이드 배경으로 원본 PDF 이미지를 삽입합니다.",
    )

    dpi = sidebar.slider(
        "이미지 품질 (DPI)",
        min_value=72,
        max_value=300,
        value=200,
        step=18,
        help="PDF를 이미지로 변환할 때의 해상도입니다. 높을수록 품질이 좋지만 처리 시간이 길어집니다.",
    )

    return {
        "use_auto_color": use_auto_color,
        "bg_color": bg_color,
        "text_color": text_color,
        "font_name": font_name,
        "include_background": include_background,
        "dpi": dpi,
    }


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _render_pdf_conversion_page(settings, conversion_settings: Dict[str, Any]) -> None:
    """Render PDF to PPT conversion workflow."""

    st.title("📄 PDF → PPT 변환")
    st.markdown(
        "OpenAI Vision을 사용하여 PDF를 지능적으로 분석하고, "
        "원본 페이지를 배경으로 편집 가능한 PPT를 생성합니다."
    )

    # Check API key
    if not settings.openai_api_key:
        st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. PDF 변환 기능을 사용하려면 API 키가 필요합니다.")
        return

    # Log panel setup
    log_panel = st.expander("📜 실행 로그", expanded=True)
    log_placeholder = log_panel.empty()
    log_queue, log_buffer = _initialise_log_state()
    _attach_streamlit_log_handler(log_queue)
    _refresh_ui_logs(log_placeholder, log_buffer)

    uploaded_file = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        key="pdf_conversion_uploader",
        help="최대 %dMB까지 업로드 가능합니다." % settings.max_upload_size_mb,
    )

    state = _get_pdf_conversion_state()

    # Check if file changed
    if uploaded_file:
        if state["file_name"] != uploaded_file.name:
            state["result_buffer"] = None
            state["file_name"] = uploaded_file.name
            state["pages_processed"] = 0
            state["text_blocks_count"] = 0

    convert_clicked = st.button(
        "🔄 PPT로 변환",
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
            pdf_buffer = io.BytesIO(uploaded_file.getvalue())
            pdf_buffer.seek(0)

            with st.spinner("OpenAI Vision으로 PDF를 분석하는 중... (페이지당 약 5-10초 소요)"):
                try:
                    # Clear log buffer for fresh start
                    log_buffer.clear()
                    st.session_state[LOG_DIRTY_KEY] = True
                    _render_log_panel(log_placeholder, log_buffer)

                    # Initialize PDF processor
                    processor = PDFProcessor(
                        api_key=settings.openai_api_key,
                        model="gpt-5.1",
                        dpi=conversion_settings["dpi"],
                    )

                    # Process PDF
                    LOGGER.info("PDF 처리 시작 (Vision-First): %s", uploaded_file.name)
                    _refresh_ui_logs(log_placeholder, log_buffer)

                    ocr_results = processor.process_pdf(pdf_buffer)
                    _refresh_ui_logs(log_placeholder, log_buffer)

                    if not ocr_results:
                        st.warning("PDF에서 페이지를 추출할 수 없습니다.")
                        return

                    # Create PPT with precise positioning
                    if conversion_settings["use_auto_color"]:
                        # Auto color: Pass None so backend uses adaptive logic
                        text_style = TextBoxStyle(
                            font_name=conversion_settings["font_name"],
                            background_color=None, # Signal for adaptive
                            text_color=None        # Signal for adaptive
                        )
                    else:
                        # Manual color
                        text_style = TextBoxStyle(
                            font_name=conversion_settings["font_name"],
                            background_color=_hex_to_rgb(conversion_settings["bg_color"]),
                            text_color=_hex_to_rgb(conversion_settings["text_color"]),
                        )

                    writer = PDFToPPTWriter(text_style=text_style)
                    output_buffer = writer.create_presentation(
                        ocr_results,
                        include_background=conversion_settings["include_background"]
                    )
                    _refresh_ui_logs(log_placeholder, log_buffer)

                    # Update state
                    total_blocks = sum(len(r.text_blocks) for r in ocr_results)
                    state["result_buffer"] = output_buffer
                    state["pages_processed"] = len(ocr_results)
                    state["text_blocks_count"] = total_blocks

                    LOGGER.info(
                        "변환 완료: %d페이지, %d개 텍스트 블록",
                        len(ocr_results),
                        total_blocks,
                    )
                    _refresh_ui_logs(log_placeholder, log_buffer)

                    st.success(
                        f"✅ 변환 완료! {len(ocr_results)}페이지에서 {total_blocks}개의 텍스트 블록을 추출했습니다."
                    )

                except ValueError as e:
                    LOGGER.error("설정 오류: %s", e)
                    st.error(str(e))
                except ImportError as e:
                    LOGGER.error("필수 라이브러리가 설치되지 않았습니다: %s", e)
                    st.error(
                        "필수 라이브러리가 설치되지 않았습니다. "
                        "`pip install PyMuPDF langchain-openai` 명령어로 설치해주세요."
                    )
                except Exception as exc:
                    LOGGER.exception("PDF 변환 실패: %s", exc)
                    st.error("PDF 변환 중 오류가 발생했습니다. 파일을 다시 확인해주세요.")
                finally:
                    pdf_buffer.close()

    # Download button
    if state["result_buffer"] is not None:
        original_name = Path(state["file_name"] or "document").stem
        safe_name = _sanitize_for_filename(original_name, "document")
        timestamp = datetime.now().strftime("%Y%m%d")
        download_name = f"{safe_name}_converted_{timestamp}.pptx"

        st.download_button(
            label="📥 변환된 PPT 다운로드",
            data=state["result_buffer"].getvalue(),
            file_name=download_name,
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

        st.caption(
            f"📊 {state['pages_processed']}페이지, {state['text_blocks_count']}개 텍스트 블록"
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

            parser = PPTParser()
            ppt_buffer.seek(0)
            paragraphs, presentation = parser.extract_paragraphs(ppt_buffer)
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
                        from src.utils.security import sanitize_html_content
                        
                        # 각 텍스트를 안전하게 이스케이프
                        safe_rows = []
                        for text, count in preview:
                            safe_text = sanitize_html_content(text, max_length=500)
                            safe_rows.append(f"<strong>{count}×</strong>: {safe_text}")
                        
                        rows = "<br>".join(safe_rows)
                        st.markdown(
                            """
                            <div style="max-height: 280px; overflow-y: auto; padding-right: 6px;">
                                {rows}
                            </div>
                            """.format(rows=rows or "<em>중복 문장이 없습니다.</em>"),
                            unsafe_allow_html=True,
                        )
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
                model_name=settings_state.get("model", "gpt-5.1"),
                source_lang=source_language,
                target_lang=target_language,
                user_prompt=settings_state.get("user_prompt"),
            )

            try:
                LOGGER.info(
                    "Starting translation with concurrency=%d and model=%s.",
                    safe_concurrency,
                    settings_state.get("model", "gpt-5.1"),
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

            # Explicitly clear large objects to help GC
            paragraphs = None
            presentation = None
            translated_texts = None
            if repetition_plan is not None:
                translated_unique = None

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
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-size: 0.9rem; color: #6b7280; margin-top: -0.25rem; margin-bottom: 0.5rem;">
            (Version 2.2, last updated: {APP_LAST_UPDATED})
        </div>
        """,
        unsafe_allow_html=True,
    )

    if CAT_IMAGE_SCALED is not None:
        st.sidebar.image(CAT_IMAGE_SCALED)
    elif CAT_IMAGE is not None:
        st.sidebar.image(CAT_IMAGE)

    st.sidebar.markdown("### 기능 선택")
    feature = st.sidebar.radio(
        "기능 선택",
        options=("PPT 번역", "텍스트 추출", "PDF → PPT 변환"),
        index=0,
        label_visibility="collapsed",
    )

    if feature == "텍스트 추출":
        extraction_options = render_extraction_settings(st.sidebar)
        _render_text_extraction_page(settings, extraction_options)
    elif feature == "PDF → PPT 변환":
        conversion_settings = _render_pdf_conversion_settings(st.sidebar)
        _render_pdf_conversion_page(settings, conversion_settings)
    else:
        translation_settings = render_settings(st.sidebar)
        _render_translation_page(settings, translation_settings)


if __name__ == "__main__":
    main()
