"""FastAPI server for PPT translation API."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import os
import resource
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.core.text_extractor import ExtractionOptions, docs_to_markdown, extract_pptx_to_docs
from src.services import (
    TranslationProgress,
    TranslationRequest,
    TranslationResult,
    TranslationService,
    get_job_manager,
    Job,
    JobState,
    JobType,
)
from src.services.models import MODEL_REGISTRY, DEFAULT_LIGHT_MODEL, DEFAULT_TRANSLATION_MODEL, TextFitMode
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.security import sanitize_filename, validate_pptx_file

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Limit thread pool to prevent runaway thread creation
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    """Set a bounded thread pool and start periodic job cleanup on startup."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_thread_pool)
    job_manager = get_job_manager()
    job_manager.start_cleanup_loop()
    yield
    await job_manager.stop_cleanup_loop()
    _thread_pool.shutdown(wait=False)


app = FastAPI(
    title="PPT 번역캣 API",
    description="PowerPoint translation API using OpenAI GPT and Anthropic Claude models",
    version="2.4.0",
    lifespan=lifespan,
)

# CORS middleware - read allowed origins from environment
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
# Desktop (Tauri) build: the WebView origin varies by platform/mode and the
# server only ever binds loopback, so allow any origin when CORS_ALLOW_ALL=1.
# allow_credentials must be False with a wildcard origin per the CORS spec; we
# don't use cookie auth, so that's fine.
_cors_allow_all = os.getenv("CORS_ALLOW_ALL") == "1"
_cors_kwargs: Dict[str, Any] = (
    {"allow_origins": ["*"], "allow_credentials": False}
    if _cors_allow_all
    else {"allow_origins": _cors_origins, "allow_credentials": True}
)
app.add_middleware(
    CORSMiddleware,
    **_cors_kwargs,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Translation-Source-Lang",
        "X-Translation-Target-Lang",
        "X-Translation-Total-Paragraphs",
        "X-Translation-Unique-Paragraphs",
        "X-Translation-Batch-Count",
        "X-Translation-Elapsed-Seconds",
        "Content-Disposition",
    ],
)

# ============================================================================
# Pydantic Models for API responses
# ============================================================================


class JobsHealthInfo(BaseModel):
    """Job concurrency info for health check."""

    running: int
    pending: int
    max_running: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    openai_api_key_configured: bool
    anthropic_api_key_configured: bool
    jobs: Optional[JobsHealthInfo] = None
    memory_usage_mb: Optional[float] = None


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    name: str
    provider: str


class ModelsResponse(BaseModel):
    """Available models response."""

    models: List[ModelInfo]


class LanguageInfo(BaseModel):
    """Language information."""

    code: str
    name: str


class LanguagesResponse(BaseModel):
    """Available languages response."""

    languages: List[LanguageInfo]


class ConfigResponse(BaseModel):
    """Configuration response."""

    max_upload_size_mb: int
    providers: List[str]
    default_provider: str
    default_model: str


class JobCreateResponse(BaseModel):
    """Job creation response."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    job_type: str
    state: str
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    progress: Optional[Dict[str, Any]]
    error_message: Optional[str]


class ExtractionResponse(BaseModel):
    """Text extraction response."""

    markdown: str
    slide_count: int


class SummarizeRequest(BaseModel):
    """Summarization request."""

    markdown: str
    provider: str = "anthropic"
    model: str = DEFAULT_LIGHT_MODEL["anthropic"]


class SummarizeResponse(BaseModel):
    """Summarization response."""

    summary: str


class FilenameSettings(BaseModel):
    """Filename generation settings."""

    mode: Literal["auto", "custom"] = "auto"
    includeLanguage: bool = True
    includeOriginalName: bool = True
    includeModel: bool = False
    includeDate: bool = True
    componentOrder: List[Literal["language", "originalName", "model", "date"]] = Field(
        default_factory=lambda: ["language", "originalName", "model", "date"]
    )
    customName: str = ""


def get_language_code(language: str) -> str:
    """Convert language name to English abbreviation for filenames."""
    code_map = {
        "한국어": "KR",
        "영어": "EN",
        "일본어": "JP",
        "중국어": "CN",
        "스페인어": "ES",
        "프랑스어": "FR",
        "독일어": "DE",
    }
    return code_map.get(language, language)


def get_model_display_name(model_id: str) -> str:
    """Convert model ID to display name for filenames."""
    for models in SUPPORTED_MODELS.values():
        for model in models:
            if model.id == model_id:
                return model.name
    return model_id


def generate_output_filename(
    filename_settings: FilenameSettings,
    original_filename: str,
    target_language: str,
    model: str,
) -> str:
    """Generate output filename based on settings."""
    if filename_settings.mode == "custom" and filename_settings.customName.strip():
        safe_custom = sanitize_filename(filename_settings.customName.strip(), fallback="translated")
        return f"{safe_custom}.pptx"

    # Auto mode
    timestamp = datetime.now().strftime("%Y%m%d")

    original_name = Path(original_filename).stem
    safe_original = sanitize_filename(original_name, fallback="presentation")
    # Use English abbreviation for language code
    lang_code = get_language_code(target_language)
    # Use display name instead of model ID
    model_display_name = get_model_display_name(model)
    safe_model = sanitize_filename(model_display_name, fallback="model")

    part_values = {
        "language": lang_code,
        "originalName": safe_original,
        "model": safe_model,
        "date": timestamp,
    }
    include_flags = {
        "language": filename_settings.includeLanguage,
        "originalName": filename_settings.includeOriginalName,
        "model": filename_settings.includeModel,
        "date": filename_settings.includeDate,
    }
    default_order = ["language", "originalName", "model", "date"]

    ordered_parts: list[str] = []
    for part in filename_settings.componentOrder:
        if part in default_order and part not in ordered_parts:
            ordered_parts.append(part)
    for part in default_order:
        if part not in ordered_parts:
            ordered_parts.append(part)

    parts = [part_values[part] for part in ordered_parts if include_flags[part]]

    if not parts:
        return "translated.pptx"

    return f"{'_'.join(parts)}.pptx"


# ============================================================================
# Configuration Data
# ============================================================================

# Built from the single source of truth in src/services/models.py so adding
# or bumping a model is a one-file change.
SUPPORTED_MODELS: Dict[str, List[ModelInfo]] = {
    provider: [
        ModelInfo(id=model_id, name=name, provider=provider)
        for model_id, name in entries
    ]
    for provider, entries in MODEL_REGISTRY.items()
}

def validate_model(provider: str, model: str) -> None:
    """Raise HTTP 400 if the model is not in the provider's allowlist.

    Prevents callers from charging arbitrary (non-exposed) models to our keys.
    Assumes the provider has already been validated against SUPPORTED_MODELS.
    """
    allowed = {m.id for m in SUPPORTED_MODELS.get(provider, [])}
    if model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}' for provider '{provider}'. Must be one of: {sorted(allowed)}",
        )


SUPPORTED_LANGUAGES: List[LanguageInfo] = [
    LanguageInfo(code="Auto", name="Auto (자동 감지)"),
    LanguageInfo(code="한국어", name="한국어"),
    LanguageInfo(code="영어", name="English"),
    LanguageInfo(code="일본어", name="日本語"),
    LanguageInfo(code="중국어", name="中文"),
    LanguageInfo(code="스페인어", name="Español"),
    LanguageInfo(code="프랑스어", name="Français"),
    LanguageInfo(code="독일어", name="Deutsch"),
]

# Language code mapping for filenames (Korean name -> English abbreviation)
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "한국어": "KR",
    "영어": "EN",
    "일본어": "JP",
    "중국어": "CN",
    "스페인어": "ES",
    "프랑스어": "FR",
    "독일어": "DE",
}


# ============================================================================
# Health & Config Endpoints
# ============================================================================


def _get_memory_usage_mb() -> float:
    """Return current process RSS memory usage in MB (Linux/macOS)."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS returns bytes, Linux returns kilobytes
        import sys

        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024 * 1024)
        return ru.ru_maxrss / 1024
    except Exception:
        return 0.0


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    job_manager = get_job_manager()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        openai_api_key_configured=bool(settings.openai_api_key),
        anthropic_api_key_configured=bool(settings.anthropic_api_key),
        jobs=JobsHealthInfo(
            running=job_manager.get_running_count(),
            pending=job_manager.get_pending_count(),
            max_running=job_manager.max_running,
        ),
        memory_usage_mb=round(_get_memory_usage_mb(), 1),
    )


@app.get("/api/v1/models", response_model=ModelsResponse)
async def get_models(provider: Optional[str] = None) -> ModelsResponse:
    """Get available models, optionally filtered by provider."""
    if provider:
        if provider not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {provider}. Must be one of: {list(SUPPORTED_MODELS.keys())}",
            )
        return ModelsResponse(models=SUPPORTED_MODELS[provider])

    all_models = []
    for models in SUPPORTED_MODELS.values():
        all_models.extend(models)
    return ModelsResponse(models=all_models)


@app.get("/api/v1/languages", response_model=LanguagesResponse)
async def get_languages() -> LanguagesResponse:
    """Get supported languages."""
    return LanguagesResponse(languages=SUPPORTED_LANGUAGES)


@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get application configuration."""
    settings = get_settings()
    return ConfigResponse(
        max_upload_size_mb=settings.max_upload_size_mb,
        providers=list(SUPPORTED_MODELS.keys()),
        default_provider="anthropic",
        default_model="claude-sonnet-4-6",
    )


PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_UPLOAD_CHUNK_SIZE = 1024 * 1024


async def _save_upload_to_temp_path(
    upload: UploadFile,
    directory: Path,
    *,
    suffix: str,
    max_size_mb: int,
) -> tuple[Path, int]:
    """Stream an uploaded file to a temp path and enforce size while reading."""
    directory.mkdir(parents=True, exist_ok=True)
    max_bytes = max_size_mb * 1024 * 1024
    written = 0
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        dir=directory,
        delete=False,
    )
    path = Path(temp_file.name)

    try:
        with temp_file:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds limit ({max_size_mb}MB)",
                    )
                temp_file.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return path, written


def _validate_ppt_path(path: Path) -> None:
    """Validate PPT/PPTX signature from a filesystem path."""
    with path.open("rb") as file_obj:
        is_valid, error_msg = validate_pptx_file(file_obj)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg or "Invalid PPT/PPTX file format")


# ============================================================================
# Job System Endpoints
# ============================================================================


def _create_progress_callback(job_id: str):
    """Create a progress callback for the job."""
    job_manager = get_job_manager()

    def callback(progress: TranslationProgress) -> None:
        job_manager.update_job_progress(job_id, progress)

    return callback


async def _run_translation_job(
    job_id: str,
    ppt_path: Path,
    work_dir: Path,
    filename: str,
    source_lang: str,
    target_lang: str,
    provider: str,
    model: str,
    context: Optional[str],
    instructions: Optional[str],
    preprocess_repetitions: bool,
    translate_notes: bool,
    glossary: Optional[Dict[str, str]],
    filename_settings: FilenameSettings,
    text_fit_mode: TextFitMode = TextFitMode.NONE,
    min_font_ratio: int = 80,
    length_limit: Optional[int] = None,
) -> None:
    """Run translation job in background, respecting concurrency semaphore."""
    job_manager = get_job_manager()
    settings = get_settings()

    # Wait for a concurrency slot before actually starting work.
    # The job is in RUNNING state, but it waits here until a slot is free.
    async with job_manager.running_semaphore:
        try:
            output_path = work_dir / "translated.pptx"
            with ppt_path.open("rb") as ppt_file_obj:
                request = TranslationRequest(
                    ppt_file=ppt_file_obj,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    provider=provider,
                    model=model,
                    context=context,
                    instructions=instructions,
                    glossary=glossary,
                    preprocess_repetitions=preprocess_repetitions,
                    translate_notes=translate_notes,
                    text_fit_mode=text_fit_mode,
                    min_font_ratio=min_font_ratio,
                    length_limit=length_limit,
                    output_path=output_path,
                )

                progress_callback = _create_progress_callback(job_id)
                service = TranslationService(settings=settings, progress_callback=progress_callback)

                result = await service.translate_async(request)

            if not result.success:
                await job_manager.fail_job(job_id, result.error_message or "Translation failed")
                return

            if result.output_path is None and result.output_file is None:
                await job_manager.fail_job(job_id, "Translation succeeded but no output file was generated")
                return

            # Generate output filename using settings
            download_name = generate_output_filename(
                filename_settings=filename_settings,
                original_filename=filename,
                target_language=result.target_language_used,
                model=model,
            )

            await job_manager.complete_job(
                job_id,
                result=result,
                output_file=result.output_file,
                output_path=result.output_path,
                output_filename=download_name,
            )

        except asyncio.CancelledError:
            LOGGER.info("Translation job %s was cancelled", job_id)
            raise
        except Exception as exc:
            LOGGER.exception("Translation job %s failed: %s", job_id, exc)
            await job_manager.fail_job(job_id, "번역 중 오류가 발생했습니다.")


@app.post("/api/v1/jobs", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    ppt_file: UploadFile = File(..., description="PPTX or PPT file to translate"),
    glossary_file: Optional[UploadFile] = File(None, description="Optional Excel glossary file"),
    source_lang: str = Form("Auto", description="Source language"),
    target_lang: str = Form("Auto", description="Target language"),
    provider: str = Form("anthropic", description="LLM provider"),
    model: str = Form(DEFAULT_TRANSLATION_MODEL, description="Model to use"),
    context: Optional[str] = Form(None, description="Background information about the presentation"),
    instructions: Optional[str] = Form(None, description="Translation style/tone guidelines"),
    preprocess_repetitions: bool = Form(False, description="Deduplicate repeated phrases"),
    translate_notes: bool = Form(False, description="Also translate speaker notes"),
    filename_settings: Optional[str] = Form(None, description="Filename settings as JSON"),
    text_fit_mode: str = Form("none", description="Text fitting mode: none, auto_shrink, expand_box"),
    min_font_ratio: int = Form(80, description="Minimum font size ratio (50-100) for auto_shrink mode"),
    compress_images: str = Form("none", description="Image compression preset: none, high, medium, low"),
    length_limit: Optional[int] = Form(None, description="Translation length limit as percentage of original (110, 130, 150)"),
) -> JobCreateResponse:
    """Create a new translation job."""
    settings = get_settings()
    job_manager = get_job_manager()
    job: Optional[Job] = None
    try:
        # Validate provider
        if provider not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {provider}. Must be one of: {list(SUPPORTED_MODELS.keys())}",
            )
        validate_model(provider, model)

        # Validate API key
        if provider == "openai" and not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
        if provider == "anthropic" and not settings.anthropic_api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

        # Validate file
        if not ppt_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        filename_lower = ppt_file.filename.lower()
        if not (filename_lower.endswith(".pptx") or filename_lower.endswith(".ppt")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .ppt and .pptx files are supported.",
            )

        # Reserve a job slot before the first await (file read) to avoid
        # over-admission when many requests arrive at the same time.
        max_allowed = settings.max_running_jobs + settings.max_queued_jobs
        job = job_manager.try_create_job(JobType.TRANSLATION, max_active=max_allowed)
        if job is None:
            raise HTTPException(
                status_code=429,
                detail="서버가 바쁩니다. 잠시 후 다시 시도해주세요.",
            )

        work_dir = Path(tempfile.mkdtemp(prefix=f"ppt-translator-{job.id}-"))
        job.work_dir = work_dir

        # Stream upload to disk so large PPT files are not retained in memory
        # for the full queued/running job lifetime.
        try:
            suffix = ".pptx" if filename_lower.endswith(".pptx") else ".ppt"
            ppt_path, written_bytes = await _save_upload_to_temp_path(
                ppt_file,
                work_dir,
                suffix=suffix,
                max_size_mb=settings.max_upload_size_mb,
            )
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("Failed to read uploaded file: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to read uploaded file")

        size_mb = written_bytes / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({size_mb:.1f}MB) exceeds limit ({settings.max_upload_size_mb}MB)",
            )

        _validate_ppt_path(ppt_path)

        # Compress images if requested
        if compress_images and compress_images != "none":
            from src.utils.image_compressor import compress_pptx_images_to_file

            try:
                compressed_path = work_dir / "compressed.pptx"
                with ppt_path.open("rb") as source_file:
                    compressed_result = compress_pptx_images_to_file(
                        source_file,
                        preset=compress_images,
                        output_path=compressed_path,
                    )
                if isinstance(compressed_result, Path):
                    ppt_path = compressed_result
            except Exception as exc:
                LOGGER.warning("Image compression failed, using original: %s", exc)

        # Load glossary if provided
        glossary = None
        if glossary_file and glossary_file.filename:
            try:
                glossary_content = await glossary_file.read()
                glossary_buffer = io.BytesIO(glossary_content)
                glossary_loader = GlossaryLoader()
                glossary = glossary_loader.load_glossary(glossary_buffer)
                LOGGER.info("Loaded glossary with %d terms", len(glossary))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Glossary error: {str(exc)}")
            except Exception as exc:
                LOGGER.exception("Failed to load glossary: %s", exc)
                raise HTTPException(status_code=400, detail="Failed to load glossary file")

        # Parse filename settings
        parsed_filename_settings = FilenameSettings()
        if filename_settings:
            try:
                settings_data = json.loads(filename_settings)
                parsed_filename_settings = FilenameSettings(**settings_data)
            except (json.JSONDecodeError, ValueError) as exc:
                LOGGER.warning("Invalid filename_settings JSON, using defaults: %s", exc)

        # Parse text fit mode
        try:
            parsed_text_fit_mode = TextFitMode(text_fit_mode)
        except ValueError:
            LOGGER.warning("Invalid text_fit_mode '%s', using default 'none'", text_fit_mode)
            parsed_text_fit_mode = TextFitMode.NONE

        # Clamp min_font_ratio
        clamped_min_font_ratio = max(50, min(100, min_font_ratio))

        # Validate length_limit
        parsed_length_limit: Optional[int] = None
        if length_limit is not None:
            if length_limit in (110, 130, 150):
                parsed_length_limit = length_limit
            else:
                LOGGER.warning("Invalid length_limit '%s', ignoring", length_limit)

        # Start background task
        task = asyncio.create_task(
            _run_translation_job(
                job_id=job.id,
                ppt_path=ppt_path,
                work_dir=work_dir,
                filename=ppt_file.filename,
                source_lang=source_lang,
                target_lang=target_lang,
                provider=provider,
                model=model,
                context=context,
                instructions=instructions,
                preprocess_repetitions=preprocess_repetitions,
                translate_notes=translate_notes,
                glossary=glossary,
                filename_settings=parsed_filename_settings,
                text_fit_mode=parsed_text_fit_mode,
                min_font_ratio=clamped_min_font_ratio,
                length_limit=parsed_length_limit,
            )
        )
        job_manager.start_job(job.id, task)

        LOGGER.info(
            "Created translation job %s: file=%s, provider=%s, model=%s",
            job.id,
            ppt_file.filename,
            provider,
            model,
        )

        return JobCreateResponse(job_id=job.id, status=job.state.value)
    except HTTPException:
        if job is not None:
            await job_manager.delete_job(job.id)
        raise
    except Exception as exc:
        if job is not None:
            LOGGER.exception("Failed to create translation job %s: %s", job.id, exc)
            await job_manager.delete_job(job.id)
        else:
            LOGGER.exception("Failed to create translation job: %s", exc)
        raise HTTPException(status_code=500, detail="번역 작업 생성 중 오류가 발생했습니다.")


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get job status."""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    progress_dict = None
    if job.progress:
        progress_dict = {
            "status": job.progress.status.value,
            "current_batch": job.progress.current_batch,
            "total_batches": job.progress.total_batches,
            "current_sentence": job.progress.current_sentence,
            "total_sentences": job.progress.total_sentences,
            "percent": job.progress.percent,
            "message": job.progress.message,
        }

    return JobStatusResponse(
        job_id=job.id,
        job_type=job.job_type.value,
        state=job.state.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=progress_dict,
        error_message=job.error_message,
    )


@app.get("/api/v1/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Stream job events via SSE."""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        async for event in job_manager.stream_events(job_id):
            data = json.dumps(
                {
                    "type": event.event_type,
                    "data": event.data,
                    "timestamp": event.timestamp,
                }
            )
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/jobs/{job_id}/result")
async def download_job_result(job_id: str) -> Response:
    """Download completed job result."""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current state: {job.state.value}",
        )

    if job.output_path is None and job.output_file is None:
        raise HTTPException(status_code=500, detail="No output file available")

    filename = job.output_filename or "translated.pptx"
    encoded_filename = quote(filename, safe="")

    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
    }

    if job.result:
        headers.update(
            {
                "X-Translation-Source-Lang": quote(job.result.source_language_detected, safe=""),
                "X-Translation-Target-Lang": quote(job.result.target_language_used, safe=""),
                "X-Translation-Total-Paragraphs": str(job.result.total_paragraphs),
                "X-Translation-Unique-Paragraphs": str(job.result.unique_paragraphs),
                "X-Translation-Batch-Count": str(job.result.batch_count),
                "X-Translation-Elapsed-Seconds": f"{job.result.elapsed_seconds:.2f}",
            }
        )

    if job.output_path is not None:
        if not job.output_path.exists():
            raise HTTPException(status_code=500, detail="Output file is no longer available")
        return FileResponse(
            path=job.output_path,
            media_type=PPTX_MEDIA_TYPE,
            headers=headers,
        )

    job.output_file.seek(0)
    content = job.output_file.read()
    return Response(
        content=content,
        media_type=PPTX_MEDIA_TYPE,
        headers=headers,
    )


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel/delete a job."""
    job_manager = get_job_manager()
    success = await job_manager.delete_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"status": "cancelled", "job_id": job_id}


# ============================================================================
# Text Extraction Endpoint
# ============================================================================


@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_text(
    ppt_file: UploadFile = File(..., description="PPTX file to extract text from"),
    figures: Literal["omit", "placeholder"] = Form("omit", description="How to handle figures"),
    charts: Literal["labels", "placeholder", "omit"] = Form("labels", description="How to handle charts"),
    with_notes: bool = Form(False, description="Include speaker notes"),
    table_header: bool = Form(True, description="Treat first row as table header"),
) -> ExtractionResponse:
    """Extract text from PPT as markdown."""
    settings = get_settings()

    # Validate file
    if not ppt_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = ppt_file.filename.lower()
    if not filename_lower.endswith(".pptx"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .pptx files are supported for extraction.",
        )

    # Read file content
    try:
        file_content = await ppt_file.read()
    except Exception as exc:
        LOGGER.exception("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    # Validate file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({size_mb:.1f}MB) exceeds limit ({settings.max_upload_size_mb}MB)",
        )

    # Validate file signature
    ppt_buffer = io.BytesIO(file_content)
    is_valid, error_msg = validate_pptx_file(ppt_buffer)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg or "Invalid PPTX file format")
    ppt_buffer.seek(0)

    # Extract text
    try:
        options = ExtractionOptions(
            figures=figures,
            charts=charts,
            table_header=table_header,
            with_notes=with_notes,
        )
        docs = extract_pptx_to_docs(ppt_buffer, options)
        markdown = docs_to_markdown(docs, options)

        LOGGER.info("Extracted text from %s: %d slides", ppt_file.filename, len(docs))

        return ExtractionResponse(markdown=markdown, slide_count=len(docs))

    except Exception as exc:
        LOGGER.exception("Failed to extract text: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to extract text")


# ============================================================================
# Summarization Endpoint
# ============================================================================


@app.post("/api/v1/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize presentation content for translation context.

    Takes extracted markdown text and generates a concise summary
    that helps maintain consistency during translation.
    """
    from src.chains.summarization_chain import summarize_presentation

    settings = get_settings()

    # Validate provider
    if request.provider not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {request.provider}. Must be one of: {list(SUPPORTED_MODELS.keys())}",
        )

    # Validate API key
    if request.provider == "openai" and not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    if request.provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")
    validate_model(request.provider, request.model)

    # Validate markdown
    if not request.markdown or not request.markdown.strip():
        raise HTTPException(status_code=400, detail="Markdown content is empty")

    try:
        summary = await summarize_presentation(
            markdown=request.markdown,
            provider=request.provider,
            model=request.model,
        )
        LOGGER.info(
            "Generated summary: provider=%s, model=%s, input_chars=%d, output_chars=%d",
            request.provider,
            request.model,
            len(request.markdown),
            len(summary),
        )
        return SummarizeResponse(summary=summary)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("Failed to summarize text: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to summarize text")


# ============================================================================
# Instructions Generation Endpoint
# ============================================================================


class GenerateInstructionsRequest(BaseModel):
    """Request for generating translation instructions."""

    target_lang: str
    markdown: str
    provider: str = "anthropic"
    model: str = DEFAULT_LIGHT_MODEL["anthropic"]


class GenerateInstructionsResponse(BaseModel):
    """Response with generated translation instructions."""

    instructions: str


def _build_markdown_preview(markdown: str, max_chars: int = 6000) -> str:
    """Keep both the beginning and end of long extracted markdown."""
    if len(markdown) <= max_chars:
        return markdown

    head_chars = int(max_chars * 0.65)
    tail_chars = max_chars - head_chars
    return (
        markdown[:head_chars].rstrip()
        + "\n\n...[middle content omitted for instruction generation]...\n\n"
        + markdown[-tail_chars:].lstrip()
    )


@app.post("/api/v1/generate-instructions", response_model=GenerateInstructionsResponse)
async def generate_instructions(request: GenerateInstructionsRequest) -> GenerateInstructionsResponse:
    """Generate translation instructions based on target language.

    Creates style/tone guidelines appropriate for the target language and culture.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from src.chains.llm_factory import create_llm

    settings = get_settings()

    # Validate provider
    if request.provider not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {request.provider}. Must be one of: {list(SUPPORTED_MODELS.keys())}",
        )

    # Validate API key
    if request.provider == "openai" and not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    if request.provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")
    validate_model(request.provider, request.model)

    # Validate target language
    if not request.target_lang or request.target_lang == "Auto":
        raise HTTPException(status_code=400, detail="Target language must be specified")

    try:
        api_key = settings.openai_api_key if request.provider == "openai" else settings.anthropic_api_key
        llm = create_llm(
            provider=request.provider,
            model_name=request.model,
            api_key=api_key,
            temperature=0.2,
            max_tokens=700,
        )

        markdown_preview = _build_markdown_preview(request.markdown)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """번역 지침 설계자로서 PPT 번역에 바로 사용할 지침을 **500자 이내**로 생성하세요.

**출력 규칙:**
- 반드시 한국어로 4-6개 bullet만 출력하세요.
- 실제 번역을 하지 말고, 번역자가 따라야 할 정책만 쓰세요.
- 문서 내용 요약이 아니라 톤, 용어, 보존 규칙, 문장 스타일 중심으로 쓰세요.
- 근거가 부족한 고유명사는 임의 번역 지시하지 말고 원문 유지로 안내하세요.
- 숫자, 단위, 날짜, 버전, URL, 코드, 변수, 괄호 안 토큰 보존 규칙을 포함하세요.
- 슬라이드 제목/불릿/표 셀은 간결한 PPT 문장으로 번역하도록 안내하세요.

**예시:**
- 격식체, 전문적 톤 유지
- Binary Spot, Heist Royale 등 게임 모드/이벤트명은 원문 유지
- 숫자, %, 날짜, 버전, URL, 괄호 안 토큰은 원문 형식 보존
- 제목/불릿/표 셀은 짧고 명확한 문장으로 정리"""),
            ("user", """타겟 언어: {target_lang}

문서 내용:
{markdown}

번역 지침:"""),
        ])

        # Generate instructions
        chain = prompt | llm
        result = await chain.ainvoke({"target_lang": request.target_lang, "markdown": markdown_preview})

        instructions = result.content if hasattr(result, "content") else str(result)
        instructions = instructions.strip()

        LOGGER.info(
            "Generated instructions: target_lang=%s, provider=%s, model=%s, result_length=%d, result=%s",
            request.target_lang,
            request.provider,
            request.model,
            len(instructions),
            instructions[:200] if instructions else "(empty)",
        )

        # Fallback if empty
        if not instructions:
            LOGGER.warning("Empty instructions generated, using default")
            instructions = f"- {request.target_lang}에 적합한 자연스러운 표현 사용\n- 전문 용어는 문맥에 맞게 번역\n- 명확하고 간결한 문장 유지"

        return GenerateInstructionsResponse(instructions=instructions)

    except Exception as exc:
        LOGGER.exception("Failed to generate instructions: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate instructions")


# ============================================================================
# Legacy Sync Translate Endpoint (kept for backwards compatibility)
# ============================================================================


@app.post("/translate")
async def translate_ppt(
    ppt_file: UploadFile = File(..., description="PPTX or PPT file to translate"),
    glossary_file: Optional[UploadFile] = File(
        None, description="Optional Excel glossary file"
    ),
    source_lang: str = Form("Auto", description="Source language (Auto for detection)"),
    target_lang: str = Form("Auto", description="Target language (Auto for inference)"),
    provider: str = Form("anthropic", description="LLM provider (openai or anthropic)"),
    model: str = Form(DEFAULT_TRANSLATION_MODEL, description="Model to use (depends on provider)"),
    context: Optional[str] = Form(None, description="Background information about the presentation"),
    instructions: Optional[str] = Form(None, description="Translation style/tone guidelines"),
    preprocess_repetitions: bool = Form(
        False, description="Deduplicate repeated phrases"
    ),
    translate_notes: bool = Form(
        False, description="Also translate speaker notes"
    ),
    compress_images: str = Form("none", description="Image compression preset: none, high, medium, low"),
) -> Response:
    """Translate a PowerPoint presentation (synchronous).

    Returns the translated PPTX file as a binary response with metadata in headers.
    """
    settings = get_settings()

    # Validate provider
    if provider not in ("openai", "anthropic"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {provider}. Must be 'openai' or 'anthropic'.",
        )
    validate_model(provider, model)

    # Validate API key for selected provider
    if provider == "openai" and not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server",
        )
    if provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not configured on the server",
        )

    # Validate file type
    if not ppt_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename_lower = ppt_file.filename.lower()
    if not (filename_lower.endswith(".pptx") or filename_lower.endswith(".ppt")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .ppt and .pptx files are supported.",
        )

    # Read file content
    try:
        file_content = await ppt_file.read()
    except Exception as exc:
        LOGGER.exception("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    # Validate file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({size_mb:.1f}MB) exceeds limit ({settings.max_upload_size_mb}MB)",
        )

    # Validate file signature. BytesIO copies the bytes, so the original
    # buffer can be released immediately to avoid holding two copies.
    ppt_buffer = io.BytesIO(file_content)
    del file_content
    is_valid, error_msg = validate_pptx_file(ppt_buffer)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_msg or "Invalid PPT/PPTX file format",
        )
    ppt_buffer.seek(0)

    # Compress images if requested
    if compress_images and compress_images != "none":
        from src.utils.image_compressor import compress_pptx_images
        try:
            ppt_buffer = compress_pptx_images(ppt_buffer, preset=compress_images)
            ppt_buffer.seek(0)
        except Exception as exc:
            LOGGER.warning("Image compression failed, using original: %s", exc)
            ppt_buffer.seek(0)

    # Load glossary if provided
    glossary = None
    if glossary_file and glossary_file.filename:
        try:
            glossary_content = await glossary_file.read()
            glossary_buffer = io.BytesIO(glossary_content)
            glossary_loader = GlossaryLoader()
            glossary = glossary_loader.load_glossary(glossary_buffer)
            LOGGER.info("Loaded glossary with %d terms", len(glossary))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Glossary error: {str(exc)}")
        except Exception as exc:
            LOGGER.exception("Failed to load glossary: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to load glossary file")

    # Create translation request
    request = TranslationRequest(
        ppt_file=ppt_buffer,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider,
        model=model,
        context=context,
        instructions=instructions,
        glossary=glossary,
        preprocess_repetitions=preprocess_repetitions,
        translate_notes=translate_notes,
    )

    # Execute translation
    LOGGER.info(
        "Starting translation: file=%s, source=%s, target=%s, provider=%s, model=%s",
        ppt_file.filename,
        source_lang,
        target_lang,
        provider,
        model,
    )

    service = TranslationService(settings=settings)
    result = await service.translate_async(request)

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=result.error_message or "Translation failed",
        )

    if result.output_file is None:
        raise HTTPException(
            status_code=500,
            detail="Translation succeeded but no output file was generated",
        )

    # Prepare response
    original_name = Path(ppt_file.filename).stem
    safe_name = sanitize_filename(original_name, fallback="presentation")
    safe_model = sanitize_filename(model, fallback="model")
    safe_target = sanitize_filename(result.target_language_used, fallback="target")
    timestamp = datetime.now().strftime("%Y%m%d")
    download_name = f"{safe_target}_{safe_name}_{safe_model}_{timestamp}.pptx"

    LOGGER.info(
        "Translation completed: %d paragraphs, %.1f seconds",
        result.total_paragraphs,
        result.elapsed_seconds,
    )

    # URL-encode non-ASCII characters for HTTP headers
    encoded_download_name = quote(download_name, safe="")
    encoded_source_lang = quote(result.source_language_detected, safe="")
    encoded_target_lang = quote(result.target_language_used, safe="")

    return Response(
        content=result.output_file.getvalue(),
        media_type=PPTX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_download_name}",
            "X-Translation-Source-Lang": encoded_source_lang,
            "X-Translation-Target-Lang": encoded_target_lang,
            "X-Translation-Total-Paragraphs": str(result.total_paragraphs),
            "X-Translation-Unique-Paragraphs": str(result.unique_paragraphs),
            "X-Translation-Batch-Count": str(result.batch_count),
            "X-Translation-Elapsed-Seconds": f"{result.elapsed_seconds:.2f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
