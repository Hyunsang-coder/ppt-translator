"""FastAPI server for PPT translation API."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import io
import json
import logging
import os

try:
    import resource  # Unix-only stdlib module; absent on Windows.
except ModuleNotFoundError:  # pragma: no cover - Windows has no `resource`
    resource = None
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.chains.translation_chain import TranslationCancelled
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
from src.services.quality_records import QualityRecorder
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.rules_loader import RulesLoader
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


class FragmentFinding(BaseModel):
    """A detection badge attached to a fragment (WP-C5)."""

    type: str
    severity: str
    description: str
    suggested_fix: Optional[str] = None
    related_location: Optional[Dict[str, Any]] = None


class StyleSegment(BaseModel):
    """One target span carrying a source run style."""

    text: str
    group_index: int
    color: Optional[str] = None
    scheme: Optional[str] = None
    bold: bool = False
    italic: bool = False


class FragmentItem(BaseModel):
    """One reviewable fragment (source/target + badges)."""

    index: int
    slide: int
    shape: int
    paragraph: int
    slide_title: Optional[str] = None
    is_note: bool
    source: str
    target: str
    repeat_count: int
    length_budget: Optional[int] = None
    findings: List[FragmentFinding] = []
    edited: bool = False
    style_segments: List[StyleSegment] = []
    style_status: Literal["single_style", "preserved", "partial", "dropped"] = "single_style"


class FragmentsResponse(BaseModel):
    """Review-screen fragment list for a completed job."""

    job_id: str
    total: int
    fragments: List[FragmentItem]
    revision: int = 0
    committed_revision: int = 0
    dirty: bool = False


class FragmentEditRequest(BaseModel):
    """Edit or re-translate a single fragment (WP-C5)."""

    action: Literal["edit", "retranslate", "ignore"] = "edit"
    # edit: the new target text (direct inline edit).
    target: Optional[str] = None
    # retranslate: free-form instruction (e.g. "더 짧게", "용어 X 사용").
    instruction: Optional[str] = None
    # Propagate the change to fragments with an identical source.
    propagate_identical: bool = False
    # For ignore: the finding type being dismissed (for the rejected record).
    finding_type: Optional[str] = None


class FragmentEditResponse(BaseModel):
    """Result of a fragment edit/re-translate."""

    index: int
    target: str
    changed_indices: List[int]
    partial_candidates: List[Dict[str, Any]] = []
    revision: int = 0


class FragmentProposalRequest(BaseModel):
    """Generate a direct-edit or retranslation candidate without applying it."""

    action: Literal["edit", "retranslate"]
    target: Optional[str] = None
    instruction: Optional[str] = None
    propagate_identical: bool = False


class FragmentProposalResponse(BaseModel):
    proposal_id: str
    index: int
    base_revision: int
    old_target: str
    target: str
    changed_indices: List[int]
    style_segments: List[StyleSegment]
    style_status: str
    partial_candidates: List[Dict[str, Any]] = []
    over_budget: bool = False


class ApplyProposalRequest(BaseModel):
    expected_revision: int


class ApplyProposalResponse(BaseModel):
    index: int
    target: str
    changed_indices: List[int]
    partial_candidates: List[Dict[str, Any]] = []
    revision: int
    dirty: bool


class PartialApplyRequest(BaseModel):
    indices: List[int]
    old_phrase: str
    new_phrase: str
    expected_revision: int


class ReviewRevisionRequest(BaseModel):
    expected_revision: int


class ReviewMutationResponse(BaseModel):
    changed_indices: List[int] = []
    revision: int
    committed_revision: int
    dirty: bool
    findings_count: int = 0


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
    """Return current process RSS memory usage in MB (Linux/macOS; 0 on Windows)."""
    if resource is None:
        return 0.0
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
        default_model=DEFAULT_TRANSLATION_MODEL,
    )


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
    ppt_buffer: io.BytesIO,
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
    team_rules: Optional[Dict] = None,
) -> None:
    """Run translation job in background, respecting concurrency semaphore."""
    job_manager = get_job_manager()
    settings = get_settings()

    # Wait for a concurrency slot before actually starting work.
    # The job is in RUNNING state, but it waits here until a slot is free.
    async with job_manager.running_semaphore:
        # C-1: pass the job's cancel flag into the request so a DELETE can stop
        # the worker thread's LLM calls at the next batch boundary. A slot may
        # have been cancelled while queued; bail before doing any work.
        job = job_manager.get_job(job_id)
        cancel_event = job.cancel_event if job is not None else None
        if cancel_event is not None and cancel_event.is_set():
            LOGGER.info("Translation job %s cancelled before start", job_id)
            return
        try:
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
                text_fit_mode=text_fit_mode,
                min_font_ratio=min_font_ratio,
                length_limit=length_limit,
                team_rules=team_rules,
                cancel_event=cancel_event,
            )

            progress_callback = _create_progress_callback(job_id)
            service = TranslationService(settings=settings, progress_callback=progress_callback)

            # Run translation in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, service.translate, request)

            if not result.success:
                await job_manager.fail_job(job_id, result.error_message or "Translation failed")
                return

            if result.output_file is None:
                await job_manager.fail_job(job_id, "Translation succeeded but no output file was generated")
                return

            # Quality ledger (WP-C2): one run row + one record per sweep finding.
            # Best-effort — a ledger failure must never fail the translation.
            try:
                recorder = QualityRecorder(quality_dir=settings.quality_dir)
                doc_ref = f"deck:{filename}"
                recorder.record_run(
                    job_id=job_id,
                    model=model,
                    source_lang=result.source_language_detected,
                    target_lang=result.target_language_used,
                    doc_words=result.total_paragraphs,
                    findings=result.findings,
                )
                recorder.record_findings(
                    result.findings,
                    job_id=job_id,
                    doc_ref=doc_ref,
                    source_lang=result.source_language_detected,
                    target_lang=result.target_language_used,
                    model=model,
                )
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Quality ledger recording failed for job %s", job_id)

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
                output_filename=download_name,
            )

        except asyncio.CancelledError:
            LOGGER.info("Translation job %s was cancelled", job_id)
        except TranslationCancelled:
            # C-1: cooperative cancel from the worker thread. delete_job already
            # set the job CANCELLED; don't mark it failed.
            LOGGER.info("Translation job %s stopped by cancellation flag", job_id)
        except Exception as exc:
            LOGGER.exception("Translation job %s failed: %s", job_id, exc)
            await job_manager.fail_job(job_id, "번역 중 오류가 발생했습니다.")


@app.post("/api/v1/jobs", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    ppt_file: UploadFile = File(..., description="PPTX or PPT file to translate"),
    glossary_file: Optional[UploadFile] = File(None, description="Optional Excel glossary file"),
    rules_file: Optional[UploadFile] = File(None, description="Optional team translation-rules JSON"),
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
        # buffer can be released immediately to avoid holding two copies for
        # the entire job duration (memory pressure in the local sidecar/server).
        ppt_buffer = io.BytesIO(file_content)
        del file_content
        is_valid, error_msg = validate_pptx_file(ppt_buffer)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg or "Invalid PPT/PPTX file format")
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

        # Load team translation rules if provided (WP-C1). No file -> feature
        # off, existing behavior unchanged. A malformed file surfaces as a 400
        # rather than silently degrading translation quality.
        team_rules = None
        if rules_file and rules_file.filename:
            try:
                rules_content = await rules_file.read()
                rules_buffer = io.BytesIO(rules_content)
                team_rules = RulesLoader().load_rules(rules_buffer)
                LOGGER.info("Loaded team translation rules from %s", rules_file.filename)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Rules error: {str(exc)}")
            except Exception as exc:
                LOGGER.exception("Failed to load rules file: %s", exc)
                raise HTTPException(status_code=400, detail="Failed to load rules file")

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
                ppt_buffer=ppt_buffer,
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
                team_rules=team_rules,
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

    if job.output_file is None:
        raise HTTPException(status_code=500, detail="No output file available")

    job.output_file.seek(0)
    content = job.output_file.read()

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

    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers=headers,
    )


@app.get("/api/v1/jobs/{job_id}/fragments", response_model=FragmentsResponse)
async def get_job_fragments(job_id: str) -> FragmentsResponse:
    """List reviewable fragments (source/target + detection badges) for a job."""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current state: {job.state.value}",
        )
    if job.review_session is None:
        raise HTTPException(status_code=404, detail="No review session for this job")

    session = job.review_session
    items: List[FragmentItem] = []
    for view in session.fragments():
        items.append(
            FragmentItem(
                index=view.index,
                slide=view.slide,
                shape=view.shape,
                paragraph=view.paragraph,
                slide_title=view.slide_title,
                is_note=view.is_note,
                source=view.source,
                target=view.target,
                repeat_count=view.repeat_count,
                length_budget=view.length_budget,
                findings=[FragmentFinding(**f) for f in view.findings],
                edited=view.edited,
                style_segments=[StyleSegment(**segment) for segment in view.style_segments],
                style_status=view.style_status,
            )
        )
    return FragmentsResponse(
        job_id=job_id,
        total=len(items),
        fragments=items,
        revision=session.revision,
        committed_revision=session.committed_revision,
        dirty=session.dirty,
    )


@app.post("/api/v1/jobs/{job_id}/fragments/{index}", response_model=FragmentEditResponse)
async def edit_job_fragment(
    job_id: str, index: int, body: FragmentEditRequest
) -> FragmentEditResponse:
    """Edit, re-translate, or ignore a single fragment (WP-C5).

    Compatibility endpoint that stages an edit in the review draft. New clients
    should use the proposal endpoints so the user can compare before applying.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != JobState.COMPLETED or job.review_session is None:
        raise HTTPException(status_code=400, detail="Job has no editable review session")

    session = job.review_session
    if not (0 <= index < len(session.paragraphs)):
        raise HTTPException(status_code=400, detail="Fragment index out of range")

    settings = get_settings()
    recorder = QualityRecorder(quality_dir=settings.quality_dir)
    doc_ref = f"deck:{job.output_filename or 'deck.pptx'}"
    original_target = session.translated_texts[index]
    source_text = session.paragraphs[index].original_text or ""

    # --- ignore: dismiss a finding, record as rejected -------------------
    if body.action == "ignore":
        session.dismiss_finding(index, body.finding_type or "")
        _record_edit(
            recorder, session, index, source_text, original_target,
            corrected=None, disposition="rejected", doc_ref=doc_ref,
            finding_type=body.finding_type,
        )
        return FragmentEditResponse(
            index=index, target=original_target, changed_indices=[],
            revision=session.revision,
        )

    # Compatibility path: create and immediately stage a server-side proposal.
    # The published output remains unchanged until /review/commit.
    async with job.review_lock:
        try:
            loop = asyncio.get_running_loop()
            proposal = await loop.run_in_executor(
                None,
                functools.partial(
                    session.create_proposal,
                    index,
                    action=body.action,
                    target=body.target,
                    instruction=body.instruction,
                    propagate_identical=body.propagate_identical,
                    model=session.model or DEFAULT_TRANSLATION_MODEL,
                    provider=session.provider,
                ),
            )
            session.apply_proposal(proposal.id, session.revision)
            session.run_final_sweep()
        except Exception as exc:
            LOGGER.exception("Failed to stage review edit: %s", exc)
            raise HTTPException(status_code=500, detail="수정 준비에 실패했습니다.")

        new_target = proposal.target
        changed = proposal.changed_indices

    # Record the accepted edit with the corrected triplet.
    _record_edit(
        recorder, session, index, source_text, original_target,
        corrected=new_target, disposition="accepted", doc_ref=doc_ref,
        propagated=len(changed) - 1,
    )

    partial = proposal.partial_candidates

    return FragmentEditResponse(
        index=index,
        target=new_target,
        changed_indices=changed,
        partial_candidates=partial,
        revision=session.revision,
    )


@app.post(
    "/api/v1/jobs/{job_id}/fragments/{index}/proposals",
    response_model=FragmentProposalResponse,
)
async def propose_job_fragment_edit(
    job_id: str, index: int, body: FragmentProposalRequest
) -> FragmentProposalResponse:
    """Generate a review candidate without mutating the draft."""
    job = get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state != JobState.COMPLETED or job.review_session is None:
        raise HTTPException(status_code=400, detail="Job has no editable review session")
    session = job.review_session
    if not (0 <= index < len(session.paragraphs)):
        raise HTTPException(status_code=400, detail="Fragment index out of range")
    if body.action == "edit" and body.target is None:
        raise HTTPException(status_code=400, detail="edit proposal requires target")

    async with job.review_lock:
        try:
            loop = asyncio.get_running_loop()
            proposal = await loop.run_in_executor(
                None,
                functools.partial(
                    session.create_proposal,
                    index,
                    action=body.action,
                    target=body.target,
                    instruction=body.instruction,
                    propagate_identical=body.propagate_identical,
                    model=session.model or DEFAULT_TRANSLATION_MODEL,
                    provider=session.provider,
                ),
            )
        except Exception as exc:
            LOGGER.exception("Fragment proposal failed: %s", exc)
            raise HTTPException(status_code=500, detail="수정 후보 생성에 실패했습니다.")

    return FragmentProposalResponse(
        proposal_id=proposal.id,
        index=proposal.index,
        base_revision=proposal.base_revision,
        old_target=proposal.old_target,
        target=proposal.target,
        changed_indices=proposal.changed_indices,
        style_segments=[StyleSegment(**segment) for segment in proposal.style_segments],
        style_status=proposal.style_status,
        partial_candidates=proposal.partial_candidates,
        over_budget=proposal.over_budget,
    )


@app.post(
    "/api/v1/jobs/{job_id}/proposals/{proposal_id}/apply",
    response_model=ApplyProposalResponse,
)
async def apply_job_fragment_proposal(
    job_id: str, proposal_id: str, body: ApplyProposalRequest
) -> ApplyProposalResponse:
    """Stage a previously previewed candidate using optimistic revision checks."""
    job = get_job_manager().get_job(job_id)
    if job is None or job.review_session is None:
        raise HTTPException(status_code=404, detail="Review session not found")
    session = job.review_session
    async with job.review_lock:
        try:
            proposal = session.apply_proposal(proposal_id, body.expected_revision)
            session.run_final_sweep()
        except KeyError:
            raise HTTPException(status_code=404, detail="Proposal not found")
        except RuntimeError:
            raise HTTPException(status_code=409, detail="검토 내용이 변경되었습니다. 다시 확인해주세요.")

    index = proposal.index
    _record_edit(
        QualityRecorder(quality_dir=get_settings().quality_dir),
        session,
        index,
        session.paragraphs[index].original_text or "",
        proposal.old_target,
        corrected=proposal.target,
        disposition="accepted",
        doc_ref=f"deck:{job.output_filename or 'deck.pptx'}",
        propagated=max(0, len(proposal.changed_indices) - 1),
    )
    return ApplyProposalResponse(
        index=index,
        target=proposal.target,
        changed_indices=proposal.changed_indices,
        partial_candidates=proposal.partial_candidates,
        revision=session.revision,
        dirty=session.dirty,
    )


@app.post(
    "/api/v1/jobs/{job_id}/review/partial",
    response_model=ReviewMutationResponse,
)
async def apply_review_partial_candidates(
    job_id: str, body: PartialApplyRequest
) -> ReviewMutationResponse:
    """Apply a reviewed phrase replacement to selected candidate fragments."""
    job = get_job_manager().get_job(job_id)
    if job is None or job.review_session is None:
        raise HTTPException(status_code=404, detail="Review session not found")
    session = job.review_session
    async with job.review_lock:
        previous_targets = {
            index: session.translated_texts[index]
            for index in body.indices
            if 0 <= index < len(session.translated_texts)
        }
        try:
            loop = asyncio.get_running_loop()
            changed = await loop.run_in_executor(
                None,
                functools.partial(
                    session.apply_partial_candidates,
                    body.indices,
                    old_phrase=body.old_phrase,
                    new_phrase=body.new_phrase,
                    expected_revision=body.expected_revision,
                    model=session.model or DEFAULT_TRANSLATION_MODEL,
                    provider=session.provider,
                ),
            )
            await loop.run_in_executor(None, session.run_final_sweep)
        except RuntimeError:
            raise HTTPException(status_code=409, detail="검토 내용이 변경되었습니다. 다시 확인해주세요.")
    recorder = QualityRecorder(quality_dir=get_settings().quality_dir)
    for index in changed:
        _record_edit(
            recorder,
            session,
            index,
            session.paragraphs[index].original_text or "",
            previous_targets[index],
            corrected=session.translated_texts[index],
            disposition="accepted",
            doc_ref=f"deck:{job.output_filename or 'deck.pptx'}",
        )
    return ReviewMutationResponse(
        changed_indices=changed,
        revision=session.revision,
        committed_revision=session.committed_revision,
        dirty=session.dirty,
    )


@app.post(
    "/api/v1/jobs/{job_id}/review/undo",
    response_model=ReviewMutationResponse,
)
async def undo_review_change(
    job_id: str, body: ReviewRevisionRequest
) -> ReviewMutationResponse:
    job = get_job_manager().get_job(job_id)
    if job is None or job.review_session is None:
        raise HTTPException(status_code=404, detail="Review session not found")
    session = job.review_session
    async with job.review_lock:
        try:
            changed = session.undo(body.expected_revision)
            session.run_final_sweep()
        except RuntimeError:
            raise HTTPException(status_code=409, detail="검토 내용이 변경되었습니다. 다시 확인해주세요.")
    return ReviewMutationResponse(
        changed_indices=changed,
        revision=session.revision,
        committed_revision=session.committed_revision,
        dirty=session.dirty,
    )


@app.post(
    "/api/v1/jobs/{job_id}/review/commit",
    response_model=ReviewMutationResponse,
)
async def commit_review_draft(
    job_id: str, body: ReviewRevisionRequest
) -> ReviewMutationResponse:
    """Atomically render and publish the current review draft."""
    job = get_job_manager().get_job(job_id)
    if job is None or job.review_session is None:
        raise HTTPException(status_code=404, detail="Review session not found")
    session = job.review_session
    async with job.review_lock:
        if body.expected_revision != session.revision:
            raise HTTPException(status_code=409, detail="검토 내용이 변경되었습니다. 다시 확인해주세요.")

        def _render_and_check():
            buffer = session.render()
            findings = session.run_final_sweep()
            return buffer, findings

        try:
            loop = asyncio.get_running_loop()
            new_buffer, findings = await loop.run_in_executor(None, _render_and_check)
        except Exception as exc:
            LOGGER.exception("Review commit failed: %s", exc)
            raise HTTPException(status_code=500, detail="최종 반영에 실패했습니다. 기존 파일은 유지됩니다.")

        # Publish only after render + QA both succeeded.
        job.output_file = new_buffer
        session.mark_committed()

    return ReviewMutationResponse(
        revision=session.revision,
        committed_revision=session.committed_revision,
        dirty=session.dirty,
        findings_count=len(findings),
    )


def _record_edit(
    recorder,
    session,
    index: int,
    source: str,
    original_target: str,
    *,
    corrected: Optional[str],
    disposition: str,
    doc_ref: str,
    propagated: int = 0,
    finding_type: Optional[str] = None,
) -> None:
    """Record a review-loop edit/ignore as a quality record (best-effort)."""
    try:
        info = session.paragraphs[index]
        row = {
            "id": None,  # filled by recorder
            "source": source,
            "output": original_target,
            "corrected": corrected,
        }
        recorder.record_review_edit(
            job_id=doc_ref,  # doc_ref carries the deck; project id set inside
            doc_ref=doc_ref,
            source_lang=session.source_lang,
            target_lang=session.target_lang,
            model=session.model,
            location={
                "slide": info.slide_index + 1,
                "shape": info.shape_index,
                "paragraph": info.paragraph_index,
            },
            segment=row,
            disposition=disposition,
            finding_type=finding_type,
            propagated=propagated,
        )
    except Exception:  # pylint: disable=broad-except
        LOGGER.exception("Failed to record review edit for fragment %d", index)


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


@app.post("/api/v1/generate-instructions", response_model=GenerateInstructionsResponse)
async def generate_instructions(request: GenerateInstructionsRequest) -> GenerateInstructionsResponse:
    """Generate translation instructions based on target language.

    Creates style/tone guidelines appropriate for the target language and culture.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

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
        # Create LLM with max_tokens limit
        if request.provider == "openai":
            from src.chains.llm_factory import create_llm

            llm = create_llm(
                provider="openai",
                model_name=request.model,
                max_tokens=512,
                api_key=settings.openai_api_key,
            )
        else:
            llm = ChatAnthropic(
                model=request.model,
                api_key=settings.anthropic_api_key,
                temperature=0.7,
                max_tokens=512,  # Limit for concise output (~300 chars)
            )

        # Truncate markdown if too long (keep first ~2000 chars for context)
        markdown_preview = request.markdown[:2000] if len(request.markdown) > 2000 else request.markdown

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """번역 스타일 전문가로서 문서에 맞는 번역 지침을 **300자 이내**로 생성하세요.

**출력 형식 (한국어, 3-4개 bullet):**
- 톤/문체 (격식체/비격식체 등)
- 용어 처리 방침 (원문 유지할 용어, 번역할 용어)
- 문장 스타일 (간결하게, 자연스럽게 등)

**예시:**
- 격식체, 전문적 톤 유지
- 게임 용어(Binary Spot, Heist Royale) 원문 유지
- 간결한 문장, 명확한 표현 사용"""),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
