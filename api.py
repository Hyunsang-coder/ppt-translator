"""FastAPI server for PPT translation API."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from src.services import TranslationRequest, TranslationService
from src.utils.config import get_settings
from src.utils.glossary_loader import GlossaryLoader
from src.utils.security import sanitize_filename, validate_pptx_file

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="PPT 번역캣 API",
    description="PowerPoint translation API using OpenAI GPT models",
    version="2.2.0",
)

# For AWS Lambda deployment
try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(settings.openai_api_key),
    }


@app.post("/translate")
async def translate_ppt(
    ppt_file: UploadFile = File(..., description="PPTX or PPT file to translate"),
    glossary_file: Optional[UploadFile] = File(
        None, description="Optional Excel glossary file"
    ),
    source_lang: str = Form("Auto", description="Source language (Auto for detection)"),
    target_lang: str = Form("Auto", description="Target language (Auto for inference)"),
    model: str = Form("gpt-5.1", description="OpenAI model to use"),
    user_prompt: Optional[str] = Form(None, description="Custom translation instructions"),
    preprocess_repetitions: bool = Form(
        False, description="Deduplicate repeated phrases"
    ),
) -> Response:
    """Translate a PowerPoint presentation.

    Returns the translated PPTX file as a binary response with metadata in headers.
    """
    settings = get_settings()

    # Validate API key
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server",
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

    # Validate file signature
    ppt_buffer = io.BytesIO(file_content)
    is_valid, error_msg = validate_pptx_file(ppt_buffer)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_msg or "Invalid PPT/PPTX file format",
        )
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
        model=model,
        user_prompt=user_prompt,
        glossary=glossary,
        preprocess_repetitions=preprocess_repetitions,
    )

    # Execute translation
    LOGGER.info(
        "Starting translation: file=%s, source=%s, target=%s, model=%s",
        ppt_file.filename,
        source_lang,
        target_lang,
        model,
    )

    service = TranslationService(settings=settings)
    result = service.translate(request)

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
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
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
