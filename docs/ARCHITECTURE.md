# Architecture

## Entry Points
- `src-tauri/`: Tauri desktop shell
  - Stores API keys in the OS keychain
  - Spawns the bundled Python sidecar
  - Emits the sidecar port to the WebView
- `desktop/sidecar.py`: Sidecar launcher for the local FastAPI server
- `api.py`: FastAPI REST API app used by the sidecar and local development
  - `GET /health`: Health check
  - `GET /api/v1/config`: Configuration (models, languages)
  - `GET /api/v1/models`: Available models list
  - `GET /api/v1/languages`: Supported languages list
  - `POST /api/v1/jobs`: Create translation job
    - Uploads are streamed into a per-job temporary directory instead of being retained as `BytesIO`
    - Optional image compression writes the compressed PPTX back to the job directory
  - `GET /api/v1/jobs/{job_id}`: Get job status
  - `GET /api/v1/jobs/{job_id}/events`: SSE progress stream
  - `GET /api/v1/jobs/{job_id}/result`: Download result via `FileResponse` when the output is stored on disk
  - `DELETE /api/v1/jobs/{job_id}`: Cancel job and clean temporary files
  - `POST /translate`: Synchronous translation endpoint (legacy compatibility)
  - `POST /api/v1/extract`: Text extraction
  - `POST /api/v1/summarize`: Context summary generation
  - `POST /api/v1/generate-instructions`: Translation style instructions
    - Uses the shared LLM factory/rate limiter and a deterministic low temperature
    - Long markdown previews keep both document head and tail
- `main.py`: Placeholder entry point (uv/pyproject.toml default)
- `glossary_template.xlsx`: Sample glossary file
- `scripts/verify_color_matching.py`: Live color matching verification loop
  - Builds a multi-color PPTX fixture
  - Runs color distribution through configured provider models
  - Re-opens the output PPTX and checks anchor substrings against expected colors
  - Supports multiple iterations for prompt/model regression checks

## Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (shapes, tables, groups)
- `ppt_writer.py`: Applies translations preserving run formatting; text fit (auto-shrink, expand-box) and color distribution. Can return an in-memory buffer for legacy callers or save directly to an output path for job results.
- `text_extractor.py`: PPTX to structured markdown with `ExtractionOptions`

## Service Layer (`src/services/`)
- `models.py`: Data models (`TranslationRequest`, `TranslationResult`, `TranslationProgress`, `TranslationStatus`, `TextFitMode`, `ProgressCallback`) + `MODEL_REGISTRY` (single source of truth for supported models / default model IDs)
- `translation_service.py`: `TranslationService` class with `ServiceProgressTracker` for progress callbacks
  - Primary runtime path is `translate_async()`
  - Legacy `translate()` is a sync wrapper for non-async callers
  - PPT parse/write uses `asyncio.to_thread()` because `python-pptx` is synchronous
  - LLM calls use LangChain async APIs so job cancellation can propagate into provider calls
  - Builds global presentation context plus per-batch nearby context
  - Uses token-estimated batching while preserving paragraph-count limits
- `job_manager.py`: Async job management
  - `JobManager`: In-memory store (max 100 jobs, 1h TTL; periodic background cleanup started in the FastAPI lifespan)
  - `Job`: State tracking (pending/running/completed/failed/cancelled), output path, output filename, and temporary work directory
  - Concurrency: `running_semaphore` + `try_create_job()` atomic admission (429 on overflow)
  - Thread-safe: `add_event()` uses `call_soon_threadsafe` for worker→event-loop bridging
  - Terminal state guards: `complete_job`/`fail_job` skip if already CANCELLED
  - Cancellation: cancels the running task, awaits completion, marks the job cancelled, and removes temporary files
  - Cleanup: old terminal jobs remove their work directories before being dropped

## Translation Chain (`src/chains/`)
- `llm_factory.py`: LLM factory (OpenAI/Anthropic)
  - Supported models come from `MODEL_REGISTRY`
  - Default rate limiter is shared by provider/API-key fingerprint so separate LLM instances do not bypass request pacing
- `translation_chain.py`: LangChain translation pipeline with structured output (`TranslationOutput`)
  - `translate_with_progress_async()` uses `abatch_as_completed()` for real-time progress and cancellable provider calls
  - Sync `translate_with_progress()` remains for compatibility/tests
  - Tenacity retries preserve successful batch results and only resubmit unfinished batches
  - Count mismatches retry the affected batch once, then pad/trim with original text as a safe fallback
  - Prompt includes presentation context, current batch context, glossary priority, PPT-specific preservation rules, and concise slide-text guidance
- `color_distribution_chain.py`: LLM-based color/format distribution for multi-color paragraphs
  - Provides sync and async paths
  - Uses lightweight provider defaults for post-processing
  - Falls back to ratio-based splitting when distribution fails validation
- `context_manager.py`: Global and per-batch context for consistency
  - `build_global_context()` emits a compact slide outline
  - `build_batch_context()` marks nearby text vs. current-batch text for disambiguation
- `summarization_chain.py`: Translation-focused context summary generation
  - Produces a compact, structured summary covering document purpose, audience, key terms/proper nouns, and translation cautions
  - Uses async LangChain invocation and low temperature for stable output

## Utilities (`src/utils/`)
- `config.py`: Settings from environment
  - Upload size, batch sizing, token batch cap, job concurrency, TPM limit, and rate limiter tuning
- `glossary_loader.py`: Excel glossary with `\b` matching for Latin terms, plain replace for CJK
- `language_detector.py`: langdetect with Korean↔English rules
- `repetition.py`: Deduplication to reduce API calls
- `helpers.py`: Batch chunking, token-estimated batch splitting, text segmentation
- `security.py`: File validation for file-like objects, filename sanitization, HTML escaping
- `image_compressor.py`: ZIP-level image compression (high/medium/low presets)
  - Supports both in-memory compatibility and file-output compression for lower job memory peaks
- `color_match_verifier.py`: Reusable color matching verification helpers used by the live script and local tests

## Frontend (`frontend/`)
Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, Zustand 5.

### Pages (`src/app/`)
- `page.tsx`: Public Vercel download 안내 page
- `translate/page.tsx`, `extract/page.tsx`, `settings/page.tsx`: Desktop app screens
- `layout.tsx`: Root layout with ThemeProvider

### Components (`src/components/`)
- **shared/**: `Header.tsx`, `FileUploader.tsx`
- **translation/**: `TranslationForm.tsx`, `SettingsPanel.tsx`, `ProgressPanel.tsx`, `LogViewer.tsx`
- **extraction/**: `ExtractionForm.tsx`, `MarkdownPreview.tsx`
- `desktop-shell.tsx`: Desktop-only route wrapper; redirects hosted web app routes back to the public root
- `sidecar-provider.tsx`: Tauri sidecar port bootstrap for desktop app screens
- **ui/**: Shadcn/Radix UI components

### State & Hooks
- `stores/translation-store.ts`: Zustand store. `resetJobState()` preserves file/settings. Defaults: `textFitMode: "expand_box"`, `imageCompression: "medium"`, `lengthLimit: null`
- `stores/extraction-store.ts`: Extraction state
- `hooks/useTranslation.ts`: Translation workflow with `retranslate()`. Uses `getState()` to avoid stale closures
- `hooks/useExtraction.ts`: Extraction workflow
- `hooks/useConfig.ts`: Config fetching with graceful fallback
- `lib/api-base.ts`: Tauri sidecar URL resolution with local browser fallback
- `lib/api-client.ts`: REST client
- `lib/keychain.ts`: Tauri keychain command wrappers
- `lib/save-file.ts`: Native save dialog in Tauri, browser download fallback
- `lib/sse-client.ts`: Polling client for progress updates

### Styling (`globals.css`)
CSS variables in OKLch color space with light/dark mode variants. Semantic tokens + glass morphism.

## Tests (`tests/`)
- `test_translation.py`: Translation helpers, language detection, prompt/context regression tests, async translation retry/cancellation helpers
- `test_api.py`: FastAPI endpoints
- `test_color_distribution.py`: Color distribution, format grouping
- `test_color_match_verifier.py`: Color matching verifier helper tests
- `test_text_fit.py`: Text fit modes, width expansion, auto_size preservation, placeholder regression
- `test_job_manager.py`: Job state, cancellation, temp-file cleanup, concurrency admission
- `test_batch_size.py`: Batch size edge cases
- `test_security_fix.py`: HTML sanitization, XSS prevention
- `test_image_compressor.py`: Image compression, transparency

Current local regression suite: `194` tests. Provider smoke tests have also been run manually against `.env` API keys for Anthropic `claude-sonnet-4-6` and OpenAI `gpt-5.5-2026-04-23` using a minimal PPTX through the full `TranslationService.translate_async()` path. Color matching verification has also passed 3 iterations each for Anthropic `claude-haiku-4-5-20251001` and OpenAI `gpt-5.4-mini-2026-03-17` via `scripts/verify_color_matching.py`.
