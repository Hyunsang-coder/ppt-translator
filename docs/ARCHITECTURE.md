# Architecture

## Entry Points
- `api.py`: FastAPI REST API server
  - `GET /health`: Health check
  - `GET /api/v1/config`: Configuration (models, languages)
  - `GET /api/v1/models`: Available models list
  - `GET /api/v1/languages`: Supported languages list
  - `POST /api/v1/jobs`: Create translation job
  - `GET /api/v1/jobs/{job_id}`: Get job status
  - `GET /api/v1/jobs/{job_id}/events`: SSE progress stream
  - `GET /api/v1/jobs/{job_id}/result`: Download result
  - `DELETE /api/v1/jobs/{job_id}`: Cancel job
  - `POST /api/v1/translate`: Synchronous translation (legacy)
  - `POST /api/v1/extract`: Text extraction
  - `POST /api/v1/summarize`: Context summary generation
  - `POST /api/v1/generate-instructions`: Translation style instructions
  - Supports AWS Lambda deployment via Mangum
- `app.py`: Streamlit redirect page (points to Vercel frontend)
- `main.py`: Placeholder entry point (uv/pyproject.toml default)
- `glossary_template.xlsx`: Sample glossary file

## Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (shapes, tables, groups)
- `ppt_writer.py`: Applies translations preserving run formatting; text fit (auto-shrink, expand-box) and color distribution
- `text_extractor.py`: PPTX to structured markdown with `ExtractionOptions`

## Service Layer (`src/services/`)
- `models.py`: Data models (`TranslationRequest`, `TranslationResult`, `TranslationProgress`, `TranslationStatus`, `TextFitMode`, `ProgressCallback`)
- `translation_service.py`: `TranslationService` class with `ServiceProgressTracker` for progress callbacks
- `job_manager.py`: Async job management
  - `JobManager`: In-memory store (max 100 jobs, 1h cleanup)
  - `Job`: State tracking (pending/running/completed/failed/cancelled)
  - Concurrency: `running_semaphore` + `try_create_job()` atomic admission (429 on overflow)
  - Thread-safe: `add_event()` uses `call_soon_threadsafe` for worker→event-loop bridging
  - Terminal state guards: `complete_job`/`fail_job` skip if already CANCELLED

## Translation Chain (`src/chains/`)
- `llm_factory.py`: LLM factory (OpenAI/Anthropic) with `InMemoryRateLimiter`
- `translation_chain.py`: LangChain pipeline with structured output (`TranslationOutput`), batch API, tenacity retry. Fail-fast validation on missing results
- `color_distribution_chain.py`: LLM-based color/format distribution for multi-color paragraphs
- `context_manager.py`: Global presentation context for consistency
- `summarization_chain.py`: Context/instructions generation (GPT-5 Mini / Haiku 4.5)

## Utilities (`src/utils/`)
- `config.py`: Settings from environment
- `glossary_loader.py`: Excel glossary with `\b` matching for Latin terms, plain replace for CJK
- `language_detector.py`: langdetect with Korean↔English rules
- `repetition.py`: Deduplication to reduce API calls
- `helpers.py`: Batch chunking, text segmentation
- `security.py`: File validation, filename sanitization, HTML escaping
- `image_compressor.py`: ZIP-level image compression (high/medium/low presets)

## Frontend (`frontend/`)
Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, Zustand 5.

### Pages (`src/app/`)
- `page.tsx`: Home (redirects to translate)
- `translate/page.tsx`, `extract/page.tsx`, `how-it-works/page.tsx`, `patch-notes/page.tsx`
- `layout.tsx`: Root layout with ThemeProvider

### Components (`src/components/`)
- **shared/**: `Header.tsx`, `FileUploader.tsx`
- **translation/**: `TranslationForm.tsx`, `SettingsPanel.tsx`, `ProgressPanel.tsx`, `LogViewer.tsx`
- **extraction/**: `ExtractionForm.tsx`, `MarkdownPreview.tsx`
- **how-it-works/**: `PipelineTimeline.tsx`, `PipelineStep.tsx`
- **patch-notes/**: `PatchNotesList.tsx`, `PatchNoteCard.tsx`
- **ui/**: Shadcn/Radix UI components

### State & Hooks
- `stores/translation-store.ts`: Zustand store. `resetJobState()` preserves file/settings. Defaults: `textFitMode: "expand_box"`, `imageCompression: "medium"`, `lengthLimit: null`
- `stores/extraction-store.ts`: Extraction state
- `hooks/useTranslation.ts`: Translation workflow with markdown caching, `retranslate()`. Uses `getState()` to avoid stale closures
- `hooks/useExtraction.ts`: Extraction workflow
- `hooks/useConfig.ts`: Config fetching with graceful fallback
- `lib/api-client.ts`: REST client
- `lib/sse-client.ts`: Polling client for progress updates

### Styling (`globals.css`)
CSS variables in OKLch color space with light/dark mode variants. Semantic tokens + glass morphism.

## Tests (`tests/`)
- `test_translation.py`: Translation helpers, language detection
- `test_api.py`: FastAPI endpoints
- `test_color_distribution.py`: Color distribution, format grouping
- `test_text_fit.py`: Text fit modes, width expansion, auto_size preservation, placeholder regression
- `test_job_manager.py`: Job state, concurrency admission
- `test_batch_size.py`: Batch size edge cases
- `test_security_fix.py`: HTML sanitization, XSS prevention
- `test_image_compressor.py`: Image compression, transparency
