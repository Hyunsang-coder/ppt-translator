# Architecture

## Entry Points
- `src-tauri/`: Tauri desktop shell
  - Stores API keys in the OS keychain
  - Spawns the bundled Python sidecar
  - Emits the sidecar port to the WebView
  - In-app auto-update via `tauri-plugin-updater` + `tauri-plugin-process` (relaunch); see [`docs/CICD.md`](CICD.md) "Auto-update"
- `desktop/sidecar.py`: Sidecar launcher for the local FastAPI server
- `api.py`: FastAPI REST API app used by the sidecar and local development
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
- `main.py`: Placeholder entry point (uv/pyproject.toml default)
- `glossary_template.xlsx`: Sample glossary file

## Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (shapes, tables, groups)
- `ppt_writer.py`: Applies translations preserving run formatting; text fit (auto-shrink, expand-box) and color distribution
- `text_extractor.py`: PPTX to structured markdown with `ExtractionOptions`

## Service Layer (`src/services/`)
- `models.py`: Data models (`TranslationRequest`, `TranslationResult`, `TranslationProgress`, `TranslationStatus`, `TextFitMode`, `ProgressCallback`) + `MODEL_REGISTRY` (single source of truth for supported models / default model IDs)
- `translation_service.py`: `TranslationService` class with `ServiceProgressTracker` for progress callbacks
- `job_manager.py`: Async job management
  - `JobManager`: In-memory store (max 100 jobs, 1h TTL; periodic background cleanup started in the FastAPI lifespan)
  - `Job`: State tracking (pending/running/completed/failed/cancelled)
  - Concurrency: `running_semaphore` + `try_create_job()` atomic admission (429 on overflow)
  - Thread-safe: `add_event()` uses `call_soon_threadsafe` for worker→event-loop bridging
  - Terminal state guards: `complete_job`/`fail_job` skip if already CANCELLED

## Translation Chain (`src/chains/`)
- `llm_factory.py`: LLM factory (OpenAI/Anthropic) with `InMemoryRateLimiter`
- `translation_chain.py`: LangChain pipeline with structured output (`TranslationOutput`), batch API, tenacity retry. Fail-fast validation on missing results
- `color_distribution_chain.py`: LLM-based simultaneous translation + semantic style mapping for multi-color paragraphs
- `context_manager.py`: Global presentation context for consistency
- `summarization_chain.py`: Context/instructions generation (GPT-5.4 Mini / Haiku 4.5)

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
- `page.tsx`: Public Vercel download 안내 page
- `translate/page.tsx`, `extract/page.tsx`: Desktop app screens
- `settings/page.tsx`: API key management (keychain) + "앱 업데이트" card (current version via `getVersion()`, manual update check)
- `layout.tsx`: Root layout with ThemeProvider

### Components (`src/components/`)
- **shared/**: `Header.tsx`, `FileUploader.tsx`
- **translation/**: `TranslationForm.tsx`, `SettingsPanel.tsx`, `ProgressPanel.tsx`, `LogViewer.tsx`
- **extraction/**: `ExtractionForm.tsx`, `MarkdownPreview.tsx`
- `desktop-shell.tsx`: Desktop-only route wrapper; redirects hosted web app routes back to the public root. Mounts the auto-update gate (`AutoUpdateGate`) in Tauri — shows `UpdateModal` on startup-check or manual `app:update-found` event
- `sidecar-provider.tsx`: Tauri sidecar port bootstrap for desktop app screens
- **ui/**: Shadcn/Radix UI components. `update-modal.tsx`: self-contained update dialog (download progress, skip-version)

### State & Hooks
- `stores/translation-store.ts`: Zustand store. `resetJobState()` preserves file/settings. Defaults: `textFitMode: "expand_box"`, `imageCompression: "medium"`, `lengthLimit: null`
- `stores/extraction-store.ts`: Extraction state
- `hooks/useTranslation.ts`: Translation workflow with `retranslate()`. Uses `getState()` to avoid stale closures
- `hooks/useExtraction.ts`: Extraction workflow
- `hooks/useConfig.ts`: Config fetching with graceful fallback
- `hooks/useAutoUpdate.ts`: Tauri auto-update lifecycle — startup check (production only), download/install with progress + relaunch, skip-version via localStorage. No-op outside Tauri
- `lib/api-base.ts`: Tauri sidecar URL resolution with local browser fallback
- `lib/api-client.ts`: REST client
- `lib/keychain.ts`: Tauri keychain command wrappers
- `lib/save-file.ts`: Native save dialog in Tauri, browser download fallback
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
