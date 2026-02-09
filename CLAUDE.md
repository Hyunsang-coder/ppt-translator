# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPT 번역캣 is a PowerPoint translation application using LangChain with OpenAI GPT and Anthropic Claude models. It provides:
- **Streamlit web UI** (legacy, `app.py`)
- **FastAPI REST API** (`api.py`) with async job management
- **Next.js frontend** (`frontend/`) - modern React-based UI (migration in progress)

Features include slide text translation while preserving original formatting, glossary support, auto language detection, real-time progress streaming via SSE, and detailed logging.

**Requires Python >= 3.12** (see `pyproject.toml`)

## Development Commands

```bash
# Backend - Run the Streamlit app (legacy)
streamlit run app.py

# Backend - Run the FastAPI server
uvicorn api:app --reload --port 8000

# Backend - Run all tests
pytest tests/ -v

# Backend - Run a specific test
pytest tests/test_translation.py::HelperTestCase::test_chunk_paragraphs_preserves_order -v

# Backend - Run slow tests (require API calls)
pytest tests/ -v -m slow

# Frontend - Install dependencies
cd frontend && npm install

# Frontend - Run development server
cd frontend && npm run dev

# Frontend - Build for production
cd frontend && npm run build

# Frontend - Type check & lint
cd frontend && npx tsc --noEmit
```

## Environment Setup

### Backend (.env)
1. Create `.env` from `.env.example` and set API keys:
   - `OPENAI_API_KEY` - Required for OpenAI models (GPT)
   - `ANTHROPIC_API_KEY` - Required for Anthropic models (Claude)
2. Optional:
   - `CORS_ALLOWED_ORIGINS` - Comma-separated allowed origins (default: `http://localhost:3000,http://127.0.0.1:3000`)
3. Optional environment variables for tuning:
   - `TRANSLATION_MAX_CONCURRENCY` (default: 8)
   - `TRANSLATION_BATCH_SIZE` (default: 80)
   - `TRANSLATION_MIN_BATCH_SIZE` (default: 60) / `TRANSLATION_MAX_BATCH_SIZE` (default: 100)
   - `TRANSLATION_TARGET_BATCH_COUNT` (default: 5) - Target number of batches for batch size calculation
   - `TRANSLATION_WAVE_MULTIPLIER` (default: 1.2) - Wave multiplier for concurrency scheduling
   - `TRANSLATION_TPM_LIMIT` (default: 30000) - Tokens per minute limit
   - `TRANSLATION_RATE_LIMIT_RPS` (default: 1.0) - Requests per second
   - `TRANSLATION_RATE_LIMIT_CHECK_INTERVAL` (default: 0.1) - Rate check interval in seconds
   - `TRANSLATION_RATE_LIMIT_MAX_BUCKET` (default: 10) - Token bucket size for rate limiting

### Frontend (.env.local)
- `NEXT_PUBLIC_API_URL` - Backend API URL (default: empty string for Vercel proxy support; set to `http://localhost:8000` for local development)

## Deployment

### Docker (EC2)
- **Dockerfile**: Multi-stage build (Python 3.12-slim), non-root `appuser`, healthcheck via `/health`
- **docker-compose.yml**: Maps port 80→8000, 1536M memory limit, env-based CORS config
- Build & run: `docker compose up -d --build`

### Vercel (Frontend)
- **`frontend/vercel.json`**: Rewrites `/api/*` and `/health` to EC2 backend, enabling same-origin API calls
- `NEXT_PUBLIC_API_URL` defaults to empty string so relative paths use Vercel rewrites

## Architecture

### Entry Points
- `glossary_template.xlsx`: Sample glossary file for term substitution
- `app.py`: Streamlit UI orchestrating two workflows:
  - PPT Translation (main feature)
  - Text Extraction (PPT → Markdown)
- `api.py`: FastAPI REST API server
  - `GET /health`: Health check endpoint
  - `GET /api/v1/config`: Configuration (models, languages)
  - `GET /api/v1/models`: Available models list
  - `GET /api/v1/languages`: Supported languages list
  - `POST /api/v1/jobs`: Create translation job
  - `GET /api/v1/jobs/{job_id}`: Get job status
  - `GET /api/v1/jobs/{job_id}/events`: SSE progress stream
  - `GET /api/v1/jobs/{job_id}/result`: Download result
  - `DELETE /api/v1/jobs/{job_id}`: Cancel job
  - `POST /api/v1/translate`: Synchronous translation (legacy)
  - `POST /api/v1/extract`: Text extraction endpoint
  - `POST /api/v1/summarize`: Generate context summary for translation
  - `POST /api/v1/generate-instructions`: Generate translation style instructions
  - Supports AWS Lambda deployment via Mangum

### Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (handles shapes, tables, groups)
- `ppt_writer.py`: Applies translations back using run-based text distribution to preserve formatting; includes text fit (auto-shrink, expand-box) and color distribution support
- `text_extractor.py`: Converts PPTX to structured markdown with `ExtractionOptions`

### Service Layer (`src/services/`)
- `models.py`: Data models (`TranslationRequest`, `TranslationResult`, `TranslationProgress`, `TranslationStatus`, `TextFitMode`, `ProgressCallback`)
- `translation_service.py`: `TranslationService` class encapsulating translation workflow logic
  - Shared by both Streamlit (`app.py`) and FastAPI (`api.py`)
  - `ServiceProgressTracker`: Adapter for progress callbacks in service layer
  - Progress callback support for real-time updates
- `job_manager.py`: Async job management for FastAPI
  - `JobManager`: In-memory job store with event streaming (max 100 jobs, 1h cleanup)
  - `Job`: Job state tracking (pending/running/completed/failed/cancelled)
  - `JobType`: TRANSLATION / EXTRACTION
  - `JobState`: PENDING / RUNNING / COMPLETED / FAILED / CANCELLED
  - `JobEvent`: SSE event types for progress updates
  - `get_job_manager()`: Global singleton accessor
  - Cancellation keeps jobs in store (state=CANCELLED, completed_at set) for status queries; `_cleanup_old_jobs` removes after 1h
  - Terminal state guards: `complete_job`/`fail_job` skip if already CANCELLED, preventing race conditions

### Translation Chain (`src/chains/`)
- `llm_factory.py`: Factory for creating LLM instances (OpenAI/Anthropic) with provider-specific configuration, includes built-in rate limiting via `InMemoryRateLimiter`
- `translation_chain.py`: LangChain pipeline using structured output (`TranslationOutput` Pydantic model) for type-safe parsing, LangChain batch API for concurrent execution with tenacity retry logic
- `color_distribution_chain.py`: LLM-based color/format distribution for multi-color paragraphs — maps translated text segments back to original format groups
- `context_manager.py`: Builds global presentation context for consistent translations
- `summarization_chain.py`: AI-powered context and instructions generation (uses lightweight models: GPT-5 Mini / Haiku 4.5)

### Utilities (`src/utils/`)
- `config.py`: Settings dataclass loaded from environment
- `glossary_loader.py`: Excel glossary loading and term substitution (pre/post translation). Uses word boundary (`\b`) matching for Latin terms to prevent partial matches (e.g., "AI" won't match inside "MAIL"), falls back to plain string replace for CJK terms and terms with special characters (e.g., "C++", ".NET")
- `language_detector.py`: Uses langdetect with Korean↔English inference rules
- `repetition.py`: Deduplicates repeated phrases to reduce API calls
- `helpers.py`: Batch chunking, text segmentation for run distribution
- `security.py`: File validation, filename sanitization (preserves Unicode/spaces, collapses whitespace), HTML content escaping

### UI Components (`src/ui/`)
- `progress_tracker.py`: Streamlit progress bar and log updates
- `settings_panel.py`: Translation settings sidebar
- `file_handler.py`: Upload validation and buffer management
- `extraction_settings.py`: Text extraction options UI

### Frontend (`frontend/`)
Next.js 16 with React 19, TypeScript 5, Tailwind CSS 4, and Zustand 5 state management. Includes graceful fallback when backend is unavailable.

#### Pages (`src/app/`)
- `page.tsx`: Home page (redirects to translate)
- `translate/page.tsx`: Translation workflow page
- `extract/page.tsx`: Text extraction page
- `layout.tsx`: Root layout with ThemeProvider

#### Components (`src/components/`)
- **shared/**: `Header.tsx` (navigation + theme toggle), `FileUploader.tsx` (drag & drop)
- **translation/**: `TranslationForm.tsx`, `SettingsPanel.tsx` (includes `FilenameSettingsSection`), `ProgressPanel.tsx`, `LogViewer.tsx`
- **extraction/**: `ExtractionForm.tsx`, `MarkdownPreview.tsx`
- **ui/**: Shadcn/Radix UI components (button, card, checkbox, input, label, progress, radio-group, select, separator, sonner, switch, tabs, textarea, tooltip)

#### State Management (`src/stores/`)
- `translation-store.ts`: Zustand store for translation state (file, settings, progress, logs). `resetJobState()` resets only job-related state (jobId, status, progress, error, logs) while preserving file/settings/context for retranslation.
- `extraction-store.ts`: Zustand store for extraction state

#### API Integration (`src/lib/`)
- `api-client.ts`: REST API client with typed endpoints
- `sse-client.ts`: Server-Sent Events client for real-time updates
- `utils.ts`: Utility functions (cn for classnames)

#### Hooks (`src/hooks/`)
- `useTranslation.ts`: Translation workflow logic (includes auto context/instructions generation with markdown caching via file key, `retranslate()` for re-running with same file/settings). Uses `useTranslationStore.getState()` for `jobId` in `downloadResult`/`cancelTranslation` to avoid stale closure references. SSE connection is cleaned up on component unmount via `useEffect` to prevent memory leaks.
- `useExtraction.ts`: Extraction workflow logic
- `useConfig.ts`: Configuration data fetching with graceful fallback (fallback models/languages when backend unavailable). Backend connection errors are handled at API call time rather than pre-checked.

#### Types (`src/types/`)
- `api.ts`: TypeScript type definitions for API responses, settings, and job states

#### Deployment (`vercel.json`)
- Rewrites `/api/:path*` and `/health` to EC2 backend for same-origin API proxy

#### Styling (`src/app/globals.css`)
Centralized color management with CSS variables:
- **Semantic tokens**: `--background`, `--foreground`, `--primary`, `--secondary`, `--destructive`, `--success`, `--warning`, `--info`
- **Brand colors**: `--brand-black`, `--brand-white`
- **Glass morphism**: `--glass-bg`, `--glass-border`, `--glass-shadow`
- All colors defined in OKLch color space with light/dark mode variants

### Tests (`tests/`)
- `test_translation.py`: Unit tests for translation helpers and language detection
- `test_api.py`: FastAPI endpoint tests
- `test_color_distribution.py`: Color distribution validation and format grouping tests
- `test_text_fit.py`: Text fit modes (auto_shrink, expand_box, shrink_then_expand) and width expansion tests

## Key Patterns

### Translation Flow
1. `PPTParser.extract_paragraphs()` → `List[ParagraphInfo]` + `Presentation`
2. `ContextManager.build_global_context()` for cross-slide consistency
3. Optional: `build_repetition_plan()` to deduplicate identical text
4. `chunk_paragraphs()` creates batches with context/glossary
5. `translate_with_progress()` handles concurrent API calls
6. `expand_translations()` maps unique results back to duplicates
7. `_fix_color_distributions()` preserves multi-color formatting via LLM-based segment mapping
8. `PPTWriter.apply_translations()` writes back preserving run formatting, applies text fit and color distributions

### Async Job Flow (FastAPI + Next.js)
1. Frontend calls `POST /api/v1/jobs` with file, settings, and filename_settings (JSON)
2. Backend creates job, returns `job_id`
3. Frontend connects to `GET /api/v1/jobs/{job_id}/events` (SSE)
4. Backend streams events: `started`, `progress`, `complete`, `error`, `cancelled`, `keepalive`
5. Frontend tracks progress via SSE events and updates UI
6. Frontend calls `GET /api/v1/jobs/{job_id}/result` to download translated file
7. Output filename generated server-side based on `FilenameSettings` (auto/custom mode)

### Formatting Preservation
`PPTWriter` uses `split_text_into_segments()` with character-length weights to distribute translated text across original runs, preserving bold/italic/font styling. For multi-color paragraphs, `color_distribution_chain` uses LLM to split translated text by meaning and map segments back to original format groups.

### Text Fit
`TextFitMode` controls how translated text fits in text boxes:
- `none`: No adjustment
- `auto_shrink`: Reduce font size (down to `min_font_ratio`%) if text overflows
- `expand_box`: Widen text box to accommodate longer text (skips rotated/grouped/table shapes)
- `shrink_then_expand`: Try shrinking first, then expand if still overflowing

Width expansion is applied before text fit for all non-NONE modes. Font sizes are rounded to the nearest whole point (1 pt = 12700 EMU) to avoid fractional values in output PPTX.

### Progress Tracking
`TranslationProgress.percent` provides monotonic overall progress (never resets between phases):
- Parsing 2% → Language detection 5% → Batch prep 8% → Translation 10–80% → Color fix 80–90% → Apply 95% → Complete 100%

### Error Handling
- Translation chain uses tenacity with exponential backoff (3 attempts)
- Structured output via Pydantic ensures type-safe translation parsing (no JSON fallback needed)
- LangChain batch API handles concurrent execution with built-in rate limiting
- Background translation tasks catch `asyncio.CancelledError` separately from `Exception` to avoid treating cancellation as failure
- Frontend relies on API call-time error handling (catch blocks) rather than pre-flight backend connection checks, avoiding false positives from stale health snapshots (e.g. behind Vercel proxy)

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Get available models
curl http://localhost:8000/api/v1/models

# Create translation job
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=anthropic" \
  -F "model=claude-sonnet-4-5-20250929"

# Stream job progress (SSE)
curl -N http://localhost:8000/api/v1/jobs/{job_id}/events

# Download result
curl http://localhost:8000/api/v1/jobs/{job_id}/result -o translated.pptx

# Legacy synchronous translation
curl -X POST http://localhost:8000/translate \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=anthropic" \
  -F "model=claude-sonnet-4-5-20250929" \
  -o translated.pptx

# Generate context summary
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"markdown": "...", "provider": "openai", "model": "gpt-5-mini"}'

# Generate translation instructions
curl -X POST http://localhost:8000/api/v1/generate-instructions \
  -H "Content-Type: application/json" \
  -d '{"target_lang": "한국어", "markdown": "...", "provider": "openai", "model": "gpt-5-mini"}'
```

## Supported Models

### Translation Models
| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.2` | GPT-5.2 |
| OpenAI | `gpt-5-mini` | GPT-5 Mini |
| Anthropic | `claude-opus-4-6` | Claude Opus 4.6 |
| Anthropic | `claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

### Context/Instructions Generation
Lightweight models are used for auto-generating context and instructions:
- **OpenAI**: GPT-5 Mini
- **Anthropic**: Claude Haiku 4.5

## Libraries

### Backend
- **LangChain**: Translation chain with `ChatOpenAI`/`ChatAnthropic` and `PromptTemplate`
- **langchain-anthropic**: Anthropic Claude model integration
- **python-pptx**: PPTX parsing and writing
- **Streamlit**: Web UI with session state management
- **FastAPI**: REST API server with automatic OpenAPI docs (v2.4.0)
- **Mangum**: AWS Lambda adapter for FastAPI
- **tenacity**: Retry logic with exponential backoff
- **langdetect**: Automatic language detection
- **sse-starlette**: Server-Sent Events for streaming
- **Pillow**: Image processing
- **pandas + openpyxl**: Excel glossary file handling
- **PyMuPDF + pytesseract + opencv-python-headless**: PDF/image extraction support

### Frontend
- **Next.js 16**: React framework with App Router (`next@16.1.6`)
- **React 19**: UI library (`react@19.2.3`)
- **TypeScript 5**: Type safety
- **Tailwind CSS 4**: Utility-first CSS with `@tailwindcss/postcss`
- **Zustand 5**: Lightweight state management
- **Radix UI**: Accessible component primitives (via `radix-ui` package)
- **Lucide React**: Icon library
- **react-dropzone**: File upload handling
- **next-themes**: Dark/light mode toggle
- **sonner**: Toast notifications
- **class-variance-authority + tailwind-merge + clsx**: Component styling utilities

## Claude Code Commands & Agents

### Custom Commands (`.claude/commands/`)
- `/commit` - Review changes and create well-formed git commit
- `/push` - Push local commits to remote with safety checks
- `/dev-backend` - Start FastAPI development server on port 8000
- `/dev-frontend` - Start Next.js development server
- `/deploy-ec2` - Deploy backend to EC2 (git pull + docker compose rebuild + health check)
- `/check-ec2` - Check EC2 server status without deploying (SSH connection, container status, health)
- `/update-docs` - Analyze codebase and update CLAUDE.md

### Custom Agents (`.claude/agents/`)
- `code-reviewer` - Reviews code changes for quality, security, and best practices
- `test-runner` - Runs tests and reports results concisely
- `translation-qa` - Reviews translation logic and prompt quality

## Claude Code Customization Best Practices

### Skills
커스텀 슬래시 명령어 정의 시 준수사항:
- **명확한 트리거**: `/command` 형식의 직관적 명령어명 사용
- **단일 책임**: 하나의 skill은 하나의 명확한 작업만 수행
- **문서화**: 각 skill의 목적, 입력, 출력을 명시
- **에러 핸들링**: 실패 시 명확한 피드백 제공
- **멱등성**: 동일 입력에 동일 결과 보장

```markdown
# Example skill definition in .claude/skills/
name: translate-slide
description: Translate a single slide with glossary support
trigger: /translate-slide
```

### Subagents (Task Tool)
전문화된 에이전트 위임 시 준수사항:
- **적절한 agent type 선택**:
  - `Explore`: 코드베이스 탐색, 파일 검색
  - `Plan`: 구현 전략 설계
  - `Bash`: Git, 빌드, 시스템 명령
  - `general-purpose`: 복합 멀티스텝 작업
- **명확한 프롬프트**: 컨텍스트와 기대 결과를 구체적으로 명시
- **병렬 실행**: 독립적인 작업은 동시에 여러 Task 호출
- **결과 검증**: 서브에이전트 결과를 맹신하지 않고 검증

```python
# Parallel task example
Task(subagent_type="Explore", prompt="Find all translation-related files")
Task(subagent_type="Explore", prompt="Find all test files")  # 병렬 호출
```

### Hooks
이벤트 기반 자동화 시 준수사항:
- **최소 권한**: 필요한 작업만 수행, 과도한 권한 요청 금지
- **빠른 실행**: 훅은 빠르게 완료되어야 함 (blocking 최소화)
- **실패 허용**: 훅 실패가 전체 워크플로우를 중단시키지 않도록 설계
- **로깅**: 디버깅을 위한 적절한 로그 출력
- **조건부 실행**: 불필요한 실행을 피하기 위한 조건 검사

```json
// .claude/settings.json hook example
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "echo 'File modification detected'"
      }
    ]
  }
}
```

### 공통 원칙
1. **보안 우선**: 민감 정보(API 키, 자격 증명) 노출 금지
2. **테스트 가능성**: 커스텀 확장은 독립적으로 테스트 가능하게 설계
3. **버전 관리**: `.claude/` 설정 파일은 git으로 관리
4. **점진적 개선**: 작게 시작하여 필요에 따라 확장
5. **문서화**: 팀원이 이해할 수 있도록 충분한 주석과 README 제공
