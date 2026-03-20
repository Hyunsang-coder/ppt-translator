# Environment & Deployment

## Backend (.env)
```
OPENAI_API_KEY=         # Required for OpenAI models
ANTHROPIC_API_KEY=      # Required for Anthropic models
CORS_ALLOWED_ORIGINS=   # Comma-separated (default: http://localhost:3000,http://127.0.0.1:3000)
```

### Tuning Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSLATION_MAX_CONCURRENCY` | 8 | Max concurrent API calls |
| `TRANSLATION_BATCH_SIZE` | 80 | Default batch size |
| `TRANSLATION_MIN_BATCH_SIZE` | 60 | Min batch size |
| `TRANSLATION_MAX_BATCH_SIZE` | 100 | Max batch size |
| `TRANSLATION_TARGET_BATCH_COUNT` | 5 | Target number of batches |
| `TRANSLATION_WAVE_MULTIPLIER` | 1.2 | Concurrency wave multiplier |
| `TRANSLATION_TPM_LIMIT` | 30000 | Tokens per minute limit |
| `TRANSLATION_MAX_RUNNING_JOBS` | 2 | Max concurrent running jobs |
| `TRANSLATION_MAX_QUEUED_JOBS` | 5 | Max queued jobs |
| `TRANSLATION_RATE_LIMIT_RPS` | 1.0 | Requests per second |
| `TRANSLATION_RATE_LIMIT_CHECK_INTERVAL` | 0.1 | Rate check interval (s) |
| `TRANSLATION_RATE_LIMIT_MAX_BUCKET` | 10 | Token bucket size |

## Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=    # Default: empty (Vercel proxy); local dev: http://localhost:8000
```

## Deployment

### Docker (EC2)
- **Dockerfile**: Multi-stage (Python 3.12-slim), non-root `appuser`, healthcheck `/health`
- **docker-compose.yml**: Port 80→8000, 1536M memory, env-based CORS
- `docker compose up -d --build`

### Vercel (Frontend)
- `frontend/vercel.json`: Rewrites `/api/*` and `/health` to EC2 backend
- `NEXT_PUBLIC_API_URL` empty → relative paths use Vercel rewrites

### CI/CD (`.github/workflows/`)
- `ci.yml`: PR + manual. Backend pytest (Python 3.12), frontend `tsc --noEmit` (Node 20)
- `predeploy.yml`: Pre-deployment validation

## Supported Models

| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.4-2026-03-05` | GPT-5.4 |
| OpenAI | `gpt-5.4-mini-2026-03-17` | GPT-5.4 Mini |
| Anthropic | `claude-opus-4-6` | Claude Opus 4.6 |
| Anthropic | `claude-sonnet-4-6` | Claude Sonnet 4.6 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

Context/instructions generation uses lightweight models: GPT-5.4 Mini / Haiku 4.5.

## Libraries

### Backend
LangChain, langchain-anthropic, python-pptx, FastAPI (v2.4.0), Mangum, tenacity, langdetect, sse-starlette, Pillow, pandas + openpyxl, PyMuPDF + pytesseract + opencv-python-headless

### Frontend
Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, Zustand 5, Radix UI, Lucide React, react-dropzone, next-themes, sonner, class-variance-authority + tailwind-merge + clsx

## API Usage
```bash
# Health check
curl http://localhost:8000/health

# Create translation job
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=anthropic" \
  -F "model=claude-sonnet-4-6"

# Stream job progress
curl -N http://localhost:8000/api/v1/jobs/{job_id}/events

# Download result
curl http://localhost:8000/api/v1/jobs/{job_id}/result -o translated.pptx

# Generate context summary
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"markdown": "...", "provider": "anthropic", "model": "claude-haiku-4-5-20251001"}'

# Generate translation instructions
curl -X POST http://localhost:8000/api/v1/generate-instructions \
  -H "Content-Type: application/json" \
  -d '{"target_lang": "한국어", "markdown": "...", "provider": "anthropic", "model": "claude-haiku-4-5-20251001"}'
```
