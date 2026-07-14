# Environment & Deployment

## API Keys

**Desktop app**: API keys are saved through Tauri commands into the OS keychain
(macOS Keychain / Windows Credential Manager). When the Python sidecar starts,
the Rust shell reads those keys and injects them as `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` environment variables.

**Local API dev**: `uvicorn --reload` reads a local `.env` via `load_dotenv()`.

## Sidecar env vars
```
OPENAI_API_KEY=         # Injected by Tauri, or .env for local API dev
ANTHROPIC_API_KEY=      # Injected by Tauri, or .env for local API dev
CORS_ALLOWED_ORIGINS=   # Comma-separated (default: http://localhost:3000,http://127.0.0.1:3000)
CORS_ALLOW_ALL=1        # Set by Tauri for loopback-only desktop sidecar
MAX_UPLOAD_SIZE_MB=1024 # Max uploaded PPT/PPTX size
```

### Tuning Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | 1024 | Max uploaded PPT/PPTX size |
| `TRANSLATION_MAX_CONCURRENCY` | 8 | Max concurrent API calls |
| `TRANSLATION_BATCH_SIZE` | 80 | Default batch size |
| `TRANSLATION_MIN_BATCH_SIZE` | 60 | Min batch size |
| `TRANSLATION_MAX_BATCH_SIZE` | 100 | Max batch size |
| `TRANSLATION_TARGET_BATCH_COUNT` | 5 | Minimum target number of translation batches |
| `TRANSLATION_WAVE_MULTIPLIER` | 1.2 | Concurrency wave multiplier |
| `TRANSLATION_TPM_LIMIT` | 30000 | Tokens per minute limit |
| `TRANSLATION_MAX_RUNNING_JOBS` | 2 | Max concurrent running jobs |
| `TRANSLATION_MAX_QUEUED_JOBS` | 5 | Max queued jobs |
| `TRANSLATION_RATE_LIMIT_RPS` | 1.0 | Requests per second |
| `TRANSLATION_RATE_LIMIT_CHECK_INTERVAL` | 0.1 | Rate check interval (s) |
| `TRANSLATION_RATE_LIMIT_MAX_BUCKET` | 10 | Token bucket size |

## Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=    # Browser-only local dev fallback; Tauri uses runtime sidecar port
```

## Deployment

### Desktop (Tauri)
- `node desktop/build-sidecar.mjs`: Builds and stages the Python sidecar into `src-tauri/resources/sidecar/` (cross-platform; auto-creates the desktop venv on first run)
- `TAURI_BUILD=1 cargo tauri build`: Builds the static frontend export and packages the desktop app
- Build the sidecar on each target OS because PyInstaller output is platform-specific
- **Auto-update**: the app self-updates from GitHub Releases (`tauri-plugin-updater`). Releases must carry signed updater bundles + `latest.json`, produced by `desktop-release.yml`. Requires the `TAURI_SIGNING_*` (updater) and `APPLE_*` (macOS notarization) secrets — see [`docs/CICD.md`](CICD.md) "Auto-update". The updater public key lives in `src-tauri/tauri.conf.json` → `plugins.updater.pubkey`

### Vercel
- Public web deployment is a download 안내 page only
- No API rewrites or hosted translation backend are used
- The root page shows the latest GitHub Release tag and published date
- Production deploys are handled by `deploy-web.yml` (see [`docs/CICD.md`](CICD.md))

### CI/CD (`.github/workflows/`)
- `ci.yml`: PR + manual. Backend pytest (Python 3.12), frontend `tsc --noEmit` (Node 20)
- `predeploy.yml`: Full validation on `main`; triggers `deploy-web.yml` on success
- `deploy-web.yml`: Vercel production deploy for the public download page
- `desktop-release.yml`: Tag/manual. Builds + signs desktop installers (`tauri-action`), generates the `latest.json` auto-update manifest, publishes the GitHub Release, then deploys web
- See [`docs/CICD.md`](CICD.md) for bootstrap and release commands

## Supported Models

Single source of truth: `MODEL_REGISTRY` in `src/services/models.py`. `api.py`
(`SUPPORTED_MODELS`, validation) and `llm_factory.get_models_for_provider` derive
from it, and the frontend loads the registry through the API at startup.

| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.6-sol` | GPT-5.6 Sol (High) |
| OpenAI | `gpt-5.6-luna` | GPT-5.6 Luna (High) |
| Anthropic | `claude-opus-4-8` | Claude Opus 4.8 |
| Anthropic | `claude-sonnet-5` | Claude Sonnet 5 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

## Libraries

### Backend
LangChain, langchain-anthropic, python-pptx, FastAPI, tenacity, langdetect, sse-starlette, Pillow, pandas + openpyxl

### Frontend
Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, Zustand 5, Radix UI, Lucide React, react-dropzone, next-themes, sonner, class-variance-authority + tailwind-merge + clsx, @tauri-apps/api + plugin-dialog/plugin-fs/plugin-updater/plugin-process

## API Usage
```bash
# Health check
curl http://localhost:8000/health

# Create translation job
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=anthropic" \
  -F "model=claude-sonnet-5"

# Stream job progress
curl -N http://localhost:8000/api/v1/jobs/{job_id}/events

# Download result
curl http://localhost:8000/api/v1/jobs/{job_id}/result -o translated.pptx

```
