# CLAUDE.md

## Source of Truth

This file is the single source of truth for project-level agent instructions. `AGENTS.md` intentionally references this file instead of duplicating instructions. Update this file when project commands, agent roles, or reference docs change.

## Project Overview

PPT 번역캣 — Tauri desktop app for PowerPoint translation (LangChain + OpenAI/Anthropic). The UI lives in `frontend/`, the desktop shell in `src-tauri/`, and the local Python sidecar uses `api.py` + `src/`. Preserves original formatting, glossary support, auto language detection, and real-time progress via polling.

**Python >= 3.12** required.

## Development Commands

```bash
# Backend
uvicorn api:app --reload --port 8000
pytest tests/ -v                    # all tests
pytest tests/ -v -m slow            # slow tests (API calls)

# Frontend
cd frontend && npm install
cd frontend && npm run dev
cd frontend && npm run build
cd frontend && npx tsc --noEmit     # type check

# Desktop (cross-platform Node sidecar build; Windows/macOS/Linux)
node desktop/build-sidecar.mjs
cd src-tauri && cargo tauri dev
TAURI_BUILD=1 cargo tauri build
```

## Claude Code Commands

- `/commit` — Review & commit
- `/push` — Push with safety checks
- `/dev-backend` / `/dev-frontend` — Start dev servers
- `/update-docs` — Update CLAUDE.md and any affected docs

### Agents
- `code-reviewer` — Code quality & security review
- `test-runner` — Run tests concisely
- `translation-qa` — Translation logic & prompt review

## Reference Docs

Detailed documentation in `docs/`:
- [`docs/CICD.md`](docs/CICD.md) — CI/CD pipelines, Vercel bootstrap, desktop release commands
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Backend/frontend structure, components, tests
- [`docs/KEY_PATTERNS.md`](docs/KEY_PATTERNS.md) — Translation flow, job flow, text fit, thread safety, error handling
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — Env vars, deployment, models, libraries, API usage
